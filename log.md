# TRENCH-IDS — Implementation Log

A dated, chronological changelog: what changed, when, and why. This is the third leg alongside `project-metrics.md` (every hard number) and `flow.md` (how the pieces connect architecturally, right now, independent of when each piece was built). Append a new dated entry whenever something is implemented, redesigned, or fixed — don't edit past entries except to correct a factual error in them; supersede old decisions with a new entry instead of rewriting history.

---

## 2026-07-13 — Second design respec: 2 datasets/6 classes/4 tasks → 3 datasets/10 classes/6 tasks

Per the professor's revised guidance: rather than minimizing complexity, build a richer continual-learning benchmark (more tasks, more attack classes, no pre-graph sampling cap).

- **NF-BoT-IoT-v2 back in the dataset pool**, restricted to Reconnaissance only via `CLASS_DATASETS` (it also independently contains large DDoS/DoS volumes that must not leak into those canonical classes).
- **NF-UNSW-NB15-v2 stays excluded** — its only relevant class, Reconnaissance, is 99.5% supplied by BoT-IoT anyway once BoT-IoT is back in.
- **Step 2 redesigned**: one large `HeteroData` graph per task replaced with many small mini-graphs (`graph_size`, default 300 flows), split by train/val/test before chunking. CL task boundaries (T1→T6) unchanged.
- Every flow gains a stable `flow_id` (`f"{source_dataset}-{original_csv_row_number}"`), assigned in Step 1, carried through to the Step 2 Flow node.
- Relation-type strings renamed for clarity (pure rename, no schema change): `host--sends-->flow` → `host--originates-->flow`, `flow--received_by-->host` → `flow--terminates_at-->host`, `flow--uses_port-->port` → `flow--targets_port-->port`, `host--talks_to-->host` → `host--communicates_with-->host`.
- Step 1's `attack_per_class_cap` removed entirely — every row passing `CLASS_DATASETS`/`EXCLUDED_CLASSES` is kept in full.
- `benign_ratio` becomes a Step 2 sweep knob (`configs/graph.yaml: sampling.benign_ratio`, candidates 2.0/3.0/4.0) instead of a Step 1 concern.
- New file: `src/trench_ids/vocab.py` (global Protocol/L7_PROTO vocabulary across all 3 retained datasets).

---

## 2026-07-14 — Benchmark finalization (third round)

Four changes made and validated against a full rerun (Step 1 once, Step 2 three times):

1. **Task table rebalanced**: `task_design.py`'s `min_weight_grouping`/`assign_groups` gained an optional size-aware tie-break (`sizes` param) — among the 78 threshold-valid pairings of the 8 non-clique classes, it now minimizes the largest resulting task's size instead of total pairwise similarity. New table hardcoded into `labels.py`: T1 Scanning (isolated), T2 Reconnaissance (isolated), T3 DDoS+Infiltration, T4 DoS+Injection, T5 Password+Bot, T6 XSS+BruteForce. Largest:smallest task-size ratio dropped from 14× to 2.9×.
2. **Corrupted-row filter added**: `preprocess.py`'s `_drop_corrupted_rows` drops any row with NaN/±inf/float32-overflow in any numeric column, run before dedup. Resolved the previously-known inf-value issue — dropped exactly 54 rows, all in task 3, matching the prior known count; a full tensor scan afterward confirmed zero non-finite flow-feature values across all ~213K saved graphs.
3. **Downsampling safety cap added**: `graphs.py`'s `_downsample_tasks` + `configs/*.yaml`'s `sampling.max_task_ratio: 3.0` caps any task at 3× the smallest task's graph count — currently a no-op given the ~2.9× natural ratio.
4. **`benign_ratio` sweep actually produced** (not just picked on paper): three full graph sets built — `data/graphs/` (ratio 3.0, 69,737 graphs), `data/graphs_ratio2/` (ratio 2.0, 78,452 graphs), `data/graphs_ratio4/` (ratio 4.0, 65,378 graphs).

Test count grew 39 → 48 (finalization round: `test_task_design.py` +4, `test_preprocess.py` +3, `test_graphs.py` +2) → 64 (added `test_graph_composition.py`, +16).

**Steps 1 and 2 declared frozen** as of this date — no further data-pipeline redesign unless the professor requests it or a genuine bug is discovered.

---

## 2026-07-15 — GPU-preference established for future compute-heavy work

Confirmed via direct check (`torch.cuda.is_available()` → `False`, 0 devices) that this dev machine has no CUDA device. Recorded as a standing preference: default to `device = "cuda" if torch.cuda.is_available() else "cpu"` for Step 3+ training/embedding code, but noted this doesn't matter for Step 1/2 or I/O-bound diagnostics (confirmed the graph-composition script's ~40 min runtime is ~90% disk I/O, not compute). Saved to `CLAUDE.md` and the auto-memory system.

---

## 2026-07-16 — Diagnostics fixed, documentation overhauled, Step 3 implemented

**Diagnostics**
- Found `data/graphs_ratio4/graph_composition.json` was stale (missing `average_class_proportion`); regenerated via `python -m trench_ids.graph_composition --graphs-dir data/graphs_ratio4`.
- Corrected a stale test-count claim in the docs (was reporting an outdated number); reran the suite and confirmed the real count.

**`project-metrics.md` documentation pass** (in response to review feedback)
- Added §8 (graph composition analysis: exact composition, dominant class, average class proportion, 11×11 co-occurrence matrix, interpretation) and its follow-up note flagging Task 4 as markedly more class-balanced than Tasks 3/5/6 — worth controlling for when comparing per-task continual-learning results later.
- Added a "Benchmark objective" paragraph, a "Benchmark at a glance" summary table, and an ASCII pipeline-overview diagram near the top.
- Added explicit node-type/relation-type lists and a plain-percentage explanation of what `benign_ratio` means (`benign_share = 1/(benign_ratio+1)`) to the Step 2 section.

**Step 3 implemented: relation-specific heterogeneous GNN + attention fusion**
- New module `src/trench_ids/model/`: `relation_conv.py` (`RelationSpecificConv` — one `nn.Linear` per edge type, mean-aggregated, kept separate per relation, never collapsed), `attention_fusion.py` (`SemanticAttention` — HAN-style global per-relation attention), `rhgnn.py` (`NodeFeatureEncoders`, `RelationSpecificLayer`, `RelationSpecificHeteroGNN`, `RelationSpecificOutput`, `to_bidirectional`, `port_bucket`), `__init__.py`.
- New config `configs/model.yaml` (`hidden_dim=64`, `attn_dim=128`, `num_layers=1`, `port_buckets=32`; vocab sizes read from the graph set's `vocab.json` at construction time).
- New `tests/test_model.py` — 16 tests against a synthetic graph mirroring the real schema; full suite grew 64 → 80 passing, ruff clean.
- **Bug found and fixed during integration testing**: raw NetFlow byte/packet-count features are unnormalized and span huge dynamic ranges — an integration test on real saved graphs (`data/graphs_ratio4/task_4_train.pt`) showed gradients ~1e10–1e12 without normalization. Fixed by adding `LayerNorm` before the Flow/Host linear projection in `NodeFeatureEncoders`; re-verified the same test afterward (loss ~1.08, gradients dropped to a stable, trainable range).
- Verified end-to-end against real data: real saved mini-graphs, real global vocab sizes (Protocol=5, Service=216), batched via PyG's `DataLoader`, full forward+backward pass with a classifier head.
- Confirmed Steps 1/2 fully untouched throughout — no changes to `graphs.py`, `preprocess.py`, or `labels.py`; no downsampling logic added anywhere in Step 3.

**Step 3 metrics added to `project-metrics.md` (§13)**
- Architecture size (6 stored relations → 11 after in-memory `ToUndirected`, per-node-type incoming-relation counts, real vocab sizes).
- Parameter counts for the default config and two comparison configs (107,026 params at `hidden_dim=64`, 1 layer).
- Real-data integration numbers (loss, forward+backward wall time, gradient magnitudes before/after the `LayerNorm` fix, untrained attention weights).
- Test suite and design-choice summary.
- Polished per follow-up feedback: added an opening summary sentence, a one-line justification for `ToUndirected`, moved the `LayerNorm` mention into the architecture overview (detail stays in the measurement subsection), added the O(R × H²) parameter-scaling relationship, reworded one sentence to be more formal.

**New documentation structure established**
- Created `flow.md`: the architecture/relationship document — a whole-system ASCII map (showing fan-out, shared inputs, dead-end diagnostics, and the one real feedback loop between `task_design.py` and `labels.py`), a per-stage narrative walkthrough, the full Step 3 architecture diagram, and a status table across every component.
- Created `log.md` (this file): the chronological changelog, split out from `flow.md` so architecture documentation and dated history don't have to compete in the same document.

**Status at end of this entry**: Steps 1–2 frozen (since 2026-07-14); Step 3 implemented and verified; Step 4 (relation-specific memory bank, transferability estimation, relation-aware EWC, classifier) not yet started.

---

## 2026-07-17 — Step 3 revised: reverse edges removed per professor's instruction

The professor flagged that Step 3's `to_bidirectional()`/PyG `ToUndirected()` step (added 2026-07-16 to give every node type multiple incoming relations) violates a real constraint: the six stored relations are semantically one-way (`host--originates-->flow` means "the host generated the flow"), and a synthetic `rev_originates` edge has no such meaning — it would exist purely to make message passing easier, undercutting the relation-specific premise the whole architecture is built on.

- Removed `to_bidirectional()` and the `ToUndirected` import entirely from `src/trench_ids/model/rhgnn.py`. `RelationSpecificHeteroGNN.forward` now consumes `graph.edge_index_dict` exactly as Step 2 stores it — 6 one-directional relations, no transform applied.
- Updated docstrings in `rhgnn.py` and `relation_conv.py` to describe the one-directional design and its consequence: Flow now has exactly 1 incoming relation (`originates`), Host has 2 (`terminates_at`, `communicates_with`), Protocol/Service/Port each have 1. Attention fusion is a genuine multi-relation combination only for Host; every other node type's fusion is a trivial single-relation identity (β=1.0) — an honest property of the directed schema, not a bug.
- Removed `to_bidirectional` from `src/trench_ids/model/__init__.py`'s exports.
- Updated `tests/test_model.py`: deleted the 3 tests specific to `to_bidirectional` (schema check, host-host symmetrization, non-mutation), and updated expected relation sets in the remaining conv/layer/model tests to match the one-directional schema. Suite: 80 → 77 passing, `ruff check` clean.
- Re-measured everything real-data-dependent against the corrected model: parameter count dropped from 107,026 → 86,226 (`hidden_dim=64`, 1 layer — fewer relations means fewer per-relation `nn.Linear` weight matrices, R went from 11 → 6), forward+backward integration check on `data/graphs_ratio4/task_4_train.pt` re-run (loss 1.0663, gradient abs-sum 23.39 post-LayerNorm-fix vs. ~1e12 pre-fix — the LayerNorm fix itself is untouched by this change and remains correct), and the untrained attention-weight table redone per node type.
- Updated `project-metrics.md` §13 (architecture size, parameter count, integration numbers, test count, design choices) and `flow.md` §8 (file table, architecture diagram, design decisions) to match.
