# TRENCH-IDS — Implementation Flow

This is the narrative companion to `project-metrics.md` and `log.md`. **`project-metrics.md` holds every hard number** (row counts, parameter counts, test counts, timings — always cited to a real file on disk). **`log.md` holds the dated changelog** (what changed, when, why). **This file holds the architecture**: what each piece of code does, how the pieces connect right now, which connections are a straight pipeline and which branch, feed back, or run sideways — independent of when each piece was built. Update it whenever a new component is implemented or a connection between existing components changes.

Nothing here is one-way. Some stages fan out (one Step 2 run produces three graph sets), some fan in (three graph sets share one vocabulary), some are dead-end diagnostics (they read pipeline output but nothing reads them back), and one relationship is a genuine divergent feedback loop (§7). The map in §1 shows all of it at once; §2 onward walks each stage in the order it was built.

---

## 1. Whole-system map

```
                    ┌─────────────────────────────────────────┐
                    │   Raw NetFlow v2 CSVs (5 datasets/disk)  │
                    │   ToN-IoT, CSE-CIC-IDS2018, BoT-IoT,     │
                    │   UNSW-NB15 (excluded), UQ-NIDS (excluded)│
                    └───────────────────┬───────────────────────┘
                                        │
                                        v
                    ┌─────────────────────────────────────────┐
                    │           labels.py                       │
                    │  canonical label map, CLASS_DATASETS      │
                    │  (per-class source restriction),          │
                    │  EXCLUDED_CLASSES, CANONICAL_TO_TASK       │
                    │  (T1-T6 hardcoded table — SOURCE OF TRUTH) │
                    └──────┬───────────────────────┬────────────┘
                           │                       │
              (used to validate/                (feeds directly)
               originally derive)                    │
                           │                          v
                           │           ┌─────────────────────────────┐
                           │           │   preprocess.py (Step 1)     │
                           │           │  2-pass stream: count →       │
                           │           │  sample+dedup+corrupted-row   │
                           │           │  filter → stratified split    │
                           │           └──────────────┬────────────────┘
                           │                          │
                           v                          v
              ┌─────────────────────┐   ┌──────────────────────────────┐
              │ similarity.py        │   │  data/processed/               │
              │ task_design.py        │   │  task_{1..6}.parquet + manifest│
              │ (cosine-similarity    │   └──────────────┬────────────────┘
              │ isolate-and-bundle,   │                  │
              │ NOT called at runtime │                  v
              │ — see §7 feedback)    │   ┌──────────────────────────────┐
              └──────────────────────┘   │   graphs.py (Step 2)          │
                                          │  per task: chunk into         │
                                          │  mini-graphs (graph_size=300), │
                                          │  sample benign_ratio,          │
                                          │  apply max_task_ratio cap      │
                                          └──────────────┬────────────────┘
                                                          │
                          ┌────────────────────────────────┼────────────────────────────────┐
                          │                                │                                │
                          v                                v                                v
              ┌───────────────────┐          ┌───────────────────┐          ┌───────────────────┐
              │ data/graphs/        │          │ data/graphs_ratio2/│          │ data/graphs_ratio4/│
              │ benign_ratio=3.0     │          │ benign_ratio=2.0    │          │ benign_ratio=4.0    │
              │ 69,737 mini-graphs   │          │ 78,452 mini-graphs  │          │ 65,378 mini-graphs  │
              └───────────┬─────────┘          └──────────┬──────────┘          └──────────┬──────────┘
                          │                                │                                │
                          └────────────────────────────────┼────────────────────────────────┘
                                                            │
                                       ┌────────────────────┴─────────────────────┐
                                       │        vocab.py (global vocab)            │
                                       │  built once across all 3 datasets,        │
                                       │  shared by all 3 graph sets identically   │
                                       │  (Protocol=5, Service/L7_PROTO=216 rows)   │
                                       └────────────────────┬─────────────────────┘
                                                            │
                          ┌─────────────────────────────────┼─────────────────────────────────┐
                          │                                 │                                 │
                          v                                 v                                 │
              ┌───────────────────────┐        ┌─────────────────────────┐                    │
              │ graph_composition.py    │        │  src/trench_ids/model/    │◄───────────────┘
              │ (diagnostic only —      │        │  Step 3: relation-       │
              │ reads graphs, writes     │        │  specific encoder +      │
              │ analysis JSON/CSV,       │        │  attention fusion         │
              │ nothing downstream        │        │  (any of the 3 graph      │
              │ consumes it — dead end)   │        │  sets can be the input)   │
              └─────────────────────────┘        └────────────┬─────────────┘
                                                                │
                                                    ┌────────────┴─────────────┐
                                                    │                          │
                                                    v                          v
                                        ┌───────────────────────┐  ┌───────────────────────┐
                                        │ output.relations        │  │ output.fused            │
                                        │ (per-relation embeds,    │  │ (final node embedding,   │
                                        │ per-node-type)           │  │ one per node)            │
                                        │                          │  │                          │
                                        │ → Step 4 (not built):     │  │ → Step 4 (not built):     │
                                        │   relation-specific       │  │   classifier head          │
                                        │   memory bank,             │  │                          │
                                        │   transferability          │  │                          │
                                        │   estimation, relation-    │  │                          │
                                        │   aware EWC                │  │                          │
                                        └───────────────────────┘  └───────────────────────┘
```

Reading the branches:
- **1-to-many**: Step 2 runs once per `benign_ratio` candidate, producing three independent graph sets from the same Step 1 output.
- **many-to-1**: all three graph sets share a single global vocabulary (`vocab.py`), built once, not regenerated per set.
- **dead end**: `graph_composition.py` reads a graph set and writes an analysis file; nothing later in the pipeline reads that analysis back in. It exists purely to inform human decisions (documented in `project-metrics.md` §8).
- **any-of-many-to-1**: Step 3's model can be pointed at any one of the three graph sets — nothing about the architecture is set specifically for one `benign_ratio` value. Which set is used for real training is still an open choice (`project-metrics.md` §11).
- **divergent feedback**: `task_design.py` originally *generated* the T1–T6 task table; `labels.py` now hardcodes that table directly. See §7 — this is the one place the "pipeline" isn't really a pipeline anymore.

---

## 2. Datasets and label harmonization — `labels.py`

**What it does**: maps every dataset's raw attack-label strings onto one shared canonical vocabulary (10 kept classes + Benign + 5 excluded classes), restricts each canonical class to the specific source dataset(s) it's allowed to come from (`CLASS_DATASETS`), and hardcodes the final T1–T6 task assignment (`CANONICAL_TO_TASK`).

**Why the per-class dataset restriction matters**: without it, a class like DDoS would be pooled from every dataset that happens to use that label — but BoT-IoT's own DDoS/DoS rows (18.3M / 16.7M) must never leak into the canonical DDoS/DoS classes; BoT-IoT is deliberately restricted to contributing Reconnaissance only.

**Status**: frozen. This is the single source of truth for label/task decisions — any relabeling or task reshuffle changes this file, nothing else.

---

## 3. Task design — `similarity.py`, `task_design.py`

**What it does**: computes a 10×10 cosine-similarity matrix between attack classes' standardized mean feature vectors, then assigns tasks via "isolate-and-bundle" — classes that are all pairwise above the similarity threshold (a conflict clique) each get a singleton task; the rest pair off via minimum-weight matching under a size-aware tie-break (minimizes the largest resulting task rather than total similarity).

**Why similarity-driven**: co-locating similar attacks in one task (e.g. DoS+DDoS together) gives a continual-learning model nothing to transfer — both get learned directly from labels in the same task. Separating similar-but-not-identical classes into different tasks is what creates a genuine transfer/forgetting signal to measure later.

**Status**: the *design* (the table it produced) is frozen and lives in `labels.py`. The *code* itself is not wired into any runtime path anymore — see §7.

---

## 4. Step 1 — raw traffic preprocessing — `preprocess.py`

**What it does**: two-pass streaming pipeline. Pass 1 counts every `CLASS_DATASETS`-restricted attack row plus per-dataset benign rows. Pass 2 samples every attack row in full (no cap), subsamples benign per-dataset, drops corrupted rows (NaN/±inf/float32-overflow, `_drop_corrupted_rows`), deduplicates, then does a per-task stratified 70/15/15 train/val/test split.

**Output**: `data/processed/task_{1..6}.parquet` + `manifest.json` — one row per flow, with a stable `flow_id` (`f"{source_dataset}-{original_csv_row_number}"`) assigned before any concatenation, so flow identity survives from raw CSV through to the Step 2 graph.

**Status**: frozen (2026-07-14). `attack_per_class_cap` was removed entirely in this design round — every row passing the class/dataset restriction is kept.

---

## 5. Step 2 — heterogeneous graph construction — `graphs.py`, `vocab.py`

**What it does**: for each task/split, chunks flows into mini-graphs of at most `graph_size` (300) flows, builds a `HeteroData` object per chunk with 5 node types (Flow, Host, Protocol, Service, Port) and 6 one-directional relations, samples each chunk's benign share to hit a target `sampling.benign_ratio`, and applies `_downsample_tasks` (a `max_task_ratio=3.0` safety cap — currently a no-op, since the task table's natural size ratio is ~2.9×).

**Why it fans out into three sets**: `benign_ratio` (candidates 2.0/3.0/4.0) controls what fraction of each mini-graph is Benign traffic (`benign_share = 1/(benign_ratio+1)`). Rather than pick one value on paper, all three were actually run end-to-end so Step 3+ can be evaluated against each and the best one chosen empirically once real continual-learning metrics exist.

**`vocab.py`**: builds the global Protocol/Service (`L7_PROTO`) vocabulary once, across all 3 retained datasets — not per graph set — so a Protocol/Service value maps to the same embedding row regardless of which `benign_ratio` set or task is being used. Port intentionally has no such vocabulary (chunk-local identity, no cross-task comparability requirement).

**Status**: frozen (2026-07-14). Relation-type strings were renamed for clarity in this round (`host--sends-->flow` → `host--originates-->flow`, etc.) — a pure rename, no schema change.

---

## 6. Diagnostics that read Step 2 output — `graph_composition.py`

**What it does**: for a given graph set, computes per-task class-combination counts, dominant class per graph, mean class share per graph, and an 11×11 cross-task co-occurrence matrix. Purely descriptive — it exists to answer "is graph-level downsampling worth doing, and would it have anything meaningful to select against."

**Where it sits in the flow**: it's a **branch, not a stage** — it reads `data/graphs_ratio4/*.pt`, writes `graph_composition.json`/`graph_composition_matrix.csv`, and nothing downstream (Step 3 or otherwise) reads those files back in. Its value is informing the human decisions documented in `project-metrics.md` §8 and §11 (e.g. the finding that Task 4 is far more class-balanced than Tasks 3/5/6, worth controlling for when comparing per-task CL results later).

**Status**: a standing diagnostic, rerun whenever a graph set changes.

---

## 7. The one real feedback loop: `task_design.py` vs. `labels.py`

This is the one place in the whole project where "pipeline" is the wrong word.

`task_design.py`'s `min_weight_grouping`/`assign_groups` originally *produced* the T1–T6 task table by computing similarity-based pairings. That output was then hand-verified and hardcoded directly into `labels.py`'s `CANONICAL_TO_TASK`, which is what every runtime path (`preprocess.py`, `graphs.py`) actually reads.

Since then, `task_design.py` gained a size-aware tie-break parameter (`sizes=...`) that was used to re-derive the *current* (rebalanced) table — but that parameter was only ever passed manually/interactively when regenerating the table by hand. `task_design.py`'s own CLI entry point (`run()`) still calls the grouping function *without* `sizes`, so running it bare today would reproduce the **old, worse-balanced** pairing (T3=XSS+DDoS at 5.88M rows), not the current one.

**Net effect**: `labels.py` is correct and is the actual source of truth; `task_design.py`'s CLI is stale and would silently regress the design if run without modification. This is tracked as a known follow-up (see `CLAUDE.md`), not a live bug — nothing currently calls `task_design.py --run` in any automated path.

---

## 8. Step 3 — relation-specific heterogeneous GNN + attention fusion — `src/trench_ids/model/`

### 8.1 What it does, in one sentence

Converts Step 2's heterogeneous mini-graphs into relation-aware node embeddings: every node learns one embedding per relation it participates in, and those per-relation embeddings are fused via a trainable attention mechanism into one final embedding per node — both the per-relation embeddings and the fused embedding are retained.

### 8.2 Files

| File | Role |
|---|---|
| `model/rhgnn.py` | `port_bucket`, `NodeFeatureEncoders`, `RelationSpecificLayer`, `RelationSpecificHeteroGNN`, `RelationSpecificOutput` |
| `model/relation_conv.py` | `RelationSpecificConv` — one `nn.Linear` per edge type, mean-aggregated, kept separate per relation |
| `model/attention_fusion.py` | `SemanticAttention` — HAN-style (Wang et al., WWW 2019) global per-relation attention |
| `model/__init__.py` | public exports |
| `configs/model.yaml` | `hidden_dim`, `attn_dim`, `num_layers`, `port_buckets` (vocab sizes read from the graph set's `vocab.json` at construction time, not hardcoded here) |
| `tests/test_model.py` | 13 tests against a synthetic graph mirroring the real 5-node-type/6-relation schema |

### 8.3 Architecture diagram

```
HeteroData mini-graph (Step 2 output, read-only — data/graphs*/task_{t}_{split}.pt)
        │             used exactly as stored: 6 one-directional relations,
        │             no reverse edges added (professor's explicit instruction —
        │             a relation encodes one real-world direction; a synthetic
        │             reverse edge has no such meaning)
        v
┌─────────────────────────── NodeFeatureEncoders ───────────────────────────┐
│  Flow raw features  ──► LayerNorm ──► Linear ──┐                          │
│  Host raw features  ──► LayerNorm ──► Linear ──┤                          │
│  Protocol vocab_id   ──► Embedding (size 5)  ──┼──► x_dict[node_type]       │
│  Service  vocab_id   ──► Embedding (size 216)──┤     all in hidden_dim      │
│  Port     port_number──► log1p-bucket ──► Emb ─┘                          │
└──────────────────────────────────┬──────────────────────────────────────┘
                                    v
        ┌────────────────── RelationSpecificLayer (× num_layers) ──────────────────┐
        │                                                                            │
        │   RelationSpecificConv:                                                    │
        │     for each of the 6 stored edge types (src, relation, dst):               │
        │       message = W_relation · x_dict[src]        ← own weight matrix,        │
        │       aggregated = mean-scatter(message) at dst    never shared across       │
        │                                                     relations               │
        │     -> relations: {node_type: {relation_name: Tensor[N, hidden_dim]}}         │
        │        Flow: 1 incoming (originates)   Host: 2 (terminates_at,                │
        │        Protocol/Service/Port: 1 each      communicates_with)                  │
        │                                                                            │
        │   SemanticAttention (per node type; only Host has >1 relation to fuse —      │
        │   the rest short-circuit to β=1.0, an honest identity, not a bug):           │
        │     w_r    = tanh(W_proj · z_r + b)             shared MLP across relations   │
        │     s_r    = q^T · w_r,  averaged over nodes    one score per relation        │
        │     β_r    = softmax(s_r)                       relation-importance weight    │
        │     fused  = Σ_r β_r · z_r                       one embedding per node        │
        │                                                                            │
        └──────────────────────────────────┬───────────────────────────────────────┘
                                            v
                             RelationSpecificOutput
                   ┌────────────────────────┼─────────────────────────┐
                   v                        v                         v
           .relations                  .fused                   .attention
     {node_type:{relation:      {node_type: embedding}     {node_type:{relation: β}}
        embedding}}             → next layer's input,       → inspectable now,
     → kept for future           or final representation      reusable later for
       memory bank /             if last layer                relation-importance
       transferability                                        estimation
       estimation (Step 4)
```

### 8.4 Design decisions (why, not just what)

- **Relation-specific weights, never shared**: the entire point of the architecture is that a node's Flow→Port relation and Flow→Protocol relation learn *independent* transformations. `RelationSpecificConv` gives each of the 6 stored edge types its own `nn.Linear(hidden_dim, hidden_dim)`; parameter count scales as O(R × H²).
- **No reverse edges — relations stay one-way** (revised 2026-07-17, professor's explicit instruction): Step 2's stored graphs are one-directional (e.g. `flow--terminates_at-->host` only), and each relation encodes a specific real-world direction — `host--originates-->flow` means "the host generated the flow"; a synthetic `rev_originates` edge has no such meaning, it would exist only to make message passing easier. Since the whole point of Step 3 is *relation-specific* representations, preserving each relation's semantics takes priority over giving every node type a rich multi-relation fusion. The consequence — Flow, Protocol, Service, and Port each end up with exactly one incoming relation, so their attention fusion is a trivial identity — is accepted as an honest property of the directed schema, not engineered around with artificial edges.
- **HAN-style semantic attention over GAT-style node-conditioned attention**: produces one importance weight per *relation* (not per node), which is exactly the granularity the later transferability-estimation and relation-aware-EWC steps need — reusable directly rather than needing to be summarized from a node-level map.
- **LayerNorm before the Flow/Host linear projection**: raw NetFlow byte/packet-count features span huge dynamic ranges; without normalization, an integration test on real saved graphs showed gradients around 1e12. Added `LayerNorm`, confirmed the fix reduced gradients to a stable, trainable range (`project-metrics.md` §13.3).
- **Log-bucketed Port embedding instead of a raw 65536-row table**: Port has no global vocabulary by Step 2's design (chunk-local identity), so a raw-port embedding would carry all the parameter cost with no generalization benefit across mini-graphs.

### 8.5 What it hands off to Step 4 (not yet built)

- `output.relations` → intended input to a future relation-specific memory bank (per-attack-class mean per-relation embedding) and transferability estimation (cosine similarity between new and stored per-relation means).
- `output.attention` (the β weights) → intended input to a future lightweight MLP that turns transferability scores into relation-importance weights, which in turn will selectively regularize a relation-aware EWC.
- `output.fused` → intended input to a classifier head (not yet built).

None of this is implemented yet — Step 3 only produces and exposes the embeddings; nothing currently consumes `output.relations` or `output.attention` downstream.

---

## 9. Status summary

| Stage | Status | Frozen since |
|---|---|---|
| Label harmonization (`labels.py`) | Implemented, frozen | 2026-07-14 (task table finalization) |
| Task design (`similarity.py`, `task_design.py`) | Design frozen; code has a known stale-CLI issue (§7) | 2026-07-14 |
| Step 1 preprocessing (`preprocess.py`) | Implemented, frozen | 2026-07-14 |
| Step 2 graph construction (`graphs.py`, `vocab.py`) | Implemented, frozen | 2026-07-14 |
| Graph composition diagnostic (`graph_composition.py`) | Implemented, standing (rerun as needed) | — |
| Step 3 relation-specific encoder + attention fusion (`model/`) | Implemented, verified | 2026-07-16 |
| Step 4 memory bank / transferability estimation / relation-aware EWC | Not started | — |

See `project-metrics.md` for every number backing the claims above (row counts, similarity scores, graph statistics, parameter counts, test results, real-data integration measurements).
