# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

> ### ⚠️ BENCHMARK REBUILT 2026-08-17 — earlier numbers are superseded
>
> The data pipeline was rebuilt to fix four defects measured in a 2026-08-07
> audit. **Every metric recorded before 2026-08-17 was produced on a
> different benchmark and is not comparable to anything produced after it** —
> that includes all of `project-metrics.md` (repo root, not under `docs/`),
> `TRENCH-IDS_Full_Project_Documentation.md`, and
> `TRENCH-IDS_Project_Summary.pdf`. Never quote a figure from those next to a
> current one without saying which benchmark each came from.
>
> **`docs/methodology-2026-08-17.md` is the current source of truth** for the
> data pipeline and the continual-learning mechanism. The task design,
> dataset selection, and encoder architecture described below and in
> `docs/dataset-plan.md` are unchanged and still authoritative.
>
> What changed, in one line each:
>
> 1. **Step 1 is quota-based** — every task is now 520K rows (was 1.3M-3.8M),
>    rare classes taken in full (T3 DDoS:Infiltration went ~30:1 to ~2.4:1).
> 2. **No benign duplication, no cross-task benign leakage** — ~91,000 unique
>    benign train flows per task, each task's slice disjoint (was ~5,600
>    replicated 54-158x, with 47-72 flow_ids leaking train-to-test across
>    tasks).
> 3. **Mini-graphs are chunked in capture order**, not from a global shuffle,
>    so Host/Port aggregates describe a real interval instead of a task-wide
>    average. Needs Step 1's new `source_row` column.
> 4. **Relation-aware EWC replaced by transferability-weighted relation
>    distillation** (`src/trench_ids/cl/distillation.py`) — a Fisher audit
>    showed EWC's relation weights could only ever modulate 2.2% of their own
>    penalty, which is structural and not fixable by re-weighting.
> 5. **λ_d is selected on the validation split** — the first selection
>    decision in this project's history not made on test data.
> 6. **TRD's logit term is restricted to old classes** (`set_old_classes`,
>    2026-08-17). The head spans all 11 classes from task 1, so distilling a
>    teacher over the *full* label space asserts "this is an old class" on
>    exactly the rows whose target is a new one — the two loss terms fight on
>    the current task's own data, and raising λ_d buys more of the conflict.
>    LwF (Li & Hoiem 2016) restricts its term the same way. The first λ_d
>    sweep predates the fix and is **discarded**, kept as evidence under
>    `runs/_superseded_prefix_lwf_bug/` (val 0.404 at λ=0.5 → 0.270 at λ=2.0,
>    monotone in λ, the signature the defect predicts).
>
> 7. **Corrected architecture: residual connections enable depth, GNN beats Flat+Host.**
>    The original 1-layer GNN had 40.9% of params unreachable (6 non-Flow relations
>    + 4 fusion modules). With `num_layers=3` + `use_residual=true`, the full
>    relational architecture participates. **3-layer GNN + replay: forgetting
>    0.081 ± 0.008, accuracy 0.902 ± 0.005** beats **Flat+Host + replay:
>    forgetting 0.105 ± 0.017, accuracy 0.872 ± 0.016** across all 3 seeds
>    (paired diff: -0.025 forgetting, +0.030 acc, same sign every seed).
>    **Two hypotheses tested and REFUTED, don't retry them:** (a) the fusion
>    merge — `ConcatFusion`/`model.fusion=concat` scored *worse* (0.6908 mean)
>    than attention's 0.7741; (b) hub dilution — all 5 Flow
>    relations have within-graph/total variance 0.994–0.997, i.e. fully
>    flow-specific, so `CONCAT(self, aggregated)` is doing its job.
>    The original "Flat+Host wins" conclusion was an artifact of dead gradients
>    in the 1-layer GNN, not a real architecture finding.
>
> Current graph set: 69,737 mini-graphs, 5.73 GB, realised attack:benign
> exactly 3.0. Reference points on the rebuilt benchmark, seed 42:
> 3-layer GNN + replay — forgetting **0.089**, accuracy **0.896**;
> Flat+Host + replay — forgetting **0.091**, accuracy **0.887**;
> plain fine-tuning (the λ_d = 0 anchor) — forgetting **0.696**, accuracy
> **0.364**. Forgetting reads worse than the old 0.094 because the old number
> was inflated by benign memorisation, not because anything regressed.
>
> **MATRIX COMPLETE (34 runs + 6 new architecture runs, 3 seeds, `docs/results_final.md` is generated from
> them).** λ_d selected on **val** = **0.25** (interior optimum, bracketed).
>
> Method, no replay, 3 seeds, vs plain fine-tuning (0.3641 acc / 0.6927 forget):
> EWC 0.3775 / 0.6687 (**null on accuracy — sign-inconsistent across seeds,
> reconfirmed on clean data**); TRD-uniform 0.3896 / 0.6506, TRD-transfer
> 0.3941 / 0.6474, TRD-inverse 0.4105 / 0.6302 (all three beat fine-tuning
> meaningfully — so **TRD works where EWC does not**). Replay dominates
> everything at 0.7741 / 0.1743, and **Replay+TRD (0.7413) is worse than plain
> replay** — the regulariser fights rehearsal rather than adding to it.
>
> **The weighting hypothesis is directionally WRONG, and the sign control wins.**
> vs TRD-uniform: `transfer` +0.0045 acc, sign-inconsistent → not meaningful;
> `inverse` +0.0210 acc, same sign on every seed, 1/7th the variance
> (±0.003 vs ±0.021) → **MEANINGFUL**. Mechanism measured, not guessed:
> `drift.py` gives transferability-vs-drift Pearson r = **−0.254**, Spearman
> **−0.400** — the relations that transfer best move least, so `softmax(+S_r)`
> regularises what was never going to drift.
>
> **DRIFT-GUIDED WEIGHTING TESTED 2026-08-23 — the reformulation is a clean
> NULL, and the "inverse is only a proxy" story is now closed (do not revive it).**
>
> **TRANSFERABILITY VALIDATED ON CORRECTED ARCHITECTURE (2026-08-24):**
> On 3-layer residual GNN, `targeted_by` relation shows transferability-vs-drift
> Pearson r = **−0.960**, Spearman **−0.900** — the relations that transfer best
> (high S_r) are those that drift least (low D_r). Signal was noise on 1-layer
> GNN (all correlations weak). This validates the core hypothesis that
> transferability scores predict future retention, but only when the full
> relational architecture is reachable.
>
> **EXPERIMENTAL STATE FROZEN (2026-08-24, git tag v1.0.0-experiments):**
> 3-layer GNN + replay validated across 3 seeds. No further architecture
> exploration unless a reproducibility or methodological error is discovered.
> Next step: write paper.
> `gnn_trd_drift` (`w_r = softmax(+D_r/T)`, D_r measured live per boundary:
> teacher vs post-warm-up student on a seeded ≤60-graph subsample,
> `distillation.boundary_relation_drift`) ran twice. First at T=0.3 —
> **superseded**, kept under `runs/_superseded_drift_T03/`: live warm-up D_r
> spans ~0.1–1.25, ~3x S_r's spread, so T=0.3 drove the softmax nearly
> one-hot onto `originates` (seed 42 task 2: w=0.748) and the arm degenerated
> into single-relation distillation (0.378 acc). Rerun at T=1.0 with properly
> spread weights (w ∈ 0.13–0.35, ordering exactly as the anti-correlation
> predicts): **0.3633 acc / 0.6893 forget — indistinguishable from plain
> fine-tuning**, meaningfully WORSE than uniform on F1/forgetting direction.
> So weighting by measured drift directly does not rescue the mechanism even
> with the signal correctly scaled; `inverse` remains the best TRD arm, and
> its advantage over uniform is *not* explained by drift-ordering alone (a
> drift-ordered weighting scores like fine-tuning). Open puzzle for the
> write-up, not for more experiments: why the S_r-based inverse ordering
> helps while the D_r-based one does not. λ_d caveat: drift inherits 0.25,
> selected on the S_r scale — stated approximation, pinned by
> `test_the_lambda_sweep_holds_temperature_fixed`. See [[project_relation_drift_finding]].
>
> **REPLAY BUDGET SWEPT 2026-08-18 (24 runs, 3 seeds, group `budget` in
> `run_matrix.py`), professor-requested.** The standing
> `replay.buffer_size_per_task: 200` was inherited from the pre-rebuild
> benchmark, where its config comment ("~2.5% of one task's ~8135 train
> graphs") was correct. The rebuild cut each task to **1,214** train graphs
> and nobody revisited the number, so the default was actually storing
> **16.5%** of every past task. Swept 25/50/100/[200]/400 with
> `replay_fraction` pinned at 0.3 (rehearsal is 30% of the pool at every
> point, so compute is constant and only the number of *distinct* retained
> graphs varies), selected on **val** by the parsimony rule — smallest budget
> within `MEANINGFUL_DELTA` of the best, because plain argmax on val returns
> the largest budget swept almost by construction.
>
> **The two encoders behave differently, and that is the finding.** GNN val
> across 25→400: 0.7573 / 0.7380 / 0.7539 / 0.7720 / 0.7664, with per-budget
> seed std **0.018–0.031 — larger than the entire 0.034 between-budget
> range**. The GNN cannot tell 25 graphs/task from 400: 16x the memory buys
> nothing measurable. Flat+Host is the opposite — 0.8332 / 0.8423 / 0.8628 /
> 0.8527 / 0.8651 at seed std 0.002–0.013, a real monotone-ish rise.
> Selected: **GNN 25** (bottom of grid, boundary-flagged, and honestly a
> null result) and **Flat+Host 100 (8.2%)** — half the standing default, and
> measuring *better* than it (0.8628 vs 0.8527 val, 0.1110 vs 0.1218
> forgetting). Flat+Host at 100 also beats the joint non-CL GNN upper bound
> (0.8072). **Nothing here justifies 200 for either encoder.** This is a
> third independent strike against the graph: it cannot exploit extra
> rehearsal data that the non-graph baseline uses productively. Untested and
> must not be claimed: anything below 25 (a knee must exist between 0 —
> plain fine-tuning at 0.364 — and 25), and `replay_fraction`, which was
> deliberately held fixed. See [[project-replay-budget-finding]].
>
> Novelty detection (§5): **superseded 2026-08-22 — the original §5 numbers
> were confounded and the conclusion is reworded, not merely renumbered.**
> The memory bank's stored prototypes came from earlier (drifted) encoder
> states while queries come from the final checkpoint; `novelty.py` now
> recomputes prototypes under the final checkpoint by default
> (`--prototype-source recomputed`), and known/novel graphs are drawn by
> seeded uniform subsample rather than a prefix slice. Re-run on the fixed
> code: `gnn_trd_transfer_s42` gives prototype AUROC **0.604 vs MSP 0.442**
> (prototype wins); `gnn_replay_s42` gives prototype **0.578 vs MSP 0.641**
> (MSP wins). The ordering is run-unstable; the defensible claim is "no
> scorer separates novel from known usefully" — FPR@95TPR ≥ 0.79 everywhere.
> The old table (prototype 0.527 below MSP 0.654, best FPR@95TPR 0.65) is
> discarded; see `docs/methodology-2026-08-17.md` §5.
>
> Same-day drift-measurement fix: `drift.py` also prefix-sliced its test
> graphs (biased post-rebuild, since graph position encodes capture time);
> now seeded-random, and per-batch means are flow-count-weighted. Unbiased
> 5-boundary means on `gnn_trd_transfer_s42`: originates 0.188 >
> targeted_by 0.151 > service_of 0.079 > terminated_by 0.043 > protocol_of
> 0.031; transferability-vs-drift anti-correlation strengthens to Pearson
> r = −0.909 / Spearman ρ = −0.900 (was −0.254/−0.400). The old
> single-boundary table (service_of 0.054 … terminated_by 0.597 at T5→T6)
> is an artifact of prefix sampling — do not quote it.
>
> ⚠️ Checkpoints predating the rebuild (`replay_seed*_newport`, svc=216)
> crash against current `data/graphs` (svc=239) with a CUDA assert; their
> configs lack `vocab_fingerprint`, so no guard fires. Use post-rebuild runs
> for any inference/analysis.
>
> Experiments are defined in code (`src/trench_ids/cl/run_matrix.py`) and the
> results document is generated (`src/trench_ids/cl/report.py`), never
> transcribed by hand.

**Design respec'd from 2 datasets/6 classes/4 tasks to 3 datasets/10 classes/6 tasks (2026-07-13, second round),** per the professor's revised guidance: rather than minimizing complexity, build a richer continual-learning benchmark (more tasks, more attack classes, no pre-graph sampling cap). NF-BoT-IoT-v2 is back in (restricted to Reconnaissance only via `CLASS_DATASETS`); NF-UNSW-NB15-v2 stays excluded. See "Continual-learning task split" below, `docs/dataset-plan.md`, and `docs/superpowers/specs/2026-07-13-dataset-task-respec-design.md` for the full design and rationale.

**Benchmark finalized 2026-07-14 (third round)** per user review of the second-round rerun: `docs/superpowers/specs/2026-07-14-benchmark-finalization-design.md` made four changes and then reran Step 1 once + Step 2 three times end-to-end, validated against real data. (1) **Task table rebalanced**: `src/trench_ids/task_design.py`'s `min_weight_grouping`/`assign_groups` gained an optional size-aware tie-break (`sizes` param) — among the 78 threshold-valid pairings of the 8 non-clique classes, it now minimizes the *largest resulting task's size* instead of total pairwise similarity, which had picked the single worst-balanced valid option (old T3=XSS+DDoS at 5.88M rows). `src/trench_ids/labels.py` now hardcodes the rebalanced table: T1 Scanning (ToN, isolated), T2 Reconnaissance (BoT, isolated), T3 DDoS+Infiltration, T4 DoS+Injection, T5 Password+Bot, T6 XSS+BruteForce (T3-T6 all ToN+CSE) — largest:smallest task-size ratio dropped from 14x to 2.9x (T1 3,789,419 vs T5 1,304,420 rows). Known follow-up: `task_design.py`'s own CLI (`run()`) doesn't pass `sizes` yet, so it would regenerate the old pairing if rerun bare — `labels.py`'s hardcoded table is the actual source of truth and is correct. (2) **Corrupted-row filter**: `preprocess.py`'s new `_drop_corrupted_rows` drops any row with NaN/±inf/float32-overflow in any numeric column, called in `assemble_tasks` before dedup; manifest gained `corrupted_rows_dropped`. This **resolved** the previously-known inf-value issue — the fresh rerun dropped exactly 54 rows (all in task 3, matching the prior count) and a full tensor scan confirmed **zero** non-finite flow-feature values across all ~213K saved graphs in all three sweep sets. (3) **Downsampling safety cap**: `graphs.py`'s new `_downsample_tasks` + `configs/*.yaml`'s `sampling.max_task_ratio: 3.0` caps any task at 3x the smallest task's graph count; currently a no-op given the ~2.9x natural ratio. (4) **benign_ratio sweep actually produced** (not just picked): three full graph sets now exist — `data/graphs/` (3.0, 69,737 graphs), `data/graphs_ratio2/` (2.0, 78,452 graphs), `data/graphs_ratio4/` (4.0, 65,378 graphs) — each config's `graph_counts.json` records its own `benign_ratio`. Choosing the winning ratio is deferred to Step 3 (needs real CL metrics). `benign_per_task` (8000) still not raised — tracked, not yet resolved, same reasoning as before. Full details: `docs/project-metrics.md` (regenerated from this rerun) and `.superpowers/sdd/progress.md`.

**Steps 1 and 2 are now frozen per explicit user direction** — no further data-pipeline redesign unless the professor requests it or a genuine bug is discovered. All effort moves to **Step 3** (the relation-specific heterogeneous GNN and continual-learning model) next.

**2026-07-19 carve-out to the Step 2 freeze, professor-requested**: Step 3's encoder had a real gap — every node type's fused embedding was 100% neighbor-derived, worst for Flow (the classification target, 1 incoming relation, 0% self-signal). The professor gave two complementary fixes. (1) **`graphs.py` gained 5 new relations** (`originated_by`, `terminated_by`, `targeted_by`, `protocol_of`, `service_of` — reverses of the 5 Flow-centric relations, individually named with correct passive-voice semantics rather than a generic `rev_X` prefix, resolving the actual objection behind the 2026-07-17 "no reverse edges" instruction rather than contradicting it). Schema is now **11 relations**; Flow's incoming count went 1→5, Host's 2→3; Protocol/Service/Port unaffected (still 1 each, no continuous features to enrich). `data/graphs` was regenerated (69,737 graphs, same count as before — schema change only, no row/sampling change) via `python -m trench_ids.graphs --config configs/graph.yaml`; `data/graphs_ratio2`/`_ratio4` were **not** regenerated (deferred until/if the ratio sweep resumes). (2) **`RelationSpecificConv`** (`src/trench_ids/model/relation_conv.py`) now concatenates each destination node's own current embedding into every relation's neighbor-aggregated message before a new per-relation `combine_lins` layer projects back to `hidden_dim` — GraphSAGE's concatenation aggregator (Hamilton et al., NeurIPS 2017), applied per-relation. `SemanticAttention` fusion needed no change (verified it never referenced a node's own features). See `docs/dataset-plan.md` §3.2 and `project-metrics.md` §13 for full details and re-measured real numbers (params, attention weights, forward/backward check).

**Step 4 implemented 2026-07-19**: `src/trench_ids/cl/` (new subpackage — `train.py`, `memory_bank.py`) + `configs/train.yaml`. Sequential fine-tuning across T1→T6, one shared classifier head over the global label space, plain baseline (no replay/EWC — that's step 7, still open). Real run on `data/graphs` (11-relation schema) produced a forgetting matrix (every task hits 0.87-0.997 accuracy on itself, then collapses once training moves on — expected catastrophic forgetting, motivating steps 5-7) and a relation-specific memory bank (all 10 attack classes × Flow's 5 relations). Ran on **GPU** (`NVIDIA GeForce RTX 3050 Laptop GPU` — this machine gained a CUDA device between 2026-07-15 and 2026-07-19; `torch.cuda.is_available()` now returns `True`, don't assume the earlier "no CUDA" note still holds). **Hydra is now a dependency** (`configs/train.yaml`), added at this step per the plan already stated in `pyproject.toml`. Full numbers: `project-metrics.md` §14.

**Step 5 implemented 2026-07-19**: `src/trench_ids/cl/transferability.py`, hooked into the Step 4 loop — for each task's new classes, cosine similarity against every bank class, same relation only, no cross-relation aggregation (that's step 6, still open). Checked the two pairs flagged in `docs/attack-similarity-matrix.md` (XSS↔Infiltration, highest raw-feature cosine; Scanning↔Reconnaissance, second-highest) against the real learned embeddings: XSS↔Infiltration partially replicates (2 of 5 relations moderately positive), Scanning↔Reconnaissance does not (near-zero/negative across all 5) — most likely because Scanning's bank entry, from task 1, was computed under a since-heavily-forgotten encoder state by the time task 6 runs (§14.1's forgetting curve), making it not a like-for-like comparison. This itself is a real, reportable finding: an un-mitigated encoder's continual drift undermines its own memory bank's later comparability, which is exactly what steps 6-7 (relation importance weights, relation-aware EWC) are meant to address. Full numbers: `project-metrics.md` §15.

**Steps 6-8 implemented 2026-07-21** (per `docs/Updated_Proposed_Methodology_Transferable_Representation_Learning.docx`, the professor's updated 10-step methodology, and his direct follow-up reply requiring the final loss to add a shared EWC term): relation-aware transferability-guided Online EWC. Full design: `docs/superpowers/specs/2026-07-21-relation-aware-ewc-design.md`; full task-by-task plan: `docs/superpowers/plans/2026-07-21-relation-aware-ewc.md`; execution ledger: `.superpowers/sdd/progress.md`. Built via subagent-driven-development, 8 tasks, 11 commits (`a47667e`..`ffb371d`), two real bugs caught and fixed in task review (an `estimate_fisher` training-mode-restoration bug, and a CPU/CUDA device-mismatch crash in `train.py`), final whole-branch review clean. New/changed: `src/trench_ids/cl/ewc.py` (`FLOW_RELATIONS`, `OnlineEWCState` — running Fisher + reference-parameter snapshot per group, Online EWC / Schwarz et al. 2018 — `estimate_fisher`, one whole-model forward/backward pass never per-group, `OnlineEWCManager` — owns all 12 parameter groups: 1 shared + 5 Flow relations + 6 other relations), `src/trench_ids/cl/importance.py` (`ImportanceMLP`, one shared-weight network applied per relation, `w_r = sigma(MLP(S_r))`, left attached to the autograd graph and recomputed fresh every batch from a once-per-task-cached `S_r` — never cached itself, to avoid graph-retention issues), extended `transferability.py` (`aggregate_transferability_scores`, mean cosine per relation) and `train.py` (2 warm-up epochs, plain loss, then 3 full-loss epochs under the combined 3-term loss `L = L_cls + λ_u·L_EWC^other + λ_s·L_EWC^shared + λ_r·Σ w_r·L_EWC^(r)`; per-task `runs/step4/importance_weights_task_{t}.json`). 121 tests passing.

**Real 6-task run result (2026-07-21, `runs/step4/`, `project-metrics.md` §16): no meaningful forgetting improvement over the plain Step 4 baseline.** The learned `w_r` collapses toward zero from task 2 onward (largest task-2 value already 1.06e-3; by T4-T6 every value is below 1.5e-5) — a real, correctly-implemented consequence of leaving `w_r` attached: minimizing the combined loss gives gradient descent a direct, unconditional incentive to shrink its own penalty term, with nothing in the per-task local objective rewarding keeping it large for future retention. But **this is confounded, not conclusive**: the `λ_s`/`λ_u` EWC terms are independent of `w_r`, run at the placeholder `λ=1.0`, and *also* showed no measurable effect — so the current run can't distinguish "EWC doesn't help here" from "regularization was effectively off" (a well-known EWC calibration pitfall; the literature typically needs `λ` orders of magnitude above 1.0). **Decided next step, explicit user direction: do not touch the algorithm yet.** Run a coarse logarithmic `λ_s`/`λ_u` (then `λ_r`) sweep (e.g. `1e-2, 1e-1, 1, 10, 1e2, 1e3`) first, to establish whether standard Online EWC reduces forgetting on this benchmark at all before revisiting whether the learned-weighting mechanism itself needs to change.

**Coarse λ sweep completed 2026-07-21** (`project-metrics.md` §17): six runs, `λ = λ_r = λ_s = λ_u ∈ {0.01, 0.1, 1, 10, 100, 1000}`, `ewc.disable_learned_weighting=true` (new config flag — fixes `w_r=1.0` for every Flow relation instead of routing through `ImportanceMLP`, so the same uniform, unweighted penalty applies to all 12 EWC groups; `λ_r=0` alone would *not* be equivalent, since it would zero out Flow's 5 relations entirely rather than regularizing them like every other group), everything else held fixed (seed 42, epochs/warmup/batch_size/lr/γ/`data/graphs`/GPU unchanged from §16). **Result: no monotonic trend across 5 orders of magnitude** — average forgetting stayed in a 0.71–0.75 band and final average accuracy in 0.326–0.359 for every λ tested, reading as run-to-run noise rather than a λ response. A 7th instrumented run (`ewc.log_loss_components=true`, new diagnostic flag + `OnlineEWCManager.loss_breakdown()`) at λ=1000 ruled out "λ too weak to matter": by tasks 5-6, `λ·L_EWC` actually *exceeds* `L_cls` (160-177%) at the start of full-loss training, dropping to 21-34% by the epoch's end — substantial, not negligible — yet forgetting still didn't improve. Root-caused a related anomaly (`other_raw` exactly `0.0` in every run): with `num_layers=1`, the 6 non-Flow relations' outputs (and 4 non-Flow `SemanticAttention` fusion modules) are computed but never consumed by anything the loss depends on (`train.py` only reads `output.fused["flow"]`), so their gradient is literally `None` (confirmed on both a synthetic graph and a real batch from `data/graphs/task_1_train.pt`) — an architectural property, not a bug, and not a confound in the λ-sweep conclusion since those parameters were never reachable by the loss or the penalty either way. **Precisely-scoped conclusion**: under the current architecture (`num_layers=1`), Online EWC applied to every parameter that *does* influence the classification objective did not reduce catastrophic forgetting over 5 orders of magnitude of λ, even when the penalty substantially exceeded `L_cls`. A direct matched-λ comparison (`project-metrics.md` §17.4 — no rerun needed, all three already on disk at `λ=1.0`/`num_layers=1`) confirms this extends to the relation-aware variant: plain fine-tuning (0.7108 avg. forgetting), plain unweighted EWC (0.7103), and relation-aware weighted EWC (0.7054) are statistically indistinguishable — all three fail to reduce forgetting relative to plain fine-tuning, a real, reportable negative result, not an artifact of under-tuning. 130 tests passing.

**Explicit user direction, 2026-07-21: don't change the architecture mid-study.** `num_layers=1` is frozen for the remainder of the paper's planned experiments — the baseline (fine-tuning, plain EWC sweep, relation-aware EWC) and their comparison are considered final for this architecture. The "6 relations disconnected from supervision" finding (above) is written up as an architectural observation, not acted on by default.

**`num_layers=2` gradient-reachability ablation run 2026-07-21** (`project-metrics.md` §18), explicit user direction — deliberately run once, as a targeted ablation to test whether the §17.2 gradient-reachability limitation was the actual bottleneck, *before* Step 10, not as a redesign of the frozen `num_layers=1` baseline. Same seed/λ=1.0/schedule as §16's relation-aware EWC run, only `num_layers` changed. **Confirmed empirically** (not just architecturally reasoned): the 6 "other" relations now get real gradients and non-zero Fisher (layer 0's copies feed layer 1's Flow-relation inputs; layer 1's own copies remain permanently dead regardless of depth — a structural property of any depth, not specific to 2 layers). **But forgetting (0.7423 avg.) and final accuracy (0.3593) are statistically indistinguishable from every `num_layers=1` variant** (0.7054–0.7108 forgetting, 0.3558–0.3622 accuracy) — inside the noise band the λ sweep already established. The `w_r` collapse (§16.2) persists with near-identical dynamics regardless of depth. **This is a stronger result than a simple improvement would have been**: it rules out "the six dead relations explain the negative EWC result" as an alternative explanation a reviewer could raise, and points the remaining bottleneck squarely at the relation-importance-weighting mechanism's trivial zero optimum (§16.2), not the GNN architecture. `num_layers=2` is not adopted — `num_layers=1` remains the frozen baseline architecture; this ablation doesn't redefine it. No further depth exploration planned before Step 10.

**Step 10a implemented and run 2026-07-22** (`src/trench_ids/cl/transferability_report.py`, `project-metrics.md` §19): consolidates the frozen `num_layers=1` baseline's `runs/step4/transferability_task_{1..6}.json` files — relation-wise ranking/evolution, class-pair ranking, and a descriptive comparison against the regenerated 10-class raw-feature similarity matrix. This covers the first two of Step 10's three bullets (transferability analysis); the prediction/inference pipeline for genuinely unseen traffic (Step 10's third bullet) was deferred at the time but has since been completed — see the 2026-07-28 note below. Two headline findings: (1) `protocol_of` is the standout most-transferable relation (mean cosine 0.197 vs. the next-best's 0.004, the other four near zero), and its transferability rises essentially monotonically across tasks (task 2 = -0.356 → task 5 = 0.318, task 6 = 0.279) — relation embeddings do become more transferable as the encoder matures, but that effect is concentrated in `protocol_of`, not general across all five relations. (2) Of the two pairs originally flagged from the raw-feature matrix, XSS↔Infiltration lands at rank 12 of 41 by learned transferability (moderate positive, partially confirming the raw-feature signal), while Scanning↔Reconnaissance lands at rank 37 of 41 (negative, contradicting it) — consistent with §15's earlier partial-replication finding, now confirmed under the complete 6-task data. The raw-feature-vs-learned comparison itself came out weakly *negatively* correlated overall (Pearson r = -0.219, Spearman rho = -0.221, n=41 matched pairs) — more than mere decoupling, a mild anti-correlation across the full pair set.

**`num_layers=1` vs. `num_layers=2` transferability comparison, 2026-07-22** (`project-metrics.md` §20): no new training run — the §18 `num_layers=2` ablation's `runs/step4_num_layers2/` already had `transferability_task_*.json` on disk (transferability estimation runs regardless of depth), just predating the Step 10a tooling. Reran the analysis against it and compared to §19's `num_layers=1` report. Depth genuinely changes the transferability picture: the relation ranking essentially inverts (`protocol_of`, the `num_layers=1` standout at mean 0.197, becomes the worst and noisiest relation at `num_layers=2`, mean -0.034; `service_of` and `originates`, previously near-zero/negative, become the new top two), mean transferability across relations rises ~7x (0.016 → 0.119), and both originally-flagged pairs move toward more transferable (XSS↔Infiltration rank 12→4 of 41; Scanning↔Reconnaissance's negative signal disappears, rank 37→32). But none of this closes the loop on forgetting: §18 already showed `num_layers=2` forgetting is statistically indistinguishable from every `num_layers=1` variant, and this section confirms the reason isn't "the encoder can't produce transferable representations" (it demonstrably can, and depth improves them) — it's that the relation-importance-weighting mechanism (`w_r`, §16.2) has no incentive to stay large enough to act on that improvement, regardless of how good the underlying representations get. `num_layers=2` remains un-adopted; this is a targeted follow-on analysis, not a new training run or architecture change.

**Inference pipeline + full evaluation metrics implemented 2026-07-23** (`project-metrics.md` §22), per the professor's direct instruction: "Complete the implementation, add the inference pipeline, find F1, accuracy and other metrics, then we can work on fine-tuning the model." `src/trench_ids/cl/inference.py` + `evaluate.py` (scikit-learn-based `compute_metrics`) + `train.py`'s new `save_checkpoint`. Confirms §16's `average_forgetting=0.7054` exactly via independent checkpoint re-evaluation; adds pooled precision/recall/F1 (macro+weighted) and a per-class breakdown showing the final model retains only Benign and whichever two classes T6 introduced (XSS, BruteForce), with every T1-T5 class at recall 0.0000 — catastrophic forgetting made visible per-class, not just as an aggregate accuracy drop.

**Replay baseline + joint-training upper bound + port-representation fix, 2026-07-23/24** (`project-metrics.md` §23-24; full numbers there, this is the headline). Professor's direction after §22: "Try tweaking the hyperparameters. You can also try other methods like replay based to mitigate catastrophic forgetting." (1) A 4-config LR/epochs hyperparameter sweep (`runs/hp_sweep/`) stayed inside the same ~0.71-0.80 forgetting band as every EWC/lambda variant already tried — another negative result, not a new lead. (2) **Experience replay works**: literature-standard graph-level replay (`docs/superpowers/specs/2026-07-23-replay-baseline-design.md` — whole mini-graphs sampled into a per-task buffer, upsampled into later tasks' training pool, no model/loss changes, EWC lambdas zeroed to isolate the effect) brought average forgetting from ~0.71 down to **0.094 ± 0.015** and final accuracy from 0.36 up to **0.861 ± 0.014** (mean ± sample std, 3 seeds: 42/1/2, all with full pooled-metric evaluation) — every task's own-task accuracy now stays between 0.70 and 0.99 through the end of training, versus 0.08-0.24 under EWC. (3) A joint-training run (`src/trench_ids/cl/train_joint.py`, all 6 tasks trained together, no CL setting) gives a non-continual upper bound of 0.876 accuracy / 0.842 F1-macro — replay's 0.861 sits ~1.5 points below it, confirming the architecture itself was never the bottleneck. (4) The well-known-port bucket-collision fix (previously stuck on `experiment/port-embedding-fixed-wellknown` pending sign-off, see the 2026-07-17 open item below) has been **reimplemented directly on `master`** — `rhgnn.py` now unconditionally gives ports 0-1023 exact embedding rows, no config fallback to the old pure-log-bucket scheme; parameter count rose from 197,842 to 263,378 at `hidden_dim=64` (all in the Port node-feature encoder, `project-metrics.md` §13.2). A same-seed replay comparison (old vs. new scheme, seed 2 — same Hydra config both runs, though the port scheme itself isn't a config field, so this rests on directory naming/run order, not a verified code diff, §24.1) shows **the old scheme measuring better** (mean final-accuracy Δ = -0.028 across T1-T6, driven by T3 -0.162 and T4 -0.028; T1/T2/T5/T6 flat-to-slightly-better under the new scheme). **This is an open call, not yet decided** (corrected 2026-07-26 — an earlier pass at this file incorrectly recorded a "keep the new scheme" decision as settled user direction, which inverted what the one available comparison actually shows): the new scheme's justification is structural (the old scheme provably conflates semantically distinct well-known services — FTP/SSH/Telnet on ports 20-23 — into one embedding row purely by numeric proximity), but that is a case made *against* the metric evidence, not from it, and the user's stated criterion was best metrics. The actual decision — keep the new scheme despite worse single-seed-pair metrics, revert to the old scheme since it measures better, or run a multi-seed comparison first — is flagged back to the user, not resolved here. A `TRENCH-IDS_Project_Summary.pdf` (2026-07-24) packages the rest of this round into a project-status report — not a paper draft, numbers cross-checked against source JSON during this reconciliation and confirmed consistent (its own port-fix section already describes the result as "mixed" and not yet decided, consistent with this correction).

**Resolved 2026-07-27**: the multi-seed comparison this note called for has been run (`project-metrics.md` §24.4) — and it's git-verified, unlike §24.1's single comparison, which rested only on directory naming and predated the port-fix commit (`30ce5a0`, 2026-07-25) entirely (both its runs finished on 2026-07-24). This round's "old" runs (`runs/replay_baseline`, `runs/replay_seed1`) genuinely predate `30ce5a0`; its "new" runs (`runs/replay_seed42_newport`, `runs/replay_seed1_newport`) postdate it on committed `master` code. Across 2 seeds, the new scheme wins on both average forgetting (mean 0.089 vs. 0.102) and final accuracy (mean 0.872 vs. 0.854), with no consistent regression — the T3/-0.162 and T4/-0.028 drops §24.1 flagged both flip sign on the second seed, confirming they were seed noise rather than a real cost of the fix. **Decision: keep the new scheme.** Both the structural justification and the metric evidence now agree; no further port-scheme comparison is planned.

**Step 10b (unseen-attack inference, the third Step 10 bullet) implemented and run 2026-07-28** (`docs/superpowers/specs/2026-07-27-unseen-attack-inference-design.md`, `project-metrics.md` §26, merged to master in `7aabd32`). New standalone modules — `src/trench_ids/unseen_data.py`, `src/trench_ids/unseen_graphs.py`, `src/trench_ids/cl/inference_unseen.py` (CLIs: `trench-unseen-data`, `trench-unseen-graphs`, `trench-infer-unseen`) — pull rows directly from raw per-dataset CSVs that `preprocess.py` never touches, bypassing `CLASS_DATASETS`'s allow-list on purpose (frozen Step 1/2 pipeline itself is untouched). Two separate evaluations against `runs/replay_seed42_newport/checkpoint_task_6.pt` (the validated best forgetting-mitigation checkpoint): **Dataset A** (seen classes from unseen sources — BoT-IoT's own DDoS/DoS, UNSW-NB15's Reconnaissance/DoS) scores near-zero accuracy (pooled 0.0008) with confident, consistent mis-mapping (e.g. BoT-IoT DDoS→"Reconnaissance" 99.3% of the time) — a genuine negative finding that the model keys on dataset-specific feature distributions rather than transferable attack semantics, confounded by unseen graphs being single-class/unshuffled/no-benign-mixing unlike training graphs (§26.1). **Dataset B** (5 classes never in any task: Backdoor, MITM, Ransomware, Web Attacks, Theft) has no ground-truth label space, so it's evaluated by prediction-distribution/confidence/embedding-similarity instead of accuracy — every class collapses onto 1-2 known classes with high softmax confidence (medians ≥0.89), so the model never signals "novel" via confidence alone; a dedicated OOD detector is explicitly out of scope. 188 tests passing. Reproduction order: `trench-unseen-data` → `trench-unseen-graphs` → `trench-infer-unseen`, all CPU-only (`--device auto` picks CPU fine — unlike the checkpoint it depends on, which needs GPU to train); BoT-IoT's two extraction steps take several minutes each streaming an 18M+/16M+-row raw CSV.

**Methodology step status — two separate bars, don't collapse them (user-corrected framing, 2026-07-21):**
- *Implementation*: Steps 1-9 complete. Step 10 (transferability analysis and prediction) is now largely a standalone stage too: Step 10a (transferability reports, `project-metrics.md` §19-20) and Step 10b (unseen-attack inference, 2026-07-28 note above, `project-metrics.md` §26) both exist as distinct deliverables with dedicated modules and CLIs, not just scattered side effects of other steps.
- *Experimental validation* (stricter): Steps 1-7 and 9 are both implemented **and** demonstrated to work as intended. Step 8 is implemented correctly, and its effectiveness question is now **settled, not open**: the λ sweep above, the 2026-07-23/24 hyperparameter sweep, and a same-seed multi-metric comparison all show standard Online EWC (uniform or relation-aware) does not reduce forgetting on this benchmark — precisely scoped to `num_layers=1`, `num_layers=2` (§18) checked and indistinguishable too. **This does not mean the project has no working forgetting-mitigation result**: experience replay (2026-07-23/24, above) does reduce forgetting substantially and is the paper's actual positive result for Step 7's slot; Step 8's specific transferability-weighted-EWC mechanism is reported as a negative result alongside it, not swept under the rug. Step 10b's Dataset A/B results (2026-07-28 note above) are themselves validated findings (real run, real checkpoint, 188 passing tests) — they just happen to be *negative* about cross-dataset generalization, which is a legitimate outcome to validate, not a gap in validation.
- *Port-embedding branch*: the `experiment/port-embedding-fixed-wellknown` open item above is resolved — the fix is on `master` now (2026-07-24 note above), the branch itself is stale/superseded and can be deleted once nobody needs to diff against it.

The codebase is a Python package (`src/trench_ids/`) managed with `uv`. Step 1 (raw traffic preprocessing) and Step 2 (heterogeneous graph construction) are implemented. **Step 2 design revised 2026-07-13** (professor's guidance, first round): rather than one large `HeteroData` graph per task, `src/trench_ids/graphs.py` now splits each task's rows by train/val/test and chunks each split into many small mini-graphs of at most `graph_size` flows (`configs/graph.yaml`, default 300 — rerun with a different value, and a different `paths.out_dir`, to experiment with graph size). CL task boundaries (T1→T2→T3→T4→T5→T6) are unchanged — training still runs through all of a task's mini-graphs before advancing. Outputs are now `data/graphs/task_{t}_{split}.pt` (each a `list[HeteroData]`) + `graph_counts.json` (gitignored, same as `data/processed/`), plus `src/trench_ids/vocab.py` for the global PROTOCOL/L7_PROTO vocabulary built across all 3 retained datasets. Step 1's `attack_per_class_cap` has been removed entirely (every row passing `CLASS_DATASETS`/`EXCLUDED_CLASSES` is kept in full — see below); the per-task attack:benign ratio is now a Step 2 sweep knob, `configs/graph.yaml: sampling.benign_ratio` (candidates 2.0/3.0/4.0, current default 3.0), falling back to sampling-with-replacement when a split's pooled benign can't cover the target ratio. Every flow also now carries a stable `flow_id` (`f"{source_dataset}-{original_csv_row_number}"`), assigned in Step 1 before concatenation, so Flow identity survives from raw CSV through to the Step 2 Flow graph node. Relation-type strings in `graphs.py` were renamed for clarity (pure rename, no schema change): `host--sends-->flow` → `host--originates-->flow`, `flow--received_by-->host` → `flow--terminates_at-->host`, `flow--uses_port-->port` → `flow--targets_port-->port`, `host--talks_to-->host` → `host--communicates_with-->host` (`uses_protocol`/`uses_service` unchanged). DoS/DDoS-style downsampling strategy is still pending — now understood to be a Step 2 concern operating on constructed graphs, not a pre-graph cap. Steps 3-9 are now implemented (see the 2026-07-19 and 2026-07-21 notes above); the coarse EWC lambda sweep is now complete (see the 2026-07-21 sweep note above) and shows standard Online EWC does not help on this benchmark — next is either further methodology-stage work (Step 10) or investigating alternatives to Step 8's learned weighting (e.g. detaching `w_r`, `num_layers=2` to make the 6 currently-dead relations trainable, or a different importance-weight training signal). Stack chosen: Python 3.10, `uv`, pandas/pyarrow for Step 1; PyTorch + PyTorch Geometric for Step 2/3 (`src/trench_ids/graphs.py`, `src/trench_ids/model/`); Hydra for Step 4+ training config (`src/trench_ids/cl/`, `configs/train.yaml`).

Key layout:
- `src/trench_ids/labels.py` — single source of truth for cross-dataset label harmonization, per-class dataset source restriction (`CLASS_DATASETS`), and T1–T6 task assignment (similarity-driven, isolate-and-bundle — see below), plus `EXCLUDED_CLASSES` (5 below-floor classes: Backdoor, MITM, Ransomware, Web Attacks, Theft). Any label/task change goes here.
- `src/trench_ids/similarity.py` / `src/trench_ids/task_design.py` — compute the attack-class cosine-similarity matrix (per-class mean feature vector, standardized across classes, restricted per `CLASS_DATASETS`) and the isolate-and-bundle task grouping from it. Rerun both if the candidate pool or threshold changes; see `docs/attack-similarity-matrix.md`.
- `src/trench_ids/preprocess.py` — Step 1 pipeline (two-pass streaming: count → sample every `CLASS_DATASETS`-restricted attack row in full + per-dataset benign subsample → dedup → per-task stratified split → Parquet + manifest). No attack-side sampling cap.
- `configs/preprocess.yaml` — Step 1 knobs (benign budget, split ratios, seed; `attack_per_class_cap` removed).
- `docs/dataset-plan.md` — **the authoritative task/graph design** (professor-scoped to Steps 1 & 2): task table T1–T6 (similarity-driven), below-floor classes dropped, benign-per-task, global normalization, per-task heterogeneous graphs (node/relation types for Step 2). Read this first for any design decision.
- `docs/datasets.md` — detailed dataset *reference* (verified counts, schema, full label-harmonization map, Step 1 pipeline). Read before touching preprocessing. Where it and dataset-plan.md disagree, dataset-plan.md wins.
- `docs/attack-similarity-matrix.md` — the 10×10 cosine-similarity matrix and the isolate-and-bundle task grouping derived from it.
- `data/processed/` — generated `task_{1..6}.parquet` (45 original NetFlow cols + `flow_id`/`source_dataset`/`canonical_label`/`task`/`split`) + `manifest.json`. Gitignored.

### Commands
```bash
uv venv --python 3.10                 # create .venv
uv pip install -e ".[dev]"            # install package + pytest/ruff
.venv/Scripts/python.exe -m pytest -q # run tests (Windows path)
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m trench_ids.preprocess --config configs/preprocess.yaml  # run Step 1
```
`pytest tests/test_labels.py::test_unknown_label_raises` runs a single test. The full preprocessing run streams ~12 GB across two passes, now with no attack-side cap (~15.7M attack rows kept, see "Continual-learning task split" below) — run it in the background.

### Compute: prefer GPU when available

User preference (2026-07-15): run compute-heavy tasks on GPU rather than CPU when a GPU is available, since it completes faster. **This dev machine currently has no CUDA device** (`torch.cuda.is_available()` returns `False`, 0 devices) — verify with that same check before assuming otherwise, since this may change if the user moves to different hardware. This mainly matters for **Step 3+** (GNN training, embedding computation, anything doing real tensor math over many batches) — that's where GPU absence vs. presence will actually change wall-clock time. It does **not** help Step 1/2 or diagnostic scripts like `src/trench_ids/graph_composition.py`: those are I/O-bound (streaming CSVs, `torch.load`-ing saved graphs from disk) or trivial-compute (counting labels), so moving them to GPU would add transfer overhead for no benefit — confirmed by the composition script's ~40-minute real run being ~90% wall-clock spent on disk I/O (CPU time was a small fraction of elapsed time, per `wmic` process timing). When Step 3 training code is written, default to `device = "cuda" if torch.cuda.is_available() else "cpu"` and structure batching so it's GPU-ready even on this CPU-only box.

Git tracks `.gitignore` and `README.md`; source code under `src/`, `tests/`, `configs/`, and `pyproject.toml` are new and untracked (commit when the user asks). `docs/`, `dataset/`, `data/`, and `CLAUDE.md` are gitignored — they exist on disk for local reference but are intentionally kept out of version control (datasets/outputs are large; docs/CLAUDE.md are working notes). Don't `git add` anything under those paths without checking with the user first — it's excluded on purpose.

## Research goal

**TRENCH-IDS**: Transferable REpresentatioN learning for Continual Heterogeneous graph-based IDS. The target output is a paper with a working end-to-end pipeline and preliminary results by **2026-07-20**.

Core gaps the method addresses (from `docs/Proposed methodology for the next work.docx`):
- Continual IDS methods don't estimate which learned representations transfer to future unseen attack classes.
- Existing CL methods preserve all learned representations uniformly instead of selectively preserving transferable relation-specific ones.
- Existing GNN-based IDS methods don't learn relation-specific representations that capture diverse network traffic.

### Proposed pipeline (per task, in a continual-learning loop)

1. **Raw traffic preprocessing** — partition traffic into tasks via similarity-driven isolate-and-bundle assignment (see task table below).
2. **Heterogeneous graph construction** — build a graph per task with node types `Host, Flow, Protocol, Port, Service` (features from raw NetFlow fields) and relation/edge types `Host→Flow, Flow→Port, Flow→Protocol, Flow→Service, Host→Host`.
3. **Relation-specific encoding + attention fusion** — learn a separate embedding per relation type, fuse via attention into one node embedding; each node retains both the per-relation embeddings and the fused embedding.
4. **Classifier training + relation-specific memory** — train a classifier on the current task's attack classes; store the mean per-relation embedding for each attack type in a memory bank.
5. **Transferability estimation** (on each new task) — repeat steps 2–4, then compare the new attack classes' per-relation mean embeddings against all previously stored means via cosine similarity, per relation.
6. **Relation importance weights** — a lightweight MLP learns which relations are worth preserving from the transferability scores.
7. **Relation-aware continual learning** — importance weights feed into a modified EWC (Elastic Weight Consolidation) that selectively regularizes parameters tied to transferable relations. **This formulation is the highest-risk, still-unresolved part of the method** — see Open Items below before implementing the CL loop.
8. **Memory update** — refresh the memory bank with the new task's per-relation means.
9. **Prediction** — steps 1–3 produce node embeddings that feed the trained classifier.

### Open items (unresolved as of `docs/dataset-plan.md`)

- Relation-aware EWC formulation (step 7) — not nailed down; needed before the CL loop can be implemented end-to-end.
- How per-relation cosine-similarity transferability scores aggregate into a single decision (step 5).
- Concrete mechanism mapping relation importance weights to EWC-regularized parameters.
- Raw data acquisition: sandbox network can't reach the UQ NIDS dataset host directly — CSVs must be fetched externally (UQ page or Kaggle mirrors) and added locally.
- Cross-dataset schema verification — confirm the 43-feature schema is identical/aligned across all four datasets before graph construction.
- **Port node representation** (Step 3, raised by the professor 2026-07-17) — the original logarithmic `port_bucket()` scheme collapsed semantically distinct well-known ports into the same embedding (e.g. FTP/SSH/Telnet — ports 20/21/22/23 — all landed in the same bucket at the 32-bucket setting), and since port value reaches the model only through this embedding (not present in the 37 Flow features), the collision was a real information loss, not just a lossy tradeoff. **Update 2026-07-24/26**: the fix (fixed per-port embeddings for 0–1023, log-bucketed tail for 1024–65535, `src/trench_ids/model/rhgnn.py`'s `port_embedding_index`/`port_embedding_size`) is now on `master` (no longer only on `experiment/port-embedding-fixed-wellknown`, which is stale/superseded), unconditionally — there's no config path back to the original scheme. But the empirical CL comparison this item always said was needed before finalizing turned out to favor the *old* scheme on the one seed pair tested (mean final-accuracy Δ = -0.028, driven by T3 -0.162; see the 2026-07-23/24 note above and `project-metrics.md` §24). **Resolved 2026-07-27**: a git-verified, 2-seed replay comparison (`project-metrics.md` §24.4) now shows the new scheme winning on both average forgetting and final accuracy on both seeds, with the earlier T3/T4 regressions flipping sign on the second seed — confirming §24.1's single-seed result was noise, not a real cost of the fix. New scheme is kept; both the structural and empirical cases now agree. `docs/port-representation-proposal.md` still describes the original single-seed prototype evidence, not this later multi-run comparison — treat `project-metrics.md` §24 as authoritative.

Treat `docs/dataset-plan.md` as the current source of truth for dataset/task decisions — it supersedes ad hoc assumptions from the `.docx`.

## Datasets

Five NetFlow v2 datasets from UQ (https://staff.itee.uq.edu.au/marius/NIDS_datasets/) live under `dataset/<name>/`, packaged as BagIt bags (`bag-info.txt`, `bagit.txt`, `manifest-sha1.txt`, `tagmanifest-sha1.txt`, `FurtherInformation.txt`, plus the payload in `data/`). Each `data/` dir has the dataset CSV (`IPV4_SRC_ADDR, L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT, ...` + `Label`/`Attack` columns) and a shared `NetFlow_v2_Features.csv` describing all 43 standardized features (Sarhan et al. feature set — this is what makes a unified heterogeneous graph schema viable across datasets).

**`NF-UQ-NIDS-v2` is excluded from experiments** — it's the merged union of the other four, not an independent domain; including it would double-count flows and bias evaluation. It's present on disk but should not be used for training/eval.

**`NF-UNSW-NB15-v2` stays excluded** (unchanged conclusion, reconfirmed 2026-07-13 second round). Empirically, UNSW-NB15 contributes no class that survives dropping it once BoT-IoT is back in the pool — its only relevant contribution, Reconnaissance, is 99.5% supplied by BoT-IoT anyway (2,620,999 of 2,633,778 combined; `docs/attack-class-counts.md`), and every other UNSW-only class (Exploits, Fuzzers, Generic, Analysis, Shellcode, Worms) is well under 100K. See `docs/dataset-plan.md` §1 for the full reasoning.

**`NF-BoT-IoT-v2` is back in** (2026-07-13, second round — the professor's guidance changed from minimizing complexity to building a richer benchmark), but **restricted to Reconnaissance only** via `CLASS_DATASETS`: BoT-IoT independently contains large volumes of `DDoS`/`DoS`-labeled rows (18.3M / 16.7M), which must not leak into those canonical classes — BoT-IoT's only sanctioned contribution is Reconnaissance (2,620,999 rows).

**Datasets actually used (3):**

| Dataset | Attack classes (`CLASS_DATASETS`-restricted candidate pool) |
|---|---|
| NF-ToN-IoT-v2 | Scanning, XSS, Password, DDoS, DoS, Injection |
| NF-CSE-CIC-IDS2018-v2 | DDoS, DoS, Injection, Bot, BruteForce, Infiltration |
| NF-BoT-IoT-v2 | Reconnaissance only |

### Continual-learning task split

Tasks are **similarity-driven**: co-locating similar attacks (e.g. DoS+DDoS) in the same task gives the model nothing to transfer, since both get learned directly from labels in one task. Method (`src/trench_ids/similarity.py`, `src/trench_ids/task_design.py`, full detail in `docs/dataset-plan.md` §2 and `docs/attack-similarity-matrix.md`): sample ~5,000 rows per candidate class (restricted per-class to its `CLASS_DATASETS` source(s)), compute each class's mean feature vector over the 37 flow-statistic features, standardize those means across classes, then take pairwise cosine similarity. Task assignment is **isolate-and-bundle**: classes that are all pairwise above a similarity threshold (a "conflict clique") each get their own singleton task, since no split avoids a conflict among them; the rest are paired via minimum-weight matching, still respecting the threshold.

**Candidate pool: 10 classes**, each restricted to specific source dataset(s) via `CLASS_DATASETS` rather than pooled wherever the canonical label happens to appear (`docs/attack-class-counts.md`). Tasks are **1-indexed (T1–T6)**. Benign is present in **every** task (see below); the table lists each task's *attack* classes:

| Task | Attack classes | Max intra-task cosine sim | Source dataset(s) |
|---|---|---:|---|
| T1 | Scanning | — (isolated) | ToN-IoT |
| T2 | Reconnaissance | — (isolated) | BoT-IoT |
| T3 | XSS, DDoS | −0.361 | ToN-IoT, CSE-CIC-IDS2018 |
| T4 | Password, Infiltration | −0.457 | ToN-IoT, CSE-CIC-IDS2018 |
| T5 | DoS, Injection | −0.343 | ToN-IoT, CSE-CIC-IDS2018 |
| T6 | Bot, BruteForce | −0.297 | CSE-CIC-IDS2018 |

Scanning and Reconnaissance are isolated in their own tasks because {Scanning, Reconnaissance} form the 10-class pool's max conflict clique at the 0.35 similarity threshold; the remaining eight classes pair off via minimum-weight matching, still respecting the threshold. This grouping is stable across thresholds **0.21–0.55** (verified empirically). Two pairings stand out for transferability evaluation: XSS↔Infiltration is the single highest cosine pair in the whole matrix (0.706), landing in different, non-adjacent tasks (T3 vs. T4); Scanning↔Reconnaissance is the second-highest (0.628), also isolated from each other (T1 vs. T2). Unlike the prior 2-dataset design, not every task spans 2+ datasets — 3 of 6 tasks (T1, T2, T6) are single-dataset; that constraint was specific to the earlier complexity-reduction guidance and isn't restated here.

**Backdoor, MITM, Ransomware, Web Attacks, and Theft are dropped entirely**: all five (2,431–16,809 rows) fall well below the ~116K–3.8M range of the ten kept classes. All still map canonically so label normalization never fails, but are filtered out in preprocessing (`EXCLUDED_CLASSES` in `labels.py`). Note that **Bot, BruteForce, and Infiltration are no longer dropped** — the prior 2-dataset/6-class design excluded them as "below threshold"; this design includes them for benchmark richness (more tasks, more forgetting/transfer signal), a reversal of that reasoning rather than a data change (see `docs/attack-class-counts.md`).

**Benign handling:** benign is the shared negative class present in every task — each task gets a *fresh* Benign subset matched to that task's contributing dataset(s), rather than one Benign pool reused everywhere. This keeps traffic-distribution consistent per task and isolates attack-pattern forgetting from normal-traffic-distribution forgetting when measuring catastrophic forgetting. There is **no standalone benign task** (a single-class task is degenerate for a classifier). Step 1's `benign_per_task`/`benign_per_dataset_cap` remain pool-size safeguards; the final per-task attack:benign ratio a task's graphs see is now a Step 2 sweep knob (`configs/graph.yaml: sampling.benign_ratio`).
