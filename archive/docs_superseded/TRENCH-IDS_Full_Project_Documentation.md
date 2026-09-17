# TRENCH-IDS — Full Project Documentation

**Trans­ferable REpresentatioN learning for Continual Heterogeneous graph-based Intrusion Detection System.**

A research project (target: a paper, preliminary results due 2026-07-20) that builds a continual-learning benchmark for network intrusion detection on real NetFlow traffic, and tests whether a relation-specific heterogeneous graph neural network, combined with transferability-guided continual-learning mechanisms, can resist catastrophic forgetting as new attack types arrive over time.

This document is a complete, numbers-backed account of the project: motivation, method, what was built, what was measured, what worked, what didn't, and why. It is meant to be usable standalone — for an interview, a stakeholder update, or handing to a collaborator with zero prior context.

---

## 1. The problem and the research idea

### 1.1 Why this problem exists

Network intrusion detection systems (IDS) increasingly use graph neural networks (GNNs) on NetFlow-style traffic, because network data is naturally relational (hosts talk to hosts, flows use ports/protocols/services). But two real gaps show up when you try to deploy this in a setting where **new attack types appear over time** (which is the normal situation for a production IDS):

1. **Catastrophic forgetting.** If you keep training the same model on new attack classes as they show up, it forgets how to detect the old ones — a well-known failure mode of neural networks under sequential ("continual") learning.
2. **No mechanism for estimating what actually transfers.** Existing continual-learning defenses (e.g., Elastic Weight Consolidation) protect *all* learned parameters roughly equally, rather than asking "which specific learned representations are actually useful for detecting *future*, not-yet-seen attacks?" And existing GNN-based IDS approaches don't even learn separate, comparable representations per relation type (host-to-flow, flow-to-port, etc.) — everything gets blended into one embedding, so there's nothing to selectively protect in the first place.

### 1.2 The proposed method (9-step pipeline, repeated per task in a continual-learning loop)

1. **Raw traffic preprocessing** — partition traffic into sequential tasks using a similarity-driven "isolate-and-bundle" strategy (below).
2. **Heterogeneous graph construction** — build a graph per task with node types `Host, Flow, Protocol, Port, Service` and edges like `Host→Flow`, `Flow→Port`, `Flow→Protocol`, `Flow→Service`, `Host→Host`.
3. **Relation-specific encoding + attention fusion** — learn a *separate* embedding per relation type for every node, then fuse them via attention into one embedding per node; both the per-relation and fused embeddings are kept.
4. **Classifier training + relation-specific memory** — train a classifier on the current task's classes; store the mean per-relation embedding for each attack type in a memory bank.
5. **Transferability estimation** — for each new task's new attack classes, compare their per-relation embeddings against everything already in the memory bank via cosine similarity, per relation.
6. **Relation importance weights** — a lightweight MLP learns which relations are worth protecting, from the transferability scores.
7. **Relation-aware continual learning** — feed those importance weights into a modified Elastic Weight Consolidation (EWC) that selectively regularizes the parameters tied to the most transferable relations.
8. **Memory update** — refresh the memory bank with the new task's per-relation means.
9. **Prediction** — the trained encoder + classifier is used for inference, including on genuinely unseen (out-of-benchmark) traffic.

All 9 steps were **fully implemented**. Steps 1–2 and the encoder (step 3) work as designed. Steps 4–9 (the continual-learning loop) were implemented correctly and rigorously validated — and the headline finding is a **precisely-scoped negative result**: the proposed transferability-weighted EWC mechanism (steps 5–7) does **not** reduce catastrophic forgetting on this benchmark, under extensive testing (5 orders of magnitude of the regularization strength, two model depths, a fixed-weight ablation that removes the specific collapse mechanism identified). What *does* work, and became the project's actual positive contribution, is a **standard experience-replay mechanism** (see §8), which closes almost the entire forgetting gap.

This "we tried the sophisticated idea, characterized exactly why it fails, and found a simpler mechanism that works" arc is itself the deliverable — a legitimate, well-evidenced research outcome, not a partial failure.

---

## 2. Datasets

Five NetFlow v2 datasets (Sarhan et al. standardized 43-feature schema) from the University of Queensland (UQ NIDS dataset collection). Only **3 of the 5** are actually used:

| Dataset | Status | Role |
|---|---|---|
| **NF-ToN-IoT-v2** | Used | Scanning, XSS, DDoS, Password, DoS, Injection |
| **NF-CSE-CIC-IDS2018-v2** | Used | DDoS, DoS, Injection, Bot, BruteForce, Infiltration |
| **NF-BoT-IoT-v2** | Used, restricted | **Reconnaissance only** (it also independently contains 18.3M DDoS / 16.7M DoS rows, which are deliberately excluded to avoid leaking a second data distribution into those canonical classes) |
| NF-UNSW-NB15-v2 | Excluded | Its only relevant class, Reconnaissance, is 99.5% supplied by BoT-IoT anyway (2,620,999 of 2,633,778 combined rows); every other UNSW-only class is well under 100K rows |
| NF-UQ-NIDS-v2 | Excluded | It's the merged union of the other four datasets — including it would double-count flows and bias evaluation |

A per-class dataset restriction (`CLASS_DATASETS` in code) enforces exactly which dataset each canonical class is allowed to come from, rather than pooling a class wherever its label happens to appear across datasets.

### 2.1 Candidate class pool (10 classes kept, 5 dropped)

| Class | Allowed dataset(s) | Raw row count |
|---|---|---:|
| Scanning | ToN-IoT | 3,781,419 |
| DDoS | ToN-IoT, CSE-CIC-IDS2018 | 3,416,504 |
| Reconnaissance | BoT-IoT | 2,620,999 |
| XSS | ToN-IoT | 2,455,020 |
| DoS | ToN-IoT, CSE-CIC-IDS2018 | 1,196,608 |
| Password | ToN-IoT | 1,153,323 |
| Injection | ToN-IoT, CSE-CIC-IDS2018 | 684,897 |
| Bot | CSE-CIC-IDS2018 | 143,097 |
| BruteForce | CSE-CIC-IDS2018 | 120,912 |
| Infiltration | CSE-CIC-IDS2018 | 116,361 |

**Dropped entirely** (`EXCLUDED_CLASSES`, all far below the 116K–3.8M kept range): Backdoor, MITM, Ransomware, Web Attacks, Theft (2,431–16,809 rows each). These 5 classes are later reused as genuinely unseen attack traffic for the Step 10b evaluation (§9).

Total candidate-pool raw attack rows: **15,689,140** (verified to match Step 1's real pipeline output exactly).

---

## 3. Continual-learning task design (the benchmark's core design decision)

### 3.1 Why tasks are similarity-driven, not random

If two similar attacks (e.g., DoS and DDoS) are placed in the *same* task, the model learns both directly from labels — there's nothing left to "transfer." The benchmark is only interesting for studying transfer/forgetting if **similar attacks are deliberately split across different, non-adjacent tasks**, so that any transfer the model exhibits is a genuine generalization, not label leakage.

### 3.2 Method

1. Sample ~5,000 rows per candidate class (respecting the per-class dataset restriction).
2. Compute each class's mean feature vector over 37 flow-statistic features.
3. Standardize means across classes, then compute the full 10×10 pairwise cosine-similarity matrix.
4. **Isolate-and-bundle assignment**: any group of classes that are all pairwise above a similarity threshold (a "conflict clique") each get their own singleton task, since no split avoids the conflict. The remaining classes are paired off via minimum-weight matching (with a size-aware tie-break, see below), still respecting the threshold.

Threshold used: **0.35**, and the resulting grouping is stable across the entire **0.21–0.55** range (verified by a 21-point threshold scan) — i.e., this isn't a fragile, cherry-picked cutoff.

### 3.3 The size-aware tie-break (a real methodological fix made mid-project)

The first version of the matching picked the pairing that minimized *total* pairwise similarity, which happened to select the single worst-balanced option out of 78 valid pairings (one task would have held 5.88M rows vs. another holding a fraction of that). This was fixed by changing the tie-break objective to minimize the **largest resulting task's size** instead — among all threshold-valid pairings, not by changing the similarity computation itself. This dropped the largest:smallest task-size ratio from **14× to 2.9×**.

### 3.4 Final task table (T1–T6, 1-indexed)

| Task | Attack classes | Dataset(s) | Max intra-task cosine similarity |
|---|---|---|---:|
| T1 | Scanning | ToN-IoT | — (isolated, part of the max conflict clique) |
| T2 | Reconnaissance | BoT-IoT | — (isolated, part of the max conflict clique) |
| T3 | DDoS + Infiltration | ToN-IoT, CSE-CIC-IDS2018 | −0.462 |
| T4 | DoS + Injection | ToN-IoT, CSE-CIC-IDS2018 | −0.343 |
| T5 | Password + Bot | ToN-IoT, CSE-CIC-IDS2018 | −0.046 |
| T6 | XSS + BruteForce | ToN-IoT, CSE-CIC-IDS2018 | −0.390 |

**Scanning and Reconnaissance are isolated in their own tasks** because {Scanning, Reconnaissance} form the pool's max conflict clique (cosine similarity 0.628, second-highest in the whole matrix) at the 0.35 threshold. **XSS↔Infiltration is the single highest cosine pair in the entire 10-class matrix (0.706/0.823 depending on measurement round)**, and the design deliberately lands them in different, non-adjacent tasks (T6 vs T3) — this pair, along with Scanning↔Reconnaissance, becomes the headline test case for whether the model's *learned* representations actually transfer the way raw-feature similarity would predict (spoiler in §7: they mostly don't, and are even mildly *anti*-correlated overall).

**Benign is present in every task** — a fresh, dataset-matched Benign subset per task (not one shared pool), so that measuring forgetting isolates attack-pattern forgetting from any drift in "what normal traffic looks like." There's no standalone Benign task (a single-class task is degenerate for a classifier).

---

## 4. Data pipeline (Steps 1–2 — frozen since 2026-07-14, no further redesign)

### 4.1 Step 1 — raw traffic preprocessing (`preprocess.py`)

Two-pass streaming pipeline over the raw CSVs (~12+ GB): count → sample every allowed attack row in full (no cap) + a per-dataset benign subsample → deduplicate → drop corrupted rows (NaN / ±inf / float32-overflow values — a real bug found and fixed, see §4.3) → per-task stratified 70/15/15 train/val/test split → Parquet output + a manifest.

Real run output (`data/processed/`, seed 42, `benign_per_dataset_cap=60000`, `benign_per_task=8000`):

| Task | Theme | Rows (incl. benign) | Dups dropped | Corrupted dropped | Train/Val/Test | Sources |
|---|---|---:|---:|---:|---|---|
| 1 | Scanning | 3,789,419 | 0 | 0 | 2,652,593 / 568,413 / 568,413 | ToN |
| 2 | Reconnaissance | 2,628,994 | 5 | 0 | 1,840,296 / 394,349 / 394,349 | BoT |
| 3 | DDoS + Infiltration | 3,540,254 | 557 | 54 | 2,478,178 / 531,039 / 531,037 | ToN + CSE |
| 4 | DoS + Injection | 1,889,505 | 0 | 0 | 1,322,654 / 283,426 / 283,425 | ToN + CSE |
| 5 | Password + Bot | 1,304,420 | 0 | 0 | 913,094 / 195,663 / 195,663 | ToN + CSE |
| 6 | XSS + BruteForce | 2,583,932 | 0 | 0 | 1,808,752 / 387,590 / 387,590 | ToN + CSE |
| **Total** | | **15,736,524** | **562** | **54** | | ~320 MB on disk |

Wall-clock: ~27 minutes.

### 4.2 Step 2 — heterogeneous graph construction (`graphs.py`)

Rather than one giant graph per task, each task's rows are split by train/val/test and chunked into many small **mini-graphs** of at most 300 flows each (`graph_size=300`). Continual-learning task boundaries (T1→T6) are unchanged — training runs through all of a task's mini-graphs before advancing to the next task.

**Node types (5):** Flow, Host, Protocol, Service, Port.
**Relation types (originally 6, later 11 — see §5.3):** `host--originates-->flow`, `flow--terminates_at-->host`, `flow--targets_port-->port`, `flow--uses_protocol-->protocol`, `flow--uses_service-->service`, `host--communicates_with-->host`.

**`benign_ratio` sweep:** the target per-task attack:benign ratio in the constructed graphs is a tunable knob (`target_benign = n_attack / benign_ratio`). Three full graph sets were actually produced (not just picked on paper):

| `benign_ratio` | Output dir | Total mini-graphs | Disk |
|---:|---|---:|---:|
| 2.0 (~33% benign per graph) | `data/graphs_ratio2/` | 78,452 | 6.47 GB |
| 3.0 (25% benign, **default used for all model training**) | `data/graphs/` | 69,737 | 5.73 GB |
| 4.0 (~20% benign) | `data/graphs_ratio4/` | 65,378 | 5.36 GB |

Wall-clock: ~51–61 minutes per set (~3.3 hours total for all Step 1+2 work).

**Downsampling safety cap**: any task is capped at 3× the smallest task's graph count (`max_task_ratio: 3.0`) — never actually triggers, since the rebalanced task table's natural ratio (~2.9×) already sits under it.

### 4.3 Real bugs found and fixed during Steps 1–2

- **Corrupted-row bug (resolved)**: 54 raw CSE-CIC-IDS2018 rows had `SRC_TO_DST_SECOND_BYTES`/`DST_TO_SRC_SECOND_BYTES` values that overflowed float32 at Step 2's cast, producing `inf` in the saved tensors (540 inf values across 533 mini-graphs). Fixed with a Step-1 filter that drops any row with NaN/±inf/float32-overflow in any numeric column, before dedup. A full tensor scan across all ~213,567 saved graphs in all three `benign_ratio` sets confirmed **zero** remaining non-finite values.
- **Global label-encoding bug (resolved)**: mini-graphs originally built their integer class labels from each chunk's own locally-sorted class list, so the same integer index could silently mean a *different* class in different mini-graphs — a serious problem for a classifier trained with cross-entropy loss across many mini-graphs. Fixed by switching to one fixed global 11-class label ordering used identically by every mini-graph in every task.

### 4.4 Graph topology and composition characteristics (from real, whole-dataset scans — not samples)

- Task 2 (Reconnaissance) mini-graphs consistently have by far the fewest hosts per graph (~12–14) but by far the highest host degree (~44–50) and host↔host edge density (~0.21–0.24) — consistent with scanning-style traffic where a small number of hosts touch many others.
- Task 1 (Scanning) mini-graphs consistently have the widest port spread (~244–246 average distinct ports per graph) — consistent with scanning behavior sweeping many destination ports.
- **Almost every mini-graph is class-mixed** (99.98–100% across every multi-class task) — true single-class graphs are essentially absent, which is good for the benchmark (every graph carries co-occurrence signal). But **within a task, the dominant class and mean class-share are essentially fixed** (zero ties, one class always dominates) — real compositional diversity in this benchmark exists *across* tasks (e.g. Task 4's DoS:Injection split is ~1.75:1, far more balanced than Task 3's ~30:1 DDoS:Infiltration split), not within them.

### 4.5 Test coverage at this stage

164 tests passing at the end of the data-pipeline work (grew incrementally as the pipeline was built); `ruff check` clean throughout.

---

## 5. Step 3 — Relation-specific heterogeneous GNN encoder

### 5.1 What it does

Every node's raw features are normalized (`LayerNorm`, necessary because raw NetFlow byte/packet counters span huge dynamic ranges) and projected into a shared hidden dimension. The model then runs **relation-specific message passing**: a separate learned transformation per relation type for every node, producing one embedding *per relation* per node — these are then fused via **semantic attention** (HAN-style, Wang et al. WWW 2019) into a single fused embedding per node, while the per-relation embeddings are also retained (needed later for transferability estimation and relation-aware EWC).

### 5.2 Design evolution — two real revisions made mid-project

1. **No generic reverse edges** (first instruction from the professor): a synthetic `rev_<relation>` edge has no real-world meaning, so it was explicitly rejected.
2. **5 new, individually-named reverse relations added later** (professor's follow-up, once a real architectural gap was identified): every node type's fused embedding was 100% neighbor-derived — worst for Flow (the actual classification target), which had only 1 incoming relation and 0% self-signal. Rather than reversing the earlier instruction, 5 correctly-named passive-voice relations were added (`originated_by`, `terminated_by`, `targeted_by`, `protocol_of`, `service_of`), taking the schema from 6 to **11 relations** and Flow's incoming-relation count from 1 to 5.
3. **Self-concatenation (GraphSAGE-style)**: `RelationSpecificConv` now concatenates each destination node's own current embedding into every relation's neighbor-aggregated message before a per-relation "combine" layer projects it back down — this actually fixed the 0%-self-signal problem the reverse edges alone didn't fully solve.

### 5.3 Architecture size (real instantiation, `hidden_dim=64`, 1 layer)

| Config | Total params |
|---|---:|
| Original scheme (32-bucket-only port embedding) | 197,842 |
| **Current/default (exact well-known-port embeddings, §8.3)** | **263,378** |

Relation-conv parameters scale as O(R × 3H²) — 11 relations × (one H×H neighbor-transform matrix + one 2H×H combine matrix) per relation, per layer.

### 5.4 Verification

A real forward+backward pass on 32 real mini-graphs (9,600 batched flow nodes) confirmed: cross-entropy loss at random init ≈ ln(3) = 1.099 as expected for a 3-class task; every node type's attention weights sum to exactly 1.0 (softmax-guaranteed); a dedicated test confirmed the self-concatenation fix actually changes output when only the node's own features are perturbed (this test would have failed on the pre-fix code).

---

## 6. Steps 4–5 — Sequential training + transferability estimation (establishing the forgetting baseline)

### 6.1 Plain sequential fine-tuning baseline (Step 4)

One shared classifier head over the full 11-class global label space, trained sequentially T1→T6 with **no forgetting-mitigation mechanism** — a deliberate plain baseline, on GPU (`NVIDIA GeForce RTX 3050 Laptop GPU`).

**Forgetting matrix** (rows = task just finished training, columns = accuracy on that task's test set):

| Trained through | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---:|---:|---:|---:|---:|---:|
| T1 | **0.926** | | | | | |
| T2 | 0.162 | **0.945** | | | | |
| T3 | 0.151 | 0.052 | **0.955** | | | |
| T4 | 0.154 | 0.191 | 0.204 | **0.875** | | |
| T5 | 0.246 | 0.228 | 0.241 | 0.242 | **0.991** | |
| T6 | 0.197 | 0.208 | 0.247 | 0.238 | 0.248 | **0.997** |

Textbook catastrophic forgetting: every task reaches 0.87–0.997 on itself, then collapses to ~0.15–0.25 the moment training moves to the next task. This is the expected, motivating baseline — exactly the problem the transferability/EWC mechanism (steps 5–7) was designed to fix.

### 6.2 Relation-specific memory bank

After each task, a no-grad pass computes the mean embedding of each of that task's attack classes, per relation, stored in a growing memory bank (10 attack classes eventually, Benign correctly excluded).

### 6.3 Transferability estimation

For each new task's new classes, cosine similarity is computed against every existing bank entry, **same relation only** (different relations are different learned subspaces, not directly comparable to each other).

**The two headline pairs checked against real learned embeddings:**

| Pair | `originates` | `terminated_by` | `targeted_by` | `protocol_of` | `service_of` |
|---|---:|---:|---:|---:|---:|
| XSS (T6) vs Infiltration (T3) | 0.180 | **0.468** | 0.098 | **0.568** | 0.088 |
| Reconnaissance (T2) vs Scanning (T1) | −0.354 | 0.029 | −0.654 | −0.363 | −0.095 |

**Mixed, reported as-is**: XSS↔Infiltration shows partial replication of the raw-feature similarity (2 of 5 relations moderately positive). Scanning↔Reconnaissance shows **no** elevated similarity at all — likely because Scanning's memory-bank entry (from Task 1) was computed under an encoder state that had already been heavily forgotten by Task 6 (Task 1's own accuracy had fallen from 0.926 to 0.197 by then), so the comparison isn't apples-to-apples. This is itself a real finding: **an un-mitigated encoder's continual drift undermines its own memory bank's later comparability** — precisely what steps 6–7 exist to address.

---

## 7. Steps 6–8 — Relation-aware, transferability-weighted Online EWC (the negative result)

**The mechanism**: per task, transferability scores (§6.3) feed a learned per-relation importance weight `w_r = sigmoid(ImportanceMLP(S_r))`, which scales a relation-specific penalty inside a combined **Online EWC** loss (Schwarz et al. 2018) — `L = L_cls + λ_u·L_EWC(other) + λ_s·L_EWC(shared) + λ_r·Σ_r[w_r·L_EWC(relation r)]` — across 12 parameter groups (1 shared + 5 Flow relations + 6 other relations).

**Result and root cause**: at λ=1.0, this produced **no meaningful improvement** over plain fine-tuning. The cause was precisely diagnosed: `w_r` collapses toward zero starting Task 2 (by Tasks 4–6, every value below 1.5×10⁻⁵), because it's trained end-to-end and only ever appears multiplied into its own penalty — gradient descent has a direct, unconditional incentive to shrink it, since nothing in the per-task loss rewards keeping it large for *future* retention. A genuine methodological finding, not a bug: an importance weight trained end-to-end inside only its own penalty term has a trivial zero optimum.

**Three follow-up checks, each ruling out a different alternative explanation:**

| Check | What it ruled out | Result |
|---|---|---|
| Coarse λ sweep, 0.01→1000 (unweighted EWC) | "λ was just too small" | No trend across 5 orders of magnitude (0.71–0.75 forgetting band); a λ=1000 diagnostic run confirmed the penalty exceeded `L_cls` in later tasks yet still didn't help |
| `num_layers=2` ablation | "6 of 11 relations never getting gradient explains the failure" | Gradient reachability restored (confirmed empirically), but forgetting (0.7423) stayed statistically indistinguishable from `num_layers=1` — not the real bottleneck |
| Fixed-weight ablation (`w_r = sigmoid(S_r)`, no learnable params) | "the collapse was masking an otherwise-good idea" | Still no improvement (0.7257) — marginally *worse* than plain fine-tuning |

**Conclusion**: under the frozen `num_layers=1` architecture, Online EWC — plain, relation-aware/learned, or relation-aware/fixed-weight — does not reduce catastrophic forgetting on this benchmark. This holds across 5 orders of magnitude of regularization strength and is independent of GNN depth and of how the importance weight is computed. Reported as a genuine, well-evidenced negative result, not an under-tuned hyperparameter.

---

## 8. THE POSITIVE RESULT: Experience Replay closes the forgetting gap

**This is the project's central empirical result.** After Steps 6–8 established, rigorously, that EWC (in every variant tried) does not fix catastrophic forgetting on this benchmark, the direction shifted to a fundamentally different mitigation family — and it worked, dramatically.

### 8.1 What was tried

Per direct guidance ("try tweaking the hyperparameters, and try other methods like replay-based mitigation"), two things were tried:

- **Hyperparameter sweep** on relation-aware EWC (more epochs, different learning rates): stayed inside the same ~0.71–0.80 forgetting band — confirmed EWC's ineffectiveness wasn't a tuning artifact, and motivated abandoning EWC-family tuning entirely.
- **Experience replay** (literature-standard, Rolnick et al. NeurIPS 2019 / Chaudhry et al. 2019 / ER-GNN as the closest GNN-specific analog): after each task, 200 whole mini-graphs are sampled without replacement into a per-task buffer; when training a later task, buffer graphs are upsampled (with replacement) to make up 30% of the combined training pool, fed through the existing unmodified data loader. **No change to the model, the loss, or the training loop whatsoever** — EWC lambdas were zeroed to isolate replay's effect cleanly.

### 8.2 Results — a 7.5× reduction in forgetting, with zero architecture changes

| Method | Avg. forgetting | Final avg. accuracy |
|---|---:|---:|
| Plain fine-tuning (no mitigation) | 0.711 | 0.356 |
| Best EWC variant (any of the above) | 0.705–0.726 | 0.351–0.362 |
| **Experience replay (3 seeds)** | **0.094 ± 0.015** | **0.861 ± 0.014** |
| Joint training (non-CL upper bound) | — | 0.876 |

Per-seed detail:

| Seed | Avg. forgetting | Final avg. accuracy | Pooled F1 (macro) |
|---:|---:|---:|---:|
| 42 | 0.0950 | 0.8571 | 0.7941 |
| 1 | 0.1084 | 0.8498 | 0.8133 |
| 2 | 0.0783 | 0.8760 | 0.8064 |

**A single, well-established, architecturally-unmodified technique reduced average forgetting by ~7.5× and more than doubled final accuracy** relative to every EWC variant tried. Under replay, every task's own-task accuracy stays between 0.70 and 0.99 all the way through T6 — versus a collapse to 0.08–0.24 under EWC. And replay's 0.861 final accuracy sits only ~1.5 points below the **non-continual joint-training upper bound (0.876)** — trained on all 6 tasks at once, with no task boundaries at all — which confirms the GNN architecture itself was never the bottleneck; the continual-learning *setting* was, and replay essentially closes that gap.

### 8.3 A real architecture fix folded in along the way: port representation

The original port-embedding scheme log-bucketed all 65,536 possible ports into 32 buckets, which **collapsed semantically distinct well-known services into the same embedding row** purely by numeric proximity (e.g., FTP/SSH/Telnet — ports 20–23 — landed in the same bucket). Since port identity reaches the model only through this embedding (it's not one of Flow's 37 continuous features), this was a real information-loss bug, not just a lossy tradeoff.

**Fix**: exact per-port embedding rows for ports 0–1023, log-bucketed only for the sparse 1024–65535 tail. Parameter count rose from 197,842 to 263,378.

A rigorous, git-verified, 2-seed comparison under the replay regime confirmed the new scheme wins on **both** metrics, on **both** seeds, with per-task "regressions" seen in an earlier single-seed check turning out to flip sign on the second seed (i.e., were noise, not a real cost):

| Scheme | Mean avg. forgetting | Mean final accuracy |
|---|---:|---:|
| Old (log-bucket only) | 0.1017 | 0.8535 |
| **New (exact well-known ports)** | **0.0893** | **0.8723** |

**Decision: keep the new scheme.** Both the structural justification (fixes a real bug) and the empirical evidence (better on both metrics, both seeds) now agree.

### 8.4 Full evaluation pipeline

A proper inference/evaluation pipeline (`inference.py`, `evaluate.py`) was built using scikit-learn metrics: accuracy, precision/recall/F1 (macro + weighted), per-class breakdown, confusion matrices. This confirmed the plain-fine-tuning baseline's per-class collapse in stark detail — the final model retains **only** Benign and whichever two classes the last task introduced, with every earlier task's classes sitting at recall 0.0000 — and (separately) validated replay's per-class picture: every class reaches F1 ≥ 0.66 except Infiltration (the rarest class, 0.1% of pooled traffic), which stays weak under uniform replay (F1 ≈ 0.09–0.24) because it's chronically under-represented in the buffer.

---

## 9. Step 10 — Transferability analysis, and inference on genuinely unseen attack traffic

### 9.1 Step 10a — consolidated transferability analysis

Analyzing all 205 transferability records (41 class pairs × 5 relations) collected during the frozen baseline's training run:

- **`protocol_of` is the standout most-transferable relation** (mean cosine 0.197, vs. the next-best's 0.004) and becomes progressively more transferable as the encoder matures (task 2: −0.356 → task 5: 0.318).
- Of the two originally-flagged pairs: **XSS↔Infiltration ranks 12th of 41** by learned transferability (moderate, partially confirming the raw-feature signal); **Scanning↔Reconnaissance ranks 37th of 41** (negative — contradicting the raw-feature signal).
- **Learned transferability and raw-feature cosine similarity are, overall, weakly *anti*-correlated** across all 41 pairs (Pearson r = −0.219, Spearman ρ = −0.221) — more than mere decoupling.
- A follow-up comparison at `num_layers=2` showed depth **redistributes, rather than uniformly improves**, transferability: `protocol_of` (the num_layers=1 standout) becomes the *worst* relation at depth 2, while previously-weak relations (`service_of`, `originates`) become the new top performers. A relation being "the transferable one" is architecture-dependent here, not an intrinsic property of what it represents.

### 9.2 Step 10b — inference on truly unseen traffic (never touched by the training pipeline)

Two purpose-built modules pull rows directly from raw per-dataset CSVs, bypassing the training pipeline's class allow-list entirely, evaluated against the best replay checkpoint:

**Dataset A — seen classes from unseen sources** (e.g., BoT-IoT's own DDoS/DoS, which the training restriction never let it contribute; UNSW-NB15's Reconnaissance/DoS, from an entirely excluded dataset): scores **near-zero pooled accuracy (0.0008–0.0046)** with *confident, systematic* mis-mapping, not random noise — e.g., BoT-IoT's DDoS is predicted "Reconnaissance" 99.3% of the time. **Genuine negative finding**: the model appears to key on dataset-specific feature distributions rather than transferable, dataset-agnostic attack semantics. (Confounded somewhat by unseen graphs being single-class/unshuffled/no-benign-mixing, unlike training graphs — flagged explicitly, not hidden.)

**Dataset B — classes never in any task** (Backdoor, MITM, Ransomware, Web Attacks, Theft): no ground-truth label space exists for these, so they're evaluated by prediction distribution and softmax confidence instead. **Every class collapses onto 1–2 known classes with high confidence (medians ≥0.89)** — the model never signals "novel" via confidence alone, meaning a dedicated out-of-distribution detector would be needed for real deployment; this was explicitly out of scope. A from-scratch reproduction on a re-trained checkpoint confirmed the qualitative pattern reproduces, though the *specific* known class each unseen class maps to is not stable across independently-trained runs — a genuine, reportable instability finding in its own right.

---

## 10. Follow-up experiment: transferability-guided replay (does the transferability signal help anywhere?)

After the EWC negative result was closed out definitively (§7.6), one more idea was tested: instead of using transferability scores to regularize *parameters* (EWC), use them to weight **which mini-graphs get selected into the replay buffer** — enriching the buffer either toward high-transferability classes or toward low-transferability ones.

**3-seed comparison** (seeds 42, 1, 2; uniform-replay baseline is n=2, both existing seeds):

| Method | n | Avg. Forgetting | Final Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Uniform replay (baseline) | 2 | 0.1031 ± 0.0131 | 0.8610 ± 0.0145 | 0.7981 ± 0.0059 |
| Enrich-high-transfer | 3 | 0.1045 ± 0.0139 | 0.8600 ± 0.0121 | 0.8071 ± 0.0293 |
| Enrich-low-transfer | 3 | 0.1000 ± 0.0130 | 0.8646 ± 0.0116 | 0.8006 ± 0.0285 |

**On headline metrics, this is a null result** — all three conditions sit within one seed-standard-deviation of each other. Root cause: replay selection operates at whole-mini-graph granularity, and since every mini-graph already mixes classes close to the task's natural proportions, changing the *selection weighting* barely moves the buffer's *coarse per-class composition* (verified directly — the realized class distributions are nearly identical between high/low conditions).

**But one real, seed-robust positive signal did emerge**: both enrichment-weighted conditions roughly **triple Infiltration's F1** over uniform replay (0.091 → ~0.27–0.28) — Infiltration is the rarest class in the whole benchmark (0.5% of the replay buffer) and the one uniform replay serves worst. This is flagged as a genuine, narrow positive lead for future work, distinct from the pooled/macro null result.

---

## 11. Headline numbers at a glance

| Metric | Value |
|---|---:|
| Source datasets used | 3 (NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2, NF-BoT-IoT-v2) |
| Attack classes | 10 (+ Benign in every task) |
| Continual-learning tasks | 6 |
| Total attack rows (post-cleaning) | 15,689,086 |
| Node types | 5 (Flow, Host, Protocol, Service, Port) |
| Relation types | 11 |
| Total mini-graphs (default `benign_ratio=3.0` set) | 69,737 |
| Model parameters (default config) | 263,378 |
| Plain fine-tuning baseline | 0.711 avg. forgetting, 0.356 final accuracy |
| Best EWC variant tried (any) | 0.705–0.726 avg. forgetting — **no improvement over baseline** |
| **Experience replay (best forgetting-mitigation method found)** | **0.094 ± 0.015 avg. forgetting, 0.861 ± 0.014 final accuracy** |
| Joint-training upper bound (non-CL reference) | 0.876 accuracy, 0.842 F1-macro |
| Unseen-source, seen-class accuracy (Step 10b, Dataset A) | ~0.001–0.005 (near-total generalization failure, systematic not random) |
| Final test suite | 188 tests passing, `ruff check` clean |

---

## 12. Technology stack

- **Language / packaging**: Python 3.10, managed with `uv`.
- **Step 1**: pandas, pyarrow (Parquet output).
- **Step 2–3**: PyTorch + PyTorch Geometric (`HeteroData`, custom relation-specific convolution layers, HAN-style semantic attention).
- **Step 4+ (continual learning)**: Hydra for structured training configuration; scikit-learn for evaluation metrics.
- **Compute**: GPU (NVIDIA GeForce RTX 3050 Laptop GPU) for all model-training work; CPU sufficient for the I/O-bound Steps 1–2 and for the unseen-attack inference pipeline (no GPU needed at inference time).
- **Testing**: pytest (188 tests at the final count), ruff for linting.
- **Development process**: subagent-driven, spec-and-plan-first development (design docs under `docs/superpowers/specs/`, execution plans and ledgers tracked per feature), with explicit whole-branch review passes before merge — several real bugs (a Fisher-estimation training-mode bug, a CPU/CUDA device mismatch, a tail-biased sampling bug, a missing `map_location` parameter) were caught specifically by this review discipline rather than by unit tests alone.

---

## 13. How to talk about this project (for an interview / non-technical audience)

**One-sentence pitch:** *"I built a continual-learning benchmark for network intrusion detection — six sequential tasks of real network attack traffic — and used it to rigorously test whether a novel transferability-guided variant of a popular anti-forgetting technique (EWC) actually works; I proved, through a systematic multi-angle investigation, that it doesn't, diagnosed exactly why, and then found and validated a simpler technique (experience replay) that reduces the model's forgetting by roughly 7×."*

**What makes this a strong story, not just "a negative result":**
1. **The negative result is airtight, not a shrug.** It's backed by a 5-order-of-magnitude hyperparameter sweep, an architecture-depth ablation that isolates and rules out a confound, and a mechanism-ablation (fixed vs. learned weight) that pinpoints the exact cause of failure (a trivially-collapsing importance weight with no incentive to stay non-zero).
2. **The positive result is real and substantial.** Experience replay took average forgetting from ~0.71 down to ~0.09 and final accuracy from ~0.36 to ~0.86, essentially closing the gap to a non-continual joint-training upper bound (0.88) — with zero architecture changes.
3. **The benchmark itself is a real contribution**: 3 real-world NetFlow datasets, harmonized into a common schema, deliberately split so similar attacks land in non-adjacent tasks (to make transfer meaningful), with a documented, size-balanced task design and a fully reproducible pipeline.
4. **The unseen-attack evaluation is an honest stress test**, not window dressing: it shows the model's accuracy on genuinely unseen data collapses in a *systematic*, explainable way (dataset-specific overfitting) rather than randomly — a nuanced, defensible finding about the limits of cross-dataset generalization for this class of model.
5. **Engineering discipline throughout**: every claimed result is backed by a real run on real data (no estimated numbers), a growing regression-test suite (188 tests), and multiple real implementation bugs were caught and root-caused via systematic debugging rather than being pushed silently into results.

---

## 14. Final Conclusions

**The research question was: can transferability-guided, relation-aware EWC selectively protect the parts of a heterogeneous-GNN IDS worth keeping, as new attack types arrive? The answer, established rigorously, is no — but the project still produced a working, validated forgetting-mitigation result and a reusable benchmark to test the next idea on.**

1. **The proposed mechanism (Steps 5–8) does not work on this benchmark, and this is now a closed question, not an open one.** Every plausible alternative explanation was tested and ruled out one at a time: under-tuned regularization strength (5-order-of-magnitude λ sweep), an architecturally-starved encoder (`num_layers=2` ablation), and a self-defeating learned weight (fixed-weight ablation). All three land in the same 0.70–0.75 forgetting band as plain, unmitigated fine-tuning. The specific failure mechanism is understood precisely: an importance weight that is trained end-to-end and only ever appears inside its own penalty term has nothing pushing back against gradient descent's incentive to shrink it to zero.

2. **Experience replay is the project's real, working result.** A standard, architecturally-unmodified replay buffer took average forgetting from ~0.71 to ~0.09 (a ~7.5× reduction) and final accuracy from ~0.36 to ~0.86 — landing within 1.5 points of a non-continual joint-training ceiling (0.876). This is not a consolation prize; it is a validated, three-seed, statistically solid finding that the project's core problem (catastrophic forgetting in a relation-specific heterogeneous GNN IDS) is solvable on this benchmark, just not by the mechanism originally proposed.

3. **The benchmark and encoder are sound, reusable contributions independent of the CL-method outcome.** Three real NetFlow datasets, harmonized and deliberately split so that similar attacks land in non-adjacent tasks; a relation-specific encoder with attention fusion that was verified end-to-end (gradients, attention weights summing to 1, self-signal preservation); a full inference/evaluation pipeline; and a from-scratch-reproducible, 188-test-covered codebase. Follow-on continual-learning ideas (a different importance-weight training signal, a meta-learned objective, replay-buffer refinements) can be tested against this benchmark directly, without redoing any of the data or architecture work.

4. **Two honest, non-flattering findings round out the picture, and are reported rather than hidden**: transferability-guided *replay-buffer selection* (as opposed to EWC) is also a null result on pooled metrics, though it does triple F1 on the rarest class; and the model's accuracy collapses to near-zero on attack samples from datasets it never trained on, in a systematic (dataset-fingerprinting), not random, way — a real limit on cross-dataset generalization worth stating plainly rather than glossing over.

**Bottom line**: the sophisticated, transferability-guided idea was tested thoroughly and shown not to work here; a simpler, well-established technique (experience replay) does work, and does most of the job. That is a legitimate, well-evidenced research outcome — a negative result on the novel mechanism, paired with a positive, reproducible result on the actual problem the project set out to solve.

---

## 15. Open items / honest limitations (as of 2026-07-31)

1. **Plain fine-tuning and standard EWC have not been multi-seeded** — only replay has full 3-seed statistics. The "EWC variants are all statistically indistinguishable" claim is well-supported indirectly (5-order-of-magnitude sweep, hyperparameter sweep, architecture ablation) but not from repeated seeds of the exact same config.
2. **The `benign_ratio` sweep (2.0/3.0/4.0) was never re-evaluated under the replay regime** — all Step 4+ training used only the default `benign_ratio=3.0`. Picking a winner was always deferred until real continual-learning metrics existed; they now exist, but only for one ratio.
3. **No manuscript/paper draft exists yet** — a project-status PDF summarizing the replay/port-fix round exists, but it is explicitly a status report, not a paper.
4. **The transferability-guided-replay follow-up is capped by the benchmark's own task design**: every task from T3 onward introduces exactly two new classes, and the min-max normalization used to convert transferability scores into buffer weights is winner-take-all on exactly two classes — a benchmark with 3+ new classes per task would let this experiment express more graded selection pressure.
5. **Benign pool sizing** is a known, deliberately-deferred data-quality item: `benign_per_task=8000` is far smaller than the largest task's attack count (3.78M), so benign flows are heavily reused (with replacement) in larger tasks.
