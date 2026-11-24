# TRENCH Task-Sequence Design — Final Detailed Specification (Draft)

- **Status:** **Draft final detailed specification** — pending reviewer approval. Step 1 is **not yet**
  declared complete; the user has decided to proceed with this detailed specification, and will approve it
  before Step 1 completion is declared.
- **Source evidence:** frozen cross-dataset comparison (`outputs/cross_dataset_analysis/`) and the frozen
  per-dataset artifacts it draws from:
  - UNSW: `outputs/unsw_full/`, `outputs/unsw_uncertainty/`, `outputs/unsw_window_sweep/`,
    `outputs/unsw_rate_ablation/`, `outputs/diagnostics/`.
  - ToN: `outputs/ton_full/`, `outputs/ton_class_contiguous/`, `outputs/ton_class_contiguous_benignmatched/`,
    `outputs/ton_uncertainty/`, `outputs/ton_window_sweep/`.
- **Scope:** research objective, design principles, formal task definitions, candidate sequences, benign
  allocation, sample/window allocation, train/val/test protocol, forward-transfer & forgetting evaluation,
  controls, feasibility constraints, limitations, reproducibility.

---

## 1. Research objective and design principles

### 1.1 Research objective

The central TRENCH-IDS research question is **forward transfer across attack classes**: whether a
representation or behavior learned on one attack class supports learning of a new, previously unseen
attack class, and whether the observed similarity is *causal* or merely *associative*.

The task-sequence design must answer, with experimental conditions:

1. **Similar → new:** Does learning a similar attack class help learn a new attack? (High-similarity conditions.)
2. **Dissimilar → new:** Does learning a dissimilar attack class provide any forward-transfer benefit? (Low-similarity conditions.)
3. **Relation-specific:** Does the similarity of a particular graph relation predict its contribution to
   forward transfer? (Relation-specific conditions.)

These three must remain the organizing questions. The design must not become a mere collection of
attack-class pairs for its own sake.

### 1.2 Design principles

1. **No automatic clustering** of similar attacks into the same task.
2. **Unseen-class transitions must remain possible**; transferability is tested through task sequences that
   introduce new attack classes over previously learned ones.
3. All candidate sequences respect dataset-specific temporal structure and class-feasibility constraints.
4. Normal traffic is handled per task with matched benign subsets (allocation policy in §4).
5. Bootstrap uncertainty must be respected (UNSW primary block 900 s; ToN primary 30 min) and unresolved
   cases flagged (UNSW Generic; ToN Injection).

---

## 2. Formal task definitions

- **Task k:** the designated attack class is introduced as a discriminative task over matched benign
  traffic; other attack classes are excluded from the task graph; previously introduced classes are not
  replayed unless replay is explicitly evaluated.

- **Task graph:** `G_k = (V_k, E_k)` contains the designated attack-class flows from the task's training
  windows plus the benign subset allocated to task `k`. Evaluation windows are held-out windows of the
  designated class with their matched benign context; all other attack classes are excluded.

- **Pair condition (A → B):** task A followed by task B. Forward transfer is evaluated as the benefit of
  the A-trained state for learning B; forgetting is evaluated as retention of A after B.

---

## 3. Candidate sequences for both datasets

### 3.1 Corrected candidate matrix

| Condition | UNSW | ToN-IoT |
|---|---|---|
| High similarity | DoS → Fuzzers | Injection → Password |
| Secondary high similarity | — | Injection → XSS |
| Low similarity | Generic → Exploits | DoS → Backdoor |
| Relation-specific | Exploits → Fuzzers | Injection → DDoS |
| Random-order control | Seeded permutations of 5 eligible classes | Seeded permutations of 7 eligible classes |

### 3.2 Per-dataset detail and decisions

**NF-UNSW-NB15-v3** (eligible classes: DoS, Exploits, Fuzzers, Generic, Reconnaissance):

| condition | sequence | status | basis |
|---|---|---|---|
| High similarity | DoS → Fuzzers | **Approved provisionally** | Cosine 0.986; bootstrap verified at final 900 s config (top-5 prob 0.754, rank mean 4.02, 95% CI [0.949, 0.991]); CI does not include 1.0 membership, so kept **provisional**. |
| Secondary high | — | n/a | — |
| Low similarity | Generic → Exploits | Approved | Cosine 0.539 (lowest among eligible); flag Generic unresolved long-range dependence (min block 1,140 s > 900 s primary). |
| Relation-specific | Exploits → Fuzzers | Approved candidate | Block-level host-degree similarity (≈0.002/0.055) with port-level divergence (src ≈ 0.871, dst ≈ 0.643); approved as the relation-contrast candidate. |
| Random control | Seeded permutations of 5 eligible classes | Redesign | Multiple permutations, comparable budgets. |
| Order reversal | Selected pairs, both directions | Redesign | Order-reversal probes + random-order controls are separate. |

**NF-ToN-IoT-v3** (eligible classes: Backdoor, ddos, dos, injection, password, scanning, xss):

| condition | sequence | status | basis |
|---|---|---|---|
| High similarity | Injection → Password | Approved | Cosine 0.936; relation 0.709; flags 0.996; bootstrap top-5 1.00 at 30 min; both eligible. |
| Secondary high | Injection → XSS | Pending approval | Cosine 0.909; relation 0.699; optional second contrast. |
| Low similarity | DoS → Backdoor | Approved (order reversed) | First-occurrence order respected (DoS precedes Backdoor); cosine symmetric at 0.169. |
| Relation-specific | Injection → DDoS | Approved (final direction) | Sourced from screened relation evidence: relation similarity 0.664 vs cosine 0.336; ToN relation-block JS distance 0.387 (defined); strongest relation/low-cosine contrast candidate; direction set to chronology-respecting **Injection → DDoS**. |
| Random control | Seeded permutations of 7 eligible classes | Redesign | Synthetic class-order controls; individual temporal order preserved, introduction sequence not necessarily chronological. |
| Order reversal | Selected pairs, both directions | Redesign | Order-reversal probes + random-order controls are separate. |

### 3.3 Final screening-pass confirmations (recap)

1. UNSW DoS → Fuzzers bootstrap stability at final 900 s: verified (top-5 0.754; CI [0.949, 0.991]) — **provisional**.
2. UNSW Exploits → Fuzzers relation contrast: confirmed (host-similarity vs port-divergence).
3. ToN relation contrast: **Injection → DDoS** confirmed via screened relation evidence.

---

## 4. Normal-traffic allocation policy (primary, as agreed — preserved)

- Each task receives its designated attack class and a **matched benign subset**.
- Benign windows are allocated to **at most one task** (disjoint benign allocation).
  - UNSW: benign windows must be audited and allocated without overlap across tasks.
  - ToN: benign-matching audit already provides the foundation (matched by window membership).
- Other attack classes are excluded from the task graph.
- Previously introduced attack classes are **not replayed** unless replay is explicitly being evaluated.
- **No repeated Normal exposure** in primary tasks.
- A **separate repeated-Normal-exposure ablation** is retained.

---

## 5. Actual sample and graph-window allocation numbers

All numbers below are taken from frozen experiment artifacts; **no balanced/invented sample sizes were
created**. Where per-task benign/attack splits are required for pair conditions, they are derived from the
task-filtered window statistics and benign-matching artifacts, not invented.

### 5.1 UNSW (global chronological split; train partition)

| quantity | value | source |
|---|---|---|
| Train benign flows | 1,590,548 | `unsw_full/` |
| Train attack flows | 65,181 | `unsw_full/` |
| Train benign:attack ratio | ≈ 24.4 : 1 | `unsw_full/` |
| Task-filtered (benign + designated) train windows | ≈ 6,197 @ 10 s; ≈ 12,275 @ 5 s | `unsw_window_sweep/task_filtered_window_statistics.csv` |
| Median designated-class flows per task graph @ 10 s (high-volume classes) | Exploits 7, Fuzzers 11, Generic 4, Reconnaissance 3 | `unsw_window_sweep/` |
| Strong-supervision windows (≥20 designated-class flows) @ 10 s | Exploits 155, Fuzzers 239, Generic 40 | `unsw_window_sweep/window_5s_vs_10s_comparison.csv` |

### 5.2 ToN (per-class contiguous chronological split, benign-matched; train partition)

| quantity | value | source |
|---|---|---|
| Train benign flows | 14,732,002 | `ton_class_contiguous_benignmatched/` |
| Train attack flows | 7,453,293 | `ton_class_contiguous_benignmatched/` |
| Train benign:attack ratio | ≈ 1.98 : 1 | `ton_class_contiguous_benignmatched/` |
| Per-class benign/attack (train) | injection 67,149 / 266,120; xss 283,026 / 1,977,968; mitm 1,755,629 / 4,140; scanning 8,327,687 / 905,605 | `ton_class_contiguous_benignmatched/benign_task_matching.csv` |
| Task-filtered train windows (≥5 class flows) | ddos 27,798 @1 s → 553 @60 s; password 36,394 → 648; xss 19,446 → 334; injection 5,452 → 141; dos 14,348 → 368; scanning 5,452 → 425; Backdoor 2,583 → 1,176 (3 windows ≥10 @1 s) | `ton_window_sweep/task_filtered_window_statistics.csv` |
| Median flows/window (train) by window size | 1 s: 27; 5 s: 110; 10 s: 202; 30 s: 514; 60 s: 943 | `ton_window_sweep/` |

### 5.3 Allocation notes

- The benign:attack imbalance is extreme and class-dependent (notably for ToN).
- ToN attack classes are temporally disjoint → 0 windows train for more than one class; benign allocation
  is therefore largely non-overlapping by construction, but the matched-benign protocol still assigns a
  per-task benign subset.
- No balanced sample sizes were invented; task graphs are built from designated-class + matched benign only.

---

## 6. Training, validation and test protocol

- Per task: training on the task graph (designated class + matched benign) from the task's training
  windows; validation on the task's validation windows; test on the task's held-out windows.
- Pair conditions run as sequential tasks (A then B).
- Training budgets must be comparable across conditions within each dataset.
- Window/bootstrap configuration for training graphs: primary 10 s / sensitivity 5 s / exploratory 30 s
  (UNSW); ToN task graphs use the per-class contiguous split with the chosen bootstrap block lengths
  (UNSW primary 900 s; ToN primary 30 min / sensitivity 60 min).
- Bootstrap intervals from different block configurations are not pooled across datasets (different
  estimands); they inform uncertainty, not task design directly.

---

## 7. Forward-transfer and forgetting evaluation

- **Forward transfer:** train task A → evaluate on task B (unseen class). High-similarity conditions
  (DoS → Fuzzers; Injection → Password) test whether similar-class learning helps; low-similarity
  conditions (Generic → Exploits; DoS → Backdoor) test whether dissimilar-class learning yields any
  benefit; relation-specific conditions (Exploits → Fuzzers; Injection → DDoS) test whether a particular
  graph relation drives transfer.
- **Forgetting:** after task B, assess retention of task A (replay not assumed; replay is an explicit
  optional ablation).
- The relation-specific axis uses the screened relation structure (e.g., Exploits → Fuzzers: host-degree
  similarity with port-level divergence; Injection → DDoS: relation 0.664 vs cosine 0.336).

---

## 8. Order-reversal and random-order controls

- **Order-reversal probes:** both directions of selected pairs (e.g., Injection → Password / Password →
  Injection; DoS → Backdoor / Backdoor → DoS; DoS → Fuzzers / Fuzzers → DoS).
- **Random-order controls:** multiple seeded permutations of the eligible class set (UNSW: 5 classes; ToN:
  7 classes); same classes and comparable training budgets; ToN controls explicitly labeled synthetic
  class-order controls. Pairwise transition experiments are defined separately from full multi-class
  random-order sequences.

---

## 9. Feasibility constraints and exclusions

- Only primary-eligible classes are used in candidate task sequences:
  - UNSW: DoS, Exploits, Fuzzers, Generic, Reconnaissance.
  - ToN: Backdoor, ddos, dos, injection, password, scanning, xss.
- Supplementary classes are excluded from candidate sequences pending explicit justification:
  - UNSW: Backdoor, Shellcode (train-supported, floor-failing).
  - ToN: mitm, ransomware (train-supported, floor-failing).
- Low-training-support classes are excluded: UNSW — Analysis, Worms.
- UNSW Generic carries unresolved long-range temporal dependence (min block 1,140 s > 900 s primary).
- ToN Injection carries limited bootstrap blocks (6 @ 30 min primary; 4 @ 60 min sensitivity) → cautious
  interpretation.

---

## 10. Limitations and reproducibility configuration

- Different split estimands across datasets (UNSW global chronological; ToN per-class contiguous) →
  descriptive, not pooled.
- Only Backdoor and DoS/dos treated as shared concepts; raw labels preserved; Reconnaissance/Scanning and
  ddos/dos not silently merged.
- ToN populations: attack-present vs benign-only where available; UNSW attack-present only.
- Bootstrap intervals not pooled across datasets.
- Single-seed bootstrap caveats; window-size differences.
- Reproducibility: frozen window config (10 s primary / 5 s sensitivity / 30 s exploratory), frozen
  bootstrap blocks (UNSW 900 s / 300 s; ToN 30 min / 60 min), seeds, split manifests, artifact paths.

---

## Deliverables status and next step

- Cross-dataset comparison: **complete** (after the three report corrections were applied and verified).
- Final candidate-screening pass: **complete**.
- Final detailed task-sequence specification: **draft** (this document), now containing the corrected
  candidate matrix, preserved normal-traffic policy, central research questions, and required sections.
- **Step 1 declared complete:** pending reviewer approval of this final detailed specification.

**Next:** after approval, begin **Step 2 — heterogeneous graph construction**, using the exact traffic
observations, attack classes, benign allocations, and experimental conditions defined here.
