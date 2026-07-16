# Attack Class Sample Counts (3 Datasets, Per-Class Source-Restricted)

**Purpose:** Input for the similarity analysis - selecting the candidate attack classes by sample
count before computing per-class centroids / cosine similarity for task assignment.

**Scope:** 3 datasets (NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2, NF-BoT-IoT-v2), per the professor's
revised guidance (2026-07-13, second round) to build a richer continual-learning benchmark instead
of minimizing complexity. NF-UNSW-NB15-v2 stays excluded - see "Why UNSW-NB15 is excluded" below
for the (unchanged) empirical justification.

**Per-class dataset restriction:** each class's count below is only summed over the dataset(s)
listed in `src/trench_ids/labels.py`'s `CLASS_DATASETS` for that class, not every dataset the
canonical label happens to appear in. This matters because BoT-IoT independently contains
`DDoS`-labeled (18,331,847) and `DoS`-labeled (16,673,183) rows - both excluded from the DDoS/DoS
counts below because BoT-IoT's only sanctioned contribution to this benchmark is Reconnaissance.

**Source:** `trench_ids.similarity.count_classes` against `configs/similarity.yaml` (exact
streaming count over the three retained datasets' `Attack` column, restricted per `CLASS_DATASETS`,
2026-07-14).

**Scope:** Attack classes only (Benign excluded - it's the shared negative class present in every
task, not a task/class under consideration for the similarity analysis).

| Rank | Canonical class | `CLASS_DATASETS` source(s) | Combined count (measured) | Task |
|---:|---|---|---:|---|
| 1 | Scanning | ToN | 3,781,419 | T1 |
| 2 | DDoS | ToN, CSE | 3,416,504 | T3 |
| 3 | Reconnaissance | BoT | 2,620,999 | T2 |
| 4 | XSS | ToN | 2,455,020 | T3 |
| 5 | DoS | ToN, CSE | 1,196,608 | T5 |
| 6 | Password | ToN | 1,153,323 | T4 |
| 7 | Injection | ToN, CSE | 684,897 | T5 |
| 8 | Bot | CSE | 143,097 | T6 |
| 9 | BruteForce | CSE | 120,912 | T6 |
| 10 | Infiltration | CSE | 116,361 | T4 |

**Total (10-class candidate pool, per-dataset-restricted):** 15,689,140. This is the source of the
"~15.7M attack rows, uncapped" scale-up figure in `docs/dataset-plan.md` §3.3 - removing
`attack_per_class_cap` (see `docs/superpowers/specs/2026-07-13-dataset-task-respec-design.md` §3)
means Step 1 now keeps every one of these rows in full, rather than Bernoulli-subsampling the
abundant classes down to a cap.

Dropped classes and their (unrestricted-count) reference values, for completeness - not part of the
candidate pool:

| Canonical class | Source dataset | Count | Reason dropped |
|---|---|---:|---|
| Backdoor | ToN | 16,809 | below candidate-pool floor |
| MITM | ToN | 7,723 | below candidate-pool floor |
| Ransomware | ToN | 3,425 | below candidate-pool floor |
| Web Attacks | CSE | 3,070 | below candidate-pool floor |
| Theft | BoT | 2,431 | below candidate-pool floor (BoT-IoT's only other class; not sanctioned for the pool) |

For reference, Benign totals (unrestricted, all three retained datasets) are 6,099,469 (ToN) +
16,635,567 (CSE) + 135,037 (BoT) = 22,870,073 combined, not included in the ranking above. Step 1's
per-task benign pool is drawn fresh per task from that task's contributing dataset(s) only (see
`docs/dataset-plan.md` §2.1).

## Candidate pool: 10 classes

**Decision: keep all 10 classes above the candidate-pool floor** (Scanning through Infiltration).
The pool splits into two bands - seven classes from 685K-3.8M, and three smaller classes
(Bot/BruteForce/Infiltration) from 116K-143K - but all ten comfortably clear the "enough usable
samples for a stratified train/val/test split" bar. Reasoning:

- Matches the professor's revised guidance (2026-07-13, second round): build a richer benchmark - ~6
  tasks, more attack classes, no artificial pre-graph sampling cap - rather than minimizing
  complexity as the prior (superseded) 2-dataset/6-class/4-task design did.
- The 10-class pool's conflict structure resolves into a 6-task isolate-and-bundle grouping
  (`trench_ids.task_design`), stable across similarity thresholds **0.21-0.55** (see
  `docs/attack-similarity-matrix.md`) - a wider stable band than the prior 6-class design's
  0.30-0.40.
- The pool spans a much wider range of similarity behavior than the 6-class pool did: the highest
  cosine pair in the whole matrix is XSS<->Infiltration (0.706), landing in different, non-adjacent
  tasks (T3 vs. T4); the second-highest is Scanning<->Reconnaissance (0.628), also isolated from
  each other (T1 vs. T2) - giving the transferability-estimation module (Step 5) much more signal to
  detect than the prior design's single positive pair (Password+Injection, 0.420).

**Selected 10:** Scanning, DDoS, Reconnaissance, XSS, DoS, Password, Injection, Bot, BruteForce,
Infiltration.

**Dropped:** Backdoor, MITM, Ransomware (ToN), Web Attacks (CSE), Theft (BoT) - all below the
candidate-pool floor (2,431-16,809 vs. the pool's 116K-15.7M range).

## Why Bot, BruteForce, and Infiltration are included now

The **prior** (superseded) 2-dataset/6-class/4-task design dropped Bot, BruteForce, and Infiltration
(116K-143K combined) as "below threshold" against the six classes it kept (685K-3.8M). That
reasoning is reversed here: the professor's guidance changed from minimizing complexity to building
a richer benchmark, and all three classes comfortably clear the "enough usable samples for
train/val/test" bar - an order of magnitude below the other seven, but still far above a viable
per-class sample floor (5,000 for the similarity analysis; well over 100K for the full task itself).
Including them is a **benchmark-richness decision, not a data change**: nothing about these three
classes' counts moved between the two designs - only the professor's complexity-vs-richness guidance
did. Including them also completes a sixth task (Bot + BruteForce, T6) and keeps the isolate-and-
bundle grouping's stable threshold band wide (0.21-0.55) rather than narrow.

## Why UNSW-NB15 is excluded

UNSW-NB15's 10-class-pool-relevant contribution depended entirely on **Reconnaissance** (2,633,778
combined across UNSW+BoT-IoT) and a marginal slice of DoS (UNSW contributed just 5,794 of the
unrestricted DoS total). Checking the per-dataset breakdown:

- **Reconnaissance**: UNSW contributes only 12,779 of the combined 2,633,778 - the rest (2,620,999,
  99.5%) comes from BoT-IoT alone. With BoT-IoT back in the pool (this design), Reconnaissance is
  fully covered by BoT-IoT; UNSW's 12,779 rows add nothing that changes the candidate pool or task
  structure.
- **Exploits, Fuzzers, Generic, Analysis, Shellcode, Worms**: UNSW-only classes, all well under 100K
  combined - none clears the candidate-pool floor even considered alone.
- **DoS**: UNSW's 5,794 contribution is a rounding error against ToN+CSE's already-substantial
  1,196,608 (CLASS_DATASETS-restricted; BoT-IoT's own DoS rows are excluded from this class - see
  the restriction note above).

So UNSW-NB15 contributes **no class that survives dropping it, once BoT-IoT is back in the pool** -
adding it back would mean a whole extra dataset's preprocessing complexity for zero new tasks or
classes. This is the same conclusion the prior 2-dataset design reached (there, in the other
direction: UNSW added nothing once BoT-IoT was *also* dropped) - restated here because it still
holds with BoT-IoT reinstated.
