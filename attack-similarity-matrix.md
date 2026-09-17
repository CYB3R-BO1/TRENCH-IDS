# Attack-Class Similarity Matrix (10×10)

**Purpose:** raw input for task grouping — lets you eyeball which attack classes are behaviorally close (high cosine similarity, must land in *different* tasks) versus already dissimilar (safe to bundle together).

**Scope:** 3 datasets (NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2, NF-BoT-IoT-v2), 10 attack classes — the richer benchmark reinstated (2026-07-13, second round) after the professor's guidance changed from minimizing complexity back to building a richer continual-learning benchmark. NF-UNSW-NB15-v2 stays excluded — see `docs/attack-class-counts.md` for why (no class survives dropping it once NF-BoT-IoT-v2 is back in the pool).

**Method** (matches `docs/dataset-plan.md` §2.2 / `src/trench_ids/similarity.py`):
1. ~5,000 rows sampled per class (exact, no replacement) from the 3 datasets, each class restricted to its sanctioned source dataset(s) via `CLASS_DATASETS` (`src/trench_ids/labels.py`) — e.g. BoT-IoT's own `DDoS`/`DoS` rows are never pooled into those classes; BoT-IoT's only sanctioned contribution is Reconnaissance. Candidate pool: the **10 classes** with enough usable samples once each is source-restricted (`docs/attack-class-counts.md`).
2. Per-class **mean** feature vector over the 37 flow-statistic features (excludes IP/port identifiers and `PROTOCOL`/`L7_PROTO`, which seed Step 2 graph-node identity instead).
3. The 10 class-mean vectors are **z-score standardized across each other** (not across the underlying samples) — the reference distribution for scaling is between-class.
4. Pairwise cosine similarity between the standardized means.

**Task assignment is isolate-and-bundle** (`src/trench_ids/task_design.py`): a set of classes that are *all* pairwise above a similarity threshold (a "conflict clique") cannot avoid a conflicting co-location no matter how it's split, so every member gets its own singleton task; the rest are paired via minimum-weight matching, still respecting the threshold.

Regenerate with: `.venv/Scripts/python.exe -m trench_ids.similarity --config configs/similarity.yaml` then `.venv/Scripts/python.exe -m trench_ids.task_design --config configs/similarity.yaml --threshold 0.35` (writes `data/similarity/similarity_matrix.csv` / `similarity_pairs.csv` / `task_assignment.csv`, which this file mirrors).

## Matrix

|  |Scanning|DDoS|Reconnaissance|XSS|DoS|Password|Injection|Bot|BruteForce|Infiltration|
|---|---|---|---|---|---|---|---|---|---|---|
| **Scanning** |1.000|-0.291|0.628|0.148|-0.166|-0.363|-0.334|0.009|-0.532|0.401|
| **DDoS** |-0.291|1.000|-0.154|-0.361|-0.183|0.021|-0.058|0.193|-0.005|-0.462|
| **Reconnaissance** |0.628|-0.154|1.000|-0.096|-0.057|-0.195|-0.383|-0.023|-0.451|0.197|
| **XSS** |0.148|-0.361|-0.096|1.000|0.235|-0.301|-0.028|-0.252|-0.390|0.707|
| **DoS** |-0.166|-0.183|-0.057|0.235|1.000|-0.137|-0.343|-0.415|0.022|0.161|
| **Password** |-0.363|0.021|-0.195|-0.301|-0.137|1.000|0.468|-0.046|0.045|-0.457|
| **Injection** |-0.334|-0.058|-0.383|-0.028|-0.343|0.468|1.000|-0.066|0.062|-0.186|
| **Bot** |0.009|0.193|-0.023|-0.252|-0.415|-0.046|-0.066|1.000|-0.297|-0.075|
| **BruteForce** |-0.532|-0.005|-0.451|-0.390|0.022|0.045|0.062|-0.297|1.000|-0.516|
| **Infiltration** |0.401|-0.462|0.197|0.707|0.161|-0.457|-0.186|-0.075|-0.516|1.000|

## All pairs, sorted by cosine similarity descending

| Rank | Class A | Class B | Cosine similarity |
|---|---|---|---:|
| 1 | XSS | Infiltration | 0.707 |
| 2 | Scanning | Reconnaissance | 0.628 |
| 3 | Password | Injection | 0.468 |
| 4 | Scanning | Infiltration | 0.401 |
| 5 | XSS | DoS | 0.235 |
| 6 | Reconnaissance | Infiltration | 0.197 |
| 7 | DDoS | Bot | 0.193 |
| 8 | DoS | Infiltration | 0.161 |
| 9 | Scanning | XSS | 0.148 |
| 10 | Injection | BruteForce | 0.062 |
| 11 | Password | BruteForce | 0.045 |
| 12 | DoS | BruteForce | 0.022 |
| 13 | Password | DDoS | 0.021 |
| 14 | Scanning | Bot | 0.009 |
| 15 | DDoS | BruteForce | -0.005 |
| 16 | Reconnaissance | Bot | -0.023 |
| 17 | XSS | Injection | -0.028 |
| 18 | Password | Bot | -0.046 |
| 19 | DoS | Reconnaissance | -0.057 |
| 20 | DDoS | Injection | -0.058 |
| 21 | Injection | Bot | -0.066 |
| 22 | Bot | Infiltration | -0.075 |
| 23 | XSS | Reconnaissance | -0.096 |
| 24 | Password | DoS | -0.137 |
| 25 | DDoS | Reconnaissance | -0.154 |
| 26 | Scanning | DoS | -0.166 |
| 27 | DDoS | DoS | -0.183 |
| 28 | Injection | Infiltration | -0.186 |
| 29 | Password | Reconnaissance | -0.195 |
| 30 | XSS | Bot | -0.252 |
| 31 | Scanning | DDoS | -0.291 |
| 32 | Bot | BruteForce | -0.297 |
| 33 | XSS | Password | -0.301 |
| 34 | Scanning | Injection | -0.334 |
| 35 | DoS | Injection | -0.343 |
| 36 | XSS | DDoS | -0.361 |
| 37 | Scanning | Password | -0.363 |
| 38 | Injection | Reconnaissance | -0.383 |
| 39 | XSS | BruteForce | -0.390 |
| 40 | DoS | Bot | -0.415 |
| 41 | Reconnaissance | BruteForce | -0.451 |
| 42 | Password | Infiltration | -0.457 |
| 43 | DDoS | Infiltration | -0.462 |
| 44 | BruteForce | Infiltration | -0.516 |
| 45 | Scanning | BruteForce | -0.532 |

**XSS + Infiltration (0.707) is the single highest cosine pair in the whole matrix**, followed by Scanning + Reconnaissance (0.628) and Password + Injection (0.468). Both of the top two pairs land in different, non-adjacent tasks in the locked design (T6 vs. T3, T1 vs. T2 respectively) — see the task table below — giving the transferability-estimation module (Step 5) substantially more signal than the prior 6-class design's single positive pair (Password+Injection, 0.420).

## Threshold scan (0.15–0.55)

Isolate-and-bundle grouping recomputed at each threshold (`trench_ids.task_design.assign_groups`). The grouping changes once (at 0.21) and is then **stable for the entire 0.21–0.55 range** — a much wider stable band than the prior 6-class design's 0.30–0.40.

| Threshold | Tasks | Groups |
|---:|---:|---|
| 0.15 | 6 | Scanning \| Reconnaissance \| Infiltration \| XSS+Password+DDoS \| DoS+Injection \| Bot+BruteForce |
| 0.17 | 6 | Scanning \| Reconnaissance \| Infiltration \| XSS+Password+DDoS \| DoS+Injection \| Bot+BruteForce |
| 0.19 | 6 | Scanning \| Reconnaissance \| Infiltration \| XSS+Password+DDoS \| DoS+Injection \| Bot+BruteForce |
| **0.21** | 6 | **Scanning \| Reconnaissance \| XSS+DDoS \| Password+Infiltration \| DoS+Injection \| Bot+BruteForce** |
| 0.23–0.55 | 6 | Scanning \| Reconnaissance \| XSS+DDoS \| Password+Infiltration \| DoS+Injection \| Bot+BruteForce |

**Note:** the threshold scan above shows the raw minimum-weight grouping (`sizes=None` in `task_design.min_weight_grouping`), i.e.\ total-similarity-only optimization. The **locked design** uses the size-aware tie-break (`sizes=class_counts`), which among equally-balanced valid pairings picks the option minimizing the largest task's size: T3=DDoS+Infiltration (−0.462), T4=DoS+Injection (−0.343), T5=Password+Bot (−0.046), T6=XSS+BruteForce (−0.390). See `docs/dataset-plan.md` §2.2 and `src/trench_ids/task_design.py` `min_weight_grouping(..., sizes=...)`.

## Task assignment (from `src/trench_ids/labels.py`, τ = 0.35)

| Task | Classes | Max intra-task cosine similarity | Contributing dataset(s) |
|---|---|---|---|
| T1 | Scanning | — (isolated) | ToN |
| T2 | Reconnaissance | — (isolated) | BoT |
| T3 | DDoS + Infiltration | −0.462 | ToN, CSE |
| T4 | DoS + Injection | −0.343 | ToN, CSE |
| T5 | Password + Bot | −0.046 | ToN, CSE |
| T6 | XSS + BruteForce | −0.390 | ToN, CSE |

If you want to manually override any grouping, use the matrix above to check the new grouping doesn't reintroduce a high-similarity co-location — then update `CANONICAL_TO_TASK` / `TASK_THEMES` / `TASK_DATASETS` in `labels.py` (and rerun `trench-preprocess`) to match.
