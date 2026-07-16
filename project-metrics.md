# TRENCH-IDS - Project Metrics

This file is the single place to find every hard number about the project - dataset scale, class distribution, task design, pipeline output, and test/runtime stats. Update it whenever Step 1/Step 2 are rerun or the class/task design changes; treat every number here as sourced from an actual file on disk (cited inline), never estimated.

**Benchmark objective:** TRENCH-IDS is a benchmark for **task-incremental continual learning on heterogeneous, graph-structured network intrusion detection data** - 6 sequential tasks, each introducing new attack classes (plus a fresh Benign subset) built from real NetFlow v2 traffic, feeding into a relation-specific heterogeneous GNN trained under continual learning (full proposed pipeline: `CLAUDE.md` "Research goal").

### Benchmark at a glance

| Metric | Value |
|---|---:|
| Source datasets | 3 (NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2, NF-BoT-IoT-v2) |
| Attack classes | 10 |
| Continual-learning tasks | 6 |
| Total attack rows (post corrupted-row filter) | 15,689,086 |
| Total rows incl. benign (Step 1 output) | 15,736,524 |
| Node types | 5 (Flow, Host, Protocol, Service, Port) |
| Relation types | 6 |
| Graph size (max flows/mini-graph) | 300 |
| Total mini-graphs, `benign_ratio=4.0` set | 65,378 |
| `benign_ratio` sweep candidates produced | 2.0 / 3.0 / 4.0 |

### Pipeline overview

```
Raw NetFlow v2 CSVs (5 datasets on disk, 3 used)
        |
        v
Class filtering + label harmonization      labels.py: CLASS_DATASETS, EXCLUDED_CLASSES  (S2)
        |
        v
Task assignment (similarity-driven          similarity.py, task_design.py               (S3-4)
  isolate-and-bundle, 6 tasks)
        |
        v
Step 1: sample + dedup + corrupted-row      preprocess.py -> data/processed/*.parquet    (S5)
  filter + stratified train/val/test split
        |
        v
Step 2: heterogeneous graph construction +  graphs.py -> data/graphs*/task_*_*.pt        (S6)
  benign_ratio sampling + mini-graph chunking
        |
        v
Mini-graphs (list[HeteroData] per task/split, node/relation schema in S6)
        |
        v
Step 3 (not yet started): relation-specific GNN + continual-learning loop
```

---

## 1. Datasets

| Dataset | Code | Status | Role |
|---|---|---|---|
| NF-ToN-IoT-v2 | `ToN` | Used | Scanning, XSS, DDoS, Password, DoS, Injection |
| NF-CSE-CIC-IDS2018-v2 | `CSE` | Used | DDoS, DoS, Injection, Bot, BruteForce, Infiltration |
| NF-BoT-IoT-v2 | `BoT` | Used | Reconnaissance **only** (also contains DDoS/DoS rows, deliberately excluded - see `CLASS_DATASETS`) |
| NF-UNSW-NB15-v2 | - | Excluded | No candidate class survives dropping it once BoT-IoT is back (its only relevant class, Reconnaissance, is 99.5% supplied by BoT-IoT) |
| NF-UQ-NIDS-v2 | - | Excluded | Merged union of the other four - including it would double-count flows |

Source: `configs/preprocess.yaml`, `src/trench_ids/labels.py`.

---

## 2. Candidate class pool (10 classes)

Per-dataset restriction (`CLASS_DATASETS` in `src/trench_ids/labels.py`) and measured raw counts (real full-file scan, `docs/superpowers/specs/2026-07-13-dataset-task-respec-design.md` §1):

| Class | Allowed dataset(s) | Raw count (measured) |
|---|---|---:|
| Scanning | ToN | 3,781,419 |
| DDoS | ToN, CSE | 3,416,504 |
| Reconnaissance | BoT | 2,620,999 |
| XSS | ToN | 2,455,020 |
| DoS | ToN, CSE | 1,196,608 |
| Password | ToN | 1,153,323 |
| Injection | ToN, CSE | 684,897 |
| Bot | CSE | 143,097 |
| BruteForce | CSE | 120,912 |
| Infiltration | CSE | 116,361 |

**Total candidate-pool raw attack rows: 15,689,140** (sum above) - matches the real Step 1 output attack total exactly (§5). Pass 1 and Pass 2 of the real run also matched each other exactly (both reported 15,689,140), which is expected: Pass 2 no longer applies any cap/probability to attack rows, only the same `CLASS_DATASETS` restriction Pass 1 already counted under.

Excluded entirely (`EXCLUDED_CLASSES`): Backdoor, MITM, Ransomware, Web Attacks, Theft (2,431–16,809 rows each, all well below the 116K–3.8M candidate-pool range).

---

## 3. Class-similarity matrix (top pairs)

Cosine similarity between standardized per-class mean feature vectors (37 flow-statistic features, 5,000-row sample/class). Full 10×10 matrix: `similarity_matrix.csv` (spike output, see design spec §1 for provenance). Top and bottom pairs:

| Class A | Class B | Cosine similarity |
|---|---|---:|
| XSS | Infiltration | **0.7066** (highest overall) |
| Scanning | Reconnaissance | 0.6279 |
| Password | Injection | 0.4681 |
| Scanning | Infiltration | 0.4013 |
| XSS | DoS | 0.2346 |
| ... | ... | (45 pairs total) |
| Scanning | BruteForce | -0.5315 (lowest overall) |
| BruteForce | Infiltration | -0.5160 |

Task-design threshold: **0.35**, stable across the entire **0.21–0.55** band (21-point scan, `threshold_scan.csv`).

---

## 4. Continual-learning task design (6 tasks)

Isolate-and-bundle grouping on the 10-class pool, with a **size-aware tie-break** (`src/trench_ids/task_design.py`'s `min_weight_grouping(..., sizes=...)`, adopted 2026-07-14): among all pairings satisfying the 0.35 similarity threshold, the tie-break minimizes the largest resulting task's size instead of total pairwise similarity. This replaced the original min-total-similarity pairing (T3=XSS+DDoS at 5,879,524 rows, T4=Password+Infiltration, T6=Bot+BruteForce), which happened to select the single worst-balanced valid option out of 78 threshold-valid pairings of the 8 non-clique classes. See `docs/superpowers/specs/2026-07-14-benchmark-finalization-design.md` §1 for the full enumeration.

| Task | Attack classes | Dataset(s) | Max intra-task cosine sim |
|---|---|---|---:|
| T1 | Scanning | ToN | - (isolated) |
| T2 | Reconnaissance | BoT | - (isolated) |
| T3 | DDoS + Infiltration | ToN, CSE | -0.462 |
| T4 | DoS + Injection | ToN, CSE | -0.343 |
| T5 | Password + Bot | ToN, CSE | -0.046 |
| T6 | XSS + BruteForce | ToN, CSE | -0.390 |

Benign is present in every task (fresh per-task subset from that task's contributing dataset(s)). Since T1 (ToN-only) and T2 (BoT-only) are unchanged, and T3-T6 each now pair a class from one dataset with a class from the other, **4 of 6 tasks span both datasets** (previously 3 of 6 - T6 was CSE-only under the old pairing).

---

## 5. Step 1 output - `data/processed/` (real run, 2026-07-14, ~27 min)

Source: `data/processed/manifest.json`. `seed=42`, `benign_per_dataset_cap=60000`, `benign_per_task=8000`, split `70/15/15`, dedup on. Row counts below **include the fixed 8,000-benign-per-task addition** (i.e. they're the full assembled task, not just the raw attack rows) - subtract the Benign class-count column to get raw attack rows alone. `corrupted_rows_dropped` is new this round (rows with NaN/±inf/float32-overflow values, filtered before dedup - see §11).

| Task | Theme | Rows (incl. benign) | Class counts | Dups dropped | Corrupted dropped | Train / Val / Test | Sources | File size |
|---|---|---:|---|---:|---:|---|---|---:|
| 1 | Scanning | 3,789,419 | Scanning 3,781,419 / Benign 8,000 | 0 | 0 | 2,652,593 / 568,413 / 568,413 | ToN 3,789,419 | 48.3 MB |
| 2 | Reconnaissance | 2,628,994 | Reconnaissance 2,620,994 / Benign 8,000 | 5 | 0 | 1,840,296 / 394,349 / 394,349 | BoT 2,628,994 | 60.2 MB |
| 3 | DDoS + Infiltration | 3,540,254 | DDoS 3,416,452 / Infiltration 115,804 / Benign 7,998 | 557 | 54 | 2,478,178 / 531,039 / 531,037 | ToN 2,030,232 / CSE 1,510,022 | 79.6 MB |
| 4 | DoS + Injection | 1,889,505 | DoS 1,196,608 / Injection 684,897 / Benign 8,000 | 0 | 0 | 1,322,654 / 283,426 / 283,425 | ToN 1,401,074 / CSE 488,431 | 42.1 MB |
| 5 | Password + Bot | 1,304,420 | Password 1,153,323 / Bot 143,097 / Benign 8,000 | 0 | 0 | 913,094 / 195,663 / 195,663 | ToN 1,157,323 / CSE 147,097 | 26.5 MB |
| 6 | XSS + BruteForce | 2,583,932 | XSS 2,455,020 / BruteForce 120,912 / Benign 8,000 | 0 | 0 | 1,808,752 / 387,590 / 387,590 | ToN 2,459,020 / CSE 124,912 | 63.5 MB |
| **Total** | | **15,736,524** | | **562** | **54** | | | **320 MB** |

Attack-only total across all 6 tasks: **15,689,140 − 54 = 15,689,086** after the corrupted-row filter (all 54 drops came from task 3's DDoS/Infiltration rows, the known CSE-CIC-IDS2018 `SRC_TO_DST_SECOND_BYTES`/`DST_TO_SRC_SECOND_BYTES` overflow issue - see §10). Benign total across tasks: 47,998 (task 3 lost 2 benign rows to dedup, everywhere else the full 8,000 survived).

Task-size imbalance is now **2.9× largest:smallest** (T1 3,789,419 vs. T5 1,304,420) - down from **14×** under the old task pairing (T1 16,807 vs. T6 1,176 *graphs*, i.e. ~4.6M vs ~272K rows) - the direct result of the size-aware tie-break in §4.

Pass-1 raw benign counts (before per-dataset cap): ToN 6,099,469 / CSE 16,635,567 / BoT 135,037. Pass-2 benign pools (after `benign_per_dataset_cap=60000`): ToN 59,614 / CSE 59,847 / BoT 59,948.

---

## 6. Step 2 output - three `benign_ratio` sweep sets (real runs, 2026-07-14)

Per the finalization design (§4), Step 2 was run three times - once per `benign_ratio` candidate - to actually produce all three sweep sets rather than pick one default. Each also applies the new `max_task_ratio=3.0` downsampling safety cap (`_downsample_tasks`); in all three sets `"trimmed": {}` - the cap never triggered, since the rebalanced task table's natural largest:smallest ratio (~2.92×, task 1 vs. task 5) already sits comfortably under 3.0×. Source: `graph_counts.json` in each `out_dir`. `graph_size=300` for all three.

**Node types (5):** Flow, Host, Protocol, Service, Port - features derived from the raw NetFlow v2 fields of the flows in each mini-graph (`src/trench_ids/graphs.py`).

**Relation types (6):** `host--originates-->flow`, `flow--terminates_at-->host`, `flow--targets_port-->port`, `flow--uses_protocol-->protocol`, `flow--uses_service-->service`, `host--communicates_with-->host` (the last is the only host-to-host, non-flow-incident relation - built per mini-graph from shared flows between host pairs).

**What `benign_ratio` means:** `target_benign = n_attack / benign_ratio`, so a graph's benign share works out to `1 / (benign_ratio + 1)`. Concretely: `benign_ratio=4.0` -> ~20% benign / ~80% attack flows per graph; `benign_ratio=3.0` -> 25% benign; `benign_ratio=2.0` -> ~33% benign. §8.3 confirms this empirically - Benign sits at ~20.0% in every task's `benign_ratio=4.0` mini-graphs.

### 6.1 `benign_ratio=3.0` - `data/graphs/` (`configs/graph.yaml`, ~58 min)

| Task | Train | Val | Test | Total | Disk (train/val/test) |
|---|---:|---:|---:|---:|---|
| 1 Scanning | 11,765 | 2,521 | 2,521 | 16,807 | 1002 MB / 214 MB / 215 MB |
| 2 Reconnaissance | 8,155 | 1,748 | 1,748 | 11,651 | 665 MB / 142 MB / 142 MB |
| 3 DDoS + Infiltration | 10,990 | 2,355 | 2,355 | 15,700 | 932 MB / 199 MB / 199 MB |
| 4 DoS + Injection | 5,854 | 1,255 | 1,255 | 8,364 | 496 MB / 106 MB / 106 MB |
| 5 Password + Bot | 4,034 | 865 | 865 | 5,764 | 341 MB / 73 MB / 73 MB |
| 6 XSS + BruteForce | 8,015 | 1,718 | 1,718 | 11,451 | 674 MB / 144 MB / 144 MB |
| **Total** | **48,813** | **10,462** | **10,462** | **69,737** | **~5.73 GB** |

Downsampling: `min_task_total=5,764` (task 5), `cap=17,292` - task 1 (16,807) is closest to the cap but stays under it; nothing trimmed.

### 6.2 `benign_ratio=2.0` - `data/graphs_ratio2/` (`configs/graph_ratio2.yaml`, ~61 min)

Lower `benign_ratio` = more benign per attack (`target_benign = n_attack / benign_ratio`), so this set has the *most* total graphs of the three.

| Task | Train | Val | Test | Total | Disk (train/val/test) |
|---|---:|---:|---:|---:|---|
| 1 Scanning | 13,235 | 2,837 | 2,837 | 18,909 | 1121 MB / 240 MB / 240 MB |
| 2 Reconnaissance | 9,174 | 1,966 | 1,966 | 13,106 | 746 MB / 160 MB / 160 MB |
| 3 DDoS + Infiltration | 12,363 | 2,650 | 2,650 | 17,663 | 1058 MB / 226 MB / 226 MB |
| 4 DoS + Injection | 6,586 | 1,412 | 1,412 | 9,410 | 562 MB / 120 MB / 120 MB |
| 5 Password + Bot | 4,538 | 973 | 973 | 6,484 | 387 MB / 83 MB / 83 MB |
| 6 XSS + BruteForce | 9,016 | 1,932 | 1,932 | 12,880 | 765 MB / 163 MB / 164 MB |
| **Total** | **54,912** | **11,770** | **11,770** | **78,452** | **~6.47 GB** |

Downsampling: `min_task_total=6,484` (task 5), `cap=19,452` - nothing trimmed.

### 6.3 `benign_ratio=4.0` - `data/graphs_ratio4/` (`configs/graph_ratio4.yaml`, ~51 min)

Higher `benign_ratio` = less benign per attack, so this set has the *fewest* total graphs of the three.

| Task | Train | Val | Test | Total | Disk (train/val/test) |
|---|---:|---:|---:|---:|---|
| 1 Scanning | 11,030 | 2,364 | 2,364 | 15,758 | 941 MB / 202 MB / 202 MB |
| 2 Reconnaissance | 7,645 | 1,639 | 1,639 | 10,923 | 624 MB / 134 MB / 134 MB |
| 3 DDoS + Infiltration | 10,303 | 2,208 | 2,208 | 14,719 | 869 MB / 186 MB / 186 MB |
| 4 DoS + Injection | 5,488 | 1,176 | 1,176 | 7,840 | 463 MB / 99 MB / 99 MB |
| 5 Password + Bot | 3,782 | 811 | 811 | 5,404 | 318 MB / 68 MB / 68 MB |
| 6 XSS + BruteForce | 7,514 | 1,610 | 1,610 | 10,734 | 629 MB / 134 MB / 135 MB |
| **Total** | **45,762** | **9,808** | **9,808** | **65,378** | **~5.36 GB** |

Downsampling: `min_task_total=5,404` (task 5), `cap=16,212` - nothing trimmed.

Flows/graph mean is ~300 (capped by `graph_size`) for every task/split/set except the last (remainder) chunk. Choosing the winning `benign_ratio` is deferred to Step 3 (needs real CL training/forgetting metrics, not available yet) - see §11.

---

## 7. Graph topology statistics

Computed over every saved mini-graph across **all three `benign_ratio` sets** (213,567 graphs total, not a sample) - `torch.load` each `task_{t}_{split}.pt`, average per-graph node/edge counts, host degree (from the precomputed `total_flows` host feature), and host↔host subgraph density (`edges / (nodes×(nodes-1))`, directed, no self-loops). Train split shown per task (largest, most stable); val/test are consistently within ~1% of train for every task, so only the "overall" rows include all splits.

### 7.1 Primary set - `benign_ratio=3.0` (`data/graphs/`)

| Task | Graphs (train) | Avg flow | Avg host | Avg protocol | Avg service | Avg port | Avg total nodes | Avg total edges | Avg host degree | Avg host↔host density |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Scanning | 11,765 | 300.0 | 63.0 | 2.2 | 16.7 | 244.6 | 626.4 | 1,597.2 | 9.5 | 0.0253 |
| 2 Reconnaissance | 8,155 | 300.0 | 12.7 | 2.1 | 8.2 | 197.0 | 520.0 | 1,531.7 | 47.3 | 0.2237 |
| 3 DDoS + Infiltration | 10,990 | 300.0 | 96.7 | 2.3 | 13.9 | 43.3 | 456.3 | 1,599.8 | 6.2 | 0.0110 |
| 4 DoS + Injection | 5,854 | 300.0 | 88.8 | 2.2 | 14.5 | 54.3 | 459.7 | 1,601.0 | 6.8 | 0.0132 |
| 5 Password + Bot | 4,034 | 299.9 | 89.0 | 2.2 | 14.8 | 42.6 | 448.5 | 1,594.2 | 6.7 | 0.0123 |
| 6 XSS + BruteForce | 8,015 | 300.0 | 83.8 | 2.1 | 14.4 | 42.7 | 442.9 | 1,588.2 | 7.2 | 0.0130 |
| **Overall (69,737 graphs, all splits)** | 69,737 | 300.0 | 70.5 | 2.2 | 13.8 | 118.4 | 504.8 | 1,585.2 | 8.5 | 0.0499 |

### 7.2 `benign_ratio=2.0` (`data/graphs_ratio2/`)

| Task | Graphs (train) | Avg flow | Avg host | Avg protocol | Avg service | Avg port | Avg total nodes | Avg total edges | Avg host degree | Avg host↔host density |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Scanning | 13,235 | 300.0 | 60.7 | 2.2 | 17.0 | 246.0 | 625.9 | 1,596.6 | 9.9 | 0.0271 |
| 2 Reconnaissance | 9,174 | 300.0 | 13.6 | 2.1 | 8.4 | 204.3 | 528.5 | 1,534.2 | 44.1 | 0.2070 |
| 3 DDoS + Infiltration | 12,363 | 300.0 | 114.8 | 2.4 | 15.8 | 54.5 | 487.4 | 1,615.8 | 5.2 | 0.0090 |
| 4 DoS + Injection | 6,586 | 300.0 | 107.0 | 2.2 | 16.4 | 64.5 | 490.1 | 1,616.2 | 5.6 | 0.0104 |
| 5 Password + Bot | 4,538 | 300.0 | 106.8 | 2.2 | 16.9 | 54.2 | 480.1 | 1,609.4 | 5.6 | 0.0099 |
| 6 XSS + BruteForce | 9,016 | 300.0 | 102.3 | 2.2 | 16.3 | 54.3 | 475.0 | 1,605.2 | 5.9 | 0.0103 |
| **Overall (78,452 graphs, all splits)** | 78,452 | 300.0 | 80.6 | 2.2 | 15.0 | 126.3 | 524.2 | 1,594.6 | 7.4 | 0.0461 |

### 7.3 `benign_ratio=4.0` (`data/graphs_ratio4/`)

| Task | Graphs (train) | Avg flow | Avg host | Avg protocol | Avg service | Avg port | Avg total nodes | Avg total edges | Avg host degree | Avg host↔host density |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Scanning | 11,030 | 300.0 | 64.3 | 2.1 | 16.4 | 243.5 | 626.3 | 1,597.1 | 9.3 | 0.0243 |
| 2 Reconnaissance | 7,645 | 300.0 | 12.0 | 2.2 | 8.0 | 192.4 | 514.6 | 1,530.1 | 50.0 | 0.2376 |
| 3 DDoS + Infiltration | 10,303 | 300.0 | 85.5 | 2.3 | 12.7 | 36.5 | 437.0 | 1,589.8 | 7.0 | 0.0127 |
| 4 DoS + Injection | 5,488 | 300.0 | 77.5 | 2.1 | 13.2 | 47.9 | 440.7 | 1,591.3 | 7.7 | 0.0158 |
| 5 Password + Bot | 3,782 | 299.9 | 77.9 | 2.1 | 13.4 | 35.5 | 428.9 | 1,584.5 | 7.7 | 0.0145 |
| 6 XSS + BruteForce | 7,514 | 300.0 | 72.3 | 2.1 | 13.0 | 35.6 | 422.9 | 1,577.5 | 8.3 | 0.0155 |
| **Overall (65,378 graphs, all splits)** | 65,378 | 300.0 | 64.1 | 2.2 | 13.0 | 113.5 | 492.6 | 1,579.0 | 9.4 | 0.0532 |

**Reading these:** the qualitative pattern is stable across all three `benign_ratio` sets - Task 2 (Reconnaissance) always has by far the fewest hosts per graph (12.0-13.6) but the highest host degree (44-50) and host↔host density (0.21-0.24), consistent with scanning-style traffic where a small number of hosts each touch many others. Task 1 (Scanning) always has the widest port spread (~244-246 avg ports/graph) since scanning traffic sweeps many destination ports by nature. Raising `benign_ratio` (less benign per attack) *shrinks* host/port counts per graph for the attack-heavy tasks (e.g. task 3's avg hosts: 114.8 at ratio=2.0 → 96.7 at ratio=3.0 → 85.5 at ratio=4.0) - fewer distinct benign IPs get pulled into each graph as their share shrinks. `flow`/`protocol` counts and the five flow-incident relation edge counts (`originates`/`terminates_at`/`targets_port`/`uses_protocol`/`uses_service`, each tracking 1:1 with flow count by construction) are essentially invariant across all three sets, since `graph_size=300` is unchanged - only the *composition* of each graph's 300 flows shifts with `benign_ratio`, not the count.

Raw per-task/per-split JSON (all three sets): `C:\Users\cherr\.claude\jobs\a5cada13\tmp\graph_topology_stats_all.json` (not checked in - regenerate against `data/graphs*/` whenever Step 2 is rerun).

---

## 8. Graph composition analysis - class mixing within mini-graphs (`benign_ratio=4.0`)

Motivation: every task besides T1/T2 has 2+ attack classes, so a mini-graph can be "pure" (one class only) or "mixed" (multiple classes co-occurring). This matters for any future **graph-level downsampling** decision - naive downsampling could disproportionately remove mixed graphs (which carry co-occurrence signal a pure graph doesn't) or graphs dominated by a minority class. `src/trench_ids/graph_composition.py` computes, per task, across all `65,378` mini-graphs of the `benign_ratio=4.0` set (train+val+test combined): (1) exact class-combination counts, (2) which class dominates each graph's flow count, (3) the mean per-graph class share, and (4) an 11×11 (Benign + 10 attack classes) co-occurrence matrix summed across all tasks. Source: `data/graphs_ratio4/graph_composition.json`, `data/graphs_ratio4/graph_composition_matrix.csv`.

### 8.1 Exact composition table (class-combination counts per task)

Since a class belongs to exactly one task (`CANONICAL_TO_TASK`), a 2-attack-class task has at most 7 possible combinations (2^3 - 1); a 1-attack-class task (T1, T2) has at most 3. In practice almost every graph lands in the single "all classes present" combination:

| Task | Combination | Count | Fraction |
|---|---|---:|---:|
| 1 Scanning | mix of Benign + Scanning | 15,758 | 100.00% |
| 2 Reconnaissance | mix of Benign + Reconnaissance | 10,923 | 100.00% |
| 3 DDoS + Infiltration | mix of Benign + DDoS + Infiltration | 14,716 | 99.98% |
| 3 DDoS + Infiltration | mix of Benign + DDoS | 3 | 0.02% |
| 4 DoS + Injection | mix of Benign + DoS + Injection | 7,840 | 100.00% |
| 5 Password + Bot | mix of Benign + Bot + Password | 5,404 | 100.00% |
| 6 XSS + BruteForce | mix of Benign + BruteForce + XSS | 10,734 | 100.00% |

Only task 3 has any graphs missing a class (3 of 14,719 graphs lack Infiltration - Infiltration is the smallest class in the whole 10-class pool, 116,361 raw rows, so a 300-row chunk very occasionally misses it entirely). Every other task's graphs are **100% mixed** - a genuinely "pure" single-class graph essentially does not occur once benign is present in every task and `graph_size=300` is this large relative to per-task class floors.

### 8.2 Dominant class per graph

The class with the most flows in each graph (ties would report "Equal"; none occurred):

| Task | Dominant class | Count | Fraction |
|---|---|---:|---:|
| 1 Scanning | Scanning | 15,758 | 100.00% |
| 2 Reconnaissance | Reconnaissance | 10,923 | 100.00% |
| 3 DDoS + Infiltration | DDoS | 14,719 | 100.00% |
| 4 DoS + Injection | DoS | 7,840 | 100.00% |
| 5 Password + Bot | Password | 5,404 | 100.00% |
| 6 XSS + BruteForce | XSS | 10,734 | 100.00% |

`dominant_class_totals` across all 65,378 graphs: Scanning 15,758 / DDoS 14,719 / Reconnaissance 10,923 / XSS 10,734 / DoS 7,840 / Password 5,404 - **zero "Equal" ties anywhere**. Every task has exactly one dominant class in 100% of its graphs; the other attack class in each 2-class task (Infiltration, Injection, Bot, BruteForce) never wins a single graph.

### 8.3 Average class proportion (mean per-graph share, each graph weighted equally)

Distinguishes a lopsided split from a near-even one - dominant-class alone can't tell a 51/49 graph from a 95/5 one:

| Task | Class shares (mean per graph) |
|---|---|
| 1 Scanning | Scanning 80.0% / Benign 20.0% |
| 2 Reconnaissance | Reconnaissance 80.0% / Benign 20.0% |
| 3 DDoS + Infiltration | DDoS 77.4% / Benign 20.0% / Infiltration 2.6% |
| 4 DoS + Injection | DoS 50.9% / Injection 29.1% / Benign 20.0% |
| 5 Password + Bot | Password 71.2% / Benign 20.0% / Bot 8.8% |
| 6 XSS + BruteForce | XSS 76.2% / Benign 20.0% / BruteForce 3.8% |

Benign sits at ~20.0% in every task, as expected from `benign_ratio=4.0` (`target_benign = n_attack / 4.0`, i.e. benign ≈ 1/(4+1) = 20% of each graph's flows). The attack-side split within each 2-class task tracks each class's natural size ratio within its task's raw row counts (§5) - e.g. task 4's DoS (1,196,608 raw rows) vs. Injection (684,897 raw rows) is a ~1.75:1 ratio, close to the observed 50.9:29.1 (~1.75:1) mean graph share. Task 4 is the most balanced (50.9/29.1 within the attack side); tasks 3, 5, and 6 are far more lopsided (a ~30:1, ~8:1, and ~20:1 attack-side split respectively).

### 8.4 11×11 class co-occurrence matrix (summed across all 6 tasks)

Diagonal = pure-class graph counts (structurally ~0 for every attack class, since Benign is in every graph); off-diagonal = graphs containing both classes, regardless of a third class also present. Cross-task cells are structurally zero - no two classes from different tasks ever co-occur in the same mini-graph:

| | Benign | Scanning | Reconn. | DDoS | Infiltr. | DoS | Inject. | Password | Bot | XSS | BruteF. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Benign** | 0 | 15,758 | 10,923 | 14,719 | 14,716 | 7,840 | 7,840 | 5,404 | 5,404 | 10,734 | 10,734 |
| **Scanning** | 15,758 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Reconn.** | 10,923 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **DDoS** | 14,719 | 0 | 0 | 0 | 14,716 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Infiltr.** | 14,716 | 0 | 0 | 14,716 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **DoS** | 7,840 | 0 | 0 | 0 | 0 | 0 | 7,840 | 0 | 0 | 0 | 0 |
| **Inject.** | 7,840 | 0 | 0 | 0 | 0 | 7,840 | 0 | 0 | 0 | 0 | 0 |
| **Password** | 5,404 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5,404 | 0 | 0 |
| **Bot** | 5,404 | 0 | 0 | 0 | 0 | 0 | 0 | 5,404 | 0 | 0 | 0 |
| **XSS** | 10,734 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10,734 |
| **BruteF.** | 10,734 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10,734 | 0 |

### 8.5 Interpretation

**Nearly every graph is class-mixed (good for the benchmark)** - true single-class graphs are essentially absent (3 of 65,378, all in task 3), so every graph carries co-occurrence signal between its task's classes and Benign. **But within a task, composition is essentially fixed** - one class dominates 100% of a task's graphs with zero ties, and the mean class shares (§8.3) barely vary graph-to-graph, because removing Step 1's `attack_per_class_cap` (see project history) left large natural frequency gaps between paired classes within a task (e.g. task 3's DDoS at 3.4M raw rows vs. Infiltration at 116K - a ~30:1 ratio) - at `graph_size=300`, a random draw from a task's pool essentially always reproduces that same skewed ratio. **Practical implication for downsampling**: a composition-aware or dominant-class-aware downsampling pass would have almost nothing to select against, since virtually 100% of a task's graphs already share the same combination and same dominant class - it would reduce to plain random sampling within a task. Real compositional diversity in this benchmark is **across tasks, not within them** (task 4 is markedly more balanced than tasks 3/5/6) - this is worth stating explicitly as a methodological note if downsampling is revisited, rather than treating it as a gap to fix.

**Relevant for Step 3 interpretation:** task 4's attack-side split (DoS 50.9% / Injection 29.1%, ~1.75:1) is markedly more balanced than task 3 (~30:1), task 5 (~8:1), and task 6 (~20:1, §8.3). When comparing continual-learning forgetting/transfer results across tasks in Step 3, keep in mind that per-task performance differences may partly reflect this composition imbalance (how skewed a task's two attack classes are toward each other) rather than purely the classes' semantic (cosine) similarity from §3-4 - worth flagging, and potentially controlling for, when reporting per-task results.

Source: `src/trench_ids/graph_composition.py`, `data/graphs_ratio4/graph_composition.json`, `data/graphs_ratio4/graph_composition_matrix.csv` (all gitignored outputs; regenerate via `python -m trench_ids.graph_composition --graphs-dir data/graphs_ratio4`).

---

## 9. Pipeline runtime & disk

| Stage | Wall-clock | Output size |
|---|---:|---:|
| Step 1 (preprocessing, 2 passes over ~12+ GB raw CSVs) | ~27 min | 320 MB (`data/processed/`) |
| Step 2, `benign_ratio=3.0` (69,737 mini-graphs) | ~58 min | 5.73 GB (`data/graphs/`) |
| Step 2, `benign_ratio=2.0` (78,452 mini-graphs) | ~61 min | 6.47 GB (`data/graphs_ratio2/`) |
| Step 2, `benign_ratio=4.0` (65,378 mini-graphs) | ~51 min | 5.36 GB (`data/graphs_ratio4/`) |
| **Total (useful compute)** | **~3.3 hr** | **~17.9 GB** |

No crashes, OOM, or data-level timeouts at real scale - all three Step 2 runs completed cleanly and independently reproduce as reported. One environment-level issue during this rerun: the machine running the pipeline has only 16 GB RAM, and two of the three Step 2 runs were killed by external resource pressure partway through (both immediately after the largest task, task 1, finished writing - the peak-memory moment of each run) when free memory dropped to ~3.8 GB with no other process attributable in the pipeline's own logs. Both were resolved by freeing RAM on the host and re-running the affected config alone (no code or data changes needed) - not a pipeline bug, but worth flagging for anyone rerunning this on similarly memory-constrained hardware.

---

## 10. Test suite

**64 tests passing** (`.venv/Scripts/python.exe -m pytest -q`), 0 failures, 2 pre-existing unrelated `torch.jit.script` deprecation warnings - up from 48 after this round added `tests/test_graph_composition.py` (16 new: composition/dominant-class/proportion/matrix logic for §8). Prior jump was 39→48 when the finalization round added `tests/test_task_design.py` (4: size-aware tie-break) and extended `test_preprocess.py` (+3: corrupted-row filter) and `test_graphs.py` (+2: downsampling cap). Covers: `test_labels.py`, `test_preprocess.py`, `test_similarity.py`, `test_graphs.py`, `test_vocab.py`, `test_task_design.py`, `test_graph_composition.py`.

---

## 11. Known open issues

| Issue | Status | Detail |
|---|---|---|
| `inf`/NaN/overflow values in task 3 | **Resolved 2026-07-14** | Previously: 540 `inf` values across 533 of task_3's mini-graphs, from ~54 raw CSE-CIC-IDS2018 rows with corrupted `SRC_TO_DST_SECOND_BYTES`/`DST_TO_SRC_SECOND_BYTES` values overflowing `float32` at Step 2's cast. Fixed by `_drop_corrupted_rows` in Step 1 (`preprocess.py`), which rejects NaN/±inf/float32-overflow across every numeric raw column before dedup - the fresh rerun dropped exactly 54 rows (all in task 3, matching the prior count) and a direct tensor scan of every saved graph across all three `benign_ratio` sets confirms **zero** non-finite flow-feature values (§5, §6). Safe to proceed to Step 3 training. |
| Benign pool sizing | Tracked, not yet resolved | `benign_per_task=8000` (Step 1 safeguard) is still far smaller than a task's attack count (T1 alone is 3.78M attack rows), so `_select_benign`'s with-replacement fallback still reuses each pooled benign flow heavily for the larger tasks. Not raised in this round - the finalization design scoped only the 3-way `benign_ratio` sweep (now actually produced, §6), not a `benign_per_task` increase. Track as a future Step 3 data-quality item if benign-flow repetition turns out to matter for training. |
| Choosing the winning `benign_ratio` | Deferred to Step 3 | All three candidate sets (2.0/3.0/4.0) now exist in full (§6, §7) with `benign_ratio` and `max_task_ratio` recorded in each `graph_counts.json`. Picking the best one needs real continual-learning training/forgetting metrics, which don't exist until Step 3's model is built. |

**Step 1 and Step 2 are now frozen** (per user direction, 2026-07-14) - no further data-pipeline redesign unless the professor requests it or a genuine bug is discovered. Effort moves to Step 3 (the relation-specific heterogeneous GNN and continual-learning model) next.

---

## 12. Where these numbers come from

- Dataset/class/task design: `docs/dataset-plan.md`, `docs/attack-class-counts.md`, `docs/attack-similarity-matrix.md`, `src/trench_ids/labels.py`, `src/trench_ids/task_design.py`
- Step 1 output: `data/processed/manifest.json` (gitignored, regenerate via `trench_ids.preprocess`)
- Step 2 output: `data/graphs/graph_counts.json`, `data/graphs_ratio2/graph_counts.json`, `data/graphs_ratio4/graph_counts.json` (gitignored, regenerate via `trench_ids.graphs --config <configs/graph*.yaml>`)
- Graph composition (§8): `src/trench_ids/graph_composition.py`, `data/graphs_ratio4/graph_composition.json`, `data/graphs_ratio4/graph_composition_matrix.csv` (gitignored, regenerate via `python -m trench_ids.graph_composition --graphs-dir data/graphs_ratio4`)
- Design rationale: `docs/superpowers/specs/2026-07-13-dataset-task-respec-design.md` (original respec), `docs/superpowers/specs/2026-07-14-benchmark-finalization-design.md` (task rebalancing, corrupted-row filter, downsampling cap, benign_ratio sweep)
