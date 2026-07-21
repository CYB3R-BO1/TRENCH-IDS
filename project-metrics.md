# TRENCH-IDS - Project Metrics

This file is the single place to find every hard number about the project - dataset scale, class distribution, task design, pipeline output, and test/runtime stats. Update it whenever Step 1/Step 2 are rerun or the class/task design changes; treat every number here as sourced from an actual file on disk (cited inline), never estimated. For the narrative/architecture picture - what each component does and how they connect (including branches, shared inputs, dead-end diagnostics, and the one real feedback loop) - see `flow.md`. For a dated changelog of when each piece was built and why - see `log.md`.

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
| Step 3 model parameters (default config: `hidden_dim=64`, 1 layer) | 86,226 |
| Step 3 relations used (one-directional, no reverse edges) | 6 |

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
Step 3: relation-specific encoding + attention   src/trench_ids/model/*.py       (S7, see §13)
  fusion (implemented) -> per-relation + fused
  node embeddings
        |
        v
Step 4 (not yet started): classifier + relation-specific memory bank,
  transferability estimation, relation-aware EWC continual-learning loop
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

### 6.1 `benign_ratio=3.0` - `data/graphs/` (`configs/graph.yaml`, ~58 min; **regenerated 2026-07-18** to pick up the global label-mapping fix, §11 - table below unchanged, only `flow.y`'s encoding changed)

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

### 6.4 Node type schema - features and relations (per node type)

**Node types and features:**

| Node type | Features |
|---|---|
| Flow | 37 raw NetFlow flow-statistics features + class label |
| Host | 4 features: total_flows, avg_bytes_as_src, avg_bytes_as_dst, unique_ports_contacted |
| Protocol | none - categorical ID only (vocab size 5) |
| Service | none - categorical ID only (vocab size 216) |
| Port | none - categorical ID only (32 buckets) |

**Relations:**

| Relation | From -> To |
|---|---|
| originates | Host -> Flow |
| terminates_at | Flow -> Host |
| targets_port | Flow -> Port |
| uses_protocol | Flow -> Protocol |
| uses_service | Flow -> Service |
| communicates_with | Host -> Host |

Flow's 37 features (`configs/graph.yaml: features`): `IN_BYTES`, `IN_PKTS`, `OUT_BYTES`, `OUT_PKTS`, `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`, `FLOW_DURATION_MILLISECONDS`, `DURATION_IN`, `DURATION_OUT`, `MIN_TTL`, `MAX_TTL`, `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`, `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`, `RETRANSMITTED_IN_BYTES`, `RETRANSMITTED_IN_PKTS`, `RETRANSMITTED_OUT_BYTES`, `RETRANSMITTED_OUT_PKTS`, `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT`, `NUM_PKTS_UP_TO_128_BYTES`, `NUM_PKTS_128_TO_256_BYTES`, `NUM_PKTS_256_TO_512_BYTES`, `NUM_PKTS_512_TO_1024_BYTES`, `NUM_PKTS_1024_TO_1514_BYTES`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`, `ICMP_TYPE`, `ICMP_IPV4_TYPE`, `DNS_QUERY_ID`, `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER`, `FTP_COMMAND_RET_CODE`.

Note: Protocol, Service, and Port have no continuous feature vector - they're identified only by which embedding-table row they index. Only Flow and Host carry real multi-dimensional feature vectors.

Source: `configs/graph.yaml`, `src/trench_ids/graphs.py`, `src/trench_ids/model/rhgnn.py`, `src/trench_ids/vocab.py`.

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
| Flow label encoding was mini-graph-local, not global | **Resolved 2026-07-18** (default set only - see caveat) | `build_task_graph()` (`graphs.py`) built `label_names`/`y` from each mini-graph chunk's own `sorted(canonical_label.unique())`, so the same integer index could mean a different class in different mini-graphs (e.g. index 1 = DDoS in one chunk, XSS in another) - a real problem for any classifier trained with `CrossEntropyLoss` across mini-graphs, since the output-neuron-to-class mapping must stay fixed. Fixed by switching to the fixed global `labels.canonical_classes()` order (`Benign, Scanning, Reconnaissance, DDoS, Infiltration, DoS, Injection, Password, Bot, XSS, BruteForce` - `graphs.py`'s new `LABEL_NAMES`/`LABEL_LOOKUP` module constants), so every mini-graph in every task now uses the identical 11-class mapping (`Benign` always index 0). Verified against the regenerated `data/graphs/`: `label_names` matches across all 6 tasks, each task's present attack classes match `labels.py`'s task table exactly. **Caveat: only `data/graphs/` (`benign_ratio=3.0`, the default) was regenerated** - `data/graphs_ratio2/` and `data/graphs_ratio4/` still have the old local/inconsistent `y` encoding baked in and must be regenerated the same way before use in Step 4 training. |

**Step 1 and Step 2 are now frozen** (per user direction, 2026-07-14) - no further data-pipeline redesign unless the professor requests it or a genuine bug is discovered. The label-encoding fix above (2026-07-18) is exactly that carve-out: a genuine correctness bug in Step 2's `graphs.py`, not a design change, so it was fixed in place rather than deferred. Step 3 (relation-specific encoder + attention fusion, §13) is implemented and verified; effort moves to Step 4 (classifier + relation-specific memory bank, transferability estimation, relation-aware EWC) next. Picking the winning `benign_ratio` still awaits real continual-learning training/forgetting curves from Step 4, not just Step 3's encoder existing.

---

## 13. Step 3 - relation-specific heterogeneous GNN + attention fusion

Implemented 2026-07-16 in `src/trench_ids/model/` (`relation_conv.py`, `attention_fusion.py`, `rhgnn.py`, `__init__.py`) + `configs/model.yaml` + `tests/test_model.py`. Reads Step 2's saved `HeteroData` mini-graphs unmodified. **Revised 2026-07-17**: removed the in-memory `ToUndirected`/reverse-edge step per the professor's explicit instruction that relations must stay one-way - a synthetic generic `rev_<relation>` edge has no real-world meaning. **Revised again 2026-07-19** (professor's guidance, two parts): (1) `graphs.py` gained 5 new, individually-named reverse relations (`originated_by`, `terminated_by`, `targeted_by`, `protocol_of`, `service_of` - this *is* a Step 2 change, the one carve-out to the Step 1/2 freeze the professor explicitly requested), each with correct passive-voice semantics rather than a generic `rev_X` prefix, giving Flow 5 incoming relations instead of 1 and Host 3 instead of 2; (2) `RelationSpecificConv` now concatenates each destination node's own current embedding into every one of its relation-specific aggregations before a per-relation combine layer (GraphSAGE-style, Hamilton et al. NeurIPS 2017) - previously a relation's stored embedding was 100% neighbor signal, 0% self, worst for Flow (the classification target). All numbers below are freshly re-measured against the regenerated real saved graphs (`data/graphs`, 69,737 graphs, unchanged count from before the schema change) and the real full test suite on the 11-relation, self-preserving model.

Step 3 converts the heterogeneous mini-graphs produced by Step 2 into relation-aware node embeddings: it first learns a separate representation per relation for every node, then fuses those representations through semantic attention into one final embedding per node.

### 13.1 Architecture size

Every node's raw Step 2 features are first normalized and projected into a common hidden dimension (Flow/Host: `LayerNorm -> Linear`; Protocol/Service: vocab-indexed `Embedding`; Port: bucketed `Embedding`) - see §13.3 for why the `LayerNorm` step was necessary. The model operates on Step 2's **11 relations** (the original 6 one-directional relations plus 5 reverse relations added 2026-07-19). Confirmed against a regenerated real saved graph (`data/graphs/task_4_test.pt`):

| Node type | Incoming relations |
|---|---:|
| Flow | 5 (`originates` from Host, `targeted_by` from Port, `protocol_of` from Protocol, `service_of` from Service, `terminated_by` from Host) |
| Host | 3 (`terminates_at` from Flow, `originated_by` from Flow, `communicates_with` from Host) |
| Protocol | 1 (`uses_protocol`, from Flow) |
| Service | 1 (`uses_service`, from Flow) |
| Port | 1 (`targets_port`, from Flow) |

Flow went from the most starved node type (1 incoming relation, the classification target) to the richest; Host from 2 to 3. Protocol/Service/Port are unaffected - the new relations only enrich Flow's and Host's incoming side, consistent with those three node types having no continuous feature vector to enrich (§6.4). Semantic attention fusion is now a genuine multi-relation combination for both Flow and Host, not just Host.

Global vocab sizes actually used (`data/graphs/vocab.json`): **Protocol = 5**, **Service (`L7_PROTO`) = 216**.

### 13.2 Parameter count (`RelationSpecificHeteroGNN`, real instantiation against regenerated `task_4_train.pt`)

| Config | Total params | Node-feature encoders | Relation-conv (per layer) | Attention fusion (per layer) |
|---|---:|---:|---:|---:|
| `hidden_dim=64`, 1 layer (**current `configs/model.yaml` default**) | **197,842** | 19,026 | 136,576 | 42,240 |

Relation-conv parameters scale as **O(R × 3H²)** now, where R is the number of relations (11, up from 6) and H is `hidden_dim` - each relation now has *two* full-rank matrices, `rel_lins[r]` (`H×H`, transforms neighbor messages) and the new `combine_lins[r]` (`2H×H`, folds the destination's own embedding + aggregated neighbor message back down to `H`), vs. one `H×H` matrix per relation before. Relation-conv params rose from 24,960 (6 relations, one `H×H` each) to 136,576 (11 relations, one `H×H` + one `2H×H` each) - both the relation count and the per-relation cost increased.

### 13.3 Real-data forward/backward integration check

32 real mini-graphs from the regenerated `data/graphs/task_4_train.pt` (task 4 = DoS + Injection + Benign, 3 classes), batched via PyG's `DataLoader`, real vocab sizes, `hidden_dim=64`, 1 layer, plus a `nn.Linear(64, 3)` classifier head on `output.fused["flow"]`:

| Metric | Value |
|---|---:|
| Batched flow nodes | 9,600 (32 graphs × 300 flows) |
| Cross-entropy loss (random init, 3 classes) | 1.0768 (≈ ln 3 = 1.099, as expected untrained) |
| Forward + backward wall time | 0.205 s (CPU) - up from 0.045s pre-change: more relations (11 vs 6) and the added concat+combine step per relation |
| `host--originates-->flow` weight grad, abs-sum | 1.617 |

Attention weights (`output.attention`) at random initialization, per node type. Flow and Host now each have a genuine multi-relation fusion (near-uniform weights across relations, as expected before any training); Protocol/Service/Port are still a trivial single-relation identity (β=1.0), unaffected by this change:

| Node type | Relation | β (untrained) |
|---|---|---:|
| Flow | `originates` | 0.191 |
| Flow | `targeted_by` | 0.194 |
| Flow | `protocol_of` | 0.214 |
| Flow | `service_of` | 0.194 |
| Flow | `terminated_by` | 0.206 |
| Host | `terminates_at` | 0.342 |
| Host | `communicates_with` | 0.317 |
| Host | `originated_by` | 0.341 |
| Protocol | `uses_protocol` | 1.000 |
| Service | `uses_service` | 1.000 |
| Port | `targets_port` | 1.000 |

Each node type's weights sum to 1.000 (softmax-guaranteed, also asserted directly in `test_relation_specific_layer_attention_sums_to_one_per_node_type`). A new integration test, `test_relation_specific_conv_incorporates_destination_own_features`, confirms the self-preservation fix directly: perturbing `x_dict["flow"]` while holding all neighbor messages fixed changes every one of Flow's 5 relation embeddings (on the pre-fix code this test would fail, since neighbor messages alone determined the output).

### 13.4 Test suite

**78 tests passing** (`.venv/Scripts/python.exe -m pytest -q`, 0 failures, same 2 pre-existing `torch.jit.script` deprecation warnings as §10) - up from 77 with the addition of `test_relation_specific_conv_incorporates_destination_own_features` in `tests/test_model.py` (now **14 tests**). `tests/test_graphs.py` gained assertions (not new tests) for the 5 new reverse-relation `edge_index` tensors and `edge_counts` entries. `ruff check src tests` passes clean on all files.

### 13.5 Design choices (brief - see code docstrings in `src/trench_ids/model/` for full justification)

- **11 relations, individually-named reverses** (revised 2026-07-19, supersedes the 2026-07-17 "no reverse edges" note below): the earlier rejection was specifically of a *generic* `rev_<relation>` edge with no real-world meaning (e.g. "flow originated this host"?), not of reverse edges in general. The professor's 5 new relations (`originated_by`, `terminated_by`, `targeted_by`, `protocol_of`, `service_of`) are individually named with correct passive-voice semantics ("this flow was originated by this host"), so they don't have that problem. This resolves the actual objection (bad naming / blurred semantics) rather than reversing the instruction.
- **Per-relation self-concatenation** (new 2026-07-19): `RelationSpecificConv` concatenates each destination node's own current embedding into every relation's neighbor-aggregated message before a per-relation combine layer projects back to `hidden_dim` (GraphSAGE-style, Hamilton et al. NeurIPS 2017). Fixes the prior 0%-self-signal gap in every relation-specific and fused embedding (worst for Flow). `SemanticAttention` fusion needed no change - it never had a self-term to begin with.
- ~~**No reverse edges** (2026-07-17, superseded above): relations stay exactly as Step 2 stores them, one-way.~~
- **HAN-style semantic attention** (Wang et al., WWW 2019) over GAT-style node-conditioned attention: produces one importance weight per relation (not per node), directly reusable for the later transferability-estimation/relation-aware-EWC steps.
- **Port bucketing** (log1p-scaled, 32 buckets) instead of a raw 65536-row `nn.Embedding`: Port has no global vocabulary (chunk-local by Step 2 design), so a raw-port embedding would never generalize across mini-graphs.
- **LayerNorm on raw Flow/Host features** before the linear projection: required because raw NetFlow byte/packet counters span huge dynamic ranges (§13.3).

Source: `src/trench_ids/model/relation_conv.py`, `src/trench_ids/model/attention_fusion.py`, `src/trench_ids/model/rhgnn.py`, `configs/model.yaml`, `tests/test_model.py`; parameter counts and integration-check numbers measured directly against the regenerated `data/graphs/task_4_train.pt` and `data/graphs/vocab.json` (not estimated).

---

## 14. Step 4 - classifier training + relation-specific memory bank

Implemented 2026-07-19 in `src/trench_ids/cl/` (`train.py`, `memory_bank.py`, `__init__.py`) + `configs/train.yaml` (Hydra-composed - added as a dependency this step, per pyproject.toml's own stated plan) + `tests/test_train.py` + `tests/test_memory_bank.py`. Sequential fine-tuning across T1-T6 in order: one shared classifier head (`nn.Linear(hidden_dim, 11)`, over the full global label space including Benign) on top of Step 3's `output.fused["flow"]`, trained jointly with the encoder via cross-entropy, **no replay/EWC** (that's step 7, still an open item - this is a plain baseline). Real run: `data/graphs` (regenerated, 11-relation schema), `hidden_dim=64`, `epochs_per_task=5`, `batch_size=8`, Adam `lr=1e-3`, on **GPU** (`NVIDIA GeForce RTX 3050 Laptop GPU`, this machine gained a CUDA device between 2026-07-15 and 2026-07-19 - resolved automatically via `torch.cuda.is_available()`, no config change needed to use it).

### 14.1 Forgetting matrix (real run, `runs/step4/forgetting_matrix.json`)

Rows = task just finished training; columns = task being evaluated (test split); diagonal = that task's own post-training accuracy, off-diagonal = accuracy on a previously-learned task's classes after training has moved on.

| Trained through | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---:|---:|---:|---:|---:|---:|
| T1 | **0.926** | | | | | |
| T2 | 0.162 | **0.945** | | | | |
| T3 | 0.151 | 0.052 | **0.955** | | | |
| T4 | 0.154 | 0.191 | 0.204 | **0.875** | | |
| T5 | 0.246 | 0.228 | 0.241 | 0.242 | **0.991** | |
| T6 | 0.197 | 0.208 | 0.247 | 0.238 | 0.248 | **0.997** |

Every task reaches high accuracy on itself when trained (0.87-0.997, diagonal), and every task's accuracy collapses once training moves to the next task (off-diagonal, e.g. T1 drops from 0.926 to 0.162 as soon as T2 is trained) - textbook catastrophic forgetting from plain sequential fine-tuning with a shared classifier head and no forgetting-mitigation mechanism. This is the expected baseline result, not a bug: it is exactly the problem steps 5-7 (transferability estimation, relation importance weights, relation-aware EWC) exist to address, and it's the forgetting curve CLAUDE.md flagged as needed before the `benign_ratio` sweep (§6) can be decided.

### 14.2 Relation-specific memory bank (real run, `runs/step4/memory_bank.pt`)

All 10 attack classes present (Benign correctly excluded - attack-type-only memory), each with a mean vector (`hidden_dim=64`) for all 5 of Flow's incoming relations (`originates`, `targeted_by`, `protocol_of`, `service_of`, `terminated_by`), computed in a no-grad pass over that class's task's train split right after training on it: `Scanning`, `Reconnaissance`, `DDoS`, `Infiltration`, `DoS`, `Injection`, `Password`, `Bot`, `XSS`, `BruteForce`.

### 14.3 Test suite

**86 tests passing** (`.venv/Scripts/python.exe -m pytest -q`, 0 failures) - up from 78 (§13.4) with `tests/test_memory_bank.py` (**5 tests**: hand-computed per-class-per-relation means, cross-batch accumulation, all-Benign-batch produces empty means, merge adds new classes, merge rejects a duplicate class) and `tests/test_train.py` (**3 tests**: `forgetting_row` bookkeeping - evaluates every previously-seen task, single-entry after the first task, calls the accuracy function exactly once per task - stubbed, no real model/graphs needed). `ruff check src tests` passes clean.

### 14.4 Design choices (brief)

- **Class-incremental, not task-incremental**: one shared classifier head over the fixed global label space (`trench_ids.labels.canonical_classes()`, already built into Step 2's `graph["flow"].y`) - no task ID at inference. This is what makes the forgetting matrix meaningful (accuracy on old classes without being told which task they belong to).
- **No replay/EWC in this step**: deliberately a plain sequential-fine-tuning baseline. Steps 5-7 (transferability estimation, importance weights, relation-aware EWC) are what's supposed to fix the forgetting shown in §14.1 - measuring the un-mitigated baseline first is the point.
- **Memory bank computed post-hoc per task**, not accumulated during training: one no-grad pass over the task's train split right after its training epochs finish, so the stored means reflect the just-trained encoder rather than an average over encoder states from earlier, less-trained epochs.
- **Hydra added this step** (`configs/train.yaml`), matching pyproject.toml's standing plan to add it "when the training loop (Step 4+) begins." `hydra.job.chdir: false` keeps the working directory at wherever the command was invoked from, so `data/graphs`-relative paths behave the same as every earlier step's plain-argparse config.

Source: `src/trench_ids/cl/`, `configs/train.yaml`, `tests/test_memory_bank.py`, `tests/test_train.py`; forgetting matrix and memory bank measured directly from a real run (`runs/step4/`, gitignored, rerun via `trench-train` or `python -m trench_ids.cl.train` to reproduce). Re-running with the same seed (42) reproduces the forgetting matrix in §14.1 exactly (verified: values match to displayed precision across two independent runs).

---

## 15. Step 5 - transferability estimation

Implemented 2026-07-19 in `src/trench_ids/cl/transferability.py` + `tests/test_transferability.py`. Per CLAUDE.md's proposed pipeline, step 5: after computing task *t*'s new per-class-per-relation means (§14.2) and *before* merging them into the bank, compare each new class against every class already in the bank, **same relation only** (different relations are different learned subspaces - a `terminated_by` mean and a `protocol_of` mean for the same pair of classes aren't comparable to each other). No aggregation into a single transferability score - that's step 6 (relation importance weights), still an open item, out of scope here. Hooked directly into the Step 4 loop (`train.py`): `runs/step4/transferability_task_{t}.json` per task, gitignored, regenerate via the same `trench-train` run that produces §14's outputs. Task 1 produces an empty report (`{"Scanning": {}}`) - nothing exists in the bank yet, expected, not a bug.

### 15.1 The two flagged pairs, checked against the real embeddings

`docs/attack-similarity-matrix.md` flagged two pairs as standouts in the raw-feature cosine-similarity matrix used for task assignment: XSS↔Infiltration (0.706, the highest in the whole 10-class matrix, landing in non-adjacent tasks T6 vs T3) and Scanning↔Reconnaissance (0.628, second-highest, isolated in T1 vs T2). Checking whether that raw-feature-level similarity carries over into the learned per-relation embeddings (`runs/step4/transferability_task_6.json`, `transferability_task_2.json`):

| Pair | `originates` | `terminated_by` | `targeted_by` | `protocol_of` | `service_of` |
|---|---:|---:|---:|---:|---:|
| XSS (T6) vs Infiltration (T3) | 0.180 | **0.468** | 0.098 | **0.568** | 0.088 |
| Reconnaissance (T2) vs Scanning (T1) | -0.354 | 0.029 | -0.654 | -0.363 | -0.095 |

**Mixed result, reported as-is rather than cherry-picked**: XSS↔Infiltration shows a *partial* match - 2 of 5 relations (`terminated_by`, `protocol_of`) moderately positive, roughly consistent with the raw-feature finding, the other 3 near zero. Scanning↔Reconnaissance shows **no** elevated similarity at all - every relation is at or below zero, the opposite of what the raw-feature similarity would predict. The most plausible explanation is §14.1's forgetting result itself: Scanning's bank entry was computed right after task 1, before 5 more tasks of un-mitigated sequential fine-tuning further changed the encoder (task 1's own test accuracy had already fallen from 0.926 to 0.197 by task 6, §14.1), so by the time Reconnaissance's mean is computed under the task-6 encoder, it's being compared against a Scanning mean computed under a substantially different, earlier version of the same encoder - not a like-for-like comparison. This is itself a real finding worth flagging to the professor: **a memory bank frozen at each task's end becomes progressively less comparable to older entries as an un-mitigated encoder keeps drifting**, which is exactly the problem steps 6-7 (relation importance weights, relation-aware EWC) are meant to fix by regularizing which parameters are allowed to drift.

### 15.2 Test suite

**95 tests passing** (`.venv/Scripts/python.exe -m pytest -q`, 0 failures) - up from 86 (§14.3) with `tests/test_transferability.py` (**9 tests**: cosine similarity identical/orthogonal/opposite/known-angle/zero-vector cases, empty-bank report shape, same-relation-only comparison, relations not shared by both classes are skipped, every bank class is covered). `ruff check src tests` passes clean.

Source: `src/trench_ids/cl/transferability.py`, `tests/test_transferability.py`; real transferability reports measured directly from `runs/step4/transferability_task_*.json` (gitignored, regenerate via `trench-train`).

---

## 16. Steps 6-8 - relation-aware transferability-guided EWC

Implemented 2026-07-21 in `src/trench_ids/cl/ewc.py` (`FLOW_RELATIONS`, `OnlineEWCState`, `estimate_fisher`, `OnlineEWCManager`), `src/trench_ids/cl/importance.py` (`ImportanceMLP`), extensions to `src/trench_ids/cl/transferability.py` (`aggregate_transferability_scores`) and `src/trench_ids/cl/train.py` (warm-up epochs + the three-term combined EWC loss), plus `configs/train.yaml` (`train.warmup_epochs: 2`, new `ewc:` block). Full design: `docs/superpowers/specs/2026-07-21-relation-aware-ewc-design.md`. Per task: 2 warm-up epochs (plain classification loss only) -> temporary prototypes -> transferability against the bank as of the previous task -> `S_r` (mean cosine per Flow relation) -> `w_r = sigma(ImportanceMLP(S_r))` -> 3 full-loss epochs under `L = L_cls + lambda_u * L_EWC^other + lambda_s * L_EWC^shared + lambda_r * sum_r(w_r * L_EWC^(r))` -> one whole-model Fisher pass -> final prototypes (discarding the warm-up's temporary ones) merged into the bank. Online EWC (Schwarz et al. 2018): one running Fisher + one reference-parameter snapshot per of 12 parameter groups (1 shared, 5 Flow relations, 6 other relations), blended across tasks via `gamma`, never a separate Fisher per task. Placeholder lambdas/gamma (not tuned): `lambda_r = lambda_s = lambda_u = 1.0`, `gamma = 0.9`. Real run: same `data/graphs` (11-relation schema), `hidden_dim=64`, `epochs_per_task=5` (2 warm-up + 3 full-loss), `batch_size=8`, Adam `lr=1e-3` (now also covering `ImportanceMLP`'s parameters), GPU (`NVIDIA GeForce RTX 3050 Laptop GPU`), seed 42.

### 16.1 Forgetting matrix (real run, `runs/step4/forgetting_matrix.json`) vs the plain-fine-tuning baseline

Baseline (§14.1, no EWC) saved to `runs/step4_baseline/forgetting_matrix.json` before this run overwrote `runs/step4/`.

| Trained through | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---:|---:|---:|---:|---:|---:|
| T1 | **0.924** | | | | | |
| T2 | 0.193 | **0.944** | | | | |
| T3 | 0.170 | 0.109 | **0.954** | | | |
| T4 | 0.179 | 0.172 | 0.204 | **0.891** | | |
| T5 | 0.232 | 0.076 | 0.238 | 0.215 | **0.990** | |
| T6 | 0.235 | 0.221 | 0.240 | 0.240 | 0.240 | **0.997** |

Final-row (T6) retention, EWC run vs. baseline (§14.1):

| Evaluated task | Baseline (no EWC) | EWC run | Difference |
|---|---:|---:|---:|
| T1 | 0.197 | 0.235 | +0.038 |
| T2 | 0.208 | 0.221 | +0.013 |
| T3 | 0.247 | 0.240 | -0.007 |
| T4 | 0.238 | 0.240 | +0.002 |
| T5 | 0.248 | 0.240 | -0.008 |
| T6 (own task) | 0.996 | 0.997 | +0.001 |

**No meaningful improvement over the plain-fine-tuning baseline** - every difference is within noise (largest is +0.038 on T1), and the shape of the forgetting curve (near-total collapse of every earlier task's accuracy once training moves past it) is essentially unchanged. Root cause identified in §16.2.

### 16.2 Learned relation-importance weights (real run, `runs/step4/importance_weights_task_{t}.json`)

| Task | `originates` | `terminated_by` | `targeted_by` | `protocol_of` | `service_of` |
|---|---:|---:|---:|---:|---:|
| T1 | 0.5355 | 0.5355 | 0.5355 | 0.5355 | 0.5355 |
| T2 | 1.10e-4 | 1.06e-3 | 4.24e-5 | 7.84e-5 | 3.60e-4 |
| T3 | 2.26e-5 | 7.96e-6 | 2.08e-5 | 3.46e-6 | 9.99e-6 |
| T4 | 5.06e-7 | 5.19e-6 | 6.46e-6 | 4.17e-6 | 6.76e-6 |
| T5 | 2.73e-6 | 4.72e-7 | 5.98e-7 | 1.44e-5 | 8.65e-7 |
| T6 | 8.65e-7 | 2.70e-6 | 8.81e-7 | 4.78e-6 | 6.45e-7 |

T1's five identical values are expected: with an empty bank, `S_r = 0.0` for every relation (§0's design-agreed default), so `ImportanceMLP` at its initial weights produces the same output for every relation - not yet a meaningful learned signal. From T2 onward, **every `w_r` collapses to a value within a few orders of magnitude of zero**, and the collapse only deepens task over task (T2's largest value, 1.06e-3, is already three orders of magnitude below 1.0; by T4-T6 every value is below 1.5e-5).

**Why this happens, and why it directly explains §16.1's flat result:** `w_r` is left attached to the autograd graph and trained end-to-end via backprop through the combined loss's `lambda_r * w_r * L_EWC^(r)` term (design's explicit, deliberate choice - see the design doc §3 and the "leave attached" decision made during design review). But minimizing that same combined loss gives gradient descent a direct, unconditional incentive to shrink `w_r` toward 0: doing so strictly reduces the loss (the penalty term shrinks) without any corresponding cost to `L_cls`, since `w_r` has no direct bearing on the current task's own classification accuracy - only on how strongly *old* parameters are protected. There is no term in the per-task local objective that rewards keeping `w_r` large for the sake of *future* retention, so nothing opposes the collapse. This is a genuine, reportable methodological finding rather than an implementation bug: **an end-to-end-trained importance weight that only ever appears multiplied into its own penalty term has a trivial optimum at zero**, which functionally disables the weighted-EWC term almost immediately and reduces the method to something very close to the plain sequential-fine-tuning baseline - consistent with §16.1 showing no meaningful improvement. Fixing this would need either (a) detaching `w_r` per-task (the alternative considered and rejected during design, which trades this failure mode for an untrained, permanently-at-initialization MLP) or (b) a genuinely different training signal for the MLP, e.g. a meta-objective that rewards `w_r` choices retrospectively based on measured forgetting - noted in the design doc's Explicitly Out of Scope section as a possible future direction, now with empirical motivation behind it.

### 16.3 Test suite

**120 tests passing** (`.venv/Scripts/python.exe -m pytest -q`, 0 failures) - up from 95 (§15.2) with `tests/test_ewc.py` (**16 tests**: `FLOW_RELATIONS`/parameter partitioning across single- and multi-layer models, `OnlineEWCState` gamma-blending and theta*-overwrite hand-computed math, `estimate_fisher` single-whole-model-pass coverage and independent train/eval mode restoration for `model` and `classifier`, `OnlineEWCManager`'s three-way loss weighting), `tests/test_importance.py` (**4 tests**: shape, output range, weight-sharing across relations, gradient flow), 3 new tests in `tests/test_transferability.py` (`aggregate_transferability_scores` mean-across-pairs, empty-bank zero default, full relation coverage), and 3 new tests in `tests/test_train.py` (`split_warmup_and_full_loss_epochs` bookkeeping, a device-mismatch regression test for `S_r`). `ruff check src tests` passes clean.

Source: `src/trench_ids/cl/ewc.py`, `src/trench_ids/cl/importance.py`, `src/trench_ids/cl/transferability.py`, `src/trench_ids/cl/train.py`, `configs/train.yaml`; forgetting matrix, memory bank, transferability reports, and importance weights measured directly from a real run (`runs/step4/`, gitignored, rerun via `trench-train` or `python -m trench_ids.cl.train` to reproduce; the pre-EWC baseline is preserved at `runs/step4_baseline/forgetting_matrix.json`, also gitignored).

---

## 17. Where these numbers come from

- Dataset/class/task design: `docs/dataset-plan.md`, `docs/attack-class-counts.md`, `docs/attack-similarity-matrix.md`, `src/trench_ids/labels.py`, `src/trench_ids/task_design.py`
- Step 1 output: `data/processed/manifest.json` (gitignored, regenerate via `trench_ids.preprocess`)
- Step 2 output: `data/graphs/graph_counts.json`, `data/graphs_ratio2/graph_counts.json`, `data/graphs_ratio4/graph_counts.json` (gitignored, regenerate via `trench_ids.graphs --config <configs/graph*.yaml>`)
- Graph composition (§8): `src/trench_ids/graph_composition.py`, `data/graphs_ratio4/graph_composition.json`, `data/graphs_ratio4/graph_composition_matrix.csv` (gitignored, regenerate via `python -m trench_ids.graph_composition --graphs-dir data/graphs_ratio4`)
- Design rationale: `docs/superpowers/specs/2026-07-13-dataset-task-respec-design.md` (original respec), `docs/superpowers/specs/2026-07-14-benchmark-finalization-design.md` (task rebalancing, corrupted-row filter, downsampling cap, benign_ratio sweep)
- Step 3 model + metrics (§13): `src/trench_ids/model/`, `configs/model.yaml`, `tests/test_model.py` (parameter counts and integration numbers measured directly, not checked into git - rerun the snippets in §13 to reproduce)
- Step 4 training + memory bank (§14): `src/trench_ids/cl/`, `configs/train.yaml` (gitignored `runs/step4/` output, rerun via `trench-train` to reproduce)
- Step 5 transferability estimation (§15): `src/trench_ids/cl/transferability.py` (gitignored `runs/step4/transferability_task_*.json` output, produced by the same `trench-train` run as Step 4)
- Steps 6-8 relation-aware EWC (§16): `src/trench_ids/cl/ewc.py`, `src/trench_ids/cl/importance.py`, `docs/superpowers/specs/2026-07-21-relation-aware-ewc-design.md` (gitignored `runs/step4/importance_weights_task_*.json` and `runs/step4_baseline/forgetting_matrix.json` output, produced by the same `trench-train` run)
