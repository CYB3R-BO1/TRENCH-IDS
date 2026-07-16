# Datasets Reference -NetFlow v2 NIDS

**Project:** TRENCH-IDS (Transferable Representation Learning for Continual Heterogeneous Graph-Based IDS)
**Scope:** Detailed reference for the four NetFlow v2 datasets on disk -their internals, class distributions, and how raw CSVs are turned into task-partitioned training data (Step 1: *Raw traffic preprocessing*). **As of 2026-07-13 (second round), the pipeline uses 3 of the four** (NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2, NF-BoT-IoT-v2) -NF-UNSW-NB15-v2 stays excluded (see `docs/dataset-plan.md` §1). NF-BoT-IoT-v2 is back in the active pipeline as of this revision, **restricted to its Reconnaissance class only** via `src/trench_ids/labels.py`'s `CLASS_DATASETS` mechanism -its own DDoS/DoS/Theft rows are not part of the candidate pool. §3.1 below (UNSW internals) remains measured reference only, not part of the active pipeline; §3.3 (BoT-IoT internals) is now active reference for the one class BoT-IoT contributes.
**Last verified:** 2026-07-10 (all row/class counts below were measured directly from the local CSVs, not copied from the README); BoT-IoT's active status and `CLASS_DATASETS` restriction updated 2026-07-14.

---

## 1. Overview

All datasets come from the University of Queensland NIDS collection
(https://staff.itee.uq.edu.au/marius/NIDS_datasets/) and follow the **standardized NetFlow v2 schema** of Sarhan et al. -the same 43 features are extracted from every source pcap, which is exactly what makes a *unified* heterogeneous graph schema viable across otherwise unrelated network testbeds.

On disk each dataset is a **BagIt bag** (`dataset/<name>/`):

```
<name>/
  bagit.txt, bag-info.txt          # BagIt metadata
  manifest-sha1.txt                # SHA-1 checksums of payload
  tagmanifest-sha1.txt
  FurtherInformation.txt           # link to UQ eSpace record + license
  data/
    <name>.csv                     # the flow records (the actual dataset)
    NetFlow_v2_Features.csv        # feature dictionary (43 features)
```

### Datasets on disk (4; 3 actually used -- see scope note above)

| Dataset | Rows | Attack rows | Benign rows | File size |
|---|---:|---:|---:|---:|
| NF-UNSW-NB15-v2 | 2,390,275 | 95,053 (3.98%) | 2,295,222 (96.02%) | 0.44 GB |
| NF-ToN-IoT-v2 | 16,940,496 | 10,841,027 (63.99%) | 6,099,469 (36.01%) | 2.68 GB |
| NF-BoT-IoT-v2 | 37,763,497 | 37,628,460 (99.64%) | 135,037 (0.36%) | 6.05 GB |
| NF-CSE-CIC-IDS2018-v2 | 18,893,708 | 2,258,141 (11.95%) | 16,635,567 (88.05%) | 3.22 GB |
| **Pooled (4)** | **75,987,976** | **50,822,681 (66.88%)** | **25,165,295 (33.12%)** | ~12.4 GB |

### Why NF-UQ-NIDS-v2 is excluded

The pooled totals above (75,987,976 / 50,822,681 / 25,165,295) are **identical** to NF-UQ-NIDS-v2's totals. NF-UQ-NIDS-v2 is literally the row-wise union of these four datasets under a unified label taxonomy -it is not an independent domain. Including it would double-count every flow and bias evaluation, so it is present on disk but never used for training/eval. See `docs/dataset-plan.md` §1.

---

## 2. Common schema (43 features + 2 labels = 45 columns)

Every dataset CSV has an **identical 45-column header** (verified byte-for-byte across all four): the 43 NetFlow v2 features followed by `Label` and `Attack`.

- **`Label`** -binary: `0` = benign, `1` = attack.
- **`Attack`** -string attack category (or `Benign`). Raw strings are **not** harmonized across datasets -see §4.

### 2.1 Feature groups

The 43 features split into a small set of **flow-key/identifier** columns (which become graph *nodes*) and the **flow-statistic** columns (which become the Flow node's feature vector). This grouping is the bridge between the raw CSV and the heterogeneous graph node types (Host / Flow / Protocol / Port / Service).

| Group | Features | Role in graph |
|---|---|---|
| **Flow-key / identifiers (4)** | `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT` | IPs → **Host** nodes; ports → **Port** nodes. Not used as numeric Flow features (identifiers cause leakage/overfitting). |
| **Protocol / service (2)** | `PROTOCOL` (L4 proto byte), `L7_PROTO` (L7 numeric) | `PROTOCOL` → **Protocol** node; `L7_PROTO` → **Service** node. |
| **Volume (4)** | `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS` | Flow features |
| **Timing (3)** | `FLOW_DURATION_MILLISECONDS`, `DURATION_IN`, `DURATION_OUT` | Flow features |
| **TCP flags (3)** | `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS` | Flow features (cumulative flag bitmasks) |
| **TTL (2)** | `MIN_TTL`, `MAX_TTL` | Flow features |
| **Packet size (4)** | `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN` | Flow features |
| **Rate / throughput (4)** | `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`, `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT` | Flow features |
| **Retransmission (4)** | `RETRANSMITTED_IN_BYTES`, `RETRANSMITTED_IN_PKTS`, `RETRANSMITTED_OUT_BYTES`, `RETRANSMITTED_OUT_PKTS` | Flow features |
| **Packet-size histogram (5)** | `NUM_PKTS_UP_TO_128_BYTES`, `NUM_PKTS_128_TO_256_BYTES`, `NUM_PKTS_256_TO_512_BYTES`, `NUM_PKTS_512_TO_1024_BYTES`, `NUM_PKTS_1024_TO_1514_BYTES` | Flow features |
| **TCP window (2)** | `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT` | Flow features |
| **ICMP (2)** | `ICMP_TYPE` (type*256+code), `ICMP_IPV4_TYPE` | Flow features (0 for non-ICMP flows) |
| **DNS (3)** | `DNS_QUERY_ID`, `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER` | Flow features (0 for non-DNS flows) |
| **FTP (1)** | `FTP_COMMAND_RET_CODE` | Flow features (0 for non-FTP flows) |

Total: 4 + 2 + 37 = **43**. The 37 flow-statistic features form the numeric Flow-node vector; the 6 identifier/proto/port fields seed the other node types.

> **Note on protocol-specific fields.** ICMP/DNS/FTP columns are `0` (not NaN) when the flow's protocol doesn't apply -the datasets are dense, with no missing values, but these columns are extremely sparse and near-constant for most flows. Worth flagging for feature scaling/selection.

---

## 3. Per-dataset internals

Attack-class counts below are measured from the local CSVs. Percentages are of that dataset's total rows.

### 3.1 NF-UNSW-NB15-v2 (excluded from current pipeline)
- **Status:** stays excluded in the 3-dataset/10-class/6-task design (unchanged conclusion from the prior 2-dataset design) -- confirmed no class survives dropping it once NF-BoT-IoT-v2 is back in the pool; see `docs/dataset-plan.md` §1 and `docs/attack-class-counts.md`.
- **Origin:** UNSW-NB15 testbed (IXIA PerfectStorm traffic generator, mixed benign + synthetic attacks). NetFlow-re-extracted from the original pcaps.
- **Character:** heavily benign-dominated (96%), broadest attack taxonomy (9 classes), but several classes are tiny (Worms = 164 flows). The only source of `Fuzzers`, `Analysis`, `Shellcode`, `Generic`, `Worms`, `Exploits`.
- **Raw `Attack` labels:** PascalCase.

| Attack (raw) | Count | % of dataset |
|---|---:|---:|
| Benign | 2,295,222 | 96.02% |
| Exploits | 31,551 | 1.320% |
| Fuzzers | 22,310 | 0.933% |
| Generic | 16,560 | 0.693% |
| Reconnaissance | 12,779 | 0.535% |
| DoS | 5,794 | 0.242% |
| Analysis | 2,299 | 0.096% |
| Backdoor | 2,169 | 0.091% |
| Shellcode | 1,427 | 0.060% |
| Worms | 164 | 0.007% |

### 3.2 NF-ToN-IoT-v2
- **Origin:** ToN-IoT IoT/IIoT testbed (Telemetry of IoT devices, Operating systems, Network). NetFlow re-extracted from public pcaps.
- **Character:** attack-majority (64%), large and fairly balanced across several attack types. Dominant source of `xss`, `scanning`, `injection`, `password`, `mitm`, `ransomware`. Only source of `MITM`, `Password`, `Ransomware`, `Injection`, `XSS` at scale.
- **Raw `Attack` labels:** **lowercase** (`dos`, `ddos`, `xss`, …) -differs from every other dataset.

| Attack (raw) | Count | % of dataset |
|---|---:|---:|
| Benign | 6,099,469 | 36.01% |
| scanning | 3,781,419 | 22.32% |
| xss | 2,455,020 | 14.49% |
| ddos | 2,026,234 | 11.96% |
| password | 1,153,323 | 6.81% |
| dos | 712,609 | 4.21% |
| injection | 684,465 | 4.04% |
| backdoor | 16,809 | 0.099% |
| mitm | 7,723 | 0.046% |
| ransomware | 3,425 | 0.020% |

### 3.3 NF-BoT-IoT-v2 (active, restricted to Reconnaissance only)
- **Status:** back in the active pipeline as of the 3-dataset/10-class/6-task design. Restricted via `CLASS_DATASETS` to contribute **Reconnaissance only** (2,620,999 rows, task T2, isolated) -- its own `DDoS` (18,331,847) and `DoS` (16,673,183) rows below are measured reference but are **not** pooled into the DDoS/DoS canonical classes (those stay ToN+CSE-only), and `Theft` (2,431) is excluded entirely (`EXCLUDED_CLASSES`, below the candidate-pool floor). Benign is also not drawn from BoT-IoT for any task other than T2 (Reconnaissance), since T2 is the only task BoT-IoT contributes to.
- **Origin:** BoT-IoT botnet testbed. NetFlow re-extracted from public pcaps.
- **Character:** almost entirely attack (99.64% -only 135k benign flows in 37.8M rows). Massive `DoS`/`DDoS`/`Reconnaissance` volume; `Theft` is tiny (2,431). Extreme benign scarcity makes it a poor benign source but the dominant DoS/DDoS/Recon source -- though only Reconnaissance is sanctioned for this benchmark.
- **Raw `Attack` labels:** PascalCase.

| Attack (raw) | Count | % of dataset |
|---|---:|---:|
| DDoS | 18,331,847 | 48.54% |
| DoS | 16,673,183 | 44.15% |
| Reconnaissance | 2,620,999 | 6.94% |
| Benign | 135,037 | 0.36% |
| Theft | 2,431 | 0.006% |

### 3.4 NF-CSE-CIC-IDS2018-v2
- **Origin:** CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity / AWS testbed). NetFlow re-extracted from original pcaps.
- **Character:** benign-dominated (88%), the **most granular raw labels** -14 distinct attack strings that map to a handful of families. Only source of `Bot` and `Infiltration`; a major source of `BruteForce`, `Web Attacks`, `DoS`, `DDoS`.
- **Raw `Attack` labels:** verbose and inconsistent -mixed casing (`DDoS` vs `DDOS`), stray spaces (`Brute Force -XSS`), and a **misspelling** (`Infilteration`). These require explicit normalization (§4).

| Attack (raw) | Count | % of dataset | Maps to family |
|---|---:|---:|---|
| Benign | 16,635,567 | 88.05% | Benign |
| DDOS attack-HOIC | 1,080,858 | 5.72% | DDoS |
| DoS attacks-Hulk | 432,648 | 2.29% | DoS |
| DDoS attacks-LOIC-HTTP | 307,300 | 1.63% | DDoS |
| Bot | 143,097 | 0.76% | Bot |
| Infilteration *(sic)* | 116,361 | 0.62% | Infiltration |
| SSH-Bruteforce | 94,979 | 0.50% | BruteForce |
| DoS attacks-GoldenEye | 27,723 | 0.147% | DoS |
| FTP-BruteForce | 25,933 | 0.137% | BruteForce |
| DoS attacks-SlowHTTPTest | 14,116 | 0.075% | DoS |
| DoS attacks-Slowloris | 9,512 | 0.050% | DoS |
| Brute Force -Web | 2,143 | 0.011% | Web Attacks |
| DDOS attack-LOIC-UDP | 2,112 | 0.011% | DDoS |
| Brute Force -XSS | 927 | 0.005% | Web Attacks |
| SQL Injection | 432 | 0.002% | Injection |

---

## 4. Cross-dataset label harmonization

The raw `Attack` strings are inconsistent across datasets (case, spelling, granularity). Preprocessing **must** normalize them to a single canonical taxonomy before task partitioning. The canonical classes and their continual-learning task assignment come from `docs/dataset-plan.md` §2.

**Canonical label → raw strings that map to it:**

| Canonical class | `CLASS_DATASETS` source(s) | Task | Raw strings (by dataset) |
|---|---|---|---|
| Benign | -(all tasks) | -(all tasks) | `Benign` (ToN, CSE, BoT) -shared negative class, injected fresh into every task from that task's own contributing dataset(s); not a task of its own |
| Scanning | ToN | T1 | ToN `scanning` |
| Reconnaissance | BoT | T2 | BoT `Reconnaissance` |
| XSS | ToN | T3 | ToN `xss` |
| DDoS | ToN, CSE | T3 | ToN `ddos`; CSE `DDoS attacks-LOIC-HTTP`, `DDOS attack-HOIC`, `DDOS attack-LOIC-UDP`. **Not** BoT's own `DDoS` rows -- `RAW_TO_CANONICAL` maps BoT's raw `DDoS` string to the same canonical `DDoS` class, but `CLASS_DATASETS` restricts the DDoS *class's* sanctioned sources to `{ToN, CSE}`, so BoT's 18,331,847 `DDoS`-labeled rows are excluded from this class (they'd otherwise dwarf ToN+CSE's combined 3,416,504). |
| Password | ToN | T4 | ToN `password` |
| Infiltration | CSE | T4 | CSE `Infilteration` *(sic)* |
| DoS | ToN, CSE | T5 | ToN `dos`; CSE `DoS attacks-Hulk`, `DoS attacks-GoldenEye`, `DoS attacks-SlowHTTPTest`, `DoS attacks-Slowloris`. **Not** BoT's own `DoS` rows -- same `CLASS_DATASETS` restriction as DDoS above (BoT's 16,673,183 `DoS`-labeled rows excluded). |
| Injection | ToN, CSE | T5 | ToN `injection`; CSE `SQL Injection` |
| Bot | CSE | T6 | CSE `Bot` |
| BruteForce | CSE | T6 | CSE `FTP-BruteForce`, `SSH-Bruteforce` |
| Backdoor | -(dropped) | -(dropped) | ToN `backdoor` -below the candidate-pool floor; see dataset-plan.md §2 |
| MITM | -(dropped) | -(dropped) | ToN `mitm` -below floor |
| Ransomware | -(dropped) | -(dropped) | ToN `ransomware` -below floor |
| Web Attacks | -(dropped) | -(dropped) | CSE `Brute Force -Web`, `Brute Force -XSS` -below floor |
| Theft | -(dropped) | -(dropped) | BoT `Theft` -below floor (2,431 rows; BoT-IoT's only other class besides Reconnaissance, not sanctioned for the pool) |

Exploits, Fuzzers, Generic, Analysis, Shellcode, Worms are UNSW-NB15-only classes and don't appear in any retained dataset -- see §3.1 (kept as reference; NF-UNSW-NB15-v2 is not part of the active pipeline).

> **Design note.** Task assignment is **isolate-and-bundle** (dataset-plan.md §2.2): classes forming a mutual "conflict clique" (every pair above a similarity threshold) cannot avoid a conflicting co-location no matter how they're split, so each gets its own singleton task. For this 10-class pool at threshold 0.35, **{Scanning} and {Reconnaissance} are the max conflict clique** and are each isolated into singleton tasks; the remaining eight classes pair off via minimum-weight matching (XSS+DDoS, Password+Infiltration, DoS+Injection, Bot+BruteForce). Stable across thresholds 0.21-0.55 -- see `docs/attack-similarity-matrix.md`.
>
> **No benign-only task.** Benign has no task of its own -it is present in every task as the negative class (see §5 step 5). A standalone single-class benign task is degenerate for a classifier and was dropped.
>
> **Below-floor classes dropped.** Backdoor, MITM, Ransomware (ToN), Web Attacks (CSE), Theft (BoT) fall below the candidate-pool floor (2,431-16,809 vs. the ~116K-3.8M range of the kept ten -- see dataset-plan.md §2.1) and are dropped from all tasks. All still map canonically (so label normalization never fails) but are filtered out during preprocessing (`EXCLUDED_CLASSES` in `labels.py`). Note Bot, BruteForce, and Infiltration are **not** in this dropped list -- unlike the prior 2-dataset design, they're now part of the active candidate pool (see dataset-plan.md §2.1's reversal note).

### Continual-learning task table

Authoritative task design is **dataset-plan.md §2.2**; this mirrors it. Benign is present in **every** task (fresh per-task subset, §5 step 5); the table lists each task's *attack* classes.

| Task | Attack classes | Max intra-task cosine similarity | Contributing datasets |
|---|---|---|---|
| T1 | Scanning | -(isolated) | ToN |
| T2 | Reconnaissance | -(isolated) | BoT |
| T3 | XSS, DDoS | -0.361 | ToN, CSE |
| T4 | Password, Infiltration | -0.457 | ToN, CSE |
| T5 | DoS, Injection | -0.343 | ToN, CSE |
| T6 | Bot, BruteForce | -0.297 | CSE |

---

## 5. Step 1 -Raw traffic preprocessing (how the data is processed)

This is the concrete pipeline that turns the three raw CSVs into task-partitioned, model-ready flow tables. Graph construction (Step 2) consumes its output; it is *not* part of Step 1. Pipeline shape (per `docs/superpowers/specs/2026-07-13-dataset-task-respec-design.md` §3): load -> assign `flow_id` -> merge (concat across datasets) -> clean (label harmonize, class-source restrict, dedup) -> split -> persist.

1. **Stream & assign `flow_id`.** Read each dataset CSV in chunks (files are up to 6 GB -do not load whole into RAM). Tag every row with a `source_dataset` column (`ToN` / `CSE` / `BoT`) so per-domain provenance survives pooling, and a stable `flow_id` (`f"{source_dataset}-{original_csv_row_number}"`) assigned during this streaming read, before concatenation across datasets -- gives every flow an identity that survives through to the Step 2 Flow graph node, unlike the prior "row position within a mini-graph chunk" identity.

2. **Merge.** Concatenate the per-dataset chunks (still per-class/per-dataset restricted at this point, see step 3) into the pooled working set.

3. **Clean: normalize labels.** Map raw `Attack` strings → canonical class via the §4 table (case-fold, strip spaces, fix `Infilteration`→`Infiltration`). Fail loudly on any unmapped string so new/unexpected labels can't slip through silently.

4. **Clean: restrict class sources & assign task IDs.** Restrict each canonical class's rows to its sanctioned dataset(s) via `CLASS_DATASETS` (e.g. BoT-IoT's own `DDoS`/`DoS` rows are dropped here, even though they mapped canonically in step 3 -- BoT-IoT's only sanctioned class is Reconnaissance). Map canonical *attack* class → task `T1…T6` via the §4 task table. Benign has no task (drawn per-task in step 6); excluded classes (Backdoor, MITM, Ransomware, Web Attacks, Theft) are filtered out here, not assigned. **No pre-graph per-class sampling cap is applied** -- every row that passes the `CLASS_DATASETS` restriction and isn't excluded is kept in full (`attack_per_class_cap` has been removed from `configs/preprocess.yaml`; see the scale-up note below).

5. **Separate feature roles.** Reserve the 4 identifier fields (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`) + `PROTOCOL` + `L7_PROTO` for graph-node construction (Step 2); keep the 37 flow-statistic features as the numeric Flow vector. Do **not** feed raw IP/port as numeric model inputs.

6. **Per-task benign sampling.** For each task, draw a **fresh** benign subset from the *same dataset(s)* that contribute that task's attack classes (not one shared benign pool) -- T1 draws only from ToN, T2 only from BoT, T6 only from CSE; T3/T4/T5 draw from both ToN and CSE. Rationale (dataset-plan.md §2/§3): keeps traffic distribution consistent within a task and isolates attack-pattern forgetting from benign-distribution forgetting during CL evaluation. `benign_per_dataset_cap`/`benign_per_task` (`configs/preprocess.yaml`) remain Step 1 pool-size safeguards -- they are no longer what determines the final attack:benign ratio a task's graphs see; that's now a Step 2 concern (`configs/graph.yaml: sampling.benign_ratio`, see dataset-plan.md §3.3).

7. **Clean: dedup.** Drop exact-duplicate flow records (identical across all original NetFlow columns) from the per-task frame (`dedup: true`).

8. **Splits.** Produce train/val/test per task with a fixed seed, stratified by canonical class (rare classes still land in every split). Split assignment stored in a `split` column.

9. **Persist.** Write one Parquet per task under `data/processed/` (all original NetFlow columns + `flow_id`/`source_dataset`/`canonical_label`/`task`/`split`), plus `manifest.json` recording per-class / per-split / per-source counts and seed. Feature values are stored **raw**.

> **Normalization (deferred to Step 2).** dataset-plan.md §2 specifies **global** normalization statistics, computed once from the full training pool across all datasets (not per-dataset) to avoid dataset-identity shortcuts. Step 1 stores raw features; the global scaler is fit on the pooled train split and applied when the Flow-node feature vector is assembled in Step 2 (keeps the fit-on-train, leak-safe property while matching the graph-build stage).

> **Scale-up from cap removal.** Removing `attack_per_class_cap` grows Step 1's attack-row output from the prior design's low-hundred-thousands to **~15.7M rows** (sum of the 10-class pool in `docs/attack-class-counts.md`, minus the held-out val/test share). Step 2's mini-graph count and `.pt` file sizes grow proportionally -- flagged, not fully resolved: see dataset-plan.md §3.3.

### Resolved preprocessing decisions (implemented in `configs/preprocess.yaml`)
- **Per-class cap:** none -- `attack_per_class_cap` has been removed; every row passing the `CLASS_DATASETS`/`EXCLUDED_CLASSES` filters is kept in full. Class-level balancing is now a Step 2 concern (`sampling.benign_ratio`; DoS/DDoS-style downsampling still pending, to operate on constructed graphs -- see dataset-plan.md §3.3).
- **Benign per task:** `benign_per_task: 8000`, drawn from each task's contributing datasets (`benign_per_dataset_cap: 60000` pool per dataset).
- **Dedup:** `dedup: true`.
- **Output granularity:** one Parquet per task; provenance retained via the `source_dataset` column; flow identity retained via `flow_id`.
- **Excluded classes:** Backdoor, MITM, Ransomware, Web Attacks, Theft (`EXCLUDED_CLASSES` in `labels.py`). Bot, BruteForce, and Infiltration are **no longer excluded** -- they're part of the active 10-class candidate pool (task T6 / T4).

---

## 6. Data-quality notes

- **No missing values**, but ICMP/DNS/FTP columns are `0`-filled and near-constant for the vast majority of flows (sparse, low-variance -candidates for careful scaling or de-emphasis).
- **Label-string inconsistency** across datasets (case, spacing, `Infilteration` misspelling, `DDoS`/`DDOS`) -handled in §5 step 3; treat the §4 map as the single source of truth.
- **Class imbalance** within and across the three retained datasets (benign ranges 0.36%-88% for BoT/ToN/CSE). Backdoor, MITM, Ransomware (ToN), Web Attacks (CSE), and Theft (BoT) (2,431-16,809) fall well short of the 10-class pool's ~116K-3.8M range and are **dropped** (§4); the rarest *kept* class is Infiltration (116,361), the most abundant is Scanning (3,781,419).
- **Identifier leakage risk** -IPs/ports are testbed-specific; using them as numeric features would let the model memorize the testbed rather than learn attack behavior. They belong in the graph structure, not the feature vector.
- **Duplicate flows** -the v2 NetFlow datasets are known to contain repeated flow records; decide on dedup policy during preprocessing.

