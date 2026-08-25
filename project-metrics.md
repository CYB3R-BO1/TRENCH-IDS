# TRENCH-IDS - Project Metrics

This file is the single place to find every hard number about the project - dataset scale, class distribution, task design, pipeline output, and test/runtime stats. Update it whenever Step 1/Step 2 are rerun or the class/task design changes; treat every number here as sourced from an actual file on disk (cited inline), never estimated. For the narrative/architecture picture - what each component does and how they connect (including branches, shared inputs, dead-end diagnostics, and the one real feedback loop) - see `flow.md`. For a dated changelog of when each piece was built and why - see `log.md`.

**Benchmark objective:** TRENCH-IDS is a benchmark for **task-incremental continual learning on heterogeneous, graph-structured network intrusion detection data** - 6 sequential tasks, each introducing new attack classes (plus a fresh Benign subset) built from real NetFlow v2 traffic, feeding into a relation-specific heterogeneous GNN trained under continual learning (full proposed pipeline: `CLAUDE.md` "Research goal").

> ⚠️ **Every number recorded before 2026-08-17 was produced on a different benchmark and is not comparable to anything produced after it** — that includes all of `project-metrics.md` (repo root, not under `docs/`), `TRENCH-IDS_Full_Project_Documentation.md`, and `TRENCH-IDS_Project_Summary.pdf`. Never quote a figure from those next to a current one without saying which benchmark each came from. `docs/methodology-2026-08-17.md` is the current source of truth for the data pipeline and the continual-learning mechanism. The task design, dataset selection, and encoder architecture described below and in `docs/dataset-plan.md` are unchanged and still authoritative.

### Benchmark at a glance (current benchmark, post-2026-08-17 rebuild)

| Metric | Value |
|---|---:|
| Source datasets | 3 (NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2, NF-BoT-IoT-v2) |
| Attack classes | 10 |
| Continual-learning tasks | 6 |
| Total attack rows (post corrupted-row filter) | 15,689,086 |
| Total rows incl. benign (Step 1 output) | 15,736,524 |
| Node types | 5 (Flow, Host, Protocol, Service, Port) |
| Relation types | 11 (6 original + 5 individually-named reverse relations, added 2026-07-19) |
| Graph size (max flows/mini-graph) | 300 |
| Total mini-graphs, `benign_ratio=3.0` set | 69,737 |
| `benign_ratio` sweep candidates produced | 2.0 / 3.0 / 4.0 |
| Step 3 model parameters (default config: `hidden_dim=64`, 1 layer, 11 relations, exact well-known-port embedding) | 263,378 |
| Test suite | 372 tests passing, `ruff check` clean |

### Best CL method (post-rebuild benchmark, 3 seeds)

| Method | Forgetting | Accuracy | Notes |
|---|---:|---:|---|
| **3-layer GNN + residual + replay** | **0.081 ± 0.008** | **0.902 ± 0.005** | Beats Flat+Host on all 3 seeds |
| **Flat+Host + replay** | 0.105 ± 0.017 | 0.872 ± 0.016 | Non-graph baseline |
| Original 1-layer GNN + replay | 0.202 ± 0.015 | 0.764 ± 0.014 | 40.9% unreachable params |
| Fine-tuning (no replay) | 0.693 ± 0.005 | 0.364 ± 0.001 | Anchor |
| Joint-training upper bound | — | 0.876 | Non-continual ceiling |

### Pipeline overview

```
Raw NetFlow v2 CSVs (5 datasets on disk, 3 used)
        |
        v
Class filtering + label harmonization      labels.py: CLASS_DATASETS, EXCLUDED_CLASSES
        |
        v
Task assignment (similarity-driven          similarity.py, task_design.py
  isolate-and-bundle, 6 tasks)
        |
        v
Step 1: sample + dedup + corrupted-row      preprocess.py -> data/processed/*.parquet
  filter + stratified train/val/test split
        |
        v
Step 2: heterogeneous graph construction +  graphs.py -> data/graphs*/task_*_*.pt
  benign_ratio sampling + mini-graph chunking
        |
        v
Mini-graphs (list[HeteroData] per task/split, 11-relation schema)
        |
        v
Step 3: relation-specific encoding + attention   src/trench_ids/model/*.py
  fusion -> per-relation + fused node embeddings
        |
        v
Step 4: sequential fine-tuning T1->T6 +          src/trench_ids/cl/train.py,
  relation-specific memory bank                    memory_bank.py
        |
        v
Step 5: transferability estimation (per-task,    transferability.py
  per-relation cosine sim. vs. memory bank)
        |
        v
Steps 6-8: relation importance weights           importance.py, ewc.py
  (ImportanceMLP) + relation-aware Online EWC -
  implemented; SETTLED as not reducing forgetting
  on this benchmark (Fisher audit: 97.8% in shared)
        |
        v
Step 9: memory bank refresh - folded into the
  Step 4/6-8 training loop
        |
        v
Step 10a: transferability analysis - relation    transferability_report.py
  ranking, class-pair ranking, raw-feature
  comparison, validated on corrected 3-layer GNN
        |
        v
Step 10b: unseen-attack inference pipeline       unseen_data.py, unseen_graphs.py,
  for genuinely unseen traffic - completed        inference_unseen.py
        |
        v
Professor direction: try replay-based forgetting   train.py (replay),
mitigation. Experience replay closes almost all     train_joint.py
of the forgetting gap; joint-training run gives
the non-CL upper bound.
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

Per-dataset restriction (`CLASS_DATASETS` in `src/trench_ids/labels.py`) and measured raw counts:

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

**Total candidate-pool raw attack rows: 15,689,140**. Excluded entirely (`EXCLUDED_CLASSES`): Backdoor, MITM, Ransomware, Web Attacks, Theft (2,431–16,809 rows each, all well below the 116K–3.8M candidate-pool range).

---

## 3. Class-similarity matrix (top pairs)

Cosine similarity between standardized per-class mean feature vectors (37 flow-statistic features, 5,000-row sample/class). Full 10×10 matrix: `similarity_matrix.csv`. Top and bottom pairs:

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

Task-design threshold: **0.35**, stable across the entire **0.21–0.55** band.

---

## 4. Continual-learning task design (6 tasks)

Isolate-and-bundle grouping on the 10-class pool, with a **size-aware tie-break** (`src/trench_ids/task_design.py`'s `min_weight_grouping(..., sizes=...)`): among all pairings satisfying the 0.35 similarity threshold, the tie-break minimizes the largest resulting task's size instead of total pairwise similarity.

| Task | Attack classes | Dataset(s) | Max intra-task cosine sim |
|---|---|---|---:|
| T1 | Scanning | ToN | - (isolated) |
| T2 | Reconnaissance | BoT | - (isolated) |
| T3 | DDoS + Infiltration | ToN, CSE | -0.462 |
| T4 | DoS + Injection | ToN, CSE | -0.343 |
| T5 | Password + Bot | ToN, CSE | -0.046 |
| T6 | XSS + BruteForce | ToN, CSE | -0.390 |

Benign is present in every task (fresh per-task subset from that task's contributing dataset(s)). Task-size imbalance: **2.9× largest:smallest** (T1 3,789,419 vs. T5 1,304,420 rows) - down from 14× under the old pairing.

---

## 5. Step 1 output - `data/processed/` (real run, `seed=42`)

| Task | Theme | Rows (incl. benign) | Class counts | Dups dropped | Corrupted dropped | Train / Val / Test | Sources | File size |
|---|---|---:|---|---:|---:|---|---|---:|
| 1 | Scanning | 3,789,419 | Scanning 3,781,419 / Benign 8,000 | 0 | 0 | 2,652,593 / 568,413 / 568,413 | ToN 3,789,419 | 48.3 MB |
| 2 | Reconnaissance | 2,628,994 | Reconnaissance 2,620,994 / Benign 8,000 | 5 | 0 | 1,840,296 / 394,349 / 394,349 | BoT 2,628,994 | 60.2 MB |
| 3 | DDoS + Infiltration | 3,540,254 | DDoS 3,416,452 / Infiltration 115,804 / Benign 7,998 | 557 | 54 | 2,478,178 / 531,039 / 531,037 | ToN 2,030,232 / CSE 1,510,022 | 79.6 MB |
| 4 | DoS + Injection | 1,889,505 | DoS 1,196,608 / Injection 684,897 / Benign 8,000 | 0 | 0 | 1,322,654 / 283,426 / 283,425 | ToN 1,401,074 / CSE 488,431 | 42.1 MB |
| 5 | Password + Bot | 1,304,420 | Password 1,153,323 / Bot 143,097 / Benign 8,000 | 0 | 0 | 913,094 / 195,663 / 195,663 | ToN 1,157,323 / CSE 147,097 | 26.5 MB |
| 6 | XSS + BruteForce | 2,583,932 | XSS 2,455,020 / BruteForce 120,912 / Benign 8,000 | 0 | 0 | 1,808,752 / 387,590 / 387,590 | ToN 2,459,020 / CSE 124,912 | 63.5 MB |
| **Total** | | **15,736,524** | | **562** | **54** | | | **320 MB** |

---

## 6. Step 2 output - three `benign_ratio` sweep sets

Node types (5): Flow, Host, Protocol, Service, Port. Relations (11): 6 original + 5 reverse. `graph_size=300` for all three.

### 6.1 `benign_ratio=3.0` - `data/graphs/` (default, regenerated 2026-07-19 for 11-relation schema)

| Task | Train | Val | Test | Total | Disk (train/val/test) |
|---|---:|---:|---:|---:|---|
| 1 Scanning | 11,765 | 2,521 | 2,521 | 16,807 | 1002 / 214 / 215 MB |
| 2 Reconnaissance | 8,155 | 1,748 | 1,748 | 11,651 | 665 / 142 / 142 MB |
| 3 DDoS + Infiltration | 10,990 | 2,355 | 2,355 | 15,700 | 932 / 199 / 199 MB |
| 4 DoS + Injection | 5,854 | 1,255 | 1,255 | 8,364 | 496 / 106 / 106 MB |
| 5 Password + Bot | 4,034 | 865 | 865 | 5,764 | 341 / 73 / 73 MB |
| 6 XSS + BruteForce | 8,015 | 1,718 | 1,718 | 11,451 | 674 / 144 / 144 MB |
| **Total** | **48,813** | **10,462** | **10,462** | **69,737** | **~5.73 GB** |

### 6.2 `benign_ratio=2.0` - `data/graphs_ratio2/`

Total: **78,452** graphs (~6.47 GB)

### 6.3 `benign_ratio=4.0` - `data/graphs_ratio4/`

Total: **65,378** graphs (~5.36 GB)

---

## 7. Graph topology statistics (primary set: `benign_ratio=3.0`)

| Task | Graphs (train) | Avg flow | Avg host | Avg protocol | Avg service | Avg port | Avg total nodes | Avg total edges | Avg host degree | Avg host↔host density |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Scanning | 11,765 | 300.0 | 63.0 | 2.2 | 16.7 | 244.6 | 626.4 | 1,597.2 | 9.5 | 0.0253 |
| 2 Reconnaissance | 8,155 | 300.0 | 12.7 | 2.1 | 8.2 | 197.0 | 520.0 | 1,531.7 | 47.3 | 0.2237 |
| 3 DDoS + Infiltration | 10,990 | 300.0 | 96.7 | 2.3 | 13.9 | 43.3 | 456.3 | 1,599.8 | 6.2 | 0.0110 |
| 4 DoS + Injection | 5,854 | 300.0 | 88.8 | 2.2 | 14.5 | 54.3 | 459.7 | 1,601.0 | 6.8 | 0.0132 |
| 5 Password + Bot | 4,034 | 299.9 | 89.0 | 2.2 | 14.8 | 42.6 | 448.5 | 1,594.2 | 6.7 | 0.0123 |
| 6 XSS + BruteForce | 8,015 | 300.0 | 83.8 | 2.1 | 14.4 | 42.7 | 442.9 | 1,588.2 | 7.2 | 0.0130 |
| **Overall** | 69,737 | 300.0 | 70.5 | 2.2 | 13.8 | 118.4 | 504.8 | 1,585.2 | 8.5 | 0.0499 |

---

## 8. Step 3 - Relation-specific heterogeneous GNN

**Architecture:** 5 node types, 11 relations. Flow: 5 incoming. Host: 3 incoming. Protocol/Service/Port: 1 each.

### 7.1 Parameter count (real instantiation against regenerated graph)

| Config | Total params | Node encoders | Relation-conv | Attention fusion |
|---|---:|---:|---:|---:|
| `hidden_dim=64`, 1 layer, **exact well-known-port scheme** | **263,378** | 84,562 | 136,576 | 42,240 |
| `hidden_dim=64`, 2 layers, residual | 389,218 | 84,562 | 2×136,576 | 2×42,240 |
| `hidden_dim=64`, 3 layers, residual | 515,058 | 84,562 | 3×136,576 | 3×42,240 |

### 7.2 Key architectural fixes (2026-07-19)

1. **5 reverse relations** added: `originated_by`, `terminated_by`, `targeted_by`, `protocol_of`, `service_of` — individually named with passive-voice semantics.
2. **Per-relation self-concatenation** (GraphSAGE-style): `RelationSpecificConv` concatenates destination node's own embedding into each relation's aggregated message before a per-relation combine layer. Fixes 0%-self-signal gap in every relation embedding (worst for Flow).
3. **Well-known port embedding**: exact row per port 0-1023 + 32 log-buckets for 1024-65535 tail (replaces pure log-bucket scheme that collapsed distinct services).

---

## 9. Continual-learning results (final, 3 seeds)

### 9.1 Main method comparison (replay vs fine-tuning vs EWC vs TRD)

| Method | Forgetting | Accuracy | F1-macro | Note |
|---|---:|---:|---:|---|
| Fine-tuning | 0.693 ± 0.005 | 0.364 ± 0.001 | 0.167 | Anchor |
| EWC (λ=1.0) | 0.669 ± 0.040 | 0.378 ± 0.026 | 0.190 | Not meaningful |
| TRD-uniform | 0.651 ± 0.027 | 0.390 ± 0.021 | 0.229 | Without replay |
| TRD-transfer | 0.647 ± 0.029 | 0.394 ± 0.021 | 0.236 | Without replay |
| TRD-inverse | 0.630 ± 0.008 | 0.411 ± 0.003 | 0.263 | **Meaningful** vs uniform |
| TRD-drift | 0.689 ± 0.008 | 0.363 ± 0.004 | 0.168 | Null |
| **Replay (1-layer GNN)** | **0.202 ± 0.015** | **0.764 ± 0.014** | 0.734 | **Strongest** |
| Replay+TRD | 0.211 ± 0.048 | 0.731 ± 0.037 | 0.679 | Worse than replay alone |

### 9.2 Architecture ablation (all with replay, 3 seeds)

| Architecture | Forgetting | Accuracy | F1-macro | Params (reachable) |
|---|---:|---:|---:|---:|
| Flat (per-flow only) | 0.209 ± 0.006 | 0.739 ± 0.005 | 0.661 | 263,586 |
| **Flat+Host** (per-flow + host agg) | **0.105 ± 0.017** | **0.872 ± 0.016** | 0.832 | 263,442 |
| Flat+Host (matched reachable) | 0.145 ± 0.021 | 0.832 ± 0.020 | 0.809 | 156,577 |
| GNN (1-layer, attention fusion) | 0.174 ± 0.032 | 0.774 ± 0.026 | 0.734 | 156,562* |
| GNN (1-layer, concat fusion) | 0.251 ± 0.029 | 0.691 ± 0.022 | 0.624 | 156,562* |
| GNN (2-layer, residual) | 0.115 ± 0.016 | 0.851 ± 0.015 | 0.812 | 389,218 |
| **GNN (3-layer, residual)** | **0.081 ± 0.008** | **0.902 ± 0.005** | 0.867 | 515,058 |

*\*1-layer GNN reachable params: 156,562 (40.9% of 264,850 nominal — 6 non-Flow relations + 4 fusion modules get zero gradient)*

**Key finding:** Original 1-layer GNN loses to Flat+Host. 2-layer GNN with residual restores gradient flow but does **not** beat Flat+Host (forgetting 0.115 vs 0.105, accuracy 0.851 vs 0.872). **3-layer GNN with residual connections beats Flat+Host** consistently across all 3 seeds (paired diff: -0.025 forgetting, +0.030 accuracy, same sign on every seed). Depth matters: 2 layers restores gradient flow but isn't enough; 3 layers enables the full relational architecture to participate.

### 9.3 Per-seed breakdown (3-layer GNN + residual vs Flat+Host, both with replay)

| Seed | GNN Forgetting | Flat+Host Forgetting | GNN Acc | Flat+Host Acc | Forget Diff | Acc Diff |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.089 | 0.091 | 0.896 | 0.887 | -0.002 | +0.009 |
| 1 | 0.082 | 0.095 | 0.904 | 0.879 | -0.013 | +0.024 |
| 2 | 0.070 | 0.129 | 0.908 | 0.850 | -0.059 | +0.057 |
| **Mean** | **0.081** | **0.105** | **0.902** | **0.872** | **-0.025** | **+0.030** |

### 9.4 Per-seed breakdown (2-layer GNN + residual vs Flat+Host, both with replay)

| Seed | GNN Forgetting | Flat+Host Forgetting | GNN Acc | Flat+Host Acc | Forget Diff | Acc Diff |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.130 | 0.091 | 0.845 | 0.887 | +0.039 | -0.042 |
| 1 | 0.093 | 0.095 | 0.872 | 0.879 | -0.002 | -0.007 |
| 2 | 0.121 | 0.129 | 0.838 | 0.850 | -0.008 | -0.013 |
| **Mean** | **0.115** | **0.105** | **0.851** | **0.872** | **+0.010** | **-0.021** |

**Note:** 2-layer GNN + residual does not consistently beat Flat+Host — paired diffs are mixed across seeds. Only 3-layer + residual achieves consistent improvement.

---

## 10. Mechanistic findings

### 10.1 EWC Fisher mass audit (2026-08-23)

| Parameter group | Fisher mass | % of total |
|---|---:|---:|
| `shared` (encoders + fusion + classifier) | 97.8% | 97.8% |
| Flow relations (5) | 2.2% | 2.2% |
| Other relations (6) | 0.0% | 0.0% |

**Conclusion:** Relation-aware EWC could only ever modulate 2.2% of its own penalty. Structural limitation, not tuning issue. 5-order λ sweep + `num_layers=2` ablation both confirmed null.

### 10.2 Transferability validation (corrected 3-layer GNN)

| Relation | S_r(t) vs future forgetting correlation |
|---|---:|
| `targeted_by` | **-0.960** (strong: higher transferability → less forgetting) |
| `terminated_by` | -0.567 |
| `originates` | -0.381 |
| `protocol_of` | -0.320 |
| `service_of` | -0.069 |

**Conclusion:** On corrected architecture, `targeted_by` transferability score strongly predicts future forgetting. Signal was noise on 1-layer GNN (all correlations weak).

### 10.3 Why replay beats EWC

| Aspect | EWC | Replay |
|---|---|---|
| Preserves | Parameters (weighted by Fisher) | Behavior on old data |
| Gradient reach | <3% (only relation-specific projections) | 100% (full encoder) |
| Scale | Requires λ ~ 100–1000; penalty > L_cls at those λ | Same scale as L_cls; no λ tuning |
| Result | Forgetting 0.669–0.742 (all configs) | **Forgetting 0.081** (3-layer GNN) |

---

## 11. Cross-dataset generalization (Step 10b)

| Dataset | Classes | Accuracy | Note |
|---|---|---:|---|
| BoT-IoT DDoS/DoS (unseen source) | Seen classes | **0.0008** | Confident mis-mapping (e.g. DDoS→Recon 99.3%) |
| UNSW-NB15 Recon/DoS (unseen source) | Seen classes | ~0 | Similar |
| 5 novel classes (Backdoor, MITM, Ransomware, Web Attacks, Theft) | Novel | N/A | Collapses to 1-2 known classes with softmax ≥0.89 |

**Conclusion:** Strong CL performance does not imply cross-dataset generalization. Model keys on dataset-specific feature distributions, not transferable attack semantics.

---

## 12. Test suite

**372 tests passing** (`.venv/Scripts/python.exe -m pytest -q`), 0 failures. Covers: labels, preprocessing, similarity, task design, vocab, graphs, model, flat model, EWC, distillation, importance, transferability, memory bank, train, train_flat, train_joint, inference, inference_unseen, evaluate, novelty, drift, replay selection, compare_runs, report, reproducibility, integration, graph_composition.

---

## 13. Reproducibility

| Run | Command | Seeds |
|---|---|---|
| 3-layer GNN + replay | `trench-train model.num_layers=3 model.use_residual=true replay.enabled=true ewc.lambda_r=0 ewc.lambda_s=0 ewc.lambda_u=0 train.warmup_epochs=1 paths.out_dir=runs/gnn_3layer_residual_s{seed}` | 42, 1, 2 |
| 2-layer GNN + replay | `trench-train model.num_layers=2 model.use_residual=true replay.enabled=true ewc.lambda_r=0 ewc.lambda_s=0 ewc.lambda_u=0 train.warmup_epochs=1 paths.out_dir=runs/gnn_2layer_residual_s{seed}` | 42, 1, 2 |
| Flat+Host + replay | `trench-train-flat replay.enabled=true model.use_host_features=true model.mlp_hidden=254 paths.out_dir=runs/flathost_replay_s{seed}` | 42, 1, 2 |
| Fine-tuning (anchor) | `trench-train replay.enabled=false ewc.lambda_r=0 ewc.lambda_s=0 ewc.lambda_u=0` | 42, 1, 2 |
| Joint training (upper bound) | `trench-train-joint` | 42 |

---

## 14. Final experimental state (git tag)

```bash
git tag v1.0.0-experiments
```

Commit: `e7888e6` - "Freeze final experimental state: 3-layer residual GNN + replay validated across 3 seeds"

All results reproducible from run directories under `runs/`. `docs/results_final.md` (generated by `trench_ids.cl.report`) is the canonical source of CL metrics.

---

*Last updated: 2026-08-24 — reflects final experimental state after 2/3-layer residual GNN multi-seed validation.*