# TRENCH-IDS: Transferable representation learning for continual heterogeneous graph-based IDS

> Transferable REpresentatioN learning for Continual Heterogeneous graph-based IDS.

**Status:** Benchmark frozen 2026-09-07. Headline: 3-layer residual HeteroGNN + replay, forgetting = 0.081 ± 0.008, final accuracy = 0.902 ± 0.005 (3 seeds). The full ablation across TCTRL, RFR, and FACT showed no forward-compatibility mechanism improved upon the replay baseline. See `project-metrics.md`.

## Summary

This project builds a continual-learning benchmark for network intrusion detection using heterogeneous graphs, and evaluates forgetting-mitigation methods on it.

**Key result:** A corrected 3-layer relation-specific heterogeneous GNN with residual connections + experience replay reduces catastrophic forgetting to **0.081 ± 0.008** (final accuracy **0.902 ± 0.005**), consistently beating a strong non-graph Flat+Host baseline (**0.105 ± 0.017** forgetting, **0.872 ± 0.016** accuracy) across all 3 seeds.

The original 1-layer GNN had **40.9% of its parameters unreachable** from the classification loss (6 non-Flow relations + 4 fusion modules), which distorted earlier conclusions. With residual connections enabling depth, the full relational architecture participates and outperforms the non-message-passing baseline.

## Canonical baseline

The headline scientific result is the **3-seed mean** of the 3-layer residual HeteroGNN + replay configuration:

> **Forgetting = 0.081 ± 0.008, Final accuracy = 0.902 ± 0.005**
> (seeds 42 / 1 / 2; see `runs/gnn_3layer_residual_s{42,1,2}/summary.json`)

A secondary single-seed configuration (`runs/strong_replay_baseline/`, seed 42, `warmup_epochs=2`) achieves **forgetting = 0.0673, accuracy = 0.9172** — same architecture, different `warmup_epochs`. The 3-seed mean is the canonical baseline because it carries an actual variance estimate; the single-seed number is a sensitivity run, not the headline.

## Datasets

Five NetFlow v2 datasets from UQ (https://staff.itee.uq.edu.au/marius/NIDS_datasets/):

| Dataset | Status | Contribution |
|---|---|---|
| NF-ToN-IoT-v2 | Used | Scanning, XSS, DDoS, Password, DoS, Injection |
| NF-CSE-CIC-IDS2018-v2 | Used | DDoS, DoS, Injection, Bot, BruteForce, Infiltration |
| NF-BoT-IoT-v2 | Used | Reconnaissance only (DDoS/DoS rows excluded) |
| NF-UNSW-NB15-v2 | Excluded | No surviving class after BoT-IoT reinstated |
| NF-UQ-NIDS-v2 | Excluded | Merged union — would double-count |

## Continual-learning benchmark

**6 tasks, 10 attack classes, similarity-driven isolate-and-bundle assignment:**

| Task | Attack classes | Max intra-task cosine sim |
|---|---|---:|
| T1 | Scanning | — (isolated) |
| T2 | Reconnaissance | — (isolated) |
| T3 | DDoS + Infiltration | -0.462 |
| T4 | DoS + Injection | -0.343 |
| T5 | Password + Bot | -0.046 |
| T6 | XSS + BruteForce | -0.390 |

Benign is present in every task (fresh per-task subset, no cross-task leakage).  
Step 1 quota-based sampling: each task gets 390K attack + 130K benign rows, waterfilled evenly across classes/datasets.  
Step 2: mini-graphs of ≤300 flows, chunked in capture order, 11 relations (6 original + 5 reverse).

## Methods evaluated

| Method | Forgetting | Accuracy | Notes |
|---|---:|---:|---|
| Fine-tuning (no replay) | 0.693 | 0.364 | Anchor |
| EWC (relation-aware) | 0.669 | 0.378 | Fisher audit: 97.8% in shared encoder |
| TRD (distillation) | 0.630–0.689 | 0.363–0.411 | Inverse weighting best; drift null |
| **Replay (1-layer GNN)** | **0.202** | **0.764** | Strongest |
| **Flat+Host + replay** | 0.105 | 0.872 | Non-graph baseline |
| **3-layer GNN + residual + replay** | **0.081** | **0.902** | Beats Flat+Host on all 3 seeds |
| Joint training (upper bound) | — | 0.876 | Non-continual ceiling |

## Mechanistic findings

1. **EWC is structurally limited** — Fisher mass audit: 97.8% in `shared` encoder (encoders + fusion + classifier), only 2.2% reachable by relation-specific weights. No λ setting helped across 5 orders of magnitude.

2. **1-layer GNN dead gradients** — 40.9% of params (6 non-Flow relations + 4 fusion modules) received zero gradient. Residual connections fix this.

3. **Transferability predicts forgetting on corrected architecture** — `targeted_by` relation: r = -0.960 (higher transferability → less future forgetting).

4. **Replay works by preserving behavior, not parameters** — Full gradient path through entire encoder on replayed data vs. EWC's parameter-space proxy.

5. **Cross-dataset generalization fails** — 0.08% accuracy on unseen-source attacks; confident wrong predictions.

## Installation

The test suite is CPU-only; the canonical benchmark requires GPU. Two environments are supported:

### Option A — `uv` venv (CPU; for tests and small models)

```bash
uv venv --python 3.10
uv pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest -q   # 373/373 pass
```

The `uv` venv installs `torch` without CUDA wheels by default. Tests pass in this environment. Full 3-layer training (3 seeds × ~10 min on RTX 3050) will take **10–20× longer on CPU** and is not recommended for reproducing the headline numbers.

### Option B — base conda (CUDA; for benchmark reproduction)

The headline numbers were produced with `PyTorch 2.14.0+cu130` on an `NVIDIA GeForce RTX 3050 A Laptop GPU` (compute capability 8.9, 4 GB). To use the same environment:

```bash
conda activate base       # or any env with torch CUDA wheels installed
cd TRENCH-IDS
pip install -e .[dev]
python -m trench_ids.cl.train model.num_layers=3 model.use_residual=true \
    replay.enabled=true ewc.lambda_r=0 ewc.lambda_s=0 ewc.lambda_u=0 \
    train.warmup_epochs=1 train.seed=42 paths.out_dir=runs/gnn_3layer_residual_s42
```

If you are on a CUDA-less machine, the same command still runs (PyTorch silently falls back to CPU) but takes much longer; the **numbers will still reproduce** because the training loop is deterministic given the seed.

If you prefer to keep the `uv` venv but want CUDA, install the matching CUDA wheel before `pip install -e .`:

```bash
uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -e ".[dev]"
```

**Random seeds:** `train.seed=42|1|2` is the full set used for the headline 3-seed baseline. `seed_everything()` (in `trench_ids/cl/train.py`) seeds `random`, `numpy`, `torch`, and `torch.cuda` together — do not seed only `torch` if you need bit-exact reproduction.

## Running experiments

```bash
# 3-layer GNN + replay (3 seeds)
trench-train model.num_layers=3 model.use_residual=true replay.enabled=true ewc.lambda_r=0 ewc.lambda_s=0 ewc.lambda_u=0 train.warmup_epochs=1 train.seed=42 paths.out_dir=runs/gnn_3layer_residual_s42
trench-train model.num_layers=3 model.use_residual=true replay.enabled=true ewc.lambda_r=0 ewc.lambda_s=0 ewc.lambda_u=0 train.warmup_epochs=1 train.seed=1 paths.out_dir=runs/gnn_3layer_residual_s1
trench-train model.num_layers=3 model.use_residual=true replay.enabled=true ewc.lambda_r=0 ewc.lambda_s=0 ewc.lambda_u=0 train.warmup_epochs=1 train.seed=2 paths.out_dir=runs/gnn_3layer_residual_s2

# Flat+Host + replay (3 seeds)
trench-train-flat replay.enabled=true model.use_host_features=true model.mlp_hidden=254 train.seed=42 paths.out_dir=runs/flathost_replay_s42
trench-train-flat replay.enabled=true model.use_host_features=true model.mlp_hidden=254 train.seed=1 paths.out_dir=runs/flathost_replay_s1
trench-train-flat replay.enabled=true model.use_host_features=true model.mlp_hidden=254 train.seed=2 paths.out_dir=runs/flathost_replay_s2
```

## Generate results report

```bash
python -m trench_ids.cl.report --out docs/results_final.md
```

## No-Cap Experiment (Robustness Benchmark)

**Purpose**: Evaluate whether the replay result survives when attack-volume capping is removed and natural task-size imbalance returns.

### Prerequisites
- UQ NIDS datasets in `dataset/` with structure:
  ```
  dataset/
    NF-ToN-IoT-v2/data/NF-ToN-IoT-v2.csv
    NF-CSE-CIC-IDS2018-v2/data/NF-CSE-CIC-IDS2018-v2.csv
    NF-BoT-IoT-v2/data/NF-BoT-IoT-v2.csv
  ```
- GPU with ≥8 GB VRAM recommended (67K graphs, batch_size=1–2)

### Run Full Pipeline
```bash
# Step 1: Preprocess (all attack rows, no quota cap)
conda run -n base python -m trench_ids.preprocess_no_cap --config configs/no-cap/preprocess.yaml

# Step 2: Graph construction (67,274 graphs, no downsampling)
conda run -n base python -m trench_ids.graphs --config configs/no-cap/graph.yaml

# Step 3: Training (4 phases × 3 seeds = 12 runs)
conda run -n base python scripts/run_no_cap.py --phase train
```

### Expected Outputs
- `data/no-cap/processed/task_{1..6}.parquet` + `manifest.json`
- `data/no-cap/graphs/task_{t}_{split}.pt` + `graph_counts.json` (67,274 graphs)
- `runs/no-cap/gnn_finetune_s{42,1,2}/summary.json`
- `runs/no-cap/gnn_replay_s{42,1,2}/summary.json`
- `runs/no-cap/gnn_ewc_s{42,1,2}/summary.json`
- `runs/no-cap/gnn_trd_transfer_s{42,1,2}/summary.json`

### Key Differences from Main Benchmark
| Aspect | Main (Capped) | No-Cap |
|--------|---------------|--------|
| Attack rows/task | 390K (quota) | All available |
| Task sizes | 520K each (1.0×) | 1.7M–5.0M (2.9× imbalance) |
| T2 benign ratio | 3:1 (exact) | 19.4:1 (pool exhausted) |
| Total graphs | 10,402 | 67,274 |
| Replay proportion | 16.5% of train | 1.2–3.4% of train |

## Test suite

```bash
.venv/Scripts/python.exe -m pytest -q
# 373 tests passing (CPU .venv is sufficient for the suite; PyTorch CUDA is not required)
```

## Final experimental state

The benchmark is frozen at `v1.0.0-experiments` (existing git tag). No further experiments are planned; further changes belong to the paper write-up.

- Headline results: `project-metrics.md`
- Complete run directory tree: `runs/`
- Per-class and per-task F1 for each arm: `docs/results_final.md` (regenerable with `python -m trench_ids.cl.report --out docs/results_final.md`)
- Pipeline metrics and methodology: `project-metrics.md`, `CLAUDE.md`