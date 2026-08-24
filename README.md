# TRENCH-IDS: Transferable representation learning for continual heterogeneous graph-based IDS

> Transferable REpresentatioN learning for Continual Heterogeneous graph-based IDS.

## Summary

This project builds a continual-learning benchmark for network intrusion detection using heterogeneous graphs, and evaluates forgetting-mitigation methods on it.

**Key result:** A corrected 3-layer relation-specific heterogeneous GNN with residual connections + experience replay reduces catastrophic forgetting to **0.081 ± 0.008** (final accuracy **0.902 ± 0.005**), consistently beating a strong non-graph Flat+Host baseline (**0.105 ± 0.017** forgetting, **0.872 ± 0.016** accuracy) across all 3 seeds.

The original 1-layer GNN had **40.9% of its parameters unreachable** from the classification loss (6 non-Flow relations + 4 fusion modules), which distorted earlier conclusions. With residual connections enabling depth, the full relational architecture participates and outperforms the non-message-passing baseline.

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

```bash
uv venv --python 3.10
uv pip install -e ".[dev]"
```

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

## Test suite

```bash
.venv/Scripts/python.exe -m pytest -q
# 372 tests passing
```

## Final experimental state

```bash
git tag v1.0.0-experiments
```

All results in `docs/results_final.md` and run directories under `runs/`. See `project-metrics.md` for complete pipeline metrics.