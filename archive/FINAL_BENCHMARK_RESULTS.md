# TRENCH-IDS Final Benchmark Results

## Executive Summary

This benchmark establishes a strong replay baseline for continual heterogeneous graph-based intrusion detection and systematically evaluates forward-compatibility mechanisms. **No tested forward-compatibility mechanism improved upon the strong replay baseline.**

The central scientific finding is the clear separation between:

- **Backward compatibility** (retention of old tasks) — excellent
- **Forward transfer** (efficient learning of future tasks) — highly transition-dependent and deteriorating

---

## The Core Result: Two Distinct Phenomena

```
BACKWARD COMPATIBILITY
        ↓
Forgetting = 0.081 ± 0.008
        ✅ Strong — replay solves retention

FORWARD TRANSFER
        ↓
AULC = 0.288 → 0.403 → 0.187 → 0.053 → 0.080
        ⚠ Strongly transition-dependent
```

The model can retain old knowledge extremely well while simultaneously becoming progressively worse at learning new knowledge quickly. This is a very clean conceptual distinction that ordinary final-accuracy/forgetting tables hide.

---

## Locked Baseline (3-layer Residual Hetero GNN + Replay)

**Primary baseline — 3-seed mean:**

| Metric | Value |
|--------|------:|
| **Average Forgetting** | **0.081 ± 0.008** |
| **Final Average Accuracy** | **0.902 ± 0.005** |
| **Final Average Val Accuracy** | **0.896 ± 0.005** |
| **Mean B1 AULC(1-3)** | **0.202** |
| **Runtime (per seed)** | ~10 min on RTX 3050 |

Architecture: 3-layer residual HeteroGNN (hidden_dim=64, attn_dim=128), replay buffer=200, replay_fraction=0.3, warmup_epochs=1, `train.warmup_epochs=1`, 5 epochs/task (1 warmup + 4 full), batch=8, lr=0.001, seeds {42, 1, 2}.

Source: `runs/gnn_3layer_residual_s{42,1,2}/summary.json`.

**Secondary sensitivity run — single seed, warmup_epochs=2:**

| Metric | Value |
|--------|------:|
| Average Forgetting | 0.0673 |
| Final Average Accuracy | 0.9172 |
| Final Average Val Accuracy | 0.9157 |
| Runtime | 1,483s |

Same architecture, same seed (42), `warmup_epochs=2` instead of 1. Source: `runs/strong_replay_baseline/summary.json`. Reproduced within run-to-run noise by the independent 2026-09-07 audit (`VERIFICATION_REPORT.md`).

The 3-seed mean is the canonical scientific result (it carries an actual variance estimate). The single-seed `warmup_epochs=2` run is reported as a sensitivity check, not as a separate claim.

---

## Forward Transfer (B1 LOTO) - Mean AULC(1-3) over 5 Transitions × 2 Seeds

| Transition | Mean AULC | 95% CI |
|------------|----------:|----------|
| T1→T2 | 0.288 | [0.284, 0.292] |
| T2→T3 | 0.403 | [0.397, 0.408] |
| T3→T4 | 0.187 | [0.075, 0.299] |
| T4→T5 | 0.053 | [0.036, 0.069] |
| T5→T6 | 0.080 | [0.072, 0.088] |
| **Mean** | **0.202** | [0.175, 0.229] |

**Key finding**: Forward transfer peaks at T2→T3 (0.403) then collapses sharply after T3.

### B1 LOTO Transition Details (Baseline)

| Transition | AULC | Final Acc | Final F1 | Dataset Shift |
|------------|-----:|----------:|---------:|---------------|
| T1→T2 | 0.288 | 0.915 | 0.159 | ToN→BoT |
| T2→T3 | 0.403 | 0.858 | 0.220 | BoT→CSE |
| T3→T4 | 0.187 | 0.250 | 0.036 | CSE→Mixed |
| T4→T5 | 0.053 | 0.250 | 0.036 | CSE→CSE |
| T5→T6 | 0.080 | 0.250 | 0.036 | CSE→Mixed |

---

## Deliberate Ablation: Three Hypotheses About Why Future Transfer Might Be Limited

Rather than presenting TCTRL, RFR, and FACT as random failed ideas, this benchmark tested **three qualitatively different hypotheses** about the source of the forward-transfer limitation:

```
HYPOTHESIS 1: Task-conditioned adaptation
  "When T4 arrives, look at T4 and modify the old representation so it fits T4 better"
  → TCTRL (target-conditioned adapter + FOMAML)
  → NO IMPROVEMENT over baseline (final_acc ≈ 0.758, AULC ≈ 0.150)

HYPOTHESIS 2: Feature richness
  "Make the representation richer during initial training so it has capacity for future classes"
  → RFR (effective-rank regularization)
  → NO MEANINGFUL IMPROVEMENT (0.064 vs 0.067 forgetting, same AULC)

HYPOTHESIS 3: Reserved embedding space
  "Explicitly reserve space for future classes via virtual prototypes"
  → FACT (forward-compatible training)
  → HARMFUL (-31% AULC: 0.202 → 0.140)
```

**Result**: None of these mechanisms resolved the forward-transfer limitation. This systematic ablation across qualitatively different hypotheses is stronger than testing a single family of methods.

---

## Method Comparison

| Method | Forgetting | Final Acc | Mean AULC | Hypothesis Tested |
|--------|-----------:|----------:|-----------:|-------------------|
| **Replay (3L residual, 3-seed)** | **0.081 ± 0.008** | **0.902 ± 0.005** | **0.202** | **Canonical baseline** |
| Replay (3L residual, single-seed `warmup=2`) | 0.0673 | 0.9172 | — | Sensitivity check |
| Replay (1L) | 0.371 | 0.533 | 0.202 | Architecture diagnostic |
| TCTRL (B0 POC, final epoch) | — | **0.758** | **0.150** | Task-conditioned adaptation |
| RFR λ=0.1 (1L) | 0.365 | 0.534 | 0.162 | Feature richness (wrong arch) |
| RFR λ=0.1 (3L) | 0.0635 | 0.9165 | ~0.202 | Feature richness (correct arch) |
| Relation-RFR | 0.442 | 0.473 | 0.125 | Per-relation richness |
| **FACT (λ=1.0)** | 0.059 | 0.918 | 0.140 | Virtual prototype reservation |

Sources on disk: `runs/gnn_3layer_residual_s{42,1,2}` (3-seed baseline), `runs/strong_replay_baseline` (sensitivity), `runs/final_baseline` (1L), `runs/meta_transfer/B0_poc/summary.json` (TCTRL, see `meta_trained/metrics.json` for per-epoch trajectory), `runs/rfr_final` and `runs/rfr_strong` (RFR), `runs/fact_strong` (FACT). The TCTRL row uses the **on-disk final-epoch numbers** (`final_acc=0.758`, `aulc_1_3=0.150` from `meta_trained/metrics.json`); intermediate single-episode checkpoints within B0 (e.g. epoch 0 F1 ≈ 0.036) are not reported as headline numbers.

---

## Experimental Timeline

1. **Stage A (Transfer Analysis)**: Counterfactual relation-reset experiments proved transferable information exists in relation-specific representations
2. **Stage B0 (TCTRL)**: Task-conditioned adapter with FOMAML — final-epoch accuracy/AULC not better than baseline
3. **Stage B0 (RFR)**: Effective-rank regularization — no meaningful improvement
4. **Stage B0 (FACT)**: Virtual prototypes — degraded performance
5. **Stage B1 LOTO**: Established benchmark on strong replay baseline

---

## Key Scientific Conclusions

1. **Replay is the dominant mechanism** for catastrophic forgetting mitigation (0.081 ± 0.008 forgetting across 3 seeds)

2. **Forward transfer exists but is highly transition-dependent** — peaks at T2→T3, collapses after T3

3. **Three distinct forward-compatibility mechanisms all failed** to improve upon replay:
   - TCTRL (task-conditioned adaptation): no improvement
   - RFR (feature richness): no improvement
   - FACT (virtual prototypes): harmful

4. **The results are consistent with progressive representation specialization as a possible source of the late-sequence forward-transfer collapse.** The experiments demonstrate the collapse clearly but do not yet causally prove that specialization is the mechanism; other explanations (intrinsic task difficulty, dataset shift interaction with replay, etc.) remain possible.

---

## Final Scientific Position

The benchmark establishes that:

> **Replay is highly effective for backward retention, but forward transfer remains strongly transition-dependent and deteriorates later in the task sequence. TCTRL, RFR, and FACT did not improve that forward-transfer behavior.**

The results **characterize** a forward-transfer limitation and show that several intuitive mechanisms do not resolve it. This is a legitimate research outcome — the project characterized a phenomenon, tested diverse hypotheses about its cause, and documented which mechanisms fail.

---

## Reproducibility

All runs use:
- **Seeds**: 42, 1, 2 (three seeds for variance estimation on the canonical baseline)
- **Architecture**: 3-layer residual HeteroGNN, hidden=64, attn_dim=128
- **Replay**: 200 graphs/task, 30% fraction, uniform selection
- **Training**: 5 epochs/task (1 warmup + 4 full), batch=8, lr=0.001
- **Device**: NVIDIA GeForce RTX 3050 Laptop GPU (PyTorch 2.14.0+cu130, CUDA 13.0)
- **Data**: 6 tasks, 3 datasets (ToN, BoT, CSE), ~1,214 train graphs/task

**Independent audit:** `VERIFICATION_REPORT.md` (2026-09-07) re-ran the canonical 3-layer GNN + replay training (seed 42) on the same hardware and reproduced both the 3-seed baseline (within ~0.004) and the single-seed sensitivity run (within ~0.008). The 8 previously-blocked tests in `tests/test_train_flat.py` were unblocked by a one-line import fix (drop the `tests.` prefix on line 9); the full suite now reports 373/373 pass.

---

## Benchmark frozen: 2026-09-07
