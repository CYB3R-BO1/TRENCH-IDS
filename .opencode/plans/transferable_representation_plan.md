# TRENCH-IDS: Forward-Compatible Representation Learning for Continual Heterogeneous Graph-based IDS

## Research & Design Plan (Revised: Minimal Falsifiable Experiment First)

**Status**: Pre-implementation literature/design study phase  
**Core Principle**: One falsifiable experiment at a time. No implementation until the minimal test is designed.

---

## 1. Current Evidence Summary

| Finding | Evidence | Implication |
|---------|----------|-------------|
| **Replay works** | Forgetting 0.081 ± 0.008, Acc 0.902 ± 0.005 (3 seeds) | **Keep as baseline** |
| **Transfer exists** | Counterfactual relation-reset experiments showed functional forward transfer | Transferable info IS in representations |
| **TCTRL failed** | Oracle 74% vs Baseline 95% - adapter path fundamentally broken | Meta-learning not the answer |

**Key insight from counterfactuals**: Transferable information lives in **relation-specific representations** (your counterfactuals showed this). The question is how to make those representations **persist and transfer** to future tasks.

---

## 2. Research Question (Refined)

> **Can forward-compatible, relation-aware representation learning improve forward transfer to unseen attack classes in continual heterogeneous graph-based intrusion detection?**

**Operational definition of "forward-compatible"**: The representation learned during current tasks contains richer, more diverse features that enable faster/better adaptation to future unseen attack classes — without requiring future-task information during training.

---

## 3. Minimal Falsifiable Experiment Sequence

### Phase 0: Global RFR (Week 1-2)
**Hypothesis**: Adding RFR effective-rank loss during T1 training improves forward transfer on T2→T6.

```
Current TRENCH-IDS (3-layer hetero GNN + Replay)
        │
        ├── Baseline (Replay only)
        │
        └── + Global RFR (L_RFR = -log(erank) on T1 fused embeddings)
              │
              └── B1 LOTO: compare AULC on T2→T6
```

**RFR Core** (from Kim et al. 2024):
- During T1 (base session): `L = L_CE + λ_RFR * (-log(erank(Z)))`
- `erank` = exp(-Σ(p_i log p_i)) where p_i = normalized singular values of normalized representation batch
- Only applied during T1 (base session); novel sessions use standard CE + replay

**Success Criteria**:
- erank(Z) increases during T1 training
- T1 accuracy does not collapse
- Forward-transfer AULC on T2→T6 improves vs replay baseline
- Forgetting ≤ baseline (0.081)

### Phase 1: Relation-Aware RFR (Week 3-4)
**Hypothesis**: Per-relation effective rank regularization outperforms global RFR on heterogeneous graph.

```
Replay baseline
       │
       ├── Global RFR (fused embeddings only)
       │
       └── Relation-aware RFR: Σ_r -log(erank(Z_r))  (uniform weights)
             │
             └── Compare: does per-relation richness help heterogeneous transfer?
```

**Relation-Aware RFR Loss** (uniform weights, no Stage A leakage):
```
L_RFR_rel = Σ_{r∈{5 relations}} -log(erank(Z_r))
```
Applied to each relation's embeddings independently during T1.

**Success Criteria**:
- Per-relation erank increases
- Relation-aware > Global RFR on forward-transfer AULC
- Forgetting ≤ baseline

### Phase 2: Controlled B0-Style Test (Before B1 LOTO)
**Before spending compute on full B1 LOTO**, run a controlled small-scale test:

```
T3→T4 transition only (same as old B0)
        │
        ├── Replay baseline
        │
        ├── Global RFR + Replay
        │
        └── Relation-aware RFR + Replay
              │
              └── Check: AULC(1-3), forgetting, erank increase
```

Only if this shows clear signal → proceed to full B1 LOTO (5 transitions × 2 seeds).

---

## 4. What We Explicitly Defer

| Deferred Mechanism | Reason |
|--------------------|--------|
| **FACT virtual prototypes** | Different hypothesis (reserve space vs enrich representation) |
| **HKD attention distillation** | Different mechanism (preserve vs enrich) |
| **Stage A T_r weighting** | Leakage: T_r measured from future transitions |
| **FACT virtual prototypes per relation** | Different mechanism (reserve space vs enrich) |
| **HKD attention distillation** | Different mechanism (preserve attention vs enrich rank) |
| **Dense embeddings** | Independent improvement; test after core hypothesis |
| **SSL pretraining (GraphIDS)** | Different paradigm (SSL vs supervised rank regularization) |
| **B1 LOTO (5×2)** | Only after Phase 0+1 show signal |

---

## 5. Minimal Experiment Design (Phase 0)

### Experiment: Global RFR on TRENCH-IDS

**Setup**:
- Same benchmark: 6 tasks, 3 datasets, replay=200, 3-layer residual GNN
- T1 (Scanning) = base session
- T2→T6 = novel sessions
- Replay buffer = 200 (unchanged)

**Implementation**:
```python
# In train.py, during T1 (task_id == 0) only:
def get_erank(reps):
    Z = F.normalize(reps, p=2, dim=1)  # [N, 64]
    _, s, _ = torch.svd(Z)
    eig_val = s * s / Z.shape[0]
    logged_erank = - (eig_val * torch.log(eig_val + 1e-8)).sum()
    return logged_erank

# In T1 training loop:
L_CE = F.cross_entropy(logits, y)
L_RFR = - get_erank(fused_embeddings)  # maximize erank = minimize -log(erank)
loss = L_CE + lambda_rfr * L_RFR
```

**Lambda Sweep**: λ_RFR ∈ {0.01, 0.1, 1.0} (start with 0.1 per paper)

**Monitoring** (per epoch):
- `erank` value (should increase)
- T1 classification accuracy
- T1 loss components

### Evaluation Protocol (Phase 0)
| Metric | Target |
|--------|--------|
| **erank(Z)** | Increases during T1 training |
| **T1 accuracy** | Within 2% of baseline |
| **Forward-transfer AULC** (T2→T6) | > Baseline (Δ > 0.0037) |
| **Forgetting** | ≤ 0.081 (replay baseline) |

---

## 6. Implementation Scope (Phase 0 Only)

### Files to Modify
| File | Change |
|------|--------|
| `src/trench_ids/rfr.py` | **NEW**: `get_erank()`, `rfr_loss()` utilities |
| `src/trench_ids/cl/train.py` | Add RFR loss in T1 training loop only |
| `configs/train.yaml` | Add `rfr_enabled`, `rfr_lambda` |

### Config Addition
```yaml
# configs/train.yaml
rfr:
  enabled: true
  lambda: 0.1
  apply_only_base: true  # only during T1
```

---

## 7. What We Explicitly Do NOT Implement (Yet)

| Mechanism | Status |
|-----------|--------|
| Relation-aware RFR (per-relation erank) | After Phase 0 succeeds |
| FACT virtual prototypes | Deferred - different hypothesis |
| HKD attention distillation | Deferred - different mechanism |
| Stage A T_r weighting | Leakage - never |
| FACT virtual prototypes per relation | Deferred |
| HKD attention distillation | Deferred |
| Dense embeddings (Protocol/Service/Port) | Independent - after core works |
| SSL pretraining (GraphIDS) | Different paradigm |
| B1 LOTO (5×2) | Only after Phase 0+1 signal |

---

## 8. Success Criteria (Phase 0)

| Metric | Threshold |
|--------|-----------|
| **erank(Z)** | Increases monotonically during T1 |
| **T1 accuracy** | Within 2% of baseline (95.1%) |
| **Forward-transfer AULC** (T2→T6) | > Baseline + 0.0037 |
| **Forgetting** | ≤ 0.081 (replay baseline) |
| **T1→T2 controlled test** | AULC improvement visible |

---

## 8. Next Step

**No implementation yet.** 

**Next step**: Confirm the Phase 0 design above is the correct minimal falsifiable test. Specifically:

1. **RFR loss formulation correct?** `L_RFR = -log(erank)` on T1 fused embeddings only
2. **Lambda sweep range?** {0.01, 0.1, 1.0} with 0.1 as default
3. **Apply only during T1?** Yes - base session only per RFR paper
3. **Monitor erank per epoch?** Yes - sanity check it increases
4. **Phase 0 evaluation** = full B1 LOTO or just T3→T4 controlled?
5. **Proceed to implementation** of `src/trench_ids/rfr.py` + `train.py` integration?

**Once you confirm**, I'll implement the minimal `src/trench_ids/rfr.py` and integrate into `train.py` for Phase 0.