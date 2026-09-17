# Literature Extraction: Core Papers for Forward-Compatible Representation Learning

---

## 1. RFR — "Improving Forward Compatibility in Class Incremental Learning by Increasing Representation Rank and Feature Richness" (Neural Networks, 2024)

**Authors**: Jaeill Kim, Wonseok Lee, Moonjung Eo, Wonjong Rhee  
**Code**: https://github.com/wonseok1017/NN-RFR

### Core Mathematical Formulation

#### Effective Rank (erank) — Roy & Vetterli (2007)
For a batch of representations `R ∈ ℝ^(N×d)`:

1. **L2 Normalize**: `Z = normalize(R)` where each row has unit L2 norm
2. **Covariance**: `A = Z^T Z / N` (where N = batch size)
3. **SVD**: `_, s, _ = torch.svd(Z)` → singular values `s`
4. **Eigenvalues**: `eig_val = s^2 / N` (normalized singular values squared)
5. **Logged Effective Rank** (Shannon Entropy):
   ```
   logged_erank = - (eig_val * log(eig_val)).sum()
   erank = exp(logged_erank)
   ```

#### RFR Loss (added during base session only)
```
L_RFR = - logged_erank  # = -log(erank)
```

**Total Base Session Loss**:
```
L_total = L_CE + λ_RFR * L_RFR
```

Where `λ_RFR` is a hyperparameter (default 0.1 in their code).

### Key Theoretical Result
- **Theorem**: Shannon entropy of representation is maximized when effective rank is maximized
- Since entropy = information content, maximizing erank → richer features
- Connection: `H(R) = log(erank)` where H is Shannon entropy

### Training Protocol (CIL Setting)
- **Base Session**: Train with `L_CE + λ_RFR * L_RFR` (RFR loss applied)
- **Novel Sessions**: Train WITHOUT RFR loss (only CE + distillation losses)
- **Evaluation**: Freeze feature extractor, train only new classification heads for novel tasks

### Key Results (CIFAR-100, ImageNet-Subset)
| Metric | Improvement |
|--------|-------------|
| Novel Task AIC | +2.5% to +7.8% (split 10/5/2) |
| Catastrophic Forgetting | -1.3% to -2.9% reduction |
| Average Incremental Acc | +0.36% to +3.74% across 11 CIL methods |

### RFR Code Reference (from `rfr_ucir.py`)
```python
def get_erank(self, reps):
    try:
        _, s, _ = torch.svd(torch.nn.functional.normalize(reps))
        eig_val = s * s / reps.shape[0]
        logged_erank = - (eig_val * torch.log(eig_val)).nansum()
    except:
        logged_erank = 0
    return logged_erank

# In train_epoch:
if t == 0:  # base session only
    loss_rfr = -1 * self.get_erank(self.hook_fc.input)
    loss = loss_ucir + self.rfr_coef * loss_rfr
```

---

## 2. FACT — "Forward Compatible Few-Shot Class-Incremental Learning" (CVPR 2022)

**Authors**: Da-Wei Zhou, Fu-Yun Wang, Han-Jia Ye, Liang Ma, Shiliang Pu, De-Chuan Zhan  
**Code**: https://github.com/zhoudw-zdw/CVPR22-Fact

### Core Concept: Forward Compatibility
> **Backward Compatibility**: New model compatible with old data (resists forgetting)
> **Forward Compatibility**: Current model prepared for future classes (growable + provident)

> *"If the former model works poorly, the latter model would degrade consequently. It is impossible to maintain backward compatibility with limited instances in incremental stages."*

### Two Mechanisms

#### 1. Growable — Virtual Prototypes (Reserve Space)
- Pre-assign `V` virtual prototypes `P_v ∈ ℝ^(d×V)` as "virtual classes"
- `V = N_B` (number of novel classes expected)

**Virtual Loss (L_v)**:
```
f_v(x) = [W, P_v]^T φ(x)  # output logits with virtual prototypes

ŷ = argmax_v P_v^T φ(x)  # nearest virtual prototype

L_v = CE(f_v(x), y) + CE(mask(f_v(x)), ŷ)
```
- First term: standard classification
- Second term: match non-target logits to nearest virtual prototype `ŷ`
- Effect: Squeezes known classes, reserves space for virtual classes

#### 2. Provident — Virtual Instances (Forecast Future)
- Generate virtual instances via manifold mixup: `z = η·φ(x_i) + (1-η)·φ(x_j)` for `y_i ≠ y_j`
- Treat mixed instance as virtual new class

**Forecast Loss (L_f)**:
```
L_f = CE(f_v(z), ŷ_v) + CE(mask(f_v(z)), y_nearest_known)
```
- First term: push mixed instance toward virtual prototype
- Second term: trade-off between virtual and known classes

**Total FACT Loss**:
```
L_FACT = L_v + L_f = L_1 + L_2 + L_3 + L_4
```

### Incremental Inference with Virtual Prototypes
During inference, virtual prototypes participate in classification:
```
p(y_i|x) = exp(w_i^T(φ(x)+p_v)) / Σ_j exp(w_j^T(φ(x)+p_v))
```
Virtual prototypes act as informative basis vectors.

### Key Ablation Results
| Loss Components | Effect |
|-----------------|--------|
| L_1 (CE only) | Overspreads embedding, no forward compatibility |
| + L_2 (virtual prototypes) | Growable: reserves space, compactifies known classes |
| + L_3 (virtual instances) | Provident: forecasts future, better compatibility |
| + L_4 (symmetric) | More compact embedding, better FSCIL |

---

## 3. TagFex — "Task-Agnostic Guided Feature Expansion" (CVPR 2025)

**Authors**: Zheng et al.  
**Code**: https://github.com/bwnzheng/TagFex_CVPR2025

### Core Idea
Maintain **task-agnostic feature learner** alongside task-specific representations.

```
Task-Agnostic Feature Learner (frozen/shared)
        ↓
   Guides expansion of
   Task-Specific Features
```

### Core Mechanism
- **TagNet**: Learns task-agnostic features from all seen data
- **Feature Expansion**: Task-specific features expanded using TagNet guidance
- **Loss**: `L = L_CE + λ * L_expand` where `L_expand` encourages diversity

---

## 4. NEGSC — "Applying Self-Supervised Learning to Network Intrusion Detection" (Computer Networks, 2024)

**Authors**: Renjie Xu et al.  
**Code**: https://github.com/renj-xu/NEGSC

### Core Contribution
First **multiclass unsupervised** GNN for NIDS using flow features + graph topology.

### Architecture: NEGAT + NEGSC
- **NEGAT Encoder**: Edge-featured GAT (NetFlow-Edge Graph Attention Network)
  - Edge features as primary, nodes as auxiliary relays
  - Attention on edges, not nodes
- **NEGSC**: Generative Subgraph Contrast
  - Generative module: creates contrastive subgraphs from center node + neighbors
  - Interpolated graph for negative samples
  - Structured contrastive loss on edge features + local topology

### Key for TRENCH-IDS
- **Edge-centric** (matches your heterogeneous graph: Flow = edges in their formulation)
- **Multiclass unsupervised** (not just binary anomaly detection)
- **Local topology awareness** (subgraph contrast)

---

## 5. GraphIDS — "Self-Supervised Learning of Graph Representations for Network Intrusion Detection" (NeurIPS 2025)

**Authors**: Guerra et al.

### Core Idea
**Joint GNN + Transformer Masked Autoencoder** on benign traffic only.

### Architecture
1. **E-GraphSAGE**: Inductive GNN encoding flows with local topological context
2. **Transformer Masked Autoencoder**: Reconstructs masked flow embeddings
   - 15-30% masking ratio
   - MSE reconstruction loss
   - Joint GNN + Transformer training (end-to-end gradients)

### Results
- 99.98% PR-AUC, 99.61% macro F1
- Outperforms baselines by 5-25%
- **Anomaly detection** via reconstruction error threshold

---

## 5. GCP — "Graph-Contrastive Pretraining for Payload-Free Encrypted-Traffic Intrusion Detection" (Algorithms, 2026)

**Authors**: Arcos-Argudo et al.

### Core Contribution
Graph-contrastive pretraining (InfoNCE) + **cross-dataset OOD transfer evaluation**.

### Key Finding
> "OOD transfer is strongly source–target dependent, with the most reliable transfer within closely related DoH domains, highlighting dataset shift as a first-class evaluation criterion for encrypted-traffic IDS."

### Evaluation Protocol
- Frozen-embedding linear probing
- Cross-dataset OOD transfer (CICIDS2017 ↔ UNSW-NB15 ↔ DoH-Combined)
- 5 seeds, 95% CI

---

## 6. HERO/MKD — Heterogeneous Continual Graph Learning (2025)

**Authors**: Sun et al. (HERO), related MKD paper

### Core Components
1. **G-MM**: Gradient-based Meta-learning Module for rapid adaptation
2. **E-HSS**: Efficient Heterogeneous Subgraph Sampling (DiSCo)
   - Diversity sampling on target nodes + metapath expansion
3. **HKD**: Heterogeneity-aware Knowledge Distillation
   - Logit-level distillation
   - **Semantic-level**: Metapath attention distillation

### Semantic-Level Distillation (HKD)
```
L_semantic = Σ_m MSE(α^(T)_{P_m}(v_i), α^(S)_{P_m}(v_i))
```
where `α_{P_m}(v_i)` = attention coefficient for metapath `P_m` at node `v_i`

### Key Ablation
| Component Removed | AP Drop | FR Increase |
|-------------------|---------|-------------|
| Experience Replay | 1.5-11% | - |
| HKD (semantic KD) | 3.6-12% | 3.6-12% |
| Meta-learning (G-MM) | 0.3-3.4% | - |

**Key**: Semantic-level distillation (metapath attention) is the core forgetting mitigation mechanism.

---

## Summary: Mathematical Objectives for TRENCH-IDS

### Option A: RFR-Inspired (Relation-Aware Effective Rank)
For each relation `r ∈ {originates, terminated_by, targeted_by, protocol_of, service_of}`:
```
L_rank(r) = - log(erank(Z_r))  # Z_r = normalized relation embeddings
L_RFR = Σ_r λ_r * L_rank(r)    # λ_r could be uniform or weighted by Stage A transfer potential
```

### Option B: FACT-Inspired (Virtual Prototypes per Relation)
For each relation `r`:
- Add `V_r` virtual prototypes `P_v^r ∈ ℝ^(d×V)`
- Virtual loss per relation: `L_v^r = CE + CE_virtual`
- Reserve embedding space per relation

### Option C: RFR + FACT Hybrid (Recommended)
```
L = L_CE 
  + λ_rank * Σ_r (-log(erank(Z_r)))           # RFR: feature richness
  + λ_virtual * Σ_r L_virtual^r               # FACT: reserve space per relation
  + λ_attention * Σ_r MSE(β_r^curr, β_r^old)  # HKD: relation attention distillation
```

### Option D: TagFex-Inspired (Task-Agnostic Feature Learner)
- Add separate **task-agnostic encoder** (shared across all tasks)
- Task-specific encoders guided by task-agnostic features
- `L_expand = MSE(task_specific_feat, task_agnostic_feat_projected)`

---

## Key Design Decisions for TRENCH-IDS

| Decision | Recommendation | Rationale |
|----------|----------------|-----------|
| **Base objective** | RFR (erank maximization) | Mathematically grounded, differentiable, proven |
| **Relation weighting** | **Uniform** initially (not Stage A weighted) | Avoids future-information leakage |
| **Virtual prototypes** | Per-relation (optional) | Matches heterogeneous schema |
| **Attention distillation** | Add MSE on `SemanticAttention.beta` | Your existing attention, HKD shows it works |
| **Where applied** | Base task (T1) + all subsequent | RFR: base only; FACT: all; Hybrid: all |
| **Global vs per-relation** | **Per-relation** (Option C) | Your Stage A shows transfer is relation-specific |

---

## Minimal Forward-Compatible Loss for TRENCH-IDS

```python
def forward_compatible_loss(model, batch, task_id, prev_model=None, stage_a_scores=None):
    """
    Forward-compatible loss for TRENCH-IDS heterogeneous GNN.
    
    Args:
        model: Current 3-layer residual RGNN
        batch: Mini-batch of heterogeneous graphs
        task_id: Current task (0 = T1 base session)
        prev_model: Previous task's model (for KD)
        stage_a_scores: Stage A transfer scores (for weighting, OPTIONAL)
    """
    outputs = model(batch)
    flow_fused = outputs.fused['flow']  # [N, 64]
    relation_embeds = outputs.relations['flow']  # dict of 5 relations
    attention_weights = outputs.attention['flow']  # dict of 5 relation betas
    
    # 1. Standard classification loss
    logits = classifier_head(flow_fused)
    L_CE = F.cross_entropy(logits, batch['flow'].y)
    
    # 2. RFR: Feature richness per relation (effective rank)
    L_RFR = 0
    for r, embeds in relation_embeds.items():
        if r in FLOW_RELATIONS:
            Z = F.normalize(embeds, p=2, dim=1)  # [N, 64]
            # Compute logged effective rank (log(erank))
            _, s, _ = torch.svd(Z)
            eig_val = s * s / Z.shape[0]
            logged_erank = - (eig_val * torch.log(eig_val + 1e-8)).sum()
            L_RFR += -logged_erank  # maximize erank = minimize -log(erank)
    
    # 3. Optional: FACT virtual prototypes per relation
    L_VIRTUAL = 0
    if use_virtual_prototypes:
        for r in FLOW_RELATIONS:
            # Add virtual prototypes P_v^r, compute virtual loss
            pass
    
    # 4. Knowledge distillation from previous task (if available)
    L_KD = 0
    if prev_model is not None:
        with torch.no_grad():
            prev_outputs = prev_model(batch)
            prev_fused = prev_outputs.fused['flow']
            prev_attention = prev_outputs.attention['flow']
        
        # Logit distillation (LwF)
        L_KD += F.kl_div(
            F.log_softmax(logits / T, dim=1),
            F.softmax(prev_classifier(prev_fused) / T, dim=1),
            reduction='batchmean'
        ) * T * T
        
        # Relation attention distillation (HKD)
        for r in FLOW_RELATIONS:
            if r in attention_weights and r in prev_attention:
                L_KD += F.mse_loss(attention_weights[r], prev_attention[r])
    
    # Total loss
    if task_id == 0:  # Base session (T1)
        total = L_CE + λ_RFR * L_RFR
    else:
        total = L_CE + λ_KD * L_KD
        # Optionally add RFR in all tasks: + λ_RFR * L_RFR
    
    return total, {
        'L_CE': L_CE.item(),
        'L_RFR': L_RFR.item() if task_id == 0 else 0,
        'L_KD': L_KD.item() if task_id > 0 else 0,
        'erank_per_rel': {r: compute_erank(embeds) for r, embeds in relation_embeds.items()}
    }
```

---

## Evaluation Protocol (Locked)

| Metric | Protocol |
|--------|----------|
| **Primary** | AULC(1-3) on B1 LOTO (5 transitions × seeds {42, 1}) |
| **Threshold** | ΔAULC > 0.0037, 95% CI excludes zero |
| **Sanity** | Oracle sanity check: frozen encoder + linear probe on unseen attacks > 70% |
| **Forgetting** | ≤ Replay baseline (0.081) |
| **Ablation** | RFR only / RFR+KD / Full / No forward-compat |

---

## Decision Points for You

1. **Which base objective?** RFR (erank) vs FACT (virtual prototypes) vs Hybrid?
2. **Per-relation or global?** Per-relation (leverages Stage A) vs Global?
3. **Virtual prototypes?** Add FACT virtual loss per relation?
3. **Where to apply?** Base only (RFR) or all tasks (FACT)?
4. **Weighting scheme?** Uniform vs Stage A transfer potential (but careful about leakage)?
4. **Dense embeddings?** Add for Protocol/Service/Port (Gu et al. 2024)?
5. **SSL pretraining first?** GraphIDS-style masked AE before continual?

---

**Next Step**: You confirm the design choices above, then I'll implement the minimal forward-compatible loss module.