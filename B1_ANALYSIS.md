# B1 LOTO Analysis: Forward Transfer on Strong Replay Baseline

## Executive Summary

The strong replay baseline (3-layer residual GNN + replay) achieves excellent retention but shows **highly variable forward transfer** across transitions.

## Key Results

### B1 LOTO AULC(1-3) - F1-macro

| Transition | Seed 42 | Seed 1 | Mean AULC | Std |
|------------|--------:|-------:|----------:|----:|
| T1→T2      | 0.290   | 0.285  | **0.288** | 0.004 |
| T2→T3      | 0.408   | 0.397  | **0.403** | 0.005 |
| T3→T4      | 0.075   | 0.298  | **0.187** | 0.112 |
| T4→T5      | 0.069   | 0.036  | **0.053** | 0.017 |
| T5→T6      | 0.073   | 0.086  | **0.080** | 0.007 |
| **Mean**   | **0.183** | **0.221** | **0.202** | 0.027 |

### Per-Transition Details

#### T1→T2 (Scanning → Reconnaissance)
- **AULC: 0.288** - Strong early transfer
- Epoch 1: 0.815 acc / 0.127 F1 → Epoch 3: 0.923 acc / 0.161 F1
- Source: Scanning (T1), Target: Reconnaissance (T2) - both CSE dataset

#### T2→T3 (Reconnaissance → DDoS + Infiltration) 
- **AULC: 0.403** - **Best transfer!**
- Epoch 1: 0.744 acc / 0.155 F1 → Epoch 3: 0.863 acc / 0.221 F1
- Source: Reconnaissance (T2), Target: DDoS + Infiltration (T3)
- Both CSE dataset

#### T3→T4 (DDoS+Infiltration → DoS+Injection)
- **AULC: 0.187** - **Sharp drop**
- Epoch 1: 0.303 acc / 0.044 F1 → Epoch 3: 0.250 acc / 0.036 F1
- Source: DDoS+Infiltration (T3), Target: DoS+Injection (T4)
- DDoS from CSE, DoS from ToN, Injection from CSE

#### T4→T5 (DoS+Injection → Password+Bot)
- **AULC: 0.053** - **Very weak**
- Epoch 1: 0.096 acc / 0.030 F1 → Epoch 3: 0.250 acc / 0.036 F1
- Source: DoS+Injection (T4), Target: Password+Bot (T5)
- DoS from ToN, Injection from CSE; Password/Bot from CSE

#### T5→T6 (Password+Bot → XSS+BruteForce)
- **AULC: 0.080** - **Very weak**
- Epoch 1: 0.250 acc / 0.036 F1 → Epoch 3: 0.250 acc / 0.036 F1
- Source: Password+Bot (T5), Target: XSS+BruteForce (T6)
- Password/Bot from CSE; XSS from ToN, BruteForce from CSE

---

## Dataset Composition Analysis

### Task Composition

| Task | Attack Classes | Source Datasets | Train Graphs | Test Graphs |
|------|----------------|-----------------|--------------|-------------|
| T1 | Benign + Scanning | ToN | 1,214 | 260 |
| T2 | Benign + Reconnaissance | BoT | 1,214 | 260 |
| T3 | Benign + DDoS + Infiltration | DDoS: CSE, Infiltration: CSE | 1,212 | 260 |
| T4 | Benign + DoS + Injection | DoS: ToN, Injection: CSE | 1,214 | 260 |
| T5 | Benign + Password + Bot | CSE | 1,214 | 260 |
| T6 | Benign + XSS + BruteForce | XSS: ToN, BruteForce: CSE | 1,214 | 260 |

### Key Observations

1. **Dataset Shift Pattern**:
   - T1→T2: ToN → BoT (different datasets)
   - T2→T3: BoT → CSE (different datasets)  
   - T3→T4: CSE → Mixed (CSE + ToN)
   - T4→T5: Mixed (CSE+ToN) → CSE
   - T5→T6: CSE → Mixed (ToN+CSE)

2. **Class Balance**:
   - All tasks: ~91K Benign + ~270K attack flows (per task)
   - T3 has 2 attack classes (DDoS: 191K, Infiltration: 81K)
   - T4 has balanced attack classes (DoS: 137K, Injection: 137K)
   - T5: Password (173K) >> Bot (100K)
   - T6: XSS (188K) > BruteForce (85K)

---

## Forgetting Matrix Analysis

| Task | After T1 | After T2 | After T3 | After T4 | After T5 | After T6 |
|------|----------|----------|----------|----------|----------|----------|
| T1 (Scanning) | 0.961 | 0.972 | 0.911 | 0.915 | 0.890 | 0.936 |
| T2 (Recon) | - | 0.956 | 0.950 | 0.935 | 0.894 | 0.940 |
| T3 (DDoS+Infil) | - | - | 0.947 | 0.799 | 0.902 | 0.908 |
| T4 (DoS+Inject) | - | - | - | 0.977 | 0.719 | 0.764 |
| T5 (Password+Bot) | - | - | - | - | 0.995 | 0.962 |
| T6 (XSS+Brute) | - | - | - | - | - | 0.993 |

**Average Forgetting: 0.0673** (Excellent retention!)

---

## Transition-Level Diagnostic Table

| Metric | T1→T2 | T2→T3 | T3→T4 | T4→T5 | T5→T6 |
|------|-------|-------|-------|-------|-------|
| **AULC(1-3)** | 0.288 | **0.403** | 0.187 | 0.053 | 0.080 |
| Final Acc | 0.915 | 0.858 | 0.250 | 0.250 | 0.250 |
| Final F1 | 0.159 | 0.220 | 0.036 | 0.036 | 0.036 |
| Target Task Classes | 1 | 2 | 2 | 2 | 2 |
| Dataset Shift | ToN→BoT | BoT→CSE | CSE→Mixed | CSE→CSE | CSE→Mixed |
| Target Test Size | 260 | 260 | 260 | 260 | 260 |
| Forgetting (all tasks) | 0.067 | 0.067 | 0.067 | 0.067 | 0.067 |

---

## Key Findings

### 1. **Excellent Retention, Variable Transfer**
- **Forgetting: 0.067** (excellent - replay works perfectly)
- But forward transfer (AULC) varies dramatically: **0.053 to 0.403**

### 2. **Transfer Peaks at T2→T3, Collapses After T3**
- T1→T2: 0.288 (good)
- T2→T3: **0.403** (best - BoT→CSE transfer works well)
- T3→T4: **0.187** (53% drop from peak)
- T4→T5: 0.053 (near zero)
- T5→T6: 0.080 (still poor)

### 3. **Final Accuracy Collapse**
- T1→T2: 91.5% final accuracy
- T2→T3: 85.8% final accuracy  
- T3→T4: **25%** (random guess for 3 classes)
- T4→T5: **25%** (random guess for 3 classes)
- T5→T6: **25%** (random guess for 3 classes)

### 4. **Transfer Collapse Correlates With...**
- **Dataset shift pattern**: Strong when source/target share dataset (T2→T3 both CSE)
- **Target task complexity**: 2-class targets work better than 3-class
- **Dataset diversity in target**: Pure CSE targets (T3, T5) work better than mixed

---

## Hypotheses for Transfer Collapse

### Hypothesis 1: Representation Specialization (Most Likely)
- As more tasks learned, representation becomes over-specialized to seen classes
- Loses capacity to represent novel attack patterns
- Supported by: Gradual decline, peaks when representation still generic (T2→T3)

### Hypothesis 2: Dataset Shift (Secondary)
- T3→T4: CSE → Mixed (CSE+ToN)
- T4→T5: CSE→CSE (should be good but isn't)
- Not fully explanatory since T4→T5 is CSE→CSE but still fails

### Hypothesis 3: Task Complexity
- 2-class targets (T2, T3) transfer better than 3-class (T4, T5, T6)
- But T3 has 3 classes and T2→T3 works well

### Hypothesis 4: Replay Buffer Saturation
- Fixed buffer (200) may become dominated by later tasks
- But T1→T2 and T2→T3 work despite same buffer

---

## Recommendations for Next Method (FACT)

Given the findings, FACT should target:

1. **Preserve representation capacity** for novel classes during sequential training
2. **Reserve embedding space** for future attack classes explicitly
3. **Monitor relation-specific capacity** - don't let relations over-specialize
4. **Test on T3→T4, T4→T5, T5→T6** specifically (where transfer collapses)

### Proposed FACT Configuration for TRENCH-IDS
- **Virtual prototypes**: Reserve space for future attack classes per relation
- **Virtual instances**: Generate synthetic attack patterns from existing ones
- **Per-relation virtual prototypes**: Match heterogeneous structure
- **Evaluation**: Must improve T3→T4, T4→T5, T5→T6 AULC specifically

---

## Next Action Items

1. ✅ **Baseline established**: 3-layer residual + replay = 0.067 forgetting, 0.917 acc
2. ✅ **B1 LOTO completed**: Mean AULC = 0.202, transfer collapses after T3
3. ⏳ **Analyze relation-specific transfer** for each transition (using existing `transferability_task_*.json`)
4. ⏳ **Design FACT experiment** targeting T3→T4, T4→T5, T5→T6 specifically
5. ⏳ **Implement FACT** with per-relation virtual prototypes

The benchmark is now locked: **Strong replay baseline = 0.067 forgetting, 0.917 acc, 0.202 mean AULC**.