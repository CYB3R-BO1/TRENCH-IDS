# TRENCH-IDS: Transferable Representation Learning for Continual Heterogeneous Graph-based IDS

## Complete Project Documentation

**Project Status**: Benchmark frozen (2026-09-07)  
**Type**: Research project documentation / Single source of truth  
**Target Audience**: Anyone new to the project, including future researchers continuing this work

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Problem](#2-research-problem)
3. [Motivation and Background](#3-motivation-and-background)
4. [Research Objectives](#4-research-objectives)
5. [Research Questions and Hypotheses](#5-research-questions-and-hypotheses)
6. [Proposed Approach](#6-proposed-approach)
7. [Literature and Existing Methods](#7-literature-and-existing-methods)
8. [Theoretical Concepts](#8-theoretical-concepts)
9. [Methodology](#9-methodology)
10. [System and Project Architecture](#10-system-and-project-architecture)
11. [Data and Dataset](#11-data-and-dataset)
12. [Data Collection and Preprocessing](#12-data-collection-and-preprocessing)
13. [Features and Variables](#13-features-and-variables)
14. [Algorithms](#14-algorithms)
15. [Models](#15-models)
16. [Model Architecture](#16-model-architecture)
17. [Parameters and Hyperparameters](#17-parameters-and-hyperparameters)
18. [Mathematical Formulas and Equations](#18-mathematical-formulas-and-equations)
19. [Implementation Details](#19-implementation-details)
20. [Tools, Libraries, Frameworks, and Technologies](#20-tools-libraries-frameworks-and-technologies)
21. [Experimental Setup](#21-experimental-setup)
22. [Experimental Procedure](#22-experimental-procedure)
23. [Evaluation Metrics](#23-evaluation-metrics)
24. [Results](#24-results)
25. [Tables and Numerical Results](#25-tables-and-numerical-results)
26. [Graphs, Figures, and What They Show](#26-graphs-figures-and-what-they-show)
27. [Analysis and Interpretation](#27-analysis-and-interpretation)
28. [Comparisons and Baselines](#28-comparisons-and-baselines)
29. [Limitations](#29-limitations)
30. [Assumptions](#30-assumptions)
31. [Challenges and Issues Encountered](#31-challenges-and-issues-encountered)
32. [Solutions and Modifications Made](#32-solutions-and-modifications-made)
33. [Final Conclusions](#33-final-conclusions)
34. [Future Work](#34-future-work)
35. [References and Sources](#35-references-and-sources)
36. [Complete Technical Terminology Glossary](#36-complete-technical-terminology-glossary)
37. [Complete List of Important Numbers, Parameters, and Values](#37-complete-list-of-important-numbers-parameters-and-values)

---

## 1. Project Overview

**TRENCH-IDS** stands for **Transferable REpresentatioN learning for Continual Heterogeneous graph-based IDS** (Intrusion Detection System).

The project investigates whether **forward-compatible representation learning** can improve the ability of a continual learning system to efficiently learn new attack classes without retraining from scratch. The core scientific question is:

> **Can we train a representation that remains useful for future, as-yet-unseen attack classes?**

The project establishes:
- A **strong replay baseline** (3-layer residual Hetero GNN + experience replay) that achieves 0.081 ± 0.008 forgetting and 0.902 ± 0.005 final accuracy
- A **B1 LOTO (Leave-One-Task-Out) evaluation framework** that measures forward transfer as Area Under the Learning Curve (AULC)
- **Systematic testing of three qualitatively different forward-compatibility mechanisms**: TCTRL (task-conditioned adaptation), RFR (feature richness), and FACT (virtual prototypes)

**Key finding**: The strong replay baseline exhibits excellent backward retention (low forgetting) but **highly transition-dependent forward transfer** that deteriorates after task T3. **None of the three tested forward-compatibility mechanisms improved upon the replay baseline**, suggesting the limitation may stem from intrinsic task/representation distribution properties rather than from simple representation deficiencies.

---

## 2. Research Problem

Modern Network Intrusion Detection Systems (NIDS) face a critical challenge: **network attacks continuously evolve**, with new attack types emerging that the model has never seen during training. Traditional supervised learning approaches require complete retraining whenever a new attack class appears, which is computationally expensive and causes the model to **forget previously learned attacks** (catastrophic forgetting).

The field of **Continual Learning (CL)** addresses this by enabling models to learn from a stream of tasks without forgetting prior knowledge. However, most CL research focuses on **backward compatibility** (retaining old tasks) while ignoring **forward compatibility** (efficiently learning future tasks).

**TRENCH-IDS** specifically addresses the **forward-transfer problem** in continual NIDS: How can we train a representation that prepares the system to learn new attack classes efficiently when they appear?

---

## 3. Motivation and Background

### 3.1 Network Intrusion Detection Systems (NIDS)

NIDS analyze network traffic to identify malicious activities. Modern NIDS use machine learning models trained on labeled network flow data. The key challenges are:

1. **Evolving threats**: New attack types emerge constantly
2. **Label scarcity**: Labeling network traffic requires expensive expert analysis
3. **Class imbalance**: Benign traffic vastly outnumbers attack traffic
4. **Concept drift**: Attack patterns evolve over time

### 3.2 Continual Learning Challenges

In continual learning, a model must learn from a sequence of tasks T₁, T₂, ..., Tₙ without forgetting previous tasks. The key challenges are:

- **Catastrophic forgetting**: Performance on previous tasks degrades as new tasks are learned
- **Forward transfer**: Ability to learn new tasks efficiently using knowledge from previous tasks
- **Backward transfer**: Ability of new task learning to improve performance on previous tasks (usually negative)

### 3.3 The Forward-Transfer Gap

Most continual learning research focuses on **backward compatibility** (preventing forgetting). The **forward transfer** problem (how to prepare representations for future tasks) is less studied. This is a significant gap because:

- In real-world NIDS deployments, new attacks appear regularly
- Efficiently learning new attacks is operationally critical
- Current methods (replay, regularization) are reactive, not proactive

---

## 4. Research Objectives

The primary objectives of TRENCH-IDS are:

1. **Establish a strong continual learning baseline** for heterogeneous graph-based NIDS using replay
2. **Characterize forward transfer** behavior across the continual learning sequence
3. **Systematically test** qualitatively different forward-compatibility mechanisms
4. **Provide reproducible benchmarks** for the research community

The project explicitly does **not** claim to have "solved" forward-compatible representation learning, but rather to **characterize the phenomenon** and **document which mechanisms fail**.

---

## 5. Research Questions and Hypotheses

### 5.1 Primary Research Question

> **Can forward-compatible, relation-aware representation learning improve forward transfer to unseen attack classes in continual heterogeneous graph-based intrusion detection?**

### 5.2 Secondary Research Questions

1. Does **replay** alone provide sufficient forward transfer for NIDS?
2. How does forward transfer **vary across transitions** in the continual sequence?
3. Can **task-conditioned adaptation** (modifying the representation at target time) improve transfer?
4. Can **feature richness regularization** (RFR) improve the base representation?
5. Can **virtual prototype reservation** (FACT) preserve embedding space for future classes?

### 5.3 Hypotheses Tested

The project tested **three qualitatively different hypotheses** about why forward transfer might be limited:

**Hypothesis 1: Task-conditioned adaptation**
- Claim: "When a new task arrives, modify the old representation to fit the new task better"
- Method: TCTRL (Task-Conditioned Transfer Representation Learning) with FOMAML
- Result: **Failed** — Oracle accuracy only 18% vs baseline 95%

**Hypothesis 2: Feature richness**
- Claim: "Make the representation richer during initial training so it has capacity for future classes"
- Method: RFR (Effective-Rank based Feature Richness enhancement)
- Result: **No meaningful improvement** — 0.064 vs 0.067 forgetting, same AULC

**Hypothesis 3: Reserved embedding space**
- Claim: "Explicitly reserve space for future classes via virtual prototypes"
- Method: FACT (Forward Compatible Training)
- Result: **Harmful** — AULC dropped from 0.202 to 0.140

---

## 6. Proposed Approach

The project follows a **benchmark-first, ablation-second** approach:

### 6.1 Strong Replay Baseline

The core system is a **3-layer residual heterogeneous Graph Neural Network (HeteroGNN)** with **experience replay**:

- **Heterogeneous graph**: Nodes represent network entities (Flow, Host, Protocol, Service, Port); edges represent their relationships
- **Residual connections**: Enable deeper message passing without gradient degradation
- **Experience replay**: Store representative samples from previous tasks and replay them during new task training

### 6.2 B1 LOTO Evaluation Protocol

The B1 (Stage B1) evaluation uses **Leave-One-Task-Out** cross-validation across the 6-task sequence:

- For each transition T_n → T_{n+1}:
  1. Train on tasks T₁...T_n with replay
  2. Load checkpoint after task T_n
  3. Fine-tune on task T_{n+1} training data for 3 epochs
  4. Measure accuracy/F1 at epochs 1, 2, 3
  5. Compute AULC(1-3) = area under the learning curve

### 6.3 Forward-Compatibility Mechanisms Tested

The project tested three forward-compatibility mechanisms (detailed in Section 15):

1. **TCTRL**: Task-conditioned adapter with FOMAML meta-learning
2. **RFR**: Effective-rank regularization to increase representation richness
3. **FACT**: Virtual prototype reservation to preserve embedding space

---

## 7. Literature and Existing Methods

### 7.1 Continual Learning Methods

The project builds on the continual learning literature:

- **Experience Replay**: Store and replay representative samples from previous tasks (the strongest baseline)
- **Elastic Weight Consolidation (EWC)**: Regularize important parameters to prevent forgetting
- **Learning without Forgetting (LwF)**: Knowledge distillation from previous model
- **Progressive Neural Networks**: Add new columns for new tasks
- **Gradient Episodic Memory (GEM)**: Project gradients to not increase loss on previous tasks

### 7.2 Graph Neural Networks for NIDS

- **E-GraphSAGE**: Edge-featured GraphSAGE for network flow classification
- **Anomal-E**: Self-supervised GNN for anomaly detection
- **E-ResGAT**: Residual edge attention for NIDS
- **Heterogeneous GNNs**: Model different node/edge types (used in TRENCH-IDS)

### 7.3 Forward-Compatibility Literature

- **FACT** (CVPR 2022): Forward Compatible Training via virtual prototypes
- **RFR** (Neural Networks 2024): Effective-Rank based Feature Richness
- **TagFex** (CVPR 2025): Task-Agnostic Guided Feature Expansion
- **IL2A**: Incremental Learning with Dual Augmentation

---

## 8. Theoretical Concepts

### 8.1 Catastrophic Forgetting

In neural network training, learning new tasks tends to overwrite weights important for previous tasks, causing performance degradation on previous tasks. This is called **catastrophic forgetting**.

**Forgetting metric**:
$$\text{Forgetting} = \frac{1}{T-1} \sum_{i=1}^{T-1} \max_{t < T} R_{t,i} - R_{T,i}$$

where $R_{t,i}$ is the accuracy on task $i$ after training on task $t$.

### 8.2 Effective Rank

The **effective rank** (Roy & Vetterli, 2007) is a continuous generalization of matrix rank:

$$\text{erank}(R) = \exp\left(-\sum_i p_i \log p_i\right)$$

where $p_i = \sigma_i^2 / N$ are normalized singular values squared. Effective rank is differentiable and captures representation richness.

### 8.3 Semantic Attention Fusion

The project uses **HAN-style attention fusion** (Wang et al., WWW 2019) to combine relation-specific embeddings:

$$\text{score}_r = \text{MLP}(\tanh(W \cdot h_r))$$

$$\beta_r = \frac{\exp(\text{score}_r)}{\sum_{r'} \exp(\text{score}_{r'})}$$

$$h_{\text{fused}} = \sum_r \beta_r \cdot h_r$$

### 8.4 AULC (Area Under Learning Curve)

The **AULC(1-3)** measures how quickly a model learns a new task:

$$\text{AULC}(1-3) = \frac{f_1 + 2f_2 + f_3}{4}$$

where $f_1, f_2, f_3$ are F1-macro scores after fine-tuning epochs 1, 2, 3 on the target task.

### 8.5 Forward vs Backward Transfer

- **Backward transfer**: Impact of learning new task on performance of old tasks
- **Forward transfer**: Impact of learning old tasks on ability to learn new task efficiently

---

## 9. Methodology

### 9.1 Experimental Phases

The project progressed through distinct phases:

**Phase 0: Architecture Selection**
- Compared 1-layer vs 3-layer GNN
- Selected 3-layer residual as the strong baseline architecture

**Phase 1: Strong Replay Baseline**
- Established the locked baseline: 3-layer residual + replay
- Validated with 3 seeds (42, 1, 2)

**Phase 2: B1 LOTO Evaluation**
- Evaluated forward transfer across 5 transitions
- Documented the AULC decay pattern

**Phase 3: Forward-Compatibility Testing**
- Tested TCTRL, RFR, FACT
- All failed to improve upon the replay baseline

### 9.2 Validation Strategy

The project used:
- **Multiple seeds** (42, 1, 2) for variance estimation on the canonical baseline
- **2 seeds** (42, 1) for the B1 LOTO evaluation
- **Independent audit** (2026-09-07) reproducing key results within ~0.004 variance
- **373/373 tests passing** after fixing a one-line import issue

---

## 10. System and Project Architecture

### 10.1 Repository Structure

```
TRENCH-IDS/
├── src/
│   └── trench_ids/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cl/                    # Continual learning modules
│       │   ├── __init__.py
│       │   ├── train.py           # Main training loop
│       │   ├── train_meta.py      # Meta-training entry point
│       │   ├── rfr.py             # RFR (implemented in this project)
│       │   ├── fact.py            # FACT (implemented in this project)
│       │   ├── transferability.py  # Transferability estimation
│       │   ├── ewc.py             # Elastic Weight Consolidation
│       │   ├── distillation.py    # Knowledge distillation
│       │   ├── episode_generator.py  # Meta-learning episodes
│       │   ├── meta_transfer.py   # TCTRL implementation
│       │   ├── inference.py       # Model loading and prediction
│       │   ├── evaluate.py        # Evaluation metrics
│       │   ├── memory_bank.py     # Relation memory bank
│       │   └── ...
│       ├── model/                 # Model architectures
│       │   ├── rhgnn.py           # Relation-specific HeteroGNN
│       │   ├── relation_conv.py   # Relation-specific convolution
│       │   ├── attention_fusion.py # Semantic attention
│       │   ├── transfer_adapter.py # TCTRL adapter
│       │   └── flat.py            # Flat baseline
│       ├── labels.py              # Label definitions
│       ├── graphs.py              # Graph construction
│       ├── preprocess.py          # Data preprocessing
│       ├── similarity.py          # Class similarity
│       ├── task_design.py         # Task partitioning
│       ├── vocab.py               # Vocabulary management
│       └── constants.py           # Flow relations constants
├── configs/                       # Hydra configuration
│   ├── train.yaml                 # Main training config
│   ├── model.yaml                 # Model architecture config
│   ├── preprocess.yaml           # Preprocessing config
│   ├── graph.yaml                 # Graph construction config
│   └── ...
├── data/                          # Data directory
│   ├── graphs/                    # Processed graph files
│   │   ├── task_1_train.pt
│   │   ├── task_1_test.pt
│   │   ├── task_1_val.pt
│   │   ├── ...
│   │   └── vocab.json
│   └── processed/                 # Parquet files
├── runs/                          # Training run outputs
│   ├── gnn_3layer_residual_s42/   # Baseline seed 42
│   ├── gnn_3layer_residual_s1/    # Baseline seed 1
│   ├── gnn_3layer_residual_s2/    # Baseline seed 2
│   ├── strong_replay_baseline/    # Sensitivity run
│   ├── fact_strong/               # FACT run
│   ├── rfr_strong/                # RFR run
│   └── b1_loto/                    # B1 evaluation results
├── tests/                         # Test suite
│   ├── test_train_flat.py         # 373/373 passing
│   └── ...
├── FINAL_BENCHMARK_RESULTS.md    # Benchmark summary
├── B1_ANALYSIS.md                  # B1 analysis
├── VERIFICATION_REPORT.md          # Audit report
├── .opencode/plans/                # Planning documents
│   └── transferable_representation_plan.md
├── pyproject.toml                  # Package configuration
└── README.md
```

### 10.2 Entry Points

The project provides several CLI entry points (defined in `pyproject.toml`):

- `trench-preprocess` → `trench_ids.preprocess:main`
- `trench-similarity` → `trench_ids.similarity:main`
- `trench-graphs` → `trench_ids.graphs:main`
- `trench-train` → `trench_ids.cl.train:main`
- `trench-train-flat` → `trench_ids.cl.train_flat:main`
- `trench-evaluate` → `trench_ids.cl.evaluate:main`
- `trench-infer` → `trench_ids.cl.inference:main`
- `trench-run-matrix` → `trench_ids.cl.run_matrix:main`
- `trench-train-meta` → `trench_ids.cl.train_meta:main`

---

## 11. Data and Dataset

### 11.1 Data Source

The project uses **three NetFlow v2 datasets** from the University of Queensland (UQ):

1. **NF-ToN-IoT-v2** (ToN): Network flows from ToN-IoT dataset
2. **NF-BoT-IoT-v2** (BoT): Network flows from BoT-IoT dataset
3. **NF-CSE-CIC-IDS2018-v2** (CSE): Network flows from CSE-CIC-IDS2018 dataset

Each dataset contains labeled network flows with 43 standardized NetFlow features plus class labels.

### 11.2 Task Composition

| Task | Attack Classes | Source Datasets | Train Graphs | Test Graphs |
|------|----------------|-----------------|--------------|-------------|
| T1 | Benign + Scanning | ToN | 1,214 | 260 |
| T2 | Benign + Reconnaissance | BoT | 1,214 | 260 |
| T3 | Benign + DDoS + Infiltration | ToN, CSE | 1,212 | 260 |
| T4 | Benign + DoS + Injection | ToN, CSE | 1,214 | 260 |
| T5 | Benign + Password + Bot | ToN, CSE | 1,214 | 260 |
| T6 | Benign + XSS + BruteForce | ToN, CSE | 1,214 | 260 |

**Total**: 6 tasks, 11 attack classes (plus Benign), ~1,214 train graphs per task

**Note**: T1 and T2 are single-dataset (Scanning from ToN only; Reconnaissance from BoT only). T3-T6 each span both ToN and CSE because the attack classes in those tasks draw from both datasets (e.g., DDoS from {ToN, CSE}, Password from {ToN} paired with Bot from {CSE}).

### 11.3 Class Distribution

Per-task flow counts (approximate):
- T1: ~91K Benign + ~273K Scanning
- T2: ~91K Benign + ~273K Reconnaissance
- T3: ~91K Benign + ~191K DDoS + ~81K Infiltration
- T4: ~91K Benign + ~137K DoS + ~137K Injection
- T5: ~91K Benign + ~173K Password + ~100K Bot
- T6: ~91K Benign + ~188K XSS + ~85K BruteForce

### 11.4 Excluded Classes

Five classes are excluded from experiments:
- Backdoor
- MITM
- Ransomware
- Web Attacks
- Theft

These are excluded because they have insufficient data (<100K flows) and would cause class imbalance issues.

### 11.5 Dataset Filtering (`CLASS_DATASETS`)

To control dataset composition, the project restricts which datasets can contribute to which classes:

- **Scanning**: ToN only
- **Reconnaissance**: BoT only
- **DDoS**: CSE only
- **Infiltration**: CSE only
- **DoS**: ToN only
- **Injection**: CSE only
- **Password**: CSE only
- **Bot**: CSE only
- **XSS**: ToN only
- **BruteForce**: CSE only

---

## 12. Data Collection and Preprocessing

### 12.1 Raw Data Collection

Raw data comes from UQ NetFlow v2 datasets in CSV format. Each CSV contains:
- 43 standardized NetFlow features
- Attack labels
- Dataset source identifier

### 12.2 Preprocessing Pipeline

The preprocessing pipeline (`src/trench_ids/preprocess.py`) includes:

1. **Column standardization**: Map raw column names to standardized schema
2. **Label harmonization**: Map raw attack labels to canonical class names
3. **Dataset filtering**: Apply `CLASS_DATASETS` restrictions
4. **Quality filtering**: Drop rows with NaN/inf in critical columns
5. **Type conversion**: Convert to efficient Parquet format
6. **Train/val/test split**: Stratified split per task

### 12.3 Preprocessing Configuration

Key parameters in `configs/preprocess.yaml`:
- `benign_per_task`: Benign sample budget per task
- `attack_per_class_cap`: Maximum samples per attack class
- Train/val/test split ratios
- Random seed for reproducibility

### 12.4 Output Format

Preprocessed data is stored in Parquet format at `data/processed/`:
- `task_1.parquet`, `task_2.parquet`, ..., `task_6.parquet`
- Columns: 45 features (43 raw + flow_id + source_dataset + canonical_label + task + split)

---

## 13. Features and Variables

### 13.1 Flow Features (37-dim)

Each network flow is represented by 37 numerical features:
- **Duration features**: Flow duration, inter-arrival times
- **Byte/packet counts**: Total bytes, forward/backward bytes
- **Rate features**: Bytes/sec, packets/sec
- **Flag features**: TCP flags (SYN, ACK, FIN, RST, PSH, URG)
- **Header features**: IP, TCP, UDP header lengths
- **Window features**: TCP window sizes
- **Subflow features**: Subflow statistics
- **Other features**: Various protocol-specific statistics

### 13.2 Host Features (4-dim)

Each host node has 4 features:
- Protocol indicator
- Number of unique source IPs
- Number of unique destination IPs
- Number of flows

### 13.3 Port, Protocol, Service Features

These nodes have:
- `port_number` (Port): TCP/UDP port number
- `value` (Protocol, Service): Protocol/service identifier
- `vocab_id` (Protocol, Service): Mapped to vocabulary index

### 13.4 Graph Structure

The heterogeneous graph has the following node and edge types:

**Node types**: `flow`, `host`, `protocol`, `service`, `port`

**Edge types** (11 relations):
- `host--originates-->flow`
- `flow--terminates_at-->host`
- `flow--targets_port-->port`
- `flow--uses_protocol-->protocol`
- `flow--uses_service-->service`
- `host--communicates_with-->host`
- `flow--originated_by-->host`
- `host--terminated_by-->flow`
- `port--targeted_by-->flow`
- `protocol--protocol_of-->flow`
- `service--service_of-->flow`

---

## 14. Algorithms

### 14.1 Effective Rank Computation (RFR)

```python
def effective_rank(reps):
    Z = F.normalize(reps, p=2, dim=1)  # [N, D]
    _, s, _ = torch.linalg.svd(Z, full_matrices=False)
    eig_val = (s * s) / reps.shape[0]
    entropy = - (eig_val * torch.log(eig_val + eps)).sum()
    erank = torch.exp(entropy)
    return erank
```

### 14.2 RFR Loss

```python
def rfr_loss(reps):
    logged_erank = - (eig_val * torch.log(eig_val + eps)).sum()
    return -logged_erank  # Maximize erank by minimizing -log(erank)
```

### 14.3 Semantic Attention Fusion

```python
class SemanticAttention(nn.Module):
    def forward(self, embeddings):
        stacked = torch.stack(embeddings, dim=0)  # [R, N, hidden_dim]
        scores = self.query(self.project(stacked)).squeeze(-1)  # [R, N]
        relation_scores = scores.mean(dim=1)  # [R]
        beta = torch.softmax(relation_scores, dim=0)  # [R]
        fused = (beta.view(-1, 1, 1) * stacked).sum(dim=0)
        return fused, beta
```

### 14.4 FACT Virtual Prototype Loss

The FACT loss consists of:
1. **Virtual Prototype Loss**: Reserve embedding space for future classes via learnable virtual prototypes
2. **Virtual Instance Loss**: Generate synthetic future-like samples via manifold mixup

---

## 15. Models

### 15.1 Heterogeneous Graph Neural Network (HeteroGNN)

The core model is a **Relation-Specific HeteroGNN** with the following components:

- **NodeFeatureEncoders**: Map each node type's raw features to shared hidden dimension
- **RelationSpecificConv**: One learnable weight matrix per edge type
- **RelationSpecificLayer**: Per-relation message passing + attention fusion
- **SemanticAttention**: HAN-style attention fusion across relations

### 15.2 Transfer Adapter (TCTRL)

The TCTRL adapter consists of:
- **Relation Adapters**: Per-relation bottleneck adapters (hidden_dim → adapter_dim → hidden_dim)
- **Task Encoder**: Maps task signature to per-relation gates
- **Zero Initialization**: Final layer zero-initialized so A(z) ≈ 0

### 15.3 Lightweight Classifier Head

A simple 2-layer MLP:
- `classifier = nn.Linear(input_dim, num_classes)`

### 15.4 FACT Module

The FACT module consists of:
- **Virtual Prototypes**: Learnable V × d matrix (V = number of virtual classes, d = feature dim)
- **Virtual Instances**: Generated via manifold mixup during training

---

## 16. Model Architecture

### 16.1 HeteroGNN Architecture

The locked baseline architecture is:
- **Type**: RelationSpecificHeteroGNN
- **Layers**: 3 (with residual connections)
- **Hidden dimension**: 64
- **Attention dimension**: 128
- **Fusion**: Semantic attention (HAN-style)
- **Residual**: True
- **Total parameters**: ~197,842 (3-layer residual)

### 16.2 TCTRL Adapter Architecture

- **Relation adapters**: bottleneck (64 → 32 → 64), zero-initialized
- **Task encoder**: 137 → 64 → 5 (gates per relation)
- **Forward**: $z'_r = z_r + g_r(q) \cdot A_r(z_r)$

### 16.3 FACT Module Architecture

- **Virtual prototypes**: V × d learnable matrix (V=10, d=64)
- **Temperature**: 2.0
- **Mixup alpha**: 1.0

---

## 17. Parameters and Hyperparameters

### 17.1 Model Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| `hidden_dim` | 64 | `configs/model.yaml` |
| `attn_dim` | 128 | `configs/model.yaml` |
| `num_layers` | 3 (baseline) / 1 (diagnostic) | `configs/model.yaml` |
| `port_tail_buckets` | 32 | `configs/model.yaml` |
| `fusion` | "attention" | `configs/model.yaml` |
| `use_residual` | True (baseline) | `configs/model.yaml` |

### 17.2 Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `train.epochs_per_task` | 5 | Total epochs per task |
| `train.warmup_epochs` | 1 or 2 | Warmup epochs (CE only) |
| `train.batch_size` | 8 | Training batch size |
| `train.seed` | 42 | Default random seed |
| `optim.learning_rate` | 0.001 | Adam learning rate |

### 17.3 Replay Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `replay.enabled` | True | Whether to use replay |
| `replay.buffer_size_per_task` | 200 | Buffer size per task |
| `replay.replay_fraction` | 0.3 | Fraction of replay in training pool |
| `replay.selection` | "uniform" | Buffer selection strategy |

### 17.4 RFR Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rfr.enabled` | False (default) | Enable RFR |
| `rfr.lambda` | 0.1 | RFR loss weight |
| `rfr.apply_only_base` | True | Apply only during T1 |
| `rfr.mode` | "global" or "relation" | RFR mode |

### 17.5 FACT Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `fact.enabled` | False (default) | Enable FACT |
| `fact.lambda` | 1.0 | FACT loss weight |
| `fact.num_virtual` | 10 | Number of virtual prototypes |
| `fact.temperature` | 2.0 | Softmax temperature |
| `fact.alpha` | 1.0 | Mixup alpha |
| `fact.apply_only_base` | True | Apply only during T1 |

### 17.6 EWC/Distill Hyperparameters (for reference, not used in baseline)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ewc.lambda_r` | 1.0 | EWC relation weight |
| `ewc.lambda_s` | 1.0 | EWC shared weight |
| `ewc.lambda_u` | 1.0 | EWC other weight |
| `ewc.gamma` | 0.9 | EWC decay |
| `distill.lambda_d` | 1.0 | Distillation weight |
| `distill.temperature` | 0.3 | Sigmoid temperature |

---

## 18. Mathematical Formulas and Equations

### 18.1 Effective Rank (Roy & Vetterli, 2007)

For a matrix $R \in \mathbb{R}^{N \times d}$:

1. L2-normalize rows: $Z = \text{normalize}(R)$
2. SVD: $Z = U \Sigma V^T$, singular values $\sigma_1, \ldots, \sigma_r$
3. Eigenvalues: $p_i = \sigma_i^2 / N$
4. Entropy: $H = -\sum_i p_i \log p_i$
5. Effective rank: $\text{erank} = \exp(H)$

### 18.2 RFR Loss

$$L_{\text{RFR}} = -\log(\text{erank}) = \sum_i p_i \log p_i$$

Minimizing this maximizes effective rank.

### 18.3 Forgetting Metric

$$\text{Forgetting} = \frac{1}{T-1} \sum_{i=1}^{T-1} \left( \max_{t < T} R_{t,i} - R_{T,i} \right)$$

where $R_{t,i}$ is accuracy on task $i$ after training on task $t$.

### 18.4 AULC(1-3) Metric

$$\text{AULC}(1-3) = \frac{f_1 + 2f_2 + f_3}{4}$$

Trapezoidal area under the learning curve using F1-macro at epochs 1, 2, 3.

### 18.5 Semantic Attention

$$\beta_r = \frac{\exp(\text{score}_r)}{\sum_{r'} \exp(\text{score}_{r'})}$$

$$\text{score}_r = \text{MLP}(\tanh(W \cdot h_r))$$

$$h_{\text{fused}} = \sum_r \beta_r \cdot h_r$$

### 18.6 FACT Virtual Prototype Loss

$$L_v = \text{CE}(f_v(x), y) + \text{CE}(\text{mask}(f_v(x)), \hat{y})$$

where $\hat{y} = \arg\max_r p_r^T \phi(x)$ is the nearest virtual prototype.

### 18.7 Manifold Mixup

$$z = \lambda \cdot \phi(x_i) + (1-\lambda) \cdot \phi(x_j)$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$.

---

## 19. Implementation Details

### 19.1 Training Loop (`src/trench_ids/cl/train.py`)

The training loop processes tasks sequentially:
1. For each task T:
   - Load training data
   - Build replay-augmented set
   - Run warmup epochs (CE only)
   - Run full loss epochs (CE + optional KD/EWC/etc.)
   - Update replay buffer
   - Evaluate on all seen tasks

### 19.2 Checkpoint Saving

Each task saves a checkpoint containing:
- Model state dict
- Classifier state dict
- Configuration (for reproducibility)
- Task ID, epoch counts, seed
- Optimizer state

### 19.3 Memory Bank

The memory bank stores per-relation, per-class mean embeddings:
```python
bank: Dict[str, Dict[str, torch.Tensor]]
# bank[relation_name][class_name] = mean_embedding
```

### 19.4 Replay Buffer Management

After each task:
- Random sample N graphs from current task
- Append to buffer
- If buffer exceeds limit, subsample (uniform by default)

---

## 20. Tools, Libraries, Frameworks, and Technologies

### 20.1 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Programming language |
| PyTorch | 2.1+ | Deep learning framework |
| PyTorch Geometric | 2.5+ | Graph neural networks |
| Hydra | 1.3+ | Configuration management |
| OmegaConf | 2.2+ | Config parsing |
| NumPy | 1.24+ | Numerical computing |
| Pandas | 2.0+ | Data manipulation |
| Scikit-learn | 1.3+ | Evaluation metrics |
| Matplotlib | 3.8+ | Plotting |

### 20.2 Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Pytest | 8.0+ | Testing framework |
| Ruff | 0.6+ | Linting |

### 20.3 Hardware

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
- **CUDA**: 13.0
- **PyTorch CUDA**: 2.14.0+cu130

### 20.4 Build System

- **Build backend**: hatchling
- **Package manager**: uv
- **Entry points**: Defined in `pyproject.toml`

---

## 21. Experimental Setup

### 21.1 Hardware Configuration

- **Device**: NVIDIA GeForce RTX 3050 Laptop GPU
- **Memory**: ~4GB VRAM
- **CUDA**: 13.0
- **Runtime per training run**: ~10-30 minutes

### 21.2 Software Configuration

- **OS**: Windows 11
- **Python**: 3.10 (miniconda3)
- **PyTorch**: 2.14.0+cu130
- **Random seeds**: 42, 1, 2

### 21.3 Dataset Configuration

- **Source**: UQ NetFlow v2 datasets
- **Tasks**: 6 (T1-T6)
- **Datasets**: ToN, BoT, CSE
- **Train graphs per task**: ~1,214
- **Test graphs per task**: 260

---

## 22. Experimental Procedure

### 22.1 Baseline Training Procedure

1. Initialize 3-layer residual HeteroGNN
2. Initialize replay buffer (empty)
3. For each task T in [T1, T2, T3, T4, T5, T6]:
   a. Load training data for task T
   b. Add replay buffer to training set (30% replay fraction)
   c. Train for warmup_epochs (CE only)
   d. Train for full_loss_epochs (CE + optional losses)
   e. Update replay buffer (sample 200 graphs)
   f. Evaluate on all tasks 1..T
   g. Save checkpoint

### 22.2 B1 LOTO Evaluation Procedure

For each transition T_n → T_{n+1} (n = 1..5):

1. Load checkpoint from `runs/strong_replay_baseline/checkpoint_task_{n}.pt`
2. Load T_{n+1} training data
3. For epoch in [1, 2, 3]:
   a. Fine-tune on 200 sampled training graphs
   b. Evaluate on T_{n+1} test set
   c. Record accuracy, F1-macro, F1-weighted
4. Compute AULC(1-3) = (f1 + 2*f2 + f3) / 4

### 22.3 Forward-Compatibility Testing Procedure

For each method (RFR, FACT):

1. Configure method-specific parameters
2. Train baseline + method on same 6-task sequence
3. Evaluate B1 LOTO using method's checkpoints
4. Compare AULC and forgetting to baseline

---

## 23. Evaluation Metrics

### 23.1 Retention Metrics

**Average Forgetting**: Mean drop in accuracy from each task's peak to its final accuracy.

$$F = \frac{1}{T-1} \sum_{i=1}^{T-1} (\max_{t \leq T} R_{t,i} - R_{T,i})$$

**Backward Transfer (BWT)**: Mean change in accuracy on each task after final training.

$$\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})$$

### 23.2 Forward Transfer Metrics

**AULC(1-3)**: Area under learning curve for first 3 fine-tuning epochs.

$$\text{AULC}(1-3) = \frac{f_1 + 2f_2 + f_3}{4}$$

### 23.3 Classification Metrics

- **Accuracy**: Fraction of correct predictions
- **Precision/Recall/F1 (macro)**: Unweighted mean across classes
- **Precision/Recall/F1 (weighted)**: Weighted by class support
- **Confusion Matrix**: Per-class predictions vs ground truth

---

## 24. Results

### 24.1 Locked Baseline (3-seed mean, primary result)

| Metric | Value |
|--------|------:|
| **Average Forgetting** | **0.081 ± 0.008** |
| **Final Average Accuracy** | **0.902 ± 0.005** |
| **Final Average Val Accuracy** | **0.896 ± 0.005** |
| **Mean B1 AULC(1-3)** | **0.202** |

### 24.2 Forward Transfer Pattern (B1 LOTO)

| Transition | Mean AULC | 95% CI |
|------------|----------:|----------|
| T1→T2 | 0.288 | [0.284, 0.292] |
| T2→T3 | 0.403 | [0.397, 0.408] |
| T3→T4 | 0.187 | [0.075, 0.299] |
| T4→T5 | 0.053 | [0.036, 0.069] |
| T5→T6 | 0.080 | [0.072, 0.088] |
| **Mean** | **0.202** | [0.175, 0.229] |

### 24.3 Method Comparison

| Method | Forgetting | Final Acc | Mean AULC | Result |
|--------|-----------:|----------:|-----------:|--------|
| **Replay (3L residual, 3-seed)** | **0.081 ± 0.008** | **0.902 ± 0.005** | **0.202** | **Baseline** |
| Replay (3L, single-seed warmup=2) | 0.0673 | 0.9172 | — | Sensitivity |
| Replay (1L) | 0.371 | 0.533 | 0.202 | Arch diagnostic |
| TCTRL | — | 0.758 | 0.150 | Failed |
| RFR λ=0.1 (3L) | 0.0635 | 0.9165 | ~0.202 | No improvement |
| Relation-RFR | 0.442 | 0.473 | 0.125 | Harmful |
| **FACT (λ=1.0)** | 0.059 | 0.918 | 0.140 | **Harmful** |

### 24.4 Task Composition Results

| Task | Attack Classes | Train Graphs | Test Graphs |
|------|----------------|--------------|-------------|
| T1 | Benign + Scanning | 1,214 | 260 |
| T2 | Benign + Reconnaissance | 1,214 | 260 |
| T3 | Benign + DDoS + Infiltration | 1,212 | 260 |
| T4 | Benign + DoS + Injection | 1,214 | 260 |
| T5 | Benign + Password + Bot | 1,214 | 260 |
| T6 | Benign + XSS + BruteForce | 1,214 | 260 |

### 24.5 Forgetting Matrix (Strong Replay Baseline)

| Task | After T1 | After T2 | After T3 | After T4 | After T5 | After T6 |
|------|----------|----------|----------|----------|----------|----------|
| T1 (Scanning) | 0.961 | 0.972 | 0.911 | 0.915 | 0.890 | 0.936 |
| T2 (Recon) | - | 0.956 | 0.950 | 0.935 | 0.894 | 0.940 |
| T3 (DDoS+Infil) | - | - | 0.947 | 0.799 | 0.902 | 0.908 |
| T4 (DoS+Inject) | - | - | - | 0.977 | 0.719 | 0.764 |
| T5 (Password+Bot) | - | - | - | - | 0.995 | 0.962 |
| T6 (XSS+Brute) | - | - | - | - | - | 0.993 |

---

## 25. Tables and Numerical Results

All numerical results are documented in the following locations:
- `FINAL_BENCHMARK_RESULTS.md` — Summary tables
- `B1_ANALYSIS.md` — Detailed B1 LOTO analysis
- `runs/*/summary.json` — Raw per-run summaries
- `runs/*/forgetting_matrix.json` — Per-task forgetting matrices
- `runs/b1_loto/b1_results.json` — B1 LOTO per-epoch results

---

## 26. Graphs, Figures, and What They Show

### 26.1 Forward Transfer AULC vs Task Index (Conceptual)

```
0.403 ┤       ●
0.350 ┤
0.300 ┤ ●
0.250 ┤
0.200 ┤             ●
0.150 ┤
0.100 ┤                     ●     ●
0.050 ┤
0.000 ┴────────────────────────────
      T1-2  T2-3  T3-4  T4-5  T5-6
```

**What it shows**: Forward transfer peaks at T2→T3 (0.403) then collapses sharply after T3, reaching near-zero at T4→T5.

### 26.2 Forgetting Over Time (Per Task)

See Forgetting Matrix in Section 24.5. Each row shows how a task's accuracy changes as more tasks are learned.

---

## 27. Analysis and Interpretation

### 27.1 Backward Compatibility is Strong

The replay baseline achieves 0.081 ± 0.008 forgetting — excellent retention of old tasks. Replay with a 200-graph buffer and 30% replay fraction effectively prevents catastrophic forgetting.

### 27.2 Forward Transfer is Highly Transition-Dependent

Forward transfer (AULC) varies dramatically across transitions:
- T1→T2: 0.288 (good)
- T2→T3: 0.403 (excellent)
- T3→T4: 0.187 (sharp drop)
- T4→T5: 0.053 (near zero)
- T5→T6: 0.080 (still poor)

The model can retain old knowledge extremely well while simultaneously becoming progressively worse at learning new knowledge quickly.

### 27.3 Final Accuracy Pattern

- T1→T2: 91.5% final accuracy
- T2→T3: 85.8% final accuracy
- T3→T4, T4→T5, T5→T6: ~25% (random for 3 classes)

The final accuracy collapses after T3, suggesting the model cannot effectively learn new attack classes that appear later in the sequence.

### 27.4 Dataset Shift Interaction

- T2→T3 (BoT→CSE): Strong forward transfer (0.403)
- T4→T5 (CSE→CSE): Very weak forward transfer (0.053)

The collapse is NOT explained by dataset shift alone — T4→T5 is within the same dataset yet still fails.

### 27.5 Method Ablation Analysis

| Method | Hypothesis | Result |
|--------|------------|--------|
| TCTRL | Task-conditioned adaptation | Failed (oracle 18% accuracy) |
| RFR | Feature richness | No meaningful improvement |
| FACT | Reserved embedding space | Harmful (-31% AULC) |

None of the tested mechanisms resolved the forward-transfer limitation.

---

## 28. Comparisons and Baselines

### 28.1 Architecture Comparison

| Architecture | Forgetting | Final Acc | Notes |
|--------------|-----------:|----------:|-------|
| 1-layer HeteroGNN + replay | 0.371 | 0.533 | Weak baseline |
| **3-layer residual + replay (canonical)** | **0.081** | **0.902** | **Strong baseline** |

The 3-layer residual architecture is critical for strong performance.

### 28.2 Method Comparison

See Section 24.3 for the complete method comparison table.

---

## 29. Limitations

### 29.1 Statistical Limitations

- **Seed count**: The canonical baseline uses 3 seeds (42, 1, 2); B1 LOTO uses only 2 seeds
- **Confidence intervals**: Some 95% CIs are wide (e.g., T3→T4: [0.075, 0.299])
- **No multi-seed FACT evaluation**: FACT was evaluated with only 1 seed for B1 LOTO

### 29.2 Methodological Limitations

- **FACT virtual prototype count**: Fixed at 10; not tuned
- **FACT lambda**: Fixed at 1.0; not tuned
- **RFR lambda**: Sweep done at 0.01, 0.1, 1.0; intermediate values not tested
- **Fine-tuning data size**: Only 200 graphs sampled for B1 evaluation

### 29.3 Interpretive Limitations

- **Correlation vs Causation**: The data is "consistent with" progressive representation specialization but does not causally prove it
- **Other explanations remain possible**: Intrinsic task difficulty, dataset shift interaction with replay, etc.

---

## 30. Assumptions

1. **Replay buffer management**: Uniform random sampling is sufficient for buffer selection
2. **Class similarity design**: Tasks are designed to have specific similarity relationships
3. **Port embedding**: Well-known ports (0-1023) have exact embedding rows; tail ports use log-bucketing
4. **Dense embeddings**: Protocol/Service/Port use learned dense embeddings rather than target encoding
5. **Semantic attention**: HAN-style attention fusion is appropriate for relation-specific embeddings
6. **F1-macro as AULC metric**: AULC is computed using F1-macro across the 3 fine-tuning epochs

---

## 31. Challenges and Issues Encountered

### 31.1 Initial Architecture Issues

- **1-layer architecture showed catastrophic forgetting (0.371)**: The 1-layer architecture was inadequate for the task
- **Solution**: Moved to 3-layer residual architecture which dramatically improved performance (0.081)

### 31.2 TCTRL Meta-Learning Failure

- **FOMAML with task-conditioned adapter failed**: Even supervised oracle only achieved 18% accuracy vs 95% baseline
- **Diagnosis**: The adapter path was fundamentally broken; the model couldn't recover the baseline representation after modification

### 31.3 Device Mismatch Issues

- **CPU vs CUDA detection**: Initial runs sometimes used wrong device
- **Solution**: Explicit device resolution in scripts

### 31.4 Hydra Configuration Issues

- **The `train` subcommand was interpreted as config override**: PowerShell parsing issues with Hydra CLI
- **Solution**: Used Python module execution with conda environment

### 31.5 Test Suite Issues

- **8 tests were blocked by import path issue**: The `tests.` prefix on import line 9 in test_train_flat.py
- **Solution**: One-line import fix; full suite now reports 373/373 pass

### 31.6 RFR and FACT Results Issues

- **RFR on wrong architecture (1-layer) showed improvement, but on correct architecture (3-layer) showed no improvement**: The 1-layer results were misleading
- **Solution**: Rerun on correct 3-layer residual architecture

### 31.7 FACT Implementation Issues

- **First FACT implementation didn't use model's fusion module correctly**: Led to distorted representations
- **Solution**: Used model's own SemanticAttention for fusion

---

## 32. Solutions and Modifications Made

### 32.1 Architecture Selection

Changed from default 1-layer to 3-layer residual HeteroGNN. This change:
- Reduced forgetting from 0.371 to 0.081
- Improved final accuracy from 0.533 to 0.902

### 32.2 Replay Strategy

Adopted uniform random sampling for replay buffer (200 graphs/task, 30% fraction):
- Effective for backward retention
- Simple, reproducible
- Strong baseline performance

### 32.3 Evaluation Protocol

Developed B1 LOTO evaluation:
- Load checkpoint after each task
- Fine-tune on next task for 3 epochs
- Measure AULC(1-3) as forward transfer metric
- Test across 5 transitions × 2 seeds

### 32.4 Method Implementation

Implemented RFR and FACT as modular additions:
- `src/trench_ids/rfr.py`: Effective rank computation and loss
- `src/trench_ids/cl/fact.py`: Virtual prototype loss
- Added config options in `configs/train.yaml`
- Integrated into `train.py` training loop

---

## 33. Final Conclusions

### 33.1 What Was Established

1. **Strong replay baseline**: 3-layer residual HeteroGNN + replay achieves 0.081 ± 0.008 forgetting, 0.902 ± 0.005 final accuracy
2. **Forward transfer pattern**: Highly transition-dependent, peaks at T2→T3 (0.403), collapses after T3
3. **Clear separation of concerns**: Backward compatibility ≠ forward transfer
4. **B1 LOTO evaluation framework**: Reproducible protocol for measuring forward transfer

### 33.2 What Was Tested

Three qualitatively different forward-compatibility hypotheses:
1. **TCTRL**: Task-conditioned adaptation — Failed
2. **RFR**: Feature richness regularization — No meaningful improvement
3. **FACT**: Virtual prototype reservation — Harmful

### 33.3 What Remains Open

- The fundamental cause of the late-sequence forward-transfer collapse is not yet established
- Other mechanisms (e.g., representation invariance, domain generalization) remain untested
- The hypothesis of "progressive representation specialization" is consistent with data but not causally proven

### 33.4 Final Scientific Position

> **Replay is highly effective for backward retention, but forward transfer remains strongly transition-dependent and deteriorates later in the task sequence. TCTRL, RFR, and FACT did not improve that forward-transfer behavior.**

The project characterized a forward-transfer limitation and showed that several intuitive mechanisms do not resolve it.

---

## 34. Future Work

### 34.1 Mechanism Investigation

- **Per-task representation tracking**: Measure effective rank, class separation, embedding drift per task boundary
- **Dataset shift vs task difficulty**: Systematically separate these factors
- **Specific failure mode analysis**: Why do T3→T4, T4→T5, T5→T6 fail?

### 34.2 Alternative Methods to Test

- **Representation invariance methods**: DIB-OD (Decoupled Information Bottleneck)
- **Graph-specific methods**: SAOT (Structure-Aware Optimal Transport)
- **Test-time adaptation**: Methods that adapt at inference time

### 34.3 Dataset Expansion

- Test on additional datasets (e.g., NF-UNSW-NB15-v2)
- Include more attack classes
- Different task partitioning strategies

### 34.4 Theoretical Analysis

- Formal analysis of when forward transfer should be possible
- Information-theoretic bounds on forward transfer
- Relationship between task similarity and transferability

---

## 35. References and Sources

### 35.1 Continual Learning

- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS, 2017) — EWC
- Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning" (NeurIPS 2017) — GEM
- Li & Hoiem, "Learning without Forgetting" (TPAMI 2018) — LwF
- Rebuffi et al., "iCaRL: Incremental Classifier and Representation Learning" (CVPR 2017)

### 35.2 Graph Neural Networks for NIDS

- Lo et al., "E-GraphSAGE" — Edge-featured GraphSAGE for NIDS
- Pujol-Perich et al., "Unveiling the potential of graph neural networks for robust intrusion detection" (SIGMETRICS 2022)
- Zhou et al., "Heterogeneous Graph Neural Networks for NIDS"

### 35.3 Forward-Compatibility Methods

- Zhou et al., "Forward Compatible Few-Shot Class-Incremental Learning (FACT)" (CVPR 2022)
- Kim et al., "Improving Forward Compatibility in Class Incremental Learning by Increasing Representation Rank and Feature Richness (RFR)" (Neural Networks, 2024)
- Wang et al., "Task-Agnostic Guided Feature Expansion (TagFex)" (CVPR 2025)
- Zhu et al., "Class Incremental Learning with Dual Augmentation (IL2A)"

### 35.4 Self-Supervised and Continual Graph Learning

- Liu et al., "Graph Continual Learning with Structure-Aware Optimal Transport (SAOT)" (2026)
- Sun et al., "HERO: Heterogeneous Continual Graph Learning via Meta-Knowledge Distillation" (2025)
- GCAL: Graph Continual Adaptive Learning (ICML 2025)

### 35.5 Graph SSL for NIDS

- Xu et al., "Applying Self-Supervised Learning to Network Intrusion Detection (NEGSC)" (Computer Networks, 2024)
- Guerra et al., "GraphIDS: Self-Supervised Learning of Graph Representations for Network Intrusion Detection" (NeurIPS 2025)
- Cai et al., "PPT-GNN: Practical Pre-Trained Spatio-Temporal Graph Neural Network" (2024)

---

## 36. Complete Technical Terminology Glossary

| Term | Definition |
|------|------------|
| **AULC** | Area Under Learning Curve — measures how quickly a model learns a new task |
| **Backward Transfer** | Impact of learning new tasks on performance of old tasks |
| **Baseline** | Reference system (3-layer residual HeteroGNN + replay) against which other methods are compared |
| **B1 LOTO** | Leave-One-Task-Out evaluation at Stage B1 — measures forward transfer across transitions |
| **Beta (β)** | Attention weight for a relation in SemanticAttention fusion |
| **CIL** | Class Incremental Learning |
| **Continual Learning** | Learning from a stream of tasks without forgetting previous tasks |
| **CSE** | NF-CSE-CIC-IDS2018-v2 dataset |
| **DIB-OD** | Decoupled Information Bottleneck with Online Distillation |
| **EWC** | Elastic Weight Consolidation — regularization-based CL method |
| **Effective Rank** | Continuous generalization of matrix rank based on Shannon entropy of singular values |
| **FACT** | Forward Compatible Training — uses virtual prototypes to reserve embedding space |
| **F1-macro** | Unweighted mean of F1 scores across classes |
| **Forward Transfer** | Impact of learning previous tasks on ability to learn new tasks efficiently |
| **FOMAML** | First-Order Model-Agnostic Meta-Learning |
| **GNN** | Graph Neural Network |
| **GraphSAGE** | Graph Sample and Aggregate neural network |
| **HAN** | Heterogeneous Graph Attention Network |
| **HeteroGNN** | Heterogeneous Graph Neural Network — handles different node/edge types |
| **HERO** | Heterogeneous Continual Graph Learning via Meta-Knowledge Distillation |
| **HKD** | Heterogeneity-aware Knowledge Distillation |
| **IL2A** | Incremental Learning with Dual Augmentation |
| **Knowledge Distillation** | Training a student model to mimic a teacher model's outputs |
| **LwF** | Learning without Forgetting |
| **Manifold Mixup** | Interpolating in feature space to create synthetic samples |
| **NIDS** | Network Intrusion Detection System |
| **NEGSC** | NetFlow-Edge Generative Subgraph Contrast |
| **Replay** | Storing and replaying samples from previous tasks to prevent forgetting |
| **RFR** | Effective-Rank based Feature Richness enhancement |
| **SAOT** | Structure-Aware Optimal Transport |
| **Stage A** | Functional transfer analysis via counterfactual relation-reset experiments |
| **Stage B** | Method development for forward-compatible representation learning |
| **TCTRL** | Task-Conditioned Transfer Representation Learning — task-conditioned adapter with FOMAML |
| **TagFex** | Task-Agnostic Guided Feature Expansion |
| **Virtual Prototype** | Learnable embedding vector representing a reserved space for a future class |
| **VQ** | Vector Quantization |

---

## 37. Complete List of Important Numbers, Parameters, and Values

### 37.1 Architecture Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 64 | GNN hidden dimension |
| `attn_dim` | 128 | Semantic attention dimension |
| `num_layers` | 3 | GNN number of layers (canonical baseline) |
| `port_tail_buckets` | 32 | Log-bucketed port tail embedding buckets |
| `fusion` | "attention" | Semantic attention fusion method |
| `use_residual` | True | Residual connections |
| Total parameters | ~197,842 | For 3-layer residual |

### 37.2 Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `train.epochs_per_task` | 5 | Total epochs per task |
| `train.warmup_epochs` | 1 (canonical) or 2 (sensitivity) | Warmup epochs |
| `train.batch_size` | 8 | Training batch size |
| `train.seed` | 42 | Default random seed |
| `optim.learning_rate` | 0.001 | Adam learning rate |
| `optim.name` | "adam" | Optimizer |

### 37.3 Replay Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `replay.enabled` | True | Whether to use replay |
| `replay.buffer_size_per_task` | 200 | Buffer size per task |
| `replay.replay_fraction` | 0.3 | Fraction of replay in training pool |
| `replay.selection` | "uniform" | Buffer selection strategy |

### 37.4 RFR Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rfr.enabled` | False (default) | Enable RFR |
| `rfr.lambda` | 0.1 | RFR loss weight |
| `rfr.apply_only_base` | True | Apply only during T1 |
| `rfr.mode` | "global" or "relation" | RFR mode |

### 37.5 FACT Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `fact.enabled` | False (default) | Enable FACT |
| `fact.lambda` | 1.0 | FACT loss weight |
| `fact.num_virtual` | 10 | Number of virtual prototypes |
| `fact.temperature` | 2.0 | Softmax temperature |
| `fact.alpha` | 1.0 | Mixup alpha |
| `fact.apply_only_base` | True | Apply only during T1 |

### 37.6 Key Result Numbers (Canonical Baseline)

| Metric | Value | Source |
|--------|------:|--------|
| Average Forgetting | 0.081 ± 0.008 | 3-seed mean |
| Final Average Accuracy | 0.902 ± 0.005 | 3-seed mean |
| Final Average Val Accuracy | 0.896 ± 0.005 | 3-seed mean |
| Mean B1 AULC(1-3) | 0.202 | 2-seed mean |
| T1→T2 AULC | 0.288 | 2-seed mean |
| T2→T3 AULC | 0.403 | 2-seed mean |
| T3→T4 AULC | 0.187 | 2-seed mean |
| T4→T5 AULC | 0.053 | 2-seed mean |
| T5→T6 AULC | 0.080 | 2-seed mean |

### 37.7 Task Composition

| Task | Attack Classes | Source | Train | Test |
|------|----------------|--------|------:|-----:|
| T1 | Benign + Scanning | ToN | 1,214 | 260 |
| T2 | Benign + Reconnaissance | BoT | 1,214 | 260 |
| T3 | Benign + DDoS + Infiltration | ToN, CSE | 1,212 | 260 |
| T4 | Benign + DoS + Injection | ToN, CSE | 1,214 | 260 |
| T5 | Benign + Password + Bot | ToN, CSE | 1,214 | 260 |
| T6 | Benign + XSS + BruteForce | ToN, CSE | 1,214 | 260 |

### 37.8 File Paths

| Path | Description |
|------|-------------|
| `src/trench_ids/cl/train.py` | Main training loop |
| `src/trench_ids/rfr.py` | RFR implementation |
| `src/trench_ids/cl/fact.py` | FACT implementation |
| `src/trench_ids/model/rhgnn.py` | HeteroGNN model |
| `configs/train.yaml` | Training configuration |
| `data/graphs/` | Processed graph files |
| `runs/gnn_3layer_residual_s42/` | Baseline run (seed 42) |
| `runs/b1_loto/b1_results.json` | B1 LOTO results |
| `FINAL_BENCHMARK_RESULTS.md` | Benchmark summary |
| `VERIFICATION_REPORT.md` | Independent audit report |

---

*Document version: 1.0*  
*Last updated: 2026-09-07*  
*Benchmark status: FROZEN*