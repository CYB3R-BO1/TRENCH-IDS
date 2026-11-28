# TRENCH-IDS
## Transferable Representation Learning for Continual Heterogeneous Graph-Based Network Intrusion Detection

**Version:** 1.0  
**Date:** 2026-11-28  
**Status:** Proposed final research methodology; project-owner approved for repository documentation, pending professor review. No new experimental outcomes are claimed.

This document defines the research direction and validation plan, not a predetermined winning algorithm. The [initial proposal](archive/INITIAL_METHODOLOGY.md) and [frozen Step 1 research](archive/step-1/STEP1_FINAL_REPORT.md) are historical evidence. The original reproducible layout is available at the [step1-frozen tag](https://github.com/CYB3R-BO1/TRENCH-IDS/tree/step1-frozen). New replay arms, natural-traffic evaluation and changes to frozen task construction require an explicit protocol amendment; do not silently modify the archive.

## 1. Motivation and research objective

Network intrusion detectors trained on known attack categories must adapt to evolving attack behavior. Sequential learning may cause catastrophic forgetting, negative transfer, or costly repeated training. Network flows also expose different relational contexts: host communication, ports, protocols and services. These may contain different amounts of reusable knowledge across attack categories.

**Objective:** To investigate, quantify and exploit relation-specific knowledge transfer between sequentially introduced network attack categories using heterogeneous graph representations, with the aim of improving adaptation to new attacks while retaining previously learned intrusion detection capabilities.

### Research questions and hypotheses

| ID | Research question | Hypothesis |
|---|---|---|
| RQ1 / H1 | Does graph context add useful detection information beyond independent flow features? | Heterogeneous graph context contains useful complementary information. |
| RQ2 / H2 | Does previous attack learning improve detection and subsequent learning of a withheld attack? | Previous representations can improve target learning. |
| RQ3 / H3 | Which relations contribute to positive or negative transfer? | Transfer varies by attack pair, direction and relation group. |
| RQ4 / H4 | Can learned relation representations estimate transfer before full target training? | Relation measurements can predict measured transfer effects. |
| RQ5 / H5 | Can validated estimates improve continual learning? | Relation guidance can improve the stability–plasticity trade-off under resource constraints. |

All hypotheses may fail. Negative and inconclusive results are valid and must be reported. H5 is conditional on a defensible estimator, not a promise of an improvement.

## 2. Scope and terminology

| Component | Scope |
|---|---|
| Domain | Network intrusion detection from NetFlow records |
| Primary dataset | NF-UNSW-NB15-v3 |
| Secondary dataset | NF-ToN-IoT-v3 |
| Learning | Supervised, domain-incremental binary classification with sequential attack-category introduction |
| Training task boundary | Known |
| Inference task identity | Not required or provided |
| Prediction unit | Flow |
| Graph unit | Temporal traffic window |
| Detection setting | Retrospective completed-flow records |
| Primary detection metric | Average Precision (AP) |

The output label space is fixed: benign = 0, attack = 1. Raw attack categories define tasks and evaluation slices, not classifier outputs. A new attack is a category absent from earlier model training; this is not a claim of real-world zero-day detection. Zero-shot detection occurs before target-dependent fitting. Forward-transfer advantage compares target learning with and without prior source learning. Forgetting is deterioration on previous tasks. Relation contribution is conditional on controlled architectural comparisons. A transferability estimate is a candidate predictor, not actual transfer itself.

Out of scope initially: task-free learning, automatic attack discovery, multiclass attack attribution, live packet inspection and cross-organization learning.

---

## Phase 0 — Research design and experimental protocol

**Objective:** Establish information access, evaluation, reproducibility and development rules.

1. Approve the research questions, hypotheses, scope and success criteria.
2. Review archived Step 1 evidence and its limitations; preserve the frozen tag and archive.
3. Specify dataset-specific splits, task membership, model inputs and experiment families.
4. Define information access, development versus confirmatory evaluation, paired seeds, metrics and resource constraints.
5. Measure engineering feasibility before committing to a large experiment matrix; freeze exact budgets, selection criteria and configurations before confirmatory runs.
6. Document protocol amendments and obtain approval before implementing changes to frozen decisions.

The researcher may use dataset metadata and training-only screening statistics to design tasks. The learner may access current-task training data and labels, previously learned parameters, and only the historical memory permitted by its method. Validation access must follow the declared protocol. Future-task training features, labels, fitted statistics and representations must not enter the learner before task introduction. Offline all-class analysis must not become model input. Reserve test results from architecture, hyperparameter, estimator and adaptive-mechanism selection. Development uses synthetic data, training subsets and permitted validation only. Any inspection of held-out data for integrity or support must be predefined and must not use test performance.

**Output/gate:** Approved protocol, information-access policy and resource/verification plan.

## Phase 1 — Dataset acquisition and understanding

**Objective:** Understand every raw field and collection limitation.

1. Acquire the original UNSW v3 dataset from the publisher; record version, filename, source, SHA-256, row/column counts and feature definitions.
2. Understand the flow record, collection environment and measurement availability, including timestamps, direction and completed-flow semantics.
3. For every column document meaning, units, type, valid domain, missing-value semantics, structural role, predictive role, transformation and leakage risk.
4. Examine all raw attack categories and benign traffic, class support, temporal coverage and collection-specific shortcuts.
5. Separate schema/integrity inspection from distribution-based choices; make the latter with training/allowed validation data.
6. Repeat dataset-specific understanding for ToN before secondary experiments.

Keep raw data immutable and gitignored. Recorded historical counts are 2,365,424 UNSW flows and 27,520,260 ToN flows, each with 55 columns; verify against the actual local files rather than assuming these counts.

**Output/gate:** Verified dataset profile and complete feature dictionary understood by the researcher.

## Phase 2 — Data validation and preprocessing

**Objective:** Produce validated, traceable flow records.

1. Validate schema, required columns, labels and raw category consistency.
2. Validate timestamp units and ranges, start/end order and duration semantics.
3. Validate hosts, ports, protocol/service codes and numerical domains. Distinguish genuine zero from invalid or unavailable values.
4. Audit missing/invalid values and duplicates; do not blindly fill with zero or drop identical records. Define explicit retention, exclusion and missingness rules; reconcile row counts.
5. Assign stable flow identifiers for tracing raw row → validated row → split → task → window → graph node → prediction. Never use these identifiers as predictive features.
6. Establish chronological splits, audit boundary-crossing flows and prevent physical-window split overlap.
7. Partition traffic by raw attack category and benign category, preserving original values, timestamps, identifiers and split membership. Separate CSVs are permitted for inspection; storage format is not a research claim.
8. Define and verify feature transformations and exclusions. Reconcile all membership and data counts.

### Split policy

UNSW initially uses the archived global chronological 70/85-percentile cuts aligned to 60 seconds (approximately 70/15/15 train/validation/test); use the authoritative graph split manifest, not the separate similarity-screening cut. ToN initially uses the archived per-class contiguous chronological cuts; verify physical-window split integrity. ToN task introductions are synthetic, not a global historical timeline. Audit long flows crossing split boundaries and require a purged or otherwise temporally isolated comparison if overlap is material. Start-time windows with completed-flow features are retrospective, not claims of real-time detection.

### Feature policy

- Label and Attack are supervision/task/evaluation metadata, never inputs.
- IPs are window-local connectivity keys, not global learned identifiers; absolute timestamps primarily organize windows.
- Numerical transformations must be justified by semantics (e.g., fixed log1p for appropriate nonnegative magnitudes).
- Ports, protocols and services require validated categorical/semantic representations; distinguish source and destination roles.
- Decode TCP bitmasks where justified; exclude unsupported identifier-like features.
- Prefer fixed domain-defined transformations. Any fitted statistics, vocabulary or normalization must use only data available at that learning stage; do not reuse offline all-class similarity normalization in the model.

**Output/gate:** Validated flow partitions, traceable IDs, approved split and feature policy; no silent duplication or leakage.

## Phase 3 — Attack-class analysis and continual learning task construction

**Objective:** Construct reproducible task populations and sequences.

1. Check eligibility by per-split attack support, temporal coverage, attack-present graph count, benign support and graph density. The archived threshold was 1,000 attack flows per split; review rather than silently change it.
2. Review training-only screening evidence: standardized mean-profile cosine, Wasserstein, Jensen–Shannon, relation profiles and TCP flag bits. These describe statistical similarity, not learned transfer.
3. Retain 10-second primary windows, 5-second sensitivity, and 30-second exploratory windows as initial candidates. Assign epoch-aligned, half-open physical windows using floor(start_timestamp / window_duration), with consistent units.
4. Form a designated-attack task graph from every eligible attack-present window. Keep that class's attacks, exclude other attack classes (including unlabeled context), and allocate same-window benign traffic.
5. Preserve deterministic unique benign ownership across the entire eligible class set, independently per dataset and resolution. Traverse windows in ascending order; among eligible classes sharing a window award all benign flows to the lowest exact fraction of already-awarded benign windows over attack-present windows in that split; tie-break by smaller denominator, larger attack count, then ascending raw label. Preserve attack-only and single-flow graphs. Benign-only and excluded-attack-only windows are not allocated in this controlled protocol.
6. Verify physical-window split integrity, row membership, unique benign ownership, reversibility of task order and both labels across each evaluated task split. Do not confuse graph count with independent sample count.
7. Define directional pairwise transfers and full task orders. Use the archived [task specification](archive/step-1/TRENCH_TASK_SEQUENCE_DESIGN.md) and [protocol](archive/step-1/step1_protocol.json) as references. Repeated-benign allocation is a sensitivity condition, not the primary protocol.

Initial eligible UNSW categories: DoS, Exploits, Fuzzers, Generic, Reconnaissance. Archived contrasts: DoS → Fuzzers (high numerical similarity), Generic → Exploits (lower similarity), Exploits → Fuzzers (host/port contrast), including reverse directions. Additional pairs help validate estimators; the selected contrasts alone do not support general prediction claims. Use archived ToN contrasts and full orders when feasible. The archived seeds and orders are starting specifications, not an obligation to run an infeasible matrix.

Controlled task graphs are label-conditioned research populations, not deployment graphs. Complement them with natural, unfiltered traffic graphs for final evaluation.

**Output/gate:** Deterministic inventory of dataset, split, raw attack class, physical window, flow IDs, benign ownership and task sequences; reconcile with [archived feasibility evidence](archive/step-1/evidence/task_feasibility_10s.csv) or document differences.

## Phase 4 — Heterogeneous graph construction

**Objective:** Expose meaningful network relationships to flow-level prediction.

1. Specify the graph schema, node attributes, edge directions and semantic relation groups.
2. Manually construct a small synthetic window and compute expected node/edge counts and feature values.
3. Implement deterministic graph conversion, flow labels and metadata; verify against the example before scaling.
4. Audit graph size, degrees, connectivity, isolated flows, relation density, RAM and storage. Define oversized-graph behavior explicitly; no silent truncation or dropping.

Conceptual node types: Flow, Host, Port, Protocol and Service. Additional structural entities are allowed only if needed for a justified relation. Seven candidate semantic groups: source host, destination host, host pair, source port, destination port, protocol and service. Source/destination roles must remain distinguishable even if node types are shared. Define forward/reverse computational edges so that messages reach supervised flow nodes; a host-to-host edge alone does not establish a usable host-pair branch.

Flow nodes use approved traffic features. Other nodes use fixed semantic features, constants or identity-free aggregates where justified. Do not expose ground truth in aggregates or use global host embeddings. Require one flow node per included record, valid endpoints and labels, no cross-window or cross-split edges, deterministic construction and no future-task information.

**Output/gate:** Verified graph representation and measured resource needs; the researcher can manually reproduce a small graph.

## Phase 5 — Static intrusion detection and representation learning

**Objective:** Establish a functional IDS and justify the graph architecture before continual learning.

1. Build a flat flow-feature MLP, a host-aware graph model and a heterogeneous relation-specific model. A flat model augmented with structural statistics is a useful additional control when feasible.
2. Share data, evaluation and comparable optimization conditions across models. Audit parameter counts and differences in inputs.
3. Begin with a small model (e.g., PyTorch/PyG, 32–64 hidden dimensions, lightweight message passing); these are provisional engineering choices, not frozen hyperparameters.
4. For each flow i, obtain base embedding h_i and separate relation embeddings z_{i,r} = f_r(G,X;theta_r)_i. Fuse these through simple mean aggregation or concatenation/projection initially; pass the result to a shared binary classification head. Attention is optional if justified, not an explanation of transfer by itself.
5. Expose relation-specific embeddings and identifiable parameter groups; account for shared flow encoder, fusion and head.
6. Use binary cross-entropy with logits initially (unweighted per archived recipe). Verify gradients, finite outputs, small-subset learning and correct evaluation. Consider imbalance handling only as a controlled validation-based change.
7. Select a common architecture and training recipe on allowed development data; freeze before confirmatory comparisons.

Graph context may fail to improve static AP yet help transfer/retention; investigate this rather than assuming the GNN is necessary. If no benefit is observed, report it or narrow the claim.

**Output/gate:** Reproducible binary classifier, static baselines and understandable relation-specific representation pipeline.

## Phase 6 — Knowledge transfer investigation

**Objective:** Measure actual source-to-target transfer before developing preservation mechanisms.

1. Use Exploits → Fuzzers as the first engineering pilot (adequate archived support and relation contrast); one seed/small predetermined subset for debugging, not scientific proof.
2. Train source only; preserve encoder, classifier and configuration.
3. Measure zero-shot detection before target-dependent fitting; compare with a seed-matched untrained detector and report prevalence.
4. Compare matched arms:

| Arm | Initialization |
|---|---|
| Target-only scratch | Random encoder and fresh head |
| Full warm-start | Source encoder and source head |
| Encoder-only transfer | Source encoder and fresh head paired with scratch |
| Frozen encoder transfer | Frozen source encoder and trainable fresh head |
| Untrained reference | No learning; zero-shot reference |

Reset target optimizer state consistently. Match target examples, update budget, learning-rate schedule, traversal and validation access. Report source pretraining cost separately. The archived 10-second budget is U = max eligible training graph count, one graph per update, chronological cyclic traversal; validate computational feasibility before freezing a revised recipe. The archived checkpoint fractions are 0, 1%, 10%, 25%, 50%, 100%, rounded up and deduplicated. Zero is the zero-shot comparison; positive checkpoints measure matched target learning.

Define directional transfer T(A→B) = AP(pretrained on A, then trained on B) − AP(scratch trained on B). Final matched-budget AP advantage is the primary transfer endpoint, early learning curves secondary. Encoder-only comparisons isolate representation reuse from classifier reuse. Record positive, negative and negligible transfer, including reverse directions. Use validation for development measurements; seal confirmatory test outputs.

**Output/gate:** Directional transfer measurements and learning curves; quantify where previous knowledge helps, harms or has no effect.

## Phase 7 — Relation-specific transfer analysis

**Objective:** Identify conditional contributions of relations.

1. Frozen relation probes: extract source-trained relation embeddings on target graphs; train a small linear classifier on permitted target training data. Compare with equivalent probes of untrained representations and evaluate on held-out data.
2. Relation masking: consistently remove the designated relation in source, target and scratch arms. Define C_r = T_full − T_without_r. Distinguish edge-only ablation (message-passing contribution) from removal of the relation and its duplicated direct features (information-group contribution). Keep the supervised flow population constant and control remaining architecture/initialization as far as possible.
3. Selective parameter transfer (where feasible): reuse all source parameters except a reinitialized relation branch; compare with full encoder transfer using matched target conditions.
4. Record source, target, relation, seed, probe AP/gain, full and masked transfer advantage, and selective transfer effect when run.

The probe measures discriminative information; masking and reinitialization measure conditional incremental effects, not independent causal importance. Shared representations, capacity and correlated relations limit attribution.

**Output/gate:** Quantitative relation-level transfer evidence.

## Phase 8 — Transferability estimation and prototypes

**Objective:** Test whether inexpensive representation measurements predict actual transfer.

1. Freeze a trained source encoder; compute source and target embeddings in the same representation space.
2. For class/task c and relation r, compute attack prototype p+ = mean(z | attack) and benign prototype p− = mean(z | benign); discriminative prototype d = p+ − p−.
3. Define weighting (flow-weighted initially; graph-weighted sensitivity if justified), minimum support, treatment of missing/zero-norm vectors and encoder version. Do not silently assign a similarity to undefined cases.
4. Compare candidate scores: archived raw statistical similarity, attack-prototype cosine, discriminative-prototype cosine and frozen linear-probe gain. LogME or gradient alignment are optional alternatives. Cosine(a,b) = dot(a,b)/(norm(a)*norm(b)) for nonzero vectors.
5. Use a predetermined permitted labeled target-training prefix/sample or budget, identical across estimators. This is not zero-shot. Account for estimator data access and computational cost; do not use target test data.
6. Compare scores against measured directional transfer and C_r using rank correlation, ability to identify useful/harmful transfer and, if a calibrated predictor is fitted, prediction error. Group validation by attack category or transfer condition; repeated seeds, flows and relations are not independent transfer experiments. Hold out target categories where feasible; with few classes, restrict claims to exploratory evidence.
7. Optional warm-up comparison: before adaptation versus after a small prespecified target warm-up. Count its exposure and updates in the target budget; recompute both prototypes compatibly. Post-full-training estimates are diagnostic, not prospective.

Same-encoder comparison avoids immediate coordinate drift; cosine similarity alone does not establish transfer, and attack-only similarity may ignore benign separability or embedding collapse. If scores are not predictive, try one justified alternative or report a negative finding. Do not force a weighting mechanism.

**Output/gate:** Validated estimator with known limits and cost, or documented evidence of insufficient predictive validity.

## Phase 9 — Conventional continual learning baselines

**Objective:** Compare standard strategies on the same data, encoder and task schedule.

- Sequential fine-tuning: retain parameters; train on current data without explicit retention.
- Standard EWC: preserve parameter reference theta*, estimate diagonal Fisher F from permitted prior training graphs, add (lambda/2) sum_i F_i (theta_i − theta_i*)² to current loss. Specify and verify Fisher estimation (do not equate squared averaged minibatch gradient with per-example Fisher), parameter scope, shared parameters, strength and separate-task versus online consolidation policy.
- Ordinary experience replay: retain complete training graph windows initially, preserving graph context. Define buffer capacity, selection, eviction, task coverage, frequency, replay loss weight and memory bytes. Current loss + gamma * replay loss is the initial objective. No test/validation examples in replay.
- Offline joint training: train using all relevant task training examples as an offline reference, not a streaming competitor or guaranteed upper bound.

Implement independently, one at a time; validate on two tasks before full sequences. DER++ is optional if justified; hybrid EWC+replay is deferred. Replay is an essential practical comparator, not presumed superior. The archived no-replay protocol must be explicitly extended for these arms.

Match architecture, target exposure, task order, evaluation graphs, seeds and comparable tuning opportunities. Account for source training, Fisher estimation, replay exposures and storage. Compare persistent method-state bytes (parameters/Fisher, graphs, labels, prototypes, extra encoders, logits as applicable); graph count alone is insufficient. Where feasible include separate compute- and memory-controlled comparisons.

**Output/gate:** Reliable adaptation, retention and resource comparison; identify a viable parent method.

## Phase 10 — Relation-guided continual learning

**Objective:** Investigate whether validated relation information improves a conventional parent strategy. Select one primary adaptive mechanism based on prior evidence, not final test results.

### Candidate A: transferability-guided EWC

Partition parameters into relation groups Theta_r and shared parameters. Candidate objective:

L = L_current + (lambda/2) sum_r w_r sum_{i in Theta_r} F_i (theta_i − theta_i*)² + L_shared.

Test bounded deterministic weights: uniform, similarity-guided, inverse-guided and shuffled-score controls, with comparable effective consolidation strength. Define how shared parameters are preserved. Higher compatibility does not necessarily require stronger preservation; the direction is a hypothesis.

### Candidate B: transferability-guided replay

Use validated relation scores to guide historical graph selection or allocation of replay opportunities, retaining minimum coverage of previous attacks. Compare ordinary replay and matched non-guided/shuffled-score controls under the same buffer, replay frequency, replay loss weight and target exposure. Define how relation scores aggregate into graph or task priorities. If justified later, investigate replay-supported relation embedding preservation (distillation) as an extension; account for teacher state and additional loss.

### Decision policy

If replay-free learning is required, prioritize A and retain replay as external comparator. If replay is allowed and is the stronger practical foundation, prioritize B. Hybrid methods require demonstrated complementary benefit. Defer the relation-importance MLP until sufficient independent transfer conditions, target supervision, loss and held-out validation exist. Do not optimize unconstrained weights merely to lower the new-task penalty. If no valid estimator or no benefit over the parent baseline, report the limitation; avoid indefinite method searching. Resolve Phase 11 memory feasibility before running full adaptive sequences.

**Output/gate:** Evidence-supported adaptive mechanism and controls, or a defensible negative finding.

## Phase 11 — Prototype memory and sequential representation maintenance

**Objective:** Keep historical representations valid within the chosen method's information and resource constraints.

1. Retain compact prototypes only where needed; record class/task, relation, attack/benign means, support, dimensions, encoder version and checkpoint.
2. Define update schedule and drift checks; do not compare embeddings from incompatible changing encoders without correction.
3. For replay, recompute historical prototypes using permitted retained graphs and the current encoder; measure support/representativeness of the buffer.
4. For replay-free learning, evaluate a fixed reference encoder or validated alignment strategy; do not assume an old reference stays informative. Account for any retained checkpoint, reference encoder and alignment cost.
5. Finalize lifecycle before full adaptive sequential runs. Restrict or reject mechanisms that cannot maintain meaningful scores under the stated memory policy.

**Output/gate:** Reproducible, resource-accounted memory lifecycle or explicit rejection of an infeasible candidate.

## Phase 12 — Final evaluation and research validation

**Objective:** Answer each research question using frozen confirmatory conditions.

### Detection and transfer

Primary detection metric: AP with attack prevalence/no-skill reference. Secondary: AUROC (no-skill 0.5), recall, FPR, precision and F1. Use fixed thresholds (archived 0.5) or thresholds chosen only on allowed validation. Report recall and realized test FPR at a threshold selected on validation to maximize recall under an empirical 1% FPR cap, including the no-detection option and archived tie rules. Task-specific thresholds are diagnostic only; use a task-agnostic threshold for operational results.

Report zero-shot detection, matched final transfer advantage, early learning curves, encoder-only transfer and direction-specific effects.

### Continual learning

Let R_{i,j} be AP (or separately AUROC) on immutable task j evaluation graphs after learning phase i. Report per-task trajectories, average seen-task performance, final macro average, current-task adaptation and backward transfer:

BWT = mean_{j<T}(R_{T,j} − R_{j,j}).

Forgetting F_j = R_{j,j} − R_{T,j}; retain negative values as improvement. This end-of-own-phase definition is not peak-to-final forgetting. Report per-task and average values.

### Natural traffic evaluation

Construct natural graphs from all valid flows in held-out physical windows, without using labels to choose membership or build features. Include mixed and benign-only windows. Group predictions afterward by ground truth to report seen and unseen attacks, and overall benign/attack performance. Count each benign flow once in pooled metrics. This evaluates retrospective traffic, not live deployment.

### Resources and uncertainty

Report parameter count, persistent memory in bytes, peak RAM/GPU memory, graph construction, source training, adaptation, method overhead and inference time. Report individual paired-seed results, means, sample standard deviations and suitable uncertainty. Account for temporal dependence; do not treat flows, windows, seeds or relation scores as independent attack-level repetitions. Distinguish screening bootstrap uncertainty from training-run variability. Preserve direction-specific results; use equal target weighting within full orders and equal order weighting if using the archived aggregation.

### Ablations and sensitivity

Compare parent method vs adaptive method and meaningful vs non-guided/shuffled relation scores, under matched conditions. Select scientifically relevant tests of window size (10 vs 5 seconds), task order, seeds, relation masking, estimator data budget, benign allocation, replay capacity or EWC strength. Avoid a full factorial matrix. Keep attention/fusion comparisons optional. Use a separately declared feasible secondary ToN study to test cross-dataset generalization of findings, not necessarily cross-dataset model transfer. Document dataset-specific changes and limitations.

**Output/gate:** Reproducible results, conclusions for each RQ, resource comparisons and limitations.

---

## 3. Experiment families and development tiers

| Family | Purpose |
|---|---|
| E1 | Static flat/host/heterogeneous comparison (RQ1) |
| E2–E4 | Zero-shot, matched target learning and encoder-only transfer (RQ2) |
| E5–E6 | Relation probes, masking and optional selective transfer (RQ3) |
| E7 | Transfer estimator validation (RQ4) |
| E8–E10 | Fine-tuning, EWC, replay and offline joint reference (RQ5 baselines) |
| E11–E12 | One justified adaptive method and its controls (RQ5) |
| E13–E15 | Natural traffic, secondary dataset and resources |

Tier 0: synthetic correctness and hand calculations. Tier 1: UNSW engineering pilot, one task/seed. Tier 2: two-attack end-to-end pilot. Tier 3: primary UNSW investigations with approved paired seeds and directions. Tier 4: frozen confirmatory runs and secondary dataset evaluation. Start small and measure per-update runtime, graph sizes and total cost before scheduling full experiments. The archived matrix (five full orders and five seeds per dataset plus directed pairs) is ambitious; retain scientific controls and reduce optional breadth if needed, documenting amendments before final tests.

Core: UNSW pipeline, static baselines, matched transfer, probes/masking, prototype estimator validation, conventional CL comparison, one adaptive candidate if justified, natural evaluation and resource accounting. Extended: all pairs, large-scale ToN, selective parameter transfer, attention, LogME/gradient alignment, warm-up, DER++, hybrid and learned weighting. ToN remains a planned secondary dataset; report if resource limits prevent it.

## 4. Implementation and verification

The project will be developed incrementally with human approval, not generated as an entire codebase. Before implementing any operation define its purpose, input, assumptions, algorithm/mathematics, alternatives, expected output, manual example, verification and failure handling. Keep a single [operational flow](FLOW.md); add code and artifacts only as required. Preserve the archive. Trace experiments to dataset checksums, code commit, configuration and seeds. Verify and understand each operation before advancing. Passing tests alone is insufficient.

## 5. Completion and expected contributions

Scientific completion means a validated flow/graph pipeline, reliable baselines, controlled attack-transfer measurements, relation contribution analysis, estimator validation, fair conventional CL comparison, evaluation of a justified adaptive candidate if supported, and documented detection, forgetting, efficiency and limitations. Success is not defined as beating all baselines.

Expected (not yet established) contributions: a controlled heterogeneous graph transfer framework for attack categories; empirical characterization of positive/negative directional and relation-level transfer; evaluation of representation-based estimators; resource-aware conventional and relation-guided learning comparisons; evidence of conditions under which relation guidance is useful or ineffective. Novelty and effectiveness require validation; literature alone cannot guarantee either.

## 6. Selected methodological references

- [UQ NetFlow datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) — provenance and feature definitions.
- [Kirkpatrick et al. (2017), Overcoming Catastrophic Forgetting in Neural Networks](https://doi.org/10.1073/pnas.1611835114) — EWC.
- [Schlichtkrull et al. (2018), Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) — relation-specific message passing.
- [Yu et al. (2020), Semantic Drift Compensation for Class-Incremental Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Semantic_Drift_Compensation_for_Class-Incremental_Learning_CVPR_2020_paper.html) — representation drift.
- [Buzzega et al. (2020), Dark Experience Replay](https://arxiv.org/abs/2004.07211) — replay-based comparison.
- [Nguyen et al. (2020), LEEP](https://proceedings.mlr.press/v119/nguyen20b.html) — transferability estimation.
- [You et al. (2021), LogME](https://proceedings.mlr.press/v139/you21b.html) — representation transferability.
- [van de Ven et al. (2022), Three Types of Incremental Learning](https://doi.org/10.1038/s42256-022-00568-3) — continual learning taxonomy.
- [On the Computation of Fisher Information in Continual Learning (2025)](https://iclr-blogposts.github.io/2025/blog/fisher/) — Fisher implementation choices.

References motivate the design; they do not establish the novelty or expected performance of TRENCH-IDS.

## 7. Decision register

**Stable research direction:** domain-incremental binary flow detection, UNSW primary/ToN secondary, graph-based relation transfer, matched comparisons, AP, fine-tuning/EWC/replay comparators, leakage-controlled evaluation.

**Provisional:** archived splits/allocation/window durations, graph node/edge schema, feature policy, update budget, exact experimental scale and resource caps. Amendments require approval and renewed validation where applicable.

**Experimental:** graph benefit, relation contribution, estimator validity, prototype design, warm-up, fusion, EWC vs replay, weighting direction, memory lifecycle and final adaptive mechanism.

Phase 0 remains pending formal protocol review and professor approval; no implementation operation is approved by this document alone.
