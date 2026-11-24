# TRENCH-IDS Step 1 task-sequence specification

Status: approved Step 1 task-sequence specification, frozen 2026-11-24. Step 1 is closed from a
research-design perspective. The ten-step research proposal remains under review.

## 1. Task and label space

Each task contains one raw attack class and its allocated benign traffic. Predict a binary label for
each flow using one shared sigmoid head, with benign coded 0 and the designated attack coded 1.
The class name identifies the data and evaluation slice. Keep it, the ground-truth label, split
membership and task position out of model inputs. Use the same detector and score definition for
every task and checkpoint. The binary head allows scoring a previously unseen attack before it
has appeared in training and measuring retention of earlier classes on their own held-out data.
The experiment estimates attack-versus-benign transfer, with attack-type classification outside
this protocol.

Group flows by the start timestamp into half-open, epoch-aligned windows. Use 10 s graphs for
primary experiments, 5 s for sensitivity and 30 s for exploration. A task graph contains every
designated attack flow in its window plus benign flows if the task owns that window's allocation.
Exclude other attack flows, including their use as unlabeled context. Keep single-flow and
attack-only graphs. Treat host identities as window-local graph keys. Keep graph construction
and fitted statistics isolated by split. The training schedule may carry learned parameters
across tasks as specified below.

## 2. Frozen splits and normalization

- **NF-UNSW-NB15-v3 (primary).** Use the global 70/85-percentile chronological cuts already
  aligned to 60 s in the [graph-sweep manifest](outputs/unsw_window_sweep/split_manifest.json).
  Train is before the first cut, validation is from the first cut through just before the
  second, and test starts at the second. The [similarity baseline](outputs/unsw_full/split_manifest.json)
  used its own frozen 5 s-aligned cuts. Treat those values as screening evidence and the
  60 s-aligned manifest as authoritative for graph training and feasibility.
- **NF-ToN-IoT-v3 (secondary).** Use each raw class's contiguous chronological 70/85-percentile
  cuts from the [corrected manifest](outputs/ton_class_contiguous_benignmatched/split_manifest.json).
  The [graph-sweep cuts](outputs/ton_window_sweep/split_manifest.json) are identical and aligned
  to 60 s. These per-class splits estimate synthetic class-introduction effects.
- Preserve the corrected ToN similarity normalization. At the original 60 s matching scale,
  compute the global mean and standard deviation from attack-train flows and benign flows in
  the union of per-class train windows. Count each eligible benign flow in that reference
  once, including matched supplementary classes as in the existing analysis. The manifest
  records 14,732,002 train benign flows, 920,895 validation and 902,091 test. The 10 s graph
  allocation below is a separate inventory. For model evaluation, use fixed per-flow
  transforms with no dataset-fitted moments, category frequencies, class prototypes or
  vocabularies drawn from future tasks. In particular, keep the offline all-class similarity
  reference out of the model inputs.

## 3. Deterministic benign allocation

Run separately per dataset and graph-window size, over the **entire frozen eligible class set**
before creating any pair, reverse direction or full order. Let the physical window key be
`floor(FLOW_START_MILLISECONDS / window_ms)`. Work from all observed attack classes when checking
physical window split integrity, then restrict ownership candidates to the eligible set.

1. Count the positive flows per eligible class and physical window. For each class and split,
   count its attack-present windows before assigning benign. Verify that each physical window
   belongs to one split and each class/window has a single record. Reject a conflicting split.
2. Visit physical windows in ascending key order. Where benign and one or more eligible attack
   classes co-occur, award all benign flows in that window to exactly one of those classes.
   Use the smallest exact fraction of already awarded benign windows divided by attack-present
   windows for the class in that split. Break ties by the smaller denominator, then by the
   larger designated attack count in that window, then by ascending case-sensitive raw label.
   The counters are independent per split. No sampling or seed is used.
3. Retain each class's attacks in every window in which it occurs. Leave benign-only windows,
   and benign in windows with only excluded attack classes, unallocated. In a shared UNSW
   physical window, the other task graphs still keep their own distinct attack rows. In ToN
   the attack-present windows are class-disjoint. Allocation is fixed across all runs at the
   same resolution. Reversing a pair changes presentation order and preserves graph membership.
4. For the repeated-benign ablation, copy the same-window benign set to every eligible class
   present in that physical window within its split. Retain the primary attack membership,
   chronology, schedules and seeds. This changes UNSW exposure in mixed windows. ToN's disjoint
   class windows make this ablation identical to its primary allocation.

The [10 s feasibility table](evidence/task_feasibility_10s.csv) contains all 36 eligible
class/split rows. It reports usable nonempty graphs, attack thresholds of 5, 10 and 20, median
attack flows over attack-present windows, and assigned benign flows and windows. Thresholds are
descriptive. Every row has both labels across the split. UNSW DoS has a median of one attack
per graph and just one test graph with at least 20 attacks. ToN scanning has 54 validation
and 47 test graphs. Inspect these counts alongside the performance estimates.

## 4. Frozen sequences

| Dataset | Condition | Direction | Evidence |
|---|---|---|---|
| UNSW | Primary high | DoS → Fuzzers | Cosine 0.986, 900 s bootstrap CI [0.949, 0.991], top-five frequency 0.754. Retained as a high-cosine contrast with uncertain rank. |
| UNSW | Primary low | Generic → Exploits | Cosine 0.539. This is a low-cosine contrast with unresolved Generic dependence. |
| UNSW | Primary relation | Exploits → Fuzzers | Small host-degree distances and large source- and destination-port distances. The common Fuzzers target also supports a descriptive comparison with DoS → Fuzzers. |
| ToN | Primary high | injection → password | Canonical cosine 0.936, relation similarity 0.709, flag similarity 0.996. Cosine top-five frequency is 1.00 at 30 min. |
| ToN | Secondary high | injection → xss | Canonical cosine 0.909 and relation similarity 0.699. Provides a high-cosine comparison whose within-window destination-port profile differs from password. |
| ToN | Primary low | dos → Backdoor | Canonical cosine 0.169 and relation similarity 0.316. This tests a low-score contrast on both measures. |
| ToN | Primary relation | injection → ddos | Canonical overall relation similarity 0.664 versus cosine 0.336. Retained for the protocol/source-port versus host-pair contrast documented below. |

Run the reverse of every listed pair as a separate direction. This yields six UNSW and eight
ToN directed pair conditions. Keep `ddos` and `dos` distinct. UNSW Analysis and Worms are
excluded for low train support. UNSW Backdoor and Shellcode and ToN mitm and ransomware are
supplementary and fail the frozen 1,000-attack-flows-per-split eligibility rule.

Run five unique full permutations per dataset from seeds **101, 227, 307, 401, 503**. They are
written explicitly in [step1_protocol.json](step1_protocol.json) and generated by sorting on
the SHA-256 hex digest of the UTF-8 string `seed:raw_label`, with raw label as a tie-break.
There are 120 possible UNSW orders over five classes and 5,040 possible ToN orders over seven.
Use all five training seeds **11, 23, 37, 53, 71** for each order and directed pair. This gives
25 full-order runs per dataset and 30 UNSW and 40 ToN pair runs before matched controls and
ablations. Full sequences preserve within-class chronology and represent synthetic
class introductions. ToN's per-class chronological splits and task order do not define a
single global historical timeline.

## 5. Checkpoints, metrics and matched comparisons

Let `U` be the maximum number of usable train graphs among the frozen eligible tasks at that
dataset and resolution. Use `U` optimizer updates per class phase, one graph per update. At
10 s, `U` is 2,297 for UNSW and 7,018 for ToN. Traverse each task in ascending window order,
cycling from the start after exhaustion. Use the complete current graph and average unweighted
binary cross-entropy over its labeled flows. This budgets update count equally. Distinct
graph counts, per-phase repetitions and flow exposures differ by class and must be logged.
The matched target arms consume exactly the same graph sequence and target examples.

- **Before a new class appears:** evaluate the frozen detector from the preceding task on
  the target test graphs. Compute target average precision (AP), AUROC, and recall and
  false-positive rate (FPR) at the fixed threshold of 0.5. Compare with a seed-matched,
  untrained detector using the same test graphs. Report attack prevalence as the AP no-skill
  reference and 0.5 as the AUROC no-skill reference. Use no target validation, calibration
  or fitting before introduction. Seal test results from tuning and training decisions.
- **After introducing a new class:** compare the continued shared detector against a
  seed-matched target-only detector trained from scratch. Both use the same architecture,
  target training schedule, optimizer initialization, learning-rate schedule, update
  budget, target validation access and evaluation graphs. Carry the encoder and head in
  the continued arm and reset optimizer state at every class boundary. Keep the
  pretraining cost of the continued arm explicit. Record checkpoints at updates
  `ceil(f * U)` for `f = 0, 0.01, 0.10, 0.25, 0.50, 1.0`, deduplicating integer counts.
  Zero belongs to the zero-shot comparison. At positive updates, report paired AP
  differences (continued minus scratch). Prespecify final-update AP advantage as the
  primary endpoint, with earlier checkpoints showing learning speed. At each positive
  checkpoint also report AUROC and fixed-threshold recall/FPR. At `U` report recall and
  realized FPR at a threshold picked separately per arm on target validation to maximize
  recall under 1% empirical FPR. Break ties by lower validation FPR, then higher threshold.
  Include an above-maximum threshold as the no-detections option. Add an encoder-only
  warm-start control using the identical fresh head initialization as its paired scratch
  arm to distinguish encoder reuse from binary-head reuse.
- **Retention and forgetting:** record a class's AP and AUROC on its own immutable test
  graphs immediately after its phase (`U`) and after every subsequent phase. Forgetting
  is end-of-own-phase score minus later score. Positive values indicate degradation.
  Report fixed-threshold recall and FPR on the same graphs and retain negative forgetting
  values as improvements. No replay or target-dependent changes to old test graphs.
- **Tuning and aggregation:** preregister a single architecture and training recipe per dataset
  and resolution before running the comparisons. Use the same recipe across arms and directions,
  with no per-arm tuning, test-based selection or early stopping. Compute AP and AUROC over all held-out flows within
  a task and report prevalence and counts. Report each direction's paired differences across
  the five training seeds as individual values, mean and sample standard deviation. For
  full orders, average equally over target tasks after the first, average training seeds
  within each order, and then weight the five orders equally. For final forgetting in a
  full order, average over all classes except the last, alongside per-class forgetting
  trajectories. Use separate dataset, resolution and experimental-condition summaries.
  Keep the two directions visible, since they condition on different targets and prevalences.

## 6. Relation-specific controls

The [relation-distance table](evidence/relation_distances.csv) uses base-2 Jensen–Shannon
distance on training-only class profiles. Smaller values indicate closer distributions.
The host blocks compare sorted flow shares, while the port, protocol and service blocks
compare empirical value frequencies using validated categorical codes. These measures
screen candidate pairs. Transfer effects require model comparisons.

| Relation | UNSW Exploits / Fuzzers | ToN injection / ddos, canonical | ToN injection / ddos, global split sensitivity |
|---|---:|---:|---:|
| Source host | 0.002488 | 0.323789 | 0.386905 |
| Destination host | 0.055281 | 0.429852 | 0.447639 |
| Host pair | 0.136597 | 0.485957 | 0.543157 |
| Source port | 0.871018 | 0.169184 | 0.155408 |
| Destination port | 0.642789 | 0.452384 | 0.394651 |
| Protocol | 0.189865 | 0.046434 | 0.044791 |
| Service | 0.565390 | 0.445602 | 0.386196 |

UNSW supplies a host-versus-port contrast. Source-host and destination-host distributions are
especially close, while both port distributions and service diverge. ToN supplies a complementary
contrast. Protocol and source port are closest, while host pair is most distant in both split
variants. Destination port and service also diverge in the canonical split. The previously
reported 0.387 ToN value was the source-host block under the global split. The canonical
per-class run stored only the aggregate, so the seven canonical blocks were computed from
the two classes' training rows. Their mean reproduces the stored aggregate similarity of
0.663828 after subtracting from one. The two candidates support distinct probes, so both are
retained. [UNSW stored blocks](outputs/unsw_full/relation_block_distance.csv)
[ToN stored global blocks](outputs/ton_global_chrono/relation_block_distance.csv)
[ToN canonical aggregate](outputs/ton_class_contiguous_benignmatched/similarity_results.json)

For both relation pairs and both directions, run the intact and each of the seven single-relation
mask conditions with the same seeds, task graphs, feature transforms, head initialization and
target budgets. In a mask condition disable the designated relation in source, target and
scratch arms, including direct features that duplicate the masked relation. Distinguish source-
and destination-port roles. Retain all supervised flow nodes. Use the encoder-only warm-start
control to compute final AP advantage for full and masked graphs, then subtract the masked
advantage from the full advantage within seed and direction. This estimates the incremental
contribution of a relation conditional on the remaining relations. Shared information across
relations and changes to model capacity limit mechanistic attribution.

## 7. Temporal uncertainty

| Dataset | Primary bootstrap block | Sensitivity block | Scope and constraint |
|---|---:|---:|---|
| UNSW | 900 s | 300 s | 73 primary blocks. Generic's estimated 1,140 s dependence exceeds the primary block length. |
| ToN | 30 min | 60 min | Resample inside each class's training sequence. injection has 6/4 blocks at the two lengths, and Backdoor raw autocorrelation never met the decay criterion within 180 min. |

These bootstrap blocks estimate uncertainty in screening metrics. Graph training always uses
the graph windows in §1. The frozen analyses use 500 replicates and seed 0, with within-run
train-only normalization. The ToN block sizes trade dependence coverage against the number of
blocks. Block counts are nominal and effective independent counts may be smaller. The seed
summaries for future training runs measure run-to-run variability separately from temporal
uncertainty in the screening evidence.

## 8. Reproduction

From `step-1/`, run `python task_protocol.py --out <new-output-directory> --write-allocations`
to reproduce the 10 s evidence and export each task's window membership and benign ownership.
Add `--recompute-ton-relations` for the complete relation-distance table. The primary run
reads two columns from the UNSW CSV and reuses the canonical ToN window table. The optional
ToN pass reads only the columns needed for the missing canonical relation blocks. Historical
outputs stay in place. See the [Step 1 final report](STEP1_FINAL_REPORT.md) for verification
and remaining limitations.
