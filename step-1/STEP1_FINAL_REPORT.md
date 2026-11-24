# Step 1 final report

**Verdict: APPROVED.** Step 1 is closed from a research-design perspective as of 2026-11-24.
The Step 1 methodology and evidence were reviewed and approved; no further analysis or design
changes are required.

## Closure record

- **Methodology review:** complete. The task-sequence specification, frozen allocation and
  evaluation design, and the reported evidence were reviewed and approved for Step 1 closure.
  The ten-step research proposal remains under review.
- **Code verification:** separate from the methodology decision. Independent verification of
  `task_protocol.py` and the generated evidence was performed as a separate check; it confirms
  the reported results and is recorded under Verification. It is not a condition of the
  research-design approval.
- **Freeze:** the Step 1 specification, protocol, implementation, tests, evidence and this
  report are frozen. SHA-256 hashes are recorded in
  [`FROZEN_STEP1.sha256`](FROZEN_STEP1.sha256). Do not edit these files without explicitly
  reopening Step 1.

## Completed

- Froze single-attack binary flow tasks and separate measurements for zero-shot detection,
  forward-transfer advantage after matched target training, and forgetting. The five training
  seeds are paired across experiment arms.
- Retained all six primary pairs and the secondary ToN injection → xss comparison. The protocol
  specifies both directions of each pair and five unique full orders per dataset.
- Implemented deterministic same-window benign ownership across the entire eligible set.
  Reversal preserves each task's inputs. The corrected ToN 60 s benign-matched normalization
  reference remains frozen for the similarity analysis.
- Verified the 10 s task inventory and the seven relation comparisons. UNSW Exploits/Fuzzers
  contrasts host and port distributions. ToN injection/ddos contrasts protocol and source
  port with host pair. The recomputed ToN canonical blocks reproduce the stored overall
  relation similarity. These are candidate-screening results. [Specification](TRENCH_TASK_SEQUENCE_DESIGN.md)
  [Configuration](step1_protocol.json) [Feasibility](evidence/task_feasibility_10s.csv)
  [Relation distances](evidence/relation_distances.csv)

## Verification

Independent verification was performed separately from the methodology review on 2026-11-24.

- Reviewed `task_protocol.py` against the allocation, split and relation rules in the
  specification. The exact-fraction priority, split-local counters, reversal invariance,
  frozen-cut checks, train-only relation blocks and the stored-aggregate assertion match the
  specified design. No correctness defect was found.
- Reran the documented reproduction command from the raw bags. The 36-row feasibility CSV and
  the 21-row relation-distance CSV reproduced byte-for-byte. Both raw dataset SHA-256 hashes
  matched the frozen sweep manifests. The exported allocation ledgers had no cross-split
  physical windows, reused benign ownership or reversal changes.
- All 81 synthetic tests passed, including seven new protocol tests. The existing CSV smoke
  tests emitted resource warnings when their row-capped reader exited early.
- Counts, medians and attack thresholds matched the frozen sweeps. The ToN graph cuts also
  matched the corrected normalization manifest. UNSW had 17,888 task graphs and 4,608 unique
  benign windows. ToN had 28,811 task graphs and 28,438 unique benign windows. Every
  class/split had both labels.
- Synchronized Step 1 in the Markdown proposal and its Word source and verified the Word
  package integrity and the four updated paragraphs. The proposal remains under review.

## Remaining limitations

- UNSW Generic dependence exceeds the 900 s bootstrap block, and DoS has sparse attack
  supervision. ToN injection has only six 30-minute blocks and four 60-minute blocks.
  ToN Backdoor remains strongly autocorrelated and scanning has only 47 test graphs at 10 s.
- The protocol scores flows in windows selected for attack presence and uses class-dependent
  benign prevalence. It estimates transfer under the frozen task construction. Shared UNSW
  training windows may capture common capture conditions, and ToN task order is synthetic.
- The feasibility run verified primary 10 s support. The 5 s sensitivity and 30 s exploratory
  settings are specified. Model resource requirements, relation effects and transfer metrics
  await downstream experiments.
