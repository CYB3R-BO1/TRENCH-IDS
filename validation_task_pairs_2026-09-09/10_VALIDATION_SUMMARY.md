# Task-pair consistency validation — summary

Intended (locked) task configuration = `src/trench_ids/labels.py` CANONICAL_TO_TASK
(locked by `tests/test_labels.py::test_attack_classes_for_task_matches_locked_six_task_table`):
T1 Scanning; T2 Reconnaissance; T3 DDoS+Infiltration; T4 DoS+Injection;
T5 Password+Bot; T6 XSS+BruteForce.

## Files checked (entire file, line by line)
- src/trench_ids/labels.py (231 lines), constants.py (21), similarity.py (246),
  task_design.py (215), preprocess.py (649), vocab.py (100), graphs.py (610),
  model/relation_conv.py (94), model/attention_fusion.py (132),
  model/rhgnn.py (409), model/flat.py (264), cl/device.py (20),
  cl/train.py (830), cl/memory_bank.py (103), cl/transferability.py (109),
  cl/inference.py (204), cl/evaluate.py (312)
- Configs: similarity.yaml, preprocess.yaml, graph.yaml, model.yaml, train.yaml

## Runs (all outputs in this folder; originals untouched)
- 01 labels/constants dump — CPU (conda base) — 01_labels_constants.json
- 02 similarity.yaml vs labels CLASS_DATASETS + 37-feature match graph.yaml — CPU
- 03 task_design assign_groups with/without sizes on repo matrix — CPU —
  with-sizes == locked (True), without == old pairing (XSS+DDoS,
  Password+Infiltration, DoS+Injection, Bot+BruteForce)
- 04 processed parquets per-task classes vs locked — CPU — MATCHES_LOCKED True
- 05 vocab sizes + fingerprint + key coercion — CPU — PROTOCOL 5, L7 239
- 06 graphs .pt per-task classes + 11 relations — CPU — MATCHES_LOCKED True
- 07 model 3-layer residual + flat forward on task-3 graph — GPU (RTX 3050) —
  fused (300,64), 5 flow relations
- 08 checkpoint_task_6 eval per task test — GPU — classes match locked;
  recomputed forgetting/acc == summary.json exactly
- 09 memory bank + transferability new-classes per task — CPU — match locked

## Errors found and fixed
1. attack-similarity-matrix.md:82 — "(T3 vs. T4, T1 vs. T2)" → "(T6 vs. T3, T1 vs. T2)"
   (XSS=T6, Infiltration=T3).
2. attack-class-counts.md:70-71 — "tasks (T3 vs. T4)" → "tasks (T6 vs. T3)". Same cause.
3. attack-class-counts.md:91 — "(Bot + BruteForce, T6)" → "(XSS + BruteForce, T6)".
   Bot+BruteForce never co-occur in locked design.
Reran text search: old patterns gone; labels tests 18/18 pass; full suite 373 passed.

## Notes (not task-pair errors, left as-is)
- archive/analysis_scripts/analyze_transferability.py TASK_CLASSES uses the old
  total-similarity-only pairing; it is archived, not core pipeline.
- attack-similarity-matrix.md threshold-scan table intentionally shows the raw
  (sizes=None) grouping vs the locked size-aware design; note already explains this.
- data/graphs on disk holds 10,402 graphs (~1.1 GB), not the 69,737 / 5.73 GB
  quoted in CLAUDE.md/supplement; row counts (3.12M/300) support ~10.4k, so the
  doc figure looks stale from the pre-quota benchmark. Task pairs unaffected.
- 3-layer param count measured 622,482 (vocab 239) vs supplement 515,058
  (vocab 216 era); same cause (vocab growth), not a pair issue.
