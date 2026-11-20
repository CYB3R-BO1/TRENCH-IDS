# Step 1 — Attack-Class Similarity Analysis

Evidence for characterising attack classes, computed from a **NetFlow-v3** dataset:
**NF-UNSW-NB15-v3** or **NF-ToN-IoT-v3** (select with `--dataset`). Attack classes are
**discovered from the data's `Attack` column** — no labels are hardcoded.

> **Similarity is not transfer.** These matrices characterise how close two classes look.
> Whether a representation learned on one helps another is directional and is measured
> later with a trained model; nothing here tests transfer.

> **Status:** full **NF-UNSW-NB15-v3** run completed and validated. The baseline is frozen
> at `outputs/unsw_full/` (dataset sha256 `4ebb97bd…68f9b5`, 2,365,424 flows, seed 0).
> Do not overwrite this directory; new experiments write to separate output directories.

## UNSW baseline (2026-11-20)

- **Similarity-supported (7):** Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode.
- **Low training support / excluded from matrices:** Analysis (646 train flows), Worms (83).
- **Meet the three-split floor (≥1,000 in every split, 5):** DoS, Exploits, Fuzzers, Generic,
  Reconnaissance. Backdoor (val 146) and Shellcode (test 406) are similarity-supported but
  fail the all-split task floor.
- **Global split:** train 1,655,729 (69.997%), val 354,848 (15.001%), test 354,847 (15.001%),
  cuts aligned to 5 s windows.
- **Observed caveats:** `SRC/DST_TO_SRC_SECOND_BYTES` have large invalid (divide-by-zero)
  shares for Generic/Recon/Shellcode; `L7_PROTO` has fractional encodings; 5 s windows are
  heavily mixed across attack classes.

## Feature-quality investigations (Step 1.2)

Run via `python diagnostics.py ...`; evidence in `outputs/diagnostics/`.

- **Rate-feature sensitivity (cosine ablation, 37 -> 35 magnitude features).** Dropping
  `SRC_TO_DST_SECOND_BYTES` and `DST_TO_SRC_SECOND_BYTES` barely moves the cosine:
  mean |delta| 0.006, max |delta| 0.022 (Exploits-Generic), Spearman rho 0.999,
  Kendall tau 0.991, top-5 pairs unchanged, Generic still clearly dissimilar (mean
  0.765 -> 0.758). The cosine findings do **not** depend on the two high-invalidity
  rate features. Ablation run: `outputs/unsw_rate_ablation/`.
- **Fractional `L7_PROTO`.** 23,866 rows (1.009%) are non-integral across 10 distinct
  values (`7.37` = 18,765; `10.16` = 4,877; rest are near-integer variants), present in
  benign and attack traffic (Fuzzers 11.7% of its rows, Backdoor 5.9%, DoS 1.7%).
  They are **not** correlated with other anomalies (share of `OUT_BYTES == 0` is 0.003
  among fractional rows vs 0.028 overall). Repeated, consistent, uncorrelated values are
  consistent with legitimate encoder output (likely upstream-averaged protocol ids), not
  random corruption.
- **Categorical-validation policy (proposed, not yet applied):** never round. Treat each
  distinct fractional `L7_PROTO` value as a valid exact categorical category (keyed by its
  exact value/string) rather than invalid. This also removes the current Fuzzers-specific
  truncation (~12% of its `L7_PROTO` mass is dropped before JS). Until approved, fractional
  values remain excluded and reported in `invalid_categorical_values.csv`.

## Window-size feasibility sweep (Step 1.2)

Descriptive sweep over 1/5/10/30/60 s with **60 s-aligned split cuts** (run:
`outputs/unsw_window_sweep/`; splits 69.99% / 14.99% / 15.02%). Every window size divides the
alignment, so no physical window crosses a partition boundary.

| window | median flows/window (train) | % attack windows multi-class | train classes with >=100 windows & >=5 class flows |
| --- | --- | --- | --- |
| 1 s | 25 | 44% single-class (least mixing) | 3 |
| 5 s | 136 | 93% multi-class | 6 |
| 10 s | 274 | 95% multi-class | 6 |
| 30 s | 826 | 95% multi-class | 5 |
| 60 s | 1,637 | 94% multi-class | 5 |

- **Trade-off is explicit.** 1 s minimises mixing but is sparse (median 25 flows/window, only
  3 classes have 100+ windows with >=5 class flows). 5-10 s keeps 6 classes feasible with real
  graph size, at the cost of 93-95% of attack-bearing windows being multi-class.
- **Train vs val/test asymmetry (important).** Under the chronological quantile split, ~60% of
  *train* windows contain no attack at all, whereas val/test windows are 92-97% multi-attack.
  Mixing is a property of the split, not only the window size.
- **Task-filtered policy (provisional).** Graphs keep benign + one designated class only;
  retained attack-flow share per class ranges from 0.99% (Analysis) to 32.85% (Exploits).
  Unlabeled-context traffic is reported (`other_attack_flows_dropped`) but deferred.
- **Window size is not auto-selected.** It must be chosen from occupancy, mixing and
  task-filtered feasibility together (see `window_sweep_report.md` for the full evidence).
- **5 s vs 10 s (task-filtered, benign + one class).** At 10 s the median designated-class
  flows per graph roughly double for the high-volume classes (Exploits 3 -> 7, Fuzzers
  6 -> 11, Generic 2 -> 4, Reconnaissance 2 -> 3), and strong-supervision windows rise
  sharply (train windows with >=20 class flows: Exploits 34 -> 155, Fuzzers 56 -> 239,
  Generic 6 -> 40). The cost is ~half the graph windows (12,275 -> 6,197 train) and lower
  absolute window counts for bursty low-volume classes at the >=5 threshold (Backdoor
  197 -> 114; Fuzzers 1670 -> 1163). Worms has no usable graph at either size.
  Evidence: `window_5s_vs_10s_comparison.csv`.

## Frozen window configuration (Step 1.3)

- **Primary: 10 s** graph windows (main TRENCH-IDS configuration). **5 s** = sensitivity
  (finer resolution), **30 s** = exploratory (aggregation / heterogeneity). Kept identical
  across all attack classes; Backdoor's 5 s behaviour is a recorded class-specific finding.
- Wording: 10 s yields **larger graphs, not more samples** (6,197 vs 12,275 train windows);
  the reduced sample count is examined by the bootstrap.

## Step 1.3 - uncertainty and temporal stability

Run via `python step1_3_uncertainty.py {bootstrap|relation-windows}`; outputs in
`outputs/unsw_uncertainty/`.

### Temporal block bootstrap (`bootstrap/`)

- Contiguous training-time blocks; all classes (incl. benign) move together; the same blocks
  are used for every pair; **gmean/gstd recomputed per replicate**; categorical universe and
  host padding fixed to full-train values (padding length does not affect JS).
- Primary: **900 s blocks, 500 replicates** (only **73 coarse blocks** - a real limit on
  effective temporal information); sensitivity at 300 s and 60 s (500 replicates each).
- **Block-length sensitivity (mean 95% CI width):** 60 s < 300 s < 900 s for every metric
  (cosine 0.078 / 0.118 / 0.127; wasserstein 0.226 / 0.339 / 0.365; JS 0.054 / 0.079 / 0.100;
  relation 0.062 / 0.083 / 0.101; flags 0.018 / 0.018 / 0.021).
- **Temporal dependence (60 s-block ACF, extended to 30 lags; first lag with |r| < 0.2):**
  Backdoor **12** lags (~12 min), Generic **19** lags (~19 min), Fuzzers 6, DoS 4, Exploits 2,
  Reconnaissance 2, Shellcode 1. So 900 s covers Backdoor but **not Generic**, whose dependence
  exceeds the primary block - the honest residual limitation.
- **Rank stability (prob. in top-5):** with 900 s blocks only **JS and relation_profile** keep a
  rock-solid pair (**DoS-Exploits, 1.00** in both). Everything else softens versus 300 s/60 s:
  cosine Backdoor-DoS 0.65 (900 s) vs 0.75 (300 s) vs 0.96 (60 s); JS Backdoor-DoS 0.84 vs 0.96
  vs 1.00. Wasserstein remains the noisiest (Backdoor-Fuzzers observed 0.147, 95% CI
  [0.147, 0.440]). Narrower short-block intervals were over-confident, not better.
- Three configurations are kept separate and **must not be merged into one interval**:
  900 s = primary conservative estimate; 300 s = sensitivity; 60 s = exploratory.
- Bootstrap output is reliability evidence only; it does **not** drive task construction.

### Per-window relation statistics (`temporal_relations/`)

Task-filtered (designated class's flows per window; windows without the class excluded),
at 10 s and 5 s, with distributions and CVs rather than a single aggregate.

- Per-window connectivity is strongly concentrated for almost every class: median
  `dst_port_entropy` 0 and median `host_pair_top_share` 1.0 (all flows in a window go to one
  host pair).
- **Exploits is the clear exception** - median entropy 0.84, top host-pair share 0.38, and the
  lowest temporal CV (0.29): diverse *and* stable across windows.
- **Backdoor differs by window size** (the anticipated 5 s finding): median `dst_port_entropy`
  0.0 at 10 s but 0.79 at 5 s, top host-pair share 1.0 vs 0.46 - the coarser window merges
  temporally separate contacts into one dominant pair.
- **Worms** (CV ~9.1) and **Shellcode** (CV ~2.1-6.0) are the most temporally variable; they
  are also the weakest-supported classes.

## Step 1.4 - independent NF-ToN-IoT-v3 analysis

Two ToN runs, kept separate; the UNSW 10 s/900 s configuration is **not** transferred.

### Reference: global chronological split (`outputs/ton_global_chrono/`, also `ton_full/`)

Run: `python attack_class_similarity.py --dataset ton --out outputs/ton_full --seed 0`
(27,520,260 flows, sha256 `53ec8f46...`; split 69.99/15.01/15.00).

- Only **4 of 9 classes are train-supported** (ddos, dos, injection, scanning); the other five
  have zero training flows, and the classes are **nearly temporally disjoint** across the split
  (dos/injection/scanning train-only; ddos train+val; xss val+test; password val-only;
  Backdoor/mitm/ransomware test-only). Useful for studying temporal emergence, unusable for a
  conventional 9-class supervised / continual-learning experiment.

### Primary: per-class contiguous chronological split (`outputs/ton_class_contiguous/`)

Run: `python ton_class_contiguous.py --dataset ton --out outputs/ton_class_contiguous --seed 0`.
Each attack class gets its own 70/15/15 cut points (60 s aligned), so every class keeps a
temporal train -> val -> test direction. Safeguards: (A) 60 s-aligned cuts, so no physical
window is split for a class; (B) benign matched per window with reuse audited; (C) minimum
support/feasibility kept; (D) per-class coverage + first-occurrence ordering reported.

- **All 9 classes now have train/val/test.** 7 meet the 3-split floor (>= 1000 in every split);
  **mitm** (val 956) and **ransomware** (596) are train-supported but fail the all-split floor.
- Classes run in sequence (first occurrence: scanning -> dos -> injection -> ddos -> password ->
  xss -> ransomware -> Backdoor -> mitm, 2019-04-23 .. 04-29); per-class spans 2-26 h.
- **No shared-normal exposure:** because the classes are temporally disjoint, **0 windows are
  train for more than one class**. (Benign val/test counts are small under the pooling
  precedence policy; per-class window-matched benign for task construction is still a follow-up.)
- **Similarity (9x9).** A tight, metric-consistent cluster emerges: **injection-password-xss**
  (cosine 0.91-0.94; JS 0.21-0.37; relation 0.58-0.71; flags 0.996-0.998). **scanning** is the
  outlier (cosine negative with every class, -0.38 .. -0.89). ddos/dos are much less similar
  than the global-split run implied (cosine 0.25).
- **Estimand caveat:** this is not a globally forward-in-time deployment split (one class's
  train period may follow another's test period). Do **not** compare its numbers directly with
  the UNSW global-chronological run.

### Benign partition audit and normalisation-scope correction

Read-only audit (`ton_benign_audit.py`) -> `benign_task_matching.csv`,
`benign_partition_overlap_audit.csv`, `task_split_temporal_audit.csv`,
`normalization_scope.json`, `benign_partition_audit.md` (in `ton_class_contiguous/`).

- **Partition isolation holds.** Classes are temporally disjoint, so **0 windows are used by more
  than one class** and **0 windows have conflicting train/val/test assignments**:
  B_train n B_val = B_train n B_test = B_val n B_test = empty.
- **Per-class matched benign varies enormously** (`benign_task_matching.csv`): e.g. mitm train
  1,755,629 benign for only 4,140 attack flows, scanning 8,327,687 / 905,605, versus injection
  67,149 / 266,120 and xss 283,026 / 1,977,968. Benign:attack imbalance is class-dependent.
- **The precedence policy was wrong.** It put **16,772,031** benign flows into "train" -
  including **2,040,029** that are validation/test for some class - so val/test benign leaked
  into the global mean/std.
- **Correction applied** (new outputs, current run preserved): benign is matched by window
  membership, so the global mean/std uses benign from per-class TRAIN windows only ->
  **train 14,732,002 / val 920,895 / test 902,091**, with 0 val/test benign in training stats.
  See `outputs/ton_class_contiguous_benignmatched/`.
- **Impact:** relation and JS similarities are unchanged (they never used the global benign);
  cosine changes by mean |delta| 0.0068 / max 0.0155. Conclusions are unchanged
  (injection-password-xss cluster; scanning the outlier). Per-class means and the similarity
  inputs were always train-only.

### Window-size feasibility sweep (`outputs/ton_window_sweep/`)

Run: `python ton_window_sweep.py --dataset ton --out outputs/ton_window_sweep --seed 0`
on the canonical benign-matched split; sizes 1/5/10/30/60 s, 60 s-aligned class cuts so all
sizes stay isolated. Three views: natural, task-filtered, unlabeled context. No benign
downsampling.

- **Mixing is 0 by construction.** ToN attack classes occupy disjoint time windows, so no window
  contains two attack classes - a fundamental difference from UNSW. Unlabeled context is ~0.
- **Occupancy is very large.** Median flows/window (train): 27 (1 s), 110 (5 s), 202 (10 s),
  514 (30 s), 943 (60 s); p90 up to 12,752 at 60 s. 1 s train has 179,461 windows (~3.95M flows
  sit in windows containing no class and are not attributed to a split).
- **Task-filtered feasibility is class-dependent** (train windows with >= 5 class flows):
  ddos 27,798 (1 s) - 553 (60 s); password 36,394 - 648; xss 19,446 - 334; injection 5,452 - 141;
  dos 14,348 - 368; scanning 5,452 - 425; Backdoor 2,583 - 1,176 but only **3** windows with >= 10
  at 1 s. mitm (182 - 64) and **ransomware (100 - 51) are supplementary**, not forced.
- **Benign:attack imbalance is extreme and class-dependent** (median, train): scanning 42 (1 s)
  -> 228 (60 s); password/injection/xss 0.1-0.4 (attack-heavy); ddos ~1.4-2.2; Backdoor 0.5-1.3.
- **The injection-password-xss cluster is feasible at every resolution** (all three have
  thousands of >= 5 windows), so it can be tested across window sizes.

### Still to do for ToN

Per-class temporal autocorrelation, then a ToN-specific bootstrap block length, then per-window
relation distributions and uncertainty - all computed independently (no UNSW 10 s/900 s transfer).


## Design decisions
- **Evidence only.** No task groupings, no task sequences, no model. (Per the agreed plan.)
- **One global chronological split** from all-flow `FLOW_START` quantiles
  (`--split 0.70 0.85`). Every similarity input uses the **training partition only**, so
  the test period cannot influence task-design evidence.
- **Feature roles** (`feature_schema.py`): only MAGNITUDE (continuous/count) features feed
  cosine/Wasserstein; identifier-like codes (`ICMP_*`, `DNS_QUERY_*`, `FTP_COMMAND_RET_CODE`)
  are excluded; TCP flags are analysed as bits; ports/protocol/service as categories.
- **Invalid values are excluded, never zero-filled**, and reported per class/feature.
- No cross-dataset pooling; same-name labels are not assumed equivalent.

## What this produces

| # | Deliverable | File(s) | Read it as |
|---|---|---|---|
| 1 | Counts + taxonomy | `attack_class_counts.csv`, `taxonomy.csv` | support and label meaning |
| 2 | Standardised mean-profile cosine | `standardized_mean_profile_cosine.csv` | higher = more similar (train-only) |
| 3 | Wasserstein distance | `wasserstein_distance.csv` | lower = more similar (valid samples) |
| 4 | Jensen-Shannon distance | `jensen_shannon_distance.csv` (+ `_per_field.csv`) | lower = more similar |
| 5 | Relation-profile similarity | `relation_profile_similarity.csv` (+ `relation_block_distance.csv`) | higher = more similar (identity-free) |
| 6 | Feasibility | `sample_feasibility.csv`, `window_feasibility.csv`, `window_occupancy.csv` | who can form a task |

Also: `feature_schema.json`, `provenance.json` (sha256), `split_manifest.json`,
`feature_validity.csv`, `invalid_categorical_values.csv`, `invalid_flag_values.csv`,
`flag_support_gaps.csv`, `tcp_flag_bit_prevalence.csv`, `tcp_flag_valid_counts.csv`,
`tcp_flag_bits_similarity.csv`, `host_identity_overlap.csv`, `cosine_feature_support.csv`,
`cosine_pair_coverage.csv`, `cosine_consistent_features.json`, `wasserstein_per_feature.csv`,
`wasserstein_feature_support.csv`, `wasserstein_pair_coverage.csv`, `class_scope.csv`,
`undefined_pairs.csv`, JS coverage tables, `categorical_compression.csv`,
`categorical_universe.json`, per-metric `rankings_*.csv`, `pair_comparison.csv`, heatmaps in
`figures/`, and `similarity_report.md`.

The matrices are reported **side by side** — never collapsed into one weighted score.

## How to run

```bash
pip install -r ../requirements.txt
python attack_class_similarity.py                          # default dataset (unsw)
python attack_class_similarity.py --dataset ton            # NF-ToN-IoT v3
python attack_class_similarity.py --data <csv> --out <dir>  # any NetFlow-v3 CSV
python -m unittest discover -s tests                       # synthetic tests, no scan
```

Key flags: `--dataset {unsw,ton}`, `--data`, `--taxonomy-csv`, `--split`, `--split-floor`,
`--min-train-flows`, `--cat-topk` (0 = full support), `--min-feature-samples`,
`--window-ms`, `--window-sweep-ms`, `--min-graph-flows`, `--reservoir-cap`, `--seed`,
`--max-rows` (smoke cap, honoured by every pass). Similarity is always train-only.

**Schema audit (sample):** both NF-UNSW-NB15-v3 and NF-ToN-IoT-v3 have 55 columns, all
required fields present, `Label ∈ {0,1}`, integral ordered timestamps, and non-blank hosts.

## Method

- **Standardised mean-profile cosine** — magnitude features, `log1p`, global train mean/std;
  cosine on a **consistent feature set** (features with enough valid observations in every
  class), so pair values are comparable; pair coverage is still reported. Zero-vector policy:
  two zero profiles are identical, zero vs non-zero is 0 (reported).
- **Wasserstein distance** — mean 1-D distance over 13 continuous features **standardised
  with train mean/std**, using valid reservoir samples only; per-feature matrices, feature
  support and per-pair coverage are saved, so unequal feature scales cannot dominate.
- **Jensen-Shannon distance** — categorical fields with strict validation (finite, integral,
  in-range; no rounding); `TCP_FLAGS` reuses the flag validator. A pair averages over
  jointly-valid fields; coverage is saved.
- **Relation-profile similarity** — identity-free sorted flow-share distributions for
  `source_host_degree`, `dest_host_degree`, `host_pair_degree`, `flow_to_src/dst_port`,
  `flow_to_protocol`, `flow_to_service`. These are structural summaries, **not** graph
  topology or learned relation embeddings; individual blocks are the primary evidence.
- **TCP flag bits** — per-bit prevalence with valid denominators; invalid flags excluded.
- **Feasibility** — per global split: flow counts, per-split host cardinality, attack-floor
  and benign-support flags; per-window occupancy and class mixing (`window_feasibility.csv`).

## Tests

Synthetic only, no dataset read: reservoir sampling, strict categorical validation,
probability vectors, identity-free relation profiles, episode/interval counting, flag
validity and denominators, Label/Attack checks, zero-support and JS-coverage handling,
zero-vector cosine, defined-cosine, continuous invalid masks, timestamp integrity (2^53),
host-identity integrity, data-driven class discovery, feature-schema partition, global
split cuts, window feasibility, and a pipeline/output-writer smoke test.

Similarity matrices are restricted to **train-supported** classes (training flows ≥
`--split-floor`). Low/zero-training-support classes remain in `sample_feasibility.csv` and
are listed with reasons in `class_scope.csv` / `undefined_pairs.csv`; they are never silently
dropped from the report.

```bash
python -m unittest discover -s tests -v     # 57 tests
```

## Not yet implemented (next phase)

- Bootstrap confidence intervals for pairwise metrics (`--bootstrap`).
- Per-window distributions of relation statistics (beyond the whole-split block distances).

## Files

```
step-1/
  attack_class_similarity.py   CLI / orchestration / metrics / report
  feature_schema.py            feature roles (MAGNITUDE / IDENTIFIER / BITMASK / CATEGORICAL)
  windows.py                   per-window feasibility
  similarity_report.md         generated narrative (after a run)
  README.md
  tests/test_attack_class_similarity.py
  outputs/                     (populated by a run; see "What this produces")
```
