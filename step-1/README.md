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
