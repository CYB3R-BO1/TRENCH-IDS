# Cross-Dataset Report — UNSW vs ToN-IoT (Step 1 outputs)

*Generated from stored Step-1 artifacts only; no new raw-data computation was performed for this pass.*
*Outputs in this directory: `dataset_comparison.csv`, `class_feasibility_comparison.csv`,
`similarity_comparison.csv`, `temporal_comparison.csv`, `relation_comparison.csv`,
`experimental_configuration.json`, `cross_dataset_report.md`, `figures/`.*

## 0. Scope, estimands and comparability caveat (read first)

- The two compared datasets are **NF-UNSW-NB15-v3** (UNSW) and **NF-ToN-IoT-v3** (ToN).
- UNSW evidence comes from the **global chronological split** (all-flow timestamp quantiles
  0.70/0.85, 60 s aligned; train 69.997%, val/test ~15.001% each) — stored in
  `outputs/unsw_full/split_manifest.json`.
- ToN evidence comes from the **per-class contiguous chronological split** with benign
  window-matching and the precedence-policy correction (`outputs/ton_class_contiguous_benignmatched/`)
  — stored in `outputs/ton_class_contiguous_benignmatched/split_manifest.json`.
- These are **different sampling protocols / estimands**. Similarity, feasibility, temporal and
  relation statistics from the two datasets are therefore **within-dataset comparisons**, reported
  descriptively, and are **not pooled or averaged across datasets**. Cross-dataset comparisons of
  matched labels (Backdoor–Backdoor, DoS/dos) are descriptive only and are not treated as
  interchangeable measurements.
- The only labels treated as shared concepts are **Backdoor** and **DoS/dos**. Raw attack labels
  were preserved (UNSW TitleCase vs ToN lowercase), and Reconnaissance/Scanning or ddos/dos were
  **not** silently merged.
- Similarity values discussed here are **model-independent descriptive statistics** computed on
  training data; they must not be presented as if produced under identical sampling protocols.

## 1. Dataset characteristics

From `dataset_comparison.csv` and stored class counts.

| aspect | NF-UNSW-NB15-v3 | NF-ToN-IoT-v3 |
| --- | --- | --- |
| flows (total) | 2,365,424 | 27,520,260 |
| benign flows | 2,237,731 (94.60%) | 16,792,214 (61.02%) |
| attack flows | 127,693 (5.40%) | 10,728,046 (38.98%) |
| temporal coverage | 648.66 h (2015-01-22 → 2015-02-18) | 144.60 h (2019-04-23 → 2019-04-29) |
| attack classes | 9 | 9 |
| train attack flows | 65,181 | 7,453,293 (attributed; benign train 14,732,002) |
| train benign:attack ratio | 24.40 : 1 | 1.98 : 1 |
| windows with >1 attack class (10 s train) | 37.0% multi-attack | 0.0% (disjoint by construction) |

- UNSW is benign-dominated and attack volume is low and concentrated (Exploits 33.5%, Fuzzers 26.5% of
  attacks; Worms 0.12%). Attack presence is common: ~94.6% of attack-bearing windows contain multiple
  attack classes at the 10 s primary window.
- ToN is attack-heavy overall (38.98% attacks) but extreme/class-dependent in benign:attack terms and
  attack classes occupy temporally disjoint windows (0% multi-attack windows by construction).
- The very different coverage (648.66 h vs 144.60 h) and benign:attack profiles are descriptive
  characteristics, not comparability judgements.

## 2. Evaluation protocols

Frozen configuration (from `experimental_configuration.json`).

### 2.1 Window configuration (both datasets)

| setting | value |
| --- | --- |
| primary window | 10 s |
| sensitivity window | 5 s |
| exploratory window | 30 s |
| graph flow thresholds | [2, 5, 10, 20] |

### 2.2 Bootstrap block configuration

| setting | NF-UNSW-NB15-v3 | NF-ToN-IoT-v3 |
| --- | --- | --- |
| primary block | 900 s (15 min) | 30 min (1,800 s) |
| sensitivity block | 300 s | 60 min (3,600 s) |
| exploratory block (UNSW only) | 60 s | — |
| replicates | 500 | 500 |
| resampling | contiguous training-time blocks, all classes together; coarse blocks @900 s = 73 | within each class's own contiguous training sequence; never concatenated/shared |
| ACF lag range | 30 lags | >= 30 (extended to 180) |
| normalisation | training-only gmean/gstd recomputed per replicate; fixed categorical universe and host padding | training-only gmean/gstd recomputed per replicate; within-class resampling |

### 2.3 Why the bootstrap configurations differ

- The bootstrap block length is driven by **dataset-specific temporal dependence** (ACF):
  - UNSW (global split): sustained |ACF| < 0.2 first crossed at lags 1–19 depending on class; block
    lengths in seconds (60/300/900 s) were derived from these lags (min blocks needed: Backdoor 720 s,
    Generic 1140 s, …). Primary conservative block = 900 s (73 coarse blocks).
  - ToN (per-class contiguous split): sustained crossings in 60 s bins ranged from 8 lags (480 s) for
    injection up to 76 lags (4,560 s) for ddos/password, and Backdoor never decayed raw. A rule-based
    suggestion was 76 min (4,560 s); the chosen primary was instead **30 min (1,800 s)** with 60 min
    (3,600 s) as sensitivity, as an operational compromise to keep ≥6 bootstrap blocks per primary
    class.
- Separately, the **estimand differs**: UNSW blocks span one global training sequence; ToN blocks are
  within-class sequences. These are independent reasons the frozen bootstrap configs are not the same.
- The two datasets therefore use different block lengths, different resampling scopes and different
  numbers of blocks; interval widths are not directly pooled across the two datasets.

## 3. Attack-class feasibility

From `class_feasibility_comparison.csv`.

Three feasibility tiers are used consistently for both datasets:

| tier | meaning | NF-UNSW-NB15-v3 | NF-ToN-IoT-v3 |
| --- | --- | --- | --- |
| primary eligible | passes the all-splits / per-class floor (>= 1,000 flows per split) with benign context OK | DoS, Exploits, Fuzzers, Generic, Reconnaissance | Backdoor, ddos, dos, injection, password, scanning, xss |
| train-supported, but fails the three-split floor | has meaningful training support, but some split falls below the floor; treated as supplementary | Backdoor, Shellcode | mitm, ransomware |
| low training support | negligible training support; essentially unusable for supervised task design | Analysis, Worms | — |

- UNSW: 5 classes pass the 1,000-flows-per-split attack floor; Backdoor fails on val (146 flows) and
  Shellcode on test (406), i.e. both are **train-supported but floor-failing** (Backdoor train 3,253;
  Shellcode train 1,407); Analysis (min split 286) and Worms (min split 37) are **low training support**.
- ToN: 7 classes pass the floor; mitm (val 956) and ransomware (test 596) are train-supported but fail the
  floor (treated as supplementary).
- A striking cross-dataset contrast for the shared Backdoor concept: **UNSW Backdoor is supplementary**
  under the global split (train-supported, floor-failing), whereas **ToN Backdoor is primary eligible**
  under the per-class contiguous split.

## 4. Similarity evidence

From `similarity_comparison.csv` (long format, within-dataset pairs).

- NF-UNSW-NB15-v3 (7 classes, 21 pairs): mean cosine 0.895 (range 0.539–0.995); most similar pair
  Backdoor–DoS (0.995), then Backdoor–Fuzzers (0.991), Reconnaissance–Shellcode (0.991).
- NF-ToN-IoT-v3 (9 classes, 36 pairs): mean cosine 0.224 (range −0.897–0.936); tight cluster
  injection–password–xss (cosine 0.91–0.94); **scanning is the outlier** (negative cosine against every
  other class, −0.39…−0.90); ddos–dos much less similar than the UNSW global-split run implied (0.25).

### 4.1 Matched-label pair (Backdoor–DoS) — descriptive, not pooled

| metric | NF-UNSW-NB15-v3 (Backdoor–DoS) | NF-ToN-IoT-v3 (Backdoor–dos) |
| --- | --- | --- |
| cosine | 0.995 | 0.169 |
| relation_similarity | 0.676 | 0.316 |
| flag_bits_similarity | 0.996 | 0.908 |

- The same conceptual pair is near-duplicate in UNSW and clearly different in ToN; the disagreement is
  partly estimand-driven (global vs per-class split) and must not be read causally.

## 5. Temporal reliability

From `temporal_comparison.csv`, ACF reports and bootstrap reports.

- UNSW (primary block 900 s): sustained |ACF| < 0.2 first reached at lags: Backdoor 12, DoS 4, Exploits 2,
  Fuzzers 6, Generic 19, Reconnaissance 2, Shellcode 1 (min block needed: Backdoor 720 s, Generic 1140 s).
  The **900 s primary block therefore covers Backdoor but not Generic** (Generic unresolved within
  primary).
- UNSW bootstrap (500 replicates): mean 95% CI width at primary block: cosine 0.127, Wasserstein 0.365,
  Jensen–Shannon 0.100, relation 0.101, flag bits 0.021. Rank stability at 900 s: only JS and
  relation-profile keep a strong pair (DoS–Exploits = 1.00); cosine Backdoor–DoS top-5 probability 0.65.
- ToN (primary 30 min / sensitivity 60 min): Backdoor never decayed raw (lag-1 0.956); sustained
  crossings (60 s bins): ddos 76, dos 57, injection 8, password 76, scanning 11, xss 19, mitm 10,
  ransomware 11. Suggested conservative block 76 min (4,560 s); chosen 30 min primary / 60 min sensitivity
  (30 min keeps ≥6 bootstrap blocks for every primary class; at 60 min Injection has 4 blocks).
- ToN bootstrap: mean CI width at 30 min: cosine 0.215, Wasserstein 0.444, JS 0.068, relation 0.106,
  flag bits 0.091; at 60 min: cosine 0.255, Wasserstein 0.473, JS 0.055, relation 0.099, flag bits 0.072.

### 5.1 Explicit discussion — UNSW Generic and ToN Injection

- **UNSW Generic (unresolved within primary):** the standardised mean-profile cosine of Generic is low
  against Exploits (0.539) and moderate against Backdoor/DoS (0.74–0.76); its ACF first crossing is at
  19 lags (1,140 s) exceeding the 900 s primary block, so the primary-block uncertainty analysis cannot
  resolve Generic–related pairs within the primary block; rank stability for Generic is weak (mean top-5
  ≈ 0.06).
- **ToN Injection (limited blocks):** Injection is part of the tight injection–password–xss similarity
  cluster and is primary eligible, but at the primary bootstrap block (30 min) it has only **6 effective
  temporal blocks** (the 60-minute sensitivity configuration falls to **4 blocks**). This warrants cautious
  interpretation; note that the below-five-block caution flag in the stored bootstrap configuration
  applies to mitm/ransomware at 30 min and to injection/mitm/ransomware at 60 min. Its attack-present vs
  benign-only graph statistics show large benign-context effects (e.g. dst-port entropy attack-present 0.46
  vs benign-only 0.90; benign:attack graph flows 20 vs 285 in attack-present windows).

## 6. Graph structure

From `relation_comparison.csv` and `relation_window_statistics`.

- Task-filtered per-window relation statistics (designated class present; 10 s; train):
  - UNSW: median dst-port entropy 0 and median host-pair top-share 1.0 for most classes; Exploits is
    the exception (median entropy 0.84, top share 0.38, lowest CV ~0.29).
  - ToN: attack-only dst-port entropy medians: injection 0.135, password 0.000, xss 0.921 (cluster
    members differ sharply at window level).
- ToN attack-present vs benign-only (population A vs B) deltas: injection dst-port entropy
  median_A 0.46 vs median_B 0.90 (Δ −0.44); password 0.13 vs 0.70 (Δ −0.57); xss 0.26 vs 0.94 (Δ −0.68);
  graph flows injection 285 vs 20 (Δ 265).

## Findings

### Finding A — Different temporal structures / co-occurrence

UNSW shows substantial attack co-occurrence (37.0% of train windows at 10 s contain multiple attack
classes; ~94.6% of attack-bearing windows are multi-class). ToN has disjoint attack windows (0%
multi-attack) by construction. Temporal reliability estimates therefore apply to different empirical
questions.

### Finding B — Dataset-dependent similarity

Similarity is not transferable as an absolute: the Backdoor–DoS cosine is 0.995 in UNSW but 0.169 for
ToN Backdoor–dos; the injection–password–xss cluster exists only in ToN. Mean cosine differs strongly
across datasets (UNSW 0.895 vs ToN 0.224).

The mean-cosine difference is **descriptive only**. The two datasets have different supported class
populations, normalization scopes and splitting protocols, so the comparison must not be read as
demonstrating an intrinsic property of the datasets. It identifies that the *measured similarity
structure* differs, not that the *attack concepts* differ.

### Finding C — Aggregate vs per-window structure for Injection–Password–XSS

The aggregate similarity cluster (injection–password–xss cosine 0.91–0.94) is not contradicted by
per-window evidence; rather, the cluster aggregates heterogeneous window contents. At the attack-only
10 s level, dst-port entropy medians are injection 0.135, password 0.000, xss 0.921; ToN bootstrap top-5
similarity probability for injection–password–xss pairs is 1.00 at 30 min.

### Finding D — Different uncertainty requirements

- UNSW Generic: dependence (min block 1,140 s) exceeds the primary block (900 s) → unresolved at primary.
- ToN Injection: limited bootstrap blocks (6 at 30 min primary; 4 at 60 min sensitivity) → cautious intervals.

### Finding E — Different task feasibility

UNSW primary-eligible set vs ToN primary-eligible set differ; shared Backdoor concept has different
feasibility across datasets.

## Implications for TRENCH-IDS task construction

*This section is scoping only; task sequences are not finalized in this pass.*

- **Primary eligible classes:** NF-ToN-IoT-v3: Backdoor, ddos, dos, injection, password, scanning, xss.
  NF-UNSW-NB15-v3: DoS, Exploits, Fuzzers, Generic, Reconnaissance. (Shared-concept Backdoor is primary
  eligible in ToN but low support in UNSW; DoS is primary eligible in both.)
- **Supplementary low-support classes:** NF-ToN-IoT-v3: mitm, ransomware; NF-UNSW-NB15-v3: Backdoor,
  Shellcode (train-supported but failing the floor). NF-UNSW-NB15-v3: Analysis and Worms are low
  training support.
- **Cross-class similarity relationships:** UNSW — Backdoor–DoS/Fuzzers, Reconnaissance–Shellcode cluster;
  ToN — injection–password–xss cluster, scanning outlier, ddos–dos low.
- **Relation-specific differences:** Backdoor–DoS relation similarity 0.68 (UNSW) vs 0.32 (ToN); flag-bit
  similarity 0.996 vs 0.908.
- **Limitations:** different split estimands; benign-handling correction; ToN A/B only; UNSW attack-present
  only; single-seed bootstrap caveats; window-size differences.

## Limitations

- No task sequences finalized here.
- Bootstrap interval estimates across datasets not pooled (different block configs/estimands).
- ToN uncertainty for supplementary classes (<5 blocks) is cautious.

## Deliverables

- `dataset_comparison.csv`, `class_feasibility_comparison.csv`, `similarity_comparison.csv`,
  `temporal_comparison.csv`, `relation_comparison.csv`, `experimental_configuration.json`,
  `cross_dataset_report.md`, `figures/*.png`.
