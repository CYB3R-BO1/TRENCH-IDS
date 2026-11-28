# Step 1 -- Attack-Class Similarity Analysis (NF-UNSW-NB15-v3)

*Generated; evidence only. All similarity metrics use the TRAINING partition.*

## 0. Scope, provenance and checks

- **Dataset:** NF-UNSW-NB15-v3, 2,365,424 flows, sha256 `4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5`.
- **Global split (all-flow timestamp quantiles):** train < 1424238130000, val < 1424250495000, test >= 1424250495000.
  Similarity uses **train only** (1,655,729 rows); test never influences task-design evidence.
- **Run:** seed 0, window 5000 ms, split (0.7, 0.85),
  floor 1000, reservoir 20000, cat-topk 0.
- **Classes compared:** 7 discovered attack classes.
- **Labels:** 0 -> 2,237,731, 1 -> 127,693; Label/Attack mismatches = 0 (run stops if > 0).
- **Checks:** zero-support attack classes = none;
  invalid categorical cells = 23866; invalid flag cells = 0.
- **Similarity is symmetric by construction; transfer is directional and is NOT tested here.**
- Datasets are analysed separately and same-name labels are not assumed equivalent.

## 1. Attack-class counts and taxonomy

| class | family | flows |
| --- | --- | --- |
| Analysis | Analysis | 1226 |
| Backdoor | Backdoor | 4659 |
| DoS | DoS | 5980 |
| Exploits | Exploits | 42748 |
| Fuzzers | Fuzzers | 33816 |
| Generic | Generic | 19651 |
| Reconnaissance | Reconnaissance | 17074 |
| Shellcode | Shellcode | 2381 |
| Worms | Worms | 158 |

## 2. Standardised mean-profile cosine (train-only, magnitude features)

Uses only MAGNITUDE features (35); identifier codes and
bitmasks are excluded. Zero standardised-deviation profiles: none.

| class | Backdoor | DoS | Exploits | Fuzzers | Generic | Reconnaissance | Shellcode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Backdoor | 1.000 | 0.995 | 0.950 | 0.990 | 0.748 | 0.956 | 0.936 |
| DoS | 0.995 | 1.000 | 0.957 | 0.985 | 0.729 | 0.946 | 0.936 |
| Exploits | 0.950 | 0.957 | 1.000 | 0.923 | 0.517 | 0.826 | 0.807 |
| Fuzzers | 0.990 | 0.985 | 0.923 | 1.000 | 0.773 | 0.972 | 0.956 |
| Generic | 0.748 | 0.729 | 0.517 | 0.773 | 1.000 | 0.892 | 0.889 |
| Reconnaissance | 0.956 | 0.946 | 0.826 | 0.972 | 0.892 | 1.000 | 0.990 |
| Shellcode | 0.936 | 0.936 | 0.807 | 0.956 | 0.889 | 0.990 | 1.000 |

Most similar: Backdoor-DoS (0.995), Backdoor-Fuzzers (0.991), Reconnaissance-Shellcode (0.990), DoS-Fuzzers (0.985).

## 3. Wasserstein distance (valid samples, train-only)

Mean 1-D Wasserstein over 13 continuous features using **valid**
observations only (invalid values are excluded, never zero-filled).

| class | Backdoor | DoS | Exploits | Fuzzers | Generic | Reconnaissance | Shellcode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Backdoor | 0.000 | 0.205 | 0.342 | 0.147 | 1.094 | 0.387 | 0.469 |
| DoS | 0.205 | 0.000 | 0.270 | 0.166 | 1.037 | 0.346 | 0.401 |
| Exploits | 0.342 | 0.270 | 0.000 | 0.302 | 1.280 | 0.579 | 0.640 |
| Fuzzers | 0.147 | 0.166 | 0.302 | 0.000 | 1.037 | 0.315 | 0.375 |
| Generic | 1.094 | 1.037 | 1.280 | 1.037 | 0.000 | 0.773 | 0.704 |
| Reconnaissance | 0.387 | 0.346 | 0.579 | 0.315 | 0.773 | 0.000 | 0.208 |
| Shellcode | 0.469 | 0.401 | 0.640 | 0.375 | 0.704 | 0.208 | 0.000 |

## 4. Jensen-Shannon distance (train-only categorical)

A pair averages only over fields valid for both classes; per-field coverage is in
`js_defined_field_count.csv`. Closest: DoS-Exploits (0.340), Backdoor-DoS (0.362), Backdoor-Fuzzers (0.366), Backdoor-Exploits (0.373).

| class | Backdoor | DoS | Exploits | Fuzzers | Generic | Reconnaissance | Shellcode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Backdoor | 0.000 | 0.362 | 0.373 | 0.366 | 0.604 | 0.411 | 0.480 |
| DoS | 0.362 | 0.000 | 0.340 | 0.419 | 0.596 | 0.450 | 0.496 |
| Exploits | 0.373 | 0.340 | 0.000 | 0.452 | 0.666 | 0.505 | 0.551 |
| Fuzzers | 0.366 | 0.419 | 0.452 | 0.000 | 0.634 | 0.438 | 0.470 |
| Generic | 0.604 | 0.596 | 0.666 | 0.634 | 0.000 | 0.583 | 0.616 |
| Reconnaissance | 0.411 | 0.450 | 0.505 | 0.438 | 0.583 | 0.000 | 0.476 |
| Shellcode | 0.480 | 0.496 | 0.551 | 0.470 | 0.616 | 0.476 | 0.000 |

## 5. Relation-profile similarity (identity-free, train-only)

source_host_degree / dest_host_degree / host_pair_degree / flow_to_src_port / flow_to_dst_port / flow_to_protocol / flow_to_service. These are identity-free structural summaries, **not**
graph topology and **not** learned relation embeddings. Individual block matrices are
the primary evidence (`relation_block_distance.csv`); the average is secondary.

| class | Backdoor | DoS | Exploits | Fuzzers | Generic | Reconnaissance | Shellcode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Backdoor | 1.000 | 0.676 | 0.627 | 0.659 | 0.462 | 0.620 | 0.622 |
| DoS | 0.676 | 1.000 | 0.703 | 0.657 | 0.445 | 0.627 | 0.582 |
| Exploits | 0.627 | 0.703 | 1.000 | 0.648 | 0.396 | 0.614 | 0.521 |
| Fuzzers | 0.659 | 0.657 | 0.648 | 1.000 | 0.396 | 0.614 | 0.579 |
| Generic | 0.462 | 0.445 | 0.396 | 0.396 | 1.000 | 0.417 | 0.454 |
| Reconnaissance | 0.620 | 0.627 | 0.614 | 0.614 | 0.417 | 1.000 | 0.555 |
| Shellcode | 0.622 | 0.582 | 0.521 | 0.579 | 0.454 | 0.555 | 1.000 |

## 6. TCP flag bits

Flag values must be finite, integral and 0..255; invalid flags are excluded from bits
and denominators. Missing (class, flag-column) support: none.

## 7. Feasibility (shared global partition)

Support per global split, plus per-window occupancy (`window_feasibility.csv`).

| class | flows | train_flows | val_flows | test_flows | min_split_flows | attack_floor_ok | benign_ok_train | benign_ok_val | benign_ok_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benign | 2237731 | 1590548 | 323445 | 323738 | 323445 |  | True | True | True |
| Analysis | 1226 | 646 | 286 | 294 | 286 | False | True | True | True |
| Backdoor | 4659 | 3253 | 146 | 1260 | 146 | False | True | True | True |
| DoS | 5980 | 3237 | 1509 | 1234 | 1234 | True | True | True | True |
| Exploits | 42748 | 21414 | 10519 | 10815 | 10519 | True | True | True | True |
| Fuzzers | 33816 | 17597 | 8832 | 7387 | 7387 | True | True | True | True |
| Generic | 19651 | 9241 | 5284 | 5126 | 5126 | True | True | True | True |
| Reconnaissance | 17074 | 8303 | 4221 | 4550 | 4221 | True | True | True | True |
| Shellcode | 2381 | 1407 | 568 | 406 | 406 | False | True | True | True |
| Worms | 158 | 83 | 38 | 37 | 37 | False | True | True | True |

## 8. Pair comparison

| class_a | class_b | cosine | wasserstein | js_distance | relation_similarity | src_host_jaccard | dst_host_jaccard | flag_bits_similarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Backdoor | DoS | 0.9949 | 0.2045 | 0.3623 | 0.6764 | 1.0000 | 1.0000 | 0.9956 |
| Backdoor | Exploits | 0.9504 | 0.3422 | 0.3732 | 0.6267 | 1.0000 | 1.0000 | 0.9853 |
| Backdoor | Fuzzers | 0.9905 | 0.1474 | 0.3661 | 0.6588 | 1.0000 | 1.0000 | 0.9924 |
| Backdoor | Generic | 0.7477 | 1.0936 | 0.6037 | 0.4616 | 1.0000 | 1.0000 | 0.9955 |
| Backdoor | Reconnaissance | 0.9557 | 0.3871 | 0.4108 | 0.6196 | 1.0000 | 1.0000 | 0.9923 |
| Backdoor | Shellcode | 0.9361 | 0.4685 | 0.4804 | 0.6222 | 1.0000 | 1.0000 | 0.9715 |
| DoS | Exploits | 0.9567 | 0.2701 | 0.3401 | 0.7030 | 1.0000 | 1.0000 | 0.9923 |
| DoS | Fuzzers | 0.9853 | 0.1660 | 0.4188 | 0.6570 | 1.0000 | 1.0000 | 0.9820 |
| DoS | Generic | 0.7293 | 1.0366 | 0.5956 | 0.4452 | 1.0000 | 1.0000 | 0.9993 |
| DoS | Reconnaissance | 0.9465 | 0.3462 | 0.4499 | 0.6273 | 1.0000 | 1.0000 | 0.9865 |
| DoS | Shellcode | 0.9360 | 0.4007 | 0.4960 | 0.5819 | 1.0000 | 1.0000 | 0.9733 |
| Exploits | Fuzzers | 0.9231 | 0.3020 | 0.4523 | 0.6481 | 1.0000 | 1.0000 | 0.9578 |
| Exploits | Generic | 0.5174 | 1.2801 | 0.6656 | 0.3958 | 1.0000 | 1.0000 | 0.9878 |
| Exploits | Reconnaissance | 0.8260 | 0.5787 | 0.5049 | 0.6137 | 1.0000 | 1.0000 | 0.9613 |
| Exploits | Shellcode | 0.8067 | 0.6399 | 0.5512 | 0.5207 | 1.0000 | 1.0000 | 0.9439 |
| Fuzzers | Generic | 0.7728 | 1.0372 | 0.6343 | 0.3957 | 1.0000 | 1.0000 | 0.9851 |
| Fuzzers | Reconnaissance | 0.9724 | 0.3150 | 0.4377 | 0.6135 | 1.0000 | 1.0000 | 0.9986 |
| Fuzzers | Shellcode | 0.9556 | 0.3748 | 0.4696 | 0.5788 | 1.0000 | 1.0000 | 0.9813 |
| Generic | Reconnaissance | 0.8916 | 0.7733 | 0.5825 | 0.4166 | 1.0000 | 1.0000 | 0.9899 |
| Generic | Shellcode | 0.8889 | 0.7045 | 0.6159 | 0.4540 | 1.0000 | 1.0000 | 0.9758 |
| Reconnaissance | Shellcode | 0.9902 | 0.2083 | 0.4757 | 0.5553 | 1.0000 | 1.0000 | 0.9876 |
