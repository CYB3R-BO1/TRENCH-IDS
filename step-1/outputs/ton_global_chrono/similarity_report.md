# Step 1 -- Attack-Class Similarity Analysis (NF-ToN-IoT-v3)

*Generated; evidence only. All similarity metrics use the TRAINING partition.*

## 0. Scope, provenance and checks

- **Dataset:** NF-ToN-IoT-v3, 27,520,260 flows, sha256 `53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78`.
- **Global split (all-flow timestamp quantiles):** train < 1556248665000, val < 1556353755000, test >= 1556353755000.
  Similarity uses **train only** (19,261,376 rows); test never influences task-design evidence.
- **Run:** seed 0, window 5000 ms, split (0.7, 0.85),
  floor 1000, reservoir 20000, cat-topk 0.
- **Classes compared:** 4 discovered attack classes.
- **Labels:** 0 -> 16,792,214, 1 -> 10,728,046; Label/Attack mismatches = 0 (run stops if > 0).
- **Checks:** zero-support attack classes = none;
  invalid categorical cells = 0; invalid flag cells = 0.
- **Similarity is symmetric by construction; transfer is directional and is NOT tested here.**
- Datasets are analysed separately and same-name labels are not assumed equivalent.

## 1. Attack-class counts and taxonomy

| class | family | flows |
| --- | --- | --- |
| Backdoor | Backdoor | 203384 |
| ddos | ddos | 4141256 |
| dos | dos | 203456 |
| injection | injection | 381777 |
| mitm | mitm | 6013 |
| password | password | 1594777 |
| ransomware | ransomware | 3971 |
| scanning | scanning | 1358977 |
| xss | xss | 2834435 |

## 2. Standardised mean-profile cosine (train-only, magnitude features)

Uses only MAGNITUDE features (37); identifier codes and
bitmasks are excluded. Zero standardised-deviation profiles: none.

| class | ddos | dos | injection | scanning |
| --- | --- | --- | --- | --- |
| ddos | 1.000 | 0.359 | 0.513 | -0.799 |
| dos | 0.359 | 1.000 | 0.341 | -0.502 |
| injection | 0.513 | 0.341 | 1.000 | -0.695 |
| scanning | -0.799 | -0.502 | -0.695 | 1.000 |

Most similar: ddos-injection (0.513), ddos-dos (0.359), dos-injection (0.341), dos-scanning (-0.502).

## 3. Wasserstein distance (valid samples, train-only)

Mean 1-D Wasserstein over 13 continuous features using **valid**
observations only (invalid values are excluded, never zero-filled).

| class | ddos | dos | injection | scanning |
| --- | --- | --- | --- | --- |
| ddos | 0.000 | 0.858 | 2.076 | 0.622 |
| dos | 0.858 | 0.000 | 2.407 | 0.959 |
| injection | 2.076 | 2.407 | 0.000 | 2.693 |
| scanning | 0.622 | 0.959 | 2.693 | 0.000 |

## 4. Jensen-Shannon distance (train-only categorical)

A pair averages only over fields valid for both classes; per-field coverage is in
`js_defined_field_count.csv`. Closest: ddos-injection (0.230), ddos-scanning (0.372), injection-scanning (0.402), dos-scanning (0.429).

| class | ddos | dos | injection | scanning |
| --- | --- | --- | --- | --- |
| ddos | 0.000 | 0.510 | 0.230 | 0.372 |
| dos | 0.510 | 0.000 | 0.501 | 0.429 |
| injection | 0.230 | 0.501 | 0.000 | 0.402 |
| scanning | 0.372 | 0.429 | 0.402 | 0.000 |

## 5. Relation-profile similarity (identity-free, train-only)

source_host_degree / dest_host_degree / host_pair_degree / flow_to_src_port / flow_to_dst_port / flow_to_protocol / flow_to_service. These are identity-free structural summaries, **not**
graph topology and **not** learned relation embeddings. Individual block matrices are
the primary evidence (`relation_block_distance.csv`); the average is secondary.

| class | ddos | dos | injection | scanning |
| --- | --- | --- | --- | --- |
| ddos | 1.000 | 0.403 | 0.663 | 0.473 |
| dos | 0.403 | 1.000 | 0.417 | 0.445 |
| injection | 0.663 | 0.417 | 1.000 | 0.492 |
| scanning | 0.473 | 0.445 | 0.492 | 1.000 |

## 6. TCP flag bits

Flag values must be finite, integral and 0..255; invalid flags are excluded from bits
and denominators. Missing (class, flag-column) support: Backdoor/TCP_FLAGS, Backdoor/CLIENT_TCP_FLAGS, Backdoor/SERVER_TCP_FLAGS, mitm/TCP_FLAGS, mitm/CLIENT_TCP_FLAGS, mitm/SERVER_TCP_FLAGS, password/TCP_FLAGS, password/CLIENT_TCP_FLAGS, password/SERVER_TCP_FLAGS, ransomware/TCP_FLAGS, ransomware/CLIENT_TCP_FLAGS, ransomware/SERVER_TCP_FLAGS, xss/TCP_FLAGS, xss/CLIENT_TCP_FLAGS, xss/SERVER_TCP_FLAGS.

## 7. Feasibility (shared global partition)

Support per global split, plus per-window occupancy (`window_feasibility.csv`).

| class | flows | train_flows | val_flows | test_flows | min_split_flows | attack_floor_ok | benign_ok_train | benign_ok_val | benign_ok_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benign | 16792214 | 13642780 | 877181 | 2272253 | 877181 |  | True | True | True |
| Backdoor | 203384 | 0 | 0 | 203384 | 0 | False | True | True | True |
| ddos | 4141256 | 3674386 | 466870 | 0 | 0 | False | True | True | True |
| dos | 203456 | 203456 | 0 | 0 | 0 | False | True | True | True |
| injection | 381777 | 381777 | 0 | 0 | 0 | False | True | True | True |
| mitm | 6013 | 0 | 0 | 6013 | 0 | False | True | True | True |
| password | 1594777 | 0 | 1594777 | 0 | 0 | False | True | True | True |
| ransomware | 3971 | 0 | 0 | 3971 | 0 | False | True | True | True |
| scanning | 1358977 | 1358977 | 0 | 0 | 0 | False | True | True | True |
| xss | 2834435 | 0 | 1191930 | 1642505 | 0 | False | True | True | True |

## 8. Pair comparison

| class_a | class_b | cosine | wasserstein | js_distance | relation_similarity | src_host_jaccard | dst_host_jaccard | flag_bits_similarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ddos | dos | 0.3589 | 0.8577 | 0.5097 | 0.4026 | 0.2222 | 0.1119 | 0.7301 |
| ddos | injection | 0.5126 | 2.0764 | 0.2298 | 0.6630 | 0.3333 | 0.2391 | 0.8273 |
| ddos | scanning | -0.7993 | 0.6223 | 0.3717 | 0.4734 | 0.1053 | 0.0368 | 0.7771 |
| dos | injection | 0.3411 | 2.4072 | 0.5008 | 0.4174 | 0.2500 | 0.0963 | 0.5687 |
| dos | scanning | -0.5019 | 0.9586 | 0.4294 | 0.4450 | 0.2632 | 0.0478 | 0.9742 |
| injection | scanning | -0.6952 | 2.6932 | 0.4017 | 0.4922 | 0.1905 | 0.0255 | 0.6127 |
