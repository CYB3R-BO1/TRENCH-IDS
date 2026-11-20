# Step 1.2 -- Window-size feasibility sweep (NF-UNSW-NB15-v3)

*Descriptive only; no similarity definition or baseline was changed.*

## Boundary policy

- Split cuts are global chronological quantiles [0.7, 0.85] aligned to **60 s**; every window size divides the alignment, so no physical window crosses a partition boundary.
- Seed 0; dataset sha256 `4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5`.
- Graph-flow thresholds: [2, 5, 10, 20, 50].

## Actual split sizes (60 s aligned)

| window | split | windows | flows | % flows |
| --- | --- | --- | --- | --- |
| 1s | train | 59,567 | 1,655,502 | 69.988 |
| 1s | val | 11,739 | 354,595 | 14.991 |
| 1s | test | 11,645 | 355,327 | 15.022 |
| 5s | train | 12,275 | 1,655,502 | 69.988 |
| 5s | val | 2,397 | 354,595 | 14.991 |
| 5s | test | 2,370 | 355,327 | 15.022 |
| 10s | train | 6,197 | 1,655,502 | 69.988 |
| 10s | val | 1,208 | 354,595 | 14.991 |
| 10s | test | 1,192 | 355,327 | 15.022 |
| 30s | train | 2,095 | 1,655,502 | 69.988 |
| 30s | val | 407 | 354,595 | 14.991 |
| 30s | test | 401 | 355,327 | 15.022 |
| 60s | train | 1,059 | 1,655,502 | 69.988 |
| 60s | val | 206 | 354,595 | 14.991 |
| 60s | test | 202 | 355,327 | 15.022 |

## Headline comparison (train)

| window | median flows | p90 | max | %multi-attack | %single-attack | classes w/ >=100 windows & >=5 flows |
| --- | --- | --- | --- | --- | --- | --- |
| 1s | 25.0 | 42.0 | 770 | 21.4% | 17.0% | 3 |
| 5s | 136.0 | 178.0 | 860 | 36.6% | 2.6% | 6 |
| 10s | 274.0 | 342.0 | 1071 | 37.0% | 2.1% | 6 |
| 30s | 826.0 | 975.0 | 1713 | 37.1% | 2.1% | 5 |
| 60s | 1637.0 | 1921.0 | 2945 | 37.6% | 2.3% | 5 |

## Task-filtered feasibility (train, benign + one class)

| window | class | windows with >=5 class flows | >=10 | >=20 | median filtered total | % of attack flows retained |
| --- | --- | --- | --- | --- | --- | --- |
| 1s | Analysis | 0 | 0 | 0 | 24.0 | 0.99% |
| 1s | Backdoor | 204 | 12 | 1 | 24.0 | 4.99% |
| 1s | DoS | 21 | 8 | 0 | 24.0 | 4.96% |
| 1s | Exploits | 674 | 53 | 0 | 24.0 | 32.85% |
| 1s | Fuzzers | 348 | 34 | 0 | 24.0 | 27.00% |
| 1s | Generic | 46 | 1 | 0 | 24.0 | 14.17% |
| 1s | Reconnaissance | 69 | 8 | 0 | 24.0 | 12.74% |
| 1s | Shellcode | 19 | 0 | 0 | 24.0 | 2.16% |
| 1s | Worms | 0 | 0 | 0 | 24.0 | 0.13% |
| 5s | Analysis | 58 | 0 | 0 | 131.0 | 0.99% |
| 5s | Backdoor | 197 | 155 | 55 | 131.0 | 4.99% |
| 5s | DoS | 127 | 33 | 0 | 131.0 | 4.96% |
| 5s | Exploits | 1574 | 554 | 34 | 132.0 | 32.85% |
| 5s | Fuzzers | 1670 | 545 | 56 | 132.0 | 27.00% |
| 5s | Generic | 645 | 102 | 6 | 131.0 | 14.17% |
| 5s | Reconnaissance | 317 | 79 | 14 | 131.0 | 12.74% |
| 5s | Shellcode | 47 | 17 | 5 | 131.0 | 2.16% |
| 5s | Worms | 0 | 0 | 0 | 130.0 | 0.13% |
| 10s | Analysis | 59 | 26 | 0 | 262.0 | 0.99% |
| 10s | Backdoor | 114 | 98 | 80 | 263.0 | 4.99% |
| 10s | DoS | 145 | 59 | 10 | 263.0 | 4.96% |
| 10s | Exploits | 1799 | 851 | 155 | 266.0 | 32.85% |
| 10s | Fuzzers | 1163 | 804 | 239 | 266.0 | 27.00% |
| 10s | Generic | 750 | 291 | 40 | 264.0 | 14.17% |
| 10s | Reconnaissance | 420 | 142 | 30 | 264.0 | 12.74% |
| 10s | Shellcode | 47 | 22 | 8 | 262.0 | 2.16% |
| 10s | Worms | 0 | 0 | 0 | 262.0 | 0.13% |
| 30s | Analysis | 33 | 27 | 14 | 793.0 | 0.99% |
| 30s | Backdoor | 41 | 38 | 35 | 793.0 | 4.99% |
| 30s | DoS | 157 | 72 | 32 | 794.0 | 4.96% |
| 30s | Exploits | 778 | 759 | 510 | 803.0 | 32.85% |
| 30s | Fuzzers | 502 | 461 | 371 | 802.0 | 27.00% |
| 30s | Generic | 399 | 318 | 184 | 798.0 | 14.17% |
| 30s | Reconnaissance | 692 | 245 | 75 | 796.0 | 12.74% |
| 30s | Shellcode | 34 | 25 | 10 | 793.0 | 2.16% |
| 30s | Worms | 0 | 0 | 0 | 792.0 | 0.13% |
| 60s | Analysis | 24 | 21 | 13 | 1584.0 | 0.99% |
| 60s | Backdoor | 22 | 20 | 20 | 1586.0 | 4.99% |
| 60s | DoS | 225 | 71 | 31 | 1587.0 | 4.96% |
| 60s | Exploits | 401 | 393 | 381 | 1603.0 | 32.85% |
| 60s | Fuzzers | 301 | 284 | 252 | 1599.0 | 27.00% |
| 60s | Generic | 296 | 199 | 159 | 1592.0 | 14.17% |
| 60s | Reconnaissance | 383 | 360 | 111 | 1592.0 | 12.74% |
| 60s | Shellcode | 39 | 17 | 13 | 1584.0 | 2.16% |
| 60s | Worms | 0 | 0 | 0 | 1584.0 | 0.13% |

## Notes

- Mixing is measured on natural traffic; task-filtered graphs drop all attack classes outside the designated one (provisional primary policy).
- Unlabeled-context traffic is reported as `other_attack_flows_dropped`; it is a future ablation, not part of the primary experiment.
- Window size is **not** auto-selected. Choose from occupancy, mixing and task-filtered feasibility together.
