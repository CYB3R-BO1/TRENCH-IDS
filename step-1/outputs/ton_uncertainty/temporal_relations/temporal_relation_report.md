# Step 1.5 - ToN per-window relation statistics (task-filtered)

- Canonical benign-matched split; windows 10s, 5s.
- Graph = benign + designated class; other attack classes excluded.
- Population A = class present; B = benign-only windows in the same class interval.
- Data sha256 `53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78`.

## Window counts

| window | split | class | A | B |
| --- | --- | --- | --- | --- |
| 10s | test | Backdoor | 1144 | 0 |
| 10s | test | ddos | 731 | 0 |
| 10s | test | dos | 447 | 0 |
| 10s | test | injection | 158 | 0 |
| 10s | test | mitm | 41 | 86 |
| 10s | test | password | 545 | 0 |
| 10s | test | ransomware | 56 | 271 |
| 10s | test | scanning | 47 | 7 |
| 10s | test | xss | 402 | 0 |
| 10s | train | Backdoor | 7018 | 145 |
| 10s | train | ddos | 3145 | 0 |
| 10s | train | dos | 2204 | 0 |
| 10s | train | injection | 791 | 93 |
| 10s | train | mitm | 365 | 162 |
| 10s | train | password | 3865 | 18 |
| 10s | train | ransomware | 233 | 194 |
| 10s | train | scanning | 2764 | 2451 |
| 10s | train | xss | 1991 | 8 |
| 10s | val | Backdoor | 1145 | 1 |
| 10s | val | ddos | 610 | 0 |
| 10s | val | dos | 450 | 0 |
| 10s | val | injection | 292 | 8 |
| 10s | val | mitm | 28 | 54 |
| 10s | val | password | 618 | 0 |
| 10s | val | ransomware | 42 | 18 |
| 10s | val | scanning | 54 | 0 |
| 10s | val | xss | 390 | 0 |
| 5s | test | Backdoor | 2287 | 0 |
| 5s | test | ddos | 1456 | 0 |
| 5s | test | dos | 893 | 0 |
| 5s | test | injection | 315 | 0 |
| 5s | test | mitm | 49 | 174 |
| 5s | test | password | 1089 | 0 |
| 5s | test | ransomware | 78 | 572 |
| 5s | test | scanning | 91 | 11 |
| 5s | test | xss | 803 | 0 |
| 5s | train | Backdoor | 14007 | 318 |
| 5s | train | ddos | 6227 | 0 |
| 5s | train | dos | 4406 | 0 |
| 5s | train | injection | 1466 | 302 |
| 5s | train | mitm | 428 | 607 |
| 5s | train | password | 7720 | 43 |
| 5s | train | ransomware | 347 | 504 |
| 5s | train | scanning | 4548 | 5881 |
| 5s | train | xss | 3965 | 28 |
| 5s | val | Backdoor | 2290 | 1 |
| 5s | val | ddos | 1213 | 0 |
| 5s | val | dos | 900 | 0 |
| 5s | val | injection | 574 | 26 |
| 5s | val | mitm | 39 | 104 |
| 5s | val | password | 1236 | 0 |
| 5s | val | ransomware | 60 | 60 |
| 5s | val | scanning | 103 | 0 |
| 5s | val | xss | 780 | 0 |

## 5 s vs 10 s, Backdoor / Injection / Password / XSS (train, population A, graph)

| class | window | dst-port entropy | pair top-share | src-port distinct | service entropy |
| --- | --- | --- | --- | --- | --- |
| Backdoor | 10s | 0.888 | 0.333 | 8.0 | 0.873 |
| Backdoor | 5s | 0.943 | 0.375 | 5.0 | 0.913 |
| injection | 10s | 0.463 | 0.382 | 219.0 | 0.393 |
| injection | 5s | 0.483 | 0.415 | 123.5 | 0.400 |
| password | 10s | 0.129 | 0.637 | 338.0 | 0.122 |
| password | 5s | 0.158 | 0.624 | 168.0 | 0.151 |
| xss | 10s | 0.265 | 0.380 | 766.0 | 0.427 |
| xss | 5s | 0.310 | 0.405 | 364.0 | 0.483 |

## Notes

- Undefined (no flows) is NaN and counted separately from genuine 0 entropy.
- Empty graphs are excluded from distributions; their counts are derivable from the interval
  length minus observed A+B windows.
- Similarity of aggregate profiles does not imply per-window similarity; compare the
  distributions and the present-vs-absent deltas.
