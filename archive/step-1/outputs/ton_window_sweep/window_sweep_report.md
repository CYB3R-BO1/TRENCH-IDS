# Step 1.4 - NF-ToN-IoT-v3 window-size sweep (per-class contiguous split)

*Canonical split: per-class contiguous, benign window-matched.*

- Dataset sha256 `53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78`; rows 27,520,260; 60 s-aligned class cuts.
- Primary classes (3-split floor): Backdoor, ddos, dos, injection, password, scanning, xss.
- Supplementary (train-supported, fail floor): mitm, ransomware.
- Thresholds for class flows: [2, 5, 10, 20].
- Mixing is 0 by construction: ToN attack classes occupy disjoint time windows, so no
  window contains two attack classes. Benign:attack imbalance is left untouched.

## Headline (train)

| window | windows | median flows | p90 | max |
| --- | --- | --- | --- | --- |
| 1s | 179,461 | 27.0 | 230.0 | 16,790 |
| 5s | 43,114 | 110.0 | 1043.0 | 62,014 |
| 10s | 22,376 | 202.5 | 2088.0 | 116,323 |
| 30s | 7,873 | 514.0 | 6508.0 | 253,130 |
| 60s | 4,087 | 943.0 | 12751.6 | 470,866 |

## Task-filtered feasibility (train) - windows with >= 5 class flows

| window | class | primary | windows>=5 | >=10 | >=20 | median filtered total | median b:a ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1s | Backdoor | True | 2,583 | 3 | 1 | 4.0 | 0.5 |
| 1s | ddos | True | 27,798 | 25,868 | 22,404 | 170.0 | 2.2 |
| 1s | dos | True | 14,348 | 3,449 | 637 | 9.0 | 0.5 |
| 1s | injection | True | 5,452 | 4,902 | 3,746 | 34.0 | 0.3 |
| 1s | mitm | False | 182 | 91 | 59 | 5.0 | 1.0 |
| 1s | password | True | 36,394 | 34,484 | 29,916 | 32.0 | 0.1 |
| 1s | ransomware | False | 100 | 97 | 72 | 3.0 | 1.0 |
| 1s | scanning | True | 5,452 | 4,382 | 3,540 | 217.5 | 42.1 |
| 1s | xss | True | 19,446 | 19,364 | 19,054 | 100.0 | 0.1 |
| 5s | Backdoor | True | 13,274 | 8,427 | 48 | 18.0 | 1.0 |
| 5s | ddos | True | 6,222 | 6,216 | 6,202 | 912.0 | 1.7 |
| 5s | dos | True | 4,403 | 4,376 | 4,002 | 47.0 | 0.5 |
| 5s | injection | True | 1,313 | 1,268 | 1,121 | 152.0 | 0.3 |
| 5s | mitm | False | 100 | 54 | 33 | 13.0 | 5.0 |
| 5s | password | True | 7,699 | 7,680 | 7,631 | 170.0 | 0.1 |
| 5s | ransomware | False | 95 | 90 | 77 | 14.0 | 3.0 |
| 5s | scanning | True | 2,017 | 1,341 | 1,135 | 639.0 | 128.4 |
| 5s | xss | True | 3,946 | 3,933 | 3,916 | 548.0 | 0.1 |
| 10s | Backdoor | True | 6,991 | 6,767 | 3,287 | 38.0 | 1.0 |
| 10s | ddos | True | 3,142 | 3,139 | 3,137 | 1874.0 | 1.6 |
| 10s | dos | True | 2,203 | 2,202 | 2,201 | 95.0 | 0.5 |
| 10s | injection | True | 696 | 667 | 638 | 285.0 | 0.4 |
| 10s | mitm | False | 84 | 44 | 28 | 20.0 | 9.5 |
| 10s | password | True | 3,857 | 3,854 | 3,852 | 342.0 | 0.1 |
| 10s | ransomware | False | 110 | 80 | 69 | 36.0 | 3.7 |
| 10s | scanning | True | 1,429 | 956 | 675 | 1152.5 | 147.9 |
| 10s | xss | True | 1,983 | 1,980 | 1,973 | 1128.0 | 0.1 |
| 30s | Backdoor | True | 2,342 | 2,337 | 2,329 | 128.0 | 1.2 |
| 30s | ddos | True | 1,079 | 1,079 | 1,078 | 5952.0 | 1.4 |
| 30s | dos | True | 735 | 735 | 735 | 297.0 | 0.5 |
| 30s | injection | True | 261 | 251 | 239 | 819.0 | 0.3 |
| 30s | mitm | False | 70 | 48 | 20 | 69.5 | 12.6 |
| 30s | password | True | 1,289 | 1,288 | 1,288 | 1027.0 | 0.1 |
| 30s | ransomware | False | 79 | 63 | 50 | 112.5 | 6.4 |
| 30s | scanning | True | 692 | 535 | 412 | 2080.0 | 180.0 |
| 30s | xss | True | 666 | 665 | 664 | 3359.5 | 0.1 |
| 60s | Backdoor | True | 1,176 | 1,175 | 1,171 | 248.0 | 1.3 |
| 60s | ddos | True | 553 | 553 | 553 | 12169.0 | 1.4 |
| 60s | dos | True | 368 | 368 | 368 | 607.0 | 0.5 |
| 60s | injection | True | 141 | 138 | 133 | 1790.0 | 0.3 |
| 60s | mitm | False | 64 | 47 | 26 | 131.0 | 10.9 |
| 60s | password | True | 648 | 646 | 646 | 2056.5 | 0.1 |
| 60s | ransomware | False | 51 | 46 | 43 | 226.5 | 5.9 |
| 60s | scanning | True | 425 | 336 | 270 | 2390.5 | 227.9 |
| 60s | xss | True | 334 | 334 | 333 | 6834.0 | 0.1 |

## Notes

- Unlabeled context is ~0 (classes disjoint), kept as a descriptive reference only.
- No benign downsampling; the extreme per-class imbalance is preserved.
- Window size is not auto-selected; review occupancy + task-filtered feasibility.
