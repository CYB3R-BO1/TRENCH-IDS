# ToN-IoT benign partition / normalisation audit

*Read-only: no similarity matrices or split files were modified.*

## Per-class matched benign (windows containing the class)

| class | split | benign matched | attack flows | windows |
| --- | --- | --- | --- | --- |
| Backdoor | train | 185,570 | 142,242 | 1,181 |
| Backdoor | val | 16,980 | 30,560 | 191 |
| Backdoor | test | 17,858 | 30,582 | 191 |
| ddos | train | 3,868,282 | 2,897,088 | 553 |
| ddos | val | 811,940 | 622,335 | 103 |
| ddos | test | 615,360 | 621,833 | 125 |
| dos | train | 77,944 | 142,098 | 368 |
| dos | val | 17,939 | 30,730 | 75 |
| dos | test | 17,732 | 30,628 | 75 |
| injection | train | 67,149 | 266,120 | 145 |
| injection | val | 11,664 | 58,189 | 50 |
| injection | test | 10,507 | 57,468 | 27 |
| mitm | train | 1,755,629 | 4,140 | 77 |
| mitm | val | 7,513 | 956 | 12 |
| mitm | test | 10,391 | 917 | 19 |
| password | train | 151,860 | 1,113,551 | 648 |
| password | val | 8,013 | 240,391 | 103 |
| password | test | 7,719 | 240,835 | 91 |
| ransomware | train | 14,855 | 2,681 | 64 |
| ransomware | val | 1,500 | 694 | 10 |
| ransomware | test | 3,551 | 596 | 17 |
| scanning | train | 8,327,687 | 905,605 | 716 |
| scanning | val | 21,004 | 191,427 | 11 |
| scanning | test | 190,977 | 261,945 | 9 |
| xss | train | 283,026 | 1,979,768 | 335 |
| xss | val | 24,342 | 428,391 | 65 |
| xss | test | 27,996 | 426,276 | 67 |

## Cross-task partition isolation

- Benign windows used by any class task: **5,328** (16,554,988 benign flows).
- Windows used by >1 class: **0**.
- Windows with conflicting train/val/test assignment across classes: **0**.
- Therefore B_train n B_val = B_train n B_test = B_val n B_test = empty for the shared benign observations.

## Normalisation scope (the issue)

- Precedence policy puts **16,772,031** benign flows into 'train' - essentially all benign, including time that is validation/test for some classes.
- Window-matched train benign = **14,732,002**; **2,040,029** benign flows are wrongly included by precedence.
- So the current global mean/std is contaminated; per-class means and the similarity inputs themselves remain train-only.

## Recommended correction

1. Assign benign per window to the class whose windows it belongs to (window membership).
2. Build the global mean/std from benign in the union of per-class TRAIN windows only.
3. Re-run the similarity matrices with that scope (new outputs; keep the current run).
