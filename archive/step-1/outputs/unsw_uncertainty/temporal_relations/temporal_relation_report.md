# Step 1.3 - per-window relation statistics (task-filtered)

Per window, statistics are computed on the designated class's flows; the window must contain
the class (windows where it is absent are excluded). `n_benign_flows` is the benign traffic in
the same window under the task-filtered policy. Windows are reported at 10 s (primary) and 5 s.

## 10s - train dst-port entropy (median per class)

| class | windows | median | IQR | p90 |
| --- | --- | --- | --- | --- |
| Analysis | 128 | 0.000 | 0.000 | 0.000 |
| Backdoor | 277 | 0.000 | 0.831 | 0.916 |
| DoS | 1317 | 0.000 | 0.722 | 1.000 |
| Exploits | 2297 | 0.840 | 0.196 | 0.971 |
| Fuzzers | 1411 | 0.000 | 0.894 | 0.985 |
| Generic | 1715 | 0.000 | 0.790 | 1.000 |
| Reconnaissance | 2149 | 0.000 | 0.918 | 1.000 |
| Shellcode | 678 | 0.000 | 0.000 | 1.000 |
| Worms | 82 | 0.000 | 0.000 | 0.000 |

## 5s - train dst-port entropy (median per class)

| class | windows | median | IQR | p90 |
| --- | --- | --- | --- | --- |
| Analysis | 209 | 0.000 | 0.000 | 0.000 |
| Backdoor | 391 | 0.789 | 0.911 | 0.968 |
| DoS | 1678 | 0.000 | 0.000 | 1.000 |
| Exploits | 4452 | 0.865 | 0.475 | 1.000 |
| Fuzzers | 2560 | 0.000 | 0.918 | 0.985 |
| Generic | 2836 | 0.000 | 0.722 | 0.960 |
| Reconnaissance | 3547 | 0.000 | 0.722 | 1.000 |
| Shellcode | 795 | 0.000 | 0.000 | 0.502 |
| Worms | 83 | 0.000 | 0.000 | 0.000 |

