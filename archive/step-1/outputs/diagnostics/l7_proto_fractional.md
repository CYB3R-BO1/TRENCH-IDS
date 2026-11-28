# Fractional `L7_PROTO` investigation (NF-UNSW-NB15-v3)

- Rows scanned: 2,365,424
- Fractional (non-integral, finite) values: 23,866 (1.0090%)
- Distinct fractional values: 10 (range 7.1260..10.4100)
- NaN L7_PROTO: 0

## Fractional values

| value | count |
| --- | --- |
| 7.37 | 18,765 |
| 10.16 | 4,877 |
| 7.253 | 146 |
| 7.376 | 45 |
| 7.5 | 8 |
| 7.212 | 7 |
| 7.291 | 7 |
| 10.41 | 7 |
| 7.126 | 2 |
| 7.18 | 2 |

## By class

| class | rows | fractional | % |
| --- | --- | --- | --- |
| Benign | 2,237,731 | 18,829 | 0.841 |
| Fuzzers | 33,816 | 3,964 | 11.722 |
| Exploits | 42,748 | 522 | 1.221 |
| Backdoor | 4,659 | 277 | 5.945 |
| Reconnaissance | 17,074 | 108 | 0.633 |
| DoS | 5,980 | 102 | 1.706 |
| Generic | 19,651 | 63 | 0.321 |
| Shellcode | 2,381 | 1 | 0.042 |
| Analysis | 1,226 | 0 | 0.000 |
| Worms | 158 | 0 | 0.000 |

## Co-occurrence among fractional rows

| condition | fractional rows | share | all-row base rate |
| --- | --- | --- | --- |
| TCP_FLAGS == 0 | 65 | 0.003 | 0.237 |
| IN_BYTES == 0 | 0 | 0.000 | 0.000 |
| OUT_BYTES == 0 | 66 | 0.003 | 0.028 |
| FLOW_DURATION_MILLISECONDS == 0 | 68 | 0.003 | 0.052 |
