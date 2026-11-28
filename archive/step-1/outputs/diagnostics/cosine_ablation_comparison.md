# Cosine ablation: 37 vs 35 magnitude features

Baseline excludes nothing; ablation drops `SRC_TO_DST_SECOND_BYTES` and `DST_TO_SRC_SECOND_BYTES`. Cosine is otherwise identical.

- Pairs compared: 21
- Mean |delta|: 0.0058
- Max |delta|: 0.0220 (Exploits-Generic)
- Spearman rho: 0.9987
- Kendall tau: 0.9905
- Top-5 pair overlap: 5/5
- Generic mean similarity: baseline 0.7651 -> ablated 0.7579

| rank | baseline pair | cos | ablated pair | cos |
| --- | --- | --- | --- | --- |
| 1 | Backdoor-DoS | 0.9953 | Backdoor-DoS | 0.9949 |
| 2 | Backdoor-Fuzzers | 0.9911 | Backdoor-Fuzzers | 0.9905 |
| 3 | Reconnaissance-Shellcode | 0.9907 | Reconnaissance-Shellcode | 0.9902 |
| 4 | DoS-Fuzzers | 0.9862 | DoS-Fuzzers | 0.9853 |
| 5 | Fuzzers-Reconnaissance | 0.9745 | Fuzzers-Reconnaissance | 0.9724 |

## Largest changes

| pair | baseline | ablated | delta |
| --- | --- | --- | --- |
| Exploits-Generic | 0.5394 | 0.5174 | -0.0220 |
| Exploits-Shellcode | 0.8252 | 0.8067 | -0.0185 |
| Exploits-Reconnaissance | 0.8413 | 0.8260 | -0.0154 |
| DoS-Generic | 0.7375 | 0.7293 | -0.0083 |
| Backdoor-Generic | 0.7553 | 0.7477 | -0.0076 |
| Fuzzers-Generic | 0.7795 | 0.7728 | -0.0067 |
| Exploits-Fuzzers | 0.9289 | 0.9231 | -0.0058 |
| DoS-Shellcode | 0.9417 | 0.9360 | -0.0056 |
