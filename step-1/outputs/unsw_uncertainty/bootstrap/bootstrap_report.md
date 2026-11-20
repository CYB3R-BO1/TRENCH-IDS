# Step 1.3 - temporal block bootstrap

- Primary block length 300s, 500 replicates; seed 0.
- Training blocks: 1059; classes: Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode.
- Blocks are contiguous training time spans; all classes (incl. benign) move together;
  the same blocks are used for every pair; gmean/gstd are recomputed per replicate.
- Categorical universe and host-degree padding are FIXED to the full-train values.

## 95% CI width by block length (mean over pairs)

| metric | 300s | 60s | 900s |
| --- | --- | --- | --- |
| cosine | 0.1192 | 0.0736 | 0.1253 |
| wasserstein | 0.3392 | 0.2087 | 0.3537 |
| jensen_shannon | 0.0818 | 0.0528 | 0.1021 |
| relation_profile | 0.0865 | 0.0651 | 0.1040 |
| flag_bit_similarity | 0.0187 | 0.0179 | 0.0214 |

## Top pairs (primary block length)

**cosine**
- Backdoor-DoS: observed 0.9953, 95% CI [0.9281, 0.9957]
- Backdoor-Fuzzers: observed 0.9911, 95% CI [0.9106, 0.9938]
- Reconnaissance-Shellcode: observed 0.9907, 95% CI [0.9564, 0.9975]
- DoS-Fuzzers: observed 0.9862, 95% CI [0.9598, 0.9905]
- Fuzzers-Reconnaissance: observed 0.9745, 95% CI [0.9460, 0.9917]

**wasserstein**
- Backdoor-Fuzzers: observed 0.1474, 95% CI [0.1287, 0.5252]
- DoS-Fuzzers: observed 0.1660, 95% CI [0.1385, 0.2781]
- Backdoor-DoS: observed 0.2045, 95% CI [0.1786, 0.5381]
- Reconnaissance-Shellcode: observed 0.2083, 95% CI [0.0969, 0.3825]
- DoS-Exploits: observed 0.2701, 95% CI [0.1115, 0.4416]

**jensen_shannon**
- DoS-Exploits: observed 0.3401, 95% CI [0.3351, 0.3852]
- Backdoor-DoS: observed 0.3623, 95% CI [0.3625, 0.4674]
- Backdoor-Fuzzers: observed 0.3661, 95% CI [0.3624, 0.5072]
- Backdoor-Exploits: observed 0.3732, 95% CI [0.3482, 0.4422]
- Backdoor-Reconnaissance: observed 0.4108, 95% CI [0.4014, 0.5093]

**relation_profile**
- DoS-Exploits: observed 0.7030, 95% CI [0.6442, 0.7227]
- Backdoor-DoS: observed 0.6764, 95% CI [0.5958, 0.6983]
- Backdoor-Fuzzers: observed 0.6588, 95% CI [0.5862, 0.6752]
- DoS-Fuzzers: observed 0.6570, 95% CI [0.6075, 0.6607]
- Exploits-Fuzzers: observed 0.6481, 95% CI [0.6024, 0.6481]

**flag_bit_similarity**
- DoS-Generic: observed 0.9993, 95% CI [0.9957, 0.9997]
- Fuzzers-Reconnaissance: observed 0.9986, 95% CI [0.9944, 0.9996]
- Backdoor-DoS: observed 0.9956, 95% CI [0.9898, 0.9981]
- Backdoor-Generic: observed 0.9955, 95% CI [0.9892, 0.9982]
- Backdoor-Fuzzers: observed 0.9924, 95% CI [0.9855, 0.9940]

