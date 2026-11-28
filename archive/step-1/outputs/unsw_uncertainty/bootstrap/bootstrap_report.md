# Step 1.3 - temporal block bootstrap

- Primary block length 900s, 500 replicates; seed 0.
- Training fine (60 s) blocks: 1059; coarse blocks: 60s=1059, 300s=216, 900s=73.
- **Caveat:** coarse-block count limits effective temporal information (primary = 73 blocks); replicate count improves Monte-Carlo precision only, not the number of independent blocks.
- Classes: Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode.
- Blocks are contiguous training time spans; all classes (incl. benign) move together;
  the same blocks are used for every pair; gmean/gstd are recomputed per replicate.
- Categorical universe and host-degree padding are FIXED to the full-train values.

## Defined coverage (primary)

- All pairs defined in 100.0%-100.0% of replicates.

## 95% CI width by block length (mean over pairs)

| metric | 900s | 60s | 300s |
| --- | --- | --- | --- |
| cosine | 0.1272 | 0.0783 | 0.1182 |
| wasserstein | 0.3646 | 0.2255 | 0.3392 |
| jensen_shannon | 0.0996 | 0.0537 | 0.0787 |
| relation_profile | 0.1006 | 0.0617 | 0.0830 |
| flag_bit_similarity | 0.0213 | 0.0176 | 0.0175 |

## Top pairs (primary block length)

**cosine**
- Backdoor-DoS: observed 0.9953, 95% CI [0.9097, 0.9959]
- Backdoor-Fuzzers: observed 0.9911, 95% CI [0.9311, 0.9926]
- Reconnaissance-Shellcode: observed 0.9907, 95% CI [0.9596, 0.9975]
- DoS-Fuzzers: observed 0.9862, 95% CI [0.9487, 0.9909]
- Fuzzers-Reconnaissance: observed 0.9745, 95% CI [0.9485, 0.9897]

**wasserstein**
- Backdoor-Fuzzers: observed 0.1474, 95% CI [0.1470, 0.4399]
- DoS-Fuzzers: observed 0.1660, 95% CI [0.1438, 0.3256]
- Backdoor-DoS: observed 0.2045, 95% CI [0.1850, 0.5087]
- Reconnaissance-Shellcode: observed 0.2083, 95% CI [0.0928, 0.3791]
- DoS-Exploits: observed 0.2701, 95% CI [0.0806, 0.4920]

**jensen_shannon**
- DoS-Exploits: observed 0.3401, 95% CI [0.3319, 0.3941]
- Backdoor-DoS: observed 0.3623, 95% CI [0.3579, 0.5224]
- Backdoor-Fuzzers: observed 0.3661, 95% CI [0.3529, 0.5337]
- Backdoor-Exploits: observed 0.3732, 95% CI [0.3493, 0.5328]
- Backdoor-Reconnaissance: observed 0.4108, 95% CI [0.4022, 0.5553]

**relation_profile**
- DoS-Exploits: observed 0.7030, 95% CI [0.6327, 0.7263]
- Backdoor-DoS: observed 0.6764, 95% CI [0.5595, 0.7028]
- Backdoor-Fuzzers: observed 0.6588, 95% CI [0.5645, 0.6776]
- DoS-Fuzzers: observed 0.6570, 95% CI [0.6018, 0.6631]
- Exploits-Fuzzers: observed 0.6481, 95% CI [0.5953, 0.6452]

**flag_bit_similarity**
- DoS-Generic: observed 0.9993, 95% CI [0.9960, 0.9997]
- Fuzzers-Reconnaissance: observed 0.9986, 95% CI [0.9943, 0.9996]
- Backdoor-DoS: observed 0.9956, 95% CI [0.9864, 0.9980]
- Backdoor-Generic: observed 0.9955, 95% CI [0.9897, 0.9983]
- Backdoor-Fuzzers: observed 0.9924, 95% CI [0.9827, 0.9984]


## Rank stability across block lengths (probability of top-5)

| metric | pair | 900s | 60s | 300s |
| --- | --- | --- | --- | --- |
| cosine | Reconnaissance-Shellcode | 0.93 | 0.92 | 0.92 |
| cosine | DoS-Fuzzers | 0.75 | 0.92 | 0.83 |
| cosine | Backdoor-DoS | 0.65 | 0.96 | 0.75 |
| cosine | Fuzzers-Reconnaissance | 0.61 | 0.59 | 0.56 |
| cosine | Backdoor-Fuzzers | 0.51 | 0.96 | 0.69 |
| cosine | DoS-Exploits | 0.41 | 0.19 | 0.31 |
| wasserstein | DoS-Fuzzers | 0.90 | 1.00 | 0.97 |
| wasserstein | Reconnaissance-Shellcode | 0.83 | 0.80 | 0.80 |
| wasserstein | DoS-Exploits | 0.62 | 0.61 | 0.61 |
| wasserstein | Backdoor-Fuzzers | 0.50 | 0.94 | 0.64 |
| wasserstein | Backdoor-DoS | 0.46 | 0.89 | 0.58 |
| wasserstein | Fuzzers-Reconnaissance | 0.31 | 0.19 | 0.27 |
| jensen_shannon | DoS-Exploits | 1.00 | 1.00 | 1.00 |
| jensen_shannon | Backdoor-DoS | 0.84 | 1.00 | 0.96 |
| jensen_shannon | Backdoor-Exploits | 0.79 | 0.99 | 0.96 |
| jensen_shannon | Backdoor-Fuzzers | 0.85 | 1.00 | 0.94 |
| jensen_shannon | DoS-Fuzzers | 0.48 | 0.27 | 0.36 |
| jensen_shannon | Backdoor-Reconnaissance | 0.60 | 0.73 | 0.61 |
| relation_profile | DoS-Exploits | 1.00 | 1.00 | 1.00 |
| relation_profile | Backdoor-DoS | 0.81 | 0.99 | 0.94 |
| relation_profile | DoS-Fuzzers | 0.82 | 0.90 | 0.83 |
| relation_profile | Backdoor-Fuzzers | 0.64 | 0.99 | 0.87 |
| relation_profile | Exploits-Fuzzers | 0.59 | 0.70 | 0.56 |
| relation_profile | Backdoor-Exploits | 0.50 | 0.28 | 0.49 |
| flag_bit_similarity | DoS-Generic | 1.00 | 1.00 | 1.00 |
| flag_bit_similarity | Fuzzers-Reconnaissance | 0.99 | 1.00 | 0.99 |
| flag_bit_similarity | Backdoor-Generic | 0.84 | 0.87 | 0.82 |
| flag_bit_similarity | Backdoor-DoS | 0.76 | 0.84 | 0.79 |
| flag_bit_similarity | DoS-Exploits | 0.31 | 0.31 | 0.37 |
| flag_bit_similarity | Backdoor-Fuzzers | 0.26 | 0.25 | 0.22 |
