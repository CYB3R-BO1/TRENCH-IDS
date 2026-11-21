# Step 1.4 - ToN-IoT temporal block bootstrap (within-class blocks)

- Blocks primary 30 min, sensitivity 60 min; 500 replicates.
- Resampling is within each class's own sequence; classes never concatenated or shared.
- Training-only normalisation recomputed per replicate.
- Data sha256 `53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78`.

## Classes with < 5 available blocks (cautious interpretation)

- 30 min: mitm (4), ransomware (3)
- 60 min: injection (4), mitm (2), ransomware (2)

## Blocks per class

| class | 30 min | 60 min |
| --- | --- | --- |
| Backdoor | 41 | 21 |
| ddos | 20 | 12 |
| dos | 13 | 7 |
| injection | 6 | 4 |
| mitm | 4 | 2 |
| password | 22 | 12 |
| ransomware | 3 | 2 |
| scanning | 31 | 16 |
| xss | 12 | 7 |

## Notes

- Rank stability and CI widths are reported per block length (separate CSVs).
- Few blocks (e.g. injection at 60 min; mitm/ransomware) mean wide, cautious intervals;
  500 replicates add Monte-Carlo precision only, not independent temporal observations.
- ToN's disjoint attack periods (vs UNSW co-occurrence) are a dataset characteristic, not
  a nuisance: resampling never mixes classes.
