# ToN-IoT per-class temporal autocorrelation (60 s bins)

## Configuration

- Resolution 60s; per class, its own training interval (zero bins included,
  outside-period excluded); classes never concatenated.
- Lags: >= 30, extended to 180 if not decayed; detrending + log1p sensitivity.
- Data sha256 `53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78`.
- **Suggested conservative block length: 76 min (4560 s)** = longest sustained dependence among primary classes.

## Class temporal dependence

| class | primary | interval | bins | lag1 | first<0.2 | sustained<0.2 | detrended sustained |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Backdoor | True | 2019-04-28 07:29:00+00:00 .. 2019-04-29 03:24:00+00:00 | 1195 | 0.956 | None | None | 77 |
| ddos | True | 2019-04-25 14:48:00+00:00 .. 2019-04-26 01:13:00+00:00 | 625 | 0.521 | 69 | 76 | 30 |
| dos | True | 2019-04-24 18:13:00+00:00 .. 2019-04-25 00:21:00+00:00 | 368 | 0.742 | 20 | 57 | 165 |
| injection | True | 2019-04-25 05:49:00+00:00 .. 2019-04-25 08:58:00+00:00 | 189 | 0.518 | 8 | 8 | 30 |
| mitm | False | 2019-04-29 12:20:00+00:00 .. 2019-04-29 14:00:00+00:00 | 100 | 0.258 | 2 | 10 | 10 |
| password | True | 2019-04-26 13:45:00+00:00 .. 2019-04-27 00:34:00+00:00 | 649 | 0.789 | 76 | 76 | 17 |
| ransomware | False | 2019-04-28 03:49:00+00:00 .. 2019-04-28 05:01:00+00:00 | 72 | 0.147 | 1 | 11 | 11 |
| scanning | True | 2019-04-23 14:09:00+00:00 .. 2019-04-24 06:11:00+00:00 | 962 | 0.874 | 11 | 11 | 11 |
| xss | True | 2019-04-27 04:54:00+00:00 .. 2019-04-27 10:29:00+00:00 | 335 | 0.679 | 19 | 19 | 16 |

## Sensitivity (first / sustained lag where |r|<0.2)

| class | raw | detrended | log1p |
| --- | --- | --- | --- |
| Backdoor | None/None | 77/77 | 25/43 |
| ddos | 69/76 | 21/30 | 53/53 |
| dos | 20/57 | 3/165 | 28/50 |
| injection | 8/8 | 6/30 | 17/17 |
| mitm | 2/10 | 2/10 | 3/25 |
| password | 76/76 | 11/17 | 27/131 |
| ransomware | 1/11 | 1/11 | 5/8 |
| scanning | 11/11 | 11/11 | None/None |
| xss | 19/19 | 5/16 | 7/16 |

## Blocks available (per class)

| class | 5min | 10min | 15min | 30min | 60min |
| --- | --- | --- | --- | --- | --- |
| Backdoor | 239 | 119 | 79 | 39 | 19 |
| ddos | 125 | 62 | 41 | 20 | 10 |
| dos | 73 | 36 | 24 | 12 | 6 |
| injection | 37 | 18 | 12 | 6 | 3 |
| mitm | 20 | 10 | 6 | 3 | 1 |
| password | 129 | 64 | 43 | 21 | 10 |
| ransomware | 14 | 7 | 4 | 2 | 1 |
| scanning | 192 | 96 | 64 | 32 | 16 |
| xss | 67 | 33 | 22 | 11 | 5 |

## Notes

- Bootstrap (later) must resample WITHIN each class's own temporal sequence; classes do
  not share contemporaneous blocks.
- Supplementary classes (mitm, ransomware) reported separately; their intervals are short
  (~2 h), so few blocks are available at long block lengths.
