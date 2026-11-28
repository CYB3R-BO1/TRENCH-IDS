# TRENCH-IDS

Transferable representation learning for continual heterogeneous graph-based network intrusion detection.

**Current status:** Clean research workspace; methodology v1.0 documented for professor review. Phase 0 protocol discussion is next. No new implementation or experimental results are claimed.

The project investigates whether representations learned from previous attack categories help a binary intrusion detector adapt to new attack categories, which network relations contribute to transfer, and whether validated relation guidance improves continual learning. EWC and experience replay will be compared rather than assuming a winner.

## Start here

| Path | Purpose |
|---|---|
| [METHODOLOGY.md](METHODOLOGY.md) | Authoritative proposed research methodology, hypotheses, phases and evaluation |
| [FLOW.md](FLOW.md) | Operational roadmap and approval status |
| [AGENTS.md](AGENTS.md) | Human-guided implementation rules |
| [archive/](archive/) | Previous research, code, notebooks and evidence |
| data/ | Local raw datasets (gitignored) |

The previous Step 1 investigation is preserved in [archive/step-1](archive/step-1/). Its original reproducible paths are available at the [step1-frozen tag](https://github.com/CYB3R-BO1/TRENCH-IDS/tree/step1-frozen). Moving files into the archive may affect old relative paths; do not edit frozen artifacts to work around this. See the [closure report](archive/step-1/STEP1_FINAL_REPORT.md).

## Development

Start with NF-UNSW-NB15-v3 only. Discuss, specify, approve, implement and verify each operation before proceeding. Create files only when required. The raw data remain local. No new source-code structure or dependencies have been introduced.
