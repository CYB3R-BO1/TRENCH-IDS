# TRENCH-IDS

Research repository for **TRENCH-IDS**: transferable representation learning for continual,
heterogeneous-graph-based intrusion detection. The current work focuses on **Step 1**:
characterising attack-class similarity and designing a reproducible task sequence for
task-incremental learning across attack classes.

> **Status:** Step 1 is approved and frozen as of 2026-11-24. The broader ten-step research
> proposal remains under review.

## Repository layout

| Path | Purpose |
|---|---|
| `INITIAL_METHODOLOGY.md` | Primary ten-step research proposal (under review). |
| `docs/` | Methodology drafts, including the Word source kept in sync with the Markdown. |
| `step-1/` | Step 1 code, tests, evidence and the approved task-sequence specification. |
| `analysis/` | Dataset analysis notebooks. |
| `data/` | Local raw NetFlow-v3 dataset bags (gitignored; never committed). |
| `requirements.txt` | Python dependencies for the Step 1 analyses. |

This is a research and analysis repository, not the older V2/V3 implementation repository.

## Step 1

Step 1 establishes the evidence base for the task sequence:

- attack-class similarity from standardised mean-profile cosine, Wasserstein distance,
  Jensen-Shannon distance, relation-profile blocks and TCP flag bits;
- window-size feasibility, class mixing and per-window relation statistics;
- temporal block-bootstrap uncertainty;
- dataset-specific splits for **NF-UNSW-NB15-v3** and **NF-ToN-IoT-v3**;
- a deterministic same-window benign-allocation protocol and a frozen 10 s task inventory.

Key documents:

- [Task-sequence specification](step-1/TRENCH_TASK_SEQUENCE_DESIGN.md) — approved and frozen.
- [Step 1 final report](step-1/STEP1_FINAL_REPORT.md) — closure record, verification and
  remaining limitations.
- [Step 1 evidence README](step-1/README.md) — per-run status, findings and reproduction notes.

Frozen evidence:

- [`task_feasibility_10s.csv`](step-1/evidence/task_feasibility_10s.csv) — 36 eligible
  class/split rows with graph counts, attack thresholds, medians and allocated benign flows.
- [`relation_distances.csv`](step-1/evidence/relation_distances.csv) — seven relation blocks
  for the three comparison scopes.

Freeze hashes are recorded in [`step-1/FROZEN_STEP1.sha256`](step-1/FROZEN_STEP1.sha256).
The Step 1 files are marked read-only; clear the read-only attribute before reopening Step 1.

The approved baseline is commit `4e172e0`, tagged `step1-frozen`.

## Setup

Python 3.10+ with the dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

There is no build system. Tests use the standard-library `unittest` runner plus numpy, pandas,
scipy, matplotlib and seaborn.

## Data

Raw datasets are **not** committed. `data/` is gitignored and contains local NetFlow-v3 bags
for UNSW-NB15, ToN-IoT, BoT-IoT and CICIDS2018. Step 1 uses:

- `data/UNSW/data/NF-UNSW-NB15-v3.csv`
- `data/ToN-IoT/NF-ToN-IoT-v3/data/NF-ToN-IoT-v3.csv`

Keep new raw data drops out of Git. Commit only code, specifications and derived evidence.

## Reproduce Step 1

From `step-1/`, with the raw bags in the paths above:

```bash
python task_protocol.py --out <output-directory> --write-allocations
python task_protocol.py --out <output-directory> --recompute-ton-relations
```

The first command reproduces the 10 s feasibility evidence and exports each task's window
membership and benign ownership. The second adds the canonical ToN relation blocks and scans
the large ToN CSV. Use an output directory outside the frozen Step 1 tree.

The full command history and expected outputs are documented in
[`step-1/README.md`](step-1/README.md).

## Tests

From `step-1/`:

```bash
python -m unittest discover -s tests
```

The suite is synthetic and does not read the raw datasets. The frozen Step 1 baseline passed
all 81 tests.

## Documentation map

- [`INITIAL_METHODOLOGY.md`](INITIAL_METHODOLOGY.md) — primary ten-step proposal, under review.
- [`docs/Updated_Proposed_Methodology_Transferable_Representation_Learning.docx`](docs/Updated_Proposed_Methodology_Transferable_Representation_Learning.docx)
  — Word source kept in sync with the Markdown proposal.
- [`step-1/TRENCH_TASK_SEQUENCE_DESIGN.md`](step-1/TRENCH_TASK_SEQUENCE_DESIGN.md) — approved
  Step 1 task-sequence specification.
- [`step-1/STEP1_FINAL_REPORT.md`](step-1/STEP1_FINAL_REPORT.md) — Step 1 closure report.
