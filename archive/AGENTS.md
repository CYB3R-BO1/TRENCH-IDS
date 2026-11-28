# AGENTS.md

## Repository scope

- Research repository for TRENCH-IDS: transferable representation learning for continual,
  heterogeneous-graph-based intrusion detection (task-incremental learning across attack classes).
- This repository is the **current** working research repo for the Step-1 attack-class similarity /
  task-design line of work. It is a research + analysis repo, not the old V2/V3 implementation
  repo.
- It contains:
  - `INITIAL_METHODOLOGY.md` — primary 10-step proposal (explicitly **under review, not finalized**).
  - `docs/` — methodology drafts (`.docx` binaries).
  - `step-1/` — Step-1 implementation: similarity metrics, window feasibility, uncertainty /
    bootstrap analyses, ToN variants; `tests/`; generated `outputs/`; `README.md`;
    `TRENCH_TASK_SEQUENCE_DESIGN.md` (draft task-sequence specification).
  - `analysis/` — dataset analysis notebooks.
  - `requirements.txt` — Python dependencies for the Step-1 analyses.
  - `data/` - raw NetFlow-v3 dataset bags (local data drops; gitignored, not committed; inside
    this repository).
- There is no build system. Tests are stdlib `unittest` plus numpy/pandas/scipy/matplotlib/seaborn
  (see `requirements.txt`). Run the synthetic tests from `step-1/`: `python -m unittest discover -s tests`.

## Document map

- `INITIAL_METHODOLOGY.md` — primary 10-step proposal.
- `docs/Updated_Proposed_Methodology_Transferable_Representation_Learning.docx` — Word source
  containing the same 10 steps; keep it in sync with the Markdown.
- `docs/Proposed methodology for the next work.docx` — earlier, shorter draft; superseded, not current.
- `step-1/README.md` — Step-1 evidence, per-run status, how to reproduce.
- `step-1/TRENCH_TASK_SEQUENCE_DESIGN.md` — approved Step 1 task-sequence specification
  (frozen 2026-11-24; Step 1 is closed from a research-design perspective).

## Related material outside this repository

- `..\old_TRENCH-IDS_research` - pre-separation research archive (old V2 implementation,
  V3 rebuild, V2 data drops, audit documents). Treat as read-only/frozen.
- Inside it: `old/` (frozen V2 archive), `new/` (V3 rebuild), `v2/` (raw V2 bags), `project-docs/`,
  `PROJECT_AUDIT.md`, `scratchpad/`, `figures/`.
- `..\old_TRENCH-IDS_research\project-docs\README.md` — entry-point research record for the old work.

## Editing rules

- Decide first whether the request concerns **this proposal** or the **V3 implementation**; they are
  separate. Do not merge designs or copy V3 claims into the proposal.
- When changing the methodology, update both `INITIAL_METHODOLOGY.md` and the updated `.docx`,
  preserving step numbering and terms. If they cannot be kept in sync, flag the divergence instead
  of silently choosing one.
- Keep the "under review" framing and open questions. Do not add experiment results or claims that
  are not backed by a stored artifact.
