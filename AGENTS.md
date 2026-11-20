# AGENTS.md

## Repository scope

- Research-design/documentation repo for TRENCH-IDS, not the implementation. Apart from this file, it contains only `INITIAL_METHODOLOGY.md`, two `.docx` files under `docs/`, and an empty `WRITEUP.md`; there is no source, tests, build, lint, or dependency manifest.
- There are no build/test/lint commands. Do not invent a stack or add scaffolding unless the user asks.
- `INITIAL_METHODOLOGY.md` is explicitly a proposal **under review, not finalized**. Do not implement it, describe it as settled, or add unsupported results.

## Document map

- `INITIAL_METHODOLOGY.md` — primary 10-step proposal.
- `docs/Updated_Proposed_Methodology_Transferable_Representation_Learning.docx` — Word source containing the same 10 steps; keep it in sync with the Markdown.
- `docs/Proposed methodology for the next work.docx` — earlier, shorter draft; superseded, not current.
- `WRITEUP.md` — empty placeholder (0 bytes); no content or claims yet.
- `.docx` files are binary: extract text with Python's stdlib (`zipfile` → `word/document.xml`), not by reading them directly. Write extracted output as UTF-8; the default Windows console (cp1252) cannot print the arrows/curly quotes in these files.

## Related material outside this repo

- `../new/` — V3 clean implementation and artifacts. Read `../new/info/project-state/README.md` first; `../new/AGENTS.md` has useful rules, but its status claims are partly stale per `../project-docs/README.md`.
- `../old/` — frozen V2 archive; do not modify. `../PROJECT_AUDIT.md` and `../project-docs/` hold the verified status, known defects, and claim limits.
- `../new/plans/v3-temporal-benchmark-spec.md` is the frozen V3 spec. It defines a **different experiment** (matched-budget A→B transfer) from this repo's prototype-memory / transferability-weighted-EWC proposal; V3 results do not validate this proposal.
- V3 Step-3 transfer results are single-seed and affected by known graph-node and macro-F1 label-set defects. Do not cite them as validated. Never compare V2 and V3 numbers directly — the data, tasks, and metrics differ.

## Editing rules

- Decide first whether the request concerns **this proposal** or the **V3 implementation**; they are separate. Do not merge designs or copy V3 claims into the proposal.
- When changing the methodology, update both `INITIAL_METHODOLOGY.md` and the updated `.docx`, preserving step numbering and terms. If they cannot be kept in sync, flag the divergence instead of silently choosing one.
- Keep the "under review" framing and open questions. Do not add experiment results or claims that are not backed by a stored artifact.
