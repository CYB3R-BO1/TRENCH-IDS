# AGENTS.md

## Scope

This is the active clean-rebuild research workspace for TRENCH-IDS. Read [METHODOLOGY.md](METHODOLOGY.md) and [FLOW.md](FLOW.md) before making changes. The old research is in archive/; the original frozen Step 1 layout is available at tag step1-frozen.

## Human-guided workflow

- The researcher owns methodological choices. Discuss the next operation and present a precise approach before implementation; obtain explicit approval. Do not infer approval to implement an entire phase from approval of the methodology.
- Implement only the approved operation. Do not advance, add speculative features, run expensive experiments, or create extra scripts/reports/configuration without approval.
- Explain purpose, inputs, assumptions, mathematics, alternatives, output, a small manual example and verification/failure conditions before coding.
- Prefer readable, direct functions and minimal dependencies. Create files only when they serve an approved operation. Explain every changed file.
- After implementation, show code/diff, intermediate outputs and tests. Stop for the researcher's understanding and approval before proceeding. Do not commit or push unless requested.
- Preserve raw datasets and archived artifacts. Do not edit archive/ or frozen tags without explicit approval. Old relative paths may be invalid after archiving; use the frozen tag for original reproduction.
- Respect the information-access policy: no future-task or test leakage, labels not model features, no tuning on final test results. Document any change to archived splits, allocation or no-replay protocol before implementation.
- Distinguish hypotheses, engineering pilots, validation results and confirmatory evidence. Report failures and negative results; do not invent claims.
- Keep large generated artifacts, datasets, caches and checkpoints out of Git by default. Record compact evidence, configurations and provenance only as needed.
- Keep METHODOLOGY.md authoritative and FLOW.md the approved operation sequence. Avoid redundant reports.

Current next step: Phase 0 research protocol discussion; no code implementation has been approved for the rebuild.
