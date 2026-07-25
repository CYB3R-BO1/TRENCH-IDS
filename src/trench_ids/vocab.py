"""Global categorical vocabulary for PROTOCOL / L7_PROTO node identity.

Built once across every task's processed Parquet (not the raw datasets — the
processed tables already contain every value that will ever reach a task
graph), so a given PROTOCOL/L7_PROTO value maps to the same node identity
regardless of which task it appears in. This is required for relation-
specific memory (Step 4) to compare per-relation embeddings across tasks
(docs/dataset-plan.md §3.1). Host and Port identity are deliberately *not*
globalized here — Host never persists across tasks by design. Port's
log-bucketed tail (1024-65535, see rhgnn.port_embedding_index) has no
cross-task comparability requirement either; its well-known range (0-1023)
is a special case, comparable across tasks by construction (fixed
IANA-standard indices) without needing a vocab.py-style scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

VOCAB_COLUMNS = ("PROTOCOL", "L7_PROTO")


def build_vocab(
    parquet_paths: list[Path], columns: tuple[str, ...] = VOCAB_COLUMNS
) -> dict[str, dict[str, int]]:
    """Scan every task Parquet and assign each column's raw values a stable index."""
    values: dict[str, set] = {c: set() for c in columns}
    for path in parquet_paths:
        frame = pd.read_parquet(path, columns=list(columns))
        for c in columns:
            values[c].update(frame[c].unique().tolist())
    return {c: {str(v): i for i, v in enumerate(sorted(vs))} for c, vs in values.items()}


def save_vocab(vocab: dict[str, dict[str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, indent=2))


def load_vocab(path: Path) -> dict[str, dict[str, int]]:
    return json.loads(Path(path).read_text())
