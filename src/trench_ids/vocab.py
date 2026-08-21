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

import hashlib
import json
from pathlib import Path

import pandas as pd

VOCAB_COLUMNS = ("PROTOCOL", "L7_PROTO")


def vocab_key(value: object) -> str:
    """Canonical string for a categorical value, robust to float coercion.

    A Parquet column that is integer almost everywhere parses as int64, but
    one NaN anywhere in the source forces pandas to float64 -- and then the
    same protocol contributes ``"6.0"`` from that file and ``"6"`` from
    every other. Keying with raw ``str(v)`` would give one value two vocab
    entries and two distinct Protocol node identities, silently splitting
    its message-passing bucket. Normalising integral floats to their integer
    string makes both spellings agree; non-integral floats and strings pass
    through ``str()`` unchanged.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_vocab(
    parquet_paths: list[Path], columns: tuple[str, ...] = VOCAB_COLUMNS
) -> dict[str, dict[str, int]]:
    """Scan every task Parquet and assign each column's raw values a stable index."""
    values: dict[str, set] = {c: set() for c in columns}
    for path in parquet_paths:
        frame = pd.read_parquet(path, columns=list(columns))
        for c in columns:
            values[c].update(frame[c].unique().tolist())
    return {
        c: {vocab_key(v): i for i, v in enumerate(sorted(vs))} for c, vs in values.items()
    }


def save_vocab(vocab: dict[str, dict[str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, indent=2))


def load_vocab(path: Path) -> dict[str, dict[str, int]]:
    return json.loads(Path(path).read_text())


def vocab_fingerprint(vocab: dict[str, dict[str, int]]) -> str:
    """A short, stable hash of a vocab's exact value -> id mapping.

    ``build_vocab`` assigns ids by position in the sorted value set
    (``enumerate(sorted(vs))``), so a vocabulary that changes size or
    content renumbers every value sorting after an inserted one -- a
    rebuild can silently produce a *different* vocabulary that still looks
    valid (same columns, every id still in range). This has already
    happened once on this project: the 2026-08-17 rebuild took L7_PROTO
    from 216 entries to 239, which means anything built against the old
    vocab (e.g. unseen-attack graphs, or a checkpoint's trained embedding
    table) carries service ids that are still in-range but now point at
    different services.

    Stored in a trained checkpoint's config (``trench_ids.cl.train.main``,
    ``trench_ids.cl.train_flat.main``) and re-checked against the
    ``vocab.json`` an inference/unseen-data run actually loads
    (``trench_ids.cl.inference.load_checkpoint``), so a mismatch raises
    instead of silently mapping a Protocol/Service value to the wrong
    embedding row.
    """
    canonical = json.dumps(vocab, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
