from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trench_ids.vocab import build_vocab, load_vocab, save_vocab


def test_build_vocab_assigns_stable_sorted_indices(tmp_path: Path) -> None:
    p1 = tmp_path / "task_1.parquet"
    p2 = tmp_path / "task_2.parquet"
    pd.DataFrame({"PROTOCOL": [17, 6], "L7_PROTO": [2, 1]}).to_parquet(p1)
    pd.DataFrame({"PROTOCOL": [6, 1], "L7_PROTO": [1, 5]}).to_parquet(p2)

    vocab = build_vocab([p1, p2])

    # Union of both files' values, sorted ascending -> 0, 1, 2, ...
    assert vocab["PROTOCOL"] == {"1": 0, "6": 1, "17": 2}
    assert vocab["L7_PROTO"] == {"1": 0, "2": 1, "5": 2}


def test_build_vocab_preserves_fractional_l7_proto_values(tmp_path: Path) -> None:
    p1 = tmp_path / "task_1.parquet"
    pd.DataFrame({"PROTOCOL": [6, 17], "L7_PROTO": [0.0, 5.119]}).to_parquet(p1)

    vocab = build_vocab([p1])

    # Keys must be the str() of the *native* float value, not a truncated int —
    # this is what graphs.py's _index_categorical + build_task_graph will look up.
    assert vocab["L7_PROTO"] == {"0.0": 0, "5.119": 1}
    assert vocab["PROTOCOL"] == {"6": 0, "17": 1}


def test_save_and_load_vocab_roundtrip(tmp_path: Path) -> None:
    vocab = {"PROTOCOL": {"6": 0, "17": 1}, "L7_PROTO": {"1": 0}}
    path = tmp_path / "vocab.json"

    save_vocab(vocab, path)
    loaded = load_vocab(path)

    assert loaded == vocab
    assert json.loads(path.read_text()) == vocab
