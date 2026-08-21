from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trench_ids.vocab import build_vocab, load_vocab, save_vocab, vocab_fingerprint, vocab_key


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

    # Fractional values must keep their exact str() -- never truncated --
    # while integral floats normalise to their integer string (vocab_key):
    # this is what graphs.py's build_task_graph looks up via vocab_key too,
    # so both sides agree either way.
    assert vocab["L7_PROTO"] == {"0": 0, "5.119": 1}
    assert vocab["PROTOCOL"] == {"6": 0, "17": 1}


def test_vocab_key_unifies_int_and_float_coercions_of_one_value() -> None:
    """The failure mode vocab_key exists for: one NaN forces a file's integer
    column to float64, so the same protocol arrives as 6 (int) from one file
    and 6.0 (float) from another. Both must key identically or the value
    splits into two vocab entries and two Protocol node identities."""
    assert vocab_key(6) == vocab_key(6.0) == "6"
    assert vocab_key(5.119) == "5.119"  # fractional floats are never truncated
    assert vocab_key("tcp") == "tcp"


def test_build_vocab_merges_across_float_coerced_files(tmp_path: Path) -> None:
    p1 = tmp_path / "task_1.parquet"  # int64 column
    p2 = tmp_path / "task_2.parquet"  # same values, float64 via one NaN
    pd.DataFrame({"PROTOCOL": [6, 17], "L7_PROTO": [2, 1]}).to_parquet(p1)
    pd.DataFrame({"PROTOCOL": [6.0, 17.0], "L7_PROTO": [1.0, 2.0]}).to_parquet(p2)

    vocab = build_vocab([p1, p2])

    assert vocab["PROTOCOL"] == {"6": 0, "17": 1}
    assert vocab["L7_PROTO"] == {"1": 0, "2": 1}


def test_save_and_load_vocab_roundtrip(tmp_path: Path) -> None:
    vocab = {"PROTOCOL": {"6": 0, "17": 1}, "L7_PROTO": {"1": 0}}
    path = tmp_path / "vocab.json"

    save_vocab(vocab, path)
    loaded = load_vocab(path)

    assert loaded == vocab
    assert json.loads(path.read_text()) == vocab


def test_vocab_fingerprint_is_stable_for_the_same_content() -> None:
    vocab = {"PROTOCOL": {"6": 0, "17": 1}, "L7_PROTO": {"1": 0}}

    assert vocab_fingerprint(vocab) == vocab_fingerprint(dict(vocab))


def test_vocab_fingerprint_is_stable_across_key_order() -> None:
    """A rebuild re-serialising the same mapping in a different dict order
    must not look like a different vocabulary."""
    a = {"PROTOCOL": {"6": 0, "17": 1}, "L7_PROTO": {"1": 0}}
    b = {"L7_PROTO": {"1": 0}, "PROTOCOL": {"17": 1, "6": 0}}

    assert vocab_fingerprint(a) == vocab_fingerprint(b)


def test_vocab_fingerprint_changes_when_ids_are_renumbered() -> None:
    """The exact failure mode this exists to catch: a rebuild that inserts
    a new value shifts every id sorting after it, changing the mapping
    while every id stays in-range -- shape alone would not catch this."""
    before = {"PROTOCOL": {"6": 0, "17": 1}}
    after = {"PROTOCOL": {"1": 0, "6": 1, "17": 2}}  # "1" inserted, others shifted

    assert vocab_fingerprint(before) != vocab_fingerprint(after)
