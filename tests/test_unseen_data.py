"""Tests for Step 10 unseen-attack data extraction (trench_ids.unseen_data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trench_ids.unseen_data import extract_unseen_class, run


def _write_csv(root: Path, dir_name: str, rows: list[dict]) -> Path:
    data_dir = root / dir_name / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{dir_name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _rows(attack: str, n: int, in_bytes: float = 100.0) -> list[dict]:
    return [{"IN_BYTES": in_bytes, "Attack": attack} for _ in range(n)]


def test_extract_unseen_class_keeps_only_matching_label(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_csv(
        root,
        "NF-ToN-IoT-v2",
        _rows("backdoor", 3) + _rows("scanning", 2) + _rows("Benign", 1),
    )
    rng = np.random.default_rng(1)

    frame, manifest_entry = extract_unseen_class(
        root, "NF-ToN-IoT-v2", "ToN", "Backdoor",
        chunk_size=2, sample_cap=None, rng=rng,
    )

    assert len(frame) == 3
    assert set(frame["canonical_label"]) == {"Backdoor"}
    assert manifest_entry["rows_found"] == 3
    assert manifest_entry["rows_kept"] == 3
    assert manifest_entry["corrupted_rows_dropped"] == 0
    assert manifest_entry["sampled"] is False
    # flow_id = f"{dataset_code}-{original_csv_row_number}", 0-indexed
    assert sorted(frame["flow_id"].tolist()) == ["ToN-0", "ToN-1", "ToN-2"]


def test_extract_unseen_class_drops_corrupted_rows(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    rows = _rows("mitm", 2)
    rows.append({"IN_BYTES": float("inf"), "Attack": "mitm"})
    _write_csv(root, "NF-ToN-IoT-v2", rows)
    rng = np.random.default_rng(1)

    frame, manifest_entry = extract_unseen_class(
        root, "NF-ToN-IoT-v2", "ToN", "MITM",
        chunk_size=10, sample_cap=None, rng=rng,
    )

    assert len(frame) == 2
    assert manifest_entry["rows_found"] == 3
    assert manifest_entry["corrupted_rows_dropped"] == 1
    assert manifest_entry["rows_kept"] == 2


def test_extract_unseen_class_applies_sample_cap(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_csv(root, "NF-BoT-IoT-v2", _rows("DDoS", 20))
    rng = np.random.default_rng(1)

    frame, manifest_entry = extract_unseen_class(
        root, "NF-BoT-IoT-v2", "BoT", "DDoS",
        chunk_size=5, sample_cap=5, rng=rng,
    )

    assert len(frame) == 5
    assert manifest_entry["rows_found"] == 20
    assert manifest_entry["rows_kept"] == 5
    assert manifest_entry["sampled"] is True


def test_extract_unseen_class_bypasses_class_datasets_restriction(tmp_path: Path) -> None:
    # BoT-IoT's DDoS rows are NOT CLASS_DATASETS-allowed for BoT (only
    # Reconnaissance is) -- extract_unseen_class must keep them anyway,
    # since bypassing that restriction is the entire point of this module.
    root = tmp_path / "raw"
    _write_csv(root, "NF-BoT-IoT-v2", _rows("DDoS", 4))
    rng = np.random.default_rng(1)

    frame, manifest_entry = extract_unseen_class(
        root, "NF-BoT-IoT-v2", "BoT", "DDoS",
        chunk_size=10, sample_cap=None, rng=rng,
    )

    assert len(frame) == 4


def test_extract_unseen_class_skips_unmapped_labels(tmp_path: Path) -> None:
    # Unmapped labels (e.g., UNSW-NB15's Exploits, Fuzzers) should not cause
    # extract_unseen_class to crash; they should be silently skipped. This
    # reproduces the critical bug fix: pre-filtering to RAW_TO_CANONICAL.
    root = tmp_path / "raw"
    rows = (
        _rows("Exploits", 2)  # unmapped, not in RAW_TO_CANONICAL
        + _rows("dos", 3)     # mapped, in RAW_TO_CANONICAL
        + _rows("Fuzzers", 1)  # unmapped
    )
    _write_csv(root, "NF-UNSW-NB15-v2", rows)
    rng = np.random.default_rng(1)

    frame, manifest_entry = extract_unseen_class(
        root, "NF-UNSW-NB15-v2", "UNSW", "DoS",
        chunk_size=10, sample_cap=None, rng=rng,
    )

    # Should extract only the mapped dos/DoS rows, not crash on Exploits/Fuzzers
    assert len(frame) == 3
    assert set(frame["canonical_label"]) == {"DoS"}
    assert manifest_entry["rows_found"] == 3
    assert manifest_entry["rows_kept"] == 3


def test_extract_unseen_class_tracks_sampled_cumulatively(tmp_path: Path) -> None:
    # sampled should be True whenever the total clean matching-row count
    # exceeds sample_cap, regardless of how many chunks it took to get
    # there (i.e. reservoir sampling actually replaced rows along the way),
    # not just based on comparing the final frame's size to sample_cap
    # (which is always exactly sample_cap once sampling happens, so that
    # comparison alone can never distinguish "sampled" from "not sampled").
    # With 525 rows and sample_cap=25, the final frame is exactly 25 rows
    # either way -- the assertion below is what actually distinguishes them.
    root = tmp_path / "raw"
    _write_csv(root, "NF-BoT-IoT-v2", _rows("ddos", 525))
    rng = np.random.default_rng(42)

    frame, manifest_entry = extract_unseen_class(
        root, "NF-BoT-IoT-v2", "BoT", "DDoS",
        chunk_size=105,
        sample_cap=25,
        rng=rng,
    )

    assert len(frame) == 25
    assert manifest_entry["rows_found"] == 525
    assert manifest_entry["rows_kept"] == 25
    # This is the key assertion that would fail with the old buggy code
    assert manifest_entry["sampled"] is True


def test_extract_unseen_class_tracks_corrupted_cumulatively(tmp_path: Path) -> None:
    # corrupted_rows_dropped should count ALL corrupted rows across every
    # chunk processed during streaming (reservoir sampling touches many
    # chunks over the life of one extraction), not just the last one seen.
    # Create 500+ rows with an inf value in an early chunk, plus normal rows
    # after; the early inf should be counted in corrupted_rows_dropped.
    root = tmp_path / "raw"
    early_rows = _rows("ddos", 100)
    # Insert one corrupted row
    early_rows.append({"IN_BYTES": float("inf"), "Attack": "ddos"})
    later_rows = _rows("ddos", 450)
    _write_csv(root, "NF-BoT-IoT-v2", early_rows + later_rows)
    rng = np.random.default_rng(42)

    frame, manifest_entry = extract_unseen_class(
        root, "NF-BoT-IoT-v2", "BoT", "DDoS",
        chunk_size=150,
        sample_cap=25,
        rng=rng,
    )

    # The corrupted row should be counted even though it was seen in an
    # early chunk, long before the stream (and reservoir sampling) finishes.
    assert manifest_entry["rows_found"] == 551
    assert manifest_entry["corrupted_rows_dropped"] == 1
    assert manifest_entry["rows_kept"] == 25


def test_extract_unseen_class_sample_is_not_tail_biased(tmp_path: Path) -> None:
    # Finding #1 (Critical): the old periodic-consolidation approach drew
    # its final sample almost entirely from the last chunk read, i.e. one
    # contiguous slice near the end of the stream, not a uniform sample
    # across the full matching population. Use a monotonically increasing
    # id column (encoded via IN_BYTES) and enough rows to force many
    # reservoir-sampling decisions; a correct uniform sample's min index
    # should land near the start of the range and its max index near the
    # end. The old tail-biased implementation would have every sampled
    # index clustered in roughly the last `sample_cap / (sample_cap*20)`
    # fraction of the stream, failing both bounds below.
    root = tmp_path / "raw"
    n_rows = 20000
    rows = [{"IN_BYTES": float(i), "Attack": "ddos"} for i in range(n_rows)]
    _write_csv(root, "NF-BoT-IoT-v2", rows)
    sample_cap = 200
    rng = np.random.default_rng(7)

    frame, manifest_entry = extract_unseen_class(
        root, "NF-BoT-IoT-v2", "BoT", "DDoS",
        chunk_size=1000, sample_cap=sample_cap, rng=rng,
    )

    assert len(frame) == sample_cap
    assert manifest_entry["rows_found"] == n_rows
    assert manifest_entry["sampled"] is True

    sampled_indices = frame["IN_BYTES"].to_numpy()
    assert sampled_indices.min() < 0.20 * n_rows
    assert sampled_indices.max() > 0.80 * n_rows


def test_extract_unseen_class_reservoir_sampling_preserves_dtypes(tmp_path: Path) -> None:
    # The dict round-trip inside reservoir sampling (to_dict/from_records)
    # can silently upcast an integer column to float if not corrected -- and
    # this is a real per-chunk risk, not a from_records quirk: pandas infers
    # each CSV chunk's dtype independently, so a chunk containing one blank
    # PROTOCOL value (later dropped by _drop_corrupted_rows) gets PROTOCOL
    # parsed as float64 for that whole chunk, while an unaffected chunk gets
    # int64 -- mixing int-origin and float-origin dict records in the
    # reservoir then makes from_records infer float64 overall. That would
    # make unseen_graphs.drop_vocab_gaps's str(int) vocab lookup fail for
    # every row ("6.0" not in vocab), dropping the entire class as a false
    # "vocab gap".
    #
    # Write the CSV by hand (not via the DataFrame-round-trip _write_csv
    # helper, which would itself force a single float64 dtype across the
    # whole file and defeat the point of this test) so chunk 1's PROTOCOL
    # column is genuinely int-typed on disk and chunk 2's contains a blank.
    root = tmp_path / "raw"
    data_dir = root / "NF-BoT-IoT-v2" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    lines = ["IN_BYTES,PROTOCOL,Attack"]
    lines += [f"{float(i)},6,ddos" for i in range(5)]  # chunk 1: clean, int PROTOCOL
    lines.append("100.0,,ddos")  # chunk 2: blank PROTOCOL -> whole chunk parsed as float64
    lines += [f"{float(100 + i)},6,ddos" for i in range(4)]  # chunk 2: rest, still clean
    (data_dir / "NF-BoT-IoT-v2.csv").write_text("\n".join(lines) + "\n")
    rng = np.random.default_rng(3)

    # sample_cap large enough that every clean row is kept via the fill
    # phase alone (no random replacement needed) -- deterministic mix of
    # int-origin (chunk 1) and float-origin (chunk 2) reservoir rows.
    frame, manifest_entry = extract_unseen_class(
        root, "NF-BoT-IoT-v2", "BoT", "DDoS",
        chunk_size=5, sample_cap=20, rng=rng,
    )

    assert len(frame) == 9  # 10 rows total, 1 dropped for blank/non-finite PROTOCOL
    assert manifest_entry["corrupted_rows_dropped"] == 1
    assert frame["PROTOCOL"].dtype == np.int64
    assert (frame["PROTOCOL"] == 6).all()


def test_run_writes_parquet_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_csv(root, "NF-ToN-IoT-v2", _rows("backdoor", 3) + _rows("Benign", 2))
    out_dir = tmp_path / "out"
    config_path = tmp_path / "unseen.yaml"
    config_path.write_text(f"""
seed: 42
paths:
  raw_dir: {root}
  out_dir: {out_dir}
chunk_size: 500000
sample_cap: 5000
classes:
  - canonical_label: Backdoor
    dataset_dir: NF-ToN-IoT-v2
    dataset_code: ToN
    evaluation_type: unseen_class
""")

    manifest = run(config_path)

    assert (out_dir / "ToN_backdoor.parquet").exists()
    assert (out_dir / "manifest.json").exists()
    entry = manifest["classes"]["ToN_backdoor"]
    assert entry["rows_kept"] == 3
    assert entry["used_in_training"] is False
    assert entry["evaluation_type"] == "unseen_class"
