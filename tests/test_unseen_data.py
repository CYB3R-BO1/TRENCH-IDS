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
    # Catches Finding 1: sampled should be True if ANY downsampling occurred
    # (periodic consolidation OR final downsample), not just based on final frame size.
    # With 525 rows, chunk_size=105, sample_cap=25:
    # - 5 chunks of 105 rows each → consolidation triggers after chunk 5 → downsamples to 25
    # - Final frame is exactly 25 rows (no additional final downsample)
    # - Old buggy code: final check sees 25 > 25? No → sampled=False (WRONG!)
    # - Fixed code: sampled_total=True from consolidation → sampled=True (CORRECT!)
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
    # Catches Finding 2: corrupted_rows_dropped should count ALL corrupted rows
    # across all consolidation cycles, not just the final frame.
    # Create 500+ rows with an inf value in an early chunk that gets consolidated,
    # plus normal rows after. The early inf should be counted in corrupted_rows_dropped.
    root = tmp_path / "raw"
    early_rows = _rows("ddos", 100)
    # Insert one corrupted row
    early_rows.append({"IN_BYTES": float("inf"), "Attack": "ddos"})
    # Add enough rows to trigger consolidation (threshold is 500 with sample_cap=25)
    later_rows = _rows("ddos", 450)
    _write_csv(root, "NF-BoT-IoT-v2", early_rows + later_rows)
    rng = np.random.default_rng(42)

    frame, manifest_entry = extract_unseen_class(
        root, "NF-BoT-IoT-v2", "BoT", "DDoS",
        chunk_size=150,
        sample_cap=25,
        rng=rng,
    )

    # The corrupted row should be counted even though it was dropped during
    # periodic consolidation, not the final consolidation
    assert manifest_entry["rows_found"] == 551
    assert manifest_entry["corrupted_rows_dropped"] == 1
    assert manifest_entry["rows_kept"] == 25


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
