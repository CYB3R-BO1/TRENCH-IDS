"""Tests for Step 10 unseen-attack data extraction (trench_ids.unseen_data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
