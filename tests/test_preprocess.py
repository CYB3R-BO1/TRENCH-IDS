"""Tests for Step 1 CLASS_DATASETS restriction and flow_id assignment
(trench_ids.preprocess)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trench_ids.preprocess import (
    _drop_corrupted_rows,
    _waterfill,
    compute_quotas,
    pass1_counts,
    pass2_sample,
)


def _write_csv(root: Path, dir_name: str, rows: list[tuple[int, str]]) -> None:
    data_dir = root / dir_name / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["IN_BYTES", "Attack"]).to_csv(
        data_dir / f"{dir_name}.csv", index=False
    )


def _cfg(root: Path) -> dict:
    return {
        "seed": 1,
        "paths": {"raw_dir": str(root), "out_dir": str(root / "out")},
        "datasets": {"NF-ToN-IoT-v2": "ToN", "NF-BoT-IoT-v2": "BoT"},
        "sampling": {
            "attack_per_task": 100,
            "benign_per_task": 100,
            "chunk_size": 2,
        },
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "dedup": True,
    }


@pytest.fixture
def two_dataset_root(tmp_path: Path) -> Path:
    # ToN rows: ddos, ddos, scanning, Benign -- all allowed (Scanning/DDoS
    # are ToN-sanctioned per trench_ids.labels.CLASS_DATASETS).
    _write_csv(
        tmp_path,
        "NF-ToN-IoT-v2",
        [(100, "ddos"), (100, "ddos"), (50, "scanning"), (10, "Benign")],
    )
    # BoT rows: DDoS, DDoS (NOT allowed -- BoT-IoT's only sanctioned class
    # is Reconnaissance), Reconnaissance (allowed), Benign (always allowed).
    _write_csv(
        tmp_path,
        "NF-BoT-IoT-v2",
        [(200, "DDoS"), (200, "DDoS"), (30, "Reconnaissance"), (5, "Benign")],
    )
    return tmp_path


def test_pass1_counts_drops_disallowed_dataset_class_combo(two_dataset_root: Path) -> None:
    cfg = _cfg(two_dataset_root)
    attack_counts, benign_counts, original_cols = pass1_counts(cfg)

    # BoT-IoT's 2 DDoS rows are NOT counted -- DDoS is ToN/CSE-only.
    # Counts are keyed by (class, dataset) so compute_quotas can split a
    # multi-source class's quota by per-source availability.
    assert attack_counts == Counter(
        {("DDoS", "ToN"): 2, ("Scanning", "ToN"): 1, ("Reconnaissance", "BoT"): 1}
    )
    assert benign_counts == Counter({"ToN": 1, "BoT": 1})
    assert original_cols == ["IN_BYTES", "Attack"]


def test_pass2_sample_drops_disallowed_and_assigns_flow_id(two_dataset_root: Path) -> None:
    cfg = _cfg(two_dataset_root)
    attack_counts, benign_counts, _ = pass1_counts(cfg)
    rng = np.random.default_rng(cfg["seed"])
    quotas = compute_quotas(attack_counts, benign_counts, 100, 100)

    attacks, benign_pools = pass2_sample(cfg, attack_counts, benign_counts, quotas, rng)

    # 4 kept attack rows: ToN ddos x2 + ToN scanning x1 + BoT reconnaissance
    # x1. BoT's 2 DDoS rows are dropped entirely -- a hard class-dataset
    # restriction, not a probability.
    assert len(attacks) == 4
    assert set(attacks["canonical_label"]) == {"DDoS", "Scanning", "Reconnaissance"}
    assert attacks[attacks["source_dataset"] == "BoT"]["canonical_label"].tolist() == [
        "Reconnaissance"
    ]

    # flow_id = f"{dataset_code}-{original_csv_row_number}", 0-indexed
    # excluding header, stable across the chunked read (chunk_size=2 forces
    # 2 chunks per file).
    ton_rows = attacks[attacks["source_dataset"] == "ToN"].sort_values("flow_id")
    assert ton_rows["flow_id"].tolist() == ["ToN-0", "ToN-1", "ToN-2"]
    bot_rows = attacks[attacks["source_dataset"] == "BoT"]
    assert bot_rows["flow_id"].tolist() == ["BoT-2"]  # Reconnaissance is CSV row 2
    assert attacks["flow_id"].is_unique

    assert len(benign_pools["ToN"]) == 1
    assert len(benign_pools["BoT"]) == 1


def test_pass2_sample_keeps_every_allowed_row_when_quota_exceeds_supply(
    two_dataset_root: Path,
) -> None:
    """A quota larger than the data keeps everything allowed, and no more."""
    cfg = _cfg(two_dataset_root)
    attack_counts, benign_counts, _ = pass1_counts(cfg)
    rng = np.random.default_rng(cfg["seed"])
    quotas = compute_quotas(attack_counts, benign_counts, 100, 100)

    attacks, _ = pass2_sample(cfg, attack_counts, benign_counts, quotas, rng)

    assert (attacks["canonical_label"] == "DDoS").sum() == 2


def test_pass2_sample_honours_a_binding_quota_exactly(two_dataset_root: Path) -> None:
    """A quota below the available supply is met exactly, not approximately.

    Selection is planned from Pass 1's counts rather than drawn per row, so
    realised counts cannot drift off the plan -- which is what keeps every
    task the same size.
    """
    cfg = _cfg(two_dataset_root)
    attack_counts, benign_counts, _ = pass1_counts(cfg)
    rng = np.random.default_rng(cfg["seed"])
    quotas = compute_quotas(attack_counts, benign_counts, attack_per_task=1, benign_per_task=1)

    attacks, benign_pools = pass2_sample(cfg, attack_counts, benign_counts, quotas, rng)

    # Task 3 (DDoS + Infiltration) gets 1 attack row total; only DDoS exists
    # in this fixture, so DDoS supplies it.
    assert (attacks["canonical_label"] == "DDoS").sum() == 1
    # T1 draws 1 benign from ToN, T2 draws 1 from BoT.
    assert len(benign_pools["ToN"]) == 1
    assert len(benign_pools["BoT"]) == 1


def test_waterfill_saturates_short_keys_and_redistributes() -> None:
    assert _waterfill({"rare": 5, "common": 1000}, 100) == {"rare": 5, "common": 95}
    assert _waterfill({"a": 3, "b": 3}, 100) == {"a": 3, "b": 3}
    assert sum(_waterfill({"a": 100, "b": 100, "c": 100}, 10).values()) == 10


def test_compute_quotas_gives_every_task_the_same_budget() -> None:
    attack_counts = Counter(
        {
            ("Scanning", "ToN"): 3_781_419,
            ("Reconnaissance", "BoT"): 2_620_999,
            ("DDoS", "ToN"): 2_030_232,
            ("DDoS", "CSE"): 1_386_220,
            ("Infiltration", "CSE"): 116_361,
            ("DoS", "ToN"): 712_609,
            ("DoS", "CSE"): 483_999,
            ("Injection", "ToN"): 684_897,
            ("Injection", "CSE"): 68_280,
            ("Password", "ToN"): 1_153_323,
            ("Bot", "CSE"): 143_097,
            ("XSS", "ToN"): 2_455_020,
            ("BruteForce", "CSE"): 120_912,
        }
    )
    benign_counts = Counter({"ToN": 6_099_469, "CSE": 16_635_567, "BoT": 135_037})

    quotas = compute_quotas(attack_counts, benign_counts, 390_000, 130_000)

    for task, by_class in quotas["attack_per_task_by_class"].items():
        assert sum(by_class.values()) == 390_000, task
        assert sum(quotas["benign_alloc"][task].values()) == 130_000, task
    # The rare classes are taken in full rather than sampled at their natural
    # (~30:1) proportion, which is the point of the waterfill.
    assert quotas["attack_per_task_by_class"][3]["Infiltration"] == 116_361
    assert quotas["attack_per_task_by_class"][6]["BruteForce"] == 120_912
    # BoT-IoT's whole benign supply is the binding constraint on T2.
    assert quotas["benign_alloc"][2] == {"BoT": 130_000}


def test_drop_corrupted_rows_removes_overflow_and_nonfinite_values() -> None:
    frame = pd.DataFrame(
        {
            "IN_BYTES": [100.0, 6.588e304, np.inf, np.nan, 200.0],
            "OUT_BYTES": [50.0, 50.0, 50.0, 50.0, 50.0],
            "Attack": ["ddos"] * 5,
        }
    )
    original_cols = ["IN_BYTES", "OUT_BYTES", "Attack"]

    clean, dropped = _drop_corrupted_rows(frame, original_cols)

    # 3 bad rows removed: float32-overflow (6.588e304), +inf, NaN. The
    # non-numeric "Attack" column is untouched by the numeric-overflow/
    # non-finite check.
    assert dropped == 3
    assert clean["IN_BYTES"].tolist() == [100.0, 200.0]
    assert clean.index.tolist() == [0, 1]  # reset_index(drop=True)


def test_drop_corrupted_rows_keeps_everything_when_clean() -> None:
    frame = pd.DataFrame({"IN_BYTES": [100.0, 200.0], "OUT_BYTES": [50.0, 60.0]})

    clean, dropped = _drop_corrupted_rows(frame, ["IN_BYTES", "OUT_BYTES"])

    assert dropped == 0
    assert len(clean) == 2


def test_drop_corrupted_rows_negative_overflow_also_removed() -> None:
    frame = pd.DataFrame({"IN_BYTES": [100.0, -6.588e304], "OUT_BYTES": [50.0, 50.0]})

    clean, dropped = _drop_corrupted_rows(frame, ["IN_BYTES", "OUT_BYTES"])

    assert dropped == 1
    assert clean["IN_BYTES"].tolist() == [100.0]
