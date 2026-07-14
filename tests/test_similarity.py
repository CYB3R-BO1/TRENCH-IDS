"""Tests for per-dataset class-source restriction in trench_ids.similarity."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trench_ids.similarity import count_classes, sample_classes


def _write_csv(root: Path, dir_name: str, rows: list[tuple[int, str]]) -> None:
    data_dir = root / dir_name / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["IN_BYTES", "Attack"]).to_csv(
        data_dir / f"{dir_name}.csv", index=False
    )


@pytest.fixture
def two_dataset_root(tmp_path: Path) -> Path:
    _write_csv(
        tmp_path,
        "NF-ToN-IoT-v2",
        [(100, "ddos"), (100, "ddos"), (50, "scanning"), (10, "password"), (10, "password")],
    )
    _write_csv(
        tmp_path,
        "NF-BoT-IoT-v2",
        [(200, "DDoS"), (200, "DDoS"), (30, "Reconnaissance"), (20, "password")],
    )
    return tmp_path


def _cfg(root: Path) -> dict:
    return {
        "paths": {"raw_dir": str(root)},
        "chunk_size": 2,
        "samples_per_class": 5,
        "datasets": {"NF-ToN-IoT-v2": "ToN", "NF-BoT-IoT-v2": "BoT"},
    }


# "Password" is deliberately allowed in BOTH fixture datasets here (unlike the
# other three classes, each restricted to exactly one) so at least one test
# exercises multi-dataset pooling -- the actual value proposition of the
# per-dataset restriction feature: a class allowed in 2+ datasets must pool
# rows from ALL of them, not just the first match.
TEST_CLASS_DATASETS = {
    "DDoS": {"ToN"},
    "Scanning": {"ToN"},
    "Reconnaissance": {"BoT"},
    "Password": {"ToN", "BoT"},
}


_SINGLE_DATASET_CLASSES = ["DDoS", "Scanning", "Reconnaissance"]


def test_count_classes_restricts_to_allowed_datasets(two_dataset_root: Path) -> None:
    cfg = _cfg(two_dataset_root)
    counts = count_classes(cfg, _SINGLE_DATASET_CLASSES, TEST_CLASS_DATASETS)

    # BoT-IoT's 2 DDoS rows are not counted -- DDoS is ToN-only here.
    assert counts == Counter({"DDoS": 2, "Scanning": 1, "Reconnaissance": 1})


def test_sample_classes_restricts_to_allowed_datasets(two_dataset_root: Path) -> None:
    cfg = _cfg(two_dataset_root)
    counts = count_classes(cfg, _SINGLE_DATASET_CLASSES, TEST_CLASS_DATASETS)
    rng = np.random.default_rng(0)

    samples = sample_classes(cfg, _SINGLE_DATASET_CLASSES, counts, rng, TEST_CLASS_DATASETS)

    assert len(samples["DDoS"]) == 2
    assert set(samples["DDoS"]["source_dataset"]) == {"ToN"}
    assert len(samples["Reconnaissance"]) == 1
    assert set(samples["Reconnaissance"]["source_dataset"]) == {"BoT"}


def test_count_and_sample_pool_across_multiple_allowed_datasets(two_dataset_root: Path) -> None:
    """Password is allowed in BOTH ToN and BoT in TEST_CLASS_DATASETS (unlike
    DDoS/Scanning/Reconnaissance, each restricted to one dataset above). This
    is the feature's actual value proposition: a class allowed in 2+ datasets
    must pool rows from ALL of them, not just the first dataset that matches
    -- a regression that only took the first matching dataset would not be
    caught by the single-dataset-restriction tests above.
    """
    cfg = _cfg(two_dataset_root)
    counts = count_classes(cfg, ["Password"], TEST_CLASS_DATASETS)

    # 2 "password" rows in ToN + 1 in BoT, summed across both datasets.
    assert counts == Counter({"Password": 3})

    rng = np.random.default_rng(0)
    samples = sample_classes(cfg, ["Password"], counts, rng, TEST_CLASS_DATASETS)

    assert len(samples["Password"]) == 3
    # Both source datasets must be represented, not just one.
    assert set(samples["Password"]["source_dataset"]) == {"ToN", "BoT"}
    assert Counter(samples["Password"]["source_dataset"]) == Counter({"ToN": 2, "BoT": 1})


def test_count_and_sample_default_to_labels_class_datasets(two_dataset_root: Path) -> None:
    """No class_datasets arg -- must fall back to trench_ids.labels.CLASS_DATASETS."""
    from trench_ids.labels import CLASS_DATASETS as REAL_CLASS_DATASETS

    assert REAL_CLASS_DATASETS["DDoS"] == {"ToN", "CSE"}
    assert REAL_CLASS_DATASETS["Reconnaissance"] == {"BoT"}

    cfg = _cfg(two_dataset_root)
    # DDoS is real-world ToN+CSE-allowed, but only ToN is present in this
    # fixture's cfg["datasets"], so BoT's DDoS rows must still be dropped.
    counts = count_classes(cfg, ["DDoS", "Reconnaissance"])
    assert counts == Counter({"DDoS": 2, "Reconnaissance": 1})

    # sample_classes must fall back identically -- exercise it with no
    # class_datasets arg too (this previously went untested: the function's
    # own default-fallback ternary line was never executed by this test).
    rng = np.random.default_rng(0)
    samples = sample_classes(cfg, ["DDoS", "Reconnaissance"], counts, rng)

    assert len(samples["DDoS"]) == 2
    assert set(samples["DDoS"]["source_dataset"]) == {"ToN"}
    assert len(samples["Reconnaissance"]) == 1
    assert set(samples["Reconnaissance"]["source_dataset"]) == {"BoT"}
