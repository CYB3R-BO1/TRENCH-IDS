"""Tests for cross-dataset label harmonization (trench_ids.labels).

The raw-string inventories below are the *measured* distinct ``Attack`` values
in each dataset (full-file scans, 2026-07-10, recorded in docs/datasets.md §3).
If a future dataset drop changes them, update both places.
"""

from __future__ import annotations

import pytest

from trench_ids import labels
from trench_ids.labels import (
    BENIGN,
    CANONICAL_TO_TASK,
    EXCLUDED_CLASSES,
    RAW_TO_CANONICAL,
    TASK_DATASETS,
    TASK_THEMES,
    UnknownAttackLabel,
    canonical_label,
    task_of,
)

# Measured distinct raw Attack strings per dataset.
RAW_BY_DATASET = {
    "UNSW": [
        "Benign", "Exploits", "Fuzzers", "Generic", "Reconnaissance",
        "DoS", "Analysis", "Backdoor", "Shellcode", "Worms",
    ],
    "ToN": [
        "Benign", "scanning", "xss", "ddos", "password",
        "dos", "injection", "backdoor", "mitm", "ransomware",
    ],
    "BoT": ["Benign", "DDoS", "DoS", "Reconnaissance", "Theft"],
    "CSE": [
        "Benign", "DDOS attack-HOIC", "DoS attacks-Hulk", "DDoS attacks-LOIC-HTTP",
        "Bot", "Infilteration", "SSH-Bruteforce", "DoS attacks-GoldenEye",
        "FTP-BruteForce", "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris",
        "Brute Force -Web", "DDOS attack-LOIC-UDP", "Brute Force -XSS", "SQL Injection",
    ],
}


@pytest.mark.parametrize("dataset,raws", RAW_BY_DATASET.items())
def test_every_measured_raw_string_maps(dataset: str, raws: list[str]) -> None:
    for raw in raws:
        assert canonical_label(raw)  # non-empty canonical, no exception


def test_unknown_label_raises() -> None:
    with pytest.raises(UnknownAttackLabel):
        canonical_label("TotallyNewAttack")


def test_infiltration_misspelling_is_normalized() -> None:
    assert canonical_label("Infilteration") == "Infiltration"


def test_case_variants_collapse_to_same_class() -> None:
    # DoS/dos and DDoS/ddos/DDOS all collapse across datasets.
    assert canonical_label("DoS") == canonical_label("dos") == "DoS"
    assert (
        canonical_label("DDoS")
        == canonical_label("ddos")
        == canonical_label("DDOS attack-HOIC")
        == "DDoS"
    )


def test_every_attack_class_has_a_task() -> None:
    # Active attack classes (excluding Benign and dropped classes) all have a task.
    active = set(RAW_TO_CANONICAL.values()) - {BENIGN} - set(EXCLUDED_CLASSES)
    for c in active:
        assert c in CANONICAL_TO_TASK, f"{c} missing from CANONICAL_TO_TASK"


def test_excluded_classes_have_no_task() -> None:
    # Below the 100K-sample candidate-pool threshold (docs/attack-class-counts.md)
    # or (Worms) too few samples for a held-out split — still map canonically
    # (RAW_TO_CANONICAL never fails) but are assigned no task.
    assert EXCLUDED_CLASSES == frozenset(
        {
            "Worms",
            "MITM",
            "Ransomware",
            "Web Attacks",
            "Theft",
            "Analysis",
            "Shellcode",
            "Exploits",
            "Fuzzers",
            "Backdoor",
            "Generic",
        }
    )
    assert canonical_label("Worms") == "Worms"
    for c in EXCLUDED_CLASSES:
        assert c not in CANONICAL_TO_TASK
        assert labels.is_excluded(c)


def test_benign_has_no_task() -> None:
    # Benign is the shared negative class present in every task, not a task itself.
    assert BENIGN not in CANONICAL_TO_TASK
    with pytest.raises(UnknownAttackLabel):
        task_of(BENIGN)


def test_tasks_are_1_indexed_1_to_6() -> None:
    assert set(CANONICAL_TO_TASK.values()) == set(range(1, 7))
    assert set(TASK_THEMES) == set(range(1, 7))
    assert len(TASK_THEMES) == 6


def test_every_task_has_theme_and_datasets() -> None:
    for task in TASK_THEMES:
        assert TASK_THEMES[task]
        assert TASK_DATASETS[task]


def test_attack_classes_for_task_excludes_benign() -> None:
    for task in TASK_THEMES:
        assert BENIGN not in labels.attack_classes_for_task(task)
    # Isolate-and-bundle assignment (docs/attack-class-counts.md, task_design.py):
    # classes forming a mutual "conflict clique" (DDoS, Reconnaissance) each get
    # a singleton task; the rest are paired via minimum-weight matching, which
    # deliberately splits up the classes the professor flagged as
    # too-similar-to-co-locate (DoS/DDoS, Reconnaissance/Scanning,
    # BruteForce/Injection).
    assert set(labels.attack_classes_for_task(1)) == {"DDoS"}
    assert set(labels.attack_classes_for_task(2)) == {"Reconnaissance"}
    assert set(labels.attack_classes_for_task(3)) == {"DoS", "Injection"}
    assert set(labels.attack_classes_for_task(4)) == {"Scanning", "BruteForce"}
    assert set(labels.attack_classes_for_task(5)) == {"XSS", "Bot"}
    assert set(labels.attack_classes_for_task(6)) == {"Password", "Infiltration"}
    # Excluded classes appear in no task.
    for c in EXCLUDED_CLASSES:
        assert all(c not in labels.attack_classes_for_task(t) for t in TASK_THEMES)
