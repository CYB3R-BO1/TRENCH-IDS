"""Tests for cross-dataset label harmonization (trench_ids.labels).

Raw-string inventories are the *measured* distinct ``Attack`` values in each
dataset (docs/datasets.md §3; BoT-IoT strings re-verified via the spike
script in the design spec, docs/superpowers/specs/2026-07-13-dataset-task-
respec-design.md §1). If a future dataset drop changes them, update both
places.
"""

from __future__ import annotations

import pytest

from trench_ids import labels
from trench_ids.labels import (
    BENIGN,
    CANONICAL_TO_TASK,
    CLASS_DATASETS,
    EXCLUDED_CLASSES,
    NUM_TASKS,
    RAW_TO_CANONICAL,
    TASK_DATASETS,
    TASK_THEMES,
    UnknownAttackLabel,
    canonical_label,
    task_of,
)

# Measured distinct raw Attack strings per dataset.
RAW_BY_DATASET = {
    "ToN": [
        "Benign", "scanning", "xss", "ddos", "password",
        "dos", "injection", "backdoor", "mitm", "ransomware",
    ],
    "CSE": [
        "Benign", "DDOS attack-HOIC", "DoS attacks-Hulk", "DDoS attacks-LOIC-HTTP",
        "Bot", "Infilteration", "SSH-Bruteforce", "DoS attacks-GoldenEye",
        "FTP-BruteForce", "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris",
        "Brute Force -Web", "DDOS attack-LOIC-UDP", "Brute Force -XSS", "SQL Injection",
    ],
    "BoT": ["Benign", "DDoS", "DoS", "Reconnaissance", "Theft"],
}

CANDIDATE_CLASSES = {
    "Scanning", "XSS", "Password", "DDoS", "DoS", "Injection",
    "Reconnaissance", "Bot", "BruteForce", "Infiltration",
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
    assert canonical_label("DoS attacks-Hulk") == canonical_label("dos") == "DoS"
    assert (
        canonical_label("ddos")
        == canonical_label("DDOS attack-HOIC")
        == canonical_label("DDoS attacks-LOIC-HTTP")
        == "DDoS"
    )


def test_bot_iot_raw_strings_map_correctly() -> None:
    assert canonical_label("DDoS") == "DDoS"
    assert canonical_label("DoS") == "DoS"
    assert canonical_label("Reconnaissance") == "Reconnaissance"
    assert canonical_label("Theft") == "Theft"


def test_class_datasets_covers_every_candidate_class() -> None:
    assert set(CLASS_DATASETS) == CANDIDATE_CLASSES


def test_class_datasets_restricts_bot_iot_to_reconnaissance_only() -> None:
    # BoT-IoT also contains DDoS/DoS-labeled rows, but its only sanctioned
    # contribution is Reconnaissance -- those rows must never be attributed
    # to the DDoS/DoS classes.
    assert CLASS_DATASETS["Reconnaissance"] == {"BoT"}
    assert "BoT" not in CLASS_DATASETS["DDoS"]
    assert "BoT" not in CLASS_DATASETS["DoS"]


def test_class_datasets_matches_two_or_three_way_classes() -> None:
    assert CLASS_DATASETS["Scanning"] == {"ToN"}
    assert CLASS_DATASETS["XSS"] == {"ToN"}
    assert CLASS_DATASETS["Password"] == {"ToN"}
    assert CLASS_DATASETS["DDoS"] == {"ToN", "CSE"}
    assert CLASS_DATASETS["DoS"] == {"ToN", "CSE"}
    assert CLASS_DATASETS["Injection"] == {"ToN", "CSE"}
    assert CLASS_DATASETS["Bot"] == {"CSE"}
    assert CLASS_DATASETS["BruteForce"] == {"CSE"}
    assert CLASS_DATASETS["Infiltration"] == {"CSE"}


def test_every_candidate_class_has_a_task() -> None:
    for c in CANDIDATE_CLASSES:
        assert c in CANONICAL_TO_TASK, f"{c} missing from CANONICAL_TO_TASK"


def test_excluded_classes_have_no_task() -> None:
    # Backdoor/MITM/Ransomware/Web Attacks: below the candidate-pool threshold.
    # Theft: BoT-IoT class not in the 10-class candidate pool.
    assert EXCLUDED_CLASSES == frozenset(
        {"Backdoor", "MITM", "Ransomware", "Web Attacks", "Theft"}
    )
    assert canonical_label("mitm") == "MITM"
    assert canonical_label("Theft") == "Theft"
    for c in EXCLUDED_CLASSES:
        assert c not in CANONICAL_TO_TASK
        assert labels.is_excluded(c)
    # Bot/BruteForce/Infiltration are NOT excluded any more (reversal from
    # the prior 4-task design) -- they're in the candidate pool now.
    for c in ("Bot", "BruteForce", "Infiltration"):
        assert c not in EXCLUDED_CLASSES
        assert not labels.is_excluded(c)


def test_benign_has_no_task() -> None:
    assert BENIGN not in CANONICAL_TO_TASK
    with pytest.raises(UnknownAttackLabel):
        task_of(BENIGN)


def test_tasks_are_1_indexed_1_to_6() -> None:
    assert set(CANONICAL_TO_TASK.values()) == set(range(1, 7))
    assert set(TASK_THEMES) == set(range(1, 7))
    assert NUM_TASKS == 6
    assert len(TASK_THEMES) == 6


def test_every_task_has_theme_and_datasets() -> None:
    for task in TASK_THEMES:
        assert TASK_THEMES[task]
        assert TASK_DATASETS[task]


def test_attack_classes_for_task_matches_locked_six_task_table() -> None:
    # Isolate-and-bundle grouping from the spike computation (design spec §2):
    # threshold 0.35, stable across 0.21-0.55.
    for task in TASK_THEMES:
        assert BENIGN not in labels.attack_classes_for_task(task)
    assert set(labels.attack_classes_for_task(1)) == {"Scanning"}
    assert set(labels.attack_classes_for_task(2)) == {"Reconnaissance"}
    assert set(labels.attack_classes_for_task(3)) == {"XSS", "DDoS"}
    assert set(labels.attack_classes_for_task(4)) == {"Password", "Infiltration"}
    assert set(labels.attack_classes_for_task(5)) == {"DoS", "Injection"}
    assert set(labels.attack_classes_for_task(6)) == {"Bot", "BruteForce"}
    for c in EXCLUDED_CLASSES:
        assert all(c not in labels.attack_classes_for_task(t) for t in TASK_THEMES)


def test_task_datasets_reflect_single_vs_multi_dataset_tasks() -> None:
    # T1 (Scanning) is ToN-only, T2 (Reconnaissance) is BoT-IoT-only, T6
    # (Bot+BruteForce) is CSE-only; T3/T4/T5 span both ToN and CSE.
    assert TASK_DATASETS[1] == ("ToN",)
    assert TASK_DATASETS[2] == ("BoT",)
    assert set(TASK_DATASETS[3]) == {"ToN", "CSE"}
    assert set(TASK_DATASETS[4]) == {"ToN", "CSE"}
    assert set(TASK_DATASETS[5]) == {"ToN", "CSE"}
    assert TASK_DATASETS[6] == ("CSE",)


def test_raw_to_canonical_has_no_gaps_for_configured_datasets() -> None:
    # Every raw string measured across ToN/CSE/BoT must appear (used as a
    # regression guard against a future dataset drop losing an entry).
    all_raws = {r for raws in RAW_BY_DATASET.values() for r in raws}
    assert all_raws <= set(RAW_TO_CANONICAL)
