from __future__ import annotations

import json
from pathlib import Path

import pytest

from trench_ids.cl.transferability_report import (
    class_pair_summary,
    load_transferability_records,
    rank_class_pairs,
    rank_relations,
    relation_wise_summary,
    relation_wise_task_evolution,
)


def _write_task_file(run_dir: Path, task: int, content: dict) -> None:
    (run_dir / f"transferability_task_{task}.json").write_text(json.dumps(content))


def test_load_transferability_records_flattens_and_skips_empty_task1(tmp_path):
    _write_task_file(tmp_path, 1, {"Scanning": {}})
    _write_task_file(tmp_path, 2, {
        "Reconnaissance": {"Scanning": {"originates": 0.5, "unusual_relation": -0.2}},
    })

    records = load_transferability_records(tmp_path, num_tasks=2)

    assert records == [
        {"task": 2, "new_class": "Reconnaissance", "bank_class": "Scanning",
         "relation": "originates", "cosine": 0.5},
        {"task": 2, "new_class": "Reconnaissance", "bank_class": "Scanning",
         "relation": "unusual_relation", "cosine": -0.2},
    ]


def test_relation_wise_summary_matches_hand_computed_values():
    records = [
        {"task": 2, "new_class": "A", "bank_class": "B", "relation": "originates", "cosine": 0.2},
        {"task": 3, "new_class": "C", "bank_class": "B", "relation": "originates", "cosine": 0.6},
        {
            "task": 3,
            "new_class": "C",
            "bank_class": "A",
            "relation": "custom_relation",
            "cosine": -0.4,
        },
    ]

    summary = relation_wise_summary(records)

    assert set(summary.keys()) == {"originates", "custom_relation"}
    assert summary["originates"]["n"] == 2
    assert summary["originates"]["mean"] == pytest.approx(0.4)
    assert summary["originates"]["std"] == pytest.approx(0.2)  # pstdev([0.2, 0.6])
    assert summary["custom_relation"]["n"] == 1
    assert summary["custom_relation"]["mean"] == pytest.approx(-0.4)
    assert summary["custom_relation"]["std"] == pytest.approx(0.0)


def test_relation_wise_task_evolution_groups_by_relation_then_task():
    records = [
        {"task": 2, "new_class": "A", "bank_class": "B", "relation": "originates", "cosine": 0.2},
        {"task": 2, "new_class": "A", "bank_class": "C", "relation": "originates", "cosine": 0.4},
        {"task": 3, "new_class": "D", "bank_class": "B", "relation": "originates", "cosine": 0.9},
    ]

    evolution = relation_wise_task_evolution(records)

    assert evolution["originates"][2] == pytest.approx(0.3)
    assert evolution["originates"][3] == pytest.approx(0.9)


def test_rank_relations_sorts_descending_by_mean():
    summary = {
        "a": {"mean": 0.1, "std": 0.0, "n": 1},
        "b": {"mean": 0.9, "std": 0.0, "n": 1},
        "c": {"mean": 0.5, "std": 0.0, "n": 1},
    }

    ranking = rank_relations(summary)

    assert [row["relation"] for row in ranking] == ["b", "c", "a"]
    assert ranking[0] == {"relation": "b", "mean": 0.9, "std": 0.0, "n": 1}


def test_class_pair_summary_averages_across_relations_and_picks_best_worst():
    records = [
        {"task": 2, "new_class": "Reconnaissance", "bank_class": "Scanning",
         "relation": "originates", "cosine": 0.6},
        {"task": 2, "new_class": "Reconnaissance", "bank_class": "Scanning",
         "relation": "terminated_by", "cosine": -0.2},
    ]

    summary = class_pair_summary(records)
    key = tuple(sorted(("Reconnaissance", "Scanning")))

    assert set(summary.keys()) == {key}
    entry = summary[key]
    assert entry["new_class"] == "Reconnaissance"
    assert entry["bank_class"] == "Scanning"
    assert entry["mean_across_relations"] == pytest.approx(0.2)
    assert entry["best_relation"] == "originates"
    assert entry["worst_relation"] == "terminated_by"
    assert entry["per_relation"] == {"originates": 0.6, "terminated_by": -0.2}


def test_class_pair_summary_key_is_order_independent():
    records_a = [{"task": 2, "new_class": "X", "bank_class": "Y", "relation": "r", "cosine": 0.5}]
    records_b = [{"task": 2, "new_class": "Y", "bank_class": "X", "relation": "r", "cosine": 0.5}]

    assert set(class_pair_summary(records_a).keys()) == set(class_pair_summary(records_b).keys())


def test_class_pair_summary_omits_missing_relation_rather_than_zero_filling():
    records = [
        {"task": 2, "new_class": "A", "bank_class": "B", "relation": "originates", "cosine": 0.3},
    ]

    entry = next(iter(class_pair_summary(records).values()))

    assert "protocol_of" not in entry["per_relation"]


def test_rank_class_pairs_sorts_descending_by_mean_and_includes_per_relation():
    records = [
        {"task": 2, "new_class": "A", "bank_class": "B", "relation": "originates", "cosine": 0.1},
        {"task": 3, "new_class": "C", "bank_class": "D", "relation": "originates", "cosine": 0.9},
    ]

    ranking = rank_class_pairs(class_pair_summary(records))

    assert [row["mean"] for row in ranking] == pytest.approx([0.9, 0.1])
    assert ranking[0]["per_relation"] == {"originates": 0.9}
