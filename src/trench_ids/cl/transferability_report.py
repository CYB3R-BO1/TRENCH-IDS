"""Step 10a — transferability analysis (docs/Updated_Proposed_Methodology_
Transferable_Representation_Learning.docx, Step 10, first two bullets: which
learned representations transfer across attack classes, and how relation-wise
transferability evolves across successive tasks). Consolidates the frozen
runs/step4 baseline's per-task transferability_task_*.json files -- no new
training run is required, all source data already exists on disk.

The prediction/inference pipeline (Step 10's third bullet) is explicitly out
of scope here -- deferred to a separate future round.

Run: python -m trench_ids.cl.transferability_report
  or: trench-transferability-report
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from trench_ids.labels import NUM_TASKS


def load_transferability_records(
    run_dir: Path, num_tasks: int = NUM_TASKS
) -> list[dict[str, Any]]:
    """Flattens transferability_task_{1..num_tasks}.json into rows. Task 1
    contributes nothing (empty bank -- {"Scanning": {}} in the real data).
    Each unordered class pair appears exactly once, in the direction
    dictated by task order (the newer class is always "new_class")."""
    records: list[dict[str, Any]] = []
    run_dir = Path(run_dir)
    for task in range(1, num_tasks + 1):
        data = json.loads((run_dir / f"transferability_task_{task}.json").read_text())
        for new_class, bank_entries in data.items():
            for bank_class, per_relation in bank_entries.items():
                for relation, cosine in per_relation.items():
                    records.append({
                        "task": task,
                        "new_class": new_class,
                        "bank_class": bank_class,
                        "relation": relation,
                        "cosine": cosine,
                    })
    return records


def relation_wise_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """{relation: {"mean", "std", "n"}} across every recorded pair,
    regardless of task. "std" is population standard deviation
    (statistics.pstdev), matching numpy's default ddof=0."""
    by_relation: dict[str, list[float]] = {}
    for record in records:
        by_relation.setdefault(record["relation"], []).append(record["cosine"])
    return {
        relation: {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "n": len(values),
        }
        for relation, values in by_relation.items()
    }


def relation_wise_task_evolution(records: list[dict[str, Any]]) -> dict[str, dict[int, float]]:
    """{relation: {task: mean_cosine_that_task}} -- only tasks where that
    relation has at least one observation that task."""
    by_relation_task: dict[str, dict[int, list[float]]] = {}
    for record in records:
        by_relation_task.setdefault(record["relation"], {}).setdefault(
            record["task"], []
        ).append(record["cosine"])
    return {
        relation: {task: statistics.fmean(values) for task, values in task_map.items()}
        for relation, task_map in by_relation_task.items()
    }


def rank_relations(summary: dict[str, dict[str, float | int]]) -> list[dict[str, Any]]:
    """[{relation, mean, std, n}, ...] sorted by mean descending."""
    return sorted(
        (
            {"relation": relation, "mean": stats["mean"], "std": stats["std"], "n": stats["n"]}
            for relation, stats in summary.items()
        ),
        key=lambda row: row["mean"],
        reverse=True,
    )


def class_pair_summary(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Keyed by tuple(sorted((new_class, bank_class))) -- normalized,
    order-independent, so lookups against the raw-feature matrix (also
    symmetric) can't silently miss a pair due to argument order. Missing
    relations for a pair are simply absent from "per_relation", never
    zero-filled."""
    per_pair_relations: dict[tuple[str, str], dict[str, float]] = {}
    pair_classes: dict[tuple[str, str], tuple[str, str]] = {}
    for record in records:
        key = tuple(sorted((record["new_class"], record["bank_class"])))
        per_pair_relations.setdefault(key, {})[record["relation"]] = record["cosine"]
        pair_classes[key] = (record["new_class"], record["bank_class"])

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, per_relation in per_pair_relations.items():
        new_class, bank_class = pair_classes[key]
        result[key] = {
            "new_class": new_class,
            "bank_class": bank_class,
            "mean_across_relations": statistics.fmean(per_relation.values()),
            "per_relation": per_relation,
            "best_relation": max(per_relation, key=per_relation.get),
            "worst_relation": min(per_relation, key=per_relation.get),
        }
    return result


def rank_class_pairs(summary: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """[{class_a, class_b, mean, per_relation, best_relation, worst_relation}, ...]
    sorted by mean descending."""
    return sorted(
        (
            {
                "class_a": key[0],
                "class_b": key[1],
                "mean": value["mean_across_relations"],
                "per_relation": value["per_relation"],
                "best_relation": value["best_relation"],
                "worst_relation": value["worst_relation"],
            }
            for key, value in summary.items()
        ),
        key=lambda row: row["mean"],
        reverse=True,
    )
