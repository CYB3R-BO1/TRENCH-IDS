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
