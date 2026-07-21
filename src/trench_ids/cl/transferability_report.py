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
