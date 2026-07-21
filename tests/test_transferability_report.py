from __future__ import annotations

import json
from pathlib import Path

from trench_ids.cl.transferability_report import load_transferability_records


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
