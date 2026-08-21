from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from trench_ids.cl.compare_runs import (
    MEANINGFUL_DELTA,
    build_report,
    load_run,
    paired_comparison,
    parse_arm,
    write_csv,
)
from trench_ids.cl.train import average_forgetting, backward_transfer


def _write_run(tmp_path, name, seed, final_accs, own_accs=None, pooled=None):
    """A run directory with a synthetic 3-task forgetting matrix.

    ``final_accs`` is the last row's first two cells (R[3][1], R[3][2]);
    ``own_accs`` is the full diagonal (R[1][1], R[2][2], R[3][3]). Those are
    the two things forgetting and BWT are built from, so a test can dial
    either independently. Defaulting the diagonal to ``final_accs`` plus a
    fixed R[3][3] makes a run with no forgetting the zero-argument case.
    """
    own_accs = own_accs or [*final_accs, 0.9]
    run_dir = tmp_path / name
    run_dir.mkdir()
    matrix = {
        "1": {"1": own_accs[0]},
        "2": {"1": own_accs[0], "2": own_accs[1]},
        "3": {"1": final_accs[0], "2": final_accs[1], "3": own_accs[2]},
    }
    (run_dir / "forgetting_matrix.json").write_text(json.dumps(matrix))
    (run_dir / "summary.json").write_text(json.dumps({"seed": seed, "runtime_seconds": 1.0}))
    if pooled is not None:
        eval_dir = run_dir / "eval"
        eval_dir.mkdir()
        (eval_dir / "pooled_final_metrics.json").write_text(json.dumps(pooled))
    return run_dir


def _pooled(f1_macro):
    return {
        "accuracy": 0.9,
        "f1_macro": f1_macro,
        "f1_weighted": 0.9,
        "per_class": {"Benign": {"f1": 0.8}},
    }


def test_backward_transfer_is_negative_when_tasks_degrade() -> None:
    # Task 1 learned at 0.9, ends at 0.5; task 2 learned at 0.8, ends at 0.6.
    matrix = {1: {1: 0.9}, 2: {1: 0.7, 2: 0.8}, 3: {1: 0.5, 2: 0.6, 3: 0.9}}

    assert backward_transfer(matrix) == pytest.approx(((0.5 - 0.9) + (0.6 - 0.8)) / 2)


def test_backward_transfer_is_positive_when_later_tasks_help() -> None:
    """Replay rehearses task 1, so it ends *above* where it was when first
    learned. Both metrics register this (forgetting goes negative too, since
    its peak excludes the final row) -- they are not redundant because their
    reference points differ, not because one is sign-limited."""
    matrix = {1: {1: 0.6}, 2: {1: 0.7, 2: 0.8}, 3: {1: 0.8, 2: 0.8, 3: 0.9}}

    assert backward_transfer(matrix) > 0
    assert average_forgetting(matrix) < 0


def test_forgetting_and_bwt_disagree_when_a_task_peaks_off_the_diagonal() -> None:
    """The case that makes reporting both worthwhile: task 1 is introduced
    at 0.60, climbs to 0.90 mid-sequence, then ends at 0.70. BWT says it
    improved overall (+0.10); forgetting says 0.20 of its best was lost.
    Both are true, and either alone is misleading."""
    matrix = {1: {1: 0.6}, 2: {1: 0.9, 2: 0.8}, 3: {1: 0.7, 2: 0.8, 3: 0.9}}

    assert backward_transfer(matrix) == pytest.approx(((0.7 - 0.6) + (0.8 - 0.8)) / 2)
    assert average_forgetting(matrix) == pytest.approx(((0.9 - 0.7) + (0.8 - 0.8)) / 2)


def test_backward_transfer_is_zero_for_a_single_task() -> None:
    assert backward_transfer({1: {1: 0.9}}) == 0.0


def test_load_run_recomputes_metrics_without_reading_them_from_summary(tmp_path) -> None:
    """The pre-existing GNN runs' summary.json predates backward_transfer;
    metrics must come from the matrix so those runs need no re-training."""
    run_dir = _write_run(tmp_path, "r", seed=42, final_accs=[0.5, 0.6], own_accs=[0.9, 0.8, 0.9])
    assert "backward_transfer" not in json.loads((run_dir / "summary.json").read_text())

    record = load_run(run_dir)

    assert record["backward_transfer"] == pytest.approx(-0.3)
    assert record["seed"] == 42


def test_load_run_tolerates_a_missing_eval_directory(tmp_path) -> None:
    record = load_run(_write_run(tmp_path, "r", seed=1, final_accs=[0.5, 0.6]))

    assert "f1_macro" not in record
    assert "average_forgetting" in record


def test_load_run_raises_for_an_unfinished_run(tmp_path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="did this run finish"):
        load_run(tmp_path / "empty")


def test_paired_comparison_uses_only_seeds_present_in_both_arms(tmp_path) -> None:
    base = [load_run(_write_run(tmp_path, f"b{s}", s, [0.5, 0.5])) for s in (42, 1, 2)]
    other = [load_run(_write_run(tmp_path, f"o{s}", s, [0.6, 0.6])) for s in (42, 1)]

    result = paired_comparison(base, other, "final_average_accuracy")

    assert result["seeds"] == [1, 42]


def test_paired_comparison_needs_at_least_two_shared_seeds(tmp_path) -> None:
    base = [load_run(_write_run(tmp_path, "b", 42, [0.5, 0.5]))]
    other = [load_run(_write_run(tmp_path, "o", 1, [0.6, 0.6]))]

    assert paired_comparison(base, other, "final_average_accuracy") is None


def test_a_large_consistent_difference_is_flagged_meaningful(tmp_path) -> None:
    base = [load_run(_write_run(tmp_path, f"b{s}", s, [0.50, 0.50])) for s in (42, 1, 2)]
    other = [load_run(_write_run(tmp_path, f"o{s}", s, [0.70, 0.70])) for s in (42, 1, 2)]

    result = paired_comparison(base, other, "final_average_accuracy")

    assert result["mean_delta"] > MEANINGFUL_DELTA
    assert result["sign_consistent_across_seeds"]
    assert result["improves_baseline"]
    assert result["meaningful"]


def test_an_inconsistent_difference_is_not_meaningful_even_if_large(tmp_path) -> None:
    """The pre-registered rule requires magnitude AND sign consistency --
    one seed swinging the mean is exactly the failure mode n=3 invites."""
    base = [load_run(_write_run(tmp_path, f"b{s}", s, [0.50, 0.50])) for s in (42, 1, 2)]
    flipped = {42: 0.90, 1: 0.45, 2: 0.45}
    other = [
        load_run(_write_run(tmp_path, f"o{s}", s, [v, v])) for s, v in flipped.items()
    ]

    result = paired_comparison(base, other, "final_average_accuracy")

    assert abs(result["mean_delta"]) >= MEANINGFUL_DELTA
    assert not result["sign_consistent_across_seeds"]
    assert not result["meaningful"]


def test_forgetting_direction_is_inverted_when_scoring_improvement(tmp_path) -> None:
    """Lower forgetting is better; the report must not call a decrease a
    regression."""
    base = [
        load_run(_write_run(tmp_path, f"b{s}", s, [0.3, 0.3], own_accs=[0.9, 0.9, 0.9]))
        for s in (42, 1, 2)
    ]
    other = [
        load_run(_write_run(tmp_path, f"o{s}", s, [0.8, 0.8], own_accs=[0.9, 0.9, 0.9]))
        for s in (42, 1, 2)
    ]

    result = paired_comparison(base, other, "average_forgetting")

    assert result["mean_delta"] < 0  # less forgetting
    assert result["improves_baseline"]


def test_build_report_aggregates_every_arm_and_pairs_against_the_baseline(tmp_path) -> None:
    arms = {}
    for name, acc, f1 in (("Flat", 0.50, 0.50), ("GNN", 0.70, 0.72)):
        arms[name] = [
            _write_run(tmp_path, f"{name}{s}", s, [acc, acc], pooled=_pooled(f1))
            for s in (42, 1, 2)
        ]

    report = build_report(arms, "Flat")

    assert report["aggregates"]["Flat"]["final_average_accuracy"]["n"] == 3
    assert report["aggregates"]["GNN"]["f1_macro"]["mean"] == pytest.approx(0.72)
    assert report["aggregates"]["Flat"]["final_average_accuracy"]["std"] == pytest.approx(0.0)
    comparison = report["paired_comparisons"]["GNN vs Flat"]
    assert {c["metric"] for c in comparison} >= {"final_average_accuracy", "f1_macro"}


def test_single_run_arm_reports_undefined_std_not_zero(tmp_path) -> None:
    arms = {"Flat": [_write_run(tmp_path, "f", 42, [0.5, 0.5])]}

    report = build_report(arms, "Flat")

    assert report["aggregates"]["Flat"]["final_average_accuracy"]["std"] is None


def test_build_report_rejects_a_baseline_that_is_not_an_arm(tmp_path) -> None:
    arms = {"Flat": [_write_run(tmp_path, "f", 42, [0.5, 0.5])]}

    with pytest.raises(ValueError, match="not among"):
        build_report(arms, "GNN")


def test_parse_arm_splits_name_and_directories() -> None:
    name, dirs = parse_arm("Flat+Host=runs/a, runs/b ")

    assert name == "Flat+Host"
    # Compared as Path, not str -- Path normalises separators per platform.
    assert dirs == [Path("runs/a"), Path("runs/b")]


def test_parse_arm_rejects_a_spec_without_a_name() -> None:
    with pytest.raises(ValueError, match="--arm expects"):
        parse_arm("runs/a,runs/b")


def test_write_csv_arm_column_uses_the_report_arm_not_summary_json(tmp_path) -> None:
    """train_flat.py writes summary.json["arm"] as a bare "flat"/"flat+host"
    model-type label, which cannot distinguish two arms that share a model
    type but differ in configuration -- e.g. flathost_replay vs.
    flathost_small_replay, the parameter-matching control pair, both of
    which are "flat+host" in summary.json. The CSV's "arm" column must show
    the caller-supplied, unambiguous report-arm name instead."""
    run_dirs = {}
    for name in ("Flat+Host", "Flat+Host (matched)"):
        run_dir = _write_run(tmp_path, name.replace(" ", "_"), 42, [0.5, 0.5])
        summary = json.loads((run_dir / "summary.json").read_text())
        summary["arm"] = "flat+host"  # collides regardless of the real arm
        (run_dir / "summary.json").write_text(json.dumps(summary))
        run_dirs[name] = [run_dir]

    report = build_report(run_dirs, "Flat+Host")
    out_path = tmp_path / "comparison.csv"
    write_csv(report, out_path)

    with out_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    arms = {row["arm"] for row in rows}
    assert arms == {"Flat+Host", "Flat+Host (matched)"}
