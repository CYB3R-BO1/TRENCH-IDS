"""Tests for the results-document builder."""

from __future__ import annotations

import json
from pathlib import Path

from trench_ids.cl import report as report_mod
from trench_ids.cl.report import COMPARISONS, arm_dirs, build, lambda_selection_table


def _write_run(root: Path, name: str, *, val: float, test: float, forget: float, lam: float
               ) -> None:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "final_average_val_accuracy": val,
                "final_average_accuracy": test,
                "average_forgetting": forget,
                "distill_lambda_d": lam,
                "seed": 42,
            }
        )
    )


def test_arm_dirs_skips_seeds_that_have_not_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    _write_run(tmp_path, "gnn_replay_s42", val=0.7, test=0.7, forget=0.2, lam=0.0)
    _write_run(tmp_path, "gnn_replay_s2", val=0.7, test=0.7, forget=0.2, lam=0.0)

    dirs = arm_dirs("gnn_replay")

    assert [d.name for d in dirs] == ["gnn_replay_s42", "gnn_replay_s2"]


def test_arm_dirs_ignores_a_run_that_is_still_in_flight(tmp_path: Path, monkeypatch) -> None:
    """run_matrix creates the directory before training starts, so a
    directory-only check would return a run with no forgetting_matrix.json
    and make build() raise -- defeating the point of being able to generate
    the report partway through a multi-hour sweep."""
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    _write_run(tmp_path, "gnn_replay_s42", val=0.7, test=0.7, forget=0.2, lam=0.0)
    (tmp_path / "gnn_replay_s1").mkdir(parents=True)  # started, not finished

    dirs = arm_dirs("gnn_replay")

    assert [d.name for d in dirs] == ["gnn_replay_s42"]


def test_lambda_is_selected_on_validation_not_test(tmp_path: Path, monkeypatch) -> None:
    """The whole reason this group exists. If the ranking ever silently
    switched to the test column, every earlier criticism of this project's
    selection hygiene would apply again."""
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    # lambda 3 wins on val; lambda 10 wins on test. Val must decide.
    _write_run(tmp_path, "lambda_trd_3_s42", val=0.80, test=0.70, forget=0.10, lam=3.0)
    _write_run(tmp_path, "lambda_trd_10_s42", val=0.60, test=0.95, forget=0.20, lam=10.0)

    text = "\n".join(lambda_selection_table())

    assert "Selected: **λ_d = 3**" in text


def test_no_candidate_beating_the_anchor_is_reported_as_a_negative_result(
    tmp_path: Path, monkeypatch
) -> None:
    """If every lambda is worse than switching the penalty off, the table must
    say so. Naming a 'winner' from a set that plain fine-tuning beats is how a
    negative result gets written up as a success."""
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    _write_run(tmp_path, "lambda_trd_0.5_s42", val=0.40, test=0.40, forget=0.63, lam=0.5)
    _write_run(tmp_path, "lambda_trd_2_s42", val=0.27, test=0.27, forget=0.71, lam=2.0)
    _write_run(tmp_path, "gnn_finetune_s42", val=0.55, test=0.54, forget=0.50, lam=0.0)

    rows = lambda_selection_table()
    text = "\n".join(rows)

    assert "No candidate beat the λ_d = 0 anchor" in text
    assert "negative result" in text
    assert "plain fine-tuning" in text


def test_a_candidate_beating_the_anchor_is_reported_as_a_win(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    _write_run(tmp_path, "lambda_trd_0.5_s42", val=0.60, test=0.59, forget=0.30, lam=0.5)
    _write_run(tmp_path, "lambda_trd_2_s42", val=0.27, test=0.27, forget=0.71, lam=2.0)
    _write_run(tmp_path, "gnn_finetune_s42", val=0.55, test=0.54, forget=0.50, lam=0.0)

    text = "\n".join(lambda_selection_table())

    assert "Selected: **λ_d = 0.5**" in text
    assert "beats the λ_d = 0 anchor by +0.0500" in text
    assert "No candidate beat" not in text


def test_a_winner_at_the_edge_of_the_range_is_flagged_as_a_boundary_result(
    tmp_path: Path, monkeypatch
) -> None:
    """A sweep whose best value is its smallest value has not bracketed the
    optimum, and the document should not present it as though it had."""
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    _write_run(tmp_path, "lambda_trd_0.5_s42", val=0.60, test=0.59, forget=0.30, lam=0.5)
    _write_run(tmp_path, "lambda_trd_2_s42", val=0.27, test=0.27, forget=0.71, lam=2.0)

    text = "\n".join(lambda_selection_table())

    assert "boundary result" in text


def test_an_interior_winner_is_not_flagged_as_a_boundary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    _write_run(tmp_path, "lambda_trd_0.1_s42", val=0.30, test=0.30, forget=0.70, lam=0.1)
    _write_run(tmp_path, "lambda_trd_0.5_s42", val=0.60, test=0.59, forget=0.30, lam=0.5)
    _write_run(tmp_path, "lambda_trd_2_s42", val=0.27, test=0.27, forget=0.71, lam=2.0)

    text = "\n".join(lambda_selection_table())

    assert "boundary result" not in text


def test_lambda_table_is_empty_before_the_sweep_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)

    assert lambda_selection_table() == []


def test_each_comparison_names_a_baseline_that_is_one_of_its_own_arms() -> None:
    """A baseline outside the arm set would make build_report raise at the
    worst possible moment -- after every run has finished."""
    for name, spec in COMPARISONS.items():
        assert spec["baseline"] in spec["arms"], name


def test_the_weighting_comparison_is_baselined_on_uniform() -> None:
    """The hypothesis is 'transferability-weighted beats equally-weighted',
    so uniform is the baseline. Baselining on replay instead would let the
    distillation mechanism's own benefit be mistaken for the weighting's."""
    assert COMPARISONS["weighting"]["baseline"] == "TRD-uniform"
    assert set(COMPARISONS["weighting"]["arms"]) == {
        "TRD-uniform",
        "TRD-transfer",
        "TRD-inverse",
    }


def test_the_method_comparison_is_baselined_on_plain_finetuning() -> None:
    """Every regularised arm (EWC, the three TRD weightings) runs without
    replay, so the shared reference point they all differ from by exactly one
    thing is plain fine-tuning. Replay and Replay+TRD sit in the same table
    to show the stronger baseline and whether TRD adds on top of it, but they
    change the training data as well as the loss, so they are not the arm the
    regularisers should be paired against."""
    assert COMPARISONS["method"]["baseline"] == "Finetune"
    assert "Replay+TRD" in COMPARISONS["method"]["arms"]


def test_build_writes_a_document_even_with_no_runs_on_disk(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path / "runs")
    out = tmp_path / "results.md"

    text = build(out)

    assert out.exists()
    assert text.startswith("# TRENCH-IDS results")


def _write_full_run(root: Path, name: str, seed: int, acc: float, forget: float) -> None:
    """A run complete enough for compare_runs.load_run to read it."""
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "final_average_accuracy": acc,
                "final_average_val_accuracy": acc,
                "average_forgetting": forget,
                "backward_transfer": -forget,
            }
        )
    )
    (run / "forgetting_matrix.json").write_text(json.dumps({"1": {"1": acc}}))


def test_a_verdict_on_two_seeds_is_marked_provisional(tmp_path: Path, monkeypatch) -> None:
    """On 2 seeds 'same sign on both' is nearly vacuous, so a row that clears
    the delta threshold must not read as settled just because the third seed
    has not finished."""
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    for seed in (42, 1, 2):
        _write_full_run(tmp_path, f"flat_replay_s{seed}", seed, 0.70, 0.20)
    for seed in (42, 1):  # third seed still running
        _write_full_run(tmp_path, f"flathost_replay_s{seed}", seed, 0.85, 0.12)

    text = build(tmp_path / "results.md")

    assert "provisional (n=2)" in text
    assert "**MEANINGFUL**" not in text.split("Architecture comparison")[1].split("##")[0]


def test_a_verdict_on_all_three_seeds_is_not_marked_provisional(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    for seed in (42, 1, 2):
        _write_full_run(tmp_path, f"flat_replay_s{seed}", seed, 0.70, 0.20)
        _write_full_run(tmp_path, f"flathost_replay_s{seed}", seed, 0.85, 0.12)

    text = build(tmp_path / "results.md")

    assert "**MEANINGFUL**" in text
    assert "provisional" not in text.split("| Comparison |")[1]


def _write_budget_run(
    root: Path, name: str, seed: int, *, val: float, test: float, forget: float
) -> None:
    """Like _write_full_run but lets val and test disagree, which is the
    whole point of a selection-on-val test."""
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "final_average_accuracy": test,
                "final_average_val_accuracy": val,
                "average_forgetting": forget,
                "backward_transfer": -forget,
            }
        )
    )
    (run / "forgetting_matrix.json").write_text(json.dumps({"1": {"1": test}}))


def test_the_smallest_indistinguishable_budget_wins_not_the_highest_scoring() -> None:
    """Val accuracy is weakly increasing in memory, so plain argmax returns
    the largest budget swept almost by construction and the sweep would only
    ever ratify wherever the grid stopped. Memory is the cost being
    minimised, so the rule has to prefer the cheapest budget that cannot be
    distinguished from the best."""
    rows = [
        (25, 3, 0.800, 0.79, 0.20),
        (50, 3, 0.815, 0.81, 0.18),
        (100, 3, 0.820, 0.82, 0.17),
        (200, 3, 0.820, 0.82, 0.17),
        (400, 3, 0.825, 0.83, 0.16),
    ]

    assert report_mod.select_budget(rows) == 50
    assert max(rows, key=lambda r: r[2])[0] == 400  # what argmax would have said


def test_a_budget_outside_the_noise_floor_is_not_selected() -> None:
    """25 is 0.05 below the best here -- well past the pre-registered
    threshold -- so cheapness must not rescue it."""
    rows = [
        (25, 3, 0.770, 0.77, 0.25),
        (50, 3, 0.820, 0.82, 0.17),
    ]

    assert report_mod.select_budget(rows) == 50


def test_select_budget_is_none_before_any_run_finishes() -> None:
    assert report_mod.select_budget([]) is None


def test_budget_is_selected_on_validation_not_test(tmp_path: Path, monkeypatch) -> None:
    """The 400 arm is much better on test and worse on val. If the table ever
    ranks on test it will pick 400, and the sweep will have made a selection
    decision on the test split -- the exact defect this project spent
    2026-08-16 removing from lambda_d."""
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    for seed in (42, 1, 2):
        _write_budget_run(tmp_path, f"budget_gnn_25_s{seed}", seed, val=0.85, test=0.60, forget=0.2)
        _write_budget_run(
            tmp_path, f"budget_gnn_400_s{seed}", seed, val=0.60, test=0.95, forget=0.1
        )

    rows = report_mod.budget_rows({25: "budget_gnn_25", 400: "budget_gnn_400"})

    assert report_mod.select_budget(rows) == 25


def test_the_standing_default_is_read_from_the_existing_replay_arms(
    tmp_path: Path, monkeypatch
) -> None:
    """200 is not re-run under a budget_* name; gnn_replay/flathost_replay
    already are that point. If the table stops sourcing it from there the
    curve loses the only budget every earlier result was produced at."""
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    for seed in (42, 1, 2):
        _write_budget_run(tmp_path, f"budget_gnn_25_s{seed}", seed, val=0.80, test=0.79, forget=0.2)
        _write_budget_run(tmp_path, f"gnn_replay_s{seed}", seed, val=0.82, test=0.81, forget=0.17)

    text = "\n".join(report_mod.budget_selection_table())

    assert "standing default" in text
    assert "| 200" in text


def test_a_budget_selected_at_the_bottom_of_the_grid_is_flagged(
    tmp_path: Path, monkeypatch
) -> None:
    """If the smallest swept budget wins, the curve has not been shown to
    turn and the reported memory requirement is an upper bound, not a
    measured minimum."""
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)
    for seed in (42, 1, 2):
        _write_budget_run(tmp_path, f"budget_gnn_25_s{seed}", seed, val=0.82, test=0.81, forget=0.2)
        _write_budget_run(tmp_path, f"gnn_replay_s{seed}", seed, val=0.82, test=0.81, forget=0.17)

    text = "\n".join(report_mod.budget_selection_table())

    assert "smallest one swept" in text


def test_budget_table_is_empty_before_the_sweep_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_mod, "RUNS_DIR", tmp_path)

    assert report_mod.budget_selection_table() == []
