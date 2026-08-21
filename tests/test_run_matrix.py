"""Tests for the experiment-matrix definition and its resume behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trench_ids.cl import run_matrix
from trench_ids.cl.run_matrix import Run, arm_of, build_matrix, collect, execute


def test_every_run_name_is_unique() -> None:
    """Two runs sharing a name would silently overwrite each other's
    directory, and the second would then be 'skipped' as already finished."""
    runs = build_matrix(lambda_d=1.0)

    names = [r.name for r in runs]
    assert len(names) == len(set(names))


def test_non_ewc_arms_zero_every_lambda() -> None:
    """Leaving the config default (1.0) in place would apply a penalty to all
    12 EWC groups, confounding every arm that is not about EWC with a
    regulariser nobody asked for."""
    runs = build_matrix(lambda_d=1.0)

    for run in runs:
        if run.module != "trench_ids.cl.train" or "ewc" in run.name:
            continue
        overrides = " ".join(run.overrides)
        assert "ewc.lambda_r=0" in overrides, run.name
        assert "ewc.lambda_s=0" in overrides, run.name
        assert "ewc.lambda_u=0" in overrides, run.name


def test_both_regularised_arms_share_the_same_warmup_window() -> None:
    """EWC and TRD have to be active for the same fraction of training, or
    the comparison between them measures the window rather than the
    mechanism."""
    runs = build_matrix(lambda_d=1.0)
    regularised = [r for r in runs if "ewc" in r.name or "trd" in r.name]

    assert regularised
    for run in regularised:
        assert "train.warmup_epochs=1" in " ".join(run.overrides), run.name


def test_the_three_weightings_differ_only_in_the_weighting_flag() -> None:
    """The ablation is only interpretable if lambda, temperature, replay and
    warm-up are identical across the three arms."""
    runs = {r.name: r for r in build_matrix(lambda_d=3.0)}

    def without_weighting(run: Run) -> set[str]:
        return {o for o in run.overrides if not o.startswith("distill.weighting=")}

    uniform = without_weighting(runs["gnn_trd_uniform_s42"])
    transfer = without_weighting(runs["gnn_trd_transfer_s42"])
    inverse = without_weighting(runs["gnn_trd_inverse_s42"])

    assert uniform == transfer == inverse
    assert "distill.lambda_d=3.0" in uniform


def test_the_lambda_sweep_holds_temperature_fixed() -> None:
    """A lambda selected under one temperature does not transfer to another,
    so the sweep and the final arms have to agree on it."""
    runs = build_matrix(lambda_d=1.0)
    sweep = [r for r in runs if r.group == "lambda"]
    final = [r for r in runs if "trd" in r.name and r.group == "method"]

    assert sweep and final
    temperatures = {o for r in sweep + final for o in r.overrides if "temperature" in o}
    assert len(temperatures) == 1


def test_every_seed_appears_in_every_multi_seed_arm() -> None:
    runs = build_matrix(lambda_d=1.0, seeds=(42, 1, 2))
    by_prefix: dict[str, set[str]] = {}
    for run in runs:
        if run.group == "lambda" or run.name == "joint":
            continue
        prefix, _, seed = run.name.rpartition("_s")
        by_prefix.setdefault(prefix, set()).add(seed)

    assert by_prefix
    for prefix, seeds in by_prefix.items():
        assert seeds == {"42", "1", "2"}, prefix


def test_the_lambda_sweep_brackets_the_winner_on_both_sides() -> None:
    """A sweep whose best value sits at the edge of the range cannot tell an
    optimum from a truncation. The first pass ran 0.5-50 and 0.5 won, so the
    range was extended downward; if a later edit ever narrows it back, this
    fails rather than silently reintroducing a boundary result."""
    runs = build_matrix(lambda_d=1.0)
    candidates = sorted(
        float(o.split("=")[1])
        for r in runs
        if r.group == "lambda"
        for o in r.overrides
        if o.startswith("distill.lambda_d=")
    )

    assert len(candidates) >= 5
    assert min(candidates) <= 0.1
    assert max(candidates) / min(candidates) >= 100


def test_the_budget_sweep_brackets_the_standing_default_on_both_sides() -> None:
    """200 graphs/task is the value every replay result on record used. A
    sweep that only went downward from it could show 200 winning without
    ever establishing it is an optimum rather than the largest value tried
    -- the exact truncation the lambda_d sweep had to be re-run to escape."""
    budgets = sorted(run_matrix.BUDGETS)

    assert min(budgets) < 200 < max(budgets)


def test_the_budget_sweep_does_not_re_run_the_default() -> None:
    """gnn_replay/flathost_replay already are the 200 point at identical
    settings; a second copy under a budget_* name would anchor the curve
    with a different seed draw than the runs it is compared against."""
    assert 200 not in run_matrix.BUDGETS


def test_budget_arms_vary_only_the_buffer_size() -> None:
    """The sweep isolates memory from compute: rehearsal stays at 30% of the
    training pool at every budget, so gradient steps and pool composition
    are identical and the only variable is how many distinct past graphs
    that 30% is drawn from. If replay_fraction ever drifts between arms the
    curve stops measuring memory at all."""
    runs = [r for r in build_matrix(lambda_d=1.0) if r.group == "budget"]
    assert runs

    for encoder in ("gnn", "flathost"):
        arms = [r for r in runs if f"budget_{encoder}_" in r.name and r.name.endswith("_s42")]
        assert len(arms) == len(run_matrix.BUDGETS)

        def without_budget(run: Run) -> set[str]:
            return {o for o in run.overrides if not o.startswith("replay.buffer_size_per_task=")}

        assert len({frozenset(without_budget(r)) for r in arms}) == 1
        for run in arms:
            assert "replay.replay_fraction=0.3" in run.overrides, run.name
            assert "replay.enabled=true" in run.overrides, run.name


def test_budget_sweep_covers_both_encoders_at_every_seed() -> None:
    """The ladder currently puts Flat+Host ahead of the GNN, so a budget
    chosen on the GNN alone could fix a number on the arm that does not
    ship."""
    runs = [r for r in build_matrix(lambda_d=1.0, seeds=(42, 1, 2)) if r.group == "budget"]

    expected = {
        f"budget_{encoder}_{budget}_s{seed}"
        for encoder in ("gnn", "flathost")
        for budget in run_matrix.BUDGETS
        for seed in (42, 1, 2)
    }
    assert {r.name for r in runs} == expected


def test_budget_arms_use_the_right_trainer_module() -> None:
    runs = {r.name: r for r in build_matrix(lambda_d=1.0)}

    assert runs["budget_gnn_25_s42"].module == "trench_ids.cl.train"
    assert runs["budget_flathost_25_s42"].module == "trench_ids.cl.train_flat"


def test_only_filter_does_not_leak_across_arms_sharing_a_prefix() -> None:
    """``gnn_replay`` is a string prefix of ``gnn_replay_trd``, so a naive
    prefix filter aimed at plain replay also selects the replay+TRD arm --
    which depends on a selected lambda_d and would run at the placeholder
    default, writing a result indistinguishable from a real one. Several
    workers share this matrix concurrently, so the filter has to be exact
    at the arm level."""
    runs = build_matrix(lambda_d=1.0)
    wanted = {"gnn_finetune", "gnn_replay"}

    selected = [r for r in runs if arm_of(r.name) in wanted]

    assert {r.name for r in selected} == {
        "gnn_finetune_s42", "gnn_finetune_s1", "gnn_finetune_s2",
        "gnn_replay_s42", "gnn_replay_s1", "gnn_replay_s2",
    }


def test_arm_of_strips_only_the_seed_suffix() -> None:
    assert arm_of("gnn_replay_trd_s42") == "gnn_replay_trd"
    assert arm_of("lambda_trd_0.25_s42") == "lambda_trd_0.25"
    assert arm_of("joint") == "joint"


def test_a_finished_run_is_skipped_not_rerun(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_matrix, "RUNS_DIR", tmp_path)
    run = Run(name="already_done", module="trench_ids.cl.train")
    (tmp_path / "already_done").mkdir(parents=True)
    (tmp_path / "already_done" / "summary.json").write_text("{}")
    (tmp_path / "already_done" / "eval").mkdir()
    (tmp_path / "already_done" / "eval" / "pooled_final_metrics.json").write_text("{}")

    result = execute([run])

    assert result.skipped == ["already_done"]
    assert result.completed == []


def test_collect_gathers_only_finished_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_matrix, "RUNS_DIR", tmp_path)
    done = Run(name="done", module="m", group="method", note="n")
    missing = Run(name="missing", module="m", group="method")
    (tmp_path / "done").mkdir(parents=True)
    (tmp_path / "done" / "summary.json").write_text(json.dumps({"average_forgetting": 0.1}))

    collected = collect([done, missing])

    assert set(collected) == {"done"}
    assert collected["done"]["average_forgetting"] == pytest.approx(0.1)
    assert collected["done"]["group"] == "method"


def test_joint_training_is_not_sent_to_the_forgetting_matrix_evaluator(
    tmp_path: Path, monkeypatch
) -> None:
    """train_joint writes no per-task checkpoints, so the evaluator would
    fail on it; the guard has to be on the module, not on the run name."""
    calls: list[str] = []
    monkeypatch.setattr(run_matrix.subprocess, "run", lambda *a, **k: calls.append(a) or None)

    run_matrix._evaluate(Run(name="joint", module="trench_ids.cl.train_joint"))

    assert calls == []


def test_joint_run_is_recognised_by_its_own_summary_filename(
    tmp_path: Path, monkeypatch
) -> None:
    """train_joint writes joint_summary.json, not summary.json. Checking only
    the latter made the joint run non-idempotent -- re-run on every sweep, and
    reported as outstanding by --dry-run after it had already finished."""
    monkeypatch.setattr(run_matrix, "RUNS_DIR", tmp_path)
    joint = Run(name="joint", module="trench_ids.cl.train_joint", group="joint")
    (tmp_path / "joint").mkdir(parents=True)
    (tmp_path / "joint" / "joint_summary.json").write_text(json.dumps({"pooled_accuracy": 0.8}))

    assert joint.summary_path().name == "joint_summary.json"
    result = execute([joint])

    assert result.skipped == ["joint"]
    assert result.completed == []


def test_sequential_runs_still_use_the_plain_summary_filename(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(run_matrix, "RUNS_DIR", tmp_path)
    run = Run(name="gnn_replay_s42", module="trench_ids.cl.train")

    assert run.summary_path().name == "summary.json"


def test_collect_reads_the_joint_runs_summary_too(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_matrix, "RUNS_DIR", tmp_path)
    joint = Run(name="joint", module="trench_ids.cl.train_joint", group="joint")
    (tmp_path / "joint").mkdir(parents=True)
    (tmp_path / "joint" / "joint_summary.json").write_text(json.dumps({"pooled_accuracy": 0.807}))

    collected = collect([joint])

    assert collected["joint"]["pooled_accuracy"] == pytest.approx(0.807)
