"""Tests for the per-relation drift analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trench_ids.cl.drift import correlate_with_transferability, load_mean_transferability


def test_correlation_reports_the_measured_direction() -> None:
    """The measured relationship on real runs is *negative*: the relations
    that score highest on transferability drift least. A test that only
    checked "a number comes back" would not catch the sign flipping."""
    drift = {"a": 0.6, "b": 0.4, "c": 0.25, "d": 0.1, "e": 0.05}
    transferability = {"a": 0.0, "b": -0.02, "c": 0.19, "d": 0.4, "e": 0.5}

    result = correlate_with_transferability(drift, transferability)

    assert result["n"] == 5
    assert result["pearson_r"] < 0
    assert result["spearman_rho"] < 0


def test_correlation_reports_n_so_five_points_are_never_read_as_many() -> None:
    result = correlate_with_transferability(
        {"a": 0.1, "b": 0.2, "c": 0.3}, {"a": 0.3, "b": 0.2, "c": 0.1}
    )

    assert result["n"] == 3
    assert set(result["relations"]) == {"a", "b", "c"}


def test_correlation_is_skipped_rather_than_faked_below_three_points() -> None:
    result = correlate_with_transferability({"a": 0.1}, {"a": 0.2})

    assert result == {"n": 1}


def test_correlation_uses_only_relations_present_in_both_inputs() -> None:
    result = correlate_with_transferability(
        {"a": 0.1, "b": 0.2, "c": 0.3, "extra": 0.9}, {"a": 0.3, "b": 0.2, "c": 0.1}
    )

    assert result["relations"] == ["a", "b", "c"]


def test_mean_transferability_averages_across_every_task_file(tmp_path: Path) -> None:
    for task, value in ((1, 0.2), (2, 0.4), (3, 0.6)):
        (tmp_path / f"distillation_task_{task}.json").write_text(
            json.dumps({"s_r": {"originates": value, "protocol_of": 1.0}})
        )

    means = load_mean_transferability(tmp_path)

    assert means["originates"] == pytest.approx(0.4)
    assert means["protocol_of"] == pytest.approx(1.0)


def test_mean_transferability_is_empty_when_no_run_recorded_any(tmp_path: Path) -> None:
    assert load_mean_transferability(tmp_path) == {}
