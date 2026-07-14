"""Tests for size-aware task-pairing tie-break (trench_ids.task_design)."""

from __future__ import annotations

import pandas as pd
import pytest

from trench_ids.task_design import assign_groups, min_weight_grouping


def _four_class_sim() -> pd.DataFrame:
    """A and B are large classes, C and D are small. A-B and C-D are the
    most DISSIMILAR pairs (lowest cosine), so minimizing total similarity
    alone picks A+B and C+D -- pairing the two large classes together and
    producing the worst-balanced valid grouping. A-C, A-D, B-C, B-D are all
    more similar (0.4) but still under any reasonable threshold, so every
    grouping is threshold-valid; only the tie-break differs."""
    return pd.DataFrame(
        {
            "A": [0.00, 0.05, 0.40, 0.40],
            "B": [0.05, 0.00, 0.40, 0.40],
            "C": [0.40, 0.40, 0.00, 0.05],
            "D": [0.40, 0.40, 0.05, 0.00],
        },
        index=["A", "B", "C", "D"],
    )


def test_min_weight_grouping_without_sizes_minimizes_total_similarity() -> None:
    sim = _four_class_sim()

    groups, total = min_weight_grouping(sim, threshold=0.5)

    assert {frozenset(g) for g in groups} == {frozenset(("A", "B")), frozenset(("C", "D"))}
    assert total == pytest.approx(0.1)


def test_min_weight_grouping_with_sizes_avoids_pairing_two_large_classes() -> None:
    sim = _four_class_sim()
    sizes = {"A": 100.0, "B": 100.0, "C": 1.0, "D": 1.0}

    groups, total = min_weight_grouping(sim, threshold=0.5, sizes=sizes)

    # A+B (size 200) is no longer chosen -- A and B must land in different
    # groups once size is the primary objective.
    group_sets = {frozenset(g) for g in groups}
    assert frozenset(("A", "B")) not in group_sets
    assert frozenset(("C", "D")) not in group_sets
    max_size = max(sum(sizes[c] for c in g) for g in groups)
    assert max_size == pytest.approx(101.0)
    # Tie-break still applied: among the two size-101 options (A+C/B+D vs.
    # A+D/B+C), both have total_sim=0.8 in this fixture, so either is valid.
    assert total == pytest.approx(0.8)


def test_min_weight_grouping_sizes_none_matches_no_sizes_arg() -> None:
    """Explicit sizes=None must be identical to omitting the argument."""
    sim = _four_class_sim()

    groups_omitted, total_omitted = min_weight_grouping(sim, threshold=0.5)
    groups_none, total_none = min_weight_grouping(sim, threshold=0.5, sizes=None)

    assert groups_omitted == groups_none
    assert total_omitted == total_none


def test_assign_groups_threads_sizes_into_remainder_grouping() -> None:
    # No clique at this threshold (every pair <= 0.5), so all 4 classes are
    # "remaining" and go through min_weight_grouping directly.
    sim = _four_class_sim()
    sizes = {"A": 100.0, "B": 100.0, "C": 1.0, "D": 1.0}

    groups, _ = assign_groups(sim, threshold=0.5, sizes=sizes)

    group_sets = {frozenset(g) for g in groups}
    assert frozenset(("A", "B")) not in group_sets
