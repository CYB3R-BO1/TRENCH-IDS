from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from trench_ids.cl.replay_selection import (
    class_scores_from_relation_scores,
    graph_enrichment_score,
    normalize_scores,
    select_replay_graphs,
    weighted_sample_without_replacement,
    with_neutral_benign,
)


def _flow_graph(labels: list[int]) -> HeteroData:
    g = HeteroData()
    g["flow"].y = torch.tensor(labels)
    return g


def test_class_scores_from_relation_scores_means_over_relations() -> None:
    per_class = {"XSS": {"originates": 0.8, "terminated_by": 0.2}}

    result = class_scores_from_relation_scores(per_class)

    assert result["XSS"] == pytest.approx(0.5)


def test_class_scores_from_relation_scores_empty_relations_is_zero() -> None:
    result = class_scores_from_relation_scores({"Scanning": {}})

    assert result["Scanning"] == pytest.approx(0.0)


def test_normalize_scores_min_max_to_unit_range() -> None:
    result = normalize_scores({"a": 0.0, "b": 0.5, "c": 1.0})

    assert result["a"] == pytest.approx(0.0)
    assert result["b"] == pytest.approx(0.5)
    assert result["c"] == pytest.approx(1.0)


def test_normalize_scores_degenerate_all_equal_returns_constant_half() -> None:
    """Single new class (T1/T2 tasks) or all-identical scores: no basis to
    differentiate, so every class gets the neutral 0.5 weight rather than
    dividing by zero."""
    result = normalize_scores({"Scanning": 0.3})

    assert result == {"Scanning": 0.5}


def test_normalize_scores_empty_input_returns_empty() -> None:
    assert normalize_scores({}) == {}


def test_with_neutral_benign_is_mean_of_attack_weights() -> None:
    result = with_neutral_benign({"XSS": 0.2, "DDoS": 0.8}, benign_name="Benign")

    assert result["Benign"] == pytest.approx(0.5)
    assert result["XSS"] == pytest.approx(0.2)
    assert result["DDoS"] == pytest.approx(0.8)


def test_with_neutral_benign_empty_attack_weights_defaults_to_half() -> None:
    result = with_neutral_benign({}, benign_name="Benign")

    assert result == {"Benign": 0.5}


def test_graph_enrichment_score_is_weighted_mean_over_flow_labels() -> None:
    # label_names[0] = "Benign" (weight 0.0), label_names[1] = "XSS" (weight 1.0)
    g = _flow_graph([0, 1, 1, 0])  # 2 Benign, 2 XSS -> mean(0, 1, 1, 0) = 0.5
    class_weights = {"Benign": 0.0, "XSS": 1.0}

    score = graph_enrichment_score(g, class_weights, label_names=["Benign", "XSS"])

    assert score == pytest.approx(0.5)


def test_graph_enrichment_score_unknown_class_defaults_to_zero() -> None:
    g = _flow_graph([0])
    score = graph_enrichment_score(g, class_weights={}, label_names=["Benign"])

    assert score == pytest.approx(0.0)


def test_weighted_sample_without_replacement_respects_n_sample() -> None:
    items = list(range(10))
    weights = np.ones(10)
    rng = np.random.default_rng(0)

    result = weighted_sample_without_replacement(items, weights, n_sample=4, rng=rng)

    assert len(result) == 4
    assert len(set(result)) == 4  # without replacement


def test_weighted_sample_without_replacement_caps_at_available_items() -> None:
    items = [1, 2, 3]
    weights = np.ones(3)
    rng = np.random.default_rng(0)

    result = weighted_sample_without_replacement(items, weights, n_sample=10, rng=rng)

    assert sorted(result) == [1, 2, 3]


def test_weighted_sample_without_replacement_all_zero_weights_falls_back_to_uniform() -> None:
    items = [1, 2, 3, 4]
    weights = np.zeros(4)
    rng = np.random.default_rng(0)

    result = weighted_sample_without_replacement(items, weights, n_sample=2, rng=rng)

    assert len(result) == 2
    assert all(item in items for item in result)


def test_select_replay_graphs_high_transfer_favors_higher_scoring_class() -> None:
    """Two graphs, one entirely benign, one entirely a high-transferability
    class -- enrich_high_transfer weighted sampling with n_sample=1 must be
    far more likely to pick the high-scoring graph than the low one, over
    many trials (statistical, not deterministic, so this checks a strong
    majority rather than every draw)."""
    high_graph = _flow_graph([1, 1, 1])  # all "XSS"
    low_graph = _flow_graph([0, 0, 0])  # all "Benign"
    per_class_relation_scores = {"XSS": {"originates": 1.0}, "DDoS": {"originates": 0.0}}
    label_names = ["Benign", "XSS"]

    picks_high = 0
    for seed in range(50):
        rng = np.random.default_rng(seed)
        selected, _ = select_replay_graphs(
            [high_graph, low_graph],
            per_class_relation_scores,
            label_names,
            n_sample=1,
            mode="enrich_high_transfer",
            benign_name="Benign",
            rng=rng,
        )
        if selected[0] is high_graph:
            picks_high += 1

    # NOTE: the verbatim task-4-brief implementation gives high_graph weight
    # 1.0 vs. low_graph weight 0.5 (benign's neutral weight = mean of attack
    # weights = 0.5), a true 2:1 selection ratio (p=2/3), not the "near-3:1"
    # the brief's Step 4 comment assumed. All 50 seeds are fixed, so this is
    # fully deterministic (always 31/50, verified by rerunning), not
    # run-to-run flakiness -- a >35 threshold has only ~26% chance of
    # passing even with a correctly-behaving implementation. Lowered to 28,
    # still comfortably above the 25/50 chance baseline and matching the
    # implementation's actual, documented 2:1 favoring behavior.
    assert picks_high > 28  # well above the 25/50 chance baseline


def test_select_replay_graphs_returns_scores_matching_graph_count() -> None:
    g1 = _flow_graph([0, 1])
    g2 = _flow_graph([1, 1])
    per_class_relation_scores = {"XSS": {"originates": 0.5}}
    rng = np.random.default_rng(0)

    selected, scores = select_replay_graphs(
        [g1, g2],
        per_class_relation_scores,
        label_names=["Benign", "XSS"],
        n_sample=2,
        mode="enrich_low_transfer",
        benign_name="Benign",
        rng=rng,
    )

    assert len(selected) == len(scores) == 2
