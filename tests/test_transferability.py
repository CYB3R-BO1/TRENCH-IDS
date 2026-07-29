from __future__ import annotations

import math

import pytest
import torch

from trench_ids.cl.transferability import (
    aggregate_per_class_relation_scores,
    aggregate_transferability_scores,
    estimate_transferability,
    relation_cosine_similarity,
)


def test_relation_cosine_similarity_identical_vectors_is_one() -> None:
    v = torch.tensor([1.0, 2.0, 3.0])
    assert relation_cosine_similarity(v, v.clone()) == pytest.approx(1.0)


def test_relation_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert relation_cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_relation_cosine_similarity_opposite_vectors_is_negative_one() -> None:
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([-1.0, -2.0])
    assert relation_cosine_similarity(a, b) == pytest.approx(-1.0)


def test_relation_cosine_similarity_known_angle() -> None:
    # (1, 0) vs (1, 1): cos(45 deg) = 1/sqrt(2)
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([1.0, 1.0])
    assert relation_cosine_similarity(a, b) == pytest.approx(1.0 / math.sqrt(2))


def test_relation_cosine_similarity_zero_vector_returns_zero_not_nan() -> None:
    a = torch.tensor([0.0, 0.0])
    b = torch.tensor([1.0, 2.0])
    assert relation_cosine_similarity(a, b) == 0.0


def test_estimate_transferability_against_empty_bank_is_empty_per_class() -> None:
    new_means = {"Scanning": {"originates": torch.tensor([1.0, 0.0])}}

    report = estimate_transferability(new_means, {})

    assert report == {"Scanning": {}}


def test_estimate_transferability_compares_same_relation_only() -> None:
    new_means = {
        "XSS": {
            "originates": torch.tensor([1.0, 0.0]),
            "targeted_by": torch.tensor([0.0, 1.0]),
        }
    }
    bank = {
        "Infiltration": {
            "originates": torch.tensor([1.0, 0.0]),  # identical -> 1.0
            "targeted_by": torch.tensor([1.0, 0.0]),  # orthogonal to [0,1] -> 0.0
        }
    }

    report = estimate_transferability(new_means, bank)

    assert report["XSS"]["Infiltration"]["originates"] == pytest.approx(1.0)
    assert report["XSS"]["Infiltration"]["targeted_by"] == pytest.approx(0.0, abs=1e-6)


def test_estimate_transferability_skips_relations_not_shared_by_both_classes() -> None:
    new_means = {"XSS": {"originates": torch.tensor([1.0, 0.0])}}
    bank = {"Infiltration": {"terminated_by": torch.tensor([1.0, 0.0])}}

    report = estimate_transferability(new_means, bank)

    assert report["XSS"]["Infiltration"] == {}


def test_estimate_transferability_covers_every_bank_class() -> None:
    new_means = {"XSS": {"originates": torch.tensor([1.0, 0.0])}}
    bank = {
        "Infiltration": {"originates": torch.tensor([1.0, 0.0])},
        "Scanning": {"originates": torch.tensor([0.0, 1.0])},
    }

    report = estimate_transferability(new_means, bank)

    assert set(report["XSS"].keys()) == {"Infiltration", "Scanning"}
    assert report["XSS"]["Infiltration"]["originates"] == pytest.approx(1.0)
    assert report["XSS"]["Scanning"]["originates"] == pytest.approx(0.0, abs=1e-6)


def test_aggregate_transferability_scores_means_across_all_pairs() -> None:
    report = {
        "XSS": {
            "Infiltration": {"originates": 0.8, "terminated_by": 0.2},
            "Scanning": {"originates": 0.4},
        }
    }

    s_r = aggregate_transferability_scores(report, flow_relations=["originates", "terminated_by"])

    # originates: mean(0.8, 0.4) = 0.6
    assert s_r["originates"].item() == pytest.approx(0.6)
    # terminated_by: only one pair -> 0.2
    assert s_r["terminated_by"].item() == pytest.approx(0.2)


def test_aggregate_transferability_scores_defaults_to_zero_for_empty_bank() -> None:
    s_r = aggregate_transferability_scores({}, flow_relations=["originates", "terminated_by"])

    assert s_r["originates"].item() == pytest.approx(0.0)
    assert s_r["terminated_by"].item() == pytest.approx(0.0)


def test_aggregate_transferability_scores_covers_every_flow_relation() -> None:
    report = {"XSS": {"Infiltration": {"originates": 1.0}}}

    s_r = aggregate_transferability_scores(
        report, flow_relations=["originates", "terminated_by", "targeted_by"]
    )

    assert set(s_r.keys()) == {"originates", "terminated_by", "targeted_by"}


def test_aggregate_per_class_relation_scores_means_over_old_classes_only() -> None:
    report = {
        "XSS": {
            "Infiltration": {"originates": 0.8, "terminated_by": 0.2},
            "Scanning": {"originates": 0.4, "terminated_by": 0.6},
        }
    }

    result = aggregate_per_class_relation_scores(
        report, flow_relations=["originates", "terminated_by"]
    )

    # Per-relation mean across old classes (Infiltration, Scanning),
    # relations kept separate -- NOT collapsed to one scalar.
    assert result["XSS"]["originates"] == pytest.approx(0.6)  # mean(0.8, 0.4)
    assert result["XSS"]["terminated_by"] == pytest.approx(0.4)  # mean(0.2, 0.6)


def test_aggregate_per_class_relation_scores_one_entry_per_new_class() -> None:
    report = {
        "XSS": {"Infiltration": {"originates": 1.0}},
        "DDoS": {"Infiltration": {"originates": 0.0}},
    }

    result = aggregate_per_class_relation_scores(report, flow_relations=["originates"])

    assert set(result.keys()) == {"XSS", "DDoS"}
    assert result["XSS"]["originates"] == pytest.approx(1.0)
    assert result["DDoS"]["originates"] == pytest.approx(0.0)


def test_aggregate_per_class_relation_scores_empty_bank_defaults_to_zero() -> None:
    """Task 1: report[new_class] = {} for every new class (no old classes
    to compare against yet) -- every relation defaults to 0.0, matching
    aggregate_transferability_scores' documented empty-bank convention."""
    report = {"Scanning": {}}

    result = aggregate_per_class_relation_scores(
        report, flow_relations=["originates", "terminated_by"]
    )

    assert result == {"Scanning": {"originates": 0.0, "terminated_by": 0.0}}


def test_aggregate_per_class_relation_scores_relation_with_no_shared_pairs_is_zero() -> None:
    report = {"XSS": {"Infiltration": {"targeted_by": 0.5}}}

    result = aggregate_per_class_relation_scores(
        report, flow_relations=["originates", "targeted_by"]
    )

    assert result["XSS"]["originates"] == pytest.approx(0.0)
    assert result["XSS"]["targeted_by"] == pytest.approx(0.5)
