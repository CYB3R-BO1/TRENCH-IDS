from __future__ import annotations

import math

import pytest
import torch

from trench_ids.cl.transferability import estimate_transferability, relation_cosine_similarity


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
