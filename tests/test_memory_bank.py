from __future__ import annotations

import pytest
import torch

from trench_ids.cl.memory_bank import RelationMeanAccumulator, merge_into_bank


def test_relation_mean_accumulator_matches_hand_computed_means() -> None:
    # 4 flows: classes Benign, DDoS, DDoS, Injection (indices 0, 1, 1, 2).
    label_names = ["Benign", "DDoS", "Injection"]
    y = torch.tensor([0, 1, 1, 2])
    relation_embeds = {
        "originates": torch.tensor([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [10.0, 0.0]]),
        "targeted_by": torch.tensor([[0.0, 1.0], [0.0, 3.0], [0.0, 5.0], [0.0, 9.0]]),
    }

    acc = RelationMeanAccumulator()
    acc.update(relation_embeds, y, label_names)
    means = acc.means()

    # Benign is excluded entirely -- attack-type-only memory.
    assert set(means.keys()) == {"DDoS", "Injection"}
    # DDoS: rows 1,2 -> originates mean = (2+4)/2 = 3; targeted_by mean = (3+5)/2 = 4.
    assert torch.allclose(means["DDoS"]["originates"], torch.tensor([3.0, 0.0]))
    assert torch.allclose(means["DDoS"]["targeted_by"], torch.tensor([0.0, 4.0]))
    # Injection: row 3 only -> means equal that row exactly.
    assert torch.allclose(means["Injection"]["originates"], torch.tensor([10.0, 0.0]))
    assert torch.allclose(means["Injection"]["targeted_by"], torch.tensor([0.0, 9.0]))


def test_relation_mean_accumulator_accumulates_across_multiple_updates() -> None:
    label_names = ["Benign", "DDoS"]
    relation_embeds_batch1 = {"originates": torch.tensor([[2.0], [4.0]])}
    y_batch1 = torch.tensor([1, 1])
    relation_embeds_batch2 = {"originates": torch.tensor([[9.0]])}
    y_batch2 = torch.tensor([1])

    acc = RelationMeanAccumulator()
    acc.update(relation_embeds_batch1, y_batch1, label_names)
    acc.update(relation_embeds_batch2, y_batch2, label_names)
    means = acc.means()

    # 3 flows total, all DDoS: (2 + 4 + 9) / 3 = 5.0
    assert torch.allclose(means["DDoS"]["originates"], torch.tensor([5.0]))


def test_relation_mean_accumulator_all_benign_produces_empty_means() -> None:
    label_names = ["Benign", "DDoS"]
    relation_embeds = {"originates": torch.tensor([[1.0], [2.0]])}
    y = torch.tensor([0, 0])

    acc = RelationMeanAccumulator()
    acc.update(relation_embeds, y, label_names)

    assert acc.means() == {}


def test_merge_into_bank_adds_new_classes() -> None:
    bank = {"Scanning": {"originates": torch.tensor([1.0])}}
    new_means = {"Reconnaissance": {"originates": torch.tensor([2.0])}}

    merged = merge_into_bank(bank, new_means)

    assert set(merged.keys()) == {"Scanning", "Reconnaissance"}
    # Original bank dict is untouched (merge_into_bank returns a new dict).
    assert set(bank.keys()) == {"Scanning"}


def test_merge_into_bank_rejects_duplicate_class() -> None:
    bank = {"DDoS": {"originates": torch.tensor([1.0])}}
    new_means = {"DDoS": {"originates": torch.tensor([2.0])}}

    with pytest.raises(ValueError, match="already in the memory bank"):
        merge_into_bank(bank, new_means)
