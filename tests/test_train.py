from __future__ import annotations

import pytest

from trench_ids.cl.train import forgetting_row, split_warmup_and_full_loss_epochs


def test_forgetting_row_evaluates_every_previously_seen_task() -> None:
    fake_accuracies = {1: 0.9, 2: 0.8, 3: 0.7}

    row = forgetting_row(lambda t: fake_accuracies[t], up_to_task=3)

    assert row == {1: 0.9, 2: 0.8, 3: 0.7}


def test_forgetting_row_after_first_task_is_a_single_entry() -> None:
    row = forgetting_row(lambda t: 1.0, up_to_task=1)

    assert row == {1: 1.0}


def test_forgetting_row_calls_accuracy_fn_once_per_task() -> None:
    calls: list[int] = []

    def accuracy_fn(task: int) -> float:
        calls.append(task)
        return float(task)

    forgetting_row(accuracy_fn, up_to_task=4)

    assert calls == [1, 2, 3, 4]


def test_split_warmup_and_full_loss_epochs_matches_config() -> None:
    warmup, full = split_warmup_and_full_loss_epochs(epochs_per_task=5, warmup_epochs=2)
    assert warmup == 2
    assert full == 3


def test_split_warmup_and_full_loss_epochs_rejects_warmup_exceeding_total() -> None:
    with pytest.raises(ValueError, match="warmup_epochs"):
        split_warmup_and_full_loss_epochs(epochs_per_task=3, warmup_epochs=5)
