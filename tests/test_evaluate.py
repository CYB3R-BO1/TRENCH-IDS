from __future__ import annotations

import pytest

from trench_ids.cl.evaluate import compute_metrics, forgetting_metrics_table, pool_predictions


def test_compute_metrics_handles_never_predicted_and_never_true_classes() -> None:
    label_names = ["Benign", "A", "B"]
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]  # class "B" (idx 2) never true, never predicted

    metrics = compute_metrics(y_true, y_pred, label_names)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["per_class"]["B"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0,
    }
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    assert metrics["label_names"] == label_names


def test_compute_metrics_precision_recall_f1_hand_computed() -> None:
    label_names = ["Benign", "A"]
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = compute_metrics(y_true, y_pred, label_names)

    # Benign: precision 1/1=1.0, recall 1/2=0.5, f1 2*1*0.5/1.5=0.6667
    assert metrics["per_class"]["Benign"]["precision"] == pytest.approx(1.0)
    assert metrics["per_class"]["Benign"]["recall"] == pytest.approx(0.5)
    assert metrics["per_class"]["Benign"]["f1"] == pytest.approx(2 / 3)
    # A: precision 2/3=0.6667, recall 2/2=1.0
    assert metrics["per_class"]["A"]["precision"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["A"]["recall"] == pytest.approx(1.0)


def test_pool_predictions_concatenates_without_reweighting() -> None:
    label_names = ["Benign", "A"]
    p1 = {
        "y_true": [0, 1],
        "y_pred": [0, 0],
        "y_prob": [[0.9, 0.1], [0.6, 0.4]],
        "label_names": label_names,
    }
    p2 = {
        "y_true": [1, 1],
        "y_pred": [1, 1],
        "y_prob": [[0.1, 0.9], [0.2, 0.8]],
        "label_names": label_names,
    }

    pooled = pool_predictions([p1, p2])

    assert pooled["y_true"] == [0, 1, 1, 1]
    assert pooled["y_pred"] == [0, 0, 1, 1]
    assert pooled["label_names"] == label_names


def test_pool_predictions_rejects_mismatched_label_names() -> None:
    p1 = {"y_true": [0], "y_pred": [0], "y_prob": [[1.0]], "label_names": ["Benign"]}
    p2 = {"y_true": [0], "y_pred": [0], "y_prob": [[1.0, 0.0]], "label_names": ["Benign", "A"]}

    with pytest.raises(ValueError, match="label_names"):
        pool_predictions([p1, p2])


def test_forgetting_metrics_table_flattens_and_sorts() -> None:
    eval_matrix = {
        2: {1: {"accuracy": 0.8, "f1_macro": 0.7}, 2: {"accuracy": 0.9, "f1_macro": 0.85}},
        1: {1: {"accuracy": 0.95, "f1_macro": 0.9}},
    }

    rows = forgetting_metrics_table(eval_matrix)

    assert rows == [
        {"trained_up_to": 1, "evaluated_task": 1, "accuracy": 0.95, "f1_macro": 0.9},
        {"trained_up_to": 2, "evaluated_task": 1, "accuracy": 0.8, "f1_macro": 0.7},
        {"trained_up_to": 2, "evaluated_task": 2, "accuracy": 0.9, "f1_macro": 0.85},
    ]
