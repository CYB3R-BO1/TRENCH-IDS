from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from trench_ids.cl.evaluate import (
    build_eval_matrix,
    compute_metrics,
    forgetting_metrics_table,
    pool_predictions,
    run,
)
from trench_ids.cl.train import save_checkpoint
from trench_ids.model.rhgnn import FLOW_FEATURE_DIM, HOST_FEATURE_DIM, RelationSpecificHeteroGNN


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


def _tiny_graph() -> HeteroData:
    g = HeteroData()
    g["flow"].x = torch.randn(4, FLOW_FEATURE_DIM)
    g["flow"].y = torch.tensor([0, 1, 0, 1])
    g["host"].num_nodes = 2
    g["host"].x = torch.randn(2, HOST_FEATURE_DIM)
    g["protocol"].num_nodes = 1
    g["protocol"].vocab_id = torch.tensor([0])
    g["service"].num_nodes = 1
    g["service"].vocab_id = torch.tensor([0])
    g["port"].num_nodes = 2
    g["port"].port_number = torch.tensor([22, 80])

    src_host = torch.tensor([0, 1, 0, 1])
    dst_host = torch.tensor([1, 0, 1, 0])
    protocol_idx = torch.tensor([0, 0, 0, 0])
    service_idx = torch.tensor([0, 0, 0, 0])
    port_idx = torch.tensor([0, 1, 0, 1])

    g["host", "originates", "flow"].edge_index = torch.stack([src_host, torch.arange(4)])
    g["flow", "terminates_at", "host"].edge_index = torch.stack([torch.arange(4), dst_host])
    g["flow", "targets_port", "port"].edge_index = torch.stack([torch.arange(4), port_idx])
    g["flow", "uses_protocol", "protocol"].edge_index = torch.stack([torch.arange(4), protocol_idx])
    g["flow", "uses_service", "service"].edge_index = torch.stack([torch.arange(4), service_idx])
    g["host", "communicates_with", "host"].edge_index = torch.tensor([[0], [1]])
    g["flow", "originated_by", "host"].edge_index = torch.stack([torch.arange(4), src_host])
    g["host", "terminated_by", "flow"].edge_index = torch.stack([dst_host, torch.arange(4)])
    g["port", "targeted_by", "flow"].edge_index = torch.stack([port_idx, torch.arange(4)])
    g["protocol", "protocol_of", "flow"].edge_index = torch.stack([protocol_idx, torch.arange(4)])
    g["service", "service_of", "flow"].edge_index = torch.stack([service_idx, torch.arange(4)])
    return g


def test_build_eval_matrix_and_run_end_to_end(tmp_path) -> None:
    device = torch.device("cpu")
    label_names = ["a", "b"]
    g = _tiny_graph()

    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1,
    ).to(device)
    classifier = torch.nn.Linear(8, 2).to(device)

    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    for task in (1, 2):
        for split in ("train", "test"):
            torch.save([g, g], graphs_dir / f"task_{task}_{split}.pt")

    run_dir = tmp_path / "run"
    config = {
        "hidden_dim": 8, "num_layers": 1, "attn_dim": 128, "port_buckets": 32,
        "protocol_vocab_size": 2, "service_vocab_size": 2, "label_names": label_names,
    }
    for task in (1, 2):
        save_checkpoint(
            run_dir / f"checkpoint_task_{task}.pt", task_id=task, epochs_per_task=1,
            warmup_epochs=0, seed=42, model=model, classifier=classifier, config=config,
        )

    eval_matrix, predictions = build_eval_matrix(
        run_dir, graphs_dir, device, batch_size=4, num_tasks=2
    )

    assert set(eval_matrix.keys()) == {1, 2}
    assert set(eval_matrix[1].keys()) == {1}
    assert set(eval_matrix[2].keys()) == {1, 2}
    for row in eval_matrix.values():
        for cell in row.values():
            assert cell["label_names"] == label_names
            assert len(cell["confusion_matrix"]) == 2

    report = run(run_dir, graphs_dir, device, batch_size=4, num_tasks=2)
    assert "pooled_final_metrics" in report
    assert "forgetting_metrics" in report
    assert len(report["forgetting_metrics"]) == 3  # 1 (trained_up_to=1) + 2 (trained_up_to=2) cells
