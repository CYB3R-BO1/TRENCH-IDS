"""Tests for Step 10 Dataset B behavioral analysis
(trench_ids.cl.inference_unseen)."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from trench_ids.cl.inference_unseen import (
    compute_class_relation_means,
    confidence_stats,
    prediction_distribution,
    similarity_report,
    similarity_rows,
)
from trench_ids.model.rhgnn import FLOW_FEATURE_DIM, HOST_FEATURE_DIM, RelationSpecificHeteroGNN


def _tiny_graph() -> HeteroData:
    g = HeteroData()
    g["flow"].x = torch.randn(4, FLOW_FEATURE_DIM)
    g["flow"].y = torch.tensor([-1, -1, -1, -1])
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


def test_prediction_distribution_sums_to_100_percent() -> None:
    y_pred = [0, 0, 0, 1]
    dist = prediction_distribution(y_pred, ["Benign", "DoS"])

    assert sum(row["percentage"] for row in dist) == 100.0
    assert dist[0]["predicted_class"] == "Benign"
    assert dist[0]["count"] == 3
    assert dist[0]["percentage"] == 75.0
    assert dist[1]["predicted_class"] == "DoS"
    assert dist[1]["count"] == 1


def test_prediction_distribution_empty_input() -> None:
    assert prediction_distribution([], ["Benign", "DoS"]) == []


def test_confidence_stats_computes_mean_median_std() -> None:
    y_prob = [[0.9, 0.1], [0.6, 0.4], [1.0, 0.0]]
    stats = confidence_stats(y_prob)

    assert abs(stats["mean_max_softmax"] - (0.9 + 0.6 + 1.0) / 3) < 1e-9
    assert stats["median_max_softmax"] == 0.9
    assert stats["std_max_softmax"] > 0.0


def test_confidence_stats_empty_input() -> None:
    stats = confidence_stats([])
    assert stats == {"mean_max_softmax": 0.0, "median_max_softmax": 0.0, "std_max_softmax": 0.0}


def test_compute_class_relation_means_returns_one_vector_per_relation() -> None:
    device = torch.device("cpu")
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=1, service_vocab_size=1, num_layers=1,
    ).to(device)

    means = compute_class_relation_means(model, [g, g], device, batch_size=2)

    assert set(means.keys()) == {"originates", "terminated_by", "targeted_by", "protocol_of", "service_of"}
    for vec in means.values():
        assert vec.shape == (8,)


def test_similarity_report_renests_by_relation() -> None:
    unseen_means = {"Backdoor": {"originates": torch.tensor([1.0, 0.0])}}
    bank = {
        "Bot": {"originates": torch.tensor([1.0, 0.0])},
        "DoS": {"originates": torch.tensor([0.0, 1.0])},
    }

    nested = similarity_report(unseen_means, bank)

    assert nested["Backdoor"]["originates"]["Bot"] == 1.0
    assert nested["Backdoor"]["originates"]["DoS"] == 0.0


def test_similarity_rows_flattens_nested_report() -> None:
    nested = {"Backdoor": {"originates": {"Bot": 0.87, "DoS": 0.1}}}
    rows = similarity_rows(nested)

    assert ("Backdoor", "originates", "Bot", 0.87) in rows
    assert ("Backdoor", "originates", "DoS", 0.1) in rows
    assert len(rows) == 2
