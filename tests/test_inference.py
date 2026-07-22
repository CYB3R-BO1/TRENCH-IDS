from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.train import save_checkpoint
from trench_ids.model.rhgnn import FLOW_FEATURE_DIM, HOST_FEATURE_DIM, RelationSpecificHeteroGNN


def _tiny_graph() -> HeteroData:
    """Real-size feature dims (FLOW_FEATURE_DIM/HOST_FEATURE_DIM) so a model
    built via RelationSpecificHeteroGNN.from_graph's default encoders (no
    manual override, unlike tests/test_train.py's reduced-dim fixture)
    matches load_checkpoint's real reconstruction path exactly."""
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


def test_predict_returns_well_formed_output() -> None:
    device = torch.device("cpu")
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1,
    ).to(device)
    classifier = torch.nn.Linear(8, 2).to(device)
    label_names = ["a", "b"]

    result = predict(model, classifier, [g, g], device, batch_size=2, label_names=label_names)

    assert len(result["y_true"]) == 8  # 4 flow nodes * 2 graphs
    assert len(result["y_pred"]) == 8
    assert len(result["y_prob"]) == 8
    assert all(len(row) == 2 for row in result["y_prob"])
    assert all(abs(sum(row) - 1.0) < 1e-5 for row in result["y_prob"])
    assert all(p in (0, 1) for p in result["y_pred"])
    assert result["label_names"] == label_names


def test_load_checkpoint_rebuilds_model_and_predicts(tmp_path) -> None:
    device = torch.device("cpu")
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1,
    ).to(device)
    classifier = torch.nn.Linear(8, 2).to(device)
    config = {
        "hidden_dim": 8, "num_layers": 1, "attn_dim": 128, "port_buckets": 32,
        "protocol_vocab_size": 2, "service_vocab_size": 2, "label_names": ["a", "b"],
    }

    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    torch.save([g, g], graphs_dir / "task_1_train.pt")

    checkpoint_path = tmp_path / "checkpoint_task_1.pt"
    save_checkpoint(
        checkpoint_path, task_id=1, epochs_per_task=1, warmup_epochs=0, seed=42,
        model=model, classifier=classifier, config=config,
    )

    loaded_model, loaded_classifier, checkpoint = load_checkpoint(
        checkpoint_path, graphs_dir, device
    )

    assert checkpoint["task_id"] == 1
    assert checkpoint["config"]["label_names"] == ["a", "b"]

    result = predict(
        loaded_model, loaded_classifier, [g], device, batch_size=2,
        label_names=["a", "b"],
    )
    assert len(result["y_true"]) == 4


def test_load_checkpoint_rejects_unsupported_version(tmp_path) -> None:
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    g = _tiny_graph()
    torch.save([g], graphs_dir / "task_1_train.pt")

    checkpoint_path = tmp_path / "checkpoint_bad.pt"
    torch.save({"checkpoint_version": 999, "config": {}}, checkpoint_path)

    import pytest

    with pytest.raises(ValueError, match="checkpoint_version"):
        load_checkpoint(checkpoint_path, graphs_dir, torch.device("cpu"))
