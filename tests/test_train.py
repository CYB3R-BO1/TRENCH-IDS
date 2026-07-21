from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from trench_ids.cl.ewc import FLOW_RELATIONS, OnlineEWCManager
from trench_ids.cl.importance import ImportanceMLP
from trench_ids.cl.train import forgetting_row, split_warmup_and_full_loss_epochs, train_one_task
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN

FLOW_DIM = 3
HOST_DIM = 2


def _tiny_graph() -> HeteroData:
    """Minimal HeteroData mirroring the real 5-node-type/11-relation schema
    (same shape convention as tests/test_ewc.py's _tiny_graph, kept
    self-contained here so test_train.py has no cross-test-file import)."""
    g = HeteroData()
    g["flow"].x = torch.randn(4, FLOW_DIM)
    g["flow"].y = torch.tensor([0, 1, 0, 1])
    g["host"].num_nodes = 2
    g["host"].x = torch.randn(2, HOST_DIM)
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="regression test for a device mismatch that only manifests with a real non-CPU device",
)
def test_train_one_task_moves_s_r_onto_device_before_importance_mlp() -> None:
    """Regression test: aggregate_transferability_scores builds S_r via
    plain torch.tensor(...) with no device= argument, so it always lands on
    CPU. train_one_task must move S_r onto `device` before importance_mlp
    (already moved to `device`) consumes it, or this raises a device-mismatch
    RuntimeError on any machine with a real GPU (this dev box included --
    CLAUDE.md notes it gained a CUDA device 2026-07-19)."""
    device = torch.device("cuda")
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1,
    )
    from trench_ids.model.rhgnn import NodeFeatureEncoders

    model.encoders = NodeFeatureEncoders(
        8, protocol_vocab_size=2, service_vocab_size=2,
        flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM,
    )
    model = model.to(device)
    classifier = torch.nn.Linear(8, 2).to(device)
    importance_mlp = ImportanceMLP().to(device)
    ewc_manager = OnlineEWCManager(
        model, classifier, FLOW_RELATIONS, gamma=0.9, lambda_r=1.0, lambda_s=1.0, lambda_u=1.0,
    )
    trainable_params = (
        list(model.parameters())
        + list(classifier.parameters())
        + list(importance_mlp.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    # bank={} -> S_r comes back as CPU tensors of 0.0 (empty-average default).
    # Without the fix, importance_mlp(s_r) inside train_one_task raises
    # "Expected all tensors to be on the same device" the first time the
    # full-loss loop runs.
    train_one_task(
        model,
        classifier,
        [g, g],
        device,
        batch_size=2,
        warmup_epochs=1,
        full_loss_epochs=1,
        optimizer=optimizer,
        ewc_manager=ewc_manager,
        importance_mlp=importance_mlp,
        label_names=["a", "b"],
        bank={},
    )
