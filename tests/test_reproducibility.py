"""Reproducibility test: the same seed must produce byte-identical training
results. seed_everything (train.py) was fixed 2026-08-07 to seed Python's
random and NumPy in addition to torch -- before that, the replay buffer's
random.sample draws ran off OS entropy and same-config runs diverged. This
test guards that fix: if any RNG consumer in the train_one_task path starts
drawing from an unseeded source, two same-seed runs stop matching.
"""

from __future__ import annotations

import copy

import torch
from torch_geometric.data import HeteroData

from trench_ids.cl.ewc import FLOW_RELATIONS, OnlineEWCManager
from trench_ids.cl.importance import ImportanceMLP
from trench_ids.cl.train import seed_everything, train_one_task
from trench_ids.constants import FLOW_FEATURE_DIM, HOST_FEATURE_DIM
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN

HIDDEN_DIM = 8
LABEL_NAMES = ["a", "b"]


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
    g["flow", "uses_protocol", "protocol"].edge_index = torch.stack(
        [torch.arange(4), protocol_idx]
    )
    g["flow", "uses_service", "service"].edge_index = torch.stack(
        [torch.arange(4), service_idx]
    )
    g["host", "communicates_with", "host"].edge_index = torch.tensor([[0], [1]])
    g["flow", "originated_by", "host"].edge_index = torch.stack([torch.arange(4), src_host])
    g["host", "terminated_by", "flow"].edge_index = torch.stack([dst_host, torch.arange(4)])
    g["port", "targeted_by", "flow"].edge_index = torch.stack([port_idx, torch.arange(4)])
    g["protocol", "protocol_of", "flow"].edge_index = torch.stack(
        [protocol_idx, torch.arange(4)]
    )
    g["service", "service_of", "flow"].edge_index = torch.stack(
        [service_idx, torch.arange(4)]
    )
    return g


def _run_one_task(graphs: list[HeteroData]) -> dict[str, torch.Tensor]:
    """One full train_one_task call on freshly seeded modules; returns the
    trained model's state_dict."""
    device = torch.device("cpu")
    seed_everything(42)
    # Graph tensors are read-only inside train_one_task, but copy anyway so
    # neither run can ever observe the other's buffers.
    run_graphs = [copy.deepcopy(g) for g in graphs]

    model = RelationSpecificHeteroGNN.from_graph(
        run_graphs[0],
        hidden_dim=HIDDEN_DIM,
        protocol_vocab_size=2,
        service_vocab_size=2,
        num_layers=1,
    ).to(device)
    classifier = torch.nn.Linear(HIDDEN_DIM, len(LABEL_NAMES)).to(device)
    importance_mlp = ImportanceMLP().to(device)
    ewc_manager = OnlineEWCManager(
        model, classifier, FLOW_RELATIONS, gamma=0.9, lambda_r=1.0, lambda_s=1.0, lambda_u=1.0,
    )
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(classifier.parameters())
        + list(importance_mlp.parameters()),
        lr=1e-3,
    )
    bank = {"attack_a": {relation: torch.randn(HIDDEN_DIM) for relation in FLOW_RELATIONS}}

    train_one_task(
        model,
        classifier,
        run_graphs,
        device,
        batch_size=2,
        warmup_epochs=1,
        full_loss_epochs=1,
        optimizer=optimizer,
        ewc_manager=ewc_manager,
        importance_mlp=importance_mlp,
        label_names=LABEL_NAMES,
        bank=bank,
        task_id=1,
        seed=42,
    )
    return copy.deepcopy(model.state_dict())


def test_same_seed_produces_identical_model_state() -> None:
    seed_everything(7)
    graphs = [_tiny_graph(), _tiny_graph(), _tiny_graph()]

    state_a = _run_one_task(graphs)
    state_b = _run_one_task(graphs)

    assert set(state_a) == set(state_b)
    for name in state_a:
        assert torch.equal(state_a[name], state_b[name]), f"parameter {name} diverged"
