from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.cl.ewc import (
    FLOW_RELATIONS,
    OnlineEWCManager,
    all_named_parameters,
    estimate_fisher,
    partition_parameter_names,
)
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN

FLOW_DIM = 3
HOST_DIM = 2


def _tiny_graph() -> HeteroData:
    """Minimal HeteroData mirroring the real 5-node-type/11-relation schema
    (same shape convention as tests/test_model.py's _synthetic_graph, kept
    self-contained here so test_ewc.py has no cross-test-file import)."""
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


def _tiny_model_and_classifier(num_layers: int = 1):
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=num_layers,
    )
    from trench_ids.model.rhgnn import NodeFeatureEncoders
    model.encoders = NodeFeatureEncoders(
        8, protocol_vocab_size=2, service_vocab_size=2,
        flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM,
    )
    classifier = torch.nn.Linear(8, 2)
    return g, model, classifier


def test_flow_relations_has_exactly_five_names() -> None:
    assert set(FLOW_RELATIONS) == {
        "originates", "terminated_by", "targeted_by", "protocol_of", "service_of",
    }


def test_all_named_parameters_prefixes_model_and_classifier() -> None:
    _, model, classifier = _tiny_model_and_classifier()
    params = all_named_parameters(model, classifier)
    assert any(name.startswith("model.encoders.") for name in params)
    assert any(name.startswith("classifier.") for name in params)


def test_partition_parameter_names_separates_flow_and_other_relations() -> None:
    _, model, classifier = _tiny_model_and_classifier()
    groups = partition_parameter_names(model, classifier, FLOW_RELATIONS)

    assert set(groups.keys()) == {
        "shared",
        "originates", "terminated_by", "targeted_by", "protocol_of", "service_of",
        "other::terminates_at", "other::communicates_with", "other::originated_by",
        "other::targets_port", "other::uses_protocol", "other::uses_service",
    }
    # Every relation-specific group is non-empty (rel_lins + combine_lins weight+bias = 4 names).
    assert len(groups["originates"]) == 4
    assert len(groups["other::terminates_at"]) == 4
    # Encoders/fusion/classifier land in "shared", never in a relation group.
    assert any("encoders" in name for name in groups["shared"])
    assert any("fusion" in name for name in groups["shared"])
    assert any(name.startswith("classifier.") for name in groups["shared"])


def test_partition_parameter_names_groups_across_multiple_layers() -> None:
    _, model, classifier = _tiny_model_and_classifier(num_layers=2)
    groups = partition_parameter_names(model, classifier, FLOW_RELATIONS)
    # 2 layers x 4 names (rel_lins/combine_lins weight+bias) = 8 for one relation.
    assert len(groups["originates"]) == 8


def test_online_ewc_state_penalty_is_zero_before_first_update() -> None:
    from trench_ids.cl.ewc import OnlineEWCState

    state = OnlineEWCState(["a"], gamma=0.9)
    current = {"a": torch.tensor([1.0, 2.0])}
    assert state.penalty(current).item() == 0.0


def test_online_ewc_state_penalty_matches_hand_computed_value_after_update() -> None:
    from trench_ids.cl.ewc import OnlineEWCState

    state = OnlineEWCState(["a"], gamma=0.9)
    fisher = {"a": torch.tensor([2.0, 4.0])}
    theta_star = {"a": torch.tensor([1.0, 1.0])}
    state.update(fisher, theta_star)

    current = {"a": torch.tensor([3.0, 1.0])}
    # sum(fisher * (current - theta_star)**2) = 2*(3-1)^2 + 4*(1-1)^2 = 8 + 0 = 8
    assert state.penalty(current).item() == pytest.approx(8.0)


def test_online_ewc_state_gamma_blends_fisher_across_two_updates() -> None:
    from trench_ids.cl.ewc import OnlineEWCState

    state = OnlineEWCState(["a"], gamma=0.5)
    state.update({"a": torch.tensor([4.0])}, {"a": torch.tensor([0.0])})
    state.update({"a": torch.tensor([2.0])}, {"a": torch.tensor([0.0])})
    # fisher = 0.5*4 + 2 = 4.0
    assert state.fisher["a"].item() == pytest.approx(4.0)


def test_online_ewc_state_theta_star_is_overwritten_not_blended() -> None:
    from trench_ids.cl.ewc import OnlineEWCState

    state = OnlineEWCState(["a"], gamma=0.9)
    state.update({"a": torch.tensor([1.0])}, {"a": torch.tensor([5.0])})
    state.update({"a": torch.tensor([1.0])}, {"a": torch.tensor([9.0])})
    assert state.theta_star["a"].item() == pytest.approx(9.0)


def test_estimate_fisher_covers_every_named_parameter() -> None:
    g, model, classifier = _tiny_model_and_classifier()
    loader = DataLoader([g, g], batch_size=1)
    device = torch.device("cpu")

    fisher = estimate_fisher(model, classifier, loader, device)

    expected_names = set(all_named_parameters(model, classifier).keys())
    assert set(fisher.keys()) == expected_names


def test_estimate_fisher_values_are_nonnegative() -> None:
    g, model, classifier = _tiny_model_and_classifier()
    loader = DataLoader([g, g], batch_size=1)
    device = torch.device("cpu")

    fisher = estimate_fisher(model, classifier, loader, device)

    for tensor in fisher.values():
        assert (tensor >= 0).all()


def test_estimate_fisher_restores_training_mode() -> None:
    g, model, classifier = _tiny_model_and_classifier()
    loader = DataLoader([g, g], batch_size=1)
    device = torch.device("cpu")
    model.train()
    classifier.train()

    estimate_fisher(model, classifier, loader, device)

    assert model.training
    assert classifier.training


def test_estimate_fisher_restores_mismatched_training_modes_independently() -> None:
    g, model, classifier = _tiny_model_and_classifier()
    loader = DataLoader([g, g], batch_size=1)
    device = torch.device("cpu")
    model.train()
    classifier.eval()

    estimate_fisher(model, classifier, loader, device)

    assert model.training
    assert not classifier.training


def test_online_ewc_manager_loss_is_zero_before_any_update() -> None:
    _, model, classifier = _tiny_model_and_classifier()
    manager = OnlineEWCManager(
        model, classifier, FLOW_RELATIONS, gamma=0.9, lambda_r=1.0, lambda_s=1.0, lambda_u=1.0,
    )
    w_r = {relation: torch.tensor(1.0) for relation in FLOW_RELATIONS}

    loss = manager.loss(model, classifier, w_r)

    assert loss.item() == pytest.approx(0.0)


def test_online_ewc_manager_loss_nonzero_after_update_and_param_change() -> None:
    g, model, classifier = _tiny_model_and_classifier()
    loader = DataLoader([g, g], batch_size=1)
    device = torch.device("cpu")
    manager = OnlineEWCManager(
        model, classifier, FLOW_RELATIONS, gamma=0.9, lambda_r=1.0, lambda_s=1.0, lambda_u=1.0,
    )

    manager.update_all(model, classifier, loader, device)
    with torch.no_grad():
        for p in model.parameters():
            p += 1.0  # move every parameter away from its just-recorded theta*

    w_r = {relation: torch.tensor(1.0) for relation in FLOW_RELATIONS}
    loss = manager.loss(model, classifier, w_r)

    assert loss.item() > 0.0


def test_online_ewc_manager_weights_flow_relations_by_w_r() -> None:
    g, model, classifier = _tiny_model_and_classifier()
    loader = DataLoader([g, g], batch_size=1)
    device = torch.device("cpu")
    manager = OnlineEWCManager(
        model, classifier, FLOW_RELATIONS, gamma=0.9, lambda_r=1.0, lambda_s=0.0, lambda_u=0.0,
    )
    manager.update_all(model, classifier, loader, device)
    with torch.no_grad():
        for p in model.parameters():
            p += 1.0

    w_zero = {relation: torch.tensor(0.0) for relation in FLOW_RELATIONS}
    w_one = {relation: torch.tensor(1.0) for relation in FLOW_RELATIONS}

    loss_zero = manager.loss(model, classifier, w_zero)
    loss_one = manager.loss(model, classifier, w_one)

    assert loss_zero.item() == pytest.approx(0.0)
    assert loss_one.item() > 0.0
