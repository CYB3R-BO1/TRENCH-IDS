from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from trench_ids.cl.ewc import FLOW_RELATIONS, all_named_parameters, partition_parameter_names
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
