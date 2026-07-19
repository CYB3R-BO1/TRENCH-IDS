from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from trench_ids.model.attention_fusion import SemanticAttention
from trench_ids.model.relation_conv import RelationSpecificConv
from trench_ids.model.rhgnn import (
    NodeFeatureEncoders,
    RelationSpecificHeteroGNN,
    RelationSpecificLayer,
    port_bucket,
)

FLOW_DIM = 5
HOST_DIM = 2


def _synthetic_graph() -> HeteroData:
    """A small HeteroData mirroring the real Step 2 schema exactly (5 node
    types, 11 relations -- the original 6 one-directional relations plus 5
    reverse relations) -- small enough for fast unit tests, same shape
    conventions as `trench_ids.graphs.build_task_graph`."""
    g = HeteroData()

    g["flow"].x = torch.randn(6, FLOW_DIM)
    g["flow"].y = torch.tensor([0, 0, 1, 1, 0, 1])

    g["host"].num_nodes = 3
    g["host"].x = torch.randn(3, HOST_DIM)

    g["protocol"].num_nodes = 2
    g["protocol"].vocab_id = torch.tensor([0, 1])

    g["service"].num_nodes = 2
    g["service"].vocab_id = torch.tensor([3, 7])

    g["port"].num_nodes = 3
    g["port"].port_number = torch.tensor([22, 80, 50000])

    src_host = torch.tensor([0, 0, 1, 1, 2, 2])
    dst_host = torch.tensor([1, 2, 0, 2, 0, 1])
    protocol_idx = torch.tensor([0, 0, 1, 1, 0, 1])
    service_idx = torch.tensor([0, 1, 0, 1, 0, 1])
    port_idx = torch.tensor([0, 1, 2, 0, 1, 2])

    g["host", "originates", "flow"].edge_index = torch.stack(
        [src_host, torch.arange(6)]
    )
    g["flow", "terminates_at", "host"].edge_index = torch.stack(
        [torch.arange(6), dst_host]
    )
    g["flow", "targets_port", "port"].edge_index = torch.stack(
        [torch.arange(6), port_idx]
    )
    g["flow", "uses_protocol", "protocol"].edge_index = torch.stack(
        [torch.arange(6), protocol_idx]
    )
    g["flow", "uses_service", "service"].edge_index = torch.stack(
        [torch.arange(6), service_idx]
    )
    g["host", "communicates_with", "host"].edge_index = torch.tensor(
        [[0, 1], [1, 2]]
    )

    # 5 reverse relations (mirror the 5 flow-centric relations above).
    g["flow", "originated_by", "host"].edge_index = torch.stack(
        [torch.arange(6), src_host]
    )
    g["host", "terminated_by", "flow"].edge_index = torch.stack(
        [dst_host, torch.arange(6)]
    )
    g["port", "targeted_by", "flow"].edge_index = torch.stack(
        [port_idx, torch.arange(6)]
    )
    g["protocol", "protocol_of", "flow"].edge_index = torch.stack(
        [protocol_idx, torch.arange(6)]
    )
    g["service", "service_of", "flow"].edge_index = torch.stack(
        [service_idx, torch.arange(6)]
    )

    return g


# ---------------------------------------------------------------------------
# port_bucket
# ---------------------------------------------------------------------------


def test_port_bucket_stays_in_range() -> None:
    ports = torch.tensor([0, 22, 80, 1023, 8080, 65535])
    buckets = port_bucket(ports, num_buckets=32)
    assert buckets.min() >= 0
    assert buckets.max() < 32


def test_port_bucket_is_monotonic_nondecreasing() -> None:
    ports = torch.tensor([0, 10, 100, 1000, 10000, 65535])
    buckets = port_bucket(ports, num_buckets=16)
    diffs = buckets[1:] - buckets[:-1]
    assert (diffs >= 0).all()


def test_port_bucket_clamps_out_of_range_values() -> None:
    ports = torch.tensor([-5, 70000])
    buckets = port_bucket(ports, num_buckets=8)
    assert buckets[0] == port_bucket(torch.tensor([0]), num_buckets=8)[0]
    assert buckets[1] == port_bucket(torch.tensor([65535]), num_buckets=8)[0]


# ---------------------------------------------------------------------------
# NodeFeatureEncoders
# ---------------------------------------------------------------------------


def test_node_feature_encoders_produce_hidden_dim_for_every_node_type() -> None:
    g = _synthetic_graph()
    encoders = NodeFeatureEncoders(
        hidden_dim=16,
        protocol_vocab_size=5,
        service_vocab_size=10,
        flow_feature_dim=FLOW_DIM,
        host_feature_dim=HOST_DIM,
        port_buckets=32,
    )
    x_dict = encoders(g)
    assert x_dict["flow"].shape == (6, 16)
    assert x_dict["host"].shape == (3, 16)
    assert x_dict["protocol"].shape == (2, 16)
    assert x_dict["service"].shape == (2, 16)
    assert x_dict["port"].shape == (3, 16)


# ---------------------------------------------------------------------------
# RelationSpecificConv
# ---------------------------------------------------------------------------


def test_relation_specific_conv_keeps_relations_separate_per_node_type() -> None:
    g = _synthetic_graph()
    encoders = NodeFeatureEncoders(16, protocol_vocab_size=5, service_vocab_size=10,
                                    flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM)
    x_dict = encoders(g)
    node_types, edge_types = g.metadata()
    conv = RelationSpecificConv(edge_types, hidden_dim=16)

    relation_embeds = conv(x_dict, g.edge_index_dict)

    assert set(relation_embeds["flow"].keys()) == {
        "originates",
        "targeted_by",
        "protocol_of",
        "service_of",
        "terminated_by",
    }
    for tensor in relation_embeds["flow"].values():
        assert tensor.shape == (6, 16)
    assert set(relation_embeds["host"].keys()) == {
        "terminates_at",
        "communicates_with",
        "originated_by",
    }
    assert set(relation_embeds["protocol"].keys()) == {"uses_protocol"}


def test_relation_specific_conv_incorporates_destination_own_features() -> None:
    """The self-preservation fix: a relation's stored embedding must depend
    on the destination node's own current embedding, not just neighbor
    messages -- on the pre-fix code this test would fail (aggregated
    neighbor messages alone, x_dict[dst_type] never referenced)."""
    g = _synthetic_graph()
    encoders = NodeFeatureEncoders(
        16,
        protocol_vocab_size=5,
        service_vocab_size=10,
        flow_feature_dim=FLOW_DIM,
        host_feature_dim=HOST_DIM,
    )
    x_dict = encoders(g)
    node_types, edge_types = g.metadata()
    conv = RelationSpecificConv(edge_types, hidden_dim=16)

    baseline = conv(x_dict, g.edge_index_dict)

    perturbed_x_dict = dict(x_dict)
    perturbed_x_dict["flow"] = x_dict["flow"] + 100.0
    perturbed = conv(perturbed_x_dict, g.edge_index_dict)

    # Relations whose destination is "flow" must change when flow's own
    # embedding changes, even though neighbor messages (from host/port/
    # protocol/service) are untouched.
    for relation in ("originates", "targeted_by", "protocol_of", "service_of", "terminated_by"):
        assert not torch.allclose(baseline["flow"][relation], perturbed["flow"][relation])


def test_relation_specific_conv_uses_distinct_weights_per_relation() -> None:
    g = _synthetic_graph()
    node_types, edge_types = g.metadata()
    conv = RelationSpecificConv(edge_types, hidden_dim=8)

    originates_w = conv.rel_lins["host__originates__flow"].weight
    terminates_w = conv.rel_lins["flow__terminates_at__host"].weight

    assert originates_w is not terminates_w
    assert not torch.allclose(originates_w, terminates_w)


# ---------------------------------------------------------------------------
# SemanticAttention
# ---------------------------------------------------------------------------


def test_semantic_attention_single_relation_is_identity() -> None:
    fusion = SemanticAttention(hidden_dim=8)
    z = torch.randn(4, 8)
    fused, beta = fusion([z])
    assert torch.equal(fused, z)
    assert beta.tolist() == pytest.approx([1.0])


def test_semantic_attention_weights_sum_to_one_and_shapes_match() -> None:
    fusion = SemanticAttention(hidden_dim=8)
    embeds = [torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)]
    fused, beta = fusion(embeds)
    assert fused.shape == (4, 8)
    assert beta.shape == (3,)
    assert beta.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (beta >= 0).all()


def test_semantic_attention_gradients_flow_to_all_relations() -> None:
    fusion = SemanticAttention(hidden_dim=8)
    embeds = [torch.randn(4, 8, requires_grad=True) for _ in range(3)]
    fused, _ = fusion(embeds)
    fused.sum().backward()
    for z in embeds:
        assert z.grad is not None
        assert not torch.all(z.grad == 0)


# ---------------------------------------------------------------------------
# RelationSpecificLayer / full model
# ---------------------------------------------------------------------------


def test_relation_specific_layer_attention_sums_to_one_per_node_type() -> None:
    g = _synthetic_graph()
    node_types, edge_types = g.metadata()
    encoders = NodeFeatureEncoders(16, protocol_vocab_size=5, service_vocab_size=10,
                                    flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM)
    x_dict = encoders(g)
    layer = RelationSpecificLayer(edge_types, node_types, hidden_dim=16)

    output = layer(x_dict, g.edge_index_dict)

    for _node_type, betas in output.attention.items():
        if not betas:
            continue
        assert sum(betas.values()) == pytest.approx(1.0, abs=1e-5)
    assert output.fused["flow"].shape == (6, 16)
    assert output.fused["port"].shape == (3, 16)


def test_full_model_forward_pass_shapes() -> None:
    g = _synthetic_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=16, protocol_vocab_size=5, service_vocab_size=10, num_layers=2
    )
    # Patch feature dims to match the small synthetic graph.
    model.encoders = NodeFeatureEncoders(
        16, protocol_vocab_size=5, service_vocab_size=10,
        flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM,
    )

    output = model(g)

    assert output.fused["flow"].shape == (6, 16)
    assert output.fused["host"].shape == (3, 16)
    assert output.fused["protocol"].shape == (2, 16)
    assert output.fused["service"].shape == (2, 16)
    assert output.fused["port"].shape == (3, 16)
    # Relation-specific embeddings from the *last* layer are still exposed.
    assert "originates" in output.relations["flow"]


def test_full_model_gradients_reach_relation_specific_weights() -> None:
    g = _synthetic_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=16, protocol_vocab_size=5, service_vocab_size=10, num_layers=1
    )
    model.encoders = NodeFeatureEncoders(
        16, protocol_vocab_size=5, service_vocab_size=10,
        flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM,
    )

    output = model(g)
    loss = output.fused["flow"].pow(2).mean()
    loss.backward()

    originates_weight = model.layers[0].conv.rel_lins["host__originates__flow"].weight
    assert originates_weight.grad is not None
    assert not torch.all(originates_weight.grad == 0)


def test_relation_specific_hetero_gnn_from_graph_matches_metadata() -> None:
    g = _synthetic_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=5, service_vocab_size=10
    )
    node_types, edge_types = g.metadata()
    conv_edge_keys = set(model.layers[0].conv.rel_lins.keys())
    expected_keys = {"__".join(et) for et in edge_types}
    assert conv_edge_keys == expected_keys
    assert set(model.layers[0].fusion.keys()) == set(node_types)
