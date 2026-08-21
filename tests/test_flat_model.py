from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.model.flat import (
    _FLOW_TO_DST_HOST,
    _FLOW_TO_SRC_HOST,
    DEFAULT_MLP_HIDDEN,
    DEFAULT_MLP_HIDDEN_WITH_HOST,
    FlatFlowEncoder,
    _per_flow_values,
)
from trench_ids.model.rhgnn import (
    FLOW_FEATURE_DIM,
    HOST_FEATURE_DIM,
    RelationSpecificHeteroGNN,
    port_embedding_index,
)

# The real vocabulary sizes from data/graphs/vocab.json. Hardcoded rather
# than read from disk so the parameter-match assertion runs without the
# generated dataset present.
REAL_PROTOCOL_VOCAB = 5
REAL_SERVICE_VOCAB = 216


def _tiny_graph(shuffle_edges: bool = False) -> HeteroData:
    """Mirrors the real Step 2 schema at real feature dims. When
    ``shuffle_edges`` is set, the flow->port/protocol/service edge lists are
    permuted so row 0 of edge_index is no longer sorted -- graphs.py always
    emits them sorted, so this checks _per_flow_values genuinely gathers
    through the edge index instead of relying on that ordering."""
    g = HeteroData()
    g["flow"].x = torch.randn(4, FLOW_FEATURE_DIM)
    g["flow"].y = torch.tensor([0, 1, 0, 1])
    g["host"].num_nodes = 2
    g["host"].x = torch.randn(2, HOST_FEATURE_DIM)
    g["protocol"].num_nodes = 2
    g["protocol"].vocab_id = torch.tensor([0, 1])
    g["service"].num_nodes = 2
    g["service"].vocab_id = torch.tensor([3, 7])
    g["port"].num_nodes = 3
    g["port"].port_number = torch.tensor([22, 80, 50000])

    src_host = torch.tensor([0, 1, 0, 1])
    dst_host = torch.tensor([1, 0, 1, 0])
    flow = torch.arange(4)
    protocol_idx = torch.tensor([0, 1, 1, 0])
    service_idx = torch.tensor([1, 0, 1, 0])
    port_idx = torch.tensor([0, 1, 2, 1])

    order = torch.tensor([2, 0, 3, 1]) if shuffle_edges else flow
    g["flow", "targets_port", "port"].edge_index = torch.stack([flow[order], port_idx[order]])
    g["flow", "uses_protocol", "protocol"].edge_index = torch.stack(
        [flow[order], protocol_idx[order]]
    )
    g["flow", "uses_service", "service"].edge_index = torch.stack(
        [flow[order], service_idx[order]]
    )

    g["host", "originates", "flow"].edge_index = torch.stack([src_host, flow])
    g["flow", "terminates_at", "host"].edge_index = torch.stack([flow, dst_host])
    g["host", "communicates_with", "host"].edge_index = torch.tensor([[0], [1]])
    g["flow", "originated_by", "host"].edge_index = torch.stack([flow, src_host])
    g["host", "terminated_by", "flow"].edge_index = torch.stack([dst_host, flow])
    g["port", "targeted_by", "flow"].edge_index = torch.stack([port_idx, flow])
    g["protocol", "protocol_of", "flow"].edge_index = torch.stack([protocol_idx, flow])
    g["service", "service_of", "flow"].edge_index = torch.stack([service_idx, flow])
    return g


def _encoder(**overrides) -> FlatFlowEncoder:
    kwargs = {
        "hidden_dim": 8,
        "protocol_vocab_size": 2,
        "service_vocab_size": 8,
        "mlp_hidden": 16,
    }
    kwargs.update(overrides)
    return FlatFlowEncoder(**kwargs)


BOTH_ARMS = pytest.mark.parametrize("use_host_features", [False, True], ids=["flat", "flat+host"])


@BOTH_ARMS
def test_forward_returns_relation_specific_output_shape(use_host_features: bool) -> None:
    g = _tiny_graph()
    out = _encoder(use_host_features=use_host_features)(g)

    assert set(out.fused) == {"flow"}
    assert out.fused["flow"].shape == (4, 8)
    # No relations exist in a model with no message passing -- this must be
    # empty rather than absent, so consumers that iterate it (e.g. the
    # memory-bank accumulator) degrade to a no-op instead of a KeyError.
    assert out.relations == {"flow": {}}
    assert out.attention == {"flow": {}}


@BOTH_ARMS
def test_output_does_not_depend_on_batch_composition(use_host_features: bool) -> None:
    """With no message passing, a flow's embedding must not change with
    which other graphs share its batch. The GNN deliberately violates this
    (SemanticAttention averages its scores over every node in the batch);
    both baseline arms must not -- including Flat+Host, whose host features
    are precomputed per mini-graph and gathered through offset edge
    indices."""
    g = _tiny_graph()
    encoder = _encoder(use_host_features=use_host_features).eval()
    with torch.no_grad():
        full = encoder(g).fused["flow"]
        batched = encoder(next(iter(DataLoader([g, g], batch_size=2)))).fused["flow"]

    torch.testing.assert_close(batched[:4], full)
    torch.testing.assert_close(batched[4:], full)


def test_categorical_lookup_follows_edge_index_not_row_order() -> None:
    """Permuting the edge lists must not change any flow's embedding --
    otherwise the encoder is silently reading port/protocol/service
    positionally and would mis-assign them under a reordering."""
    torch.manual_seed(0)
    sorted_graph = _tiny_graph(shuffle_edges=False)
    torch.manual_seed(0)
    shuffled_graph = _tiny_graph(shuffle_edges=True)

    encoder = _encoder().eval()
    with torch.no_grad():
        torch.testing.assert_close(
            encoder(sorted_graph).fused["flow"], encoder(shuffled_graph).fused["flow"]
        )


@BOTH_ARMS
def test_host_features_are_consumed_only_by_the_host_arm(use_host_features: bool) -> None:
    """The single variable separating the two baseline rungs. Flat must be
    provably blind to host aggregates; Flat+Host must provably use them --
    otherwise the ladder does not decompose anything."""
    g = _tiny_graph()
    encoder = _encoder(use_host_features=use_host_features).eval()
    with torch.no_grad():
        before = encoder(g).fused["flow"]
        g["host"].x = torch.randn(2, HOST_FEATURE_DIM) * 1000
        after = encoder(g).fused["flow"]

    if use_host_features:
        assert not torch.allclose(before, after)
    else:
        torch.testing.assert_close(before, after)


def test_host_arm_reads_source_and_destination_endpoints_in_the_right_order() -> None:
    """A swapped src/dst gather would still train and still look plausible,
    so pin the mapping down directly against the fixture's known topology
    (src_host=[0,1,0,1], dst_host=[1,0,1,0])."""
    g = _tiny_graph()
    g["host"].x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    src = _per_flow_values(g, _FLOW_TO_SRC_HOST, g["host"].x, 4)
    dst = _per_flow_values(g, _FLOW_TO_DST_HOST, g["host"].x, 4)

    torch.testing.assert_close(src, g["host"].x[torch.tensor([0, 1, 0, 1])])
    torch.testing.assert_close(dst, g["host"].x[torch.tensor([1, 0, 1, 0])])


def test_host_arm_shares_one_encoder_across_both_endpoint_roles() -> None:
    """Matches the GNN, which encodes every Host node with a single
    LayerNorm+Linear regardless of the role it plays in a flow -- separate
    per-role weights would be a capacity difference, not just a wiring
    one."""
    encoder = _encoder(use_host_features=True)
    host_params = {name for name, _ in encoder.named_parameters() if name.startswith("host")}

    assert host_params == {"host_norm.weight", "host_norm.bias", "host.weight", "host.bias"}


def test_flat_arm_has_no_host_parameters_at_all() -> None:
    encoder = _encoder(use_host_features=False)

    assert not [name for name, _ in encoder.named_parameters() if name.startswith("host")]


def test_port_embedding_table_matches_the_gnns() -> None:
    """Both arms must resolve ports through the same well-known/tail scheme,
    or the comparison is confounded by the 2026-07-24 port fix rather than
    isolating the graph."""
    flat = _encoder(port_tail_buckets=32)
    gnn = RelationSpecificHeteroGNN.from_graph(
        _tiny_graph(), hidden_dim=8, protocol_vocab_size=2, service_vocab_size=8,
        num_layers=1, port_tail_buckets=32,
    )
    assert flat.port.weight.shape == gnn.encoders.port.weight.shape
    # The fix's actual content: adjacent well-known ports are distinct rows.
    assert port_embedding_index(torch.tensor([22]), 32).item() != (
        port_embedding_index(torch.tensor([23]), 32).item()
    )


def test_distinct_ports_produce_distinct_embeddings() -> None:
    """Ports reach the model only through this embedding (L4_DST_PORT is not
    among the 37 Flow features), so a flow's port must actually move its
    representation."""
    g = _tiny_graph()
    encoder = _encoder().eval()
    with torch.no_grad():
        before = encoder(g).fused["flow"]
        # flow 0 and flow 3 point at port rows 0 (22) and 1 (80); flow 1 also
        # points at row 1, so remapping row 0 must move flow 0 alone.
        g["port"].port_number = torch.tensor([443, 80, 50000])
        after = encoder(g).fused["flow"]

    assert not torch.allclose(before[0], after[0])
    torch.testing.assert_close(before[1], after[1])


@BOTH_ARMS
def test_gradients_reach_every_parameter(use_host_features: bool) -> None:
    g = _tiny_graph()
    encoder = _encoder(use_host_features=use_host_features)
    classifier = torch.nn.Linear(8, 2)
    logits = classifier(encoder(g).fused["flow"])
    torch.nn.functional.cross_entropy(logits, g["flow"].y).backward()

    dead = [
        name
        for name, p in encoder.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    # Embedding rows for unused categories legitimately get zero gradient,
    # but every parameter tensor must at least receive one -- unlike the
    # GNN, where 6 of 11 relation groups are unreachable at num_layers=1.
    assert dead == []


@BOTH_ARMS
def test_every_arm_is_parameter_matched_to_the_reference_gnn(use_host_features: bool) -> None:
    """The fairness control that lets a win be attributed to the graph
    rather than to capacity. Both rungs must match, or a three-way
    comparison silently becomes a capacity sweep. Asserted at the real
    configuration, with mlp_hidden left to resolve per arm."""
    flat = FlatFlowEncoder(
        hidden_dim=64,
        protocol_vocab_size=REAL_PROTOCOL_VOCAB,
        service_vocab_size=REAL_SERVICE_VOCAB,
        port_tail_buckets=32,
        use_host_features=use_host_features,
    )
    gnn = RelationSpecificHeteroGNN.from_graph(
        _tiny_graph(),
        hidden_dim=64,
        protocol_vocab_size=REAL_PROTOCOL_VOCAB,
        service_vocab_size=REAL_SERVICE_VOCAB,
        num_layers=1,
        attn_dim=128,
        port_tail_buckets=32,
    )
    flat_params = sum(p.numel() for p in flat.parameters())
    gnn_params = sum(p.numel() for p in gnn.parameters())

    assert abs(flat_params - gnn_params) / gnn_params < 0.01, (
        f"arm use_host_features={use_host_features}: flat={flat_params:,} vs "
        f"gnn={gnn_params:,} -- retune flat.DEFAULT_MLP_HIDDEN"
        f"{'_WITH_HOST' if use_host_features else ''} to restore the match"
    )


def test_each_arm_resolves_its_own_default_mlp_width() -> None:
    """A single shared default would leave one arm unbalanced; the widths
    differ because Flat+Host spends parameters on two extra input channels
    plus the host encoder."""
    assert _encoder(mlp_hidden=None).mlp_hidden == DEFAULT_MLP_HIDDEN
    assert (
        _encoder(mlp_hidden=None, use_host_features=True).mlp_hidden
        == DEFAULT_MLP_HIDDEN_WITH_HOST
    )
    assert DEFAULT_MLP_HIDDEN != DEFAULT_MLP_HIDDEN_WITH_HOST
    # An explicit value still wins, so a deliberate mismatch stays possible.
    assert _encoder(mlp_hidden=7, use_host_features=True).mlp_hidden == 7
