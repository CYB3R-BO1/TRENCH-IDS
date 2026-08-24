"""Non-graph baseline encoder -- the ablation that removes message passing.

Answers the question no experiment in this project has asked yet: **does the
heterogeneous graph contribute anything over the per-flow features?** Every
result so far compares GNN variants against other GNN variants (EWC vs.
replay vs. joint training), so "a relation-specific heterogeneous GNN is a
good IDS representation" has never been tested against a model that has no
graph at all.

What this keeps, identical to ``rhgnn.NodeFeatureEncoders``:

  * the 37 Flow features, through the same ``LayerNorm -> Linear`` pathway,
  * the Port embedding, through the *same* ``port_embedding_index`` /
    ``port_embedding_size`` functions (so the well-known-port fix of
    2026-07-24 applies identically here -- the comparison is not confounded
    by the port scheme),
  * the Protocol and Service embeddings, indexed by the same global vocab
    (``trench_ids.vocab``).

What this removes -- and *only* this, which is what makes it a clean
ablation rather than a different model:

  * all message passing (``RelationSpecificConv``): no relation-specific
    neighbour aggregation, no 11 relations,
  * the semantic-attention fusion (``SemanticAttention``),
  * optionally the Host node type -- see ``use_host_features`` below.

**The three-arm decomposition.** Host aggregates and message passing are two
distinct things the graph buys, and a single flat-vs-GNN comparison confounds
them: if the GNN wins, there is no way to say which one was responsible. So
``use_host_features`` splits the baseline in two, giving a ladder in which
exactly one capability is added per rung:

  ===============  =====================  =================  ===============
  Arm              Host aggregates        Message passing    Class
  ===============  =====================  =================  ===============
  Flat             no                     no                 this, flag off
  Flat+Host        yes (as flow columns)  no                 this, flag on
  GNN              yes (as graph nodes)   yes                RelationSpecific-
                                                             HeteroGNN
  ===============  =====================  =================  ===============

With all three, ``Flat+Host - Flat`` isolates the value of host statistics
and ``GNN - Flat+Host`` isolates the value of propagation, in either
direction of outcome.

When enabled, each flow gets its source and destination host's 4 aggregate
features (``total_flows``, ``avg_bytes_as_src``, ``avg_bytes_as_dst``,
``unique_ports_contacted``) through the *same* ``LayerNorm(4) -> Linear(4,
hidden)`` encoder the GNN applies to Host nodes, with weights shared between
the two roles exactly as the GNN shares them. The difference from the GNN is
then purely that these values are read off the flow's own two endpoints
instead of being propagated and attention-fused.

Those host aggregates are computed per mini-graph, and since the 2026-08-17
rebuild ``graphs.py`` chunks in capture order rather than from a global
shuffle, they describe a real contiguous capture interval instead of a random
task-wide subsample. Both this arm and the GNN read the identical
``graph["host"].x``, so the comparison stays apples-to-apples -- but the
aggregates are now genuinely informative rather than noise, which is worth
knowing when reading the size of the Flat -> Flat+Host gap.

Because both arms read the same per-mini-graph tensor, and a mini-graph
belongs entirely to one split, there is no train/test leakage route through
these features for either arm.

Port/Protocol/Service reach a Flow through edges even though they are 1:1,
and Host reaches it through the ``originated_by``/``terminates_at`` edges, so
all of them are gathered back per flow via ``_per_flow_values`` rather than
read positionally -- see that function for why the edge lookup is done
defensively.

The four channels are **concatenated** (not summed) before the MLP, mirroring
``RelationSpecificConv``'s own ``CONCAT(self, aggregated)`` combine step, so
the baseline is not handicapped relative to the GNN by an information-losing
merge.

``forward`` returns a ``RelationSpecificOutput`` -- the same type
``RelationSpecificHeteroGNN`` returns -- with ``fused["flow"]`` populated and
``relations["flow"]`` deliberately empty (there are no relations). That makes
this a drop-in encoder for every existing consumer
(``trench_ids.cl.train.evaluate_accuracy``, ``trench_ids.cl.inference.predict``,
and therefore ``trench_ids.cl.evaluate``), so the flat baseline is scored by
byte-identical evaluation code rather than a parallel implementation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import HeteroData

from trench_ids.model.rhgnn import (
    FLOW_FEATURE_DIM,
    HOST_FEATURE_DIM,
    RelationSpecificOutput,
    port_embedding_index,
    port_embedding_size,
)

# MLP widths chosen so each arm parameter-matches RelationSpecificHeteroGNN
# at the real configuration (hidden_dim=64, port_tail_buckets=32,
# protocol_vocab_size=5, service_vocab_size=216 -- data/graphs/vocab.json),
# where the GNN encoder has 263,378 parameters.
#
# ⚠️ SUPERSEDED as a fairness control (2026-08-17 audit): 263,378 is the
# GNN's *nominal* count, but ~40.9% of those parameters (108,288) receive
# zero/None gradient from the classification loss -- the 6 non-Flow relations
# plus 4 unused fusion modules, computed every forward pass but never
# consumed since training reads only output.fused["flow"]. Matching against
# the nominal count compared ~157K trainable params against ~263K (1.7x).
# The corrected control is flathost_small_replay (mlp_hidden=123 -> 156,577
# vs the GNN's 156,562 *reachable* parameters); these defaults are kept only
# for backward compatibility with the recorded runs.
#
# Cost shared by both arms (LayerNorm 74 + Linear(37->64) 2,432 + port
# 1,056x64 + protocol 5x64 + service 216x64) = 84,234.
#
#   Flat      : 4x64=256-wide MLP input, budget 179,144. The 3-layer MLP
#               costs h^2 + 322h + 64; h=292 spends 179,352 -> 263,586
#               total, 1.0008x the GNN.
#   Flat+Host : adds LayerNorm(4) 8 + Linear(4->64) 320 and widens the MLP
#               input to 6x64=384 (flow, port, protocol, service, src host,
#               dst host), budget 178,816. The MLP costs h^2 + 450h + 64;
#               h=254 spends 178,880 -> 263,442 total, 1.0002x the GNN.
#
# Parameter matching is a fairness control, not a load-bearing constant: it
# exists so a GNN win cannot be attributed to extra capacity. Both are
# asserted in tests/test_flat_model.py, and every run's summary.json records
# the realised counts, so drift (e.g. a regenerated vocabulary) shows up in
# the results rather than silently biasing them.
DEFAULT_MLP_HIDDEN = 292
DEFAULT_MLP_HIDDEN_WITH_HOST = 254

_FLOW_TO_PORT = ("flow", "targets_port", "port")
_FLOW_TO_PROTOCOL = ("flow", "uses_protocol", "protocol")
_FLOW_TO_SERVICE = ("flow", "uses_service", "service")
_FLOW_TO_SRC_HOST = ("flow", "originated_by", "host")
_FLOW_TO_DST_HOST = ("flow", "terminates_at", "host")


def _per_flow_values(
    graph: HeteroData,
    edge_type: tuple[str, str, str],
    source_values: torch.Tensor,
    num_flow: int,
) -> torch.Tensor:
    """Gather a 1:1 neighbour node's value back onto each Flow row.

    Handles both scalar sources (``port_number``, ``vocab_id`` -> one value
    per flow) and vector sources (``host.x`` -> one feature row per flow),
    since ``source_values.shape[1:]`` is carried through unchanged.

    ``graphs.build_task_graph`` builds every edge type used here with
    ``np.arange(num_flow)`` on the Flow side, so row 0 of ``edge_index`` is
    already sorted and a positional read would work on an unbatched graph.
    This scatters through ``edge_index`` anyway: it stays correct under
    ``DataLoader`` batching (which concatenates and offsets edge indices,
    so a positional read of ``host.x`` would silently pick the wrong
    graph's hosts), under any future edge reordering, and under the
    unseen-traffic graphs built by ``trench_ids.unseen_graphs``.

    Flows with no such edge keep a zero row. That cannot happen on graphs
    this project builds (every flow has exactly one protocol, service,
    destination port, source host and destination host by construction) --
    it is a defined fallback rather than an uninitialised read, so a
    malformed graph degrades to a known value instead of producing garbage.
    """
    out = torch.zeros(
        (num_flow, *source_values.shape[1:]),
        dtype=source_values.dtype,
        device=source_values.device,
    )
    edge_index = graph[edge_type].edge_index
    if edge_index.numel():
        out[edge_index[0]] = source_values[edge_index[1]]
    return out


class FlatFlowEncoder(nn.Module):
    """Per-flow MLP over the same inputs the GNN gets, minus the graph.

    Same constructor signature shape as ``RelationSpecificHeteroGNN`` for the
    arguments they share, so ``trench_ids.cl.inference.load_checkpoint`` can
    dispatch between them on the checkpoint's ``model_type`` without special
    -casing anything else.
    """

    def __init__(
        self,
        hidden_dim: int,
        protocol_vocab_size: int,
        service_vocab_size: int,
        flow_feature_dim: int = FLOW_FEATURE_DIM,
        host_feature_dim: int = HOST_FEATURE_DIM,
        port_tail_buckets: int = 32,
        use_host_features: bool = False,
        mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.use_host_features = use_host_features
        # Resolved rather than defaulted in the signature: the two arms have
        # different parameter budgets, so a single default would silently
        # unbalance whichever arm did not set it explicitly.
        self.mlp_hidden = mlp_hidden if mlp_hidden is not None else (
            DEFAULT_MLP_HIDDEN_WITH_HOST if use_host_features else DEFAULT_MLP_HIDDEN
        )

        self.flow_norm = nn.LayerNorm(flow_feature_dim)
        self.flow = nn.Linear(flow_feature_dim, hidden_dim)
        self.protocol = nn.Embedding(protocol_vocab_size, hidden_dim)
        self.service = nn.Embedding(service_vocab_size, hidden_dim)
        self.port = nn.Embedding(port_embedding_size(port_tail_buckets), hidden_dim)
        # Scaled init, matching NodeFeatureEncoders: the default N(0,1)
        # embedding rows would numerically dominate the LayerNorm -> Linear
        # flow/host channels after concatenation.
        for embedding in (self.protocol, self.service, self.port):
            nn.init.normal_(embedding.weight, std=0.02)
        self.port_tail_buckets = port_tail_buckets

        num_channels = 4
        if use_host_features:
            # One shared encoder for both endpoint roles, matching how the
            # GNN's NodeFeatureEncoders encodes every Host node with a single
            # LayerNorm+Linear regardless of the role it plays in a flow.
            self.host_norm = nn.LayerNorm(host_feature_dim)
            self.host = nn.Linear(host_feature_dim, hidden_dim)
            num_channels += 2  # source host, destination host

        self.mlp = nn.Sequential(
            nn.Linear(num_channels * hidden_dim, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, hidden_dim),
        )

    def forward(self, graph: HeteroData) -> RelationSpecificOutput:
        x = graph["flow"].x
        num_flow = x.size(0)

        port_number = _per_flow_values(graph, _FLOW_TO_PORT, graph["port"].port_number, num_flow)
        protocol_id = _per_flow_values(
            graph, _FLOW_TO_PROTOCOL, graph["protocol"].vocab_id, num_flow
        )
        service_id = _per_flow_values(graph, _FLOW_TO_SERVICE, graph["service"].vocab_id, num_flow)

        channels = [
            self.flow(self.flow_norm(x)),
            self.port(port_embedding_index(port_number, self.port_tail_buckets)),
            self.protocol(protocol_id),
            self.service(service_id),
        ]
        if self.use_host_features:
            host_x = graph["host"].x
            for edge_type in (_FLOW_TO_SRC_HOST, _FLOW_TO_DST_HOST):
                endpoint = _per_flow_values(graph, edge_type, host_x, num_flow)
                channels.append(self.host(self.host_norm(endpoint)))

        return RelationSpecificOutput(
            relations={"flow": {}},
            fused={"flow": self.mlp(torch.cat(channels, dim=-1))},
            attention={"flow": {}},
        )
