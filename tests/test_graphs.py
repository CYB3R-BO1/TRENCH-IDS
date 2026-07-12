from __future__ import annotations

import pandas as pd
import pytest
import torch

from trench_ids.graphs import (
    _host_features,
    _host_host_edges,
    _host_ids,
    _index_categorical,
    build_task_graph,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_dataset": ["ToN", "ToN", "ToN", "ToN"],
            "IPV4_SRC_ADDR": ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.1"],
            "IPV4_DST_ADDR": ["10.0.0.2", "10.0.0.3", "10.0.0.1", "10.0.0.2"],
            "L4_SRC_PORT": [1000, 1001, 2000, 1002],
            "L4_DST_PORT": [80, 443, 80, 80],
            "PROTOCOL": [6, 6, 17, 6],
            "L7_PROTO": [1, 2, 1, 1],
            "IN_BYTES": [100, 200, 10, 300],
            "OUT_BYTES": [50, 20, 5, 30],
            "FLOW_DURATION_MILLISECONDS": [10, 20, 5, 15],
            "canonical_label": ["Benign", "DDoS", "Benign", "DDoS"],
            "split": ["train", "train", "val", "test"],
        }
    )


def test_index_categorical_sorted_stable() -> None:
    idx, uniques = _index_categorical(pd.Series([6, 6, 17, 6]))
    assert uniques == [6, 17]
    assert idx.tolist() == [0, 0, 1, 0]


def test_host_ids_dedupes_across_src_and_dst_roles() -> None:
    frame = _sample_frame()
    src_idx, dst_idx, hosts = _host_ids(frame)

    assert hosts == [("ToN", "10.0.0.1"), ("ToN", "10.0.0.2"), ("ToN", "10.0.0.3")]
    assert src_idx.tolist() == [0, 0, 1, 0]
    assert dst_idx.tolist() == [1, 2, 0, 1]


def test_host_features_matches_hand_computed_stats() -> None:
    frame = _sample_frame()
    src_idx, dst_idx, hosts = _host_ids(frame)

    feats = _host_features(frame, src_idx, dst_idx, len(hosts))

    # host 0 = 10.0.0.1: src in rows 0,1,3 (total_flows contribution 3),
    # dst in row 2 (contribution 1) -> total_flows = 4.
    assert feats[0, 0] == pytest.approx(4.0)
    # avg_bytes_as_src for host 0: rows 0,1,3 -> (150 + 220 + 330) / 3
    assert feats[0, 1] == pytest.approx(700.0 / 3.0)
    # avg_bytes_as_dst for host 0: row 2 only -> 15
    assert feats[0, 2] == pytest.approx(15.0)
    # unique dst ports contacted as src: rows 0,1,3 -> {80, 443} -> 2
    assert feats[0, 3] == pytest.approx(2.0)

    # host 1 = 10.0.0.2: src in row 2 only, dst in rows 0,3.
    assert feats[1, 0] == pytest.approx(3.0)
    assert feats[1, 1] == pytest.approx(15.0)
    assert feats[1, 2] == pytest.approx(240.0)
    assert feats[1, 3] == pytest.approx(1.0)

    # host 2 = 10.0.0.3: dst in row 1 only, never a src.
    assert feats[2, 0] == pytest.approx(1.0)
    assert feats[2, 1] == pytest.approx(0.0)
    assert feats[2, 2] == pytest.approx(220.0)
    assert feats[2, 3] == pytest.approx(0.0)


def test_host_host_edges_aggregates_repeated_pairs() -> None:
    frame = _sample_frame()
    src_idx, dst_idx, _ = _host_ids(frame)

    edge_index, edge_attr = _host_host_edges(frame, src_idx, dst_idx)

    assert edge_index.shape == (2, 3)  # (0,1), (0,2), (1,0) -- (0,1) appears twice, collapses
    pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    assert set(pairs) == {(0, 1), (0, 2), (1, 0)}

    row = pairs.index((0, 1))
    assert edge_attr[row, 0] == pytest.approx(2.0)     # flow_count: rows 0 and 3
    assert edge_attr[row, 1] == pytest.approx(480.0)   # total_bytes: 150 + 330
    assert edge_attr[row, 2] == pytest.approx(12.5)    # mean_duration: (10 + 15) / 2


def test_build_task_graph_full_structure() -> None:
    frame = _sample_frame()
    # Global vocab deliberately ordered differently from this task's local
    # sorted order, to prove vocab_id comes from the global vocab and not a
    # fresh per-task re-index.
    vocab = {"PROTOCOL": {"17": 0, "6": 1, "1": 2}, "L7_PROTO": {"1": 0, "2": 1, "5": 2}}
    features = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]

    graph, counts = build_task_graph(frame, features, vocab)

    # Flow nodes: one per row, features in the given order.
    assert graph["flow"].x.shape == (4, 3)
    assert torch.equal(graph["flow"].x[0], torch.tensor([100.0, 50.0, 10.0]))
    assert graph["flow"].label_names == ["Benign", "DDoS"]
    assert graph["flow"].y.tolist() == [0, 1, 0, 1]
    assert graph["flow"].train_mask.tolist() == [True, True, False, False]
    assert graph["flow"].val_mask.tolist() == [False, False, True, False]
    assert graph["flow"].test_mask.tolist() == [False, False, False, True]

    # Host / Protocol / Service / Port node counts.
    assert graph["host"].num_nodes == 3
    assert graph["host"].x.shape == (3, 4)
    assert graph["protocol"].num_nodes == 2   # values {6, 17}
    assert graph["service"].num_nodes == 2    # values {1, 2}
    assert graph["port"].num_nodes == 2       # values {80, 443}

    # Global vocab_id: local protocol_values sorted = [6, 17];
    # vocab maps "6"->1, "17"->0, so vocab_id should be [1, 0].
    assert graph["protocol"].vocab_id.tolist() == [1, 0]
    # local service_values sorted = [1, 2]; vocab maps "1"->0, "2"->1.
    assert graph["service"].vocab_id.tolist() == [0, 1]

    # Edge relations: 5 relations track 1:1 with flow rows, host-host collapses.
    assert graph["host", "sends", "flow"].edge_index.shape == (2, 4)
    assert graph["flow", "received_by", "host"].edge_index.shape == (2, 4)
    assert graph["flow", "uses_port", "port"].edge_index.shape == (2, 4)
    assert graph["flow", "uses_protocol", "protocol"].edge_index.shape == (2, 4)
    assert graph["flow", "uses_service", "service"].edge_index.shape == (2, 4)
    assert graph["host", "talks_to", "host"].edge_index.shape == (2, 3)
    assert graph["host", "talks_to", "host"].edge_attr.shape == (3, 3)

    # Counts report mirrors the graph.
    assert counts["node_counts"] == {"flow": 4, "host": 3, "protocol": 2, "service": 2, "port": 2}
    assert counts["edge_counts"]["host_talks_to_host"] == 3
    assert counts["flow_class_counts"] == {"Benign": 2, "DDoS": 2}
    assert counts["host_degree"]["max"] == pytest.approx(4.0)
    assert counts["host_degree"]["min"] == pytest.approx(1.0)
