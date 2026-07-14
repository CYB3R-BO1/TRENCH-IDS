from __future__ import annotations

import pandas as pd
import pytest
import torch

from trench_ids.graphs import (
    _chunk_frame,
    _host_features,
    _host_host_edges,
    _host_ids,
    _index_categorical,
    build_split_graphs,
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
            "flow_id": ["ToN-0", "ToN-1", "ToN-2", "ToN-3"],
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
    assert graph["flow"].flow_id == ["ToN-0", "ToN-1", "ToN-2", "ToN-3"]
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
    assert graph["host", "originates", "flow"].edge_index.shape == (2, 4)
    assert graph["flow", "terminates_at", "host"].edge_index.shape == (2, 4)
    assert graph["flow", "targets_port", "port"].edge_index.shape == (2, 4)
    assert graph["flow", "uses_protocol", "protocol"].edge_index.shape == (2, 4)
    assert graph["flow", "uses_service", "service"].edge_index.shape == (2, 4)
    assert graph["host", "communicates_with", "host"].edge_index.shape == (2, 3)
    assert graph["host", "communicates_with", "host"].edge_attr.shape == (3, 3)

    # Counts report mirrors the graph.
    assert counts["node_counts"] == {"flow": 4, "host": 3, "protocol": 2, "service": 2, "port": 2}
    assert counts["edge_counts"]["host_communicates_with_host"] == 3
    assert counts["flow_class_counts"] == {"Benign": 2, "DDoS": 2}
    assert counts["host_degree"]["max"] == pytest.approx(4.0)
    assert counts["host_degree"]["min"] == pytest.approx(1.0)


def test_chunk_frame_splits_into_consecutive_blocks() -> None:
    frame = _sample_frame()  # 4 rows

    chunks = _chunk_frame(frame, size=3)

    assert len(chunks) == 2
    assert len(chunks[0]) == 3
    assert len(chunks[1]) == 1  # remainder chunk kept, not dropped or padded
    # Row order preserved, no shuffling.
    assert chunks[0]["IPV4_SRC_ADDR"].tolist() == ["10.0.0.1", "10.0.0.1", "10.0.0.2"]


def test_chunk_frame_exact_multiple_has_no_remainder_chunk() -> None:
    frame = _sample_frame()  # 4 rows

    chunks = _chunk_frame(frame, size=2)

    assert len(chunks) == 2
    assert all(len(c) == 2 for c in chunks)


def test_build_split_graphs_produces_one_mini_graph_per_chunk() -> None:
    frame = _sample_frame()
    vocab = {"PROTOCOL": {"17": 0, "6": 1, "1": 2}, "L7_PROTO": {"1": 0, "2": 1, "5": 2}}
    features = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]

    graphs, split_report = build_split_graphs(
        frame, features, vocab, graph_size=3, seed=42, benign_ratio=1.0
    )

    assert len(graphs) == 2
    assert graphs[0]["flow"].x.shape[0] == 3
    assert graphs[1]["flow"].x.shape[0] == 1
    # Each mini-graph is self-contained: host identity does not span chunks.
    assert graphs[0]["host"].num_nodes <= 3
    assert graphs[1]["host"].num_nodes <= 2

    assert split_report["num_graphs"] == 2
    assert split_report["flows_per_graph"] == {"min": 1, "max": 3, "mean": pytest.approx(2.0)}
    # Total flows across chunks' class counts matches the original frame.
    assert sum(split_report["class_counts"].values()) == len(frame)


def _class_ordered_frame(n_per_class: int = 20) -> pd.DataFrame:
    """Rows grouped by class, like Step 1's real output (per-class sampling
    then concatenation) -- reproduces the ordering that breaks unshuffled
    chunking."""
    n = n_per_class * 2
    labels = ["A"] * n_per_class + ["B"] * n_per_class
    return pd.DataFrame(
        {
            "source_dataset": ["ToN"] * n,
            "IPV4_SRC_ADDR": [f"10.0.0.{i % 250}" for i in range(n)],
            "IPV4_DST_ADDR": [f"10.0.1.{i % 250}" for i in range(n)],
            "L4_SRC_PORT": list(range(1000, 1000 + n)),
            "L4_DST_PORT": [80] * n,
            "PROTOCOL": [6] * n,
            "L7_PROTO": [1] * n,
            "IN_BYTES": [100] * n,
            "OUT_BYTES": [50] * n,
            "FLOW_DURATION_MILLISECONDS": [10] * n,
            "canonical_label": labels,
            "split": ["train"] * n,
            "flow_id": [f"ToN-{i}" for i in range(n)],
        }
    )


def test_build_split_graphs_shuffles_before_chunking() -> None:
    frame = _class_ordered_frame(n_per_class=20)  # rows 0-19 = "A", 20-39 = "B"
    vocab = {"PROTOCOL": {"6": 0}, "L7_PROTO": {"1": 0}}
    features = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]

    graphs, _ = build_split_graphs(
        frame, features, vocab, graph_size=10, seed=42, benign_ratio=1.0
    )

    # Without shuffling, every chunk would be monolithic (all "A" or all "B").
    # With shuffling, at least one chunk must mix both classes.
    assert any(len(g["flow"].label_names) > 1 for g in graphs)


def _mixed_frame(n_attack: int, n_benign: int) -> pd.DataFrame:
    n = n_attack + n_benign
    labels_ = ["DDoS"] * n_attack + ["Benign"] * n_benign
    return pd.DataFrame(
        {
            "source_dataset": ["ToN"] * n,
            "IPV4_SRC_ADDR": [f"10.0.0.{i % 200}" for i in range(n)],
            "IPV4_DST_ADDR": [f"10.0.1.{i % 200}" for i in range(n)],
            "L4_SRC_PORT": list(range(1000, 1000 + n)),
            "L4_DST_PORT": [80] * n,
            "PROTOCOL": [6] * n,
            "L7_PROTO": [1] * n,
            "IN_BYTES": [100] * n,
            "OUT_BYTES": [50] * n,
            "FLOW_DURATION_MILLISECONDS": [10] * n,
            "canonical_label": labels_,
            "split": ["train"] * n,
            "flow_id": [f"ToN-{i}" for i in range(n)],
        }
    )


def test_build_split_graphs_subsamples_benign_to_target_ratio() -> None:
    # 20 attack rows, 20 benign rows available; ratio 4:1 -> target 5 benign kept.
    frame = _mixed_frame(n_attack=20, n_benign=20)
    vocab = {"PROTOCOL": {"6": 0}, "L7_PROTO": {"1": 0}}
    features = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]

    graphs, split_report = build_split_graphs(
        frame, features, vocab, graph_size=100, seed=42, benign_ratio=4.0
    )

    assert len(graphs) == 1
    assert graphs[0]["flow"].x.shape[0] == 25  # 20 attack + 5 benign (20 / 4.0)
    assert split_report["class_counts"]["Benign"] == 5
    assert split_report["class_counts"]["DDoS"] == 20


def test_build_split_graphs_reuses_benign_with_replacement_when_pool_too_small() -> None:
    # 20 attack rows but only 2 benign rows available; ratio 1:1 needs 20
    # benign -- must reuse (sample with replacement) rather than error or
    # silently under-fill.
    frame = _mixed_frame(n_attack=20, n_benign=2)
    vocab = {"PROTOCOL": {"6": 0}, "L7_PROTO": {"1": 0}}
    features = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]

    graphs, split_report = build_split_graphs(
        frame, features, vocab, graph_size=100, seed=42, benign_ratio=1.0
    )

    assert split_report["class_counts"]["Benign"] == 20
    assert split_report["class_counts"]["DDoS"] == 20
