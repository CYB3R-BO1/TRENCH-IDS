"""Step 2 — heterogeneous graph construction.

Builds one self-contained PyTorch Geometric ``HeteroData`` graph per task
from that task's processed Parquet (``data/processed/task_{t}.parquet``),
per the node/relation design in docs/dataset-plan.md §3:

  Node types: Host, Flow, Protocol, Port, Service.
  Relations : host->flow, flow->host, flow->port, flow->protocol,
              flow->service, host->host (aggregated within the task only).

Flow node identity is one row of the task Parquet (no cross-row merging).
Host identity is (source_dataset, IP), aggregated within the task only —
it never persists across tasks. Protocol/L7_PROTO node identity uses the
global vocabulary from ``trench_ids.vocab`` so the same value maps to the
same category across every task; Port stays task-local (raw destination
port number).

Run:  trench-graphs --config configs/graph.yaml
  or: python -m trench_ids.graphs --config configs/graph.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def _index_categorical(values: pd.Series) -> tuple[np.ndarray, list]:
    """Task-local index for a categorical column: sorted unique values -> 0..N-1."""
    uniques = sorted(values.unique().tolist())
    lookup = {v: i for i, v in enumerate(uniques)}
    idx = values.map(lookup).to_numpy()
    return idx, uniques


def _host_ids(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """(source_dataset, IP) identity for every flow's src and dst host role."""
    src_keys = list(zip(frame["source_dataset"], frame["IPV4_SRC_ADDR"], strict=True))
    dst_keys = list(zip(frame["source_dataset"], frame["IPV4_DST_ADDR"], strict=True))
    uniques = sorted(set(src_keys) | set(dst_keys))
    lookup = {k: i for i, k in enumerate(uniques)}
    src_idx = np.array([lookup[k] for k in src_keys], dtype=np.int64)
    dst_idx = np.array([lookup[k] for k in dst_keys], dtype=np.int64)
    return src_idx, dst_idx, uniques


def _host_features(
    frame: pd.DataFrame, src_idx: np.ndarray, dst_idx: np.ndarray, num_hosts: int
) -> np.ndarray:
    """[num_hosts, 4]: total_flows, avg_bytes_as_src, avg_bytes_as_dst, unique_ports_contacted."""
    bytes_total = (frame["IN_BYTES"] + frame["OUT_BYTES"]).to_numpy(dtype=np.float64)
    dst_port = frame["L4_DST_PORT"].to_numpy()

    total_flows = np.zeros(num_hosts, dtype=np.float64)
    np.add.at(total_flows, src_idx, 1)
    np.add.at(total_flows, dst_idx, 1)

    src_bytes_sum = np.zeros(num_hosts, dtype=np.float64)
    src_bytes_cnt = np.zeros(num_hosts, dtype=np.float64)
    np.add.at(src_bytes_sum, src_idx, bytes_total)
    np.add.at(src_bytes_cnt, src_idx, 1)

    dst_bytes_sum = np.zeros(num_hosts, dtype=np.float64)
    dst_bytes_cnt = np.zeros(num_hosts, dtype=np.float64)
    np.add.at(dst_bytes_sum, dst_idx, bytes_total)
    np.add.at(dst_bytes_cnt, dst_idx, 1)

    avg_bytes_as_src = np.divide(
        src_bytes_sum, src_bytes_cnt, out=np.zeros(num_hosts), where=src_bytes_cnt > 0
    )
    avg_bytes_as_dst = np.divide(
        dst_bytes_sum, dst_bytes_cnt, out=np.zeros(num_hosts), where=dst_bytes_cnt > 0
    )

    unique_ports: list[set] = [set() for _ in range(num_hosts)]
    for h, p in zip(src_idx, dst_port, strict=True):
        unique_ports[h].add(p)
    ports_contacted = np.array([len(s) for s in unique_ports], dtype=np.float64)

    return np.stack([total_flows, avg_bytes_as_src, avg_bytes_as_dst, ports_contacted], axis=1)


def _host_host_edges(
    frame: pd.DataFrame, src_idx: np.ndarray, dst_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate repeated (src_host, dst_host) pairs into one edge per pair."""
    bytes_total = (frame["IN_BYTES"] + frame["OUT_BYTES"]).to_numpy(dtype=np.float64)
    duration = frame["FLOW_DURATION_MILLISECONDS"].to_numpy(dtype=np.float64)
    pair_df = pd.DataFrame(
        {"src": src_idx, "dst": dst_idx, "bytes": bytes_total, "duration": duration}
    )
    agg = pair_df.groupby(["src", "dst"], sort=False).agg(
        flow_count=("bytes", "size"),
        total_bytes=("bytes", "sum"),
        mean_duration=("duration", "mean"),
    )
    edge_index = agg.index.to_frame(index=False)[["src", "dst"]].to_numpy(dtype=np.int64).T
    edge_attr = agg[["flow_count", "total_bytes", "mean_duration"]].to_numpy(dtype=np.float64)
    return edge_index, edge_attr


def build_task_graph(
    frame: pd.DataFrame, features: list[str], vocab: dict[str, dict[str, int]]
) -> tuple[HeteroData, dict[str, Any]]:
    """Build one task's HeteroData graph; also return a JSON-able counts report."""
    frame = frame.reset_index(drop=True)
    num_flow = len(frame)

    src_idx, dst_idx, hosts = _host_ids(frame)
    num_hosts = len(hosts)
    host_x = _host_features(frame, src_idx, dst_idx, num_hosts)

    protocol_idx, protocol_values = _index_categorical(frame["PROTOCOL"])
    service_idx, service_values = _index_categorical(frame["L7_PROTO"])
    port_idx, port_values = _index_categorical(frame["L4_DST_PORT"])

    protocol_vocab_ids = [vocab["PROTOCOL"][str(v)] for v in protocol_values]
    service_vocab_ids = [vocab["L7_PROTO"][str(v)] for v in service_values]

    label_names = sorted(frame["canonical_label"].unique().tolist())
    label_lookup = {name: i for i, name in enumerate(label_names)}
    y = frame["canonical_label"].map(label_lookup).to_numpy(dtype=np.int64)

    hh_edge_index, hh_edge_attr = _host_host_edges(frame, src_idx, dst_idx)

    graph = HeteroData()

    graph["flow"].x = torch.tensor(frame[features].to_numpy(dtype=np.float32))
    graph["flow"].y = torch.tensor(y)
    graph["flow"].label_names = label_names
    for split in ("train", "val", "test"):
        graph["flow"][f"{split}_mask"] = torch.tensor((frame["split"] == split).to_numpy())

    graph["host"].num_nodes = num_hosts
    graph["host"].x = torch.tensor(host_x, dtype=torch.float32)
    graph["host"].identity = hosts

    graph["protocol"].num_nodes = len(protocol_values)
    graph["protocol"].value = torch.tensor(protocol_values, dtype=torch.int64)
    graph["protocol"].vocab_id = torch.tensor(protocol_vocab_ids, dtype=torch.int64)

    graph["service"].num_nodes = len(service_values)
    graph["service"].value = torch.tensor(service_values, dtype=torch.int64)
    graph["service"].vocab_id = torch.tensor(service_vocab_ids, dtype=torch.int64)

    graph["port"].num_nodes = len(port_values)
    graph["port"].port_number = torch.tensor(port_values, dtype=torch.int64)

    graph["host", "sends", "flow"].edge_index = torch.tensor(
        np.stack([src_idx, np.arange(num_flow)]), dtype=torch.int64
    )
    graph["flow", "received_by", "host"].edge_index = torch.tensor(
        np.stack([np.arange(num_flow), dst_idx]), dtype=torch.int64
    )
    graph["flow", "uses_port", "port"].edge_index = torch.tensor(
        np.stack([np.arange(num_flow), port_idx]), dtype=torch.int64
    )
    graph["flow", "uses_protocol", "protocol"].edge_index = torch.tensor(
        np.stack([np.arange(num_flow), protocol_idx]), dtype=torch.int64
    )
    graph["flow", "uses_service", "service"].edge_index = torch.tensor(
        np.stack([np.arange(num_flow), service_idx]), dtype=torch.int64
    )
    graph["host", "talks_to", "host"].edge_index = torch.tensor(hh_edge_index, dtype=torch.int64)
    graph["host", "talks_to", "host"].edge_attr = torch.tensor(hh_edge_attr, dtype=torch.float32)

    host_degree = host_x[:, 0]
    counts: dict[str, Any] = {
        "node_counts": {
            "flow": num_flow,
            "host": num_hosts,
            "protocol": len(protocol_values),
            "service": len(service_values),
            "port": len(port_values),
        },
        "edge_counts": {
            "host_sends_flow": num_flow,
            "flow_received_by_host": num_flow,
            "flow_uses_port": num_flow,
            "flow_uses_protocol": num_flow,
            "flow_uses_service": num_flow,
            "host_talks_to_host": int(hh_edge_index.shape[1]),
        },
        "flow_class_counts": {
            k: int(v) for k, v in frame["canonical_label"].value_counts().items()
        },
        "host_degree": {
            "min": float(host_degree.min()) if num_hosts else 0.0,
            "mean": float(host_degree.mean()) if num_hosts else 0.0,
            "median": float(np.median(host_degree)) if num_hosts else 0.0,
            "max": float(host_degree.max()) if num_hosts else 0.0,
        },
    }
    return graph, counts
