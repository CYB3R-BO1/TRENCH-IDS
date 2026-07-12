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

import numpy as np
import pandas as pd


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
