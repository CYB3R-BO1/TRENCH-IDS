"""Step 2 — heterogeneous graph construction.

Builds many small, self-contained PyTorch Geometric ``HeteroData`` graphs
per task from that task's processed Parquet
(``data/processed/task_{t}.parquet``), per the node/relation design in
docs/dataset-plan.md §3:

  Node types: Host, Flow, Protocol, Port, Service.
  Relations : host->flow (originates), flow->host (terminates_at),
              flow->port (targets_port), flow->protocol (uses_protocol),
              flow->service (uses_service), host->host (communicates_with,
              aggregated within the mini-graph only).

Each task's rows are split by train/val/test (the ``split`` column from
Step 1), then each split's rows are shuffled (seeded by ``seed``) and
chunked into consecutive mini-graphs of at most ``graph_size`` flows
(configs/graph.yaml). Chunking within a split keeps every mini-graph on one
side of the train/val/test boundary; shuffling before chunking keeps each
mini-graph a representative class mix, since the source Parquet is ordered
by canonical_label from Step 1's per-class sampling. Flow
node identity is one row of the chunk; Host identity is (source_dataset,
IP), aggregated within that mini-graph only — it never persists across
mini-graphs, splits, or tasks. Protocol/L7_PROTO node identity uses the
global vocabulary from ``trench_ids.vocab`` so the same value maps to the
same category everywhere; Port stays chunk-local (raw destination port
number).

Continual-learning task boundaries (T1..T4) are unaffected: training still
proceeds through all of a task's mini-graphs before moving to the next
task.

Run:  trench-graphs --config configs/graph.yaml
  or: python -m trench_ids.graphs --config configs/graph.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData

from trench_ids.labels import BENIGN
from trench_ids.vocab import build_vocab, save_vocab


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


def _chunk_frame(frame: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    """Split a frame into consecutive chunks of at most `size` rows each."""
    return [
        frame.iloc[start : start + size].reset_index(drop=True)
        for start in range(0, len(frame), size)
    ]


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

    graph["host", "originates", "flow"].edge_index = torch.tensor(
        np.stack([src_idx, np.arange(num_flow)]), dtype=torch.int64
    )
    graph["flow", "terminates_at", "host"].edge_index = torch.tensor(
        np.stack([np.arange(num_flow), dst_idx]), dtype=torch.int64
    )
    graph["flow", "targets_port", "port"].edge_index = torch.tensor(
        np.stack([np.arange(num_flow), port_idx]), dtype=torch.int64
    )
    graph["flow", "uses_protocol", "protocol"].edge_index = torch.tensor(
        np.stack([np.arange(num_flow), protocol_idx]), dtype=torch.int64
    )
    graph["flow", "uses_service", "service"].edge_index = torch.tensor(
        np.stack([np.arange(num_flow), service_idx]), dtype=torch.int64
    )
    graph["host", "communicates_with", "host"].edge_index = torch.tensor(
        hh_edge_index, dtype=torch.int64
    )
    graph["host", "communicates_with", "host"].edge_attr = torch.tensor(
        hh_edge_attr, dtype=torch.float32
    )

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
            "host_originates_flow": num_flow,
            "flow_terminates_at_host": num_flow,
            "flow_targets_port": num_flow,
            "flow_uses_protocol": num_flow,
            "flow_uses_service": num_flow,
            "host_communicates_with_host": int(hh_edge_index.shape[1]),
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


def _select_benign(
    attack_rows: pd.DataFrame, benign_rows: pd.DataFrame, benign_ratio: float, seed: int
) -> pd.DataFrame:
    """Subsample benign_rows to hit attack:benign == benign_ratio:1.

    Samples without replacement when the pool covers the target; falls back
    to sampling with replacement when the ratio demands more benign rows
    than the pool holds (an uncapped task's attack count can now far exceed
    Step 1's fixed benign_per_task pool -- see
    docs/superpowers/specs/2026-07-13-dataset-task-respec-design.md §3a).
    Returns an empty frame if there are no attack rows or no benign pool.
    """
    n_attack = len(attack_rows)
    if n_attack == 0 or benign_rows.empty:
        return benign_rows.iloc[0:0]
    target = max(1, round(n_attack / benign_ratio))
    if target <= len(benign_rows):
        return benign_rows.sample(n=target, random_state=seed).reset_index(drop=True)
    reps = -(-target // len(benign_rows))  # ceil division
    pool = pd.concat([benign_rows] * reps, ignore_index=True)
    return pool.sample(n=target, random_state=seed).reset_index(drop=True)


def build_split_graphs(
    frame: pd.DataFrame,
    features: list[str],
    vocab: dict[str, dict[str, int]],
    graph_size: int,
    seed: int,
    benign_ratio: float,
) -> tuple[list[HeteroData], dict[str, Any]]:
    """Chunk one task/split's rows into mini-graphs of at most `graph_size` flows.

    Benign rows are first subsampled to hit `benign_ratio` (attack:benign,
    see _select_benign) -- attack rows are always kept in full; only benign
    rows are ever removed (or repeated). Rows are then shuffled (seeded)
    before chunking: the source Parquet is ordered by canonical_label (Step
    1 samples per class, then concatenates), so chunking without shuffling
    first would produce mini-graphs that are almost entirely one class
    instead of a representative mix.
    """
    is_benign = frame["canonical_label"] == BENIGN
    attack_rows = frame[~is_benign].reset_index(drop=True)
    benign_rows = frame[is_benign].reset_index(drop=True)
    benign_selected = _select_benign(attack_rows, benign_rows, benign_ratio, seed)

    frame = pd.concat([attack_rows, benign_selected], ignore_index=True)
    frame = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    chunks = _chunk_frame(frame, graph_size)
    graphs: list[HeteroData] = []
    flow_counts: list[int] = []
    host_degree_maxes: list[float] = []
    class_counts: dict[str, int] = {}
    for chunk in chunks:
        graph, counts = build_task_graph(chunk, features, vocab)
        graphs.append(graph)
        flow_counts.append(counts["node_counts"]["flow"])
        host_degree_maxes.append(counts["host_degree"]["max"])
        for label, n in counts["flow_class_counts"].items():
            class_counts[label] = class_counts.get(label, 0) + n

    split_report: dict[str, Any] = {
        "num_graphs": len(graphs),
        "flows_per_graph": {
            "min": min(flow_counts) if flow_counts else 0,
            "max": max(flow_counts) if flow_counts else 0,
            "mean": float(np.mean(flow_counts)) if flow_counts else 0.0,
        },
        "host_degree_max_mean": float(np.mean(host_degree_maxes)) if host_degree_maxes else 0.0,
        "class_counts": class_counts,
    }
    return graphs, split_report


def run(config_path: str | Path) -> dict[str, Any]:
    """Build every task's mini-graphs, save them, and write a combined counts report."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    processed_dir = Path(cfg["paths"]["processed_dir"])
    out_dir = Path(cfg["paths"]["out_dir"])
    vocab_path = Path(cfg["paths"]["vocab_path"])
    features = cfg["features"]
    graph_size = cfg["graph_size"]
    seed = cfg["seed"]
    benign_ratio = cfg["sampling"]["benign_ratio"]

    task_paths = sorted(processed_dir.glob("task_*.parquet"))
    if not task_paths:
        raise FileNotFoundError(f"No task_*.parquet files found under {processed_dir}")

    vocab = build_vocab(task_paths)
    save_vocab(vocab, vocab_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"graph_size": graph_size}
    total_graphs = 0
    for path in task_paths:
        task_id = int(path.stem.split("_")[1])
        frame = pd.read_parquet(path)
        task_report: dict[str, Any] = {}
        for split in ("train", "val", "test"):
            split_frame = frame[frame["split"] == split].reset_index(drop=True)
            graphs, split_report = build_split_graphs(
                split_frame, features, vocab, graph_size, seed, benign_ratio
            )
            torch.save(graphs, out_dir / f"task_{task_id}_{split}.pt")
            task_report[split] = split_report
            total_graphs += split_report["num_graphs"]
        report[str(task_id)] = task_report
        print(
            f"[graphs] task {task_id}: "
            + ", ".join(
                f"{split}={task_report[split]['num_graphs']} graphs "
                f"(flows/graph mean={task_report[split]['flows_per_graph']['mean']:.0f})"
                for split in ("train", "val", "test")
            ),
            flush=True,
        )

    (out_dir / "graph_counts.json").write_text(json.dumps(report, indent=2))
    print(
        f"[done] wrote {total_graphs} graphs (graph_size={graph_size}) "
        f"+ graph_counts.json to {out_dir}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="TRENCH-IDS Step 2 graph construction.")
    parser.add_argument("--config", default="configs/graph.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
