"""Step 2 — heterogeneous graph construction.

Builds many small, self-contained PyTorch Geometric ``HeteroData`` graphs
per task from that task's processed Parquet
(``data/processed/task_{t}.parquet``), per the node/relation design in
docs/dataset-plan.md §3:

  Node types: Host, Flow, Protocol, Port, Service.
  Relations : host->flow (originates), flow->host src role (originated_by),
              flow->host dst role (terminates_at), host->flow dst role
              (terminated_by), flow->port (targets_port), port->flow
              (targeted_by), flow->protocol (uses_protocol), protocol->flow
              (protocol_of), flow->service (uses_service), service->flow
              (service_of), host->host (communicates_with, aggregated
              within the mini-graph only).

  The 5 reverse relations (originated_by/terminated_by/targeted_by/
  protocol_of/service_of) were added per the professor's explicit
  instruction so Flow -- the classification target -- receives messages
  along 5 incoming relations instead of 1 (``originates`` was previously
  its only incoming edge); Host goes from 2 incoming relations to 3
  (``originated_by`` added). ``communicates_with`` (host->host) has no
  reverse -- only the 5 Flow-centric relations do. Protocol/Service/Port
  still have exactly 1 incoming relation each (unaffected), consistent with
  them carrying no continuous feature vector to enrich (see
  ``trench_ids.model.rhgnn``).

Each task's rows are split by train/val/test (the ``split`` column from
Step 1), then each split's rows are put back into **capture order**
(``source_dataset``, ``source_row``) and chunked into consecutive
mini-graphs of at most ``graph_size`` flows (configs/graph.yaml). Chunking
within a split keeps every mini-graph on one side of the train/val/test
boundary; capture-order chunking (replacing the global random shuffle used
until 2026-08-16) makes each mini-graph a contiguous window of one dataset's
traffic, so Host/Port aggregation describes a real interval of activity
rather than a task-wide average -- see ``_capture_order`` for the full
rationale and for why class mixing survives the change. Flow
node identity is one row of the chunk; Host identity is (source_dataset,
IP), aggregated within that mini-graph only — it never persists across
mini-graphs, splits, or tasks. Protocol/L7_PROTO node identity uses the
global vocabulary from ``trench_ids.vocab`` so the same value maps to the
same category everywhere; Port stays chunk-local (raw destination port
number).

Continual-learning task boundaries (T1..T6) are unaffected: training still
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

from trench_ids.labels import BENIGN, canonical_classes
from trench_ids.vocab import build_vocab, save_vocab, vocab_key

# Fixed global class -> index mapping for Flow node labels (graph["flow"].y),
# same order/meaning in every mini-graph across every task/split/graph_size/
# benign_ratio set. Must NOT be built per mini-graph from that chunk's local
# canonical_label.unique() -- a classifier trained across mini-graphs needs
# one stable label space (index 4 always means the same class everywhere),
# or CrossEntropyLoss gets contradictory supervision batch to batch.
LABEL_NAMES = canonical_classes()
LABEL_LOOKUP = {name: i for i, name in enumerate(LABEL_NAMES)}


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

    protocol_vocab_ids = [vocab["PROTOCOL"][vocab_key(v)] for v in protocol_values]
    service_vocab_ids = [vocab["L7_PROTO"][vocab_key(v)] for v in service_values]

    y = frame["canonical_label"].map(LABEL_LOOKUP).to_numpy(dtype=np.int64)

    hh_edge_index, hh_edge_attr = _host_host_edges(frame, src_idx, dst_idx)

    graph = HeteroData()

    graph["flow"].x = torch.tensor(frame[features].to_numpy(dtype=np.float32))
    graph["flow"].y = torch.tensor(y)
    graph["flow"].label_names = LABEL_NAMES
    graph["flow"].flow_id = frame["flow_id"].tolist()
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

    # 5 new reverse relations (professor-specified) so Flow -- the
    # classification target -- receives messages along 5 incoming relations
    # instead of 1, and Host along 3 instead of 2. No reverse is added for
    # communicates_with (host->host).
    graph["flow", "originated_by", "host"].edge_index = torch.tensor(
        np.stack([np.arange(num_flow), src_idx]), dtype=torch.int64
    )
    graph["host", "terminated_by", "flow"].edge_index = torch.tensor(
        np.stack([dst_idx, np.arange(num_flow)]), dtype=torch.int64
    )
    graph["port", "targeted_by", "flow"].edge_index = torch.tensor(
        np.stack([port_idx, np.arange(num_flow)]), dtype=torch.int64
    )
    graph["protocol", "protocol_of", "flow"].edge_index = torch.tensor(
        np.stack([protocol_idx, np.arange(num_flow)]), dtype=torch.int64
    )
    graph["service", "service_of", "flow"].edge_index = torch.tensor(
        np.stack([service_idx, np.arange(num_flow)]), dtype=torch.int64
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
            "flow_originated_by_host": num_flow,
            "host_terminated_by_flow": num_flow,
            "port_targeted_by_flow": num_flow,
            "protocol_protocol_of_flow": num_flow,
            "service_service_of_flow": num_flow,
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
) -> tuple[pd.DataFrame, float]:
    """Subsample benign_rows to hit attack:benign == benign_ratio:1.

    **Sampling with replacement was removed 2026-08-16.** The old fallback
    repeated benign rows whenever the ratio demanded more than the pool
    held, which at the previous Step 1 sizing meant every benign flow
    appeared 54-158 times: benign is ~25% of every pooled metric, so a
    quarter of every reported number rested on ~7,200 distinct flows, and a
    model could memorise them. Step 1's quota planner now sizes each task's
    benign pool so this branch is unnecessary
    (``benign_per_task * benign_ratio == attack_per_task`` exactly); if a
    pool still falls short, the shortfall is taken honestly (all available
    rows, each used once) and surfaced in the report as a realised ratio
    that differs from the requested one, rather than being papered over
    with duplicates.

    Returns (selected_benign_rows, realised_attack_to_benign_ratio).
    """
    n_attack = len(attack_rows)
    if n_attack == 0 or benign_rows.empty:
        return benign_rows.iloc[0:0], float("inf")
    target = max(1, round(n_attack / benign_ratio))
    if target <= len(benign_rows):
        selected = benign_rows.sample(n=target, random_state=seed).reset_index(drop=True)
    else:
        selected = benign_rows.reset_index(drop=True)
    return selected, n_attack / max(len(selected), 1)


def _capture_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Order rows the way the traffic was actually captured.

    Sorts by ``(source_dataset, source_row)`` -- the row's 0-indexed line
    number in its own raw CSV, which for these captures is acquisition
    order. Two consequences, both of which the previous global
    ``frame.sample(frac=1)`` shuffle destroyed:

    * **Mini-graph windows become time-local.** A 300-flow chunk now spans a
      contiguous stretch of one capture instead of being drawn uniformly
      from the whole task. Everything the graph is supposed to contribute
      over per-flow features depends on this: a Host node's aggregate
      degree/port-spread and a Port node's in-degree only mean "burst" if
      the window is a real interval. Under a global shuffle they instead
      estimate the *task-wide average rate*, which carries far less signal
      about what is happening right now, and which per-flow features
      partially encode anyway.
    * **Mini-graphs stop mixing datasets.** Sorting by dataset first keeps
      each window inside a single capture, so a graph no longer splices two
      unrelated testbeds' hosts into one "network" -- physically meaningless
      topology that the model was nonetheless asked to aggregate over. Only
      the one chunk straddling each dataset boundary is mixed.

    Class mixing, the reason the shuffle existed, survives: the raw CSVs
    interleave attack and benign traffic, so capture-order windows are
    still multi-class (measured on real data: 2-3 distinct classes per
    300-flow window in T3, versus the near-single-class windows that
    chunking the class-ordered Parquet directly would give).
    """
    if "source_row" not in frame.columns:
        raise KeyError(
            "capture-order chunking needs the 'source_row' column written by "
            "Step 1 (trench_ids.preprocess). Re-run Step 1 -- Parquet files "
            "produced before 2026-08-16 do not have it."
        )
    return frame.sort_values(["source_dataset", "source_row"], kind="stable").reset_index(
        drop=True
    )


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
    rows are ever dropped, never repeated. The combined rows are then put
    back into **capture order** (see :func:`_capture_order`) before
    chunking, so each mini-graph is a contiguous window of one dataset's
    traffic rather than a uniform random sample of the whole task.
    """
    is_benign = frame["canonical_label"] == BENIGN
    attack_rows = frame[~is_benign].reset_index(drop=True)
    benign_rows = frame[is_benign].reset_index(drop=True)
    benign_selected, realised_ratio = _select_benign(
        attack_rows, benign_rows, benign_ratio, seed
    )

    frame = pd.concat([attack_rows, benign_selected], ignore_index=True)
    frame = _capture_order(frame)
    chunks = _chunk_frame(frame, graph_size)
    graphs: list[HeteroData] = []
    flow_counts: list[int] = []
    host_degree_maxes: list[float] = []
    host_counts: list[int] = []
    capture_spans: list[float] = []
    class_counts: dict[str, int] = {}
    for chunk in chunks:
        graph, counts = build_task_graph(chunk, features, vocab)
        graphs.append(graph)
        flow_counts.append(counts["node_counts"]["flow"])
        host_counts.append(counts["node_counts"]["host"])
        host_degree_maxes.append(counts["host_degree"]["max"])
        rows = chunk["source_row"].to_numpy()
        capture_spans.append(float(rows.max() - rows.min()) if rows.size else 0.0)
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
        "hosts_per_graph_mean": float(np.mean(host_counts)) if host_counts else 0.0,
        # Mean (max - min) source_row inside a mini-graph: how wide a slice
        # of the original capture one window covers. Small relative to the
        # task's total row span == time-local windows; under the old global
        # shuffle this equalled the whole capture by construction.
        "capture_span_mean": float(np.mean(capture_spans)) if capture_spans else 0.0,
        "benign_unique_rows": int(len(benign_selected)),
        "realised_attack_benign_ratio": realised_ratio,
        "class_counts": class_counts,
    }
    return graphs, split_report


def _downsample_tasks(
    out_dir: Path, report: dict[str, Any], max_task_ratio: float, seed: int
) -> dict[str, Any]:
    """Cap any task's total graph count (train+val+test) at
    max_task_ratio times the smallest task's total.

    A task exceeding the cap has its train/val/test lists each randomly
    subsampled by the same shrink factor (preserving existing split
    proportions), re-saved to the same task_{t}_{split}.pt paths, and
    report[t][split]["num_graphs"] updated to match. Tasks within the cap
    are untouched (file and report both). Returns a JSON-able summary; an
    empty "trimmed" dict means the cap never triggered.
    """
    task_keys = [k for k in report if k not in ("graph_size", "benign_ratio")]
    totals = {
        k: sum(report[k][split]["num_graphs"] for split in ("train", "val", "test"))
        for k in task_keys
    }
    min_total = min(totals.values())
    cap = min_total * max_task_ratio

    summary: dict[str, Any] = {
        "max_task_ratio": max_task_ratio,
        "min_task_total": min_total,
        "cap": cap,
        "trimmed": {},
    }
    for k in task_keys:
        if totals[k] <= cap:
            continue
        shrink = cap / totals[k]
        rng = np.random.default_rng(seed)
        after_total = 0
        for split in ("train", "val", "test"):
            path = out_dir / f"task_{k}_{split}.pt"
            graphs = torch.load(path, weights_only=False)  # trusted, first-party output
            target = max(1, round(len(graphs) * shrink))
            if target < len(graphs):
                idx = sorted(rng.choice(len(graphs), size=target, replace=False))
                graphs = [graphs[i] for i in idx]
                torch.save(graphs, path)
            report[k][split]["num_graphs"] = len(graphs)
            after_total += len(graphs)
        summary["trimmed"][k] = {"before_total": totals[k], "after_total": after_total}
    return summary


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
    max_task_ratio = cfg["sampling"]["max_task_ratio"]

    task_paths = sorted(processed_dir.glob("task_*.parquet"))
    if not task_paths:
        raise FileNotFoundError(f"No task_*.parquet files found under {processed_dir}")

    vocab = build_vocab(task_paths)
    save_vocab(vocab, vocab_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"graph_size": graph_size, "benign_ratio": benign_ratio}
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

    downsampling = _downsample_tasks(out_dir, report, max_task_ratio, seed)
    report["downsampling"] = downsampling

    (out_dir / "graph_counts.json").write_text(json.dumps(report, indent=2))
    final_total = sum(
        report[k][s]["num_graphs"]
        for k in report
        if k not in ("graph_size", "benign_ratio", "downsampling")
        for s in ("train", "val", "test")
    )
    print(
        f"[done] wrote {final_total} graphs (graph_size={graph_size}, "
        f"benign_ratio={benign_ratio}) + graph_counts.json to {out_dir}"
    )
    if downsampling["trimmed"]:
        print(
            f"[downsample] max_task_ratio={max_task_ratio}: trimmed tasks "
            f"{sorted(downsampling['trimmed'])}"
        )
    else:
        print(
            f"[downsample] max_task_ratio={max_task_ratio}: no task exceeded "
            f"the cap, nothing trimmed"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="TRENCH-IDS Step 2 graph construction.")
    parser.add_argument("--config", default="configs/graph.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
