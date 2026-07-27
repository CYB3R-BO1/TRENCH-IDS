"""Step 10 (third bullet) -- build inference-only mini-graphs for
unseen-attack data (trench_ids.unseen_data's output).

Reuses trench_ids.graphs.build_task_graph completely unchanged -- these
graphs are built exactly the same way as the continual-learning pipeline's;
the only difference is they are never used during training, only passed
through a trained checkpoint for inference and analysis.

Two adjustments live here, not in graphs.py:

  - Vocabulary check: every PROTOCOL/L7_PROTO value in the extracted data
    must already exist in the trained vocab (data/graphs/vocab.json).
    NetFlow v2 standardizes these fields across all UQ datasets, so this is
    expected to pass -- but it fails loudly (raises) rather than silently
    extending the vocab, since an unverified new value would produce a
    meaningless, never-trained embedding row.

  - y=-1 sentinel: build_task_graph maps canonical_label -> global class
    index via graphs.LABEL_LOOKUP, which has no entry for Dataset B's
    classes (Backdoor, MITM, Ransomware, Web Attacks, Theft). This module
    substitutes BENIGN (present in LABEL_LOOKUP) purely to satisfy that
    cast, then immediately overwrites graph["flow"].y with -1 -- a
    placeholder that exists only to satisfy the graph schema. It is never
    passed to CrossEntropyLoss or any training routine (this pipeline is
    inference-only). The true label survives separately as reporting
    metadata: graph["flow"].true_label.

Run: trench-unseen-graphs --config configs/unseen_graphs.yaml
  or: python -m trench_ids.unseen_graphs --config configs/unseen_graphs.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData

from trench_ids.graphs import LABEL_LOOKUP, build_task_graph
from trench_ids.labels import BENIGN
from trench_ids.vocab import load_vocab

VOCAB_COLUMNS = ("PROTOCOL", "L7_PROTO")


def check_vocab_coverage(frame: pd.DataFrame, vocab: dict[str, dict[str, int]]) -> None:
    """Raise if any PROTOCOL/L7_PROTO value in `frame` is missing from
    `vocab` -- extending the vocab silently would produce an embedding row
    for a value the model never trained on."""
    for column in VOCAB_COLUMNS:
        known = vocab[column]
        seen = frame[column].astype(str).unique()
        missing = sorted(v for v in seen if v not in known)
        if missing:
            raise ValueError(
                f"{column} value(s) {missing} not present in the trained vocab "
                f"({column}) -- refusing to silently extend it."
            )


def _chunk(frame: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    return [
        frame.iloc[start : start + size].reset_index(drop=True)
        for start in range(0, len(frame), size)
    ]


def build_unseen_graphs(
    frame: pd.DataFrame,
    features: list[str],
    vocab: dict[str, dict[str, int]],
    graph_size: int,
) -> list[tuple[HeteroData, dict[str, Any]]]:
    """Builds one or more HeteroData mini-graphs from `frame` (every row
    shares one canonical_label, since trench_ids.unseen_data writes one
    parquet per class)."""
    check_vocab_coverage(frame, vocab)
    if frame.empty:
        return []

    true_label = frame["canonical_label"].iloc[0]
    is_known = true_label in LABEL_LOOKUP

    out: list[tuple[HeteroData, dict[str, Any]]] = []
    for chunk in _chunk(frame, graph_size):
        working = chunk.copy()
        working["split"] = "test"
        if not is_known:
            working["canonical_label"] = BENIGN  # sentinel-only, overwritten below
        graph, counts = build_task_graph(working, features, vocab)
        if not is_known:
            graph["flow"].y = torch.full((len(chunk),), -1, dtype=torch.int64)
            counts["flow_class_counts"] = {true_label: len(chunk)}
        graph["flow"].true_label = [true_label] * len(chunk)
        out.append((graph, counts))
    return out


def run(config_path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    unseen_dir = Path(cfg["paths"]["unseen_dir"])
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab(Path(cfg["paths"]["vocab_path"]))
    features = cfg["features"]
    graph_size = cfg["graph_size"]

    manifest = json.loads((unseen_dir / "manifest.json").read_text())
    summary: dict[str, Any] = {}
    for slug, entry in manifest["classes"].items():
        frame = pd.read_parquet(entry["output_path"])
        graphs_and_counts = build_unseen_graphs(frame, features, vocab, graph_size)
        graphs = [g for g, _ in graphs_and_counts]
        torch.save(graphs, out_dir / f"{slug}.pt")
        summary[slug] = {
            "num_graphs": len(graphs),
            "num_flows": sum(len(g["flow"].true_label) for g in graphs),
        }
    (out_dir / "graph_counts.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build inference-only mini-graphs for Step 10 unseen-attack data."
    )
    parser.add_argument("--config", default="configs/unseen_graphs.yaml")
    args = parser.parse_args()
    summary = run(args.config)
    for slug, info in summary.items():
        print(f"{slug}: {info['num_graphs']} graphs, {info['num_flows']} flows")


if __name__ == "__main__":
    main()
