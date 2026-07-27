"""Tests for Step 10 unseen-attack graph construction (trench_ids.unseen_graphs)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from trench_ids.unseen_graphs import build_unseen_graphs, drop_vocab_gaps, run

FEATURES = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]


def _vocab() -> dict[str, dict[str, int]]:
    return {"PROTOCOL": {"6": 0, "17": 1}, "L7_PROTO": {"1": 0, "2": 1}}


def _frame(canonical_label: str, n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "IN_BYTES": [10.0] * n,
        "OUT_BYTES": [20.0] * n,
        "FLOW_DURATION_MILLISECONDS": [5.0] * n,
        "PROTOCOL": [6] * n,
        "L7_PROTO": [1] * n,
        "L4_DST_PORT": [80] * n,
        "IPV4_SRC_ADDR": ["10.0.0.1"] * n,
        "IPV4_DST_ADDR": ["10.0.0.2"] * n,
        "flow_id": [f"x-{i}" for i in range(n)],
        "source_dataset": ["X"] * n,
        "canonical_label": [canonical_label] * n,
    })


def test_drop_vocab_gaps_passes_when_all_values_known() -> None:
    frame = _frame("Benign", 2)
    filtered, dropped = drop_vocab_gaps(frame, _vocab())
    assert len(filtered) == 2
    assert dropped == 0


def test_drop_vocab_gaps_drops_unknown_protocol() -> None:
    frame = _frame("Benign", 3)
    frame.loc[1, "PROTOCOL"] = 999  # unknown protocol
    filtered, dropped = drop_vocab_gaps(frame, _vocab())
    assert len(filtered) == 2
    assert dropped == 1


def test_drop_vocab_gaps_drops_unknown_l7_proto() -> None:
    frame = _frame("Benign", 3)
    frame.loc[2, "L7_PROTO"] = 999  # unknown L7_PROTO
    filtered, dropped = drop_vocab_gaps(frame, _vocab())
    assert len(filtered) == 2
    assert dropped == 1


def test_build_unseen_graphs_known_class_keeps_real_label() -> None:
    frame = _frame("Reconnaissance", 3)
    graphs_and_counts, vocab_gap_dropped = build_unseen_graphs(
        frame, FEATURES, _vocab(), graph_size=300
    )

    assert len(graphs_and_counts) == 1
    assert vocab_gap_dropped == 0
    graph, counts = graphs_and_counts[0]
    assert graph["flow"].true_label == ["Reconnaissance"] * 3
    # Reconnaissance is a known class -- y must be its real global index,
    # not the -1 sentinel.
    assert (graph["flow"].y != -1).all()


def test_build_unseen_graphs_unknown_class_gets_sentinel_y() -> None:
    frame = _frame("Backdoor", 3)
    graphs_and_counts, vocab_gap_dropped = build_unseen_graphs(
        frame, FEATURES, _vocab(), graph_size=300
    )

    assert vocab_gap_dropped == 0
    graph, counts = graphs_and_counts[0]
    assert graph["flow"].true_label == ["Backdoor"] * 3
    assert (graph["flow"].y == -1).all()
    assert counts["flow_class_counts"] == {"Backdoor": 3}


def test_build_unseen_graphs_chunks_by_graph_size() -> None:
    frame = _frame("Backdoor", 5)
    graphs_and_counts, vocab_gap_dropped = build_unseen_graphs(
        frame, FEATURES, _vocab(), graph_size=2
    )

    assert vocab_gap_dropped == 0
    assert len(graphs_and_counts) == 3  # chunks of 2, 2, 1
    total_flows = sum(g["flow"].y.shape[0] for g, _ in graphs_and_counts)
    assert total_flows == 5


def test_run_writes_one_pt_file_per_class(tmp_path: Path) -> None:
    unseen_dir = tmp_path / "unseen"
    unseen_dir.mkdir()
    frame = _frame("Backdoor", 3)
    frame.to_parquet(unseen_dir / "ToN_backdoor.parquet", index=False)
    manifest = {
        "classes": {
            "ToN_backdoor": {
                "canonical_label": "Backdoor",
                "output_path": str(unseen_dir / "ToN_backdoor.parquet"),
                "evaluation_type": "unseen_class",
            }
        }
    }
    (unseen_dir / "manifest.json").write_text(json.dumps(manifest))

    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(_vocab()))

    out_dir = tmp_path / "unseen_graphs"
    config_path = tmp_path / "unseen_graphs.yaml"
    config_path.write_text(f"""
paths:
  unseen_dir: {unseen_dir}
  vocab_path: {vocab_path}
  out_dir: {out_dir}
graph_size: 300
features: {FEATURES}
""")

    run(config_path)

    graphs = torch.load(out_dir / "ToN_backdoor.pt", weights_only=False)
    assert len(graphs) == 1
    assert graphs[0]["flow"].true_label == ["Backdoor"] * 3
