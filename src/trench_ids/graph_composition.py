"""Graph composition analysis — per-task class-mix breakdown of saved mini-graphs.

Reports, for each task's saved mini-graphs (``task_{t}_{split}.pt`` from
``trench_ids.graphs``), how many graphs are "pure" (every flow the same
canonical class, attack or Benign) versus "mixed" (two or more classes
present together in the same mini-graph), and which class numerically
dominates each mixed graph's flow count. Intended as context for
downsampling decisions -- an indiscriminate downsampling pass could
disproportionately remove mixed graphs (which carry co-occurrence signal a
pure graph doesn't) or graphs dominated by a minority class.

Three outputs, all keyed by task, combined across train/val/test:

  - Exact composition table: every class-combination actually observed in a
    task's graphs, with both the raw graph count and its fraction of the
    task's total -- the fraction is what a proportional downsampling pass
    would preserve. Since class assignment is task-exclusive
    (``trench_ids.labels.CANONICAL_TO_TASK`` — a class belongs to exactly
    one task), a 2-attack-class task has at most 2**3 - 1 = 7 combinations
    (2 attack classes + Benign); a 1-attack-class task (T1, T2) has at most
    3 (attack-only, Benign-only, attack+Benign mixed).
  - Dominant-class summary: for each graph, the canonical class with the
    most flows in it (by raw flow count within that graph, not just
    presence/absence); "Equal" when two or more classes tie for the max.
    Distinguishes a mixed graph that's actually lopsided toward one class
    from one that's a genuinely balanced mix.
  - Average class proportion: per task, the mean per-graph class share
    (each graph weighted equally, not flow-weighted), e.g. "DDoS 91%,
    Infiltration 5%, Benign 4%". Dominant-class alone can't tell a 51/49
    split from a 95/5 one -- this can.
  - An 11x11 (Benign + 10 attack classes, ``trench_ids.labels.
    canonical_classes()`` order) symmetric matrix, summed across every
    task: diagonal = pure-class graph counts; off-diagonal[i][j] = graphs
    containing BOTH class i and class j (inclusive of a third class also
    being present -- see the exact composition table above for the
    unambiguous three-way count). Cross-task cells are structurally zero,
    since no two classes from different tasks ever appear in the same
    mini-graph.

Run:  python -m trench_ids.graph_composition --graphs-dir data/graphs_ratio4
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from trench_ids.labels import canonical_classes


def graph_class_set(graph: HeteroData) -> frozenset[str]:
    """The distinct canonical labels present in one mini-graph."""
    return frozenset(graph["flow"].label_names)


def graph_class_counts(graph: HeteroData) -> Counter:
    """Flow count per canonical class within one mini-graph.

    ``flow.y`` is an index into ``flow.label_names`` (see
    ``trench_ids.graphs.build_task_graph``); this maps back to class names
    to get an actual per-class tally, not just which classes are present.
    """
    label_names = graph["flow"].label_names
    counts: Counter = Counter()
    for idx in graph["flow"].y.tolist():
        counts[label_names[idx]] += 1
    return counts


def dominant_class(counts: Counter) -> str:
    """The class with the most flows in a graph; 'Equal' on a tie for the max."""
    if not counts:
        return "Equal"
    top_count = max(counts.values())
    leaders = [c for c, n in counts.items() if n == top_count]
    return leaders[0] if len(leaders) == 1 else "Equal"


def graph_class_proportions(graph: HeteroData) -> dict[str, float]:
    """Fraction of this graph's flows belonging to each class present."""
    counts = graph_class_counts(graph)
    total = sum(counts.values())
    return {c: n / total for c, n in counts.items()} if total else {}


def load_task_graphs(out_dir: Path, task_id: int) -> list[HeteroData]:
    """All of one task's mini-graphs, combined across train/val/test."""
    graphs: list[HeteroData] = []
    for split in ("train", "val", "test"):
        path = out_dir / f"task_{task_id}_{split}.pt"
        if path.exists():
            graphs.extend(torch.load(path, weights_only=False))  # trusted, first-party output
    return graphs


def task_combination_counts(graphs: list[HeteroData]) -> Counter:
    """Count of graphs per exact class-combination (pure or mixed)."""
    return Counter(graph_class_set(g) for g in graphs)


def task_dominant_class_counts(graphs: list[HeteroData]) -> Counter:
    """Count of graphs per dominant class (by flow count), 'Equal' on ties."""
    return Counter(dominant_class(graph_class_counts(g)) for g in graphs)


def task_average_class_proportions(graphs: list[HeteroData]) -> dict[str, float]:
    """Mean per-graph class share across a task's graphs, each graph weighted
    equally (not flow-weighted -- a task's smaller-attack-count graphs count
    the same as its larger ones).

    Complements dominant_class: two tasks can share the same dominant class
    while one is a lopsided 95/5 split and the other a near-even 51/49 one.
    """
    sums: Counter = Counter()
    n = len(graphs)
    if n == 0:
        return {}
    for g in graphs:
        for cls, share in graph_class_proportions(g).items():
            sums[cls] += share
    return {cls: total / n for cls, total in sums.items()}


def combination_label(combo: frozenset[str]) -> str:
    """Human-readable label for a combination: 'pure X' or 'mix of A + B'."""
    names = sorted(combo)
    if len(names) == 1:
        return f"pure {names[0]}"
    return "mix of " + " + ".join(names)


def build_matrix(task_combos: dict[int, Counter], classes: list[str]) -> pd.DataFrame:
    """Symmetric NxN matrix (N = len(classes)) summed across every task.

    Diagonal[i] = graphs pure in class i. Off-diagonal[i][j] = graphs
    containing both class i and class j, regardless of any other class also
    present in that graph -- a three-way mix increments all three of its
    pairwise cells, so this matrix alone cannot distinguish a two-way mix
    from a three-way one; use the exact per-task combination table for that.
    """
    matrix = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    for combos in task_combos.values():
        for combo, count in combos.items():
            names = sorted(combo)
            if len(names) == 1:
                matrix.loc[names[0], names[0]] += count
            else:
                for i, a in enumerate(names):
                    for b in names[i + 1 :]:
                        matrix.loc[a, b] += count
                        matrix.loc[b, a] += count
    return matrix


def run(graphs_dir: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    """Build and write the per-task composition report + the class matrix."""
    graphs_dir = Path(graphs_dir)
    out_dir = Path(out_dir) if out_dir else graphs_dir

    task_ids = sorted({int(p.stem.split("_")[1]) for p in graphs_dir.glob("task_*_*.pt")})
    if not task_ids:
        raise FileNotFoundError(f"No task_*_*.pt files found under {graphs_dir}")

    classes = canonical_classes()
    task_combos: dict[int, Counter] = {}
    dominant_totals: Counter = Counter()
    report: dict[str, Any] = {"source": str(graphs_dir), "tasks": {}}

    for task_id in task_ids:
        graphs = load_task_graphs(graphs_dir, task_id)
        total = len(graphs)

        combos = task_combination_counts(graphs)
        task_combos[task_id] = combos
        ranked = sorted(combos.items(), key=lambda kv: -kv[1])

        dominant = task_dominant_class_counts(graphs)
        dominant_totals.update(dominant)
        dominant_ranked = sorted(dominant.items(), key=lambda kv: -kv[1])

        avg_proportions = task_average_class_proportions(graphs)
        avg_ranked = sorted(avg_proportions.items(), key=lambda kv: -kv[1])

        report["tasks"][str(task_id)] = {
            "total_graphs": total,
            "combinations": {
                combination_label(combo): {"count": count, "fraction": count / total}
                for combo, count in ranked
            },
            "dominant_class": {name: count for name, count in dominant_ranked},
            "average_class_proportion": {cls: share for cls, share in avg_ranked},
        }

        print(f"[composition] task {task_id}: {total:,} graphs")
        for combo, count in ranked:
            print(f"    {combination_label(combo):40s} {count:>7,}  ({count / total:>6.1%})")
        print("    dominant class:")
        for name, count in dominant_ranked:
            print(f"      {name:30s} {count:>7,}  ({count / total:>6.1%})")
        print("    average class proportion (mean per-graph share):")
        for cls, share in avg_ranked:
            print(f"      {cls:30s} {share:>6.1%}")

    report["dominant_class_totals"] = dict(
        sorted(dominant_totals.items(), key=lambda kv: -kv[1])
    )

    matrix = build_matrix(task_combos, classes)
    report["matrix"] = matrix.to_dict()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph_composition.json").write_text(json.dumps(report, indent=2))
    matrix.to_csv(out_dir / "graph_composition_matrix.csv")
    print(f"[done] wrote graph_composition.json + graph_composition_matrix.csv to {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-task graph class-composition breakdown (pure vs. mixed)."
    )
    parser.add_argument("--graphs-dir", default="data/graphs_ratio4")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run(args.graphs_dir, args.out_dir)


if __name__ == "__main__":
    main()
