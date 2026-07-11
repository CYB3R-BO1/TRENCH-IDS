"""Similarity-driven continual-learning task assignment.

Turns the attack-class cosine-similarity matrix (trench_ids.similarity) into a
concrete task grouping: classes with a pairwise similarity above ``threshold``
must never land in the same task -- co-locating them gives the model nothing
to transfer, since both would be learned directly from labels in one task.

Method: keep the existing 7-task cadence (14 classes / 2 per task), and among
every perfect pairing that respects the threshold, pick the one minimizing
total within-task similarity via exact brute-force enumeration (a "conflict
graph" of 14 nodes has at most 135,135 perfect matchings -- small enough to
enumerate directly, no networkx/scipy dependency needed).

Run:  python -m trench_ids.task_design --config configs/similarity.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_similarity_matrix(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def assign_pairs(sim: pd.DataFrame, threshold: float = 0.5) -> tuple[list[tuple[str, str]], float]:
    """Best perfect pairing: minimum total similarity, no pair over threshold.

    Requires an even number of classes. Raises if the conflict graph is too
    dense for a valid perfect pairing to exist (consider a higher threshold or
    larger group size).
    """
    classes = list(sim.index)
    if len(classes) % 2 != 0:
        raise ValueError(f"assign_pairs needs an even class count, got {len(classes)}")

    best: dict[str, Any] = {"pairs": None, "total": float("inf")}

    def recurse(remaining: list[str], pairs: list[tuple[str, str]], total: float) -> None:
        if not remaining:
            if total < best["total"]:
                best["pairs"], best["total"] = list(pairs), total
            return
        a = remaining[0]
        rest = remaining[1:]
        for b in rest:
            s = sim.loc[a, b]
            if s > threshold:
                continue  # hard constraint: never co-locate a similar pair
            recurse([c for c in rest if c != b], pairs + [(a, b)], total + s)

    recurse(classes, [], 0.0)
    if best["pairs"] is None:
        raise ValueError(
            f"No valid pairing respects threshold={threshold}; the conflict graph is "
            f"too dense for pure pairs. Try a higher threshold or larger group size."
        )
    return best["pairs"], best["total"]


def run(config_path: str | Path, threshold: float = 0.5) -> list[tuple[str, str]]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["paths"]["out_dir"])
    sim = load_similarity_matrix(out_dir / "similarity_matrix.csv")

    pairs, total = assign_pairs(sim, threshold)
    print(f"[task_design] threshold={threshold}, total intra-task similarity={total:.4f}")
    for i, (a, b) in enumerate(pairs, start=1):
        print(f"  T{i}: {a} + {b}  (cosine sim = {sim.loc[a, b]:.3f})")

    out_path = out_dir / "task_assignment.csv"
    pd.DataFrame(
        [
            {"task": f"T{i}", "class_a": a, "class_b": b, "cosine_similarity": sim.loc[a, b]}
            for i, (a, b) in enumerate(pairs, start=1)
        ]
    ).to_csv(out_path, index=False)
    print(f"[task_design] wrote {out_path}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Similarity-driven task assignment.")
    parser.add_argument("--config", default="configs/similarity.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args.config, args.threshold)


if __name__ == "__main__":
    main()
