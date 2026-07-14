"""Similarity-driven continual-learning task assignment.

Turns the attack-class cosine-similarity matrix (trench_ids.similarity) into a
concrete task grouping: classes with a pairwise similarity above ``threshold``
must never land in the same task -- co-locating them gives the model nothing
to transfer, since both would be learned directly from labels in one task.

Method ("isolate-and-bundle"): a set of classes that are *all* pairwise
above threshold (a "conflict clique") cannot avoid a conflict no matter how
it's split up, so every member of the largest such clique gets its own
singleton task. The remaining classes -- which by construction each have at
least one compatible partner -- are paired off via minimum-weight perfect
matching (still respecting the threshold) to fill out the rest of the
tasks. When called with ``sizes`` (see ``min_weight_grouping``), the
matching's primary objective is minimizing the largest resulting task's
size, with total similarity as a tie-break among equally-balanced options --
multiple pairings can satisfy the same similarity threshold while differing
sharply in balance, so total-similarity-only optimization is free to (and,
empirically, did) pick the worst-balanced valid option. Without ``sizes``,
the objective is total-similarity-only.

Run:  python -m trench_ids.task_design --config configs/similarity.yaml
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_similarity_matrix(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _group_ok(sim: pd.DataFrame, group: tuple[str, ...], threshold: float) -> bool:
    return all(
        sim.loc[a, b] <= threshold for i, a in enumerate(group) for b in group[i + 1 :]
    )


def _group_score(sim: pd.DataFrame, group: tuple[str, ...]) -> float:
    return sum(sim.loc[a, b] for i, a in enumerate(group) for b in group[i + 1 :])


def min_weight_grouping(
    sim: pd.DataFrame, threshold: float = 0.5, sizes: dict[str, float] | None = None
) -> tuple[list[tuple[str, ...]], float]:
    """Minimum-weight grouping into pairs, no group over threshold.

    An odd number of classes is handled by allowing exactly one group of three
    (otherwise one class would be left over); every other group is a pair.
    Raises if the conflict graph is too dense for a valid grouping to exist.

    If ``sizes`` is given (class -> weight, e.g. flow count), the objective
    becomes lexicographic: primarily minimize the largest group's total
    size, then minimize total pairwise similarity as a tie-break among
    groupings that achieve the same minimal max size. This exists because
    multiple groupings can satisfy the same similarity threshold while
    differing sharply in how balanced the resulting tasks are -- minimizing
    total similarity alone is free to pick among them arbitrarily, and can
    pick the worst-balanced one (see
    docs/superpowers/specs/2026-07-14-benchmark-finalization-design.md §1).
    If ``sizes`` is omitted, the objective is total-similarity-only (prior
    behavior) -- every candidate's max group size collapses to 0.0, so the
    lexicographic key reduces to total similarity alone.
    """
    classes = list(sim.index)
    allow_triple = len(classes) % 2 == 1

    def group_size(group: tuple[str, ...]) -> float:
        return sum(sizes[c] for c in group) if sizes else 0.0

    best: dict[str, Any] = {"groups": None, "total": float("inf"), "max_size": float("inf")}

    def recurse(
        remaining: list[str], groups: list[tuple[str, ...]], total: float, triple_used: bool
    ) -> None:
        if not remaining:
            max_size = max((group_size(g) for g in groups), default=0.0)
            if (max_size, total) < (best["max_size"], best["total"]):
                best["groups"], best["total"], best["max_size"] = list(groups), total, max_size
            return
        a = remaining[0]
        rest = remaining[1:]
        for b in rest:
            if sim.loc[a, b] > threshold:
                continue  # hard constraint: never co-locate a similar pair
            recurse(
                [c for c in rest if c != b], groups + [(a, b)], total + sim.loc[a, b], triple_used
            )
        if allow_triple and not triple_used and len(rest) >= 2:
            for b, c in itertools.combinations(rest, 2):
                triple = (a, b, c)
                if not _group_ok(sim, triple, threshold):
                    continue
                remaining2 = [x for x in rest if x not in (b, c)]
                recurse(remaining2, groups + [triple], total + _group_score(sim, triple), True)

    recurse(classes, [], 0.0, False)
    if best["groups"] is None:
        raise ValueError(
            f"No valid grouping respects threshold={threshold}; the conflict graph is "
            f"too dense. Try a higher threshold or a larger candidate pool."
        )
    return best["groups"], best["total"]


def find_max_clique(sim: pd.DataFrame, threshold: float) -> list[str]:
    """Largest set of classes that are all pairwise above ``threshold``.

    Every member of this set conflicts with every other member, so no split
    into 2+ tasks can avoid at least one conflicting co-location within a
    group larger than 1 -- each member must get a task of its own. Brute-force
    over subsets (largest first); fine for the class counts here (<=20).
    """
    classes = list(sim.index)
    conflict = {
        frozenset((a, b))
        for i, a in enumerate(classes)
        for b in classes[i + 1 :]
        if sim.loc[a, b] > threshold
    }

    def is_clique(group: tuple[str, ...]) -> bool:
        return all(
            frozenset((a, b)) in conflict for i, a in enumerate(group) for b in group[i + 1 :]
        )

    for r in range(len(classes), 1, -1):
        for combo in itertools.combinations(classes, r):
            if is_clique(combo):
                return list(combo)
    return []


def assign_groups(
    sim: pd.DataFrame, threshold: float = 0.5, sizes: dict[str, float] | None = None
) -> tuple[list[list[str]], list[float]]:
    """Isolate-and-bundle task assignment (see module docstring).

    Returns (groups, per-group max internal similarity). Singleton groups
    (isolated clique members) report 0.0 since there is no internal pair.
    ``sizes``, if given, is threaded into the non-clique remainder's
    grouping (see ``min_weight_grouping``'s ``sizes`` parameter) -- clique
    members are singleton tasks regardless of size, so sizes never affects
    which classes get isolated, only how the remainder pairs up.
    """
    clique = find_max_clique(sim, threshold)
    remaining = [c for c in sim.index if c not in clique]

    groups: list[list[str]] = [[c] for c in clique]
    scores: list[float] = [0.0] * len(clique)

    if remaining:
        rem_sim = sim.loc[remaining, remaining]
        rem_sizes = {c: sizes[c] for c in remaining} if sizes else None
        rem_groups, _ = min_weight_grouping(rem_sim, threshold, rem_sizes)
        for group in rem_groups:
            groups.append(list(group))
            scores.append(max(sim.loc[a, b] for i, a in enumerate(group) for b in group[i + 1 :]))

    return groups, scores


def run(config_path: str | Path, threshold: float = 0.35) -> list[list[str]]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["paths"]["out_dir"])
    sim = load_similarity_matrix(out_dir / "similarity_matrix.csv")

    groups, scores = assign_groups(sim, threshold)
    print(f"[task_design] threshold={threshold}, isolate-and-bundle grouping ({len(groups)} tasks)")
    for i, (group, score) in enumerate(zip(groups, scores, strict=True), start=1):
        label = " + ".join(group)
        suffix = f"  (max intra-task cosine sim = {score:.3f})" if len(group) > 1 else " (isolated)"
        print(f"  T{i}: {label}{suffix}")

    out_path = out_dir / "task_assignment.csv"
    pd.DataFrame(
        [
            {"task": f"T{i}", "classes": " + ".join(group), "max_intra_task_similarity": score}
            for i, (group, score) in enumerate(zip(groups, scores, strict=True), start=1)
        ]
    ).to_csv(out_path, index=False)
    print(f"[task_design] wrote {out_path}")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Similarity-driven task assignment.")
    parser.add_argument("--config", default="configs/similarity.yaml")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    run(args.config, args.threshold)


if __name__ == "__main__":
    main()
