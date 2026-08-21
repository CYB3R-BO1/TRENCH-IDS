"""Transferability-guided replay-buffer selection (2026-07-29 design,
docs/superpowers/specs/2026-07-29-transferability-followup-design.md,
Experiment 2).

Mini-graphs are usually not class-pure (2-3 distinct classes per 300-flow
window even under the 2026-08-17 rebuild's capture-order chunking, per
``docs/methodology-2026-08-17.md`` -- though a single-class window is now
possible where a real capture has a long same-class run, e.g. a flood-style
attack; see ``weighted_sample_without_replacement``'s handling of that), so
this module implements *enrichment-weighted* sampling rather than
class-pure selection: each candidate graph gets a score based on how much
of its flow-node composition favors high/low-transferability classes, then
the replay buffer is filled via weighted sampling without replacement using
that score.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData


def class_scores_from_relation_scores(
    per_class_relation_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Collapse ``aggregate_per_class_relation_scores``' per-relation
    output to one scalar per class (mean over relations) -- v1's
    simplification, see design §Experiment 2 step 2. ``0.0`` for a class
    with no relations to average (matches the empty-bank convention
    upstream in ``transferability.py``)."""
    return {
        class_name: (sum(relations.values()) / len(relations) if relations else 0.0)
        for class_name, relations in per_class_relation_scores.items()
    }


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize into [0, 1]. All-equal input (including a single
    class, e.g. T1/T2's one new attack class) has no basis to
    differentiate classes, so it maps to a constant 0.5 rather than
    dividing by zero."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {name: 0.5 for name in scores}
    return {name: (value - lo) / (hi - lo) for name, value in scores.items()}


def with_neutral_benign(class_weights: dict[str, float], benign_name: str) -> dict[str, float]:
    """Benign has no entry in the attack-only memory bank, so it gets a
    neutral weight equal to the mean of the (already mode-specific, i.e.
    already high- or low-transfer-adjusted) attack-class weights, keeping
    it from being starved out of the buffer regardless of which direction
    is being tested."""
    weights = dict(class_weights)
    weights[benign_name] = (
        sum(class_weights.values()) / len(class_weights) if class_weights else 0.5
    )
    return weights


def graph_enrichment_score(
    graph: HeteroData, class_weights: dict[str, float], label_names: list[str]
) -> float:
    """Weighted mean of ``class_weights`` over one graph's flow-node
    labels -- how much this graph's composition favors the weighted
    classes. A class present in ``label_names`` but absent from
    ``class_weights`` defaults to 0.0 (shouldn't happen once
    ``with_neutral_benign`` has been applied upstream, but avoids a
    KeyError if called directly in a test)."""
    y = graph["flow"].y
    per_index_weight = torch.tensor(
        [class_weights.get(name, 0.0) for name in label_names], dtype=torch.float32
    )
    return per_index_weight[y].mean().item()


def weighted_sample_without_replacement(
    items: list, weights: np.ndarray, n_sample: int, rng: np.random.Generator
) -> list:
    """Sample ``min(n_sample, len(items))`` items without replacement,
    weighted by ``weights``. Falls back to uniform sampling -- rather than
    letting ``rng.choice`` raise -- whenever the weighting cannot support a
    without-replacement draw of this size:

    * an all-zero (or otherwise non-positive-sum) weight vector (e.g. every
      candidate graph scored 0.0), or
    * fewer nonzero-weight items than ``n_sample``. This is a real risk, not
      a theoretical one: graphs.py chunks mini-graphs in *capture order*
      (2026-08-17 rebuild), not from a global shuffle, so a 300-flow window
      landing entirely inside a flood-style attack's burst can be purely the
      min-scoring class with zero Benign flows -- which scores exactly 0.0
      under this module's min-max normalisation (see
      ``graph_enrichment_score``). If enough of a task's graphs do that,
      fewer than ``n_sample`` carry positive weight and an unguarded
      ``rng.choice(..., replace=False, p=probs)`` raises
      ``ValueError: Fewer non-zero entries in p than size``.

    Both are degenerate weighting, not errors worth raising over -- a
    replay buffer that falls back to uniform for one task is still a
    correct buffer, just an unweighted one for that draw.
    """
    n_sample = min(n_sample, len(items))
    total = float(weights.sum())
    nonzero = int((weights > 0).sum())
    if total > 0 and nonzero >= n_sample:
        probs = weights / total
    else:
        probs = np.full(len(items), 1.0 / len(items))
    indices = rng.choice(len(items), size=n_sample, replace=False, p=probs)
    return [items[i] for i in indices]


def select_replay_graphs(
    graphs: list[HeteroData],
    per_class_relation_scores: dict[str, dict[str, float]],
    label_names: list[str],
    n_sample: int,
    mode: str,
    benign_name: str,
    rng: np.random.Generator,
) -> tuple[list[HeteroData], list[float]]:
    """Full pipeline: per-class-per-relation scores -> per-class scalar
    -> normalized -> (for enrich_low_transfer) inverted -> neutral benign
    weight added -> per-graph enrichment score -> weighted sample without
    replacement. Returns ``(selected_graphs, selected_graph_scores)`` --
    the scores are logged by the caller for the buffer-composition
    diagnostics (design §Experiment 2 Diagnostics), not recomputed there.

    ``mode`` must be ``"enrich_high_transfer"`` or ``"enrich_low_transfer"``
    -- ``"uniform"`` is handled by the caller as an exact passthrough to
    the pre-existing ``random.sample`` code path and never reaches this
    function (design §Experiment 2 step 6).
    """
    raw = class_scores_from_relation_scores(per_class_relation_scores)
    normalized = normalize_scores(raw)
    if mode == "enrich_low_transfer":
        normalized = {name: 1.0 - value for name, value in normalized.items()}
    elif mode != "enrich_high_transfer":
        raise ValueError(f"Unknown replay selection mode: {mode!r}")
    class_weights = with_neutral_benign(normalized, benign_name)

    graph_scores = np.array(
        [graph_enrichment_score(g, class_weights, label_names) for g in graphs]
    )
    selected = weighted_sample_without_replacement(graphs, graph_scores, n_sample, rng)
    selected_indices = {id(g): score for g, score in zip(graphs, graph_scores, strict=True)}
    selected_scores = [selected_indices[id(g)] for g in selected]
    return selected, selected_scores
