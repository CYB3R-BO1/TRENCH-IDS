"""Per-relation representation drift across task boundaries.

Forgetting is usually reported as an accuracy drop, which says *that* the
model lost something but not *where*. This measures the where: for every
consecutive pair of checkpoints, it re-embeds a fixed set of already-seen
flows under both encoders and reports, per relation, how far the embedding
moved (1 - cosine similarity).

The measurement pays for itself twice over.

**It sizes the problem the regulariser is aimed at.** On the rebuilt
benchmark the relations differ by an order of magnitude -- ``service_of``
moves 0.05 across the T5->T6 boundary while ``terminated_by`` moves 0.60 --
so "the encoder drifts" is not one phenomenon but five, and a penalty
weighted uniformly over them is not doing what the words suggest.

**It exposes a confound in the method's own hypothesis.** The relations that
score highest on transferability (``S_r``) turn out to be the ones that drift
*least*. Weighting a drift penalty by ``softmax(S_r)`` therefore concentrates
it where there is least drift to prevent -- which is why the first TRD
configuration tried came out inert and slightly worse than plain replay.
``correlate_with_transferability`` computes that relationship explicitly, so
the uniform/transfer/inverse ablation is read with it in view rather than
against an unstated assumption that the two are independent.

Run:  python -m trench_ids.cl.drift --run-dir runs/gnn_replay_s42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader

from trench_ids.cl.ewc import FLOW_RELATIONS
from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split, sample_graphs
from trench_ids.labels import NUM_TASKS


@torch.no_grad()
def boundary_drift(
    earlier_model: torch.nn.Module,
    earlier_classifier: torch.nn.Module,
    later_model: torch.nn.Module,
    later_classifier: torch.nn.Module,
    graphs: list,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, float]:
    """Drift between two encoders, measured on the same flows.

    Cosine rather than L2 because the relation embeddings are unnormalised
    and their scales differ across relations; an L2 comparison would mostly
    report which relation has the largest activations. Prediction agreement
    and logit KL are reported alongside so encoder drift can be separated
    from decision-boundary drift -- they are different failure modes with
    different fixes, and an accuracy drop alone cannot tell them apart.
    """
    sums: dict[str, float] = {relation: 0.0 for relation in FLOW_RELATIONS}
    fused_sum = 0.0
    agree_sum = 0.0
    kl_sum = 0.0
    batches = 0
    for batch in DataLoader(graphs, batch_size=batch_size):
        batch = batch.to(device)
        before, after = earlier_model(batch), later_model(batch)
        for relation in FLOW_RELATIONS:
            a = before.relations["flow"].get(relation)
            b = after.relations["flow"].get(relation)
            if a is None or b is None:
                continue
            sums[relation] += float(F.cosine_similarity(a, b, dim=-1).mean())
        fused_sum += float(
            F.cosine_similarity(before.fused["flow"], after.fused["flow"], dim=-1).mean()
        )
        before_logits = earlier_classifier(before.fused["flow"])
        after_logits = later_classifier(after.fused["flow"])
        agree_sum += float((before_logits.argmax(-1) == after_logits.argmax(-1)).float().mean())
        kl_sum += float(
            F.kl_div(
                F.log_softmax(after_logits, dim=-1),
                F.softmax(before_logits, dim=-1),
                reduction="batchmean",
            )
        )
        batches += 1
    if not batches:
        return {}
    result = {f"drift_{relation}": 1.0 - sums[relation] / batches for relation in FLOW_RELATIONS}
    result["drift_fused"] = 1.0 - fused_sum / batches
    result["prediction_agreement"] = agree_sum / batches
    result["logit_kl"] = kl_sum / batches
    return result


def correlate_with_transferability(
    mean_drift: dict[str, float], mean_s_r: dict[str, float]
) -> dict[str, Any]:
    """Is a relation's transferability related to how much it drifts?

    Reported with n stated, because n is the number of *relations* (5), not
    the number of measurements -- a correlation over five points is a
    direction to note, never a result to lean on, and writing it without the
    n invites exactly that mistake.
    """
    relations = sorted(set(mean_drift) & set(mean_s_r))
    if len(relations) < 3:
        return {"n": len(relations)}
    drift = [mean_drift[r] for r in relations]
    scores = [mean_s_r[r] for r in relations]
    pearson = pearsonr(scores, drift)
    spearman = spearmanr(scores, drift)
    return {
        "n": len(relations),
        "relations": relations,
        "mean_drift": {r: mean_drift[r] for r in relations},
        "mean_transferability": {r: mean_s_r[r] for r in relations},
        "pearson_r": float(pearson[0]),
        "pearson_p": float(pearson[1]),
        "spearman_rho": float(spearman[0]),
        "spearman_p": float(spearman[1]),
    }


def load_mean_transferability(run_dir: Path) -> dict[str, float]:
    """Mean ``S_r`` per relation over every task that recorded one."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for path in sorted(run_dir.glob("distillation_task_*.json")):
        record = json.loads(path.read_text()).get("s_r", {})
        for relation, value in record.items():
            sums[relation] = sums.get(relation, 0.0) + float(value)
            counts[relation] = counts.get(relation, 0) + 1
    return {r: sums[r] / counts[r] for r in sums if counts[r]}


def run(
    run_dir: Path,
    graphs_dir: Path,
    device: torch.device,
    batch_size: int = 8,
    max_graphs: int = 60,
    seed: int = 0,
) -> dict[str, Any]:
    """Drift at every task boundary, plus the transferability correlation."""
    boundaries: dict[str, Any] = {}
    accumulated: dict[str, list[float]] = {}
    for task in range(1, NUM_TASKS):
        earlier_path = run_dir / f"checkpoint_task_{task}.pt"
        later_path = run_dir / f"checkpoint_task_{task + 1}.pt"
        if not (earlier_path.exists() and later_path.exists()):
            continue
        earlier_model, earlier_classifier, _ = load_checkpoint(earlier_path, graphs_dir, device)
        later_model, later_classifier, _ = load_checkpoint(later_path, graphs_dir, device)
        # Measured on the *earlier* task's own test flows: drift only matters
        # where the model was already known to work. Seeded random subsample,
        # not a prefix slice -- mini-graphs are chunked in capture order, so
        # ``[:max_graphs]`` would measure one specific early-capture window.
        graphs = sample_graphs(
            load_split(graphs_dir, task, "test"),
            max_graphs,
            seed=f"{seed}:drift:task_{task}",
        )
        measured = boundary_drift(
            earlier_model,
            earlier_classifier,
            later_model,
            later_classifier,
            graphs,
            device,
            batch_size,
        )
        boundaries[f"{task}->{task + 1}"] = measured
        for key, value in measured.items():
            accumulated.setdefault(key, []).append(value)

    mean_over_boundaries = {k: sum(v) / len(v) for k, v in accumulated.items() if v}
    mean_drift = {
        relation: mean_over_boundaries[f"drift_{relation}"]
        for relation in FLOW_RELATIONS
        if f"drift_{relation}" in mean_over_boundaries
    }
    report: dict[str, Any] = {
        "run": run_dir.name,
        "boundaries": boundaries,
        "mean_over_boundaries": mean_over_boundaries,
        "relation_drift_ranking": sorted(mean_drift, key=mean_drift.get, reverse=True),
    }
    mean_s_r = load_mean_transferability(run_dir)
    if mean_s_r:
        report["transferability_vs_drift"] = correlate_with_transferability(mean_drift, mean_s_r)
    (run_dir / "drift_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-relation representation drift analysis.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--graphs-dir", default="data/graphs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-graphs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    report = run(
        Path(args.run_dir), Path(args.graphs_dir), device, args.batch_size, args.max_graphs,
        seed=args.seed,
    )
    mean = report["mean_over_boundaries"]
    print(f"[drift] {report['run']}, mean over {len(report['boundaries'])} task boundaries")
    for relation in report["relation_drift_ranking"]:
        print(f"   {relation:>14}: {mean['drift_' + relation]:.4f}")
    print(f"   {'fused':>14}: {mean.get('drift_fused', float('nan')):.4f}")
    print(
        f"   prediction agreement {mean.get('prediction_agreement', float('nan')):.4f}, "
        f"logit KL {mean.get('logit_kl', float('nan')):.4f}"
    )
    correlation = report.get("transferability_vs_drift")
    if correlation and correlation.get("n", 0) >= 3:
        print(
            f"[drift] transferability vs drift: Pearson r = {correlation['pearson_r']:.3f}, "
            f"Spearman rho = {correlation['spearman_rho']:.3f} (n = {correlation['n']} relations)"
        )


if __name__ == "__main__":
    main()
