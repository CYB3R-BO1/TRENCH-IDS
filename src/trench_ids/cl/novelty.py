"""Novel-attack detection from the relation-specific memory bank.

The Step 10b evaluation established a real limitation: shown attack traffic
from five classes that appear in no task (Backdoor, MITM, Ransomware, Web
Attacks, Theft), the model maps every one of them onto a known class *with
high confidence* -- median softmax ≥ 0.89. Read as a deployment result that
is bad news: the classifier never says "I have not seen this before".

That evaluation only ever asked the **classifier head**, though. This module
asks the **memory bank** instead -- the per-class, per-relation prototype
store the method already maintains for transferability estimation -- and
turns it into a novelty score:

    novelty(v) = 1 - Σ_r w_r · max_c cos( h_r(v), μ_(c,r) )

i.e. how far a flow's relation-specific embeddings sit from the nearest
known class prototype, in each relation's own subspace, combined with the
same transferability-derived weights ``w_r`` the distillation penalty uses.

Why this can work where confidence cannot: a softmax over a fixed label
space is normalised *across known classes only*, so it measures which known
class fits best, never whether any of them fit at all -- an input far from
every class can still produce a peaked distribution. Distance to the nearest
prototype is not normalised that way and has no such blind spot. This is the
standard argument behind distance-based OOD detection (Lee et al., NeurIPS
2018, Mahalanobis; Sun et al., ICML 2022, KNN-OOD); what is specific here is
that the prototypes are **per relation**, so the score decomposes into which
relation found the input unfamiliar -- and that the store it reads was built
for another purpose entirely and costs nothing extra.

Three baselines are scored alongside it on identical inputs, so any claimed
advantage is measured rather than asserted:

  ``msp``       1 - max softmax probability (Hendrycks & Gimpel, ICLR 2017).
                The one Step 10b implicitly used.
  ``max_logit`` negated max logit -- drops softmax's normalisation.
  ``energy``    negated logsumexp of the logits (Liu et al., NeurIPS 2020).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.cl.device import resolve_device
from trench_ids.cl.ewc import FLOW_RELATIONS
from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.memory_bank import RelationMeanAccumulator, load_memory_bank
from trench_ids.cl.train import load_split, sample_graphs
from trench_ids.labels import BENIGN, NUM_TASKS, attack_classes_for_task, canonical_classes

SCORE_NAMES = ("prototype", "msp", "max_logit", "energy")


def build_prototypes(
    bank: dict[str, dict[str, torch.Tensor]],
    relations: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """``{relation: [num_classes, hidden]}`` of L2-normalised prototypes.

    Classes missing a given relation are skipped for that relation only --
    the bank is built incrementally per task and a relation with no stored
    mean for a class carries no information about it, which is different
    from carrying a zero vector (a zero row would score cosine 0 and quietly
    act as a mid-range match).
    """
    prototypes: dict[str, list[torch.Tensor]] = {relation: [] for relation in relations}
    for _class_name, per_relation in sorted(bank.items()):
        for relation in relations:
            vector = per_relation.get(relation)
            if vector is not None:
                prototypes[relation].append(vector.to(device))
    return {
        relation: F.normalize(torch.stack(vectors), dim=-1)
        for relation, vectors in prototypes.items()
        if vectors
    }


@torch.no_grad()
def compute_benign_prototype_means(
    model: torch.nn.Module,
    graphs: list[HeteroData],
    device: torch.device,
    batch_size: int,
    label_names: list[str],
) -> dict[str, torch.Tensor]:
    """Mean per-relation Flow embedding over Benign flows in ``graphs``.

    ``memory_bank.RelationMeanAccumulator`` deliberately excludes Benign --
    it is an attack-type memory for continual-learning transferability, not
    a general class memory, and that design is correct for training. But
    this module's "known" evaluation set is ordinary test traffic, which is
    mostly-but-not-only attacks (the realised attack:benign ratio is ~3:1),
    and scoring a Benign flow against zero Benign prototype means
    ``prototype_novelty`` can only ever measure its distance to the nearest
    *attack* prototype -- systematically reading real Benign traffic as
    novel, a confound specific to the prototype detector, since the
    msp/max_logit/energy baselines all have a real Benign output unit to
    score against. Computing a local Benign prototype here -- rather than
    changing what the bank stores for training -- keeps the CL mechanism
    untouched and gives the prototype detector the same fair shot at Benign
    the baselines already have.
    """
    model.eval()
    benign_idx = label_names.index(BENIGN)
    sums: dict[str, torch.Tensor] = {}
    count = 0
    for batch in DataLoader(graphs, batch_size=batch_size):
        batch = batch.to(device)
        output = model(batch)
        mask = batch["flow"].y == benign_idx
        n = int(mask.sum().item())
        if n == 0:
            continue
        count += n
        for relation, embed in output.relations["flow"].items():
            contribution = embed[mask].sum(dim=0).detach()
            if relation in sums:
                sums[relation] = sums[relation] + contribution
            else:
                sums[relation] = contribution.clone()
    return {relation: total / count for relation, total in sums.items()} if count else {}


@torch.no_grad()
def recompute_bank_prototypes(
    model: torch.nn.Module,
    graphs_dir: Path,
    device: torch.device,
    batch_size: int,
    max_graphs_per_task: int,
    seed: int = 0,
) -> dict[str, dict[str, torch.Tensor]]:
    """Re-derive every attack class's per-relation means under ``model``.

    The stored bank's prototypes were accumulated at each task's own training
    time, i.e. under encoder states that have since drifted -- while every
    query embedding (and the benign prototype) comes from the final encoder.
    Cosine distances between embeddings from different encoder vintages are
    not comparable within a relation, and that staleness alone can push the
    prototype scorer toward chance. Recomputing all class means under the
    *final* checkpoint puts prototypes and queries in the same embedding
    space; each class belongs to exactly one task
    (``labels.CANONICAL_TO_TASK``), so streaming that task's train split and
    masking by class index recovers the same means the bank would have
    stored, had it been built here.

    Uses at most ``max_graphs_per_task`` graphs per task (seeded random, not
    a prefix slice -- see ``train.sample_graphs``); pass 0 to use a task's
    full train split.
    """
    model.eval()
    label_names = canonical_classes()
    accumulator = RelationMeanAccumulator()
    for task in range(1, NUM_TASKS + 1):
        classes = attack_classes_for_task(task)
        if not classes:
            continue
        graphs = sample_graphs(
            load_split(graphs_dir, task, "train"),
            max_graphs_per_task,
            seed=f"{seed}:prototypes:task_{task}",
        )
        for batch in DataLoader(graphs, batch_size=batch_size):
            batch = batch.to(device)
            output = model(batch)
            accumulator.update(output.relations["flow"], batch["flow"].y, label_names)
    return accumulator.means()


def prototype_novelty(
    relation_embeds: dict[str, torch.Tensor],
    prototypes: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Per-flow novelty in [0, 2]; higher means further from everything known.

    ``weights`` defaults to uniform over the relations that have prototypes.
    They are renormalised over exactly those relations, so a relation absent
    from the bank does not silently shrink every score toward zero.
    """
    usable = [r for r in prototypes if r in relation_embeds]
    if not usable:
        raise ValueError("no relation has both an embedding and a prototype")
    if weights is None:
        weights = {relation: 1.0 for relation in usable}
    total_weight = sum(weights.get(relation, 0.0) for relation in usable)
    if total_weight <= 0:
        raise ValueError("relation weights sum to zero over the usable relations")

    similarity = torch.zeros(
        next(iter(relation_embeds.values())).shape[0],
        device=next(iter(relation_embeds.values())).device,
    )
    for relation in usable:
        embeds = F.normalize(relation_embeds[relation], dim=-1)
        best = (embeds @ prototypes[relation].T).max(dim=-1).values
        similarity = similarity + (weights.get(relation, 0.0) / total_weight) * best
    return 1.0 - similarity


def logit_scores(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """The three classifier-head baselines, all oriented so higher = more novel."""
    return {
        "msp": 1.0 - torch.softmax(logits, dim=-1).max(dim=-1).values,
        "max_logit": -logits.max(dim=-1).values,
        "energy": -torch.logsumexp(logits, dim=-1),
    }


@torch.no_grad()
def score_graphs(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    graphs: list[HeteroData],
    prototypes: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 8,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Every score in ``SCORE_NAMES``, per flow, over ``graphs``."""
    model.eval()
    classifier.eval()
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in SCORE_NAMES}
    for batch in DataLoader(graphs, batch_size=batch_size):
        batch = batch.to(device)
        output = model(batch)
        logits = classifier(output.fused["flow"])
        collected["prototype"].append(
            prototype_novelty(output.relations["flow"], prototypes, weights).cpu()
        )
        for name, value in logit_scores(logits).items():
            collected[name].append(value.cpu())
    return {name: torch.cat(parts) for name, parts in collected.items() if parts}


def detection_metrics(
    known: dict[str, torch.Tensor], novel: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    """AUROC / average precision / FPR-at-95%-TPR for every score.

    FPR@95 is reported because it is the number that decides whether a
    detector is deployable: at the threshold that catches 95% of novel
    attacks, what share of ordinary known traffic gets flagged. AUROC alone
    can look respectable while that figure is unusable.
    """
    results: dict[str, dict[str, float]] = {}
    for name in known:
        if name not in novel:
            continue
        scores = torch.cat([known[name], novel[name]]).numpy()
        labels = torch.cat(
            [torch.zeros(known[name].numel()), torch.ones(novel[name].numel())]
        ).numpy()
        order = novel[name].numpy()
        threshold = float(torch.quantile(novel[name].float(), 0.05))
        results[name] = {
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "fpr_at_95_tpr": float((known[name].numpy() >= threshold).mean()),
            "mean_known": float(known[name].mean()),
            "mean_novel": float(order.mean()),
        }
    return results


def run(
    run_dir: Path,
    graphs_dir: Path,
    unseen_graph_dir: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 8,
    checkpoint_task: int = NUM_TASKS,
    max_graphs_per_source: int = 200,
    prototype_source: str = "recomputed",
    seed: int = 0,
) -> dict[str, Any]:
    """Score known test traffic against genuinely unseen attack classes.

    ``prototype_source``: ``"recomputed"`` (default) re-derives every attack
    class's per-relation means under the final checkpoint, so prototypes and
    query embeddings come from the same encoder state; ``"bank"`` reads the
    stored ``memory_bank.pt`` as-is -- those means were computed at each
    task's own training time under a since-drifted encoder, which is a
    confound for the prototype scorer (kept available only so the
    stale-prototype variant remains reproducible).
    """
    if prototype_source not in ("recomputed", "bank"):
        raise ValueError(
            f"prototype_source must be 'recomputed' or 'bank', got {prototype_source!r}"
        )
    model, classifier, checkpoint = load_checkpoint(
        run_dir / f"checkpoint_task_{checkpoint_task}.pt", graphs_dir, device
    )
    # Everything below -- the memory bank, the prototype scorer's per-relation
    # embeddings -- needs a RelationSpecificHeteroGNN checkpoint.
    # train_flat.py never writes memory_bank.pt (its own docstring: "No EWC,
    # no memory bank"), so pointing this at a flat run would otherwise fail
    # with a bare FileNotFoundError from load_memory_bank below -- true, but
    # not the actual reason, and not obviously so to someone re-running this
    # against the wrong run directory.
    model_type = checkpoint["config"].get("model_type", "rhgnn")
    if model_type != "rhgnn":
        raise ValueError(
            f"novelty detection needs a RelationSpecificHeteroGNN checkpoint (a memory "
            f"bank and per-relation embeddings), but {run_dir} has model_type={model_type!r} "
            "-- train_flat.py runs have neither. Point --run-dir at a GNN run instead."
        )
    bank = load_memory_bank(run_dir / "memory_bank.pt")

    # The bank has no Benign prototype by design (see
    # compute_benign_prototype_means), but "known" below includes Benign
    # test flows. Built from TRAIN splits, like every other prototype in the
    # bank, so it isn't fit on the same flows it's later scored against.
    label_names = canonical_classes()
    benign_source_graphs: list[HeteroData] = []
    for task in range(1, NUM_TASKS + 1):
        benign_source_graphs.extend(
            sample_graphs(
                load_split(graphs_dir, task, "train"),
                max_graphs_per_source,
                seed=f"{seed}:benign:task_{task}",
            )
        )
    benign_means = compute_benign_prototype_means(
        model, benign_source_graphs, device, batch_size, label_names
    )
    if prototype_source == "recomputed":
        fresh = recompute_bank_prototypes(
            model,
            graphs_dir,
            device,
            batch_size,
            max_graphs_per_source,
            seed=seed,
        )
        bank_for_prototypes = {**fresh, BENIGN: benign_means} if benign_means else fresh
    else:
        bank_for_prototypes = {**bank, BENIGN: benign_means} if benign_means else bank
    prototypes = build_prototypes(bank_for_prototypes, FLOW_RELATIONS, device)

    weights_path = run_dir / f"distillation_task_{checkpoint_task}.json"
    weights = None
    if weights_path.exists():
        weights = json.loads(weights_path.read_text()).get("weights")

    known_graphs: list[HeteroData] = []
    for task in range(1, NUM_TASKS + 1):
        known_graphs.extend(
            sample_graphs(
                load_split(graphs_dir, task, "test"),
                max_graphs_per_source,
                seed=f"{seed}:known:task_{task}",
            )
        )
    known = score_graphs(
        model, classifier, known_graphs, prototypes, device, batch_size, weights
    )

    per_source: dict[str, Any] = {}
    novel_parts: dict[str, list[torch.Tensor]] = {name: [] for name in SCORE_NAMES}
    for path in sorted(unseen_graph_dir.glob("*.pt")):
        graphs = sample_graphs(
            torch.load(path, weights_only=False),
            max_graphs_per_source,
            seed=f"{seed}:novel:{path.stem}",
        )
        scores = score_graphs(
            model, classifier, graphs, prototypes, device, batch_size, weights
        )
        per_source[path.stem] = detection_metrics(known, scores)
        for name, value in scores.items():
            novel_parts[name].append(value)

    novel = {name: torch.cat(parts) for name, parts in novel_parts.items() if parts}
    report = {
        "checkpoint_task": checkpoint_task,
        "prototype_source": prototype_source,
        "sample_seed": seed,
        "relation_weights": weights,
        "num_known_flows": int(next(iter(known.values())).numel()),
        "num_novel_flows": int(next(iter(novel.values())).numel()),
        "pooled": detection_metrics(known, novel),
        "per_source": per_source,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "novelty_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Novel-attack detection from the relation-specific memory bank."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--graphs-dir", default="data/graphs")
    parser.add_argument("--unseen-graph-dir", default="data/unseen/graphs")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-graphs-per-source", type=int, default=200)
    parser.add_argument(
        "--prototype-source",
        choices=("recomputed", "bank"),
        default="recomputed",
        help="recomputed: re-derive attack prototypes under the final checkpoint "
        "(default; the stored bank's means come from drifted earlier encoders). "
        "bank: read memory_bank.pt as-is (stale-vintage confound, kept for reproducibility).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    report = run(
        run_dir=run_dir,
        graphs_dir=Path(args.graphs_dir),
        unseen_graph_dir=Path(args.unseen_graph_dir),
        out_dir=Path(args.out_dir) if args.out_dir else run_dir / "novelty",
        device=device,
        batch_size=args.batch_size,
        max_graphs_per_source=args.max_graphs_per_source,
        prototype_source=args.prototype_source,
        seed=args.seed,
    )
    for name, metrics in report["pooled"].items():
        print(
            f"[novelty] {name:>10}: AUROC {metrics['auroc']:.4f}  "
            f"AP {metrics['average_precision']:.4f}  "
            f"FPR@95TPR {metrics['fpr_at_95_tpr']:.4f}"
        )


if __name__ == "__main__":
    main()
