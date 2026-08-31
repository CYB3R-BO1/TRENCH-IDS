"""Step 4 — classifier training loop + relation-specific memory bank.

Sequential fine-tuning across T1..T6 (``trench_ids.labels.NUM_TASKS``, in
task-number order): one shared classifier head over the full global label
space (``trench_ids.labels.canonical_classes()``), trained on each task's
data in turn -- plain sequential fine-tuning, no replay/EWC (that's step 7,
CLAUDE.md's Open Items, explicitly out of scope here). After each task, the
shared classifier is evaluated on that task's test split *and* every
previously-seen task's test split, building a forgetting matrix --
CLAUDE.md flags this as a required Step 4 deliverable ("picking the winning
benign_ratio ... awaits real continual-learning training/forgetting curves
from Step 4"), not an optional nicety.

Also builds the relation-specific memory bank (CLAUDE.md's proposed
pipeline, step 4): for each task's attack classes (Benign excluded), the
mean of Flow's relation-specific embeddings, computed in a no-grad pass over
that task's train split right after training on it finishes.

Run:  trench-train
  or: python -m trench_ids.cl.train
  or, to override a config value: python -m trench_ids.cl.train train.epochs_per_task=10
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.cl.device import (
    resolve_device,  # re-exported: train_flat/train_joint import it from here
)
from trench_ids.cl.distillation import RelationDistiller, boundary_relation_drift
from trench_ids.cl.ewc import FLOW_RELATIONS, OnlineEWCManager
from trench_ids.cl.importance import ImportanceMLP
from trench_ids.cl.memory_bank import RelationMeanAccumulator, merge_into_bank, save_memory_bank
from trench_ids.cl.replay_selection import select_replay_graphs
from trench_ids.cl.transferability import (
    aggregate_per_class_relation_scores,
    aggregate_transferability_scores,
    estimate_transferability,
)
from trench_ids.labels import (
    BENIGN,
    NUM_TASKS,
    attack_classes_for_task,
    canonical_classes,
)
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab, vocab_fingerprint


def seed_everything(seed: int) -> None:
    """Seed torch, Python's ``random``, and NumPy's global RNG.

    Until 2026-08-07 only ``torch.manual_seed`` was called here, so the
    replay buffer's ``random.sample``/``random.choices`` draws (below) ran
    off OS entropy: two runs of the *same* config produced different replay
    buffers. That made every replay run non-reproducible, and made any
    between-condition difference smaller than that hidden variance
    uninterpretable -- including the port-scheme decision
    (``project-metrics.md`` §24.4) and the transferability-guided-replay
    comparison (§30), both of which turn on gaps of ~0.01 average
    forgetting.

    Runs recorded before this fix cannot be reproduced exactly; re-run them
    before comparing their numbers against anything produced afterwards.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def save_checkpoint(
    path: Path,
    task_id: int,
    epochs_per_task: int,
    warmup_epochs: int,
    seed: int,
    model: torch.nn.Module,
    classifier: torch.nn.Linear,
    config: dict,
) -> None:
    """Saves an evaluation-only checkpoint -- no optimizer state, since
    these are inference/evaluation artifacts, not resumable training
    snapshots (design doc: "Optimizer state is intentionally omitted").

    ``model`` is typed as a plain ``nn.Module`` rather than
    ``RelationSpecificHeteroGNN`` because the non-graph baseline
    (``trench_ids.model.flat.FlatFlowEncoder``) checkpoints through this
    same function. ``config`` should carry a ``model_type`` key so
    ``trench_ids.cl.inference.load_checkpoint`` can rebuild the right class;
    checkpoints written before that key existed are treated as
    ``"rhgnn"``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 1,
            "task_id": task_id,
            "epochs_per_task": epochs_per_task,
            "warmup_epochs": warmup_epochs,
            "random_seed": seed,
            "git_commit": _git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "config": config,
        },
        path,
    )


def load_split(graphs_dir: Path, task: int, split: str) -> list[HeteroData]:
    return torch.load(graphs_dir / f"task_{task}_{split}.pt", weights_only=False)


def sample_graphs(
    graphs: list[HeteroData], max_graphs: int, seed: int | str = 0
) -> list[HeteroData]:
    """Seeded uniform subsample of a split's mini-graphs.

    Since the 2026-08-17 rebuild, mini-graphs are chunked in capture order,
    so position in a split file encodes real time -- ``graphs[:max_graphs]``
    samples one specific early-capture window (and whichever classes run
    there) rather than a random subset. Every analysis that subsamples a
    split should go through this instead of a prefix slice. ``max_graphs <=
    0`` (or a split smaller than the cap) returns the input unchanged, so a
    caller can request "everything" with 0. ``seed`` may be a string so
    callers can derive per-purpose, per-task determinism without colliding.
    """
    if max_graphs <= 0 or len(graphs) <= max_graphs:
        return graphs
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(graphs)), max_graphs))
    return [graphs[i] for i in indices]


def evaluate_accuracy(
    model: torch.nn.Module,
    classifier: torch.nn.Linear,
    graphs: list[HeteroData],
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    loader = DataLoader(graphs, batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            logits = classifier(output.fused["flow"])
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch["flow"].y).sum().item())
            total += batch["flow"].y.numel()
    return correct / total if total else 0.0


def forgetting_row(accuracy_fn: Callable[[int], float], up_to_task: int) -> dict[int, float]:
    """``{evaluated_task: accuracy}`` for every task ``1..up_to_task``.

    Isolated from ``evaluate_accuracy``/real graph loading so the
    bookkeeping shape (evaluate every previously-seen task, not just the
    current one) can be unit-tested with a stub.
    """
    return {
        evaluated_task: accuracy_fn(evaluated_task) for evaluated_task in range(1, up_to_task + 1)
    }


def average_forgetting(forgetting_matrix: dict[int, dict[int, float]]) -> float:
    """Mean, over every task before the final one, of that task's peak
    accuracy (across all evaluations from its own training through the
    final task) minus its accuracy after the final task -- the standard
    forgetting-matrix definition (Lopez-Paz & Ranzato, GEM). 0.0 when only
    one task has been trained (nothing earlier to have forgotten)."""
    final_task = max(forgetting_matrix)
    if final_task == 1:
        return 0.0
    per_task_forgetting = []
    for evaluated_task in range(1, final_task):
        peak = max(
            forgetting_matrix[trained_up_to][evaluated_task]
            for trained_up_to in range(evaluated_task, final_task)
        )
        final_accuracy = forgetting_matrix[final_task][evaluated_task]
        per_task_forgetting.append(peak - final_accuracy)
    return sum(per_task_forgetting) / len(per_task_forgetting)


def backward_transfer(forgetting_matrix: dict[int, dict[int, float]]) -> float:
    """Backward transfer (Lopez-Paz & Ranzato, GEM, NeurIPS 2017):
    ``BWT = mean over i<T of (R[T][i] - R[i][i])`` -- how much learning
    every later task changed performance on task ``i``, relative to the
    moment task ``i`` finished training. Negative means forgetting;
    *positive* means later tasks improved an earlier one.

    Not redundant with ``average_forgetting``, despite both summarising the
    same matrix, but the distinction is *not* "one can go negative and the
    other cannot" -- both can. ``average_forgetting`` takes the peak over
    ``l`` in ``[i, T-1]``, which excludes the final row, so it too comes out
    negative when a task ends above every earlier measurement.

    The real difference is the reference point. Forgetting measures the drop
    from each task's *best observed* accuracy ("how much of our best did we
    lose"); BWT measures the change from ``R[i][i]`` specifically ("are we
    better or worse than when this task was introduced"). They coincide only
    when a task's peak sits on the diagonal, and diverge whenever an earlier
    task kept improving after its own training -- plausible under replay,
    where later tasks' buffers keep rehearsing task ``i``. Reporting both is
    what makes that case legible instead of averaging it away.

    0.0 when only one task has been trained (no earlier task to transfer
    back to), matching ``average_forgetting``'s convention.

    Note that forward transfer (FWT) is *not* derivable from this matrix:
    it needs each task's accuracy before that task was trained, plus a
    random-init reference, and the matrix only stores cells with
    ``evaluated_task <= trained_up_to``.
    """
    final_task = max(forgetting_matrix)
    if final_task == 1:
        return 0.0
    deltas = [
        forgetting_matrix[final_task][evaluated_task]
        - forgetting_matrix[evaluated_task][evaluated_task]
        for evaluated_task in range(1, final_task)
    ]
    return sum(deltas) / len(deltas)


def final_average_accuracy(forgetting_matrix: dict[int, dict[int, float]]) -> float:
    """Mean test accuracy, over every task, as measured right after the
    final task finishes training -- the forgetting matrix's last row."""
    final_task = max(forgetting_matrix)
    accuracies = forgetting_matrix[final_task].values()
    return sum(accuracies) / len(accuracies)


def split_warmup_and_full_loss_epochs(epochs_per_task: int, warmup_epochs: int) -> tuple[int, int]:
    """``(warmup_epochs, full_loss_epochs)`` -- every task runs this same
    split (design §5's uniform per-task flow: no ``if task == 1`` special
    case anywhere, including here)."""
    if warmup_epochs > epochs_per_task:
        raise ValueError(
            f"warmup_epochs ({warmup_epochs}) cannot exceed epochs_per_task ({epochs_per_task})"
        )
    return warmup_epochs, epochs_per_task - warmup_epochs


def build_replay_augmented_set(
    current_task_graphs: list[HeteroData],
    replay_buffer: list[HeteroData],
    replay_fraction: float,
) -> list[HeteroData]:
    """Combined training set for a task, mixing in the accumulated replay
    buffer (design doc §Mechanism/Mixing into training). ``replay_buffer``
    graphs are upsampled with replacement so they make up ``replay_fraction``
    of the returned list; ``current_task_graphs`` appear once each, unchanged
    (no replay for an empty buffer, e.g. task 1)."""
    if not replay_buffer:
        return current_task_graphs
    n_current = len(current_task_graphs)
    n_replay = round(n_current * replay_fraction / (1 - replay_fraction))
    upsampled = random.choices(replay_buffer, k=n_replay)
    return current_task_graphs + upsampled


def train_one_task(
    model: RelationSpecificHeteroGNN,
    classifier: torch.nn.Linear,
    graphs: list[HeteroData],
    device: torch.device,
    batch_size: int,
    warmup_epochs: int,
    full_loss_epochs: int,
    optimizer: torch.optim.Optimizer,
    ewc_manager: OnlineEWCManager,
    importance_mlp: ImportanceMLP,
    label_names: list[str],
    bank: dict[str, dict[str, torch.Tensor]],
    disable_learned_weighting: bool = False,
    w_r_mode: str = "learned",
    epoch_log_path: Path | None = None,
    distiller: RelationDistiller | None = None,
    task_id: int = 0,
    seed: int = 0,
    drift_max_graphs: int = 60,
) -> tuple[float, dict[str, float], dict[str, Any]]:
    """Runs one task's full warm-up + full-loss training (design §5, steps
    1-5). Returns ``(final_epoch_mean_loss, final_w_r, distill_report)``.

    ``final_w_r`` is logged by the caller for reproducibility (design §7).
    ``distiller``, when given, adds the transferability-weighted relation
    distillation penalty of ``trench_ids.cl.distillation`` to the full-loss
    epochs; its weights are set here (not by the caller) because they depend
    on ``S_r``, which is only known after the warm-up pass -- or, for the
    ``drift`` weighting mode, on live per-relation drift ``D_r`` between the
    teacher and the post-warm-up student (measured here on a seeded
    subsample of at most ``drift_max_graphs`` of this task's graphs, since
    before training starts the student *is* the teacher and every D_r is
    trivially zero; after warm-up it is exactly "what this task's adaptation
    is moving"). ``distill_report`` carries that task's realised weights and
    per-relation distances for the run record, and is empty when no
    distiller is in use."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    # Step 1: warm-up, plain classification loss only, every task.
    model.train()
    classifier.train()
    for _epoch in range(warmup_epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            logits = classifier(output.fused["flow"])
            loss = F.cross_entropy(logits, batch["flow"].y)
            loss.backward()
            optimizer.step()

    # Step 2: fresh no-grad pass -> temporary prototypes.
    #
    # Skipped entirely when nothing downstream consumes S_r -- i.e. when EWC
    # is off (all lambdas zero) and no distiller is attached. That is the
    # case for 24 of the 30 runs in the experiment matrix, and this pass
    # costs a full sweep over the task's training set, so running it anyway
    # would spend hours computing a number no loss ever reads. The *final*
    # prototypes (computed by the caller after training) are unaffected --
    # those feed the memory bank and the transferability report, which every
    # run does produce.
    needs_s_r = distiller is not None or ewc_manager.any_lambda_nonzero()
    if needs_s_r:
        temp_means = compute_task_memory_means(model, graphs, device, batch_size, label_names)
        transferability = estimate_transferability(temp_means, bank)
        s_r = aggregate_transferability_scores(transferability, FLOW_RELATIONS)
        s_r = {relation: value.to(device) for relation, value in s_r.items()}
    else:
        s_r = {relation: torch.zeros((), device=device) for relation in FLOW_RELATIONS}

    def _compute_w_r() -> dict[str, torch.Tensor]:
        # Three w_r sources, in priority order:
        # 1. disable_learned_weighting=True (legacy flag, kept for
        #    backward compatibility with existing configs/runs) or
        #    w_r_mode="disabled": fixed at 1.0 for every relation, so
        #    lambda_r=lambda_s=lambda_u applies the same uniform,
        #    unweighted penalty to all 12 EWC groups.
        # 2. w_r_mode="fixed" (2026-07-29 ablation,
        #    docs/superpowers/specs/2026-07-29-transferability-followup-design.md):
        #    w_r = sigmoid(S_r) directly, no ImportanceMLP call -- S_r
        #    already carries no gradient graph (aggregate_transferability_scores
        #    wraps plain Python floats), so this value is a pure constant
        #    from the loss's perspective, structurally incapable of the
        #    w_r-collapses-to-zero pathology diagnosed in importance.py.
        # 3. w_r_mode="learned" (default, original Step 7 behavior):
        #    w_r = ImportanceMLP(S_r).
        if disable_learned_weighting or w_r_mode == "disabled":
            return {relation: torch.ones((), device=device) for relation in s_r}
        if w_r_mode == "fixed":
            return {relation: torch.sigmoid(value) for relation, value in s_r.items()}
        return importance_mlp(s_r)

    distill_report: dict[str, Any] = {}
    if distiller is not None:
        if distiller.weighting == "drift" and distiller.active:
            drift_sample = sample_graphs(
                graphs, drift_max_graphs, seed=f"{seed}:distill-drift:task_{task_id}"
            )
            d_r = boundary_relation_drift(distiller.teacher, model, drift_sample, device)
            distill_report["d_r"] = d_r
            distill_report["weights"] = distiller.set_weights(d_r)
        else:
            distill_report["weights"] = distiller.set_weights(s_r)
        distill_report["s_r"] = {relation: float(v) for relation, v in s_r.items()}
        distill_report["active"] = distiller.active

    # Step 5: full-loss epochs. w_r is recomputed fresh from the cached S_r
    # every batch (not cached itself) so the MLP trains via backprop without
    # retaining a graph across batches (design §3, revised after user review).
    model.train()
    classifier.train()
    last_epoch_loss = 0.0
    for epoch in range(full_loss_epochs):
        total_loss = 0.0
        total_flows = 0
        component_sums = {
            "l_cls": 0.0, "shared_raw": 0.0, "flow_raw": 0.0,
            "other_raw": 0.0, "total_weighted": 0.0,
        }
        distill_sum = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            logits = classifier(output.fused["flow"])
            w_r = _compute_w_r()
            cls_loss = F.cross_entropy(logits, batch["flow"].y)
            breakdown = ewc_manager.loss_breakdown(model, classifier, w_r)
            loss = cls_loss + breakdown["total_weighted"]
            if distiller is not None and distiller.active:
                distill_term = distiller.penalty(
                    batch, output.relations["flow"], classifier=classifier
                )
                loss = loss + distill_term
                distill_sum += distill_term.item() * batch["flow"].y.numel()
            loss.backward()
            optimizer.step()
            n = batch["flow"].y.numel()
            total_loss += loss.item() * n
            total_flows += n
            if epoch_log_path is not None:
                component_sums["l_cls"] += cls_loss.item() * n
                for key in ("shared_raw", "flow_raw", "other_raw", "total_weighted"):
                    component_sums[key] += breakdown[key].item() * n
        last_epoch_loss = total_loss / total_flows if total_flows else 0.0
        if total_flows:
            distill_report.setdefault("mean_penalty_per_epoch", []).append(
                distill_sum / total_flows
            )
        if epoch_log_path is not None and total_flows:
            record = {"epoch": epoch, **{k: v / total_flows for k, v in component_sums.items()}}
            record["loss_total"] = last_epoch_loss
            record["distill"] = distill_sum / total_flows
            with epoch_log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    final_w_r = {relation: value.item() for relation, value in _compute_w_r().items()}
    return last_epoch_loss, final_w_r, distill_report


def compute_task_memory_means(
    model: RelationSpecificHeteroGNN,
    graphs: list[HeteroData],
    device: torch.device,
    batch_size: int,
    label_names: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    model.eval()
    accumulator = RelationMeanAccumulator()
    loader = DataLoader(graphs, batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            accumulator.update(output.relations["flow"], batch["flow"].y, label_names)
    return accumulator.means()


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.train.device)
    seed_everything(cfg.train.seed)
    print(f"[train] device = {device}")

    graphs_dir = Path(cfg.paths.graphs_dir)
    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(graphs_dir / "vocab.json")
    protocol_vocab_size = len(vocab["PROTOCOL"])
    service_vocab_size = len(vocab["L7_PROTO"])
    vocab_fingerprint_value = vocab_fingerprint(vocab)
    label_names = canonical_classes()

    sample_graphs = load_split(graphs_dir, 1, "train")
    model = RelationSpecificHeteroGNN.from_graph(
        sample_graphs[0],
        hidden_dim=cfg.model.hidden_dim,
        protocol_vocab_size=protocol_vocab_size,
        service_vocab_size=service_vocab_size,
        num_layers=cfg.model.num_layers,
        attn_dim=cfg.model.attn_dim,
        port_tail_buckets=cfg.model.port_tail_buckets,
        fusion=cfg.model.get("fusion", "attention"),
        use_residual=cfg.model.get("use_residual", False),
    ).to(device)
    classifier = torch.nn.Linear(cfg.model.hidden_dim, len(label_names)).to(device)

    ewc_manager = OnlineEWCManager(
        model,
        classifier,
        FLOW_RELATIONS,
        gamma=cfg.ewc.gamma,
        lambda_r=cfg.ewc.lambda_r,
        lambda_s=cfg.ewc.lambda_s,
        lambda_u=cfg.ewc.lambda_u,
    )
    importance_mlp = ImportanceMLP().to(device)
    distiller = (
        RelationDistiller(
            FLOW_RELATIONS,
            lambda_d=cfg.distill.lambda_d,
            weighting=cfg.distill.weighting,
            temperature=cfg.distill.temperature,
            objective=cfg.distill.objective,
            logit_temperature=cfg.distill.logit_temperature,
        )
        if cfg.distill.enabled
        else None
    )
    trainable_params = (
        list(model.parameters()) + list(classifier.parameters()) + list(importance_mlp.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.optim.learning_rate)

    memory_bank: dict[str, dict[str, torch.Tensor]] = {}
    forgetting_matrix: dict[int, dict[int, float]] = {}
    val_matrix: dict[int, dict[int, float]] = {}
    replay_buffer: list[HeteroData] = []
    run_start = time.monotonic()

    for task in range(1, NUM_TASKS + 1):
        train_graphs = sample_graphs if task == 1 else load_split(graphs_dir, task, "train")
        warmup_epochs, full_loss_epochs = split_warmup_and_full_loss_epochs(
            cfg.train.epochs_per_task, cfg.train.warmup_epochs
        )

        training_set = (
            build_replay_augmented_set(train_graphs, replay_buffer, cfg.replay.replay_fraction)
            if cfg.replay.enabled
            else train_graphs
        )

        epoch_log_path = (
            out_dir / f"loss_components_task_{task}.jsonl"
            if cfg.ewc.log_loss_components
            else None
        )

        if distiller is not None:
            # The teacher is the encoder+head as of task t-1, so the only
            # classes it can speak about are Benign (present in every task)
            # plus the attack classes of tasks 1..t-1. Telling the distiller
            # which those are keeps its logit term off the classes this task
            # is here to learn -- see RelationDistiller.set_old_classes.
            seen = {BENIGN}
            for earlier in range(1, task):
                seen.update(attack_classes_for_task(earlier))
            distiller.set_old_classes(
                [i for i, name in enumerate(label_names) if name in seen]
            )
        final_loss, final_w_r, distill_report = train_one_task(
            model,
            classifier,
            training_set,
            device,
            cfg.train.batch_size,
            warmup_epochs,
            full_loss_epochs,
            optimizer,
            ewc_manager,
            importance_mlp,
            label_names,
            memory_bank,
            disable_learned_weighting=cfg.ewc.disable_learned_weighting,
            w_r_mode=cfg.ewc.w_r_mode,
            epoch_log_path=epoch_log_path,
            distiller=distiller,
            task_id=task,
            seed=cfg.train.seed,
            drift_max_graphs=cfg.distill.get("drift_max_graphs", 60),
        )
        print(f"[train] task {task}: final epoch mean loss = {final_loss:.4f}")
        # Mirrors train_one_task's own needs_s_r check: when neither EWC nor
        # the distiller is attached, S_r is never computed there and
        # final_w_r is derived from an all-zero placeholder, not a real
        # transferability signal (see train_one_task's Step 2 comment). That
        # is the case for most of the experiment matrix's runs (plain replay
        # included), so the file has to say so -- an unmarked
        # importance_weights_task_*.json reading identically to a real one is
        # exactly the kind of silent placeholder that produced the earlier,
        # genuine w_r-collapse finding (CLAUDE.md, 2026-07-21) from real data.
        s_r_is_real = distiller is not None or ewc_manager.any_lambda_nonzero()
        importance_record: dict[str, Any] = {"weights": final_w_r, "s_r_is_real": s_r_is_real}
        if not s_r_is_real:
            importance_record["note"] = (
                "S_r was not computed this task (no EWC lambda nonzero, no "
                "distiller attached), so these weights were derived from an "
                "all-zero placeholder S_r, not a real transferability signal. "
                "Do not read them as evidence about relation importance."
            )
        (out_dir / f"importance_weights_task_{task}.json").write_text(
            json.dumps(importance_record, indent=2)
        )
        if distiller is not None:
            print(f"[distill] task {task}: w_r = {distill_report.get('weights')}")
            (out_dir / f"distillation_task_{task}.json").write_text(
                json.dumps(distill_report, indent=2)
            )

        # Step 8: final prototypes (post full-loss weights) -- discards the
        # warm-up's temporary prototypes, which only existed to drive this
        # task's transferability/importance-weighting (design §5, step 8).
        final_means = compute_task_memory_means(
            model, train_graphs, device, cfg.train.batch_size, label_names
        )
        transferability = estimate_transferability(final_means, memory_bank)
        (out_dir / f"transferability_task_{task}.json").write_text(
            json.dumps(transferability, indent=2)
        )

        if cfg.replay.enabled:
            n_sample = min(cfg.replay.buffer_size_per_task, len(train_graphs))
            if cfg.replay.selection == "uniform" or task == 1:
                # Exact passthrough -- task 1 always falls back here too,
                # since memory_bank is empty and there is no transferability
                # signal yet (design §Experiment 2 step 3).
                replay_buffer.extend(random.sample(train_graphs, k=n_sample))
            else:
                per_class_relation_scores = aggregate_per_class_relation_scores(
                    transferability, FLOW_RELATIONS
                )
                rng = np.random.default_rng(cfg.train.seed + task)
                selected, selected_scores = select_replay_graphs(
                    train_graphs,
                    per_class_relation_scores,
                    label_names,
                    n_sample,
                    mode=cfg.replay.selection,
                    benign_name=BENIGN,
                    rng=rng,
                )
                replay_buffer.extend(selected)
                class_counts: dict[str, int] = {}
                for g in selected:
                    for class_idx in g["flow"].y.tolist():
                        name = label_names[class_idx]
                        class_counts[name] = class_counts.get(name, 0) + 1
                total_flows = sum(class_counts.values())
                composition = {
                    "class_counts": class_counts,
                    "class_fractions": (
                        {k: v / total_flows for k, v in class_counts.items()}
                        if total_flows
                        else {}
                    ),
                    "mean_enrichment": sum(selected_scores) / len(selected_scores)
                    if selected_scores
                    else 0.0,
                    "per_class_relation_scores": per_class_relation_scores,
                }
                (out_dir / f"replay_buffer_composition_task_{task}.json").write_text(
                    json.dumps(composition, indent=2)
                )

        memory_bank = merge_into_bank(memory_bank, final_means)

        # The next task's distillation target is this task's finished
        # encoder -- snapshotted after the memory bank update so the teacher
        # and the bank describe the same parameter state.
        if distiller is not None:
            distiller.snapshot(model, classifier)

        # Step 6-7: Fisher/theta* update for the next task's EWC penalty.
        # Only the EWC arms read it -- see OnlineEWCManager.any_lambda_nonzero.
        if ewc_manager.any_lambda_nonzero():
            eval_loader = DataLoader(train_graphs, batch_size=cfg.train.batch_size)
            ewc_manager.update_all(model, classifier, eval_loader, device)

        def _accuracy_for(evaluated_task: int, split: str) -> float:
            split_graphs = load_split(graphs_dir, evaluated_task, split)
            return evaluate_accuracy(model, classifier, split_graphs, device, cfg.train.batch_size)

        forgetting_matrix[task] = forgetting_row(lambda t: _accuracy_for(t, "test"), task)
        for evaluated_task, acc in forgetting_matrix[task].items():
            print(f"[eval] after task {task}, task {evaluated_task} test accuracy = {acc:.4f}")

        val_matrix[task] = forgetting_row(lambda t: _accuracy_for(t, "val"), task)
        for evaluated_task, acc in val_matrix[task].items():
            print(f"[eval] after task {task}, task {evaluated_task} val accuracy = {acc:.4f}")

        # Save replay buffer and optimizer state for intervention experiments
        (out_dir / f"replay_buffer_task_{task}.pt").write_text("")
        torch.save(replay_buffer, out_dir / f"replay_buffer_task_{task}.pt")
        torch.save(optimizer.state_dict(), out_dir / f"optimizer_task_{task}.pt")
        
        save_checkpoint(
            out_dir / f"checkpoint_task_{task}.pt",
            task_id=task,
            epochs_per_task=cfg.train.epochs_per_task,
            warmup_epochs=cfg.train.warmup_epochs,
            seed=cfg.train.seed,
            model=model,
            classifier=classifier,
            config={
                "hidden_dim": cfg.model.hidden_dim,
                "num_layers": cfg.model.num_layers,
                "attn_dim": cfg.model.attn_dim,
                "port_tail_buckets": cfg.model.port_tail_buckets,
                # Recorded because the fusion mode changes the parameter set,
                # not just its values: a concat-fusion checkpoint cannot be
                # loaded into an attention-fusion model at all. Absent in
                # checkpoints written before this key existed, which were all
                # attention-fusion runs -- hence build_model's default.
                "fusion": cfg.model.get("fusion", "attention"),
                "use_residual": cfg.model.get("use_residual", False),
                "protocol_vocab_size": protocol_vocab_size,
                "service_vocab_size": service_vocab_size,
                # Re-checked by inference.load_checkpoint against whatever
                # vocab.json a later run loads -- see vocab.vocab_fingerprint.
                "vocab_fingerprint": vocab_fingerprint_value,
                "label_names": label_names,
                "replay_enabled": cfg.replay.enabled,
                "replay_buffer_size_per_task": cfg.replay.buffer_size_per_task,
                "replay_fraction": cfg.replay.replay_fraction,
                "replay_selection": cfg.replay.selection,
            },
        )

    # Save final replay buffer and optimizer state for intervention experiments
    torch.save(replay_buffer, out_dir / "replay_buffer.pt")
    torch.save(optimizer.state_dict(), out_dir / "optimizer_final.pt")
    
    save_memory_bank(memory_bank, out_dir / "memory_bank.pt")
    (out_dir / "forgetting_matrix.json").write_text(json.dumps(forgetting_matrix, indent=2))
    (out_dir / "val_matrix.json").write_text(json.dumps(val_matrix, indent=2))

    summary = {
        "average_forgetting": average_forgetting(forgetting_matrix),
        "backward_transfer": backward_transfer(forgetting_matrix),
        "final_average_accuracy": final_average_accuracy(forgetting_matrix),
        "final_average_val_accuracy": final_average_accuracy(val_matrix),
        "runtime_seconds": time.monotonic() - run_start,
        "lambda_r": cfg.ewc.lambda_r,
        "lambda_s": cfg.ewc.lambda_s,
        "lambda_u": cfg.ewc.lambda_u,
        "gamma": cfg.ewc.gamma,
        "disable_learned_weighting": cfg.ewc.disable_learned_weighting,
        "w_r_mode": cfg.ewc.w_r_mode,
        "replay_enabled": cfg.replay.enabled,
        "replay_buffer_size_per_task": cfg.replay.buffer_size_per_task,
        "replay_fraction": cfg.replay.replay_fraction,
        "replay_selection": cfg.replay.selection,
        "distill_enabled": cfg.distill.enabled,
        "distill_lambda_d": cfg.distill.lambda_d,
        "distill_weighting": cfg.distill.weighting,
        "distill_temperature": cfg.distill.temperature,
        "distill_objective": cfg.distill.objective,
        "distill_logit_temperature": cfg.distill.logit_temperature,
        "seed": cfg.train.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote memory_bank.pt, forgetting_matrix.json, and summary.json to {out_dir}")


if __name__ == "__main__":
    main()
