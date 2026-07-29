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

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.cl.ewc import FLOW_RELATIONS, OnlineEWCManager
from trench_ids.cl.importance import ImportanceMLP
from trench_ids.cl.memory_bank import RelationMeanAccumulator, merge_into_bank, save_memory_bank
from trench_ids.cl.transferability import aggregate_transferability_scores, estimate_transferability
from trench_ids.labels import NUM_TASKS, canonical_classes
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


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
    model: RelationSpecificHeteroGNN,
    classifier: torch.nn.Linear,
    config: dict,
) -> None:
    """Saves an evaluation-only checkpoint -- no optimizer state, since
    these are inference/evaluation artifacts, not resumable training
    snapshots (design doc: "Optimizer state is intentionally omitted")."""
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


def evaluate_accuracy(
    model: RelationSpecificHeteroGNN,
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
) -> tuple[float, dict[str, float]]:
    """Runs one task's full warm-up + full-loss training (design §5, steps
    1-5). Returns ``(final_epoch_mean_loss, final_w_r)`` -- ``final_w_r`` is
    logged by the caller for reproducibility (design §7)."""
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
    temp_means = compute_task_memory_means(model, graphs, device, batch_size, label_names)

    # Step 3-4: transferability against the bank as of task t-1 -> S_r -> w_r.
    transferability = estimate_transferability(temp_means, bank)
    s_r = aggregate_transferability_scores(transferability, FLOW_RELATIONS)
    s_r = {relation: value.to(device) for relation, value in s_r.items()}

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
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            logits = classifier(output.fused["flow"])
            w_r = _compute_w_r()
            cls_loss = F.cross_entropy(logits, batch["flow"].y)
            breakdown = ewc_manager.loss_breakdown(model, classifier, w_r)
            loss = cls_loss + breakdown["total_weighted"]
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
        if epoch_log_path is not None and total_flows:
            record = {"epoch": epoch, **{k: v / total_flows for k, v in component_sums.items()}}
            record["loss_total"] = last_epoch_loss
            with epoch_log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    final_w_r = {relation: value.item() for relation, value in _compute_w_r().items()}
    return last_epoch_loss, final_w_r


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
    torch.manual_seed(cfg.train.seed)
    print(f"[train] device = {device}")

    graphs_dir = Path(cfg.paths.graphs_dir)
    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(graphs_dir / "vocab.json")
    protocol_vocab_size = len(vocab["PROTOCOL"])
    service_vocab_size = len(vocab["L7_PROTO"])
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
    trainable_params = (
        list(model.parameters()) + list(classifier.parameters()) + list(importance_mlp.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.optim.learning_rate)

    memory_bank: dict[str, dict[str, torch.Tensor]] = {}
    forgetting_matrix: dict[int, dict[int, float]] = {}
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
        final_loss, final_w_r = train_one_task(
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
        )
        print(f"[train] task {task}: final epoch mean loss = {final_loss:.4f}")
        (out_dir / f"importance_weights_task_{task}.json").write_text(
            json.dumps(final_w_r, indent=2)
        )

        # Step 8: final prototypes (post full-loss weights) -- discards the
        # warm-up's temporary prototypes, which only existed to drive this
        # task's transferability/importance-weighting (design §5, step 8).
        final_means = compute_task_memory_means(
            model, train_graphs, device, cfg.train.batch_size, label_names
        )
        if cfg.replay.enabled:
            n_sample = min(cfg.replay.buffer_size_per_task, len(train_graphs))
            replay_buffer.extend(random.sample(train_graphs, k=n_sample))
        transferability = estimate_transferability(final_means, memory_bank)
        (out_dir / f"transferability_task_{task}.json").write_text(
            json.dumps(transferability, indent=2)
        )
        memory_bank = merge_into_bank(memory_bank, final_means)

        # Step 6-7: Fisher/theta* update for the next task's EWC penalty.
        eval_loader = DataLoader(train_graphs, batch_size=cfg.train.batch_size)
        ewc_manager.update_all(model, classifier, eval_loader, device)

        def _accuracy_for(evaluated_task: int) -> float:
            test_graphs = load_split(graphs_dir, evaluated_task, "test")
            return evaluate_accuracy(model, classifier, test_graphs, device, cfg.train.batch_size)

        forgetting_matrix[task] = forgetting_row(_accuracy_for, task)
        for evaluated_task, acc in forgetting_matrix[task].items():
            print(f"[eval] after task {task}, task {evaluated_task} test accuracy = {acc:.4f}")

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
                "protocol_vocab_size": protocol_vocab_size,
                "service_vocab_size": service_vocab_size,
                "label_names": label_names,
                "replay_enabled": cfg.replay.enabled,
                "replay_buffer_size_per_task": cfg.replay.buffer_size_per_task,
                "replay_fraction": cfg.replay.replay_fraction,
            },
        )

    save_memory_bank(memory_bank, out_dir / "memory_bank.pt")
    (out_dir / "forgetting_matrix.json").write_text(json.dumps(forgetting_matrix, indent=2))

    summary = {
        "average_forgetting": average_forgetting(forgetting_matrix),
        "final_average_accuracy": final_average_accuracy(forgetting_matrix),
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
        "seed": cfg.train.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote memory_bank.pt, forgetting_matrix.json, and summary.json to {out_dir}")


if __name__ == "__main__":
    main()
