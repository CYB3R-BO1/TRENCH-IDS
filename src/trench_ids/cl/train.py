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
from collections.abc import Callable
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.cl.memory_bank import RelationMeanAccumulator, merge_into_bank, save_memory_bank
from trench_ids.labels import NUM_TASKS, canonical_classes
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


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


def train_one_task(
    model: RelationSpecificHeteroGNN,
    classifier: torch.nn.Linear,
    graphs: list[HeteroData],
    device: torch.device,
    batch_size: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Plain sequential fine-tuning on one task's train split. Returns the
    final epoch's mean per-flow cross-entropy loss."""
    model.train()
    classifier.train()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    last_epoch_loss = 0.0
    for _epoch in range(epochs):
        total_loss = 0.0
        total_flows = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            logits = classifier(output.fused["flow"])
            loss = F.cross_entropy(logits, batch["flow"].y)
            loss.backward()
            optimizer.step()
            n = batch["flow"].y.numel()
            total_loss += loss.item() * n
            total_flows += n
        last_epoch_loss = total_loss / total_flows if total_flows else 0.0
    return last_epoch_loss


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
        port_buckets=cfg.model.port_buckets,
    ).to(device)
    classifier = torch.nn.Linear(cfg.model.hidden_dim, len(label_names)).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()), lr=cfg.optim.learning_rate
    )

    memory_bank: dict[str, dict[str, torch.Tensor]] = {}
    forgetting_matrix: dict[int, dict[int, float]] = {}

    for task in range(1, NUM_TASKS + 1):
        train_graphs = sample_graphs if task == 1 else load_split(graphs_dir, task, "train")

        final_loss = train_one_task(
            model,
            classifier,
            train_graphs,
            device,
            cfg.train.batch_size,
            cfg.train.epochs_per_task,
            optimizer,
        )
        print(f"[train] task {task}: final epoch mean loss = {final_loss:.4f}")

        task_means = compute_task_memory_means(
            model, train_graphs, device, cfg.train.batch_size, label_names
        )
        memory_bank = merge_into_bank(memory_bank, task_means)

        def _accuracy_for(evaluated_task: int) -> float:
            test_graphs = load_split(graphs_dir, evaluated_task, "test")
            return evaluate_accuracy(model, classifier, test_graphs, device, cfg.train.batch_size)

        forgetting_matrix[task] = forgetting_row(_accuracy_for, task)
        for evaluated_task, acc in forgetting_matrix[task].items():
            print(f"[eval] after task {task}, task {evaluated_task} test accuracy = {acc:.4f}")

    save_memory_bank(memory_bank, out_dir / "memory_bank.pt")
    (out_dir / "forgetting_matrix.json").write_text(json.dumps(forgetting_matrix, indent=2))
    print(f"[done] wrote memory_bank.pt and forgetting_matrix.json to {out_dir}")


if __name__ == "__main__":
    main()
