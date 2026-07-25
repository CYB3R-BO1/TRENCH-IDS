"""Joint-training upper bound (2026-07-23).

Pools all 6 tasks' train graphs into one dataset and trains the same
``RelationSpecificHeteroGNN`` + linear-classifier architecture on all of
them simultaneously -- no sequential task order, no EWC, no memory bank,
no importance weighting. This isolates a single question from the Step 4
continual-learning result (``runs/step4/forgetting_matrix.json``, average
forgetting 0.7054, final average accuracy 0.36): is the architecture
capable of solving all 10 attack classes + benign when catastrophic
forgetting is removed from the picture entirely?

Not a continual-learning run -- no forgetting matrix, no checkpoint per
task. One checkpoint, one evaluation pass over every task's test split via
``evaluate_accuracy`` (reused from ``trench_ids.cl.train``).

Run:  python -m trench_ids.cl.train_joint
  or: python -m trench_ids.cl.train_joint train.epochs=10
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from trench_ids.cl.evaluate import compute_metrics
from trench_ids.cl.train import evaluate_accuracy, load_split, resolve_device, save_checkpoint
from trench_ids.labels import NUM_TASKS, canonical_classes
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train_joint")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.train.device)
    torch.manual_seed(cfg.train.seed)
    print(f"[train_joint] device = {device}")

    graphs_dir = Path(cfg.paths.graphs_dir)
    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(graphs_dir / "vocab.json")
    protocol_vocab_size = len(vocab["PROTOCOL"])
    service_vocab_size = len(vocab["L7_PROTO"])
    label_names = canonical_classes()

    print("[train_joint] loading and pooling all tasks' train graphs")
    pooled_train_graphs = []
    for task in range(1, NUM_TASKS + 1):
        pooled_train_graphs.extend(load_split(graphs_dir, task, "train"))
    print(f"[train_joint] pooled train graphs = {len(pooled_train_graphs)}")

    model = RelationSpecificHeteroGNN.from_graph(
        pooled_train_graphs[0],
        hidden_dim=cfg.model.hidden_dim,
        protocol_vocab_size=protocol_vocab_size,
        service_vocab_size=service_vocab_size,
        num_layers=cfg.model.num_layers,
        attn_dim=cfg.model.attn_dim,
        port_tail_buckets=cfg.model.port_tail_buckets,
    ).to(device)
    classifier = torch.nn.Linear(cfg.model.hidden_dim, len(label_names)).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()), lr=cfg.optim.learning_rate
    )

    loader = DataLoader(pooled_train_graphs, batch_size=cfg.train.batch_size, shuffle=True)
    run_start = time.monotonic()
    model.train()
    classifier.train()
    for epoch in range(cfg.train.epochs):
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
        mean_loss = total_loss / total_flows if total_flows else 0.0
        print(f"[train_joint] epoch {epoch}: mean loss = {mean_loss:.4f}")

    save_checkpoint(
        out_dir / "checkpoint_joint.pt",
        task_id=NUM_TASKS,
        epochs_per_task=cfg.train.epochs,
        warmup_epochs=0,
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
        },
    )

    print("[train_joint] evaluating on every task's test split")
    per_task_accuracy = {}
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    for task in range(1, NUM_TASKS + 1):
        test_graphs = load_split(graphs_dir, task, "test")
        acc = evaluate_accuracy(model, classifier, test_graphs, device, cfg.train.batch_size)
        per_task_accuracy[task] = acc
        print(f"[eval] task {task} test accuracy = {acc:.4f}")

        model.eval()
        classifier.eval()
        eval_loader = DataLoader(test_graphs, batch_size=cfg.train.batch_size)
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                output = model(batch)
                logits = classifier(output.fused["flow"])
                preds = logits.argmax(dim=-1)
                all_y_true.extend(batch["flow"].y.tolist())
                all_y_pred.extend(preds.tolist())

    pooled_metrics = compute_metrics(all_y_true, all_y_pred, label_names)

    summary = {
        "per_task_accuracy": per_task_accuracy,
        "pooled_accuracy": pooled_metrics["accuracy"],
        "pooled_f1_macro": pooled_metrics["f1_macro"],
        "pooled_f1_weighted": pooled_metrics["f1_weighted"],
        "runtime_seconds": time.monotonic() - run_start,
        "epochs": cfg.train.epochs,
        "seed": cfg.train.seed,
    }
    (out_dir / "joint_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "joint_pooled_metrics.json").write_text(json.dumps(pooled_metrics, indent=2))
    print(f"[done] wrote checkpoint_joint.pt, joint_summary.json to {out_dir}")


if __name__ == "__main__":
    main()
