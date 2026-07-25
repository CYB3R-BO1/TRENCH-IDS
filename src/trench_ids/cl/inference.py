"""Inference -- load a trained checkpoint and run predictions on graphs.

Decoupled from metric computation (trench_ids.cl.evaluate) so this module
is reusable for a future genuinely-unseen-raw-traffic pipeline (Step 10's
third bullet, explicitly deferred) without any API change -- predict()
never computes anything beyond the model's own predictions, so a caller
that only has raw graphs (no ground truth at all) can still use it.

Run: python -m trench_ids.cl.inference --checkpoint runs/step4/checkpoint_task_3.pt \
  --graphs-dir data/graphs --task 3 --split test
  or: trench-infer ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.cl.train import load_split
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN

SUPPORTED_CHECKPOINT_VERSION = 1


def load_checkpoint(
    path: Path, graphs_dir: Path, device: torch.device
) -> tuple[RelationSpecificHeteroGNN, torch.nn.Linear, dict]:
    """Rebuilds the model + classifier from a checkpoint saved by
    trench_ids.cl.train.save_checkpoint. Reads task 1's train split's first
    graph purely for edge/node-type metadata
    (RelationSpecificHeteroGNN.from_graph) -- every mini-graph in this
    benchmark shares the same schema, so any graph would do; task 1's train
    split is guaranteed to exist for any run that reached checkpointing."""
    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )  # trusted, first-party output (same convention as memory_bank.load_memory_bank)
    version = checkpoint.get("checkpoint_version")
    if version != SUPPORTED_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint_version {version!r} (expected "
            f"{SUPPORTED_CHECKPOINT_VERSION}) in {path}"
        )
    config = checkpoint["config"]
    sample_graph = load_split(graphs_dir, 1, "train")[0]
    model = RelationSpecificHeteroGNN.from_graph(
        sample_graph,
        hidden_dim=config["hidden_dim"],
        protocol_vocab_size=config["protocol_vocab_size"],
        service_vocab_size=config["service_vocab_size"],
        num_layers=config["num_layers"],
        attn_dim=config["attn_dim"],
        port_tail_buckets=config["port_tail_buckets"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classifier = torch.nn.Linear(config["hidden_dim"], len(config["label_names"])).to(device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    classifier.eval()

    return model, classifier, checkpoint


def predict(
    model: RelationSpecificHeteroGNN,
    classifier: torch.nn.Linear,
    graphs: list[HeteroData],
    device: torch.device,
    batch_size: int,
    label_names: list[str],
) -> dict:
    """No-grad forward pass over `graphs`. Returns y_true/y_pred (global
    class indices, trench_ids.labels.canonical_classes() order, matching
    how graph["flow"].y is built in Step 2) and y_prob (softmax
    probabilities, same order) as plain Python lists."""
    model.eval()
    classifier.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    loader = DataLoader(graphs, batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            logits = classifier(output.fused["flow"])
            probs = F.softmax(logits, dim=-1)
            y_true.extend(batch["flow"].y.tolist())
            y_pred.extend(logits.argmax(dim=-1).tolist())
            y_prob.extend(probs.tolist())
    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "label_names": label_names}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for one checkpoint/task/split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graphs-dir", default="data/graphs")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    graphs_dir = Path(args.graphs_dir)
    model, classifier, checkpoint = load_checkpoint(Path(args.checkpoint), graphs_dir, device)
    graphs = load_split(graphs_dir, args.task, args.split)
    result = predict(
        model, classifier, graphs, device, args.batch_size, checkpoint["config"]["label_names"]
    )

    correct = sum(
        int(t == p) for t, p in zip(result["y_true"], result["y_pred"], strict=True)
    )
    accuracy = correct / len(result["y_true"]) if result["y_true"] else 0.0

    print(f"Checkpoint : {Path(args.checkpoint).name}")
    print(f"Task       : {args.task}")
    print(f"Split      : {args.split}")
    print(f"Samples    : {len(result['y_true'])}")
    print(f"Accuracy   : {accuracy:.1%}")


if __name__ == "__main__":
    main()
