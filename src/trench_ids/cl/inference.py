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
from trench_ids.model.flat import FlatFlowEncoder
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab, vocab_fingerprint

SUPPORTED_CHECKPOINT_VERSION = 1

# Checkpoints written before the non-graph baseline existed carry no
# "model_type" key; they are all RelationSpecificHeteroGNN runs.
DEFAULT_MODEL_TYPE = "rhgnn"


def build_model(
    config: dict, sample_graph, device: torch.device
) -> torch.nn.Module:
    """Rebuild the encoder a checkpoint's ``config`` describes.

    Dispatches on ``config["model_type"]``: ``"rhgnn"`` (the default, and
    what every pre-2026-08-07 checkpoint implicitly is) builds the
    relation-specific heterogeneous GNN; ``"flat"`` builds the non-graph
    baseline. ``sample_graph`` supplies edge/node-type metadata and is used
    only by the GNN path -- the flat encoder has no schema to match, which
    is the whole point of it.
    """
    model_type = config.get("model_type", DEFAULT_MODEL_TYPE)
    if model_type == DEFAULT_MODEL_TYPE:
        return RelationSpecificHeteroGNN.from_graph(
            sample_graph,
            hidden_dim=config["hidden_dim"],
            protocol_vocab_size=config["protocol_vocab_size"],
            service_vocab_size=config["service_vocab_size"],
            num_layers=config["num_layers"],
            attn_dim=config["attn_dim"],
            port_tail_buckets=config["port_tail_buckets"],
            # Defaults to "attention" for checkpoints predating the key --
            # all of which were attention-fusion runs. Getting this wrong
            # would not silently degrade accuracy, it would fail to load:
            # the two modes have different parameter sets.
            fusion=config.get("fusion", "attention"),
        ).to(device)
    if model_type == "flat":
        use_host_features = config.get("use_host_features", False)
        return FlatFlowEncoder(
            hidden_dim=config["hidden_dim"],
            protocol_vocab_size=config["protocol_vocab_size"],
            service_vocab_size=config["service_vocab_size"],
            port_tail_buckets=config["port_tail_buckets"],
            use_host_features=use_host_features,
            # None lets FlatFlowEncoder pick the width matching this arm;
            # every checkpoint train_flat.py writes records the resolved
            # value, so this fallback only covers hand-built configs.
            mlp_hidden=config.get("mlp_hidden"),
        ).to(device)
    raise ValueError(f"Unknown model_type {model_type!r} in checkpoint config")


def load_checkpoint(
    path: Path, graphs_dir: Path, device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Linear, dict]:
    """Rebuilds the model + classifier from a checkpoint saved by
    trench_ids.cl.train.save_checkpoint. Reads task 1's train split's first
    graph purely for edge/node-type metadata
    (RelationSpecificHeteroGNN.from_graph) -- every mini-graph in this
    benchmark shares the same schema, so any graph would do; task 1's train
    split is guaranteed to exist for any run that reached checkpointing.

    The returned encoder's concrete class depends on the checkpoint's
    ``model_type`` (see ``build_model``); both satisfy the same
    ``forward(graph) -> RelationSpecificOutput`` contract, so ``predict``
    and everything downstream of it (``trench_ids.cl.evaluate``) is
    unchanged either way."""
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
    # Both encoders' Protocol/Service embeddings are indexed by vocab.json's
    # ids, which are assigned by *position* in a sorted value set
    # (vocab.build_vocab) -- a Step 1/2 rerun after this checkpoint was
    # trained renumbers them, so a value ends up pointing at a different,
    # untrained embedding row without any shape mismatch to raise on. Only
    # checked when the checkpoint recorded a fingerprint (added alongside
    # this check; older checkpoints predate it and are not retroactively
    # checkable).
    expected_fingerprint = config.get("vocab_fingerprint")
    if expected_fingerprint is not None:
        current_fingerprint = vocab_fingerprint(load_vocab(graphs_dir / "vocab.json"))
        if current_fingerprint != expected_fingerprint:
            raise ValueError(
                f"{graphs_dir / 'vocab.json'} does not match the vocabulary {path} was "
                "trained with (vocab_fingerprint mismatch). Rebuilding Step 1/2 renumbers "
                "Protocol/Service ids, so loading this checkpoint against the current "
                "vocab would silently map values to the wrong embedding rows. Point "
                "graphs_dir at the vocab this checkpoint was actually trained with, or "
                "retrain against the current one."
            )
    sample_graph = load_split(graphs_dir, 1, "train")[0]
    model = build_model(config, sample_graph, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classifier = torch.nn.Linear(config["hidden_dim"], len(config["label_names"])).to(device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    classifier.eval()

    return model, classifier, checkpoint


def predict(
    model: torch.nn.Module,
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
