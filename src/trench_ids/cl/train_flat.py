"""Non-graph baseline under the identical continual-learning protocol.

Runs ``trench_ids.model.flat.FlatFlowEncoder`` (per-flow MLP, no message
passing, no Host node type) through the *same* T1..T6 sequence, the same
optimizer/epochs/batch size, the same optional experience replay, and the
same evaluation code as ``trench_ids.cl.train``'s GNN runs. The only
variable between this and the GNN replay baseline is the encoder.

**Why this run exists.** Every result in ``project-metrics.md`` compares GNN
variants against other GNN variants -- EWC vs. replay vs. joint training --
so the project's foundational claim ("a relation-specific heterogeneous GNN
is a good IDS representation") has never been tested against a model with no
graph. A read-only probe on 2026-08-07 found a gradient-boosted tree on the
same per-flow features scoring accuracy 0.9346 / F1-macro 0.8745, *above*
the GNN's own joint-training upper bound (0.876 / 0.842) and its best
continual-learning result (0.861 / 0.798). That probe was not
capacity-matched or protocol-matched; this module is, so the comparison can
actually be reported.

**Decision rule, fixed before running** (so the outcome is not
rationalised after the fact):

  * flat >= GNN under CL  -> the graph is not earning its place; the next
    priority is graph *construction* (``graphs.py`` shuffles rows before
    chunking, so mini-graph topology is a sampling artifact), not more CL
    variants.
  * GNN > flat under CL, despite the GNN losing on the joint upper bound
    -> graph structure aids *retention* rather than raw accuracy, which is
    a genuine and publishable finding, and becomes the paper's spine.

Fairness controls, all deliberate:

  * **Parameter-matched** -- 263,586 vs. the GNN's 263,378 at the real
    configuration (see ``flat.DEFAULT_MLP_HIDDEN``); both counts are
    recorded in ``summary.json`` so the match is auditable per run rather
    than assumed.
  * **Same inputs minus structure** -- identical Flow-feature pathway, and
    the identical port/protocol/service embeddings, so the port-representation
    fix (2026-07-24) applies to both arms.
  * **Same gradient budget** -- ``epochs_per_task`` here equals the GNN's
    ``warmup_epochs + full_loss_epochs`` (2 + 3 = 5). The GNN's warm-up
    split exists only to stage its EWC penalty; with no EWC there is
    nothing to warm up, so 5 plain epochs is the matched quantity.
  * **Same evaluation** -- checkpoints are written in the same format and
    scored by the unmodified ``trench_ids.cl.evaluate``.

No EWC, no memory bank, no transferability estimation: all three are defined
over Flow's *relation-specific* embeddings, which a model without relations
does not have. This run is the baseline those mechanisms should have been
compared against, not a place to reimplement them.

Run:  trench-train-flat
  or: python -m trench_ids.cl.train_flat
  or: python -m trench_ids.cl.train_flat train.seed=1 replay.enabled=true \\
        paths.out_dir=runs/flat_replay_seed1
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.cl.train import (
    average_forgetting,
    backward_transfer,
    build_replay_augmented_set,
    evaluate_accuracy,
    final_average_accuracy,
    forgetting_row,
    load_split,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from trench_ids.labels import NUM_TASKS, canonical_classes
from trench_ids.model.flat import FlatFlowEncoder
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab, vocab_fingerprint


def count_parameters(*modules: torch.nn.Module) -> int:
    return sum(p.numel() for module in modules for p in module.parameters())


def reference_gnn_parameter_count(
    sample_graph: HeteroData,
    hidden_dim: int,
    protocol_vocab_size: int,
    service_vocab_size: int,
    num_layers: int,
    attn_dim: int,
    port_tail_buckets: int,
) -> int:
    """Parameter count of the GNN encoder this baseline is matched against.

    Built and discarded purely to record the comparison in ``summary.json``
    -- the point of the flat baseline is that a capacity difference cannot
    explain whichever arm wins, and that argument is only checkable if both
    numbers are written down next to the result they justify.
    """
    reference = RelationSpecificHeteroGNN.from_graph(
        sample_graph,
        hidden_dim=hidden_dim,
        protocol_vocab_size=protocol_vocab_size,
        service_vocab_size=service_vocab_size,
        num_layers=num_layers,
        attn_dim=attn_dim,
        port_tail_buckets=port_tail_buckets,
    )
    return count_parameters(reference)


def train_one_task(
    model: torch.nn.Module,
    classifier: torch.nn.Linear,
    graphs: list[HeteroData],
    device: torch.device,
    batch_size: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Plain cross-entropy training over one task. Returns the final epoch's
    flow-weighted mean loss (weighted by flows per batch, matching how
    ``trench_ids.cl.train.train_one_task`` reports it, so the two runs' loss
    curves are directly comparable)."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    model.train()
    classifier.train()
    last_epoch_loss = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        total_flows = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = classifier(model(batch).fused["flow"])
            loss = F.cross_entropy(logits, batch["flow"].y)
            loss.backward()
            optimizer.step()
            n = batch["flow"].y.numel()
            total_loss += loss.item() * n
            total_flows += n
        last_epoch_loss = total_loss / total_flows if total_flows else 0.0
        print(f"[train_flat]   epoch {epoch}: mean loss = {last_epoch_loss:.4f}", flush=True)
    return last_epoch_loss


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train_flat")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.train.device)
    seed_everything(cfg.train.seed)
    print(f"[train_flat] device = {device}, seed = {cfg.train.seed}")

    graphs_dir = Path(cfg.paths.graphs_dir)
    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(graphs_dir / "vocab.json")
    protocol_vocab_size = len(vocab["PROTOCOL"])
    service_vocab_size = len(vocab["L7_PROTO"])
    vocab_fingerprint_value = vocab_fingerprint(vocab)
    label_names = canonical_classes()

    sample_graphs = load_split(graphs_dir, 1, "train")
    model = FlatFlowEncoder(
        hidden_dim=cfg.model.hidden_dim,
        protocol_vocab_size=protocol_vocab_size,
        service_vocab_size=service_vocab_size,
        port_tail_buckets=cfg.model.port_tail_buckets,
        use_host_features=cfg.model.use_host_features,
        mlp_hidden=cfg.model.mlp_hidden,
    ).to(device)
    classifier = torch.nn.Linear(cfg.model.hidden_dim, len(label_names)).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()), lr=cfg.optim.learning_rate
    )

    flat_params = count_parameters(model)
    gnn_params = reference_gnn_parameter_count(
        sample_graphs[0],
        hidden_dim=cfg.model.hidden_dim,
        protocol_vocab_size=protocol_vocab_size,
        service_vocab_size=service_vocab_size,
        num_layers=cfg.reference_model.num_layers,
        attn_dim=cfg.reference_model.attn_dim,
        port_tail_buckets=cfg.model.port_tail_buckets,
    )
    arm = "flat+host" if model.use_host_features else "flat"
    print(
        f"[train_flat] arm = {arm}, mlp_hidden = {model.mlp_hidden}, "
        f"encoder parameters: {flat_params:,} vs. reference GNN "
        f"{gnn_params:,} ({flat_params / gnn_params:.4f}x)"
    )
    print(f"[train_flat] replay enabled = {bool(cfg.replay.enabled)}")

    checkpoint_config = {
        "model_type": "flat",
        "use_host_features": model.use_host_features,
        "hidden_dim": cfg.model.hidden_dim,
        # The resolved width, not cfg.model.mlp_hidden -- that may be null,
        # in which case the encoder picked the width matching this arm.
        "mlp_hidden": model.mlp_hidden,
        "port_tail_buckets": cfg.model.port_tail_buckets,
        "protocol_vocab_size": protocol_vocab_size,
        "service_vocab_size": service_vocab_size,
        # Re-checked by inference.load_checkpoint against whatever
        # vocab.json a later run loads -- see vocab.vocab_fingerprint.
        "vocab_fingerprint": vocab_fingerprint_value,
        "label_names": label_names,
        "replay_enabled": bool(cfg.replay.enabled),
        "replay_buffer_size_per_task": cfg.replay.buffer_size_per_task,
        "replay_fraction": cfg.replay.replay_fraction,
    }

    forgetting_matrix: dict[int, dict[int, float]] = {}
    val_matrix: dict[int, dict[int, float]] = {}
    replay_buffer: list[HeteroData] = []
    run_start = time.monotonic()

    for task in range(1, NUM_TASKS + 1):
        train_graphs = sample_graphs if task == 1 else load_split(graphs_dir, task, "train")
        training_set = (
            build_replay_augmented_set(train_graphs, replay_buffer, cfg.replay.replay_fraction)
            if cfg.replay.enabled
            else train_graphs
        )
        print(
            f"[train_flat] task {task}: {len(train_graphs)} own graphs, "
            f"{len(training_set)} after replay mixing",
            flush=True,
        )

        final_loss = train_one_task(
            model,
            classifier,
            training_set,
            device,
            cfg.train.batch_size,
            cfg.train.epochs_per_task,
            optimizer,
        )
        print(f"[train_flat] task {task}: final epoch mean loss = {final_loss:.4f}")

        if cfg.replay.enabled:
            # Uniform sampling, identical to trench_ids.cl.train's
            # replay.selection="uniform" path, so the buffer-fill policy is
            # not a second variable alongside the encoder change.
            n_sample = min(cfg.replay.buffer_size_per_task, len(train_graphs))
            replay_buffer.extend(random.sample(train_graphs, k=n_sample))

        def _accuracy_for(evaluated_task: int, split: str) -> float:
            split_graphs = load_split(graphs_dir, evaluated_task, split)
            return evaluate_accuracy(model, classifier, split_graphs, device, cfg.train.batch_size)

        forgetting_matrix[task] = forgetting_row(lambda t: _accuracy_for(t, "test"), task)
        for evaluated_task, acc in forgetting_matrix[task].items():
            print(f"[eval] after task {task}, task {evaluated_task} test accuracy = {acc:.4f}")

        val_matrix[task] = forgetting_row(lambda t: _accuracy_for(t, "val"), task)
        for evaluated_task, acc in val_matrix[task].items():
            print(f"[eval] after task {task}, task {evaluated_task} val accuracy = {acc:.4f}")

        save_checkpoint(
            out_dir / f"checkpoint_task_{task}.pt",
            task_id=task,
            epochs_per_task=cfg.train.epochs_per_task,
            warmup_epochs=0,  # no EWC penalty to stage -- see module docstring
            seed=cfg.train.seed,
            model=model,
            classifier=classifier,
            config=checkpoint_config,
        )

    (out_dir / "forgetting_matrix.json").write_text(json.dumps(forgetting_matrix, indent=2))
    (out_dir / "val_matrix.json").write_text(json.dumps(val_matrix, indent=2))
    summary = {
        "model_type": "flat",
        "arm": arm,
        "use_host_features": model.use_host_features,
        "average_forgetting": average_forgetting(forgetting_matrix),
        "backward_transfer": backward_transfer(forgetting_matrix),
        "final_average_accuracy": final_average_accuracy(forgetting_matrix),
        "final_average_val_accuracy": final_average_accuracy(val_matrix),
        "runtime_seconds": time.monotonic() - run_start,
        "encoder_parameters": flat_params,
        "reference_gnn_encoder_parameters": gnn_params,
        "parameter_ratio_flat_over_gnn": flat_params / gnn_params,
        "epochs_per_task": cfg.train.epochs_per_task,
        "batch_size": cfg.train.batch_size,
        "learning_rate": cfg.optim.learning_rate,
        "mlp_hidden": model.mlp_hidden,
        "hidden_dim": cfg.model.hidden_dim,
        "replay_enabled": bool(cfg.replay.enabled),
        "replay_buffer_size_per_task": cfg.replay.buffer_size_per_task,
        "replay_fraction": cfg.replay.replay_fraction,
        "graphs_dir": str(graphs_dir),
        "seed": cfg.train.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[done] avg forgetting = {summary['average_forgetting']:.4f}, "
        f"final avg accuracy = {summary['final_average_accuracy']:.4f} "
        f"-> {out_dir}"
    )


if __name__ == "__main__":
    main()
