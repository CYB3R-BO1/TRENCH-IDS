"""End-to-end integration test: synthetic graphs -> 2-task continual
training (with replay + memory bank) -> checkpoint round-trip -> inference.

Complements the per-module unit tests by exercising the *wiring* the real
pipeline depends on: save/load of task split files, the train_one_task ->
compute_task_memory_means -> merge_into_bank -> replay-buffer -> checkpoint
sequence train.py's main() runs, and inference.load_checkpoint rebuilding a
predictable model from that checkpoint. Runs on CPU with tiny graphs and
real feature dimensions, in well under a second.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import yaml
from torch_geometric.data import HeteroData

from trench_ids.cl.ewc import FLOW_RELATIONS, OnlineEWCManager
from trench_ids.cl.importance import ImportanceMLP
from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.memory_bank import merge_into_bank
from trench_ids.cl.train import (
    average_forgetting,
    build_replay_augmented_set,
    compute_task_memory_means,
    evaluate_accuracy,
    final_average_accuracy,
    forgetting_row,
    load_split,
    save_checkpoint,
    seed_everything,
    train_one_task,
)
from trench_ids.constants import FLOW_FEATURE_DIM, HOST_FEATURE_DIM
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN

HIDDEN_DIM = 8
# Global label space mirroring the real pipeline's structure: Benign (index 0)
# is present in every task and excluded from the memory bank; each attack
# class belongs to exactly one task (merge_into_bank enforces this).
LABEL_NAMES = ["Benign", "attack_a", "attack_b"]
TASK_CLASSES = {1: [0, 1], 2: [0, 2]}
REPLAY_FRACTION = 0.3


def _tiny_graph(class_mix: list[int]) -> HeteroData:
    """Real-dimension mini-graph mirroring the 5-node-type/11-relation Step 2
    schema (real FLOW_FEATURE_DIM/HOST_FEATURE_DIM so from_graph's default
    encoders match, same shape convention as tests/test_inference.py).
    ``class_mix`` holds one global class index per Flow node."""
    g = HeteroData()
    g["flow"].x = torch.randn(4, FLOW_FEATURE_DIM)
    g["flow"].y = torch.tensor(class_mix)
    g["host"].num_nodes = 2
    g["host"].x = torch.randn(2, HOST_FEATURE_DIM)
    g["protocol"].num_nodes = 1
    g["protocol"].vocab_id = torch.tensor([0])
    g["service"].num_nodes = 1
    g["service"].vocab_id = torch.tensor([0])
    g["port"].num_nodes = 2
    g["port"].port_number = torch.tensor([22, 80])

    src_host = torch.tensor([0, 1, 0, 1])
    dst_host = torch.tensor([1, 0, 1, 0])
    protocol_idx = torch.tensor([0, 0, 0, 0])
    service_idx = torch.tensor([0, 0, 0, 0])
    port_idx = torch.tensor([0, 1, 0, 1])

    g["host", "originates", "flow"].edge_index = torch.stack([src_host, torch.arange(4)])
    g["flow", "terminates_at", "host"].edge_index = torch.stack([torch.arange(4), dst_host])
    g["flow", "targets_port", "port"].edge_index = torch.stack([torch.arange(4), port_idx])
    g["flow", "uses_protocol", "protocol"].edge_index = torch.stack(
        [torch.arange(4), protocol_idx]
    )
    g["flow", "uses_service", "service"].edge_index = torch.stack(
        [torch.arange(4), service_idx]
    )
    g["host", "communicates_with", "host"].edge_index = torch.tensor([[0], [1]])
    g["flow", "originated_by", "host"].edge_index = torch.stack([torch.arange(4), src_host])
    g["host", "terminated_by", "flow"].edge_index = torch.stack([dst_host, torch.arange(4)])
    g["port", "targeted_by", "flow"].edge_index = torch.stack([port_idx, torch.arange(4)])
    g["protocol", "protocol_of", "flow"].edge_index = torch.stack(
        [protocol_idx, torch.arange(4)]
    )
    g["service", "service_of", "flow"].edge_index = torch.stack(
        [service_idx, torch.arange(4)]
    )
    return g


def _write_splits(graphs_dir: Path, task: int) -> list[HeteroData]:
    """Write one task's train/val/test split files; returns the train split."""
    benign, attack = TASK_CLASSES[task]
    class_mix = [benign, attack, benign, attack]
    train = [_tiny_graph(class_mix), _tiny_graph(class_mix)]
    val = [_tiny_graph(class_mix)]
    test = [_tiny_graph(class_mix)]
    for split, graphs in (("train", train), ("val", val), ("test", test)):
        torch.save(graphs, graphs_dir / f"task_{task}_{split}.pt")
    return train


def test_constants_match_graph_config_and_schema() -> None:
    """trench_ids.constants must agree with the two places the schema is
    independently defined: configs/graph.yaml's features list (length) and
    the fixture graph's Flow-incoming relation set."""
    config_path = Path(__file__).parents[1] / "configs" / "graph.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f)
    assert len(config["features"]) == FLOW_FEATURE_DIM

    fixture_relations = {
        edge_type[1]
        for edge_type in _tiny_graph([0, 1, 0, 1]).edge_types
        if edge_type[2] == "flow"
    }
    assert set(FLOW_RELATIONS) == fixture_relations


def test_full_pipeline_two_tasks_with_replay(tmp_path) -> None:
    device = torch.device("cpu")
    seed_everything(42)

    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    task_train = {1: _write_splits(graphs_dir, 1)}
    task_train[2] = _write_splits(graphs_dir, 2)

    sample = load_split(graphs_dir, 1, "train")[0]
    model = RelationSpecificHeteroGNN.from_graph(
        sample,
        hidden_dim=HIDDEN_DIM,
        protocol_vocab_size=2,
        service_vocab_size=2,
        num_layers=1,
    ).to(device)
    classifier = torch.nn.Linear(HIDDEN_DIM, len(LABEL_NAMES)).to(device)
    importance_mlp = ImportanceMLP().to(device)
    # EWC is the superseded mechanism -- all lambdas zero, exactly like most
    # arms of the experiment matrix; this also exercises the S_r-skip path.
    ewc_manager = OnlineEWCManager(
        model, classifier, FLOW_RELATIONS, gamma=0.9, lambda_r=0.0, lambda_s=0.0, lambda_u=0.0,
    )
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(classifier.parameters())
        + list(importance_mlp.parameters()),
        lr=1e-3,
    )

    memory_bank: dict[str, dict[str, torch.Tensor]] = {}
    replay_buffer: list[HeteroData] = []
    forgetting_matrix: dict[int, dict[int, float]] = {}
    checkpoint_paths: dict[int, Path] = {}

    for task in (1, 2):
        training_set = build_replay_augmented_set(task_train[task], replay_buffer, REPLAY_FRACTION)
        if task == 2:
            # Replay graphs are upsampled with replacement into the pool.
            assert len(training_set) > len(task_train[task])

        _, _, distill_report = train_one_task(
            model,
            classifier,
            training_set,
            device,
            batch_size=2,
            warmup_epochs=1,
            full_loss_epochs=1,
            optimizer=optimizer,
            ewc_manager=ewc_manager,
            importance_mlp=importance_mlp,
            label_names=LABEL_NAMES,
            bank=memory_bank,
            disable_learned_weighting=True,
            task_id=task,
            seed=42,
        )
        # No distiller attached: no weights/d_r are ever set, and the
        # per-epoch penalty log is present but identically zero (train_one_task
        # records mean_penalty_per_epoch unconditionally).
        assert "weights" not in distill_report
        assert "d_r" not in distill_report
        assert distill_report["mean_penalty_per_epoch"] == [0.0]

        final_means = compute_task_memory_means(model, task_train[task], device, 2, LABEL_NAMES)
        memory_bank = merge_into_bank(memory_bank, final_means)
        # Benign is excluded from the bank (attack-type memory, not a general
        # class memory); each task adds exactly its own attack class, and the
        # bank accumulates across tasks.
        assert set(memory_bank) == {f"attack_{chr(96 + t)}" for t in range(1, task + 1)}
        assert set(memory_bank[f"attack_{chr(96 + task)}"]) == set(FLOW_RELATIONS)

        replay_buffer.extend(random.sample(task_train[task], k=len(task_train[task])))

        def accuracy_for(evaluated_task: int) -> float:
            graphs = load_split(graphs_dir, evaluated_task, "test")
            return evaluate_accuracy(model, classifier, graphs, device, batch_size=2)

        forgetting_matrix[task] = forgetting_row(accuracy_for, task)
        for acc in forgetting_matrix[task].values():
            assert 0.0 <= acc <= 1.0

        checkpoint_path = tmp_path / f"checkpoint_task_{task}.pt"
        save_checkpoint(
            checkpoint_path,
            task_id=task,
            epochs_per_task=2,
            warmup_epochs=1,
            seed=42,
            model=model,
            classifier=classifier,
            config={
                "hidden_dim": HIDDEN_DIM,
                "num_layers": 1,
                "attn_dim": 128,
                "port_tail_buckets": 32,
                "fusion": "attention",
                "protocol_vocab_size": 2,
                "service_vocab_size": 2,
                "label_names": LABEL_NAMES,
            },
        )
        checkpoint_paths[task] = checkpoint_path

    assert average_forgetting(forgetting_matrix) <= 1.0
    assert 0.0 <= final_average_accuracy(forgetting_matrix) <= 1.0

    # Round-trip through the inference path: rebuild from the task-2
    # checkpoint against the on-disk splits, then predict.
    loaded_model, loaded_classifier, checkpoint = load_checkpoint(
        checkpoint_paths[2], graphs_dir, device
    )
    assert checkpoint["task_id"] == 2
    result = predict(
        loaded_model,
        loaded_classifier,
        load_split(graphs_dir, 2, "test"),
        device,
        batch_size=2,
        label_names=checkpoint["config"]["label_names"],
    )
    assert len(result["y_true"]) == 4
    assert all(p in (0, 1, 2) for p in result["y_pred"])
