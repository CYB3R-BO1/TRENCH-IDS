from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.cl.ewc import FLOW_RELATIONS, OnlineEWCManager
from trench_ids.cl.importance import ImportanceMLP
from trench_ids.cl.train import (
    average_forgetting,
    build_replay_augmented_set,
    final_average_accuracy,
    forgetting_row,
    save_checkpoint,
    split_warmup_and_full_loss_epochs,
    train_one_task,
)
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN

FLOW_DIM = 3
HOST_DIM = 2


def _tiny_graph() -> HeteroData:
    """Minimal HeteroData mirroring the real 5-node-type/11-relation schema
    (same shape convention as tests/test_ewc.py's _tiny_graph, kept
    self-contained here so test_train.py has no cross-test-file import)."""
    g = HeteroData()
    g["flow"].x = torch.randn(4, FLOW_DIM)
    g["flow"].y = torch.tensor([0, 1, 0, 1])
    g["host"].num_nodes = 2
    g["host"].x = torch.randn(2, HOST_DIM)
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
    g["flow", "uses_protocol", "protocol"].edge_index = torch.stack([torch.arange(4), protocol_idx])
    g["flow", "uses_service", "service"].edge_index = torch.stack([torch.arange(4), service_idx])
    g["host", "communicates_with", "host"].edge_index = torch.tensor([[0], [1]])
    g["flow", "originated_by", "host"].edge_index = torch.stack([torch.arange(4), src_host])
    g["host", "terminated_by", "flow"].edge_index = torch.stack([dst_host, torch.arange(4)])
    g["port", "targeted_by", "flow"].edge_index = torch.stack([port_idx, torch.arange(4)])
    g["protocol", "protocol_of", "flow"].edge_index = torch.stack([protocol_idx, torch.arange(4)])
    g["service", "service_of", "flow"].edge_index = torch.stack([service_idx, torch.arange(4)])
    return g


def test_forgetting_row_evaluates_every_previously_seen_task() -> None:
    fake_accuracies = {1: 0.9, 2: 0.8, 3: 0.7}

    row = forgetting_row(lambda t: fake_accuracies[t], up_to_task=3)

    assert row == {1: 0.9, 2: 0.8, 3: 0.7}


def test_forgetting_row_after_first_task_is_a_single_entry() -> None:
    row = forgetting_row(lambda t: 1.0, up_to_task=1)

    assert row == {1: 1.0}


def test_forgetting_row_calls_accuracy_fn_once_per_task() -> None:
    calls: list[int] = []

    def accuracy_fn(task: int) -> float:
        calls.append(task)
        return float(task)

    forgetting_row(accuracy_fn, up_to_task=4)

    assert calls == [1, 2, 3, 4]


def test_split_warmup_and_full_loss_epochs_matches_config() -> None:
    warmup, full = split_warmup_and_full_loss_epochs(epochs_per_task=5, warmup_epochs=2)
    assert warmup == 2
    assert full == 3


def test_split_warmup_and_full_loss_epochs_rejects_warmup_exceeding_total() -> None:
    with pytest.raises(ValueError, match="warmup_epochs"):
        split_warmup_and_full_loss_epochs(epochs_per_task=3, warmup_epochs=5)


def test_train_one_task_runs_end_to_end_on_cpu_with_nonempty_bank() -> None:
    """CPU-visible integration test for the full warm-up -> temp-prototypes
    -> S_r -> w_r -> combined-loss -> optimizer-step pipeline.

    The only other test that runs train_one_task end-to-end
    (test_train_one_task_moves_s_r_onto_device_before_importance_mlp, above)
    is skipped whenever torch.cuda.is_available() is False, which leaves this
    entire pipeline untested on any CPU-only machine (this repo's dev box was
    CPU-only until 2026-07-19, and CI/other contributors' machines may still
    be). This test uses a non-empty bank (one class's per-relation Flow
    means) so at least one OnlineEWCState is initialized=True and the
    weighted EWC term is genuinely non-zero, not trivially skipped, and
    asserts importance_mlp actually received gradients during the call.
    """
    device = torch.device("cpu")
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1,
    )
    from trench_ids.model.rhgnn import NodeFeatureEncoders

    model.encoders = NodeFeatureEncoders(
        8, protocol_vocab_size=2, service_vocab_size=2,
        flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM,
    )
    model = model.to(device)
    classifier = torch.nn.Linear(8, 2).to(device)
    importance_mlp = ImportanceMLP().to(device)
    ewc_manager = OnlineEWCManager(
        model, classifier, FLOW_RELATIONS, gamma=0.9, lambda_r=1.0, lambda_s=1.0, lambda_u=1.0,
    )
    # Prime the EWC manager with a Fisher/theta* estimate so its shared and
    # per-relation states are initialized (loss() would otherwise be zero).
    loader = DataLoader([g, g], batch_size=2)
    ewc_manager.update_all(model, classifier, loader, device)

    trainable_params = (
        list(model.parameters())
        + list(classifier.parameters())
        + list(importance_mlp.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    # Non-empty bank: one class's per-relation Flow means, matching the shape
    # produced by RelationMeanAccumulator.means() in memory_bank.py.
    bank = {
        "attack_a": {relation: torch.randn(8) for relation in FLOW_RELATIONS},
    }

    train_one_task(
        model,
        classifier,
        [g, g],
        device,
        batch_size=2,
        warmup_epochs=1,
        full_loss_epochs=2,
        optimizer=optimizer,
        ewc_manager=ewc_manager,
        importance_mlp=importance_mlp,
        label_names=["a", "b"],
        bank=bank,
    )

    # The full-loss epochs' combined loss (L_cls + weighted EWC term) must
    # have backpropagated into importance_mlp at some point -- optimizer.step()
    # doesn't clear .grad, so it's still readable here after the call returns.
    assert importance_mlp.net[0].weight.grad is not None
    assert torch.any(importance_mlp.net[0].weight.grad != 0.0)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="regression test for a device mismatch that only manifests with a real non-CPU device",
)
def test_train_one_task_moves_s_r_onto_device_before_importance_mlp() -> None:
    """Regression test: aggregate_transferability_scores builds S_r via
    plain torch.tensor(...) with no device= argument, so it always lands on
    CPU. train_one_task must move S_r onto `device` before importance_mlp
    (already moved to `device`) consumes it, or this raises a device-mismatch
    RuntimeError on any machine with a real GPU (this dev box included --
    CLAUDE.md notes it gained a CUDA device 2026-07-19)."""
    device = torch.device("cuda")
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1,
    )
    from trench_ids.model.rhgnn import NodeFeatureEncoders

    model.encoders = NodeFeatureEncoders(
        8, protocol_vocab_size=2, service_vocab_size=2,
        flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM,
    )
    model = model.to(device)
    classifier = torch.nn.Linear(8, 2).to(device)
    importance_mlp = ImportanceMLP().to(device)
    ewc_manager = OnlineEWCManager(
        model, classifier, FLOW_RELATIONS, gamma=0.9, lambda_r=1.0, lambda_s=1.0, lambda_u=1.0,
    )
    trainable_params = (
        list(model.parameters())
        + list(classifier.parameters())
        + list(importance_mlp.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    # bank={} -> S_r comes back as CPU tensors of 0.0 (empty-average default).
    # Without the fix, importance_mlp(s_r) inside train_one_task raises
    # "Expected all tensors to be on the same device" the first time the
    # full-loss loop runs.
    train_one_task(
        model,
        classifier,
        [g, g],
        device,
        batch_size=2,
        warmup_epochs=1,
        full_loss_epochs=1,
        optimizer=optimizer,
        ewc_manager=ewc_manager,
        importance_mlp=importance_mlp,
        label_names=["a", "b"],
        bank={},
    )


def test_train_one_task_disabled_weighting_fixes_w_r_and_skips_importance_mlp_grad() -> None:
    """When disable_learned_weighting=True (the plain-Online-EWC baseline),
    w_r must be fixed at 1.0 for every Flow relation rather than routed
    through importance_mlp, so lambda_r=lambda_s=lambda_u gives every one of
    the 12 EWC groups the same uniform, unweighted penalty. Verified two
    ways: importance_mlp's parameters receive no gradient (it's never called
    on the forward path), and final_w_r in the returned tuple is exactly 1.0
    for every relation."""
    device = torch.device("cpu")
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1,
    )
    from trench_ids.model.rhgnn import NodeFeatureEncoders

    model.encoders = NodeFeatureEncoders(
        8, protocol_vocab_size=2, service_vocab_size=2,
        flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM,
    )
    model = model.to(device)
    classifier = torch.nn.Linear(8, 2).to(device)
    importance_mlp = ImportanceMLP().to(device)
    ewc_manager = OnlineEWCManager(
        model, classifier, FLOW_RELATIONS, gamma=0.9, lambda_r=1.0, lambda_s=1.0, lambda_u=1.0,
    )
    loader = DataLoader([g, g], batch_size=2)
    ewc_manager.update_all(model, classifier, loader, device)

    trainable_params = (
        list(model.parameters())
        + list(classifier.parameters())
        + list(importance_mlp.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    bank = {
        "attack_a": {relation: torch.randn(8) for relation in FLOW_RELATIONS},
    }

    _, final_w_r = train_one_task(
        model,
        classifier,
        [g, g],
        device,
        batch_size=2,
        warmup_epochs=1,
        full_loss_epochs=2,
        optimizer=optimizer,
        ewc_manager=ewc_manager,
        importance_mlp=importance_mlp,
        label_names=["a", "b"],
        bank=bank,
        disable_learned_weighting=True,
    )

    assert final_w_r == {relation: 1.0 for relation in FLOW_RELATIONS}
    assert importance_mlp.net[0].weight.grad is None or torch.all(
        importance_mlp.net[0].weight.grad == 0.0
    )


def test_train_one_task_writes_loss_components_log_when_path_given(tmp_path) -> None:
    """Diagnostic instrumentation (2026-07-21 lambda-sweep follow-up): with
    epoch_log_path given, train_one_task must append one JSON line per
    full-loss epoch recording L_cls and the raw/weighted EWC breakdown, so
    a run's log can show whether lambda*L_EWC is actually large enough to
    influence optimization rather than just re-running more lambda values."""
    device = torch.device("cpu")
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1,
    )
    from trench_ids.model.rhgnn import NodeFeatureEncoders

    model.encoders = NodeFeatureEncoders(
        8, protocol_vocab_size=2, service_vocab_size=2,
        flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM,
    )
    model = model.to(device)
    classifier = torch.nn.Linear(8, 2).to(device)
    importance_mlp = ImportanceMLP().to(device)
    ewc_manager = OnlineEWCManager(
        model, classifier, FLOW_RELATIONS,
        gamma=0.9, lambda_r=1000.0, lambda_s=1000.0, lambda_u=1000.0,
    )
    loader = DataLoader([g, g], batch_size=2)
    ewc_manager.update_all(model, classifier, loader, device)

    trainable_params = (
        list(model.parameters())
        + list(classifier.parameters())
        + list(importance_mlp.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    bank = {
        "attack_a": {relation: torch.randn(8) for relation in FLOW_RELATIONS},
    }
    log_path = tmp_path / "loss_components.jsonl"

    train_one_task(
        model,
        classifier,
        [g, g],
        device,
        batch_size=2,
        warmup_epochs=1,
        full_loss_epochs=2,
        optimizer=optimizer,
        ewc_manager=ewc_manager,
        importance_mlp=importance_mlp,
        label_names=["a", "b"],
        bank=bank,
        epoch_log_path=log_path,
    )

    import json

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["epoch"] == i
        assert set(record.keys()) == {
            "epoch", "l_cls", "shared_raw", "flow_raw", "other_raw",
            "total_weighted", "loss_total",
        }


def test_build_replay_augmented_set_returns_current_graphs_unchanged_for_empty_buffer() -> None:
    """Task 1 (no buffer yet) must be byte-identical to plain training --
    regression guard so replay is provably a no-op until a buffer exists."""
    current = [object(), object(), object()]

    result = build_replay_augmented_set(current, replay_buffer=[], replay_fraction=0.3)

    assert result is current


def test_build_replay_augmented_set_hits_target_replay_fraction() -> None:
    current = [object() for _ in range(100)]
    buffer = [object() for _ in range(10)]

    result = build_replay_augmented_set(current, buffer, replay_fraction=0.3)

    n_replay = len(result) - len(current)
    assert all(g in current for g in result[: len(current)])
    assert all(g in buffer for g in result[len(current) :])
    # replay_fraction of the *combined* pool, not of current alone:
    # n_replay / (len(current) + n_replay) ~= 0.3
    assert n_replay / len(result) == pytest.approx(0.3, abs=0.01)


def test_average_forgetting_returns_zero_for_single_task() -> None:
    assert average_forgetting({1: {1: 0.9}}) == 0.0


def test_average_forgetting_is_peak_minus_final_averaged_over_earlier_tasks() -> None:
    forgetting_matrix = {
        1: {1: 0.9},
        2: {1: 0.8, 2: 0.85},
        3: {1: 0.6, 2: 0.7, 3: 0.95},
    }

    # Task 1's peak accuracy across evaluations 1..2 (its own eval + task 2's
    # re-eval) is 0.9, final (task 3's re-eval) is 0.6 -> forgetting 0.3.
    # Task 2's peak across evaluations 2..2 is 0.85, final is 0.7 -> 0.15.
    # Average of [0.3, 0.15] = 0.225.
    assert average_forgetting(forgetting_matrix) == pytest.approx(0.225)


def test_final_average_accuracy_averages_last_row() -> None:
    forgetting_matrix = {
        1: {1: 0.9},
        2: {1: 0.8, 2: 0.85},
    }

    assert final_average_accuracy(forgetting_matrix) == pytest.approx((0.8 + 0.85) / 2)


def test_save_checkpoint_writes_loadable_state(tmp_path: Path) -> None:
    device = torch.device("cpu")
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1,
    )
    from trench_ids.model.rhgnn import NodeFeatureEncoders

    model.encoders = NodeFeatureEncoders(
        8, protocol_vocab_size=2, service_vocab_size=2,
        flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM,
    )
    model = model.to(device)
    classifier = torch.nn.Linear(8, 2).to(device)
    config = {
        "hidden_dim": 8, "num_layers": 1, "attn_dim": 128, "port_tail_buckets": 32,
        "protocol_vocab_size": 2, "service_vocab_size": 2, "label_names": ["a", "b"],
    }
    path = tmp_path / "checkpoint_task_1.pt"

    save_checkpoint(
        path, task_id=1, epochs_per_task=5, warmup_epochs=2, seed=42,
        model=model, classifier=classifier, config=config,
    )

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["checkpoint_version"] == 1
    assert checkpoint["task_id"] == 1
    assert checkpoint["epochs_per_task"] == 5
    assert checkpoint["warmup_epochs"] == 2
    assert checkpoint["random_seed"] == 42
    assert checkpoint["config"] == config
    assert "timestamp" in checkpoint
    assert "git_commit" in checkpoint

    fresh_model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1,
    )
    fresh_model.encoders = NodeFeatureEncoders(
        8, protocol_vocab_size=2, service_vocab_size=2,
        flow_feature_dim=FLOW_DIM, host_feature_dim=HOST_DIM,
    )
    fresh_model.load_state_dict(checkpoint["model_state_dict"])  # no error

    fresh_classifier = torch.nn.Linear(8, 2)
    fresh_classifier.load_state_dict(checkpoint["classifier_state_dict"])  # no error
