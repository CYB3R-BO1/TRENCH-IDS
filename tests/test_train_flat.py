from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from tests.test_flat_model import _tiny_graph
from trench_ids.cl.inference import DEFAULT_MODEL_TYPE, build_model, load_checkpoint, predict
from trench_ids.cl.train import save_checkpoint, seed_everything
from trench_ids.cl.train_flat import count_parameters, train_one_task
from trench_ids.model.flat import FlatFlowEncoder
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN

FLAT_CONFIG = {
    "model_type": "flat",
    "hidden_dim": 8,
    "mlp_hidden": 16,
    "port_tail_buckets": 32,
    "protocol_vocab_size": 2,
    "service_vocab_size": 8,
    "label_names": ["a", "b"],
}


def _flat_encoder(use_host_features: bool = False) -> FlatFlowEncoder:
    return FlatFlowEncoder(
        hidden_dim=8,
        protocol_vocab_size=2,
        service_vocab_size=8,
        mlp_hidden=16,
        use_host_features=use_host_features,
    )


def _graphs_dir(tmp_path):
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    torch.save([_tiny_graph(), _tiny_graph()], graphs_dir / "task_1_train.pt")
    return graphs_dir


def test_seed_everything_seeds_all_three_generators() -> None:
    """The bug this fixes: only torch was seeded, so the replay buffer's
    random.sample draws were unseeded and runs of the same config were not
    reproducible."""
    seed_everything(7)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(7)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_train_one_task_reduces_loss_and_updates_parameters() -> None:
    torch.manual_seed(0)
    encoder = _flat_encoder()
    classifier = torch.nn.Linear(8, 2)
    graphs = [_tiny_graph() for _ in range(4)]
    before = [p.detach().clone() for p in encoder.parameters()]
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=0.01
    )

    loss = train_one_task(
        encoder, classifier, graphs, torch.device("cpu"), batch_size=2, epochs=3,
        optimizer=optimizer,
    )

    assert loss > 0.0
    assert any(
        not torch.equal(b, a) for b, a in zip(before, encoder.parameters(), strict=True)
    )


@pytest.mark.parametrize("use_host_features", [False, True], ids=["flat", "flat+host"])
def test_flat_checkpoint_round_trips_through_the_shared_evaluation_path(
    tmp_path, use_host_features: bool
) -> None:
    """Both baseline arms must be scored by the *unmodified* evaluation
    code, so their checkpoints have to load through
    inference.load_checkpoint like any GNN run's -- and the reconstructed
    encoder has to be the same arm it was saved as, since a Flat/Flat+Host
    mix-up would silently compare the wrong rungs."""
    device = torch.device("cpu")
    encoder = _flat_encoder(use_host_features=use_host_features)
    classifier = torch.nn.Linear(8, 2)
    graphs_dir = _graphs_dir(tmp_path)
    checkpoint_path = tmp_path / "checkpoint_task_1.pt"
    config = {**FLAT_CONFIG, "use_host_features": use_host_features}

    save_checkpoint(
        checkpoint_path, task_id=1, epochs_per_task=5, warmup_epochs=0, seed=42,
        model=encoder, classifier=classifier, config=config,
    )
    loaded, loaded_classifier, checkpoint = load_checkpoint(checkpoint_path, graphs_dir, device)

    assert isinstance(loaded, FlatFlowEncoder)
    assert checkpoint["config"]["model_type"] == "flat"
    assert loaded.use_host_features is use_host_features

    g = _tiny_graph()
    with torch.no_grad():
        expected = classifier(encoder.eval()(g).fused["flow"])
        actual = loaded_classifier(loaded(g).fused["flow"])
    torch.testing.assert_close(expected, actual)

    result = predict(loaded, loaded_classifier, [g], device, batch_size=2, label_names=["a", "b"])
    assert len(result["y_pred"]) == 4


def test_flat_checkpoints_without_the_host_flag_load_as_the_flat_arm(tmp_path) -> None:
    """Arm-1 checkpoints were written before use_host_features existed; they
    must keep loading as the plain Flat encoder rather than erroring."""
    encoder = _flat_encoder()
    graphs_dir = _graphs_dir(tmp_path)
    checkpoint_path = tmp_path / "checkpoint_task_1.pt"
    assert "use_host_features" not in FLAT_CONFIG

    save_checkpoint(
        checkpoint_path, task_id=1, epochs_per_task=5, warmup_epochs=0, seed=42,
        model=encoder, classifier=torch.nn.Linear(8, 2), config=FLAT_CONFIG,
    )
    loaded, _, _ = load_checkpoint(checkpoint_path, graphs_dir, torch.device("cpu"))

    assert loaded.use_host_features is False


def test_checkpoints_without_model_type_still_build_the_gnn(tmp_path) -> None:
    """Backward compatibility: every checkpoint written before the flat
    baseline existed (runs/replay_seed*_newport, etc.) has no model_type
    key and must keep loading as the GNN."""
    legacy_config = {
        "hidden_dim": 8, "num_layers": 1, "attn_dim": 128, "port_tail_buckets": 32,
        "protocol_vocab_size": 2, "service_vocab_size": 8, "label_names": ["a", "b"],
    }
    assert "model_type" not in legacy_config

    model = build_model(legacy_config, _tiny_graph(), torch.device("cpu"))

    assert isinstance(model, RelationSpecificHeteroGNN)
    assert legacy_config.get("model_type", DEFAULT_MODEL_TYPE) == "rhgnn"


def test_build_model_rejects_an_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_model({"model_type": "nope"}, _tiny_graph(), torch.device("cpu"))


def test_count_parameters_sums_across_modules() -> None:
    encoder = _flat_encoder()
    classifier = torch.nn.Linear(8, 2)

    assert count_parameters(encoder, classifier) == count_parameters(
        encoder
    ) + count_parameters(classifier)
