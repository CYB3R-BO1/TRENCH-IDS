"""Tests for memory-bank novelty detection."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from trench_ids.cl.novelty import (
    build_prototypes,
    compute_benign_prototype_means,
    detection_metrics,
    logit_scores,
    prototype_novelty,
)
from trench_ids.cl.novelty import run as run_novelty
from trench_ids.cl.train import save_checkpoint
from trench_ids.model.flat import FlatFlowEncoder
from trench_ids.model.rhgnn import (
    FLOW_FEATURE_DIM,
    HOST_FEATURE_DIM,
    NodeFeatureEncoders,
    RelationSpecificHeteroGNN,
)

DEVICE = torch.device("cpu")
FLOW_DIM = 3
HOST_DIM = 2


def _tiny_graph(y: list[int]) -> HeteroData:
    """Minimal HeteroData mirroring the real 5-node-type/11-relation schema,
    same construction as tests/test_distillation.py's fixture of the same
    name."""
    n = len(y)
    g = HeteroData()
    g["flow"].x = torch.randn(n, FLOW_DIM)
    g["flow"].y = torch.tensor(y)
    g["host"].num_nodes = 2
    g["host"].x = torch.randn(2, HOST_DIM)
    g["protocol"].num_nodes = 1
    g["protocol"].vocab_id = torch.tensor([0])
    g["service"].num_nodes = 1
    g["service"].vocab_id = torch.tensor([0])
    g["port"].num_nodes = 2
    g["port"].port_number = torch.tensor([22, 80])

    src_host = torch.arange(n) % 2
    dst_host = 1 - src_host
    protocol_idx = torch.zeros(n, dtype=torch.long)
    service_idx = torch.zeros(n, dtype=torch.long)
    port_idx = torch.arange(n) % 2

    g["host", "originates", "flow"].edge_index = torch.stack([src_host, torch.arange(n)])
    g["flow", "terminates_at", "host"].edge_index = torch.stack([torch.arange(n), dst_host])
    g["flow", "targets_port", "port"].edge_index = torch.stack([torch.arange(n), port_idx])
    g["flow", "uses_protocol", "protocol"].edge_index = torch.stack([torch.arange(n), protocol_idx])
    g["flow", "uses_service", "service"].edge_index = torch.stack([torch.arange(n), service_idx])
    g["host", "communicates_with", "host"].edge_index = torch.tensor([[0], [1]])
    g["flow", "originated_by", "host"].edge_index = torch.stack([torch.arange(n), src_host])
    g["host", "terminated_by", "flow"].edge_index = torch.stack([dst_host, torch.arange(n)])
    g["port", "targeted_by", "flow"].edge_index = torch.stack([port_idx, torch.arange(n)])
    g["protocol", "protocol_of", "flow"].edge_index = torch.stack([protocol_idx, torch.arange(n)])
    g["service", "service_of", "flow"].edge_index = torch.stack([service_idx, torch.arange(n)])
    return g


def _tiny_model(graph: HeteroData) -> RelationSpecificHeteroGNN:
    """Same construction as tests/test_distillation.py's fixture of the same
    name -- from_graph reads the schema off the graph but not the feature
    widths, which default to the real 37/4, so the encoders are swapped for
    ones sized to this fixture."""
    model = RelationSpecificHeteroGNN.from_graph(
        graph, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1
    )
    model.encoders = NodeFeatureEncoders(
        8,
        protocol_vocab_size=2,
        service_vocab_size=2,
        flow_feature_dim=FLOW_DIM,
        host_feature_dim=HOST_DIM,
    )
    return model


def _bank() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "DDoS": {
            "originates": torch.tensor([1.0, 0.0]),
            "protocol_of": torch.tensor([0.0, 1.0]),
        },
        "Scanning": {
            "originates": torch.tensor([0.0, 1.0]),
            "protocol_of": torch.tensor([1.0, 0.0]),
        },
    }


def test_prototypes_are_stacked_per_relation_and_normalised() -> None:
    prototypes = build_prototypes(_bank(), ["originates", "protocol_of"], DEVICE)

    assert set(prototypes) == {"originates", "protocol_of"}
    assert prototypes["originates"].shape == (2, 2)
    assert torch.allclose(prototypes["originates"].norm(dim=-1), torch.ones(2))


def test_a_relation_missing_from_every_class_is_absent_not_zero() -> None:
    """A zero row would score cosine 0 and act as a mid-range match, quietly
    dragging every novelty score toward the middle. Absence has to stay
    absence."""
    prototypes = build_prototypes(_bank(), ["originates", "service_of"], DEVICE)

    assert "service_of" not in prototypes


def test_a_flow_sitting_on_a_known_prototype_scores_zero_novelty() -> None:
    prototypes = build_prototypes(_bank(), ["originates"], DEVICE)
    embeds = {"originates": torch.tensor([[3.0, 0.0]])}  # same direction as DDoS

    score = prototype_novelty(embeds, prototypes)

    assert float(score[0]) == pytest.approx(0.0, abs=1e-6)


def test_a_flow_orthogonal_to_every_prototype_scores_maximally_novel() -> None:
    prototypes = build_prototypes({"DDoS": {"originates": torch.tensor([1.0, 0.0, 0.0])}},
                                 ["originates"], DEVICE)
    embeds = {"originates": torch.tensor([[0.0, 1.0, 0.0]])}

    score = prototype_novelty(embeds, prototypes)

    assert float(score[0]) == pytest.approx(1.0, abs=1e-6)


def test_novelty_takes_the_nearest_class_not_the_average() -> None:
    """Distance to the *nearest* prototype is the whole point: a flow that
    matches one known class exactly is not novel just because it looks
    nothing like the others."""
    prototypes = build_prototypes(_bank(), ["originates"], DEVICE)
    embeds = {"originates": torch.tensor([[0.0, 5.0]])}  # exactly Scanning

    assert float(prototype_novelty(embeds, prototypes)[0]) == pytest.approx(0.0, abs=1e-6)


def test_weights_are_renormalised_over_usable_relations_only() -> None:
    """A weight naming a relation with no prototypes must not shrink the
    score -- otherwise a partially-populated bank silently reports
    everything as familiar."""
    prototypes = build_prototypes(_bank(), ["originates"], DEVICE)
    embeds = {"originates": torch.tensor([[0.0, 1.0, 0.0][:2]])}
    weights = {"originates": 0.2, "service_of": 0.8}

    with_weights = prototype_novelty(embeds, prototypes, weights)
    without = prototype_novelty(embeds, prototypes)

    assert float(with_weights[0]) == pytest.approx(float(without[0]))


def test_prototype_novelty_needs_at_least_one_usable_relation() -> None:
    with pytest.raises(ValueError, match="no relation"):
        prototype_novelty({"a": torch.zeros(1, 2)}, {"b": torch.zeros(1, 2)})


def test_every_logit_baseline_is_oriented_so_higher_means_more_novel() -> None:
    confident = torch.tensor([[10.0, 0.0, 0.0]])
    uncertain = torch.tensor([[0.1, 0.0, -0.1]])

    sharp = logit_scores(confident)
    flat = logit_scores(uncertain)

    for name in ("msp", "max_logit", "energy"):
        assert float(flat[name]) > float(sharp[name]), name


def test_detection_metrics_report_perfect_separation_as_auroc_one() -> None:
    known = {"prototype": torch.tensor([0.0, 0.1, 0.2])}
    novel = {"prototype": torch.tensor([0.8, 0.9, 1.0])}

    metrics = detection_metrics(known, novel)["prototype"]

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["fpr_at_95_tpr"] == pytest.approx(0.0)
    assert metrics["mean_novel"] > metrics["mean_known"]


def test_detection_metrics_report_no_separation_as_auroc_one_half() -> None:
    same = torch.tensor([0.0, 0.5, 1.0])

    metrics = detection_metrics({"msp": same}, {"msp": same.clone()})["msp"]

    assert metrics["auroc"] == pytest.approx(0.5)


def test_compute_benign_prototype_means_averages_only_benign_rows() -> None:
    """The memory bank has no Benign entry by design (memory_bank.py), so
    this is the module's own source of a Benign prototype -- it must match
    what RelationMeanAccumulator would compute if it did not skip Benign,
    i.e. the mean embedding restricted to Benign-labeled flow rows only."""
    label_names = ["Benign", "DDoS"]
    graph = _tiny_graph(y=[0, 1, 0, 1])  # 2 Benign, 2 DDoS
    model = _tiny_model(graph)
    model.eval()

    with torch.no_grad():
        embeds = model(graph).relations["flow"]
    benign_mask = graph["flow"].y == 0
    expected = {
        relation: embed[benign_mask].mean(dim=0) for relation, embed in embeds.items()
    }

    means = compute_benign_prototype_means(
        model, [graph], DEVICE, batch_size=8, label_names=label_names
    )

    assert set(means) == set(expected)
    for relation, vector in expected.items():
        assert torch.allclose(means[relation], vector, atol=1e-5)


def test_compute_benign_prototype_means_is_empty_with_no_benign_rows() -> None:
    label_names = ["Benign", "DDoS"]
    graph = _tiny_graph(y=[1, 1, 1, 1])  # no Benign flows at all
    model = _tiny_model(graph)
    model.eval()

    means = compute_benign_prototype_means(
        model, [graph], DEVICE, batch_size=8, label_names=label_names
    )

    assert means == {}


def test_run_rejects_a_non_gnn_checkpoint(tmp_path) -> None:
    """train_flat.py never writes memory_bank.pt (no EWC, no memory bank),
    so pointing this at a flat run would otherwise fail with a bare
    FileNotFoundError from load_memory_bank -- true, but not the actual
    reason. Must raise a clear model_type error first.

    Uses the real 37/4 flow/host feature widths, not this file's tiny
    FLOW_DIM/HOST_DIM fixture size: load_checkpoint rebuilds a fresh
    FlatFlowEncoder from config alone (inference.build_model), which always
    uses the real widths, so the saved state_dict has to match those or
    loading fails on a shape mismatch before this test's guard is even
    reached.
    """
    graph = HeteroData()
    graph["flow"].x = torch.randn(4, FLOW_FEATURE_DIM)
    graph["flow"].y = torch.tensor([0, 1, 0, 1])
    graph["host"].num_nodes = 2
    graph["host"].x = torch.randn(2, HOST_FEATURE_DIM)
    graph["protocol"].num_nodes = 1
    graph["protocol"].vocab_id = torch.tensor([0])
    graph["service"].num_nodes = 1
    graph["service"].vocab_id = torch.tensor([0])
    graph["port"].num_nodes = 2
    graph["port"].port_number = torch.tensor([22, 80])
    src_host = torch.tensor([0, 1, 0, 1])
    dst_host = torch.tensor([1, 0, 1, 0])
    zero4 = torch.tensor([0, 0, 0, 0])
    port_idx = torch.tensor([0, 1, 0, 1])
    graph["flow", "targets_port", "port"].edge_index = torch.stack([torch.arange(4), port_idx])
    graph["flow", "uses_protocol", "protocol"].edge_index = torch.stack([torch.arange(4), zero4])
    graph["flow", "uses_service", "service"].edge_index = torch.stack([torch.arange(4), zero4])
    graph["flow", "originated_by", "host"].edge_index = torch.stack([torch.arange(4), src_host])
    graph["flow", "terminates_at", "host"].edge_index = torch.stack([torch.arange(4), dst_host])

    label_names = ["Benign", "DDoS"]
    model = FlatFlowEncoder(
        hidden_dim=8, protocol_vocab_size=1, service_vocab_size=1,
        flow_feature_dim=FLOW_FEATURE_DIM, host_feature_dim=HOST_FEATURE_DIM,
    ).to(DEVICE)
    classifier = torch.nn.Linear(8, len(label_names)).to(DEVICE)
    config = {
        "hidden_dim": 8, "protocol_vocab_size": 1, "service_vocab_size": 1,
        "port_tail_buckets": 32, "label_names": label_names, "model_type": "flat",
    }

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    torch.save([graph], graphs_dir / "task_1_train.pt")
    save_checkpoint(
        run_dir / "checkpoint_task_6.pt", task_id=6, epochs_per_task=1, warmup_epochs=0,
        seed=42, model=model, classifier=classifier, config=config,
    )

    with pytest.raises(ValueError, match="model_type"):
        run_novelty(
            run_dir=run_dir,
            graphs_dir=graphs_dir,
            unseen_graph_dir=tmp_path,
            out_dir=tmp_path / "out",
            device=DEVICE,
        )


def test_recompute_bank_prototypes_uses_the_final_encoder_and_excludes_benign(tmp_path) -> None:
    """The stored bank's means were computed at each task's own training time
    under a since-drifted encoder; recompute_bank_prototypes must instead
    reproduce, per class, exactly what the *final* checkpoint's encoder
    emits for that class's train flows -- with Benign excluded, like the
    bank itself."""
    from trench_ids.cl.novelty import recompute_bank_prototypes
    from trench_ids.cl.train import load_split
    from trench_ids.labels import BENIGN, attack_classes_for_task, canonical_classes

    label_names = canonical_classes()
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    for task in range(1, 7):
        classes = attack_classes_for_task(task)
        y = [label_names.index(c) for c in classes] + [0]
        torch.save([_tiny_graph(y)], graphs_dir / f"task_{task}_train.pt")

    model = _tiny_model(_tiny_graph([0]))
    means = recompute_bank_prototypes(
        model, graphs_dir, DEVICE, batch_size=2, max_graphs_per_task=0
    )

    # Every non-benign class across all six tasks is present; Benign is not.
    assert BENIGN not in means
    assert set(means) == {
        name for task in range(1, 7) for name in attack_classes_for_task(task)
    }

    # Spot-check one class against a manual pass under the same encoder.
    from torch_geometric.loader import DataLoader

    t1_class = attack_classes_for_task(1)[0]
    scan_idx = label_names.index(t1_class)
    stored = load_split(graphs_dir, 1, "train")[0]
    model.eval()
    with torch.no_grad():
        batch = next(iter(DataLoader([stored], batch_size=2))).to(DEVICE)
        embeds = model(batch).relations["flow"]
    mask = batch["flow"].y == scan_idx
    for relation, embed in embeds.items():
        expected = embed[mask].mean(dim=0)
        assert torch.allclose(means[t1_class][relation], expected, atol=1e-6)
