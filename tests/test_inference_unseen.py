"""Tests for Step 10 Dataset B behavioral analysis
(trench_ids.cl.inference_unseen)."""

from __future__ import annotations

import json as _json

import torch
from torch_geometric.data import HeteroData

from trench_ids.cl.inference_unseen import (
    compute_class_relation_means,
    confidence_stats,
    prediction_distribution,
    similarity_report,
    similarity_rows,
)
from trench_ids.cl.inference_unseen import (
    run as run_inference_unseen,
)
from trench_ids.cl.memory_bank import save_memory_bank
from trench_ids.cl.train import save_checkpoint
from trench_ids.model.rhgnn import FLOW_FEATURE_DIM, HOST_FEATURE_DIM, RelationSpecificHeteroGNN


def _tiny_graph() -> HeteroData:
    g = HeteroData()
    g["flow"].x = torch.randn(4, FLOW_FEATURE_DIM)
    g["flow"].y = torch.tensor([-1, -1, -1, -1])
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
    g["flow", "uses_protocol", "protocol"].edge_index = torch.stack([torch.arange(4), protocol_idx])
    g["flow", "uses_service", "service"].edge_index = torch.stack([torch.arange(4), service_idx])
    g["host", "communicates_with", "host"].edge_index = torch.tensor([[0], [1]])
    g["flow", "originated_by", "host"].edge_index = torch.stack([torch.arange(4), src_host])
    g["host", "terminated_by", "flow"].edge_index = torch.stack([dst_host, torch.arange(4)])
    g["port", "targeted_by", "flow"].edge_index = torch.stack([port_idx, torch.arange(4)])
    g["protocol", "protocol_of", "flow"].edge_index = torch.stack([protocol_idx, torch.arange(4)])
    g["service", "service_of", "flow"].edge_index = torch.stack([service_idx, torch.arange(4)])
    return g


def test_prediction_distribution_sums_to_100_percent() -> None:
    y_pred = [0, 0, 0, 1]
    dist = prediction_distribution(y_pred, ["Benign", "DoS"])

    # round(pct, 4) doesn't guarantee an exact sum to 100.0 in general (e.g.
    # thirds), so compare with a small tolerance rather than exact equality.
    assert abs(sum(row["percentage"] for row in dist) - 100.0) < 0.01
    assert dist[0]["predicted_class"] == "Benign"
    assert dist[0]["count"] == 3
    assert dist[0]["percentage"] == 75.0
    assert dist[1]["predicted_class"] == "DoS"
    assert dist[1]["count"] == 1


def test_prediction_distribution_empty_input() -> None:
    assert prediction_distribution([], ["Benign", "DoS"]) == []


def test_confidence_stats_computes_mean_median_std() -> None:
    y_prob = [[0.9, 0.1], [0.6, 0.4], [1.0, 0.0]]
    stats = confidence_stats(y_prob)

    assert abs(stats["mean_max_softmax"] - (0.9 + 0.6 + 1.0) / 3) < 1e-9
    assert stats["median_max_softmax"] == 0.9
    assert stats["std_max_softmax"] > 0.0


def test_confidence_stats_empty_input() -> None:
    stats = confidence_stats([])
    assert stats == {"mean_max_softmax": 0.0, "median_max_softmax": 0.0, "std_max_softmax": 0.0}


def test_compute_class_relation_means_returns_one_vector_per_relation() -> None:
    device = torch.device("cpu")
    g = _tiny_graph()
    model = RelationSpecificHeteroGNN.from_graph(
        g, hidden_dim=8, protocol_vocab_size=1, service_vocab_size=1, num_layers=1,
    ).to(device)

    means = compute_class_relation_means(model, [g, g], device, batch_size=2)

    assert set(means.keys()) == {"originates", "terminated_by", "targeted_by", "protocol_of", "service_of"}
    for vec in means.values():
        assert vec.shape == (8,)


def test_similarity_report_renests_by_relation() -> None:
    unseen_means = {"Backdoor": {"originates": torch.tensor([1.0, 0.0])}}
    bank = {
        "Bot": {"originates": torch.tensor([1.0, 0.0])},
        "DoS": {"originates": torch.tensor([0.0, 1.0])},
    }

    nested = similarity_report(unseen_means, bank)

    assert nested["Backdoor"]["originates"]["Bot"] == 1.0
    assert nested["Backdoor"]["originates"]["DoS"] == 0.0


def test_similarity_rows_flattens_nested_report() -> None:
    nested = {"Backdoor": {"originates": {"Bot": 0.87, "DoS": 0.1}}}
    rows = similarity_rows(nested)

    assert ("Backdoor", "originates", "Bot", 0.87) in rows
    assert ("Backdoor", "originates", "DoS", 0.1) in rows
    assert len(rows) == 2


def test_run_writes_dataset_a_and_dataset_b_outputs(tmp_path) -> None:
    device = torch.device("cpu")
    g_known = _tiny_graph()
    g_known["flow"].y = torch.tensor([0, 1, 0, 1])  # Dataset A: real labels
    g_known["flow"].true_label = ["Benign", "DoS", "Benign", "DoS"]

    g_unknown = _tiny_graph()  # Dataset B: y already -1 from _tiny_graph()
    g_unknown["flow"].true_label = ["Backdoor"] * 4

    label_names = ["Benign", "DoS"]
    model = RelationSpecificHeteroGNN.from_graph(
        g_known, hidden_dim=8, protocol_vocab_size=1, service_vocab_size=1, num_layers=1,
    ).to(device)
    classifier = torch.nn.Linear(8, len(label_names)).to(device)
    config = {
        "hidden_dim": 8, "num_layers": 1, "attn_dim": 128, "port_tail_buckets": 32,
        "protocol_vocab_size": 1, "service_vocab_size": 1, "label_names": label_names,
    }

    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    torch.save([g_known], graphs_dir / "task_1_train.pt")

    checkpoint_path = tmp_path / "checkpoint_task_6.pt"
    save_checkpoint(
        checkpoint_path, task_id=6, epochs_per_task=1, warmup_epochs=0, seed=42,
        model=model, classifier=classifier, config=config,
    )

    unseen_graphs_dir = tmp_path / "unseen_graphs"
    unseen_graphs_dir.mkdir()
    torch.save([g_known], unseen_graphs_dir / "UNSW_dos.pt")
    torch.save([g_unknown], unseen_graphs_dir / "ToN_backdoor.pt")

    manifest = {
        "classes": {
            "UNSW_dos": {"canonical_label": "DoS", "evaluation_type": "seen_class_unseen_samples"},
            "ToN_backdoor": {"canonical_label": "Backdoor", "evaluation_type": "unseen_class"},
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_json.dumps(manifest))

    bank = {"DoS": {r: torch.randn(8) for r in
        ["originates", "terminated_by", "targeted_by", "protocol_of", "service_of"]}}
    bank_path = tmp_path / "memory_bank.pt"
    save_memory_bank(bank, bank_path)

    out_dir = tmp_path / "unseen_eval"

    result = run_inference_unseen(
        checkpoint_path=checkpoint_path,
        graphs_dir_for_schema=graphs_dir,
        unseen_graphs_dir=unseen_graphs_dir,
        unseen_manifest_path=manifest_path,
        memory_bank_path=bank_path,
        out_dir=out_dir,
        device=device,
        batch_size=2,
    )

    assert (out_dir / "dataset_a_metrics.json").exists()
    assert (out_dir / "dataset_b_predictions.json").exists()
    assert (out_dir / "dataset_b_predictions.csv").exists()
    assert (out_dir / "dataset_b_confidence.json").exists()
    assert (out_dir / "dataset_b_confidence.csv").exists()
    assert (out_dir / "dataset_b_similarity.json").exists()
    assert (out_dir / "dataset_b_similarity.csv").exists()

    assert result["dataset_a"][0]["slug"] == "UNSW_dos"
    assert "accuracy" in result["dataset_a"][0]["metrics"]
    # Finding #2: pooled Dataset A metrics, alongside the per-class list.
    assert "accuracy" in result["dataset_a_pooled"]
    assert "ToN_backdoor" in result["dataset_b_predictions"]
    assert "ToN_backdoor" in result["dataset_b_confidence"]
    # Finding #3: Dataset B outputs are all keyed by slug now (joinable across
    # files), with canonical_label carried as a sibling field for robustness.
    assert "ToN_backdoor" in result["dataset_b_similarity"]
    assert result["dataset_b_similarity"]["ToN_backdoor"]["canonical_label"] == "Backdoor"
    assert result["dataset_b_predictions"]["ToN_backdoor"][0]["canonical_label"] == "Backdoor"
    assert result["dataset_b_confidence"]["ToN_backdoor"]["canonical_label"] == "Backdoor"
