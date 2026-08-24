"""Tests for transferability-weighted relation distillation.

The two properties worth defending with tests are the two that the EWC
formulation this replaces provably lacked (see
``trench_ids.cl.distillation``'s module docstring): the weights must reach
*all* the parameters, and they must not be able to collapse. Everything
else here is ordinary mechanism coverage.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from trench_ids.cl.distillation import (
    RelationDistiller,
    boundary_relation_drift,
    penalty_attribution,
    transferability_weights,
)
from trench_ids.cl.ewc import FLOW_RELATIONS
from trench_ids.model.rhgnn import NodeFeatureEncoders, RelationSpecificHeteroGNN

FLOW_DIM = 3
HOST_DIM = 2


def _tiny_graph() -> HeteroData:
    """Minimal HeteroData mirroring the real 5-node-type/11-relation schema."""
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


def _tiny_model(graph: HeteroData) -> RelationSpecificHeteroGNN:
    model = RelationSpecificHeteroGNN.from_graph(
        graph, hidden_dim=8, protocol_vocab_size=2, service_vocab_size=2, num_layers=1
    )
    # from_graph reads the schema off the graph but not the feature widths,
    # which default to the real 37/4; swap in encoders sized for the tiny
    # fixture, same pattern as tests/test_train.py.
    model.encoders = NodeFeatureEncoders(
        8,
        protocol_vocab_size=2,
        service_vocab_size=2,
        flow_feature_dim=FLOW_DIM,
        host_feature_dim=HOST_DIM,
    )
    return model


def test_uniform_weighting_is_equal_and_normalised() -> None:
    s_r = {"a": 0.9, "b": -0.4, "c": 0.1}

    weights = transferability_weights(s_r, mode="uniform")

    assert set(weights) == {"a", "b", "c"}
    assert weights["a"] == pytest.approx(1 / 3)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_transfer_weighting_favours_the_most_transferable_relation() -> None:
    s_r = {"low": -0.5, "high": 0.5, "mid": 0.0}

    weights = transferability_weights(s_r, mode="transfer", temperature=0.1)

    assert weights["high"] > weights["mid"] > weights["low"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_inverse_weighting_is_the_exact_mirror_of_transfer() -> None:
    """The sign control has to be a genuine mirror, or a null result between
    the two modes could be blamed on them not actually differing."""
    s_r = {"low": -0.5, "high": 0.5, "mid": 0.0}

    transfer = transferability_weights(s_r, mode="transfer", temperature=0.1)
    inverse = transferability_weights(s_r, mode="inverse", temperature=0.1)

    assert inverse["low"] == pytest.approx(transfer["high"])
    assert inverse["high"] == pytest.approx(transfer["low"])
    assert sum(inverse.values()) == pytest.approx(1.0)


def test_drift_weighting_favours_the_most_drifted_relation() -> None:
    """The reformulation drift.py motivates: transferability is anti-
    correlated with per-relation drift, so weight by drift directly. The
    scores here are D_r values (bigger = moved more), weighted positively."""
    d_r = {"still": 0.03, "moving": 0.19, "mid": 0.08}

    weights = transferability_weights(d_r, mode="drift", temperature=0.1)

    assert weights["moving"] > weights["mid"] > weights["still"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_drift_mode_is_not_the_mirror_of_anything() -> None:
    """Drift's hypothesis is one-sided -- preserve what moves -- so it must
    weight positively even for scores that would flip under ``inverse``."""
    d_r = {"low": 0.03, "high": 0.19}

    drift = transferability_weights(d_r, mode="drift", temperature=1.0)
    inverse = transferability_weights(d_r, mode="inverse", temperature=1.0)

    assert drift["high"] > drift["low"]
    assert inverse["high"] < inverse["low"]


def test_every_mode_applies_the_same_total_regularisation_pressure() -> None:
    """Weights are a softmax, so mode changes *where* the pressure goes, never
    *how much* there is -- which is what makes the ablation interpretable."""
    s_r = {r: (i - 2) * 0.3 for i, r in enumerate(FLOW_RELATIONS)}

    for mode in ("uniform", "transfer", "inverse", "drift"):
        assert sum(transferability_weights(s_r, mode=mode).values()) == pytest.approx(1.0)


def test_unknown_weighting_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="weighting must be one of"):
        transferability_weights({"a": 1.0}, mode="magic")


def test_weights_are_plain_floats_and_carry_no_gradient() -> None:
    """The failure mode this replaces: an importance weight left attached to
    the autograd graph appears only as a multiplier on its own penalty, so
    gradient descent drives it to zero (measured: <1.5e-5 by task 4). These
    weights cannot be optimised at all."""
    s_r = {r: torch.tensor(0.2, requires_grad=True) for r in FLOW_RELATIONS}

    weights = transferability_weights(s_r, mode="transfer")

    assert all(isinstance(w, float) for w in weights.values())


def test_penalty_is_zero_before_any_teacher_exists() -> None:
    graph = _tiny_graph()
    model = _tiny_model(graph)
    distiller = RelationDistiller(FLOW_RELATIONS)

    output = model(graph)

    assert not distiller.active
    assert float(distiller.penalty(graph, output.relations["flow"])) == 0.0


def test_penalty_is_zero_when_the_student_still_matches_the_teacher() -> None:
    graph = _tiny_graph()
    model = _tiny_model(graph)
    distiller = RelationDistiller(FLOW_RELATIONS)
    distiller.snapshot(model)

    output = model(graph)
    penalty = distiller.penalty(graph, output.relations["flow"])

    assert distiller.active
    assert float(penalty) == pytest.approx(0.0, abs=1e-5)


def test_penalty_grows_once_the_student_drifts_from_the_teacher() -> None:
    graph = _tiny_graph()
    model = _tiny_model(graph)
    distiller = RelationDistiller(FLOW_RELATIONS)
    distiller.snapshot(model)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    penalty = distiller.penalty(graph, model(graph).relations["flow"])

    assert float(penalty) > 1e-3


def test_penalty_is_differentiable_into_the_encoder() -> None:
    graph = _tiny_graph()
    model = _tiny_model(graph)
    distiller = RelationDistiller(FLOW_RELATIONS)
    distiller.snapshot(model)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    penalty = distiller.penalty(graph, model(graph).relations["flow"])
    penalty.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_the_teacher_is_frozen_and_excluded_from_optimisation() -> None:
    graph = _tiny_graph()
    model = _tiny_model(graph)
    distiller = RelationDistiller(FLOW_RELATIONS)
    distiller.snapshot(model)

    assert distiller.teacher is not None
    assert all(not p.requires_grad for p in distiller.teacher.parameters())
    assert not distiller.teacher.training
    # And it is a copy, not an alias: mutating the student must not move it.
    before = next(iter(distiller.teacher.parameters())).clone()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    assert torch.equal(next(iter(distiller.teacher.parameters())), before)


def test_lambda_d_scales_the_penalty_linearly() -> None:
    graph = _tiny_graph()
    model = _tiny_model(graph)
    weak = RelationDistiller(FLOW_RELATIONS, lambda_d=1.0)
    strong = RelationDistiller(FLOW_RELATIONS, lambda_d=3.0)
    weak.snapshot(model)
    strong.snapshot(model)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    relations = model(graph).relations["flow"]

    assert float(strong.penalty(graph, relations)) == pytest.approx(
        3.0 * float(weak.penalty(graph, relations)), rel=1e-5
    )


def test_relation_terms_reach_the_shared_encoder_parameters() -> None:
    """**The structural test.** A Fisher-mass audit of the EWC formulation
    found 97.8% of the penalty mass in the shared group and 2.2% across all
    five relation groups combined -- so the relation weights could not steer
    the objective no matter how they were set. Here each relation's term
    back-propagates through the shared encoder, so a large majority of
    parameters must receive gradient from at least one relation term.
    """
    graph = _tiny_graph()
    model = _tiny_model(graph)
    distiller = RelationDistiller(FLOW_RELATIONS)
    distiller.snapshot(model)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    classifier = torch.nn.Linear(8, 2)
    attribution = penalty_attribution(model, graph, distiller, classifier=classifier)

    # Like-for-like against EWC's 2.2%: of the parameters the classification
    # loss can reach, the relation terms reach the overwhelming majority.
    # The only exception is Flow's semantic-attention fusion module (3
    # tensors), which sits *downstream* of the per-relation embeddings the
    # penalty is defined on -- a real and deliberate gap, not an oversight:
    # regularising it would mean constraining the fused output, which is a
    # different (non-relation-specific) claim. Parameters outside the
    # supervised set are the six non-Flow relations, unreachable from any
    # loss at num_layers=1 -- an architectural property that constrains both
    # mechanisms identically, which is why they are excluded from the
    # denominator.
    assert attribution["coverage_of_supervised"] > 0.85
    assert attribution["supervised_fraction"] > 0.0
    # Every Flow relation contributes a non-trivial gradient of its own.
    per_relation = [attribution[r] for r in FLOW_RELATIONS if r in attribution]
    assert len(per_relation) == len(FLOW_RELATIONS)
    assert all(norm > 0.0 for norm in per_relation)


def test_set_weights_restricts_to_the_distiller_s_relations() -> None:
    distiller = RelationDistiller(["originates", "protocol_of"])

    weights = distiller.set_weights(
        {"originates": 0.5, "protocol_of": -0.5, "not_mine": 9.0}
    )

    assert set(weights) == {"originates", "protocol_of"}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_boundary_relation_drift_is_zero_for_an_unchanged_student() -> None:
    """At a task boundary *before any training*, the student is byte-for-byte
    the teacher -- so every D_r must read zero, not noise."""
    graph = _tiny_graph()
    model = _tiny_model(graph)
    distiller = RelationDistiller(FLOW_RELATIONS)
    distiller.snapshot(model)

    d_r = boundary_relation_drift(distiller.teacher, model, [graph], torch.device("cpu"))

    assert set(d_r) == set(FLOW_RELATIONS)
    assert all(value == pytest.approx(0.0, abs=1e-6) for value in d_r.values())


def test_boundary_relation_drift_grows_when_the_student_moves() -> None:
    graph = _tiny_graph()
    model = _tiny_model(graph)
    distiller = RelationDistiller(FLOW_RELATIONS)
    distiller.snapshot(model)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    d_r = boundary_relation_drift(distiller.teacher, model, [graph], torch.device("cpu"))

    assert len(d_r) == len(FLOW_RELATIONS)
    assert all(value > 1e-4 for value in d_r.values())


def test_boundary_relation_drift_restores_the_student_s_training_mode() -> None:
    """The measurement runs under eval (dropout would measure stochasticity,
    not drift); a training loop calling it mid-task must get its train-mode
    student back."""
    graph = _tiny_graph()
    model = _tiny_model(graph)
    model.train()
    teacher = _tiny_model(graph)

    boundary_relation_drift(teacher, model, [graph], torch.device("cpu"))

    assert model.training


def test_a_drift_weighted_distiller_sets_its_weights_from_measured_d_r() -> None:
    """End-to-end shape of the new arm: snapshot -> student moves -> live
    measurement -> softmax weights, agreeing with the pure-function path."""
    graph = _tiny_graph()
    model = _tiny_model(graph)
    distiller = RelationDistiller(FLOW_RELATIONS, weighting="drift", temperature=0.3)
    distiller.snapshot(model)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    d_r = boundary_relation_drift(distiller.teacher, model, [graph], torch.device("cpu"))
    weights = distiller.set_weights(d_r)

    assert weights == transferability_weights(d_r, mode="drift", temperature=0.3)
    assert sum(weights.values()) == pytest.approx(1.0)


def _teacher_of(graph: HeteroData):
    """A distiller whose teacher is a snapshot of a freshly-built model."""
    model = _tiny_model(graph)
    head = torch.nn.Linear(8, 4)
    distiller = RelationDistiller(FLOW_RELATIONS, lambda_d=1.0, objective="logit")
    distiller.snapshot(model, head)
    return model, head, distiller


def test_logit_objective_ignores_classes_the_teacher_never_saw() -> None:
    """The head spans the whole label space from task 1, so at task t the
    teacher has untrained logits for task t's classes. Including them in the
    KL makes the penalty argue "this is an old class" on exactly the rows
    whose label is a new one. Perturbing only the new classes' logits must
    therefore leave the penalty untouched."""
    torch.manual_seed(0)
    graph = _tiny_graph()
    model, head, distiller = _teacher_of(graph)
    distiller.set_old_classes([0, 1])  # classes 2 and 3 belong to the current task

    student_relations = model(graph).relations["flow"]
    before = float(distiller.penalty(graph, student_relations, classifier=head))

    with torch.no_grad():  # shift only the new classes' rows of the head
        head.weight[2:] += 5.0
        head.bias[2:] += 5.0
    after = float(distiller.penalty(graph, student_relations, classifier=head))

    assert after == pytest.approx(before, abs=1e-6)


def test_logit_objective_still_responds_to_the_old_classes() -> None:
    """The complement of the test above: restricting to old classes must not
    disable the term altogether."""
    torch.manual_seed(0)
    graph = _tiny_graph()
    model, head, distiller = _teacher_of(graph)
    distiller.set_old_classes([0, 1])

    student_relations = model(graph).relations["flow"]
    before = float(distiller.penalty(graph, student_relations, classifier=head))

    with torch.no_grad():
        head.weight[0] += 5.0
        head.bias[0] += 5.0
    after = float(distiller.penalty(graph, student_relations, classifier=head))

    assert after > before + 1e-4


def test_unrestricted_penalty_does_react_to_new_class_logits() -> None:
    """Pins the defect this fix addresses: without set_old_classes the
    penalty is sensitive to logits the teacher was never trained to produce.
    If this ever stops being true the restriction has become a no-op and the
    two tests above would pass vacuously."""
    torch.manual_seed(0)
    graph = _tiny_graph()
    model, head, distiller = _teacher_of(graph)
    distiller.set_old_classes(None)

    student_relations = model(graph).relations["flow"]
    before = float(distiller.penalty(graph, student_relations, classifier=head))

    with torch.no_grad():
        head.weight[2:] += 5.0
        head.bias[2:] += 5.0
    after = float(distiller.penalty(graph, student_relations, classifier=head))

    assert abs(after - before) > 1e-4


def test_a_single_old_class_disables_the_logit_term() -> None:
    """At task 1 only Benign has been seen; a softmax over one class is a
    constant, so the term is zero and carries no gradient. It should be
    skipped rather than silently added as a no-op."""
    torch.manual_seed(0)
    graph = _tiny_graph()
    model, head, distiller = _teacher_of(graph)
    distiller.set_old_classes([0])

    student_relations = model(graph).relations["flow"]

    assert float(distiller.penalty(graph, student_relations, classifier=head)) == pytest.approx(0.0)
