"""Transferability-weighted relation distillation (TRD).

The replacement for the relation-aware Online EWC of ``ewc.py``, which was
implemented correctly, tested exhaustively, and **does not work** on this
benchmark. Three independent diagnostics, each ruling out a different
alternative explanation, converged on the conclusion that the failure is
structural rather than a tuning problem:

1. **A 5-order-of-magnitude λ sweep** (0.01 → 1000) moved average forgetting
   by less than the seed-to-seed noise band, even at λ values where the
   penalty term exceeded the classification loss.
2. **A ``num_layers=2`` ablation** restored gradient flow to the six
   relations that are unreachable from the loss at depth 1, and changed
   nothing measurable.
3. **A Fisher-mass audit** of a trained checkpoint found 97.8% of the total
   Fisher mass sitting in the ``shared`` parameter group, 2.2% spread across
   all five Flow-relation groups, and exactly 0.0 in the six ``other``
   groups. *The relation weights could only ever modulate 2.2% of the
   penalty they were supposed to steer.*

Finding (3) is the decisive one, and it is not fixable by re-weighting: in a
relation-specific heterogeneous GNN, "the parameters belonging to relation
r" are a thin per-relation projection sitting on top of an encoder whose
parameters are shared by every relation. Parameter-space regularisation
therefore cannot express "preserve what relation r represents" -- almost all
of what relation r represents is computed by weights that are not
attributable to r at all.

**What this module does instead: regularise the function, not the
parameters.** After task *t-1* the encoder is snapshotted as a frozen
teacher. While training task *t*, each relation's Flow-node embedding is
pulled toward what the teacher produced for the same flows:

    L_TRD = Σ_r  w_r · mean_v [ 1 − cos( h_r(v), h̃_r(v) ) ]

where ``h_r(v)`` is the student's embedding of Flow node *v* along relation
*r*, ``h̃_r(v)`` the teacher's, and ``w_r`` a per-relation weight derived
from that task's measured transferability scores.

Two properties follow directly, and are exactly the two the EWC formulation
lacked:

* **The weights modulate the whole penalty.** ``h_r(v)`` is a function of
  *every* upstream parameter -- the shared node-feature encoders included --
  so scaling relation *r*'s term scales the gradient that reaches those
  shared parameters too. There is no 97.8% of the objective sitting outside
  the weights' reach. ``penalty_attribution`` measures this directly rather
  than asserting it.
* **The weights cannot collapse.** ``w_r`` is computed from transferability
  scores that carry no gradient and is never a learned parameter, so the
  degenerate optimum that sank the ``ImportanceMLP`` formulation -- a weight
  that appears only as a multiplier on its own penalty has an unconditional
  incentive to shrink to zero, and did, reaching <1.5e-5 by task 4 -- does
  not exist here.

Distillation as a continual-learning regulariser is standard (Learning
without Forgetting, Li & Hoiem, ECCV 2016; feature-level variants such as
LFL, Jung et al. 2016). What is new here is applying it *per relation* in a
heterogeneous graph and weighting the relations by an independently measured
transferability signal -- which is the project's actual hypothesis, now
attached to a mechanism that can express it.

``weighting`` selects how ``S_r`` becomes ``w_r``, and the three modes form
the ablation ladder the hypothesis needs:

  ``uniform``      every relation weighted equally -- plain per-relation
                   distillation, the "preserve everything" control.
  ``transfer``     ``softmax(S_r / temperature)`` -- the proposed method:
                   preserve the relations whose representations were
                   measured to transfer.
  ``inverse``      ``softmax(-S_r / temperature)`` -- the sign control. If
                   ``transfer`` and ``inverse`` score the same, the
                   transferability signal is not what is doing the work and
                   the hypothesis fails even if the mechanism helps.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

WEIGHTING_MODES = ("uniform", "transfer", "inverse")

# What the per-relation penalty compares.
#
#   cosine  the relation's Flow embedding itself, against the teacher's.
#           Targets representation drift, which is real and large here
#           (0.05-0.60 cosine across one task boundary, depending on
#           relation).
#   logit   what the relation *says about the classes*: the teacher's head
#           applied to the teacher's relation embedding versus the live head
#           applied to the student's, matched by KL. Targets decision-
#           boundary drift, which feature distillation cannot reach -- and
#           which is where this benchmark's residual forgetting actually
#           lives: under replay, T1/T2/T5 hold up while T3 (DDoS +
#           Infiltration) and T4 (DoS + Injection) collapse, i.e. exactly the
#           tasks whose classes are confusable with later ones. That is class
#           interference, not generic drift.
#   both    the sum. The default, because the two failure modes are
#           independent and each term leaves the other unaddressed.
OBJECTIVES = ("cosine", "logit", "both")


def transferability_weights(
    s_r: dict[str, torch.Tensor | float],
    mode: str = "transfer",
    temperature: float = 1.0,
) -> dict[str, float]:
    """Turn per-relation transferability scores into distillation weights.

    The weights are a softmax over relations, so they always sum to 1.0 and
    the total regularisation pressure is identical across modes -- only its
    *distribution* over relations changes. That is what makes the
    ``uniform``/``transfer``/``inverse`` comparison a clean test of the
    hypothesis rather than a comparison of penalty magnitudes (the mistake
    that made the earlier λ sweep hard to interpret: there, changing the
    weighting also changed how much regularisation was applied in total).

    Returned as plain floats, never tensors: these must not carry a
    gradient. See the module docstring for why.
    """
    if mode not in WEIGHTING_MODES:
        raise ValueError(f"weighting must be one of {WEIGHTING_MODES}, got {mode!r}")
    relations = sorted(s_r)
    if not relations:
        return {}
    if mode == "uniform":
        return {relation: 1.0 / len(relations) for relation in relations}
    sign = 1.0 if mode == "transfer" else -1.0

    def as_scalar(value: torch.Tensor | float) -> float:
        # S_r arrives straight off the transferability pass and can still be
        # attached to the graph. Detaching explicitly rather than letting
        # float() do it silently: float() on a requires_grad tensor warns,
        # and the warning is the only thing that would otherwise flag a
        # future caller accidentally routing a live gradient through here.
        return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)

    scores = torch.tensor(
        [sign * as_scalar(s_r[relation]) for relation in relations], dtype=torch.float32
    )
    weights = torch.softmax(scores / temperature, dim=0)
    return {relation: float(w) for relation, w in zip(relations, weights, strict=True)}


class RelationDistiller:
    """Holds the frozen teacher and computes ``L_TRD`` for a batch.

    Lifecycle, once per task, mirroring the existing ``OnlineEWCManager``
    call sites so the training loop's shape does not change:

        distiller.set_weights(s_r)          # after warm-up, S_r now known
        ...                                 # per batch: distiller.penalty(...)
        distiller.snapshot(model)           # end of task -> next task's teacher

    Before the first ``snapshot`` (i.e. during task 1) ``penalty`` returns a
    zero scalar: there is nothing learned yet to preserve, which is a real
    property of the setting rather than a special case worth branching on at
    the call site.
    """

    def __init__(
        self,
        relations: Iterable[str],
        lambda_d: float = 1.0,
        weighting: str = "transfer",
        temperature: float = 1.0,
        objective: str = "both",
        logit_temperature: float = 2.0,
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        self.relations = list(relations)
        self.lambda_d = float(lambda_d)
        self.weighting = weighting
        self.temperature = float(temperature)
        self.objective = objective
        self.logit_temperature = float(logit_temperature)
        self.teacher: nn.Module | None = None
        self.teacher_classifier: nn.Module | None = None
        self.old_class_index: torch.Tensor | None = None
        self.weights: dict[str, float] = {
            relation: 1.0 / len(self.relations) for relation in self.relations
        }

    @property
    def active(self) -> bool:
        """Whether a teacher exists yet (False during task 1)."""
        return self.teacher is not None

    def set_old_classes(self, class_indices: Sequence[int] | None) -> None:
        """Restrict the ``logit`` objective to the classes the teacher was trained on.

        Without this the KL runs over the whole global label space, and the
        head spans all 11 canonical classes from task 1 onward -- so at task
        t the teacher is asked about classes it has never seen. Its softmax
        puts almost no mass on them, and distilling that says "this flow is
        one of the old classes" on precisely the rows whose cross-entropy
        target is a *new* class. The two terms then pull against each other
        on the new task's own data, and raising lambda_d buys more of the
        conflict: penalty strength converts directly into suppression of the
        thing the task is trying to learn.

        Restricting the distribution to old classes is what LwF (Li & Hoiem,
        ECCV 2016) does for the same reason -- the teacher's outputs are only
        meaningful over the label space it was fitted on. Passing ``None``
        restores the unrestricted behaviour, which is only correct when the
        teacher has seen every class.
        """
        if class_indices is None:
            self.old_class_index = None
            return
        self.old_class_index = torch.as_tensor(sorted(set(class_indices)), dtype=torch.long)

    def set_weights(self, s_r: dict[str, torch.Tensor | float]) -> dict[str, float]:
        """Recompute ``w_r`` from this task's transferability scores."""
        restricted = {r: s_r[r] for r in self.relations if r in s_r}
        self.weights = transferability_weights(
            restricted, mode=self.weighting, temperature=self.temperature
        )
        return dict(self.weights)

    def snapshot(self, model: nn.Module, classifier: nn.Module | None = None) -> None:
        """Freeze copies of the current encoder (and head) as the next task's teacher.

        ``deepcopy`` rather than a state-dict clone so the teacher survives
        any in-place mutation of the live model, and ``requires_grad_(False)``
        plus ``eval()`` so it contributes no parameters to the optimiser and
        no dropout/batchnorm nondeterminism to the target.

        The classifier is snapshotted too because the ``logit`` objective
        needs the *old* decision boundary, not the current one read through
        old features -- the point is to preserve what each relation used to
        say about each class.
        """
        teacher = copy.deepcopy(model)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        self.teacher = teacher
        if classifier is not None:
            head = copy.deepcopy(classifier)
            head.eval()
            for param in head.parameters():
                param.requires_grad_(False)
            self.teacher_classifier = head

    def penalty(
        self,
        batch,
        student_relations: dict[str, torch.Tensor],
        classifier: nn.Module | None = None,
    ) -> torch.Tensor:
        """The weighted per-relation penalty, per ``self.objective``.

        ``student_relations`` is ``output.relations["flow"]`` from the live
        model's forward pass on ``batch`` -- passed in rather than recomputed
        so the student's graph is shared with the classification loss.
        ``classifier`` is the live head, needed only by the ``logit``
        objective; without it that term is skipped rather than silently
        computed against a stale head.
        """
        device = next(iter(student_relations.values())).device if student_relations else None
        if self.teacher is None or device is None:
            zero = torch.zeros((), device=device) if device is not None else torch.zeros(())
            return zero
        with torch.no_grad():
            teacher_out = self.teacher(batch)
        teacher_relations = teacher_out.relations["flow"]
        use_logit = self.objective in ("logit", "both") and (
            classifier is not None and self.teacher_classifier is not None
        )
        use_cosine = self.objective in ("cosine", "both")

        old_index = self.old_class_index
        if use_logit and old_index is not None:
            if old_index.numel() < 2:
                # A softmax over fewer than two classes is a constant, so the
                # KL is identically zero and carries no gradient. Skipping is
                # not an optimisation -- computing it would silently add a
                # no-op term and make the logit objective look active when it
                # is not.
                use_logit = False
            else:
                old_index = old_index.to(device)

        total = torch.zeros((), device=device)
        for relation in self.relations:
            student = student_relations.get(relation)
            target = teacher_relations.get(relation)
            if student is None or target is None:
                continue
            weight = self.weights.get(relation, 0.0)
            if use_cosine:
                distance = 1.0 - F.cosine_similarity(student, target, dim=-1)
                total = total + weight * distance.mean()
            if use_logit:
                tau = self.logit_temperature
                with torch.no_grad():
                    teacher_logits = self.teacher_classifier(target)
                    if old_index is not None:
                        teacher_logits = teacher_logits.index_select(-1, old_index)
                    teacher_p = F.softmax(teacher_logits / tau, dim=-1)
                student_logits = classifier(student)
                if old_index is not None:
                    student_logits = student_logits.index_select(-1, old_index)
                student_log_p = F.log_softmax(student_logits / tau, dim=-1)
                # tau^2 keeps the gradient magnitude comparable to the
                # unsoftened case (Hinton et al. 2015) so logit_temperature
                # does not double as a second, hidden lambda.
                kl = F.kl_div(student_log_p, teacher_p, reduction="batchmean") * (tau**2)
                total = total + weight * kl
        return self.lambda_d * total

    def penalty_components(
        self, batch, student_relations: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Per-relation *unweighted* cosine distance, for diagnostics.

        Reported alongside ``weights`` so a run can be audited for the thing
        that killed the EWC formulation: whether the quantity the weights
        multiply is actually non-trivial, per relation, rather than being
        dominated by one term or vanishing entirely.
        """
        if self.teacher is None:
            return {}
        with torch.no_grad():
            teacher_relations = self.teacher(batch).relations["flow"]
            return {
                relation: float(
                    (1.0 - F.cosine_similarity(
                        student_relations[relation], teacher_relations[relation], dim=-1
                    )).mean()
                )
                for relation in self.relations
                if relation in student_relations and relation in teacher_relations
            }


def _reached_mask(loss: torch.Tensor, params: list[torch.Tensor]) -> torch.Tensor:
    """Which of ``params`` receive a non-zero gradient from ``loss``."""
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    return torch.tensor(
        [g is not None and bool(g.abs().sum() > 0) for g in grads], dtype=torch.bool
    )


def penalty_attribution(
    model: nn.Module,
    batch,
    distiller: RelationDistiller,
    classifier: nn.Module | None = None,
) -> dict[str, float]:
    """How much of the model the relation-weighted penalty can actually steer.

    This is the direct answer to the audit that condemned the EWC
    formulation, run against the replacement rather than assumed: there, the
    five relation groups jointly owned 2.2% of the Fisher mass, so no setting
    of the relation weights could move the other 97.8% of the objective.

    The denominator is deliberately **not** "all parameters". At
    ``num_layers=1`` a known architectural property leaves the six non-Flow
    relations' weights (and four of the fusion modules) unreachable from
    *any* loss that reads ``fused["flow"]`` -- the classification loss
    included -- so counting them would understate both mechanisms equally and
    measure the architecture rather than the penalty. When ``classifier`` is
    given, ``coverage_of_supervised`` reports the share of the parameters the
    *classification* loss reaches that the distillation terms also reach;
    that ratio is the like-for-like counterpart of EWC's 2.2%.

    Returns ``{relation: gradient_l2_norm}`` plus:

      ``reachable_fraction``          share of all parameters reached
      ``supervised_fraction``         share reached by the classification
                                      loss (only with ``classifier``)
      ``coverage_of_supervised``      the ratio of the two (only with
                                      ``classifier``)
    """
    if distiller.teacher is None:
        return {}
    with torch.no_grad():
        teacher_relations = distiller.teacher(batch).relations["flow"]
    params = [p for p in model.parameters() if p.requires_grad]
    result: dict[str, float] = {}
    reached = torch.zeros(len(params), dtype=torch.bool)
    for relation in distiller.relations:
        output = model(batch)
        student = output.relations["flow"].get(relation)
        if student is None or relation not in teacher_relations:
            continue
        term = (1.0 - F.cosine_similarity(student, teacher_relations[relation], dim=-1)).mean()
        grads = torch.autograd.grad(term, params, retain_graph=False, allow_unused=True)
        norm = 0.0
        for i, g in enumerate(grads):
            if g is None:
                continue
            norm += float(g.pow(2).sum())
            if g.abs().sum() > 0:
                reached[i] = True
        result[relation] = norm**0.5
    result["reachable_fraction"] = float(reached.sum()) / max(len(params), 1)

    if classifier is not None:
        logits = classifier(model(batch).fused["flow"])
        supervised = _reached_mask(
            F.cross_entropy(logits, batch["flow"].y), params
        )
        n_supervised = int(supervised.sum())
        result["supervised_fraction"] = n_supervised / max(len(params), 1)
        result["coverage_of_supervised"] = (
            float((reached & supervised).sum()) / n_supervised if n_supervised else 0.0
        )
    return result
