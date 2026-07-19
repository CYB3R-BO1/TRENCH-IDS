"""Step 5 — transferability estimation.

CLAUDE.md's proposed pipeline, step 5: "compare the new attack classes'
per-relation mean embeddings against all previously stored means via
cosine similarity, per relation." Comparison is **same relation only** —
different relations are different learned subspaces (each has its own
``rel_lins``/``combine_lins`` weights, see ``trench_ids.model.relation_conv``),
so a cross-relation cosine similarity wouldn't mean anything. No
aggregation into a single transferability score across relations — that's
step 6 (relation importance weights), still an open item (CLAUDE.md's Open
Items), explicitly out of scope here.
"""

from __future__ import annotations

import torch


def relation_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two same-shaped vectors.

    Returns 0.0 if either vector is all-zero (cosine similarity is
    undefined there, not an error worth raising over).
    """
    a_norm = a.norm()
    b_norm = b.norm()
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(torch.dot(a, b) / (a_norm * b_norm))


def estimate_transferability(
    new_means: dict[str, dict[str, torch.Tensor]],
    bank: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    """For every new class, compare its per-relation means against every
    class already in the bank, one relation at a time.

    Returns ``{new_class: {old_class: {relation: cosine_similarity}}}``.
    Empty if `bank` is empty (e.g. the first task — nothing to compare
    against yet, expected, not a bug) or if a (new_class, old_class) pair
    shares no relation name.
    """
    report: dict[str, dict[str, dict[str, float]]] = {}
    for new_class, new_relations in new_means.items():
        report[new_class] = {}
        for old_class, old_relations in bank.items():
            shared_relations = new_relations.keys() & old_relations.keys()
            report[new_class][old_class] = {
                relation: relation_cosine_similarity(
                    new_relations[relation], old_relations[relation]
                )
                for relation in shared_relations
            }
    return report
