"""Step 4 — relation-specific memory bank: per-class, per-relation mean Flow embeddings.

CLAUDE.md's proposed pipeline, step 4: "store the mean per-relation
embedding for each attack type in a memory bank." Benign is excluded --
this is an *attack-type* memory, not a general class memory. Keyed by
canonical class name (global identity, stable across tasks) rather than
task number, since Step 5's transferability estimation needs to compare a
new task's classes against every previously-seen class regardless of which
task it came from.

No class repeats across T1-T6 (``trench_ids.labels.CANONICAL_TO_TASK`` is a
one-to-one class->task mapping), so merging a task's newly computed means
into the running bank is a plain insert, never a running average across
tasks.
"""

from __future__ import annotations

from pathlib import Path

import torch

from trench_ids.labels import BENIGN


class RelationMeanAccumulator:
    """Streaming per-class, per-relation mean accumulator over Flow's
    relation-specific embeddings.

    Update incrementally, batch by batch, so a task's full train split can
    be summarized without holding every embedding in memory at once.
    """

    def __init__(self) -> None:
        self._sums: dict[str, dict[str, torch.Tensor]] = {}
        self._counts: dict[str, int] = {}

    def update(
        self,
        relation_embeds: dict[str, torch.Tensor],
        y: torch.Tensor,
        label_names: list[str],
    ) -> None:
        """``relation_embeds``: ``{relation_name: Tensor[N, hidden_dim]}`` for
        Flow (``output.relations["flow"]``). ``y``: ``Tensor[N]`` of global
        class indices (``trench_ids.labels.canonical_classes()`` order,
        matching how Step 2's ``graphs.py`` builds ``graph["flow"].y``).
        """
        for class_idx in y.unique().tolist():
            name = label_names[class_idx]
            if name == BENIGN:
                continue
            mask = y == class_idx
            n = int(mask.sum().item())
            self._counts[name] = self._counts.get(name, 0) + n
            bucket = self._sums.setdefault(name, {})
            for relation, embed in relation_embeds.items():
                contribution = embed[mask].sum(dim=0).detach()
                if relation in bucket:
                    bucket[relation] = bucket[relation] + contribution
                else:
                    bucket[relation] = contribution.clone()

    def means(self) -> dict[str, dict[str, torch.Tensor]]:
        """``{class_name: {relation_name: mean_vector}}``."""
        return {
            name: {relation: total / self._counts[name] for relation, total in relations.items()}
            for name, relations in self._sums.items()
        }


def merge_into_bank(
    bank: dict[str, dict[str, torch.Tensor]],
    new_means: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Add a task's newly computed per-class means into the running bank.

    Raises if a class is already present -- each attack class belongs to
    exactly one task (``trench_ids.labels.CANONICAL_TO_TASK``), so a repeat
    means the caller passed the same task's means twice, not a legitimate
    update.
    """
    merged = dict(bank)
    for class_name, relations in new_means.items():
        if class_name in merged:
            raise ValueError(
                f"Class {class_name!r} is already in the memory bank -- each attack "
                "class should belong to exactly one task."
            )
        merged[class_name] = relations
    return merged


def save_memory_bank(bank: dict[str, dict[str, torch.Tensor]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)


def load_memory_bank(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    return torch.load(path, weights_only=False)  # trusted, first-party output
