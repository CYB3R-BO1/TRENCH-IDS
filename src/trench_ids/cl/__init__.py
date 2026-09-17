"""Continual-learning steps (4+): classifier training, relation-specific
memory bank, transferability estimation, meta-transfer learning. Operates on
Step 3's ``RelationSpecificHeteroGNN`` output over Step 2's saved mini-graphs.

The CL package intentionally avoids eager imports of the heavier Stage B and
transfer-predictor modules at package import time. Importing the CLI should not
pull in sci-kit-learn or other optional analysis stacks unless the caller
explicitly imports those submodules.
"""

from __future__ import annotations

__all__ = [
    "train_b0_poc",
    "build_dataset",
    "run_fomaml_episode",
    "TaskSignatureExtractor",
    "extract_task_signature",
    "SignatureStats",
    "EpisodeGenerator",
    "generate_b0_meta_data",
    "TransferAdapter",
    "RandomTransferAdapter",
    "fuse_adapted",
]


def __getattr__(name: str):
    """Lazily import the CL entry points as they are accessed."""
    if name == "train_b0_poc":
        from .meta_transfer import train_b0_poc
        return train_b0_poc
    if name == "build_dataset":
        from .transfer_predictor import build_dataset
        return build_dataset
    if name == "run_fomaml_episode":
        from .fomaml import run_fomaml_episode
        return run_fomaml_episode
    if name in {"TaskSignatureExtractor", "extract_task_signature", "SignatureStats"}:
        from .task_signature import TaskSignatureExtractor, SignatureStats, extract_task_signature
        mapping = {
            "TaskSignatureExtractor": TaskSignatureExtractor,
            "extract_task_signature": extract_task_signature,
            "SignatureStats": SignatureStats,
        }
        return mapping[name]
    if name in {"EpisodeGenerator", "generate_b0_meta_data"}:
        from .episode_generator import EpisodeGenerator, generate_b0_meta_data
        mapping = {
            "EpisodeGenerator": EpisodeGenerator,
            "generate_b0_meta_data": generate_b0_meta_data,
        }
        return mapping[name]
    if name in {"TransferAdapter", "RandomTransferAdapter", "fuse_adapted"}:
        from trench_ids.model.transfer_adapter import TransferAdapter, RandomTransferAdapter, fuse_adapted
        mapping = {
            "TransferAdapter": TransferAdapter,
            "RandomTransferAdapter": RandomTransferAdapter,
            "fuse_adapted": fuse_adapted,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
