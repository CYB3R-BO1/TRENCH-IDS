"""Shared torch.device resolution for every CLI/module entry point.

Before 2026-08-23 ``train.py`` owned ``resolve_device`` while five other
modules (drift.py, evaluate.py, inference.py, inference_unseen.py,
novelty.py) each reimplemented the same inline "auto" block. This module
is now the single implementation; ``train.py`` re-exports it so existing
``from trench_ids.cl.train import resolve_device`` imports keep working.
"""

from __future__ import annotations

import torch


def resolve_device(device_cfg: str) -> torch.device:
    """Map a config/CLI device string ("auto" | "cuda" | "cpu" | ...) to a
    ``torch.device``. "auto" picks CUDA when available, else CPU."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)
