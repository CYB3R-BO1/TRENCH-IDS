"""Step 7 — relation importance learning.

A single shared-weight MLP (input dim 1, output dim 1) applied
independently to each Flow relation's aggregated transferability score
``S_r``, producing ``w_r = sigma(MLP(S_r))`` (design §3). Sharing one
network's weights across all 5 relations -- rather than 5 independently
initialized networks, or one 5-in/5-out joint network -- avoids hard-coding
relation identity/order into the weights and needs far fewer per-task
training signals to learn anything (design's ``AskUserQuestion`` decision:
"Shared per-relation MLP").
"""

from __future__ import annotations

import torch
from torch import nn


class ImportanceMLP(nn.Module):
    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, s_r: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {relation: self.net(value.view(1)).view(()) for relation, value in s_r.items()}
