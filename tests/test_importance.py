from __future__ import annotations

import pytest
import torch

from trench_ids.cl.importance import ImportanceMLP


def test_importance_mlp_outputs_scalar_per_relation() -> None:
    mlp = ImportanceMLP()
    s_r = {"originates": torch.tensor(0.5), "terminated_by": torch.tensor(-0.2)}

    w_r = mlp(s_r)

    assert set(w_r.keys()) == {"originates", "terminated_by"}
    for value in w_r.values():
        assert value.shape == ()


def test_importance_mlp_output_is_between_zero_and_one() -> None:
    mlp = ImportanceMLP()
    s_r = {"originates": torch.tensor(3.0), "terminated_by": torch.tensor(-3.0)}

    w_r = mlp(s_r)

    for value in w_r.values():
        assert 0.0 <= value.item() <= 1.0


def test_importance_mlp_shares_weights_across_relations() -> None:
    """Same-valued S_r for two different relations must produce the exact
    same w_r -- proving one shared-weight network is applied independently
    per relation, not five separately-initialized networks."""
    mlp = ImportanceMLP()
    s_r = {"originates": torch.tensor(0.42), "terminated_by": torch.tensor(0.42)}

    w_r = mlp(s_r)

    assert w_r["originates"].item() == pytest.approx(w_r["terminated_by"].item())


def test_importance_mlp_gradients_flow_to_mlp_weights() -> None:
    mlp = ImportanceMLP()
    s_r = {"originates": torch.tensor(0.5)}

    w_r = mlp(s_r)
    w_r["originates"].backward()

    first_linear_weight = mlp.net[0].weight
    assert first_linear_weight.grad is not None
    assert not torch.all(first_linear_weight.grad == 0)
