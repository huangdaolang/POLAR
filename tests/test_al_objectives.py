from __future__ import annotations

import torch

from polar.utils import (
    aggregate_target_log_prob,
    compute_policy_target_weights,
    compute_weighted_target_rmse,
)


def test_compute_policy_target_weights_uniform_matches_average() -> None:
    target_y = torch.tensor([[0.2, -0.3, 1.1]], dtype=torch.float32)

    weights = compute_policy_target_weights(target_y, weighting="uniform", beta=3.0)

    expected = torch.full_like(target_y, 1.0 / 3.0)
    assert torch.allclose(weights, expected)


def test_compute_policy_target_weights_exp_y_prefers_larger_targets() -> None:
    target_y = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)

    weights = compute_policy_target_weights(target_y, weighting="exp_y", beta=1.0)

    assert torch.allclose(weights.sum(dim=1), torch.ones(1))
    assert float(weights[0, 2].item()) > float(weights[0, 1].item()) > float(weights[0, 0].item())


def test_compute_policy_target_weights_step_pos_prefers_positive_targets() -> None:
    target_y = torch.tensor([[-2.0, 0.0, 0.5, 3.0]], dtype=torch.float32)

    weights = compute_policy_target_weights(target_y, weighting="step_pos", beta=4.0)

    expected = torch.tensor([[0.1, 0.1, 0.4, 0.4]], dtype=torch.float32)
    assert torch.allclose(weights, expected)


def test_compute_policy_target_weights_step_pos_uses_custom_threshold() -> None:
    target_y = torch.tensor([[-2.0, 0.0, 0.5, 3.0]], dtype=torch.float32)

    weights = compute_policy_target_weights(
        target_y,
        weighting="step_pos",
        beta=4.0,
        threshold=1.0,
    )

    expected = torch.tensor([[1.0, 1.0, 1.0, 4.0]], dtype=torch.float32)
    expected = expected / expected.sum(dim=1, keepdim=True)
    assert torch.allclose(weights, expected)


def test_compute_policy_target_weights_step_pos_requires_beta_above_one() -> None:
    target_y = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    try:
        compute_policy_target_weights(target_y, weighting="step_pos", beta=1.0)
    except ValueError as exc:
        assert "step_pos" in str(exc)
    else:
        raise AssertionError("Expected step_pos weighting to require beta > 1.")


def test_aggregate_target_log_prob_matches_weighted_sum() -> None:
    target_log_prob = torch.tensor([[-3.0, -1.0, -0.5]], dtype=torch.float32)
    target_y = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)

    score = aggregate_target_log_prob(
        target_log_prob,
        target_y,
        weighting="exp_y",
        beta=1.0,
    )
    weights = torch.softmax(target_y, dim=1)
    expected = (weights * target_log_prob).sum(dim=1)

    assert torch.allclose(score, expected)


def test_aggregate_target_log_prob_matches_step_pos_weighted_sum() -> None:
    target_log_prob = torch.tensor([[-3.0, -1.0, -0.5]], dtype=torch.float32)
    target_y = torch.tensor([[-2.0, 0.5, 2.0]], dtype=torch.float32)

    score = aggregate_target_log_prob(
        target_log_prob,
        target_y,
        weighting="step_pos",
        beta=3.0,
    )
    weights = torch.tensor([[1.0, 3.0, 3.0]], dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)
    expected = (weights * target_log_prob).sum(dim=1)

    assert torch.allclose(score, expected)


def test_aggregate_target_log_prob_step_pos_uses_custom_threshold() -> None:
    target_log_prob = torch.tensor([[-3.0, -1.0, -0.5]], dtype=torch.float32)
    target_y = torch.tensor([[0.5, 1.5, 2.0]], dtype=torch.float32)

    score = aggregate_target_log_prob(
        target_log_prob,
        target_y,
        weighting="step_pos",
        beta=3.0,
        threshold=1.0,
    )
    weights = torch.tensor([[1.0, 3.0, 3.0]], dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)
    expected = (weights * target_log_prob).sum(dim=1)

    assert torch.allclose(score, expected)


def test_compute_weighted_target_rmse_uses_same_weights() -> None:
    pred_mean = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
    target_y = torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32)

    rmse = compute_weighted_target_rmse(
        pred_mean,
        target_y,
        weighting="exp_y",
        beta=1.0,
    )
    weights = torch.softmax(target_y, dim=1)
    expected = torch.sqrt((weights * (pred_mean - target_y).square()).sum(dim=1))

    assert torch.allclose(rmse, expected)
