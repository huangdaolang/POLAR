"""Shared objective helpers for loss-driven active learning."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

PolicyTargetWeighting = Literal["uniform", "exp_y", "step_pos"]


def normalize_policy_target_weighting(value: str) -> PolicyTargetWeighting:
    """Validate and normalize the policy target weighting mode."""
    weighting = str(value).strip().lower()
    if weighting not in {"uniform", "exp_y", "step_pos"}:
        raise ValueError(
            "policy_target_weighting must be one of: 'uniform', 'exp_y', 'step_pos'."
        )
    return weighting  # type: ignore[return-value]


def validate_policy_target_weight_beta(
    *,
    weighting: PolicyTargetWeighting,
    beta: float,
) -> float:
    """Validate and normalize the beta parameter for one weighting mode."""
    beta = float(beta)
    if weighting == "step_pos":
        if beta <= 1.0:
            raise ValueError(
                "policy_target_weight_beta must be > 1 when "
                "policy_target_weighting='step_pos'."
            )
        return beta
    if beta < 0.0:
        raise ValueError("policy_target_weight_beta must be >= 0.")
    return beta


def validate_policy_target_weight_threshold(threshold: float) -> float:
    """Validate and normalize the threshold used by thresholded weighting rules."""
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("policy_target_weight_threshold must be finite.")
    return threshold


def compute_policy_target_weights(
    target_y: Tensor,
    *,
    weighting: PolicyTargetWeighting = "uniform",
    beta: float = 1.0,
    threshold: float = 0.0,
) -> Tensor:
    """Return normalized per-target weights aligned with ``target_y``."""
    if target_y.ndim not in (1, 2):
        raise ValueError(
            "target_y must be 1D [N_target] or 2D [B, N_target], "
            f"got {tuple(target_y.shape)}."
        )
    if target_y.shape[-1] <= 0:
        raise ValueError("target_y must contain at least one target point.")

    weighting = normalize_policy_target_weighting(weighting)
    beta = validate_policy_target_weight_beta(weighting=weighting, beta=beta)
    threshold = validate_policy_target_weight_threshold(threshold)
    target_y = target_y.float()
    if weighting == "uniform":
        return torch.full_like(target_y, 1.0 / float(target_y.shape[-1]))

    if weighting == "step_pos":
        raw_weight = torch.where(
            target_y > threshold,
            torch.full_like(target_y, beta),
            torch.ones_like(target_y),
        )
        return raw_weight / raw_weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

    scaled = float(beta) * target_y
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values
    return torch.softmax(scaled, dim=-1)


def aggregate_target_log_prob(
    target_log_prob: Tensor,
    target_y: Tensor,
    *,
    weighting: PolicyTargetWeighting = "uniform",
    beta: float = 1.0,
    threshold: float = 0.0,
) -> Tensor:
    """Aggregate per-target log probabilities into one score per episode."""
    if target_log_prob.shape != target_y.shape:
        raise ValueError(
            "target_log_prob and target_y must have matching shape, "
            f"got {tuple(target_log_prob.shape)} vs {tuple(target_y.shape)}."
        )
    weights = compute_policy_target_weights(
        target_y,
        weighting=weighting,
        beta=beta,
        threshold=threshold,
    )
    return (weights * target_log_prob.float()).sum(dim=-1)


def compute_weighted_target_rmse(
    pred_mean: Tensor,
    target_y: Tensor,
    *,
    weighting: PolicyTargetWeighting = "uniform",
    beta: float = 1.0,
    threshold: float = 0.0,
) -> Tensor:
    """Compute RMSE with policy-aligned target weights."""
    if pred_mean.shape != target_y.shape:
        raise ValueError(
            "pred_mean and target_y must have matching shape, "
            f"got {tuple(pred_mean.shape)} vs {tuple(target_y.shape)}."
        )
    weights = compute_policy_target_weights(
        target_y,
        weighting=weighting,
        beta=beta,
        threshold=threshold,
    )
    sq_error = (pred_mean.float() - target_y.float()).square()
    return torch.sqrt((weights * sq_error).sum(dim=-1).clamp_min(0.0))
