"""Sequential EIG utilities for Bayesian Experimental Design.

Provides functions for:
- Dense per-step sPCE rewards for REINFORCE training
- Terminal sPCE / sNMC bounds from cumulative log-likelihoods
- Full-trajectory bound computation from a task log-likelihood callable

All computations are in log-space for numerical stability.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor


def dense_spce_reward(
    step_log_lik: Tensor,
    cum_log_lik: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute the dense sPCE reward for a single design step.

    The per-step reward decomposes the terminal sPCE lower bound into
    incremental contributions:
        r_t = log p(y_t | theta_0, x_t) - logsumexp_l(C_t) + logsumexp_l(C_{t-1})
    where C_t[l] = sum_{k=1}^{t} log p(y_k | theta_l, x_k).

    These rewards telescope:
        sum_t r_t = log p(h_T | theta_0) - logsumexp_l p(h_T | theta_l) + log(L + 1)

    Args:
        step_log_lik: Log-likelihoods of the new observation under each theta.
            Shape ``[L+1, B]`` where index 0 is the true theta.
        cum_log_lik: Cumulative log-likelihoods before this step.
            Shape ``[L+1, B]``. Initialize to zeros at ``t = 0``.

    Returns:
        reward: Per-step dense sPCE reward, shape ``[B]``.
        new_cum_log_lik: Updated cumulative log-likelihoods, shape ``[L+1, B]``.
    """
    prev_logsumexp = torch.logsumexp(cum_log_lik, dim=0)  # [B]
    new_cum_log_lik = cum_log_lik + step_log_lik  # [L+1, B]
    new_logsumexp = torch.logsumexp(new_cum_log_lik, dim=0)  # [B]

    reward = step_log_lik[0] - new_logsumexp + prev_logsumexp  # [B]
    return reward, new_cum_log_lik


def compute_spce(cum_log_lik: Tensor) -> Tensor:
    """Compute the terminal sPCE lower bound from cumulative log-likelihoods.

    sPCE = log p(h_T | theta_0) - log(1 / (L + 1) * sum_l p(h_T | theta_l))
         = cum_log_lik[0] - logsumexp(cum_log_lik) + log(L + 1)

    Args:
        cum_log_lik: Final cumulative log-likelihoods, shape ``[L+1, B]``.

    Returns:
        sPCE values, shape ``[B]``.
    """
    l_plus_1 = cum_log_lik.shape[0]
    return cum_log_lik[0] - torch.logsumexp(cum_log_lik, dim=0) + math.log(l_plus_1)


def compute_snmc(cum_log_lik: Tensor) -> Tensor:
    """Compute the terminal sNMC upper bound from cumulative log-likelihoods.

    sNMC = log p(h_T | theta_0) - log(1 / L * sum_{l=1}^{L} p(h_T | theta_l))
         = cum_log_lik[0] - logsumexp(cum_log_lik[1:]) + log(L)

    Args:
        cum_log_lik: Final cumulative log-likelihoods, shape ``[L+1, B]``.

    Returns:
        sNMC values, shape ``[B]``.
    """
    l_plus_1 = int(cum_log_lik.shape[0])
    if l_plus_1 < 2:
        raise ValueError("sNMC requires at least one contrastive sample (shape[0] >= 2).")
    num_contrastive = l_plus_1 - 1
    return (
        cum_log_lik[0]
        - torch.logsumexp(cum_log_lik[1:], dim=0)
        + math.log(num_contrastive)
    )


def compute_eig_bounds(cum_log_lik: Tensor) -> tuple[Tensor, Tensor]:
    """Compute the terminal sPCE lower bound and sNMC upper bound.

    Args:
        cum_log_lik: Final cumulative log-likelihoods, shape ``[L+1, B]``.

    Returns:
        Tuple ``(spce, snmc)``, each with shape ``[B]``.
    """
    return compute_spce(cum_log_lik), compute_snmc(cum_log_lik)


def compute_cumulative_log_likelihood(
    log_lik_fn: Callable[..., Tensor],
    y_trajectory: Tensor,
    x_trajectory: Tensor,
    theta_all: Tensor,
) -> Tensor:
    """Accumulate joint history log-likelihoods for each contrastive theta.

    Args:
        log_lik_fn: Callable ``(y, x, theta) -> log_prob``.
            Expected shapes: y ``[1, B]``, x ``[1, B, d]``,
            theta ``[L+1, B, ...theta_shape]`` -> ``[L+1, B]``.
        y_trajectory: Observations, shape ``[B, T]``.
        x_trajectory: Designs, shape ``[B, T, d]``.
        theta_all: Contrastive theta samples, shape ``[L+1, B, ...theta_shape]``.

    Returns:
        Final cumulative log-likelihoods, shape ``[L+1, B]``.
    """
    batch_size, steps = y_trajectory.shape
    l_plus_1 = theta_all.shape[0]
    dtype = theta_all.dtype if torch.is_floating_point(theta_all) else torch.float32
    cum_log_lik = torch.zeros(l_plus_1, batch_size, device=y_trajectory.device, dtype=dtype)

    for step_idx in range(steps):
        y_t = y_trajectory[:, step_idx]  # [B]
        x_t = x_trajectory[:, step_idx]  # [B, d]
        step_ll = log_lik_fn(
            y_t.unsqueeze(0),  # [1, B]
            x_t.unsqueeze(0),  # [1, B, d]
            theta_all,
        )  # [L+1, B]
        cum_log_lik = cum_log_lik + step_ll.to(cum_log_lik.dtype)

    return cum_log_lik


def compute_spce_from_trajectory(
    log_lik_fn: Callable[..., Tensor],
    y_trajectory: Tensor,
    x_trajectory: Tensor,
    theta_all: Tensor,
) -> Tensor:
    """Compute the terminal sPCE lower bound for full design trajectories."""
    cum_log_lik = compute_cumulative_log_likelihood(
        log_lik_fn=log_lik_fn,
        y_trajectory=y_trajectory,
        x_trajectory=x_trajectory,
        theta_all=theta_all,
    )
    return compute_spce(cum_log_lik)


def compute_snmc_from_trajectory(
    log_lik_fn: Callable[..., Tensor],
    y_trajectory: Tensor,
    x_trajectory: Tensor,
    theta_all: Tensor,
) -> Tensor:
    """Compute the terminal sNMC upper bound for full design trajectories."""
    cum_log_lik = compute_cumulative_log_likelihood(
        log_lik_fn=log_lik_fn,
        y_trajectory=y_trajectory,
        x_trajectory=x_trajectory,
        theta_all=theta_all,
    )
    return compute_snmc(cum_log_lik)


def compute_eig_bounds_from_trajectory(
    log_lik_fn: Callable[..., Tensor],
    y_trajectory: Tensor,
    x_trajectory: Tensor,
    theta_all: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute the terminal sPCE lower bound and sNMC upper bound for trajectories."""
    cum_log_lik = compute_cumulative_log_likelihood(
        log_lik_fn=log_lik_fn,
        y_trajectory=y_trajectory,
        x_trajectory=x_trajectory,
        theta_all=theta_all,
    )
    return compute_eig_bounds(cum_log_lik)


__all__ = [
    "compute_cumulative_log_likelihood",
    "compute_eig_bounds",
    "compute_eig_bounds_from_trajectory",
    "compute_snmc",
    "compute_snmc_from_trajectory",
    "compute_spce",
    "compute_spce_from_trajectory",
    "dense_spce_reward",
]
