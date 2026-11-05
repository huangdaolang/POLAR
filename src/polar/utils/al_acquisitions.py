"""Active-learning acquisitions in one place.

This file intentionally contains all AL acquisition functions used by the project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from polar.utils.al_objectives import (
    normalize_policy_target_weighting,
    validate_policy_target_weight_beta,
    validate_policy_target_weight_threshold,
)


@dataclass(frozen=True)
class ALAcquisitionInputs:
    """Inputs required by AL acquisition functions."""

    backend: str
    query_x: torch.Tensor
    query_mean: torch.Tensor
    query_variance: torch.Tensor
    target_x: torch.Tensor | None = None
    target_mean: torch.Tensor | None = None
    target_variance: torch.Tensor | None = None
    target_query_covariance: torch.Tensor | None = None
    target_weighting: str = "uniform"
    target_weight_beta: float = 1.0
    target_weight_threshold: float = 0.0


ALAcquisitionFn = Callable[[ALAcquisitionInputs], torch.Tensor]

_STEP_POS_POSTERIOR_MC_SAMPLES = 128
_STEP_POS_POSTERIOR_MC_SEED = 0


def al_random(inputs: ALAcquisitionInputs) -> torch.Tensor:
    """Random acquisition baseline."""
    return torch.rand_like(inputs.query_mean)


def al_uncertainty(inputs: ALAcquisitionInputs) -> torch.Tensor:
    """Uncertainty sampling: maximize predictive variance."""
    return inputs.query_variance


def al_variance_reduction(inputs: ALAcquisitionInputs) -> torch.Tensor:  # noqa: ARG001
    """GP variance reduction with one-step closed-form posterior updates."""
    if inputs.backend != "gp":
        raise NotImplementedError("AL variance reduction is currently implemented only for backend='gp'.")
    if inputs.target_query_covariance is None:
        raise ValueError("variance_reduction requires target_query_covariance in ALAcquisitionInputs.")

    covariance = inputs.target_query_covariance.float()
    denom = inputs.query_variance.float().clamp_min(1.0e-12)
    return covariance.square().mean(dim=0) / denom


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable standard-normal CDF."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard-normal PDF."""
    return torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)


def _step_pos_weighted_gaussian_uncertainty(
    target_mean: torch.Tensor,
    target_variance: torch.Tensor,
    *,
    beta: float,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Return U_w for w(y)=beta if y>threshold else 1 and Y ~ N(mean, variance).

    This is the weighted squared-error uncertainty from the paper:
        U_w = E[w(Y) Y^2] - E[w(Y) Y]^2 / E[w(Y)].
    """
    mean = target_mean.to(dtype=torch.float64)
    variance = target_variance.to(dtype=torch.float64).clamp_min(0.0)
    threshold = float(threshold)
    positive_variance = variance > 1.0e-12
    safe_std = torch.sqrt(variance.clamp_min(1.0e-12))
    standardized = (mean - threshold) / safe_std
    prob_pos = _standard_normal_cdf(standardized)
    pdf_pos = _standard_normal_pdf(standardized)
    positive_first_moment = mean * prob_pos + safe_std * pdf_pos
    positive_second_moment = (
        (mean.square() + variance) * prob_pos
        + (mean + threshold) * safe_std * pdf_pos
    )
    deterministic_positive = mean > threshold
    prob_pos = torch.where(
        positive_variance,
        prob_pos,
        deterministic_positive.to(mean.dtype),
    )
    positive_first_moment = torch.where(
        positive_variance,
        positive_first_moment,
        torch.where(deterministic_positive, mean, torch.zeros_like(mean)),
    )
    positive_second_moment = torch.where(
        positive_variance,
        positive_second_moment,
        torch.where(deterministic_positive, mean.square(), torch.zeros_like(mean)),
    )
    beta_minus_one = float(beta) - 1.0
    weighted_zero_moment = 1.0 + beta_minus_one * prob_pos
    weighted_first_moment = mean + beta_minus_one * positive_first_moment
    weighted_second_moment = mean.square() + variance + beta_minus_one * positive_second_moment
    return weighted_second_moment - weighted_first_moment.square() / weighted_zero_moment.clamp_min(1.0e-12)


def _fixed_antithetic_standard_normal_samples(
    *,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic common-random-number samples for posterior-label MC."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_STEP_POS_POSTERIOR_MC_SEED)
    n_base = (n_samples + 1) // 2
    base = torch.randn(n_base, generator=generator, dtype=torch.float64)
    samples = torch.cat([base, -base], dim=0)[:n_samples]
    return samples.to(device=device, dtype=dtype)


def al_weighted_variance_reduction(inputs: ALAcquisitionInputs) -> torch.Tensor:
    """Loss-driven weighted GP variance reduction.

    `exp_y` is the analytic exponential special case from the paper. `step_pos`
    follows the weighted squared-error uncertainty `U_w` and uses a deterministic
    Monte Carlo average over the hypothetical observed query label.
    """
    if inputs.backend != "gp":
        raise NotImplementedError(
            "AL weighted variance reduction is currently implemented only for backend='gp'."
        )
    if inputs.target_mean is None or inputs.target_variance is None:
        raise ValueError("weighted_variance_reduction requires target_mean and target_variance.")
    if inputs.target_query_covariance is None:
        raise ValueError("weighted_variance_reduction requires target_query_covariance.")

    weighting = normalize_policy_target_weighting(inputs.target_weighting)
    if weighting == "uniform":
        return al_variance_reduction(inputs)

    beta = validate_policy_target_weight_beta(weighting=weighting, beta=inputs.target_weight_beta)
    threshold = validate_policy_target_weight_threshold(inputs.target_weight_threshold)
    output_dtype = inputs.query_mean.dtype
    target_mean = inputs.target_mean.to(dtype=torch.float64)
    target_variance = inputs.target_variance.to(dtype=torch.float64).clamp_min(0.0)
    covariance = inputs.target_query_covariance.to(dtype=torch.float64)
    denom = inputs.query_variance.to(dtype=torch.float64).clamp_min(1.0e-12)
    variance_gain = covariance.square() / denom[None, :]

    if weighting == "exp_y":
        # For z ~ N(m, v) and w(z)=exp(beta z), the weighted squared-error
        # uncertainty simplifies exactly to U_w = E[w(z)] * v. Averaging over the
        # hypothetical query label keeps the same pre-query weight factor.
        target_weight = torch.exp(beta * target_mean + 0.5 * (beta**2) * target_variance)
        return (target_weight[:, None] * variance_gain).mean(dim=0).to(dtype=output_dtype)

    prior_uncertainty = _step_pos_weighted_gaussian_uncertainty(
        target_mean,
        target_variance,
        beta=beta,
        threshold=threshold,
    )
    posterior_variance = (target_variance[:, None] - variance_gain).clamp_min(0.0)
    posterior_mean_shift_std = covariance / torch.sqrt(denom)[None, :]
    posterior_eps = _fixed_antithetic_standard_normal_samples(
        n_samples=_STEP_POS_POSTERIOR_MC_SAMPLES,
        device=target_mean.device,
        dtype=target_mean.dtype,
    )
    posterior_mean = (
        target_mean[:, None, None]
        + posterior_mean_shift_std[:, :, None] * posterior_eps[None, None, :]
    )
    posterior_uncertainty = _step_pos_weighted_gaussian_uncertainty(
        posterior_mean,
        posterior_variance[:, :, None],
        beta=beta,
        threshold=threshold,
    )
    expected_posterior_uncertainty = posterior_uncertainty.mean(dim=-1)
    return (prior_uncertainty[:, None] - expected_posterior_uncertainty).mean(dim=0).to(
        dtype=output_dtype
    )


_AL_ACQUISITION_REGISTRY: dict[str, ALAcquisitionFn] = {
    "random": al_random,
    "uncertainty": al_uncertainty,
    "variance_reduction": al_variance_reduction,
    "weighted_variance_reduction": al_weighted_variance_reduction,
}


def list_al_acquisitions() -> list[str]:
    """List registered AL acquisitions."""
    return sorted(_AL_ACQUISITION_REGISTRY.keys())


def get_al_acquisition(name: str) -> ALAcquisitionFn:
    """Fetch acquisition function by name."""
    try:
        return _AL_ACQUISITION_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_al_acquisitions())
        raise KeyError(f"Unknown AL acquisition '{name}'. Available: {available}") from exc


def compute_al_acquisition_scores(
    *,
    name: str,
    inputs: ALAcquisitionInputs,
) -> torch.Tensor:
    """Compute acquisition scores for query candidates."""
    fn = get_al_acquisition(name)
    scores = fn(inputs)
    if scores.ndim != 1:
        raise ValueError(f"Acquisition '{name}' must return 1D scores, got shape {tuple(scores.shape)}")
    if scores.shape[0] != inputs.query_x.shape[0]:
        raise ValueError(
            f"Acquisition '{name}' returned {scores.shape[0]} scores but query has "
            f"{inputs.query_x.shape[0]} points."
        )
    return scores
