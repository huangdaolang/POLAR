"""Bayesian-optimization acquisitions in one place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class BOAcquisitionInputs:
    """Inputs required by BO acquisition functions."""

    candidate_x: torch.Tensor
    predictive_mean: torch.Tensor
    predictive_variance: torch.Tensor
    best_f: torch.Tensor | float | None = None
    maximize: bool = True
    predictive_samples: torch.Tensor | None = None
    mc_num_samples: int = 128


BOAcquisitionFn = Callable[[BOAcquisitionInputs], torch.Tensor]

# UCB hyperparameter is intentionally hardcoded for now.
UCB_ALPHA = 1.0
DEFAULT_EI_MC_SAMPLES = 128


def bo_ucb(inputs: BOAcquisitionInputs) -> torch.Tensor:
    """Upper confidence bound."""
    std = torch.sqrt(inputs.predictive_variance.clamp_min(1e-12))
    return inputs.predictive_mean + UCB_ALPHA * std


def bo_random(inputs: BOAcquisitionInputs) -> torch.Tensor:
    """Random acquisition baseline."""
    n_candidates = int(inputs.candidate_x.shape[0])
    if n_candidates <= 0:
        raise ValueError("candidate_x must contain at least one candidate.")
    return torch.rand(
        n_candidates,
        device=inputs.candidate_x.device,
        dtype=inputs.predictive_mean.dtype,
    )


def bo_ei(inputs: BOAcquisitionInputs) -> torch.Tensor:
    """Analytical Gaussian expected improvement."""
    mean = inputs.predictive_mean
    variance = inputs.predictive_variance
    best_f = _resolve_best_f(inputs, dtype=mean.dtype, device=mean.device)
    if best_f.ndim > 0:
        best_f = best_f.squeeze()

    std = torch.sqrt(variance.clamp_min(0.0))
    eps = torch.tensor(1e-12, device=mean.device, dtype=mean.dtype)
    safe_std = torch.clamp(std, min=float(eps.item()))

    if inputs.maximize:
        improvement_mean = mean - best_f
    else:
        improvement_mean = best_f - mean

    z = improvement_mean / safe_std
    normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    ei = improvement_mean * normal.cdf(z) + safe_std * torch.exp(normal.log_prob(z))

    deterministic = torch.clamp(improvement_mean, min=0.0)
    return torch.where(std <= eps, deterministic, ei)


def _resolve_best_f(inputs: BOAcquisitionInputs, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if inputs.best_f is None:
        raise ValueError("EI acquisition requires `best_f` in BOAcquisitionInputs.")
    if torch.is_tensor(inputs.best_f):
        return inputs.best_f.to(device=device, dtype=dtype)
    return torch.tensor(float(inputs.best_f), device=device, dtype=dtype)


def _resolve_mc_samples(inputs: BOAcquisitionInputs) -> torch.Tensor:
    n_candidates = int(inputs.candidate_x.shape[0])
    if n_candidates <= 0:
        raise ValueError("candidate_x must contain at least one candidate.")

    if inputs.predictive_samples is not None:
        samples = inputs.predictive_samples
        if samples.ndim != 2:
            raise ValueError(
                "predictive_samples must be 2D [num_samples, num_candidates], "
                f"got shape {tuple(samples.shape)}."
            )
        if samples.shape[1] != n_candidates:
            raise ValueError(
                "predictive_samples second dimension must match num_candidates: "
                f"{samples.shape[1]} vs {n_candidates}."
            )
        return samples

    num_samples = int(inputs.mc_num_samples) if inputs.mc_num_samples is not None else DEFAULT_EI_MC_SAMPLES
    if num_samples <= 0:
        raise ValueError("mc_num_samples must be positive.")

    mean = inputs.predictive_mean
    std = torch.sqrt(inputs.predictive_variance.clamp_min(1e-12))
    eps = torch.randn(num_samples, n_candidates, device=mean.device, dtype=mean.dtype)
    return mean.unsqueeze(0) + std.unsqueeze(0) * eps


def bo_ei_mc(inputs: BOAcquisitionInputs) -> torch.Tensor:
    """Monte Carlo expected improvement (maximize by default)."""
    samples = _resolve_mc_samples(inputs)
    best_f = _resolve_best_f(inputs, dtype=samples.dtype, device=samples.device)

    if best_f.ndim > 0:
        best_f = best_f.squeeze()

    if inputs.maximize:
        improvements = samples - best_f
    else:
        improvements = best_f - samples
    return torch.clamp(improvements, min=0.0).mean(dim=0)


_BO_ACQUISITION_REGISTRY: dict[str, BOAcquisitionFn] = {
    "random": bo_random,
    "ei": bo_ei,
    "ucb": bo_ucb,
    "ei_mc": bo_ei_mc,
}


def list_bo_acquisitions() -> list[str]:
    """List registered BO acquisitions."""
    return sorted(_BO_ACQUISITION_REGISTRY.keys())


def get_bo_acquisition(name: str) -> BOAcquisitionFn:
    """Fetch BO acquisition function by name."""
    try:
        return _BO_ACQUISITION_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_bo_acquisitions())
        raise KeyError(f"Unknown BO acquisition '{name}'. Available: {available}") from exc


def compute_bo_acquisition_scores(
    *,
    name: str,
    inputs: BOAcquisitionInputs,
) -> torch.Tensor:
    """Compute acquisition scores for BO candidates."""
    fn = get_bo_acquisition(name)
    scores = fn(inputs)
    if scores.ndim != 1:
        raise ValueError(f"Acquisition '{name}' must return 1D scores, got shape {tuple(scores.shape)}")
    if scores.shape[0] != inputs.candidate_x.shape[0]:
        raise ValueError(
            f"Acquisition '{name}' returned {scores.shape[0]} scores but candidates have "
            f"{inputs.candidate_x.shape[0]} points."
        )
    return scores
