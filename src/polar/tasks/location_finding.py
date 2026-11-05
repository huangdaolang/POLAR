"""Location Finding task sampler for Bayesian Experimental Design.

Implements the hidden source localization problem from Blau et al. (2022):
K hidden signal sources at unknown positions theta in d-dimensional space.
The experimenter selects measurement locations x and observes noisy signal
intensity to infer source positions.

Physical model:
    signal(theta, x) = log(base_signal + sum_k 1 / (max_signal + ||theta_k - x||^2))
    y | theta, x ~ N(signal(theta, x), noise_scale^2)

Prior:
    theta_k ~ N(0, I_d)  (normal)  or  Uniform([0,1]^{K x d})  (uniform)
"""

from __future__ import annotations

import math
from functools import partial
from typing import Literal

import torch
from torch import Tensor

from polar.tasks.bed import BEDEpisode, BEDTrajectoryPlotter

ThetaPrior = Literal["normal", "uniform"]

LocationFindingEpisode = BEDEpisode


def location_finding_log_likelihood(
    y: Tensor,
    x: Tensor,
    theta: Tensor,
    *,
    base_signal: float = 0.1,
    max_signal: float = 1e-4,
    noise_scale: float = 0.5,
) -> Tensor:
    """Analytical log-likelihood for the Location Finding model.

    Computes log N(y | signal(theta, x), noise_scale^2) where
    signal(theta, x) = log(base_signal + sum_k 1/(max_signal + ||theta_k - x||^2)).

    Fully vectorized over contrastive samples and batch dimensions.

    Args:
        y: Observations. Shape ``[*, B]`` (broadcastable over leading dims).
        x: Design locations. Shape ``[*, B, d]`` (broadcastable).
        theta: Source locations. Shape ``[L+1, B, K, d]``.
        base_signal: Background signal constant.
        max_signal: Maximum signal constant (prevents singularity).
        noise_scale: Observation noise std.

    Returns:
        Log-likelihood values, shape ``[L+1, B]``.
    """
    # x: [*, B, d] -> [*, B, 1, d] for broadcasting with theta [L+1, B, K, d]
    x_expanded = x.unsqueeze(-2)  # [*, B, 1, d]

    # Squared distance: [L+1, B, K]
    sq_dist = (theta - x_expanded).pow(2).sum(dim=-1)

    # Signal intensity: [L+1, B]
    signal = torch.log(base_signal + (1.0 / (max_signal + sq_dist)).sum(dim=-1))

    # Log-normal likelihood: [L+1, B]
    log_prob = (
        -0.5 * math.log(2.0 * math.pi)
        - math.log(noise_scale)
        - 0.5 * ((y - signal) / noise_scale).pow(2)
    )
    return log_prob


class LocationFindingSampler:
    """Sample BED episodes for the Location Finding task."""

    def __init__(
        self,
        *,
        dim_x: int = 2,
        K: int = 2,
        n_contrastive: int = 19,
        n_context_init: int = 1,
        n_query: int = 200,
        n_target: int = 0,
        design_min: float = -4.0,
        design_max: float = 4.0,
        base_signal: float = 0.1,
        max_signal: float = 1e-4,
        noise_scale: float = 0.5,
        theta_prior: ThetaPrior = "normal",
        seed: int | None = None,
    ) -> None:
        if dim_x <= 0:
            raise ValueError("dim_x must be positive.")
        if K <= 0:
            raise ValueError("K must be positive.")
        if n_contrastive <= 0:
            raise ValueError("n_contrastive must be positive.")
        if n_context_init < 1:
            raise ValueError("n_context_init must be >= 1.")
        if n_query <= 0:
            raise ValueError("n_query must be positive.")
        if n_target < 0:
            raise ValueError("n_target must be non-negative.")
        if design_max <= design_min:
            raise ValueError("design_max must be larger than design_min.")
        if noise_scale <= 0.0:
            raise ValueError("noise_scale must be positive.")
        if theta_prior not in {"normal", "uniform"}:
            raise ValueError(f"Unsupported theta_prior '{theta_prior}'.")

        self.dim_x = dim_x
        self.name = "location_finding"
        self.observation_type = "continuous"
        self.num_classes = None
        self.K = K
        self.n_contrastive = n_contrastive
        self.n_context_init = n_context_init
        self.n_query = n_query
        self.n_target = n_target
        self.design_min = float(design_min)
        self.design_max = float(design_max)
        self.base_signal = float(base_signal)
        self.max_signal = float(max_signal)
        self.noise_scale = float(noise_scale)
        self.theta_prior = theta_prior

        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(int(seed))

    @property
    def trajectory_plotter(self) -> BEDTrajectoryPlotter | None:
        if self.dim_x != 2:
            return None
        from polar.utils.plotting import plot_location_finding_trajectories

        return partial(
            plot_location_finding_trajectories,
            design_min=self.design_min,
            design_max=self.design_max,
        )

    def sample_episode(self) -> BEDEpisode:
        """Sample one BED episode."""
        # Sample theta: theta_all[0] is true, theta_all[1:] are contrastive
        n_theta = self.n_contrastive + 1
        theta_all = self.sample_theta(n_theta)  # [L+1, K, d]
        theta_true = theta_all[0]  # [K, d]

        # Sample all design points
        n_total = self.n_context_init + self.n_query + self.n_target
        all_x = self._sample_designs(n_total)  # [n_total, d]

        # Generate observations under theta_true
        all_y = self._observe(all_x, theta_true)  # [n_total]

        # Split into context / query / target
        ctx_end = self.n_context_init
        query_end = ctx_end + self.n_query

        context_x = all_x[:ctx_end]  # [n_context_init, d]
        context_y = all_y[:ctx_end]  # [n_context_init]
        query_x = all_x[ctx_end:query_end]  # [n_query, d]
        query_y = all_y[ctx_end:query_end]  # [n_query]

        if self.n_target > 0:
            target_x = all_x[query_end:]  # [n_target, d]
            target_y = all_y[query_end:]  # [n_target]
        else:
            target_x = None
            target_y = None

        return BEDEpisode(
            theta_all=theta_all,
            context_x=context_x,
            context_y=context_y,
            query_x=query_x,
            query_y=query_y,
            target_x=target_x,
            target_y=target_y,
        )

    def sample_batch(self, *, batch_size: int) -> list[BEDEpisode]:
        """Sample a batch of independent BED episodes."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return [self.sample_episode() for _ in range(batch_size)]

    def sample_theta(self, n: int) -> Tensor:
        """Sample n theta vectors from the prior.

        Returns: shape ``[n, K, d]``.
        """
        if self.theta_prior == "normal":
            return torch.randn(n, self.K, self.dim_x, generator=self._rng, dtype=torch.float32)
        else:  # uniform
            return torch.rand(n, self.K, self.dim_x, generator=self._rng, dtype=torch.float32)

    def log_likelihood(self, y: Tensor, x: Tensor, theta: Tensor) -> Tensor:
        """Task-specific analytical log-likelihood."""
        return location_finding_log_likelihood(
            y,
            x,
            theta,
            base_signal=self.base_signal,
            max_signal=self.max_signal,
            noise_scale=self.noise_scale,
        )

    def _sample_designs(self, n: int) -> Tensor:
        """Sample n design points uniformly in [design_min, design_max]^d.

        Returns: shape ``[n, d]``.
        """
        u = torch.rand(n, self.dim_x, generator=self._rng, dtype=torch.float32)
        return self.design_min + (self.design_max - self.design_min) * u

    def _signal(self, x: Tensor, theta: Tensor) -> Tensor:
        """Compute log-mean-signal for given designs and sources.

        Args:
            x: Design locations, shape ``[N, d]``.
            theta: Source locations, shape ``[K, d]``.

        Returns:
            Signal values, shape ``[N]``.
        """
        # x: [N, d] -> [N, 1, d], theta: [K, d] -> [1, K, d]
        sq_dist = (x.unsqueeze(1) - theta.unsqueeze(0)).pow(2).sum(dim=-1)  # [N, K]
        return torch.log(self.base_signal + (1.0 / (self.max_signal + sq_dist)).sum(dim=-1))  # [N]

    def _observe(self, x: Tensor, theta: Tensor) -> Tensor:
        """Sample noisy observations y ~ N(signal(theta, x), noise_scale^2).

        Args:
            x: Design locations, shape ``[N, d]``.
            theta: Source locations, shape ``[K, d]``.

        Returns:
            Observations, shape ``[N]``.
        """
        signal = self._signal(x, theta)  # [N]
        noise = torch.randn(signal.shape, generator=self._rng, dtype=signal.dtype)
        return signal + self.noise_scale * noise
