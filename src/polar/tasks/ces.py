"""Constant Elasticity of Substitution (CES) task for BED."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal, constraints
from torch.distributions.utils import broadcast_all

from polar.tasks.bed import BEDEpisode, BEDTrajectoryPlotter

CESEpisode = BEDEpisode
_RHO_ZERO_TOL = 1e-6


class CensoredSigmoidNormal(Distribution):
    """Sigmoid-normal distribution with clipping mass at both boundaries."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        lower: float | Tensor,
        upper: float | Tensor,
        validate_args: bool | None = None,
    ) -> None:
        lower_tensor = torch.as_tensor(lower, dtype=loc.dtype, device=loc.device)
        upper_tensor = torch.as_tensor(upper, dtype=loc.dtype, device=loc.device)
        self.loc, self.scale, self.lower, self.upper = broadcast_all(
            loc,
            scale,
            lower_tensor,
            upper_tensor,
        )
        if torch.any(self.scale <= 0):
            raise ValueError("scale must be positive.")
        if torch.any(self.upper <= self.lower):
            raise ValueError("upper must be larger than lower.")

        self._normal = Normal(self.loc, self.scale)
        self._lower_logit = torch.logit(self.lower)
        self._upper_logit = torch.logit(self.upper)
        super().__init__(batch_shape=self.loc.shape, validate_args=validate_args)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        eta = self._normal.rsample(sample_shape)
        y = torch.sigmoid(eta)
        lower = self.lower.expand_as(y)
        upper = self.upper.expand_as(y)
        return torch.maximum(torch.minimum(y, upper), lower)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)

        value = value.to(dtype=self.loc.dtype, device=self.loc.device)
        value, loc, scale, lower, upper, lower_logit, upper_logit = broadcast_all(
            value,
            self.loc,
            self.scale,
            self.lower,
            self.upper,
            self._lower_logit,
            self._upper_logit,
        )

        normal = Normal(loc, scale)
        z_lower = (lower_logit - loc) / scale
        z_upper = (upper_logit - loc) / scale

        safe_value = value.clamp(min=torch.finfo(value.dtype).tiny, max=1.0 - torch.finfo(value.dtype).eps)
        logit_value = torch.logit(safe_value)
        interior_log_prob = (
            normal.log_prob(logit_value)
            - torch.log(safe_value)
            - torch.log1p(-safe_value)
        )
        lower_log_mass = torch.special.log_ndtr(z_lower)
        upper_log_mass = torch.special.log_ndtr(-z_upper)
        neg_inf = torch.full_like(value, -torch.inf)
        at_lower = value == lower
        at_upper = value == upper
        in_open_interval = (value > lower) & (value < upper)

        return torch.where(
            at_lower,
            lower_log_mass,
            torch.where(
                at_upper,
                upper_log_mass,
                torch.where(in_open_interval, interior_log_prob, neg_inf),
            ),
        )


def ces_utility(
    baskets: Tensor,
    rho: Tensor,
    alpha: Tensor,
) -> Tensor:
    """Compute CES utility for one or more baskets."""

    weighted_sum = torch.sum(alpha * baskets.pow(rho), dim=-1, keepdim=True)
    standard_utility = weighted_sum.pow(1.0 / rho)

    return standard_utility


def ces_log_likelihood(
    y: Tensor,
    x: Tensor,
    theta: Tensor,
    *,
    n_goods: int = 3,
    noise_scale: float = 0.005,
    epsilon: float = 2.0 ** (-22),
) -> Tensor:
    """Analytical log-likelihood for the CES task."""

    if x.shape[-1] != 2 * n_goods:
        raise ValueError(
            f"Expected x.shape[-1] == {2 * n_goods} for n_goods={n_goods}, got {x.shape[-1]}."
        )
    expected_theta_dim = n_goods + 2
    if theta.shape[-1] != expected_theta_dim:
        raise ValueError(
            "Unexpected theta dimension for CES task: "
            f"expected {expected_theta_dim}, got {theta.shape[-1]}."
        )

    rho = theta[..., 0:1]
    alpha = theta[..., 1 : 1 + n_goods]
    log_u = theta[..., 1 + n_goods : 2 + n_goods]
    u = torch.exp(log_u)

    basket_a = x[..., :n_goods]
    basket_b = x[..., n_goods:]
    utility_a = ces_utility(basket_a, rho, alpha)
    utility_b = ces_utility(basket_b, rho, alpha)
    mean_eta = u * (utility_a - utility_b)

    distance = torch.norm(basket_a - basket_b, p=2, dim=-1, keepdim=True)
    std_eta = u * noise_scale * (1.0 + distance)

    dist = CensoredSigmoidNormal(mean_eta, std_eta, epsilon, 1.0 - epsilon)
    if y.ndim == x.ndim - 1:
        y = y.unsqueeze(-1)
    return dist.log_prob(y).squeeze(-1)


class CESSampler:
    """Sample BED episodes for the CES preference-learning task."""

    def __init__(
        self,
        *,
        n_goods: int = 3,
        n_contrastive: int = 19,
        n_context_init: int = 5,
        n_query: int = 200,
        n_target: int = 0,
        design_min: float = 0.0,
        design_max: float = 100.0,
        noise_scale: float = 0.005,
        epsilon: float = 2.0 ** (-22),
        seed: int | None = None,
    ) -> None:
        if n_goods <= 0:
            raise ValueError("n_goods must be positive.")
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
        if not (0.0 < epsilon < 0.5):
            raise ValueError("epsilon must be in (0, 0.5).")

        self.name = "ces"
        self.observation_type = "continuous"
        self.num_classes = None
        self.n_goods = int(n_goods)
        self.dim_x = 2 * self.n_goods
        self.theta_dim = self.n_goods + 2
        self.n_contrastive = int(n_contrastive)
        self.n_context_init = int(n_context_init)
        self.n_query = int(n_query)
        self.n_target = int(n_target)
        self.physical_design_min = float(design_min)
        self.physical_design_max = float(design_max)
        self.model_design_mean = 0.5 * (self.physical_design_min + self.physical_design_max)
        # Use the mean/std implied by the config-defined uniform design range.
        self.model_design_std = (
            self.physical_design_max - self.physical_design_min
        ) / math.sqrt(12.0)
        self.design_min = (
            self.physical_design_min - self.model_design_mean
        ) / self.model_design_std
        self.design_max = (
            self.physical_design_max - self.model_design_mean
        ) / self.model_design_std
        self.noise_scale = float(noise_scale)
        self.epsilon = float(epsilon)
        self.trajectory_plotter: BEDTrajectoryPlotter | None = None

        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(int(seed))

    def sample_theta(self, n: int) -> Tensor:
        # rho ~ Beta(1,1) with regularization: rho in [0.01, 1.0]
        rho = 0.01 + 0.99 * torch.distributions.beta.Beta(1.0, 1.0).sample((n,))
        # rho = 0.01 + 0.99 * torch.rand(n, generator=self._rng, dtype=torch.float32)
        # alpha ~ Dirichlet(1,...,1)
        alpha = torch.distributions.Dirichlet(
            torch.ones(self.n_goods)
        ).sample((n,))
        log_u = 1.0 + 3.0 * torch.randn(n, generator=self._rng, dtype=torch.float32)
        return torch.cat([rho.unsqueeze(-1), alpha, log_u.unsqueeze(-1)], dim=-1)

    def sample_episode(self) -> BEDEpisode:
        n_theta = self.n_contrastive + 1
        theta_all = self.sample_theta(n_theta)
        theta_true = theta_all[0]

        n_total = self.n_context_init + self.n_query + self.n_target
        all_x_physical = self._sample_physical_designs(n_total)
        all_x_physical = torch.clamp(
            all_x_physical,
            min=max(self.physical_design_min, 0.01),
            max=self.physical_design_max,
        )
        all_y = self._observe(all_x_physical, theta_true)
        all_x = self._to_model_designs(all_x_physical)

        context_end = self.n_context_init
        query_end = context_end + self.n_query

        target_x = all_x[query_end:] if self.n_target > 0 else None
        target_y = all_y[query_end:] if self.n_target > 0 else None
        return BEDEpisode(
            theta_all=theta_all,
            context_x=all_x[:context_end],
            context_y=all_y[:context_end],
            query_x=all_x[context_end:query_end],
            query_y=all_y[context_end:query_end],
            target_x=target_x,
            target_y=target_y,
        )

    def sample_batch(self, *, batch_size: int) -> list[BEDEpisode]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return [self.sample_episode() for _ in range(batch_size)]

    def log_likelihood(self, y: Tensor, x: Tensor, theta: Tensor) -> Tensor:
        return ces_log_likelihood(
            y,
            self._to_physical_designs(x),
            theta,
            n_goods=self.n_goods,
            noise_scale=self.noise_scale,
            epsilon=self.epsilon,
        )

    def _sample_physical_designs(self, n: int) -> Tensor:
        u = torch.rand(n, self.dim_x, generator=self._rng, dtype=torch.float32)
        return self.physical_design_min + (
            self.physical_design_max - self.physical_design_min
        ) * u

    def _to_model_designs(self, x_physical: Tensor) -> Tensor:
        x_physical = x_physical.to(dtype=torch.float32)
        return (x_physical - self.model_design_mean) / self.model_design_std

    def _to_physical_designs(self, x_model: Tensor) -> Tensor:
        x_model = x_model.to(dtype=torch.float32)
        return self.model_design_mean + self.model_design_std * x_model

    def _observe(self, x: Tensor, theta: Tensor) -> Tensor:
        x = x.to(dtype=torch.float32)
        rho = theta[0:1]
        alpha = theta[1 : 1 + self.n_goods]
        log_u = theta[1 + self.n_goods : 2 + self.n_goods]
        basket_a = x[:, : self.n_goods]
        basket_b = x[:, self.n_goods :]
        utility_a = ces_utility(
            basket_a,
            rho.view(1, 1),
            alpha.view(1, self.n_goods),
        ).squeeze(-1)
        utility_b = ces_utility(
            basket_b,
            rho.view(1, 1),
            alpha.view(1, self.n_goods),
        ).squeeze(-1)

        u = torch.exp(log_u)
        mean_eta = u * (utility_a - utility_b)
        distance = torch.norm(basket_a-basket_b, dim=-1, p=2)
        std_eta = u * self.noise_scale * (1.0 + distance)

        dist = CensoredSigmoidNormal(
            mean_eta,
            std_eta,
            self.epsilon,
            1.0 - self.epsilon,
        )
        return cast(Tensor, dist.rsample())


__all__ = [
    "CESEpisode",
    "CESSampler",
    "CensoredSigmoidNormal",
    "ces_log_likelihood",
    "ces_utility",
]
