"""Gaussian-process prior sampler for active-learning sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal

import torch

KernelType = Literal["rbf", "matern12", "matern32", "matern52"]
LengthscaleSampling = Literal["uniform", "log_uniform"]
_STD_EPS = 1e-6


def _standardize_tensor(
    values: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    eps: float = _STD_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = values.float()
    if mean is None:
        mean = target.mean()
    if std is None:
        std = target.std(unbiased=False)
    std = std.clamp_min(float(eps))
    return (target - mean) / std, mean, std


def _canonicalize_uniform_box(
    x: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = float(upper) - float(lower)
    if width <= 0.0:
        raise ValueError("upper must be larger than lower.")
    center = torch.tensor(0.5 * (float(lower) + float(upper)), dtype=x.dtype, device=x.device)
    scale = torch.tensor(width / math.sqrt(12.0), dtype=x.dtype, device=x.device)
    return (x - center) / scale, center, scale


@dataclass(frozen=True)
class GPPriorSample:
    """One sampled GP episode split into context/query/target."""

    context_x: torch.Tensor
    context_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    target_x: torch.Tensor
    target_y: torch.Tensor
    plot_x: torch.Tensor
    plot_y_true: torch.Tensor
    kernel_type: KernelType
    lengthscales: torch.Tensor
    output_scale: torch.Tensor
    raw_context_x: torch.Tensor | None = None
    raw_query_x: torch.Tensor | None = None
    raw_target_x: torch.Tensor | None = None
    raw_plot_x: torch.Tensor | None = None
    raw_context_y: torch.Tensor | None = None
    raw_query_y: torch.Tensor | None = None
    raw_target_y: torch.Tensor | None = None
    raw_plot_y_true: torch.Tensor | None = None
    x_center: torch.Tensor | None = None
    x_scale: torch.Tensor | None = None
    y_mean: torch.Tensor | None = None
    y_std: torch.Tensor | None = None
    x_standardized: bool = False
    y_standardized: bool = False
    final_jitter: float | None = None


class GPPriorALSampler:
    """Sample a single latent function and split points for AL evaluation."""

    KERNEL_ORDER: tuple[KernelType, ...] = ("rbf", "matern12", "matern32", "matern52")

    def __init__(
        self,
        *,
        dim_x: int,
        n_context_init: int,
        n_query: int,
        n_target: int,
        design_min: float,
        design_max: float,
        plot_grid_size: int,
        noise_std: float,
        lengthscale_min: float,
        lengthscale_max: float,
        scale_min: float,
        scale_max: float,
        p_isotropic: float,
        kernel_weights: dict[str, float] | None = None,
        jitter: float = 1e-4,
        cholesky_max_tries: int = 8,
        jitter_growth: float = 10.0,
        lengthscale_sampling: LengthscaleSampling = "uniform",
        standardize_x: bool = False,
        standardize_y: bool = False,
        seed: int | None = None,
    ) -> None:
        if dim_x <= 0:
            raise ValueError("dim_x must be positive.")
        if n_context_init <= 0:
            raise ValueError("n_context_init must be positive.")
        if n_query <= 0:
            raise ValueError("n_query must be positive.")
        if n_target <= 0:
            raise ValueError("n_target must be positive.")
        if design_max <= design_min:
            raise ValueError("design_max must be larger than design_min.")
        if plot_grid_size <= 1:
            raise ValueError("plot_grid_size must be > 1.")
        if lengthscale_max <= lengthscale_min:
            raise ValueError("lengthscale_max must be larger than lengthscale_min.")
        if scale_max <= scale_min:
            raise ValueError("scale_max must be larger than scale_min.")
        if not (0.0 <= p_isotropic <= 1.0):
            raise ValueError("p_isotropic must be in [0, 1].")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")
        if jitter <= 0.0:
            raise ValueError("jitter must be positive.")
        if cholesky_max_tries <= 0:
            raise ValueError("cholesky_max_tries must be positive.")
        if jitter_growth <= 1.0:
            raise ValueError("jitter_growth must be greater than 1.0.")
        if lengthscale_sampling not in {"uniform", "log_uniform"}:
            raise ValueError("lengthscale_sampling must be 'uniform' or 'log_uniform'.")

        self.dim_x = int(dim_x)
        self.n_context_init = int(n_context_init)
        self.n_query = int(n_query)
        self.n_target = int(n_target)
        self.design_min = float(design_min)
        self.design_max = float(design_max)
        self.plot_grid_size = int(plot_grid_size)
        self.noise_std = float(noise_std)
        self.lengthscale_min = float(lengthscale_min)
        self.lengthscale_max = float(lengthscale_max)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.p_isotropic = float(p_isotropic)
        self.jitter = float(jitter)
        self.cholesky_max_tries = int(cholesky_max_tries)
        self.jitter_growth = float(jitter_growth)
        self.lengthscale_sampling = lengthscale_sampling
        self.standardize_x = bool(standardize_x)
        self.standardize_y = bool(standardize_y)

        weights = kernel_weights or {
            "rbf": 1.0,
            "matern12": 0.0,
            "matern32": 0.5,
            "matern52": 0.5,
        }
        ordered = torch.tensor([float(weights.get(name, 0.0)) for name in self.KERNEL_ORDER])
        if torch.any(ordered < 0):
            raise ValueError("kernel_weights must be non-negative.")
        if float(ordered.sum().item()) <= 0.0:
            raise ValueError("kernel_weights must have at least one positive value.")
        self.kernel_weights = ordered / ordered.sum()

        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(int(seed))

        if self.standardize_x:
            x_center = torch.full((self.dim_x,), 0.5 * (self.design_min + self.design_max))
            x_scale = torch.full(
                (self.dim_x,),
                (self.design_max - self.design_min) / math.sqrt(12.0),
            )
        else:
            x_center = torch.zeros(self.dim_x, dtype=torch.float32)
            x_scale = torch.ones(self.dim_x, dtype=torch.float32)
        self._x_center = x_center.float()
        self._x_scale = x_scale.float()

    def sample_episode(self) -> GPPriorSample:
        """Sample one AL episode from a GP prior."""
        n_data = self.n_context_init + self.n_query + self.n_target
        raw_data_x = self._sample_raw_inputs(n_data)
        raw_plot_x = self._sample_raw_plot_inputs()

        data_x = self._transform_inputs(raw_data_x)
        plot_x = self._transform_inputs(raw_plot_x)
        all_x = torch.cat([plot_x, data_x], dim=0)

        lengthscales, output_scale, kernel_type = self._sample_hyperparameters()
        cov = self._kernel_matrix(
            all_x,
            all_x,
            lengthscales=lengthscales,
            output_scale=output_scale,
            kernel_type=kernel_type,
        )

        latent, final_jitter = self._sample_multivariate(cov)
        if self.noise_std > 0:
            observed = latent + self.noise_std * torch.randn(latent.shape[0], generator=self._rng)
        else:
            observed = latent

        raw_plot_y_true = latent[: self.plot_grid_size]
        raw_data_y = observed[self.plot_grid_size :]

        if self.standardize_y:
            data_y, y_mean, y_std = _standardize_tensor(raw_data_y)
            plot_y_true = (raw_plot_y_true.float() - y_mean) / y_std
        else:
            data_y = raw_data_y.float()
            plot_y_true = raw_plot_y_true.float()
            y_mean = torch.zeros((), dtype=torch.float32)
            y_std = torch.ones((), dtype=torch.float32)

        context_end = self.n_context_init
        query_end = context_end + self.n_query

        return GPPriorSample(
            context_x=data_x[:context_end],
            context_y=data_y[:context_end],
            query_x=data_x[context_end:query_end],
            query_y=data_y[context_end:query_end],
            target_x=data_x[query_end:],
            target_y=data_y[query_end:],
            plot_x=plot_x,
            plot_y_true=plot_y_true,
            kernel_type=kernel_type,
            lengthscales=lengthscales,
            output_scale=output_scale,
            raw_context_x=raw_data_x[:context_end],
            raw_query_x=raw_data_x[context_end:query_end],
            raw_target_x=raw_data_x[query_end:],
            raw_plot_x=raw_plot_x,
            raw_context_y=raw_data_y[:context_end],
            raw_query_y=raw_data_y[context_end:query_end],
            raw_target_y=raw_data_y[query_end:],
            raw_plot_y_true=raw_plot_y_true,
            x_center=self._x_center.clone(),
            x_scale=self._x_scale.clone(),
            y_mean=y_mean.clone(),
            y_std=y_std.clone(),
            x_standardized=self.standardize_x,
            y_standardized=self.standardize_y,
            final_jitter=float(final_jitter),
        )

    def sample_batch(self, *, batch_size: int) -> list[GPPriorSample]:
        """Sample a batch of independent AL episodes."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return [self.sample_episode() for _ in range(batch_size)]

    def _sample_raw_plot_inputs(self) -> torch.Tensor:
        if self.dim_x != 1:
            # For dim_x > 1, plotting as a line is undefined; still provide points.
            return self._sample_raw_inputs(self.plot_grid_size)
        grid = torch.linspace(
            self.design_min,
            self.design_max,
            self.plot_grid_size,
            dtype=torch.float32,
        )
        return grid.unsqueeze(-1)

    def _sample_raw_inputs(self, n_points: int) -> torch.Tensor:
        x = torch.rand(n_points, self.dim_x, generator=self._rng, dtype=torch.float32)
        return self.design_min + (self.design_max - self.design_min) * x

    def _transform_inputs(self, raw_x: torch.Tensor) -> torch.Tensor:
        if self.standardize_x:
            x_model, _, _ = _canonicalize_uniform_box(
                raw_x,
                lower=self.design_min,
                upper=self.design_max,
            )
            return x_model
        return raw_x.float()

    def _sample_hyperparameters(self) -> tuple[torch.Tensor, torch.Tensor, KernelType]:
        if self.lengthscale_sampling == "log_uniform":
            log_min = math.log(self.lengthscale_min)
            log_max = math.log(self.lengthscale_max)
            lengthscales_base = torch.exp(
                log_min + (log_max - log_min) * torch.rand(
                    self.dim_x,
                    generator=self._rng,
                    dtype=torch.float32,
                )
            )
        else:
            lengthscale_range = self.lengthscale_max - self.lengthscale_min
            lengthscales_base = self.lengthscale_min + lengthscale_range * torch.rand(
                self.dim_x,
                generator=self._rng,
                dtype=torch.float32,
            )

        if self.dim_x > 1:
            draw = float(torch.rand(1, generator=self._rng).item())
            if draw < self.p_isotropic:
                lengthscales_base = torch.full_like(
                    lengthscales_base,
                    float(lengthscales_base[0].item()),
                )

        lengthscales = lengthscales_base * math.sqrt(float(self.dim_x))
        scale = self.scale_min + (self.scale_max - self.scale_min) * torch.rand(
            1,
            generator=self._rng,
            dtype=torch.float32,
        )
        kernel_idx = int(torch.multinomial(self.kernel_weights, 1, generator=self._rng).item())
        kernel_type: KernelType = self.KERNEL_ORDER[kernel_idx]
        return lengthscales, scale, kernel_type

    def _kernel_matrix(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *,
        lengthscales: torch.Tensor,
        output_scale: torch.Tensor,
        kernel_type: KernelType,
    ) -> torch.Tensor:
        sq_dist = self._scaled_squared_distances(x1, x2, lengthscales)
        if kernel_type == "rbf":
            kernel = torch.exp(-0.5 * sq_dist)
        elif kernel_type == "matern12":
            dist = torch.sqrt(sq_dist.clamp_min(0.0))
            kernel = torch.exp(-dist)
        elif kernel_type == "matern32":
            dist = torch.sqrt(sq_dist.clamp_min(0.0))
            sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=torch.float32))
            kernel = (1.0 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)
        elif kernel_type == "matern52":
            dist = torch.sqrt(sq_dist.clamp_min(0.0))
            sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=torch.float32))
            kernel = (1.0 + sqrt5 * dist + (5.0 / 3.0) * dist.square()) * torch.exp(-sqrt5 * dist)
        else:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")
        return output_scale * kernel

    @staticmethod
    def _scaled_squared_distances(
        x1: torch.Tensor,
        x2: torch.Tensor,
        lengthscales: torch.Tensor,
    ) -> torch.Tensor:
        diff = x1.unsqueeze(1) - x2.unsqueeze(0)
        inv_ls2 = 1.0 / lengthscales.square().view(1, 1, -1)
        return (diff.square() * inv_ls2).sum(dim=-1)

    def _sample_multivariate(self, cov: torch.Tensor) -> tuple[torch.Tensor, float]:
        target_dtype = cov.dtype
        cov64 = cov.to(torch.float64)
        cov64 = 0.5 * (cov64 + cov64.transpose(-1, -2))

        eye = torch.eye(cov64.shape[0], dtype=cov64.dtype, device=cov64.device)
        chol, info = torch.linalg.cholesky_ex(cov64)
        if int(info.item()) == 0:
            z = torch.randn(cov64.shape[0], generator=self._rng, dtype=cov64.dtype)
            return (chol @ z).to(dtype=target_dtype), 0.0

        jitter = self.jitter

        for _ in range(self.cholesky_max_tries):
            chol, info = torch.linalg.cholesky_ex(cov64 + jitter * eye)
            if int(info.item()) == 0:
                z = torch.randn(cov64.shape[0], generator=self._rng, dtype=cov64.dtype)
                return (chol @ z).to(dtype=target_dtype), float(jitter)
            jitter *= self.jitter_growth

        raise torch._C._LinAlgError(
            "linalg.cholesky failed for GP prior covariance matrix after "
            f"{self.cholesky_max_tries} retries; final jitter={jitter:.3e}."
        )


class GPPriorALTask:
    """Expose the GP prior sampler through the shared ALTask interface."""

    name = "gp_regression"
    DEFAULT_TEST_SEEDS: tuple[str, ...] = ("test0", "test1", "test2", "test3", "test4")

    def __init__(
        self,
        *,
        dim_x: int | None = None,
        dim_x_min: int | None = None,
        dim_x_max: int | None = None,
        design_min: float,
        design_max: float,
        plot_grid_size: int,
        noise_std: float,
        lengthscale_min: float,
        lengthscale_max: float,
        scale_min: float,
        scale_max: float,
        p_isotropic: float,
        kernel_weights: dict[str, float],
        jitter: float,
        cholesky_max_tries: int,
        jitter_growth: float,
        split: str = "train",
        lengthscale_sampling: LengthscaleSampling = "uniform",
        standardize_x: bool = False,
        standardize_y: bool = False,
        seed: int | None = None,
    ) -> None:
        if dim_x_min is None and dim_x_max is None:
            if dim_x is None:
                raise ValueError("Provide dim_x or both dim_x_min/dim_x_max.")
            dim_x_min = int(dim_x)
            dim_x_max = int(dim_x)
        elif dim_x_min is None or dim_x_max is None:
            raise ValueError("dim_x_min and dim_x_max must both be provided together.")
        if int(dim_x_min) <= 0 or int(dim_x_max) <= 0:
            raise ValueError("dim_x_min and dim_x_max must be positive.")
        if int(dim_x_max) < int(dim_x_min):
            raise ValueError("dim_x_max must be >= dim_x_min.")

        self.split = split
        self.dim_x_min = int(dim_x_min)
        self.dim_x_max = int(dim_x_max)
        self.dim_x = int(self.dim_x_max)
        self.variable_dim = self.dim_x_min != self.dim_x_max
        self.design_min = float(design_min)
        self.design_max = float(design_max)
        self.plot_grid_size = int(plot_grid_size)
        self.noise_std = float(noise_std)
        self.lengthscale_min = float(lengthscale_min)
        self.lengthscale_max = float(lengthscale_max)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.p_isotropic = float(p_isotropic)
        self.kernel_weights = {str(k): float(v) for k, v in kernel_weights.items()}
        self.jitter = float(jitter)
        self.cholesky_max_tries = int(cholesky_max_tries)
        self.jitter_growth = float(jitter_growth)
        self.lengthscale_sampling = lengthscale_sampling
        self.standardize_x = bool(standardize_x)
        self.standardize_y = bool(standardize_y)
        self._base_seed = int(seed) if seed is not None else 0
        self.dataset_ids = ["gp_prior"]
        self.test_seeds = self.DEFAULT_TEST_SEEDS
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self._base_seed)

    def eligible_dataset_ids(self, *, min_total_size: int) -> list[str]:
        if int(min_total_size) <= 0:
            raise ValueError("min_total_size must be positive.")
        return list(self.dataset_ids)

    def sample_train_batch(
        self,
        *,
        batch_size: int,
        n_context: int,
        n_query: int,
        n_target: int,
    ) -> list[GPPriorSample]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        self._validate_runtime_sizes(n_context=n_context, n_query=n_query, n_target=n_target)
        sampler_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())
        sampler = self._build_sampler(
            seed=sampler_seed,
            dim_x=self._sample_batch_dim(),
            n_context=n_context,
            n_query=n_query,
            n_target=n_target,
        )
        return sampler.sample_batch(batch_size=batch_size)

    def build_test_episode(
        self,
        *,
        seed: str,
        n_context: int,
        query_pool_size: int,
        n_target: int,
    ) -> GPPriorSample:
        self._validate_runtime_sizes(
            n_context=n_context,
            n_query=query_pool_size,
            n_target=n_target,
        )
        digest = hashlib.sha256(f"{self.name}:{seed}:{self._base_seed}".encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)
        sampler = self._build_sampler(
            seed=int(seed_int),
            dim_x=self._deterministic_dim_for_seed(int(seed_int)),
            n_context=n_context,
            n_query=query_pool_size,
            n_target=n_target,
        )
        return sampler.sample_episode()

    def iter_test_episodes(
        self,
        *,
        n_context: int,
        query_pool_size: int,
        n_target: int,
        seeds: list[str] | tuple[str, ...] | None = None,
        dataset_ids: list[str] | tuple[str, ...] | None = None,
        shuffle_query_pool: bool = True,
    ):
        _ = shuffle_query_pool
        if dataset_ids is not None:
            active_ids = [str(dataset_id) for dataset_id in dataset_ids]
            if active_ids != self.dataset_ids:
                raise ValueError("GPPriorALTask only supports dataset_ids=['gp_prior'].")
        active_seeds = list(seeds) if seeds is not None else list(self.test_seeds)
        for seed in active_seeds:
            yield self.build_test_episode(
                seed=str(seed),
                n_context=n_context,
                query_pool_size=query_pool_size,
                n_target=n_target,
            )

    def _sample_batch_dim(self) -> int:
        if not self.variable_dim:
            return self.dim_x_min
        return int(
            torch.randint(
                low=self.dim_x_min,
                high=self.dim_x_max + 1,
                size=(1,),
                generator=self._rng,
            ).item()
        )

    def _deterministic_dim_for_seed(self, seed_int: int) -> int:
        if not self.variable_dim:
            return self.dim_x_min
        span = self.dim_x_max - self.dim_x_min + 1
        return self.dim_x_min + (int(seed_int) % span)

    def _build_sampler(
        self,
        *,
        seed: int,
        dim_x: int,
        n_context: int,
        n_query: int,
        n_target: int,
    ) -> GPPriorALSampler:
        return GPPriorALSampler(
            dim_x=int(dim_x),
            n_context_init=int(n_context),
            n_query=int(n_query),
            n_target=int(n_target),
            design_min=self.design_min,
            design_max=self.design_max,
            plot_grid_size=self.plot_grid_size,
            noise_std=self.noise_std,
            lengthscale_min=self.lengthscale_min,
            lengthscale_max=self.lengthscale_max,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            p_isotropic=self.p_isotropic,
            kernel_weights=self.kernel_weights,
            jitter=self.jitter,
            cholesky_max_tries=self.cholesky_max_tries,
            jitter_growth=self.jitter_growth,
            lengthscale_sampling=self.lengthscale_sampling,
            standardize_x=self.standardize_x,
            standardize_y=self.standardize_y,
            seed=seed,
        )

    @staticmethod
    def _validate_runtime_sizes(*, n_context: int, n_query: int, n_target: int) -> None:
        if int(n_context) <= 0:
            raise ValueError("n_context must be positive.")
        if int(n_query) <= 0:
            raise ValueError("n_query must be positive.")
        if int(n_target) <= 0:
            raise ValueError("n_target must be positive.")
