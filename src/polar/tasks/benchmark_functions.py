"""Benchmark-function AL task for unseen function evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Sequence

import torch

from polar.tasks.al import ALEpisode

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


def _canonicalize_box(
    x: torch.Tensor,
    *,
    lower: float | Sequence[float] | torch.Tensor,
    upper: float | Sequence[float] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lower_tensor = torch.as_tensor(lower, dtype=x.dtype, device=x.device)
    upper_tensor = torch.as_tensor(upper, dtype=x.dtype, device=x.device)
    if lower_tensor.ndim == 0:
        lower_tensor = lower_tensor.repeat(x.shape[-1])
    if upper_tensor.ndim == 0:
        upper_tensor = upper_tensor.repeat(x.shape[-1])
    if lower_tensor.shape != upper_tensor.shape:
        raise ValueError("lower and upper must have the same shape.")
    if lower_tensor.numel() != x.shape[-1]:
        raise ValueError("Bounds must match the final feature dimension of x.")
    widths = upper_tensor - lower_tensor
    if torch.any(widths <= 0):
        raise ValueError("All upper bounds must be larger than lower bounds.")
    center = 0.5 * (lower_tensor + upper_tensor)
    scale = widths / math.sqrt(12.0)
    return (x - center) / scale, center, scale


def _sobol_in_box(
    *,
    dim: int,
    n: int,
    lower: float | Sequence[float] | torch.Tensor,
    upper: float | Sequence[float] | torch.Tensor,
    scramble: bool = False,
    seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if n <= 0:
        raise ValueError("n must be positive.")
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=scramble, seed=seed)
    points = engine.draw(n, dtype=dtype)
    if device is not None:
        points = points.to(device)
    lower_tensor = torch.as_tensor(lower, dtype=dtype, device=points.device)
    upper_tensor = torch.as_tensor(upper, dtype=dtype, device=points.device)
    if lower_tensor.ndim == 0:
        lower_tensor = lower_tensor.repeat(dim)
    if upper_tensor.ndim == 0:
        upper_tensor = upper_tensor.repeat(dim)
    if lower_tensor.shape != upper_tensor.shape:
        raise ValueError("lower and upper must have the same shape.")
    if lower_tensor.numel() != dim:
        raise ValueError("Bounds must match dim.")
    return lower_tensor + (upper_tensor - lower_tensor) * points


@dataclass(frozen=True)
class BenchmarkFunctionSpec:
    """Static benchmark definition."""

    name: str
    dim_x: int
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    fn: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class BenchmarkFunctionEpisode(ALEpisode):
    """AL episode sampled from a deterministic benchmark function."""

    raw_context_x: torch.Tensor
    raw_query_x: torch.Tensor
    raw_target_x: torch.Tensor
    raw_context_y: torch.Tensor
    raw_query_y: torch.Tensor
    raw_target_y: torch.Tensor
    x_center: torch.Tensor
    x_scale: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


@dataclass(frozen=True)
class BenchmarkNormalization:
    """Fixed per-function normalization statistics."""

    x_center: torch.Tensor
    x_scale: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


def _forrester(x: torch.Tensor) -> torch.Tensor:
    x0 = x[..., 0]
    return (6.0 * x0 - 2.0).square() * torch.sin(12.0 * x0 - 4.0)


def _gramacy_lee_2012(x: torch.Tensor) -> torch.Tensor:
    x0 = x[..., 0]
    return torch.sin(10.0 * torch.pi * x0) / (2.0 * x0) + (x0 - 1.0).pow(4)


def _higdon_2002(x: torch.Tensor) -> torch.Tensor:
    x0 = x[..., 0]
    return torch.sin(2.0 * torch.pi * x0 / 10.0) + 0.2 * torch.sin(2.0 * torch.pi * x0 / 2.5)


def _gaussian_pdf(x: torch.Tensor, *, mean: float, std: float) -> torch.Tensor:
    x = x.float()
    variance = float(std) ** 2
    norm = float(std) * math.sqrt(2.0 * math.pi)
    return torch.exp(-0.5 * (x - float(mean)).square() / variance) / norm


def _sine_gaussian_bumps(x: torch.Tensor) -> torch.Tensor:
    x0 = x[..., 0]
    return (
        2.0 * torch.sin(2.0 * x0)
        + 8.0 * _gaussian_pdf(x0, mean=2.5, std=0.5)
        + 10.0 * _gaussian_pdf(x0, mean=7.5, std=0.25)
        - 6.0 * _gaussian_pdf(x0, mean=-4.5, std=0.5)
    )


def _branin(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return a * (x2 - b * x1.square() + c * x1 - r).square() + s * (1.0 - t) * torch.cos(x1) + s


def _goldstein_price(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = 1.0 + (x1 + x2 + 1.0).square() * (
        19.0 - 14.0 * x1 + 3.0 * x1.square() - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2.square()
    )
    term2 = 30.0 + (2.0 * x1 - 3.0 * x2).square() * (
        18.0 - 32.0 * x1 + 12.0 * x1.square() + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2.square()
    )
    return term1 * term2


def _three_hump_camel(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    return 2.0 * x1.square() - 1.05 * x1.pow(4) + x1.pow(6) / 6.0 + x1 * x2 + x2.square()


def _six_hump_camel(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    return (4.0 - 2.1 * x1.square() + x1.pow(4) / 3.0) * x1.square() + x1 * x2 + (-4.0 + 4.0 * x2.square()) * x2.square()


def _franke(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = 0.75 * torch.exp(-((9.0 * x1 - 2.0).square()) / 4.0 - ((9.0 * x2 - 2.0).square()) / 4.0)
    term2 = 0.75 * torch.exp(-((9.0 * x1 + 1.0).square()) / 49.0 - (9.0 * x2 + 1.0) / 10.0)
    term3 = 0.5 * torch.exp(-((9.0 * x1 - 7.0).square()) / 4.0 - ((9.0 * x2 - 3.0).square()) / 4.0)
    term4 = -0.2 * torch.exp(-(9.0 * x1 - 4.0).square() - (9.0 * x2 - 7.0).square())
    return term1 + term2 + term3 + term4


def _ackley(x: torch.Tensor) -> torch.Tensor:
    sq_mean = x.square().mean(dim=-1)
    cos_mean = torch.cos(2.0 * torch.pi * x).mean(dim=-1)
    return -20.0 * torch.exp(-0.2 * torch.sqrt(sq_mean)) - torch.exp(cos_mean) + 20.0 + math.e


_BENCHMARK_SPECS: dict[str, BenchmarkFunctionSpec] = {
    "forrester": BenchmarkFunctionSpec("forrester", 1, (0.0,), (1.0,), _forrester),
    "gramacy_lee_2012": BenchmarkFunctionSpec(
        "gramacy_lee_2012",
        1,
        (0.5,),
        (2.5,),
        _gramacy_lee_2012,
    ),
    "higdon_2002": BenchmarkFunctionSpec("higdon_2002", 1, (0.0,), (10.0,), _higdon_2002),
    "sine_gaussian_bumps": BenchmarkFunctionSpec(
        "sine_gaussian_bumps",
        1,
        (-8.0,),
        (8.0,),
        _sine_gaussian_bumps,
    ),
    "branin": BenchmarkFunctionSpec("branin", 2, (-5.0, 0.0), (10.0, 15.0), _branin),
    "goldstein_price": BenchmarkFunctionSpec(
        "goldstein_price",
        2,
        (-2.0, -2.0),
        (2.0, 2.0),
        _goldstein_price,
    ),
    "three_hump_camel": BenchmarkFunctionSpec(
        "three_hump_camel",
        2,
        (-5.0, -5.0),
        (5.0, 5.0),
        _three_hump_camel,
    ),
    "six_hump_camel": BenchmarkFunctionSpec(
        "six_hump_camel",
        2,
        (-3.0, -2.0),
        (3.0, 2.0),
        _six_hump_camel,
    ),
    "franke": BenchmarkFunctionSpec("franke", 2, (0.0, 0.0), (1.0, 1.0), _franke),
    "ackley": BenchmarkFunctionSpec(
        "ackley",
        2,
        (-32.768, -32.768),
        (32.768, 32.768),
        _ackley,
    ),
}


class BenchmarkFunctionALTask:
    """Pool-based AL task over a configurable set of benchmark functions."""

    name = "benchmark_functions"

    def __init__(
        self,
        *,
        benchmarks: list[str] | tuple[str, ...],
        split: str = "test",
        normalize_x: bool = True,
        normalize_y: bool = True,
        normalization_num_samples: int = 8192,
        num_eval_episodes: int = 50,
        group_by_dim: bool = True,
        seed: int | None = None,
    ) -> None:
        if not benchmarks:
            raise ValueError("benchmarks must be a non-empty list.")
        self.split = str(split)
        self.normalize_x = bool(normalize_x)
        self.normalize_y = bool(normalize_y)
        self.normalization_num_samples = int(normalization_num_samples)
        self.num_eval_episodes = int(num_eval_episodes)
        self.group_by_dim = bool(group_by_dim)
        if self.normalization_num_samples <= 0:
            raise ValueError("normalization_num_samples must be positive.")
        if self.num_eval_episodes <= 0:
            raise ValueError("num_eval_episodes must be positive.")

        normalized_names = [str(name).strip().lower() for name in benchmarks]
        missing = [name for name in normalized_names if name not in _BENCHMARK_SPECS]
        if missing:
            available = ", ".join(sorted(_BENCHMARK_SPECS))
            raise ValueError(f"Unknown benchmark(s): {missing}. Available: {available}")

        self.benchmarks = normalized_names
        self.dataset_ids = list(self.benchmarks)
        self.test_seeds = tuple(f"episode{i}" for i in range(self.num_eval_episodes))
        self._specs = {name: _BENCHMARK_SPECS[name] for name in self.benchmarks}
        self.dim_x = max(spec.dim_x for spec in self._specs.values())
        self._base_seed = int(seed) if seed is not None else 0
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self._base_seed)
        self._normalization = {
            name: self._build_normalization(spec)
            for name, spec in self._specs.items()
        }

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
    ) -> list[BenchmarkFunctionEpisode]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        self._validate_runtime_sizes(n_context=n_context, n_query=n_query, n_target=n_target)
        dataset_idx = int(
            torch.randint(0, len(self.dataset_ids), (1,), generator=self._rng).item()
        )
        dataset_id = self.dataset_ids[dataset_idx]
        episodes: list[BenchmarkFunctionEpisode] = []
        for _ in range(batch_size):
            episode_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())
            episodes.append(
                self._sample_episode(
                    dataset_id=dataset_id,
                    seed_int=episode_seed,
                    n_context=n_context,
                    n_query=n_query,
                    n_target=n_target,
                    split="train",
                )
            )
        return episodes

    def build_test_episode(
        self,
        *,
        dataset_id: str,
        seed: str,
        n_context: int,
        query_pool_size: int,
        n_target: int,
    ) -> BenchmarkFunctionEpisode:
        self._validate_runtime_sizes(n_context=n_context, n_query=query_pool_size, n_target=n_target)
        dataset_id = str(dataset_id).strip().lower()
        if dataset_id not in self._specs:
            raise ValueError(f"Unknown benchmark dataset_id={dataset_id!r}.")
        digest = hashlib.sha256(
            f"{self.name}:{dataset_id}:{seed}:{self._base_seed}".encode("utf-8")
        ).digest()
        seed_int = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)
        return self._sample_episode(
            dataset_id=dataset_id,
            seed_int=seed_int,
            n_context=n_context,
            n_query=query_pool_size,
            n_target=n_target,
            split="test",
            seed_name=str(seed),
        )

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
        active_ids = (
            [str(dataset_id).strip().lower() for dataset_id in dataset_ids]
            if dataset_ids is not None
            else list(self.dataset_ids)
        )
        active_seeds = list(seeds) if seeds is not None else list(self.test_seeds)
        for dataset_id in active_ids:
            if dataset_id not in self._specs:
                raise ValueError(f"Unknown benchmark dataset_id={dataset_id!r}.")
            for seed in active_seeds:
                yield self.build_test_episode(
                    dataset_id=dataset_id,
                    seed=str(seed),
                    n_context=n_context,
                    query_pool_size=query_pool_size,
                    n_target=n_target,
                )

    def benchmark_metadata(self) -> dict[str, dict[str, object]]:
        """Return normalization and bounds metadata for reporting."""
        metadata: dict[str, dict[str, object]] = {}
        for dataset_id, spec in self._specs.items():
            norm = self._normalization[dataset_id]
            metadata[dataset_id] = {
                "dim_x": spec.dim_x,
                "lower_bounds": list(spec.lower_bounds),
                "upper_bounds": list(spec.upper_bounds),
                "normalize_x": self.normalize_x,
                "normalize_y": self.normalize_y,
                "x_center": norm.x_center.tolist(),
                "x_scale": norm.x_scale.tolist(),
                "y_mean": float(norm.y_mean.item()),
                "y_std": float(norm.y_std.item()),
            }
        return metadata

    def evaluate_raw(self, dataset_id: str, x: torch.Tensor) -> torch.Tensor:
        """Evaluate a benchmark function in its original coordinate system."""
        dataset_id = str(dataset_id).strip().lower()
        if dataset_id not in self._specs:
            raise ValueError(f"Unknown benchmark dataset_id={dataset_id!r}.")
        return self._specs[dataset_id].fn(x.float()).float()

    def plot_bundle(
        self,
        dataset_id: str,
        *,
        resolution_1d: int = 400,
        resolution_2d: int = 80,
    ) -> dict[str, object]:
        """Build a dense true-function plot bundle in raw coordinates."""
        dataset_id = str(dataset_id).strip().lower()
        if dataset_id not in self._specs:
            raise ValueError(f"Unknown benchmark dataset_id={dataset_id!r}.")
        spec = self._specs[dataset_id]

        if spec.dim_x == 1:
            raw_plot_x = torch.linspace(
                spec.lower_bounds[0],
                spec.upper_bounds[0],
                int(resolution_1d),
                dtype=torch.float32,
            ).unsqueeze(-1)
            raw_plot_y_true = spec.fn(raw_plot_x).float()
            return {
                "kind": "line",
                "dim_x": 1,
                "raw_plot_x": raw_plot_x,
                "raw_plot_y_true": raw_plot_y_true,
            }

        if spec.dim_x == 2:
            x0 = torch.linspace(
                spec.lower_bounds[0],
                spec.upper_bounds[0],
                int(resolution_2d),
                dtype=torch.float32,
            )
            x1 = torch.linspace(
                spec.lower_bounds[1],
                spec.upper_bounds[1],
                int(resolution_2d),
                dtype=torch.float32,
            )
            grid_x0, grid_x1 = torch.meshgrid(x0, x1, indexing="ij")
            raw_plot_x = torch.stack([grid_x0, grid_x1], dim=-1).reshape(-1, 2)
            raw_plot_y_true = spec.fn(raw_plot_x).float()
            return {
                "kind": "surface",
                "dim_x": 2,
                "raw_plot_x": raw_plot_x,
                "raw_plot_y_true": raw_plot_y_true,
                "grid_shape": (int(resolution_2d), int(resolution_2d)),
            }

        raise ValueError(
            f"plot_bundle currently supports only 1D/2D benchmarks, got dim_x={spec.dim_x}."
        )

    def _build_normalization(self, spec: BenchmarkFunctionSpec) -> BenchmarkNormalization:
        lower = torch.tensor(spec.lower_bounds, dtype=torch.float32)
        upper = torch.tensor(spec.upper_bounds, dtype=torch.float32)
        x_center = 0.5 * (lower + upper)
        x_scale = (upper - lower) / math.sqrt(12.0)

        if self.normalize_y:
            x_samples = _sobol_in_box(
                dim=spec.dim_x,
                n=self.normalization_num_samples,
                lower=lower,
                upper=upper,
                scramble=False,
            )
            y_samples = spec.fn(x_samples).float()
            _, y_mean, y_std = _standardize_tensor(y_samples)
        else:
            y_mean = torch.zeros((), dtype=torch.float32)
            y_std = torch.ones((), dtype=torch.float32)

        if not self.normalize_x:
            x_center = torch.zeros_like(x_center)
            x_scale = torch.ones_like(x_scale)

        return BenchmarkNormalization(
            x_center=x_center,
            x_scale=x_scale,
            y_mean=y_mean,
            y_std=y_std,
        )

    def _sample_episode(
        self,
        *,
        dataset_id: str,
        seed_int: int,
        n_context: int,
        n_query: int,
        n_target: int,
        split: str,
        seed_name: str | None = None,
    ) -> BenchmarkFunctionEpisode:
        spec = self._specs[dataset_id]
        norm = self._normalization[dataset_id]
        total = int(n_context + n_query + n_target)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed_int))

        lower = torch.tensor(spec.lower_bounds, dtype=torch.float32)
        upper = torch.tensor(spec.upper_bounds, dtype=torch.float32)
        raw_x = lower + (upper - lower) * torch.rand(total, spec.dim_x, generator=generator)
        raw_y = spec.fn(raw_x).float()

        if self.normalize_x:
            x_model, _, _ = _canonicalize_box(raw_x, lower=lower, upper=upper)
        else:
            x_model = raw_x.float()
        if self.normalize_y:
            y_model = (raw_y - norm.y_mean) / norm.y_std
        else:
            y_model = raw_y

        context_end = int(n_context)
        query_end = context_end + int(n_query)
        return BenchmarkFunctionEpisode(
            dataset_id=dataset_id,
            split=split,  # type: ignore[arg-type]
            seed=seed_name,
            context_x=x_model[:context_end],
            context_y=y_model[:context_end],
            query_x=x_model[context_end:query_end],
            query_y=y_model[context_end:query_end],
            target_x=x_model[query_end:],
            target_y=y_model[query_end:],
            raw_context_x=raw_x[:context_end],
            raw_query_x=raw_x[context_end:query_end],
            raw_target_x=raw_x[query_end:],
            raw_context_y=raw_y[:context_end],
            raw_query_y=raw_y[context_end:query_end],
            raw_target_y=raw_y[query_end:],
            x_center=norm.x_center,
            x_scale=norm.x_scale,
            y_mean=norm.y_mean,
            y_std=norm.y_std,
        )

    @staticmethod
    def _validate_runtime_sizes(*, n_context: int, n_query: int, n_target: int) -> None:
        if int(n_context) <= 0:
            raise ValueError("n_context must be positive.")
        if int(n_query) <= 0:
            raise ValueError("n_query must be positive.")
        if int(n_target) <= 0:
            raise ValueError("n_target must be positive.")


__all__ = [
    "BenchmarkFunctionALTask",
    "BenchmarkFunctionEpisode",
]
