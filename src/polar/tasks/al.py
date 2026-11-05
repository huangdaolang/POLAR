"""Shared task interfaces for pool-based active learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol

import torch

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class ALEpisode:
    """One AL episode with pool-based context/query/target splits."""

    dataset_id: str
    split: SplitName
    seed: str | None
    context_x: torch.Tensor
    context_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    target_x: torch.Tensor
    target_y: torch.Tensor


class ALTask(Protocol):
    """Minimal interface required by real-data AL training and evaluation."""

    name: str
    split: SplitName
    dim_x: int
    dataset_ids: list[str]
    test_seeds: tuple[str, ...]

    def eligible_dataset_ids(self, *, min_total_size: int) -> list[str]: ...

    def sample_train_batch(
        self,
        *,
        batch_size: int,
        n_context: int,
        n_query: int,
        n_target: int,
    ) -> list[ALEpisode]: ...

    def iter_test_episodes(
        self,
        *,
        n_context: int,
        query_pool_size: int,
        n_target: int,
        seeds: list[str] | tuple[str, ...] | None = None,
        dataset_ids: list[str] | tuple[str, ...] | None = None,
        shuffle_query_pool: bool = True,
    ) -> Iterator[ALEpisode]: ...


def create_al_task(task_cfg: Any, *, seed: int | None = None) -> ALTask:
    """Instantiate an AL task from config."""

    task_name = str(getattr(task_cfg, "name")).strip().lower()

    if task_name == "gp_regression":
        from polar.tasks.gp_regression import GPPriorALTask

        kernels = getattr(task_cfg.gp_prior, "kernels", None)
        kernel_weights_cfg = getattr(task_cfg.gp_prior, "kernel_weights", None)
        if kernel_weights_cfg is None:
            kernel_weights = {}
        elif isinstance(kernel_weights_cfg, dict):
            kernel_weights = {str(k): float(v) for k, v in kernel_weights_cfg.items()}
        else:
            kernel_weights = {str(k): float(v) for k, v in kernel_weights_cfg.items()}
        if kernels is not None:
            kernel_names = [str(name) for name in kernels]
            if kernel_weights:
                kernel_weights = {name: float(kernel_weights.get(name, 0.0)) for name in kernel_names}
            else:
                kernel_weights = {name: 1.0 for name in kernel_names}
        if not kernel_weights:
            kernel_weights = {"rbf": 1.0, "matern12": 0.0, "matern32": 0.5, "matern52": 0.5}
        return GPPriorALTask(
            dim_x=int(task_cfg.dim_x) if getattr(task_cfg, "dim_x", None) is not None else None,
            dim_x_min=(
                int(task_cfg.dim_x_min)
                if getattr(task_cfg, "dim_x_min", None) is not None
                else None
            ),
            dim_x_max=(
                int(task_cfg.dim_x_max)
                if getattr(task_cfg, "dim_x_max", None) is not None
                else None
            ),
            design_min=float(task_cfg.design_min),
            design_max=float(task_cfg.design_max),
            plot_grid_size=int(task_cfg.plot_grid_size),
            noise_std=float(task_cfg.noise_std),
            lengthscale_min=float(task_cfg.gp_prior.lengthscale_min),
            lengthscale_max=float(task_cfg.gp_prior.lengthscale_max),
            scale_min=float(task_cfg.gp_prior.scale_min),
            scale_max=float(task_cfg.gp_prior.scale_max),
            p_isotropic=float(task_cfg.gp_prior.p_isotropic),
            kernel_weights={str(k): float(v) for k, v in kernel_weights.items()},
            jitter=float(task_cfg.gp_prior.jitter),
            cholesky_max_tries=int(task_cfg.gp_prior.cholesky_max_tries),
            jitter_growth=float(task_cfg.gp_prior.jitter_growth),
            split=str(getattr(task_cfg, "split", "train")),
            lengthscale_sampling=str(getattr(task_cfg.gp_prior, "lengthscale_sampling", "uniform")),
            standardize_x=bool(getattr(task_cfg.gp_prior, "standardize_x", False)),
            standardize_y=bool(getattr(task_cfg.gp_prior, "standardize_y", False)),
            seed=seed,
        )

    if task_name == "benchmark_functions":
        from polar.tasks.benchmark_functions import BenchmarkFunctionALTask

        benchmarks_cfg = getattr(task_cfg, "benchmarks", None)
        if benchmarks_cfg is None:
            raise ValueError("BenchmarkFunctionALTask requires task.benchmarks.")
        benchmarks = [str(name) for name in benchmarks_cfg]
        return BenchmarkFunctionALTask(
            benchmarks=benchmarks,
            split=str(getattr(task_cfg, "split", "test")),
            normalize_x=bool(getattr(task_cfg, "normalize_x", True)),
            normalize_y=bool(getattr(task_cfg, "normalize_y", True)),
            normalization_num_samples=int(getattr(task_cfg, "normalization_num_samples", 8192)),
            num_eval_episodes=int(getattr(task_cfg, "num_eval_episodes", 50)),
            group_by_dim=bool(getattr(task_cfg, "group_by_dim", True)),
            seed=seed,
        )

    raise ValueError(f"Unsupported AL task '{task_name}'.")


__all__ = [
    "ALEpisode",
    "ALTask",
    "SplitName",
    "create_al_task",
]
