"""Shared task interfaces for Bayesian Experimental Design."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from torch import Tensor

BEDTrajectory = dict[str, Any]
BEDTrajectoryPlotter = Callable[[Path, list[BEDTrajectory]], None]
BEDObservationType = Literal["continuous", "binary"]


@dataclass(frozen=True)
class BEDEpisode:
    """One pool-based BED episode.

    ``theta_all[0]`` is always the true latent parameter used to generate
    observations. Remaining entries are contrastive latent samples from the
    task prior.
    """

    theta_all: Tensor
    context_x: Tensor
    context_y: Tensor
    query_x: Tensor
    query_y: Tensor
    target_x: Tensor | None
    target_y: Tensor | None


class BEDTask(Protocol):
    """Minimal interface required by BED training and evaluation."""

    name: str
    dim_x: int
    n_contrastive: int
    n_context_init: int
    n_query: int
    n_target: int
    design_min: float
    design_max: float
    trajectory_plotter: BEDTrajectoryPlotter | None
    observation_type: BEDObservationType
    num_classes: int | None

    def sample_episode(self) -> BEDEpisode: ...

    def sample_batch(self, *, batch_size: int) -> list[BEDEpisode]: ...

    def sample_theta(self, n: int) -> Tensor: ...

    def log_likelihood(self, y: Tensor, x: Tensor, theta: Tensor) -> Tensor: ...


def bed_task_checkpoint_kind(task: BEDTask) -> Literal["regression", "classification"]:
    """Choose backbone checkpoint kind from the BED task observation model."""
    if task.observation_type == "binary":
        return "classification"
    if task.observation_type == "continuous":
        return "regression"
    raise ValueError(f"Unsupported BED observation_type '{task.observation_type}'.")


def create_bed_task(task_cfg: Any, *, seed: int | None = None) -> BEDTask:
    """Instantiate a BED task from config.

    BED tasks share a runtime contract, but each task keeps its own config
    fields and latent-parameter structure.
    """

    task_name = str(getattr(task_cfg, "name")).strip().lower()

    if task_name == "location_finding":
        from polar.tasks.location_finding import LocationFindingSampler

        return LocationFindingSampler(
            dim_x=int(task_cfg.dim_x),
            K=int(task_cfg.K),
            n_contrastive=int(task_cfg.n_contrastive),
            n_context_init=int(task_cfg.n_context_init),
            n_query=int(task_cfg.n_query),
            n_target=int(task_cfg.n_target),
            design_min=float(task_cfg.design_min),
            design_max=float(task_cfg.design_max),
            base_signal=float(task_cfg.signal.base_signal),
            max_signal=float(task_cfg.signal.max_signal),
            noise_scale=float(task_cfg.signal.noise_scale),
            theta_prior=str(task_cfg.theta_prior),
            seed=seed,
        )

    if task_name == "ces":
        from polar.tasks.ces import CESSampler

        return CESSampler(
            n_goods=int(task_cfg.n_goods),
            n_contrastive=int(task_cfg.n_contrastive),
            n_context_init=int(task_cfg.n_context_init),
            n_query=int(task_cfg.n_query),
            n_target=int(task_cfg.n_target),
            design_min=float(task_cfg.design_min),
            design_max=float(task_cfg.design_max),
            noise_scale=float(task_cfg.noise_scale),
            epsilon=float(task_cfg.epsilon),
            seed=seed,
        )

    raise ValueError(f"Unsupported BED task '{task_name}'.")


__all__ = [
    "BEDEpisode",
    "BEDObservationType",
    "BEDTask",
    "BEDTrajectory",
    "BEDTrajectoryPlotter",
    "bed_task_checkpoint_kind",
    "create_bed_task",
]
