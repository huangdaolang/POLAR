"""Shared task interfaces for pool-based Bayesian optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol

import torch

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class BOEpisode:
    """One BO episode with pool-based context/query/(optional)target splits."""

    search_space_id: str
    dataset_id: str
    split: SplitName
    seed: str | None
    context_x: torch.Tensor
    context_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    target_x: torch.Tensor | None
    target_y: torch.Tensor | None
    y_best: float
    y_min: float


class BOTask(Protocol):
    """Minimal interface required by BO training and evaluation."""

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
    ) -> list[BOEpisode]: ...

    def iter_test_episodes(
        self,
        *,
        n_context: int,
        query_pool_size: int,
        seeds: list[str] | tuple[str, ...] | None = None,
        dataset_ids: list[str] | tuple[str, ...] | None = None,
        shuffle_query_pool: bool = True,
    ) -> Iterator[BOEpisode]: ...


def create_bo_task(task_cfg: Any, *, seed: int | None = None) -> BOTask:
    """Instantiate a BO task from config."""

    task_name = str(getattr(task_cfg, "name")).strip().lower()

    if task_name == "hpob":
        from polar.tasks.hpob import HPOBBODataLoader

        return HPOBBODataLoader(
            root_dir=str(task_cfg.data_root),
            search_space_id=str(task_cfg.search_space_id),
            split=str(task_cfg.split),
            normalize_targets=bool(task_cfg.normalize_targets),
            seed=seed,
        )

    if task_name == "dockstring":
        from polar.tasks.dockstring import DockstringBODataLoader

        cache_path = Path(task_cfg.cache_path).expanduser()
        raw_data_path = getattr(task_cfg, "raw_data_path", None)
        train_targets = getattr(task_cfg, "train_targets", None)
        test_targets = getattr(task_cfg, "test_targets", None)

        return DockstringBODataLoader(
            cache_path=cache_path,
            raw_data_path=(Path(raw_data_path).expanduser() if raw_data_path else None),
            split=str(task_cfg.split),
            train_targets=_cfg_list_or_none(train_targets),
            test_targets=_cfg_list_or_none(test_targets),
            subset_size=int(task_cfg.subset_size),
            subset_seed=int(task_cfg.subset_seed),
            fingerprint_type=str(task_cfg.representation.type),
            fingerprint_radius=int(task_cfg.representation.radius),
            fingerprint_n_bits=int(task_cfg.representation.n_bits),
            normalize_targets=bool(task_cfg.normalize_targets),
            negate_scores=bool(task_cfg.negate_scores),
            seed=seed,
        )

    raise ValueError(f"Unsupported BO task '{task_name}'.")


def _cfg_list_or_none(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    return [str(item) for item in value]


__all__ = [
    "BOEpisode",
    "BOTask",
    "SplitName",
    "create_bo_task",
]
