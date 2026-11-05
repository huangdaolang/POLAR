"""HPO-B task loader and episode sampler for Bayesian optimization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import torch

from polar.tasks.bo import BOEpisode, SplitName

HPOBBOEpisode = BOEpisode


class HPOBBODataLoader:
    """Load one HPO-B search space and sample BO episodes."""

    name = "hpob"

    SPLIT_FILE: dict[SplitName, str] = {
        "train": "meta-train-dataset.json",
        "validation": "meta-validation-dataset.json",
        "test": "meta-test-dataset.json",
    }
    BO_INITIALIZATION_FILE = "bo-initializations.json"
    DEFAULT_TEST_SEEDS: tuple[str, ...] = ("test0", "test1", "test2", "test3", "test4")

    def __init__(
        self,
        *,
        root_dir: str | Path,
        search_space_id: str,
        split: SplitName,
        normalize_targets: bool = False,
        seed: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.search_space_id = str(search_space_id)
        self.split = split
        self.normalize_targets = bool(normalize_targets)
        self._base_seed = int(seed) if seed is not None else 0

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self._base_seed)

        self._datasets = self._load_split_datasets()
        if not self._datasets:
            raise ValueError(
                f"No datasets found for search_space_id={self.search_space_id!r} in split={self.split!r}."
            )

        self._dataset_ids = sorted(self._datasets.keys())
        first_x = self._datasets[self._dataset_ids[0]][0]
        self.dim_x = int(first_x.shape[1])

        self._bo_initializations = self._load_bo_initializations() if self.split == "test" else None

    @property
    def dataset_ids(self) -> list[str]:
        return list(self._dataset_ids)

    @property
    def test_seeds(self) -> tuple[str, ...]:
        if self._bo_initializations is None:
            return self.DEFAULT_TEST_SEEDS

        for dataset_id in self._dataset_ids:
            seed_map = self._bo_initializations.get(dataset_id, {})
            if seed_map:
                return tuple(seed_map.keys())
        return self.DEFAULT_TEST_SEEDS

    def dataset_size(self, dataset_id: str) -> int:
        x, _ = self._get_dataset(dataset_id)
        return int(x.shape[0])

    def eligible_dataset_ids(self, *, min_total_size: int) -> list[str]:
        if min_total_size <= 0:
            raise ValueError("min_total_size must be positive.")
        return [
            dataset_id
            for dataset_id in self._dataset_ids
            if self.dataset_size(dataset_id) >= int(min_total_size)
        ]

    def sample_train_episode(
        self,
        *,
        n_context: int,
        n_query: int,
        n_target: int,
        dataset_id: str | None = None,
    ) -> BOEpisode:
        """Sample one train/validation episode with random context/query/target splits."""
        if self.split not in {"train", "validation"}:
            raise ValueError("sample_train_episode is only valid for train/validation splits.")
        if n_context <= 0:
            raise ValueError("n_context must be positive.")
        if n_query <= 0:
            raise ValueError("n_query must be positive.")
        if n_target < 0:
            raise ValueError("n_target must be non-negative.")

        required_size = int(n_context + n_query + n_target)
        if dataset_id is None:
            eligible = self.eligible_dataset_ids(min_total_size=required_size)
            if not eligible:
                raise ValueError(
                    f"No dataset in search_space={self.search_space_id} has size >= {required_size}."
                )
            dataset_id = eligible[int(torch.randint(len(eligible), (1,), generator=self._rng).item())]

        x, y = self._get_dataset(dataset_id)
        n_data = int(x.shape[0])
        if n_data < required_size:
            raise ValueError(
                f"Dataset {dataset_id} has {n_data} points but needs >= {required_size} "
                f"for n_context={n_context}, n_query={n_query}, n_target={n_target}."
            )

        perm = torch.randperm(n_data, generator=self._rng)
        context_idx = perm[:n_context]
        query_idx = perm[n_context : n_context + n_query]
        if n_target > 0:
            target_idx = perm[n_context + n_query : n_context + n_query + n_target]
            target_x = x[target_idx]
            target_y = y[target_idx]
        else:
            target_x = None
            target_y = None

        return BOEpisode(
            search_space_id=self.search_space_id,
            dataset_id=dataset_id,
            split=self.split,
            seed=None,
            context_x=x[context_idx],
            context_y=y[context_idx],
            query_x=x[query_idx],
            query_y=y[query_idx],
            target_x=target_x,
            target_y=target_y,
            y_best=float(y.max().item()),
            y_min=float(y.min().item()),
        )

    def sample_train_batch(
        self,
        *,
        batch_size: int,
        n_context: int,
        n_query: int,
        n_target: int,
    ) -> list[BOEpisode]:
        """Sample a batch of train/validation episodes from eligible datasets."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        required_size = int(n_context + n_query + n_target)
        eligible = self.eligible_dataset_ids(min_total_size=required_size)
        if not eligible:
            raise ValueError(
                f"No dataset in search_space={self.search_space_id} has size >= {required_size}."
            )

        episodes: list[BOEpisode] = []
        for _ in range(batch_size):
            dataset_id = eligible[int(torch.randint(len(eligible), (1,), generator=self._rng).item())]
            episodes.append(
                self.sample_train_episode(
                    n_context=n_context,
                    n_query=n_query,
                    n_target=n_target,
                    dataset_id=dataset_id,
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
        shuffle_query_pool: bool = True,
    ) -> BOEpisode:
        """Build one test episode using official HPO-B BO initializations."""
        if self.split != "test":
            raise ValueError("build_test_episode is only valid for test split.")
        if n_context <= 0:
            raise ValueError("n_context must be positive.")

        x, y = self._get_dataset(dataset_id)
        n_data = int(x.shape[0])

        init_ids = self._get_init_indices(dataset_id=dataset_id, seed=seed)
        if not init_ids:
            raise ValueError(f"Initialization list is empty for dataset={dataset_id}, seed={seed}.")

        init_ids_tensor = torch.tensor(init_ids, dtype=torch.long)
        if torch.any(init_ids_tensor < 0) or torch.any(init_ids_tensor >= n_data):
            raise ValueError(
                f"Found invalid initialization index for dataset={dataset_id}, n_data={n_data}."
            )

        episode_rng = self._build_episode_generator(dataset_id=dataset_id, seed=seed)

        if n_context <= len(init_ids):
            context_idx = init_ids_tensor[:n_context]
        else:
            extra_needed = n_context - len(init_ids)
            pending_mask = torch.ones(n_data, dtype=torch.bool)
            pending_mask[init_ids_tensor] = False
            pending_idx = torch.nonzero(pending_mask, as_tuple=False).view(-1)
            if pending_idx.numel() < extra_needed:
                raise ValueError(
                    f"Dataset {dataset_id} has only {n_data} points; cannot build n_context={n_context}."
                )
            perm = torch.randperm(int(pending_idx.numel()), generator=episode_rng)
            extra_idx = pending_idx[perm[:extra_needed]]
            context_idx = torch.cat([init_ids_tensor, extra_idx], dim=0)

        pending_mask = torch.ones(n_data, dtype=torch.bool)
        pending_mask[context_idx] = False
        pending_idx = torch.nonzero(pending_mask, as_tuple=False).view(-1)
        pending_count = int(pending_idx.numel())
        if pending_count <= 0:
            raise ValueError(f"Dataset {dataset_id} has no remaining query points after context.")

        if query_pool_size <= 0:
            query_size = pending_count
        else:
            query_size = min(int(query_pool_size), pending_count)
        if query_size <= 0:
            raise ValueError("query_size resolved to zero.")

        if shuffle_query_pool:
            perm = torch.randperm(pending_count, generator=episode_rng)
            pending_idx = pending_idx[perm]
        query_idx = pending_idx[:query_size]

        return BOEpisode(
            search_space_id=self.search_space_id,
            dataset_id=dataset_id,
            split=self.split,
            seed=seed,
            context_x=x[context_idx],
            context_y=y[context_idx],
            query_x=x[query_idx],
            query_y=y[query_idx],
            target_x=None,
            target_y=None,
            y_best=float(y.max().item()),
            y_min=float(y.min().item()),
        )

    def iter_test_episodes(
        self,
        *,
        n_context: int,
        query_pool_size: int,
        seeds: list[str] | tuple[str, ...] | None = None,
        dataset_ids: list[str] | tuple[str, ...] | None = None,
        shuffle_query_pool: bool = True,
    ) -> Iterator[BOEpisode]:
        """Iterate test episodes over dataset_ids x seeds."""
        if self.split != "test":
            raise ValueError("iter_test_episodes is only valid for test split.")

        active_dataset_ids = list(dataset_ids) if dataset_ids is not None else self._dataset_ids
        active_seeds = list(seeds) if seeds is not None else list(self.test_seeds)

        for dataset_id in active_dataset_ids:
            for seed in active_seeds:
                if not self._has_seed(dataset_id=dataset_id, seed=seed):
                    continue
                try:
                    yield self.build_test_episode(
                        dataset_id=dataset_id,
                        seed=seed,
                        n_context=n_context,
                        query_pool_size=query_pool_size,
                        shuffle_query_pool=shuffle_query_pool,
                    )
                except ValueError:
                    # Keep iteration robust when some dataset/seed pair is too small for settings.
                    continue

    def _load_split_datasets(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        split_filename = self.SPLIT_FILE[self.split]
        split_path = self.root_dir / split_filename
        if not split_path.exists():
            raise FileNotFoundError(f"Missing HPO-B split file: {split_path}")

        with split_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        space_data = payload.get(self.search_space_id)
        if space_data is None:
            available = ", ".join(sorted(payload.keys())[:20])
            raise KeyError(
                f"Unknown search_space_id '{self.search_space_id}' in {split_filename}. "
                f"Example available ids: {available}"
            )

        datasets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        expected_dim: int | None = None
        for dataset_id, record in space_data.items():
            x = torch.tensor(record["X"], dtype=torch.float32)
            y = torch.tensor(record["y"], dtype=torch.float32).view(-1)
            if x.ndim != 2:
                raise ValueError(
                    f"Expected X to be 2D for dataset={dataset_id}, got shape={tuple(x.shape)}."
                )
            if y.ndim != 1:
                raise ValueError(
                    f"Expected y to be 1D for dataset={dataset_id}, got shape={tuple(y.shape)}."
                )
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X/y size mismatch for dataset={dataset_id}: {x.shape[0]} vs {y.shape[0]}."
                )

            if self.normalize_targets:
                y = self._normalize_targets(y)

            if expected_dim is None:
                expected_dim = int(x.shape[1])
            elif int(x.shape[1]) != expected_dim:
                raise ValueError(
                    f"Inconsistent feature dimensions within search_space_id={self.search_space_id}: "
                    f"{int(x.shape[1])} vs {expected_dim}."
                )
            datasets[str(dataset_id)] = (x, y)

        return datasets

    def _load_bo_initializations(self) -> dict[str, dict[str, list[int]]]:
        init_path = self.root_dir / self.BO_INITIALIZATION_FILE
        if not init_path.exists():
            raise FileNotFoundError(f"Missing HPO-B initialization file: {init_path}")

        with init_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        space_inits = payload.get(self.search_space_id)
        if space_inits is None:
            raise KeyError(
                f"Initialization file does not contain search_space_id={self.search_space_id}."
            )
        return space_inits

    def _get_dataset(self, dataset_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._datasets[str(dataset_id)]
        except KeyError as exc:
            raise KeyError(
                f"Unknown dataset_id '{dataset_id}' for search_space_id={self.search_space_id}."
            ) from exc

    def _get_init_indices(self, *, dataset_id: str, seed: str) -> list[int]:
        if self._bo_initializations is None:
            raise RuntimeError("Initializations are only available for test split.")
        try:
            init_ids = self._bo_initializations[dataset_id][seed]
        except KeyError as exc:
            raise KeyError(
                f"Missing initialization for dataset_id={dataset_id}, seed={seed}, "
                f"search_space_id={self.search_space_id}."
            ) from exc
        return [int(idx) for idx in init_ids]

    def _has_seed(self, *, dataset_id: str, seed: str) -> bool:
        if self._bo_initializations is None:
            return False
        dataset_inits = self._bo_initializations.get(dataset_id, {})
        return seed in dataset_inits

    def _build_episode_generator(self, *, dataset_id: str, seed: str) -> torch.Generator:
        payload = f"{self.search_space_id}:{dataset_id}:{seed}:{self._base_seed}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        seed_int = int.from_bytes(digest[:8], byteorder="little", signed=False)
        seed_int = seed_int % (2**63 - 1)
        episode_rng = torch.Generator(device="cpu")
        episode_rng.manual_seed(seed_int)
        return episode_rng

    @staticmethod
    def _normalize_targets(y: torch.Tensor) -> torch.Tensor:
        y_min = torch.min(y)
        y_max = torch.max(y)
        scale = y_max - y_min
        if float(scale.item()) <= 1e-12:
            return torch.zeros_like(y)
        return (y - y_min) / scale
