"""DOCKSTRING task loader and cache helpers for pool-based Bayesian optimization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch

from polar.tasks.bo import BOEpisode, SplitName

DOCKSTRING_CACHE_VERSION = 1


def prepare_dockstring_cache(
    *,
    raw_data_path: str | Path,
    cache_path: str | Path,
    subset_size: int,
    subset_seed: int,
    fingerprint_type: str,
    fingerprint_radius: int,
    fingerprint_n_bits: int,
) -> Path:
    """Create a reusable fixed-subset feature cache for DOCKSTRING."""

    raw_path = Path(raw_data_path).expanduser().resolve()
    output_path = Path(cache_path).expanduser().resolve()
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    if fingerprint_radius <= 0:
        raise ValueError("fingerprint_radius must be positive.")
    if fingerprint_n_bits <= 0:
        raise ValueError("fingerprint_n_bits must be positive.")

    fp_type = fingerprint_type.strip().lower()
    if fp_type != "morgan":
        raise ValueError(f"Unsupported fingerprint_type='{fingerprint_type}'.")

    if not raw_path.exists():
        raise FileNotFoundError(f"DOCKSTRING raw data file does not exist: {raw_path}")

    df = pd.read_csv(raw_path, sep="\t")
    required_columns = {"inchikey", "smiles"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"DOCKSTRING file is missing required columns: {missing_text}")

    effective_subset_size = min(int(subset_size), int(len(df)))
    sampled = (
        df.sample(n=effective_subset_size, random_state=int(subset_seed))
        .sort_index()
        .reset_index(drop=False)
        .rename(columns={"index": "source_row_index"})
    )
    target_columns = [
        col for col in sampled.columns if col not in {"source_row_index", "inchikey", "smiles"}
    ]
    feature_matrix = _compute_morgan_fingerprints(
        smiles_list=[str(smiles) for smiles in sampled["smiles"].tolist()],
        radius=int(fingerprint_radius),
        n_bits=int(fingerprint_n_bits),
    )

    payload = {
        "version": DOCKSTRING_CACHE_VERSION,
        "source_path": str(raw_path),
        "source_num_rows": int(len(df)),
        "subset_size": int(effective_subset_size),
        "subset_seed": int(subset_seed),
        "fingerprint_type": fp_type,
        "fingerprint_radius": int(fingerprint_radius),
        "fingerprint_n_bits": int(fingerprint_n_bits),
        "source_row_index": torch.tensor(sampled["source_row_index"].tolist(), dtype=torch.int64),
        "inchikey": [str(value) for value in sampled["inchikey"].tolist()],
        "smiles": [str(value) for value in sampled["smiles"].tolist()],
        "target_names": list(target_columns),
        "features": feature_matrix.to(torch.uint8),
        "targets": {
            target: torch.tensor(sampled[target].to_numpy(), dtype=torch.float32)
            for target in target_columns
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


class DockstringBODataLoader:
    """Load a fixed-subset DOCKSTRING cache and sample BO episodes across targets."""

    name = "dockstring"
    DEFAULT_TEST_SEEDS: tuple[str, ...] = ("test0", "test1", "test2", "test3", "test4")

    def __init__(
        self,
        *,
        cache_path: str | Path,
        raw_data_path: str | Path | None,
        split: SplitName,
        train_targets: Sequence[str] | None,
        test_targets: Sequence[str] | None,
        subset_size: int,
        subset_seed: int,
        fingerprint_type: str,
        fingerprint_radius: int,
        fingerprint_n_bits: int,
        normalize_targets: bool = False,
        negate_scores: bool = True,
        seed: int | None = None,
    ) -> None:
        if split == "validation":
            raise NotImplementedError(
                "DockstringBODataLoader does not define a validation split yet. "
                "Use split=train or split=test."
            )

        self.cache_path = Path(cache_path).expanduser().resolve()
        self.raw_data_path = (
            Path(raw_data_path).expanduser().resolve() if raw_data_path is not None else None
        )
        self.split = split
        self.normalize_targets = bool(normalize_targets)
        self.negate_scores = bool(negate_scores)
        self.subset_size = int(subset_size)
        self.subset_seed = int(subset_seed)
        self.fingerprint_type = str(fingerprint_type).strip().lower()
        self.fingerprint_radius = int(fingerprint_radius)
        self.fingerprint_n_bits = int(fingerprint_n_bits)
        self._base_seed = int(seed) if seed is not None else 0
        self.search_space_id = self.name

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self._base_seed)

        payload = self._load_or_prepare_cache()
        self._features = torch.as_tensor(payload["features"], dtype=torch.float32)
        if self._features.ndim != 2:
            raise ValueError(
                f"Cached features must be 2D, got shape={tuple(self._features.shape)}."
            )
        self.dim_x = int(self._features.shape[1])
        self._target_names = [str(name) for name in payload["target_names"]]
        self._target_scores = {
            str(target): torch.as_tensor(values, dtype=torch.float32).view(-1)
            for target, values in payload["targets"].items()
        }

        self._train_targets, self._test_targets = self._resolve_target_partitions(
            train_targets=train_targets,
            test_targets=test_targets,
        )
        if self.split == "train":
            self._dataset_ids = list(self._train_targets)
        else:
            self._dataset_ids = list(self._test_targets)
        if not self._dataset_ids:
            raise ValueError(f"No targets configured for split={self.split!r}.")

        self._datasets = {dataset_id: self._build_dataset(dataset_id) for dataset_id in self._dataset_ids}
        for dataset_id, (x, y) in self._datasets.items():
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Feature/target size mismatch for target={dataset_id}: "
                    f"{x.shape[0]} vs {y.shape[0]}."
                )

    @property
    def dataset_ids(self) -> list[str]:
        return list(self._dataset_ids)

    @property
    def test_seeds(self) -> tuple[str, ...]:
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
        """Sample one train episode with random context/query/target splits."""

        if self.split != "train":
            raise ValueError("sample_train_episode is only valid for split=train.")
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
                    f"No Dockstring target has size >= {required_size} for split={self.split!r}."
                )
            dataset_id = eligible[int(torch.randint(len(eligible), (1,), generator=self._rng).item())]

        x, y = self._get_dataset(dataset_id)
        n_data = int(x.shape[0])
        if n_data < required_size:
            raise ValueError(
                f"Target {dataset_id} has {n_data} points but needs >= {required_size} "
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
        """Sample a batch of train episodes from configured training targets."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        required_size = int(n_context + n_query + n_target)
        eligible = self.eligible_dataset_ids(min_total_size=required_size)
        if not eligible:
            raise ValueError(
                f"No Dockstring target has size >= {required_size} for split={self.split!r}."
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
        """Build one test episode with deterministic seeded initialization."""

        if self.split != "test":
            raise ValueError("build_test_episode is only valid for split=test.")
        if n_context <= 0:
            raise ValueError("n_context must be positive.")

        x, y = self._get_dataset(dataset_id)
        n_data = int(x.shape[0])
        if n_data <= n_context:
            raise ValueError(
                f"Target {dataset_id} has only {n_data} valid points; cannot build n_context={n_context}."
            )

        episode_rng = self._build_episode_generator(dataset_id=dataset_id, seed=seed)
        perm = torch.randperm(n_data, generator=episode_rng)
        context_idx = perm[:n_context]
        pending_idx = perm[n_context:]
        pending_count = int(pending_idx.numel())
        if pending_count <= 0:
            raise ValueError(f"Target {dataset_id} has no remaining query points after context.")

        if query_pool_size <= 0:
            query_size = pending_count
        else:
            query_size = min(int(query_pool_size), pending_count)
        if query_size <= 0:
            raise ValueError("query_size resolved to zero.")

        if shuffle_query_pool:
            query_perm = torch.randperm(pending_count, generator=episode_rng)
            pending_idx = pending_idx[query_perm]
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
            raise ValueError("iter_test_episodes is only valid for split=test.")

        active_dataset_ids = list(dataset_ids) if dataset_ids is not None else list(self._dataset_ids)
        active_seeds = list(seeds) if seeds is not None else list(self.test_seeds)

        for dataset_id in active_dataset_ids:
            for seed_name in active_seeds:
                yield self.build_test_episode(
                    dataset_id=dataset_id,
                    seed=seed_name,
                    n_context=n_context,
                    query_pool_size=query_pool_size,
                    shuffle_query_pool=shuffle_query_pool,
                )

    def _load_or_prepare_cache(self) -> dict[str, object]:
        if self.cache_path.exists():
            payload = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict payload in cache file: {self.cache_path}")
        else:
            if self.raw_data_path is None:
                raise FileNotFoundError(
                    f"Dockstring cache does not exist: {self.cache_path}. "
                    "Provide task.raw_data_path so the cache can be generated."
                )
            prepare_dockstring_cache(
                raw_data_path=self.raw_data_path,
                cache_path=self.cache_path,
                subset_size=self.subset_size,
                subset_seed=self.subset_seed,
                fingerprint_type=self.fingerprint_type,
                fingerprint_radius=self.fingerprint_radius,
                fingerprint_n_bits=self.fingerprint_n_bits,
            )
            payload = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict payload in cache file: {self.cache_path}")

        self._validate_cache_metadata(payload)
        return payload

    def _validate_cache_metadata(self, payload: dict[str, object]) -> None:
        version = int(payload.get("version", -1))
        if version != DOCKSTRING_CACHE_VERSION:
            raise ValueError(
                f"Unsupported Dockstring cache version={version}; "
                f"expected {DOCKSTRING_CACHE_VERSION}."
            )

        source_num_rows = payload.get("source_num_rows")
        observed_subset_size = payload.get("subset_size")
        if observed_subset_size is None:
            raise KeyError("Dockstring cache payload must contain 'subset_size'.")
        observed_subset_size_int = int(observed_subset_size)
        if observed_subset_size_int <= 0:
            raise ValueError(
                f"Dockstring cache has invalid subset_size={observed_subset_size_int}."
            )

        if source_num_rows is not None:
            expected_subset_size = min(self.subset_size, int(source_num_rows))
            if observed_subset_size_int != expected_subset_size:
                raise ValueError(
                    "Dockstring cache metadata mismatch for subset_size: "
                    f"expected effective subset {expected_subset_size!r}, "
                    f"found {observed_subset_size_int!r}. Regenerate or point "
                    "task.cache_path to a matching cache."
                )
        elif observed_subset_size_int > self.subset_size:
            raise ValueError(
                "Dockstring cache metadata mismatch for subset_size: "
                f"requested at most {self.subset_size!r}, found {observed_subset_size_int!r}. "
                "Regenerate or point task.cache_path to a matching cache."
            )

        expected = {
            "subset_seed": self.subset_seed,
            "fingerprint_type": self.fingerprint_type,
            "fingerprint_radius": self.fingerprint_radius,
            "fingerprint_n_bits": self.fingerprint_n_bits,
        }
        for key, expected_value in expected.items():
            observed_value = payload.get(key)
            if observed_value != expected_value:
                raise ValueError(
                    f"Dockstring cache metadata mismatch for {key}: "
                    f"expected {expected_value!r}, found {observed_value!r}. "
                    f"Regenerate or point task.cache_path to a matching cache."
                )

        if "features" not in payload or "targets" not in payload or "target_names" not in payload:
            raise KeyError(
                "Dockstring cache payload must contain 'features', 'targets', and 'target_names'."
            )

    def _resolve_target_partitions(
        self,
        *,
        train_targets: Sequence[str] | None,
        test_targets: Sequence[str] | None,
    ) -> tuple[list[str], list[str]]:
        available = set(self._target_names)
        train = _normalize_targets(train_targets)
        test = _normalize_targets(test_targets)

        missing_train = sorted(set(train).difference(available))
        missing_test = sorted(set(test).difference(available))
        if missing_train or missing_test:
            missing_all = ", ".join(missing_train + missing_test)
            raise KeyError(f"Unknown Dockstring target(s): {missing_all}")

        overlap = sorted(set(train).intersection(test))
        if overlap:
            overlap_text = ", ".join(overlap)
            raise ValueError(f"Dockstring train_targets and test_targets overlap: {overlap_text}")

        if not train and test:
            train = sorted(available.difference(test))
        if not test and train:
            test = sorted(available.difference(train))
        if not train and not test:
            raise ValueError(
                "Dockstring train_targets and test_targets are both empty. "
                "Specify at least one partition explicitly to avoid train/test leakage."
            )

        return list(train), list(test)

    def _build_dataset(self, dataset_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self._target_scores.get(dataset_id)
        if scores is None:
            raise KeyError(f"Unknown Dockstring target: {dataset_id}")

        mask = torch.isfinite(scores)
        x = self._features[mask]
        y = scores[mask]
        if self.negate_scores:
            y = -y
        if self.normalize_targets:
            y = self._normalize_targets(y)
        return x.clone(), y.clone()

    def _get_dataset(self, dataset_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._datasets[str(dataset_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset_id '{dataset_id}' for Dockstring split={self.split!r}.") from exc

    def _build_episode_generator(self, *, dataset_id: str, seed: str) -> torch.Generator:
        payload = f"{self.name}:{dataset_id}:{seed}:{self._base_seed}:{self.subset_seed}".encode(
            "utf-8"
        )
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


def _normalize_targets(targets: Sequence[str] | None) -> list[str]:
    if targets is None:
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for target in targets:
        target_name = str(target).strip()
        if not target_name or target_name in seen:
            continue
        seen.add(target_name)
        normalized.append(target_name)
    return normalized


def _compute_morgan_fingerprints(
    *,
    smiles_list: Sequence[str],
    radius: int,
    n_bits: int,
) -> torch.Tensor:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to build a Dockstring Morgan fingerprint cache. "
            "Install RDKit, then rerun the command."
        ) from exc

    fp_rows: list[torch.Tensor] = []
    invalid_smiles: list[str] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append(smiles)
            continue

        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        fp_array = np.zeros(n_bits, dtype=np.int32)
        DataStructs.ConvertToNumpyArray(bit_vect, fp_array)
        fp_rows.append(torch.from_numpy(fp_array.astype(np.uint8, copy=False)))

    if invalid_smiles:
        preview = ", ".join(invalid_smiles[:5])
        raise ValueError(
            f"Failed to parse {len(invalid_smiles)} SMILES while building Dockstring cache. "
            f"Examples: {preview}"
        )

    if not fp_rows:
        raise ValueError("No valid SMILES were found while building the Dockstring cache.")

    return torch.stack(fp_rows, dim=0)


__all__ = [
    "DOCKSTRING_CACHE_VERSION",
    "DockstringBODataLoader",
    "prepare_dockstring_cache",
]
