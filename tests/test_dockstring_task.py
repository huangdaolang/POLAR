from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from polar.tasks import DockstringBODataLoader, create_bo_task


def _write_cache(path: Path) -> None:
    payload = {
        "version": 1,
        "source_path": str(path.with_suffix(".tsv")),
        "source_num_rows": 6,
        "subset_size": 6,
        "subset_seed": 13,
        "fingerprint_type": "morgan",
        "fingerprint_radius": 2,
        "fingerprint_n_bits": 4,
        "source_row_index": torch.arange(6, dtype=torch.int64),
        "inchikey": [f"ik{i}" for i in range(6)],
        "smiles": [f"C{i}" for i in range(6)],
        "target_names": ["JAK2", "KIT", "PTK2"],
        "features": torch.tensor(
            [
                [1, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            dtype=torch.uint8,
        ),
        "targets": {
            "JAK2": torch.tensor([-8.0, -7.0, -9.5, -8.7, -6.5, -10.0], dtype=torch.float32),
            "KIT": torch.tensor([-7.5, -8.1, -8.4, -9.0, -7.8, -8.8], dtype=torch.float32),
            "PTK2": torch.tensor([-7.9, float("nan"), -8.2, -8.7, -7.4, -9.1], dtype=torch.float32),
        },
    }
    torch.save(payload, path)


def test_create_bo_task_returns_dockstring_loader(tmp_path: Path) -> None:
    cache_path = tmp_path / "dockstring_cache.pt"
    _write_cache(cache_path)

    cfg = OmegaConf.create(
        {
            "name": "dockstring",
            "cache_path": str(cache_path),
            "raw_data_path": None,
            "split": "train",
            "subset_size": 6,
            "subset_seed": 13,
            "representation": {"type": "morgan", "radius": 2, "n_bits": 4},
            "train_targets": ["JAK2", "KIT"],
            "test_targets": ["PTK2"],
            "normalize_targets": False,
            "negate_scores": True,
        }
    )

    task = create_bo_task(cfg, seed=17)
    assert isinstance(task, DockstringBODataLoader)
    assert task.dataset_ids == ["JAK2", "KIT"]
    assert task.dim_x == 4

    episodes = task.sample_train_batch(batch_size=3, n_context=2, n_query=2, n_target=1)
    assert len(episodes) == 3
    for episode in episodes:
        assert episode.dataset_id in {"JAK2", "KIT"}
        assert episode.context_x.shape == torch.Size([2, 4])
        assert episode.query_x.shape == torch.Size([2, 4])
        assert episode.target_x is not None
        assert episode.target_x.shape == torch.Size([1, 4])
        # Targets are negated, so better docking scores become larger BO rewards.
        assert float(episode.y_best) > 0.0


def test_dockstring_test_episodes_are_deterministic(tmp_path: Path) -> None:
    cache_path = tmp_path / "dockstring_cache.pt"
    _write_cache(cache_path)

    loader = DockstringBODataLoader(
        cache_path=cache_path,
        raw_data_path=None,
        split="test",
        train_targets=["JAK2", "KIT"],
        test_targets=["PTK2"],
        subset_size=6,
        subset_seed=13,
        fingerprint_type="morgan",
        fingerprint_radius=2,
        fingerprint_n_bits=4,
        normalize_targets=False,
        negate_scores=True,
        seed=5,
    )

    ep_a = loader.build_test_episode(
        dataset_id="PTK2",
        seed="test0",
        n_context=2,
        query_pool_size=2,
        shuffle_query_pool=True,
    )
    ep_b = loader.build_test_episode(
        dataset_id="PTK2",
        seed="test0",
        n_context=2,
        query_pool_size=2,
        shuffle_query_pool=True,
    )

    assert loader.dataset_ids == ["PTK2"]
    assert torch.equal(ep_a.context_x, ep_b.context_x)
    assert torch.equal(ep_a.query_x, ep_b.query_x)
    assert torch.equal(ep_a.context_y, ep_b.context_y)
    assert ep_a.context_x.shape == torch.Size([2, 4])
    assert ep_a.query_x.shape == torch.Size([2, 4])
    assert float(ep_a.y_best) > 0.0


def test_dockstring_accepts_effective_subset_smaller_than_requested(tmp_path: Path) -> None:
    cache_path = tmp_path / "dockstring_cache.pt"
    _write_cache(cache_path)

    loader = DockstringBODataLoader(
        cache_path=cache_path,
        raw_data_path=None,
        split="train",
        train_targets=["JAK2", "KIT"],
        test_targets=["PTK2"],
        subset_size=10,
        subset_seed=13,
        fingerprint_type="morgan",
        fingerprint_radius=2,
        fingerprint_n_bits=4,
        normalize_targets=False,
        negate_scores=True,
        seed=5,
    )

    assert loader.dataset_ids == ["JAK2", "KIT"]


def test_dockstring_requires_explicit_target_partition(tmp_path: Path) -> None:
    cache_path = tmp_path / "dockstring_cache.pt"
    _write_cache(cache_path)

    with pytest.raises(ValueError, match="both empty"):
        DockstringBODataLoader(
            cache_path=cache_path,
            raw_data_path=None,
            split="train",
            train_targets=None,
            test_targets=None,
            subset_size=6,
            subset_seed=13,
            fingerprint_type="morgan",
            fingerprint_radius=2,
            fingerprint_n_bits=4,
            normalize_targets=False,
            negate_scores=True,
            seed=5,
        )
