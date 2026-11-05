"""Local loader for TabPFN regression and classification checkpoints."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from polar.backbones.tabpfn.checkpoint_paths import (
    TabPFNCheckpointKind,
    TabPFNVersion,
    resolve_tabpfn_checkpoint,
    resolve_tabpfn_regressor_checkpoint,
)
from polar.backbones.tabpfn.model import ARCHITECTURES
from polar.backbones.tabpfn.model.base.bar_distribution import (
    FullSupportBarDistribution,
)


@dataclass
class LoadedTabPFN:
    """Container for a fully loaded TabPFN backbone."""

    model: nn.Module
    bar_distribution: nn.Module | None
    model_config: Any
    checkpoint_path: Path
    architecture_name: str
    checkpoint_version: str
    missing_keys: list[str]
    unexpected_keys: list[str]
    checkpoint_kind: Literal["regression", "classification"] = "regression"


def _infer_checkpoint_version(checkpoint: dict[str, Any]) -> str:
    if "inference_config" in checkpoint:
        return "v2.5_or_newer"
    if "architecture_name" in checkpoint:
        return "v2.5"
    return "v2"


def _build_regression_bar_distribution(num_buckets: int, dist_cls: type[nn.Module]) -> nn.Module:
    borders = torch.arange(num_buckets + 1, dtype=torch.float32) * 30_000.0
    return dist_cls(borders, ignore_nan_targets=True)  # type: ignore[call-arg]


def _infer_checkpoint_kind_from_model_config(model_config: Any) -> Literal["regression", "classification"]:
    max_num_classes = int(getattr(model_config, "max_num_classes", 0) or 0)
    return "classification" if max_num_classes > 0 else "regression"


def load_tabpfn_model(
    *,
    checkpoint_path: str | Path | None = None,
    version: TabPFNVersion = "v2.5",
    checkpoint_kind: TabPFNCheckpointKind = "regression",
    device: str | torch.device = "cpu",
    checkpoints_root: str | Path | None = None,
    cache_trainset_representation: bool = False,
    strict: bool = True,
) -> LoadedTabPFN:
    """Load a local TabPFN checkpoint with optional kind auto-detection."""
    resolved_path = (
        resolve_tabpfn_checkpoint(
            checkpoint_path=checkpoint_path,
            version=version,
            checkpoint_kind=checkpoint_kind,
            checkpoints_root=checkpoints_root,
            must_exist=True,
        )
        if checkpoint_kind != "regression" or checkpoint_path is not None
        else resolve_tabpfn_regressor_checkpoint(
            checkpoint_path=checkpoint_path,
            version=version,
            checkpoints_root=checkpoints_root,
            must_exist=True,
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint: dict[str, Any] = torch.load(
            resolved_path,
            map_location="cpu",
            weights_only=True,
        )

    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing required key: state_dict")
    if "config" not in checkpoint:
        raise KeyError("Checkpoint missing required key: config")

    architecture_name = checkpoint.get("architecture_name", "base")
    if architecture_name not in ARCHITECTURES:
        raise KeyError(f"Unknown architecture_name='{architecture_name}' in checkpoint.")
    architecture = ARCHITECTURES[architecture_name]

    model_config, _unused = architecture.parse_config(checkpoint["config"])
    inferred_kind = _infer_checkpoint_kind_from_model_config(model_config)
    if checkpoint_kind != "auto" and inferred_kind != checkpoint_kind:
        raise ValueError(
            "Loaded checkpoint kind does not match requested checkpoint_kind: "
            f"requested={checkpoint_kind}, inferred={inferred_kind}."
        )

    state_dict = dict(checkpoint["state_dict"])
    if inferred_kind == "regression":
        criterion_keys = [k for k in state_dict if k.startswith("criterion.")]
        if not criterion_keys:
            raise ValueError(
                "Regression checkpoint missing criterion.* keys for the bar distribution."
            )

        bar_distribution = _build_regression_bar_distribution(
            num_buckets=int(model_config.num_buckets),
            dist_cls=FullSupportBarDistribution,
        )
        criterion_state = {
            k.replace("criterion.", ""): state_dict.pop(k) for k in criterion_keys
        }
        bar_distribution.load_state_dict(criterion_state, strict=True)
        n_out = int(bar_distribution.num_bars)
    else:
        bar_distribution = None
        n_out = int(model_config.max_num_classes)

    model = architecture.get_architecture(
        model_config,
        n_out=n_out,
        cache_trainset_representation=cache_trainset_representation,
    )

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "Strict loading failed.\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )

    model = model.to(device)
    if bar_distribution is not None:
        bar_distribution = bar_distribution.to(device)
    model.eval()

    return LoadedTabPFN(
        model=model,
        bar_distribution=bar_distribution,
        model_config=model_config,
        checkpoint_path=resolved_path,
        architecture_name=architecture_name,
        checkpoint_version=_infer_checkpoint_version(checkpoint),
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        checkpoint_kind=inferred_kind,
    )


def load_tabpfn_regressor(
    *,
    checkpoint_path: str | Path | None = None,
    version: TabPFNVersion = "v2.5",
    device: str | torch.device = "cpu",
    checkpoints_root: str | Path | None = None,
    cache_trainset_representation: bool = False,
    strict: bool = True,
) -> LoadedTabPFN:
    """Load TabPFN regressor from a local checkpoint path.

    This loader is intentionally minimal:
    - regression only
    - no automatic downloading
    - no sklearn wrappers
    """
    return load_tabpfn_model(
        checkpoint_path=checkpoint_path,
        version=version,
        checkpoint_kind="regression",
        device=device,
        checkpoints_root=checkpoints_root,
        cache_trainset_representation=cache_trainset_representation,
        strict=strict,
    )

