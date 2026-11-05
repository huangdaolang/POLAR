"""Local loader for TabICL regression and classification checkpoints."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from polar.backbones.tabicl.checkpoint_paths import (
    TabICLCheckpointKind,
    TabICLVersion,
    resolve_tabicl_checkpoint,
    resolve_tabicl_regressor_checkpoint,
)
from polar.backbones.tabicl.model.tabicl import TabICL


@dataclass
class LoadedTabICL:
    """Container for a fully loaded TabICL backbone."""

    model: TabICL
    model_config: dict[str, Any]
    checkpoint_path: Path
    checkpoint_version: str
    missing_keys: list[str]
    unexpected_keys: list[str]
    checkpoint_kind: str = "regression"


def _infer_checkpoint_version(path: Path, config: dict[str, Any]) -> str:
    if "v2" in path.name:
        return "v2"
    if config.get("max_classes", None) == 0:
        return "regression_custom"
    return "custom"


def _infer_checkpoint_kind_from_model_config(
    model_config: dict[str, Any],
) -> Literal["regression", "classification"]:
    max_classes = int(model_config.get("max_classes", 0) or 0)
    return "classification" if max_classes > 0 else "regression"


def load_tabicl_model(
    *,
    checkpoint_path: str | Path | None = None,
    version: TabICLVersion = "v2",
    checkpoint_kind: TabICLCheckpointKind = "regression",
    device: str | torch.device = "cpu",
    checkpoints_root: str | Path | None = None,
    strict: bool = True,
) -> LoadedTabICL:
    """Load a local TabICL checkpoint with optional kind auto-detection."""
    resolved_path = (
        resolve_tabicl_checkpoint(
            checkpoint_path=checkpoint_path,
            version=version,
            checkpoint_kind=checkpoint_kind,
            checkpoints_root=checkpoints_root,
            must_exist=True,
        )
        if checkpoint_kind != "regression" or checkpoint_path is not None
        else resolve_tabicl_regressor_checkpoint(
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

    model_config = dict(checkpoint["config"])
    inferred_kind = _infer_checkpoint_kind_from_model_config(model_config)
    if checkpoint_kind != "auto" and inferred_kind != checkpoint_kind:
        raise ValueError(
            "Loaded checkpoint kind does not match requested checkpoint_kind: "
            f"requested={checkpoint_kind}, inferred={inferred_kind}."
        )

    model = TabICL(**model_config)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "Strict loading failed.\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )

    model = model.to(device)
    model.eval()

    return LoadedTabICL(
        model=model,
        model_config=model_config,
        checkpoint_path=resolved_path,
        checkpoint_version=_infer_checkpoint_version(resolved_path, model_config),
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        checkpoint_kind=inferred_kind,
    )


def load_tabicl_regressor(
    *,
    checkpoint_path: str | Path | None = None,
    version: TabICLVersion = "v2",
    device: str | torch.device = "cpu",
    checkpoints_root: str | Path | None = None,
    strict: bool = True,
) -> LoadedTabICL:
    """Load TabICL regressor from a local checkpoint path.

    This loader is intentionally minimal:
    - regression only
    - no automatic downloading
    - no sklearn wrappers
    """
    return load_tabicl_model(
        checkpoint_path=checkpoint_path,
        version=version,
        checkpoint_kind="regression",
        device=device,
        checkpoints_root=checkpoints_root,
        strict=strict,
    )

