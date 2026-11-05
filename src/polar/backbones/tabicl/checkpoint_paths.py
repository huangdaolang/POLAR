"""Local path utilities for pretrained TabICL checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from polar.backbones.tabpfn.checkpoint_paths import get_path_layout

TabICLVersion = Literal["v2"]
TabICLCheckpointKind = Literal["auto", "regression", "classification"]

DEFAULT_REGRESSION_CHECKPOINT = {
    "v2": "tabicl-regressor-v2-20260212.ckpt",
}
DEFAULT_CLASSIFICATION_CHECKPOINT = {
    "v2": "tabicl-classifier-v2-20260212.ckpt",
}


def default_tabicl_regressor_checkpoint_path(
    version: TabICLVersion,
    checkpoints_root: str | Path | None = None,
) -> Path:
    """Return canonical local path for the default TabICL regressor checkpoint."""
    layout = get_path_layout(checkpoints_root=checkpoints_root)
    return (
        layout.checkpoints_root
        / "pretrained"
        / "tabicl"
        / version
        / DEFAULT_REGRESSION_CHECKPOINT[version]
    )


def default_tabicl_classifier_checkpoint_path(
    version: TabICLVersion,
    checkpoints_root: str | Path | None = None,
) -> Path:
    """Return canonical local path for the default TabICL classifier checkpoint."""
    layout = get_path_layout(checkpoints_root=checkpoints_root)
    return (
        layout.checkpoints_root
        / "pretrained"
        / "tabicl"
        / version
        / DEFAULT_CLASSIFICATION_CHECKPOINT[version]
    )


def default_tabicl_checkpoint_path(
    *,
    version: TabICLVersion,
    checkpoint_kind: TabICLCheckpointKind,
    checkpoints_root: str | Path | None = None,
) -> Path:
    """Return canonical local path for a default TabICL checkpoint."""
    effective_kind = "regression" if checkpoint_kind == "auto" else checkpoint_kind
    if effective_kind == "regression":
        return default_tabicl_regressor_checkpoint_path(
            version=version,
            checkpoints_root=checkpoints_root,
        )
    return default_tabicl_classifier_checkpoint_path(
        version=version,
        checkpoints_root=checkpoints_root,
    )


def resolve_tabicl_regressor_checkpoint(
    *,
    checkpoint_path: str | Path | None,
    version: TabICLVersion,
    checkpoints_root: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve local TabICL checkpoint path with optional existence validation."""
    if checkpoint_path is None:
        resolved = default_tabicl_regressor_checkpoint_path(
            version=version,
            checkpoints_root=checkpoints_root,
        )
    else:
        resolved = Path(checkpoint_path).expanduser().resolve()

    if must_exist and not resolved.exists():
        if checkpoint_path is None:
            raise FileNotFoundError(
                "Default local TabICL checkpoint not found.\n"
                f"Expected: {resolved}\n"
                "Either place the checkpoint there, or pass model.checkpoint_path=<absolute path>."
            )
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved}")

    return resolved


def resolve_tabicl_checkpoint(
    *,
    checkpoint_path: str | Path | None,
    version: TabICLVersion,
    checkpoint_kind: TabICLCheckpointKind,
    checkpoints_root: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve local TabICL checkpoint path with kind-aware defaults."""
    if checkpoint_path is None:
        resolved = default_tabicl_checkpoint_path(
            version=version,
            checkpoint_kind=checkpoint_kind,
            checkpoints_root=checkpoints_root,
        )
    else:
        resolved = Path(checkpoint_path).expanduser().resolve()

    if must_exist and not resolved.exists():
        if checkpoint_path is None:
            effective_kind = "regression" if checkpoint_kind == "auto" else checkpoint_kind
            raise FileNotFoundError(
                "Default local TabICL checkpoint not found.\n"
                f"Expected: {resolved}\n"
                "Requested checkpoint kind: "
                f"{effective_kind}\n"
                "Either place the checkpoint there, or pass model.checkpoint_path=<absolute path>."
            )
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved}")

    return resolved
