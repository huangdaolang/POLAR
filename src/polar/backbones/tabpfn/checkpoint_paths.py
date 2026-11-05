"""Local path utilities for pretrained checkpoints and experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TabPFNVersion = Literal["v2", "v2.5"]
TabPFNCheckpointKind = Literal["auto", "regression", "classification"]

DEFAULT_REGRESSION_CHECKPOINT = {
    "v2": "tabpfn-v2-regressor-v2_default.ckpt",
    "v2.5": "tabpfn-v2.5-regressor-v2.5_default.ckpt",
}
DEFAULT_CLASSIFICATION_CHECKPOINT = {
    "v2": "tabpfn-v2-classifier-v2_default.ckpt",
    "v2.5": "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
}


def get_repo_root() -> Path:
    """Return repository root assuming src-layout installation."""
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PathLayout:
    """Canonical local storage layout for checkpoints and results."""

    checkpoints_root: Path
    pretrained_tabpfn_root: Path
    finetuned_root: Path
    policy_root: Path
    results_root: Path
    runs_root: Path
    eval_root: Path


def get_path_layout(
    checkpoints_root: str | Path | None = None,
    results_root: str | Path | None = None,
) -> PathLayout:
    """Build canonical path layout from optional roots."""
    repo_root = get_repo_root()
    checkpoints = Path(checkpoints_root) if checkpoints_root is not None else repo_root / "checkpoints"
    results = Path(results_root) if results_root is not None else repo_root / "results"

    return PathLayout(
        checkpoints_root=checkpoints,
        pretrained_tabpfn_root=checkpoints / "pretrained" / "tabpfn",
        finetuned_root=checkpoints / "finetuned",
        policy_root=checkpoints / "policy",
        results_root=results,
        runs_root=results / "runs",
        eval_root=results / "eval",
    )


def default_tabpfn_regressor_checkpoint_path(
    version: TabPFNVersion,
    checkpoints_root: str | Path | None = None,
) -> Path:
    """Return canonical local path for the default TabPFN regressor checkpoint."""
    layout = get_path_layout(checkpoints_root=checkpoints_root)
    return layout.pretrained_tabpfn_root / version / DEFAULT_REGRESSION_CHECKPOINT[version]


def default_tabpfn_classifier_checkpoint_path(
    version: TabPFNVersion,
    checkpoints_root: str | Path | None = None,
) -> Path:
    """Return canonical local path for the default TabPFN classifier checkpoint."""
    layout = get_path_layout(checkpoints_root=checkpoints_root)
    return layout.pretrained_tabpfn_root / version / DEFAULT_CLASSIFICATION_CHECKPOINT[version]


def default_tabpfn_checkpoint_path(
    *,
    version: TabPFNVersion,
    checkpoint_kind: TabPFNCheckpointKind,
    checkpoints_root: str | Path | None = None,
) -> Path:
    """Return canonical local path for a default TabPFN checkpoint."""
    effective_kind = "regression" if checkpoint_kind == "auto" else checkpoint_kind
    if effective_kind == "regression":
        return default_tabpfn_regressor_checkpoint_path(
            version=version,
            checkpoints_root=checkpoints_root,
        )
    return default_tabpfn_classifier_checkpoint_path(
        version=version,
        checkpoints_root=checkpoints_root,
    )


def resolve_tabpfn_regressor_checkpoint(
    *,
    checkpoint_path: str | Path | None,
    version: TabPFNVersion,
    checkpoints_root: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve local checkpoint path with optional existence validation."""
    if checkpoint_path is None:
        resolved = default_tabpfn_regressor_checkpoint_path(
            version=version,
            checkpoints_root=checkpoints_root,
        )
    else:
        resolved = Path(checkpoint_path).expanduser().resolve()

    if must_exist and not resolved.exists():
        if checkpoint_path is None:
            raise FileNotFoundError(
                "Default local TabPFN checkpoint not found.\n"
                f"Expected: {resolved}\n"
                "Either place the checkpoint there, or pass model.checkpoint_path=<absolute path>."
            )
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved}")

    return resolved


def resolve_tabpfn_checkpoint(
    *,
    checkpoint_path: str | Path | None,
    version: TabPFNVersion,
    checkpoint_kind: TabPFNCheckpointKind,
    checkpoints_root: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a local TabPFN checkpoint path with kind-aware defaults."""
    if checkpoint_path is None:
        resolved = default_tabpfn_checkpoint_path(
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
                "Default local TabPFN checkpoint not found.\n"
                f"Expected: {resolved}\n"
                "Requested checkpoint kind: "
                f"{effective_kind}\n"
                "Either place the checkpoint there, or pass model.checkpoint_path=<absolute path>."
            )
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved}")

    return resolved
