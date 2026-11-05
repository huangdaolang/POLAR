"""Shared runtime helpers for policy-training scripts.

This module consolidates boilerplate that used to be duplicated across
``scripts/train_*_policy.py``: run-tag generation, run/checkpoint
directory layout, single-file checkpoint saving, config-value coercion,
W&B initialization, LR extraction, and a null-safe W&B logger.

These helpers are intentionally *not* re-exported from
``polar.utils.__init__`` — they are script-level glue, and
following the same convention as ``polar.utils.seeding``, scripts
import them directly:

    from polar.utils.run_utils import (
        build_run_tag,
        build_run_dirs,
        save_checkpoint,
        init_wandb,
        log_wandb_metrics,
        extract_param_group_lrs,
    )

Epoch convention
----------------
``save_checkpoint`` stores ``epoch`` as a **0-indexed** count, i.e.
``epoch=0`` means "epoch index 0 has just completed". This matches the
convention used by PyTorch Lightning's ``current_epoch`` and the
Hugging Face Trainer. After a full run of ``num_epochs`` epochs, the
final checkpoint records ``epoch=num_epochs - 1``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from polar.backbones.tabpfn.checkpoint_paths import get_path_layout


def build_run_tag(cfg: DictConfig) -> str:
    """Return ``cfg.output.run_tag`` if set, else a local-time timestamp."""
    if cfg.output.run_tag:
        return str(cfg.output.run_tag)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_dirs(cfg: DictConfig, run_tag: str) -> tuple[Path, Path]:
    """Create and return ``(run_dir, checkpoint_dir)`` for ``run_tag``.

    Uses ``cfg.paths.checkpoints_root`` / ``cfg.paths.results_root`` +
    ``cfg.output.experiment_name``.  Both directories are created if
    missing.
    """
    layout = get_path_layout(
        checkpoints_root=cfg.paths.checkpoints_root,
        results_root=cfg.paths.results_root,
    )
    experiment_name = str(cfg.output.experiment_name)

    run_dir = layout.results_root / experiment_name / run_tag
    checkpoint_dir = layout.policy_root / experiment_name / run_tag

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, checkpoint_dir


def save_checkpoint(
    *,
    path: Path,
    epoch: int,
    cfg: DictConfig,
    policy_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    backbone_model_state_dict: dict[str, torch.Tensor] | None = None,
    backbone_checkpoint_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a single-file policy training checkpoint.

    ``epoch`` is 0-indexed: ``epoch=0`` means "epoch index 0 just
    completed".  ``metadata`` is an optional free-form dict for
    trainer-specific state (e.g. BED heavy-eval diagnostics).
    """
    payload: dict[str, Any] = {
        "epoch": epoch,
        "policy_head_state_dict": policy_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "backbone_checkpoint_path": backbone_checkpoint_path,
    }
    if backbone_model_state_dict is not None:
        payload["backbone_model_state_dict"] = backbone_model_state_dict
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, path)


def cfg_str_or_none(value: Any) -> str | None:
    """Coerce a config value to a non-empty ``str`` or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def init_wandb(cfg: DictConfig, run_dir: Path, run_tag: str) -> Any | None:
    """Initialize a W&B run if ``cfg.wandb.enabled`` is true, else ``None``.

    Raises ``ImportError`` with actionable guidance if W&B is enabled
    but the ``wandb`` package is not installed.
    """
    wandb_cfg = cfg.get("wandb")
    if wandb_cfg is None or not bool(wandb_cfg.enabled):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging is enabled (wandb.enabled=true), but 'wandb' is not installed. "
            "Run `uv sync` (or `pip install wandb`)."
        ) from exc

    tags_raw = OmegaConf.to_container(wandb_cfg.get("tags", []), resolve=True)
    tags = [str(item) for item in tags_raw] if isinstance(tags_raw, list) else None
    run_name = cfg_str_or_none(wandb_cfg.get("run_name"))
    if run_name is None:
        run_name = f"{cfg.output.experiment_name}-{run_tag}"

    wandb_run = wandb.init(
        project=str(wandb_cfg.get("project", "polar")),
        entity=cfg_str_or_none(wandb_cfg.get("entity")),
        name=run_name,
        group=cfg_str_or_none(wandb_cfg.get("group")),
        tags=tags,
        mode=str(wandb_cfg.get("mode", "online")),
        dir=cfg_str_or_none(wandb_cfg.get("dir")) or str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"wandb_run: {wandb_run.name} ({wandb_run.id})")
    return wandb_run


def log_wandb_metrics(
    wandb_run: Any | None,
    *,
    step: int,
    payload: dict[str, float],
) -> None:
    """Log ``payload`` to W&B with a null-safe guard on ``wandb_run``.

    The caller is responsible for building the full payload (including
    any ``"epoch"`` key and LR entries).  This keeps the metric schema
    fully under caller control, since different trainer families log
    different keys.
    """
    if wandb_run is None:
        return
    wandb_run.log(payload, step=step)


def extract_param_group_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Return ``{"lr/group_0": ..., "lr/group_1": ...}`` for each param group.

    Mirrors the convention used by the ``train_*_policy.py`` scripts
    (except ``train_bed_policy.py``, which uses its own ``train/lr_*``
    keys).
    """
    out: dict[str, float] = {}
    for idx, group in enumerate(optimizer.param_groups):
        lr = group.get("lr")
        if lr is not None:
            out[f"lr/group_{idx}"] = float(lr)
    return out
