"""Helpers for loading policy checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import torch
from omegaconf import DictConfig, OmegaConf

from polar.backbones import create_backbone
from polar.trainers import build_policy_head


@dataclass
class LoadedPolicy:
    """Loaded policy components."""

    backbone: Any
    policy_head: torch.nn.Module
    policy_checkpoint_path: str
    backbone_checkpoint_path: str | None
    backbone_state_source: str


def _normalize_checkpoint_path(path_like: str | Path) -> Path:
    """Prefer the user-provided path over a resolved symlink target.

    This is important on HPC/container setups where `/scratch/...` may be
    mounted inside the container while the canonical host path resolved from
    that symlink is not.
    """
    path = Path(path_like).expanduser()
    if path.exists():
        return path

    resolved = path.resolve()
    if resolved.exists():
        return resolved

    raise FileNotFoundError(
        "Checkpoint does not exist. Checked both the provided path and its resolved path: "
        f"{path} | {resolved}"
    )


def load_checkpoint_payload(path_like: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load a policy checkpoint payload from disk."""
    path = _normalize_checkpoint_path(path_like)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict checkpoint payload from {path}.")
    return payload


def extract_payload_model_backend(payload: dict[str, Any]) -> str | None:
    """Return the backbone name recorded in a checkpoint payload config, if any."""
    config_obj = payload.get("config")
    if not isinstance(config_obj, dict):
        return None
    model_obj = config_obj.get("model")
    if not isinstance(model_obj, dict):
        return None
    model_name = model_obj.get("name")
    if model_name is None:
        return None
    return str(model_name).strip().lower()


def extract_policy_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract the policy head state dict from a checkpoint payload."""
    if "policy_head_state_dict" in payload:
        state = payload["policy_head_state_dict"]
        if not isinstance(state, dict):
            raise TypeError("policy_head_state_dict must be a state-dict mapping.")
        return state
    if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload  # type: ignore[return-value]
    raise KeyError("Could not find policy state dict in checkpoint.")


def extract_backbone_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    """Extract a fine-tuned backbone state dict from a checkpoint payload when present."""
    maybe_state = payload.get("backbone_model_state_dict")
    if maybe_state is None:
        return None
    if not isinstance(maybe_state, dict):
        raise TypeError("backbone_model_state_dict must be a state-dict mapping.")
    return maybe_state


def resolve_policy_arch(
    *,
    payload: dict[str, Any],
    policy_cfg: DictConfig,
) -> dict[str, Any]:
    """Resolve policy-head architecture by merging user overrides onto payload defaults."""
    payload_config = payload.get("config", {})
    payload_train = payload_config.get("train", {}) if isinstance(payload_config, dict) else {}
    payload_head = payload_train.get("policy_head", {}) if isinstance(payload_train, dict) else {}

    def _pick(field: str, default: Any) -> Any:
        user_value = policy_cfg.get(field)
        if user_value is not None:
            return user_value
        if isinstance(payload_head, dict) and payload_head.get(field) is not None:
            return payload_head[field]
        return default

    head_type = str(_pick("type", "mlp"))
    use_remaining_budget = bool(_pick("use_remaining_budget", False))
    if head_type.strip().lower() == "target_aware_transformer" and policy_cfg.get(
        "use_remaining_budget"
    ) is None:
        # Early AL checkpoints could record an enabled MLP-only flag that the
        # target-aware head ignored. Its saved FFN width identifies the actual
        # architecture; an explicit user override still permits a warm start.
        policy_state = payload.get("policy_head_state_dict", payload)
        if isinstance(policy_state, dict):
            ffn_weight = policy_state.get("layers.0.ffn.0.weight")
            scorer_weight = policy_state.get("scorer.weight")
            if (
                isinstance(ffn_weight, torch.Tensor) and ffn_weight.ndim == 2
                and isinstance(scorer_weight, torch.Tensor) and scorer_weight.ndim == 2
                and ffn_weight.shape[1] in {scorer_weight.shape[1], scorer_weight.shape[1] + 1}
            ):
                use_remaining_budget = ffn_weight.shape[1] == scorer_weight.shape[1] + 1

    return {
        "type": head_type,
        "hidden_dim": int(_pick("hidden_dim", 128)),
        "num_layers": int(_pick("num_layers", 2)),
        "num_heads": int(_pick("num_heads", 4)),
        "ffn_dim": int(_pick("ffn_dim", 128)),
        "activation": str(_pick("activation", "gelu")),
        "dropout": float(_pick("dropout", 0.0)),
        "use_remaining_budget": use_remaining_budget,
        "budget_scale": float(
            _pick(
                "budget_scale",
                payload_train.get("num_acquisitions", 1.0)
                if isinstance(payload_train, dict) else 1.0,
            )
        ),
    }


def resolve_backbone_checkpoint_path(
    *,
    model_cfg: DictConfig,
    payload: dict[str, Any],
) -> str | None:
    """Choose the backbone checkpoint path for reconstructing a policy backbone."""
    user_path = model_cfg.get("checkpoint_path")
    if user_path is not None:
        return str(user_path)

    payload_path = payload.get("backbone_checkpoint_path")
    if isinstance(payload_path, str) and payload_path:
        resolved = Path(payload_path).expanduser()
        if resolved.exists():
            return str(resolved)
        print(
            "warning: backbone_checkpoint_path from policy checkpoint does not exist on this machine; "
            "falling back to local default checkpoint path resolution."
        )
    return None


def load_policy_checkpoint(
    *,
    backend: str,
    model_cfg: DictConfig,
    policy_cfg: DictConfig,
    device: str,
    checkpoints_root: str | Path,
    policy_checkpoint_path: str | Path,
    strict_backbone_state: bool = True,
) -> LoadedPolicy:
    """Load a policy checkpoint and reconstruct the paired backbone/policy head modules."""
    backend = str(backend).strip().lower()
    if backend == "gp":
        raise ValueError("backend='gp' does not support a learned AL policy.")

    checkpoint_path = _normalize_checkpoint_path(policy_checkpoint_path)
    payload = load_checkpoint_payload(checkpoint_path)

    payload_backend = extract_payload_model_backend(payload)
    if payload_backend is not None and payload_backend != backend:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained with model.name='{payload_backend}', "
            f"but backend='{backend}' was requested."
        )

    backbone_checkpoint_path = resolve_backbone_checkpoint_path(model_cfg=model_cfg, payload=payload)
    backbone = create_backbone(
        name=str(model_cfg.get("name", backend)),
        checkpoint_path=backbone_checkpoint_path,
        version=model_cfg.get("version"),
        device=device,
        checkpoints_root=checkpoints_root,
        cache_trainset_representation=bool(model_cfg.get("cache_trainset_representation", False)),
        strict=bool(model_cfg.get("strict_load", True)),
    )

    policy_arch = resolve_policy_arch(payload=payload, policy_cfg=policy_cfg)
    policy_head = build_policy_head(
        input_dim=int(getattr(backbone, "embedding_dim")),
        policy_head_cfg=OmegaConf.create(policy_arch),
    ).to(backbone.device)

    policy_state = extract_policy_state_dict(payload)
    policy_head.load_state_dict(policy_state, strict=True)
    policy_head.eval()

    backbone_state_source = "pretrained_backbone"
    maybe_backbone_state = extract_backbone_state_dict(payload)
    if maybe_backbone_state is not None:
        backbone.model.load_state_dict(
            maybe_backbone_state,
            strict=bool(strict_backbone_state),
        )
        backbone_state_source = "checkpoint_payload"
    backbone.model.eval()

    return LoadedPolicy(
        backbone=backbone,
        policy_head=policy_head,
        policy_checkpoint_path=str(checkpoint_path),
        backbone_checkpoint_path=str(backbone.checkpoint_path) if backbone.checkpoint_path else None,
        backbone_state_source=backbone_state_source,
    )
