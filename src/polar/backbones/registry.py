"""Backbone registry for model-agnostic construction."""

from __future__ import annotations

from typing import Any, Callable

from polar.backbones.base import TabularBackbone

_BACKBONE_REGISTRY: dict[str, type[TabularBackbone]] = {}


def register_backbone(name: str) -> Callable[[type[TabularBackbone]], type[TabularBackbone]]:
    """Register a backbone class under a stable name."""

    def decorator(cls: type[TabularBackbone]) -> type[TabularBackbone]:
        if name in _BACKBONE_REGISTRY:
            raise ValueError(f"Backbone '{name}' already registered.")
        _BACKBONE_REGISTRY[name] = cls
        return cls

    return decorator


def get_backbone_class(name: str) -> type[TabularBackbone]:
    """Fetch a backbone class from registry."""
    try:
        return _BACKBONE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKBONE_REGISTRY))
        raise KeyError(f"Unknown backbone '{name}'. Available: {available}") from exc


def create_backbone(name: str, **kwargs: Any) -> TabularBackbone:
    """Instantiate a registered backbone from pretrained weights."""
    cls = get_backbone_class(name)
    return cls.load_pretrained(**kwargs)


def list_backbones() -> list[str]:
    """List all registered backbones."""
    return sorted(_BACKBONE_REGISTRY.keys())
