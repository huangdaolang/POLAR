"""Abstract interfaces for tabular foundation-model backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Lightweight categorical distribution wrapper for classification backbones
# ---------------------------------------------------------------------------
class CategoricalClassificationDistribution:
    """Categorical predictive distribution wrapping raw class logits.

    Provides the same ``mean`` / ``variance`` / ``log_prob`` / ``sample`` API
    that regression distributions (``BarDistribution``, ``QuantileDistribution``)
    expose, so that downstream code can treat classification and regression
    outputs uniformly.
    """

    def __init__(self, logits: torch.Tensor) -> None:
        if logits.ndim not in (2, 3):
            raise ValueError(
                "logits must be 2D [N, C] or 3D [B, N, C], "
                f"got shape {tuple(logits.shape)}"
            )
        self.logits = logits
        self.probs = torch.softmax(logits, dim=-1)

    def mean(self) -> torch.Tensor:
        classes = torch.arange(
            self.probs.shape[-1],
            device=self.probs.device,
            dtype=self.probs.dtype,
        )
        return (self.probs * classes).sum(dim=-1)

    def variance(self) -> torch.Tensor:
        classes = torch.arange(
            self.probs.shape[-1],
            device=self.probs.device,
            dtype=self.probs.dtype,
        )
        mean = self.mean()
        second_moment = (self.probs * classes.square()).sum(dim=-1)
        return (second_moment - mean.square()).clamp_min(0.0)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(self.logits, dim=-1)
        gathered = log_probs.gather(dim=-1, index=value.long().unsqueeze(-1))
        return gathered.squeeze(-1)

    def sample(
        self,
        sample_shape: torch.Size | tuple[int, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = torch.distributions.Categorical(probs=self.probs)
        if sample_shape is None or torch.is_tensor(sample_shape):
            return dist.sample()
        return dist.sample(torch.Size(sample_shape))


@dataclass
class ContextState:
    """Belief state based on currently observed context points."""

    x_context: torch.Tensor
    y_context: torch.Tensor


class TabularBackbone(nn.Module, ABC):
    """Model-agnostic interface for tabular belief backbones."""

    @property
    def device(self) -> torch.device:
        """Best-effort model device resolver."""
        model = getattr(self, "model", None)
        if not isinstance(model, nn.Module):
            raise AttributeError(
                f"{self.__class__.__name__} must define `self.model` as an nn.Module."
            )

        first_param = next(model.parameters(), None)
        if first_param is not None:
            return first_param.device

        first_buffer = next(model.buffers(), None)
        if first_buffer is not None:
            return first_buffer.device

        return torch.device("cpu")

    @classmethod
    @abstractmethod
    def load_pretrained(cls, **kwargs: Any) -> "TabularBackbone":
        """Construct backbone from a pretrained checkpoint."""

    def encode_context(self, x_context: torch.Tensor, y_context: torch.Tensor) -> ContextState:
        """Encode observed context into a state object."""
        x_context = x_context.to(self.device)
        y_context = y_context.to(self.device)

        if x_context.ndim == 2:
            if y_context.ndim not in (1, 2):
                raise ValueError(f"y_context must be 1D or 2D, got shape {tuple(y_context.shape)}")
            if y_context.ndim == 2 and y_context.shape[1] != 1:
                raise ValueError("2D y_context must have shape [N, 1].")
            if y_context.shape[0] != x_context.shape[0]:
                raise ValueError("x_context and y_context must have matching first dimension.")
            return ContextState(
                x_context=x_context.float(),
                y_context=y_context.view(-1).float(),
            )

        if x_context.ndim == 3:
            if y_context.ndim not in (2, 3):
                raise ValueError(f"y_context must be 2D or 3D, got shape {tuple(y_context.shape)}")
            if y_context.ndim == 3 and y_context.shape[2] != 1:
                raise ValueError("3D y_context must have shape [B, N, 1].")
            if y_context.shape[0] != x_context.shape[0] or y_context.shape[1] != x_context.shape[1]:
                raise ValueError(
                    "For batched context, x_context and y_context must match on [B, N]."
                )
            return ContextState(
                x_context=x_context.float(),
                y_context=y_context.view(x_context.shape[0], x_context.shape[1]).float(),
            )

        raise ValueError(
            "x_context must be 2D [N, F] or 3D [B, N, F], "
            f"got shape {tuple(x_context.shape)}"
        )

    def predictive_distribution(
        self,
        x_query: torch.Tensor,
        context_state: ContextState,
    ) -> dict[str, torch.Tensor]:
        """Return query-time predictive outputs (e.g., logits, mean)."""
        logits = self._forward_logits(x_query, context_state)
        mean = self.mean_from_logits(logits)
        return {
            "logits": logits,
            "mean": mean,
        }

    def _validate_query_shape(
        self,
        x_query: torch.Tensor,
        context_state: ContextState,
    ) -> None:
        if context_state.x_context.ndim == 2:
            if x_query.ndim != 2:
                raise ValueError(f"x_query must be 2D [Q, F], got shape {tuple(x_query.shape)}")
            if x_query.shape[1] != context_state.x_context.shape[1]:
                raise ValueError(
                    "x_query feature dimension must match context feature dimension: "
                    f"{x_query.shape[1]} vs {context_state.x_context.shape[1]}"
                )
            return

        if context_state.x_context.ndim == 3:
            if x_query.ndim != 3:
                raise ValueError(
                    "For batched context, x_query must be 3D [B, Q, F], "
                    f"got shape {tuple(x_query.shape)}"
                )
            if x_query.shape[0] != context_state.x_context.shape[0]:
                raise ValueError("Batch size mismatch between x_query and context.")
            if x_query.shape[2] != context_state.x_context.shape[2]:
                raise ValueError(
                    "x_query feature dimension must match context feature dimension: "
                    f"{x_query.shape[2]} vs {context_state.x_context.shape[2]}"
                )
            return

        raise ValueError(
            "context_state.x_context must be 2D or 3D, "
            f"got shape {tuple(context_state.x_context.shape)}"
        )

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing to reduce VRAM. Override in subclasses."""

    def disable_gradient_checkpointing(self) -> None:
        """Disable activation checkpointing. Override in subclasses if needed."""

    @abstractmethod
    def _forward_outputs(
        self,
        x_query: torch.Tensor,
        context_state: ContextState,
        *,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return model-specific outputs and must include a ``logits`` key."""

    def _forward_logits(
        self,
        x_query: torch.Tensor,
        context_state: ContextState,
    ) -> torch.Tensor:
        return self._forward_outputs(
            x_query,
            context_state,
            return_embeddings=False,
        )["logits"]

    def forward_with_embeddings(
        self,
        x_query: torch.Tensor,
        context_state: ContextState,
    ) -> dict[str, torch.Tensor]:
        """Return logits/mean and optional model-specific embeddings."""
        outputs = self._forward_outputs(
            x_query,
            context_state,
            return_embeddings=True,
        )
        outputs["mean"] = self.mean_from_logits(outputs["logits"])
        return outputs

    def log_prob(
        self,
        x_query: torch.Tensor,
        y_query: torch.Tensor,
        context_state: ContextState,
    ) -> torch.Tensor:
        """Log probability under backbone-specific predictive distribution."""
        logits = self._forward_logits(x_query, context_state)
        return self.log_prob_from_logits(logits, y_query)

    def prediction_loss_from_logits(
        self,
        logits: torch.Tensor,
        y_query: torch.Tensor,
    ) -> torch.Tensor:
        """Per-point training loss used by post-training loops.

        Defaults to negative log probability, but backbones can override this
        when their native regression objective differs from log-density training.
        """
        return -self.log_prob_from_logits(logits, y_query)

    def _normalize_y_for_logits(
        self,
        logits: torch.Tensor,
        y_query: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Normalize y_query shape to match logits. Returns (normalized_y, is_batched)."""
        y_query = y_query.to(self.device).float()
        if logits.ndim == 2:
            y_query = y_query.view(-1)
            if y_query.shape[0] != logits.shape[0]:
                raise ValueError("y_query and logits must have matching first dimension.")
            return y_query, False

        if logits.ndim == 3:
            if y_query.ndim == 3 and y_query.shape[2] == 1:
                y_query = y_query.squeeze(-1)
            if y_query.ndim != 2:
                raise ValueError(
                    "For batched logits, y_query must have shape [B, N] or [B, N, 1], "
                    f"got {tuple(y_query.shape)}"
                )
            if y_query.shape[0] != logits.shape[0] or y_query.shape[1] != logits.shape[1]:
                raise ValueError("For batched logits, y_query and logits must match on [B, N].")
            return y_query, True

        raise ValueError(
            "logits must be 2D [N, C] or 3D [B, N, C], "
            f"got shape {tuple(logits.shape)}"
        )

    def _normalize_class_labels_for_logits(
        self,
        logits: torch.Tensor,
        y_query: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Normalize integer class labels to match logits batch structure."""
        y_query, is_batched = self._normalize_y_for_logits(logits, y_query)
        rounded = y_query.round()
        if not torch.allclose(y_query, rounded, atol=1e-5, rtol=0.0):
            raise ValueError("Classification targets must be integer-valued class labels.")

        y_class = rounded.long()
        if y_class.numel() > 0:
            min_class = int(y_class.min().item())
            max_class = int(y_class.max().item())
            num_classes = int(logits.shape[-1])
            if min_class < 0 or max_class >= num_classes:
                raise ValueError(
                    "Classification targets are out of range for logits: "
                    f"[{min_class}, {max_class}] vs num_classes={num_classes}."
                )
        return y_class, is_batched

    @abstractmethod
    def distribution_from_logits(self, logits: torch.Tensor) -> Any:
        """Build or return the predictive distribution object."""

    @abstractmethod
    def mean_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute predictive mean from logits."""

    @abstractmethod
    def variance_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute predictive variance from logits."""

    @abstractmethod
    def log_prob_from_logits(self, logits: torch.Tensor, y_query: torch.Tensor) -> torch.Tensor:
        """Compute pointwise log probability from precomputed logits."""
