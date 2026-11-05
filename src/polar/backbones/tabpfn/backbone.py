"""TabPFN backbone wrapper for regression and classification checkpoints."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from polar.backbones.base import ContextState, TabularBackbone
from polar.backbones.base import CategoricalClassificationDistribution
from polar.backbones.registry import register_backbone
from polar.backbones.tabpfn.loader import LoadedTabPFN, load_tabpfn_model


@register_backbone("tabpfn")
class TabPFNBackbone(TabularBackbone):
    """TabPFN backbone with a model-agnostic interface."""

    def __init__(
        self,
        components: LoadedTabPFN,
        *,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.model = components.model
        self.bar_distribution = components.bar_distribution
        self.model_config = components.model_config
        self.checkpoint_path = components.checkpoint_path
        self.architecture_name = components.architecture_name
        self.checkpoint_version = components.checkpoint_version
        self.checkpoint_kind = components.checkpoint_kind
        self.is_classification = components.checkpoint_kind == "classification"
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.embedding_dim = int(self.model_config.emsize)
        self._gradient_checkpointing_enabled = False
        self._previous_recompute_layer: bool | None = None
        if not self.is_classification and self.bar_distribution is None:
            raise ValueError("Regression TabPFN backbones require a bar distribution.")
        if self.is_classification and self.num_classes is not None and self.num_classes <= 1:
            raise ValueError("Classification TabPFN backbones require num_classes >= 2.")

    @classmethod
    def load_pretrained(cls, **kwargs: Any) -> "TabPFNBackbone":
        num_classes = kwargs.pop("num_classes", None)
        components = load_tabpfn_model(**kwargs)
        return cls(components, num_classes=num_classes)

    def _effective_num_classes(self, logits: torch.Tensor) -> int:
        if not self.is_classification:
            return int(logits.shape[-1])
        configured = (
            int(self.num_classes)
            if self.num_classes is not None
            else int(getattr(self.model_config, "max_num_classes", logits.shape[-1]))
        )
        return min(configured, int(logits.shape[-1]))

    def _trim_classification_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.is_classification:
            return logits
        return logits[..., : self._effective_num_classes(logits)]

    def enable_gradient_checkpointing(self) -> None:
        """Enable stable activation checkpointing for the TabPFN layer stack.

        We only toggle the model's built-in layer-level recomputation here.
        Runtime monkey-patching of attention/MLP submodules was more aggressive,
        but it interacts poorly with incremental per-step backward and can trip
        PyTorch checkpoint metadata checks on large BO rollouts.
        """
        if self._gradient_checkpointing_enabled:
            return

        # Ensure layer-level checkpointing is on.
        if hasattr(self.model, "recompute_layer"):
            self._previous_recompute_layer = bool(self.model.recompute_layer)
            self.model.recompute_layer = True

        self._gradient_checkpointing_enabled = True

    def disable_gradient_checkpointing(self) -> None:
        """Restore the original TabPFN checkpointing state."""
        if not self._gradient_checkpointing_enabled:
            return

        if self._previous_recompute_layer is not None and hasattr(self.model, "recompute_layer"):
            self.model.recompute_layer = self._previous_recompute_layer
        self._previous_recompute_layer = None
        self._gradient_checkpointing_enabled = False

    def _forward_outputs(
        self,
        x_query: torch.Tensor,
        context_state: ContextState,
        *,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._validate_query_shape(x_query, context_state)

        x_query = x_query.to(self.device).float()
        x_context = context_state.x_context
        y_context = context_state.y_context

        is_unbatched = x_context.ndim == 2
        if is_unbatched:
            x_full = torch.cat([x_context, x_query], dim=0).unsqueeze(1)
            y_train = y_context.unsqueeze(1)
        else:
            x_full = torch.cat([x_context, x_query], dim=1).transpose(0, 1).contiguous()
            y_train = y_context.transpose(0, 1).contiguous()

        if return_embeddings:
            raw_outputs = self.model(
                x_full,
                y_train,
                only_return_standard_out=False,
                categorical_inds=None,
            )
            if not isinstance(raw_outputs, dict):
                raise TypeError("Expected dict output when only_return_standard_out=False.")

            outputs: dict[str, torch.Tensor] = {"logits": raw_outputs["standard"]}
            for key in ("test_embeddings", "train_embeddings"):
                if key in raw_outputs:
                    outputs[key] = raw_outputs[key]
        else:
            logits = self.model(
                x_full,
                y_train,
                only_return_standard_out=True,
                categorical_inds=None,
            )
            outputs = {"logits": logits}

        if is_unbatched:
            outputs = {key: value.squeeze(1) for key, value in outputs.items()}
        else:
            outputs = {key: value.transpose(0, 1).contiguous() for key, value in outputs.items()}
        outputs["logits"] = self._trim_classification_logits(outputs["logits"])
        return outputs

    def distribution_from_logits(self, logits: torch.Tensor) -> nn.Module:
        """Return predictive distribution handle (shape-dependent operations are methods)."""
        if self.is_classification:
            return CategoricalClassificationDistribution(logits)
        _ = logits  # keep signature aligned with other backbones
        return self.bar_distribution

    def mean_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.is_classification:
            return self.distribution_from_logits(logits).mean()
        return self.bar_distribution.mean(logits)

    def variance_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.is_classification:
            return self.distribution_from_logits(logits).variance().clamp_min(1e-12)
        return self.bar_distribution.variance(logits).clamp_min(1e-12)

    def log_prob_from_logits(
        self,
        logits: torch.Tensor,
        y_query: torch.Tensor,
    ) -> torch.Tensor:
        """Log probability from precomputed logits."""
        if self.is_classification:
            y_class, _ = self._normalize_class_labels_for_logits(logits, y_query)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs.gather(dim=-1, index=y_class.unsqueeze(-1)).squeeze(-1)

        y_query, is_batched = self._normalize_y_for_logits(logits, y_query)
        if is_batched:
            logits_sf = logits.transpose(0, 1).contiguous()
            y_sf = y_query.transpose(0, 1).contiguous()
        else:
            logits_sf = logits.unsqueeze(1)
            y_sf = y_query.unsqueeze(1)

        nll = self.bar_distribution(logits_sf, y_sf)
        if is_batched:
            return -nll.transpose(0, 1).contiguous()
        return -nll.squeeze(1)

