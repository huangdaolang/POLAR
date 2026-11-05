"""TabICL backbone wrapper for regression and classification checkpoints."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from polar.backbones.base import ContextState, TabularBackbone
from polar.backbones.base import CategoricalClassificationDistribution
from polar.backbones.registry import register_backbone
from polar.backbones.tabicl.loader import LoadedTabICL, load_tabicl_model


@register_backbone("tabicl")
class TabICLBackbone(TabularBackbone):
    """TabICL backbone with a model-agnostic interface."""

    def __init__(
        self,
        components: LoadedTabICL,
        *,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.model = components.model
        self.model_config = components.model_config
        self.checkpoint_path = components.checkpoint_path
        self.checkpoint_version = components.checkpoint_version
        self.checkpoint_kind = components.checkpoint_kind
        self.is_classification = components.checkpoint_kind == "classification"
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.architecture_name = "tabicl"
        self.embedding_dim = int(self.model.embed_dim * self.model.row_num_cls)
        self._gradient_checkpointing_enabled = False
        self._recompute_modules: list[tuple[object, bool]] = []
        if self.is_classification and self.num_classes is not None and self.num_classes <= 1:
            raise ValueError("Classification TabICL backbones require num_classes >= 2.")

    @classmethod
    def load_pretrained(cls, **kwargs: Any) -> "TabICLBackbone":
        kwargs.pop("cache_trainset_representation", None)  # config compat with tabpfn
        num_classes = kwargs.pop("num_classes", None)
        components = load_tabicl_model(**kwargs)
        return cls(components, num_classes=num_classes)

    def _effective_num_classes(self, logits: torch.Tensor) -> int:
        if not self.is_classification:
            return int(logits.shape[-1])
        configured = (
            int(self.num_classes)
            if self.num_classes is not None
            else int(self.model_config.get("max_classes", logits.shape[-1]))
        )
        return min(configured, int(logits.shape[-1]))

    def _trim_classification_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.is_classification:
            return logits
        return logits[..., : self._effective_num_classes(logits)]

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing in all TabICL encoder / interaction blocks."""
        if self._gradient_checkpointing_enabled:
            return

        self._recompute_modules = []
        for module in self.model.modules():
            if hasattr(module, "recompute"):
                previous = bool(getattr(module, "recompute"))
                self._recompute_modules.append((module, previous))
                module.recompute = True
        self._gradient_checkpointing_enabled = True

    def disable_gradient_checkpointing(self) -> None:
        """Restore each TabICL submodule's original recompute setting."""
        if not self._gradient_checkpointing_enabled:
            return

        for module, previous in self._recompute_modules:
            module.recompute = previous
        self._recompute_modules.clear()
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
            x_full = torch.cat([x_context, x_query], dim=0).unsqueeze(0)
            y_train = y_context.unsqueeze(0)
            train_size = int(x_context.shape[0])
        else:
            x_full = torch.cat([x_context, x_query], dim=1)
            y_train = y_context
            train_size = int(x_context.shape[1])

        row_representations = self.model.row_interactor(
            self.model.col_embedder(
                x_full,
                y_train=y_train,
                d=None,
                embed_with_test=False,
            ),
            d=None,
        )
        if return_embeddings:
            contextual_representations = row_representations.clone()
            if self.is_classification:
                y_train_embeddings = self.model.icl_predictor.y_encoder(y_train.float())
            else:
                y_train_embeddings = self.model.icl_predictor.y_encoder(y_train.unsqueeze(-1))
            contextual_representations[:, :train_size, :] = (
                contextual_representations[:, :train_size, :] + y_train_embeddings
            )

            contextual_representations = self.model.icl_predictor.tf_icl(
                contextual_representations,
                train_size=train_size,
            )
            if self.model.icl_predictor.norm_first:
                contextual_representations = self.model.icl_predictor.ln(
                    contextual_representations
                )

            quantiles = self.model.icl_predictor.decoder(
                contextual_representations
            )[:, train_size:, :]
            outputs = {
                "logits": quantiles,
                "test_embeddings": contextual_representations[:, train_size:, :],
                "train_embeddings": contextual_representations[:, :train_size, :],
            }
        else:
            quantiles = self.model.icl_predictor(row_representations, y_train=y_train)
            outputs = {"logits": quantiles}

        if is_unbatched:
            outputs = {key: value.squeeze(0) for key, value in outputs.items()}
        outputs["logits"] = self._trim_classification_logits(outputs["logits"])
        return outputs

    def distribution_from_logits(self, logits: torch.Tensor):
        """Construct predictive distribution from TabICL outputs."""
        if self.is_classification:
            return CategoricalClassificationDistribution(logits)
        return self.model.quantile_dist(logits)

    def mean_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.distribution_from_logits(logits).mean()

    def variance_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.distribution_from_logits(logits).variance().clamp_min(1e-12)

    def log_prob_from_logits(
        self,
        logits: torch.Tensor,
        y_query: torch.Tensor,
    ) -> torch.Tensor:
        """Log probability from precomputed outputs."""
        if self.is_classification:
            y_class, _ = self._normalize_class_labels_for_logits(logits, y_query)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs.gather(dim=-1, index=y_class.unsqueeze(-1)).squeeze(-1)

        y_query, _ = self._normalize_y_for_logits(logits, y_query)
        return self.distribution_from_logits(logits).log_prob(y_query)

    def prediction_loss_from_logits(
        self,
        logits: torch.Tensor,
        y_query: torch.Tensor,
    ) -> torch.Tensor:
        """TabICL uses pinball loss for regression and NLL for classification."""
        if self.is_classification:
            return super().prediction_loss_from_logits(logits, y_query)
        y_query, _ = self._normalize_y_for_logits(logits, y_query)
        return self.distribution_from_logits(logits).pinball(y_query)

