"""Shared infrastructure for policy trainers (AL / BED / BO).

Contains the backbone protocol, policy head, optimizer/scheduler builders,
and a ``BasePolicyTrainer`` with the methods common to all three task families.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Literal, Protocol

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Backbone protocol (identical across AL / BED / BO)
# ---------------------------------------------------------------------------


class PolicyBackbone(Protocol):
    """Minimal backbone interface required by all policy trainers."""

    model: nn.Module
    device: torch.device

    def encode_context(self, x_context: Tensor, y_context: Tensor): ...

    def forward_with_embeddings(
        self,
        x_query: Tensor,
        context_state,
    ) -> dict[str, Tensor]: ...

    def log_prob_from_logits(
        self,
        logits: Tensor,
        y_query: Tensor,
    ) -> Tensor: ...

    def prediction_loss_from_logits(
        self,
        logits: Tensor,
        y_query: Tensor,
    ) -> Tensor: ...

    def enable_gradient_checkpointing(self) -> None: ...

    def disable_gradient_checkpointing(self) -> None: ...


# ---------------------------------------------------------------------------
# Policy head
# ---------------------------------------------------------------------------


def _get_activation_module(activation: Literal["relu", "gelu"]) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{activation}'.")


class CandidatePolicyHead(nn.Module):
    """Score candidates, optionally conditioning on the remaining acquisition budget.

    ``input_dim`` is always the backbone embedding dimension. Budget conditioning
    appends one feature inside the head, leaving the backbone interface unchanged.
    ``budget_scale`` is fixed when training and restored from checkpoint config.
    """

    head_type = "mlp"
    requires_target_embeddings = False

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: Literal["relu", "gelu"] = "gelu",
        dropout: float = 0.0,
        use_remaining_budget: bool = False,
        budget_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not math.isfinite(float(budget_scale)) or budget_scale <= 0:
            raise ValueError("budget_scale must be finite and positive.")

        self.embedding_dim = int(input_dim)
        self.use_remaining_budget = bool(use_remaining_budget)
        self.budget_scale = float(budget_scale)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_get_activation_module(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

        if self.use_remaining_budget:
            # Expand only after initializing the original network. This preserves
            # its weights AND RNG consumption for a same-seed budget ablation.
            first_layer = self.network[0]
            assert isinstance(first_layer, nn.Linear)
            with torch.no_grad():
                expanded_weight = torch.cat(
                    [first_layer.weight, first_layer.weight.new_zeros(hidden_dim, 1)],
                    dim=1,
                )
            first_layer.in_features = self.embedding_dim + 1
            first_layer.weight = nn.Parameter(expanded_weight)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # A legacy MLP can warm-start a budget-conditioned MLP without changing
        # its initial scores. All other shape/key mismatches remain strict errors.
        key = prefix + "network.0.weight"
        weight = state_dict.get(key)
        first_layer = self.network[0]
        assert isinstance(first_layer, nn.Linear)
        if (
            self.use_remaining_budget
            and isinstance(weight, Tensor)
            and weight.shape == (first_layer.out_features, self.embedding_dim)
        ):
            state_dict[key] = torch.cat(
                [weight, weight.new_zeros(weight.shape[0], 1)], dim=1
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(
        self,
        query_embeddings: Tensor,
        *,
        remaining_budget: int | float | None = None,
    ) -> Tensor:
        """Return one scalar score per candidate (unnormalized logit)."""
        if query_embeddings.ndim != 2:
            raise ValueError(
                "query_embeddings must be 2D [num_query, embed_dim], "
                f"got {tuple(query_embeddings.shape)}"
            )
        if self.use_remaining_budget:
            if remaining_budget is None:
                raise ValueError("remaining_budget is required for a budget-conditioned MLP.")
            remaining = float(remaining_budget)
            if not math.isfinite(remaining) or remaining < 0:
                raise ValueError("remaining_budget must be finite and non-negative.")
            budget = query_embeddings.new_full(
                (query_embeddings.shape[0], 1), remaining / self.budget_scale
            )
            query_embeddings = torch.cat([query_embeddings, budget], dim=-1)
        return self.network(query_embeddings).squeeze(-1)


class TargetAwareTransformerPolicyHead(nn.Module):
    """Cross-attend to targets, optionally conditioning each FFN on the budget.

    Budget conditioning appends one scalar to each FFN input. Attention,
    residual connections, and the final scorer retain the embedding dimension.
    """

    head_type = "target_aware_transformer"
    requires_target_embeddings = True

    def __init__(
        self,
        *,
        input_dim: int,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["relu", "gelu"] = "gelu",
        num_layers: int = 1,
        use_remaining_budget: bool = False,
        budget_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1.")
        if not math.isfinite(float(budget_scale)) or budget_scale <= 0:
            raise ValueError("budget_scale must be finite and positive.")

        self.embedding_dim = int(input_dim)
        self.use_remaining_budget = bool(use_remaining_budget)
        self.budget_scale = float(budget_scale)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(
                            embed_dim=input_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "dropout": nn.Dropout(dropout),
                        "ln_attn": nn.LayerNorm(input_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(input_dim, ffn_dim),
                            _get_activation_module(activation),
                            nn.Dropout(dropout),
                            nn.Linear(ffn_dim, input_dim),
                        ),
                        "ln_ffn": nn.LayerNorm(input_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.scorer = nn.Linear(input_dim, 1)

        if self.use_remaining_budget:
            # Initialize the complete original head first, then expand each FFN.
            # Zero columns preserve the original weights and RNG consumption.
            for layer in self.layers:
                first_layer = layer["ffn"][0]
                assert isinstance(first_layer, nn.Linear)
                with torch.no_grad():
                    expanded_weight = torch.cat(
                        [first_layer.weight, first_layer.weight.new_zeros(ffn_dim, 1)],
                        dim=1,
                    )
                first_layer.in_features = self.embedding_dim + 1
                first_layer.weight = nn.Parameter(expanded_weight)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Explicit warm starts from legacy heads only pad the new budget column.
        # All other shape/key mismatches remain strict loading errors.
        if self.use_remaining_budget:
            for index, layer in enumerate(self.layers):
                key = prefix + f"layers.{index}.ffn.0.weight"
                weight = state_dict.get(key)
                first_layer = layer["ffn"][0]
                assert isinstance(first_layer, nn.Linear)
                if (
                    isinstance(weight, Tensor)
                    and weight.shape == (first_layer.out_features, self.embedding_dim)
                ):
                    state_dict[key] = torch.cat(
                        [weight, weight.new_zeros(weight.shape[0], 1)], dim=1
                    )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(
        self,
        query_embeddings: Tensor,
        target_embeddings: Tensor | None = None,
        *,
        remaining_budget: int | float | None = None,
    ) -> Tensor:
        if target_embeddings is None:
            raise ValueError("TargetAwareTransformerPolicyHead requires target_embeddings.")

        squeeze_batch = False
        if query_embeddings.ndim == 2:
            query_embeddings = query_embeddings.unsqueeze(0)
            squeeze_batch = True
        if target_embeddings.ndim == 2:
            target_embeddings = target_embeddings.unsqueeze(0)
        if query_embeddings.ndim != 3 or target_embeddings.ndim != 3:
            raise ValueError(
                "TargetAwareTransformerPolicyHead expects query/target embeddings to be 2D or 3D."
            )
        if query_embeddings.shape[0] != target_embeddings.shape[0]:
            raise ValueError("Query and target batch size must match.")
        if query_embeddings.shape[-1] != target_embeddings.shape[-1]:
            raise ValueError("Query and target embedding dim must match.")

        budget_fraction = 0.0
        if self.use_remaining_budget:
            if remaining_budget is None:
                raise ValueError("remaining_budget is required for a budget-conditioned head.")
            remaining = float(remaining_budget)
            if not math.isfinite(remaining) or remaining < 0:
                raise ValueError("remaining_budget must be finite and non-negative.")
            budget_fraction = remaining / self.budget_scale

        hidden = query_embeddings
        for layer in self.layers:
            attn_out, _ = layer["attn"](hidden, target_embeddings, target_embeddings, need_weights=False)
            hidden = layer["ln_attn"](hidden + layer["dropout"](attn_out))
            ffn_input = hidden
            if self.use_remaining_budget:
                budget = hidden.new_full((*hidden.shape[:-1], 1), budget_fraction)
                ffn_input = torch.cat([hidden, budget], dim=-1)
            ffn_out = layer["ffn"](ffn_input)
            hidden = layer["ln_ffn"](hidden + layer["dropout"](ffn_out))

        scores = self.scorer(hidden).squeeze(-1)
        if squeeze_batch:
            return scores.squeeze(0)
        return scores


PolicyHead = nn.Module


def build_policy_head(
    *,
    input_dim: int,
    policy_head_cfg: Any,
) -> PolicyHead:
    """Build a policy head from config-like objects."""
    head_type = str(getattr(policy_head_cfg, "type", "mlp")).strip().lower()
    if head_type == "mlp":
        return CandidatePolicyHead(
            input_dim=input_dim,
            hidden_dim=int(getattr(policy_head_cfg, "hidden_dim", 128)),
            num_layers=int(getattr(policy_head_cfg, "num_layers", 2)),
            activation=str(getattr(policy_head_cfg, "activation", "gelu")),
            dropout=float(getattr(policy_head_cfg, "dropout", 0.0)),
            use_remaining_budget=bool(getattr(policy_head_cfg, "use_remaining_budget", False)),
            budget_scale=float(getattr(policy_head_cfg, "budget_scale", 1.0)),
        )
    if head_type == "target_aware_transformer":
        return TargetAwareTransformerPolicyHead(
            input_dim=input_dim,
            num_heads=int(getattr(policy_head_cfg, "num_heads", 4)),
            ffn_dim=int(getattr(policy_head_cfg, "ffn_dim", 128)),
            dropout=float(getattr(policy_head_cfg, "dropout", 0.0)),
            activation=str(getattr(policy_head_cfg, "activation", "gelu")),
            num_layers=int(getattr(policy_head_cfg, "num_layers", 1)),
            use_remaining_budget=bool(getattr(policy_head_cfg, "use_remaining_budget", False)),
            budget_scale=float(getattr(policy_head_cfg, "budget_scale", 1.0)),
        )
    raise ValueError(f"Unsupported policy_head.type '{head_type}'.")


def score_policy_head(
    policy_head: PolicyHead,
    *,
    query_embeddings: Tensor,
    target_embeddings: Tensor | None = None,
    remaining_budget: int | float | None = None,
) -> Tensor:
    """Score candidates with optional budget conditioning for either head type."""
    budget_kwargs = (
        {"remaining_budget": remaining_budget}
        if bool(getattr(policy_head, "use_remaining_budget", False))
        else {}
    )
    requires_target = bool(getattr(policy_head, "requires_target_embeddings", False))
    if requires_target:
        return policy_head(query_embeddings, target_embeddings, **budget_kwargs)

    if query_embeddings.ndim == 2:
        return policy_head(query_embeddings, **budget_kwargs)
    if query_embeddings.ndim == 3:
        batch_size, n_query, embed_dim = query_embeddings.shape
        scores_flat = policy_head(
            query_embeddings.reshape(batch_size * n_query, embed_dim), **budget_kwargs
        )
        return scores_flat.reshape(batch_size, n_query)
    raise ValueError(
        "query_embeddings must be 2D [Q, E] or 3D [B, Q, E], "
        f"got {tuple(query_embeddings.shape)}."
    )


# ---------------------------------------------------------------------------
# Optimizer & scheduler builders
# ---------------------------------------------------------------------------


def build_policy_optimizer(
    *,
    backbone: PolicyBackbone,
    policy_head: PolicyHead,
    finetune_backbone: bool,
    lr_policy: float,
    lr_backbone: float,
    weight_decay_policy: float,
    weight_decay_backbone: float,
) -> torch.optim.Optimizer:
    """Build optimizer parameter groups.

    Args:
        finetune_backbone: If ``True``, include backbone parameters in the
            optimizer (needed when ``finetune_epochs > 0``).
    """
    policy_params = [p for p in policy_head.parameters() if p.requires_grad]
    if not policy_params:
        raise ValueError("Policy head has no trainable parameters.")

    param_groups: list[dict[str, object]] = [
        {
            "params": policy_params,
            "lr": float(lr_policy),
            "weight_decay": float(weight_decay_policy),
        }
    ]

    if finetune_backbone:
        backbone_params = list(backbone.model.parameters())
        if not backbone_params:
            raise ValueError("Backbone has no parameters for finetuning.")
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(lr_backbone),
                "weight_decay": float(weight_decay_backbone),
            }
        )

    return torch.optim.AdamW(param_groups)


def build_lr_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build a cosine-annealing LR scheduler with optional linear warmup.

    Uses ``LambdaLR`` so that each parameter group independently follows
    warmup → cosine-decay with its own ``min_lr = base_lr * min_lr_ratio``.

    Args:
        optimizer: Optimizer returned by ``build_policy_optimizer``.
        num_epochs: Total training epochs.
        warmup_epochs: Linear warmup epochs (0 = no warmup).
        min_lr_ratio: Minimum LR as fraction of peak LR (0 = decay to zero).

    Returns:
        A scheduler to ``.step()`` once per epoch, or ``None`` if not needed.
    """
    if num_epochs <= 1:
        return None

    warmup_epochs = max(0, min(warmup_epochs, num_epochs - 1))
    cosine_epochs = max(num_epochs - warmup_epochs, 1)

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup: 0 → 1
            return (epoch + 1) / max(warmup_epochs, 1)
        # Cosine decay: 1 → min_lr_ratio
        progress = (epoch - warmup_epochs) / cosine_epochs
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------


class BasePolicyTrainer:
    """Shared training infrastructure for AL / BED / BO policy trainers.

    Subclasses must implement:
    - ``_rollout_batch``
    - ``_build_policy_losses``
    - ``train_one_epoch``
    """

    def __init__(
        self,
        *,
        backbone: PolicyBackbone,
        policy_head: PolicyHead,
        optimizer: torch.optim.Optimizer,
        finetune_epochs: int,
        nll_weight: float,
        policy_weight: float,
        gamma: float,
        normalize_rewards: bool,
        detach_policy_features: bool,
        grad_clip_norm: float | None,
        use_amp: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        self.backbone = backbone
        self.policy_head = policy_head
        self.optimizer = optimizer
        self.finetune_epochs = int(finetune_epochs)

        self.nll_weight = float(nll_weight)
        self.policy_weight = float(policy_weight)
        self.gamma = float(gamma)
        self.normalize_rewards = bool(normalize_rewards)
        self.detach_policy_features = bool(detach_policy_features)
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = bool(use_amp)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self._gradient_checkpointing_active = False

        if self.gamma < 0.0:
            raise ValueError("gamma must be >= 0.")

        self._backbone_trainable = self.finetune_epochs > 0
        self._configure_backbone()

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    # -- Shared helpers -----------------------------------------------------

    def _prediction_loss_from_logits(
        self,
        logits: Tensor,
        y_query: Tensor,
    ) -> Tensor:
        prediction_loss_fn = getattr(self.backbone, "prediction_loss_from_logits", None)
        if callable(prediction_loss_fn):
            return prediction_loss_fn(logits, y_query)
        return -self.backbone.log_prob_from_logits(logits, y_query)

    def _query_requires_grad(self) -> bool:
        return (
            self._backbone_trainable
            and self.policy_weight > 0.0
            and not self.detach_policy_features
        )

    def _target_requires_grad(self, *, has_targets: bool) -> bool:
        return has_targets and self._backbone_trainable and self.nll_weight > 0.0

    def _should_backward_target_nll_per_step(self, *, has_targets: bool) -> bool:
        """Backward target NLL per step when its graph is disjoint from policy logits."""
        return self._target_requires_grad(has_targets=has_targets) and not self._query_requires_grad()

    def _backward_target_nll_per_step(
        self,
        step_prediction_loss: Tensor,
        *,
        total_steps: int,
    ) -> Tensor:
        """Accumulate the per-step NLL contribution and return a detached metric tensor."""
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        (self.nll_weight * step_prediction_loss.mean() / float(total_steps)).backward()
        return step_prediction_loss.detach()

    def _step_forward_with_target_embeddings(
        self,
        query_x: Tensor,
        target_x: Tensor | None,
        context_state,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Auto-select the cheapest forward pattern for one rollout step.

        Handles three cases:
        1. No target set (``target_x is None``): query-only forward.
        2. Query and target need the same grad mode: single concatenated forward.
        3. Different grad modes: two separate forwards.

        Returns ``(query_embeddings, target_embeddings, target_logits)`` where
        the latter two values are ``None`` when ``target_x is None``.
        """
        if query_x.ndim != 3:
            raise ValueError(
                "Batched rollout expects query_x to be 3D [B, N, F], "
                f"got {tuple(query_x.shape)}."
            )

        n_query = int(query_x.shape[1])
        query_needs_grad = self._query_requires_grad()
        target_needs_grad = self._target_requires_grad(has_targets=target_x is not None)

        if target_x is None:
            context_manager = nullcontext() if query_needs_grad else torch.no_grad()
            with context_manager:
                outputs = self.backbone.forward_with_embeddings(query_x, context_state)
            query_embeddings = outputs.get("test_embeddings")
            if query_embeddings is None:
                raise KeyError("forward_with_embeddings must return 'test_embeddings'.")
            if query_embeddings.ndim != 3:
                raise ValueError(
                    "Batched rollout requires test_embeddings to be 3D [B, N_query, E], "
                    f"got {tuple(query_embeddings.shape)}."
                )
            target_embeddings = None
            target_logits = None
        elif query_needs_grad == target_needs_grad:
            context_manager = nullcontext() if query_needs_grad else torch.no_grad()
            with context_manager:
                eval_x = torch.cat([query_x, target_x], dim=1)
                outputs = self.backbone.forward_with_embeddings(eval_x, context_state)

            query_embeddings = outputs.get("test_embeddings")
            if query_embeddings is None:
                raise KeyError("forward_with_embeddings must return 'test_embeddings'.")
            if query_embeddings.ndim != 3:
                raise ValueError(
                    "Batched rollout requires test_embeddings to be 3D [B, N_query, E], "
                    f"got {tuple(query_embeddings.shape)}."
                )
            target_embeddings = query_embeddings[:, n_query:, :]
            query_embeddings = query_embeddings[:, :n_query, :]
            target_logits = outputs["logits"][:, n_query:]
        else:
            query_context = nullcontext() if query_needs_grad else torch.no_grad()
            with query_context:
                query_outputs = self.backbone.forward_with_embeddings(query_x, context_state)

            query_embeddings = query_outputs.get("test_embeddings")
            if query_embeddings is None:
                raise KeyError("forward_with_embeddings must return 'test_embeddings'.")
            if query_embeddings.ndim != 3:
                raise ValueError(
                    "Batched rollout requires test_embeddings to be 3D [B, N_query, E], "
                    f"got {tuple(query_embeddings.shape)}."
                )

            target_context = nullcontext() if target_needs_grad else torch.no_grad()
            with target_context:
                target_outputs = self.backbone.forward_with_embeddings(target_x, context_state)
            target_embeddings = target_outputs.get("test_embeddings")
            if target_embeddings is None:
                raise KeyError("forward_with_embeddings must return 'test_embeddings'.")
            if target_embeddings.ndim != 3:
                raise ValueError(
                    "Batched rollout requires target test_embeddings to be 3D [B, N_target, E], "
                    f"got {tuple(target_embeddings.shape)}."
                )
            target_logits = target_outputs["logits"]

        if self.detach_policy_features:
            query_embeddings = query_embeddings.detach()
            if target_embeddings is not None:
                target_embeddings = target_embeddings.detach()
        return query_embeddings, target_embeddings, target_logits

    def _step_forward(
        self,
        query_x: Tensor,
        target_x: Tensor | None,
        context_state,
    ) -> tuple[Tensor, Tensor | None]:
        query_embeddings, _, target_logits = self._step_forward_with_target_embeddings(
            query_x,
            target_x,
            context_state,
        )
        return query_embeddings, target_logits

    def _configure_backbone(self) -> None:
        """Set backbone grad state based on ``_backbone_trainable``."""
        should_checkpoint = self.gradient_checkpointing and self._backbone_trainable
        if should_checkpoint != self._gradient_checkpointing_active:
            if should_checkpoint:
                self.backbone.enable_gradient_checkpointing()
            else:
                self.backbone.disable_gradient_checkpointing()
            self._gradient_checkpointing_active = should_checkpoint

        if self._backbone_trainable:
            self.backbone.model.train()
            for param in self.backbone.model.parameters():
                param.requires_grad_(True)
        else:
            self.backbone.model.eval()
            for param in self.backbone.model.parameters():
                param.requires_grad_(False)
        self.policy_head.train()

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (call when epoch >= finetune_epochs)."""
        if self._backbone_trainable:
            self._backbone_trainable = False
            self._configure_backbone()

    def _backward_and_step(
        self,
        total_loss: Tensor,
        *,
        already_zeroed: bool = False,
        already_backwarded: bool = False,
    ) -> None:
        """Shared backward pass, gradient clipping, and optimizer step."""
        if not already_zeroed:
            self.optimizer.zero_grad(set_to_none=True)
        did_backward = already_backwarded
        if total_loss.requires_grad:
            total_loss.backward()
            did_backward = True
        if did_backward:
            if self.grad_clip_norm is not None and self.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self.policy_head.parameters(),
                    max_norm=float(self.grad_clip_norm),
                )
                if self._backbone_trainable:
                    nn.utils.clip_grad_norm_(
                        self.backbone.model.parameters(),
                        max_norm=float(self.grad_clip_norm),
                    )
            self.optimizer.step()
