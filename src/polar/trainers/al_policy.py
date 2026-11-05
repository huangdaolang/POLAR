"""AL policy training loop for pool-based active learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from polar.tasks.al import ALEpisode
from polar.trainers.base import (
    BasePolicyTrainer,
    PolicyBackbone,
    PolicyHead,
    score_policy_head,
)
from polar.utils.al_objectives import (
    PolicyTargetWeighting,
    aggregate_target_log_prob,
    normalize_policy_target_weighting,
    validate_policy_target_weight_beta,
    validate_policy_target_weight_threshold,
)


@dataclass
class _BatchRollout:
    """Per-batch rollout tensors used to build losses and metrics."""

    action_log_probs: Tensor
    policy_entropy: Tensor
    rewards: Tensor
    nll_losses: Tensor
    steps: int
    target_log_prob_start: Tensor
    target_log_prob_end: Tensor
    policy_target_score_start: Tensor
    policy_target_score_end: Tensor
    selected_query_y_mean: Tensor
    target_nll_backwarded: bool


class ALPolicyTrainer(BasePolicyTrainer):
    """Train AL policies with prediction loss and discounted reward-to-go.

    As in BO and BED, gamma=0 uses immediate rewards and gamma=1 uses the
    undiscounted sum of current and future rewards.
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
        reward_clamp_min: float,
        normalize_rewards: bool,
        detach_policy_features: bool,
        grad_clip_norm: float | None,
        use_amp: bool = True,
        gradient_checkpointing: bool = False,
        policy_target_weighting: PolicyTargetWeighting = "uniform",
        policy_target_weight_beta: float = 1.0,
        policy_target_weight_threshold: float = 0.0,
    ) -> None:
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        self.reward_clamp_min = float(reward_clamp_min)
        self.policy_target_weighting = normalize_policy_target_weighting(policy_target_weighting)
        self.policy_target_weight_beta = validate_policy_target_weight_beta(
            weighting=self.policy_target_weighting,
            beta=policy_target_weight_beta,
        )
        self.policy_target_weight_threshold = validate_policy_target_weight_threshold(
            policy_target_weight_threshold
        )
        super().__init__(
            backbone=backbone,
            policy_head=policy_head,
            optimizer=optimizer,
            finetune_epochs=finetune_epochs,
            nll_weight=nll_weight,
            policy_weight=policy_weight,
            gamma=gamma,
            normalize_rewards=normalize_rewards,
            detach_policy_features=detach_policy_features,
            grad_clip_norm=grad_clip_norm,
            use_amp=use_amp,
            gradient_checkpointing=gradient_checkpointing,
        )

    def train_one_epoch(
        self,
        *,
        episodes: list[ALEpisode],
        num_acquisitions: int,
    ) -> dict[str, float]:
        """Train one epoch using a batch of sampled AL episodes."""
        if not episodes:
            raise ValueError("episodes must not be empty.")
        if num_acquisitions <= 0:
            raise ValueError("num_acquisitions must be positive.")

        backward_target_nll = self._should_backward_target_nll_per_step(has_targets=True)
        self.optimizer.zero_grad(set_to_none=True)
        autocast_enabled = self.use_amp and self.device.type == "cuda"
        autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.amp.autocast(
            device_type=autocast_device_type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            rollout = self._rollout_batch(
                episodes=episodes,
                num_acquisitions=num_acquisitions,
                backward_target_nll=backward_target_nll,
            )
            policy_losses = self._build_policy_losses(
                action_log_probs=rollout.action_log_probs,
                rewards=rollout.rewards,
            )

            policy_loss = (
                policy_losses.mean()
                if policy_losses.numel() > 0
                else torch.zeros((), device=self.device)
            )
            nll_loss = (
                rollout.nll_losses.mean()
                if rollout.nll_losses.numel() > 0
                else torch.zeros((), device=self.device)
            )
            total_loss = self.nll_weight * nll_loss + self.policy_weight * policy_loss

        final_loss = self.policy_weight * policy_loss if rollout.target_nll_backwarded else total_loss
        self._backward_and_step(
            final_loss,
            already_zeroed=True,
            already_backwarded=rollout.target_nll_backwarded,
        )

        target_log_prob_gain = rollout.target_log_prob_end - rollout.target_log_prob_start
        policy_target_score_gain = rollout.policy_target_score_end - rollout.policy_target_score_start
        if rollout.policy_entropy.numel() > 0:
            policy_entropy = float(rollout.policy_entropy.detach().mean().cpu().item())
        else:
            policy_entropy = 0.0
        return {
            "steps": float(rollout.steps),
            "total_loss": float(total_loss.detach().cpu().item()),
            "nll_loss": float(nll_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "policy_entropy": policy_entropy,
            "selected_query_y_mean": float(
                rollout.selected_query_y_mean.mean().detach().cpu().item()
            ),
            "target_log_prob_start": float(
                rollout.target_log_prob_start.mean().detach().cpu().item()
            ),
            "target_log_prob_end": float(
                rollout.target_log_prob_end.mean().detach().cpu().item()
            ),
            "target_log_prob_gain": float(
                target_log_prob_gain.mean().detach().cpu().item()
            ),
            "policy_target_score_start": float(
                rollout.policy_target_score_start.mean().detach().cpu().item()
            ),
            "policy_target_score_end": float(
                rollout.policy_target_score_end.mean().detach().cpu().item()
            ),
            "policy_target_score_gain": float(
                policy_target_score_gain.mean().detach().cpu().item()
            ),
        }

    def _rollout_batch(
        self,
        *,
        episodes: list[ALEpisode],
        num_acquisitions: int,
        backward_target_nll: bool = False,
    ) -> _BatchRollout:
        """Roll out a batch of AL episodes in parallel (B episodes per forward)."""
        if not episodes:
            raise ValueError("episodes must not be empty.")

        query_len = int(episodes[0].query_x.shape[0])
        max_steps = min(int(num_acquisitions), query_len)
        if max_steps <= 0:
            raise ValueError("num_acquisitions must be positive and query set must be non-empty.")

        context_x = torch.stack([ep.context_x for ep in episodes], dim=0).to(self.device).float()
        context_y = torch.stack([ep.context_y for ep in episodes], dim=0).to(self.device).float()
        query_x = torch.stack([ep.query_x for ep in episodes], dim=0).to(self.device).float()
        query_y = torch.stack([ep.query_y for ep in episodes], dim=0).to(self.device).float()
        target_x = torch.stack([ep.target_x for ep in episodes], dim=0).to(self.device).float()
        target_y = torch.stack([ep.target_y for ep in episodes], dim=0).to(self.device).float()

        if context_y.ndim == 3 and context_y.shape[2] == 1:
            context_y = context_y.squeeze(-1)
        if query_y.ndim == 3 and query_y.shape[2] == 1:
            query_y = query_y.squeeze(-1)
        if target_y.ndim == 3 and target_y.shape[2] == 1:
            target_y = target_y.squeeze(-1)

        batch_size = int(context_x.shape[0])
        action_log_probs: list[Tensor] = []
        policy_entropies: list[Tensor] = []
        target_log_probs: list[Tensor] = []
        policy_target_scores: list[Tensor] = []
        prediction_losses: list[Tensor] = []
        selected_query_ys: list[Tensor] = []
        target_nll_backwarded = False

        for step_idx in range(max_steps):
            context_state = self.backbone.encode_context(context_x, context_y)
            n_query = int(query_x.shape[1])
            query_embeddings, target_embeddings, target_logits = (
                self._step_forward_with_target_embeddings(
                    query_x,
                    target_x,
                    context_state,
                )
            )
            if target_logits is None:
                raise RuntimeError("ALPolicyTrainer requires target logits per step.")

            policy_scores = score_policy_head(
                self.policy_head,
                query_embeddings=query_embeddings,
                target_embeddings=target_embeddings,
                remaining_budget=max_steps - step_idx,
            )
            if policy_scores.shape != (batch_size, n_query):
                raise ValueError(
                    "Policy head must return scores with shape [batch_size, n_query], "
                    f"got {tuple(policy_scores.shape)}."
                )
            action_dist = Categorical(logits=policy_scores)
            action_idx = action_dist.sample()
            action_log_probs.append(action_dist.log_prob(action_idx))
            policy_entropies.append(action_dist.entropy())

            target_log_prob = self.backbone.log_prob_from_logits(target_logits, target_y)
            if target_log_prob.ndim == 1:
                step_target_log_prob = target_log_prob
            elif target_log_prob.ndim == 2:
                step_target_log_prob = target_log_prob.mean(dim=1)
            else:
                raise ValueError(
                    "Expected target_log_prob to be 1D or 2D in batched rollout, "
                    f"got {tuple(target_log_prob.shape)}."
                )
            target_log_probs.append(step_target_log_prob.detach())
            policy_target_scores.append(
                self._aggregate_target_log_prob(target_log_prob, target_y).detach()
            )

            prediction_loss = self._prediction_loss_from_logits(target_logits, target_y)
            if prediction_loss.ndim == 1:
                step_prediction_loss = prediction_loss
            elif prediction_loss.ndim == 2:
                step_prediction_loss = prediction_loss.mean(dim=1)
            else:
                raise ValueError(
                    "Expected prediction_loss to be 1D or 2D in batched rollout, "
                    f"got {tuple(prediction_loss.shape)}."
                )
            if backward_target_nll:
                prediction_losses.append(
                    self._backward_target_nll_per_step(
                        step_prediction_loss,
                        total_steps=max_steps,
                    )
                )
                target_nll_backwarded = True
            else:
                prediction_losses.append(step_prediction_loss)

            gather_x = action_idx.view(batch_size, 1, 1).expand(-1, 1, query_x.shape[2])
            selected_x = torch.gather(query_x, dim=1, index=gather_x)
            gather_y = action_idx.view(batch_size, 1)
            selected_y = torch.gather(query_y, dim=1, index=gather_y).squeeze(1)
            selected_query_ys.append(selected_y.detach())

            context_x = torch.cat([context_x, selected_x], dim=1)
            context_y = torch.cat([context_y, selected_y.unsqueeze(1)], dim=1)

            keep_mask = (
                torch.arange(n_query, device=self.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
                != action_idx.unsqueeze(1)
            )
            query_x = query_x[keep_mask].view(batch_size, n_query - 1, query_x.shape[2])
            query_y = query_y[keep_mask].view(batch_size, n_query - 1)
            if int(query_x.shape[1]) == 0:
                break

        # The scores collected inside the loop describe D_{t-1}. Evaluate the
        # target set once more under the final context D_T so every selected
        # action, including the last one, receives its one-step utility.
        if action_log_probs:
            with torch.no_grad():
                final_context_state = self.backbone.encode_context(context_x, context_y)
                final_outputs = self.backbone.forward_with_embeddings(target_x, final_context_state)
                final_target_logits = final_outputs.get("logits")
                if final_target_logits is None:
                    raise RuntimeError("ALPolicyTrainer requires final target logits.")
                final_target_log_prob = self.backbone.log_prob_from_logits(
                    final_target_logits,
                    target_y,
                )
                if final_target_log_prob.ndim == 1:
                    final_step_target_log_prob = final_target_log_prob
                elif final_target_log_prob.ndim == 2:
                    final_step_target_log_prob = final_target_log_prob.mean(dim=1)
                else:
                    raise ValueError(
                        "Expected final target_log_prob to be 1D or 2D, "
                        f"got {tuple(final_target_log_prob.shape)}."
                    )
                target_log_probs.append(final_step_target_log_prob.detach())
                policy_target_scores.append(
                    self._aggregate_target_log_prob(
                        final_target_log_prob,
                        target_y,
                    ).detach()
                )

        steps = len(action_log_probs)
        if steps == 0:
            action_log_prob_tensor = torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)
            policy_entropy_tensor = torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)
            reward_tensor = torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)
            nll_losses = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            target_log_prob_start = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            target_log_prob_end = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            policy_target_score_start = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            policy_target_score_end = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            selected_query_y_mean = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        else:
            action_log_prob_tensor = torch.stack(action_log_probs, dim=1)
            policy_entropy_tensor = torch.stack(policy_entropies, dim=1)
            target_log_prob_tensor = torch.stack(target_log_probs, dim=1)
            policy_target_score_tensor = torch.stack(policy_target_scores, dim=1)
            prediction_loss_tensor = torch.stack(prediction_losses, dim=1)
            selected_query_y_tensor = torch.stack(selected_query_ys, dim=1)
            reward_tensor = self._build_batch_rewards(policy_target_score_tensor)
            nll_losses = prediction_loss_tensor.mean(dim=1)
            target_log_prob_start = target_log_prob_tensor[:, 0]
            target_log_prob_end = target_log_prob_tensor[:, -1]
            policy_target_score_start = policy_target_score_tensor[:, 0]
            policy_target_score_end = policy_target_score_tensor[:, -1]
            selected_query_y_mean = selected_query_y_tensor.mean(dim=1)

        return _BatchRollout(
            action_log_probs=action_log_prob_tensor,
            policy_entropy=policy_entropy_tensor,
            rewards=reward_tensor,
            nll_losses=nll_losses,
            steps=steps,
            target_log_prob_start=target_log_prob_start,
            target_log_prob_end=target_log_prob_end,
            policy_target_score_start=policy_target_score_start,
            policy_target_score_end=policy_target_score_end,
            selected_query_y_mean=selected_query_y_mean,
            target_nll_backwarded=target_nll_backwarded,
        )

    def _aggregate_target_log_prob(
        self,
        target_log_prob: Tensor,
        target_y: Tensor,
    ) -> Tensor:
        if target_log_prob.ndim == 1:
            return aggregate_target_log_prob(
                target_log_prob,
                target_y,
                weighting=self.policy_target_weighting,
                beta=self.policy_target_weight_beta,
                threshold=self.policy_target_weight_threshold,
            )
        if target_log_prob.ndim == 2:
            return aggregate_target_log_prob(
                target_log_prob,
                target_y,
                weighting=self.policy_target_weighting,
                beta=self.policy_target_weight_beta,
                threshold=self.policy_target_weight_threshold,
            )
        raise ValueError(
            "Expected target_log_prob to be 1D or 2D before aggregation, "
            f"got {tuple(target_log_prob.shape)}."
        )

    def _build_batch_rewards(self, target_log_probs: Tensor) -> Tensor:
        if target_log_probs.ndim != 2:
            raise ValueError(
                "target_log_probs must be 2D [batch, steps], "
                f"got {tuple(target_log_probs.shape)}"
            )

        batch_size, num_scores = target_log_probs.shape
        if num_scores <= 1:
            return torch.zeros((batch_size, 0), dtype=target_log_probs.dtype, device=self.device)

        reward_tensor = target_log_probs[:, 1:] - target_log_probs[:, :-1]
        reward_tensor = torch.clamp(reward_tensor, min=self.reward_clamp_min)
        return reward_tensor.detach()

    def _build_policy_losses(
        self,
        *,
        action_log_probs: Tensor,
        rewards: Tensor,
    ) -> Tensor:
        if action_log_probs.ndim != 2:
            raise ValueError(
                "action_log_probs must be 2D [batch, steps], "
                f"got {tuple(action_log_probs.shape)}"
            )
        if rewards.numel() == 0:
            return torch.zeros(action_log_probs.shape[0], dtype=action_log_probs.dtype, device=self.device)

        if action_log_probs.shape != rewards.shape:
            raise RuntimeError(
                "Action/reward shape mismatch: "
                f"{tuple(action_log_probs.shape)} vs {tuple(rewards.shape)}"
            )
        returns = rewards.detach()
        if self.gamma > 0.0:
            returns = torch.empty_like(rewards)
            running_return = torch.zeros_like(rewards[:, 0])
            for t in reversed(range(rewards.shape[1])):
                running_return = rewards[:, t].detach() + self.gamma * running_return
                returns[:, t] = running_return

        normalized_returns = returns
        if self.normalize_rewards and returns.shape[1] > 1:
            return_mean = returns.mean(dim=1, keepdim=True)
            return_std = returns.std(dim=1, unbiased=False, keepdim=True)
            normalized_returns = (returns - return_mean) / (return_std + 1e-9)

        return -(action_log_probs * normalized_returns).mean(dim=1)
