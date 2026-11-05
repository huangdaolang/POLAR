"""BO policy post-training loop for pool-based Bayesian optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.distributions import Categorical

from polar.tasks.bo import BOEpisode
from polar.trainers.base import (
    BasePolicyTrainer,
    CandidatePolicyHead,
    PolicyBackbone,
    score_policy_head,
)

RewardType = Literal["best_y", "incremental", "regret_reduction", "log_regret_reduction"]
RewardNormScope = Literal["episode", "batch"]


@dataclass
class _BatchRollout:
    """Per-batch rollout tensors used to build losses and metrics."""

    action_log_probs: Tensor
    policy_entropy: Tensor
    rewards: Tensor
    nll_losses: Tensor
    steps: int
    best_y_start: Tensor
    best_y_end: Tensor
    reward_mean: Tensor
    reward_nonzero_ratio: Tensor
    target_log_prob_start: Tensor
    target_log_prob_end: Tensor
    target_nll_backwarded: bool


class BOPolicyTrainer(BasePolicyTrainer):
    """Train BO policies with prediction loss and discounted reward-to-go.

    As in BED, gamma=0 uses immediate rewards and gamma=1 uses the
    undiscounted sum of current and future rewards.
    """

    def __init__(
        self,
        *,
        backbone: PolicyBackbone,
        policy_head: CandidatePolicyHead,
        optimizer: torch.optim.Optimizer,
        finetune_epochs: int,
        nll_weight: float,
        policy_weight: float,
        gamma: float,
        reward_type: RewardType,
        reward_log_epsilon: float,
        reward_scale_by_range: bool,
        reward_no_improve_penalty: float,
        reward_clamp_min: float | None,
        normalize_rewards: bool,
        reward_normalization_scope: RewardNormScope,
        detach_policy_features: bool,
        grad_clip_norm: float | None,
        use_amp: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        self.reward_type = reward_type
        self.reward_log_epsilon = float(reward_log_epsilon)
        self.reward_scale_by_range = bool(reward_scale_by_range)
        self.reward_no_improve_penalty = float(reward_no_improve_penalty)
        self.reward_clamp_min = reward_clamp_min
        self.reward_normalization_scope = reward_normalization_scope

        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if self.reward_log_epsilon <= 0.0:
            raise ValueError("reward_log_epsilon must be positive.")
        if self.reward_normalization_scope not in {"episode", "batch"}:
            raise ValueError(
                "reward_normalization_scope must be one of: 'episode', 'batch'."
            )
        if self.reward_type not in {"best_y", "incremental", "regret_reduction", "log_regret_reduction"}:
            raise ValueError(
                "reward_type must be one of: "
                "'best_y', 'incremental', 'regret_reduction', 'log_regret_reduction'."
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
        episodes: list[BOEpisode],
        num_acquisitions: int,
    ) -> dict[str, float]:
        """Train one epoch using a batch of pre-sampled BO episodes."""
        if not episodes:
            raise ValueError("episodes must not be empty.")
        if num_acquisitions <= 0:
            raise ValueError("num_acquisitions must be positive.")

        has_targets = all(ep.target_x is not None and ep.target_y is not None for ep in episodes)
        backward_target_nll = self._should_backward_target_nll_per_step(has_targets=has_targets)
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

        if rollout.policy_entropy.numel() > 0:
            policy_entropy = float(rollout.policy_entropy.detach().mean().cpu().item())
        else:
            policy_entropy = 0.0

        best_y_improvement = rollout.best_y_end - rollout.best_y_start
        best_y_improvement_mean = float(best_y_improvement.mean().detach().cpu().item())

        return {
            "steps": float(rollout.steps),
            "total_loss": float(total_loss.detach().cpu().item()),
            "nll_loss": float(nll_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "policy_entropy": policy_entropy,
            "reward_mean": float(rollout.reward_mean.mean().detach().cpu().item()),
            "reward_nonzero_ratio": float(
                rollout.reward_nonzero_ratio.mean().detach().cpu().item()
            ),
            "best_y_start": float(rollout.best_y_start.mean().detach().cpu().item()),
            "best_y_end": float(rollout.best_y_end.mean().detach().cpu().item()),
            "best_y_improvement": best_y_improvement_mean,
            "target_log_prob_start": float(
                rollout.target_log_prob_start.mean().detach().cpu().item()
            ),
            "target_log_prob_end": float(
                rollout.target_log_prob_end.mean().detach().cpu().item()
            ),
        }

    def _rollout_batch(
        self,
        *,
        episodes: list[BOEpisode],
        num_acquisitions: int,
        backward_target_nll: bool = False,
    ) -> _BatchRollout:
        """Roll out a batch of BO episodes in parallel (B episodes per forward)."""
        if not episodes:
            raise ValueError("episodes must not be empty.")

        first = episodes[0]
        query_len = int(first.query_x.shape[0])
        max_steps = min(int(num_acquisitions), query_len)
        if max_steps <= 0:
            raise ValueError("num_acquisitions must be positive and query set must be non-empty.")

        context_x = torch.stack([ep.context_x for ep in episodes], dim=0).to(self.device).float()
        context_y = torch.stack([ep.context_y for ep in episodes], dim=0).to(self.device).float()
        query_x = torch.stack([ep.query_x for ep in episodes], dim=0).to(self.device).float()
        query_y = torch.stack([ep.query_y for ep in episodes], dim=0).to(self.device).float()
        has_targets = all(ep.target_x is not None and ep.target_y is not None for ep in episodes)
        use_target_auxiliary = has_targets and self._backbone_trainable and self.nll_weight > 0.0
        if use_target_auxiliary:
            target_x = torch.stack(
                [ep.target_x for ep in episodes], dim=0  # type: ignore[arg-type]
            ).to(self.device).float()
            target_y = torch.stack(
                [ep.target_y for ep in episodes], dim=0  # type: ignore[arg-type]
            ).to(self.device).float()
        else:
            target_x = None
            target_y = None

        if context_y.ndim == 3 and context_y.shape[2] == 1:
            context_y = context_y.squeeze(-1)
        if query_y.ndim == 3 and query_y.shape[2] == 1:
            query_y = query_y.squeeze(-1)
        if target_y is not None and target_y.ndim == 3 and target_y.shape[2] == 1:
            target_y = target_y.squeeze(-1)

        batch_size = int(context_x.shape[0])
        action_log_probs: list[Tensor] = []
        policy_entropies: list[Tensor] = []
        rewards: list[Tensor] = []
        target_log_probs: list[Tensor] = []
        prediction_losses: list[Tensor] = []
        target_nll_backwarded = False

        best_y_start_tensor = torch.max(context_y, dim=1).values
        y_best_tensor = torch.tensor(
            [float(ep.y_best) for ep in episodes],
            device=self.device,
            dtype=context_y.dtype,
        )
        y_min_tensor = torch.tensor(
            [float(ep.y_min) for ep in episodes],
            device=self.device,
            dtype=context_y.dtype,
        )
        range_scale_tensor = torch.clamp(y_best_tensor - y_min_tensor, min=1e-12)

        for step_idx in range(max_steps):
            context_state = self.backbone.encode_context(context_x, context_y)
            n_query = int(query_x.shape[1])
            query_embeddings, target_logits = self._step_forward(query_x, target_x, context_state)

            policy_scores = score_policy_head(
                self.policy_head,
                query_embeddings=query_embeddings,
                remaining_budget=max_steps - step_idx,
            )
            action_dist = Categorical(logits=policy_scores)
            action_idx = action_dist.sample()
            action_log_probs.append(action_dist.log_prob(action_idx))
            policy_entropies.append(action_dist.entropy())

            if target_logits is not None and target_y is not None:
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

            prev_best_y = torch.max(context_y, dim=1).values
            gather_x = action_idx.view(batch_size, 1, 1).expand(-1, 1, query_x.shape[2])
            selected_x = torch.gather(query_x, dim=1, index=gather_x)
            gather_y = action_idx.view(batch_size, 1)
            selected_y = torch.gather(query_y, dim=1, index=gather_y).squeeze(1)
            new_best_y = torch.maximum(prev_best_y, selected_y)

            reward = self._build_step_reward(
                prev_best_y=prev_best_y,
                new_best_y=new_best_y,
                y_best=y_best_tensor,
                range_scale=range_scale_tensor,
            )
            rewards.append(reward)

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

        steps = len(action_log_probs)
        if steps == 0:
            action_log_prob_tensor = torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)
            policy_entropy_tensor = torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)
            reward_tensor = torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)
            nll_losses = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            target_log_prob_start = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            target_log_prob_end = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        else:
            action_log_prob_tensor = torch.stack(action_log_probs, dim=1)
            policy_entropy_tensor = torch.stack(policy_entropies, dim=1)
            reward_tensor = torch.stack(rewards, dim=1)
            if prediction_losses:
                prediction_loss_tensor = torch.stack(prediction_losses, dim=1)
                nll_losses = prediction_loss_tensor.mean(dim=1)
            else:
                nll_losses = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            if target_log_probs:
                target_log_prob_tensor = torch.stack(target_log_probs, dim=1)
                target_log_prob_start = target_log_prob_tensor[:, 0]
                target_log_prob_end = target_log_prob_tensor[:, -1]
            else:
                target_log_prob_start = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
                target_log_prob_end = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        reward_mean_tensor = (
            reward_tensor.detach().mean(dim=1)
            if reward_tensor.numel() > 0
            else torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        )
        reward_nonzero_ratio_tensor = (
            (reward_tensor.detach().abs() > 1e-12).float().mean(dim=1)
            if reward_tensor.numel() > 0
            else torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        )
        best_y_end_tensor = torch.max(context_y, dim=1).values

        return _BatchRollout(
            action_log_probs=action_log_prob_tensor,
            policy_entropy=policy_entropy_tensor,
            rewards=reward_tensor,
            nll_losses=nll_losses,
            steps=steps,
            best_y_start=best_y_start_tensor,
            best_y_end=best_y_end_tensor,
            reward_mean=reward_mean_tensor,
            reward_nonzero_ratio=reward_nonzero_ratio_tensor,
            target_log_prob_start=target_log_prob_start,
            target_log_prob_end=target_log_prob_end,
            target_nll_backwarded=target_nll_backwarded,
        )

    def _build_step_reward(
        self,
        *,
        prev_best_y: Tensor,
        new_best_y: Tensor,
        y_best: float | Tensor,
        range_scale: float | Tensor,
    ) -> Tensor:
        if torch.is_tensor(y_best):
            y_best_tensor = y_best.to(self.device, dtype=new_best_y.dtype)
        else:
            y_best_tensor = torch.tensor(y_best, device=self.device, dtype=new_best_y.dtype)

        if self.reward_type == "best_y":
            reward = new_best_y
        elif self.reward_type == "incremental":
            reward = new_best_y - prev_best_y
        elif self.reward_type == "regret_reduction":
            prev_regret = y_best_tensor - prev_best_y
            new_regret = y_best_tensor - new_best_y
            reward = prev_regret - new_regret
        elif self.reward_type == "log_regret_reduction":
            prev_regret = torch.clamp(y_best_tensor - prev_best_y, min=0.0)
            new_regret = torch.clamp(y_best_tensor - new_best_y, min=0.0)
            eps = torch.tensor(self.reward_log_epsilon, device=self.device, dtype=new_best_y.dtype)
            reward = torch.log(prev_regret + eps) - torch.log(new_regret + eps)
        else:
            raise ValueError(f"Unsupported reward_type '{self.reward_type}'.")

        if self.reward_no_improve_penalty != 0.0:
            no_improve = new_best_y <= prev_best_y + 1e-12
            penalty = torch.tensor(
                self.reward_no_improve_penalty,
                device=self.device,
                dtype=reward.dtype,
            )
            reward = torch.where(no_improve, reward + penalty, reward)

        if self.reward_scale_by_range and self.reward_type != "log_regret_reduction":
            if torch.is_tensor(range_scale):
                denom = torch.clamp(range_scale.to(self.device, dtype=reward.dtype), min=1e-12)
            else:
                denom = max(float(range_scale), 1e-12)
            reward = reward / denom

        if self.reward_clamp_min is not None:
            reward = torch.clamp(reward, min=float(self.reward_clamp_min))

        return reward

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
            return torch.zeros(
                action_log_probs.shape[0],
                dtype=action_log_probs.dtype,
                device=self.device,
            )
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
        if self.normalize_rewards:
            if self.reward_normalization_scope == "episode":
                if returns.shape[1] > 1:
                    return_mean = returns.mean(dim=1, keepdim=True)
                    return_std = returns.std(dim=1, unbiased=False, keepdim=True)
                    normalized_returns = (returns - return_mean) / (return_std + 1e-9)
            elif returns.numel() > 1:
                batch_mean = returns.mean()
                batch_std = returns.std(unbiased=False)
                normalized_returns = (returns - batch_mean) / (batch_std + 1e-9)

        return -(action_log_probs * normalized_returns).mean(dim=1)
