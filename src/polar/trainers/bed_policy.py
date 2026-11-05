"""BED policy training loop for sequential Bayesian Experimental Design.

Uses REINFORCE with dense sPCE rewards computed from an analytical
log-likelihood function. The policy selects designs from a discrete
candidate pool to maximize information about latent parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.distributions import Categorical

from polar.tasks.bed import BEDEpisode
from polar.trainers.base import (
    BasePolicyTrainer,
    CandidatePolicyHead,
    PolicyBackbone,
    score_policy_head,
)
from polar.utils.eig import (
    compute_cumulative_log_likelihood,
    compute_spce,
    dense_spce_reward,
)


@dataclass
class _BatchRollout:
    """Per-batch rollout tensors used to build losses and metrics."""

    action_log_probs: Tensor  # [B, T]
    rewards: Tensor  # [B, T]
    nll_losses: Tensor  # [B]  (zeros if no target set)
    steps: int
    spce_estimate: Tensor  # [B]
    target_log_prob_start: Tensor  # [B]
    target_log_prob_end: Tensor  # [B]
    policy_entropy: Tensor  # [B, T] — per-step Categorical entropy
    target_nll_backwarded: bool


class BEDPolicyTrainer(BasePolicyTrainer):
    """Train policy head with REINFORCE using dense sPCE rewards.

    Rewards are computed from an analytical log-likelihood function (not the
    backbone's predictions). The dense sPCE recursion starts from the fixed
    initialized history, so each reward is the incremental information gain of
    the next policy acquisition conditioned on all points seen so far. Backbone
    gradients come only from the optional NLL prediction loss on held-out
    target points (full_finetune mode).
    """

    def __init__(
        self,
        *,
        backbone: PolicyBackbone,
        policy_head: CandidatePolicyHead,
        optimizer: torch.optim.Optimizer,
        finetune_epochs: int,
        log_likelihood_fn: Callable[..., Tensor],
        # Loss weights
        nll_weight: float,
        policy_weight: float,
        # REINFORCE
        gamma: float,
        use_baseline: bool,
        baseline_momentum: float,
        normalize_rewards: bool,
        detach_policy_features: bool,
        # Optimization
        grad_clip_norm: float | None,
        use_amp: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        self.log_likelihood_fn = log_likelihood_fn
        self.use_baseline = bool(use_baseline)
        self.baseline_momentum = float(baseline_momentum)
        if not 0.0 < self.baseline_momentum < 1.0:
            raise ValueError("baseline_momentum must be in (0, 1).")

        self._baseline: Tensor | None = None

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
        episodes: list[BEDEpisode],
        num_acquisitions: int,
    ) -> dict[str, float]:
        """Train one epoch using a batch of BED episodes."""
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
            policy_loss = self._build_policy_losses(
                action_log_probs=rollout.action_log_probs,
                rewards=rollout.rewards,
            ).mean()

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

        reward_mean = (
            float(rollout.rewards.detach().mean().cpu().item())
            if rollout.rewards.numel() > 0
            else 0.0
        )
        spce_mean = float(rollout.spce_estimate.detach().mean().cpu().item())

        # Policy entropy: mean over batch and steps
        if rollout.policy_entropy.numel() > 0:
            entropy_mean = float(rollout.policy_entropy.detach().mean().cpu().item())
        else:
            entropy_mean = 0.0

        return {
            "steps": float(rollout.steps),
            "total_loss": float(total_loss.detach().cpu().item()),
            "nll_loss": float(nll_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "reward_mean": reward_mean,
            "spce_estimate": spce_mean,
            "policy_entropy": entropy_mean,
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
        episodes: list[BEDEpisode],
        num_acquisitions: int,
        backward_target_nll: bool = False,
    ) -> _BatchRollout:
        """Roll out a batch of BED episodes in parallel."""
        if not episodes:
            raise ValueError("episodes must not be empty.")

        batch_size = len(episodes)
        query_len = int(episodes[0].query_x.shape[0])
        max_steps = min(int(num_acquisitions), query_len)
        if max_steps <= 0:
            raise ValueError("num_acquisitions must be positive and query set must be non-empty.")

        # Stack episode data into batched tensors.
        # theta_all keeps the task-specific latent shape after [L+1, B, ...].
        theta_all = torch.stack(
            [ep.theta_all for ep in episodes], dim=1
        ).to(self.device).float()

        context_x = torch.stack(
            [ep.context_x for ep in episodes], dim=0
        ).to(self.device).float()
        context_y = torch.stack(
            [ep.context_y for ep in episodes], dim=0
        ).to(self.device).float()
        query_x = torch.stack(
            [ep.query_x for ep in episodes], dim=0
        ).to(self.device).float()
        query_y = torch.stack(
            [ep.query_y for ep in episodes], dim=0
        ).to(self.device).float()

        # Handle optional target set
        has_targets = all(ep.target_x is not None and ep.target_y is not None for ep in episodes)
        use_target_auxiliary = has_targets and self._backbone_trainable and self.nll_weight > 0.0
        if use_target_auxiliary:
            target_x = torch.stack(
                [ep.target_x for ep in episodes], dim=0  # type: ignore[arg-type]
            ).to(self.device).float()
            target_y = torch.stack(
                [ep.target_y for ep in episodes], dim=0  # type: ignore[arg-type]
            ).to(self.device).float()
            if target_y.ndim == 3 and target_y.shape[2] == 1:
                target_y = target_y.squeeze(-1)
        else:
            target_x = None
            target_y = None

        # Squeeze trailing y dims
        if context_y.ndim == 3 and context_y.shape[2] == 1:
            context_y = context_y.squeeze(-1)
        if query_y.ndim == 3 and query_y.shape[2] == 1:
            query_y = query_y.squeeze(-1)

        L_plus_1 = theta_all.shape[0]
        d = query_x.shape[2]

        # Keep the running history on top of the fixed initial context so the
        # rollout can report full-trajectory sPCE including the initialization.
        if int(context_x.shape[1]) > 0:
            initial_cum_log_lik = compute_cumulative_log_likelihood(
                log_lik_fn=self.log_likelihood_fn,
                y_trajectory=context_y,
                x_trajectory=context_x,
                theta_all=theta_all,
            ).to(device=self.device, dtype=torch.float32)
        else:
            initial_cum_log_lik = torch.zeros(
                L_plus_1, batch_size, device=self.device, dtype=torch.float32
            )
        cum_log_lik = initial_cum_log_lik.clone()

        action_log_probs: list[Tensor] = []
        rewards: list[Tensor] = []
        policy_entropies: list[Tensor] = []
        target_log_probs: list[Tensor] = []
        prediction_losses: list[Tensor] = []
        target_nll_backwarded = False

        for step_idx in range(max_steps):
            context_state = self.backbone.encode_context(context_x, context_y)
            n_query = int(query_x.shape[1])
            query_embeddings, target_logits = self._step_forward(query_x, target_x, context_state)

            # Policy scores and action sampling
            policy_scores = score_policy_head(
                self.policy_head,
                query_embeddings=query_embeddings,
                remaining_budget=max_steps - step_idx,
            )
            action_dist = Categorical(logits=policy_scores)
            action_idx = action_dist.sample()  # [B]
            action_log_probs.append(action_dist.log_prob(action_idx))
            policy_entropies.append(action_dist.entropy())  # [B]

            # Target prediction metrics (optional NLL)
            if target_logits is not None and target_y is not None:
                target_log_prob = self.backbone.log_prob_from_logits(target_logits, target_y)
                if target_log_prob.ndim == 1:
                    step_target_lp = target_log_prob
                elif target_log_prob.ndim == 2:
                    step_target_lp = target_log_prob.mean(dim=1)
                else:
                    raise ValueError(
                        "Expected target_log_prob to be 1D or 2D, "
                        f"got {tuple(target_log_prob.shape)}."
                    )
                target_log_probs.append(step_target_lp.detach())

                pred_loss = self._prediction_loss_from_logits(target_logits, target_y)
                if pred_loss.ndim == 1:
                    step_pred_loss = pred_loss
                elif pred_loss.ndim == 2:
                    step_pred_loss = pred_loss.mean(dim=1)
                else:
                    raise ValueError(
                        "Expected prediction_loss to be 1D or 2D, "
                        f"got {tuple(pred_loss.shape)}."
                    )
                if backward_target_nll:
                    prediction_losses.append(
                        self._backward_target_nll_per_step(
                            step_pred_loss,
                            total_steps=max_steps,
                        )
                    )
                    target_nll_backwarded = True
                else:
                    prediction_losses.append(step_pred_loss)

            # Gather selected design and observation
            gather_x = action_idx.view(batch_size, 1, 1).expand(-1, 1, d)
            x_t = torch.gather(query_x, dim=1, index=gather_x).squeeze(1)  # [B, d]
            gather_y = action_idx.view(batch_size, 1)
            y_t = torch.gather(query_y, dim=1, index=gather_y).squeeze(1)  # [B]

            # Dense sPCE increment over the running history, including any
            # fixed initialization points already present in cum_log_lik.
            step_log_lik = self.log_likelihood_fn(
                y_t.unsqueeze(0),  # [1, B]
                x_t.unsqueeze(0),  # [1, B, d]
                theta_all,  # [L+1, B, ...theta_shape]
            )  # [L+1, B]

            step_log_lik = step_log_lik.to(cum_log_lik.dtype)
            reward_t, cum_log_lik = dense_spce_reward(step_log_lik, cum_log_lik)
            rewards.append(reward_t)

            # Update context
            context_x = torch.cat([context_x, x_t.unsqueeze(1)], dim=1)
            context_y = torch.cat([context_y, y_t.unsqueeze(1)], dim=1)

            # Remove selected from pool
            keep_mask = (
                torch.arange(n_query, device=self.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
                != action_idx.unsqueeze(1)
            )
            query_x = query_x[keep_mask].view(batch_size, n_query - 1, d)
            query_y = query_y[keep_mask].view(batch_size, n_query - 1)
            if int(query_x.shape[1]) == 0:
                break

        # Assemble rollout
        steps = len(action_log_probs)
        if steps == 0:
            zero_b = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            return _BatchRollout(
                action_log_probs=torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device),
                rewards=torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device),
                nll_losses=zero_b,
                steps=0,
                spce_estimate=zero_b,
                target_log_prob_start=zero_b,
                target_log_prob_end=zero_b,
                policy_entropy=torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device),
                target_nll_backwarded=False,
            )

        action_log_prob_tensor = torch.stack(action_log_probs, dim=1)  # [B, T]
        reward_tensor = torch.stack(rewards, dim=1)  # [B, T]
        entropy_tensor = torch.stack(policy_entropies, dim=1)  # [B, T]
        spce_estimate = compute_spce(cum_log_lik)  # [B]

        if prediction_losses:
            prediction_loss_tensor = torch.stack(prediction_losses, dim=1)  # [B, T]
            nll_losses = prediction_loss_tensor.mean(dim=1)  # [B]
        else:
            nll_losses = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        if target_log_probs:
            target_lp_tensor = torch.stack(target_log_probs, dim=1)  # [B, T]
            target_log_prob_start = target_lp_tensor[:, 0]
            target_log_prob_end = target_lp_tensor[:, -1]
        else:
            target_log_prob_start = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            target_log_prob_end = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        return _BatchRollout(
            action_log_probs=action_log_prob_tensor,
            rewards=reward_tensor,
            nll_losses=nll_losses,
            steps=steps,
            spce_estimate=spce_estimate,
            target_log_prob_start=target_log_prob_start,
            target_log_prob_end=target_log_prob_end,
            policy_entropy=entropy_tensor,
            target_nll_backwarded=target_nll_backwarded,
        )

    def _build_policy_losses(
        self,
        *,
        action_log_probs: Tensor,
        rewards: Tensor,
    ) -> Tensor:
        """REINFORCE loss with discounted returns and optional baseline.

        Args:
            action_log_probs: Log-probabilities of selected actions, shape ``[B, T]``.
            rewards: Per-step PCE rewards, shape ``[B, T]``.

        Returns:
            Per-episode policy loss, shape ``[B]``.
        """
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

        B, T = rewards.shape

        # Compute discounted returns: G_t = r_t + gamma * G_{t+1}
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(B, device=self.device, dtype=rewards.dtype)
        for t in reversed(range(T)):
            running_return = rewards[:, t] + self.gamma * running_return
            returns[:, t] = running_return

        # Baseline subtraction (EMA of batch mean return)
        advantages = returns.detach()
        if self.use_baseline:
            batch_return_mean = returns.detach().mean()
            if self._baseline is None:
                self._baseline = batch_return_mean
            else:
                self._baseline = (
                    self.baseline_momentum * self._baseline
                    + (1.0 - self.baseline_momentum) * batch_return_mean
                )
            advantages = advantages - self._baseline.detach()

        # Optional normalization
        if self.normalize_rewards:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = advantages / (adv_std + 1e-8)

        return -(action_log_probs * advantages).mean(dim=1)  # [B]
