"""Reusable evaluation helpers for active-learning policy rollouts."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch

from polar.trainers import score_policy_head
from polar.utils.al_acquisitions import (
    ALAcquisitionInputs,
    compute_al_acquisition_scores,
)
from polar.utils.al_objectives import (
    PolicyTargetWeighting,
    aggregate_target_log_prob,
    compute_weighted_target_rmse,
)
from polar.utils.surrogates import Surrogate

PolicySelection = Literal["argmax", "sample"]

_GP_VARIANCE_REDUCTION_ACQUISITIONS = {
    "variance_reduction",
    "weighted_variance_reduction",
}


def snapshot_round_indices(*, max_rounds: int, stride: int) -> set[int]:
    """Choose 0-based round indices to snapshot, always keeping the final round."""
    if max_rounds <= 0:
        return set()
    if stride <= 1:
        return set(range(max_rounds))

    indices = {0, max_rounds - 1}
    next_human_round = int(stride)
    while next_human_round <= max_rounds:
        indices.add(next_human_round - 1)
        next_human_round += int(stride)
    return indices


def episode_raw_x_to_model(episode: Any, raw_x: torch.Tensor) -> torch.Tensor:
    """Map raw x coordinates into the model space used by the backbone."""
    x_model = raw_x.float()
    x_center = getattr(episode, "x_center", None)
    x_scale = getattr(episode, "x_scale", None)
    if x_center is None or x_scale is None:
        return x_model

    center = torch.as_tensor(x_center, dtype=x_model.dtype, device=x_model.device).view(1, -1)
    scale = torch.as_tensor(x_scale, dtype=x_model.dtype, device=x_model.device).view(1, -1)
    return (x_model - center) / scale


def episode_model_y_to_raw(episode: Any, y_model: torch.Tensor) -> torch.Tensor:
    """Map model-space y values back to raw function values when metadata exists."""
    y_raw = y_model.float()
    y_mean = getattr(episode, "y_mean", None)
    y_std = getattr(episode, "y_std", None)
    if y_mean is None or y_std is None:
        return y_raw

    mean = torch.as_tensor(y_mean, dtype=y_raw.dtype, device=y_raw.device)
    std = torch.as_tensor(y_std, dtype=y_raw.dtype, device=y_raw.device)
    return y_raw * std + mean


def episode_model_std_to_raw(episode: Any, std_model: torch.Tensor) -> torch.Tensor:
    """Map model-space predictive std back to raw scale when metadata exists."""
    std_raw = std_model.float()
    y_std = getattr(episode, "y_std", None)
    if y_std is None:
        return std_raw

    scale = torch.as_tensor(y_std, dtype=std_raw.dtype, device=std_raw.device)
    return std_raw * scale


def _serialize_query_point(point: torch.Tensor) -> float | list[float]:
    flat = point.detach().view(-1).cpu()
    if flat.numel() == 1:
        return float(flat.item())
    return [float(v.item()) for v in flat]


def rollout_policy_probe_episode(
    *,
    episode: Any,
    backbone: Any,
    policy_head: torch.nn.Module,
    num_acquisitions: int,
    selection: PolicySelection = "argmax",
    collect_snapshots: bool = True,
    snapshot_round_stride: int = 1,
    policy_target_weighting: PolicyTargetWeighting = "uniform",
    policy_target_weight_beta: float = 1.0,
    policy_target_weight_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one policy rollout on a fixed episode and collect diagnostic metrics."""
    if num_acquisitions <= 0:
        raise ValueError("num_acquisitions must be positive.")

    device = backbone.device
    context_x = episode.context_x.unsqueeze(0).to(device).float()
    context_y = episode.context_y.unsqueeze(0).to(device).float()
    query_x = episode.query_x.unsqueeze(0).to(device).float()
    query_y = episode.query_y.unsqueeze(0).to(device).float()
    target_x = episode.target_x.unsqueeze(0).to(device).float()
    target_y = episode.target_y.unsqueeze(0).to(device).float()

    if context_y.ndim == 3 and context_y.shape[2] == 1:
        context_y = context_y.squeeze(-1)
    if query_y.ndim == 3 and query_y.shape[2] == 1:
        query_y = query_y.squeeze(-1)
    if target_y.ndim == 3 and target_y.shape[2] == 1:
        target_y = target_y.squeeze(-1)

    plot_x = getattr(episode, "plot_x", None)
    if collect_snapshots and plot_x is None:
        raise ValueError("Probe snapshots require episodes with a plot_x attribute.")
    plot_x_model = plot_x.to(device).float() if plot_x is not None else None

    raw_plot_x = getattr(episode, "raw_plot_x", None)
    raw_plot_y_true = getattr(episode, "raw_plot_y_true", None)
    raw_context_x = getattr(episode, "raw_context_x", episode.context_x)
    raw_context_y = getattr(episode, "raw_context_y", episode.context_y)
    raw_target_x = getattr(episode, "raw_target_x", episode.target_x)
    raw_target_y = getattr(episode, "raw_target_y", episode.target_y)
    raw_query_x = getattr(episode, "raw_query_x", None)
    raw_query_y = getattr(episode, "raw_query_y", None)

    round_metrics: list[dict[str, Any]] = []
    round_snapshots: list[dict[str, Any]] = []
    acquired_raw_x: list[torch.Tensor] = []
    acquired_raw_y: list[torch.Tensor] = []

    max_rounds = min(int(num_acquisitions), int(query_x.shape[1]))
    snapshot_rounds = snapshot_round_indices(max_rounds=max_rounds, stride=snapshot_round_stride)

    for round_idx in range(max_rounds):
        n_query = int(query_x.shape[1])
        with torch.no_grad():
            context_state = backbone.encode_context(context_x, context_y)
            x_union = torch.cat([query_x, target_x], dim=1)
            outputs = backbone.forward_with_embeddings(x_union, context_state)

            union_logits = outputs["logits"]
            union_mean = outputs["mean"]
            union_embeddings = outputs.get("test_embeddings")
            if union_embeddings is None:
                raise KeyError("forward_with_embeddings must return 'test_embeddings'.")

            query_embeddings = union_embeddings[:, :n_query]
            target_embeddings = union_embeddings[:, n_query:]
            target_logits = union_logits[:, n_query:]
            target_mean = union_mean[:, n_query:]

            target_log_prob = backbone.log_prob_from_logits(target_logits, target_y)
            if target_log_prob.ndim == 1:
                target_log_prob = target_log_prob.unsqueeze(0)

            target_rmse = torch.sqrt(torch.mean((target_mean - target_y) ** 2, dim=1))
            weighted_target_rmse = compute_weighted_target_rmse(
                target_mean,
                target_y,
                weighting=policy_target_weighting,
                beta=policy_target_weight_beta,
                threshold=policy_target_weight_threshold,
            )
            target_log_prob_mean = target_log_prob.mean(dim=1)
            weighted_target_log_prob = aggregate_target_log_prob(
                target_log_prob,
                target_y,
                weighting=policy_target_weighting,
                beta=policy_target_weight_beta,
                threshold=policy_target_weight_threshold,
            )

            policy_scores = score_policy_head(
                policy_head,
                query_embeddings=query_embeddings,
                target_embeddings=target_embeddings,
                remaining_budget=max_rounds - round_idx,
            )
            if policy_scores.shape != (1, n_query):
                raise ValueError(
                    "Policy head must return scores with shape [1, N_q], "
                    f"got {tuple(policy_scores.shape)}."
                )
            if selection == "argmax":
                selected_indices = torch.argmax(policy_scores, dim=1)
            elif selection == "sample":
                action_dist = torch.distributions.Categorical(logits=policy_scores)
                selected_indices = action_dist.sample()
            else:
                raise ValueError(f"Unsupported selection mode: {selection}")

            policy_probs = torch.softmax(policy_scores, dim=1)
            policy_entropy = -torch.sum(
                policy_probs * torch.log(policy_probs.clamp_min(1.0e-12)),
                dim=1,
            )

            selected_idx = int(selected_indices[0].item())
            selected_query_x = query_x[0, selected_idx].detach().cpu()
            selected_query_y = query_y[0, selected_idx].detach().cpu()
            selected_prob = float(policy_probs[0, selected_idx].detach().cpu().item())

            selected_raw_x = None
            selected_raw_y = None
            if raw_query_x is not None and raw_query_y is not None:
                selected_raw_x = raw_query_x[selected_idx : selected_idx + 1].detach().cpu()
                selected_raw_y = raw_query_y[selected_idx : selected_idx + 1].detach().cpu()

            round_metrics.append(
                {
                    "round": int(round_idx),
                    "context_size": int(context_x.shape[1]),
                    "query_size": int(query_x.shape[1]),
                    "target_size": int(target_x.shape[1]),
                    "target_rmse": float(target_rmse[0].detach().cpu().item()),
                    "weighted_target_rmse": float(
                        weighted_target_rmse[0].detach().cpu().item()
                    ),
                    "target_log_prob_mean": float(
                        target_log_prob_mean[0].detach().cpu().item()
                    ),
                    "weighted_target_log_prob": float(
                        weighted_target_log_prob[0].detach().cpu().item()
                    ),
                    "selected_query_x": float(selected_query_x.view(-1)[0].item()),
                    "selected_query_y": float(selected_query_y.item()),
                    "selected_query_raw_y": (
                        float(selected_raw_y.view(-1)[0].item()) if selected_raw_y is not None else None
                    ),
                    "selected_prob": float(selected_prob),
                    "policy_entropy": float(policy_entropy[0].detach().cpu().item()),
                }
            )

            if collect_snapshots and plot_x_model is not None and round_idx in snapshot_rounds:
                plot_dist = backbone.predictive_distribution(
                    plot_x_model.unsqueeze(0),
                    context_state,
                )
                plot_mean = plot_dist["mean"][0].detach().cpu()
                plot_std = torch.sqrt(
                    backbone.variance_from_logits(plot_dist["logits"][0]).clamp_min(1.0e-12)
                ).detach().cpu()
                acquired_now_x = acquired_raw_x + ([selected_raw_x] if selected_raw_x is not None else [])
                acquired_now_y = acquired_raw_y + ([selected_raw_y] if selected_raw_y is not None else [])
                snapshot = {
                    "round": int(round_idx),
                    "target_rmse": float(target_rmse[0].detach().cpu().item()),
                    "weighted_target_rmse": float(weighted_target_rmse[0].detach().cpu().item()),
                    "target_log_prob_mean": float(target_log_prob_mean[0].detach().cpu().item()),
                    "weighted_target_log_prob": float(
                        weighted_target_log_prob[0].detach().cpu().item()
                    ),
                    "selected_query_x": selected_query_x,
                    "selected_query_y": selected_query_y,
                    "context_x": context_x[0].detach().cpu(),
                    "context_y": context_y[0].detach().cpu(),
                    "target_x": target_x[0].detach().cpu(),
                    "target_y": target_y[0].detach().cpu(),
                    "true_y": getattr(episode, "plot_y_true").detach().cpu(),
                    "pred_mean": plot_mean,
                    "pred_std": plot_std,
                    "score_values": policy_probs[0].detach().cpu(),
                    "score_label": "Policy prob",
                    "score_kind": "policy_prob",
                    "selected_score_value": float(policy_probs[0, selected_idx].detach().cpu().item()),
                }
                if raw_query_x is not None:
                    snapshot["score_query_raw_x"] = raw_query_x.detach().cpu()
                else:
                    snapshot["score_query_x"] = query_x[0].detach().cpu()
                if raw_plot_x is not None and raw_plot_y_true is not None:
                    snapshot.update(
                        {
                            "raw_plot_x": raw_plot_x.detach().cpu(),
                            "raw_plot_y_true": raw_plot_y_true.detach().cpu(),
                            "pred_mean_raw": episode_model_y_to_raw(episode, plot_mean),
                            "pred_std_raw": episode_model_std_to_raw(episode, plot_std),
                            "raw_context_x": raw_context_x.detach().cpu(),
                            "raw_context_y": raw_context_y.detach().cpu(),
                            "raw_target_x": raw_target_x.detach().cpu(),
                            "raw_target_y": raw_target_y.detach().cpu(),
                            "acquired_raw_x": (
                                torch.cat(acquired_now_x, dim=0)
                                if acquired_now_x
                                else torch.empty((0, raw_plot_x.shape[-1]), dtype=torch.float32)
                            ),
                            "acquired_raw_y": (
                                torch.cat(acquired_now_y, dim=0)
                                if acquired_now_y
                                else torch.empty((0,), dtype=torch.float32)
                            ),
                            "selected_query_raw_x": (
                                selected_raw_x.view(-1).detach().cpu()
                                if selected_raw_x is not None
                                else None
                            ),
                        }
                    )
                round_snapshots.append(snapshot)

        gather_x = selected_indices.view(1, 1, 1).expand(-1, 1, query_x.shape[2])
        selected_x = torch.gather(query_x, dim=1, index=gather_x)
        gather_y = selected_indices.view(1, 1)
        selected_y = torch.gather(query_y, dim=1, index=gather_y).squeeze(1)

        context_x = torch.cat([context_x, selected_x], dim=1)
        context_y = torch.cat([context_y, selected_y.unsqueeze(1)], dim=1)

        keep_mask = (
            torch.arange(n_query, device=device)
            .unsqueeze(0)
            .expand(1, -1)
            != selected_indices.unsqueeze(1)
        )
        query_x = query_x[keep_mask].view(1, n_query - 1, query_x.shape[2])
        query_y = query_y[keep_mask].view(1, n_query - 1)

        if raw_query_x is not None and raw_query_y is not None and selected_raw_x is not None and selected_raw_y is not None:
            acquired_raw_x.append(selected_raw_x)
            acquired_raw_y.append(selected_raw_y)
            keep_mask_cpu = keep_mask[0].detach().cpu()
            raw_query_x = raw_query_x[keep_mask_cpu]
            raw_query_y = raw_query_y[keep_mask_cpu]

        if int(query_x.shape[1]) == 0:
            break

    return round_metrics, round_snapshots


def evaluate_policy_episodes(
    *,
    episodes: list[Any],
    backbone: Any,
    policy_head: torch.nn.Module,
    num_acquisitions: int,
    selection: PolicySelection = "argmax",
    policy_target_weighting: PolicyTargetWeighting = "uniform",
    policy_target_weight_beta: float = 1.0,
    policy_target_weight_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Evaluate one policy across many episodes in a batched rollout loop."""
    if not episodes:
        return []
    if num_acquisitions <= 0:
        raise ValueError("num_acquisitions must be positive.")

    device = backbone.device
    batch_size = len(episodes)
    batch_idx = torch.arange(batch_size, device=device)

    context_x = torch.stack([episode.context_x for episode in episodes]).to(device).float()
    context_y = torch.stack([episode.context_y for episode in episodes]).to(device).float()
    query_x = torch.stack([episode.query_x for episode in episodes]).to(device).float()
    query_y = torch.stack([episode.query_y for episode in episodes]).to(device).float()
    target_x = torch.stack([episode.target_x for episode in episodes]).to(device).float()
    target_y = torch.stack([episode.target_y for episode in episodes]).to(device).float()

    if context_y.ndim == 3 and context_y.shape[2] == 1:
        context_y = context_y.squeeze(-1)
    if query_y.ndim == 3 and query_y.shape[2] == 1:
        query_y = query_y.squeeze(-1)
    if target_y.ndim == 3 and target_y.shape[2] == 1:
        target_y = target_y.squeeze(-1)

    raw_query_y_by_episode = [
        getattr(episode, "raw_query_y", None).detach().cpu()
        if getattr(episode, "raw_query_y", None) is not None
        else None
        for episode in episodes
    ]

    round_metrics: list[dict[str, Any]] = []
    max_rounds = min(int(num_acquisitions), int(query_x.shape[1]))

    for round_idx in range(max_rounds):
        n_query = int(query_x.shape[1])
        with torch.no_grad():
            context_state = backbone.encode_context(context_x, context_y)
            x_union = torch.cat([query_x, target_x], dim=1)
            outputs = backbone.forward_with_embeddings(x_union, context_state)

            union_logits = outputs["logits"]
            union_mean = outputs["mean"]
            union_embeddings = outputs.get("test_embeddings")
            if union_embeddings is None:
                raise KeyError("forward_with_embeddings must return 'test_embeddings'.")

            query_embeddings = union_embeddings[:, :n_query]
            target_embeddings = union_embeddings[:, n_query:]
            target_logits = union_logits[:, n_query:]
            target_mean = union_mean[:, n_query:]

            target_log_prob = backbone.log_prob_from_logits(target_logits, target_y)
            if target_log_prob.ndim == 1:
                target_log_prob = target_log_prob.unsqueeze(0)

            target_rmse = torch.sqrt(torch.mean((target_mean - target_y) ** 2, dim=1))
            weighted_target_rmse = compute_weighted_target_rmse(
                target_mean,
                target_y,
                weighting=policy_target_weighting,
                beta=policy_target_weight_beta,
                threshold=policy_target_weight_threshold,
            )
            target_log_prob_mean = target_log_prob.mean(dim=1)
            weighted_target_log_prob = aggregate_target_log_prob(
                target_log_prob,
                target_y,
                weighting=policy_target_weighting,
                beta=policy_target_weight_beta,
                threshold=policy_target_weight_threshold,
            )

            policy_scores = score_policy_head(
                policy_head,
                query_embeddings=query_embeddings,
                target_embeddings=target_embeddings,
                remaining_budget=max_rounds - round_idx,
            )
            if policy_scores.shape != (batch_size, n_query):
                raise ValueError(
                    "Policy head must return scores with shape [B, N_q], "
                    f"got {tuple(policy_scores.shape)}."
                )

            if selection == "argmax":
                selected_indices = torch.argmax(policy_scores, dim=1)
            elif selection == "sample":
                action_dist = torch.distributions.Categorical(logits=policy_scores)
                selected_indices = action_dist.sample()
            else:
                raise ValueError(f"Unsupported selection mode: {selection}")

            policy_probs = torch.softmax(policy_scores, dim=1)
            selected_probs = policy_probs[batch_idx, selected_indices]
            policy_entropy = -torch.sum(
                policy_probs * torch.log(policy_probs.clamp_min(1.0e-12)),
                dim=1,
            )

        selected_x = query_x[batch_idx, selected_indices]
        selected_y = query_y[batch_idx, selected_indices]

        for episode_idx, episode in enumerate(episodes):
            raw_query_y = raw_query_y_by_episode[episode_idx]
            raw_selected_y = None
            if raw_query_y is not None:
                raw_selected_y = float(raw_query_y[int(selected_indices[episode_idx].item())].item())

            round_metrics.append(
                {
                    "episode": int(episode_idx),
                    "round": int(round_idx),
                    "dataset_id": str(getattr(episode, "dataset_id", "gp_prior")),
                    "episode_seed": getattr(episode, "seed", None),
                    "context_size": int(context_x.shape[1]),
                    "query_size": int(n_query),
                    "target_size": int(target_x.shape[1]),
                    "target_rmse": float(target_rmse[episode_idx].detach().cpu().item()),
                    "weighted_target_rmse": float(
                        weighted_target_rmse[episode_idx].detach().cpu().item()
                    ),
                    "target_log_prob_mean": float(
                        target_log_prob_mean[episode_idx].detach().cpu().item()
                    ),
                    "weighted_target_log_prob": float(
                        weighted_target_log_prob[episode_idx].detach().cpu().item()
                    ),
                    "selected_query_x": _serialize_query_point(selected_x[episode_idx]),
                    "selected_query_y": float(selected_y[episode_idx].detach().cpu().item()),
                    "selected_query_raw_y": raw_selected_y,
                    "selected_prob": float(selected_probs[episode_idx].detach().cpu().item()),
                    "policy_entropy": float(policy_entropy[episode_idx].detach().cpu().item()),
                }
            )

        context_x = torch.cat([context_x, selected_x.unsqueeze(1)], dim=1)
        context_y = torch.cat([context_y, selected_y.unsqueeze(1)], dim=1)

        keep_mask = torch.ones(batch_size, n_query, dtype=torch.bool, device=device)
        keep_mask[batch_idx, selected_indices] = False
        query_x = query_x[keep_mask].reshape(batch_size, n_query - 1, query_x.shape[-1])
        query_y = query_y[keep_mask].reshape(batch_size, n_query - 1)

        updated_raw_query_y: list[torch.Tensor | None] = []
        keep_mask_cpu = keep_mask.detach().cpu()
        for episode_idx, raw_query_y in enumerate(raw_query_y_by_episode):
            if raw_query_y is None:
                updated_raw_query_y.append(None)
                continue
            updated_raw_query_y.append(raw_query_y[keep_mask_cpu[episode_idx]])
        raw_query_y_by_episode = updated_raw_query_y

        if n_query - 1 == 0:
            break

    return round_metrics


def rollout_surrogate_probe_episode(
    *,
    episode: Any,
    surrogate: Surrogate,
    acquisition_name: str,
    num_acquisitions: int,
    collect_snapshots: bool = True,
    snapshot_round_stride: int = 1,
    policy_target_weighting: PolicyTargetWeighting = "uniform",
    policy_target_weight_beta: float = 1.0,
    policy_target_weight_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one surrogate-based rollout on a fixed episode and collect diagnostics."""
    if num_acquisitions <= 0:
        raise ValueError("num_acquisitions must be positive.")

    device = surrogate.device
    context_x = episode.context_x.to(device).float()
    context_y = episode.context_y.to(device).float()
    query_x = episode.query_x.to(device).float()
    query_y = episode.query_y.to(device).float()
    target_x = episode.target_x.to(device).float()
    target_y = episode.target_y.to(device).float()

    if context_y.ndim == 2 and context_y.shape[1] == 1:
        context_y = context_y.squeeze(-1)
    if query_y.ndim == 2 and query_y.shape[1] == 1:
        query_y = query_y.squeeze(-1)
    if target_y.ndim == 2 and target_y.shape[1] == 1:
        target_y = target_y.squeeze(-1)

    plot_x = getattr(episode, "plot_x", None)
    if collect_snapshots and plot_x is None:
        raise ValueError("Probe snapshots require episodes with a plot_x attribute.")

    raw_plot_x = getattr(episode, "raw_plot_x", None)
    raw_plot_y_true = getattr(episode, "raw_plot_y_true", None)
    raw_context_x = getattr(episode, "raw_context_x", episode.context_x)
    raw_context_y = getattr(episode, "raw_context_y", episode.context_y)
    raw_target_x = getattr(episode, "raw_target_x", episode.target_x)
    raw_target_y = getattr(episode, "raw_target_y", episode.target_y)
    raw_query_x = getattr(episode, "raw_query_x", None)
    raw_query_y = getattr(episode, "raw_query_y", None)

    round_metrics: list[dict[str, Any]] = []
    round_snapshots: list[dict[str, Any]] = []
    acquired_raw_x: list[torch.Tensor] = []
    acquired_raw_y: list[torch.Tensor] = []

    max_rounds = min(int(num_acquisitions), int(query_x.shape[0]))
    snapshot_rounds = snapshot_round_indices(max_rounds=max_rounds, stride=snapshot_round_stride)

    for round_idx in range(max_rounds):
        n_query = int(query_x.shape[0])
        surrogate.fit(context_x, context_y)
        target_query_covariance = None
        if (
            acquisition_name in _GP_VARIANCE_REDUCTION_ACQUISITIONS
            and hasattr(surrogate, "predict_query_target_joint")
        ):
            joint_pred = surrogate.predict_query_target_joint(query_x, target_x)
            query_mean = joint_pred.query_mean
            query_variance = joint_pred.query_variance
            target_mean = joint_pred.target_mean
            target_variance = joint_pred.target_variance
            target_query_covariance = joint_pred.target_query_covariance
        else:
            x_union = torch.cat([query_x, target_x], dim=0)
            union_pred = surrogate.predict(x_union)
            query_mean = union_pred.mean[:n_query]
            query_variance = union_pred.variance[:n_query]
            target_mean = union_pred.mean[n_query:]
            target_variance = union_pred.variance[n_query:]

        target_log_prob = surrogate.point_log_prob(target_x, target_y)
        target_rmse = torch.sqrt(torch.mean((target_mean - target_y) ** 2))
        weighted_target_rmse = compute_weighted_target_rmse(
            target_mean,
            target_y,
            weighting=policy_target_weighting,
            beta=policy_target_weight_beta,
            threshold=policy_target_weight_threshold,
        )
        target_log_prob_mean = target_log_prob.mean()
        weighted_target_log_prob = aggregate_target_log_prob(
            target_log_prob,
            target_y,
            weighting=policy_target_weighting,
            beta=policy_target_weight_beta,
            threshold=policy_target_weight_threshold,
        )

        acquisition_inputs = ALAcquisitionInputs(
            backend=str(surrogate.backend_name),
            query_x=query_x,
            query_mean=query_mean,
            query_variance=query_variance,
            target_x=target_x,
            target_mean=target_mean,
            target_variance=target_variance,
            target_query_covariance=target_query_covariance,
            target_weighting=str(policy_target_weighting),
            target_weight_beta=float(policy_target_weight_beta),
            target_weight_threshold=float(policy_target_weight_threshold),
        )
        acquisition_scores = compute_al_acquisition_scores(
            name=acquisition_name,
            inputs=acquisition_inputs,
        )
        selected_idx = int(torch.argmax(acquisition_scores).item())

        selected_x = query_x[selected_idx : selected_idx + 1]
        selected_y = query_y[selected_idx : selected_idx + 1]
        selected_raw_x = None
        selected_raw_y = None
        if raw_query_x is not None and raw_query_y is not None:
            selected_raw_x = raw_query_x[selected_idx : selected_idx + 1].detach().cpu()
            selected_raw_y = raw_query_y[selected_idx : selected_idx + 1].detach().cpu()

        round_metrics.append(
            {
                "round": int(round_idx),
                "context_size": int(context_x.shape[0]),
                "query_size": int(query_x.shape[0]),
                "target_size": int(target_x.shape[0]),
                "target_rmse": float(target_rmse.detach().cpu().item()),
                "weighted_target_rmse": float(weighted_target_rmse.detach().cpu().item()),
                "target_log_prob_mean": float(target_log_prob_mean.detach().cpu().item()),
                "weighted_target_log_prob": float(weighted_target_log_prob.detach().cpu().item()),
                "selected_query_x": _serialize_query_point(selected_x[0]),
                "selected_query_y": float(selected_y[0].detach().cpu().item()),
                "selected_query_raw_y": (
                    float(selected_raw_y.view(-1)[0].item()) if selected_raw_y is not None else None
                ),
                "selected_prob": None,
                "policy_entropy": None,
            }
        )

        if collect_snapshots and plot_x is not None and round_idx in snapshot_rounds:
            plot_pred = surrogate.predict(plot_x.to(device).float())
            plot_mean = plot_pred.mean.detach().cpu()
            plot_std = torch.sqrt(plot_pred.variance.clamp_min(1.0e-12)).detach().cpu()
            acquired_now_x = acquired_raw_x + ([selected_raw_x] if selected_raw_x is not None else [])
            acquired_now_y = acquired_raw_y + ([selected_raw_y] if selected_raw_y is not None else [])
            snapshot = {
                "round": int(round_idx),
                "target_rmse": float(target_rmse.detach().cpu().item()),
                "weighted_target_rmse": float(weighted_target_rmse.detach().cpu().item()),
                "target_log_prob_mean": float(target_log_prob_mean.detach().cpu().item()),
                "weighted_target_log_prob": float(weighted_target_log_prob.detach().cpu().item()),
                "selected_query_x": selected_x[0].detach().cpu(),
                "selected_query_y": selected_y[0].detach().cpu(),
                "context_x": context_x.detach().cpu(),
                "context_y": context_y.detach().cpu(),
                "target_x": target_x.detach().cpu(),
                "target_y": target_y.detach().cpu(),
                "true_y": getattr(episode, "plot_y_true").detach().cpu(),
                "pred_mean": plot_mean,
                "pred_std": plot_std,
                "score_values": acquisition_scores.detach().cpu(),
                "score_label": str(acquisition_name).replace("_", " "),
                "score_kind": "acquisition_score",
                "selected_score_value": float(acquisition_scores[selected_idx].detach().cpu().item()),
            }
            if raw_query_x is not None:
                snapshot["score_query_raw_x"] = raw_query_x.detach().cpu()
            else:
                snapshot["score_query_x"] = query_x.detach().cpu()
            if raw_plot_x is not None and raw_plot_y_true is not None:
                snapshot.update(
                    {
                        "raw_plot_x": raw_plot_x.detach().cpu(),
                        "raw_plot_y_true": raw_plot_y_true.detach().cpu(),
                        "pred_mean_raw": episode_model_y_to_raw(episode, plot_mean),
                        "pred_std_raw": episode_model_std_to_raw(episode, plot_std),
                        "raw_context_x": raw_context_x.detach().cpu(),
                        "raw_context_y": raw_context_y.detach().cpu(),
                        "raw_target_x": raw_target_x.detach().cpu(),
                        "raw_target_y": raw_target_y.detach().cpu(),
                        "acquired_raw_x": (
                            torch.cat(acquired_now_x, dim=0)
                            if acquired_now_x
                            else torch.empty((0, raw_plot_x.shape[-1]), dtype=torch.float32)
                        ),
                        "acquired_raw_y": (
                            torch.cat(acquired_now_y, dim=0)
                            if acquired_now_y
                            else torch.empty((0,), dtype=torch.float32)
                        ),
                        "selected_query_raw_x": (
                            selected_raw_x.view(-1).detach().cpu()
                            if selected_raw_x is not None
                            else None
                        ),
                    }
                )
            round_snapshots.append(snapshot)

        context_x = torch.cat([context_x, selected_x], dim=0)
        context_y = torch.cat([context_y, selected_y], dim=0)

        keep_mask = torch.ones(query_x.shape[0], dtype=torch.bool, device=device)
        keep_mask[selected_idx] = False
        query_x = query_x[keep_mask]
        query_y = query_y[keep_mask]

        if raw_query_x is not None and raw_query_y is not None and selected_raw_x is not None and selected_raw_y is not None:
            acquired_raw_x.append(selected_raw_x)
            acquired_raw_y.append(selected_raw_y)
            keep_mask_cpu = keep_mask.detach().cpu()
            raw_query_x = raw_query_x[keep_mask_cpu]
            raw_query_y = raw_query_y[keep_mask_cpu]

        if int(query_x.shape[0]) == 0:
            break

    return round_metrics, round_snapshots


def evaluate_surrogate_episodes(
    *,
    episodes: list[Any],
    surrogate: Surrogate,
    acquisition_name: str,
    num_acquisitions: int,
    policy_target_weighting: PolicyTargetWeighting = "uniform",
    policy_target_weight_beta: float = 1.0,
    policy_target_weight_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Evaluate one surrogate-based AL method across many episodes."""
    all_round_metrics: list[dict[str, Any]] = []
    for episode_idx, episode in enumerate(episodes):
        round_metrics, _ = rollout_surrogate_probe_episode(
            episode=episode,
            surrogate=surrogate,
            acquisition_name=acquisition_name,
            num_acquisitions=num_acquisitions,
            collect_snapshots=False,
            policy_target_weighting=policy_target_weighting,
            policy_target_weight_beta=policy_target_weight_beta,
            policy_target_weight_threshold=policy_target_weight_threshold,
        )
        for metric in round_metrics:
            all_round_metrics.append(
                {
                    "episode": int(episode_idx),
                    "dataset_id": str(getattr(episode, "dataset_id", "gp_prior")),
                    "episode_seed": getattr(episode, "seed", None),
                    **metric,
                }
            )
    return all_round_metrics


def aggregate_al_eval_metrics(round_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate round-wise AL evaluation metrics across episodes."""
    if not round_metrics:
        return []

    metric_fields = (
        ("target_rmse", "target_rmse"),
        ("weighted_target_rmse", "weighted_target_rmse"),
        ("target_log_prob_mean", "target_log_prob"),
        ("weighted_target_log_prob", "weighted_target_log_prob"),
        ("selected_query_y", "selected_query_y"),
        ("selected_query_raw_y", "selected_query_raw_y"),
        ("selected_prob", "selected_prob"),
        ("policy_entropy", "policy_entropy"),
    )

    aggregated: list[dict[str, Any]] = []
    rounds = sorted({int(metric["round"]) for metric in round_metrics})
    for round_idx in rounds:
        bucket = [metric for metric in round_metrics if int(metric["round"]) == round_idx]
        row: dict[str, Any] = {
            "round": int(round_idx),
            "num_episodes": int(len(bucket)),
        }

        for field_name, prefix in metric_fields:
            values = [
                float(metric[field_name])
                for metric in bucket
                if metric.get(field_name) is not None
            ]
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float64)
            count = int(tensor.numel())
            mean = float(tensor.mean().item())
            std = float(tensor.std(unbiased=False).item()) if count > 1 else 0.0
            sem = float(std / math.sqrt(count)) if count > 0 else 0.0
            ci95 = float(1.96 * sem)
            row[f"mean_{prefix}"] = mean
            row[f"std_{prefix}"] = std
            row[f"sem_{prefix}"] = sem
            row[f"ci95_{prefix}"] = ci95

        aggregated.append(row)

    return aggregated


def aggregate_al_eval_metrics_by_dataset(
    round_metrics: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group AL evaluation metrics by dataset and aggregate each group by round."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in round_metrics:
        dataset_id = metric.get("dataset_id")
        if dataset_id is None:
            continue
        grouped.setdefault(str(dataset_id), []).append(metric)
    return {
        dataset_id: aggregate_al_eval_metrics(metrics)
        for dataset_id, metrics in grouped.items()
    }


def summarize_probe_metrics(round_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Reduce round-wise probe metrics to a compact summary for logging."""
    if not round_metrics:
        return {
            "rounds": 0.0,
            "target_rmse_start": 0.0,
            "target_rmse_end": 0.0,
            "weighted_target_rmse_start": 0.0,
            "weighted_target_rmse_end": 0.0,
            "target_log_prob_start": 0.0,
            "target_log_prob_end": 0.0,
            "weighted_target_log_prob_start": 0.0,
            "weighted_target_log_prob_end": 0.0,
            "selected_query_y_mean": 0.0,
        }

    first = round_metrics[0]
    last = round_metrics[-1]
    selected_y = torch.tensor(
        [float(metric["selected_query_y"]) for metric in round_metrics],
        dtype=torch.float32,
    )
    return {
        "rounds": float(len(round_metrics)),
        "target_rmse_start": float(first["target_rmse"]),
        "target_rmse_end": float(last["target_rmse"]),
        "weighted_target_rmse_start": float(first["weighted_target_rmse"]),
        "weighted_target_rmse_end": float(last["weighted_target_rmse"]),
        "target_log_prob_start": float(first["target_log_prob_mean"]),
        "target_log_prob_end": float(last["target_log_prob_mean"]),
        "weighted_target_log_prob_start": float(first["weighted_target_log_prob"]),
        "weighted_target_log_prob_end": float(last["weighted_target_log_prob"]),
        "selected_query_y_mean": float(selected_y.mean().item()),
        "selected_query_raw_y_mean": float(
            torch.tensor(
                [
                    float(metric["selected_query_raw_y"])
                    for metric in round_metrics
                    if metric["selected_query_raw_y"] is not None
                ],
                dtype=torch.float32,
            ).mean().item()
        )
        if any(metric["selected_query_raw_y"] is not None for metric in round_metrics)
        else 0.0,
    }
