"""Evaluation helpers for Bayesian Experimental Design policies."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.distributions import Categorical

from polar.backbones import create_backbone
from polar.tasks import create_bed_task
from polar.tasks.bed import BEDEpisode, BEDTask, bed_task_checkpoint_kind
from polar.tasks.location_finding import LocationFindingSampler
from polar.trainers.base import CandidatePolicyHead, PolicyBackbone, score_policy_head
from polar.utils.eig import compute_eig_bounds_from_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class EpisodeResult:
    """Result of a single BED evaluation episode."""

    method: str
    spce: float
    snmc: float
    theta_true: Any
    x_traj: Any
    num_steps: int


@torch.no_grad()
def rollout_bed_policy(
    *,
    backbone: PolicyBackbone,
    policy_head: CandidatePolicyHead,
    episode: BEDEpisode,
    num_acquisitions: int,
    greedy: bool,
) -> tuple[list[Tensor], list[Tensor]]:
    """Roll out a trained policy on one BED episode."""

    device = backbone.device
    context_x = episode.context_x.to(device)
    context_y = episode.context_y.to(device)
    query_x = episode.query_x.to(device)
    query_y = episode.query_y.to(device)

    max_steps = min(num_acquisitions, int(query_x.shape[0]))
    x_traj: list[Tensor] = []
    y_traj: list[Tensor] = []

    for step_idx in range(max_steps):
        context_state = backbone.encode_context(context_x, context_y)
        outputs = backbone.forward_with_embeddings(query_x, context_state)
        embeddings = outputs["test_embeddings"]
        scores = score_policy_head(
            policy_head,
            query_embeddings=embeddings,
            remaining_budget=max_steps - step_idx,
        )

        if greedy:
            action_idx = scores.argmax()
        else:
            action_idx = Categorical(logits=scores).sample()

        x_t = query_x[action_idx]
        y_t = query_y[action_idx]
        x_traj.append(x_t)
        y_traj.append(y_t)

        context_x = torch.cat([context_x, x_t.unsqueeze(0)], dim=0)
        context_y = torch.cat([context_y, y_t.unsqueeze(0)], dim=0)

        n_remaining = int(query_x.shape[0])
        keep_mask = torch.arange(n_remaining, device=device) != action_idx
        query_x = query_x[keep_mask]
        query_y = query_y[keep_mask]
        if query_x.shape[0] == 0:
            break

    return x_traj, y_traj


def _load_checkpoint_payload(
    path_like: str | Path,
    *,
    map_location: str = "cpu",
) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict checkpoint payload from {path}.")
    return payload


def _coerce_location_theta(
    theta: Any,
    *,
    expected_k: int,
    expected_dim: int,
) -> Tensor:
    theta_tensor = torch.as_tensor(theta, dtype=torch.float32)
    if theta_tensor.ndim != 2 or theta_tensor.shape != (expected_k, expected_dim):
        raise ValueError(
            f"theta must have shape [{expected_k}, {expected_dim}], got {tuple(theta_tensor.shape)}."
        )
    return theta_tensor


def _resolve_policy_arch(payload: dict[str, Any]) -> dict[str, Any]:
    train_cfg = payload.get("config", {})
    train_section = train_cfg.get("train", {}) if isinstance(train_cfg, dict) else {}
    policy_head_cfg = (
        train_section.get("policy_head", {}) if isinstance(train_section, dict) else {}
    )
    if not isinstance(policy_head_cfg, dict):
        policy_head_cfg = {}
    return {
        "hidden_dim": int(policy_head_cfg.get("hidden_dim", 128)),
        "num_layers": int(policy_head_cfg.get("num_layers", 2)),
        "activation": str(policy_head_cfg.get("activation", "gelu")),
        "dropout": float(policy_head_cfg.get("dropout", 0.0)),
        "use_remaining_budget": bool(policy_head_cfg.get("use_remaining_budget", False)),
        "budget_scale": float(
            policy_head_cfg.get(
                "budget_scale",
                train_section.get("num_acquisitions", 1.0)
                if isinstance(train_section, dict) else 1.0,
            )
        ),
    }


def load_location_policy_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    seed: int = 42,
    n_query_eval: int | None = None,
    strict_backbone_state: bool = True,
) -> dict[str, Any]:
    """Load a 2-D location-finding BED policy checkpoint for notebook evaluation."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = _load_checkpoint_payload(checkpoint_path)

    train_cfg = payload.get("config")
    if not isinstance(train_cfg, dict):
        raise TypeError("Checkpoint payload is missing a resolved training config dict.")

    task_cfg = train_cfg.get("task")
    model_cfg = train_cfg.get("model")
    train_section = train_cfg.get("train", {})
    if not isinstance(task_cfg, dict) or not isinstance(model_cfg, dict):
        raise KeyError("Checkpoint config must contain `task` and `model` sections.")
    if str(task_cfg.get("name")) != "location_finding":
        raise ValueError("This helper only supports the location_finding BED task.")
    if int(task_cfg.get("dim_x", -1)) != 2:
        raise ValueError("This helper currently supports only 2-D location_finding checkpoints.")

    task_cfg_omega = OmegaConf.create(task_cfg)
    if n_query_eval is not None:
        task_cfg_omega.n_query = int(n_query_eval)
    task = create_bed_task(task_cfg_omega, seed=int(seed))
    if not isinstance(task, LocationFindingSampler):
        raise TypeError("Expected create_bed_task to return a LocationFindingSampler.")

    backbone_checkpoint_path = model_cfg.get("checkpoint_path")
    payload_checkpoint_path = payload.get("backbone_checkpoint_path")
    if (
        isinstance(payload_checkpoint_path, str)
        and Path(payload_checkpoint_path).expanduser().exists()
    ):
        backbone_checkpoint_path = payload_checkpoint_path

    checkpoint_kind = (
        str(model_cfg.get("checkpoint_kind", "auto")).strip().lower()
        if model_cfg.get("checkpoint_kind") is not None
        else "auto"
    )
    if checkpoint_kind == "auto":
        checkpoint_kind = (
            "auto"
            if backbone_checkpoint_path is not None
            else bed_task_checkpoint_kind(task)
        )

    backbone = create_backbone(
        name=str(model_cfg.get("name", "tabicl")),
        checkpoint_path=backbone_checkpoint_path,
        version=model_cfg.get("version"),
        checkpoint_kind=checkpoint_kind,
        num_classes=None,
        device=device,
        checkpoints_root=REPO_ROOT / "checkpoints",
        cache_trainset_representation=bool(model_cfg.get("cache_trainset_representation", False)),
        strict=bool(model_cfg.get("strict_load", True)),
    )

    arch = _resolve_policy_arch(payload)
    policy_head = CandidatePolicyHead(
        input_dim=int(getattr(backbone, "embedding_dim")),
        hidden_dim=arch["hidden_dim"],
        num_layers=arch["num_layers"],
        activation=arch["activation"],
        dropout=arch["dropout"],
        use_remaining_budget=arch["use_remaining_budget"],
        budget_scale=arch["budget_scale"],
    ).to(backbone.device)

    policy_state = payload.get("policy_head_state_dict")
    if policy_state is None:
        raise KeyError("Checkpoint missing 'policy_head_state_dict'.")
    policy_head.load_state_dict(policy_state, strict=True)
    policy_head.eval()

    backbone_state = payload.get("backbone_model_state_dict")
    if backbone_state is not None:
        backbone.model.load_state_dict(
            backbone_state,
            strict=bool(strict_backbone_state),
        )
    backbone.model.eval()

    num_acquisitions = 30
    if isinstance(train_section, dict) and train_section.get("num_acquisitions") is not None:
        num_acquisitions = int(train_section["num_acquisitions"])

    return {
        "checkpoint_path": checkpoint_path,
        "payload": payload,
        "train_cfg": train_cfg,
        "task": task,
        "backbone": backbone,
        "policy_head": policy_head,
        "num_acquisitions": num_acquisitions,
    }


@torch.no_grad()
def rollout_bed_random(
    *,
    episode: BEDEpisode,
    num_acquisitions: int,
) -> tuple[list[Tensor], list[Tensor]]:
    """Random acquisition rollout on one BED episode."""

    query_x = episode.query_x.clone()
    query_y = episode.query_y.clone()

    max_steps = min(num_acquisitions, int(query_x.shape[0]))
    x_traj: list[Tensor] = []
    y_traj: list[Tensor] = []

    for _ in range(max_steps):
        idx = torch.randint(int(query_x.shape[0]), (1,)).item()
        x_traj.append(query_x[idx])
        y_traj.append(query_y[idx])

        keep_mask = torch.arange(int(query_x.shape[0])) != idx
        query_x = query_x[keep_mask]
        query_y = query_y[keep_mask]
        if query_x.shape[0] == 0:
            break

    return x_traj, y_traj


def compute_bed_eig_bounds_for_trajectory(
    *,
    context_x: Tensor | None,
    context_y: Tensor | None,
    x_traj: list[Tensor],
    y_traj: list[Tensor],
    theta_true: Tensor,
    task: BEDTask,
    n_contrastive_eval: int,
    device: torch.device,
) -> tuple[float, float]:
    """Compute full-trajectory sPCE/sNMC bounds for one rollout."""

    if not x_traj:
        return 0.0, 0.0

    theta_eval = task.sample_theta(n_contrastive_eval)
    theta_all = torch.cat([theta_true.unsqueeze(0), theta_eval], dim=0).unsqueeze(1).to(device)
    x_segments = []
    y_segments = []
    if (context_x is None) != (context_y is None):
        raise ValueError("context_x and context_y must be both provided or both omitted.")
    if context_x is not None and context_y is not None:
        if int(context_x.shape[0]) != int(context_y.shape[0]):
            raise ValueError(
                "context_x and context_y must have the same number of rows, "
                f"got {int(context_x.shape[0])} and {int(context_y.shape[0])}."
            )
        if int(context_x.shape[0]) > 0:
            x_segments.append(context_x.to(device))
            y_segments.append(context_y.to(device))

    x_segments.append(torch.stack(x_traj).to(device))
    y_segments.append(torch.stack(y_traj).to(device))
    x_trajectory = torch.cat(x_segments, dim=0).unsqueeze(0)
    y_trajectory = torch.cat(y_segments, dim=0).unsqueeze(0)
    spce, snmc = compute_eig_bounds_from_trajectory(
        task.log_likelihood,
        y_trajectory,
        x_trajectory,
        theta_all,
    )
    return float(spce.item()), float(snmc.item())


def _aggregate_metric(values: list[float], prefix: str) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    n = int(tensor.numel())
    mean = float(tensor.mean().item()) if n > 0 else 0.0
    std = float(tensor.std(unbiased=False).item()) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else 0.0
    ci95 = 1.96 * sem
    return {
        f"mean_{prefix}": mean,
        f"std_{prefix}": std,
        f"sem_{prefix}": sem,
        f"ci95_{prefix}": ci95,
    }


def aggregate_eig_bounds(results: list[EpisodeResult]) -> dict[str, float]:
    """Aggregate sPCE/sNMC statistics across evaluation episodes."""

    n = len(results)
    aggregated = {"n_episodes": float(n)}
    aggregated.update(_aggregate_metric([result.spce for result in results], "spce"))
    aggregated.update(_aggregate_metric([result.snmc for result in results], "snmc"))
    return aggregated


@torch.no_grad()
def evaluate_bed_policy(
    *,
    backbone: PolicyBackbone,
    policy_head: CandidatePolicyHead,
    task: BEDTask,
    num_acquisitions: int,
    n_eval_episodes: int,
    n_contrastive_eval: int,
    n_query_eval: int | None = None,
    greedy: bool = True,
    plot_dir: Path | None = None,
    n_plot_episodes: int = 6,
) -> dict[str, float]:
    """Evaluate a BED policy with task-specific analytical likelihoods."""

    policy_head.eval()
    was_training = backbone.model.training
    backbone.model.eval()

    plotter = task.trajectory_plotter
    should_plot = plot_dir is not None and plotter is not None
    spce_values: list[float] = []
    snmc_values: list[float] = []
    plot_data: list[dict[str, Any]] = []

    original_n_query = int(task.n_query)
    if n_query_eval is not None:
        task.n_query = int(n_query_eval)

    try:
        for _ in range(n_eval_episodes):
            episode = task.sample_episode()
            theta_true = episode.theta_all[0]
            x_traj, y_traj = rollout_bed_policy(
                backbone=backbone,
                policy_head=policy_head,
                episode=episode,
                num_acquisitions=num_acquisitions,
                greedy=greedy,
            )
            if not x_traj:
                continue

            spce_val, snmc_val = compute_bed_eig_bounds_for_trajectory(
                context_x=episode.context_x,
                context_y=episode.context_y,
                x_traj=x_traj,
                y_traj=y_traj,
                theta_true=theta_true,
                task=task,
                n_contrastive_eval=n_contrastive_eval,
                device=backbone.device,
            )
            spce_values.append(spce_val)
            snmc_values.append(snmc_val)

            if should_plot and len(plot_data) < n_plot_episodes:
                plot_data.append(
                    {
                        "theta_true": theta_true.cpu().numpy(),
                        "x_traj": torch.stack(x_traj).cpu().numpy(),
                        "spce": spce_val,
                    }
                )
    finally:
        task.n_query = original_n_query
        if was_training:
            backbone.model.train()
        policy_head.train()

    if not spce_values:
        return {
            "eval/spce_mean": 0.0,
            "eval/spce_stderr": 0.0,
            "eval/snmc_mean": 0.0,
            "eval/snmc_stderr": 0.0,
            "eval/n_episodes": 0.0,
        }

    spce_tensor = torch.tensor(spce_values)
    snmc_tensor = torch.tensor(snmc_values)
    stderr_val = (
        float(spce_tensor.std(unbiased=False).item() / math.sqrt(len(spce_values)))
        if len(spce_values) > 1
        else 0.0
    )
    snmc_stderr_val = (
        float(snmc_tensor.std(unbiased=False).item() / math.sqrt(len(snmc_values)))
        if len(snmc_values) > 1
        else 0.0
    )

    if should_plot and plot_data and plotter is not None:
        plotter(plot_dir, plot_data)

    return {
        "eval/spce_mean": float(spce_tensor.mean().item()),
        "eval/spce_stderr": stderr_val,
        "eval/snmc_mean": float(snmc_tensor.mean().item()),
        "eval/snmc_stderr": snmc_stderr_val,
        "eval/n_episodes": float(len(spce_values)),
    }


def rollout_location_trajectory_from_checkpoint(
    checkpoint_path: str | Path,
    theta: Any,
    *,
    device: str = "cpu",
    seed: int = 42,
    n_query_eval: int | None = None,
    num_acquisitions: int | None = None,
    greedy: bool = True,
    n_contrastive_eval: int | None = None,
    all_x: Any | None = None,
) -> dict[str, Any]:
    """Roll out a 2-D location-finding checkpoint on a fixed theta."""

    loaded = load_location_policy_checkpoint(
        checkpoint_path,
        device=device,
        seed=seed,
        n_query_eval=n_query_eval,
    )
    task = loaded["task"]
    theta_true = _coerce_location_theta(theta, expected_k=int(task.K), expected_dim=int(task.dim_x))

    if all_x is None:
        n_total = int(task.n_context_init + task.n_query + task.n_target)
        all_x_tensor = task._sample_designs(n_total)
    else:
        all_x_tensor = torch.as_tensor(all_x, dtype=torch.float32)
        expected_rows = int(task.n_context_init + task.n_query + task.n_target)
        if all_x_tensor.ndim != 2 or all_x_tensor.shape != (expected_rows, int(task.dim_x)):
            raise ValueError(
                "all_x must have shape "
                f"[{expected_rows}, {int(task.dim_x)}], got {tuple(all_x_tensor.shape)}."
            )

    contrastive_theta_tensor = task.sample_theta(int(task.n_contrastive))
    theta_all = torch.cat([theta_true.unsqueeze(0), contrastive_theta_tensor], dim=0)
    all_y = task._observe(all_x_tensor, theta_true)

    context_end = int(task.n_context_init)
    query_end = context_end + int(task.n_query)
    target_x = all_x_tensor[query_end:] if int(task.n_target) > 0 else None
    target_y = all_y[query_end:] if int(task.n_target) > 0 else None
    episode = BEDEpisode(
        theta_all=theta_all,
        context_x=all_x_tensor[:context_end],
        context_y=all_y[:context_end],
        query_x=all_x_tensor[context_end:query_end],
        query_y=all_y[context_end:query_end],
        target_x=target_x,
        target_y=target_y,
    )

    resolved_num_acquisitions = (
        int(num_acquisitions)
        if num_acquisitions is not None
        else int(loaded["num_acquisitions"])
    )
    x_traj, y_traj = rollout_bed_policy(
        backbone=loaded["backbone"],
        policy_head=loaded["policy_head"],
        episode=episode,
        num_acquisitions=resolved_num_acquisitions,
        greedy=bool(greedy),
    )
    if not x_traj:
        raise ValueError("Collected an empty trajectory.")

    x_traj_tensor = torch.stack(x_traj).cpu()
    y_traj_tensor = torch.stack(y_traj).cpu()

    spce: float | None = None
    snmc: float | None = None
    if n_contrastive_eval is not None:
        spce, snmc = compute_bed_eig_bounds_for_trajectory(
            context_x=episode.context_x,
            context_y=episode.context_y,
            x_traj=x_traj,
            y_traj=y_traj,
            theta_true=episode.theta_all[0],
            task=task,
            n_contrastive_eval=int(n_contrastive_eval),
            device=loaded["backbone"].device,
        )

    return {
        "x_traj": x_traj_tensor,
        "y_traj": y_traj_tensor,
        "theta_true": episode.theta_all[0].cpu(),
        "episode": episode,
        "task": task,
        "spce": spce,
        "snmc": snmc,
        "design_min": float(task.design_min),
        "design_max": float(task.design_max),
    }


__all__ = [
    "EpisodeResult",
    "aggregate_eig_bounds",
    "compute_bed_eig_bounds_for_trajectory",
    "evaluate_bed_policy",
    "load_location_policy_checkpoint",
    "rollout_location_trajectory_from_checkpoint",
    "rollout_bed_policy",
    "rollout_bed_random",
]
