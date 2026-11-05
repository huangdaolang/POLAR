#!/usr/bin/env python
"""Policy training runner (REINFORCE + dense sPCE) for BED tasks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from polar.backbones import create_backbone
from polar.evaluation import evaluate_bed_policy
from polar.tasks import create_bed_task
from polar.tasks.bed import bed_task_checkpoint_kind
from polar.trainers import (
    BEDPolicyTrainer,
    CandidatePolicyHead,
    build_lr_scheduler,
    build_policy_optimizer,
)
from polar.utils.run_utils import (
    build_run_dirs,
    build_run_tag,
    init_wandb,
    log_wandb_metrics,
    save_checkpoint,
)
from polar.utils.seeding import set_global_seed


def _log_wandb_epoch(
    *,
    wandb_run: Any | None,
    epoch: int,
    epoch_metrics: dict[str, float],
    optimizer: torch.optim.Optimizer,
) -> None:
    # `optimizer` is accepted for signature parity with other train scripts;
    # BED uses its own `train/lr_*` keys sourced from epoch_metrics instead
    # of the `lr/group_N` convention used elsewhere.
    del optimizer
    if wandb_run is None:
        return

    payload: dict[str, float] = {
        "epoch": float(epoch),
        "train/rollouts_per_epoch": float(epoch_metrics["rollouts_per_epoch"]),
        "train/total_rollouts": float(epoch_metrics["total_rollouts"]),
        "train/acquisition_evals_this_epoch": float(epoch_metrics["acquisition_evals_this_epoch"]),
        "train/target_evals_this_epoch": float(epoch_metrics["target_evals_this_epoch"]),
        "train/likelihood_evals_this_epoch": float(epoch_metrics["likelihood_evals_this_epoch"]),
        "train/total_likelihood_evals": float(epoch_metrics["total_likelihood_evals"]),
        "train/steps": float(epoch_metrics["steps"]),
        "train/total_loss": float(epoch_metrics["total_loss"]),
        "train/nll_loss": float(epoch_metrics["nll_loss"]),
        "train/policy_loss": float(epoch_metrics["policy_loss"]),
        "train/reward_mean": float(epoch_metrics["reward_mean"]),
        "train/spce_estimate": float(epoch_metrics["spce_estimate"]),
        "train/target_log_prob_start": float(epoch_metrics["target_log_prob_start"]),
        "train/target_log_prob_end": float(epoch_metrics["target_log_prob_end"]),
        "train/policy_entropy": float(epoch_metrics["policy_entropy"]),
    }
    if "lr_policy" in epoch_metrics:
        payload["train/lr_policy"] = float(epoch_metrics["lr_policy"])
    if "lr_backbone" in epoch_metrics:
        payload["train/lr_backbone"] = float(epoch_metrics["lr_backbone"])
    log_wandb_metrics(wandb_run, step=epoch, payload=payload)


def _log_wandb_eval(
    *,
    wandb_run: Any | None,
    epoch: int,
    eval_metrics: dict[str, float],
) -> None:
    if wandb_run is None:
        return
    payload = {k: float(v) for k, v in eval_metrics.items()}
    log_wandb_metrics(wandb_run, step=epoch, payload=payload)


def _parse_likelihood_eval_milestones(cfg: DictConfig) -> list[int]:
    heavy_eval_cfg = cfg.train.get("heavy_eval")
    if heavy_eval_cfg is None or not bool(heavy_eval_cfg.get("enabled", False)):
        return []

    milestones = heavy_eval_cfg.get("likelihood_eval_milestones")
    if milestones is None:
        return []

    parsed = sorted({int(value) for value in milestones})
    if any(value <= 0 for value in parsed):
        raise ValueError(
            "train.heavy_eval.likelihood_eval_milestones must all be positive integers."
        )
    return parsed


def _count_target_evals_per_episode(
    *,
    episodes: list[Any],
    backbone_trainable: bool,
    nll_weight: float,
) -> int:
    if not episodes or not backbone_trainable or nll_weight <= 0.0:
        return 0

    has_targets = all(ep.target_x is not None and ep.target_y is not None for ep in episodes)
    if not has_targets:
        return 0

    first_target_x = episodes[0].target_x
    if first_target_x is None:
        return 0
    return int(first_target_x.shape[0])


def _count_epoch_likelihood_evals(
    *,
    episodes: list[Any],
    steps_per_episode: int,
    backbone_trainable: bool,
    nll_weight: float,
) -> dict[str, int]:
    batch_size = len(episodes)
    target_evals_per_episode = _count_target_evals_per_episode(
        episodes=episodes,
        backbone_trainable=backbone_trainable,
        nll_weight=nll_weight,
    )
    acquisition_evals_this_epoch = int(batch_size * steps_per_episode)
    target_evals_this_epoch = int(batch_size * target_evals_per_episode)
    likelihood_evals_this_epoch = acquisition_evals_this_epoch + target_evals_this_epoch
    return {
        "acquisition_evals_this_epoch": acquisition_evals_this_epoch,
        "target_evals_per_episode": target_evals_per_episode,
        "target_evals_this_epoch": target_evals_this_epoch,
        "likelihood_evals_this_epoch": likelihood_evals_this_epoch,
    }


def _compute_likelihood_eval_schedule(
    *,
    batch_size: int,
    num_epochs: int,
    num_acquisitions: int,
    n_query: int,
    n_target: int,
    finetune_epochs: int,
    nll_weight: float,
    likelihood_eval_milestones: list[int],
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("train.batch_size must be positive.")
    if num_epochs <= 0:
        raise ValueError("train.num_epochs must be positive.")
    if num_acquisitions <= 0:
        raise ValueError("train.num_acquisitions must be positive.")
    if n_query <= 0:
        raise ValueError("task.n_query must be positive.")
    if n_target < 0:
        raise ValueError("task.n_target must be non-negative.")

    rollouts_per_epoch = int(batch_size)
    total_training_rollouts = int(rollouts_per_epoch * num_epochs)
    acquisition_evals_per_epoch = int(batch_size * min(num_acquisitions, n_query))
    target_evals_per_epoch = (
        int(batch_size * n_target)
        if finetune_epochs > 0 and nll_weight > 0.0 and n_target > 0
        else 0
    )
    active_target_eval_epochs = min(num_epochs, max(finetune_epochs, 0))

    total_likelihood_evals = 0
    cumulative_likelihood_evals: list[int] = []
    for epoch in range(num_epochs):
        total_likelihood_evals += acquisition_evals_per_epoch
        if target_evals_per_epoch > 0 and epoch < active_target_eval_epochs:
            total_likelihood_evals += target_evals_per_epoch
        cumulative_likelihood_evals.append(total_likelihood_evals)

    reached_totals = set(cumulative_likelihood_evals)
    non_exact_milestones = [
        milestone
        for milestone in likelihood_eval_milestones
        if milestone not in reached_totals
    ]
    unreachable_milestones = [
        milestone
        for milestone in likelihood_eval_milestones
        if milestone > total_likelihood_evals
    ]
    return {
        "rollouts_per_epoch": rollouts_per_epoch,
        "total_training_rollouts": total_training_rollouts,
        "acquisition_evals_per_epoch": acquisition_evals_per_epoch,
        "target_evals_per_epoch": target_evals_per_epoch,
        "total_training_likelihood_evals": total_likelihood_evals,
        "non_exact_milestones": non_exact_milestones,
        "unreachable_milestones": unreachable_milestones,
    }


def _resolve_reached_likelihood_eval_milestones(
    *,
    likelihood_eval_milestones: list[int],
    completed_milestones: set[int],
    current_likelihood_evals: int,
) -> list[int]:
    return [
        milestone
        for milestone in likelihood_eval_milestones
        if milestone not in completed_milestones and current_likelihood_evals >= milestone
    ]


def _save_heavy_eval_results(
    *,
    json_path: Path,
    csv_path: Path,
    results: list[dict[str, Any]],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "epoch",
        "requested_likelihood_evals",
        "total_likelihood_evals",
        "likelihood_evals_this_epoch",
        "acquisition_evals_this_epoch",
        "target_evals_this_epoch",
        "target_evals_per_episode",
        "total_rollouts",
        "rollouts_per_epoch",
        "checkpoint_path",
        "eval_spce_mean",
        "eval_spce_stderr",
        "eval_snmc_mean",
        "eval_snmc_stderr",
        "eval_n_episodes",
        "eval_n_contrastive_eval",
        "eval_n_query_eval",
        "eval_greedy",
        "plot_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fieldnames})


def _resolve_bed_backbone_options(
    cfg: DictConfig,
    task,
) -> tuple[str, int | None]:
    requested_kind = str(cfg.model.get("checkpoint_kind", "auto")).strip().lower()
    if requested_kind not in {"auto", "regression", "classification"}:
        raise ValueError(
            "model.checkpoint_kind must be one of: auto, regression, classification."
        )

    explicit_checkpoint_path = cfg.model.get("checkpoint_path")
    if requested_kind == "auto":
        checkpoint_kind = (
            "auto"
            if explicit_checkpoint_path is not None
            else bed_task_checkpoint_kind(task)
        )
    else:
        checkpoint_kind = requested_kind

    configured_num_classes = cfg.model.get("num_classes")
    if configured_num_classes is not None:
        num_classes = int(configured_num_classes)
    else:
        raw_num_classes = getattr(task, "num_classes", None)
        num_classes = int(raw_num_classes) if raw_num_classes is not None else None

    if checkpoint_kind == "classification" and num_classes is None:
        raise ValueError(
            "Classification BED backbone requires task.num_classes or model.num_classes."
        )
    if checkpoint_kind == "regression":
        num_classes = None
    return checkpoint_kind, num_classes


def _run_policy_eval(
    *,
    cfg: DictConfig,
    backbone,
    policy_head: CandidatePolicyHead,
    num_acquisitions: int,
    eval_cfg: DictConfig,
    plot_path: Path | None,
    eval_seed: int,
) -> tuple[dict[str, float], int]:
    eval_task = create_bed_task(cfg.task, seed=eval_seed)
    n_query_eval_raw = eval_cfg.get("n_query_eval")
    n_query_eval = int(n_query_eval_raw) if n_query_eval_raw is not None else None
    resolved_n_query_eval = int(n_query_eval) if n_query_eval is not None else int(eval_task.n_query)
    eval_metrics = evaluate_bed_policy(
        backbone=backbone,
        policy_head=policy_head,
        task=eval_task,
        num_acquisitions=num_acquisitions,
        n_eval_episodes=int(eval_cfg.n_eval_episodes),
        n_contrastive_eval=int(eval_cfg.n_contrastive_eval),
        n_query_eval=n_query_eval,
        greedy=bool(eval_cfg.greedy),
        plot_dir=plot_path,
        n_plot_episodes=int(eval_cfg.get("n_plot_episodes", 0)),
    )
    return eval_metrics, resolved_n_query_eval


@hydra.main(version_base="1.3", config_path="../config", config_name="exp/train_bed_policy")
def main(cfg: DictConfig) -> None:
    print(f"==== BED Policy Training ({cfg.task.name}) ====")
    print(OmegaConf.to_yaml(cfg))

    set_global_seed(int(cfg.seed))

    run_tag = build_run_tag(cfg)
    run_dir, checkpoint_dir = build_run_dirs(cfg, run_tag)
    print(f"run_dir: {run_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")

    task = create_bed_task(cfg.task, seed=int(cfg.seed))
    checkpoint_kind, num_classes = _resolve_bed_backbone_options(cfg, task)

    backbone = create_backbone(
        name=cfg.model.name,
        checkpoint_path=cfg.model.checkpoint_path,
        version=cfg.model.version,
        checkpoint_kind=checkpoint_kind,
        num_classes=num_classes,
        device=cfg.device,
        checkpoints_root=cfg.paths.checkpoints_root,
        cache_trainset_representation=cfg.model.cache_trainset_representation,
        strict=cfg.model.strict_load,
    )
    print(f"loaded_backbone: {type(backbone).__name__}")
    print(f"checkpoint_path: {backbone.checkpoint_path}")
    print(f"checkpoint_kind: {getattr(backbone, 'checkpoint_kind', checkpoint_kind)}")

    embedding_dim = int(getattr(backbone, "embedding_dim"))
    policy_head = CandidatePolicyHead(
        input_dim=embedding_dim,
        hidden_dim=int(cfg.train.policy_head.hidden_dim),
        num_layers=int(cfg.train.policy_head.num_layers),
        activation=str(cfg.train.policy_head.activation),
        dropout=float(cfg.train.policy_head.dropout),
        use_remaining_budget=bool(cfg.train.policy_head.get("use_remaining_budget", False)),
        budget_scale=float(cfg.train.policy_head.get("budget_scale", cfg.train.num_acquisitions)),
    ).to(backbone.device)

    finetune_epochs = int(cfg.train.finetune_epochs)
    optimizer = build_policy_optimizer(
        backbone=backbone,
        policy_head=policy_head,
        finetune_backbone=finetune_epochs > 0,
        lr_policy=float(cfg.train.optim.lr_policy),
        lr_backbone=float(cfg.train.optim.lr_backbone),
        weight_decay_policy=float(cfg.train.optim.weight_decay_policy),
        weight_decay_backbone=float(cfg.train.optim.weight_decay_backbone),
    )

    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        num_epochs=int(cfg.train.num_epochs),
        warmup_epochs=int(cfg.train.optim.get("warmup_epochs", 0)),
        min_lr_ratio=float(cfg.train.optim.get("min_lr_ratio", 0.0)),
    )

    log_lik_fn = task.log_likelihood
    print(f"task: {task.name} (dim_x={task.dim_x}, n_query={task.n_query})")

    trainer = BEDPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=finetune_epochs,
        log_likelihood_fn=log_lik_fn,
        nll_weight=float(cfg.train.loss.nll_weight),
        policy_weight=float(cfg.train.loss.policy_weight),
        gamma=float(cfg.train.loss.gamma),
        use_baseline=bool(cfg.train.loss.use_baseline),
        baseline_momentum=float(cfg.train.loss.baseline_momentum),
        normalize_rewards=bool(cfg.train.loss.normalize_rewards),
        detach_policy_features=bool(cfg.train.loss.detach_policy_features),
        grad_clip_norm=(
            float(cfg.train.optim.grad_clip_norm)
            if cfg.train.optim.grad_clip_norm is not None
            else None
        ),
        use_amp=bool(cfg.train.optim.get("use_amp", True)),
        gradient_checkpointing=bool(cfg.train.optim.get("gradient_checkpointing", False)),
    )

    num_epochs = int(cfg.train.num_epochs)
    batch_size = int(cfg.train.batch_size)
    num_acquisitions = int(cfg.train.num_acquisitions)
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every)
    eval_interval = int(cfg.train.eval.eval_interval)
    likelihood_eval_milestones = _parse_likelihood_eval_milestones(cfg)
    rollout_schedule = _compute_likelihood_eval_schedule(
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_acquisitions=num_acquisitions,
        n_query=int(task.n_query),
        n_target=int(task.n_target),
        finetune_epochs=finetune_epochs,
        nll_weight=float(cfg.train.loss.nll_weight),
        likelihood_eval_milestones=likelihood_eval_milestones,
    )
    rollouts_per_epoch = int(rollout_schedule["rollouts_per_epoch"])
    total_training_rollouts = int(rollout_schedule["total_training_rollouts"])
    acquisition_evals_per_epoch = int(rollout_schedule["acquisition_evals_per_epoch"])
    target_evals_per_epoch = int(rollout_schedule["target_evals_per_epoch"])
    total_training_likelihood_evals = int(rollout_schedule["total_training_likelihood_evals"])
    non_exact_milestones = list(rollout_schedule["non_exact_milestones"])
    unreachable_milestones = list(rollout_schedule["unreachable_milestones"])
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    include_backbone_state = finetune_epochs > 0
    heavy_eval_json_path = run_dir / "heavy_eval_metrics.json"
    heavy_eval_csv_path = run_dir / "heavy_eval_metrics.csv"

    print(f"rollouts_per_epoch: {rollouts_per_epoch}")
    print(f"total_training_rollouts: {total_training_rollouts}")
    print(f"acquisition_evals_per_epoch: {acquisition_evals_per_epoch}")
    if target_evals_per_epoch > 0:
        print(f"target_evals_per_epoch_during_prediction_loss: {target_evals_per_epoch}")
    print(f"total_training_likelihood_evals: {total_training_likelihood_evals}")
    if non_exact_milestones:
        print(
            "heavy_eval non-exact likelihood-eval milestones:"
            f" {non_exact_milestones} (trigger on first reached epoch)."
        )
    if unreachable_milestones:
        print(
            "warning: unreachable heavy_eval likelihood-eval milestones:"
            f" {unreachable_milestones} (exceed total_training_likelihood_evals)."
        )

    wandb_run = init_wandb(cfg, run_dir, run_tag)
    try:
        all_metrics: list[dict[str, float]] = []
        heavy_eval_results: list[dict[str, Any]] = []
        completed_heavy_eval: set[int] = set()
        current_rollouts = 0
        current_likelihood_evals = 0
        for epoch in range(num_epochs):
            if finetune_epochs > 0 and epoch == finetune_epochs:
                trainer.freeze_backbone()
                print(f"[epoch={epoch:04d}] backbone frozen (finetune_epochs={finetune_epochs})")
            episodes = task.sample_batch(batch_size=batch_size)
            epoch_metrics = trainer.train_one_epoch(
                episodes=episodes,
                num_acquisitions=num_acquisitions,
            )
            if scheduler is not None:
                scheduler.step()

            # Record current learning rates.
            current_lrs = [pg["lr"] for pg in optimizer.param_groups]
            epoch_metrics["lr_policy"] = float(current_lrs[0])
            if len(current_lrs) > 1:
                epoch_metrics["lr_backbone"] = float(current_lrs[1])

            epoch_budget = _count_epoch_likelihood_evals(
                episodes=episodes,
                steps_per_episode=int(epoch_metrics["steps"]),
                backbone_trainable=bool(getattr(trainer, "_backbone_trainable", False)),
                nll_weight=float(cfg.train.loss.nll_weight),
            )
            current_rollouts += int(len(episodes))
            current_likelihood_evals += int(epoch_budget["likelihood_evals_this_epoch"])
            epoch_metrics["epoch"] = float(epoch)
            epoch_metrics["rollouts_per_epoch"] = float(rollouts_per_epoch)
            epoch_metrics["total_rollouts"] = float(current_rollouts)
            epoch_metrics["acquisition_evals_this_epoch"] = float(
                epoch_budget["acquisition_evals_this_epoch"]
            )
            epoch_metrics["target_evals_per_episode"] = float(
                epoch_budget["target_evals_per_episode"]
            )
            epoch_metrics["target_evals_this_epoch"] = float(epoch_budget["target_evals_this_epoch"])
            epoch_metrics["likelihood_evals_this_epoch"] = float(
                epoch_budget["likelihood_evals_this_epoch"]
            )
            epoch_metrics["total_likelihood_evals"] = float(current_likelihood_evals)
            all_metrics.append(epoch_metrics)
            _log_wandb_epoch(
                wandb_run=wandb_run,
                epoch=epoch,
                epoch_metrics=epoch_metrics,
                optimizer=optimizer,
            )

            if epoch % log_every == 0 or epoch == num_epochs - 1:
                lr_str = f"lr_p={epoch_metrics.get('lr_policy', 0):.2e}"
                if "lr_backbone" in epoch_metrics:
                    lr_str += f" lr_b={epoch_metrics['lr_backbone']:.2e}"
                print(
                    f"[epoch={epoch:04d}]",
                    f"rollouts={int(epoch_metrics['total_rollouts'])}",
                    f"likelihood_evals={int(epoch_metrics['total_likelihood_evals'])}",
                    f"steps={epoch_metrics['steps']:.0f}",
                    f"total_loss={epoch_metrics['total_loss']:.6f}",
                    f"nll={epoch_metrics['nll_loss']:.6f}",
                    f"policy={epoch_metrics['policy_loss']:.6f}",
                    f"reward={epoch_metrics['reward_mean']:.6f}",
                    f"spce={epoch_metrics['spce_estimate']:.4f}",
                    f"entropy={epoch_metrics['policy_entropy']:.4f}",
                    lr_str,
                )

            if eval_interval > 0 and (
                (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1
            ):
                n_plot = int(cfg.train.eval.get("n_plot_episodes", 6))
                plot_path: Path | None = None
                if n_plot > 0:
                    plot_path = run_dir / "plots" / f"trajectories_epoch{epoch + 1:05d}.png"
                eval_metrics, _ = _run_policy_eval(
                    cfg=cfg,
                    backbone=backbone,
                    policy_head=policy_head,
                    num_acquisitions=num_acquisitions,
                    eval_cfg=cfg.train.eval,
                    plot_path=plot_path,
                    eval_seed=int(cfg.seed) + 1_000_000 + epoch + 1,
                )
                epoch_metrics.update(eval_metrics)
                print(
                    f"  [eval] spce_mean={eval_metrics['eval/spce_mean']:.4f}"
                    f" +/- {eval_metrics['eval/spce_stderr']:.4f}"
                    f" snmc_mean={eval_metrics['eval/snmc_mean']:.4f}"
                    f" +/- {eval_metrics['eval/snmc_stderr']:.4f}"
                    f" (n={int(eval_metrics['eval/n_episodes'])})"
                )
                _log_wandb_eval(
                    wandb_run=wandb_run,
                    epoch=epoch,
                    eval_metrics=eval_metrics,
                )

            reached_likelihood_eval_milestones = _resolve_reached_likelihood_eval_milestones(
                likelihood_eval_milestones=likelihood_eval_milestones,
                completed_milestones=completed_heavy_eval,
                current_likelihood_evals=current_likelihood_evals,
            )
            if reached_likelihood_eval_milestones:
                heavy_eval_cfg = cfg.train.heavy_eval
                heavy_eval_n_plot = int(heavy_eval_cfg.get("n_plot_episodes", 0))
                heavy_eval_plot_path: Path | None = None
                if heavy_eval_n_plot > 0:
                    heavy_eval_plot_path = (
                        run_dir
                        / "plots"
                        / f"heavy_eval_likelihood_evals_{current_likelihood_evals:09d}.png"
                    )
                heavy_eval_metrics, heavy_eval_n_query = _run_policy_eval(
                    cfg=cfg,
                    backbone=backbone,
                    policy_head=policy_head,
                    num_acquisitions=num_acquisitions,
                    eval_cfg=heavy_eval_cfg,
                    plot_path=heavy_eval_plot_path,
                    eval_seed=int(cfg.seed) + current_likelihood_evals,
                )
                heavy_eval_log = {
                    "heavy_eval/total_rollouts": float(current_rollouts),
                    "heavy_eval/total_likelihood_evals": float(current_likelihood_evals),
                    "heavy_eval/spce_mean": float(heavy_eval_metrics["eval/spce_mean"]),
                    "heavy_eval/spce_stderr": float(heavy_eval_metrics["eval/spce_stderr"]),
                    "heavy_eval/snmc_mean": float(heavy_eval_metrics["eval/snmc_mean"]),
                    "heavy_eval/snmc_stderr": float(heavy_eval_metrics["eval/snmc_stderr"]),
                    "heavy_eval/n_episodes": float(heavy_eval_metrics["eval/n_episodes"]),
                }
                epoch_metrics.update(heavy_eval_log)
                _log_wandb_eval(
                    wandb_run=wandb_run,
                    epoch=epoch,
                    eval_metrics=heavy_eval_log,
                )

                milestone_checkpoint = (
                    checkpoint_dir
                    / f"checkpoint_likelihood_evals_{current_likelihood_evals}.pt"
                )
                saved_checkpoint_path: str | None = None
                if bool(heavy_eval_cfg.get("save_checkpoint", True)):
                    save_checkpoint(
                        path=milestone_checkpoint,
                        epoch=epoch,
                        cfg=cfg,
                        policy_head=policy_head,
                        optimizer=optimizer,
                        backbone_model_state_dict=(
                            backbone.model.state_dict() if include_backbone_state else None
                        ),
                        backbone_checkpoint_path=str(backbone.checkpoint_path),
                        metadata={
                            "phase": "heavy_eval",
                            "total_likelihood_evals": current_likelihood_evals,
                            "requested_likelihood_evals": reached_likelihood_eval_milestones,
                            "total_rollouts": current_rollouts,
                            "heavy_eval": heavy_eval_metrics,
                        },
                    )
                    saved_checkpoint_path = str(milestone_checkpoint.resolve())

                heavy_eval_record_base: dict[str, Any] = {
                    "epoch": epoch,
                    "total_likelihood_evals": current_likelihood_evals,
                    "likelihood_evals_this_epoch": int(epoch_budget["likelihood_evals_this_epoch"]),
                    "acquisition_evals_this_epoch": int(
                        epoch_budget["acquisition_evals_this_epoch"]
                    ),
                    "target_evals_this_epoch": int(epoch_budget["target_evals_this_epoch"]),
                    "target_evals_per_episode": int(epoch_budget["target_evals_per_episode"]),
                    "total_rollouts": current_rollouts,
                    "rollouts_per_epoch": rollouts_per_epoch,
                    "checkpoint_path": saved_checkpoint_path,
                    "eval_spce_mean": float(heavy_eval_metrics["eval/spce_mean"]),
                    "eval_spce_stderr": float(heavy_eval_metrics["eval/spce_stderr"]),
                    "eval_snmc_mean": float(heavy_eval_metrics["eval/snmc_mean"]),
                    "eval_snmc_stderr": float(heavy_eval_metrics["eval/snmc_stderr"]),
                    "eval_n_episodes": float(heavy_eval_metrics["eval/n_episodes"]),
                    "eval_n_contrastive_eval": int(heavy_eval_cfg.n_contrastive_eval),
                    "eval_n_query_eval": int(heavy_eval_n_query),
                    "eval_greedy": bool(heavy_eval_cfg.greedy),
                    "plot_path": (
                        str(heavy_eval_plot_path.resolve()) if heavy_eval_plot_path is not None else None
                    ),
                }
                for requested_likelihood_evals in reached_likelihood_eval_milestones:
                    heavy_eval_record = dict(heavy_eval_record_base)
                    heavy_eval_record["requested_likelihood_evals"] = int(
                        requested_likelihood_evals
                    )
                    heavy_eval_results.append(heavy_eval_record)
                    completed_heavy_eval.add(int(requested_likelihood_evals))
                _save_heavy_eval_results(
                    json_path=heavy_eval_json_path,
                    csv_path=heavy_eval_csv_path,
                    results=heavy_eval_results,
                )
                print(
                    "  [heavy eval]",
                    f"requested_likelihood_evals={reached_likelihood_eval_milestones}",
                    f"total_likelihood_evals={current_likelihood_evals}",
                    f"total_rollouts={current_rollouts}",
                    f"spce_mean={heavy_eval_metrics['eval/spce_mean']:.4f}",
                    f"+/- {heavy_eval_metrics['eval/spce_stderr']:.4f}",
                    f"snmc_mean={heavy_eval_metrics['eval/snmc_mean']:.4f}",
                    f"+/- {heavy_eval_metrics['eval/snmc_stderr']:.4f}",
                    f"(n={int(heavy_eval_metrics['eval/n_episodes'])})",
                )

            if save_every > 0 and (epoch + 1) % save_every == 0:
                save_checkpoint(
                    path=checkpoint_path,
                    epoch=epoch,
                    cfg=cfg,
                    policy_head=policy_head,
                    optimizer=optimizer,
                    backbone_model_state_dict=(
                        backbone.model.state_dict() if include_backbone_state else None
                    ),
                    backbone_checkpoint_path=str(backbone.checkpoint_path),
                )

        save_checkpoint(
            path=checkpoint_path,
            epoch=num_epochs - 1,
            cfg=cfg,
            policy_head=policy_head,
            optimizer=optimizer,
            backbone_model_state_dict=(
                backbone.model.state_dict() if include_backbone_state else None
            ),
            backbone_checkpoint_path=str(backbone.checkpoint_path),
        )

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        if likelihood_eval_milestones:
            _save_heavy_eval_results(
                json_path=heavy_eval_json_path,
                csv_path=heavy_eval_csv_path,
                results=heavy_eval_results,
            )
        OmegaConf.save(cfg, run_dir / "config_resolved.yaml")

        if wandb_run is not None:
            wandb_run.summary["checkpoint_path"] = str(checkpoint_path)
            wandb_run.summary["contains_backbone_state"] = bool(include_backbone_state)
            wandb_run.summary["metrics_path"] = str(run_dir / "metrics.json")
            if likelihood_eval_milestones:
                wandb_run.summary["heavy_eval_metrics_path"] = str(heavy_eval_json_path)

        print("BED policy training finished.")
        print(f"metrics_path: {run_dir / 'metrics.json'}")
        if likelihood_eval_milestones:
            print(f"heavy_eval_metrics_path: {heavy_eval_json_path}")
            pending_milestones = [
                milestone
                for milestone in likelihood_eval_milestones
                if milestone not in completed_heavy_eval
            ]
            if pending_milestones:
                print(f"unreached_heavy_eval_milestones: {pending_milestones}")
        print(f"checkpoint: {checkpoint_path}")
        print(f"contains_backbone_state: {include_backbone_state}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
