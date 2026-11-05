#!/usr/bin/env python
"""Unified AL policy training runner (NLL + REINFORCE, optional probe diagnostics).

All AL training variants are expressed through config:
- ``train.policy_head.type`` selects MLP vs. target-aware transformer heads.
- ``task.dim_x_min`` / ``task.dim_x_max`` choose fixed-dim vs. mixed-dim GP training.
- ``train.loss.policy_target_weighting`` / ``train.loss.policy_target_weight_beta`` /
  ``train.loss.policy_target_weight_threshold``
  toggle the loss-driven policy objective.
- ``probe.enabled`` and ``output.save_probe_plots`` control held-out rollout diagnostics.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from polar.backbones import create_backbone
from polar.evaluation import (
    plot_policy_probe_rounds_grid,
    rollout_policy_probe_episode,
    summarize_probe_metrics,
)
from polar.tasks import create_al_task
from polar.trainers import (
    ALPolicyTrainer,
    build_lr_scheduler,
    build_policy_head,
    build_policy_optimizer,
)
from polar.utils.run_utils import (
    build_run_dirs,
    build_run_tag,
    extract_param_group_lrs,
    init_wandb,
    log_wandb_metrics,
    save_checkpoint,
)
from polar.utils.seeding import set_global_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probe_enabled(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "probe.enabled", default=False))


def _save_probe_plots(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "output.save_probe_plots", default=False))


def _log_wandb_epoch(
    *,
    wandb_run: Any | None,
    epoch: int,
    epoch_metrics: dict[str, float],
    optimizer: torch.optim.Optimizer,
    probe_summary: dict[str, float] | None = None,
) -> None:
    if wandb_run is None:
        return

    payload: dict[str, float] = {
        "epoch": float(epoch),
        "train/steps": float(epoch_metrics["steps"]),
        "train/total_loss": float(epoch_metrics["total_loss"]),
        "train/nll_loss": float(epoch_metrics["nll_loss"]),
        "train/policy_loss": float(epoch_metrics["policy_loss"]),
        "train/policy_entropy": float(epoch_metrics["policy_entropy"]),
        "train/selected_query_y_mean": float(epoch_metrics["selected_query_y_mean"]),
        "train/target_log_prob_start": float(epoch_metrics["target_log_prob_start"]),
        "train/target_log_prob_end": float(epoch_metrics["target_log_prob_end"]),
        "train/target_log_prob_gain": float(epoch_metrics["target_log_prob_gain"]),
        "train/policy_target_score_start": float(epoch_metrics["policy_target_score_start"]),
        "train/policy_target_score_end": float(epoch_metrics["policy_target_score_end"]),
        "train/policy_target_score_gain": float(epoch_metrics["policy_target_score_gain"]),
    }
    if probe_summary is not None:
        payload.update(
            {
                "probe/rounds": float(probe_summary["rounds"]),
                "probe/target_rmse_start": float(probe_summary["target_rmse_start"]),
                "probe/target_rmse_end": float(probe_summary["target_rmse_end"]),
                "probe/weighted_target_rmse_start": float(
                    probe_summary["weighted_target_rmse_start"]
                ),
                "probe/weighted_target_rmse_end": float(
                    probe_summary["weighted_target_rmse_end"]
                ),
                "probe/target_log_prob_start": float(probe_summary["target_log_prob_start"]),
                "probe/target_log_prob_end": float(probe_summary["target_log_prob_end"]),
                "probe/weighted_target_log_prob_start": float(
                    probe_summary["weighted_target_log_prob_start"]
                ),
                "probe/weighted_target_log_prob_end": float(
                    probe_summary["weighted_target_log_prob_end"]
                ),
                "probe/selected_query_y_mean": float(probe_summary["selected_query_y_mean"]),
                "probe/selected_query_raw_y_mean": float(
                    probe_summary["selected_query_raw_y_mean"]
                ),
            }
        )
    payload.update(extract_param_group_lrs(optimizer))
    log_wandb_metrics(wandb_run, step=epoch, payload=payload)


@contextmanager
def _temporary_eval_mode(*modules: torch.nn.Module) -> Iterator[None]:
    previous = [(module, bool(module.training)) for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, was_training in previous:
            module.train(was_training)


def _should_run_probe(*, epoch: int, num_epochs: int, every_epochs: int) -> bool:
    if every_epochs <= 0:
        raise ValueError("probe.every_epochs must be positive when probe.enabled=true.")
    human_epoch = int(epoch) + 1
    return human_epoch % every_epochs == 0 or human_epoch == num_epochs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path="../config", config_name="exp/train_al_policy")
def main(cfg: DictConfig) -> None:
    print(f"==== AL Policy Training ({cfg.task.name}) ====")
    print(OmegaConf.to_yaml(cfg))

    set_global_seed(int(cfg.seed))

    run_tag = build_run_tag(cfg)
    run_dir, checkpoint_dir = build_run_dirs(cfg, run_tag)
    print(f"run_dir: {run_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")

    backbone = create_backbone(
        name=cfg.model.name,
        checkpoint_path=cfg.model.checkpoint_path,
        version=cfg.model.version,
        device=cfg.device,
        checkpoints_root=cfg.paths.checkpoints_root,
        cache_trainset_representation=cfg.model.cache_trainset_representation,
        strict=cfg.model.strict_load,
    )
    print(f"loaded_backbone: {type(backbone).__name__}")
    print(f"checkpoint_path: {backbone.checkpoint_path}")

    embedding_dim = int(getattr(backbone, "embedding_dim"))
    if embedding_dim <= 0:
        raise ValueError("Backbone must expose a positive integer `embedding_dim`.")
    policy_head = build_policy_head(
        input_dim=embedding_dim,
        policy_head_cfg=cfg.train.policy_head,
    ).to(backbone.device)
    print(f"policy_head: {type(policy_head).__name__}")

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

    task = create_al_task(cfg.task, seed=int(cfg.seed))
    n_context = int(cfg.task.n_context_init)
    n_query = int(cfg.task.n_query)
    n_target = int(cfg.task.n_target)
    if n_target <= 0:
        raise ValueError("AL training requires task.n_target > 0.")

    required_size = n_context + n_query + n_target
    eligible_dataset_ids = task.eligible_dataset_ids(min_total_size=required_size)
    if not eligible_dataset_ids:
        raise ValueError(
            f"No train dataset has enough points for required_size={required_size} "
            f"(context={n_context}, query={n_query}, target={n_target})."
        )

    dim_summary = (
        f"dim_x_range=[{getattr(task, 'dim_x_min', task.dim_x)},"
        f" {getattr(task, 'dim_x_max', task.dim_x)}]"
    )
    policy_target_weighting = str(
        OmegaConf.select(cfg, "train.loss.policy_target_weighting", default="uniform")
    )
    policy_target_weight_beta = float(
        OmegaConf.select(cfg, "train.loss.policy_target_weight_beta", default=1.0)
    )
    policy_target_weight_threshold = float(
        OmegaConf.select(cfg, "train.loss.policy_target_weight_threshold", default=0.0)
    )
    print(
        "train_task:",
        f"name={cfg.task.name}",
        f"num_datasets={len(task.dataset_ids)}",
        f"eligible_datasets={len(eligible_dataset_ids)}",
        dim_summary,
        f"required_size={required_size}",
        f"weighting={policy_target_weighting}",
        f"beta={policy_target_weight_beta:.3f}",
        f"threshold={policy_target_weight_threshold:.3f}",
    )

    trainer = ALPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=finetune_epochs,
        nll_weight=float(cfg.train.loss.nll_weight),
        policy_weight=float(cfg.train.loss.policy_weight),
        gamma=float(cfg.train.loss.gamma),
        reward_clamp_min=float(cfg.train.loss.reward_clamp_min),
        normalize_rewards=bool(cfg.train.loss.normalize_rewards),
        detach_policy_features=bool(cfg.train.loss.detach_policy_features),
        grad_clip_norm=(
            float(cfg.train.optim.grad_clip_norm)
            if cfg.train.optim.grad_clip_norm is not None
            else None
        ),
        use_amp=bool(cfg.train.optim.get("use_amp", True)),
        gradient_checkpointing=bool(cfg.train.optim.get("gradient_checkpointing", False)),
        policy_target_weighting=policy_target_weighting,
        policy_target_weight_beta=policy_target_weight_beta,
        policy_target_weight_threshold=policy_target_weight_threshold,
    )

    # Optional probe episode (held-out rollout for diagnostics).
    probe_episode = None
    if _probe_enabled(cfg):
        if not hasattr(task, "build_test_episode"):
            raise ValueError("Probe diagnostics require tasks with a build_test_episode method.")
        probe_episode = task.build_test_episode(
            seed=str(cfg.probe.seed),
            n_context=n_context,
            query_pool_size=n_query,
            n_target=n_target,
        )
        print(
            "probe_episode:",
            f"seed={cfg.probe.seed}",
            f"num_acquisitions={int(cfg.probe.num_acquisitions)}",
            f"snapshot_stride={int(cfg.probe.snapshot_round_stride)}",
        )

    num_epochs = int(cfg.train.num_epochs)
    batch_size = int(cfg.train.batch_size)
    num_acquisitions = int(cfg.train.num_acquisitions)
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    include_backbone_state = finetune_epochs > 0

    probe_dir = run_dir / "probe"
    probe_history: list[dict[str, Any]] = []

    wandb_run = init_wandb(cfg, run_dir, run_tag)
    try:
        all_metrics: list[dict[str, float]] = []
        for epoch in range(num_epochs):
            if finetune_epochs > 0 and epoch == finetune_epochs:
                trainer.freeze_backbone()
                print(f"[epoch={epoch:04d}] backbone frozen (finetune_epochs={finetune_epochs})")

            episodes = task.sample_train_batch(
                batch_size=batch_size,
                n_context=n_context,
                n_query=n_query,
                n_target=n_target,
            )
            epoch_metrics = trainer.train_one_epoch(
                episodes=episodes,
                num_acquisitions=num_acquisitions,
            )
            if scheduler is not None:
                scheduler.step()

            current_lrs = [pg["lr"] for pg in optimizer.param_groups]
            epoch_metrics["lr_policy"] = float(current_lrs[0])
            if len(current_lrs) > 1:
                epoch_metrics["lr_backbone"] = float(current_lrs[1])
            epoch_metrics["epoch"] = float(epoch)
            all_metrics.append(epoch_metrics)

            probe_summary: dict[str, float] | None = None
            if probe_episode is not None and _should_run_probe(
                epoch=epoch,
                num_epochs=num_epochs,
                every_epochs=int(cfg.probe.every_epochs),
            ):
                with _temporary_eval_mode(backbone.model, policy_head):
                    round_metrics, round_snapshots = rollout_policy_probe_episode(
                        episode=probe_episode,
                        backbone=backbone,
                        policy_head=policy_head,
                        num_acquisitions=int(cfg.probe.num_acquisitions),
                        selection=str(cfg.probe.selection),
                        collect_snapshots=_save_probe_plots(cfg),
                        snapshot_round_stride=int(cfg.probe.snapshot_round_stride),
                        policy_target_weighting=policy_target_weighting,
                        policy_target_weight_beta=policy_target_weight_beta,
                        policy_target_weight_threshold=policy_target_weight_threshold,
                    )
                probe_summary = summarize_probe_metrics(round_metrics)
                probe_record: dict[str, Any] = {
                    "epoch": int(epoch),
                    "summary": probe_summary,
                    "round_metrics": round_metrics,
                }
                probe_history.append(probe_record)
                if _save_probe_plots(cfg) and round_snapshots:
                    plot_path = probe_dir / f"epoch_{epoch + 1:04d}.png"
                    plot_policy_probe_rounds_grid(
                        save_path=plot_path,
                        round_snapshots=round_snapshots,
                        method_name=f"epoch {epoch + 1}",
                        ci_std=float(cfg.probe.ci_std),
                    )
                    probe_record["plot_path"] = str(plot_path)
                    print(f"probe_plot: {plot_path}")
                print(
                    f"[probe epoch={epoch + 1:04d}]",
                    f"rmse={probe_summary['target_rmse_start']:.4f}->{probe_summary['target_rmse_end']:.4f}",
                    f"w_rmse={probe_summary['weighted_target_rmse_start']:.4f}->{probe_summary['weighted_target_rmse_end']:.4f}",
                    f"logp={probe_summary['target_log_prob_start']:.4f}->{probe_summary['target_log_prob_end']:.4f}",
                    f"w_logp={probe_summary['weighted_target_log_prob_start']:.4f}->{probe_summary['weighted_target_log_prob_end']:.4f}",
                    f"selected_y_mean={probe_summary['selected_query_y_mean']:.4f}",
                )

            _log_wandb_epoch(
                wandb_run=wandb_run,
                epoch=epoch,
                epoch_metrics=epoch_metrics,
                optimizer=optimizer,
                probe_summary=probe_summary,
            )

            if epoch % log_every == 0 or epoch == num_epochs - 1:
                lr_str = f"lr_p={epoch_metrics.get('lr_policy', 0):.2e}"
                if "lr_backbone" in epoch_metrics:
                    lr_str += f" lr_b={epoch_metrics['lr_backbone']:.2e}"
                print(
                    f"[epoch={epoch:04d}]",
                    f"steps={epoch_metrics['steps']:.2f}",
                    f"total_loss={epoch_metrics['total_loss']:.6f}",
                    f"nll={epoch_metrics['nll_loss']:.6f}",
                    f"policy={epoch_metrics['policy_loss']:.6f}",
                    f"entropy={epoch_metrics['policy_entropy']:.4f}",
                    f"logp={epoch_metrics['target_log_prob_start']:.6f}"
                    f"->{epoch_metrics['target_log_prob_end']:.6f}",
                    f"policy_score_gain={epoch_metrics['policy_target_score_gain']:.6f}",
                    lr_str,
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
                    backbone_checkpoint_path=str(backbone.checkpoint_path)
                    if backbone.checkpoint_path is not None
                    else None,
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
            backbone_checkpoint_path=str(backbone.checkpoint_path)
            if backbone.checkpoint_path is not None
            else None,
        )

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        if probe_history:
            with (run_dir / "probe_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(probe_history, f, ensure_ascii=False, indent=2)
        OmegaConf.save(cfg, run_dir / "config_resolved.yaml")

        if wandb_run is not None:
            wandb_run.summary["checkpoint_path"] = str(checkpoint_path)
            wandb_run.summary["contains_backbone_state"] = bool(include_backbone_state)
            wandb_run.summary["metrics_path"] = str(run_dir / "metrics.json")
            if probe_history:
                wandb_run.summary["probe_metrics_path"] = str(run_dir / "probe_metrics.json")

        print("AL policy training finished.")
        print(f"metrics_path: {run_dir / 'metrics.json'}")
        if probe_history:
            print(f"probe_metrics_path: {run_dir / 'probe_metrics.json'}")
        print(f"checkpoint: {checkpoint_path}")
        print(f"contains_backbone_state: {include_backbone_state}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
