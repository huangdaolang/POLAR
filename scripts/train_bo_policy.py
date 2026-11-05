#!/usr/bin/env python
"""Policy post-training runner (NLL + REINFORCE) for BO tasks."""

from __future__ import annotations

import json
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from polar.backbones import create_backbone
from polar.tasks import create_bo_task
from polar.trainers import (
    BOPolicyTrainer,
    CandidatePolicyHead,
    build_lr_scheduler,
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


def _log_wandb_epoch(
    *,
    wandb_run: Any | None,
    epoch: int,
    epoch_metrics: dict[str, float],
    optimizer: torch.optim.Optimizer,
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
        "train/reward_mean": float(epoch_metrics["reward_mean"]),
        "train/reward_nonzero_ratio": float(epoch_metrics["reward_nonzero_ratio"]),
        "train/best_y_start": float(epoch_metrics["best_y_start"]),
        "train/best_y_end": float(epoch_metrics["best_y_end"]),
        "train/best_y_improvement": float(epoch_metrics["best_y_improvement"]),
        "train/target_log_prob_start": float(epoch_metrics["target_log_prob_start"]),
        "train/target_log_prob_end": float(epoch_metrics["target_log_prob_end"]),
    }
    payload.update(extract_param_group_lrs(optimizer))
    log_wandb_metrics(wandb_run, step=epoch, payload=payload)


@hydra.main(version_base="1.3", config_path="../config", config_name="exp/train_bo_policy")
def main(cfg: DictConfig) -> None:
    print(f"==== BO Policy Training ({cfg.task.name}) ====")
    print(OmegaConf.to_yaml(cfg))

    set_global_seed(int(cfg.seed))

    if str(cfg.task.split) != "train":
        raise ValueError("BO post-training expects task.split=train.")

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

    task = create_bo_task(cfg.task, seed=int(cfg.seed))
    nll_weight = float(cfg.train.loss.nll_weight)

    n_context = int(cfg.task.n_context_init)
    n_query = int(cfg.task.n_query)
    configured_n_target = int(cfg.task.n_target)
    use_target_auxiliary = finetune_epochs > 0 and nll_weight > 0.0
    n_target = configured_n_target if use_target_auxiliary else 0
    required_size = n_context + n_query + n_target
    eligible_dataset_ids = task.eligible_dataset_ids(min_total_size=required_size)
    if not eligible_dataset_ids:
        raise ValueError(
            f"No train dataset has enough points for required_size={required_size} "
            f"(context={n_context}, query={n_query}, target={n_target})."
        )
    print(
        "train_task:",
        f"name={cfg.task.name}",
        f"num_datasets={len(task.dataset_ids)}",
        f"eligible_datasets={len(eligible_dataset_ids)}",
        f"required_size={required_size}",
        f"configured_target={configured_n_target}",
        f"effective_target={n_target}",
    )

    reward_clamp_min_cfg = cfg.train.loss.reward_clamp_min
    reward_clamp_min = float(reward_clamp_min_cfg) if reward_clamp_min_cfg is not None else None

    trainer = BOPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=finetune_epochs,
        nll_weight=nll_weight,
        policy_weight=float(cfg.train.loss.policy_weight),
        gamma=float(cfg.train.loss.gamma),
        reward_type=str(cfg.train.loss.reward_type),
        reward_log_epsilon=float(cfg.train.loss.reward_log_epsilon),
        reward_scale_by_range=bool(cfg.train.loss.reward_scale_by_range),
        reward_no_improve_penalty=float(cfg.train.loss.reward_no_improve_penalty),
        reward_clamp_min=reward_clamp_min,
        normalize_rewards=bool(cfg.train.loss.normalize_rewards),
        reward_normalization_scope=str(cfg.train.loss.reward_normalization_scope),
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
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    include_backbone_state = finetune_epochs > 0

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
                    f"steps={epoch_metrics['steps']:.2f}",
                    f"total_loss={epoch_metrics['total_loss']:.6f}",
                    f"nll={epoch_metrics['nll_loss']:.6f}",
                    f"policy={epoch_metrics['policy_loss']:.6f}",
                    f"entropy={epoch_metrics['policy_entropy']:.4f}",
                    f"reward={epoch_metrics['reward_mean']:.6f}",
                    f"improve={epoch_metrics['best_y_improvement']:.6f}",
                    f"reward_nonzero={epoch_metrics['reward_nonzero_ratio']:.4f}",
                    f"best_y={epoch_metrics['best_y_start']:.6f}"
                    f" -> {epoch_metrics['best_y_end']:.6f}",
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
        OmegaConf.save(cfg, run_dir / "config_resolved.yaml")

        if wandb_run is not None:
            wandb_run.summary["checkpoint_path"] = str(checkpoint_path)
            wandb_run.summary["contains_backbone_state"] = bool(include_backbone_state)
            wandb_run.summary["metrics_path"] = str(run_dir / "metrics.json")

        print("BO policy training finished.")
        print(f"metrics_path: {run_dir / 'metrics.json'}")
        print(f"checkpoint: {checkpoint_path}")
        print(f"contains_backbone_state: {include_backbone_state}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
