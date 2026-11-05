#!/usr/bin/env python
"""Evaluate BED policies (trained policy vs random baseline)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from polar.backbones import create_backbone
from polar.backbones.tabpfn.checkpoint_paths import get_path_layout
from polar.evaluation import (
    EpisodeResult,
    aggregate_eig_bounds,
    compute_bed_eig_bounds_for_trajectory,
    rollout_bed_policy,
    rollout_bed_random,
)
from polar.utils.policy_loading import (
    extract_backbone_state_dict,
    extract_policy_state_dict,
    load_checkpoint_payload,
    resolve_backbone_checkpoint_path,
    resolve_policy_arch,
)
from polar.tasks import create_bed_task
from polar.tasks.bed import bed_task_checkpoint_kind
from polar.trainers import CandidatePolicyHead
from polar.utils.plotting import plot_bed_spce_comparison
from polar.utils.method_names import (
    MethodSpec,
    normalize_backend_name,
    parse_method,
)
from polar.utils.seeding import set_global_seed

def _build_run_dir(cfg: DictConfig, results_root: Path) -> Path:
    suffix = (
        str(cfg.output.run_tag)
        if cfg.output.run_tag
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = results_root / str(cfg.output.experiment_name) / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _available_backends(cfg: DictConfig) -> set[str]:
    models_cfg = OmegaConf.to_container(cfg.models, resolve=True)
    if not isinstance(models_cfg, dict):
        raise ValueError("models config must be a mapping.")
    return {normalize_backend_name(str(name)) for name in models_cfg.keys()}


def _model_cfg_for_backend(cfg: DictConfig, backend: str) -> DictConfig:
    backend_cfg = cfg.models.get(backend)
    if backend_cfg is None:
        available = ", ".join(sorted(_available_backends(cfg)))
        raise ValueError(f"Missing models.{backend} config. Available: {available}")
    return backend_cfg


def _parse_method_spec(cfg: DictConfig, *, method_name: str) -> MethodSpec:
    # BED has no surrogate acquisition baselines, so only the policy and random
    # forms of the shared grammar apply here.
    return parse_method(
        method_name,
        backends=_available_backends(cfg),
        default_backend=cfg.policy.get("backend"),
        acquisitions=(),
    )


def _resolve_policy_checkpoint_path(cfg: DictConfig, *, method_name: str, backend: str) -> str:
    """Resolve a policy checkpoint by method name, then backend name, then default.

    This is the same lookup order the AL and BO evaluation scripts use, so
    keying a checkpoint by backbone (``policy.checkpoints.tabicl=...``) works
    identically in all three.
    """
    checkpoints_cfg_obj = cfg.policy.checkpoints
    if isinstance(checkpoints_cfg_obj, str) and checkpoints_cfg_obj.strip():
        return checkpoints_cfg_obj.strip()
    if OmegaConf.is_config(checkpoints_cfg_obj):
        checkpoints_cfg = OmegaConf.to_container(checkpoints_cfg_obj, resolve=True)
    elif checkpoints_cfg_obj is None:
        checkpoints_cfg = {}
    else:
        checkpoints_cfg = checkpoints_cfg_obj

    if isinstance(checkpoints_cfg, dict):
        for key in (method_name, backend):
            path = checkpoints_cfg.get(key)
            if path is not None and str(path).strip():
                return str(path)

    default_path = cfg.policy.default_checkpoint_path
    if default_path is not None and str(default_path).strip():
        return str(default_path)

    raise ValueError(
        f"Method '{method_name}' needs a policy checkpoint. Provide one of:\n"
        f"- +policy.checkpoints.{method_name}=/abs/path/to/checkpoint.pt\n"
        f"- +policy.checkpoints.{backend}=/abs/path/to/checkpoint.pt\n"
        "- policy.default_checkpoint_path=/abs/path/to/checkpoint.pt"
    )


def _resolve_bed_backbone_options(
    model_cfg: DictConfig,
    task,
    *,
    explicit_checkpoint_path: str | None,
) -> tuple[str, int | None]:
    requested_kind = str(model_cfg.get("checkpoint_kind", "auto")).strip().lower()
    if requested_kind not in {"auto", "regression", "classification"}:
        raise ValueError(
            "model.checkpoint_kind must be one of: auto, regression, classification."
        )

    if requested_kind == "auto":
        checkpoint_kind = (
            "auto"
            if explicit_checkpoint_path is not None
            else bed_task_checkpoint_kind(task)
        )
    else:
        checkpoint_kind = requested_kind

    configured_num_classes = model_cfg.get("num_classes")
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


def _load_policy(
    cfg: DictConfig, task, *, method_name: str, backend: str
) -> tuple[Any, CandidatePolicyHead, str]:
    ckpt_path_str = _resolve_policy_checkpoint_path(cfg, method_name=method_name, backend=backend)
    payload = load_checkpoint_payload(ckpt_path_str)
    model_cfg = _model_cfg_for_backend(cfg, backend)
    backbone_ckpt = resolve_backbone_checkpoint_path(model_cfg=model_cfg, payload=payload)

    checkpoint_kind, num_classes = _resolve_bed_backbone_options(
        model_cfg,
        task,
        explicit_checkpoint_path=backbone_ckpt,
    )

    backbone = create_backbone(
        name=str(model_cfg.name),
        checkpoint_path=backbone_ckpt,
        version=model_cfg.get("version"),
        checkpoint_kind=checkpoint_kind,
        num_classes=num_classes,
        device=cfg.device,
        checkpoints_root=cfg.paths.checkpoints_root,
        cache_trainset_representation=bool(model_cfg.get("cache_trainset_representation", False)),
        strict=bool(model_cfg.get("strict_load", True)),
    )

    arch = resolve_policy_arch(payload=payload, policy_cfg=cfg.policy)
    if str(arch.get("type", "mlp")).strip().lower() != "mlp":
        raise ValueError("BED evaluation only supports MLP policy heads.")
    embedding_dim = int(getattr(backbone, "embedding_dim"))
    policy_head = CandidatePolicyHead(
        input_dim=embedding_dim,
        hidden_dim=arch["hidden_dim"],
        num_layers=arch["num_layers"],
        activation=arch["activation"],
        dropout=arch["dropout"],
        use_remaining_budget=arch["use_remaining_budget"],
        budget_scale=arch["budget_scale"],
    ).to(backbone.device)

    policy_state = extract_policy_state_dict(payload)
    policy_head.load_state_dict(policy_state, strict=True)
    policy_head.eval()

    backbone_state = extract_backbone_state_dict(payload)
    if backbone_state is not None:
        backbone.model.load_state_dict(
            backbone_state,
            strict=bool(cfg.policy.strict_backbone_state),
        )
    backbone.model.eval()
    return backbone, policy_head, ckpt_path_str


@hydra.main(version_base="1.3", config_path="../config", config_name="exp/evaluate_bed")
def main(cfg: DictConfig) -> None:
    print(f"==== BED Evaluate ({cfg.task.name}) ====")
    print(OmegaConf.to_yaml(cfg))

    set_global_seed(int(cfg.seed))

    methods_cfg = OmegaConf.to_container(cfg.compare.methods, resolve=True)
    if not isinstance(methods_cfg, list) or not methods_cfg:
        raise ValueError("compare.methods must be a non-empty list.")
    method_specs = [_parse_method_spec(cfg, method_name=str(method)) for method in methods_cfg]

    seen: set[str] = set()
    for spec in method_specs:
        if spec.name in seen:
            raise ValueError(f"Duplicate method: {spec.name}")
        seen.add(spec.name)

    layout = get_path_layout(
        checkpoints_root=cfg.paths.checkpoints_root,
        results_root=cfg.paths.results_root,
    )
    run_dir = _build_run_dir(cfg, layout.results_root)
    plots_dir = run_dir / "plots"
    if bool(cfg.output.save_plots):
        plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir: {run_dir}")

    task = create_bed_task(cfg.task, seed=int(cfg.seed))
    n_query_eval_raw = cfg.eval.get("n_query_eval")
    n_query_eval = int(n_query_eval_raw) if n_query_eval_raw is not None else int(task.n_query)
    print(f"task: {task.name} (dim_x={task.dim_x}, n_query_eval={n_query_eval})")

    num_acquisitions = int(cfg.eval.num_acquisitions)
    n_eval_episodes = int(cfg.eval.n_eval_episodes)
    n_contrastive_eval = int(cfg.eval.n_contrastive_eval)
    n_plot_episodes = int(cfg.eval.get("n_plot_episodes", 6))
    greedy = str(cfg.policy.selection).strip().lower() == "argmax"

    uncertainty_str = str(cfg.eval.plot_uncertainty).strip().lower()
    if uncertainty_str not in {"ci95", "std", "sem"}:
        raise ValueError("eval.plot_uncertainty must be one of: ci95, std, sem.")
    uncertainty_mode = uncertainty_str  # type: ignore[assignment]

    original_n_query = int(task.n_query)
    task.n_query = n_query_eval
    try:
        print(f"Pre-sampling {n_eval_episodes} shared episodes (n_query={n_query_eval}) ...")
        shared_episodes = task.sample_batch(batch_size=n_eval_episodes)
    finally:
        task.n_query = original_n_query

    backbone, policy_head, policy_ckpt_path = None, None, None
    for spec in method_specs:
        if spec.mode == "policy":
            backbone, policy_head, policy_ckpt_path = _load_policy(cfg, task, method_name=spec.name, backend=str(spec.backend))
            print(f"policy_checkpoint: {policy_ckpt_path}")
            print(f"checkpoint_kind: {getattr(backbone, 'checkpoint_kind', 'unknown')}")
            break

    all_results: dict[str, list[EpisodeResult]] = {}
    method_reports: dict[str, dict[str, Any]] = {}
    device = torch.device(cfg.device)

    for spec in method_specs:
        print(f"---- method={spec.name} ----")
        results: list[EpisodeResult] = []

        for ep_idx, episode in enumerate(shared_episodes):
            theta_true = episode.theta_all[0]

            if spec.mode == "policy":
                assert backbone is not None and policy_head is not None
                x_traj, y_traj = rollout_bed_policy(
                    backbone=backbone,
                    policy_head=policy_head,
                    episode=episode,
                    num_acquisitions=num_acquisitions,
                    greedy=greedy,
                )
            else:
                x_traj, y_traj = rollout_bed_random(
                    episode=episode,
                    num_acquisitions=num_acquisitions,
                )

            spce_val, snmc_val = compute_bed_eig_bounds_for_trajectory(
                context_x=episode.context_x,
                context_y=episode.context_y,
                x_traj=x_traj,
                y_traj=y_traj,
                theta_true=theta_true,
                task=task,
                n_contrastive_eval=n_contrastive_eval,
                device=device,
            )
            results.append(
                EpisodeResult(
                    method=spec.name,
                    spce=spce_val,
                    snmc=snmc_val,
                    theta_true=theta_true.cpu().numpy(),
                    x_traj=torch.stack(x_traj).cpu().numpy() if x_traj else None,
                    num_steps=len(x_traj),
                )
            )

            if (ep_idx + 1) % 20 == 0 or ep_idx == n_eval_episodes - 1:
                running_spce = sum(result.spce for result in results) / max(len(results), 1)
                running_snmc = sum(result.snmc for result in results) / max(len(results), 1)
                print(
                    f"  [episode={ep_idx + 1:03d}/{n_eval_episodes}]"
                    f" spce={spce_val:.4f}"
                    f" snmc={snmc_val:.4f}"
                    f" running_spce={running_spce:.4f}"
                    f" running_snmc={running_snmc:.4f}"
                )

        all_results[spec.name] = results
        agg = aggregate_eig_bounds(results)
        print(
            f"[{spec.name}]"
            f" sPCE:"
            f" mean={agg['mean_spce']:.4f}"
            f" std={agg['std_spce']:.4f}"
            f" sem={agg['sem_spce']:.4f}"
            f" | sNMC:"
            f" mean={agg['mean_snmc']:.4f}"
            f" std={agg['std_snmc']:.4f}"
            f" sem={agg['sem_snmc']:.4f}"
            f" (n={int(agg['n_episodes'])})"
        )
        method_reports[spec.name] = {
            "mode": spec.mode,
            "n_episodes": int(agg["n_episodes"]),
            "mean_spce": agg["mean_spce"],
            "std_spce": agg["std_spce"],
            "sem_spce": agg["sem_spce"],
            "ci95_spce": agg["ci95_spce"],
            "per_episode_spce": [result.spce for result in results],
            "mean_snmc": agg["mean_snmc"],
            "std_snmc": agg["std_snmc"],
            "sem_snmc": agg["sem_snmc"],
            "ci95_snmc": agg["ci95_snmc"],
            "per_episode_snmc": [result.snmc for result in results],
        }
        if spec.mode == "policy":
            method_reports[spec.name]["policy_checkpoint"] = policy_ckpt_path

    if bool(cfg.output.save_plots):
        bar_path = plots_dir / "bed_evaluate_spce.png"
        plot_bed_spce_comparison(
            bar_path,
            method_stats=method_reports,
            uncertainty=uncertainty_mode,
            task_label=str(task.name).replace("_", " ").title(),
        )
        print(f"bar_plot: {bar_path}")

        plotter = task.trajectory_plotter
        if plotter is not None and n_plot_episodes > 0:
            for method_name, results in all_results.items():
                plot_results = [result for result in results if result.x_traj is not None][:n_plot_episodes]
                if not plot_results:
                    continue
                traj_path = plots_dir / f"trajectories_{method_name}.png"
                plotter(
                    traj_path,
                    [
                        {
                            "theta_true": result.theta_true,
                            "x_traj": result.x_traj,
                            "spce": result.spce,
                        }
                        for result in plot_results
                    ],
                )
                print(f"trajectory_plot ({method_name}): {traj_path}")

    report: dict[str, Any] = {
        "task": f"bed_{task.name}",
        "dim_x": int(task.dim_x),
        "n_eval_episodes": n_eval_episodes,
        "num_acquisitions": num_acquisitions,
        "n_contrastive_eval": n_contrastive_eval,
        "n_query_eval": n_query_eval,
        "methods": [spec.name for spec in method_specs],
        "method_reports": method_reports,
    }

    if bool(cfg.output.save_metrics):
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        OmegaConf.save(cfg, run_dir / "config_resolved.yaml")

    print("BED evaluation finished.")
    print(f"metrics_path: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
