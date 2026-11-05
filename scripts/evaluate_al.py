#!/usr/bin/env python
"""Unified AL evaluation runner (policy + surrogate baselines, optional weighted metrics)."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from polar.backbones import create_backbone
from polar.backbones.tabpfn.checkpoint_paths import get_path_layout
from polar.evaluation import (
    aggregate_al_eval_metrics,
    aggregate_al_eval_metrics_by_dataset,
    evaluate_policy_episodes,
    evaluate_surrogate_episodes,
    episode_raw_x_to_model,
    load_policy_checkpoint,
    plot_policy_probe_rounds_grid,
    rollout_policy_probe_episode,
    rollout_surrogate_probe_episode,
)
from polar.tasks import create_al_task
from polar.utils.surrogates import (
    GPyTorchSurrogate,
    TabICLSurrogate,
    TabPFNSurrogate,
)
from polar.utils.plotting import (
    plot_al_metric_grid as _plot_metric_grid,
    plot_benchmark_surface_rounds_grid as _plot_benchmark_surface_rounds_grid,
)
from polar.utils.method_names import (
    MethodSpec,
    normalize_backend_name as _normalize_backend_name,
    parse_method,
)
from polar.utils.seeding import set_global_seed


def _build_run_tag(cfg: DictConfig) -> str:
    if cfg.output.run_tag:
        return str(cfg.output.run_tag)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_run_dir(cfg: DictConfig, run_tag: str) -> Path:
    layout = get_path_layout(
        checkpoints_root=cfg.paths.checkpoints_root,
        results_root=cfg.paths.results_root,
    )
    run_dir = layout.results_root / str(cfg.output.experiment_name) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _cfg_str_list_or_none(value: Any) -> list[str] | None:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("Expected null, string, or list of strings.")


def _available_backends(cfg: DictConfig) -> set[str]:
    models_cfg = OmegaConf.to_container(cfg.models, resolve=True)
    if not isinstance(models_cfg, dict):
        raise ValueError("models config must be a mapping.")
    return {_normalize_backend_name(str(name)) for name in models_cfg.keys()}


AL_ACQUISITIONS = (
    "weighted_variance_reduction",
    "variance_reduction",
    "uncertainty",
    "random",
)

# Acquisitions that are only implemented on the GP surrogate.
_GP_ONLY_ACQUISITIONS = {"variance_reduction", "weighted_variance_reduction"}


def _parse_method_spec(cfg: DictConfig, *, method_name: str) -> MethodSpec:
    spec = parse_method(
        method_name,
        backends=_available_backends(cfg),
        default_backend=cfg.policy.get("backend"),
        acquisitions=AL_ACQUISITIONS,
        # Every AL method also needs a backend to fit and score the target set,
        # so a standalone `random` is underspecified here; use `<backend>_random`.
        allow_bare_random=False,
    )
    if spec.acquisition in _GP_ONLY_ACQUISITIONS and spec.backend != "gp":
        raise ValueError(
            f"Method '{spec.name}' is invalid: acquisition '{spec.acquisition}' is "
            "currently supported only with backend='gp'."
        )
    return spec


def _model_cfg_for_backend(cfg: DictConfig, backend: str) -> DictConfig:
    backend_cfg = cfg.models.get(backend)
    if backend_cfg is None:
        available = ", ".join(sorted(_available_backends(cfg)))
        raise ValueError(f"Missing models.{backend} config. Available: {available}")
    return backend_cfg


def _build_surrogate(cfg: DictConfig, *, backend: str):
    backend = _normalize_backend_name(backend)
    model_cfg = _model_cfg_for_backend(cfg, backend)

    if backend == "tabpfn":
        backbone = create_backbone(
            name=str(model_cfg.get("name", "tabpfn")),
            checkpoint_path=model_cfg.get("checkpoint_path"),
            version=model_cfg.get("version"),
            device=cfg.device,
            checkpoints_root=cfg.paths.checkpoints_root,
            cache_trainset_representation=bool(model_cfg.get("cache_trainset_representation", False)),
            strict=bool(model_cfg.get("strict_load", True)),
        )
        return TabPFNSurrogate(backbone=backbone)

    if backend == "tabicl":
        backbone = create_backbone(
            name=str(model_cfg.get("name", "tabicl")),
            checkpoint_path=model_cfg.get("checkpoint_path"),
            version=model_cfg.get("version"),
            device=cfg.device,
            checkpoints_root=cfg.paths.checkpoints_root,
            cache_trainset_representation=bool(model_cfg.get("cache_trainset_representation", False)),
            strict=bool(model_cfg.get("strict_load", True)),
        )
        return TabICLSurrogate(backbone=backbone)

    if backend == "gp":
        return GPyTorchSurrogate(
            kernel=str(model_cfg.kernel),
            training_steps=int(model_cfg.training_steps),
            learning_rate=float(model_cfg.learning_rate),
            initial_noise=float(model_cfg.initial_noise),
            device=str(cfg.device),
        )

    raise ValueError(f"Unsupported surrogate backend: {backend}")


def _resolve_policy_checkpoint_path(
    cfg: DictConfig,
    *,
    method_name: str,
    backend: str | None = None,
) -> str:
    checkpoints_cfg_obj = cfg.policy.get("checkpoints")
    checkpoints_cfg: dict[str, str]
    if checkpoints_cfg_obj is None:
        checkpoints_cfg = {}
    elif OmegaConf.is_config(checkpoints_cfg_obj):
        raw = OmegaConf.to_container(checkpoints_cfg_obj, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError("policy.checkpoints must be a mapping.")
        checkpoints_cfg = {
            str(k).strip().lower(): str(v)
            for k, v in raw.items()
            if v is not None and str(v).strip()
        }
    elif isinstance(checkpoints_cfg_obj, dict):
        checkpoints_cfg = {
            str(k).strip().lower(): str(v)
            for k, v in checkpoints_cfg_obj.items()
            if v is not None and str(v).strip()
        }
    else:
        raise ValueError("policy.checkpoints must be a mapping.")

    method_key = str(method_name).strip().lower()
    path = checkpoints_cfg.get(method_key)
    if path:
        return path
    if backend is not None:
        backend_path = checkpoints_cfg.get(str(backend).strip().lower())
        if backend_path:
            return backend_path

    default_path = cfg.policy.get("default_checkpoint_path")
    if default_path is not None and str(default_path).strip():
        return str(default_path)

    raise ValueError(
        f"Method '{method_name}' needs a policy checkpoint. Provide one of:\n"
        f"- policy.checkpoints.{method_name}=...\n"
        "- policy.default_checkpoint_path=..."
    )


def _stable_method_seed(base_seed: int, method_name: str) -> int:
    digest = hashlib.sha256(method_name.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return int((int(base_seed) + offset) % (2**31 - 1))


def _validate_eval_task_config(cfg: DictConfig) -> None:
    task_name = str(cfg.task.name).strip().lower()
    if str(cfg.task.split).strip().lower() != "test":
        raise ValueError("AL evaluation expects task.split=test.")

    if task_name == "gp_regression":
        if not bool(getattr(cfg.task.gp_prior, "standardize_x", False)):
            raise ValueError("AL evaluation expects task.gp_prior.standardize_x=true.")
        if not bool(getattr(cfg.task.gp_prior, "standardize_y", False)):
            raise ValueError("AL evaluation expects task.gp_prior.standardize_y=true.")
        return

    if task_name == "benchmark_functions":
        if not bool(getattr(cfg.task, "normalize_x", False)):
            raise ValueError("AL benchmark evaluation expects task.normalize_x=true.")
        if not bool(getattr(cfg.task, "normalize_y", False)):
            raise ValueError("AL benchmark evaluation expects task.normalize_y=true.")
        return

    raise ValueError(
        "AL evaluation currently supports only task.name=gp_regression "
        "or task.name=benchmark_functions."
    )


def _metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "ci95": 0.0,
        }
    tensor = torch.tensor(values, dtype=torch.float64)
    count = int(tensor.numel())
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item()) if count > 1 else 0.0
    sem = float(std / math.sqrt(count)) if count > 0 else 0.0
    ci95 = float(1.96 * sem)
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95": ci95,
    }


def _summarize_method_results(
    *,
    round_metrics: list[dict[str, Any]],
    aggregated_rounds: list[dict[str, Any]],
) -> dict[str, float]:
    summary: dict[str, float] = {
        "num_rounds": float(len(aggregated_rounds)),
        "num_selected_queries": float(len(round_metrics)),
    }

    trajectory_metrics = (
        "target_rmse",
        "weighted_target_rmse",
        "target_log_prob",
        "weighted_target_log_prob",
        "selected_query_y",
        "selected_query_raw_y",
    )
    if aggregated_rounds:
        first = aggregated_rounds[0]
        last = aggregated_rounds[-1]
        for metric_name in trajectory_metrics:
            key = f"mean_{metric_name}"
            sem_key = f"sem_{metric_name}"
            if key not in first or key not in last:
                continue
            start = float(first[key])
            end = float(last[key])
            summary[f"{metric_name}_start"] = start
            summary[f"{metric_name}_end"] = end
            summary[f"{metric_name}_gain"] = end - start
            if sem_key in first:
                summary[f"{metric_name}_start_sem"] = float(first[sem_key])
            if sem_key in last:
                summary[f"{metric_name}_end_sem"] = float(last[sem_key])

    selected_y_values = [float(metric["selected_query_y"]) for metric in round_metrics]
    selected_y_stats = _metric_stats(selected_y_values)
    summary["selected_query_y_mean_overall"] = selected_y_stats["mean"]
    summary["selected_query_y_sem_overall"] = selected_y_stats["sem"]

    selected_raw_y_values = [
        float(metric["selected_query_raw_y"])
        for metric in round_metrics
        if metric.get("selected_query_raw_y") is not None
    ]
    selected_raw_y_stats = _metric_stats(selected_raw_y_values)
    summary["selected_query_raw_y_mean_overall"] = selected_raw_y_stats["mean"]
    summary["selected_query_raw_y_sem_overall"] = selected_raw_y_stats["sem"]
    return summary


def _format_mean_sem(summary: dict[str, float], mean_key: str, sem_key: str) -> str:
    """Format a summary statistic as ``mean+-sem`` for console logs."""
    mean = float(summary.get(mean_key, 0.0))
    sem = float(summary.get(sem_key, 0.0))
    return f"{mean:.4f}\u00b1{sem:.4f}"


def _select_benchmark_plot_episodes(
    episodes: list[Any],
    *,
    n_plot_episodes_per_dataset: int,
) -> list[tuple[str, int, Any]]:
    """Select the first n seeds per benchmark dataset for plotting."""
    if n_plot_episodes_per_dataset < 0:
        raise ValueError("n_plot_episodes_per_dataset must be >= 0.")
    selected: list[tuple[str, int, Any]] = []
    counts: dict[str, int] = {}
    for episode in episodes:
        dataset_id = str(getattr(episode, "dataset_id", "unknown"))
        count = counts.get(dataset_id, 0)
        if count >= n_plot_episodes_per_dataset:
            continue
        selected.append((dataset_id, count, episode))
        counts[dataset_id] = count + 1
    return selected


def _attach_benchmark_plot_bundle(episode: Any, plot_bundle: dict[str, object]) -> Any:
    """Return an episode-like object enriched with dense plot metadata."""
    raw_plot_x = torch.as_tensor(plot_bundle["raw_plot_x"], dtype=torch.float32)
    raw_plot_y_true = torch.as_tensor(plot_bundle["raw_plot_y_true"], dtype=torch.float32)
    plot_x = episode_raw_x_to_model(episode, raw_plot_x)

    y_mean = torch.as_tensor(getattr(episode, "y_mean", 0.0), dtype=torch.float32)
    y_std = torch.as_tensor(getattr(episode, "y_std", 1.0), dtype=torch.float32)
    plot_y_true = (raw_plot_y_true - y_mean) / y_std

    payload = dict(getattr(episode, "__dict__", {}))
    payload.update(
        {
            "plot_x": plot_x,
            "plot_y_true": plot_y_true,
            "raw_plot_x": raw_plot_x,
            "raw_plot_y_true": raw_plot_y_true,
        }
    )
    return SimpleNamespace(**payload)


def _gp_regression_probe_plots_enabled(task: Any) -> bool:
    """Return whether 1D GP probe-fit plots are supported for this task."""
    return int(getattr(task, "dim_x", 1)) == 1


@hydra.main(version_base="1.3", config_path="../config", config_name="exp/evaluate_al")
def main(cfg: DictConfig) -> None:
    print("==== AL Evaluation ====")
    print(OmegaConf.to_yaml(cfg))

    set_global_seed(int(cfg.seed))
    _validate_eval_task_config(cfg)

    methods = _cfg_str_list_or_none(cfg.compare.methods)
    if methods is None or not methods:
        raise ValueError("compare.methods must contain at least one method.")
    method_specs = [_parse_method_spec(cfg, method_name=method_name) for method_name in methods]

    run_tag = _build_run_tag(cfg)
    run_dir = _build_run_dir(cfg, run_tag)
    print(f"run_dir: {run_dir}")

    task = create_al_task(cfg.task, seed=int(cfg.seed))
    task_name = str(cfg.task.name).strip().lower()
    eval_dataset_ids = _cfg_str_list_or_none(cfg.eval.get("dataset_ids"))
    eval_seeds = _cfg_str_list_or_none(cfg.eval.get("seeds"))
    if eval_seeds is None:
        num_episodes = int(cfg.eval.num_episodes)
        if num_episodes <= 0:
            raise ValueError("eval.num_episodes must be positive.")
        eval_seeds = [f"episode{i}" for i in range(num_episodes)]

    episodes = list(
        task.iter_test_episodes(
            n_context=int(cfg.task.n_context_init),
            query_pool_size=int(cfg.task.n_query),
            n_target=int(cfg.task.n_target),
            dataset_ids=eval_dataset_ids,
            seeds=eval_seeds,
            shuffle_query_pool=bool(cfg.eval.shuffle_query_pool),
        )
    )
    if not episodes:
        raise RuntimeError("No test episodes were generated for evaluation.")
    target_weight_threshold = float(cfg.eval.get("target_weight_threshold", 0.0))

    plot_num_episodes = int(cfg.output.gp_plot_num_episodes)
    if plot_num_episodes < 0:
        raise ValueError("output.gp_plot_num_episodes must be >= 0.")
    plot_round_stride = int(cfg.output.gp_plot_round_stride)
    if plot_round_stride <= 0:
        raise ValueError("output.gp_plot_round_stride must be positive.")
    benchmark_plot_num_episodes = int(cfg.output.get("benchmark_plot_num_episodes_per_dataset", 1))
    if benchmark_plot_num_episodes < 0:
        raise ValueError("output.benchmark_plot_num_episodes_per_dataset must be >= 0.")
    benchmark_plot_round_stride = int(cfg.output.get("benchmark_plot_round_stride", 1))
    if benchmark_plot_round_stride <= 0:
        raise ValueError("output.benchmark_plot_round_stride must be positive.")

    plots_dir = run_dir / "plots"
    if bool(cfg.output.save_plots):
        plots_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "task": {
            "name": str(cfg.task.name),
            "split": str(cfg.task.split),
            "n_context_init": int(cfg.task.n_context_init),
            "n_query": int(cfg.task.n_query),
            "n_target": int(cfg.task.n_target),
            "dim_x": (
                int(cfg.task.dim_x if cfg.task.get("dim_x") is not None else cfg.task.dim_x_max)
                if task_name == "gp_regression"
                else None
            ),
            "standardize_x": (
                bool(cfg.task.gp_prior.standardize_x) if task_name == "gp_regression" else None
            ),
            "standardize_y": (
                bool(cfg.task.gp_prior.standardize_y) if task_name == "gp_regression" else None
            ),
            "benchmarks": list(cfg.task.benchmarks) if task_name == "benchmark_functions" else None,
            "normalize_x": bool(cfg.task.normalize_x) if task_name == "benchmark_functions" else None,
            "normalize_y": bool(cfg.task.normalize_y) if task_name == "benchmark_functions" else None,
        },
        "eval": {
            "num_episodes": int(len(episodes)),
            "num_acquisitions": int(cfg.eval.num_acquisitions),
            "target_weighting": str(cfg.eval.target_weighting),
            "target_weight_beta": float(cfg.eval.target_weight_beta),
            "target_weight_threshold": target_weight_threshold,
            "selection": str(cfg.policy.selection),
            "plot_uncertainty": str(cfg.eval.plot_uncertainty),
            "benchmark_plot_num_episodes_per_dataset": float(benchmark_plot_num_episodes),
        },
        "run_dir": str(run_dir),
        "methods": {},
        "episodes": [
            {
                "index": int(idx),
                "dataset_id": str(getattr(episode, "dataset_id", "gp_prior")),
                "seed": str(getattr(episode, "seed", f"episode{idx}")),
            }
            for idx, episode in enumerate(episodes)
        ],
    }
    round_metrics_by_method: dict[str, list[dict[str, Any]]] = {}
    aggregated_by_method: dict[str, list[dict[str, Any]]] = {}
    plot_paths_by_method: dict[str, list[str]] = {}
    aggregated_by_dataset_by_method: dict[str, dict[str, list[dict[str, Any]]]] = {}

    gp_probe_plots_enabled = (
        task_name == "gp_regression" and _gp_regression_probe_plots_enabled(task)
    )
    if bool(cfg.output.save_plots) and task_name == "gp_regression" and not gp_probe_plots_enabled:
        print(
            "gp_probe_plots: skipped for gp_regression with dim_x>1; "
            "saving metric plots only."
        )

    plot_episodes = (
        list(enumerate(episodes[:plot_num_episodes]))
        if task_name == "gp_regression" and gp_probe_plots_enabled
        else []
    )
    benchmark_plot_episodes = (
        _select_benchmark_plot_episodes(
            episodes,
            n_plot_episodes_per_dataset=benchmark_plot_num_episodes,
        )
        if task_name == "benchmark_functions"
        else []
    )
    for spec in method_specs:
        method_name = spec.name
        backend = spec.backend
        model_cfg = _model_cfg_for_backend(cfg, backend)

        method_seed = _stable_method_seed(int(cfg.seed), f"{method_name}:eval")
        set_global_seed(method_seed)

        method_metadata: dict[str, Any] = {
            "mode": spec.mode,
            "backend": backend,
            "policy_label": spec.label,
            "acquisition_name": spec.acquisition,
        }

        if spec.mode == "policy":
            checkpoint_path = _resolve_policy_checkpoint_path(
                cfg,
                method_name=method_name,
                backend=backend,
            )
            loaded_policy = load_policy_checkpoint(
                backend=backend,
                model_cfg=model_cfg,
                policy_cfg=cfg.policy,
                device=str(cfg.device),
                checkpoints_root=cfg.paths.checkpoints_root,
                policy_checkpoint_path=checkpoint_path,
                strict_backbone_state=bool(cfg.policy.strict_backbone_state),
            )
            round_metrics = evaluate_policy_episodes(
                episodes=episodes,
                backbone=loaded_policy.backbone,
                policy_head=loaded_policy.policy_head,
                num_acquisitions=int(cfg.eval.num_acquisitions),
                selection=str(cfg.policy.selection),
                policy_target_weighting=str(cfg.eval.target_weighting),
                policy_target_weight_beta=float(cfg.eval.target_weight_beta),
                policy_target_weight_threshold=target_weight_threshold,
            )
            method_metadata.update(
                {
                    "policy_checkpoint_path": loaded_policy.policy_checkpoint_path,
                    "backbone_checkpoint_path": loaded_policy.backbone_checkpoint_path,
                    "backbone_state_source": loaded_policy.backbone_state_source,
                }
            )
        elif spec.mode == "surrogate":
            surrogate = _build_surrogate(cfg, backend=backend)
            round_metrics = evaluate_surrogate_episodes(
                episodes=episodes,
                surrogate=surrogate,
                acquisition_name=str(spec.acquisition),
                num_acquisitions=int(cfg.eval.num_acquisitions),
                policy_target_weighting=str(cfg.eval.target_weighting),
                policy_target_weight_beta=float(cfg.eval.target_weight_beta),
                policy_target_weight_threshold=target_weight_threshold,
            )
            method_metadata.update(
                {
                    "policy_checkpoint_path": None,
                    "backbone_checkpoint_path": None,
                    "backbone_state_source": "surrogate",
                }
            )
        else:
            raise ValueError(f"Unsupported method mode '{spec.mode}' for '{method_name}'.")

        aggregated_rounds = aggregate_al_eval_metrics(round_metrics)
        aggregated_rounds_by_dataset = aggregate_al_eval_metrics_by_dataset(round_metrics)
        method_summary = _summarize_method_results(
            round_metrics=round_metrics,
            aggregated_rounds=aggregated_rounds,
        )

        print(
            " | ".join(
                [
                    f"method={method_name}",
                    f"mode={spec.mode}",
                    f"backend={backend}",
                    "weighted_target_rmse_end="
                    f"{_format_mean_sem(method_summary, 'weighted_target_rmse_end', 'weighted_target_rmse_end_sem')}",
                    "target_rmse_end="
                    f"{_format_mean_sem(method_summary, 'target_rmse_end', 'target_rmse_end_sem')}",
                    "weighted_target_log_prob_end="
                    f"{_format_mean_sem(method_summary, 'weighted_target_log_prob_end', 'weighted_target_log_prob_end_sem')}",
                    "target_log_prob_end="
                    f"{_format_mean_sem(method_summary, 'target_log_prob_end', 'target_log_prob_end_sem')}",
                    "selected_query_y_mean_overall="
                    f"{_format_mean_sem(method_summary, 'selected_query_y_mean_overall', 'selected_query_y_sem_overall')}",
                ]
            )
        )

        method_plot_paths: list[str] = []
        if bool(cfg.output.save_plots):
            for episode_idx, episode in plot_episodes:
                plot_seed = _stable_method_seed(
                    int(cfg.seed),
                    f"{method_name}:{getattr(episode, 'seed', episode_idx)}:plot",
                )
                set_global_seed(plot_seed)
                if spec.mode == "policy":
                    _, round_snapshots = rollout_policy_probe_episode(
                        episode=episode,
                        backbone=loaded_policy.backbone,
                        policy_head=loaded_policy.policy_head,
                        num_acquisitions=int(cfg.eval.num_acquisitions),
                        selection=str(cfg.policy.selection),
                        collect_snapshots=True,
                        snapshot_round_stride=plot_round_stride,
                        policy_target_weighting=str(cfg.eval.target_weighting),
                        policy_target_weight_beta=float(cfg.eval.target_weight_beta),
                        policy_target_weight_threshold=target_weight_threshold,
                    )
                else:
                    surrogate = _build_surrogate(cfg, backend=backend)
                    _, round_snapshots = rollout_surrogate_probe_episode(
                        episode=episode,
                        surrogate=surrogate,
                        acquisition_name=str(spec.acquisition),
                        num_acquisitions=int(cfg.eval.num_acquisitions),
                        collect_snapshots=True,
                        snapshot_round_stride=plot_round_stride,
                        policy_target_weighting=str(cfg.eval.target_weighting),
                        policy_target_weight_beta=float(cfg.eval.target_weight_beta),
                        policy_target_weight_threshold=target_weight_threshold,
                    )
                episode_seed = str(getattr(episode, "seed", f"episode{episode_idx}"))
                save_path = plots_dir / (
                    f"gp_fit_episode_{episode_idx:03d}_{episode_seed}_{method_name}.png"
                )
                plot_policy_probe_rounds_grid(
                    save_path=save_path,
                    round_snapshots=round_snapshots,
                    method_name=f"{method_name} | {episode_seed}",
                    ci_std=float(cfg.eval.confidence_interval_std),
                )
                method_plot_paths.append(str(save_path))
                print(f"gp_plot[{method_name}]: {save_path}")

        round_metrics_by_method[method_name] = round_metrics
        aggregated_by_method[method_name] = aggregated_rounds
        aggregated_by_dataset_by_method[method_name] = aggregated_rounds_by_dataset
        plot_paths_by_method[method_name] = method_plot_paths
        results["methods"][method_name] = {
            **method_metadata,
            "summary": method_summary,
            "aggregated_rounds": aggregated_rounds,
            "aggregated_rounds_by_dataset": aggregated_rounds_by_dataset,
            "plot_paths": method_plot_paths,
        }

        if bool(cfg.output.save_plots) and task_name == "benchmark_functions":
            for dataset_id, _, episode in benchmark_plot_episodes:
                plot_bundle = task.plot_bundle(str(dataset_id))
                episode_for_plot = _attach_benchmark_plot_bundle(episode, plot_bundle)
                plot_seed = _stable_method_seed(
                    int(cfg.seed),
                    f"{method_name}:{dataset_id}:{getattr(episode, 'seed', 'episode')}:benchmark_plot",
                )
                set_global_seed(plot_seed)

                if spec.mode == "policy":
                    _, round_snapshots = rollout_policy_probe_episode(
                        episode=episode_for_plot,
                        backbone=loaded_policy.backbone,
                        policy_head=loaded_policy.policy_head,
                        num_acquisitions=int(cfg.eval.num_acquisitions),
                        selection=str(cfg.policy.selection),
                        collect_snapshots=True,
                        snapshot_round_stride=benchmark_plot_round_stride,
                        policy_target_weighting=str(cfg.eval.target_weighting),
                        policy_target_weight_beta=float(cfg.eval.target_weight_beta),
                        policy_target_weight_threshold=target_weight_threshold,
                    )
                else:
                    surrogate = _build_surrogate(cfg, backend=backend)
                    _, round_snapshots = rollout_surrogate_probe_episode(
                        episode=episode_for_plot,
                        surrogate=surrogate,
                        acquisition_name=str(spec.acquisition),
                        num_acquisitions=int(cfg.eval.num_acquisitions),
                        collect_snapshots=True,
                        snapshot_round_stride=benchmark_plot_round_stride,
                        policy_target_weighting=str(cfg.eval.target_weighting),
                        policy_target_weight_beta=float(cfg.eval.target_weight_beta),
                        policy_target_weight_threshold=target_weight_threshold,
                    )

                if not round_snapshots:
                    continue
                episode_seed = str(getattr(episode, "seed", "episode"))
                save_path = plots_dir / f"benchmark_fit_{dataset_id}_{episode_seed}_{method_name}.png"
                if str(plot_bundle.get("kind", "line")) == "line":
                    plot_policy_probe_rounds_grid(
                        save_path=save_path,
                        round_snapshots=round_snapshots,
                        method_name=f"{method_name} | {dataset_id} | {episode_seed}",
                        ci_std=float(cfg.eval.confidence_interval_std),
                    )
                else:
                    grid_shape = plot_bundle.get("grid_shape")
                    if not isinstance(grid_shape, tuple) or len(grid_shape) != 2:
                        raise ValueError(
                            f"2D benchmark plot bundle for {dataset_id} must define grid_shape."
                        )
                    _plot_benchmark_surface_rounds_grid(
                        save_path=save_path,
                        round_snapshots=round_snapshots,
                        method_name=method_name,
                        dataset_id=str(dataset_id),
                        episode_seed=episode_seed,
                        grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
                    )
                method_plot_paths.append(str(save_path))
                print(f"benchmark_plot[{method_name}]: {save_path}")

    metric_plot_path: Path | None = None
    if bool(cfg.output.save_plots):
        metric_plot_path = plots_dir / "al_eval_metrics.png"
        _plot_metric_grid(
            save_path=metric_plot_path,
            aggregated_by_method=aggregated_by_method,
            uncertainty_mode=str(cfg.eval.plot_uncertainty),
        )
        print(f"metric_plot: {metric_plot_path}")
        if task_name == "benchmark_functions":
            dataset_ids = sorted({str(getattr(episode, "dataset_id", "unknown")) for episode in episodes})
            for dataset_id in dataset_ids:
                by_method_for_dataset = {
                    method_name: by_dataset[dataset_id]
                    for method_name, by_dataset in aggregated_by_dataset_by_method.items()
                    if dataset_id in by_dataset
                }
                if not by_method_for_dataset:
                    continue
                dataset_metric_plot = plots_dir / f"al_eval_metrics_{dataset_id}.png"
                _plot_metric_grid(
                    save_path=dataset_metric_plot,
                    aggregated_by_method=by_method_for_dataset,
                    uncertainty_mode=str(cfg.eval.plot_uncertainty),
                )
                print(f"metric_plot[{dataset_id}]: {dataset_metric_plot}")

    results["metric_plot_path"] = str(metric_plot_path) if metric_plot_path is not None else None

    if bool(cfg.output.save_metrics):
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        round_metrics_path = run_dir / "round_metrics.json"
        with round_metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(round_metrics_by_method, handle, indent=2)

        print(f"metrics_json: {metrics_path}")
        print(f"round_metrics_json: {round_metrics_path}")


if __name__ == "__main__":
    main()
