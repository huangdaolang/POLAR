#!/usr/bin/env python
"""Evaluate BO methods (policy, surrogate, random) on shared test episodes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from polar.backbones import create_backbone
from polar.backbones.tabpfn.checkpoint_paths import get_path_layout
from polar.utils.policy_loading import (
    LoadedPolicy,
    extract_backbone_state_dict,
    extract_payload_model_backend,
    load_checkpoint_payload,
    load_policy_checkpoint,
    resolve_backbone_checkpoint_path,
)
from polar.tasks import BOEpisode, create_bo_task
from polar.trainers import CandidatePolicyHead, score_policy_head
from polar.utils.bo_acquisitions import (
    BOAcquisitionInputs,
    compute_bo_acquisition_scores,
    list_bo_acquisitions,
)
from polar.utils.plotting import plot_bo_comparison_curves as _plot_comparison_curves
from polar.utils.method_names import (
    MethodSpec,
    normalize_backend_name as _normalize_backend_name,
    parse_method,
)
from polar.utils.seeding import set_global_seed
from polar.utils.surrogates import (
    GPyTorchSurrogate,
    RandomForestSurrogate,
    Surrogate,
    TabICLSurrogate,
    TabPFNSurrogate,
)

PolicySelection = Literal["argmax", "sample"]
UncertaintyMode = Literal["ci95", "std", "sem"]
MetricName = Literal["regret", "normalized_regret"]
CurveMetricName = Literal[
    "best_y",
    "best_y_improvement",
    "oracle_best_y",
    "initial_best_y",
    "regret",
    "normalized_regret",
    "selected_y",
    "selected_score",
    "selected_prob",
    "policy_entropy",
    "query_size",
    "context_size",
]


@dataclass
class LoadedSurrogate:
    """Loaded surrogate components and backbone provenance."""

    surrogate: Surrogate
    policy_checkpoint_path: str | None
    backbone_checkpoint_path: str | None
    backbone_state_source: str


def _build_run_dir(cfg: DictConfig, results_root: Path) -> Path:
    suffix = str(cfg.output.run_tag) if cfg.output.run_tag else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / str(cfg.output.experiment_name) / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _as_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _available_backends(cfg: DictConfig) -> set[str]:
    models_cfg = OmegaConf.to_container(cfg.models, resolve=True)
    if not isinstance(models_cfg, dict):
        raise ValueError("models config must be a mapping.")
    return {_normalize_backend_name(str(name)) for name in models_cfg.keys()}


def _parse_method_spec(
    raw: str,
    *,
    acquisitions: list[str],
    backends: set[str],
    policy_backend: str | None = None,
) -> MethodSpec:
    spec = parse_method(
        raw,
        backends=backends,
        default_backend=policy_backend,
        acquisitions=acquisitions,
    )
    if spec.acquisition == "ei" and spec.backend not in {"gp", "rf"}:
        raise ValueError(
            f"Method '{spec.name}' is invalid: analytical acquisition 'ei' only supports "
            "backend in {gp, rf}. Use '<backend>_ei_mc' for other backbones."
        )
    return spec


def _model_cfg_for_backend(cfg: DictConfig, backend: str) -> DictConfig:
    backend_cfg = cfg.models.get(backend)
    if backend_cfg is None:
        available = ", ".join(sorted(_available_backends(cfg)))
        raise ValueError(f"Missing models.{backend} config. Available: {available}")
    return backend_cfg


def _parse_policy_checkpoint_config(
    cfg: DictConfig,
) -> tuple[dict[str, str], str | None]:
    checkpoints_cfg_obj = cfg.policy.checkpoints
    if isinstance(checkpoints_cfg_obj, str):
        single_path = checkpoints_cfg_obj.strip()
        checkpoints_cfg: dict[str, str] | None = {}
        default_path = single_path if single_path else None
    elif checkpoints_cfg_obj is None:
        checkpoints_cfg = None
        default_path = None
    elif OmegaConf.is_config(checkpoints_cfg_obj):
        checkpoints_cfg = OmegaConf.to_container(checkpoints_cfg_obj, resolve=True)
        default_path = None
    else:
        checkpoints_cfg = checkpoints_cfg_obj
        default_path = None

    if checkpoints_cfg is None:
        checkpoints: dict[str, str] = {}
    elif isinstance(checkpoints_cfg, dict):
        checkpoints = {
            str(k).strip().lower(): str(v)
            for k, v in checkpoints_cfg.items()
            if v is not None and str(v).strip()
        }
    else:
        raise ValueError("policy.checkpoints must be a mapping.")

    config_default_path = cfg.policy.default_checkpoint_path
    if config_default_path is not None and str(config_default_path).strip():
        default_path = str(config_default_path)

    return checkpoints, default_path


def _maybe_resolve_policy_checkpoint_path(
    cfg: DictConfig,
    *,
    method_name: str,
    backend: str,
) -> str | None:
    checkpoints, default_path = _parse_policy_checkpoint_config(cfg)
    for key in (method_name, backend):
        path = checkpoints.get(key)
        if path:
            return path

    return default_path


def _resolve_policy_checkpoint_path(
    cfg: DictConfig,
    *,
    method_name: str,
    backend: str,
) -> str:
    policy_checkpoint_path = _maybe_resolve_policy_checkpoint_path(
        cfg,
        method_name=method_name,
        backend=backend,
    )
    if policy_checkpoint_path is not None:
        return policy_checkpoint_path

    raise ValueError(
        f"Method '{method_name}' needs a policy checkpoint. Provide one of:\n"
        f"- policy.checkpoints.{method_name}=...\n"
        f"- policy.checkpoints.{backend}=...\n"
        "- policy.default_checkpoint_path=..."
    )


def _load_policy(
    cfg: DictConfig,
    *,
    method_name: str,
    backend: str,
) -> LoadedPolicy:
    if backend in {"gp", "rf"}:
        raise ValueError(f"Method '{method_name}' is invalid: backend={backend} has no policy backbone.")

    model_cfg = _model_cfg_for_backend(cfg, backend)

    policy_checkpoint_str = _resolve_policy_checkpoint_path(
        cfg,
        method_name=method_name,
        backend=backend,
    )
    return load_policy_checkpoint(
        backend=backend,
        model_cfg=model_cfg,
        policy_cfg=cfg.policy,
        device=cfg.device,
        checkpoints_root=cfg.paths.checkpoints_root,
        policy_checkpoint_path=policy_checkpoint_str,
        strict_backbone_state=bool(cfg.policy.strict_backbone_state),
    )


def _build_surrogate(
    cfg: DictConfig,
    *,
    method_name: str,
    backend: str,
    random_state: int | None = None,
) -> LoadedSurrogate:
    backend = _normalize_backend_name(backend)
    model_cfg = _model_cfg_for_backend(cfg, backend)

    max_query_chunk_cfg = cfg.eval.get("max_query_chunk", 0)
    max_query_chunk = int(max_query_chunk_cfg) if max_query_chunk_cfg else 0
    chunk_kwarg: dict[str, int] = {"max_query_chunk": max_query_chunk}

    if backend == "gp":
        return LoadedSurrogate(
            surrogate=GPyTorchSurrogate(
                kernel=str(model_cfg.kernel),
                training_steps=int(model_cfg.training_steps),
                learning_rate=float(model_cfg.learning_rate),
                initial_noise=float(model_cfg.initial_noise),
                device=str(cfg.device),
            ),
            policy_checkpoint_path=None,
            backbone_checkpoint_path=None,
            backbone_state_source="not_applicable",
        )

    if backend == "rf":
        model_random_state = model_cfg.get("random_state")
        resolved_random_state = (
            int(model_random_state)
            if model_random_state is not None
            else (int(random_state) if random_state is not None else None)
        )
        return LoadedSurrogate(
            surrogate=RandomForestSurrogate(
                n_estimators=int(model_cfg.n_estimators),
                max_depth=(int(model_cfg.max_depth) if model_cfg.get("max_depth") is not None else None),
                min_samples_leaf=int(model_cfg.min_samples_leaf),
                max_features=model_cfg.get("max_features"),
                bootstrap=bool(model_cfg.get("bootstrap", True)),
                variance_floor=float(model_cfg.get("variance_floor", 1e-6)),
                n_jobs=(int(model_cfg.n_jobs) if model_cfg.get("n_jobs") is not None else None),
                random_state=resolved_random_state,
                device=str(cfg.device),
            ),
            policy_checkpoint_path=None,
            backbone_checkpoint_path=None,
            backbone_state_source="not_applicable",
        )

    policy_checkpoint_str = _maybe_resolve_policy_checkpoint_path(
        cfg,
        method_name=method_name,
        backend=backend,
    )
    policy_payload: dict[str, Any] | None = None
    policy_checkpoint_path: str | None = None
    if policy_checkpoint_str is not None:
        resolved_policy_ckpt = Path(policy_checkpoint_str).expanduser().resolve()
        policy_payload = load_checkpoint_payload(resolved_policy_ckpt)
        payload_backend_raw = extract_payload_model_backend(policy_payload)
        payload_backend = (
            _normalize_backend_name(payload_backend_raw)
            if payload_backend_raw is not None
            else None
        )
        if payload_backend is not None and payload_backend != backend:
            raise ValueError(
                f"Surrogate method '{method_name}' requires backend='{backend}', but policy checkpoint "
                f"was trained with model.name='{payload_backend}'. Please provide a matching checkpoint."
            )
        policy_checkpoint_path = str(resolved_policy_ckpt)

    backbone_checkpoint_path = (
        resolve_backbone_checkpoint_path(model_cfg=model_cfg, payload=policy_payload)
        if policy_payload is not None
        else model_cfg.get("checkpoint_path")
    )
    backbone = create_backbone(
        name=str(model_cfg.get("name", backend)),
        checkpoint_path=backbone_checkpoint_path,
        version=model_cfg.get("version"),
        device=cfg.device,
        checkpoints_root=cfg.paths.checkpoints_root,
        cache_trainset_representation=bool(model_cfg.get("cache_trainset_representation", False)),
        strict=bool(model_cfg.get("strict_load", True)),
    )
    backbone_state_source = "pretrained_backbone"
    if policy_payload is not None:
        maybe_backbone_state = extract_backbone_state_dict(policy_payload)
        if maybe_backbone_state is None:
            print(
                f"warning: policy checkpoint for surrogate method '{method_name}' does not contain "
                "backbone_model_state_dict; using pretrained backbone weights."
            )
        else:
            backbone.model.load_state_dict(
                maybe_backbone_state,
                strict=bool(cfg.policy.strict_backbone_state),
            )
            backbone_state_source = "checkpoint_payload"
    backbone.model.eval()

    if backend == "tabpfn":
        return LoadedSurrogate(
            surrogate=TabPFNSurrogate(backbone=backbone, **chunk_kwarg),
            policy_checkpoint_path=policy_checkpoint_path,
            backbone_checkpoint_path=str(backbone.checkpoint_path) if backbone.checkpoint_path else None,
            backbone_state_source=backbone_state_source,
        )

    if backend == "tabicl":
        return LoadedSurrogate(
            surrogate=TabICLSurrogate(backbone=backbone, **chunk_kwarg),
            policy_checkpoint_path=policy_checkpoint_path,
            backbone_checkpoint_path=str(backbone.checkpoint_path) if backbone.checkpoint_path else None,
            backbone_state_source=backbone_state_source,
        )

    raise ValueError(f"Unsupported surrogate backend: {backend}")


def _round_metric(
    *,
    method_name: str,
    episode: BOEpisode,
    round_idx: int,
    context_size: int,
    query_size: int,
    best_y: float,
    initial_best_y: float,
    selected_y: float | None,
    selected_score: float | None,
    selected_prob: float | None,
    policy_entropy: float | None,
) -> dict[str, Any]:
    best = float(episode.y_best)
    y_min = float(episode.y_min)
    regret = max(0.0, best - best_y)
    denom = max(best - y_min, 1e-12)
    normalized_regret = regret / denom

    return {
        "method": method_name,
        "search_space_id": episode.search_space_id,
        "dataset_id": episode.dataset_id,
        "seed": episode.seed,
        "round": int(round_idx),
        "context_size": int(context_size),
        "query_size": int(query_size),
        "best_y": float(best_y),
        "initial_best_y": float(initial_best_y),
        "best_y_improvement": float(best_y - initial_best_y),
        "oracle_best_y": best,
        "regret": float(regret),
        "normalized_regret": float(normalized_regret),
        "selected_y": selected_y,
        "selected_score": selected_score,
        "selected_prob": selected_prob,
        "policy_entropy": policy_entropy,
    }


def _evaluate_one_episode_policy(
    *,
    method_name: str,
    episode: BOEpisode,
    backbone,
    policy_head: CandidatePolicyHead,
    num_acquisitions: int,
    selection: PolicySelection,
    max_query_chunk: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context_x = episode.context_x.to(backbone.device).float().clone()
    context_y = episode.context_y.to(backbone.device).float().clone()
    query_x = episode.query_x.to(backbone.device).float().clone()
    query_y = episode.query_y.to(backbone.device).float().clone()

    initial_best_y = _as_float(torch.max(context_y))
    if initial_best_y is None:
        raise RuntimeError("Failed to compute initial best_y.")

    metrics: list[dict[str, Any]] = [
        _round_metric(
            method_name=method_name,
            episode=episode,
            round_idx=0,
            context_size=int(context_x.shape[0]),
            query_size=int(query_x.shape[0]),
            best_y=initial_best_y,
            initial_best_y=initial_best_y,
            selected_y=None,
            selected_score=None,
            selected_prob=None,
            policy_entropy=None,
        )
    ]

    chunk = int(max_query_chunk) if max_query_chunk and int(max_query_chunk) > 0 else 0

    max_rounds = min(int(num_acquisitions), int(query_x.shape[0]))
    for round_idx in range(1, max_rounds + 1):
        with torch.no_grad():
            context_state = backbone.encode_context(context_x, context_y)

            n_query = int(query_x.shape[0])
            if chunk <= 0 or n_query <= chunk:
                outputs = backbone.forward_with_embeddings(query_x, context_state)
                query_embeddings = outputs.get("test_embeddings")
                if query_embeddings is None:
                    raise KeyError("forward_with_embeddings must return 'test_embeddings'.")
            else:
                emb_parts: list[torch.Tensor] = []
                for start in range(0, n_query, chunk):
                    chunk_out = backbone.forward_with_embeddings(
                        query_x[start : start + chunk], context_state
                    )
                    chunk_emb = chunk_out.get("test_embeddings")
                    if chunk_emb is None:
                        raise KeyError("forward_with_embeddings must return 'test_embeddings'.")
                    emb_parts.append(chunk_emb)
                query_embeddings = torch.cat(emb_parts, dim=0)

            policy_scores = score_policy_head(
                policy_head,
                query_embeddings=query_embeddings,
                remaining_budget=max_rounds - round_idx + 1,
            )
            action_dist = torch.distributions.Categorical(logits=policy_scores)
            if selection == "sample":
                selected_idx = int(action_dist.sample().item())
            elif selection == "argmax":
                selected_idx = int(torch.argmax(policy_scores).item())
            else:
                raise ValueError(f"Unsupported selection mode: {selection}")

            policy_probs = torch.softmax(policy_scores, dim=0)
            selected_prob = _as_float(policy_probs[selected_idx])
            selected_score = _as_float(policy_scores[selected_idx])
            policy_entropy = _as_float(action_dist.entropy())

        selected_x = query_x[selected_idx : selected_idx + 1]
        selected_y_tensor = query_y[selected_idx : selected_idx + 1]
        selected_y = _as_float(selected_y_tensor.squeeze())

        context_x = torch.cat([context_x, selected_x], dim=0)
        context_y = torch.cat([context_y, selected_y_tensor], dim=0)

        keep_mask = torch.ones(query_x.shape[0], dtype=torch.bool, device=query_x.device)
        keep_mask[selected_idx] = False
        query_x = query_x[keep_mask]
        query_y = query_y[keep_mask]

        best_y = _as_float(torch.max(context_y))
        if best_y is None:
            raise RuntimeError("Failed to compute best_y.")

        metrics.append(
            _round_metric(
                method_name=method_name,
                episode=episode,
                round_idx=round_idx,
                context_size=int(context_x.shape[0]),
                query_size=int(query_x.shape[0]),
                best_y=best_y,
                initial_best_y=initial_best_y,
                selected_y=selected_y,
                selected_score=selected_score,
                selected_prob=selected_prob,
                policy_entropy=policy_entropy,
            )
        )
        if int(query_x.shape[0]) == 0:
            break

    summary = {
        "search_space_id": episode.search_space_id,
        "dataset_id": episode.dataset_id,
        "seed": episode.seed,
        "num_rounds": len(metrics) - 1,
        "best_y_start": metrics[0]["best_y"],
        "best_y_end": metrics[-1]["best_y"],
        "regret_start": metrics[0]["regret"],
        "regret_end": metrics[-1]["regret"],
        "normalized_regret_start": metrics[0]["normalized_regret"],
        "normalized_regret_end": metrics[-1]["normalized_regret"],
    }
    return metrics, summary


def _evaluate_one_episode_surrogate(
    *,
    method_name: str,
    episode: BOEpisode,
    surrogate: Surrogate | None,
    acquisition_name: str,
    num_acquisitions: int,
    ei_mc_samples: int,
    maximize: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context_x = episode.context_x.clone()
    context_y = episode.context_y.clone()
    query_x = episode.query_x.clone()
    query_y = episode.query_y.clone()

    initial_best_y = _as_float(torch.max(context_y))
    if initial_best_y is None:
        raise RuntimeError("Failed to compute initial best_y.")

    metrics: list[dict[str, Any]] = [
        _round_metric(
            method_name=method_name,
            episode=episode,
            round_idx=0,
            context_size=int(context_x.shape[0]),
            query_size=int(query_x.shape[0]),
            best_y=initial_best_y,
            initial_best_y=initial_best_y,
            selected_y=None,
            selected_score=None,
            selected_prob=None,
            policy_entropy=None,
        )
    ]

    max_rounds = min(int(num_acquisitions), int(query_x.shape[0]))
    for round_idx in range(1, max_rounds + 1):
        if acquisition_name == "random":
            selected_idx = int(torch.randint(int(query_x.shape[0]), (1,)).item())
            selected_score = None
        else:
            if surrogate is None:
                raise RuntimeError("Surrogate must be available when acquisition is not random.")

            surrogate.fit(context_x, context_y)
            pred = surrogate.predict(query_x)

            predictive_samples = None
            if acquisition_name == "ei_mc":
                predictive_samples = surrogate.sample(query_x, num_samples=int(ei_mc_samples))

            acq_inputs = BOAcquisitionInputs(
                candidate_x=query_x.to(surrogate.device),
                predictive_mean=pred.mean,
                predictive_variance=pred.variance,
                best_f=torch.max(context_y).to(surrogate.device),
                maximize=bool(maximize),
                predictive_samples=predictive_samples,
                mc_num_samples=int(ei_mc_samples),
            )
            acq_scores = compute_bo_acquisition_scores(name=acquisition_name, inputs=acq_inputs)
            if acq_scores.numel() == 0:
                break

            select_fn = torch.argmax if maximize else torch.argmin
            selected_idx = int(select_fn(acq_scores).item())
            selected_score = _as_float(acq_scores[selected_idx])

        selected_x = query_x[selected_idx : selected_idx + 1]
        selected_y_tensor = query_y[selected_idx : selected_idx + 1]
        selected_y = _as_float(selected_y_tensor.squeeze())

        context_x = torch.cat([context_x, selected_x], dim=0)
        context_y = torch.cat([context_y, selected_y_tensor], dim=0)

        keep_mask = torch.ones(query_x.shape[0], dtype=torch.bool)
        keep_mask[selected_idx] = False
        query_x = query_x[keep_mask]
        query_y = query_y[keep_mask]

        best_y = _as_float(torch.max(context_y))
        if best_y is None:
            raise RuntimeError("Failed to compute best_y.")

        metrics.append(
            _round_metric(
                method_name=method_name,
                episode=episode,
                round_idx=round_idx,
                context_size=int(context_x.shape[0]),
                query_size=int(query_x.shape[0]),
                best_y=best_y,
                initial_best_y=initial_best_y,
                selected_y=selected_y,
                selected_score=selected_score,
                selected_prob=None,
                policy_entropy=None,
            )
        )
        if int(query_x.shape[0]) == 0:
            break

    summary = {
        "search_space_id": episode.search_space_id,
        "dataset_id": episode.dataset_id,
        "seed": episode.seed,
        "num_rounds": len(metrics) - 1,
        "best_y_start": metrics[0]["best_y"],
        "best_y_end": metrics[-1]["best_y"],
        "regret_start": metrics[0]["regret"],
        "regret_end": metrics[-1]["regret"],
        "normalized_regret_start": metrics[0]["normalized_regret"],
        "normalized_regret_end": metrics[-1]["normalized_regret"],
    }
    return metrics, summary


def _aggregate_round_metrics(round_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not round_metrics:
        return []

    rounds = sorted({int(m["round"]) for m in round_metrics})
    aggregated: list[dict[str, Any]] = []
    for round_idx in rounds:
        bucket = [m for m in round_metrics if int(m["round"]) == round_idx]
        regrets = torch.tensor([float(m["regret"]) for m in bucket], dtype=torch.float64)
        norm_regrets = torch.tensor([float(m["normalized_regret"]) for m in bucket], dtype=torch.float64)
        best_ys = torch.tensor([float(m["best_y"]) for m in bucket], dtype=torch.float64)
        best_y_improvements = torch.tensor(
            [float(m["best_y_improvement"]) for m in bucket],
            dtype=torch.float64,
        )

        n = int(regrets.numel())
        regret_mean = float(regrets.mean().item())
        regret_std = float(regrets.std(unbiased=False).item()) if n > 1 else 0.0
        regret_sem = float(regret_std / math.sqrt(n)) if n > 0 else 0.0
        regret_ci95 = float(1.96 * regret_sem)

        norm_regret_mean = float(norm_regrets.mean().item())
        norm_regret_std = float(norm_regrets.std(unbiased=False).item()) if n > 1 else 0.0
        norm_regret_sem = float(norm_regret_std / math.sqrt(n)) if n > 0 else 0.0
        norm_regret_ci95 = float(1.96 * norm_regret_sem)

        aggregated.append(
            {
                "round": int(round_idx),
                "num_episodes": n,
                "mean_best_y": float(best_ys.mean().item()),
                "mean_best_y_improvement": float(best_y_improvements.mean().item()),
                "mean_regret": regret_mean,
                "std_regret": regret_std,
                "sem_regret": regret_sem,
                "ci95_regret": regret_ci95,
                "mean_normalized_regret": norm_regret_mean,
                "std_normalized_regret": norm_regret_std,
                "sem_normalized_regret": norm_regret_sem,
                "ci95_normalized_regret": norm_regret_ci95,
            }
        )
    return aggregated


def _build_dense_curve_artifact(round_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not round_metrics:
        return {
            "dataset_ids": [],
            "seeds": [],
            "rounds": [],
            "shape": [0, 0, 0],
            "metrics": {},
        }

    dataset_ids = list(dict.fromkeys(str(item["dataset_id"]) for item in round_metrics))
    seeds = list(dict.fromkeys(str(item["seed"]) for item in round_metrics))
    rounds = sorted({int(item["round"]) for item in round_metrics})
    dataset_to_idx = {dataset_id: idx for idx, dataset_id in enumerate(dataset_ids)}
    seed_to_idx = {seed: idx for idx, seed in enumerate(seeds)}
    round_to_idx = {round_idx: idx for idx, round_idx in enumerate(rounds)}

    shape = (len(dataset_ids), len(seeds), len(rounds))
    metric_names: tuple[CurveMetricName, ...] = (
        "best_y",
        "best_y_improvement",
        "oracle_best_y",
        "initial_best_y",
        "regret",
        "normalized_regret",
        "selected_y",
        "selected_score",
        "selected_prob",
        "policy_entropy",
        "query_size",
        "context_size",
    )
    metrics = {
        metric_name: np.full(shape, np.nan, dtype=np.float32)
        for metric_name in metric_names
    }

    for item in round_metrics:
        dataset_idx = dataset_to_idx[str(item["dataset_id"])]
        seed_idx = seed_to_idx[str(item["seed"])]
        round_idx = round_to_idx[int(item["round"])]
        for metric_name in metric_names:
            raw_value = item.get(metric_name)
            if raw_value is None:
                continue
            metrics[metric_name][dataset_idx, seed_idx, round_idx] = float(raw_value)

    return {
        "dataset_ids": dataset_ids,
        "seeds": seeds,
        "rounds": rounds,
        "shape": list(shape),
        "metrics": metrics,
    }


def _save_method_curve_artifacts(
    *,
    method_dir: Path,
    method_name: str,
    search_space_id: str,
    round_metrics: list[dict[str, Any]],
    aggregate_by_round: list[dict[str, Any]],
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    method_dir.mkdir(parents=True, exist_ok=True)
    dense = _build_dense_curve_artifact(round_metrics)

    artifact_files: dict[str, str] = {}
    for metric_name, array in dense["metrics"].items():
        filename = f"{metric_name}.npy"
        np.save(method_dir / filename, array)
        artifact_files[metric_name] = filename

    metadata = {
        "method": method_name,
        "search_space_id": search_space_id,
        "dataset_ids": dense["dataset_ids"],
        "seeds": dense["seeds"],
        "rounds": dense["rounds"],
        "array_shape": dense["shape"],
        "includes_initial_round": bool(dense["rounds"] and dense["rounds"][0] == 0),
        "aggregate_by_round": aggregate_by_round,
        "artifact_files": artifact_files,
        **extra_metadata,
    }
    metadata_path = method_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "method_dir": str(method_dir),
        "metadata_path": str(metadata_path),
        "artifact_files": {key: str(method_dir / filename) for key, filename in artifact_files.items()},
    }


def _stable_method_seed(base_seed: int, method_name: str) -> int:
    digest = hashlib.sha256(method_name.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    # Keep seed within common 32-bit signed range.
    return int((int(base_seed) + offset) % (2**31 - 1))


def _print_method_summary(method_name: str, aggregate: list[dict[str, Any]]) -> None:
    if not aggregate:
        print(f"[{method_name}] aggregate: no data")
        return

    start = aggregate[0]
    end = aggregate[-1]
    print(
        f"[{method_name}] aggregate_regret:",
        f"round0={start['mean_regret']:.6f}",
        f"round{end['round']}={end['mean_regret']:.6f}",
    )
    print(
        f"[{method_name}] aggregate_norm_regret:",
        f"round0={start['mean_normalized_regret']:.6f}",
        f"round{end['round']}={end['mean_normalized_regret']:.6f}",
    )


@hydra.main(version_base="1.3", config_path="../config", config_name="exp/evaluate_bo")
def main(cfg: DictConfig) -> None:
    print(f"==== BO Evaluate ({cfg.task.name}) ====")
    print(OmegaConf.to_yaml(cfg))

    if str(cfg.task.split) != "test":
        raise ValueError("BO evaluation currently supports task.split=test only.")

    set_global_seed(int(cfg.seed))

    methods_cfg = OmegaConf.to_container(cfg.compare.methods, resolve=True)
    if not isinstance(methods_cfg, list) or not methods_cfg:
        raise ValueError("compare.methods must be a non-empty list.")

    available_backends = _available_backends(cfg)
    acquisitions = list_bo_acquisitions()
    method_specs = [
        _parse_method_spec(
            str(raw_method),
            acquisitions=acquisitions,
            backends=available_backends,
            policy_backend=str(cfg.policy.backend),
        )
        for raw_method in methods_cfg
    ]

    method_names = [spec.name for spec in method_specs]
    seen_method_names: set[str] = set()
    duplicate_names: set[str] = set()
    for name in method_names:
        if name in seen_method_names:
            duplicate_names.add(name)
        else:
            seen_method_names.add(name)
    if duplicate_names:
        dups = ", ".join(sorted(duplicate_names))
        raise ValueError(f"compare.methods contains duplicates: {dups}")

    metric = str(cfg.eval.metric).strip().lower()
    if metric not in {"regret", "normalized_regret"}:
        raise ValueError("eval.metric must be 'regret' or 'normalized_regret'.")

    uncertainty = str(cfg.eval.plot_uncertainty).strip().lower()
    if uncertainty not in {"ci95", "std", "sem"}:
        raise ValueError("eval.plot_uncertainty must be one of: ci95, std, sem.")
    uncertainty_mode: UncertaintyMode
    if uncertainty == "ci95":
        uncertainty_mode = "ci95"
    elif uncertainty == "std":
        uncertainty_mode = "std"
    else:
        uncertainty_mode = "sem"

    metric_name: MetricName
    if metric == "regret":
        metric_name = "regret"
    else:
        metric_name = "normalized_regret"

    layout = get_path_layout(
        checkpoints_root=cfg.paths.checkpoints_root,
        results_root=cfg.paths.results_root,
    )
    run_dir = _build_run_dir(cfg, layout.results_root)
    plots_dir = run_dir / "plots"
    if bool(cfg.output.save_plots):
        plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir: {run_dir}")

    task = create_bo_task(cfg.task, seed=int(cfg.seed))

    seeds_cfg = OmegaConf.to_container(cfg.eval.seeds, resolve=True)
    if not isinstance(seeds_cfg, list):
        raise ValueError("eval.seeds must be a list of seed names.")
    eval_seeds = [str(v) for v in seeds_cfg]

    dataset_ids_cfg_obj = cfg.eval.dataset_ids
    if dataset_ids_cfg_obj is None:
        eval_dataset_ids = None
    else:
        dataset_ids_cfg = OmegaConf.to_container(dataset_ids_cfg_obj, resolve=True)
        if not isinstance(dataset_ids_cfg, list):
            raise ValueError("eval.dataset_ids must be null or a list.")
        eval_dataset_ids = [str(v) for v in dataset_ids_cfg]

    episodes = list(
        task.iter_test_episodes(
            n_context=int(cfg.task.n_context_init),
            query_pool_size=int(cfg.eval.query_pool_size),
            seeds=eval_seeds,
            dataset_ids=eval_dataset_ids,
            shuffle_query_pool=bool(cfg.eval.shuffle_query_pool),
        )
    )
    if not episodes:
        raise RuntimeError(
            "No test episodes were created. Check task split settings, n_context_init, "
            "eval.query_pool_size, dataset_ids and eval.seeds."
        )

    print(
        "task:",
        f"name={cfg.task.name}",
        f"num_datasets={len({e.dataset_id for e in episodes})}",
        f"num_episodes={len(episodes)}",
        f"seeds={eval_seeds}",
    )
    print("methods:", ", ".join(method_names))

    num_acquisitions = int(cfg.eval.num_acquisitions)
    if num_acquisitions <= 0:
        raise ValueError("eval.num_acquisitions must be positive.")

    method_reports: dict[str, dict[str, Any]] = {}
    aggregate_by_method: dict[str, list[dict[str, Any]]] = {}

    # Track the currently loaded GPU models so we can free them between methods.
    _active_policy: LoadedPolicy | None = None
    _active_surrogate: Surrogate | None = None
    _active_backend: str | None = None

    def _release_gpu_models() -> None:
        """Move all cached models off GPU and free CUDA memory."""
        nonlocal _active_policy, _active_surrogate, _active_backend
        if _active_policy is not None:
            _active_policy.backbone.model.cpu()
            _active_policy.policy_head.cpu()
            _active_policy = None
        if _active_surrogate is not None:
            if hasattr(_active_surrogate, "backbone"):
                _active_surrogate.backbone.model.cpu()  # type: ignore[union-attr]
            _active_surrogate = None
        _active_backend = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for spec in method_specs:
        method_seed = _stable_method_seed(int(cfg.seed), spec.name)
        set_global_seed(method_seed)

        print(f"---- method={spec.name} seed={method_seed} ----")

        # Free previous method's GPU models before loading new ones.
        _release_gpu_models()

        all_round_metrics: list[dict[str, Any]] = []
        episode_summaries: list[dict[str, Any]] = []

        if spec.mode == "policy":
            if spec.backend is None:
                raise RuntimeError(f"Method '{spec.name}' has no backend.")

            loaded_policy = _load_policy(cfg, method_name=spec.name, backend=spec.backend)
            _active_policy = loaded_policy

            print(f"policy_checkpoint: {loaded_policy.policy_checkpoint_path}")
            print(f"policy_backbone_checkpoint: {loaded_policy.backbone_checkpoint_path}")
            print(f"policy_backbone_state: {loaded_policy.backbone_state_source}")

            for episode_idx, episode in enumerate(episodes):
                round_metrics, summary = _evaluate_one_episode_policy(
                    method_name=spec.name,
                    episode=episode,
                    backbone=loaded_policy.backbone,
                    policy_head=loaded_policy.policy_head,
                    num_acquisitions=num_acquisitions,
                    selection=str(cfg.policy.selection),
                    max_query_chunk=int(cfg.eval.get("max_query_chunk", 0)),
                )
                all_round_metrics.extend(round_metrics)
                episode_summaries.append(summary)
                print(
                    f"[episode={episode_idx:03d}]",
                    f"dataset={episode.dataset_id}",
                    f"seed={episode.seed}",
                    f"best_y={summary['best_y_start']:.6f}->{summary['best_y_end']:.6f}",
                    f"regret={summary['regret_start']:.6f}->{summary['regret_end']:.6f}",
                )

            aggregate = _aggregate_round_metrics(all_round_metrics)
            _print_method_summary(spec.name, aggregate)

            method_reports[spec.name] = {
                "mode": "policy",
                "backend": spec.backend,
                "acquisition": None,
                "seed": method_seed,
                "policy_checkpoint_path": loaded_policy.policy_checkpoint_path,
                "backbone_checkpoint_path": loaded_policy.backbone_checkpoint_path,
                "backbone_state_source": loaded_policy.backbone_state_source,
                "num_episodes": len(episodes),
                "num_acquisitions": num_acquisitions,
                "episode_summaries": episode_summaries,
                "aggregate_by_round": aggregate,
                "round_metrics": all_round_metrics,
            }
            aggregate_by_method[spec.name] = aggregate
            continue

        if spec.mode not in {"surrogate", "random"}:
            raise RuntimeError(f"Unknown method mode: {spec.mode}")

        acquisition_name = str(spec.acquisition) if spec.acquisition is not None else "random"
        surrogate: Surrogate | None
        if acquisition_name == "random":
            surrogate = None
            surrogate_backend_name = "none"
            print("surrogate_backend: skipped (random acquisition baseline)")
        else:
            if spec.backend is None:
                raise RuntimeError(f"Method '{spec.name}' has no backend.")
            loaded_surrogate = _build_surrogate(
                cfg,
                method_name=spec.name,
                backend=spec.backend,
                random_state=method_seed,
            )
            surrogate = loaded_surrogate.surrogate
            _active_surrogate = surrogate
            _active_backend = spec.backend
            surrogate_backend_name = surrogate.backend_name
            print(f"surrogate_backend: {surrogate_backend_name}")
            if surrogate_backend_name in {"tabpfn", "tabicl"}:
                print(f"surrogate_checkpoint: {loaded_surrogate.backbone_checkpoint_path}")
                print(f"surrogate_backbone_state: {loaded_surrogate.backbone_state_source}")
                print(f"surrogate_policy_checkpoint: {loaded_surrogate.policy_checkpoint_path}")

        if acquisition_name == "ei" and surrogate_backend_name not in {"gp", "rf"}:
            raise ValueError(
                f"Method '{spec.name}' is invalid: acquisition='ei' only supports backend in {{gp, rf}}."
            )

        for episode_idx, episode in enumerate(episodes):
            round_metrics, summary = _evaluate_one_episode_surrogate(
                method_name=spec.name,
                episode=episode,
                surrogate=surrogate,
                acquisition_name=acquisition_name,
                num_acquisitions=num_acquisitions,
                ei_mc_samples=int(cfg.eval.ei_mc_samples),
                maximize=bool(cfg.eval.maximize),
            )
            all_round_metrics.extend(round_metrics)
            episode_summaries.append(summary)
            print(
                f"[episode={episode_idx:03d}]",
                f"dataset={episode.dataset_id}",
                f"seed={episode.seed}",
                f"best_y={summary['best_y_start']:.6f}->{summary['best_y_end']:.6f}",
                f"regret={summary['regret_start']:.6f}->{summary['regret_end']:.6f}",
            )

        aggregate = _aggregate_round_metrics(all_round_metrics)
        _print_method_summary(spec.name, aggregate)

        method_reports[spec.name] = {
            "mode": "random" if spec.mode == "random" else "surrogate",
            "backend": spec.backend,
            "acquisition": acquisition_name,
            "seed": method_seed,
            "num_episodes": len(episodes),
            "num_acquisitions": num_acquisitions,
            "episode_summaries": episode_summaries,
            "aggregate_by_round": aggregate,
            "round_metrics": all_round_metrics,
        }
        if acquisition_name != "random" and surrogate_backend_name in {"tabpfn", "tabicl"}:
            method_reports[spec.name]["policy_checkpoint_path"] = loaded_surrogate.policy_checkpoint_path
            method_reports[spec.name]["backbone_checkpoint_path"] = loaded_surrogate.backbone_checkpoint_path
            method_reports[spec.name]["backbone_state_source"] = loaded_surrogate.backbone_state_source
        aggregate_by_method[spec.name] = aggregate

    plot_path: Path | None = None
    if bool(cfg.output.save_plots):
        plot_path = plots_dir / f"bo_evaluate_{metric}.png"
        _plot_comparison_curves(
            save_path=plot_path,
            aggregate_by_method=aggregate_by_method,
            metric=metric_name,
            uncertainty=uncertainty_mode,
            task_label=str(cfg.task.name),
        )
        print(f"plot_path: {plot_path}")

    report = {
        "task_name": str(cfg.task.name),
        "task_label": str(episodes[0].search_space_id) if episodes else str(cfg.task.name),
        "metric": metric,
        "plot_uncertainty": uncertainty,
        "methods": method_names,
        "num_episodes": len(episodes),
        "num_acquisitions": num_acquisitions,
        "seeds": eval_seeds,
        "dataset_ids": eval_dataset_ids,
        "method_reports": method_reports,
        "plot_path": str(plot_path) if plot_path is not None else None,
    }

    if bool(cfg.output.save_metrics):
        raw_curves_root = run_dir / "raw_curves"
        for method_name, method_report in method_reports.items():
            artifact_info = _save_method_curve_artifacts(
                method_dir=raw_curves_root / method_name,
                method_name=method_name,
                search_space_id=str(episodes[0].search_space_id) if episodes else str(cfg.task.name),
                round_metrics=method_report["round_metrics"],
                aggregate_by_round=method_report["aggregate_by_round"],
                extra_metadata={
                    "mode": method_report["mode"],
                    "backend": method_report["backend"],
                    "acquisition": method_report["acquisition"],
                    "num_episodes": method_report["num_episodes"],
                    "num_acquisitions": method_report["num_acquisitions"],
                    "seed": method_report["seed"],
                    "policy_checkpoint_path": method_report.get("policy_checkpoint_path"),
                    "backbone_checkpoint_path": method_report.get("backbone_checkpoint_path"),
                    "backbone_state_source": method_report.get("backbone_state_source"),
                },
            )
            method_report["artifact_dir"] = artifact_info["method_dir"]
            method_report["artifact_metadata_path"] = artifact_info["metadata_path"]
            method_report["artifact_files"] = artifact_info["artifact_files"]

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        OmegaConf.save(cfg, run_dir / "config_resolved.yaml")

    print("BO evaluation finished.")
    print(f"metrics_path: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
