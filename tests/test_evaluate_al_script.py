from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

from omegaconf import OmegaConf


_EVALUATE_AL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_al.py"
_EVALUATE_AL_SPEC = importlib.util.spec_from_file_location(
    "evaluate_al_script",
    _EVALUATE_AL_PATH,
)
if _EVALUATE_AL_SPEC is None or _EVALUATE_AL_SPEC.loader is None:
    raise RuntimeError(f"Failed to import evaluate_al.py from {_EVALUATE_AL_PATH}.")
_EVALUATE_AL = importlib.util.module_from_spec(_EVALUATE_AL_SPEC)
sys.modules[_EVALUATE_AL_SPEC.name] = _EVALUATE_AL
_EVALUATE_AL_SPEC.loader.exec_module(_EVALUATE_AL)

_parse_method_spec = _EVALUATE_AL._parse_method_spec
_select_benchmark_plot_episodes = _EVALUATE_AL._select_benchmark_plot_episodes
_resolve_policy_checkpoint_path = _EVALUATE_AL._resolve_policy_checkpoint_path
_summarize_method_results = _EVALUATE_AL._summarize_method_results
_validate_eval_task_config = _EVALUATE_AL._validate_eval_task_config
_gp_regression_probe_plots_enabled = _EVALUATE_AL._gp_regression_probe_plots_enabled


def test_validate_eval_task_config_accepts_scaled_1d_gp() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "name": "gp_regression",
                "split": "test",
                "dim_x": 1,
                "dim_x_min": 1,
                "dim_x_max": 1,
                "gp_prior": {
                    "standardize_x": True,
                    "standardize_y": True,
                },
            },
        }
    )

    _validate_eval_task_config(cfg)


def test_validate_eval_task_config_accepts_scaled_high_dim_gp() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "name": "gp_regression",
                "split": "test",
                "dim_x": 2,
                "dim_x_min": 2,
                "dim_x_max": 2,
                "gp_prior": {
                    "standardize_x": True,
                    "standardize_y": True,
                },
            },
        }
    )

    _validate_eval_task_config(cfg)


def test_validate_eval_task_config_rejects_unscaled_task() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "name": "benchmark_functions",
                "split": "test",
                "normalize_x": False,
                "normalize_y": True,
            },
        }
    )

    try:
        _validate_eval_task_config(cfg)
    except ValueError as exc:
        assert "normalize_x=true" in str(exc)
    else:
        raise AssertionError("Expected eval validation to reject unnormalized benchmarks.")


def test_validate_eval_task_config_accepts_benchmark_functions() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "name": "benchmark_functions",
                "split": "test",
                "normalize_x": True,
                "normalize_y": True,
            },
        }
    )

    _validate_eval_task_config(cfg)


def test_gp_regression_probe_plots_enabled_only_for_1d() -> None:
    class _Task:
        def __init__(self, dim_x: int) -> None:
            self.dim_x = dim_x

    assert _gp_regression_probe_plots_enabled(_Task(dim_x=1)) is True
    assert _gp_regression_probe_plots_enabled(_Task(dim_x=2)) is False


def test_parse_method_spec_accepts_strict_policy_name() -> None:
    cfg = OmegaConf.create(
        {
            "models": {
                "tabicl": {"name": "tabicl"},
                "tabpfn": {"name": "tabpfn"},
                "gp": {"name": "gp"},
            },
            "policy": {"backend": "tabicl"},
        }
    )

    spec = _parse_method_spec(cfg, method_name="tabicl_polar_uniform")
    assert spec.mode == "policy"
    assert spec.backend == "tabicl"
    assert spec.label == "uniform"


def test_parse_method_spec_accepts_canonical_polar_name() -> None:
    cfg = OmegaConf.create(
        {
            "models": {
                "tabicl": {"name": "tabicl"},
                "tabpfn": {"name": "tabpfn"},
                "gp": {"name": "gp"},
            },
            "policy": {"backend": "tabicl"},
        }
    )

    spec = _parse_method_spec(cfg, method_name="polar")
    assert spec.mode == "policy"
    assert spec.backend == "tabicl"
    assert spec.label is None


def test_parse_method_spec_accepts_named_polar_variant() -> None:
    cfg = OmegaConf.create(
        {
            "models": {"tabicl": {"name": "tabicl"}},
            "policy": {"backend": "tabicl"},
        }
    )

    spec = _parse_method_spec(cfg, method_name="polar_weighted")
    assert spec.mode == "policy"
    assert spec.backend == "tabicl"
    assert spec.label == "weighted"


def test_parse_method_spec_accepts_surrogate_names() -> None:
    cfg = OmegaConf.create(
        {
            "models": {
                "tabicl": {"name": "tabicl"},
                "tabpfn": {"name": "tabpfn"},
                "gp": {"name": "gp"},
            },
            "policy": {"backend": "tabicl"},
        }
    )

    gp_spec = _parse_method_spec(cfg, method_name="gp_uncertainty")
    assert gp_spec.mode == "surrogate"
    assert gp_spec.backend == "gp"
    assert gp_spec.acquisition == "uncertainty"

    tabicl_spec = _parse_method_spec(cfg, method_name="tabicl_random")
    assert tabicl_spec.mode == "surrogate"
    assert tabicl_spec.backend == "tabicl"
    assert tabicl_spec.acquisition == "random"

    gp_vr_spec = _parse_method_spec(cfg, method_name="gp_variance_reduction")
    assert gp_vr_spec.mode == "surrogate"
    assert gp_vr_spec.backend == "gp"
    assert gp_vr_spec.acquisition == "variance_reduction"

    gp_wvr_spec = _parse_method_spec(cfg, method_name="gp_weighted_variance_reduction")
    assert gp_wvr_spec.mode == "surrogate"
    assert gp_wvr_spec.backend == "gp"
    assert gp_wvr_spec.acquisition == "weighted_variance_reduction"


def test_parse_method_spec_grammar_boundaries() -> None:
    cfg = OmegaConf.create(
        {
            "models": {
                "tabicl": {"name": "tabicl"},
                "tabpfn": {"name": "tabpfn"},
                "gp": {"name": "gp"},
            },
            "policy": {"backend": "tabicl"},
        }
    )

    # `<backend>_polar` is part of the shared grammar and must parse.
    bare = _parse_method_spec(cfg, method_name="tabicl_polar")
    assert bare.mode == "policy"
    assert bare.backend == "tabicl"
    assert bare.label is None

    # The pre-release `<backend>_policy` spelling stays rejected.
    try:
        _parse_method_spec(cfg, method_name="tabicl_policy")
    except ValueError as exc:
        assert "Could not parse method" in str(exc)
    else:
        raise AssertionError("Expected legacy '<backend>_policy' to be rejected.")

    try:
        _parse_method_spec(cfg, method_name="tabicl_variance_reduction")
    except ValueError as exc:
        assert "only with backend='gp'" in str(exc)
    else:
        raise AssertionError("Expected non-GP variance reduction to be rejected.")


def test_resolve_policy_checkpoint_path_uses_per_method_mapping() -> None:
    cfg = OmegaConf.create(
        {
            "policy": {
                "checkpoints": {
                    "tabicl_polar_uniform": "/tmp/uniform.pt",
                    "tabicl_polar_weighted": "/tmp/weighted.pt",
                },
                "default_checkpoint_path": None,
            },
        }
    )

    assert (
        _resolve_policy_checkpoint_path(cfg, method_name="tabicl_polar_uniform")
        == "/tmp/uniform.pt"
    )
    assert (
        _resolve_policy_checkpoint_path(cfg, method_name="tabicl_polar_weighted")
        == "/tmp/weighted.pt"
    )


def test_summarize_method_results_tracks_weighted_and_unweighted_metrics() -> None:
    round_metrics = [
        {
            "round": 0,
            "target_rmse": 1.0,
            "weighted_target_rmse": 0.8,
            "target_log_prob_mean": -1.2,
            "weighted_target_log_prob": -1.0,
            "selected_query_y": 0.2,
            "selected_query_raw_y": 0.4,
        },
        {
            "round": 0,
            "target_rmse": 1.2,
            "weighted_target_rmse": 1.0,
            "target_log_prob_mean": -1.5,
            "weighted_target_log_prob": -1.3,
            "selected_query_y": 0.3,
            "selected_query_raw_y": 0.6,
        },
        {
            "round": 1,
            "target_rmse": 0.7,
            "weighted_target_rmse": 0.5,
            "target_log_prob_mean": -0.8,
            "weighted_target_log_prob": -0.6,
            "selected_query_y": 0.5,
            "selected_query_raw_y": 0.8,
        },
        {
            "round": 1,
            "target_rmse": 0.9,
            "weighted_target_rmse": 0.6,
            "target_log_prob_mean": -0.9,
            "weighted_target_log_prob": -0.7,
            "selected_query_y": 0.4,
            "selected_query_raw_y": 0.7,
        },
    ]
    aggregated_rounds = [
        {
            "round": 0,
            "mean_target_rmse": 1.1,
            "sem_target_rmse": 0.07,
            "mean_weighted_target_rmse": 0.9,
            "sem_weighted_target_rmse": 0.06,
            "mean_target_log_prob": -1.35,
            "sem_target_log_prob": 0.08,
            "mean_weighted_target_log_prob": -1.15,
            "sem_weighted_target_log_prob": 0.05,
            "mean_selected_query_y": 0.25,
            "sem_selected_query_y": 0.02,
            "mean_selected_query_raw_y": 0.5,
            "sem_selected_query_raw_y": 0.03,
        },
        {
            "round": 1,
            "mean_target_rmse": 0.8,
            "sem_target_rmse": 0.04,
            "mean_weighted_target_rmse": 0.55,
            "sem_weighted_target_rmse": 0.03,
            "mean_target_log_prob": -0.85,
            "sem_target_log_prob": 0.02,
            "mean_weighted_target_log_prob": -0.65,
            "sem_weighted_target_log_prob": 0.01,
            "mean_selected_query_y": 0.45,
            "sem_selected_query_y": 0.015,
            "mean_selected_query_raw_y": 0.75,
            "sem_selected_query_raw_y": 0.025,
        },
    ]

    summary = _summarize_method_results(
        round_metrics=round_metrics,
        aggregated_rounds=aggregated_rounds,
    )

    assert summary["target_rmse_start"] == 1.1
    assert summary["target_rmse_end"] == 0.8
    assert summary["target_rmse_end_sem"] == 0.04
    assert summary["weighted_target_rmse_end"] == 0.55
    assert summary["weighted_target_rmse_end_sem"] == 0.03
    assert math.isclose(summary["weighted_target_log_prob_gain"], 0.5)
    assert summary["weighted_target_log_prob_end_sem"] == 0.01
    assert summary["selected_query_y_mean_overall"] > 0.0
    assert math.isclose(summary["selected_query_y_mean_overall"], 0.35)
    assert math.isclose(summary["selected_query_y_sem_overall"], 0.0559016994, rel_tol=1e-6)


def test_select_benchmark_plot_episodes_uses_first_n_seeds_per_dataset() -> None:
    episodes = [
        OmegaConf.create({"dataset_id": "forrester", "seed": "episode0"}),
        OmegaConf.create({"dataset_id": "forrester", "seed": "episode1"}),
        OmegaConf.create({"dataset_id": "forrester", "seed": "episode2"}),
        OmegaConf.create({"dataset_id": "branin", "seed": "episode0"}),
        OmegaConf.create({"dataset_id": "branin", "seed": "episode1"}),
    ]

    selected = _select_benchmark_plot_episodes(
        episodes,
        n_plot_episodes_per_dataset=2,
    )

    assert [(dataset_id, idx, episode.seed) for dataset_id, idx, episode in selected] == [
        ("forrester", 0, "episode0"),
        ("forrester", 1, "episode1"),
        ("branin", 0, "episode0"),
        ("branin", 1, "episode1"),
    ]
