from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from polar.tasks import HPOBBOEpisode
from polar.trainers import CandidatePolicyHead
from polar.utils.surrogates import RandomForestSurrogate


_EVALUATE_BO_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_bo.py"
_EVALUATE_BO_SPEC = importlib.util.spec_from_file_location("evaluate_bo_script", _EVALUATE_BO_PATH)
if _EVALUATE_BO_SPEC is None or _EVALUATE_BO_SPEC.loader is None:
    raise RuntimeError(f"Failed to import evaluate_bo.py from {_EVALUATE_BO_PATH}.")
_EVALUATE_BO = importlib.util.module_from_spec(_EVALUATE_BO_SPEC)
sys.modules[_EVALUATE_BO_SPEC.name] = _EVALUATE_BO
_EVALUATE_BO_SPEC.loader.exec_module(_EVALUATE_BO)

_evaluate_one_episode_policy = _EVALUATE_BO._evaluate_one_episode_policy
_build_surrogate = _EVALUATE_BO._build_surrogate
_build_dense_curve_artifact = _EVALUATE_BO._build_dense_curve_artifact
_parse_method_spec = _EVALUATE_BO._parse_method_spec
_plot_comparison_curves = _EVALUATE_BO._plot_comparison_curves
_save_method_curve_artifacts = _EVALUATE_BO._save_method_curve_artifacts

matplotlib.use("Agg")


class _ChunkSafePolicyBackbone:
    def __init__(self) -> None:
        self.model = nn.Linear(1, 1)
        self._device = torch.device("cpu")
        self.embedding_dim = 3

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_context(self, x_context: torch.Tensor, y_context: torch.Tensor):
        return {
            "x_mean": x_context.mean(dim=0),
            "y_mean": y_context.mean(),
        }

    def forward_with_embeddings(self, x_query: torch.Tensor, context_state):
        x_centered = x_query - context_state["x_mean"]
        y_mean = context_state["y_mean"].expand(x_query.shape[0], 1)
        embeddings = torch.cat([x_query, x_centered.sum(dim=-1, keepdim=True) + y_mean], dim=-1)
        logits = embeddings.sum(dim=-1)
        return {
            "logits": logits,
            "mean": logits,
            "test_embeddings": embeddings,
        }


class _DummySurrogateBackbone:
    def __init__(self) -> None:
        self.model = nn.Linear(2, 1, bias=False)
        self._device = torch.device("cpu")
        self.embedding_dim = 2
        self.checkpoint_path: str | None = None

    @property
    def device(self) -> torch.device:
        return self._device


def _episode() -> HPOBBOEpisode:
    context_x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    context_y = torch.tensor([0.1, 0.4], dtype=torch.float32)
    query_x = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    query_y = torch.tensor([0.2, 0.5, 0.3, 0.6], dtype=torch.float32)
    return HPOBBOEpisode(
        search_space_id="dummy",
        dataset_id="dataset",
        split="test",
        seed="test0",
        context_x=context_x,
        context_y=context_y,
        query_x=query_x,
        query_y=query_y,
        target_x=None,
        target_y=None,
        y_best=float(torch.cat([context_y, query_y]).max().item()),
        y_min=float(torch.cat([context_y, query_y]).min().item()),
    )


@pytest.mark.parametrize("requested_budget", [2, 9])
def test_policy_eval_uses_remaining_count_including_current_action(requested_budget: int) -> None:
    backbone = _ChunkSafePolicyBackbone()
    policy_head = CandidatePolicyHead(
        input_dim=backbone.embedding_dim, hidden_dim=8, num_layers=1,
        use_remaining_budget=True, budget_scale=10,
    )
    policy_inputs = []
    policy_head.network[0].register_forward_pre_hook(
        lambda _module, args: policy_inputs.append(args[0].detach().clone())
    )
    _evaluate_one_episode_policy(
        method_name="polar", episode=_episode(), backbone=backbone, policy_head=policy_head,
        num_acquisitions=requested_budget, selection="argmax", max_query_chunk=2,
    )

    effective_budget = min(requested_budget, 4)
    assert len(policy_inputs) == effective_budget
    for features, remaining in zip(
        policy_inputs, range(effective_budget, 0, -1), strict=True
    ):
        assert features.shape[-1] == backbone.embedding_dim + 1
        torch.testing.assert_close(
            features[:, -1], features.new_full((features.shape[0],), remaining / 10)
        )


def test_policy_chunked_eval_matches_full_eval() -> None:
    torch.manual_seed(0)
    backbone = _ChunkSafePolicyBackbone()
    policy_head = CandidatePolicyHead(input_dim=backbone.embedding_dim, hidden_dim=8, num_layers=1)
    policy_head.eval()
    episode = _episode()

    full_rounds, full_summary = _evaluate_one_episode_policy(
        method_name="tabpfn_polar",
        episode=episode,
        backbone=backbone,
        policy_head=policy_head,
        num_acquisitions=3,
        selection="argmax",
        max_query_chunk=0,
    )
    chunked_rounds, chunked_summary = _evaluate_one_episode_policy(
        method_name="tabpfn_polar",
        episode=episode,
        backbone=backbone,
        policy_head=policy_head,
        num_acquisitions=3,
        selection="argmax",
        max_query_chunk=2,
    )

    assert full_summary == chunked_summary
    assert len(full_rounds) == len(chunked_rounds)
    for full_step, chunked_step in zip(full_rounds, chunked_rounds):
        assert full_step["round"] == chunked_step["round"]
        assert full_step["query_size"] == chunked_step["query_size"]
        assert full_step["context_size"] == chunked_step["context_size"]
        assert full_step["selected_y"] == chunked_step["selected_y"]
        assert full_step["selected_score"] == chunked_step["selected_score"]
        assert full_step["selected_prob"] == chunked_step["selected_prob"]
        assert full_step["policy_entropy"] == chunked_step["policy_entropy"]


def test_policy_eval_reports_running_best_y_not_oracle_best_y() -> None:
    torch.manual_seed(0)
    backbone = _ChunkSafePolicyBackbone()
    policy_head = CandidatePolicyHead(input_dim=backbone.embedding_dim, hidden_dim=8, num_layers=1)
    policy_head.eval()
    episode = _episode()

    round_metrics, summary = _evaluate_one_episode_policy(
        method_name="tabpfn_polar",
        episode=episode,
        backbone=backbone,
        policy_head=policy_head,
        num_acquisitions=1,
        selection="argmax",
        max_query_chunk=0,
    )

    assert round_metrics[0]["best_y"] == pytest.approx(0.4)
    assert round_metrics[0]["oracle_best_y"] == pytest.approx(0.6)
    assert summary["best_y_start"] == pytest.approx(0.4)
    assert summary["best_y_start"] == round_metrics[0]["best_y"]
    assert summary["best_y_end"] == round_metrics[-1]["best_y"]
    assert summary["best_y_start"] != round_metrics[0]["oracle_best_y"]


def test_parse_policy_variant_method_name() -> None:
    spec = _parse_method_spec(
        "tabpfn_polar_frozen",
        acquisitions=["ucb", "ei_mc", "ei"],
        backends={"tabpfn", "tabicl", "gp"},
    )

    assert spec.name == "tabpfn_polar_frozen"
    assert spec.mode == "policy"
    assert spec.backend == "tabpfn"
    assert spec.acquisition is None


def test_parse_canonical_polar_method_name() -> None:
    spec = _parse_method_spec(
        "polar",
        acquisitions=["ucb", "ei_mc", "ei"],
        backends={"tabpfn", "tabicl", "gp"},
        policy_backend="tabicl",
    )

    assert spec.name == "polar"
    assert spec.mode == "policy"
    assert spec.backend == "tabicl"
    assert spec.acquisition is None


def test_parse_rf_ei_method_name() -> None:
    spec = _parse_method_spec(
        "rf_ei",
        acquisitions=["ucb", "ei_mc", "ei"],
        backends={"tabpfn", "tabicl", "gp", "rf"},
    )

    assert spec.name == "rf_ei"
    assert spec.mode == "surrogate"
    assert spec.backend == "rf"
    assert spec.acquisition == "ei"


def test_surrogate_can_load_backbone_state_from_policy_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pretrained_path = tmp_path / "pretrained.ckpt"
    pretrained_path.touch()
    policy_ckpt_path = tmp_path / "policy_checkpoint.pt"

    target_weight = torch.tensor([[1.25, -0.75]], dtype=torch.float32)
    torch.save(
        {
            "config": {"model": {"name": "tabpfn"}},
            "backbone_checkpoint_path": str(pretrained_path),
            "backbone_model_state_dict": {"weight": target_weight},
        },
        policy_ckpt_path,
    )

    dummy_backbone = _DummySurrogateBackbone()

    def _fake_create_backbone(**kwargs):
        dummy_backbone.checkpoint_path = kwargs.get("checkpoint_path")
        return dummy_backbone

    monkeypatch.setattr(_EVALUATE_BO, "create_backbone", _fake_create_backbone)

    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "paths": {"checkpoints_root": str(tmp_path)},
            "eval": {"max_query_chunk": 0},
            "policy": {
                "checkpoints": {"tabpfn_ucb": str(policy_ckpt_path)},
                "default_checkpoint_path": None,
                "strict_backbone_state": True,
            },
            "models": {
                "tabpfn": {
                    "name": "tabpfn",
                    "checkpoint_path": None,
                    "version": "v2.5",
                    "cache_trainset_representation": False,
                    "strict_load": True,
                }
            },
        }
    )

    loaded = _build_surrogate(
        cfg,
        method_name="tabpfn_ucb",
        backend="tabpfn",
    )

    assert loaded.policy_checkpoint_path == str(policy_ckpt_path.resolve())
    assert loaded.backbone_checkpoint_path == str(pretrained_path)
    assert loaded.backbone_state_source == "checkpoint_payload"
    assert torch.allclose(loaded.surrogate.backbone.model.weight.detach(), target_weight)


def test_build_surrogate_supports_rf_backend(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "paths": {"checkpoints_root": str(tmp_path)},
            "eval": {"max_query_chunk": 0},
            "policy": {
                "checkpoints": {},
                "default_checkpoint_path": None,
                "strict_backbone_state": True,
            },
            "models": {
                "rf": {
                    "name": "rf",
                    "n_estimators": 16,
                    "max_depth": 5,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "variance_floor": 1.0e-6,
                    "n_jobs": 1,
                    "random_state": None,
                }
            },
        }
    )

    loaded = _build_surrogate(
        cfg,
        method_name="rf_ei",
        backend="rf",
        random_state=123,
    )

    assert isinstance(loaded.surrogate, RandomForestSurrogate)
    assert loaded.surrogate.backend_name == "rf"
    assert loaded.policy_checkpoint_path is None
    assert loaded.backbone_checkpoint_path is None
    assert loaded.backbone_state_source == "not_applicable"


def test_dense_curve_artifact_uses_dataset_seed_round_axes() -> None:
    round_metrics = [
        {
            "dataset_id": "d1",
            "seed": "s0",
            "round": 0,
            "best_y": 0.4,
            "best_y_improvement": 0.0,
            "oracle_best_y": 0.6,
            "initial_best_y": 0.4,
            "regret": 0.2,
            "normalized_regret": 0.4,
            "query_size": 4,
            "context_size": 2,
        },
        {
            "dataset_id": "d1",
            "seed": "s0",
            "round": 1,
            "best_y": 0.5,
            "best_y_improvement": 0.1,
            "oracle_best_y": 0.6,
            "initial_best_y": 0.4,
            "regret": 0.1,
            "normalized_regret": 0.2,
            "selected_y": 0.5,
            "query_size": 3,
            "context_size": 3,
        },
        {
            "dataset_id": "d2",
            "seed": "s1",
            "round": 0,
            "best_y": 0.3,
            "best_y_improvement": 0.0,
            "oracle_best_y": 0.9,
            "initial_best_y": 0.3,
            "regret": 0.6,
            "normalized_regret": 0.75,
            "query_size": 4,
            "context_size": 2,
        },
    ]

    artifact = _build_dense_curve_artifact(round_metrics)

    assert artifact["dataset_ids"] == ["d1", "d2"]
    assert artifact["seeds"] == ["s0", "s1"]
    assert artifact["rounds"] == [0, 1]
    assert artifact["shape"] == [2, 2, 2]
    regret = artifact["metrics"]["regret"]
    assert regret.shape == (2, 2, 2)
    assert regret[0, 0, 0] == pytest.approx(0.2)
    assert regret[0, 0, 1] == pytest.approx(0.1)
    assert regret[1, 1, 0] == pytest.approx(0.6)
    assert torch.isnan(torch.tensor(regret[1, 0, 0])).item()


def test_save_method_curve_artifacts_writes_npy_and_json(tmp_path: Path) -> None:
    artifact_info = _save_method_curve_artifacts(
        method_dir=tmp_path / "ours",
        method_name="ours",
        search_space_id="5860",
        round_metrics=[
            {
                "dataset_id": "d1",
                "seed": "test0",
                "round": 0,
                "best_y": 0.4,
                "best_y_improvement": 0.0,
                "oracle_best_y": 0.6,
                "initial_best_y": 0.4,
                "regret": 0.2,
                "normalized_regret": 0.4,
                "query_size": 4,
                "context_size": 2,
            }
        ],
        aggregate_by_round=[{"round": 0, "mean_regret": 0.2}],
        extra_metadata={"mode": "policy", "backend": "tabicl", "acquisition": None},
    )

    metadata_path = Path(artifact_info["metadata_path"])
    regret_path = Path(artifact_info["artifact_files"]["regret"])

    assert metadata_path.exists()
    assert regret_path.exists()
    saved_regret = torch.from_numpy(np.load(regret_path))
    assert tuple(saved_regret.shape) == (1, 1, 1)
    assert saved_regret[0, 0, 0].item() == pytest.approx(0.2)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["method"] == "ours"
    assert metadata["search_space_id"] == "5860"
    assert metadata["array_shape"] == [1, 1, 1]


def test_plot_comparison_curves_uses_log_y_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    monkeypatch.setattr(plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(plt, "close", lambda _fig: None)

    save_path = tmp_path / "comparison.png"
    _plot_comparison_curves(
        save_path=save_path,
        aggregate_by_method={
            "ours": [
                {
                    "round": 0,
                    "mean_regret": 1.0,
                    "ci95_regret": 0.2,
                    "mean_normalized_regret": 0.5,
                    "ci95_normalized_regret": 0.1,
                },
                {
                    "round": 1,
                    "mean_regret": 0.0,
                    "ci95_regret": 0.0,
                    "mean_normalized_regret": 0.05,
                    "ci95_normalized_regret": 0.02,
                },
            ]
        },
        metric="regret",
        uncertainty="ci95",
        task_label="hpob",
    )

    assert save_path.exists()
    assert ax.get_yscale() == "log"
    assert ax.lines[0].get_drawstyle() == "steps-post"
