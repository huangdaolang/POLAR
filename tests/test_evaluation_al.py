from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest
import torch
from torch import nn

from polar.evaluation import (
    aggregate_al_eval_metrics,
    aggregate_al_eval_metrics_by_dataset,
    evaluate_policy_episodes,
    evaluate_surrogate_episodes,
    plot_policy_probe_rounds_grid,
    rollout_policy_probe_episode,
    rollout_surrogate_probe_episode,
    summarize_probe_metrics,
)
from polar.tasks import GPPriorSample
from polar.trainers import CandidatePolicyHead, TargetAwareTransformerPolicyHead
from polar.utils.surrogates import GPJointPrediction, Surrogate, SurrogatePrediction

matplotlib.use("Agg")


class _ProbeBackbone:
    def __init__(self) -> None:
        self.model = nn.Linear(1, 1)
        self._device = torch.device("cpu")
        self.embedding_dim = 2

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_context(self, x_context: torch.Tensor, y_context: torch.Tensor):
        return {
            "x_mean": x_context.mean(dim=-2),
            "y_mean": y_context.mean(dim=-1, keepdim=True),
        }

    def forward_with_embeddings(self, x_query: torch.Tensor, context_state):
        x_mean = context_state["x_mean"]
        y_mean = context_state["y_mean"]
        if x_query.ndim == 3:
            centered = x_query - x_mean.unsqueeze(1)
            expanded_y = y_mean.unsqueeze(1).expand(x_query.shape[0], x_query.shape[1], 1)
            embeddings = torch.cat([x_query, centered + expanded_y], dim=-1)
            logits = embeddings.sum(dim=-1)
            return {"logits": logits, "mean": logits, "test_embeddings": embeddings}

        centered = x_query - x_mean
        embeddings = torch.cat([x_query, centered + y_mean], dim=-1)
        logits = embeddings.sum(dim=-1)
        return {"logits": logits, "mean": logits, "test_embeddings": embeddings}

    def log_prob_from_logits(self, logits: torch.Tensor, y_query: torch.Tensor) -> torch.Tensor:
        if y_query.ndim == 3 and y_query.shape[2] == 1:
            y_query = y_query.squeeze(-1)
        return -(logits - y_query).square()

    def variance_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(logits)

    def predictive_distribution(self, x_query: torch.Tensor, context_state):
        out = self.forward_with_embeddings(x_query, context_state)
        return {"logits": out["logits"], "mean": out["mean"]}


class _ProbeSurrogate(Surrogate):
    @property
    def backend_name(self) -> str:
        return "gp"

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def fit(self, context_x: torch.Tensor, context_y: torch.Tensor) -> None:
        self._x_mean = context_x.mean(dim=0, keepdim=True)
        self._y_mean = context_y.mean()

    def predict(self, x: torch.Tensor) -> SurrogatePrediction:
        mean = x.squeeze(-1) + float(self._y_mean)
        variance = 0.1 + (x.squeeze(-1) - float(self._x_mean.squeeze())).square()
        return SurrogatePrediction(mean=mean, variance=variance)

    def point_log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.predict(x)
        return -(pred.mean - y).square()

    def sample(self, x: torch.Tensor, *, num_samples: int) -> torch.Tensor:
        pred = self.predict(x)
        return pred.mean.unsqueeze(0).repeat(num_samples, 1)


class _ProbeGPSurrogate(_ProbeSurrogate):
    def predict_query_target_joint(
        self,
        query_x: torch.Tensor,
        target_x: torch.Tensor,
    ) -> GPJointPrediction:
        query_pred = self.predict(query_x)
        target_pred = self.predict(target_x)
        covariance = torch.stack(
            [
                0.05 + target_x[:, 0].abs(),
                0.03 + 0.5 * target_x[:, 0].abs(),
                0.02 + 0.25 * target_x[:, 0].abs(),
            ],
            dim=1,
        )[:, : query_x.shape[0]]
        return GPJointPrediction(
            query_mean=query_pred.mean,
            query_variance=query_pred.variance,
            target_mean=target_pred.mean,
            target_variance=target_pred.variance,
            target_query_covariance=covariance,
        )


def _episode(*, y_shift: float = 0.0) -> GPPriorSample:
    x = torch.linspace(-1.0, 1.0, steps=8).unsqueeze(-1)
    y = torch.tensor([-0.4, -0.2, 0.1, 0.3, 0.8, 0.6, 0.5, 0.4], dtype=torch.float32)
    y = y + float(y_shift)
    return GPPriorSample(
        context_x=x[:2],
        context_y=y[:2],
        query_x=x[2:5],
        query_y=y[2:5],
        target_x=x[5:7],
        target_y=y[5:7],
        plot_x=x,
        plot_y_true=y,
        kernel_type="rbf",
        lengthscales=torch.ones(1),
        output_scale=torch.ones(1),
        raw_context_x=x[:2],
        raw_query_x=x[2:5],
        raw_target_x=x[5:7],
        raw_plot_x=x,
        raw_context_y=y[:2],
        raw_query_y=y[2:5],
        raw_target_y=y[5:7],
        raw_plot_y_true=y,
        y_mean=torch.zeros(()),
        y_std=torch.ones(()),
    )


@pytest.mark.parametrize("mode", ["probe", "batch"])
@pytest.mark.parametrize("head_type", ["mlp", "target_aware_transformer"])
def test_al_evaluation_passes_remaining_budget_to_either_head(mode: str, head_type: str) -> None:
    backbone = _ProbeBackbone()
    if head_type == "mlp":
        policy_head = CandidatePolicyHead(
            input_dim=backbone.embedding_dim, hidden_dim=8, num_layers=1,
            use_remaining_budget=True, budget_scale=10,
        )
        input_layer = policy_head.network[0]
    else:
        policy_head = TargetAwareTransformerPolicyHead(
            input_dim=backbone.embedding_dim, num_heads=1, ffn_dim=8, num_layers=1,
            use_remaining_budget=True, budget_scale=10,
        )
        input_layer = policy_head.layers[0]["ffn"][0]
    policy_inputs = []
    input_layer.register_forward_pre_hook(
        lambda _module, args: policy_inputs.append(args[0].detach().clone())
    )
    if mode == "probe":
        rollout_policy_probe_episode(
            episode=_episode(), backbone=backbone, policy_head=policy_head,
            num_acquisitions=5, selection="argmax", collect_snapshots=False,
        )
    else:
        evaluate_policy_episodes(
            episodes=[_episode(), _episode(y_shift=0.25)],
            backbone=backbone, policy_head=policy_head,
            num_acquisitions=5, selection="argmax",
        )

    assert len(policy_inputs) == 3
    for features, remaining in zip(policy_inputs, [3, 2, 1], strict=True):
        assert features.shape[-1] == backbone.embedding_dim + 1
        torch.testing.assert_close(
            features[..., -1], features.new_full(features.shape[:-1], remaining / 10)
        )


def test_rollout_policy_probe_episode_collects_weighted_metrics_and_snapshots(
    tmp_path: Path,
) -> None:
    backbone = _ProbeBackbone()
    policy_head = TargetAwareTransformerPolicyHead(
        input_dim=backbone.embedding_dim,
        num_heads=1,
        ffn_dim=8,
        num_layers=1,
        dropout=0.0,
    )

    round_metrics, snapshots = rollout_policy_probe_episode(
        episode=_episode(),
        backbone=backbone,
        policy_head=policy_head,
        num_acquisitions=2,
        selection="argmax",
        collect_snapshots=True,
        snapshot_round_stride=1,
        policy_target_weighting="exp_y",
        policy_target_weight_beta=1.0,
    )

    assert len(round_metrics) == 2
    assert len(snapshots) == 2
    assert "weighted_target_rmse" in round_metrics[0]
    assert "weighted_target_log_prob" in round_metrics[0]

    summary = summarize_probe_metrics(round_metrics)
    assert summary["selected_query_y_mean"] > 0.0

    plot_path = tmp_path / "probe.png"
    plot_policy_probe_rounds_grid(
        save_path=plot_path,
        round_snapshots=snapshots,
        method_name="probe",
        ci_std=1.96,
    )
    assert plot_path.exists()


def test_rollout_policy_probe_episode_supports_step_pos_weighting() -> None:
    backbone = _ProbeBackbone()
    policy_head = TargetAwareTransformerPolicyHead(
        input_dim=backbone.embedding_dim,
        num_heads=1,
        ffn_dim=8,
        num_layers=1,
        dropout=0.0,
    )

    round_metrics, snapshots = rollout_policy_probe_episode(
        episode=_episode(),
        backbone=backbone,
        policy_head=policy_head,
        num_acquisitions=2,
        selection="argmax",
        collect_snapshots=False,
        policy_target_weighting="step_pos",
        policy_target_weight_beta=3.0,
        policy_target_weight_threshold=1.0,
    )

    assert len(round_metrics) == 2
    assert snapshots == []
    assert "weighted_target_log_prob" in round_metrics[0]


def test_evaluate_policy_episodes_batches_weighted_and_unweighted_metrics() -> None:
    backbone = _ProbeBackbone()
    policy_head = TargetAwareTransformerPolicyHead(
        input_dim=backbone.embedding_dim,
        num_heads=1,
        ffn_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    episodes = [
        _episode(y_shift=0.0),
        _episode(y_shift=0.25),
    ]

    round_metrics = evaluate_policy_episodes(
        episodes=episodes,
        backbone=backbone,
        policy_head=policy_head,
        num_acquisitions=2,
        selection="argmax",
        policy_target_weighting="exp_y",
        policy_target_weight_beta=1.0,
    )

    assert len(round_metrics) == 4
    assert "weighted_target_rmse" in round_metrics[0]
    assert "weighted_target_log_prob" in round_metrics[0]
    assert "selected_query_raw_y" in round_metrics[0]

    aggregated = aggregate_al_eval_metrics(round_metrics)
    assert len(aggregated) == 2
    assert "mean_target_rmse" in aggregated[0]
    assert "mean_weighted_target_rmse" in aggregated[0]
    assert "mean_target_log_prob" in aggregated[0]
    assert "mean_weighted_target_log_prob" in aggregated[0]
    assert "mean_selected_query_y" in aggregated[0]


def test_rollout_surrogate_probe_episode_collects_weighted_metrics_and_snapshots(
    tmp_path: Path,
) -> None:
    surrogate = _ProbeSurrogate()

    round_metrics, snapshots = rollout_surrogate_probe_episode(
        episode=_episode(),
        surrogate=surrogate,
        acquisition_name="uncertainty",
        num_acquisitions=2,
        collect_snapshots=True,
        snapshot_round_stride=1,
        policy_target_weighting="exp_y",
        policy_target_weight_beta=1.0,
    )

    assert len(round_metrics) == 2
    assert len(snapshots) == 2
    assert "weighted_target_rmse" in round_metrics[0]
    assert "weighted_target_log_prob" in round_metrics[0]
    assert round_metrics[0]["selected_prob"] is None

    plot_path = tmp_path / "surrogate_probe.png"
    plot_policy_probe_rounds_grid(
        save_path=plot_path,
        round_snapshots=snapshots,
        method_name="gp_uncertainty",
        ci_std=1.96,
    )
    assert plot_path.exists()


def test_evaluate_surrogate_episodes_batches_metrics() -> None:
    surrogate = _ProbeSurrogate()
    episodes = [
        _episode(y_shift=0.0),
        _episode(y_shift=0.25),
    ]

    round_metrics = evaluate_surrogate_episodes(
        episodes=episodes,
        surrogate=surrogate,
        acquisition_name="uncertainty",
        num_acquisitions=2,
        policy_target_weighting="exp_y",
        policy_target_weight_beta=1.0,
    )

    assert len(round_metrics) == 4
    assert "episode_seed" in round_metrics[0]
    aggregated = aggregate_al_eval_metrics(round_metrics)
    assert len(aggregated) == 2
    assert "mean_weighted_target_rmse" in aggregated[0]
    assert "mean_selected_query_y" in aggregated[0]

    grouped = aggregate_al_eval_metrics_by_dataset(round_metrics)
    assert list(grouped.keys()) == ["gp_prior"]
    assert grouped["gp_prior"] == aggregated


def test_rollout_surrogate_probe_episode_supports_gp_variance_reduction() -> None:
    surrogate = _ProbeGPSurrogate()

    round_metrics, _ = rollout_surrogate_probe_episode(
        episode=_episode(),
        surrogate=surrogate,
        acquisition_name="variance_reduction",
        num_acquisitions=2,
        collect_snapshots=False,
        policy_target_weighting="exp_y",
        policy_target_weight_beta=1.0,
    )

    assert len(round_metrics) == 2
    assert isinstance(round_metrics[0]["selected_query_y"], float)


def test_rollout_surrogate_probe_episode_supports_gp_weighted_variance_reduction() -> None:
    surrogate = _ProbeGPSurrogate()

    round_metrics, _ = rollout_surrogate_probe_episode(
        episode=_episode(),
        surrogate=surrogate,
        acquisition_name="weighted_variance_reduction",
        num_acquisitions=2,
        collect_snapshots=False,
        policy_target_weighting="exp_y",
        policy_target_weight_beta=1.0,
    )

    assert len(round_metrics) == 2
    assert isinstance(round_metrics[0]["selected_query_y"], float)


def test_rollout_surrogate_probe_episode_supports_step_pos_weighted_variance_reduction() -> None:
    surrogate = _ProbeGPSurrogate()

    round_metrics, _ = rollout_surrogate_probe_episode(
        episode=_episode(),
        surrogate=surrogate,
        acquisition_name="weighted_variance_reduction",
        num_acquisitions=2,
        collect_snapshots=False,
        policy_target_weighting="step_pos",
        policy_target_weight_beta=3.0,
        policy_target_weight_threshold=1.0,
    )

    assert len(round_metrics) == 2
    assert isinstance(round_metrics[0]["selected_query_y"], float)
