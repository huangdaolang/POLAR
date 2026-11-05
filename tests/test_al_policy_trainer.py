import pytest
import torch
from torch import nn

from polar.tasks import GPPriorALSampler, GPPriorSample
from polar.trainers import (
    ALPolicyTrainer,
    CandidatePolicyHead,
    TargetAwareTransformerPolicyHead,
    build_policy_optimizer,
)


class _DummyBackbone:
    def __init__(self) -> None:
        self.model = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1))
        self._device = torch.device("cpu")
        self.embedding_dim = 2
        self.forward_calls = 0

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_context(self, x_context: torch.Tensor, y_context: torch.Tensor):
        return {"x": x_context, "y": y_context}

    def forward_with_embeddings(self, x_query: torch.Tensor, context_state):
        _ = context_state
        self.forward_calls += 1
        if x_query.ndim == 2:
            logits = self.model(x_query).view(-1)
            return {
                "logits": logits,
                "mean": logits,
                "test_embeddings": x_query,
            }

        if x_query.ndim == 3:
            batch_size, n_points, dim_x = x_query.shape
            logits = self.model(x_query.reshape(batch_size * n_points, dim_x)).view(batch_size, n_points)
            return {
                "logits": logits,
                "mean": logits,
                "test_embeddings": x_query,
            }

        raise ValueError(f"Unexpected x_query shape: {tuple(x_query.shape)}")

    def log_prob_from_logits(self, logits: torch.Tensor, y_query: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1:
            y = y_query.view(-1)
            return -(logits - y).square()
        if logits.ndim == 2:
            y = y_query
            if y.ndim == 3 and y.shape[2] == 1:
                y = y.squeeze(-1)
            return -(logits - y).square()
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    def prediction_loss_from_logits(self, logits: torch.Tensor, y_query: torch.Tensor) -> torch.Tensor:
        return -self.log_prob_from_logits(logits, y_query)

    def enable_gradient_checkpointing(self) -> None:
        return None

    def disable_gradient_checkpointing(self) -> None:
        return None


def _episode() -> GPPriorSample:
    x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0.10, 0.20, 0.15, 0.25, 0.35, 0.30, 0.40], dtype=torch.float32)
    return GPPriorSample(
        context_x=x[:2],
        context_y=y[:2],
        query_x=x[2:5],
        query_y=y[2:5],
        target_x=x[5:],
        target_y=y[5:],
        plot_x=x,
        plot_y_true=y,
        kernel_type="rbf",
        lengthscales=torch.ones(2),
        output_scale=torch.ones(1),
    )


@pytest.mark.parametrize("use_remaining_budget", [False, True])
@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_al_policy_trainer_one_epoch_runs_in_batch_mode(
    use_remaining_budget: bool, gamma: float
) -> None:
    torch.manual_seed(0)
    backbone = _DummyBackbone()
    policy_head = CandidatePolicyHead(
        input_dim=backbone.embedding_dim, hidden_dim=16, num_layers=1,
        use_remaining_budget=use_remaining_budget, budget_scale=10,
    )
    policy_inputs = []
    policy_head.network[0].register_forward_pre_hook(
        lambda _module, args: policy_inputs.append(args[0].detach().clone())
    )
    optimizer = build_policy_optimizer(
        backbone=backbone,
        policy_head=policy_head,
        finetune_backbone=False,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )
    trainer = ALPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=0,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=gamma,
        reward_clamp_min=0.0,
        normalize_rewards=True,
        detach_policy_features=True,
        grad_clip_norm=1.0,
    )

    metrics = trainer.train_one_epoch(
        episodes=[_episode(), _episode()],
        num_acquisitions=2,
    )

    if use_remaining_budget:
        assert len(policy_inputs) == 2
        for features, remaining in zip(policy_inputs, [2, 1], strict=True):
            assert features.shape[-1] == backbone.embedding_dim + 1
            torch.testing.assert_close(
                features[:, -1], features.new_full((features.shape[0],), remaining / 10)
            )
    assert metrics["steps"] > 0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    assert torch.isfinite(torch.tensor(metrics["nll_loss"]))
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert torch.isfinite(torch.tensor(metrics["policy_entropy"]))
    assert metrics["policy_entropy"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["selected_query_y_mean"]))
    assert torch.isfinite(torch.tensor(metrics["target_log_prob_start"]))
    assert torch.isfinite(torch.tensor(metrics["target_log_prob_end"]))
    assert torch.isfinite(torch.tensor(metrics["target_log_prob_gain"]))
    assert torch.isfinite(torch.tensor(metrics["policy_target_score_start"]))
    assert torch.isfinite(torch.tensor(metrics["policy_target_score_end"]))
    assert torch.isfinite(torch.tensor(metrics["policy_target_score_gain"]))
    assert "reward_mean" not in metrics


def test_al_policy_trainer_requires_beta_above_one_for_step_pos() -> None:
    backbone = _DummyBackbone()
    policy_head = CandidatePolicyHead(input_dim=backbone.embedding_dim, hidden_dim=16, num_layers=1)
    optimizer = build_policy_optimizer(
        backbone=backbone,
        policy_head=policy_head,
        finetune_backbone=False,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )

    try:
        ALPolicyTrainer(
            backbone=backbone,
            policy_head=policy_head,
            optimizer=optimizer,
            finetune_epochs=0,
            nll_weight=1.0,
            policy_weight=1.0,
            gamma=1.0,
            reward_clamp_min=0.0,
            normalize_rewards=True,
            detach_policy_features=True,
            grad_clip_norm=1.0,
            policy_target_weighting="step_pos",
            policy_target_weight_beta=1.0,
        )
    except ValueError as exc:
        assert "step_pos" in str(exc)
    else:
        raise AssertionError("Expected step_pos weighting to require beta > 1.")


def test_al_policy_trainer_accepts_custom_step_pos_threshold() -> None:
    backbone = _DummyBackbone()
    policy_head = CandidatePolicyHead(input_dim=backbone.embedding_dim, hidden_dim=16, num_layers=1)
    optimizer = build_policy_optimizer(
        backbone=backbone,
        policy_head=policy_head,
        finetune_backbone=False,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )

    trainer = ALPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=0,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_clamp_min=0.0,
        normalize_rewards=True,
        detach_policy_features=True,
        grad_clip_norm=1.0,
        policy_target_weighting="step_pos",
        policy_target_weight_beta=3.0,
        policy_target_weight_threshold=1.0,
    )

    assert trainer.policy_target_weight_threshold == 1.0


def test_al_policy_trainer_auto_forward_strategy_matches_gradient_needs() -> None:
    episodes = [_episode()]

    backbone_head_only = _DummyBackbone()
    policy_head_head_only = CandidatePolicyHead(
        input_dim=backbone_head_only.embedding_dim,
        hidden_dim=16,
        num_layers=1,
    )
    optimizer_head_only = build_policy_optimizer(
        backbone=backbone_head_only,
        policy_head=policy_head_head_only,
        finetune_backbone=False,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )
    trainer_head_only = ALPolicyTrainer(
        backbone=backbone_head_only,
        policy_head=policy_head_head_only,
        optimizer=optimizer_head_only,
        finetune_epochs=0,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_clamp_min=0.0,
        normalize_rewards=True,
        detach_policy_features=True,
        grad_clip_norm=1.0,
    )
    trainer_head_only._rollout_batch(episodes=episodes, num_acquisitions=1)
    assert backbone_head_only.forward_calls == 2

    backbone_split = _DummyBackbone()
    policy_head_split = CandidatePolicyHead(
        input_dim=backbone_split.embedding_dim,
        hidden_dim=16,
        num_layers=1,
    )
    optimizer_split = build_policy_optimizer(
        backbone=backbone_split,
        policy_head=policy_head_split,
        finetune_backbone=True,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )
    trainer_split = ALPolicyTrainer(
        backbone=backbone_split,
        policy_head=policy_head_split,
        optimizer=optimizer_split,
        finetune_epochs=100,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_clamp_min=0.0,
        normalize_rewards=True,
        detach_policy_features=True,
        grad_clip_norm=1.0,
    )
    trainer_split._rollout_batch(episodes=episodes, num_acquisitions=1)
    assert backbone_split.forward_calls == 3

    backbone_joint = _DummyBackbone()
    policy_head_joint = CandidatePolicyHead(
        input_dim=backbone_joint.embedding_dim,
        hidden_dim=16,
        num_layers=1,
    )
    optimizer_joint = build_policy_optimizer(
        backbone=backbone_joint,
        policy_head=policy_head_joint,
        finetune_backbone=True,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )
    trainer_joint = ALPolicyTrainer(
        backbone=backbone_joint,
        policy_head=policy_head_joint,
        optimizer=optimizer_joint,
        finetune_epochs=100,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_clamp_min=0.0,
        normalize_rewards=True,
        detach_policy_features=False,
        grad_clip_norm=1.0,
    )
    trainer_joint._rollout_batch(episodes=episodes, num_acquisitions=1)
    assert backbone_joint.forward_calls == 2


def test_al_policy_trainer_rollout_backwards_target_nll_per_step() -> None:
    torch.manual_seed(1)
    backbone = _DummyBackbone()
    policy_head = CandidatePolicyHead(input_dim=backbone.embedding_dim, hidden_dim=16, num_layers=1)
    optimizer = build_policy_optimizer(
        backbone=backbone,
        policy_head=policy_head,
        finetune_backbone=True,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )
    trainer = ALPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=100,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_clamp_min=0.0,
        normalize_rewards=True,
        detach_policy_features=True,
        grad_clip_norm=1.0,
    )

    trainer.optimizer.zero_grad(set_to_none=True)
    rollout = trainer._rollout_batch(
        episodes=[_episode()],
        num_acquisitions=2,
        backward_target_nll=trainer._should_backward_target_nll_per_step(has_targets=True),
    )

    assert rollout.target_nll_backwarded is True


def test_target_aware_policy_head_changes_scores_when_targets_change() -> None:
    torch.manual_seed(0)
    head = TargetAwareTransformerPolicyHead(
        input_dim=2,
        num_heads=1,
        ffn_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    layer = head.layers[0]
    layer["ln_attn"] = nn.Identity()
    layer["ln_ffn"] = nn.Identity()
    with torch.no_grad():
        attn = layer["attn"]
        attn.in_proj_weight.zero_()
        attn.in_proj_bias.zero_()
        eye = torch.eye(2)
        attn.in_proj_weight[0:2].copy_(eye)
        attn.in_proj_weight[2:4].copy_(eye)
        attn.in_proj_weight[4:6].copy_(eye)
        attn.out_proj.weight.copy_(eye)
        attn.out_proj.bias.zero_()
        for param in layer["ffn"].parameters():
            param.zero_()
        head.scorer.weight.copy_(torch.tensor([[1.0, -0.5]], dtype=torch.float32))
        head.scorer.bias.zero_()

    query_embeddings = torch.tensor(
        [[[0.25, -0.75], [1.10, 0.40]]],
        dtype=torch.float32,
    )
    target_embeddings_a = torch.tensor(
        [[[0.40, -1.20], [1.70, 0.30]]],
        dtype=torch.float32,
    )
    target_embeddings_b = torch.tensor(
        [[[2.00, 0.50], [-0.30, 0.10]]],
        dtype=torch.float32,
    )

    scores_a = head(query_embeddings, target_embeddings_a)
    scores_b = head(query_embeddings, target_embeddings_b)

    assert scores_a.shape == (1, 2)
    assert scores_b.shape == (1, 2)
    assert not torch.allclose(scores_a, scores_b)


@pytest.mark.parametrize("gamma", [0.0, 1.0])
@pytest.mark.parametrize("use_remaining_budget", [False, True])
def test_al_policy_trainer_target_aware_one_epoch_runs(
    gamma: float, use_remaining_budget: bool
) -> None:
    torch.manual_seed(0)
    backbone = _DummyBackbone()
    policy_head = TargetAwareTransformerPolicyHead(
        input_dim=backbone.embedding_dim,
        num_heads=1,
        ffn_dim=16,
        num_layers=1,
        dropout=0.0,
        use_remaining_budget=use_remaining_budget,
        budget_scale=10,
    )
    policy_inputs = []
    policy_head.layers[0]["ffn"][0].register_forward_pre_hook(
        lambda _module, args: policy_inputs.append(args[0].detach().clone())
    )
    optimizer = build_policy_optimizer(
        backbone=backbone,
        policy_head=policy_head,
        finetune_backbone=False,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )
    trainer = ALPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=0,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=gamma,
        reward_clamp_min=0.0,
        normalize_rewards=True,
        detach_policy_features=True,
        grad_clip_norm=1.0,
    )

    metrics = trainer.train_one_epoch(
        episodes=[_episode(), _episode()],
        num_acquisitions=2,
    )

    if use_remaining_budget:
        assert len(policy_inputs) == 2
        for features, remaining in zip(policy_inputs, [2, 1], strict=True):
            assert features.shape[-1] == backbone.embedding_dim + 1
            torch.testing.assert_close(
                features[..., -1], features.new_full(features.shape[:-1], remaining / 10)
            )
    assert metrics["steps"] > 0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    assert torch.isfinite(torch.tensor(metrics["nll_loss"]))
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert torch.isfinite(torch.tensor(metrics["policy_entropy"]))
    assert torch.isfinite(torch.tensor(metrics["selected_query_y_mean"]))
    assert torch.isfinite(torch.tensor(metrics["target_log_prob_start"]))
    assert torch.isfinite(torch.tensor(metrics["target_log_prob_end"]))
    assert torch.isfinite(torch.tensor(metrics["target_log_prob_gain"]))
    assert torch.isfinite(torch.tensor(metrics["policy_target_score_start"]))
    assert torch.isfinite(torch.tensor(metrics["policy_target_score_end"]))
    assert torch.isfinite(torch.tensor(metrics["policy_target_score_gain"]))


def test_al_policy_trainer_uses_two_forwards_when_target_path_needs_grad() -> None:
    episodes = [_episode()]
    backbone = _DummyBackbone()
    policy_head = TargetAwareTransformerPolicyHead(
        input_dim=backbone.embedding_dim,
        num_heads=1,
        ffn_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    optimizer = build_policy_optimizer(
        backbone=backbone,
        policy_head=policy_head,
        finetune_backbone=True,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )
    trainer = ALPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=10,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_clamp_min=0.0,
        normalize_rewards=True,
        detach_policy_features=True,
        grad_clip_norm=1.0,
    )

    rollout = trainer._rollout_batch(episodes=episodes, num_acquisitions=1)

    assert backbone.forward_calls == 3
    assert rollout.steps == 1
    assert rollout.action_log_probs.shape == (1, 1)
    assert rollout.rewards.shape == (1, 1)


def _al_credit_trainer(gamma: float, *, normalize_rewards: bool = False) -> ALPolicyTrainer:
    backbone = _DummyBackbone()
    policy_head = CandidatePolicyHead(input_dim=2, hidden_dim=8, num_layers=1)
    return ALPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=torch.optim.SGD(policy_head.parameters(), lr=0.01),
        finetune_epochs=0,
        nll_weight=0.0,
        policy_weight=1.0,
        gamma=gamma,
        reward_clamp_min=0.0,
        normalize_rewards=normalize_rewards,
        detach_policy_features=True,
        grad_clip_norm=None,
        use_amp=False,
    )


@pytest.mark.parametrize(
    ("gamma", "expected_credit"),
    [(0.0, [0.0, 0.0, 4.0]), (0.5, [1.0, 2.0, 4.0]), (1.0, [4.0, 4.0, 4.0])],
)
def test_al_delayed_reward_credits_earlier_actions(
    gamma: float, expected_credit: list[float]
) -> None:
    trainer = _al_credit_trainer(gamma)
    target_scores = torch.tensor([[0.0, 0.0, 0.0, 4.0]], requires_grad=True)
    rewards = trainer._build_batch_rewards(target_scores)
    log_probs = torch.zeros(1, 3, requires_grad=True)

    trainer._build_policy_losses(action_log_probs=log_probs, rewards=rewards).sum().backward()

    torch.testing.assert_close(log_probs.grad, -torch.tensor([expected_credit]) / 3)
    assert target_scores.grad is None


def test_al_normalizes_reward_to_go_after_accumulation() -> None:
    trainer = _al_credit_trainer(0.5, normalize_rewards=True)
    log_probs = torch.zeros(2, 3, requires_grad=True)
    rewards = torch.tensor([[1.0, 2.0, 4.0], [2.0, 0.0, 1.0]], requires_grad=True)
    expected_returns = torch.tensor([[3.0, 4.0, 4.0], [2.25, 0.5, 1.0]])
    expected_credit = (
        expected_returns - expected_returns.mean(dim=1, keepdim=True)
    ) / (expected_returns.std(dim=1, unbiased=False, keepdim=True) + 1e-9)

    trainer._build_policy_losses(action_log_probs=log_probs, rewards=rewards).sum().backward()

    torch.testing.assert_close(log_probs.grad, -expected_credit / 3)
    assert rewards.grad is None


@pytest.mark.parametrize("normalize_rewards", [False, True])
@pytest.mark.parametrize("num_steps", [1, 3])
def test_al_myopic_credit_preserves_legacy_loss_and_gradients(
    normalize_rewards: bool, num_steps: int
) -> None:
    trainer = _al_credit_trainer(0.0, normalize_rewards=normalize_rewards)
    target_scores = torch.tensor([
        [-2.0, -1.5, -2.1, -1.0],
        [-1.0, -0.5, -0.25, -0.4],
    ])[:, :num_steps + 1]
    log_probs = torch.tensor([
        [-0.4, -0.8, -0.3], [-0.5, -0.7, -0.6],
    ])[:, :num_steps].requires_grad_()

    # The old default gamma=1 kept each clipped immediate reward unchanged.
    legacy_credit = (target_scores[:, 1:] - target_scores[:, :-1]).clamp(min=0.0).detach()
    if normalize_rewards and num_steps > 1:
        legacy_credit = (legacy_credit - legacy_credit.mean(dim=1, keepdim=True)) / (
            legacy_credit.std(dim=1, unbiased=False, keepdim=True) + 1e-9
        )
    legacy_loss = -(log_probs * legacy_credit).mean(dim=1)
    loss = trainer._build_policy_losses(
        action_log_probs=log_probs, rewards=trainer._build_batch_rewards(target_scores)
    )

    assert torch.equal(loss, legacy_loss)
    assert torch.equal(
        torch.autograd.grad(loss.sum(), log_probs)[0],
        torch.autograd.grad(legacy_loss.sum(), log_probs)[0],
    )


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("nan"), float("inf")])
def test_al_rejects_invalid_credit_discount(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
        _al_credit_trainer(gamma)


def test_gp_prior_sampler_sample_batch_returns_expected_shapes() -> None:
    sampler = GPPriorALSampler(
        dim_x=2,
        n_context_init=3,
        n_query=4,
        n_target=2,
        design_min=-1.0,
        design_max=1.0,
        plot_grid_size=8,
        noise_std=0.1,
        lengthscale_min=0.1,
        lengthscale_max=1.0,
        scale_min=0.5,
        scale_max=1.5,
        p_isotropic=0.5,
        seed=0,
    )

    episodes = sampler.sample_batch(batch_size=3)

    assert len(episodes) == 3
    for episode in episodes:
        assert episode.context_x.shape == (3, 2)
        assert episode.context_y.shape == (3,)
        assert episode.query_x.shape == (4, 2)
        assert episode.query_y.shape == (4,)
        assert episode.target_x.shape == (2, 2)
        assert episode.target_y.shape == (2,)
