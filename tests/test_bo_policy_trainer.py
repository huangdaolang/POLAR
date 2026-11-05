import pytest
import torch
from torch import nn

from polar.tasks import HPOBBOEpisode
from polar.trainers import BOPolicyTrainer, CandidatePolicyHead, build_policy_optimizer


class _DummyBackbone:
    def __init__(self) -> None:
        self.model = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1))
        self._device = torch.device("cpu")
        self.embedding_dim = 2
        self.forward_calls = 0
        self.gradient_checkpointing_enabled = False
        self.enable_gradient_checkpointing_calls = 0
        self.disable_gradient_checkpointing_calls = 0

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

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing_enabled = True
        self.enable_gradient_checkpointing_calls += 1

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing_enabled = False
        self.disable_gradient_checkpointing_calls += 1


def _episode() -> HPOBBOEpisode:
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
    y = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.35, 0.3, 0.4], dtype=torch.float32)
    return HPOBBOEpisode(
        search_space_id="dummy",
        dataset_id="d0",
        split="train",
        seed=None,
        context_x=x[:2],
        context_y=y[:2],
        query_x=x[2:5],
        query_y=y[2:5],
        target_x=x[5:],
        target_y=y[5:],
        y_best=float(y.max().item()),
        y_min=float(y.min().item()),
    )


def _episode_without_target() -> HPOBBOEpisode:
    episode = _episode()
    return HPOBBOEpisode(
        search_space_id=episode.search_space_id,
        dataset_id=episode.dataset_id,
        split=episode.split,
        seed=episode.seed,
        context_x=episode.context_x,
        context_y=episode.context_y,
        query_x=episode.query_x,
        query_y=episode.query_y,
        target_x=None,
        target_y=None,
        y_best=episode.y_best,
        y_min=episode.y_min,
    )


@pytest.mark.parametrize("use_remaining_budget", [False, True])
@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_bo_policy_trainer_one_epoch_runs(use_remaining_budget: bool, gamma: float) -> None:
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
    trainer = BOPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=0,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=gamma,
        reward_type="regret_reduction",
        reward_log_epsilon=1.0e-6,
        reward_scale_by_range=True,
        reward_no_improve_penalty=0.0,
        reward_clamp_min=None,
        normalize_rewards=True,
        reward_normalization_scope="batch",
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
    assert metrics["nll_loss"] == 0.0
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert torch.isfinite(torch.tensor(metrics["policy_entropy"]))
    assert metrics["policy_entropy"] >= 0.0
    assert metrics["best_y_improvement"] >= 0.0
    assert metrics["target_log_prob_start"] == 0.0
    assert metrics["target_log_prob_end"] == 0.0


def test_bo_policy_trainer_runs_without_target_auxiliary_loss() -> None:
    torch.manual_seed(0)
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
    trainer = BOPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=0,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_type="regret_reduction",
        reward_log_epsilon=1.0e-6,
        reward_scale_by_range=True,
        reward_no_improve_penalty=0.0,
        reward_clamp_min=None,
        normalize_rewards=True,
        reward_normalization_scope="batch",
        detach_policy_features=True,
        grad_clip_norm=1.0,
    )

    metrics = trainer.train_one_epoch(
        episodes=[_episode_without_target(), _episode_without_target()],
        num_acquisitions=2,
    )

    assert metrics["steps"] > 0
    assert metrics["nll_loss"] == 0.0
    assert torch.isfinite(torch.tensor(metrics["policy_entropy"]))
    assert metrics["best_y_improvement"] >= 0.0
    assert metrics["target_log_prob_start"] == 0.0
    assert metrics["target_log_prob_end"] == 0.0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))


def test_gradient_checkpointing_tracks_backbone_trainability() -> None:
    torch.manual_seed(2)
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
    trainer = BOPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=5,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_type="regret_reduction",
        reward_log_epsilon=1.0e-6,
        reward_scale_by_range=True,
        reward_no_improve_penalty=0.0,
        reward_clamp_min=None,
        normalize_rewards=True,
        reward_normalization_scope="batch",
        detach_policy_features=True,
        grad_clip_norm=1.0,
        gradient_checkpointing=True,
    )

    assert backbone.gradient_checkpointing_enabled is True
    assert backbone.enable_gradient_checkpointing_calls == 1
    assert backbone.disable_gradient_checkpointing_calls == 0

    trainer.freeze_backbone()

    assert backbone.gradient_checkpointing_enabled is False
    assert backbone.enable_gradient_checkpointing_calls == 1
    assert backbone.disable_gradient_checkpointing_calls == 1


def test_head_only_training_does_not_enable_gradient_checkpointing() -> None:
    torch.manual_seed(3)
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
    _trainer = BOPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=0,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_type="regret_reduction",
        reward_log_epsilon=1.0e-6,
        reward_scale_by_range=True,
        reward_no_improve_penalty=0.0,
        reward_clamp_min=None,
        normalize_rewards=True,
        reward_normalization_scope="batch",
        detach_policy_features=True,
        grad_clip_norm=1.0,
        gradient_checkpointing=True,
    )

    assert backbone.gradient_checkpointing_enabled is False
    assert backbone.enable_gradient_checkpointing_calls == 0
    assert backbone.disable_gradient_checkpointing_calls == 0


def test_bo_rollout_backwards_target_nll_per_step_when_backbone_is_split() -> None:
    torch.manual_seed(4)
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
    trainer = BOPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=5,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_type="regret_reduction",
        reward_log_epsilon=1.0e-6,
        reward_scale_by_range=True,
        reward_no_improve_penalty=0.0,
        reward_clamp_min=None,
        normalize_rewards=True,
        reward_normalization_scope="batch",
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
    assert rollout.nll_losses.requires_grad is False
    assert any(param.grad is not None for param in backbone.model.parameters())
    assert all(param.grad is None for param in policy_head.parameters())


def test_log_regret_reward_matches_formula() -> None:
    backbone = _DummyBackbone()
    policy_head = CandidatePolicyHead(input_dim=backbone.embedding_dim, hidden_dim=8, num_layers=1)
    optimizer = build_policy_optimizer(
        backbone=backbone,
        policy_head=policy_head,
        finetune_backbone=False,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )
    trainer = BOPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=0,
        nll_weight=1.0,
        policy_weight=1.0,
        gamma=1.0,
        reward_type="log_regret_reduction",
        reward_log_epsilon=1.0e-9,
        reward_scale_by_range=False,
        reward_no_improve_penalty=0.0,
        reward_clamp_min=None,
        normalize_rewards=False,
        reward_normalization_scope="episode",
        detach_policy_features=True,
        grad_clip_norm=None,
    )
    prev_best_y = torch.tensor(0.90)
    new_best_y = torch.tensor(0.99)
    reward = trainer._build_step_reward(
        prev_best_y=prev_best_y,
        new_best_y=new_best_y,
        y_best=1.0,
        range_scale=1.0,
    )
    expected = torch.log(torch.tensor(0.1 + 1.0e-9)) - torch.log(torch.tensor(0.01 + 1.0e-9))
    assert torch.allclose(reward, expected, atol=1e-6)


def _credit_trainer(
    gamma: float,
    *,
    normalize_rewards: bool = False,
    scope: str = "batch",
) -> BOPolicyTrainer:
    backbone = _DummyBackbone()
    policy_head = CandidatePolicyHead(input_dim=2, hidden_dim=8, num_layers=1)
    return BOPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=torch.optim.SGD(policy_head.parameters(), lr=0.01),
        finetune_epochs=0,
        nll_weight=0.0,
        policy_weight=1.0,
        gamma=gamma,
        reward_type="regret_reduction",
        reward_log_epsilon=1e-6,
        reward_scale_by_range=True,
        reward_no_improve_penalty=0.0,
        reward_clamp_min=None,
        normalize_rewards=normalize_rewards,
        reward_normalization_scope=scope,
        detach_policy_features=True,
        grad_clip_norm=None,
        use_amp=False,
    )


@pytest.mark.parametrize(
    ("gamma", "expected_credit"),
    [(0.0, [0.0, 0.0, 4.0]), (0.5, [1.0, 2.0, 4.0]), (1.0, [4.0, 4.0, 4.0])],
)
def test_bo_delayed_reward_credits_earlier_actions(
    gamma: float, expected_credit: list[float]
) -> None:
    trainer = _credit_trainer(gamma)
    log_probs = torch.zeros(1, 3, requires_grad=True)
    rewards = torch.tensor([[0.0, 0.0, 4.0]], requires_grad=True)
    loss = trainer._build_policy_losses(action_log_probs=log_probs, rewards=rewards).sum()
    loss.backward()

    torch.testing.assert_close(log_probs.grad, -torch.tensor([expected_credit]) / 3)
    assert rewards.grad is None


@pytest.mark.parametrize("scope", ["batch", "episode"])
def test_bo_normalizes_reward_to_go_after_accumulation(scope: str) -> None:
    trainer = _credit_trainer(0.5, normalize_rewards=True, scope=scope)
    log_probs = torch.zeros(2, 3, requires_grad=True)
    rewards = torch.tensor([[1.0, 2.0, 4.0], [2.0, 0.0, 1.0]])
    expected_returns = torch.tensor([[3.0, 4.0, 4.0], [2.25, 0.5, 1.0]])
    dim = 1 if scope == "episode" else None
    expected_credit = (expected_returns - expected_returns.mean(dim=dim, keepdim=True)) / (
        expected_returns.std(dim=dim, keepdim=True, unbiased=False) + 1e-9
    )

    trainer._build_policy_losses(action_log_probs=log_probs, rewards=rewards).sum().backward()
    torch.testing.assert_close(log_probs.grad, -expected_credit / 3)


@pytest.mark.parametrize("normalize_rewards", [False, True])
@pytest.mark.parametrize("scope", ["batch", "episode"])
def test_bo_myopic_credit_preserves_legacy_loss_and_gradients(
    normalize_rewards: bool, scope: str
) -> None:
    trainer = _credit_trainer(0.0, normalize_rewards=normalize_rewards, scope=scope)
    rewards = torch.tensor([[0.0, 0.5, 0.25], [1.0, 0.0, 0.5]])
    log_probs = torch.tensor([[-0.4, -0.8, -0.3], [-0.5, -0.7, -0.6]], requires_grad=True)

    # Before reward-to-go support, gamma=1 left immediate rewards unchanged.
    legacy_credit = rewards
    if normalize_rewards:
        dim = 1 if scope == "episode" else None
        legacy_credit = (rewards - rewards.mean(dim=dim, keepdim=True)) / (
            rewards.std(dim=dim, keepdim=True, unbiased=False) + 1e-9
        )
    legacy_loss = -(log_probs * legacy_credit).mean(dim=1)
    loss = trainer._build_policy_losses(action_log_probs=log_probs, rewards=rewards)

    assert torch.equal(loss, legacy_loss)
    assert torch.equal(
        torch.autograd.grad(loss.sum(), log_probs)[0],
        torch.autograd.grad(legacy_loss.sum(), log_probs)[0],
    )


def test_bo_myopic_rollout_keeps_rewards_after_first_step(monkeypatch) -> None:
    trainer = _credit_trainer(0.0)
    monkeypatch.setattr(
        "polar.trainers.bo_policy.Categorical.sample",
        lambda distribution: torch.zeros(
            distribution.batch_shape, dtype=torch.long, device=distribution.logits.device
        ),
    )
    rollout = trainer._rollout_batch(episodes=[_episode()], num_acquisitions=3)

    # Selecting candidates in order improves best-y from .2 to .25 and then .35.
    torch.testing.assert_close(rollout.rewards, torch.tensor([[0.0, 0.05 / 0.3, 0.1 / 0.3]]))


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("nan"), float("inf")])
def test_bo_rejects_invalid_credit_discount(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
        _credit_trainer(gamma)
