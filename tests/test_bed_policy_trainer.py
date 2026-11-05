import math
from functools import partial
from types import SimpleNamespace

import torch
from torch import nn

from polar.evaluation import (
    compute_bed_eig_bounds_for_trajectory,
    evaluate_bed_policy,
    rollout_bed_policy,
)
from polar.tasks import CESSampler, create_bed_task
from polar.tasks.bed import BEDEpisode
from polar.tasks.ces import CensoredSigmoidNormal, ces_log_likelihood
from polar.tasks.location_finding import (
    LocationFindingEpisode,
    LocationFindingSampler,
    location_finding_log_likelihood,
)
from polar.trainers import BEDPolicyTrainer, CandidatePolicyHead, build_policy_optimizer
from polar.utils.eig import (
    compute_cumulative_log_likelihood,
    compute_eig_bounds,
    compute_eig_bounds_from_trajectory,
    compute_snmc,
    compute_snmc_from_trajectory,
    compute_spce,
    compute_spce_from_trajectory,
    dense_spce_reward,
)


class _DummyBackbone:
    def __init__(self, dim_x: int = 2) -> None:
        self.model = nn.Sequential(nn.Linear(dim_x, 8), nn.GELU(), nn.Linear(8, 1))
        self._device = torch.device("cpu")
        self.embedding_dim = dim_x

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_context(self, x_context: torch.Tensor, y_context: torch.Tensor):
        return {"x": x_context, "y": y_context}

    def forward_with_embeddings(self, x_query: torch.Tensor, context_state):
        _ = context_state
        if x_query.ndim == 2:
            logits = self.model(x_query).view(-1)
            return {"logits": logits, "mean": logits, "test_embeddings": x_query}
        if x_query.ndim == 3:
            batch_size, n_query, dim_x = x_query.shape
            logits = self.model(x_query.reshape(batch_size * n_query, dim_x)).view(batch_size, n_query)
            return {"logits": logits, "mean": logits, "test_embeddings": x_query}
        raise ValueError(f"Unexpected x_query shape: {tuple(x_query.shape)}")

    def log_prob_from_logits(self, logits: torch.Tensor, y_query: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1:
            return -(logits - y_query.view(-1)).square()
        if logits.ndim == 2:
            y = y_query
            if y.ndim == 3 and y.shape[2] == 1:
                y = y.squeeze(-1)
            return -(logits - y).square()
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


_LOCATION_LOG_LIK_FN = partial(
    location_finding_log_likelihood,
    base_signal=0.1,
    max_signal=1e-4,
    noise_scale=0.5,
)


def _toy_log_lik(y: torch.Tensor, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    theta_term = theta.sum(dim=-1)
    x_term = x.squeeze(0).sum(dim=-1, keepdim=False).unsqueeze(0)
    y_term = y
    return theta_term + 0.5 * x_term - 0.25 * y_term


def _make_location_episode(
    *,
    K: int = 2,
    d: int = 2,
    L: int = 3,
    n_ctx: int = 1,
    n_query: int = 5,
    n_target: int = 0,
) -> LocationFindingEpisode:
    sampler = LocationFindingSampler(
        dim_x=d,
        K=K,
        n_contrastive=L,
        n_context_init=n_ctx,
        n_query=n_query,
        n_target=n_target,
        design_min=-4.0,
        design_max=4.0,
        seed=42,
    )
    return sampler.sample_episode()


def _make_ces_episode(*, L: int = 3, n_ctx: int = 3, n_query: int = 5, n_target: int = 0):
    sampler = CESSampler(
        n_goods=3,
        n_contrastive=L,
        n_context_init=n_ctx,
        n_query=n_query,
        n_target=n_target,
        design_min=0.0,
        design_max=100.0,
        seed=7,
    )
    return sampler.sample_episode()


def _build_trainer(
    *, log_likelihood_fn, n_target: int = 0, finetune_epochs: int = 0,
    dim_x: int = 2, use_remaining_budget: bool = False,
):
    backbone = _DummyBackbone(dim_x=dim_x)
    policy_head = CandidatePolicyHead(
        input_dim=backbone.embedding_dim, hidden_dim=16, num_layers=1,
        use_remaining_budget=use_remaining_budget, budget_scale=10,
    )
    optimizer = build_policy_optimizer(
        backbone=backbone,
        policy_head=policy_head,
        finetune_backbone=finetune_epochs > 0,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )
    nll_weight = 1.0 if (finetune_epochs > 0 and n_target > 0) else 0.0
    return BEDPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=finetune_epochs,
        log_likelihood_fn=log_likelihood_fn,
        nll_weight=nll_weight,
        policy_weight=1.0,
        gamma=1.0,
        use_baseline=True,
        baseline_momentum=0.99,
        normalize_rewards=True,
        detach_policy_features=True,
        grad_clip_norm=1.0,
    )


def test_dense_spce_reward_basic_shapes():
    L_plus_1, batch_size = 5, 3
    step_ll = torch.randn(L_plus_1, batch_size)
    cum = torch.zeros(L_plus_1, batch_size)
    reward, new_cum = dense_spce_reward(step_ll, cum)
    assert reward.shape == (batch_size,)
    assert new_cum.shape == (L_plus_1, batch_size)


def test_dense_spce_reward_telescopes_to_spce():
    torch.manual_seed(123)
    L_plus_1, batch_size, steps = 10, 4, 8
    cum = torch.zeros(L_plus_1, batch_size)
    total_reward = torch.zeros(batch_size)
    for _ in range(steps):
        step_ll = torch.randn(L_plus_1, batch_size)
        reward, cum = dense_spce_reward(step_ll, cum)
        total_reward += reward
    assert torch.allclose(total_reward, compute_spce(cum), atol=1e-5)


def test_compute_eig_bounds_matches_individual_bounds():
    torch.manual_seed(7)
    cum = torch.randn(6, 4)
    spce, snmc = compute_eig_bounds(cum)
    assert torch.allclose(spce, compute_spce(cum))
    assert torch.allclose(snmc, compute_snmc(cum))


def test_compute_spce_from_trajectory():
    torch.manual_seed(42)
    batch_size, steps, K, dim_x = 2, 5, 2, 2
    L = 9
    theta_all = torch.randn(L + 1, batch_size, K, dim_x)
    x_traj = torch.randn(batch_size, steps, dim_x)
    y_traj = torch.zeros(batch_size, steps)
    for step in range(steps):
        for batch_idx in range(batch_size):
            x = x_traj[batch_idx, step]
            theta = theta_all[0, batch_idx]
            sq = (theta - x.unsqueeze(0)).pow(2).sum(-1)
            signal = torch.log(torch.tensor(0.1) + (1.0 / (1e-4 + sq)).sum())
            y_traj[batch_idx, step] = signal + 0.5 * torch.randn(1).item()

    spce = compute_spce_from_trajectory(_LOCATION_LOG_LIK_FN, y_traj, x_traj, theta_all)
    assert spce.shape == (batch_size,)
    assert torch.all(torch.isfinite(spce))


def test_compute_eig_bounds_from_trajectory():
    torch.manual_seed(24)
    batch_size, steps, K, dim_x = 2, 4, 2, 2
    L = 5
    theta_all = torch.randn(L + 1, batch_size, K, dim_x)
    x_traj = torch.randn(batch_size, steps, dim_x)
    y_traj = torch.randn(batch_size, steps)

    spce, snmc = compute_eig_bounds_from_trajectory(
        _LOCATION_LOG_LIK_FN,
        y_traj,
        x_traj,
        theta_all,
    )

    assert spce.shape == (batch_size,)
    assert snmc.shape == (batch_size,)
    assert torch.all(torch.isfinite(spce))
    assert torch.all(torch.isfinite(snmc))


def test_location_finding_log_likelihood_shapes():
    batch_size, L, K, dim_x = 4, 9, 2, 2
    y = torch.randn(1, batch_size)
    x = torch.randn(1, batch_size, dim_x)
    theta = torch.randn(L + 1, batch_size, K, dim_x)
    ll = location_finding_log_likelihood(y, x, theta)
    assert ll.shape == (L + 1, batch_size)
    assert torch.all(torch.isfinite(ll))


def test_location_finding_log_likelihood_values():
    theta = torch.tensor([[[[0.0]]]])
    x = torch.tensor([[[0.0]]])
    signal = math.log(0.1 + 1.0 / 1e-4)
    y = torch.tensor([[signal]])
    ll = location_finding_log_likelihood(y, x, theta)
    expected = -0.5 * math.log(2 * math.pi) - math.log(0.5)
    assert ll.shape == (1, 1)
    assert abs(ll.item() - expected) < 1e-5


def test_censored_sigmoid_normal_boundary_masses():
    lower = 0.2
    upper = 0.8
    dist = CensoredSigmoidNormal(
        loc=torch.tensor(0.0),
        scale=torch.tensor(1.0),
        lower=lower,
        upper=upper,
    )
    lower_logit = math.log(lower / (1.0 - lower))
    upper_logit = math.log(upper / (1.0 - upper))
    assert torch.allclose(
        dist.log_prob(torch.tensor(lower)),
        torch.special.log_ndtr(torch.tensor(lower_logit)),
        atol=1e-6,
    )
    assert torch.allclose(
        dist.log_prob(torch.tensor(upper)),
        torch.special.log_ndtr(torch.tensor(-upper_logit)),
        atol=1e-6,
    )


def test_censored_sigmoid_normal_rsample_is_clipped():
    dist = CensoredSigmoidNormal(
        loc=torch.tensor([0.0]),
        scale=torch.tensor([1.0]),
        lower=0.1,
        upper=0.9,
    )
    samples = dist.rsample((256,))
    assert samples.shape == (256, 1)
    assert torch.all(samples >= 0.1)
    assert torch.all(samples <= 0.9)


def test_ces_log_likelihood_shapes_and_boundaries():
    y = torch.tensor([[2.3841858e-07, 0.5, 1.0 - 2.3841858e-07]], dtype=torch.float32)
    x = torch.rand(1, 3, 6) * 100.0
    theta = torch.tensor(
        [
            [[0.3, 0.2, 0.3, 0.5, 1.0], [0.6, 0.5, 0.2, 0.3, -0.5], [0.8, 0.3, 0.4, 0.3, 0.0]],
            [[0.4, 0.3, 0.3, 0.4, 0.1], [0.5, 0.2, 0.3, 0.5, 1.5], [0.7, 0.2, 0.5, 0.3, -1.0]],
        ],
        dtype=torch.float32,
    )
    ll = ces_log_likelihood(y, x, theta)
    assert ll.shape == (2, 3)
    assert torch.all(torch.isfinite(ll))


def test_location_sampler_shapes_without_targets():
    sampler = LocationFindingSampler(
        dim_x=2,
        K=2,
        n_contrastive=4,
        n_context_init=1,
        n_query=10,
        n_target=0,
        seed=0,
    )
    episode = sampler.sample_episode()
    assert episode.theta_all.shape == (5, 2, 2)
    assert episode.context_x.shape == (1, 2)
    assert episode.context_y.shape == (1,)
    assert episode.query_x.shape == (10, 2)
    assert episode.query_y.shape == (10,)
    assert episode.target_x is None
    assert episode.target_y is None


def test_location_sampler_shapes_with_targets():
    sampler = LocationFindingSampler(
        dim_x=2,
        K=2,
        n_contrastive=4,
        n_context_init=1,
        n_query=10,
        n_target=5,
        seed=0,
    )
    episode = sampler.sample_episode()
    assert episode.target_x is not None
    assert episode.target_y is not None
    assert episode.target_x.shape == (5, 2)
    assert episode.target_y.shape == (5,)


def test_location_sampler_batch():
    sampler = LocationFindingSampler(
        dim_x=2,
        K=2,
        n_contrastive=3,
        n_context_init=1,
        n_query=8,
        n_target=0,
        seed=0,
    )
    assert len(sampler.sample_batch(batch_size=3)) == 3


def test_ces_sampler_shapes_and_ranges():
    sampler = CESSampler(
        n_goods=3,
        n_contrastive=4,
        n_context_init=2,
        n_query=7,
        n_target=3,
        seed=0,
    )
    episode = sampler.sample_episode()
    assert episode.theta_all.shape == (5, 5)
    assert episode.context_x.shape == (2, 6)
    assert episode.query_x.shape == (7, 6)
    assert episode.target_x is not None and episode.target_x.shape == (3, 6)
    assert torch.all(episode.context_x >= sampler.design_min)
    assert torch.all(episode.query_x <= sampler.design_max)
    alpha = episode.theta_all[:, 1:4]
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(alpha.shape[0]), atol=1e-5)


def test_ces_sampler_log_likelihood_uses_physical_design_scale() -> None:
    sampler = CESSampler(
        n_goods=3,
        n_contrastive=4,
        n_context_init=2,
        n_query=7,
        n_target=0,
        seed=0,
    )
    x_physical = torch.tensor(
        [
            [
                [5.0, 15.0, 25.0, 75.0, 85.0, 95.0],
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                [12.5, 22.5, 32.5, 42.5, 52.5, 62.5],
                [7.0, 17.0, 27.0, 37.0, 47.0, 57.0],
            ]
        ],
        dtype=torch.float32,
    )
    x_model = sampler._to_model_designs(x_physical)
    y = torch.full((1, 4), 0.5, dtype=torch.float32)
    theta = torch.tensor(
        [
            [[0.3, 0.2, 0.3, 0.5, 1.0], [0.6, 0.5, 0.2, 0.3, -0.5], [0.8, 0.3, 0.4, 0.3, 0.0], [0.4, 0.3, 0.3, 0.4, 0.1]],
            [[0.4, 0.3, 0.3, 0.4, 0.1], [0.5, 0.2, 0.3, 0.5, 1.5], [0.7, 0.2, 0.5, 0.3, -1.0], [0.2, 0.5, 0.1, 0.4, 0.7]],
        ],
        dtype=torch.float32,
    )

    actual = sampler.log_likelihood(y, x_model, theta)
    expected = ces_log_likelihood(
        y,
        x_physical,
        theta,
        n_goods=sampler.n_goods,
        noise_scale=sampler.noise_scale,
        epsilon=sampler.epsilon,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_ces_sampler_standardizes_designs_for_model() -> None:
    sampler = CESSampler(
        n_goods=3,
        n_contrastive=4,
        n_context_init=2,
        n_query=7,
        n_target=0,
        design_min=0.0,
        design_max=100.0,
        seed=0,
    )
    x_physical = torch.tensor([[0.0, 25.0, 50.0, 75.0, 100.0, 60.0]], dtype=torch.float32)
    expected_std = (100.0 - 0.0) / math.sqrt(12.0)
    x_model = sampler._to_model_designs(x_physical)

    assert sampler.model_design_mean == 50.0
    assert math.isclose(sampler.model_design_std, expected_std)
    assert torch.isclose(x_model[0, 2], torch.tensor(0.0))
    assert torch.allclose(sampler._to_physical_designs(x_model), x_physical)


def test_create_bed_task_factory_supports_both_tasks():
    location_cfg = SimpleNamespace(
        name="location_finding",
        dim_x=2,
        K=2,
        n_contrastive=3,
        n_context_init=1,
        n_query=5,
        n_target=0,
        design_min=0.0,
        design_max=1.0,
        theta_prior="uniform",
        signal=SimpleNamespace(base_signal=0.1, max_signal=1e-4, noise_scale=0.5),
    )
    ces_cfg = SimpleNamespace(
        name="ces",
        n_goods=3,
        n_contrastive=3,
        n_context_init=2,
        n_query=5,
        n_target=0,
        design_min=0.0,
        design_max=100.0,
        noise_scale=0.005,
        epsilon=2.384185791015625e-07,
    )
    location_task = create_bed_task(location_cfg, seed=0)
    ces_task = create_bed_task(ces_cfg, seed=0)
    assert location_task.name == "location_finding"
    assert ces_task.name == "ces"


def test_bed_training_and_evaluation_share_capped_remaining_budgets() -> None:
    trainer = _build_trainer(
        log_likelihood_fn=_LOCATION_LOG_LIK_FN, use_remaining_budget=True
    )
    backbone_before = {
        key: value.clone() for key, value in trainer.backbone.model.state_dict().items()
    }
    policy_inputs = []
    trainer.policy_head.network[0].register_forward_pre_hook(
        lambda _module, args: policy_inputs.append(args[0].detach().clone())
    )
    episode = _make_location_episode(n_query=3)
    trainer.train_one_epoch(episodes=[episode, episode], num_acquisitions=5)
    rollout_bed_policy(
        backbone=trainer.backbone, policy_head=trainer.policy_head,
        episode=episode, num_acquisitions=5, greedy=True,
    )

    assert len(policy_inputs) == 6
    for features, remaining in zip(policy_inputs, [3, 2, 1, 3, 2, 1], strict=True):
        assert features.shape[-1] == trainer.backbone.embedding_dim + 1
        torch.testing.assert_close(
            features[:, -1], features.new_full((features.shape[0],), remaining / 10)
        )
    for key, before in backbone_before.items():
        torch.testing.assert_close(trainer.backbone.model.state_dict()[key], before)


def test_bed_trainer_one_epoch_head_only_location():
    torch.manual_seed(0)
    trainer = _build_trainer(log_likelihood_fn=_LOCATION_LOG_LIK_FN, n_target=0, dim_x=2)
    episodes = [_make_location_episode() for _ in range(2)]
    metrics = trainer.train_one_epoch(episodes=episodes, num_acquisitions=3)
    assert metrics["steps"] > 0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert torch.isfinite(torch.tensor(metrics["spce_estimate"]))
    assert metrics["nll_loss"] == 0.0


def test_bed_trainer_one_epoch_full_finetune_with_targets_location():
    torch.manual_seed(0)
    trainer = _build_trainer(
        log_likelihood_fn=_LOCATION_LOG_LIK_FN,
        n_target=3,
        finetune_epochs=100,
        dim_x=2,
    )
    episodes = [_make_location_episode(n_target=3) for _ in range(2)]
    metrics = trainer.train_one_epoch(episodes=episodes, num_acquisitions=3)
    assert metrics["steps"] > 0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    assert torch.isfinite(torch.tensor(metrics["nll_loss"]))
    assert metrics["nll_loss"] != 0.0


def test_bed_rollout_backwards_target_nll_per_step_for_full_finetune() -> None:
    torch.manual_seed(4)
    trainer = _build_trainer(
        log_likelihood_fn=_LOCATION_LOG_LIK_FN,
        n_target=3,
        finetune_epochs=100,
        dim_x=2,
    )
    episodes = [_make_location_episode(n_target=3)]

    trainer.optimizer.zero_grad(set_to_none=True)
    rollout = trainer._rollout_batch(
        episodes=episodes,
        num_acquisitions=3,
        backward_target_nll=trainer._should_backward_target_nll_per_step(has_targets=True),
    )

    assert rollout.target_nll_backwarded is True
    assert rollout.nll_losses.requires_grad is False
    assert any(param.grad is not None for param in trainer.backbone.model.parameters())
    assert all(param.grad is None for param in trainer.policy_head.parameters())


def test_bed_trainer_one_epoch_head_only_ces():
    torch.manual_seed(0)
    ces_task = CESSampler(
        n_goods=3,
        n_contrastive=3,
        n_context_init=2,
        n_query=6,
        n_target=0,
        seed=0,
    )
    trainer = _build_trainer(log_likelihood_fn=ces_task.log_likelihood, n_target=0, dim_x=6)
    episodes = [ces_task.sample_episode() for _ in range(2)]
    metrics = trainer.train_one_epoch(episodes=episodes, num_acquisitions=3)
    assert metrics["steps"] > 0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    assert torch.isfinite(torch.tensor(metrics["spce_estimate"]))


def test_bed_trainer_head_only_freezes_backbone():
    trainer = _build_trainer(log_likelihood_fn=_LOCATION_LOG_LIK_FN, finetune_epochs=0, dim_x=2)
    for param in trainer.backbone.model.parameters():
        assert not param.requires_grad


def test_bed_trainer_myopic_rollout_is_finite():
    torch.manual_seed(0)
    episodes = [_make_location_episode(L=3, n_query=6) for _ in range(2)]

    backbone = _DummyBackbone(dim_x=2)
    policy_head = CandidatePolicyHead(input_dim=2, hidden_dim=16, num_layers=1)
    optimizer = build_policy_optimizer(
        backbone=backbone,
        policy_head=policy_head,
        finetune_backbone=False,
        lr_policy=1e-3,
        lr_backbone=1e-5,
        weight_decay_policy=0.0,
        weight_decay_backbone=0.0,
    )
    trainer = BEDPolicyTrainer(
        backbone=backbone,
        policy_head=policy_head,
        optimizer=optimizer,
        finetune_epochs=0,
        log_likelihood_fn=_LOCATION_LOG_LIK_FN,
        nll_weight=0.0,
        policy_weight=1.0,
        gamma=0.0,
        use_baseline=False,
        baseline_momentum=0.99,
        normalize_rewards=False,
        detach_policy_features=True,
        grad_clip_norm=None,
    )
    rollout = trainer._rollout_batch(episodes=episodes, num_acquisitions=3)
    assert torch.all(torch.isfinite(rollout.rewards))
    assert torch.all(torch.isfinite(rollout.spce_estimate))


def test_bed_rollout_first_reward_uses_context_prefix_spce():
    torch.manual_seed(5)
    trainer = _build_trainer(log_likelihood_fn=_toy_log_lik, n_target=0, dim_x=2)
    episode = BEDEpisode(
        theta_all=torch.tensor(
            [[0.3, -0.1], [0.5, 0.2], [-0.4, 0.8]],
            dtype=torch.float32,
        ),
        context_x=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        context_y=torch.tensor([0.25], dtype=torch.float32),
        query_x=torch.tensor([[0.5, -1.0]], dtype=torch.float32),
        query_y=torch.tensor([-0.3], dtype=torch.float32),
        target_x=None,
        target_y=None,
    )

    rollout = trainer._rollout_batch(episodes=[episode], num_acquisitions=1)

    theta_all = episode.theta_all.unsqueeze(1)
    initial_cum = compute_cumulative_log_likelihood(
        _toy_log_lik,
        episode.context_y.unsqueeze(0),
        episode.context_x.unsqueeze(0),
        theta_all,
    )
    first_step_cum = compute_cumulative_log_likelihood(
        _toy_log_lik,
        episode.query_y.unsqueeze(0),
        episode.query_x.unsqueeze(0),
        theta_all,
    )
    expected_reward = compute_spce(initial_cum + first_step_cum) - compute_spce(initial_cum)
    expected_spce = compute_spce(initial_cum + first_step_cum)

    assert torch.allclose(rollout.rewards[:, 0], expected_reward, atol=1e-5)
    assert torch.allclose(rollout.spce_estimate, expected_spce, atol=1e-5)


def test_evaluate_bed_policy_location_greedy():
    torch.manual_seed(0)
    backbone = _DummyBackbone(dim_x=2)
    backbone.model.eval()
    for param in backbone.model.parameters():
        param.requires_grad_(False)

    policy_head = CandidatePolicyHead(input_dim=2, hidden_dim=16, num_layers=1)
    task = LocationFindingSampler(
        dim_x=2,
        K=2,
        n_contrastive=3,
        n_context_init=1,
        n_query=10,
        n_target=0,
        seed=42,
    )
    metrics = evaluate_bed_policy(
        backbone=backbone,
        policy_head=policy_head,
        task=task,
        num_acquisitions=3,
        n_eval_episodes=4,
        n_contrastive_eval=50,
        greedy=True,
    )
    assert metrics["eval/n_episodes"] == 4.0
    assert math.isfinite(metrics["eval/spce_mean"])
    assert math.isfinite(metrics["eval/spce_stderr"])
    assert math.isfinite(metrics["eval/snmc_mean"])
    assert math.isfinite(metrics["eval/snmc_stderr"])


def test_compute_bed_eig_bounds_for_trajectory_uses_full_trajectory_spce():
    theta_true = torch.tensor([0.3, -0.1], dtype=torch.float32)
    theta_eval = torch.tensor([[0.5, 0.2], [-0.4, 0.8]], dtype=torch.float32)
    context_x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    context_y = torch.tensor([0.25], dtype=torch.float32)
    x_traj = [
        torch.tensor([0.5, -1.0], dtype=torch.float32),
        torch.tensor([1.5, 0.25], dtype=torch.float32),
    ]
    y_traj = [
        torch.tensor(-0.3, dtype=torch.float32),
        torch.tensor(0.6, dtype=torch.float32),
    ]

    class _ToyTask:
        def __init__(self, theta_eval: torch.Tensor) -> None:
            self._theta_eval = theta_eval

        def sample_theta(self, n: int) -> torch.Tensor:
            assert n == int(self._theta_eval.shape[0])
            return self._theta_eval.clone()

        def log_likelihood(self, y: torch.Tensor, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            return _toy_log_lik(y, x, theta)

    task = _ToyTask(theta_eval)
    spce, snmc = compute_bed_eig_bounds_for_trajectory(
        context_x=context_x,
        context_y=context_y,
        x_traj=x_traj,
        y_traj=y_traj,
        theta_true=theta_true,
        task=task,
        n_contrastive_eval=int(theta_eval.shape[0]),
        device=torch.device("cpu"),
    )

    theta_all = torch.cat([theta_true.unsqueeze(0), theta_eval], dim=0).unsqueeze(1)
    x_full = torch.cat([context_x, torch.stack(x_traj)], dim=0).unsqueeze(0)
    y_full = torch.cat([context_y, torch.stack(y_traj)], dim=0).unsqueeze(0)
    expected_spce = compute_spce_from_trajectory(_toy_log_lik, y_full, x_full, theta_all)
    expected_snmc = compute_snmc_from_trajectory(_toy_log_lik, y_full, x_full, theta_all)

    assert math.isclose(spce, float(expected_spce.item()), rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(snmc, float(expected_snmc.item()), rel_tol=1e-6, abs_tol=1e-6)


def test_evaluate_bed_policy_location_plot(tmp_path):
    torch.manual_seed(2)
    backbone = _DummyBackbone(dim_x=2)
    backbone.model.eval()
    for param in backbone.model.parameters():
        param.requires_grad_(False)

    policy_head = CandidatePolicyHead(input_dim=2, hidden_dim=16, num_layers=1)
    task = LocationFindingSampler(
        dim_x=2,
        K=2,
        n_contrastive=3,
        n_context_init=1,
        n_query=10,
        n_target=0,
        seed=99,
    )
    plot_path = tmp_path / "plots" / "traj.png"
    metrics = evaluate_bed_policy(
        backbone=backbone,
        policy_head=policy_head,
        task=task,
        num_acquisitions=3,
        n_eval_episodes=4,
        n_contrastive_eval=30,
        greedy=True,
        plot_dir=plot_path,
        n_plot_episodes=3,
    )
    assert metrics["eval/n_episodes"] == 4.0
    assert math.isfinite(metrics["eval/snmc_mean"])
    assert plot_path.exists()


def test_evaluate_bed_policy_ces_has_no_trajectory_plot(tmp_path):
    torch.manual_seed(3)
    backbone = _DummyBackbone(dim_x=6)
    backbone.model.eval()
    for param in backbone.model.parameters():
        param.requires_grad_(False)

    policy_head = CandidatePolicyHead(input_dim=6, hidden_dim=16, num_layers=1)
    task = CESSampler(
        n_goods=3,
        n_contrastive=3,
        n_context_init=2,
        n_query=10,
        n_target=0,
        seed=88,
    )
    plot_path = tmp_path / "plots" / "traj_1d.png"
    metrics = evaluate_bed_policy(
        backbone=backbone,
        policy_head=policy_head,
        task=task,
        num_acquisitions=3,
        n_eval_episodes=3,
        n_contrastive_eval=20,
        greedy=True,
        plot_dir=plot_path,
        n_plot_episodes=2,
    )
    assert metrics["eval/n_episodes"] == 3.0
    assert math.isfinite(metrics["eval/snmc_mean"])
    assert not plot_path.exists()
