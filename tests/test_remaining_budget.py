from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from polar.trainers import (
    CandidatePolicyHead,
    TargetAwareTransformerPolicyHead,
    build_policy_head,
    score_policy_head,
)


def test_zero_budget_weights_preserve_same_seed_scores_and_rng() -> None:
    embeddings = torch.tensor([[0.2, -0.4, 1.0], [1.0, 0.0, -0.5]])
    torch.manual_seed(41)
    original = CandidatePolicyHead(input_dim=3, hidden_dim=8, num_layers=2)
    original_rng = torch.get_rng_state()
    torch.manual_seed(41)
    conditioned = CandidatePolicyHead(
        input_dim=3, hidden_dim=8, num_layers=2,
        use_remaining_budget=True, budget_scale=30,
    )

    assert torch.equal(torch.get_rng_state(), original_rng)
    assert conditioned.network[0].in_features == 4
    assert sum(p.numel() for p in conditioned.parameters()) == (
        sum(p.numel() for p in original.parameters()) + 8
    )
    for remaining in [1, 30, 60]:
        torch.testing.assert_close(
            original(embeddings), conditioned(embeddings, remaining_budget=remaining)
        )


def test_legacy_state_warm_start_preserves_scores_and_does_not_mutate_source() -> None:
    original = CandidatePolicyHead(input_dim=3, hidden_dim=8, num_layers=2)
    legacy_state = original.state_dict()
    conditioned = CandidatePolicyHead(
        input_dim=3, hidden_dim=8, num_layers=2,
        use_remaining_budget=True, budget_scale=30,
    )
    conditioned.load_state_dict(legacy_state, strict=True)
    embeddings = torch.randn(7, 3)

    assert legacy_state["network.0.weight"].shape == (8, 3)
    assert torch.count_nonzero(conditioned.network[0].weight[:, -1]) == 0
    for remaining in [1, 30]:
        torch.testing.assert_close(
            original(embeddings), conditioned(embeddings, remaining_budget=remaining)
        )

    incompatible = CandidatePolicyHead(input_dim=5, hidden_dim=8, num_layers=2)
    with pytest.raises(RuntimeError, match="size mismatch"):
        conditioned.load_state_dict(incompatible.state_dict(), strict=True)


def test_budget_can_change_candidate_ranking_and_receive_gradients() -> None:
    head = CandidatePolicyHead(
        input_dim=1, hidden_dim=2, num_layers=1, activation="relu",
        use_remaining_budget=True, budget_scale=10,
    )
    # score(x, h) = relu(x + h/10 - 1) - 0.5 * relu(x).
    with torch.no_grad():
        head.network[0].weight.copy_(torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
        head.network[0].bias.copy_(torch.tensor([-1.0, 0.0]))
        head.network[2].weight.copy_(torch.tensor([[1.0, -0.5]]))
        head.network[2].bias.zero_()
    embeddings = torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]])
    original_embeddings = embeddings.clone()
    last_step = score_policy_head(head, query_embeddings=embeddings, remaining_budget=1)
    first_step = score_policy_head(head, query_embeddings=embeddings, remaining_budget=10)

    assert last_step.shape == first_step.shape == (2, 2)
    assert last_step.argmax(-1).tolist() == [0, 0]
    assert first_step.argmax(-1).tolist() == [1, 1]
    torch.testing.assert_close(embeddings, original_embeddings)
    first_step.sum().backward()
    assert torch.count_nonzero(head.network[0].weight.grad[:, -1]) > 0


@pytest.mark.parametrize("remaining", [None, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("head_class", [CandidatePolicyHead, TargetAwareTransformerPolicyHead])
def test_conditioned_head_rejects_missing_or_invalid_budget(remaining, head_class) -> None:
    head = head_class(input_dim=4, use_remaining_budget=True)
    with pytest.raises(ValueError, match="remaining_budget"):
        score_policy_head(
            head, query_embeddings=torch.zeros(3, 4), target_embeddings=torch.zeros(2, 4),
            remaining_budget=remaining,
        )


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("head_class", [CandidatePolicyHead, TargetAwareTransformerPolicyHead])
def test_budget_scale_must_be_finite_and_positive(scale: float, head_class) -> None:
    with pytest.raises(ValueError, match="budget_scale"):
        head_class(input_dim=4, use_remaining_budget=True, budget_scale=scale)


def test_disabled_target_aware_head_ignores_budget() -> None:
    cfg = SimpleNamespace(
        type="target_aware_transformer", num_heads=1, ffn_dim=8,
        use_remaining_budget=False, budget_scale=10,
    )
    head = build_policy_head(input_dim=4, policy_head_cfg=cfg).eval()
    queries, targets = torch.randn(2, 3, 4), torch.randn(2, 5, 4)
    before = score_policy_head(
        head, query_embeddings=queries, target_embeddings=targets, remaining_budget=10
    )
    after = score_policy_head(
        head, query_embeddings=queries, target_embeddings=targets, remaining_budget=1
    )
    torch.testing.assert_close(before, after)
    assert not head.use_remaining_budget
    assert head.layers[0]["ffn"][0].in_features == 4


@pytest.mark.parametrize("task", ["al", "bed", "bo"])
def test_default_policy_config_uses_expected_budget_setting(task: str) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.create({"train": OmegaConf.load(root / f"config/train/{task}_policy.yaml")})
    cfg.train.num_acquisitions = 17
    head = build_policy_head(input_dim=4, policy_head_cfg=cfg.train.policy_head)
    assert head.use_remaining_budget == (task != "al")
    assert head.budget_scale == 17
    assert cfg.train.loss.gamma == 0
    if task == "al":
        assert isinstance(head, TargetAwareTransformerPolicyHead)
        assert head.layers[0]["ffn"][0].in_features == 4
    else:
        assert isinstance(head, CandidatePolicyHead)
        assert head.network[0].in_features == 5


@pytest.mark.parametrize("head_type", ["mlp", "target_aware_transformer"])
def test_al_config_allows_opt_in_budget_for_either_head(head_type: str) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.create({"train": OmegaConf.load(root / "config/train/al_policy.yaml")})
    cfg.train.policy_head.type = head_type
    cfg.train.num_acquisitions = 17
    default_head = build_policy_head(input_dim=4, policy_head_cfg=cfg.train.policy_head)
    assert not default_head.use_remaining_budget

    cfg.train.policy_head.use_remaining_budget = True
    head = build_policy_head(input_dim=4, policy_head_cfg=cfg.train.policy_head)
    assert head.use_remaining_budget
    assert head.budget_scale == 17


@pytest.mark.parametrize("num_layers", [1, 2])
def test_target_aware_zero_budget_weights_preserve_scores_and_rng(num_layers: int) -> None:
    queries, targets = torch.randn(2, 3, 4), torch.randn(2, 5, 4)
    torch.manual_seed(41)
    original = TargetAwareTransformerPolicyHead(
        input_dim=4, num_heads=1, ffn_dim=8, num_layers=num_layers,
    )
    original_rng = torch.get_rng_state()
    torch.manual_seed(41)
    conditioned = TargetAwareTransformerPolicyHead(
        input_dim=4, num_heads=1, ffn_dim=8, num_layers=num_layers,
        use_remaining_budget=True, budget_scale=20,
    )
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert sum(p.numel() for p in conditioned.parameters()) == (
        sum(p.numel() for p in original.parameters()) + 8 * num_layers
    )
    for remaining in [1, 20, 40]:
        torch.testing.assert_close(
            original(queries, targets),
            conditioned(queries, targets, remaining_budget=remaining),
        )


def test_target_aware_legacy_warm_start_preserves_scores_and_strict_loading() -> None:
    original = TargetAwareTransformerPolicyHead(
        input_dim=4, num_heads=1, ffn_dim=8, num_layers=2,
    )
    legacy_state = original.state_dict()
    conditioned = TargetAwareTransformerPolicyHead(
        input_dim=4, num_heads=1, ffn_dim=8, num_layers=2,
        use_remaining_budget=True, budget_scale=20,
    )
    conditioned.load_state_dict(legacy_state, strict=True)
    for index, layer in enumerate(conditioned.layers):
        assert legacy_state[f"layers.{index}.ffn.0.weight"].shape == (8, 4)
        assert torch.count_nonzero(layer["ffn"][0].weight[:, -1]) == 0
    queries, targets = torch.randn(3, 4), torch.randn(5, 4)
    for remaining in [1, 20]:
        torch.testing.assert_close(
            original(queries, targets),
            conditioned(queries, targets, remaining_budget=remaining),
        )

    incompatible = TargetAwareTransformerPolicyHead(
        input_dim=4, num_heads=1, ffn_dim=9, num_layers=2,
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        conditioned.load_state_dict(incompatible.state_dict(), strict=True)
    incomplete = dict(legacy_state)
    del incomplete["scorer.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        conditioned.load_state_dict(incomplete, strict=True)


@pytest.mark.parametrize("batched", [False, True])
def test_target_aware_budget_changes_ranking_and_receives_gradients(batched: bool) -> None:
    head = TargetAwareTransformerPolicyHead(
        input_dim=4, num_heads=1, ffn_dim=2, num_layers=1, activation="relu",
        use_remaining_budget=True, budget_scale=10,
    )
    layer = head.layers[0]
    # Keep both LayerNorms. The FFN changes the first coordinate by
    # -3 * relu(hidden[0] + remaining/10 - 1), which can reverse the ranking.
    with torch.no_grad():
        for param in layer["attn"].parameters():
            param.zero_()
        for param in layer["ffn"].parameters():
            param.zero_()
        layer["ffn"][0].weight[0, 0] = 1.0
        layer["ffn"][0].weight[0, -1] = 1.0
        layer["ffn"][0].bias[0] = -1.0
        layer["ffn"][3].weight[0, 0] = -3.0
        head.scorer.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        head.scorer.bias.zero_()
    queries = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
    targets = torch.zeros(3, 4)
    if batched:
        queries = queries.unsqueeze(0).repeat(2, 1, 1)
        targets = targets.unsqueeze(0).repeat(2, 1, 1)
    original_queries, original_targets = queries.clone(), targets.clone()
    last_step = score_policy_head(
        head, query_embeddings=queries, target_embeddings=targets, remaining_budget=1,
    )
    first_step = score_policy_head(
        head, query_embeddings=queries, target_embeddings=targets, remaining_budget=10,
    )
    assert (last_step.argmax(-1) == 0).all()
    assert (first_step.argmax(-1) == 1).all()
    assert not torch.allclose(last_step.softmax(-1), first_step.softmax(-1))
    first_step.log_softmax(-1)[..., 1].sum().backward()
    assert torch.count_nonzero(layer["ffn"][0].weight.grad[:, -1]) > 0
    torch.testing.assert_close(queries, original_queries)
    torch.testing.assert_close(targets, original_targets)
