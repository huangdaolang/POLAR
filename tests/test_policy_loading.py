from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from polar.trainers import CandidatePolicyHead, TargetAwareTransformerPolicyHead
from polar.utils import policy_loading
from polar.utils.policy_loading import load_checkpoint_payload


def test_load_checkpoint_payload_prefers_existing_input_path_over_resolved_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "policy.pt"
    torch.save({"policy_head_state_dict": {"weight": torch.tensor([1.0])}}, checkpoint_path)

    def _unexpected_resolve(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("resolve() should not be called when the provided path already exists.")

    monkeypatch.setattr(Path, "resolve", _unexpected_resolve)

    payload = load_checkpoint_payload(checkpoint_path)

    assert "policy_head_state_dict" in payload


@pytest.mark.parametrize("saved_budget,override", [(False, None), (True, None), (False, True)])
def test_policy_loader_restores_budget_config_and_supports_legacy_warm_start(
    tmp_path: Path, monkeypatch, saved_budget: bool, override: bool | None,
) -> None:
    original = CandidatePolicyHead(
        input_dim=2, hidden_dim=4, num_layers=1,
        use_remaining_budget=saved_budget, budget_scale=30,
    )
    if saved_budget:
        with torch.no_grad():
            original.network[0].weight[:, -1].fill_(0.5)
    arch = {"hidden_dim": 4, "num_layers": 1}
    if saved_budget:
        arch.update(use_remaining_budget=True, budget_scale=30)
    checkpoint = tmp_path / "policy.pt"
    torch.save({
        "config": {
            "model": {"name": "tabicl"},
            "train": {"num_acquisitions": 30, "policy_head": arch},
        },
        "policy_head_state_dict": original.state_dict(),
    }, checkpoint)
    backbone = SimpleNamespace(
        embedding_dim=2, device=torch.device("cpu"),
        model=torch.nn.Linear(2, 1), checkpoint_path=None,
    )
    monkeypatch.setattr(policy_loading, "create_backbone", lambda **kwargs: backbone)
    loaded = policy_loading.load_policy_checkpoint(
        backend="tabicl", model_cfg=OmegaConf.create({"name": "tabicl"}),
        policy_cfg=OmegaConf.create({"use_remaining_budget": override, "budget_scale": None}),
        device="cpu", checkpoints_root=tmp_path, policy_checkpoint_path=checkpoint,
    )

    assert loaded.policy_head.use_remaining_budget == (saved_budget or bool(override))
    assert loaded.policy_head.budget_scale == 30
    embeddings = torch.randn(3, 2)
    for remaining in [1, 6]:
        torch.testing.assert_close(
            original(embeddings, remaining_budget=remaining),
            loaded.policy_head(embeddings, remaining_budget=remaining),
        )


@pytest.mark.parametrize(
    ("saved_budget", "recorded_flag", "override"),
    [
        (False, None, None),
        (False, False, None),
        (False, True, None),  # Early checkpoints recorded a flag that their head ignored.
        (False, None, True),
        (False, True, True),
        (True, True, None),
        (True, None, None),
        (True, True, False),
    ],
)
def test_target_aware_loader_uses_saved_ffn_width_and_respects_overrides(
    tmp_path: Path, monkeypatch, saved_budget: bool,
    recorded_flag: bool | None, override: bool | None,
) -> None:
    original = TargetAwareTransformerPolicyHead(
        input_dim=4, num_heads=1, ffn_dim=8, num_layers=2,
        use_remaining_budget=saved_budget, budget_scale=40 if saved_budget else 30,
    )
    if saved_budget:
        with torch.no_grad():
            for layer in original.layers:
                layer["ffn"][0].weight[:, -1].copy_(torch.linspace(-0.5, 0.5, 8))
    arch = {"type": "target_aware_transformer", "num_heads": 1, "ffn_dim": 8, "num_layers": 2}
    if recorded_flag is not None:
        arch["use_remaining_budget"] = recorded_flag
    if saved_budget:
        arch["budget_scale"] = 40
    checkpoint = tmp_path / "target_aware.pt"
    torch.save({
        "config": {
            "model": {"name": "tabicl"},
            "train": {"num_acquisitions": 30, "policy_head": arch},
        },
        "policy_head_state_dict": original.state_dict(),
    }, checkpoint)
    backbone = SimpleNamespace(
        embedding_dim=4, device=torch.device("cpu"),
        model=torch.nn.Linear(4, 1), checkpoint_path=None,
    )
    monkeypatch.setattr(policy_loading, "create_backbone", lambda **kwargs: backbone)
    kwargs = dict(
        backend="tabicl", model_cfg=OmegaConf.create({"name": "tabicl"}),
        policy_cfg=OmegaConf.create({"use_remaining_budget": override, "budget_scale": None}),
        device="cpu", checkpoints_root=tmp_path, policy_checkpoint_path=checkpoint,
    )
    if saved_budget and override is False:
        with pytest.raises(RuntimeError, match="size mismatch"):
            policy_loading.load_policy_checkpoint(**kwargs)
        return
    loaded = policy_loading.load_policy_checkpoint(**kwargs)
    expected_budget = saved_budget if override is None else override
    assert loaded.policy_head.use_remaining_budget == expected_budget
    assert loaded.policy_head.budget_scale == (40 if saved_budget else 30)
    for layer in loaded.policy_head.layers:
        assert layer["ffn"][0].in_features == 4 + int(expected_budget)
    queries, targets = torch.randn(3, 4), torch.randn(5, 4)
    for remaining in [1, 6, 60]:
        torch.testing.assert_close(
            original(queries, targets, remaining_budget=remaining),
            loaded.policy_head(queries, targets, remaining_budget=remaining),
        )
