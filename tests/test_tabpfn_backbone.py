import functools
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from polar.backbones.tabpfn.backbone import TabPFNBackbone
from polar.backbones.tabpfn.loader import LoadedTabPFN
from polar.backbones.tabpfn.model.base.mlp import MLP


class _TinyTabPFNModel(nn.Module):
    def __init__(self, *, recompute: bool) -> None:
        super().__init__()
        self.recompute_layer = False
        self.mlp = MLP(
            size=4,
            hidden_size=8,
            activation="gelu",
            device=None,
            dtype=None,
            recompute=recompute,
        )


class _TinyClassifierTabPFNModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.recompute_layer = False
        self.anchor = nn.Parameter(torch.zeros(()))
        self._logits = logits

    def forward(
        self,
        x,
        y,
        *,
        only_return_standard_out: bool = True,
        categorical_inds=None,
    ):
        del x, y, categorical_inds
        if only_return_standard_out:
            return self._logits
        return {
            "standard": self._logits,
            "test_embeddings": torch.ones(
                self._logits.shape[0],
                self._logits.shape[1],
                4,
            ),
            "train_embeddings": torch.ones(1, self._logits.shape[1], 4),
        }


def _build_backbone(*, recompute: bool) -> TabPFNBackbone:
    model = _TinyTabPFNModel(recompute=recompute)
    components = LoadedTabPFN(
        model=model,
        bar_distribution=nn.Identity(),
        model_config=SimpleNamespace(emsize=4),
        checkpoint_path=Path("dummy.ckpt"),
        architecture_name="base",
        checkpoint_version="test",
        missing_keys=[],
        unexpected_keys=[],
    )
    return TabPFNBackbone(components)


def _build_classifier_backbone() -> TabPFNBackbone:
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0, -2.0]],
            [[-1.0, 1.0, 3.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    model = _TinyClassifierTabPFNModel(logits=logits)
    components = LoadedTabPFN(
        model=model,
        bar_distribution=None,
        model_config=SimpleNamespace(emsize=4, max_num_classes=4),
        checkpoint_path=Path("dummy.ckpt"),
        architecture_name="base",
        checkpoint_version="test",
        missing_keys=[],
        unexpected_keys=[],
        checkpoint_kind="classification",
    )
    return TabPFNBackbone(components, num_classes=2)


def test_enable_and_disable_gradient_checkpointing_round_trip() -> None:
    backbone = _build_backbone(recompute=False)
    original_forward = backbone.model.mlp.forward

    backbone.enable_gradient_checkpointing()

    assert backbone.model.recompute_layer is True
    assert not isinstance(backbone.model.mlp.forward, functools.partial)
    assert backbone.model.mlp.forward.__func__ is original_forward.__func__

    backbone.disable_gradient_checkpointing()

    assert backbone.model.recompute_layer is False
    assert not isinstance(backbone.model.mlp.forward, functools.partial)
    assert backbone.model.mlp.forward.__func__ is original_forward.__func__


def test_enable_gradient_checkpointing_preserves_existing_checkpoint_modules() -> None:
    backbone = _build_backbone(recompute=True)
    original_forward = backbone.model.mlp.forward

    backbone.enable_gradient_checkpointing()

    assert backbone.model.mlp.forward is original_forward

    backbone.disable_gradient_checkpointing()

    assert backbone.model.mlp.forward is original_forward


def test_tabpfn_classifier_log_prob_uses_trimmed_binary_logits() -> None:
    backbone = _build_classifier_backbone()
    logits = torch.tensor([[2.0, 0.0, -4.0, -5.0], [0.0, 1.0, 3.0, 4.0]], dtype=torch.float32)
    targets = torch.tensor([0.0, 1.0])

    actual = backbone.log_prob_from_logits(logits[..., :2], targets)
    expected = torch.log_softmax(logits[..., :2], dim=-1).gather(
        dim=-1,
        index=targets.long().unsqueeze(-1),
    ).squeeze(-1)

    assert torch.allclose(actual, expected)


def test_tabpfn_classifier_forward_with_embeddings_trims_unused_classes() -> None:
    backbone = _build_classifier_backbone()
    context_state = backbone.encode_context(
        torch.rand(1, 2),
        torch.tensor([0.0]),
    )
    outputs = backbone.forward_with_embeddings(torch.rand(2, 2), context_state)

    assert outputs["logits"].shape == torch.Size([2, 2])
