from pathlib import Path

import torch
import torch.nn.functional as F

from polar.backbones.tabicl.backbone import TabICLBackbone
from polar.backbones.tabicl.loader import LoadedTabICL
from polar.backbones.tabicl.model.inference_config import InferenceConfig
from polar.backbones.tabicl.model.tabicl import TabICL


def _build_backbone() -> TabICLBackbone:
    model = TabICL(
        max_classes=0,
        num_quantiles=11,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=4,
        ff_factor=2,
        dropout=0.0,
    )
    model.train()
    components = LoadedTabICL(
        model=model,
        model_config={},
        checkpoint_path=Path("dummy.ckpt"),
        checkpoint_version="test",
        missing_keys=[],
        unexpected_keys=[],
    )
    return TabICLBackbone(components)


def _build_classifier_backbone() -> TabICLBackbone:
    model = TabICL(
        max_classes=4,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=4,
        ff_factor=2,
        dropout=0.0,
    )
    model.eval()
    components = LoadedTabICL(
        model=model,
        model_config={"max_classes": 4},
        checkpoint_path=Path("dummy.ckpt"),
        checkpoint_version="test",
        missing_keys=[],
        unexpected_keys=[],
        checkpoint_kind="classification",
    )
    return TabICLBackbone(components, num_classes=2)


def test_tabicl_forward_with_embeddings_returns_contextualized_hidden_states() -> None:
    torch.manual_seed(0)
    backbone = _build_backbone()

    x_context = torch.rand(2, 3, 1)
    y_context = torch.sin(x_context[..., 0] * torch.pi)
    x_query = torch.rand(2, 4, 1)

    context_state = backbone.encode_context(x_context, y_context)
    outputs = backbone.forward_with_embeddings(x_query, context_state)

    train_size = x_context.shape[1]
    x_full = torch.cat([context_state.x_context, x_query], dim=1)
    row_representations = backbone.model.row_interactor(
        backbone.model.col_embedder(
            x_full,
            y_train=context_state.y_context,
            d=None,
            embed_with_test=False,
        ),
        d=None,
    )

    contextual_representations = row_representations.clone()
    y_train_embeddings = backbone.model.icl_predictor.y_encoder(
        context_state.y_context.unsqueeze(-1)
    )
    contextual_representations[:, :train_size, :] = (
        contextual_representations[:, :train_size, :] + y_train_embeddings
    )
    contextual_representations = backbone.model.icl_predictor.tf_icl(
        contextual_representations,
        train_size=train_size,
    )
    if backbone.model.icl_predictor.norm_first:
        contextual_representations = backbone.model.icl_predictor.ln(
            contextual_representations
        )

    expected_logits = backbone.model.icl_predictor.decoder(
        contextual_representations
    )[:, train_size:, :]

    assert torch.allclose(outputs["logits"], expected_logits)
    assert torch.allclose(
        outputs["test_embeddings"],
        contextual_representations[:, train_size:, :],
    )
    assert torch.allclose(
        outputs["train_embeddings"],
        contextual_representations[:, :train_size, :],
    )


def test_tabicl_prediction_loss_uses_pinball_objective() -> None:
    torch.manual_seed(1)
    backbone = _build_backbone()

    x_context = torch.rand(2, 3, 1)
    y_context = torch.sin(x_context[..., 0] * torch.pi)
    x_query = torch.rand(2, 4, 1)
    y_query = torch.cos(x_query[..., 0] * torch.pi)

    context_state = backbone.encode_context(x_context, y_context)
    outputs = backbone.forward_with_embeddings(x_query, context_state)

    expected = backbone.distribution_from_logits(outputs["logits"]).pinball(y_query)
    actual = backbone.prediction_loss_from_logits(outputs["logits"], y_query)
    assert torch.allclose(actual, expected)


def test_tabicl_classifier_prediction_loss_uses_categorical_nll() -> None:
    backbone = _build_classifier_backbone()
    logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 1.0]], [[-1.0, 3.0], [1.5, -0.5]]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    actual = backbone.prediction_loss_from_logits(logits, targets)
    expected = -F.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=targets.long().unsqueeze(-1),
    ).squeeze(-1)

    assert torch.allclose(actual, expected)


def test_tabicl_classifier_forward_with_embeddings_trims_unused_classes() -> None:
    torch.manual_seed(0)
    backbone = _build_classifier_backbone()

    x_context = torch.rand(1, 3, 1)
    y_context = torch.tensor([[0.0, 1.0, 0.0]])
    x_query = torch.rand(1, 4, 1)

    context_state = backbone.encode_context(x_context, y_context)
    outputs = backbone.forward_with_embeddings(x_query, context_state)

    assert outputs["logits"].shape == torch.Size([1, 4, 2])


def test_tabicl_inference_defaults_disable_amp() -> None:
    config = InferenceConfig()

    assert config.COL_CONFIG.use_amp is False
    assert config.ROW_CONFIG.use_amp is False
    assert config.ICL_CONFIG.use_amp is False
