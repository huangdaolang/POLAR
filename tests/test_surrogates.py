from __future__ import annotations

import torch
from torch import nn

from polar.utils.surrogates import (
    GPyTorchSurrogate,
    RandomForestSurrogate,
    TabICLSurrogate,
    TabPFNSurrogate,
    TanimotoKernel,
)


class _DummyTabularDistribution:
    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits

    def sample(self, shape) -> torch.Tensor:
        if isinstance(shape, tuple):
            return self._logits.expand(*shape, *self._logits.shape)
        return self._logits


class _DummyBackbone:
    def __init__(self) -> None:
        self.model = nn.Linear(1, 1)
        self._context_state = None
        self.last_predict_grad_enabled: bool | None = None
        self.last_log_prob_grad_enabled: bool | None = None
        self.predict_calls = 0
        self.log_prob_calls = 0

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def encode_context(self, x_context: torch.Tensor, y_context: torch.Tensor):
        self._context_state = (x_context, y_context)
        return self._context_state

    def predictive_distribution(self, x: torch.Tensor, context_state) -> dict[str, torch.Tensor]:
        del context_state
        self.predict_calls += 1
        self.last_predict_grad_enabled = torch.is_grad_enabled()
        logits = x.clone().requires_grad_(torch.is_grad_enabled())
        return {"logits": logits, "mean": logits.mean(dim=-1)}

    def variance_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.square().mean(dim=-1) + 1.0

    def log_prob(self, x: torch.Tensor, y: torch.Tensor, context_state) -> torch.Tensor:
        del context_state
        self.log_prob_calls += 1
        self.last_log_prob_grad_enabled = torch.is_grad_enabled()
        return -(x.sum(dim=-1) - y.view(-1)).square()

    def distribution_from_logits(self, logits: torch.Tensor):
        return _DummyTabularDistribution(logits.mean(dim=-1))


def test_tanimoto_kernel_matches_expected_binary_similarity() -> None:
    kernel = TanimotoKernel()
    x = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )

    cov = kernel.forward(x, x)
    expected = torch.tensor(
        [
            [1.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0],
        ]
    )

    assert torch.allclose(cov, expected, atol=1e-6)
    assert torch.allclose(kernel.forward(x, x, diag=True), torch.ones(2), atol=1e-6)


def test_gpytorch_surrogate_supports_tanimoto_kernel() -> None:
    torch.manual_seed(0)

    surrogate = GPyTorchSurrogate(
        kernel="tanimoto",
        training_steps=5,
        learning_rate=0.1,
        initial_noise=1.0e-3,
        device="cpu",
    )

    context_x = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    context_y = torch.tensor([0.9, 0.4, 0.7])
    query_x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )

    surrogate.fit(context_x, context_y)
    prediction = surrogate.predict(query_x)

    assert prediction.mean.shape == torch.Size([2])
    assert prediction.variance.shape == torch.Size([2])
    assert torch.isfinite(prediction.mean).all()
    assert torch.isfinite(prediction.variance).all()
    assert torch.all(prediction.variance > 0.0)

    draws = surrogate.sample(query_x, num_samples=3)
    assert draws.shape == torch.Size([3, 2])
    assert torch.isfinite(draws).all()


def test_random_forest_surrogate_predicts_finite_mean_variance_and_samples() -> None:
    surrogate = RandomForestSurrogate(
        n_estimators=32,
        max_depth=4,
        min_samples_leaf=1,
        max_features="sqrt",
        variance_floor=1.0e-6,
        n_jobs=1,
        random_state=0,
    )

    context_x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    context_y = torch.tensor([0.0, 0.8, 1.4, 2.1, 0.9], dtype=torch.float32)
    query_x = torch.tensor(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.25, 0.75, 0.25],
        ],
        dtype=torch.float32,
    )
    query_y = torch.tensor([0.2, 1.8, 0.7], dtype=torch.float32)

    surrogate.fit(context_x, context_y)
    prediction = surrogate.predict(query_x)

    assert prediction.mean.shape == torch.Size([3])
    assert prediction.variance.shape == torch.Size([3])
    assert torch.isfinite(prediction.mean).all()
    assert torch.isfinite(prediction.variance).all()
    assert torch.all(prediction.variance > 0.0)

    log_prob = surrogate.point_log_prob(query_x, query_y)
    assert log_prob.shape == torch.Size([3])
    assert torch.isfinite(log_prob).all()

    draws = surrogate.sample(query_x, num_samples=4)
    assert draws.shape == torch.Size([4, 3])
    assert torch.isfinite(draws).all()

    surrogate_repeated = RandomForestSurrogate(
        n_estimators=32,
        max_depth=4,
        min_samples_leaf=1,
        max_features="sqrt",
        variance_floor=1.0e-6,
        n_jobs=1,
        random_state=0,
    )
    surrogate_repeated.fit(context_x, context_y)
    repeated_draws = surrogate_repeated.sample(query_x, num_samples=4)
    assert torch.allclose(draws, repeated_draws)


def test_tabpfn_surrogate_uses_inference_mode() -> None:
    backbone = _DummyBackbone()
    surrogate = TabPFNSurrogate(backbone=backbone)  # type: ignore[arg-type]

    context_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    context_y = torch.tensor([0.5, 0.7])
    query_x = torch.tensor([[1.0, 1.0], [0.5, 0.5]])

    surrogate.fit(context_x, context_y)

    with torch.enable_grad():
        prediction = surrogate.predict(query_x)
        log_prob = surrogate.point_log_prob(query_x, torch.tensor([0.2, 0.4]))
        draws = surrogate.sample(query_x, num_samples=2)

    assert backbone.last_predict_grad_enabled is False
    assert backbone.last_log_prob_grad_enabled is False
    assert prediction.mean.requires_grad is False
    assert prediction.variance.requires_grad is False
    assert log_prob.requires_grad is False
    assert draws.requires_grad is False


def test_tabicl_surrogate_uses_inference_mode() -> None:
    backbone = _DummyBackbone()
    surrogate = TabICLSurrogate(backbone=backbone)  # type: ignore[arg-type]

    context_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    context_y = torch.tensor([0.5, 0.7])
    query_x = torch.tensor([[1.0, 1.0], [0.5, 0.5]])

    surrogate.fit(context_x, context_y)

    with torch.enable_grad():
        prediction = surrogate.predict(query_x)
        log_prob = surrogate.point_log_prob(query_x, torch.tensor([0.2, 0.4]))
        draws = surrogate.sample(query_x, num_samples=2)

    assert backbone.last_predict_grad_enabled is False
    assert backbone.last_log_prob_grad_enabled is False
    assert prediction.mean.requires_grad is False
    assert prediction.variance.requires_grad is False
    assert log_prob.requires_grad is False
    assert draws.requires_grad is False


def test_tabpfn_surrogate_chunking_matches_full_and_zero_disables_chunking() -> None:
    backbone = _DummyBackbone()
    surrogate_full = TabPFNSurrogate(backbone=backbone, max_query_chunk=0)  # type: ignore[arg-type]

    context_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    context_y = torch.tensor([0.5, 0.7])
    query_x = torch.tensor(
        [
            [1.0, 1.0],
            [0.5, 0.5],
            [0.25, 0.75],
            [0.9, 0.1],
        ]
    )
    query_y = torch.tensor([0.2, 0.4, 0.1, 0.8])

    surrogate_full.fit(context_x, context_y)
    pred_full = surrogate_full.predict(query_x)
    log_prob_full = surrogate_full.point_log_prob(query_x, query_y)
    draws_full = surrogate_full.sample(query_x, num_samples=3)

    assert backbone.predict_calls == 2
    assert backbone.log_prob_calls == 1

    backbone.predict_calls = 0
    backbone.log_prob_calls = 0

    surrogate_chunked = TabPFNSurrogate(backbone=backbone, max_query_chunk=2)  # type: ignore[arg-type]
    surrogate_chunked.fit(context_x, context_y)
    pred_chunked = surrogate_chunked.predict(query_x)
    log_prob_chunked = surrogate_chunked.point_log_prob(query_x, query_y)
    draws_chunked = surrogate_chunked.sample(query_x, num_samples=3)

    assert backbone.predict_calls == 4
    assert backbone.log_prob_calls == 2
    assert torch.allclose(pred_full.mean, pred_chunked.mean)
    assert torch.allclose(pred_full.variance, pred_chunked.variance)
    assert torch.allclose(log_prob_full, log_prob_chunked)
    assert torch.allclose(draws_full, draws_chunked)


def test_tabicl_surrogate_chunking_matches_full_and_zero_disables_chunking() -> None:
    backbone = _DummyBackbone()
    surrogate_full = TabICLSurrogate(backbone=backbone, max_query_chunk=0)  # type: ignore[arg-type]

    context_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    context_y = torch.tensor([0.5, 0.7])
    query_x = torch.tensor(
        [
            [1.0, 1.0],
            [0.5, 0.5],
            [0.25, 0.75],
            [0.9, 0.1],
        ]
    )
    query_y = torch.tensor([0.2, 0.4, 0.1, 0.8])

    surrogate_full.fit(context_x, context_y)
    pred_full = surrogate_full.predict(query_x)
    log_prob_full = surrogate_full.point_log_prob(query_x, query_y)
    draws_full = surrogate_full.sample(query_x, num_samples=3)

    assert backbone.predict_calls == 2
    assert backbone.log_prob_calls == 1

    backbone.predict_calls = 0
    backbone.log_prob_calls = 0

    surrogate_chunked = TabICLSurrogate(backbone=backbone, max_query_chunk=2)  # type: ignore[arg-type]
    surrogate_chunked.fit(context_x, context_y)
    pred_chunked = surrogate_chunked.predict(query_x)
    log_prob_chunked = surrogate_chunked.point_log_prob(query_x, query_y)
    draws_chunked = surrogate_chunked.sample(query_x, num_samples=3)

    assert backbone.predict_calls == 4
    assert backbone.log_prob_calls == 2
    assert torch.allclose(pred_full.mean, pred_chunked.mean)
    assert torch.allclose(pred_full.variance, pred_chunked.variance)
    assert torch.allclose(log_prob_full, log_prob_chunked)
    assert torch.allclose(draws_full, draws_chunked)
