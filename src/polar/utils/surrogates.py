"""Surrogate adapters shared across AL/BO runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import gpytorch
from sklearn.ensemble import RandomForestRegressor

from polar.backbones.tabicl.backbone import TabICLBackbone
from polar.backbones.tabpfn.backbone import TabPFNBackbone


@dataclass(frozen=True)
class SurrogatePrediction:
    """Predictive mean/variance for a set of query points."""

    mean: torch.Tensor
    variance: torch.Tensor


@dataclass(frozen=True)
class GPJointPrediction:
    """Joint GP predictive statistics for query/target acquisition computations."""

    query_mean: torch.Tensor
    query_variance: torch.Tensor
    target_mean: torch.Tensor
    target_variance: torch.Tensor
    target_query_covariance: torch.Tensor


class Surrogate(ABC):
    """Common interface for surrogate models."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Stable backend name."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Compute device used by the surrogate."""

    @abstractmethod
    def fit(self, context_x: torch.Tensor, context_y: torch.Tensor) -> None:
        """Update surrogate state with current context."""

    @abstractmethod
    def predict(self, x: torch.Tensor) -> SurrogatePrediction:
        """Predict mean and variance for x."""

    @abstractmethod
    def point_log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-point log probability under an independent approximation."""

    @abstractmethod
    def sample(self, x: torch.Tensor, *, num_samples: int) -> torch.Tensor:
        """Draw predictive samples with shape [num_samples, num_points]."""


class TabPFNSurrogate(Surrogate):
    """TabPFN surrogate adapter with chunked inference for large query pools.

    When the query set is larger than ``max_query_chunk``, predictions are
    split into independent chunks that each share the same context.  This
    keeps per-item attention memory (which scales as seq_len² × feature_groups)
    under control for high-dimensional inputs.
    """

    DEFAULT_MAX_QUERY_CHUNK: int = 500

    def __init__(
        self,
        backbone: TabPFNBackbone,
        *,
        max_query_chunk: int | None = None,
    ) -> None:
        self.backbone = backbone
        self._context_state = None
        chunk = self.DEFAULT_MAX_QUERY_CHUNK if max_query_chunk is None else int(max_query_chunk)
        self.max_query_chunk = chunk if chunk > 0 else 0

    @property
    def backend_name(self) -> str:
        return "tabpfn"

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    def fit(self, context_x: torch.Tensor, context_y: torch.Tensor) -> None:
        self._context_state = self.backbone.encode_context(context_x, context_y)

    # -- predict -----------------------------------------------------------

    def predict(self, x: torch.Tensor) -> SurrogatePrediction:
        if self._context_state is None:
            raise RuntimeError("TabPFNSurrogate.fit must be called before predict.")

        n_query = int(x.shape[0])
        chunk = self.max_query_chunk

        if chunk <= 0 or n_query <= chunk:
            return self._predict_single(x)

        means: list[torch.Tensor] = []
        variances: list[torch.Tensor] = []
        for start in range(0, n_query, chunk):
            pred = self._predict_single(x[start : start + chunk])
            means.append(pred.mean)
            variances.append(pred.variance)
        return SurrogatePrediction(
            mean=torch.cat(means, dim=0),
            variance=torch.cat(variances, dim=0),
        )

    def _predict_single(self, x: torch.Tensor) -> SurrogatePrediction:
        assert self._context_state is not None
        with torch.inference_mode():
            outputs = self.backbone.predictive_distribution(x, self._context_state)
            logits = outputs["logits"]
            mean = outputs["mean"]
            variance = self.backbone.variance_from_logits(logits)
        return SurrogatePrediction(mean=mean, variance=variance)

    # -- point_log_prob ----------------------------------------------------

    def point_log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._context_state is None:
            raise RuntimeError("TabPFNSurrogate.fit must be called before point_log_prob.")

        n_query = int(x.shape[0])
        chunk = self.max_query_chunk

        if chunk <= 0 or n_query <= chunk:
            with torch.inference_mode():
                return self.backbone.log_prob(x, y, self._context_state)

        parts: list[torch.Tensor] = []
        for start in range(0, n_query, chunk):
            with torch.inference_mode():
                lp = self.backbone.log_prob(
                    x[start : start + chunk],
                    y[start : start + chunk],
                    self._context_state,
                )
            parts.append(lp)
        return torch.cat(parts, dim=0)

    # -- sample ------------------------------------------------------------

    def sample(self, x: torch.Tensor, *, num_samples: int) -> torch.Tensor:
        if self._context_state is None:
            raise RuntimeError("TabPFNSurrogate.fit must be called before sample.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        n_query = int(x.shape[0])
        chunk = self.max_query_chunk

        if chunk <= 0 or n_query <= chunk:
            return self._sample_single(x, num_samples=num_samples)

        all_draws: list[torch.Tensor] = []
        for start in range(0, n_query, chunk):
            draws_chunk = self._sample_single(x[start : start + chunk], num_samples=num_samples)
            all_draws.append(draws_chunk)
        return torch.cat(all_draws, dim=1)  # [num_samples, total_query]

    def _sample_single(self, x: torch.Tensor, *, num_samples: int) -> torch.Tensor:
        assert self._context_state is not None
        with torch.inference_mode():
            outputs = self.backbone.predictive_distribution(x, self._context_state)
            logits = outputs["logits"]
            dist = self.backbone.distribution_from_logits(logits)
            draws = [dist.sample(logits).to(self.device).float() for _ in range(num_samples)]
        return torch.stack(draws, dim=0)


class TabICLSurrogate(Surrogate):
    """TabICL surrogate adapter with chunked inference."""

    DEFAULT_MAX_QUERY_CHUNK: int = 500

    def __init__(
        self,
        backbone: TabICLBackbone,
        *,
        max_query_chunk: int | None = None,
    ) -> None:
        self.backbone = backbone
        self._context_state = None
        chunk = self.DEFAULT_MAX_QUERY_CHUNK if max_query_chunk is None else int(max_query_chunk)
        self.max_query_chunk = chunk if chunk > 0 else 0

    @property
    def backend_name(self) -> str:
        return "tabicl"

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    def fit(self, context_x: torch.Tensor, context_y: torch.Tensor) -> None:
        self._context_state = self.backbone.encode_context(context_x, context_y)

    def predict(self, x: torch.Tensor) -> SurrogatePrediction:
        if self._context_state is None:
            raise RuntimeError("TabICLSurrogate.fit must be called before predict.")

        n_query = int(x.shape[0])
        chunk = self.max_query_chunk

        if chunk <= 0 or n_query <= chunk:
            return self._predict_single(x)

        means: list[torch.Tensor] = []
        variances: list[torch.Tensor] = []
        for start in range(0, n_query, chunk):
            pred = self._predict_single(x[start : start + chunk])
            means.append(pred.mean)
            variances.append(pred.variance)
        return SurrogatePrediction(
            mean=torch.cat(means, dim=0),
            variance=torch.cat(variances, dim=0),
        )

    def _predict_single(self, x: torch.Tensor) -> SurrogatePrediction:
        assert self._context_state is not None
        with torch.inference_mode():
            outputs = self.backbone.predictive_distribution(x, self._context_state)
            logits = outputs["logits"]
            mean = outputs["mean"]
            variance = self.backbone.variance_from_logits(logits)
        return SurrogatePrediction(mean=mean, variance=variance)

    def point_log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._context_state is None:
            raise RuntimeError("TabICLSurrogate.fit must be called before point_log_prob.")

        n_query = int(x.shape[0])
        chunk = self.max_query_chunk

        if chunk <= 0 or n_query <= chunk:
            with torch.inference_mode():
                return self.backbone.log_prob(x, y, self._context_state)

        parts: list[torch.Tensor] = []
        for start in range(0, n_query, chunk):
            with torch.inference_mode():
                lp = self.backbone.log_prob(
                    x[start : start + chunk],
                    y[start : start + chunk],
                    self._context_state,
                )
            parts.append(lp)
        return torch.cat(parts, dim=0)

    def sample(self, x: torch.Tensor, *, num_samples: int) -> torch.Tensor:
        if self._context_state is None:
            raise RuntimeError("TabICLSurrogate.fit must be called before sample.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        n_query = int(x.shape[0])
        chunk = self.max_query_chunk

        if chunk <= 0 or n_query <= chunk:
            return self._sample_single(x, num_samples=num_samples)

        all_draws: list[torch.Tensor] = []
        for start in range(0, n_query, chunk):
            draws_chunk = self._sample_single(x[start : start + chunk], num_samples=num_samples)
            all_draws.append(draws_chunk)
        return torch.cat(all_draws, dim=1)

    def _sample_single(self, x: torch.Tensor, *, num_samples: int) -> torch.Tensor:
        assert self._context_state is not None
        with torch.inference_mode():
            outputs = self.backbone.predictive_distribution(x, self._context_state)
            logits = outputs["logits"]
            dist = self.backbone.distribution_from_logits(logits)
            draws = dist.sample((num_samples,))
        if draws.ndim == 1:
            draws = draws.unsqueeze(0)
        return draws.to(self.device).float()


class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto similarity kernel for non-negative fingerprint vectors."""

    is_stationary = False
    has_lengthscale = False

    def __init__(self, *, eps: float = 1e-12, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = float(eps)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        del params
        x1 = x1.float()
        x2 = x2.float()

        if diag:
            intersection = (x1 * x2).sum(dim=-1)
            denominator = x1.square().sum(dim=-1) + x2.square().sum(dim=-1) - intersection
        else:
            intersection = torch.matmul(x1, x2.transpose(-2, -1))
            x1_norm = x1.square().sum(dim=-1, keepdim=True)
            x2_norm = x2.square().sum(dim=-1, keepdim=True).transpose(-2, -1)
            denominator = x1_norm + x2_norm - intersection

        similarity = intersection / denominator.clamp_min(self.eps)
        return torch.where(denominator > self.eps, similarity, torch.ones_like(similarity))


class GPyTorchSurrogate(Surrogate):
    """GPyTorch exact-GP surrogate adapter."""

    def __init__(
        self,
        *,
        kernel: Literal["rbf", "matern12", "matern32", "matern52", "tanimoto"] = "rbf",
        training_steps: int = 100,
        learning_rate: float = 0.1,
        initial_noise: float = 1e-3,
        device: str = "cpu",
    ) -> None:

        self.kernel = kernel
        self.training_steps = int(training_steps)
        self.learning_rate = float(learning_rate)
        self.initial_noise = float(initial_noise)
        self._device = torch.device(device)

        self._model: torch.nn.Module | None = None
        self._likelihood: torch.nn.Module | None = None

    @staticmethod
    def _build_base_kernel(kernel_name: str) -> gpytorch.kernels.Kernel:
        if kernel_name == "rbf":
            return gpytorch.kernels.RBFKernel()
        if kernel_name == "matern12":
            return gpytorch.kernels.MaternKernel(nu=0.5)
        if kernel_name == "matern32":
            return gpytorch.kernels.MaternKernel(nu=1.5)
        if kernel_name == "matern52":
            return gpytorch.kernels.MaternKernel(nu=2.5)
        if kernel_name == "tanimoto":
            return TanimotoKernel()
        raise ValueError(f"Unsupported GPyTorch kernel: {kernel_name}")

    @property
    def backend_name(self) -> str:
        return "gp"

    @property
    def device(self) -> torch.device:
        return self._device

    def fit(self, context_x: torch.Tensor, context_y: torch.Tensor) -> None:
        train_x = context_x.to(self.device).float()
        train_y = context_y.to(self.device).view(-1).float()

        class ExactGPRegressor(gpytorch.models.ExactGP):
            def __init__(
                self,
                train_x_inner: torch.Tensor,
                train_y_inner: torch.Tensor,
                likelihood_inner: torch.nn.Module,
                kernel_name: str,
            ) -> None:
                super().__init__(train_x_inner, train_y_inner, likelihood_inner)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    GPyTorchSurrogate._build_base_kernel(kernel_name)
                )

            def forward(self, x_inner: torch.Tensor):  # type: ignore[override]
                mean_x = self.mean_module(x_inner)
                covar_x = self.covar_module(x_inner)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._likelihood.noise = torch.tensor(float(self.initial_noise), device=train_x.device)
        self._model = ExactGPRegressor(train_x, train_y, self._likelihood, self.kernel)
        self._model = self._model.to(self.device)
        self._likelihood = self._likelihood.to(self.device)

        self._model.train()
        self._likelihood.train()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        for _ in range(self.training_steps):
            optimizer.zero_grad()
            output = self._model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    def predict(self, x: torch.Tensor) -> SurrogatePrediction:
        if self._model is None or self._likelihood is None:
            raise RuntimeError("GPyTorchSurrogate.fit must be called before predict.")

        x = x.to(self.device).float()
        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self._likelihood(self._model(x))
            mean = posterior.mean
            variance = posterior.variance.clamp_min(1e-12)
        return SurrogatePrediction(mean=mean, variance=variance)

    def predict_query_target_joint(
        self,
        query_x: torch.Tensor,
        target_x: torch.Tensor,
    ) -> GPJointPrediction:
        """Return dense joint predictive stats needed by GP variance-reduction acquisitions."""
        if self._model is None or self._likelihood is None:
            raise RuntimeError("GPyTorchSurrogate.fit must be called before predict_query_target_joint.")

        query_x = query_x.to(self.device).float()
        target_x = target_x.to(self.device).float()
        x_union = torch.cat([query_x, target_x], dim=0)
        n_query = int(query_x.shape[0])

        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad():
            posterior = self._likelihood(self._model(x_union))
            mean = posterior.mean
            covariance = posterior.covariance_matrix

        query_mean = mean[:n_query]
        target_mean = mean[n_query:]
        query_variance = torch.diagonal(covariance[:n_query, :n_query], dim1=-2, dim2=-1).clamp_min(1e-12)
        target_variance = torch.diagonal(covariance[n_query:, n_query:], dim1=-2, dim2=-1).clamp_min(1e-12)
        target_query_covariance = covariance[n_query:, :n_query]

        return GPJointPrediction(
            query_mean=query_mean,
            query_variance=query_variance,
            target_mean=target_mean,
            target_variance=target_variance,
            target_query_covariance=target_query_covariance,
        )

    def point_log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.predict(x)
        y = y.to(self.device).view(-1).float()
        variance = pred.variance.clamp_min(1e-12)
        return -0.5 * (
            torch.log(2.0 * torch.pi * variance) + (y - pred.mean).square() / variance
        )

    def sample(self, x: torch.Tensor, *, num_samples: int) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        pred = self.predict(x)
        std = torch.sqrt(pred.variance.clamp_min(1e-12))
        eps = torch.randn(
            num_samples,
            pred.mean.shape[0],
            device=self.device,
            dtype=pred.mean.dtype,
        )
        return pred.mean.unsqueeze(0) + std.unsqueeze(0) * eps


class RandomForestSurrogate(Surrogate):
    """Random-forest surrogate with ensemble mean/variance uncertainty."""

    def __init__(
        self,
        *,
        n_estimators: int = 256,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = "sqrt",
        bootstrap: bool = True,
        variance_floor: float = 1e-6,
        n_jobs: int | None = 1,
        random_state: int | None = None,
        device: str = "cpu",
    ) -> None:
        del device

        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth) if max_depth is not None else None
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.variance_floor = float(variance_floor)
        self.n_jobs = int(n_jobs) if n_jobs is not None else None
        self.random_state = int(random_state) if random_state is not None else None

        self._device = torch.device("cpu")
        self._model: RandomForestRegressor | None = None
        self._rng = np.random.default_rng(self.random_state)

    @property
    def backend_name(self) -> str:
        return "rf"

    @property
    def device(self) -> torch.device:
        return self._device

    def fit(self, context_x: torch.Tensor, context_y: torch.Tensor) -> None:
        train_x = self._to_numpy_features(context_x)
        train_y = context_y.detach().cpu().view(-1).numpy().astype(np.float32, copy=False)

        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        model.fit(train_x, train_y)

        self._model = model

    def predict(self, x: torch.Tensor) -> SurrogatePrediction:
        tree_predictions = self._predict_tree_matrix(x)
        mean = tree_predictions.mean(axis=0)
        variance = tree_predictions.var(axis=0, ddof=0)
        variance = np.maximum(variance, self.variance_floor)
        return SurrogatePrediction(
            mean=torch.from_numpy(mean).to(self.device, dtype=torch.float32),
            variance=torch.from_numpy(variance).to(self.device, dtype=torch.float32),
        )

    def point_log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.predict(x)
        target = y.to(self.device).view(-1).float()
        variance = pred.variance.clamp_min(self.variance_floor)
        return -0.5 * (
            torch.log(2.0 * torch.pi * variance) + (target - pred.mean).square() / variance
        )

    def sample(self, x: torch.Tensor, *, num_samples: int) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        tree_predictions = self._predict_tree_matrix(x)
        n_trees, n_points = tree_predictions.shape
        if n_trees <= 0:
            raise RuntimeError("RandomForestSurrogate has no fitted trees.")

        tree_indices = self._rng.integers(0, n_trees, size=(num_samples, n_points))
        point_indices = np.broadcast_to(np.arange(n_points), (num_samples, n_points))
        draws = tree_predictions[tree_indices, point_indices]
        return torch.from_numpy(draws).to(self.device, dtype=torch.float32)

    def _predict_tree_matrix(self, x: torch.Tensor) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RandomForestSurrogate.fit must be called before predict.")

        query_x = self._to_numpy_features(x)
        tree_predictions = np.stack(
            [estimator.predict(query_x).astype(np.float32, copy=False) for estimator in self._model.estimators_],
            axis=0,
        )
        return tree_predictions

    @staticmethod
    def _to_numpy_features(x: torch.Tensor) -> np.ndarray:
        x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
        if x_np.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {tuple(x_np.shape)}.")
        return x_np
