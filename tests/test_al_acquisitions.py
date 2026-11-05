import math

import torch

from polar.utils.al_acquisitions import (
    ALAcquisitionInputs,
    _step_pos_weighted_gaussian_uncertainty,
    compute_al_acquisition_scores,
    list_al_acquisitions,
)


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)


def _manual_step_pos_weighted_uncertainty(
    target_mean: torch.Tensor,
    target_variance: torch.Tensor,
    *,
    beta: float,
    threshold: float = 0.0,
) -> torch.Tensor:
    mean = target_mean.to(dtype=torch.float64)
    variance = target_variance.to(dtype=torch.float64).clamp_min(0.0)
    positive_variance = variance > 1.0e-12
    safe_std = torch.sqrt(variance.clamp_min(1.0e-12))
    standardized = (mean - float(threshold)) / safe_std
    prob_pos = _standard_normal_cdf(standardized)
    pdf_pos = _standard_normal_pdf(standardized)
    positive_first_moment = mean * prob_pos + safe_std * pdf_pos
    positive_second_moment = (
        (mean.square() + variance) * prob_pos
        + (mean + float(threshold)) * safe_std * pdf_pos
    )
    deterministic_positive = mean > float(threshold)
    prob_pos = torch.where(positive_variance, prob_pos, deterministic_positive.to(mean.dtype))
    positive_first_moment = torch.where(
        positive_variance,
        positive_first_moment,
        torch.where(deterministic_positive, mean, torch.zeros_like(mean)),
    )
    positive_second_moment = torch.where(
        positive_variance,
        positive_second_moment,
        torch.where(deterministic_positive, mean.square(), torch.zeros_like(mean)),
    )
    beta_minus_one = float(beta) - 1.0
    weighted_zero_moment = 1.0 + beta_minus_one * prob_pos
    weighted_first_moment = mean + beta_minus_one * positive_first_moment
    weighted_second_moment = mean.square() + variance + beta_minus_one * positive_second_moment
    return weighted_second_moment - weighted_first_moment.square() / weighted_zero_moment


def _manual_antithetic_standard_normal_samples(n_samples: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    n_base = (n_samples + 1) // 2
    base = torch.randn(n_base, generator=generator, dtype=torch.float64)
    return torch.cat([base, -base], dim=0)[:n_samples]


def _manual_step_pos_weighted_variance_reduction(
    inputs: ALAcquisitionInputs,
    *,
    n_samples: int = 128,
) -> torch.Tensor:
    target_mean = inputs.target_mean.to(dtype=torch.float64)
    target_variance = inputs.target_variance.to(dtype=torch.float64).clamp_min(0.0)
    covariance = inputs.target_query_covariance.to(dtype=torch.float64)
    denom = inputs.query_variance.to(dtype=torch.float64).clamp_min(1.0e-12)
    variance_gain = covariance.square() / denom[None, :]
    prior_uncertainty = _manual_step_pos_weighted_uncertainty(
        target_mean,
        target_variance,
        beta=inputs.target_weight_beta,
        threshold=inputs.target_weight_threshold,
    )
    posterior_variance = (target_variance[:, None] - variance_gain).clamp_min(0.0)
    posterior_eps = _manual_antithetic_standard_normal_samples(n_samples)
    posterior_mean = (
        target_mean[:, None, None]
        + covariance[:, :, None] / torch.sqrt(denom)[None, :, None] * posterior_eps[None, None, :]
    )
    posterior_uncertainty = _manual_step_pos_weighted_uncertainty(
        posterior_mean,
        posterior_variance[:, :, None],
        beta=inputs.target_weight_beta,
        threshold=inputs.target_weight_threshold,
    )
    return (prior_uncertainty[:, None] - posterior_uncertainty.mean(dim=-1)).mean(dim=0).float()


def _build_inputs() -> ALAcquisitionInputs:
    return ALAcquisitionInputs(
        backend="tabpfn",
        query_x=torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
        query_mean=torch.tensor([0.2, 0.1, 0.3], dtype=torch.float32),
        query_variance=torch.tensor([0.5, 0.2, 0.8], dtype=torch.float32),
    )


def test_list_al_acquisitions_contains_basics() -> None:
    names = list_al_acquisitions()
    assert "random" in names
    assert "uncertainty" in names
    assert "variance_reduction" in names
    assert "weighted_variance_reduction" in names


def test_uncertainty_returns_variance() -> None:
    inputs = _build_inputs()
    scores = compute_al_acquisition_scores(name="uncertainty", inputs=inputs)
    assert torch.allclose(scores, inputs.query_variance)


def test_variance_reduction_matches_closed_form() -> None:
    inputs = ALAcquisitionInputs(
        backend="gp",
        query_x=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        query_mean=torch.tensor([0.1, -0.2], dtype=torch.float32),
        query_variance=torch.tensor([0.5, 0.8], dtype=torch.float32),
        target_mean=torch.tensor([0.5, -1.0], dtype=torch.float32),
        target_variance=torch.tensor([0.2, 0.7], dtype=torch.float32),
        target_query_covariance=torch.tensor(
            [[0.1, 0.3], [0.4, 0.2]],
            dtype=torch.float32,
        ),
    )

    scores = compute_al_acquisition_scores(name="variance_reduction", inputs=inputs)
    expected = torch.tensor(
        [
            ((0.1**2) + (0.4**2)) / 2.0 / 0.5,
            ((0.3**2) + (0.2**2)) / 2.0 / 0.8,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(scores, expected)


def test_weighted_variance_reduction_uses_exponential_posterior_weight() -> None:
    inputs = ALAcquisitionInputs(
        backend="gp",
        query_x=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        query_mean=torch.tensor([0.1, -0.2], dtype=torch.float32),
        query_variance=torch.tensor([0.5, 0.8], dtype=torch.float32),
        target_mean=torch.tensor([0.5, -1.0], dtype=torch.float32),
        target_variance=torch.tensor([0.2, 0.7], dtype=torch.float32),
        target_query_covariance=torch.tensor(
            [[0.1, 0.3], [0.4, 0.2]],
            dtype=torch.float32,
        ),
        target_weighting="exp_y",
        target_weight_beta=1.0,
    )

    scores = compute_al_acquisition_scores(name="weighted_variance_reduction", inputs=inputs)
    weights = torch.exp(inputs.target_mean + 0.5 * inputs.target_variance)
    expected = (
        weights[:, None] * inputs.target_query_covariance.square() / inputs.query_variance[None, :]
    ).mean(dim=0)
    assert torch.allclose(scores, expected)


def test_weighted_variance_reduction_matches_step_pos_weighted_uncertainty() -> None:
    inputs = ALAcquisitionInputs(
        backend="gp",
        query_x=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        query_mean=torch.tensor([0.1, -0.2], dtype=torch.float32),
        query_variance=torch.tensor([0.5, 0.8], dtype=torch.float32),
        target_mean=torch.tensor([0.5, -1.0], dtype=torch.float32),
        target_variance=torch.tensor([0.2, 0.7], dtype=torch.float32),
        target_query_covariance=torch.tensor(
            [[0.1, 0.3], [0.4, 0.2]],
            dtype=torch.float32,
        ),
        target_weighting="step_pos",
        target_weight_beta=3.0,
    )

    scores = compute_al_acquisition_scores(name="weighted_variance_reduction", inputs=inputs)
    expected = _manual_step_pos_weighted_variance_reduction(inputs)
    assert torch.allclose(scores, expected)


def test_step_pos_weighted_uncertainty_handles_deterministic_targets() -> None:
    uncertainty = _step_pos_weighted_gaussian_uncertainty(
        torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        beta=4.0,
    )
    assert torch.allclose(uncertainty, torch.zeros(3, dtype=uncertainty.dtype))


def test_weighted_variance_reduction_step_pos_uses_custom_threshold() -> None:
    inputs = ALAcquisitionInputs(
        backend="gp",
        query_x=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        query_mean=torch.tensor([0.1, -0.2], dtype=torch.float32),
        query_variance=torch.tensor([0.5, 0.8], dtype=torch.float32),
        target_mean=torch.tensor([0.5, 1.5], dtype=torch.float32),
        target_variance=torch.tensor([0.2, 0.7], dtype=torch.float32),
        target_query_covariance=torch.tensor(
            [[0.1, 0.3], [0.4, 0.2]],
            dtype=torch.float32,
        ),
        target_weighting="step_pos",
        target_weight_beta=3.0,
        target_weight_threshold=1.0,
    )

    scores = compute_al_acquisition_scores(name="weighted_variance_reduction", inputs=inputs)
    expected = _manual_step_pos_weighted_variance_reduction(inputs)
    assert torch.allclose(scores, expected)


def test_weighted_variance_reduction_reduces_to_vr_for_uniform_weights() -> None:
    inputs = ALAcquisitionInputs(
        backend="gp",
        query_x=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        query_mean=torch.tensor([0.1, -0.2], dtype=torch.float32),
        query_variance=torch.tensor([0.5, 0.8], dtype=torch.float32),
        target_mean=torch.tensor([0.5, -1.0], dtype=torch.float32),
        target_variance=torch.tensor([0.2, 0.7], dtype=torch.float32),
        target_query_covariance=torch.tensor(
            [[0.1, 0.3], [0.4, 0.2]],
            dtype=torch.float32,
        ),
        target_weighting="uniform",
        target_weight_beta=3.0,
    )

    vr_scores = compute_al_acquisition_scores(name="variance_reduction", inputs=inputs)
    weighted_scores = compute_al_acquisition_scores(
        name="weighted_variance_reduction",
        inputs=inputs,
    )
    assert torch.allclose(weighted_scores, vr_scores)


def test_weighted_variance_reduction_requires_beta_above_one_for_step_pos() -> None:
    inputs = ALAcquisitionInputs(
        backend="gp",
        query_x=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        query_mean=torch.tensor([0.1, -0.2], dtype=torch.float32),
        query_variance=torch.tensor([0.5, 0.8], dtype=torch.float32),
        target_mean=torch.tensor([0.5, -1.0], dtype=torch.float32),
        target_variance=torch.tensor([0.2, 0.7], dtype=torch.float32),
        target_query_covariance=torch.tensor(
            [[0.1, 0.3], [0.4, 0.2]],
            dtype=torch.float32,
        ),
        target_weighting="step_pos",
        target_weight_beta=1.0,
    )

    try:
        compute_al_acquisition_scores(name="weighted_variance_reduction", inputs=inputs)
    except ValueError as exc:
        assert "step_pos" in str(exc)
    else:
        raise AssertionError("Expected step_pos weighting to require beta > 1.")


def test_unknown_acquisition_raises() -> None:
    inputs = _build_inputs()
    try:
        compute_al_acquisition_scores(name="does_not_exist", inputs=inputs)
    except KeyError:
        return
    raise AssertionError("Expected KeyError for unknown acquisition name.")
