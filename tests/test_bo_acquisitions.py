import torch

from polar.utils.bo_acquisitions import (
    BOAcquisitionInputs,
    DEFAULT_EI_MC_SAMPLES,
    UCB_ALPHA,
    compute_bo_acquisition_scores,
    list_bo_acquisitions,
)


def _build_inputs() -> BOAcquisitionInputs:
    return BOAcquisitionInputs(
        candidate_x=torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
        predictive_mean=torch.tensor([0.2, 0.1, 0.3], dtype=torch.float32),
        predictive_variance=torch.tensor([0.5, 0.2, 0.8], dtype=torch.float32),
    )


def test_list_bo_acquisitions_contains_ucb() -> None:
    names = list_bo_acquisitions()
    assert "random" in names
    assert "ei" in names
    assert "ucb" in names
    assert "ei_mc" in names


def test_random_returns_one_score_per_candidate() -> None:
    torch.manual_seed(0)
    inputs = _build_inputs()
    scores = compute_bo_acquisition_scores(name="random", inputs=inputs)
    assert scores.shape == torch.Size([3])
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_ucb_uses_hardcoded_alpha() -> None:
    inputs = _build_inputs()
    scores = compute_bo_acquisition_scores(name="ucb", inputs=inputs)
    expected = inputs.predictive_mean + UCB_ALPHA * torch.sqrt(inputs.predictive_variance)
    assert torch.allclose(scores, expected)


def test_analytical_ei_matches_closed_form() -> None:
    inputs = BOAcquisitionInputs(
        candidate_x=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        predictive_mean=torch.tensor([0.5, 0.1], dtype=torch.float32),
        predictive_variance=torch.tensor([0.04, 0.01], dtype=torch.float32),  # std=[0.2,0.1]
        best_f=0.2,
    )
    scores = compute_bo_acquisition_scores(name="ei", inputs=inputs)

    std = torch.sqrt(inputs.predictive_variance)
    z = (inputs.predictive_mean - 0.2) / std
    normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    expected = (inputs.predictive_mean - 0.2) * normal.cdf(z) + std * torch.exp(normal.log_prob(z))
    assert torch.allclose(scores, expected, atol=1e-6)


def test_analytical_ei_handles_zero_variance() -> None:
    inputs = BOAcquisitionInputs(
        candidate_x=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        predictive_mean=torch.tensor([0.1, 0.3], dtype=torch.float32),
        predictive_variance=torch.tensor([0.0, 0.0], dtype=torch.float32),
        best_f=0.2,
    )
    scores = compute_bo_acquisition_scores(name="ei", inputs=inputs)
    expected = torch.tensor([0.0, 0.1], dtype=torch.float32)
    assert torch.allclose(scores, expected, atol=1e-6)


def test_ei_mc_uses_provided_samples() -> None:
    inputs = _build_inputs()
    samples = torch.tensor(
        [
            [0.3, 0.2, 0.6],
            [0.1, 0.5, 0.2],
        ],
        dtype=torch.float32,
    )
    # best_f=0.25 => improvements:
    # col0: [0.05, 0.00] -> 0.025
    # col1: [0.00, 0.25] -> 0.125
    # col2: [0.35, 0.00] -> 0.175
    ei_inputs = BOAcquisitionInputs(
        candidate_x=inputs.candidate_x,
        predictive_mean=inputs.predictive_mean,
        predictive_variance=inputs.predictive_variance,
        best_f=0.25,
        predictive_samples=samples,
    )
    scores = compute_bo_acquisition_scores(name="ei_mc", inputs=ei_inputs)
    expected = torch.tensor([0.025, 0.125, 0.175], dtype=torch.float32)
    assert torch.allclose(scores, expected, atol=1e-6)


def test_ei_mc_without_samples_uses_mc_draws() -> None:
    torch.manual_seed(0)
    inputs = _build_inputs()
    ei_inputs = BOAcquisitionInputs(
        candidate_x=inputs.candidate_x,
        predictive_mean=inputs.predictive_mean,
        predictive_variance=inputs.predictive_variance,
        best_f=0.0,
        mc_num_samples=DEFAULT_EI_MC_SAMPLES,
    )
    scores = compute_bo_acquisition_scores(name="ei_mc", inputs=ei_inputs)
    assert scores.shape == torch.Size([3])
    assert torch.all(scores >= 0.0)


def test_unknown_acquisition_raises() -> None:
    inputs = _build_inputs()
    try:
        compute_bo_acquisition_scores(name="does_not_exist", inputs=inputs)
    except KeyError:
        return
    raise AssertionError("Expected KeyError for unknown acquisition name.")
