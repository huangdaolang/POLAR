import math

import torch
from torch.testing import assert_close

from polar.tasks import GPPriorALSampler, GPPriorALTask


def test_gp_prior_sampler_shapes() -> None:
    sampler = GPPriorALSampler(
        dim_x=1,
        n_context_init=4,
        n_query=7,
        n_target=9,
        design_min=-3.0,
        design_max=3.0,
        plot_grid_size=50,
        noise_std=0.0,
        lengthscale_min=0.2,
        lengthscale_max=1.2,
        scale_min=0.2,
        scale_max=1.0,
        p_isotropic=1.0,
        seed=123,
    )
    episode = sampler.sample_episode()

    assert episode.context_x.shape == (4, 1)
    assert episode.context_y.shape == (4,)
    assert episode.query_x.shape == (7, 1)
    assert episode.query_y.shape == (7,)
    assert episode.target_x.shape == (9, 1)
    assert episode.target_y.shape == (9,)
    assert episode.plot_x.shape == (50, 1)
    assert episode.plot_y_true.shape == (50,)
    assert isinstance(episode.kernel_type, str)
    assert episode.lengthscales.shape == (1,)
    assert episode.output_scale.shape == (1,)

    assert torch.all(episode.context_x >= -3.0)
    assert torch.all(episode.context_x <= 3.0)


def test_gp_prior_sampler_adaptive_cholesky_jitter() -> None:
    sampler = GPPriorALSampler(
        dim_x=1,
        n_context_init=2,
        n_query=2,
        n_target=2,
        design_min=-1.0,
        design_max=1.0,
        plot_grid_size=10,
        noise_std=0.0,
        lengthscale_min=0.2,
        lengthscale_max=1.2,
        scale_min=0.2,
        scale_max=1.0,
        p_isotropic=1.0,
        jitter=1e-12,
        cholesky_max_tries=12,
        seed=0,
    )

    # Rank-deficient PSD covariance (all points perfectly correlated).
    cov = torch.ones(16, 16, dtype=torch.float32)
    sample, final_jitter = sampler._sample_multivariate(cov)
    assert sample.shape == (16,)
    assert torch.isfinite(sample).all()
    assert final_jitter > 0.0


def test_gp_prior_al_task_accepts_runtime_episode_sizes() -> None:
    task = GPPriorALTask(
        dim_x=1,
        design_min=-2.0,
        design_max=2.0,
        plot_grid_size=25,
        noise_std=0.0,
        lengthscale_min=0.2,
        lengthscale_max=1.2,
        scale_min=0.2,
        scale_max=1.0,
        p_isotropic=1.0,
        kernel_weights={"rbf": 1.0, "matern12": 0.0, "matern32": 0.0, "matern52": 0.0},
        jitter=1e-4,
        cholesky_max_tries=8,
        jitter_growth=10.0,
        seed=123,
    )

    batch_a = task.sample_train_batch(batch_size=2, n_context=3, n_query=4, n_target=2)
    batch_b = task.sample_train_batch(batch_size=1, n_context=5, n_query=6, n_target=3)
    eval_episode = task.build_test_episode(
        seed="test0",
        n_context=4,
        query_pool_size=7,
        n_target=5,
    )

    assert len(batch_a) == 2
    assert batch_a[0].context_x.shape == (3, 1)
    assert batch_a[0].query_x.shape == (4, 1)
    assert batch_a[0].target_x.shape == (2, 1)
    assert batch_b[0].context_x.shape == (5, 1)
    assert batch_b[0].query_x.shape == (6, 1)
    assert batch_b[0].target_x.shape == (3, 1)
    assert eval_episode.context_x.shape == (4, 1)
    assert eval_episode.query_x.shape == (7, 1)
    assert eval_episode.target_x.shape == (5, 1)


def test_gp_prior_al_task_variable_dim_batches_use_one_dim_per_batch() -> None:
    task = GPPriorALTask(
        dim_x_min=1,
        dim_x_max=4,
        design_min=-5.0,
        design_max=5.0,
        plot_grid_size=25,
        noise_std=0.0,
        lengthscale_min=0.1,
        lengthscale_max=1.5,
        scale_min=0.2,
        scale_max=1.0,
        p_isotropic=0.5,
        kernel_weights={"rbf": 1.0, "matern12": 0.0, "matern32": 1.0, "matern52": 1.0},
        jitter=1.0e-4,
        cholesky_max_tries=8,
        jitter_growth=10.0,
        lengthscale_sampling="log_uniform",
        standardize_x=True,
        standardize_y=True,
        seed=7,
    )

    seen_dims: set[int] = set()
    expected_scale = (5.0 - (-5.0)) / math.sqrt(12.0)

    for _ in range(12):
        batch = task.sample_train_batch(batch_size=3, n_context=2, n_query=4, n_target=3)
        batch_dims = {int(episode.context_x.shape[-1]) for episode in batch}
        assert len(batch_dims) == 1

        dim = batch_dims.pop()
        seen_dims.add(dim)
        assert 1 <= dim <= 4

        for episode in batch:
            assert episode.context_x.shape == (2, dim)
            assert episode.query_x.shape == (4, dim)
            assert episode.target_x.shape == (3, dim)
            assert episode.x_standardized is True
            assert episode.y_standardized is True
            assert episode.x_center is not None
            assert episode.x_scale is not None
            assert episode.y_mean is not None
            assert episode.y_std is not None
            assert episode.raw_context_x is not None
            assert episode.raw_query_x is not None
            assert episode.raw_target_x is not None
            assert episode.raw_context_y is not None
            assert episode.raw_query_y is not None
            assert episode.raw_target_y is not None

            assert_close(episode.x_scale, torch.full((dim,), expected_scale, dtype=torch.float32))
            assert_close(episode.context_x * episode.x_scale + episode.x_center, episode.raw_context_x)
            assert_close(episode.query_x * episode.x_scale + episode.x_center, episode.raw_query_x)
            assert_close(episode.target_x * episode.x_scale + episode.x_center, episode.raw_target_x)

            all_y = torch.cat([episode.context_y, episode.query_y, episode.target_y], dim=0)
            raw_y = torch.cat([episode.raw_context_y, episode.raw_query_y, episode.raw_target_y], dim=0)
            assert_close(all_y.mean(), torch.zeros((), dtype=torch.float32), atol=1.0e-5, rtol=0.0)
            assert_close(all_y.std(unbiased=False), torch.ones((), dtype=torch.float32), atol=1.0e-5, rtol=0.0)
            assert_close(all_y * episode.y_std + episode.y_mean, raw_y, atol=1.0e-5, rtol=1.0e-5)

    assert len(seen_dims) >= 2


def test_gp_prior_al_task_fixed_dim_range_matches_legacy_dim() -> None:
    task = GPPriorALTask(
        dim_x_min=3,
        dim_x_max=3,
        design_min=-2.0,
        design_max=2.0,
        plot_grid_size=20,
        noise_std=0.0,
        lengthscale_min=0.2,
        lengthscale_max=0.8,
        scale_min=0.5,
        scale_max=1.0,
        p_isotropic=1.0,
        kernel_weights={"rbf": 1.0, "matern12": 0.0, "matern32": 0.0, "matern52": 0.0},
        jitter=1.0e-4,
        cholesky_max_tries=8,
        jitter_growth=10.0,
        seed=11,
    )

    episode = task.build_test_episode(seed="fixed", n_context=2, query_pool_size=3, n_target=4)

    assert task.variable_dim is False
    assert task.dim_x == 3
    assert episode.context_x.shape == (2, 3)
    assert episode.query_x.shape == (3, 3)
    assert episode.target_x.shape == (4, 3)
