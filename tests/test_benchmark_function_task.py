import torch
from torch.testing import assert_close

from polar.tasks import BenchmarkFunctionALTask


def test_benchmark_function_task_normalizes_x_and_y_with_recoverable_metadata() -> None:
    task = BenchmarkFunctionALTask(
        benchmarks=["forrester", "branin"],
        normalize_x=True,
        normalize_y=True,
        normalization_num_samples=1024,
        seed=13,
    )

    episode = task.build_test_episode(
        dataset_id="branin",
        seed="episode0",
        n_context=2,
        query_pool_size=3,
        n_target=4,
    )

    assert episode.dataset_id == "branin"
    assert episode.context_x.shape == (2, 2)
    assert episode.query_x.shape == (3, 2)
    assert episode.target_x.shape == (4, 2)
    assert episode.raw_context_x.shape == (2, 2)
    assert episode.raw_query_x.shape == (3, 2)
    assert episode.raw_target_x.shape == (4, 2)

    assert_close(episode.context_x * episode.x_scale + episode.x_center, episode.raw_context_x)
    assert_close(episode.query_x * episode.x_scale + episode.x_center, episode.raw_query_x)
    assert_close(episode.target_x * episode.x_scale + episode.x_center, episode.raw_target_x)
    assert_close(episode.context_y * episode.y_std + episode.y_mean, episode.raw_context_y)
    assert_close(episode.query_y * episode.y_std + episode.y_mean, episode.raw_query_y)
    assert_close(episode.target_y * episode.y_std + episode.y_mean, episode.raw_target_y)

    metadata = task.benchmark_metadata()
    assert set(metadata) == {"forrester", "branin"}
    assert metadata["branin"]["dim_x"] == 2
    assert metadata["branin"]["normalize_x"] is True
    assert metadata["branin"]["normalize_y"] is True


def test_benchmark_function_task_samples_one_dataset_per_train_batch() -> None:
    task = BenchmarkFunctionALTask(
        benchmarks=["forrester", "branin"],
        normalize_x=True,
        normalize_y=True,
        normalization_num_samples=512,
        seed=5,
    )

    batch = task.sample_train_batch(batch_size=4, n_context=2, n_query=3, n_target=2)

    dataset_ids = {episode.dataset_id for episode in batch}
    dims = {int(episode.context_x.shape[-1]) for episode in batch}
    assert len(dataset_ids) == 1
    assert len(dims) == 1


def test_benchmark_function_task_iter_test_episodes_respects_dataset_filter() -> None:
    task = BenchmarkFunctionALTask(
        benchmarks=["forrester", "branin"],
        normalize_x=True,
        normalize_y=True,
        normalization_num_samples=256,
        seed=9,
    )

    episodes = list(
        task.iter_test_episodes(
            n_context=1,
            query_pool_size=2,
            n_target=1,
            seeds=["s0", "s1"],
            dataset_ids=["forrester"],
        )
    )

    assert len(episodes) == 2
    assert [episode.dataset_id for episode in episodes] == ["forrester", "forrester"]
    assert [episode.seed for episode in episodes] == ["s0", "s1"]


def test_benchmark_function_task_plot_bundle_and_evaluate_raw() -> None:
    task = BenchmarkFunctionALTask(
        benchmarks=["forrester", "branin"],
        normalize_x=True,
        normalize_y=True,
        normalization_num_samples=256,
        seed=3,
    )

    line_bundle = task.plot_bundle("forrester", resolution_1d=32)
    surface_bundle = task.plot_bundle("branin", resolution_2d=12)
    raw_eval = task.evaluate_raw("forrester", torch.tensor([[0.5]], dtype=torch.float32))

    assert line_bundle["kind"] == "line"
    assert line_bundle["raw_plot_x"].shape == (32, 1)
    assert line_bundle["raw_plot_y_true"].shape == (32,)

    assert surface_bundle["kind"] == "surface"
    assert surface_bundle["raw_plot_x"].shape == (144, 2)
    assert surface_bundle["raw_plot_y_true"].shape == (144,)
    assert surface_bundle["grid_shape"] == (12, 12)

    assert raw_eval.shape == (1,)


def test_sine_gaussian_bumps_is_registered_and_auto_normalized() -> None:
    task = BenchmarkFunctionALTask(
        benchmarks=["sine_gaussian_bumps"],
        normalize_x=True,
        normalize_y=True,
        normalization_num_samples=2048,
        seed=17,
    )

    points = torch.tensor([[-4.5], [2.5], [7.5]], dtype=torch.float32)
    raw_eval = task.evaluate_raw("sine_gaussian_bumps", points)

    def gaussian_pdf(x: torch.Tensor, *, mean: float, std: float) -> torch.Tensor:
        variance = std**2
        norm = std * torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=x.dtype))
        return torch.exp(-0.5 * (x - mean).square() / variance) / norm

    expected = (
        2.0 * torch.sin(2.0 * points.squeeze(-1))
        + 8.0 * gaussian_pdf(points.squeeze(-1), mean=2.5, std=0.5)
        + 10.0 * gaussian_pdf(points.squeeze(-1), mean=7.5, std=0.25)
        - 6.0 * gaussian_pdf(points.squeeze(-1), mean=-4.5, std=0.5)
    )
    assert_close(raw_eval, expected, rtol=1e-5, atol=1e-6)

    episode = task.build_test_episode(
        dataset_id="sine_gaussian_bumps",
        seed="episode0",
        n_context=4,
        query_pool_size=5,
        n_target=6,
    )
    assert_close(episode.context_x * episode.x_scale + episode.x_center, episode.raw_context_x)
    assert_close(episode.query_x * episode.x_scale + episode.x_center, episode.raw_query_x)
    assert_close(episode.target_x * episode.x_scale + episode.x_center, episode.raw_target_x)
    assert_close(episode.context_y * episode.y_std + episode.y_mean, episode.raw_context_y)
    assert_close(episode.query_y * episode.y_std + episode.y_mean, episode.raw_query_y)
    assert_close(episode.target_y * episode.y_std + episode.y_mean, episode.raw_target_y)

    metadata = task.benchmark_metadata()["sine_gaussian_bumps"]
    assert metadata["dim_x"] == 1
    assert metadata["lower_bounds"] == [-8.0]
    assert metadata["upper_bounds"] == [8.0]
    assert metadata["normalize_x"] is True
    assert metadata["normalize_y"] is True
