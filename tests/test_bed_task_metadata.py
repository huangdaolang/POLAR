from polar.tasks import create_bed_task
from polar.tasks.bed import bed_task_checkpoint_kind


class _TaskCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_ces_uses_regression_checkpoint_kind() -> None:
    task = create_bed_task(
        _TaskCfg(
            name="ces",
            n_goods=3,
            n_contrastive=4,
            n_context_init=1,
            n_query=4,
            n_target=2,
            design_min=0.0,
            design_max=100.0,
            noise_scale=0.005,
            epsilon=1.0e-5,
        ),
        seed=0,
    )

    assert task.observation_type == "continuous"
    assert task.num_classes is None
    assert task.epsilon == 1.0e-5
    assert bed_task_checkpoint_kind(task) == "regression"
