from __future__ import annotations

import matplotlib
import torch

from polar.utils.plotting import (
    plot_location_trajectories_comparison_2d,
    plot_location_trajectory_2d,
)


matplotlib.use("Agg")


def test_plot_location_trajectory_2d_smoke() -> None:
    theta = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    x_traj = torch.tensor(
        [[-1.0, -1.0], [-0.5, 0.2], [0.3, 0.8]],
        dtype=torch.float32,
    )

    fig, ax = plot_location_trajectory_2d(
        theta,
        x_traj,
        design_min=-4.0,
        design_max=4.0,
        title="Smoke Test",
    )

    assert ax.get_title() == "Smoke Test"
    assert fig.axes[-1].get_ylabel() == "Acquisition step"
    assert len(ax.lines) == 0
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["True $\\theta$", "Design"]
    assert ax.get_xlabel() == "Coordinate 1"
    assert ax.get_ylabel() == "Coordinate 2"


def test_plot_location_trajectories_comparison_2d_smoke() -> None:
    theta = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    trajectories = {
        "Ours": torch.tensor([[-1.0, -1.0], [-0.5, 0.2], [0.3, 0.8]], dtype=torch.float32),
        "DAD": torch.tensor([[-0.8, -0.9], [-0.2, 0.1], [0.1, 0.7]], dtype=torch.float32),
    }

    fig, axes = plot_location_trajectories_comparison_2d(
        theta,
        trajectories,
        design_min=-4.0,
        design_max=4.0,
    )

    assert axes.shape == (1, 2)
    assert axes[0, 0].get_title() == "Ours"
    assert axes[0, 1].get_title() == "DAD"
    legend = axes[0, 0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["True $\\theta$", "Design"]
    assert axes[0, 0].get_ylabel() == "Coordinate 2"
    assert axes[0, 1].get_ylabel() == ""
    assert all(not tick.get_visible() for tick in axes[0, 1].get_yticklabels())
    assert fig.axes[-1].get_ylabel() == "Acquisition step"
