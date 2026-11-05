from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import torch

from polar.utils.plotting import (
    PAPER_COLORS,
    apply_style,
    draw_policy_probe_score_axis,
    draw_policy_probe_snapshot_axis,
)


def test_apply_style_updates_common_rcparams() -> None:
    apply_style(use_tex=False)

    assert matplotlib.rcParams["text.usetex"] is False
    assert float(matplotlib.rcParams["font.size"]) == 16.0
    assert float(matplotlib.rcParams["lines.linewidth"]) == 2.0
    assert tuple(matplotlib.rcParams["figure.figsize"]) == (3.0, 3.0)
    assert matplotlib.rcParams["axes.facecolor"] == "#fdfdfd"
    assert matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][: len(PAPER_COLORS)] == list(
        PAPER_COLORS
    )


def test_paper_colors_are_fixed_hex_values() -> None:
    assert len(PAPER_COLORS) == 8
    assert all(color.startswith("#") for color in PAPER_COLORS)
    assert PAPER_COLORS[0] == "#0072B2"
    assert PAPER_COLORS[6] == "#7F7F7F"


def test_draw_policy_probe_snapshot_axis_smoke() -> None:
    fig, ax = plt.subplots()
    snapshot = {
        "round": 4,
        "target_rmse": 0.25,
        "weighted_target_rmse": 0.2,
        "raw_plot_x": torch.linspace(-1.0, 1.0, 32).unsqueeze(-1),
        "raw_plot_y_true": torch.sin(torch.linspace(-1.0, 1.0, 32)),
        "pred_mean_raw": torch.sin(torch.linspace(-1.0, 1.0, 32)) * 0.9,
        "pred_std_raw": torch.full((32,), 0.1),
        "raw_context_x": torch.tensor([[-0.8], [0.2]], dtype=torch.float32),
        "raw_context_y": torch.tensor([-0.6, 0.25], dtype=torch.float32),
        "raw_target_x": torch.tensor([[-0.4], [0.7]], dtype=torch.float32),
        "raw_target_y": torch.tensor([-0.35, 0.5], dtype=torch.float32),
        "acquired_raw_x": torch.tensor([[-0.2], [0.5]], dtype=torch.float32),
        "acquired_raw_y": torch.tensor([-0.15, 0.45], dtype=torch.float32),
    }

    draw_policy_probe_snapshot_axis(
        ax,
        snapshot=snapshot,
        ci_std=1.96,
        title="Round 5",
    )

    handles, labels = ax.get_legend_handles_labels()
    assert ax.get_title() == "Round 5"
    assert "True" in labels
    assert "Latest query" in labels
    assert len(handles) == len(labels)
    plt.close(fig)


def test_draw_policy_probe_snapshot_axis_supports_query_only_mode() -> None:
    fig, ax = plt.subplots()
    snapshot = {
        "round": 4,
        "target_rmse": 0.25,
        "weighted_target_rmse": 0.2,
        "raw_plot_x": torch.linspace(-1.0, 1.0, 32).unsqueeze(-1),
        "raw_plot_y_true": torch.sin(torch.linspace(-1.0, 1.0, 32)),
        "pred_mean_raw": torch.sin(torch.linspace(-1.0, 1.0, 32)) * 0.9,
        "pred_std_raw": torch.full((32,), 0.1),
        "raw_context_x": torch.tensor([[-0.8], [0.2]], dtype=torch.float32),
        "raw_context_y": torch.tensor([-0.6, 0.25], dtype=torch.float32),
        "raw_target_x": torch.tensor([[-0.4], [0.7]], dtype=torch.float32),
        "raw_target_y": torch.tensor([-0.35, 0.5], dtype=torch.float32),
        "acquired_raw_x": torch.tensor([[-0.2], [0.5]], dtype=torch.float32),
        "acquired_raw_y": torch.tensor([-0.15, 0.45], dtype=torch.float32),
    }

    draw_policy_probe_snapshot_axis(
        ax,
        snapshot=snapshot,
        ci_std=1.96,
        show_target=False,
        combine_context_and_acquired=True,
        latest_query_style="vline",
    )

    handles, labels = ax.get_legend_handles_labels()
    assert "Target" not in labels
    assert "Queried" in labels
    assert "Latest query" in labels
    plt.close(fig)


def test_draw_policy_probe_score_axis_smoke() -> None:
    fig, ax = plt.subplots()
    snapshot = {
        "score_query_raw_x": torch.tensor([[0.3], [-0.2], [0.7]], dtype=torch.float32),
        "score_values": torch.tensor([0.2, 0.5, 0.1], dtype=torch.float32),
        "score_label": "Policy prob",
        "score_kind": "policy_prob",
        "selected_query_raw_x": torch.tensor([0.3], dtype=torch.float32),
        "selected_score_value": 0.2,
    }

    draw_policy_probe_score_axis(ax, snapshot=snapshot)

    handles, labels = ax.get_legend_handles_labels()
    assert ax.get_ylabel() == "Policy prob"
    assert "Policy prob" in labels
    assert "Latest query" in labels
    plt.close(fig)
