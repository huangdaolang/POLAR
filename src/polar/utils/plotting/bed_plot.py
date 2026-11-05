"""BED-specific plotting helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from polar.utils.plotting.base import PAPER_COLORS, UncertaintyMode


def _get_location_trajectory_cmap() -> Any:
    """Return a muted sequential colormap for acquisition order."""

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap("Blues")
    return LinearSegmentedColormap.from_list(
        "location_trajectory_blues",
        base(np.linspace(0.24, 0.96, 256)),
    )


def plot_location_finding_trajectories(
    save_path: Path,
    trajectories: list[dict[str, Any]],
    *,
    design_min: float,
    design_max: float,
) -> None:
    """Plot BED trajectories for the 2-D location-finding task."""

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if not trajectories:
        return

    n = len(trajectories)
    n_cols = min(n, 4)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols + 0.8, 3.8 * n_rows),
        squeeze=False,
    )

    max_steps = max(traj["x_traj"].shape[0] for traj in trajectories)
    norm = Normalize(vmin=1, vmax=max(max_steps, 2))
    cmap = _get_location_trajectory_cmap()

    for idx, traj in enumerate(trajectories):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        theta_true = traj["theta_true"]
        x_traj = traj["x_traj"]
        steps = list(range(1, x_traj.shape[0] + 1))

        ax.scatter(
            x_traj[:, 0],
            x_traj[:, 1],
            c=steps,
            cmap=cmap,
            norm=norm,
            s=40,
            edgecolors="k",
            linewidths=0.5,
            zorder=2,
        )
        ax.scatter(
            theta_true[:, 0],
            theta_true[:, 1],
            marker="*",
            s=200,
            c=PAPER_COLORS[1],
            edgecolors="k",
            linewidths=0.8,
            zorder=3,
            label=r"True $\theta$",
        )

        ax.set_xlim(design_min, design_max)
        ax.set_ylim(design_min, design_max)
        ax.set_aspect("equal")
        title = f"Episode {idx + 1}"
        spce_val = traj.get("spce")
        if spce_val is not None:
            title += f"  (sPCE={spce_val:.2f})"
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)

    for idx in range(len(trajectories), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        label="Time step",
        shrink=0.85,
    )
    fig.suptitle("BED Policy Trajectories (Location Finding)", fontsize=13)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_location_trajectory_2d(
    theta_true: Any,
    x_traj: Any,
    *,
    design_min: float,
    design_max: float,
    title: str | None = None,
    ax: Any | None = None,
    show_colorbar: bool = True,
) -> tuple[Any, Any]:
    """Plot a single 2-D location-finding trajectory with a standard colorbar."""

    import numpy as np
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt

    theta_arr = np.asarray(theta_true, dtype=float)
    x_arr = np.asarray(x_traj, dtype=float)

    if theta_arr.ndim != 2 or theta_arr.shape[1] != 2:
        raise ValueError(
            "theta_true must have shape [K, 2] for 2-D location-finding plots."
        )
    if x_arr.ndim != 2 or x_arr.shape[1] != 2:
        raise ValueError("x_traj must have shape [T, 2] for 2-D location-finding plots.")
    if x_arr.shape[0] == 0:
        raise ValueError("x_traj must contain at least one design.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.4, 4.0))
        created_fig = True
    else:
        fig = ax.figure

    norm = Normalize(vmin=1, vmax=max(int(x_arr.shape[0]), 2))
    cmap = _get_location_trajectory_cmap()
    scatter = _plot_location_trajectory_axis(
        ax=ax,
        theta_arr=theta_arr,
        x_arr=x_arr,
        norm=norm,
        cmap=cmap,
        design_min=design_min,
        design_max=design_max,
        title=title,
        show_legend=True,
        show_xlabel=True,
        show_ylabel=True,
    )

    if show_colorbar:
        colorbar = fig.colorbar(scatter, ax=ax, label="Acquisition step", shrink=0.9)
        colorbar.ax.tick_params(labelsize=15)
        colorbar.set_label(
            "Acquisition step",
            fontsize=22,
        )

    if created_fig:
        fig.tight_layout()
    return fig, ax


def plot_location_trajectories_comparison_2d(
    theta_true: Any,
    trajectories: dict[str, Any],
    *,
    design_min: float,
    design_max: float,
    show_colorbar: bool = True,
) -> tuple[Any, Any]:
    """Plot multiple 2-D location-finding trajectories with one shared colorbar."""

    import numpy as np
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt

    theta_arr = np.asarray(theta_true, dtype=float)
    if theta_arr.ndim != 2 or theta_arr.shape[1] != 2:
        raise ValueError(
            "theta_true must have shape [K, 2] for 2-D location-finding plots."
        )
    if not trajectories:
        raise ValueError("trajectories must be a non-empty dict of method -> [T, 2] arrays.")

    items = list(trajectories.items())
    max_steps = 0
    parsed_trajectories: list[tuple[str, Any]] = []
    for method, x_traj in items:
        x_arr = np.asarray(x_traj, dtype=float)
        if x_arr.ndim != 2 or x_arr.shape[1] != 2:
            raise ValueError(
                f"Trajectory for '{method}' must have shape [T, 2], got {tuple(x_arr.shape)}."
            )
        if x_arr.shape[0] == 0:
            raise ValueError(f"Trajectory for '{method}' must contain at least one design.")
        parsed_trajectories.append((method, x_arr))
        max_steps = max(max_steps, int(x_arr.shape[0]))

    n_panels = len(parsed_trajectories)
    n_cols = min(n_panels, 2)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            4.45 * n_cols + 0.95,
            3.0 * n_rows + 0.10,
        ),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.8,
        bottom=0.10,
        top=0.92,
        wspace=0.0,
        hspace=0.30,
    )

    norm = Normalize(vmin=1, vmax=max(max_steps, 2))
    cmap = _get_location_trajectory_cmap()
    visible_axes = []
    shared_scatter = None

    for idx, (method, x_arr) in enumerate(parsed_trajectories):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        shared_scatter = _plot_location_trajectory_axis(
            ax=ax,
            theta_arr=theta_arr,
            x_arr=x_arr,
            norm=norm,
            cmap=cmap,
            design_min=design_min,
            design_max=design_max,
            title=method,
            show_legend=(idx == 0),
            show_xlabel=(row == n_rows - 1),
            show_ylabel=(col == 0),
        )
        visible_axes.append(ax)

    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    if show_colorbar and shared_scatter is not None:
        colorbar = fig.colorbar(
            shared_scatter,
            ax=visible_axes,
            label="Acquisition step",
            shrink=0.9,
            pad=0.05,
        )
        colorbar.ax.tick_params(labelsize=15)
        colorbar.set_label(
            "Acquisition step",
            fontsize=24,
        )

    return fig, axes


def _plot_location_trajectory_axis(
    *,
    ax: Any,
    theta_arr: Any,
    x_arr: Any,
    norm: Any,
    cmap: Any,
    design_min: float,
    design_max: float,
    title: str | None,
    show_legend: bool,
    show_xlabel: bool,
    show_ylabel: bool,
) -> Any:
    """Draw one 2-D trajectory panel and return the scatter artist."""

    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    design_color = cmap(norm(1))
    design_size = 40.0
    theta_size = 300.0

    scatter = ax.scatter(
        x_arr[:, 0],
        x_arr[:, 1],
        c=list(range(1, x_arr.shape[0] + 1)),
        cmap=cmap,
        norm=norm,
        s=design_size,
        edgecolors="k",
        linewidths=0.6,
        zorder=2,
    )
    ax.scatter(
        theta_arr[:, 0],
        theta_arr[:, 1],
        marker="*",
        s=theta_size,
        c=PAPER_COLORS[1],
        edgecolors="k",
        linewidths=0.8,
        zorder=3,
    )

    ax.set_xlim(design_min - 0.05, design_max + 0.05)
    ax.set_ylim(design_min - 0.05, design_max + 0.05)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_aspect("equal")
    ax.set_xlabel(
        "Coordinate 1" if show_xlabel else "",
        fontsize=20,
    )
    ax.set_ylabel(
        "Coordinate 2" if show_ylabel else "",
        fontsize=20,
    )
    ax.grid(alpha=0.2)
    if title is not None:
        ax.set_title(title, fontsize=22)
    ax.tick_params(axis="x", labelsize=16)
    if show_ylabel:
        ax.tick_params(axis="y", labelsize=14)
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)

    if show_legend:
        legend_handles = [
            Line2D(
                [],
                [],
                marker="*",
                linestyle="None",
                markerfacecolor=PAPER_COLORS[1],
                markeredgecolor="k",
                markersize=12,
                label="True $\\theta$",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor=design_color,
                markeredgecolor="k",
                markersize=8,
                label="Design",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            fontsize=20,
            borderaxespad=0.2,
            borderpad=0.2,
            handletextpad=0.005,
            labelspacing=0.2,
        )

    return scatter


def plot_bed_spce_comparison(
    save_path: Path,
    *,
    method_stats: dict[str, dict[str, float]],
    uncertainty: UncertaintyMode,
    task_label: str,
) -> None:
    """Bar chart comparing sPCE across methods with error bars."""

    import matplotlib.pyplot as plt

    if not method_stats:
        return

    method_names: list[str] = []
    means: list[float] = []
    errors: list[float] = []

    for method_name, stats in method_stats.items():
        method_names.append(method_name)
        means.append(float(stats["mean_spce"]))
        if uncertainty == "ci95":
            errors.append(float(stats["ci95_spce"]))
        elif uncertainty == "std":
            errors.append(float(stats["std_spce"]))
        else:
            errors.append(float(stats["sem_spce"]))

    fig, ax = plt.subplots(figsize=(max(4.0, 1.8 * len(method_names)), 5.0))
    x = list(range(len(method_names)))
    palette = [PAPER_COLORS[idx % len(PAPER_COLORS)] for idx in range(len(x))]
    bars = ax.bar(
        x,
        means,
        yerr=errors,
        capsize=6,
        color=palette,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=11)
    ax.set_ylabel("sPCE (higher = better)", fontsize=11)

    spread_label = {"ci95": "95% CI", "std": "±1 std", "sem": "±1 sem"}[uncertainty]
    ax.set_title(f"BED Evaluation — {task_label} ({spread_label})", fontsize=13)
    ax.grid(axis="y", alpha=0.25)

    for bar, mean, error in zip(bars, means, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + error + 0.01 * abs(mean),
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "plot_bed_spce_comparison",
    "plot_location_finding_trajectories",
    "plot_location_trajectories_comparison_2d",
    "plot_location_trajectory_2d",
]
