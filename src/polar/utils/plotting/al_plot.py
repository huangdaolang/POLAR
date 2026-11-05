"""AL-specific plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from polar.utils.plotting.base import apply_style


def plot_al_metric_grid(
    *,
    save_path: Path,
    aggregated_by_method: dict[str, list[dict[str, Any]]],
    uncertainty_mode: str,
) -> None:
    """Plot a grid of AL evaluation metrics across acquisition rounds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    if not aggregated_by_method:
        return

    apply_style(use_tex=False, base_font_size=11.0)

    metric_specs = [
        ("target_rmse", "Target RMSE"),
        ("weighted_target_rmse", "Weighted Target RMSE"),
        ("target_log_prob", "Target Log Prob"),
        ("weighted_target_log_prob", "Weighted Target Log Prob"),
        ("selected_query_y", "Selected Query y"),
        ("selected_query_raw_y", "Selected Query raw y"),
    ]
    available_specs = [
        (metric_name, label)
        for metric_name, label in metric_specs
        if any(
            aggregated_rounds and f"mean_{metric_name}" in aggregated_rounds[0]
            for aggregated_rounds in aggregated_by_method.values()
        )
    ]
    if not available_specs:
        return

    uncertainty_key = {
        "ci95": "ci95",
        "std": "std",
        "sem": "sem",
    }.get(str(uncertainty_mode).strip().lower())
    if uncertainty_key is None:
        raise ValueError("eval.plot_uncertainty must be one of: ci95, std, sem.")

    n_panels = len(available_specs)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.0 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    for idx, (metric_name, label) in enumerate(available_specs):
        ax = axes[idx // n_cols][idx % n_cols]
        for method_name, aggregated_rounds in aggregated_by_method.items():
            if not aggregated_rounds:
                continue
            mean_key = f"mean_{metric_name}"
            band_key = f"{uncertainty_key}_{metric_name}"
            if mean_key not in aggregated_rounds[0]:
                continue

            rounds = [int(row["round"]) + 1 for row in aggregated_rounds]
            means = [float(row[mean_key]) for row in aggregated_rounds]
            bands = [float(row.get(band_key, 0.0)) for row in aggregated_rounds]
            ax.plot(rounds, means, linewidth=2.0, label=method_name)
            lower = [mean - band for mean, band in zip(means, bands, strict=False)]
            upper = [mean + band for mean, band in zip(means, bands, strict=False)]
            ax.fill_between(rounds, lower, upper, alpha=0.18)

        ax.set_title(label)
        ax.set_xlabel("Acquisition round")
        ax.grid(alpha=0.25)

    for idx in range(n_panels, n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
        )
    fig.suptitle("AL Evaluation", y=1.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_surface_rounds_grid(
    *,
    save_path: Path,
    round_snapshots: list[dict[str, Any]],
    method_name: str,
    dataset_id: str,
    episode_seed: str,
    grid_shape: tuple[int, int],
) -> None:
    """Plot 2D benchmark rounds as true-function contours plus predictive contours."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    if not round_snapshots:
        return

    apply_style(use_tex=False, base_font_size=11.0)

    n_cols = 3
    n_rounds = len(round_snapshots)
    n_rows = (n_rounds + n_cols - 1) // n_cols
    cmap = plt.get_cmap("viridis")
    max_acquired = max(
        int(snapshot["acquired_raw_x"].shape[0]) for snapshot in round_snapshots
    )
    norm = Normalize(vmin=1, vmax=max(max_acquired, 2))
    n0, n1 = grid_shape

    first = round_snapshots[0]
    raw_plot_x = torch.as_tensor(first["raw_plot_x"], dtype=torch.float32).detach().cpu()
    raw_plot_y_true = torch.as_tensor(first["raw_plot_y_true"], dtype=torch.float32).detach().cpu()
    x0 = raw_plot_x[:, 0].reshape(n0, n1)
    x1 = raw_plot_x[:, 1].reshape(n0, n1)
    y_grid = raw_plot_y_true.reshape(n0, n1)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 4.0 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    contour_ref = None

    for idx, snapshot in enumerate(round_snapshots):
        ax = axes[idx // n_cols][idx % n_cols]
        raw_context_x = snapshot["raw_context_x"].detach().cpu()
        raw_target_x = snapshot["raw_target_x"].detach().cpu()
        acquired_raw_x = snapshot["acquired_raw_x"].detach().cpu()
        pred_mean = snapshot["pred_mean_raw"].detach().cpu().reshape(n0, n1)

        contour_ref = ax.contourf(
            x0.numpy(),
            x1.numpy(),
            y_grid.numpy(),
            levels=24,
            cmap="Greys",
            alpha=0.85,
        )
        ax.contour(
            x0.numpy(),
            x1.numpy(),
            pred_mean.numpy(),
            levels=12,
            colors="tab:blue",
            linewidths=1.0,
            alpha=0.95,
        )
        ax.scatter(
            raw_context_x[:, 0].numpy(),
            raw_context_x[:, 1].numpy(),
            color="black",
            s=24,
            zorder=5,
        )
        ax.scatter(
            raw_target_x[:, 0].numpy(),
            raw_target_x[:, 1].numpy(),
            color="#009E73",
            marker="x",
            s=40,
            linewidths=1.2,
            zorder=6,
        )
        if acquired_raw_x.numel() > 0:
            colors = torch.arange(1, int(acquired_raw_x.shape[0]) + 1)
            ax.scatter(
                acquired_raw_x[:, 0].numpy(),
                acquired_raw_x[:, 1].numpy(),
                c=colors.numpy(),
                cmap=cmap,
                norm=norm,
                s=38,
                edgecolors="white",
                linewidths=0.5,
                zorder=7,
            )
            ax.scatter(
                [float(acquired_raw_x[-1, 0].item())],
                [float(acquired_raw_x[-1, 1].item())],
                color="#7A0019",
                s=64,
                marker="o",
                edgecolors="white",
                linewidths=0.7,
                zorder=8,
            )
        ax.set_title(
            (
                f"Round {int(snapshot['round']) + 1} | "
                f"RMSE={float(snapshot['target_rmse']):.4f} | "
                f"wRMSE={float(snapshot['weighted_target_rmse']):.4f}"
            ),
            fontsize=9,
        )
        ax.grid(alpha=0.15)

    for idx in range(n_rounds, n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    if contour_ref is not None:
        fig.colorbar(contour_ref, ax=axes.ravel().tolist(), pad=0.02, label="True function value")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        pad=0.09,
        label="Acquisition round",
    )

    handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=5, label="Context"),
        Line2D([0], [0], marker="x", color="#009E73", linestyle="None", markersize=6, label="Target"),
        Line2D([0], [0], color="tab:blue", linewidth=1.3, label="Pred contours"),
        Line2D([0], [0], marker="o", color="#7A0019", linestyle="None", markersize=6, label="Latest query"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
    )
    fig.suptitle(f"{dataset_id} | {method_name} | seed={episode_seed}", y=1.05)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _extract_1d_probe_snapshot_tensors(snapshot: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Normalize 1D probe snapshot payloads into a common tensor bundle."""
    if "pred_mean_raw" in snapshot and "raw_plot_x" in snapshot and "raw_plot_y_true" in snapshot:
        return {
            "x_line": snapshot["raw_plot_x"].squeeze(-1).detach().cpu(),
            "y_true": snapshot["raw_plot_y_true"].detach().cpu(),
            "pred_mean": snapshot["pred_mean_raw"].detach().cpu(),
            "pred_std": snapshot["pred_std_raw"].detach().cpu(),
            "context_x": snapshot["raw_context_x"].squeeze(-1).detach().cpu(),
            "context_y": snapshot["raw_context_y"].detach().cpu(),
            "target_x": snapshot["raw_target_x"].squeeze(-1).detach().cpu(),
            "target_y": snapshot["raw_target_y"].detach().cpu(),
            "acquired_x": snapshot["acquired_raw_x"].squeeze(-1).detach().cpu(),
            "acquired_y": snapshot["acquired_raw_y"].detach().cpu(),
        }

    return {
        "x_line": snapshot["context_x"].squeeze(-1).detach().cpu(),
        "y_true": snapshot["true_y"].detach().cpu(),
        "pred_mean": snapshot["pred_mean"].detach().cpu(),
        "pred_std": snapshot["pred_std"].detach().cpu(),
        "context_x": snapshot["context_x"].squeeze(-1).detach().cpu(),
        "context_y": snapshot["context_y"].detach().cpu(),
        "target_x": snapshot["target_x"].squeeze(-1).detach().cpu(),
        "target_y": snapshot["target_y"].detach().cpu(),
        "acquired_x": torch.empty((0,), dtype=torch.float32),
        "acquired_y": torch.empty((0,), dtype=torch.float32),
    }


def _extract_1d_probe_score_tensors(snapshot: dict[str, Any]) -> dict[str, torch.Tensor | float | str | None]:
    """Normalize 1D probe score payloads into a common tensor bundle."""
    raw_query_x = snapshot.get("score_curve_raw_x")
    score_values = snapshot.get("score_curve_values")
    if raw_query_x is None:
        raw_query_x = snapshot.get("score_query_raw_x")
    if score_values is None:
        score_values = snapshot.get("score_values")
    if raw_query_x is None:
        raw_query_x = snapshot.get("score_query_x")
    if raw_query_x is None:
        raise KeyError("Snapshot does not contain score_query_raw_x/score_query_x.")
    if score_values is None:
        raise KeyError("Snapshot does not contain score_values/score_curve_values.")

    selected_query_raw_x = snapshot.get("selected_query_raw_x")
    selected_query_x_scalar: float | None = None
    if isinstance(selected_query_raw_x, torch.Tensor) and selected_query_raw_x.numel() > 0:
        selected_query_x_scalar = float(selected_query_raw_x.view(-1)[0].item())

    return {
        "query_x": torch.as_tensor(raw_query_x, dtype=torch.float32).squeeze(-1).detach().cpu(),
        "score_values": torch.as_tensor(
            score_values,
            dtype=torch.float32,
        ).detach().cpu(),
        "selected_query_x": selected_query_x_scalar,
        "selected_score_value": (
            float(snapshot["selected_score_value"])
            if snapshot.get("selected_score_value") is not None
            else None
        ),
        "score_label": str(snapshot.get("score_label", "Score")),
        "score_kind": (
            str(snapshot["score_kind"])
            if snapshot.get("score_kind") is not None
            else None
        ),
    }


def draw_policy_probe_snapshot_axis(
    ax: Any,
    *,
    snapshot: dict[str, Any],
    ci_std: float,
    cmap: Any | None = None,
    norm: Any | None = None,
    title: str | None = None,
    style: dict[str, Any] | None = None,
    show_target: bool = True,
    combine_context_and_acquired: bool = False,
    latest_query_style: str = "marker",
) -> None:
    """Draw one 1D AL probe snapshot onto an existing matplotlib axis."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    tensors = _extract_1d_probe_snapshot_tensors(snapshot)
    style_cfg = {
        "true_color": "#D55E00",
        "pred_color": "tab:blue",
        "band_color": "lightskyblue",
        "band_alpha": 0.35,
        "true_linewidth": 1.8,
        "pred_linewidth": 1.6,
        "context_color": "black",
        "context_size": 12.0,
        "target_color": "#009E73",
        "target_size": 18.0,
        "target_marker": "x",
        "target_linewidths": 1.1,
        "queried_size": 24.0,
        "queried_edgecolor": "white",
        "queried_linewidths": 0.5,
        "queried_color": "black",
        "latest_edgecolor": "#CC79A7",
        "latest_linewidths": 1.4,
        "latest_size": 72.0,
        "latest_facecolors": "none",
        "latest_color": "#CC3311",
        "latest_alpha": 0.95,
        "grid_alpha": 0.2,
    }
    if style is not None:
        style_cfg.update(style)

    acquired_x = tensors["acquired_x"]
    acquired_y = tensors["acquired_y"]
    if cmap is None:
        cmap = plt.get_cmap("viridis")
    if norm is None:
        norm = Normalize(vmin=1, vmax=max(int(acquired_x.shape[0]), 2))

    x_line = tensors["x_line"]
    y_true = tensors["y_true"]
    pred_mean = tensors["pred_mean"]
    pred_std = tensors["pred_std"]
    context_x = tensors["context_x"]
    context_y = tensors["context_y"]
    target_x = tensors["target_x"]
    target_y = tensors["target_y"]

    ax.plot(
        x_line.numpy(),
        y_true.numpy(),
        color=str(style_cfg["true_color"]),
        linewidth=float(style_cfg["true_linewidth"]),
        label="True",
    )
    ax.plot(
        x_line.numpy(),
        pred_mean.numpy(),
        color=str(style_cfg["pred_color"]),
        linewidth=float(style_cfg["pred_linewidth"]),
        label="Pred mean",
    )
    lower = pred_mean - float(ci_std) * pred_std
    upper = pred_mean + float(ci_std) * pred_std
    ax.fill_between(
        x_line.numpy(),
        lower.numpy(),
        upper.numpy(),
        color=str(style_cfg["band_color"]),
        alpha=float(style_cfg["band_alpha"]),
        label=f"Pred +/- {ci_std:.1f} std",
    )
    if combine_context_and_acquired:
        if acquired_x.numel() > 0:
            queried_x = torch.cat([context_x, acquired_x], dim=0)
            queried_y = torch.cat([context_y, acquired_y], dim=0)
        else:
            queried_x = context_x
            queried_y = context_y
        ax.scatter(
            queried_x.numpy(),
            queried_y.numpy(),
            color=str(style_cfg["queried_color"]),
            s=float(style_cfg["queried_size"]),
            edgecolors=str(style_cfg["queried_edgecolor"]),
            linewidths=float(style_cfg["queried_linewidths"]),
            zorder=4,
            label="Queried",
        )
    else:
        ax.scatter(
            context_x.numpy(),
            context_y.numpy(),
            color=str(style_cfg["context_color"]),
            s=float(style_cfg["context_size"]),
            zorder=3,
            label="Context",
        )
        if acquired_x.numel() > 0:
            colors = torch.arange(1, int(acquired_x.shape[0]) + 1)
            ax.scatter(
                acquired_x.numpy(),
                acquired_y.numpy(),
                c=colors.numpy(),
                cmap=cmap,
                norm=norm,
                s=float(style_cfg["queried_size"]),
                edgecolors=str(style_cfg["queried_edgecolor"]),
                linewidths=float(style_cfg["queried_linewidths"]),
                zorder=4,
                label="Queried",
            )

    if show_target:
        ax.scatter(
            target_x.numpy(),
            target_y.numpy(),
            color=str(style_cfg["target_color"]),
            s=float(style_cfg["target_size"]),
            marker=str(style_cfg["target_marker"]),
            linewidths=float(style_cfg["target_linewidths"]),
            zorder=3,
            label="Target",
        )

    if acquired_x.numel() > 0:
        latest_x = float(acquired_x[-1].item())
        latest_y = float(acquired_y[-1].item())
        if latest_query_style == "vline":
            ax.axvline(
                latest_x,
                color=str(style_cfg["latest_color"]),
                linewidth=float(style_cfg["latest_linewidths"]),
                alpha=float(style_cfg["latest_alpha"]),
                zorder=5,
                label="Latest query",
            )
        elif latest_query_style == "marker":
            ax.scatter(
                [latest_x],
                [latest_y],
                facecolors=str(style_cfg["latest_facecolors"]),
                edgecolors=str(style_cfg["latest_edgecolor"]),
                linewidths=float(style_cfg["latest_linewidths"]),
                s=float(style_cfg["latest_size"]),
                zorder=5,
                label="Latest query",
            )
        else:
            raise ValueError(
                "latest_query_style must be one of: 'marker', 'vline'."
            )

    if title is not None:
        ax.set_title(title, fontsize=20)
    ax.grid(alpha=float(style_cfg["grid_alpha"]))


def draw_policy_probe_score_axis(
    ax: Any,
    *,
    snapshot: dict[str, Any],
    title: str | None = None,
    style: dict[str, Any] | None = None,
) -> None:
    """Draw the 1D acquisition/policy-score curve for a probe snapshot."""
    tensors = _extract_1d_probe_score_tensors(snapshot)
    style_cfg = {
        "score_color": "#0072B2",
        "score_linewidth": 1.8,
        "score_alpha": 0.95,
        "score_fill_alpha": 0.12,
        "latest_color": "#CC3311",
        "latest_linewidth": 1.4,
        "latest_alpha": 0.95,
        "selected_marker_size": 24.0,
        "selected_marker_edgecolor": "white",
        "selected_marker_linewidth": 0.4,
        "grid_alpha": 0.2,
    }
    if style is not None:
        style_cfg.update(style)

    query_x = tensors["query_x"]
    score_values = tensors["score_values"]
    if query_x.ndim != 1:
        raise ValueError("draw_policy_probe_score_axis expects 1D query_x.")
    if score_values.ndim != 1:
        raise ValueError("draw_policy_probe_score_axis expects 1D score_values.")
    if query_x.shape[0] != score_values.shape[0]:
        raise ValueError("query_x and score_values must have the same length.")

    finite_mask = torch.isfinite(query_x) & torch.isfinite(score_values)
    if not bool(finite_mask.any()):
        ax.text(
            0.5,
            0.5,
            "No finite scores",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_ylabel(str(tensors["score_label"]))
        ax.grid(alpha=float(style_cfg["grid_alpha"]))
        return

    query_x = query_x[finite_mask]
    score_values = score_values[finite_mask]
    order = torch.argsort(query_x)
    query_x_sorted = query_x[order]
    score_sorted = score_values[order]

    ax.plot(
        query_x_sorted.numpy(),
        score_sorted.numpy(),
        color=str(style_cfg["score_color"]),
        linewidth=float(style_cfg["score_linewidth"]),
        alpha=float(style_cfg["score_alpha"]),
        label=str(tensors["score_label"]),
    )
    min_score = float(score_sorted.min().item())
    if min_score >= 0.0:
        baseline = torch.zeros_like(score_sorted)
        ax.fill_between(
            query_x_sorted.numpy(),
            baseline.numpy(),
            score_sorted.numpy(),
            color=str(style_cfg["score_color"]),
            alpha=float(style_cfg["score_fill_alpha"]),
        )
        ax.set_ylim(bottom=0.0)

    selected_query_x = tensors["selected_query_x"]
    selected_score_value = tensors["selected_score_value"]
    if selected_query_x is not None:
        ax.axvline(
            float(selected_query_x),
            color=str(style_cfg["latest_color"]),
            linewidth=float(style_cfg["latest_linewidth"]),
            alpha=float(style_cfg["latest_alpha"]),
            zorder=4,
            label="Latest query",
        )
    if (
        selected_query_x is not None
        and selected_score_value is not None
        and torch.isfinite(torch.tensor(float(selected_score_value)))
    ):
        ax.scatter(
            [float(selected_query_x)],
            [float(selected_score_value)],
            color=str(style_cfg["latest_color"]),
            s=float(style_cfg["selected_marker_size"]),
            edgecolors=str(style_cfg["selected_marker_edgecolor"]),
            linewidths=float(style_cfg["selected_marker_linewidth"]),
            zorder=5,
        )

    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(str(tensors["score_label"]))
    ax.grid(alpha=float(style_cfg["grid_alpha"]))


def plot_policy_probe_rounds_grid(
    *,
    save_path: Path,
    round_snapshots: list[dict[str, Any]],
    method_name: str,
    ci_std: float,
) -> None:
    """Plot a grid of probe-round snapshots for 1D GP episodes."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    if not round_snapshots:
        return

    apply_style(use_tex=False, base_font_size=11.0)

    n_cols = 4
    n_rounds = len(round_snapshots)
    n_rows = (n_rounds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    max_acquired = max(
        int(snapshot.get("acquired_raw_x", torch.empty((0, 1))).shape[0])
        for snapshot in round_snapshots
    )
    norm = Normalize(vmin=1, vmax=max(max_acquired, 2))
    cmap = plt.get_cmap("viridis")

    for idx, snapshot in enumerate(round_snapshots):
        ax = axes[idx // n_cols][idx % n_cols]
        draw_policy_probe_snapshot_axis(
            ax,
            snapshot=snapshot,
            ci_std=ci_std,
            cmap=cmap,
            norm=norm,
            title=(
                f"Round {int(snapshot['round']) + 1} | "
                f"RMSE={float(snapshot['target_rmse']):.4f} | "
                f"wRMSE={float(snapshot['weighted_target_rmse']):.4f}"
            ),
        )

    for idx in range(n_rounds, n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    handles, labels = axes[0][0].get_legend_handles_labels()
    unique_handles: list[Any] = []
    unique_labels: list[str] = []
    for handle, label in zip(handles, labels, strict=False):
        if label in unique_labels:
            continue
        unique_handles.append(handle)
        unique_labels.append(label)
    fig.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        ncol=min(6, len(unique_labels)),
        fontsize=8,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(f"Loss-Driven AL Probe | {method_name}", y=1.04)
    fig.supxlabel("x")
    fig.supylabel("y")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "draw_policy_probe_snapshot_axis",
    "draw_policy_probe_score_axis",
    "plot_al_metric_grid",
    "plot_benchmark_surface_rounds_grid",
    "plot_policy_probe_rounds_grid",
]
