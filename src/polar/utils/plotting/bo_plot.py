"""BO-specific plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from polar.utils.plotting.base import UncertaintyMode

MetricName = Literal["regret", "normalized_regret"]


def _resolve_spread_key(metric: MetricName, uncertainty: UncertaintyMode) -> tuple[str, str]:
    suffix = "regret" if metric == "regret" else "normalized_regret"
    if uncertainty == "ci95":
        return f"ci95_{suffix}", "95% CI"
    if uncertainty == "std":
        return f"std_{suffix}", "±1 std"
    if uncertainty == "sem":
        return f"sem_{suffix}", "±1 sem"
    raise ValueError("eval.plot_uncertainty must be one of: ci95, std, sem.")


def plot_bo_comparison_curves(
    *,
    save_path: Path,
    aggregate_by_method: dict[str, list[dict[str, Any]]],
    metric: MetricName,
    uncertainty: UncertaintyMode,
    task_label: str,
) -> None:
    """Plot BO regret or normalized-regret comparison curves."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogFormatterMathtext, LogLocator
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    if not aggregate_by_method:
        return

    metric_key = "mean_regret" if metric == "regret" else "mean_normalized_regret"
    spread_key, spread_label = _resolve_spread_key(metric, uncertainty)
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    marker_every = 5

    positive_values = [
        float(item[key])
        for aggregate in aggregate_by_method.values()
        for item in aggregate
        for key in (metric_key, spread_key)
        if float(item[key]) > 0.0
    ]
    if positive_values:
        smallest_positive = min(positive_values)
        log_floor = max(smallest_positive * 0.1, 1e-12)
    else:
        log_floor = 1e-12

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for method_idx, (method_name, aggregate) in enumerate(aggregate_by_method.items()):
        if not aggregate:
            continue
        rounds = [int(item["round"]) for item in aggregate]
        means = [float(item[metric_key]) for item in aggregate]
        spreads = [float(item[spread_key]) for item in aggregate]
        lower = [max(log_floor, m - s) for m, s in zip(means, spreads)]
        upper = [max(log_floor, m + s) for m, s in zip(means, spreads)]
        means = [max(log_floor, m) for m in means]

        marker = marker_cycle[method_idx % len(marker_cycle)]
        markevery = list(range(0, len(rounds), marker_every)) if len(rounds) > marker_every else None
        (line,) = ax.step(
            rounds,
            means,
            where="post",
            linewidth=2.0,
            marker=marker,
            markevery=markevery,
            markersize=7,
            label=method_name,
        )
        ax.fill_between(rounds, lower, upper, color=line.get_color(), alpha=0.14, step="post")

    y_label = "Regret" if metric == "regret" else "Normalized Regret"
    ax.set_title(f"BO Evaluation ({task_label})")
    ax.set_xlabel("Step (acquisition round)")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.set_yscale("log")
    ax.set_ylim(bottom=log_floor)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        cols = 2 if len(handles) >= 8 else 1
        ax.legend(title=spread_label, ncol=cols, fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "MetricName",
    "plot_bo_comparison_curves",
]
