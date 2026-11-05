"""Shared plotting foundations."""

from __future__ import annotations

from typing import Literal

UncertaintyMode = Literal["ci95", "std", "sem"]

#
# Fixed paper palette:
# - Primarily based on the Okabe-Ito / Wong scientific-figure palette
#   that is widely recommended for accessible research plots, including
#   Nature Methods discussions of colorblind-safe figures.
# - Gray replaces black for a softer default on white backgrounds.
# - A deep indigo from Paul Tol's scientific palettes rounds this out to 8 colors.
#
PAPER_COLORS: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # red / vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#7F7F7F",  # gray
    "#332288",  # deep indigo
)


def apply_style(
    *,
    use_tex: bool = True,
    base_font_size: float = 16.0,
) -> None:
    """Apply the default notebook plotting style used for final figures."""

    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(
        {
            "font.family": "times",
            "font.size": base_font_size,
            "lines.linewidth": 2,
            "lines.antialiased": True,
            "axes.facecolor": "#fdfdfd",
            "axes.edgecolor": "#777777",
            "axes.linewidth": 1,
            "axes.titlesize": base_font_size + 2.0,
            "axes.labelsize": base_font_size,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=PAPER_COLORS),
            "xtick.major.size": 0,
            "xtick.minor.size": 0,
            "xtick.major.pad": 6,
            "xtick.minor.pad": 6,
            "xtick.color": "#333333",
            "xtick.labelsize": base_font_size - 1.0,
            "xtick.direction": "in",
            "ytick.major.size": 0,
            "ytick.minor.size": 0,
            "ytick.major.pad": 6,
            "ytick.minor.pad": 6,
            "ytick.color": "#333333",
            "ytick.labelsize": base_font_size - 1.0,
            "ytick.direction": "in",
            "axes.grid": False,
            "grid.alpha": 0.3,
            "grid.linewidth": 1,
            "legend.fancybox": True,
            "legend.fontsize": base_font_size - 2.0,
            "figure.figsize": (3.0, 3.0),
            "figure.facecolor": "1.0",
            "figure.edgecolor": "0.5",
            "hatch.linewidth": 0.1,
            "text.usetex": use_tex,
        }
    )


__all__ = [
    "PAPER_COLORS",
    "UncertaintyMode",
    "apply_style",
]
