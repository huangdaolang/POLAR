#!/usr/bin/env python
"""Visualize sampled 1D GP prior episodes for foundation AL sanity checks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from polar.tasks import create_al_task
from polar.utils.plotting import apply_style
from polar.utils.seeding import set_global_seed


def _resolve_save_dir(cfg: DictConfig) -> Path:
    run_tag = str(cfg.output.run_tag) if cfg.output.run_tag else datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(cfg.paths.results_root) / str(cfg.output.experiment_name) / run_tag


def _build_figure(
    *,
    episodes: list,
    show_raw: bool,
    show_standardized: bool,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = int(show_raw) + int(show_standardized)
    if cols == 0:
        raise ValueError("At least one of inspect.show_raw/show_standardized must be true.")

    apply_style(use_tex=False, base_font_size=11.0)
    fig, axes = plt.subplots(
        len(episodes),
        cols,
        figsize=(5.0 * cols, 3.2 * len(episodes)),
        squeeze=False,
    )

    for row, episode in enumerate(episodes):
        col = 0
        panels: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        if show_raw:
            panels.append(
                (
                    "Raw",
                    episode.raw_plot_x,
                    episode.raw_plot_y_true,
                    episode.raw_context_x,
                    episode.raw_context_y,
                    episode.raw_query_x,
                    episode.raw_query_y,
                    episode.raw_target_x,
                    episode.raw_target_y,
                )
            )
        if show_standardized:
            panels.append(
                (
                    "Standardized",
                    episode.plot_x,
                    episode.plot_y_true,
                    episode.context_x,
                    episode.context_y,
                    episode.query_x,
                    episode.query_y,
                    episode.target_x,
                    episode.target_y,
                )
            )

        for title_prefix, plot_x, plot_y, ctx_x, ctx_y, qry_x, qry_y, tgt_x, tgt_y in panels:
            ax = axes[row][col]
            col += 1

            ax.plot(plot_x.squeeze(-1).numpy(), plot_y.numpy(), color="#0072B2", linewidth=1.6)
            ax.scatter(ctx_x.squeeze(-1).numpy(), ctx_y.numpy(), color="#111111", s=20, label="Context", zorder=3)
            ax.scatter(qry_x.squeeze(-1).numpy(), qry_y.numpy(), color="#D55E00", s=14, alpha=0.75, label="Query", zorder=2)
            ax.scatter(tgt_x.squeeze(-1).numpy(), tgt_y.numpy(), color="#009E73", s=14, alpha=0.75, label="Target", zorder=2)

            ls_str = ", ".join(f"{float(val):.2f}" for val in episode.lengthscales)
            scale_str = float(episode.output_scale.view(-1)[0].item())
            jitter_str = (
                f"{float(episode.final_jitter):.1e}"
                if episode.final_jitter is not None
                else "n/a"
            )
            ax.set_title(
                f"{title_prefix} | {episode.kernel_type} | ls=[{ls_str}] | scale={scale_str:.2f} | jitter={jitter_str}",
                fontsize=10,
            )
            ax.grid(alpha=0.2)
            if row == 0 and col == 1:
                ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


@hydra.main(version_base="1.3", config_path="../config", config_name="exp/inspect_gp_prior")
def main(cfg: DictConfig) -> None:
    print("==== Inspect GP Prior ====")
    print(OmegaConf.to_yaml(cfg))

    dim_x = cfg.task.get("dim_x")
    dim_x_min = cfg.task.get("dim_x_min")
    dim_x_max = cfg.task.get("dim_x_max")
    if (
        (dim_x is not None and int(dim_x) != 1)
        or (dim_x_min is not None and int(dim_x_min) != 1)
        or (dim_x_max is not None and int(dim_x_max) != 1)
    ):
        raise ValueError("inspect_gp_prior.py currently supports only dim=1.")

    set_global_seed(int(cfg.seed))

    task = create_al_task(cfg.task, seed=int(cfg.seed))
    episodes = task.sample_train_batch(
        batch_size=int(cfg.inspect.num_samples),
        n_context=int(cfg.task.n_context_init),
        n_query=int(cfg.task.n_query),
        n_target=int(cfg.task.n_target),
    )

    save_dir = _resolve_save_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig = _build_figure(
        episodes=episodes,
        show_raw=bool(cfg.inspect.show_raw),
        show_standardized=bool(cfg.inspect.show_standardized),
    )
    figure_path = save_dir / "gp_prior_samples.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    print(f"figure_path: {figure_path}")

    metadata = []
    for idx, episode in enumerate(episodes):
        metadata.append(
            {
                "sample_idx": idx,
                "kernel_type": episode.kernel_type,
                "lengthscales": [float(v) for v in episode.lengthscales.tolist()],
                "output_scale": float(episode.output_scale.view(-1)[0].item()),
                "x_standardized": bool(episode.x_standardized),
                "y_standardized": bool(episode.y_standardized),
                "final_jitter": float(episode.final_jitter) if episode.final_jitter is not None else None,
                "x_center": episode.x_center.tolist() if episode.x_center is not None else None,
                "x_scale": episode.x_scale.tolist() if episode.x_scale is not None else None,
                "y_mean": float(episode.y_mean.item()) if episode.y_mean is not None else None,
                "y_std": float(episode.y_std.item()) if episode.y_std is not None else None,
            }
        )
    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
