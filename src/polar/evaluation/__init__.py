"""Evaluation helpers."""

from polar.evaluation.al import (
    aggregate_al_eval_metrics,
    aggregate_al_eval_metrics_by_dataset,
    evaluate_policy_episodes,
    evaluate_surrogate_episodes,
    episode_model_std_to_raw,
    episode_model_y_to_raw,
    episode_raw_x_to_model,
    rollout_policy_probe_episode,
    rollout_surrogate_probe_episode,
    snapshot_round_indices,
    summarize_probe_metrics,
)
from polar.evaluation.bed import (
    EpisodeResult,
    aggregate_eig_bounds,
    compute_bed_eig_bounds_for_trajectory,
    evaluate_bed_policy,
    load_location_policy_checkpoint,
    rollout_location_trajectory_from_checkpoint,
    rollout_bed_policy,
    rollout_bed_random,
)
from polar.utils.policy_loading import LoadedPolicy, load_policy_checkpoint
from polar.utils.plotting import plot_policy_probe_rounds_grid

__all__ = [
    "EpisodeResult",
    "LoadedPolicy",
    "aggregate_eig_bounds",
    "aggregate_al_eval_metrics",
    "aggregate_al_eval_metrics_by_dataset",
    "compute_bed_eig_bounds_for_trajectory",
    "evaluate_surrogate_episodes",
    "episode_model_std_to_raw",
    "episode_model_y_to_raw",
    "episode_raw_x_to_model",
    "evaluate_policy_episodes",
    "evaluate_bed_policy",
    "load_policy_checkpoint",
    "load_location_policy_checkpoint",
    "plot_policy_probe_rounds_grid",
    "rollout_location_trajectory_from_checkpoint",
    "rollout_policy_probe_episode",
    "rollout_surrogate_probe_episode",
    "rollout_bed_policy",
    "rollout_bed_random",
    "snapshot_round_indices",
    "summarize_probe_metrics",
]
