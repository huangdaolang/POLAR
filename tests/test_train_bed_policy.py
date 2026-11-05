from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf


_TRAIN_BED_POLICY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_bed_policy.py"
_TRAIN_BED_POLICY_SPEC = importlib.util.spec_from_file_location(
    "train_bed_policy_script",
    _TRAIN_BED_POLICY_PATH,
)
if _TRAIN_BED_POLICY_SPEC is None or _TRAIN_BED_POLICY_SPEC.loader is None:
    raise RuntimeError(f"Failed to import train_bed_policy.py from {_TRAIN_BED_POLICY_PATH}.")
_TRAIN_BED_POLICY = importlib.util.module_from_spec(_TRAIN_BED_POLICY_SPEC)
sys.modules[_TRAIN_BED_POLICY_SPEC.name] = _TRAIN_BED_POLICY
_TRAIN_BED_POLICY_SPEC.loader.exec_module(_TRAIN_BED_POLICY)

_compute_likelihood_eval_schedule = _TRAIN_BED_POLICY._compute_likelihood_eval_schedule
_parse_likelihood_eval_milestones = _TRAIN_BED_POLICY._parse_likelihood_eval_milestones
_resolve_reached_likelihood_eval_milestones = (
    _TRAIN_BED_POLICY._resolve_reached_likelihood_eval_milestones
)
_save_heavy_eval_results = _TRAIN_BED_POLICY._save_heavy_eval_results


def test_parse_likelihood_eval_milestones_returns_empty_when_disabled() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "heavy_eval": {
                    "enabled": False,
                    "likelihood_eval_milestones": [1000, 10000],
                }
            }
        }
    )

    assert _parse_likelihood_eval_milestones(cfg) == []


def test_parse_likelihood_eval_milestones_sorts_and_deduplicates() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "heavy_eval": {
                    "enabled": True,
                    "likelihood_eval_milestones": [10000, 1000, 10000],
                }
            }
        }
    )

    assert _parse_likelihood_eval_milestones(cfg) == [1000, 10000]


def test_compute_likelihood_eval_schedule_matches_information_budget() -> None:
    schedule = _compute_likelihood_eval_schedule(
        batch_size=50,
        num_epochs=1000,
        num_acquisitions=30,
        n_query=1000,
        n_target=50,
        finetune_epochs=1000,
        nll_weight=1.0,
        likelihood_eval_milestones=[1000, 4000, 8000, 4_000_001],
    )

    assert schedule["rollouts_per_epoch"] == 50
    assert schedule["total_training_rollouts"] == 50000
    assert schedule["acquisition_evals_per_epoch"] == 1500
    assert schedule["target_evals_per_epoch"] == 2500
    assert schedule["total_training_likelihood_evals"] == 4_000_000
    assert schedule["non_exact_milestones"] == [1000, 4_000_001]
    assert schedule["unreachable_milestones"] == [4_000_001]


def test_compute_likelihood_eval_schedule_only_counts_targets_while_active() -> None:
    schedule = _compute_likelihood_eval_schedule(
        batch_size=4,
        num_epochs=3,
        num_acquisitions=2,
        n_query=10,
        n_target=5,
        finetune_epochs=1,
        nll_weight=1.0,
        likelihood_eval_milestones=[28, 36, 44],
    )

    assert schedule["acquisition_evals_per_epoch"] == 8
    assert schedule["target_evals_per_epoch"] == 20
    assert schedule["total_training_likelihood_evals"] == 44
    assert schedule["non_exact_milestones"] == []
    assert schedule["unreachable_milestones"] == []


def test_resolve_reached_likelihood_eval_milestones_uses_first_reached_epoch() -> None:
    reached = _resolve_reached_likelihood_eval_milestones(
        likelihood_eval_milestones=[1000, 10000],
        completed_milestones=set(),
        current_likelihood_evals=1008,
    )

    assert reached == [1000]


def test_save_heavy_eval_results_writes_json_and_csv(tmp_path: Path) -> None:
    json_path = tmp_path / "heavy_eval_metrics.json"
    csv_path = tmp_path / "heavy_eval_metrics.csv"
    results = [
        {
            "epoch": 20,
            "requested_likelihood_evals": 4000,
            "total_likelihood_evals": 4000,
            "likelihood_evals_this_epoch": 4000,
            "acquisition_evals_this_epoch": 1500,
            "target_evals_this_epoch": 2500,
            "target_evals_per_episode": 50,
            "total_rollouts": 50,
            "rollouts_per_epoch": 50,
            "checkpoint_path": "/tmp/checkpoint_likelihood_evals_4000.pt",
            "eval_spce_mean": 2.5,
            "eval_spce_stderr": 0.1,
            "eval_snmc_mean": 2.9,
            "eval_snmc_stderr": 0.2,
            "eval_n_episodes": 1000.0,
            "eval_n_contrastive_eval": 1000000,
            "eval_n_query_eval": 2000,
            "eval_greedy": True,
            "plot_path": None,
        }
    ]

    _save_heavy_eval_results(
        json_path=json_path,
        csv_path=csv_path,
        results=results,
    )

    saved_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_json == results

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["requested_likelihood_evals"] == "4000"
    assert rows[0]["total_likelihood_evals"] == "4000"
    assert rows[0]["total_rollouts"] == "50"
    assert rows[0]["eval_spce_mean"] == "2.5"
    assert rows[0]["eval_snmc_mean"] == "2.9"
