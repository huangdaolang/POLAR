from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_release_checkpoints.py"
_SPEC = importlib.util.spec_from_file_location("export_release_checkpoints", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to import {_SCRIPT}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_export_tree_removes_training_state_and_private_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "hpob" / "4796" / "checkpoint.pt"
    source.parent.mkdir(parents=True)
    torch.save(
        {
            "epoch": 7,
            "policy_head_state_dict": {"weight": torch.ones(2)},
            "backbone_model_state_dict": {"weight": torch.zeros(2)},
            "optimizer_state_dict": {"state": {0: {"momentum": torch.ones(2)}}},
            "backbone_checkpoint_path": "/scratch/private/base.ckpt",
            "config": {
                "model": {"name": "tabicl"},
                "data_path": "/Users/alice/private/data.pt",
            },
        },
        source,
    )
    output_root = tmp_path / "release"

    manifest = _MODULE.export_tree(source_root, output_root)

    destination = output_root / "hpob" / "4796" / "checkpoint.pt"
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert "optimizer_state_dict" not in payload
    assert payload["backbone_checkpoint_path"] is None
    assert payload["config"]["data_path"] == "<redacted-local-path>"
    assert payload["metadata"]["release_export"]["training_state_removed"] == [
        "optimizer_state_dict"
    ]
    assert manifest["checkpoint_count"] == 1
    assert manifest["checkpoints"]["hpob/4796/checkpoint.pt"]["model_backend"] == "tabicl"
    assert (output_root / "manifest.json").is_file()
