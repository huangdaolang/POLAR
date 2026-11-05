#!/usr/bin/env python
"""Create safe, optimizer-free POLAR checkpoint payloads for distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


TRAINING_ONLY_KEYS = {
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
}
PRIVATE_PATH_PREFIXES = ("/Users/", "/home/", "/scratch/", "/pfs/")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_local_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_local_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_local_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_local_paths(item) for item in value)
    if isinstance(value, str) and value.startswith(PRIVATE_PATH_PREFIXES):
        return "<redacted-local-path>"
    return value


def export_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    """Export one checkpoint and return its manifest record."""

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict checkpoint payload: {source}")

    removed_keys = sorted(key for key in TRAINING_ONLY_KEYS if key in payload)
    release_payload = {
        key: _redact_local_paths(value)
        for key, value in payload.items()
        if key not in TRAINING_ONLY_KEYS
    }
    # Loading a full fine-tuned state still requires the public base checkpoint
    # to instantiate the architecture. Let POLAR resolve it under
    # checkpoints/pretrained instead of retaining a machine-specific path.
    if "backbone_checkpoint_path" in release_payload:
        release_payload["backbone_checkpoint_path"] = None

    metadata = release_payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        release_payload["metadata"] = metadata
    metadata["release_export"] = {
        "format_version": 1,
        "training_state_removed": removed_keys,
        "local_paths_redacted": True,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release_payload, destination)

    config = release_payload.get("config")
    model = config.get("model", {}) if isinstance(config, dict) else {}
    return {
        "source_sha256": _sha256(source),
        "release_sha256": _sha256(destination),
        "source_size_bytes": source.stat().st_size,
        "release_size_bytes": destination.stat().st_size,
        "removed_keys": removed_keys,
        "model_backend": model.get("name") if isinstance(model, dict) else None,
        "epoch": release_payload.get("epoch"),
    }


def export_tree(
    source_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export every ``.pt`` file below ``source_root`` preserving paths."""

    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Checkpoint source directory does not exist: {source_root}")
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("Output directory must be outside the source checkpoint tree.")

    sources = sorted(source_root.rglob("*.pt"))
    if not sources:
        raise FileNotFoundError(f"No .pt checkpoints found below {source_root}")

    records: dict[str, Any] = {}
    for source in sources:
        relative = source.relative_to(source_root)
        destination = output_root / relative
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Release checkpoint already exists: {destination}. Use --overwrite to replace it."
            )
        records[relative.as_posix()] = export_checkpoint(source, destination)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": source_root.name,
        "checkpoint_count": len(records),
        "checkpoints": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("checkpoints/policy"),
        help="Directory containing selected POLAR checkpoints.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New directory for optimizer-free release checkpoints and manifest.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace release files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = export_tree(
        args.source_root,
        args.output_root,
        overwrite=bool(args.overwrite),
    )
    print(
        f"Exported {manifest['checkpoint_count']} checkpoints to "
        f"{args.output_root.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
