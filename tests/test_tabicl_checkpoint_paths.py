from pathlib import Path

from polar.backbones.tabicl.checkpoint_paths import (
    default_tabicl_classifier_checkpoint_path,
    default_tabicl_regressor_checkpoint_path,
    resolve_tabicl_checkpoint,
    resolve_tabicl_regressor_checkpoint,
)


def test_default_checkpoint_path_shape() -> None:
    path = default_tabicl_regressor_checkpoint_path("v2")
    assert "checkpoints/pretrained/tabicl/v2" in str(path)
    assert path.name == "tabicl-regressor-v2-20260212.ckpt"


def test_resolve_explicit_checkpoint_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"")
    resolved = resolve_tabicl_regressor_checkpoint(
        checkpoint_path=ckpt,
        version="v2",
        must_exist=True,
    )
    assert resolved == ckpt.resolve()


def test_resolve_default_without_exists_check() -> None:
    resolved = resolve_tabicl_regressor_checkpoint(
        checkpoint_path=None,
        version="v2",
        must_exist=False,
    )
    assert resolved.name == "tabicl-regressor-v2-20260212.ckpt"


def test_default_classifier_checkpoint_path_shape() -> None:
    path = default_tabicl_classifier_checkpoint_path("v2")
    assert "checkpoints/pretrained/tabicl/v2" in str(path)
    assert path.name == "tabicl-classifier-v2-20260212.ckpt"


def test_resolve_classifier_default_without_exists_check() -> None:
    resolved = resolve_tabicl_checkpoint(
        checkpoint_path=None,
        version="v2",
        checkpoint_kind="classification",
        must_exist=False,
    )
    assert resolved.name == "tabicl-classifier-v2-20260212.ckpt"
