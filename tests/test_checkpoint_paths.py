from pathlib import Path

from polar.backbones.tabpfn.checkpoint_paths import (
    default_tabpfn_classifier_checkpoint_path,
    default_tabpfn_regressor_checkpoint_path,
    get_path_layout,
    resolve_tabpfn_checkpoint,
    resolve_tabpfn_regressor_checkpoint,
)


def test_default_checkpoint_path_shape() -> None:
    path = default_tabpfn_regressor_checkpoint_path("v2.5")
    assert "checkpoints/pretrained/tabpfn/v2.5" in str(path)
    assert path.name == "tabpfn-v2.5-regressor-v2.5_default.ckpt"


def test_resolve_explicit_checkpoint_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"")
    resolved = resolve_tabpfn_regressor_checkpoint(
        checkpoint_path=ckpt,
        version="v2.5",
        must_exist=True,
    )
    assert resolved == ckpt.resolve()


def test_resolve_default_without_exists_check() -> None:
    resolved = resolve_tabpfn_regressor_checkpoint(
        checkpoint_path=None,
        version="v2",
        must_exist=False,
    )
    assert resolved.name == "tabpfn-v2-regressor-v2_default.ckpt"


def test_default_results_layout() -> None:
    layout = get_path_layout()
    assert str(layout.results_root).endswith("results")
    assert str(layout.runs_root).endswith("results/runs")
    assert str(layout.eval_root).endswith("results/eval")


def test_default_classifier_checkpoint_path_shape() -> None:
    path = default_tabpfn_classifier_checkpoint_path("v2.5")
    assert "checkpoints/pretrained/tabpfn/v2.5" in str(path)
    assert path.name == "tabpfn-v2.5-classifier-v2.5_default-2.ckpt"


def test_resolve_classifier_default_without_exists_check() -> None:
    resolved = resolve_tabpfn_checkpoint(
        checkpoint_path=None,
        version="v2.5",
        checkpoint_kind="classification",
        must_exist=False,
    )
    assert resolved.name == "tabpfn-v2.5-classifier-v2.5_default-2.ckpt"
