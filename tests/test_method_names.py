"""Contract tests for the shared `compare.methods` grammar."""

from __future__ import annotations

import pytest

from polar.utils.method_names import MethodSpec, parse_method

BACKENDS = {"tabicl", "tabpfn", "gp", "rf"}
BO_ACQ = ("ei", "ei_mc", "ucb", "random")


def _parse(name: str, **kwargs) -> MethodSpec:
    options = {"backends": BACKENDS, "default_backend": "tabicl", "acquisitions": BO_ACQ}
    options.update(kwargs)
    return parse_method(name, **options)


@pytest.mark.parametrize(
    ("name", "mode", "backend", "label", "acquisition"),
    [
        ("random", "random", None, None, None),
        ("polar", "policy", "tabicl", None, None),
        ("polar_weighted", "policy", "tabicl", "weighted", None),
        ("tabicl_polar", "policy", "tabicl", None, None),
        ("tabpfn_polar", "policy", "tabpfn", None, None),
        ("tabpfn_polar_frozen", "policy", "tabpfn", "frozen", None),
        ("tabicl_ei_mc", "surrogate", "tabicl", None, "ei_mc"),
        ("gp_ei", "surrogate", "gp", None, "ei"),
    ],
)
def test_grammar_forms(name, mode, backend, label, acquisition) -> None:
    spec = _parse(name)
    assert (spec.name, spec.mode, spec.backend, spec.label, spec.acquisition) == (
        name,
        mode,
        backend,
        label,
        acquisition,
    )


def test_default_backend_switches_bare_polar() -> None:
    assert _parse("polar", default_backend="tabpfn").backend == "tabpfn"


def test_backend_prefix_overrides_the_default() -> None:
    # The whole point of `<backend>_polar`: one run can compare two backbones.
    assert _parse("tabicl_polar", default_backend="tabpfn").backend == "tabicl"


def test_bare_polar_requires_a_configured_backend() -> None:
    with pytest.raises(ValueError, match="policy.backend"):
        _parse("polar", default_backend=None)


def test_legacy_policy_spelling_is_rejected() -> None:
    with pytest.raises(ValueError, match="Could not parse method"):
        _parse("tabicl_policy")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        _parse("mystery_polar")


def test_backends_without_a_policy_head_are_rejected() -> None:
    with pytest.raises(ValueError, match="no learned policy head"):
        _parse("gp_polar")


def test_bare_random_can_be_disabled_for_al() -> None:
    with pytest.raises(ValueError, match="<backend>_random"):
        _parse("random", allow_bare_random=False)
    # ...but the backend-qualified form still parses as a surrogate.
    spec = _parse("gp_random", allow_bare_random=False)
    assert (spec.mode, spec.backend, spec.acquisition) == ("surrogate", "gp", "random")


def test_scripts_without_acquisitions_reject_surrogate_names() -> None:
    with pytest.raises(ValueError, match="Could not parse method"):
        _parse("tabicl_ei_mc", acquisitions=())


def test_names_are_case_and_whitespace_insensitive() -> None:
    assert _parse("  TabICL_Polar  ").backend == "tabicl"


def test_empty_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        _parse("   ")
