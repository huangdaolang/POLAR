"""Shared grammar for ``compare.methods`` entries in the evaluation scripts.

AL, BO, and BED evaluation all accept the same method-name grammar so that one
spelling works everywhere:

===============================  ==========================  ====================
Name                             Backbone                    Mode
===============================  ==========================  ====================
``random``                       none                        ``random``
``polar``                        ``policy.backend``          ``policy``
``polar_<label>``                ``policy.backend``          ``policy``
``<backend>_polar``              the backend in the name     ``policy``
``<backend>_polar_<label>``      the backend in the name     ``policy``
``<backend>_<acquisition>``      the backend in the name     ``surrogate``
===============================  ==========================  ====================

The two axes are orthogonal: the *prefix* selects a backbone and the *suffix*
labels one of several checkpoints that share a backbone. Encoding the backbone
in the name (rather than in a side map) keeps one evaluation run able to
compare several backbones, makes output directories self-describing, and lets
the existing ``method name -> backend name -> default`` checkpoint lookup find
the right file without extra configuration.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

POLAR_METHOD = "polar"
RANDOM_METHOD = "random"
_POLAR_INFIX = f"_{POLAR_METHOD}_"
_POLAR_SUFFIX = f"_{POLAR_METHOD}"

MethodMode = Literal["policy", "surrogate", "random"]


@dataclass(frozen=True)
class MethodSpec:
    """One entry of ``compare.methods`` after parsing."""

    name: str
    mode: MethodMode
    backend: str | None = None
    acquisition: str | None = None
    label: str | None = None


def normalize_backend_name(raw: str) -> str:
    """Lower-case and strip a backend name."""
    return str(raw).strip().lower()


def _fail(method: str, backends: Collection[str], acquisitions: Collection[str]) -> None:
    forms = [
        f"'{RANDOM_METHOD}'",
        f"'{POLAR_METHOD}'",
        f"'{POLAR_METHOD}_<label>'",
        "'<backend>_polar'",
        "'<backend>_polar_<label>'",
    ]
    if acquisitions:
        forms.append("'<backend>_<acquisition>'")
    detail = f"Could not parse method '{method}'. Expected one of: " + ", ".join(forms) + "."
    detail += f"\n  available backends: {', '.join(sorted(backends))}"
    if acquisitions:
        detail += f"\n  available acquisitions: {', '.join(sorted(acquisitions))}"
    raise ValueError(detail)


def parse_method(
    raw: str,
    *,
    backends: Collection[str],
    default_backend: str | None = None,
    acquisitions: Collection[str] = (),
    policy_backends_excluded: Collection[str] = ("gp", "rf"),
    allow_bare_random: bool = True,
) -> MethodSpec:
    """Parse one ``compare.methods`` entry.

    Args:
        raw: The configured method name.
        backends: Backend names available in ``cfg.models``.
        default_backend: Backbone used by the bare ``polar`` / ``polar_<label>``
            forms, i.e. ``cfg.policy.backend``.
        acquisitions: Surrogate acquisition suffixes this script supports. Pass
            an empty collection for scripts without surrogate baselines.
        policy_backends_excluded: Backends that have no learned policy head.
        allow_bare_random: Whether a standalone ``random`` method is meaningful.
            AL sets this to ``False`` because every AL method also needs a
            backend to fit and score the target set, so ``<backend>_random``
            must be used there instead.
    """
    method = str(raw).strip().lower()
    if not method:
        raise ValueError("Method name cannot be empty.")

    known = {normalize_backend_name(b) for b in backends}
    excluded = {normalize_backend_name(b) for b in policy_backends_excluded}
    acq = tuple(sorted({str(a).strip().lower() for a in acquisitions}, key=len, reverse=True))

    def _check_policy_backend(backend: str) -> str:
        if backend in excluded:
            raise ValueError(
                f"Method '{method}' is invalid: backend='{backend}' has no learned policy head."
            )
        if backend not in known:
            raise ValueError(
                f"Unknown backend '{backend}' in method '{method}'. "
                f"Available: {', '.join(sorted(known))}"
            )
        return backend

    # 1. Standalone random baseline.
    if method == RANDOM_METHOD:
        if not allow_bare_random:
            raise ValueError(
                "This script has no standalone 'random' method because every method also "
                "needs a backend for prediction. Use '<backend>_random', e.g. 'gp_random'."
            )
        return MethodSpec(name=method, mode="random")

    # 2/3. Canonical `polar` and labelled `polar_<label>`; backbone from config.
    if method == POLAR_METHOD or method.startswith(f"{POLAR_METHOD}_"):
        backend = normalize_backend_name(default_backend or "")
        if not backend:
            raise ValueError(
                f"Method '{method}' requires policy.backend to be configured."
            )
        _check_policy_backend(backend)
        label = method[len(POLAR_METHOD) + 1 :] or None
        return MethodSpec(name=method, mode="policy", backend=backend, label=label)

    # 4/5. `<backend>_polar` and `<backend>_polar_<label>`; backbone from the name.
    if method.endswith(_POLAR_SUFFIX):
        backend = _check_policy_backend(normalize_backend_name(method[: -len(_POLAR_SUFFIX)]))
        return MethodSpec(name=method, mode="policy", backend=backend, label=None)
    if _POLAR_INFIX in method:
        backend_raw, label = method.split(_POLAR_INFIX, maxsplit=1)
        if backend_raw and label:
            backend = _check_policy_backend(normalize_backend_name(backend_raw))
            return MethodSpec(name=method, mode="policy", backend=backend, label=label)

    # 6. `<backend>_<acquisition>` surrogate baselines.
    for acquisition in acq:
        suffix = f"_{acquisition}"
        if not method.endswith(suffix):
            continue
        backend = normalize_backend_name(method[: -len(suffix)])
        if not backend:
            raise ValueError(f"Invalid surrogate method '{method}'.")
        if backend not in known:
            raise ValueError(
                f"Unknown backend '{backend}' in method '{method}'. "
                f"Available: {', '.join(sorted(known))}"
            )
        return MethodSpec(name=method, mode="surrogate", backend=backend, acquisition=acquisition)

    _fail(method, known, acq)
    raise AssertionError("unreachable")  # pragma: no cover


__all__ = [
    "POLAR_METHOD",
    "RANDOM_METHOD",
    "MethodMode",
    "MethodSpec",
    "normalize_backend_name",
    "parse_method",
]
