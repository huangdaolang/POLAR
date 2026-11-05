# Vendored from TabPFN (https://github.com/PriorLabs/TabPFN).
#
# Copyright (c) Prior Labs GmbH 2025.
#
# Redistributed under the Prior Labs License v1.2 (Apache 2.0 with an
# additional attribution provision); full text in licenses/TABPFN_LICENSE.txt.
#
# This file was modified by the POLAR authors; see THIRD_PARTY_NOTICES.md.

"""Contains a collection of different model architectures.

"Architecture" refers to a PyTorch module, which is then wrapped by e.g.
TabPFNClassifier or TabPFNRegressor to form the complete model.

Each submodule in this module should contain an architecture. Each may be a directory,
or just a single file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import base

if TYPE_CHECKING:
    from polar.backbones.tabpfn.model.interface import ArchitectureModule

ARCHITECTURES: dict[str, ArchitectureModule] = {"base": base}
"""Map from architecture names to the corresponding module."""
