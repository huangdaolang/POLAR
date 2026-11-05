# Vendored from TabICL (https://github.com/soda-inria/tabicl), revision
# f719c886a586ed4a29236345e319ac1ea596c478.
#
# Copyright (c) 2025, Soda team @ Inria
#
# Redistributed under the BSD 3-Clause License. The full copyright notice, the
# list of conditions, and the disclaimer are in licenses/TABICL_LICENSE.txt.
#
# This file was modified by the POLAR authors; see THIRD_PARTY_NOTICES.md.

"""Vendored TabICL model modules for POLAR backbones."""

from polar.backbones.tabicl.model.inference_config import InferenceConfig
from polar.backbones.tabicl.model.kv_cache import TabICLCache
from polar.backbones.tabicl.model.tabicl import TabICL

__all__ = ["InferenceConfig", "TabICL", "TabICLCache"]
