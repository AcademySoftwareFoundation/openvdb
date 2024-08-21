# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations
from typing import Optional, Tuple, Union, Mapping

import torch

if torch.cuda.is_available():
    torch.cuda.init()

from . import utils
from ._Cpp import JaggedTensor, GridBatch, SparseConvPackInfo, ConvPackBackend
from ._Cpp import (
    jcat,
    sparse_grid_from_ijk,
    sparse_grid_from_points,
    sparse_grid_from_nearest_voxels_to_points,
    sparse_grid_from_dense,
    sparse_grid_from_mesh,
    load,
    save,
    scaled_dot_product_attention,
    config,
    jrand,
    jrandn,
    jones,
    jzeros,
    jempty,
)

# The following import needs to come after the GridBatch and JaggedTensor imports
# immediately above in order to avoid a circular dependency error.
from . import nn


__version__ = "0.0.1"
__version_info__ = (0, 0, 1)

__all__ = [
    "GridBatch",
    "JaggedTensor",
    "SparseConvPackInfo",
    "ConvPackBackend",
    "sparse_grid_from_ijk",
    "sparse_grid_from_points",
    "sparse_grid_from_nearest_voxels_to_points",
    "sparse_grid_from_dense",
    "sparse_grid_from_mesh",
    "load",
    "jcat",
    "scaled_dot_product_attention",
    "config",
    "save",
    "jrand",
    "jrandn",
    "jones",
    "jzeros",
    "jempty",
]
