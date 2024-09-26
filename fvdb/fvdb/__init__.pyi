# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations

from typing import List, Sequence, Union, overload

import torch

if torch.cuda.is_available():
    torch.cuda.init()

# The following import needs to come after the GridBatch and JaggedTensor imports
# immediately above in order to avoid a circular dependency error.
from . import nn
from ._Cpp import (
    ConvPackBackend,
    GridBatch,
    JaggedTensor,
    SparseConvPackInfo,
    config,
    gridbatch_from_dense,
    gridbatch_from_ijk,
    gridbatch_from_mesh,
    gridbatch_from_nearest_voxels_to_points,
    gridbatch_from_points,
    jempty,
    jones,
    jrand,
    jrandn,
    jzeros,
    load,
    save,
    scaled_dot_product_attention,
    volume_render,
)

JaggedTensorOrTensor = Union[torch.Tensor, JaggedTensor]

@overload
def jcat(grid_batches: Sequence[GridBatch]) -> GridBatch: ...
@overload
def jcat(jagged_tensors: Sequence[JaggedTensorOrTensor], dim: int | None = None) -> JaggedTensor: ...
@overload
def jcat(jagged_tensors: Sequence[JaggedTensor], dim: int | None = None) -> JaggedTensor: ...
@overload
def jcat(vdb_tensors: Sequence[nn.VDBTensor], dim: int | None = None) -> nn.VDBTensor: ...

__all__ = [
    "GridBatch",
    "JaggedTensor",
    "SparseConvPackInfo",
    "ConvPackBackend",
    "gridbatch_from_ijk",
    "gridbatch_from_points",
    "gridbatch_from_nearest_voxels_to_points",
    "gridbatch_from_dense",
    "gridbatch_from_mesh",
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
    "volume_render",
]
