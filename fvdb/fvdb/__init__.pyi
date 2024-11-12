# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
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
    gaussian_fully_fused_projection,
    gaussian_render,
    gaussian_render_depth,
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
    precompute_gaussian_render_state,
    render_pixels_from_precomputed_gaussian_render_state,
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
    "gaussian_fully_fused_projection",
    "gaussian_render",
    "precompute_gaussian_render_state",
    "render_pixels_from_precomputed_gaussian_render_state",
    "gaussian_render_depth",
]
