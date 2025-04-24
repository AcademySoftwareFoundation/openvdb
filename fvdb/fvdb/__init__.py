# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.init()

# isort: off
from ._Cpp import JaggedTensor, GridBatch, SparseConvPackInfo, ConvPackBackend, GaussianSplat3d
from ._Cpp import (
    gridbatch_from_ijk,
    gridbatch_from_points,
    gridbatch_from_nearest_voxels_to_points,
    gridbatch_from_dense,
    gridbatch_from_mesh,
    load,
    save,
    scaled_dot_product_attention,
    config,
    jrand,
    jrandn,
    jones,
    jzeros,
    jempty,
    volume_render,
    gaussian_render_jagged,
)

# The following import needs to come after the GridBatch and JaggedTensor imports
# immediately above in order to avoid a circular dependency error.
from . import nn

# isort: on


def jcat(things_to_cat, dim=None):
    if len(things_to_cat) == 0:
        raise ValueError("Cannot concatenate empty list")
    if isinstance(things_to_cat[0], GridBatch):
        if dim is not None:
            raise ValueError("GridBatch concatenation does not support dim argument")
        return _Cpp.jcat(things_to_cat)
    elif isinstance(things_to_cat[0], JaggedTensor):
        return _Cpp.jcat(things_to_cat, dim)
    elif isinstance(things_to_cat[0], nn.VDBTensor):
        if dim == 0:
            raise ValueError("VDBTensor concatenation does not support dim=0")
        grids = [t.grid for t in things_to_cat]
        data = [t.data for t in things_to_cat]
        # FIXME: Note we're not checking that the grids can be concatenated if you pass in grids with mismatching
        #        topology.
        return nn.VDBTensor(_Cpp.jcat(grids) if dim == None else grids[0], _Cpp.jcat(data, dim))
    else:
        raise ValueError("jcat() can only cat GridBatch, JaggedTensor, or VDBTensor")


from .version import __version__

__version_info__ = tuple(map(int, __version__.split(".")))

__all__ = [
    "GridBatch",
    "JaggedTensor",
    "SparseConvPackInfo",
    "ConvPackBackend",
    "GaussianSplat3d",
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
    "gaussian_render_jagged",
]
