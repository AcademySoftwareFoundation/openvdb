# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import Iterable, List, Tuple, Union

import numpy
import torch

Numeric = Union[int, float]

Vec3i = Union[torch.Tensor, numpy.ndarray, List[int], Tuple[int, int, int], torch.Size]
Vec3d = Union[torch.Tensor, numpy.ndarray, List[float], Tuple[float, float, float]]
Vec3dOrScalar = Union[Vec3d, float, int]
Vec3iOrScalar = Union[Vec3i, int]
Vec4i = Union[torch.Tensor, numpy.ndarray, List[int], Tuple[int, int, int, int]]

Vec3iBatch = Union[
    Vec3i, torch.Tensor, numpy.ndarray, List[int], List[List[int]], Tuple[int, int, int], List[Tuple[int, int, int]]
]
Vec3dBatch = Union[
    torch.Tensor,
    numpy.ndarray,
    List[float],
    List[List[float]],
    Tuple[float, float, float],
    List[Tuple[float, float, float]],
    Vec3iBatch,
    Vec3d,
]
Vec3dBatchOrScalar = Union[
    torch.Tensor,
    numpy.ndarray,
    List[float],
    List[List[float]],
    Tuple[float, float, float],
    List[Tuple[float, float, float]],
    float,
    Vec3iBatch,
    Vec3dBatch,
    int,
]

Index = Union[int, slice, type(Ellipsis), None]

GridIdentifier = Union[str, int, List[str], List[int], Tuple[str, ...], Tuple[int, ...]]

LShapeSpec = Union[Iterable[int], Iterable[Iterable[int]]]
RShapeSpec = Iterable[int]
