# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from ctypes import *

from .utils import load_library
from .compute import pnanovdb_Compute, pnanovdb_ComputeArray
from .device import pnanovdb_Device, pnanovdb_ComputeQueue

COMPUTE_LIB = "pnanovdbcompute"


class pnanovdb_Raster(Structure):
    """Definition equivalent to pnanovdb_raster_t."""
    _fields_ = [
        ("raster_interface", c_void_p),
        ("raster_points", CFUNCTYPE(POINTER(pnanovdb_ComputeArray),
                                  POINTER(pnanovdb_Compute),
                                  POINTER(pnanovdb_ComputeQueue),
                                  c_float,                         # voxel_size
                                  POINTER(pnanovdb_ComputeArray),  # means
                                  POINTER(pnanovdb_ComputeArray),  # quaternions
                                  POINTER(pnanovdb_ComputeArray),  # scales
                                  POINTER(pnanovdb_ComputeArray),  # colors
                                  POINTER(pnanovdb_ComputeArray),  # spherical_harmonics
                                  POINTER(pnanovdb_ComputeArray))), # opacities
        ("compute", POINTER(pnanovdb_Compute))
    ]


class Raster:
    """Python wrapper for pnanovdb_raster_t."""
    def __init__(self, compute: pnanovdb_Compute, device: pnanovdb_Device=None):
        lib = load_library(COMPUTE_LIB)

        get_raster_func = lib.pnanovdb_get_raster
        get_raster_func.restype = POINTER(pnanovdb_Raster)
        get_raster_func.argtypes = []

        self._raster = get_raster_func()
        if not self._raster:
            raise RuntimeError("Failed to get raster")

        self._compute = compute
        self._device = device if device else compute.device_interface().get_device()
        self._compute_queue = compute.device_interface().get_compute_queue(self._device)
        self._raster.contents.compute = compute.get_compute()

    def raster_points(
        self,
        voxel_size: float,
        means: pnanovdb_ComputeArray,
        quaternions: pnanovdb_ComputeArray,
        scales: pnanovdb_ComputeArray,
        colors: pnanovdb_ComputeArray,
        spherical_harmonics: pnanovdb_ComputeArray,
        opacities: pnanovdb_ComputeArray
    ) -> pnanovdb_ComputeArray:
        raster_points_func = self._raster.contents.raster_points

        try:
            nanovdb_array = raster_points_func(
                self._compute.get_compute(),
                self._compute_queue,
                voxel_size,
                means,
                quaternions,
                scales,
                colors,
                spherical_harmonics,
                opacities
            )
        except Exception as e:
            print(f"Error rastering points: {e}")
            raise e

        if not nanovdb_array:
            raise RuntimeError("Failed to raster points")

        return nanovdb_array.contents

    def __del__(self):
        self._raster = None
        self._compute = None
