# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from ctypes import *
from typing import Tuple
import numpy as np

from .utils import load_library
from .device import DeviceInterface, pnanovdb_DeviceInterface, pnanovdb_Device
from .compiler import Compiler, pnanovdb_Compiler

COMPUTE_LIB = "pnanovdbcompute"

VULKAN_API = 1


class pnanovdb_ShaderInterface(Structure):
    """Definition equivalent to pnanovdb_compute_shader_interface_t."""
    _fields_ = [
        ("shader_interface", c_void_p),
        ("create_shader", CFUNCTYPE(c_void_p, POINTER(c_void_p))),
        ("map_shader_build", CFUNCTYPE(c_bool, c_void_p, POINTER(POINTER(c_void_p)))),
        ("destroy_shader", CFUNCTYPE(None, c_void_p)),
    ]


class pnanovdb_ComputeArray(Structure):
    """Definition equivalent to pnanovdb_compute_array_t."""
    _fields_ = [
        ("data", c_void_p),
        ("element_size", c_uint64),
        ("element_count", c_uint64),
        ("filepath", c_char_p)
    ]

    def __init__(self, data=None, element_size=0, element_count=0, filepath=None):
        super().__init__()
        self.data = data
        self.element_size = element_size
        self.element_count = element_count
        self.filepath = filepath.encode('utf-8') if isinstance(filepath, str) else filepath


class pnanovdb_Compute(Structure):
    """Definition equivalent to pnanovdb_compute_t."""
    _fields_ = [
        ("compute", c_void_p),
        ("compiler", POINTER(pnanovdb_Compiler)),
        ("shader_interface", pnanovdb_ShaderInterface),
        ("device_interface", pnanovdb_DeviceInterface),
        ("load_nanovdb", CFUNCTYPE(POINTER(pnanovdb_ComputeArray), c_char_p)),
        ("save_nanovdb", CFUNCTYPE(None, POINTER(pnanovdb_ComputeArray), c_char_p)),
        ("create_shader_context", CFUNCTYPE(c_void_p, c_char_p)),
        ("destroy_shader_context", CFUNCTYPE(None, c_void_p)),
        ("init_shader", CFUNCTYPE(c_int,
                                c_void_p,  # POINTER(pnanovdb_Compute)
                                c_void_p,  # pnanovdb_compute_queue_t*
                                c_void_p,  # pnanovdb_shader_context_t*
                                c_void_p)), # pnanovdb_compiler_settings_t*
        ("destroy_shader", CFUNCTYPE(None,
                                   c_void_p,  # POINTER(pnanovdb_Compute)
                                   POINTER(pnanovdb_ShaderInterface),
                                   c_void_p,  # pnanovdb_compute_context_t*
                                   c_void_p)), # pnanovdb_shader_context_t*
        ("dispatch_shader", CFUNCTYPE(None,
                                    c_void_p,  # POINTER(pnanovdb_Compute)
                                    c_void_p,  # pnanovdb_compute_context_t*
                                    c_void_p,  # const pnanovdb_shader_context_t*
                                    c_void_p,  # pnanovdb_compute_resource_t*
                                    c_uint32,  # grid_dim_x
                                    c_uint32,  # grid_dim_y
                                    c_uint32,  # grid_dim_z
                                    c_char_p)), # debug_label
        ("dispatch_shader_on_array", CFUNCTYPE(c_int,
                                             c_void_p,  # POINTER(pnanovdb_Compute)
                                             POINTER(pnanovdb_Device),
                                             c_char_p,
                                             c_uint32,
                                             c_uint32,
                                             c_uint32,
                                             POINTER(pnanovdb_ComputeArray),
                                             POINTER(pnanovdb_ComputeArray),
                                             POINTER(pnanovdb_ComputeArray),
                                             c_uint32,
                                             c_uint64,
                                             c_uint64)),
        ("dispatch_shader_on_nanovdb_array", CFUNCTYPE(c_int,
                                                     c_void_p,  # POINTER(pnanovdb_Compute)
                                                     c_void_p,  # const pnanovdb_compute_device_t*
                                                     c_void_p,  # const pnanovdb_shader_context_t*
                                                     POINTER(pnanovdb_ComputeArray),  # nanovdb_array
                                                     c_int32,   # image_width
                                                     c_int32,   # image_height
                                                     c_void_p,  # background_image
                                                     c_void_p,  # upload_buffer
                                                     c_void_p,  # user_upload_buffer
                                                     POINTER(c_void_p),  # nanovdb_buffer
                                                     POINTER(c_void_p))), # readback_buffer
        ("create_array", CFUNCTYPE(POINTER(pnanovdb_ComputeArray),
                                 c_size_t,
                                 c_uint64,
                                 c_void_p)),
        ("destroy_array", CFUNCTYPE(None, POINTER(pnanovdb_ComputeArray))),
        ("map_array", CFUNCTYPE(c_void_p, POINTER(pnanovdb_ComputeArray))),
        ("unmap_array", CFUNCTYPE(None, POINTER(pnanovdb_ComputeArray))),
        ("compute_array_print_range", CFUNCTYPE(None,
                                              c_void_p,  # POINTER(pnanovdb_Compute)
                                              c_void_p,  # pnanovdb_compute_log_print_t
                                              c_char_p,  # name
                                              POINTER(pnanovdb_ComputeArray),  # arr
                                              c_uint32)),  # channel_count
        ("module", c_void_p)
    ]


class Compute:
    """Python wrapper for pnanovdb_compute_t."""
    def __init__(self, compiler: Compiler):
        """Mirrors what is in pnanovdb_compute_load"""
        self._compiler = compiler

        lib = load_library(COMPUTE_LIB)

        # only vulkan is supported for now
        self._device_interface = DeviceInterface(VULKAN_API)

        get_compute = lib.pnanovdb_get_compute
        get_compute.restype = POINTER(pnanovdb_Compute)
        get_compute.argtypes = []

        self._compute = get_compute()
        if not self._compute:
            raise RuntimeError("Failed to get compute interface")

        compiler_ptr = compiler.get_compiler()
        if not compiler_ptr:
            raise RuntimeError("Failed to get compiler interface")

        self._compute.contents.compiler = compiler_ptr
        self._compute.contents.device_interface = self._device_interface.get_device_interface().contents

        get_shader_interface = lib.pnanovdb_get_compute_shader_interface
        get_shader_interface.restype = POINTER(pnanovdb_ShaderInterface)
        get_shader_interface.argtypes = []

        self._compute.contents.shader_interface = get_shader_interface().contents

    def get_compute(self) -> POINTER(pnanovdb_Compute):
        return self._compute

    def compiler(self) -> Compiler:
        return self._compiler

    def device_interface(self) -> DeviceInterface:
        return self._device_interface

    def load_nanovdb(self, filepath: str) -> pnanovdb_ComputeArray:
        load_func = self._compute.contents.load_nanovdb
        array = load_func(filepath.encode('utf-8'))
        if not array:
            raise RuntimeError(f"Failed to load NanoVDB file: {filepath}")
        return array.contents

    def save_nanovdb(self, array: pnanovdb_ComputeArray, filepath: str) -> None:
        save_func = self._compute.contents.save_nanovdb
        save_func(pointer(array), filepath.encode('utf-8'))

    def create_array(self, data: np.ndarray) -> pnanovdb_ComputeArray:
        create_func = self._compute.contents.create_array
        array = create_func(
            data.itemsize,
            data.size,
            data.ctypes.data_as(c_void_p)
        )
        if not array:
            raise RuntimeError("Failed to create compute array")
        return array.contents

    def destroy_array(self, array: pnanovdb_ComputeArray) -> None:
        destroy_func = self._compute.contents.destroy_array
        destroy_func(pointer(array))

    def dispatch_shader_on_array(self,
                               shader_path: str,
                               grid_dims: Tuple[int, int, int],
                               data_in: pnanovdb_ComputeArray,
                               constants: pnanovdb_ComputeArray,
                               data_out: pnanovdb_ComputeArray,
                               dispatch_count: int = 1,
                               scratch_size: int = 0,
                               scratch_clear_size: int = 0) -> bool:
        if not data_in or not constants or not data_out:
            raise ValueError("ComputeArray parameters cannot be None")

        dispatch_func = self._compute.contents.dispatch_shader_on_array
        result = dispatch_func(
            self._compute,
            self._device_interface.get_device(),
            shader_path.encode('utf-8'),
            c_uint32(grid_dims[0]),
            c_uint32(grid_dims[1]),
            c_uint32(grid_dims[2]),
            pointer(data_in),
            pointer(constants),
            pointer(data_out),
            c_uint32(dispatch_count),
            c_uint64(scratch_size),
            c_uint64(scratch_clear_size)
        )

        return result == 0

    def map_array(self, array: pnanovdb_ComputeArray, np_dtype: np.dtype) -> np.ndarray:
        if array.element_size != np_dtype.itemsize:
            raise ValueError("Array element size mismatches the provided dtype")

        map_func = self._compute.contents.map_array
        data_ptr = map_func(pointer(array))
        if not data_ptr:
            raise RuntimeError("Failed to map array")

        buffer = (c_byte * (array.element_size * array.element_count)).from_address(data_ptr)
        return np.frombuffer(buffer, dtype=np_dtype)

    def unmap_array(self, array: pnanovdb_ComputeArray) -> None:
        unmap_func = self._compute.contents.unmap_array
        unmap_func(pointer(array))

    def array_exists(self, array: pnanovdb_ComputeArray) -> bool:
        return array and array.data is not None

    def __del__(self):
        self._compute = None
        self._compiler = None
        self._device_interface = None
