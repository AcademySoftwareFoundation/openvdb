# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from ctypes import *
from enum import Enum
import numpy as np
from typing import Tuple
from .utils import load_library

COMPILER_LIB = "pnanovdbcompiler"


class pnanovdb_CompileTarget(Enum):
    UNDEFINED = 0

class pnanovdb_CompileTarget(Enum):
    UNDEFINED = 0
    VULKAN = 1
    CPU = 2


class pnanovdb_CompilerSettings(Structure):
    """Definition equivalent to pnanovdb_compiler_settings_t."""
    _fields_ = [
        ("is_row_major", c_int32),
        ("use_glslang", c_int32),
        ("hlsl_output", c_int32),
        ("compile_target", c_uint32),
        ("entry_point_name", c_char * 64),
    ]


class pnanovdb_CompilerInstance(Structure):
    """Definition equivalent to pnanovdb_compiler_instance_t."""


class pnanovdb_Compiler(Structure):
    """Definition equivalent to pnanovdb_compiler_t."""
    _fields_ = [
        ("compiler", c_void_p),
        ("create_instance", CFUNCTYPE(POINTER(pnanovdb_CompilerInstance))),
        ("compile_shader_from_file",
            CFUNCTYPE(c_bool,
                     POINTER(pnanovdb_CompilerInstance),
                     c_char_p,
                     POINTER(pnanovdb_CompilerSettings),
                     POINTER(c_bool))),
        ("execute_cpu",
            CFUNCTYPE(c_bool,
                      POINTER(pnanovdb_CompilerInstance),
                      c_char_p,
                      c_uint32,
                      c_uint32,
                      c_uint32,
                      c_void_p,
                      c_void_p)),
        ("destroy_instance", CFUNCTYPE(None, POINTER(pnanovdb_CompilerInstance)))
    ]


class MemoryBuffer(Structure):
    """Python wrapper for memory buffer passed to the shader function run on CPU."""
    _fields_ = [
        ("data", c_void_p),
        ("size", c_uint64),
    ]

    def __init__(self, data: np.ndarray):
        """Initialize the MemoryBuffer with a numpy array."""
        self.data = data.ctypes.data_as(c_void_p)
        self.size = data.size

    def to_ndarray(self, dtype):
        """Convert memory buffer to numpy array of specified dtype."""
        if not self.data or not self.size:
            return np.array([], dtype=dtype)

        # Create array from buffer without copying
        return np.frombuffer(
            (c_byte * (self.size * np.dtype(dtype).itemsize)).from_address(self.data),
            dtype=dtype
        )


class Compiler:
    """Python wrapper for pnanovdb_compiler_t."""
    def __init__(self):
        """Mirrors what is in pnanovdb_compiler_load"""
        lib = load_library(COMPILER_LIB)

        get_compiler = lib.pnanovdb_get_compiler
        get_compiler.restype = POINTER(pnanovdb_Compiler)
        get_compiler.argtypes = []
        self._compiler = get_compiler()
        if not self._compiler:
            raise RuntimeError("Failed to get compiler interface")

        self._instance = None

    def get_compiler(self) -> POINTER(pnanovdb_Compiler):
        return self._compiler

    def create_instance(self) -> None:
        if self._instance:
            raise RuntimeError("Compiler instance already exists")

        create_func = self._compiler.contents.create_instance
        self._instance = create_func()

        if not self._instance:
            raise RuntimeError("Failed to create compiler instance")

    def compile_shader(self, filename: str, entry_point_name="main", is_row_major=False, compile_target=pnanovdb_CompileTarget.VULKAN) -> bool:
        if not self._instance:
            raise RuntimeError("No compiler instance exists")

        settings = pnanovdb_CompilerSettings(entry_point_name=entry_point_name.encode('utf-8'),
                                             is_row_major=is_row_major,
                                             use_glslang=False,
                                             hlsl_output=False,
                                             compile_target=(c_uint32)(compile_target.value))

        compile_func = self._compiler.contents.compile_shader_from_file
        return compile_func(
            self._instance,
            filename.encode('utf-8'),
            byref(settings),
            None
        )

    def execute_cpu(self, filename: str, grid_dims: Tuple[int, int, int], uniform_params: POINTER(c_void_p), uniform_state: POINTER(c_void_p)) -> bool:
        if not self._instance:
            raise RuntimeError("No compiler instance exists")

        execute_func = self._compiler.contents.execute_cpu
        return execute_func(
            self._instance,
            filename.encode('utf-8'),
            c_uint32(grid_dims[0]),
            c_uint32(grid_dims[1]),
            c_uint32(grid_dims[2]),
            uniform_params,
            uniform_state
        )

    def destroy_instance(self) -> None:
        if not self._instance:
            return

        destroy_func = self._compiler.contents.destroy_instance
        destroy_func(self._instance)

        self._instance = None

    def __del__(self):
        if self._instance:
            self.destroy_instance()

        self._compiler = None
