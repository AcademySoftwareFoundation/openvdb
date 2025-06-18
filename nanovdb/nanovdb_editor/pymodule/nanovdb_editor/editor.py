# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from ctypes import *

from .compute import Compute, pnanovdb_Compute, pnanovdb_ComputeArray
from .compiler import Compiler, pnanovdb_Compiler
from .device import pnanovdb_Device
from .utils import load_library

EDITOR_LIB = "pnanovdbeditor"

CALLABLE_FUNC = CFUNCTYPE(None, c_char_p)


class pnanovdb_Editor(Structure):
    """Definition equivalent to pnanovdb_editor_interface_t."""
    _fields_ = [
        ("editor", c_void_p),
        ("compiler", POINTER(pnanovdb_Compiler)),
        ("compute", POINTER(pnanovdb_Compute)),
        ("init", CFUNCTYPE(None, c_void_p)),
        ("shutdown", CFUNCTYPE(None, c_void_p)),
        ("add_nanovdb", CFUNCTYPE(None,
                                 c_void_p,
                                 POINTER(pnanovdb_ComputeArray))),
        ("add_array", CFUNCTYPE(None,
                               c_void_p,
                               POINTER(pnanovdb_ComputeArray))),
        ("add_callable", CFUNCTYPE(None, c_void_p, c_char_p, CALLABLE_FUNC)),
        ("show", CFUNCTYPE(None, c_void_p, POINTER(pnanovdb_Device))),
        ("module", c_void_p),
        ("nanovdb_array", POINTER(pnanovdb_ComputeArray)),
        ("data_array", POINTER(pnanovdb_ComputeArray)),
        ("callable_func", CALLABLE_FUNC),
        ("callable_name", c_char * 64),
    ]


class Editor:
    """Python wrapper for pnanovdb_editor_t."""
    def __init__(self, compute: Compute, compiler: Compiler):
        lib = load_library(EDITOR_LIB)

        get_editor = lib.pnanovdb_get_editor
        get_editor.restype = POINTER(pnanovdb_Editor)
        get_editor.argtypes = []

        self._editor = get_editor()
        if not self._editor:
            raise RuntimeError("Failed to get editor interface")

        self._compute = compute
        self._compiler = compiler
        self._callable_func_ref = None

        self._editor.contents.compute = compute.get_compute()
        self._editor.contents.compiler = compiler.get_compiler()

        init_func = self._editor.contents.init
        init_func(self._editor)

    def shutdown(self) -> None:
        shutdown_func = self._editor.contents.shutdown
        shutdown_func(self._editor)

    def add_callable(self, name: str, callable_func) -> None:
        self._callable_func_ref = CALLABLE_FUNC(callable_func)

        add_callable_func = self._editor.contents.add_callable
        add_callable_func(self._editor, name.encode('utf-8'), self._callable_func_ref)

    def add_nanovdb(self, array: pnanovdb_ComputeArray) -> None:
        add_nanovdb_func = self._editor.contents.add_nanovdb
        add_nanovdb_func(self._editor, pointer(array))

    def add_array(self, array: pnanovdb_ComputeArray) -> None:
        add_array_func = self._editor.contents.add_array
        add_array_func(self._editor, pointer(array))

    def show(self) -> None:
        show_func = self._editor.contents.show
        try:
            show_func(self._editor, self._compute.device_interface().get_device())
        except Exception as e:
            print(f"Error: Editor runtime error ({e})")

    def get_nanovdb(self) -> pnanovdb_ComputeArray:
        return self._editor.contents.nanovdb_array.contents

    def get_array(self) -> pnanovdb_ComputeArray:
        return self._editor.contents.data_array.contents

    def __del__(self):
        if self._editor:
            self.shutdown()

        self._editor = None
