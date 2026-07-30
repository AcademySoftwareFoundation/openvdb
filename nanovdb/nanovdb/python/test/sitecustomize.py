#!/usr/bin/env python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Configure dependent-DLL lookup for in-tree Windows Python tests."""

import os


# Keep this handle alive for the lifetime of the child Python process.
_dll_directory_handle = None

if hasattr(os, "add_dll_directory"):
    _path = os.environ.get("NANOVDB_TEST_DLL_DIRECTORY")
    if _path and os.path.isdir(_path):
        try:
            _dll_directory_handle = os.add_dll_directory(_path)
        except OSError:
            pass
