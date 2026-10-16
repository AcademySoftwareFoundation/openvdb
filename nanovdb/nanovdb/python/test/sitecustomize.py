#!/usr/bin/env python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Configure dependent-DLL lookup for in-tree Windows Python tests."""

import os


# Keep these handles alive for the lifetime of the child Python process.
_dll_directory_handles = []

if hasattr(os, "add_dll_directory"):
    for _path in os.environ.get(
            "NANOVDB_TEST_DLL_DIRECTORIES", "").split(os.pathsep):
        if os.path.isdir(_path):
            try:
                _dll_directory_handles.append(os.add_dll_directory(_path))
            except OSError:
                pass
