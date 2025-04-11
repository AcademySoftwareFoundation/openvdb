# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
import sys
if sys.platform == "win32":
    import os
    openvdb_dll_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'openvdb', 'lib')
    os.add_dll_directory(directory)

from .lib.nanovdb import *
