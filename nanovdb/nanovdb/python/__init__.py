# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
import os
import sys

if sys.platform == "win32":
    _openvdb_dll_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "openvdb", "lib"
    )
    if os.path.isdir(_openvdb_dll_directory):
        os.add_dll_directory(_openvdb_dll_directory)


def get_include():
    """Return the absolute path to the bundled NanoVDB C/C++ include directory.

    Use this from a downstream Python extension's build system so the extension
    compiles against the same NanoVDB headers the installed wheel was built with::

        import nanovdb
        ext_kwargs = dict(include_dirs=[nanovdb.get_include()])
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")


from .lib.nanovdb import *  # noqa: E402,F401,F403
