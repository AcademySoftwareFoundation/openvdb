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

# The compiled extension only exposes a `cuda` submodule when built with CUDA
# support. Attach `nanovdb.cuda.compile_options(*extra)` there so downstream
# code can obtain the NanoVDB include flag (plus any extra flags) for runtime
# CUDA compilation. Guarded so `import nanovdb` still succeeds in non-CUDA
# builds where the `cuda` submodule does not exist.
from .lib import nanovdb as _ext  # noqa: E402

if hasattr(_ext, "cuda"):

    def _cuda_compile_options(*extra):
        """Return the NanoVDB include flag followed by any extra flags.

        Suitable for passing to a runtime CUDA compiler (e.g. NVRTC) so kernels
        compile against the same NanoVDB headers as the installed wheel::

            opts = nanovdb.cuda.compile_options("-std=c++17")
        """
        return (f"-I{get_include()}",) + tuple(extra)

    _ext.cuda.compile_options = _cuda_compile_options

del _ext
