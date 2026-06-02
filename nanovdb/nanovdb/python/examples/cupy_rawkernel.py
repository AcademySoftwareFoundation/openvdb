# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Run a custom CUDA kernel over a NanoVDB device grid with CuPy.

This shows the full "bring your own kernel" path: compile a CUDA kernel
that ``#include <nanovdb/NanoVDB.h>`` against the SAME headers the nanovdb
wheel was built with (via ``nanovdb.cuda.compile_options()``), then launch
it on the device grid pointer obtained from ``grid.data_ptr()``.

Device-pointer ABI (read this before writing a kernel):

* ``handle.deviceGrid(n).data_ptr()`` is a raw DEVICE pointer (a Python
  int) to the base of a ``nanovdb::NanoGrid<BuildT>`` in GPU memory. For a
  float grid that is ``const nanovdb::NanoGrid<float>*``.
* Pass it to a kernel as a plain pointer argument. With CuPy RawKernel the
  argument tuple takes the int directly (CuPy forwards it as a pointer).
* The grid layout in device memory is exactly the C++ ``NanoGrid<BuildT>``
  ABI, so the same accessor / sampler code you would write in C++ works
  inside the kernel.
* ``grid.data_ptr()`` cannot tell host from device — only pass a pointer
  from ``deviceGrid(n)`` to a device kernel. A host pointer from
  ``grid(n)`` would dereference host memory on the GPU (garbage / fault).

``nanovdb.cuda.compile_options(*extra)`` returns ``-I<headers>`` followed
by any extra flags. The header dir is only physically present in an
installed wheel; in an in-source dev build tree it resolves but does not
exist, so this example falls back to a ``NANOVDB_INCLUDE`` env var hint.

Requires CuPy and a CUDA-capable GPU.

Run with: python cupy_rawkernel.py
"""
import os

import nanovdb

KERNEL_SRC = r"""
#include <nanovdb/NanoVDB.h>

// d_grid is the raw device pointer from FloatGrid.data_ptr(); out is a
// 2-float device buffer that receives [value@origin, activeVoxelCount].
extern "C" __global__
void inspect_float_grid(const nanovdb::NanoGrid<float>* d_grid, float* out)
{
    auto acc = d_grid->getAccessor();
    out[0] = acc.getValue(nanovdb::Coord(0, 0, 0));
    out[1] = static_cast<float>(d_grid->activeVoxelCount());
}
"""


def _include_options():
    """compile_options(), falling back to $NANOVDB_INCLUDE in a dev tree."""
    opts = list(nanovdb.cuda.compile_options("-std=c++17"))
    inc_dir = opts[0][2:]  # strip the leading -I
    if not os.path.isdir(inc_dir):
        env_inc = os.environ.get("NANOVDB_INCLUDE")
        if env_inc and os.path.isdir(env_inc):
            opts[0] = f"-I{env_inc}"
        else:
            print(f"NanoVDB headers not found at {inc_dir!r} (expected in an "
                  "installed wheel). Set NANOVDB_INCLUDE to the dir that "
                  "contains nanovdb/NanoVDB.h to run from a source tree.")
            return None
    return tuple(opts)


def main():
    if not (nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable()):
        print("This example requires a CUDA build of nanovdb and a GPU. "
              "Skipping.")
        return
    try:
        import cupy as cp
    except ImportError:
        print("This example requires CuPy. Install it with: pip install cupy")
        return

    options = _include_options()
    if options is None:
        return

    # Build a float level-set sphere directly on the device.
    handle = nanovdb.tools.cuda.createLevelSetSphere(nanovdb.GridType.Float, 20)
    handle.deviceUpload(0, True)
    device_grid = handle.deviceGrid(0)
    print(f"Device FloatGrid at {hex(device_grid.data_ptr())}")

    kernel = cp.RawKernel(
        KERNEL_SRC, "inspect_float_grid", options=options, backend="nvrtc")

    out = cp.zeros(2, dtype=cp.float32)
    # Launch with the raw device-grid pointer as the first argument.
    kernel((1,), (1,), (device_grid.data_ptr(), out.data.ptr))
    cp.cuda.runtime.deviceSynchronize()

    value, active = cp.asnumpy(out)
    print(f"  kernel read value@origin = {value}")
    print(f"  kernel read activeVoxelCount = {int(active)}")


if __name__ == "__main__":
    main()
