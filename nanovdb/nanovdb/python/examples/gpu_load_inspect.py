# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Load a NanoVDB grid onto the GPU and inspect its device buffer zero-copy.

This is the GPU counterpart to ``load_inspect.py``. It builds a grid,
writes it to disk, reads it straight onto the device with
``nanovdb.io.deviceReadGrid``, uploads it with ``deviceUpload``, and then
views the whole device buffer as a CuPy array WITHOUT copying via the
CUDA Array Interface (``cupy.asarray(handle)``).

Key facts demonstrated:

* ``handle.deviceUpload(stream, sync)`` takes a RAW CUDA stream handle as
  a Python int (0 == the default stream).
* ``handle.__cuda_array_interface__`` (and ``__dlpack__``) expose the
  whole device buffer as a 1-D ``uint8`` array; the data pointer is null
  until ``deviceUpload`` runs.
* ``cupy.asarray(handle)`` aliases ``handle.device_ptr()`` with no copy;
  ``arr.nbytes == handle.size()``.
* A single handle exposes BOTH a host grid via ``handle.grid(n)`` and a
  device grid via ``handle.deviceGrid(n)``. They have different
  ``data_ptr()`` values and the grid object cannot tell host from device
  (see the docstring on ``Grid.data_ptr``) — feed ``deviceGrid(n)`` only
  to ``nanovdb.tools.cuda.*`` device entry points.

Requires CuPy and a CUDA-capable GPU.

Run with: python gpu_load_inspect.py
"""
import os
import tempfile

import nanovdb


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

    # Build a small level-set sphere on the host, then write it so we can
    # read it back straight onto the device.
    host_handle = nanovdb.tools.createLevelSetSphere(
        radius=20.0, voxelSize=1.0, name="gpu_sphere")
    tmp = tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False)
    tmp.close()
    nanovdb.io.writeGrid(tmp.name, host_handle)

    try:
        # deviceReadGrid returns a nanovdb.cuda.DeviceGridHandle.
        handle = nanovdb.io.deviceReadGrid(tmp.name)
    finally:
        os.unlink(tmp.name)

    print(f"Read device handle: gridCount={handle.gridCount()}, "
          f"size={handle.size()} bytes")
    print(f"  device_ptr before upload = {handle.device_ptr()} "
          f"(0 == not uploaded yet)")
    print(f"  deviceGrid(0) before upload = {handle.deviceGrid(0)}")

    # Upload to the device on the default stream, synchronously.
    handle.deviceUpload(0, True)
    print(f"  device_ptr after upload  = {hex(handle.device_ptr())}")

    # Zero-copy view of the whole device buffer as a CuPy uint8 array.
    buf = cp.asarray(handle)
    print(f"  cupy.asarray(handle): shape={buf.shape}, dtype={buf.dtype}, "
          f"nbytes={buf.nbytes}")
    print(f"    aliases device_ptr (no copy): "
          f"{int(buf.data.ptr) == handle.device_ptr()}")

    # The host grid and the device grid share the handle but differ in
    # provenance: the host grid's data_ptr is a host address, the device
    # grid's data_ptr is a device address. The grid object itself cannot
    # tell them apart.
    host_grid = handle.grid(0)
    device_grid = handle.deviceGrid(0)
    print(f"  host  grid.data_ptr() = {hex(host_grid.data_ptr())} (CPU)")
    print(f"  device grid.data_ptr() = {hex(device_grid.data_ptr())} (GPU)")
    print(f"  device grid.data_ptr() == handle.device_ptr(): "
          f"{device_grid.data_ptr() == handle.device_ptr()}")

    # The device grid is the input to nanovdb.tools.cuda.* ops. Validate
    # it entirely on the device.
    print(f"  tools.cuda.isValid(device_grid) = "
          f"{nanovdb.tools.cuda.isValid(device_grid)}")

    print("WARNING: calling a host-side accessor (e.g. "
          "device_grid.getAccessor().getValue(...)) on a DEVICE grid "
          "dereferences GPU memory on the CPU and SEGFAULTS. Use host_grid "
          "for host reads.")


if __name__ == "__main__":
    main()
