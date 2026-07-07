# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Validate a grid, and its checksum and stats, entirely on the device.

GPU counterpart to ``validate.py``. Once a grid is resident on the GPU
(``deviceReadGrid`` + ``deviceUpload``), the quality-control tools run
in place on the device grid — no download required:

* ``tools.cuda.isValid(g, mode)``          — structural + checksum check.
* ``tools.cuda.evalChecksum(g, mode)``     — compute the CRC checksum.
* ``tools.cuda.updateChecksum(g, mode)``   — recompute it in place.
* ``tools.cuda.validateChecksum(g, mode)`` — verify it matches.
* ``tools.cuda.updateGridStats(g, mode)``  — recompute min/max/avg/bbox.

``CheckMode`` trades coverage for speed (``Partial`` vs ``Full``);
``StatsMode`` selects how much of the per-node stats to refresh.

Requires CuPy and a CUDA-capable GPU.

Run with: python validate_cuda.py
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
        import cupy  # noqa: F401  (only needed to confirm a usable GPU stack)
    except ImportError:
        print("This example requires CuPy. Install it with: pip install cupy")
        return

    host_handle = nanovdb.tools.createLevelSetSphere(
        radius=40.0, voxelSize=1.0, name="sphere")
    tmp = tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False)
    tmp.close()
    nanovdb.io.writeGrid(tmp.name, host_handle)
    try:
        handle = nanovdb.io.deviceReadGrid(tmp.name)
    finally:
        os.unlink(tmp.name)
    handle.deviceUpload(0, True)
    device_grid = handle.deviceGrid(0)

    # Structural validation on the device, partial and full.
    partial = nanovdb.tools.cuda.isValid(device_grid, nanovdb.CheckMode.Partial)
    full = nanovdb.tools.cuda.isValid(device_grid, nanovdb.CheckMode.Full)
    print(f"isValid: Partial={partial}, Full={full}")
    assert partial and full

    # Checksum round-trip: recompute in place, then verify it matches.
    nanovdb.tools.cuda.updateChecksum(device_grid, nanovdb.CheckMode.Full)
    checksum = nanovdb.tools.cuda.evalChecksum(device_grid, nanovdb.CheckMode.Full)
    ok = nanovdb.tools.cuda.validateChecksum(device_grid, nanovdb.CheckMode.Full)
    print(f"checksum: {checksum} validateChecksum(Full)={ok}")
    assert ok

    # Recompute per-node statistics on the device.
    nanovdb.tools.cuda.updateGridStats(device_grid)
    print("updateGridStats: OK")
    assert nanovdb.tools.cuda.isValid(device_grid, nanovdb.CheckMode.Full)

    print("OK: device-side validation, checksum, and stats all pass")


if __name__ == "__main__":
    main()
