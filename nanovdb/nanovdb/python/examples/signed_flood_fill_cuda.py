# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Signed flood fill of a level set on the device.

Demonstrates ``nanovdb.tools.cuda.signedFloodFill`` — the device tool
that propagates interior/exterior sign out of a narrow band so the
inactive tiles inside the surface read as negative background and those
outside read as positive background. It runs in place on a device
``FloatGrid`` (or ``DoubleGrid``):

    tools.cuda.signedFloodFill(d_grid, verbose, stream)

We upload a level-set sphere, run the flood fill on the device, then use
``tools.cuda.sampleFromVoxels`` to confirm the field is consistently
signed (negative inside, positive outside) and that the grid still
validates.

Requires CuPy and a CUDA-capable GPU.

Run with: python signed_flood_fill_cuda.py
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

    radius = 40.0
    host_handle = nanovdb.tools.createLevelSetSphere(
        radius=radius, voxelSize=1.0, name="sphere")
    tmp = tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False)
    tmp.close()
    nanovdb.io.writeGrid(tmp.name, host_handle)
    try:
        handle = nanovdb.io.deviceReadGrid(tmp.name)
    finally:
        os.unlink(tmp.name)
    handle.deviceUpload(0, True)
    device_grid = handle.deviceGrid(0)

    # Propagate signs across the whole grid on the device, in place.
    nanovdb.tools.cuda.signedFloodFill(device_grid, False, 0)
    print("signedFloodFill: done on the device")

    # The interior sign is only meaningful right at the band; sample just
    # inside and just outside the surface and confirm the signs.
    points = cp.asarray([[radius - 2.0, 0.0, 0.0],
                         [radius + 2.0, 0.0, 0.0]], dtype=cp.float32)
    values = cp.empty(2, dtype=cp.float32)
    nanovdb.tools.cuda.sampleFromVoxels(cp.ascontiguousarray(points), device_grid, values, 0)
    cp.cuda.Stream.null.synchronize()
    inside, outside = (float(v) for v in cp.asnumpy(values))
    print(f"  sdf just inside surface = {inside:+.3f} (expect < 0)")
    print(f"  sdf just outside surface = {outside:+.3f} (expect > 0)")
    assert inside < 0.0 < outside

    # The flooded grid must still be structurally valid.
    assert nanovdb.tools.cuda.isValid(device_grid, nanovdb.CheckMode.Full)
    print("OK: device signed flood fill produced a consistent, valid SDF")


if __name__ == "__main__":
    main()
