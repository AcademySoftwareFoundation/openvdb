# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Bake a per-voxel value array onto an index grid, on the device.

Python port of the device path of ``ex_index_grid_cuda`` (the GPU half
of ``index_grid_channels.py``). An OnIndex grid stores only topology;
the per-voxel payload lives in a separate linear array indexed by
``grid.valueCount()``. ``nanovdb.tools.cuda.indexToGrid`` fuses the two
on the device — an index grid plus a value array in, a fully-typed
value grid out:

    value_handle = tools.cuda.indexToGrid(d_index_grid, values, stream)

To fill the value array meaningfully we use
``tools.cuda.activeVoxelCoords`` to recover each value slot's
index-space coordinate (the decode companion to the index grid), then
compute a value per voxel with plain CuPy — here the distance from the
origin, turning the shell into a scalar field.

Requires CuPy and a CUDA-capable GPU.

Run with: python index_to_grid_cuda.py
"""
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

    # A hollow sphere shell of voxels, built on the GPU.
    radius = 24.0
    lin = cp.arange(-30, 31, dtype=cp.float32)
    x, y, z = cp.meshgrid(lin, lin, lin, indexing="ij")
    r = cp.sqrt(x * x + y * y + z * z)
    shell = cp.abs(r - radius) < 1.0
    coords = cp.ascontiguousarray(
        cp.stack([x[shell], y[shell], z[shell]], axis=1).astype(cp.int32))
    index_handle = nanovdb.tools.cuda.voxelsToOnIndexGrid(coords, 1.0, 0)
    index_grid = index_handle.deviceGrid(0)

    index_handle.deviceDownload(0, True)
    value_count = index_handle.grid(0).valueCount()
    print(f"OnIndex grid: {index_handle.grid(0).activeVoxelCount()} voxels, "
          f"valueCount={value_count}")

    # Recover the coordinate of every value slot, then compute a value
    # (distance from origin) per voxel with CuPy. Row 0 is the background
    # slot; leave it at zero.
    voxel_coords = cp.zeros((value_count, 3), dtype=cp.int32)
    nanovdb.tools.cuda.activeVoxelCoords(index_grid, voxel_coords, 9, 0)
    cp.cuda.Stream.null.synchronize()
    fc = voxel_coords.astype(cp.float32)
    values = cp.sqrt((fc * fc).sum(axis=1)).astype(cp.float32)
    values[0] = 0.0  # background slot
    values = cp.ascontiguousarray(values)

    value_handle = nanovdb.tools.cuda.indexToGrid(index_grid, values, 0)
    value_grid = value_handle.deviceGrid(0)
    print(f"indexToGrid -> {value_handle.gridType(0)} grid")
    assert nanovdb.tools.cuda.isValid(value_grid, nanovdb.CheckMode.Full)

    # Confirm the CuPy-computed values actually landed in the grid by
    # reading one back on the host: a shell voxel holds its radius.
    value_handle.deviceDownload(0, True)
    sampler = nanovdb.math.createTrilinearSampler(value_handle.grid(0))
    baked = sampler(nanovdb.math.Vec3f(radius, 0.0, 0.0))
    print(f"  baked value at (r,0,0) = {baked:.2f} (shell radius {radius:.0f})")
    assert abs(baked - radius) < 1.0

    print("OK: baked a CuPy-computed value array onto an index grid on device")


if __name__ == "__main__":
    main()
