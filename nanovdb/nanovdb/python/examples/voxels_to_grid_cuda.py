# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Build a NanoVDB grid directly on the GPU from a CuPy coordinate array.

GPU counterpart to ``build_grid.py`` and a Python port of
``ex_voxels_to_grid_cuda`` (and the device half of
``ex_make_custom_nanovdb_cuda``). Where the host builders write voxels
one at a time through a tree, the device rasterizers take a whole
``(N, 3)`` array already resident on the GPU and construct the grid in
one call, entirely on the device:

* ``nanovdb.tools.cuda.voxelsToOnIndexGrid(coords, voxelSize, stream)``
  rasterizes ``(N, 3)`` int32 INDEX-space voxel coordinates into a
  device ``OnIndexGrid`` handle.
* ``nanovdb.tools.cuda.pointsToGrid(points, voxelSize, stream)``
  rasterizes ``(N, 3)`` float WORLD-space positions into a device
  ``NanoGrid<Point>`` (the point coordinates are stored as blind data).

Key facts demonstrated:

* The returned handle is already DEVICE-resident: ``deviceGrid(0)`` is
  valid immediately with no ``deviceUpload``. ``grid(0)`` (the host
  grid) is ``None`` until you call ``deviceDownload``.
* The device grid feeds straight into other ``nanovdb.tools.cuda.*``
  ops (here, device validation).

Requires CuPy and a CUDA-capable GPU.

Run with: python voxels_to_grid_cuda.py
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

    # Build the voxel coordinates of a hollow sphere shell directly on the
    # GPU with CuPy, so nothing round-trips through the host.
    radius = 32.0
    lin = cp.arange(-40, 41, dtype=cp.float32)
    x, y, z = cp.meshgrid(lin, lin, lin, indexing="ij")
    r = cp.sqrt(x * x + y * y + z * z)
    shell = (cp.abs(r - radius) < 1.0)
    coords = cp.stack([x[shell], y[shell], z[shell]], axis=1).astype(cp.int32)
    coords = cp.ascontiguousarray(coords)
    print(f"Generated {coords.shape[0]} shell voxel coords on the GPU")

    # Rasterize into a device OnIndex grid in one call.
    handle = nanovdb.tools.cuda.voxelsToOnIndexGrid(coords, 1.0, 0)
    device_grid = handle.deviceGrid(0)
    print(f"voxelsToOnIndexGrid -> {handle.gridType(0)} handle, "
          f"device_grid={type(device_grid).__name__}")
    print(f"  grid(0) before download = {handle.grid(0)} (device-built)")

    # Validate entirely on the device, then download for host-side metadata.
    print(f"  tools.cuda.isValid(device_grid) = {nanovdb.tools.cuda.isValid(device_grid)}")
    handle.deviceDownload(0, True)
    host_grid = handle.grid(0)
    active = host_grid.activeVoxelCount()
    print(f"  after deviceDownload: activeVoxelCount={active}, "
          f"valueCount={host_grid.valueCount()}")
    # OnIndex value 0 is the background slot, so valueCount == active + 1.
    assert host_grid.valueCount() == active + 1
    assert active == int(coords.shape[0])

    # pointsToGrid takes WORLD-space float positions and builds a Point grid.
    world_points = coords.astype(cp.float32) * 0.5  # arbitrary world scale
    world_points = cp.ascontiguousarray(world_points)
    point_handle = nanovdb.tools.cuda.pointsToGrid(world_points, 0.5, 0)
    print(f"pointsToGrid -> {point_handle.gridType(0)} handle "
          f"(positions stored as blind data)")
    assert nanovdb.tools.cuda.isValid(point_handle.deviceGrid(0))
    point_handle.deviceDownload(0, True)
    point_active = point_handle.grid(0).activeVoxelCount()
    print(f"  point grid activeVoxelCount={point_active} (isValid on device)")
    assert point_active > 0

    print("OK: built OnIndex and Point grids on the device from CuPy arrays")


if __name__ == "__main__":
    main()
