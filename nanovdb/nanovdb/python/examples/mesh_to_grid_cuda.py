# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Rasterize a triangle mesh into a narrow-band SDF on the GPU.

Python port of the device path of ``ex_mesh_to_grid_cuda``.
``nanovdb.tools.cuda.meshToGrid`` takes device arrays of mesh vertices and
triangle indices and builds, entirely on the GPU, a narrow-band ``OnIndex``
grid plus a per-value unsigned-distance-field (UDF) sidecar. The C++ example
sources its mesh through OpenVDB; here the mesh (a closed box) is generated
with NumPy, so the example needs no OpenVDB.

The returned pieces chain naturally:

* ``handle`` — a device ``OnIndex`` grid holding the narrow-band topology.
* ``udf`` — a ``nanovdb.cuda.DeviceBuffer`` of ``valueCount`` float32 unsigned
  distances (voxel units), indexed by the grid's per-voxel value index. That
  is exactly the value array ``tools.cuda.indexToGrid`` consumes, so we bake
  the UDF into a Float distance grid and sample it back to verify.

Requires CuPy and a CUDA-capable GPU.

Run with: python mesh_to_grid_cuda.py
"""
import nanovdb


def _box_mesh(cp, half):
    """Return (vertices (8,3) float32, triangles (12,3) int32) for a cube."""
    corners = cp.asarray([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=cp.float32) * half
    # 12 triangles (2 per face), outward winding.
    tris = cp.asarray([
        [0, 3, 2], [0, 2, 1],  # -z
        [4, 5, 6], [4, 6, 7],  # +z
        [0, 1, 5], [0, 5, 4],  # -y
        [3, 7, 6], [3, 6, 2],  # +y
        [0, 4, 7], [0, 7, 3],  # -x
        [1, 2, 6], [1, 6, 5],  # +x
    ], dtype=cp.int32)
    return cp.ascontiguousarray(corners), cp.ascontiguousarray(tris)


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

    half = 40.0
    voxel_size = 1.0
    vertices, triangles = _box_mesh(cp, half)
    print(f"Box mesh: {vertices.shape[0]} vertices, {triangles.shape[0]} "
          f"triangles, half-extent {half:.0f}")

    # Build the narrow-band UDF on the device in one call.
    handle, udf = nanovdb.tools.cuda.meshToGrid(
        vertices, triangles, voxel_size, 3.0, "box_udf", 0)
    index_grid = handle.deviceGrid(0)
    assert nanovdb.tools.cuda.isValid(index_grid)

    handle.deviceDownload(0, True)
    value_count = handle.grid(0).valueCount()
    # The sidecar is a device buffer of `value_count` float32 UDF values.
    udf_values = cp.asarray(udf).view(cp.float32)[:value_count]
    udf_values = cp.ascontiguousarray(udf_values)
    print(f"meshToGrid -> {handle.gridType(0)} grid, "
          f"{handle.grid(0).activeVoxelCount()} narrow-band voxels, "
          f"UDF range [{float(udf_values[1:].min()):.2f}, "
          f"{float(udf_values[1:].max()):.2f}] voxels")

    # Bake the UDF into a Float distance grid on the device and sample it.
    dist_handle = nanovdb.tools.cuda.indexToGrid(index_grid, udf_values, 0)
    dist_grid = dist_handle.deviceGrid(0)
    assert nanovdb.tools.cuda.isValid(dist_grid, nanovdb.CheckMode.Full)

    # A point right on a face should read a near-zero unsigned distance.
    dist_handle.deviceDownload(0, True)
    sampler = nanovdb.math.createTrilinearSampler(dist_handle.grid(0))
    on_face = sampler(nanovdb.math.Vec3f(half, 0.0, 0.0))
    print(f"  baked UDF at a box face = {on_face:.3f} voxels (expect near 0)")
    assert on_face < 2.0

    print("OK: built a narrow-band SDF from a triangle mesh on the device")


if __name__ == "__main__":
    main()
