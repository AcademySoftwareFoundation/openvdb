# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Morphology and topology operators on device OnIndex grids.

Python port of the device path shared by ``ex_dilate_nanovdb_cuda``,
``ex_coarsen_nanovdb_cuda``, ``ex_refine_nanovdb_cuda`` and
``ex_merge_nanovdb_cuda``. The C++ examples read a ``.vdb`` through
OpenVDB to source the grid; here the source OnIndex grid is built
on the GPU with ``voxelsToOnIndexGrid``, so the example needs no
OpenVDB and everything stays on the device.

Each operator returns a fresh device ``GridHandle``:

* ``dilateGrid(g, op)``  — grow active topology (op 6 = faces,
  26 = faces+edges+vertices).
* ``coarsenGrid(g)``     — 2x topological downsample.
* ``refineGrid(g)``      — 2x topological upsample.
* ``mergeGrids(a, b)``   — active-mask union (strictly binary).
* ``pruneGrid(g, mask)`` — keep only voxels flagged in a per-leaf
  retain mask (one ``nanovdb::Mask<3>`` = 8 x uint64 per leaf).

Requires CuPy and a CUDA-capable GPU.

Run with: python device_topology_ops.py
"""
import nanovdb


def _active(handle):
    """Download a device handle and return its active voxel count."""
    handle.deviceDownload(0, True)
    return handle.grid(0).activeVoxelCount()


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

    # Source grid: a solid 8x8x8 block of voxels, built on the GPU.
    lin = cp.arange(8, dtype=cp.int32)
    i, j, k = cp.meshgrid(lin, lin, lin, indexing="ij")
    coords = cp.ascontiguousarray(
        cp.stack([i.ravel(), j.ravel(), k.ravel()], axis=1))
    src = nanovdb.tools.cuda.voxelsToOnIndexGrid(coords, 1.0, 0)
    src_grid = src.deviceGrid(0)
    src_active = _active(src)
    print(f"source block: {src_active} active voxels")

    dil6 = nanovdb.tools.cuda.dilateGrid(src_grid, 6, 0)
    dil26 = nanovdb.tools.cuda.dilateGrid(src_grid, 26, 0)
    coarse = nanovdb.tools.cuda.coarsenGrid(src_grid, 0)
    fine = nanovdb.tools.cuda.refineGrid(src_grid, 0)
    a6, a26 = _active(dil6), _active(dil26)
    ac, af = _active(coarse), _active(fine)
    print(f"dilate(faces)     -> {a6} active")
    print(f"dilate(faces+e+v) -> {a26} active")
    print(f"coarsen (2x down) -> {ac} active")
    print(f"refine  (2x up)   -> {af} active")
    assert a6 > src_active and a26 >= a6
    assert ac < src_active < af

    # Union with a shifted copy of the block; the merged topology must
    # cover at least the larger of the two inputs.
    shifted = nanovdb.tools.cuda.voxelsToOnIndexGrid(
        cp.ascontiguousarray(coords + cp.asarray([4, 0, 0], dtype=cp.int32)),
        1.0, 0)
    merged = nanovdb.tools.cuda.mergeGrids(src_grid, shifted.deviceGrid(0), 0)
    am = _active(merged)
    print(f"merge(block, block+4x) -> {am} active")
    assert am > src_active

    # Prune with a retain-all mask (8 uint64 per leaf); topology is
    # unchanged, demonstrating the mask-driven prune entry point.
    src.deviceDownload(0, True)
    leaf_count = src.grid(0).tree().nodeCount(0)
    retain_all = cp.full(leaf_count * 8, 0xFFFFFFFFFFFFFFFF, dtype=cp.uint64)
    pruned = nanovdb.tools.cuda.pruneGrid(src_grid, retain_all, 0)
    ap = _active(pruned)
    print(f"prune(retain-all, {leaf_count} leaves) -> {ap} active")
    assert ap == src_active

    print("OK: dilate / coarsen / refine / merge / prune on the device")


if __name__ == "__main__":
    main()
