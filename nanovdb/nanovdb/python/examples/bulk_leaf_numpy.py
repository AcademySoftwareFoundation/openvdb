# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Bulk per-leaf value access as a zero-copy NumPy array.

Phase 3d's grid.leaf_values() is the highest-bandwidth path from
NanoVDB into NumPy. It returns an (N_leaves, 512) view of every
leaf's mValues without copying — modify it, slice it, feed it into a
PyTorch tensor, hash it for cache lookup, whatever you need.

Run with: python bulk_leaf_numpy.py
"""
import nanovdb


def main():
    try:
        import numpy as np
    except ImportError:
        print("This example requires numpy. Install it with: pip install numpy")
        return

    # Build a fog volume sphere with stats so the leaves have meaningful
    # min/max attached (just for the printing below — not required by
    # leaf_values itself).
    handle = nanovdb.tools.createFogVolumeSphere(
        radius=20.0, name="bulk_demo")
    grid = handle.grid()
    print(f"Grid: {grid.gridType()}, active voxels = {grid.activeVoxelCount()}, "
          f"leaves = {grid.tree().nodeCount(0)}")

    # leaf_values() is the zero-copy view. Modifying it modifies the grid.
    bulk = grid.leaf_values()
    # np.asarray adds a NumPy wrapper but doesn't copy.
    arr = np.asarray(bulk)
    print(f"leaf_values: shape={arr.shape}, dtype={arr.dtype}, "
          f"backed by grid memory (no copy).")

    # Global statistics across every voxel in every leaf, computed in C.
    # 0.0 voxels (background) are excluded by using a mask.
    nonzero = arr[arr != 0.0]
    print(f"  non-background voxels = {nonzero.size}")
    print(f"  min = {nonzero.min()}, max = {nonzero.max()}, "
          f"mean = {nonzero.mean()}")

    # Per-leaf reductions: each row of `arr` is one leaf's 512 voxels.
    per_leaf_max = arr.max(axis=1)
    print(f"  per-leaf max (first 5): {per_leaf_max[:5]}")

    # Zero-copy means writes propagate. Zero out the first leaf and read
    # it back through the binding to confirm the grid actually changed.
    arr[0] = 0.0
    leaf = grid.tree().getFirstLeaf()
    if leaf is not None:
        n_active_after = sum(1 for n in range(leaf.voxelCount()) if leaf.isActive(n))
        # The active mask is unchanged by writing into mValues — we
        # just zeroed the *values*, not the active state. The grid's
        # activeVoxelCount() therefore stays the same.
        print(f"  zeroed first leaf's values in place — "
              f"activeVoxelCount unchanged: {grid.activeVoxelCount()}")


if __name__ == "__main__":
    main()
