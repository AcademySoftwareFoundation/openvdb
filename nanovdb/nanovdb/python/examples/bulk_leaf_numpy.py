# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Bulk per-leaf value and active-mask access as zero-copy NumPy arrays.

grid.leafValues() is the highest-bandwidth path from NanoVDB into
NumPy. It returns an (N_leaves, 512) view of every leaf's mValues
without copying — modify it, slice it, feed it into a PyTorch tensor,
hash it for cache lookup, whatever you need.

grid.leaf_active_masks() is the same idea for activity: an
(N_leaves, 8) uint64 view of every leaf's 512-bit active-value mask.
Active and value are independent in VDB — a voxel can be active with
value 0.0, or inactive with a stale nonzero value — so filtering by
value (`arr != 0.0`) is not a substitute for checking the mask.

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
    # leafValues itself).
    handle = nanovdb.tools.createFogVolumeSphere(
        radius=20.0, name="bulk_demo")
    grid = handle.grid()
    print(f"Grid: {grid.gridType()}, active voxels = {grid.activeVoxelCount()}, "
          f"leaves = {grid.tree().nodeCount(0)}")

    # leafValues() is the zero-copy view. Modifying it modifies the grid.
    bulk = grid.leafValues()
    # np.asarray adds a NumPy wrapper but doesn't copy.
    arr = np.asarray(bulk)
    print(f"leafValues: shape={arr.shape}, dtype={arr.dtype}, "
          f"backed by grid memory (no copy).")

    # leaf_active_masks() is the activity counterpart: (N_leaves, 8)
    # uint64, one 512-bit mask per leaf, same row order as leaf_values().
    # Unpack it into a (N_leaves, 512) bool array so it lines up with arr.
    mask_words = np.asarray(grid.leaf_active_masks())
    active = np.unpackbits(mask_words.view(np.uint8),
                            bitorder="little").reshape(arr.shape).astype(bool)

    # Global statistics across every active voxel, computed in C for the
    # bulk read and in NumPy for the filter. This is the correct way to
    # exclude background/inactive voxels — filtering by value (arr != 0.0)
    # would also drop legitimately active voxels whose value happens to be
    # 0.0, and would keep inactive voxels with stale nonzero values.
    active_values = arr[active]
    print(f"  active leaf voxels = {active_values.size} "
          f"(grid.activeVoxelCount() = {grid.activeVoxelCount()} — larger "
          f"here because this sphere also has active *tiles* on internal "
          f"nodes, which leaf_active_masks() does not cover)")
    print(f"  min = {active_values.min()}, max = {active_values.max()}, "
          f"mean = {active_values.mean()}")

    # Per-leaf reductions: each row of `arr` is one leaf's 512 voxels.
    per_leaf_max = arr.max(axis=1)
    print(f"  per-leaf max (first 5): {per_leaf_max[:5]}")

    # Zero-copy means writes propagate. Zero out the first leaf's values
    # and read one back through the regular accessor to confirm the grid
    # actually changed. (The active *mask* is unchanged — we wrote into
    # mValues only — so activeVoxelCount() stays the same.)
    arr[0] = 0.0
    leaf = grid.tree().getFirstLeaf()
    if leaf is not None:
        first_value_after = leaf.getFirstValue()
        print(f"  zeroed first leaf's values in place: "
              f"leaf.getFirstValue() = {first_value_after}, "
              f"activeVoxelCount unchanged: {grid.activeVoxelCount()}")


if __name__ == "__main__":
    main()
