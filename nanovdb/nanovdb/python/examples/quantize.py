# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Quantize a NanoGrid<float> down to a quantized BuildT.

Phase 5b binds tools.createNanoGridFp4 / Fp8 / Fp16 / FpN — the
quantized counterparts of the regular grid types. Fp4 / Fp8 / Fp16
use a fixed bit-width per voxel. FpN picks the bit-width per leaf so
each leaf hits a user-supplied tolerance (the "oracle"). Smaller bit
widths give smaller files at the cost of approximation error.

Run with: python quantize.py
"""
import nanovdb


def gridSize_in_kb(handle):
    return handle.gridSize(0) / 1024


def main():
    # Source: a 50-radius sphere fog volume at full float precision.
    src_handle = nanovdb.tools.createFogVolumeSphere(radius=50.0)
    src_grid = src_handle.grid()
    print(f"Source FloatGrid: {gridSize_in_kb(src_handle):.1f} KB, "
          f"active voxels = {src_grid.activeVoxelCount()}")

    # Fixed-width quantization. Each subsequent format roughly halves
    # the per-voxel storage cost; dithering optional.
    for fn_name, label in [
        ("createNanoGridFp16", "Fp16 (16-bit fixed)"),
        ("createNanoGridFp8",  "Fp8  (8-bit fixed)"),
        ("createNanoGridFp4",  "Fp4  (4-bit fixed)"),
    ]:
        h = getattr(nanovdb.tools, fn_name)(src_grid, ditherOn=True)
        print(f"  {label}: {gridSize_in_kb(h):.1f} KB, "
              f"active voxels = {h.grid().activeVoxelCount()}")

    # Variable-width FpN. The oracle picks the per-leaf bit-width to
    # meet a tolerance — AbsDiff for absolute error, RelDiff for
    # relative error. -1 tolerance means "uninitialized" so we pass
    # an explicit value.
    abs_oracle = nanovdb.tools.AbsDiff(0.05)  # ±0.05 per voxel
    h_fpn_abs = nanovdb.tools.createNanoGridFpN(src_grid, abs_oracle)
    print(f"  FpN (AbsDiff 0.05): {gridSize_in_kb(h_fpn_abs):.1f} KB")

    rel_oracle = nanovdb.tools.RelDiff(0.1)   # 10% relative error
    h_fpn_rel = nanovdb.tools.createNanoGridFpN(src_grid, rel_oracle)
    print(f"  FpN (RelDiff 0.10): {gridSize_in_kb(h_fpn_rel):.1f} KB")

    # The output is a regular NanoGrid<FpN> — read-only, but exposes
    # the standard surface (gridType, activeVoxelCount, accessor.getValue
    # returning a decoded float, etc).
    fpn = h_fpn_abs.grid()
    print(f"\nFpN grid type: {fpn.gridType()}, "
          f"voxel at (0,0,0) = {fpn.getAccessor().getValue(nanovdb.math.Coord(0, 0, 0))}")


if __name__ == "__main__":
    main()
