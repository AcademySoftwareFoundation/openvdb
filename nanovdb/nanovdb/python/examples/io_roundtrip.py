# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Write a bundle of primitive grids to a .nvdb file and read it back.

Python port of ex_write_nanovdb_grids, ex_read_nanovdb_sphere, and
ex_read_nanovdb_sphere_accessor, consolidated into one self-contained
round trip: bake five primitives, write them to a single file (with
codec fallback), inspect the file metadata without loading the grids,
re-read one grid by name and all grids at once, and split a merged
handle back apart. The point-sphere section reads the world-space
point positions through the zero-copy getBlindData() NumPy view.

Run with: python io_roundtrip.py
"""
import os
import tempfile

import nanovdb


def write_primitives(path):
    handles = [
        nanovdb.tools.createLevelSetSphere(radius=50.0, name="sphere_ls"),
        nanovdb.tools.createLevelSetTorus(majorRadius=50.0, minorRadius=20.0,
                                          name="torus_ls"),
        nanovdb.tools.createLevelSetBox(width=40.0, height=60.0, depth=80.0,
                                        name="box_ls"),
        nanovdb.tools.createLevelSetBBox(width=40.0, height=60.0, depth=80.0,
                                         thickness=10.0, name="bbox_ls"),
        nanovdb.tools.createPointSphere(pointsPerVoxel=2, radius=50.0,
                                        name="sphere_points"),
    ]
    # BLOSC gives the best compression but is a build-time option; fall
    # back to an uncompressed file when this module was built without it.
    try:
        nanovdb.io.writeGrids(path, handles, codec=nanovdb.io.Codec.BLOSC)
        codec = "BLOSC"
    except RuntimeError:
        nanovdb.io.writeGrids(path, handles, codec=nanovdb.io.Codec.NONE)
        codec = "NONE"
    print(f"Wrote {len(handles)} grids to {path} (codec={codec})")


def inspect_file(path):
    # readGridMetaData parses the per-grid file headers without loading
    # any voxel data — the cheap way to answer "what's in this file?".
    for meta in nanovdb.io.readGridMetaData(path):
        print(f"  {meta.gridName!r}: type={meta.gridType}, "
              f"class={meta.gridClass}, voxels={meta.voxelCount}")
    assert nanovdb.io.hasGrid(path, "sphere_ls")
    assert not nanovdb.io.hasGrid(path, "no_such_grid")


def read_back(path):
    # Read a single grid by name, then print the recognizable
    # ex_read_nanovdb_sphere_accessor cross-section along the x-axis.
    handle = nanovdb.io.readGrid(path, "sphere_ls")
    acc = handle.grid().getAccessor()
    for i in range(47, 54):
        ijk = nanovdb.math.Coord(i, 0, 0)
        print(f"  sphere_ls({i},0,0) = {acc.getValue(ijk):.2f}")

    # Read every grid, merge them into one multi-grid handle, and split
    # that handle back into one handle per grid.
    handles = nanovdb.io.readGrids(path)
    merged = nanovdb.mergeGrids(handles)
    print(f"  merged handle holds {merged.gridCount()} grids")
    parts = nanovdb.splitGrids(merged)
    assert len(parts) == len(handles)
    print(f"  splitGrids -> {len(parts)} single-grid handles")
    return handles


def point_positions(handles):
    try:
        import numpy as np
    except ImportError:
        print("NumPy not found. Skipping the point-positions section.")
        return
    # The point sphere stores its world-space positions as a blind-data
    # channel; getBlindData() exposes it as a zero-copy (N, 3) view.
    points = next(h.grid() for h in handles
                  if h.grid().gridName() == "sphere_points")
    positions = points.getBlindData(0)
    radii = np.linalg.norm(positions, axis=1)
    print(f"  {positions.shape[0]} points, "
          f"|p| in [{radii.min():.2f}, {radii.max():.2f}]")
    # Points are jittered within their voxel, so allow ~1.5 voxels.
    assert np.all(np.abs(radii - 50.0) < 1.5)


def main():
    out_dir = tempfile.mkdtemp(prefix="nanovdb_")
    path = os.path.join(out_dir, "primitives.nvdb")
    write_primitives(path)
    inspect_file(path)
    handles = read_back(path)
    point_positions(handles)
    print(f"Output left in {out_dir}")


if __name__ == "__main__":
    main()
