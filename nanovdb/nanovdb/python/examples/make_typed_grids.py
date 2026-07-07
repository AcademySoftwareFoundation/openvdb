# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Build grids of many value types and read them back polymorphically.

Python port of ex_make_typed_grids: a small solid sphere is baked once
per value type through the matching nanovdb.tools.build.<Suffix>Grid
mutable builder, all handles are written to a single .nvdb file, and
the file is re-read with handle.grid() returning the correct typed
subclass for each grid at runtime.

Run with: python make_typed_grids.py
"""
import os
import tempfile

import nanovdb

RADIUS = 8  # voxels; ~2.1k active voxels per grid keeps this quick


def solid_sphere_coords():
    r2 = RADIUS * RADIUS
    for i in range(-RADIUS, RADIUS + 1):
        for j in range(-RADIUS, RADIUS + 1):
            for k in range(-RADIUS, RADIUS + 1):
                if i * i + j * j + k * k < r2:
                    yield nanovdb.math.Coord(i, j, k)


def build_typed_grids():
    # (builder class, grid name, background, voxel value)
    specs = [
        (nanovdb.tools.build.FloatGrid,  "float_grid",  0.0, 1.0),
        (nanovdb.tools.build.DoubleGrid, "double_grid", 0.0, 1.0),
        (nanovdb.tools.build.Int16Grid,  "int16_grid",  0,   1),
        (nanovdb.tools.build.Int32Grid,  "int32_grid",  0,   1),
        (nanovdb.tools.build.Int64Grid,  "int64_grid",  0,   1),
        (nanovdb.tools.build.UInt32Grid, "uint32_grid", 0,   1),
        (nanovdb.tools.build.Vec3fGrid,  "vec3f_grid",
         nanovdb.math.Vec3f(0.0), nanovdb.math.Vec3f(1.0, 0.0, 0.0)),
    ]
    handles = []
    for cls, name, background, value in specs:
        grid = cls(background, name, nanovdb.GridClass.Unknown)
        for ijk in solid_sphere_coords():
            grid.setValue(ijk, value)
        handles.append(grid.to_nanovdb())
        print(f"built {name} ({cls.__name__})")
    return handles


def main():
    handles = build_typed_grids()

    out_dir = tempfile.mkdtemp(prefix="nanovdb_")
    path = os.path.join(out_dir, "custom_types.nvdb")
    nanovdb.io.writeGrids(path, handles)
    print(f"Wrote {len(handles)} grids to {path}")

    # Re-read and dispatch: handle.grid() returns FloatGrid, Int16Grid,
    # Vec3fGrid, ... according to the GridType each grid carries.
    probe = nanovdb.math.Coord(0, 0, 0)
    for handle in nanovdb.io.readGrids(path):
        grid = handle.grid()
        value = grid.getAccessor().getValue(probe)
        print(f"  {grid.gridName():<12} -> {type(grid).__name__:<12} "
              f"active={grid.activeVoxelCount()} value(0,0,0)={value}")
        assert grid.activeVoxelCount() > 0


if __name__ == "__main__":
    main()
