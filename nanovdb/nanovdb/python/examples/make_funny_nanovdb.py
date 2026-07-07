# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Build a level set from a Python function evaluated at every voxel.

Python port of ex_make_funny_nanovdb: a trigonometric interference
pattern is CSG-intersected with a sphere and clamped to a narrow band,
using the functor-based nanovdb.tools.createFloatGrid factory. The
callback is one Python call per voxel in the bbox, so this port uses a
65^3 domain (~275k calls, a few seconds) where the C++ original fills
[-500,500]^3 — scale `SIZE` up only if you are willing to wait.

Run with: python make_funny_nanovdb.py
"""
import math
import os
import tempfile

import nanovdb

SIZE = 32          # half-width of the cubic domain, in voxels
BACKGROUND = 5.0   # narrow-band half-width, in world units
FREQ = 0.8         # rescaled from the C++ 0.1 to fit the smaller domain


def funny(ijk):
    v = 4.0 + 5.0 * (math.cos(ijk.x * FREQ) * math.sin(ijk.y * FREQ) +
                     math.cos(ijk.y * FREQ) * math.sin(ijk.z * FREQ) +
                     math.cos(ijk.z * FREQ) * math.sin(ijk.x * FREQ))
    # CSG intersection with a sphere of radius SIZE.
    r = math.sqrt(ijk.x ** 2 + ijk.y ** 2 + ijk.z ** 2)
    v = max(v, r - SIZE)
    # Clamp to the narrow band.
    return max(-BACKGROUND, min(BACKGROUND, v))


def main():
    bbox = nanovdb.math.CoordBBox(nanovdb.math.Coord(-SIZE),
                                  nanovdb.math.Coord(SIZE))
    print(f"Evaluating funny() over {bbox} ...")
    handle = nanovdb.tools.createFloatGrid(
        BACKGROUND, "funny", nanovdb.GridClass.LevelSet, funny, bbox)

    grid = handle.grid()
    print(f"activeVoxelCount = {grid.activeVoxelCount()}")
    acc = grid.getAccessor()
    probe = nanovdb.math.Coord(0, 0, 0)
    print(f"value at {probe} = {acc.getValue(probe):.3f}")
    assert grid.isLevelSet()
    assert grid.activeVoxelCount() > 0

    out_dir = tempfile.mkdtemp(prefix="nanovdb_")
    path = os.path.join(out_dir, "funny.nvdb")
    try:
        nanovdb.io.writeGrid(path, handle, codec=nanovdb.io.Codec.BLOSC)
    except RuntimeError:
        nanovdb.io.writeGrid(path, handle, codec=nanovdb.io.Codec.NONE)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
