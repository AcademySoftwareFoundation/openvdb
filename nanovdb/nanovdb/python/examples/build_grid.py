# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Construct a NanoVDB grid voxel-by-voxel in pure Python.

nanovdb.tools.build.<Suffix>Grid is the mutable CPU grid builder
that mirrors nanovdb::tools::build::Grid<T>. This example shows the
three construction loops you'll typically reach for (setValue
directly, the cached ValueAccessor, and the thread-safe
WriteAccessor), then bakes each build grid into a host NanoGrid via
.to_nanovdb() and reads it back through the regular polymorphic
handle.grid() API.

Run with: python build_grid.py
"""
import nanovdb


def fill_with_setValue():
    """Simplest path: setValue() directly on the build grid."""
    g = nanovdb.tools.build.FloatGrid(
        background=0.0, name="setValue_demo",
        gridClass=nanovdb.GridClass.FogVolume)
    # Plant five active voxels along the x axis.
    for i in range(5):
        g.setValue(nanovdb.math.Coord(i, 0, 0), float(i + 1))
    return g


def fill_with_accessor():
    """Cached path: getAccessor() avoids re-walking the tree for
    each write when consecutive coordinates share a leaf."""
    g = nanovdb.tools.build.FloatGrid(background=0.0, name="accessor_demo")
    acc = g.getAccessor()
    # The accessor caches the last leaf / lower / upper node, so a
    # burst of writes in a 16^3 neighborhood only walks the tree once.
    for x in range(8):
        for y in range(8):
            for z in range(8):
                if x + y + z == 7:
                    acc.setValue(nanovdb.math.Coord(x, y, z),
                                 float(x * 64 + y * 8 + z))
    return g


def fill_with_write_accessor():
    """Thread-safe path: getWriteAccessor() buffers writes into a
    private root and merges them into the parent on destruction or
    on an explicit .merge() call. Useful when fanning out to multiple
    threads — one WriteAccessor per thread, no shared mutable state."""
    g = nanovdb.tools.build.FloatGrid(background=0.0, name="write_accessor_demo")
    wa = g.getWriteAccessor()
    wa.setValue(nanovdb.math.Coord(50, 50, 50), 9.0)
    # Before merge, the parent doesn't see the change yet.
    assert g.getValue(nanovdb.math.Coord(50, 50, 50)) == 0.0
    wa.merge()
    assert g.getValue(nanovdb.math.Coord(50, 50, 50)) == 9.0
    return g


def main():
    for builder in (fill_with_setValue, fill_with_accessor,
                    fill_with_write_accessor):
        g = builder()
        print(f"=== {g.getName()} ===")
        print(f"  nodeCount (leaf, lower, upper) = {g.nodeCount()}")

        # Bake into a host NanoGrid<float>.
        handle = g.to_nanovdb(sMode=nanovdb.tools.StatsMode.All)
        ng = handle.grid()
        print(f"  baked NanoGrid: type={ng.gridType()}, "
              f"active={ng.activeVoxelCount()}, "
              f"worldBBox={ng.worldBBox()}")

        # The build grid is left untouched — we can bake again.
        handle2 = g.to_nanovdb()
        assert handle2.grid().activeVoxelCount() == ng.activeVoxelCount()


if __name__ == "__main__":
    main()
