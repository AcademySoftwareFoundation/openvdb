# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Load a NanoVDB grid and inspect it polymorphically.

handle.grid(i) returns the correct typed Python subclass for whatever
GridType the grid carries, so a single call site can handle a mixed
bundle of grid types. This example builds two grids of different
value types into one handle, then walks the handle inspecting each
grid via the polymorphic accessor plus GridMetaData (the type-erased
introspector that answers "what's in this buffer?" without knowing
BuildT at compile time).

Run with: python load_inspect.py
"""
import nanovdb


def describe_handle(handle):
    print(f"Handle contains {handle.gridCount()} grid(s).")
    for i in range(handle.gridCount()):
        # GridType / gridSize are cheap to query on the handle itself.
        gtype = handle.gridType(i)
        gsize = handle.gridSize(i)
        print(f"  [{i}] type={gtype}, size={gsize} bytes")

        # handle.grid(i) returns the matching <Suffix>Grid subclass
        # at runtime — no isinstance dispatch needed at the call site.
        # The grid name lives on the grid itself, not the handle.
        grid = handle.grid(i)
        print(f"      name={grid.gridName()!r}, "
              f"gridClass={grid.gridClass()}")
        print(f"      activeVoxelCount={grid.activeVoxelCount()}")
        print(f"      worldBBox={grid.worldBBox()}")

        # GridMetaData is the type-erased introspector — answers
        # "what's in this buffer?" without knowing BuildT.
        meta = nanovdb.GridMetaData(grid)
        print(f"      gridSize={meta.gridSize()}, "
              f"isLevelSet={meta.isLevelSet()}, "
              f"isFogVolume={meta.isFogVolume()}")


def main():
    # Build two grids of different types into one handle so we can
    # exercise the polymorphic accessor.
    h_float = nanovdb.tools.createLevelSetSphere(
        radius=10.0, name="sphere_float")
    h_double = nanovdb.tools.createLevelSetSphere(
        gridType=nanovdb.GridType.Double, radius=10.0, name="sphere_double")
    handle = nanovdb.mergeGrids([h_float, h_double])

    describe_handle(handle)

    print()
    print("Polymorphic dispatch from a runtime GridType:")
    for i in range(handle.gridCount()):
        grid = handle.grid(i)
        # Each typed grid carries a getAccessor() that returns the
        # appropriate <Suffix>ReadAccessor — float for FloatGrid,
        # double for DoubleGrid, etc.
        acc = grid.getAccessor()
        v = acc.getValue(nanovdb.math.Coord(0, 0, 0))
        print(f"  grid[{i}] accessor.getValue(0,0,0) = {v}")


if __name__ == "__main__":
    main()
