# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Round-trip a grid between OpenVDB and NanoVDB.

Python port of ex_openvdb_to_nanovdb and ex_openvdb_to_nanovdb_accessor:
an OpenVDB level-set sphere is converted to NanoVDB with openToNanoVDB,
values are compared through both libraries' accessors, and the NanoVDB
grid is converted back with nanoToOpenVDB. Requires a build with
NANOVDB_USE_OPENVDB and an importable `openvdb` module; the script
skips (exit 0) when either is missing.

Run with: python openvdb_interop.py
"""
import nanovdb


def main():
    if not hasattr(nanovdb.tools, "openToNanoVDB"):
        print("nanovdb was built without NANOVDB_USE_OPENVDB. Skipping...")
        return
    try:
        import openvdb
    except ImportError:
        print("openvdb not found. Skipping...")
        return

    sphere = openvdb.createLevelSetSphere(100.0)
    sphere.name = "sphere"
    handle = nanovdb.tools.openToNanoVDB(sphere)
    grid = handle.grid()
    print(f"openToNanoVDB: {grid.gridName()!r}, class={grid.gridClass()}, "
          f"active={grid.activeVoxelCount()}")
    assert grid.gridClass() == nanovdb.GridClass.LevelSet

    # Compare a cross-section through both accessors, as the C++
    # accessor example does.
    open_acc = sphere.getAccessor()
    nano_acc = grid.getAccessor()
    for i in range(97, 104):
        open_v = open_acc.getValue((i, 0, 0))
        nano_v = nano_acc.getValue(nanovdb.math.Coord(i, 0, 0))
        print(f"  ({i},0,0): openvdb={open_v:.3f} nanovdb={nano_v:.3f}")
        assert abs(open_v - nano_v) < 1e-5

    # And back: NanoVDB -> OpenVDB.
    back = nanovdb.tools.nanoToOpenVDB(handle)
    print(f"nanoToOpenVDB: {back.name!r}, empty={back.empty()}")
    assert back.name == "sphere"
    assert not back.empty()


if __name__ == "__main__":
    main()
