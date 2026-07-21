# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Index grids, value channels, and blind-data authoring.

Part A ports the host half of ex_index_grid_cuda: a float level set is
re-encoded as a NanoGrid<ValueOnIndex> whose voxels store sequential
uint64 indices, with the original float values copied into a blind-data
channel — then read back by coordinate through a ChannelAccessor,
which the C++ example only does in a CUDA kernel. Part B authors a
brand-new blind-data channel on a grid with tools.CreateNanoGrid and
fills it in place through the writable getBlindData() NumPy view.

Run with: python index_grid_channels.py
"""
import nanovdb


def index_grid_with_channel():
    src = nanovdb.tools.createLevelSetSphere(radius=50.0, name="sphere")
    src_grid = src.grid()

    # channels=1 copies the source values into blind-data channel 0,
    # indexed by the per-voxel uint64 indices the OnIndex grid stores.
    handle = nanovdb.tools.createNanoGridOnIndex(src_grid, channels=1)
    grid = handle.grid()
    print(f"OnIndex grid: valueCount={grid.valueCount()}, "
          f"source activeVoxelCount={src_grid.activeVoxelCount()}")
    assert grid.valueCount() >= src_grid.activeVoxelCount()

    # createChannelAccessor inspects the channel's recorded dataType and
    # returns the matching typed accessor (here: OnIndexFloat...).
    channel = nanovdb.createChannelAccessor(grid, 0)
    print(f"channel accessor: {type(channel).__name__}, "
          f"valueCount={channel.valueCount()}")

    src_acc = src_grid.getAccessor()
    for ijk in (nanovdb.math.Coord(48, 0, 0), nanovdb.math.Coord(0, 50, 0),
                nanovdb.math.Coord(0, 0, 52)):
        via_channel = channel.getValue(ijk)
        via_source = src_acc.getValue(ijk)
        print(f"  {ijk}: channel={via_channel:.3f} source={via_source:.3f} "
              f"(linear offset {channel.getIndex(ijk)})")
        assert via_channel == via_source


def author_blind_data():
    try:
        import numpy as np
    except ImportError:
        print("NumPy not found. Skipping the blind-data authoring section.")
        return
    src = nanovdb.tools.build.FloatGrid(0.0, "authored",
                                        nanovdb.GridClass.Unknown)
    for i in range(8):
        src.setValue(nanovdb.math.Coord(i, 0, 0), float(i))

    # Declare a channel at conversion time; it is allocated zero-filled
    # in the baked grid and filled afterwards through the writable view.
    conv = nanovdb.tools.CreateNanoGrid(src)
    ch = conv.addBlindData("temperature", count=64,
                           dataType=nanovdb.GridType.Float)
    handle = conv.getHandle()
    grid = handle.grid()
    print(f"authored grid has {grid.blindDataCount()} blind-data channel(s)")

    view = grid.getBlindData(ch)
    view[:] = np.linspace(273.0, 373.0, num=64, dtype=np.float32)

    # Re-resolve the channel by name and confirm the writes persisted.
    n = grid.findBlindData("temperature")
    meta = grid.blindMetaData(n)
    stored = grid.getBlindData(n)
    print(f"  {meta.name()!r}: {meta.valueCount} x {meta.dataType}, "
          f"range [{stored.min():.1f}, {stored.max():.1f}]")
    assert n == ch
    assert stored[0] == np.float32(273.0)


def main():
    index_grid_with_channel()
    print()
    author_blind_data()


if __name__ == "__main__":
    main()
