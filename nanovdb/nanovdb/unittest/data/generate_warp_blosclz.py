# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Generate the Warp BloscLZ fixture. See README.md for pinned dependencies."""

import argparse
from pathlib import Path
import struct

import blosc
import numpy as np
import warp as wp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("warp_blosclz.nvdb"))
args = parser.parse_args()

wp.init()
blosc.set_nthreads(1)
i, j, k = np.indices((8, 8, 8))
values = (100 * i + 10 * j + k).astype(np.float32)
volume = wp.Volume.load_from_numpy(
    values, min_world=(1.0, 2.0, 3.0), voxel_size=0.5, bg_value=-1.0, device="cpu"
)
# Do not override the compressor: this test covers Warp's BloscLZ default.
volume.save_to_nvdb(str(args.output), codec="blosc")

file_data = args.output.read_bytes()
_, _, grid_count, codec = struct.unpack_from("<QIHH", file_data)
assert grid_count == 1 and codec == 2
name_size = struct.unpack_from("<I", file_data, 16 + 136)[0]
offset = 16 + 176 + name_size
chunk_size = struct.unpack_from("<Q", file_data, offset)[0]
chunk = file_data[offset + 8 : offset + 8 + chunk_size]
assert blosc.get_clib(chunk) == "BloscLZ"
assert not chunk[2] & 2, "The fixture must contain compressed blocks."
print(f"Warp {wp.__version__}; Python Blosc {blosc.__version__}; C Blosc {blosc.blosclib_version}")
print(f"Wrote {len(file_data)} bytes to {args.output}")
