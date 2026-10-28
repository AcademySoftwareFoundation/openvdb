# Warp BloscLZ fixture

`warp_blosclz.nvdb` was written by `warp.Volume.save_to_nvdb(codec="blosc")`.
Warp calls Python Blosc without selecting a compressor. Its default is BloscLZ.
The stored chunk is compressed, rather than a Blosc memcpy buffer.

The fixture contains one float grid with 512 active voxels. Coordinates range
from (0, 0, 0) through (7, 7, 7). The value at (i, j, k) is 100*i + 10*j + k.
The background is -1, the voxel size is 0.5, and the translation is (1, 2, 3).
The grid name is empty and its class is Unknown.

It was generated on macOS arm64 with Python 3.12, Warp 1.18.0, Python Blosc
1.11.4, and NumPy 2.5.4. Python Blosc reported C Blosc 1.21.7.dev
(`$Date:: 2024-06-24 #$`). The file uses NanoVDB version 32.3.3.

To regenerate it from the repository root:

```sh
python3 -m venv /tmp/warp-blosclz-fixture
/tmp/warp-blosclz-fixture/bin/python -m pip install warp-lang==1.18.0 blosc==1.11.4 numpy==2.5.4
/tmp/warp-blosclz-fixture/bin/python nanovdb/nanovdb/unittest/data/generate_warp_blosclz.py
```

The generator verifies the compressor and rejects uncompressed chunks.
Grid padding or compressor builds can change the file bytes across platforms.
The unit test checks grid contents rather than requiring byte-identical output.
CI reads the checked-in file and does not need Python, Warp, or external Blosc.

SHA-256 of the checked-in fixture:

`a741e8b4c75afdf74dcd46566efda01a0eed91806689e139202c7c154c653586`

The fixture and generator are provided under Apache-2.0, like the unit tests.
