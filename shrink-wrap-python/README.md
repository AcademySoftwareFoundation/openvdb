# OpenVDB Python Examples

This directory contains example scripts demonstrating the use of OpenVDB's Python bindings.

## Available Examples

### shrink_wrap.py

Converts USD mesh files to OpenVDB level sets with optional mesh reconstruction.

#### Features

- **USD Mesh Reading**: Uses Pixar's USD library (`pxr.Usd`) to read mesh geometry
- **Level Set Conversion**: Converts polygon soup to level set volumes with LOD pyramid generation
- **Multiple Conversion Modes**:
  - Voxel size-based (automatic or manual bbox)
  - Dimension-based with explicit bounding box
  - Supports both triangles and quads
- **Mesh Reconstruction**: Optional conversion back to mesh with adaptive polygonization
- **Output Formats**:
  - VDB files with multiple LOD grids
  - USD mesh files

#### Dependencies

```bash
# Required
pip install numpy usd-core

# OpenVDB Python bindings must be built and available
# See main OpenVDB documentation for building Python bindings
```

#### Usage

**Basic conversion to VDB** (creates `input-out.vdb`):
```bash
python shrink_wrap.py input.usd --voxel 0.1
```

**Convert to VDB with custom parameters:**
```bash
python shrink_wrap.py input.usd \
    --voxel 0.05 \
    --erode 10.0 \
    --half-width 4.0
```

**Use dimension-based sizing:**
```bash
python shrink_wrap.py input.usd \
    --dim 256 \
    --bbox-min -10 -10 -10 \
    --bbox-max 10 10 10
```

**Convert to mesh with adaptivity** (creates `input-out.usd`):
```bash
python shrink_wrap.py input.usd \
    --to-mesh \
    --voxel 0.1 \
    --adaptivity 0.005
```

**Save only finest grid:**
```bash
python shrink_wrap.py input.usd \
    --voxel 0.1 \
    --finest-only
```

#### Command-Line Arguments

**Required:**
- `input.usd`: Input USD mesh file

**Output:**
- Output filename is auto-generated from input filename:
  - Default (VDB): `<input_basename>-out.vdb`
  - With `--to-mesh`: `<input_basename>-out.usd`
  - Multi-LOD mesh: `<input_basename>-out_lod0.usd`, `<input_basename>-out_lod1.usd`, etc.
- Output is created in the same directory as the input file

**Conversion Mode:**
- `--to-mesh`: Convert grids back to mesh (output as USD)

**Level Set Parameters (one required):**
- `--voxel FLOAT`: Voxel size for finest level set (required, uses minVoxelSize + bbox)
- `--dim INT`: Grid dimension for finest level (uses dimension + bbox)
- `--bbox-min X Y Z`: Bounding box minimum (optional, auto-computed if not provided)
- `--bbox-max X Y Z`: Bounding box maximum (optional, auto-computed if not provided)
- `--erode FLOAT`: Maximum deformation allowed (default: 8.0)
- `--threshold FLOAT`: Closing threshold (default: 0.0)
- `--half-width FLOAT`: Narrow band half-width in voxels (default: 3.0)

**Mesh Conversion Parameters:**
- `--adaptivity FLOAT`: Mesh adaptivity for convertToPolygons (0-1, default: 0.0)
  - 0 = high detail (no simplification)
  - 1 = maximum simplification
- `--isovalue FLOAT`: Isovalue for mesh extraction (default: 0.0)

**Output Options:**
- `--finest-only`: Only write finest grid (do not write entire LOD pyramid)

#### Technical Details

**LOD Pyramid Generation:**

The script uses OpenVDB's `convertPolygonSoupToLevelSet` which generates multiple level-of-detail grids:
- Grid 0: Finest resolution (smallest voxel size)
- Grid 1+: Progressively coarser resolutions

The LOD pyramid automatically generates multiple resolution levels based on the input parameters.

**Two Conversion Modes:**

1. **Voxel Size Mode**:
   ```bash
   --voxel 0.1
   ```
   Specifies minimum voxel size. Bounding box is auto-computed from mesh if not provided.
   Can optionally specify explicit bbox:
   ```bash
   --voxel 0.1 --bbox-min -10 -10 -10 --bbox-max 10 10 10
   ```

2. **Dimension Mode**:
   ```bash
   --dim 256
   ```
   Uses grid dimension to compute voxel size from bounding box. Bounding box is auto-computed from mesh if not provided.
   Can optionally specify explicit bbox:
   ```bash
   --dim 256 --bbox-min -10 -10 -10 --bbox-max 10 10 10
   ```

**USD I/O:**

The script uses Pixar's USD library for both reading and writing:
- Reads arbitrary USD mesh prims
- Handles triangles, quads, and n-gons (triangulated with fan method)

**Mesh Topology:**

- Input: Supports mixed triangle/quad meshes and n-gons
- N-gons (faces with 5+ vertices) are automatically triangulated using fan triangulation

#### Examples

**Convert a simple mesh** (creates `sphere-out.vdb`):
```bash
python shrink_wrap.py sphere.usd --voxel 0.5
```

**High-resolution conversion with custom parameters** (creates `detailed_mesh-out.vdb`):
```bash
python shrink_wrap.py detailed_mesh.usd \
    --voxel 0.01 \
    --erode 5.0 \
    --half-width 5.0
```

**Convert to simplified mesh** (creates `input-out.usd`):
```bash
python shrink_wrap.py input.usd \
    --to-mesh \
    --voxel 0.1 \
    --adaptivity 0.005
```

**Create multiple LOD meshes** (creates `input-out_lod0.usd`, `input-out_lod1.usd`, etc.):
```bash
# Without --finest-only, this creates multiple LOD files
python shrink_wrap.py input.usd \
    --to-mesh \
    --voxel 0.1
```

#### Verifying Output

**VDB files:**
```bash
# View grid metadata
vdb_print input-out.vdb

# Visualize (if available)
vdb_view input-out.vdb
```

**USD files:**
```bash
# View mesh (requires usd-core with usdview)
usdview input-out.usd
```

#### Troubleshooting

**Error: "pxr.Usd not found"**
```bash
pip install usd-core
```

**Error: "openvdb module not found"**

Build OpenVDB Python bindings:
```bash
cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/openvdb -DOPENVDB_BUILD_CORE=ON -DOPENVDB_BUILD_PYTHON_MODULE=ON -DOPENVDB_BUILD_VDB_LOD=ON -DOPENVDB_BUILD_VDB_RENDER=ON -DOPENVDB_BUILD_VDB_VIEW=ON -DOPENVDB_BUILD_VDB_TOOL=ON ..
make -j4
make install
```

**Error: "No mesh found in USD file"**

Check that your USD file contains a mesh prim:
```bash
usdcat input.usd  # View USD contents
```

#### Performance Tips

- **Voxel Size**: Larger voxel sizes = faster conversion, less memory
- **Erode Parameter**: Lower values = faster but may lose thin features
- **Closing Threshold Parameter**: Feature size for closing holes
- **Finest Only**: Use `--finest-only` if you don't need LOD pyramid
- **Adaptivity**: Higher values = faster mesh output but less detail

#### See Also

- OpenVDB Documentation: https://www.openvdb.org/documentation/
- USD Documentation: https://openusd.org/
- `vdb_tool` command-line utility for additional VDB operations
