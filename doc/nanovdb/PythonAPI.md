# NanoVDB Python API Reference

This is a narrative reference to the Python surface produced by the
[bindings restructure plan](../../nanovdb/nanovdb-python-plan.md).
For full signatures and per-argument docstrings, use Python's `help()`
on any symbol (e.g. `help(nanovdb.tools.createNanoGridFpN)`); the
bindings ship `.pyi` type stubs for IDE / type-checker support.

The module structure mirrors the C++ namespaces one-for-one:

| Python | C++ |
| --- | --- |
| `nanovdb` (root) | `nanovdb::` |
| `nanovdb.math` | `nanovdb::math::` |
| `nanovdb.io` | `nanovdb::io::` |
| `nanovdb.tools` | `nanovdb::tools::` |
| `nanovdb.tools.build` | `nanovdb::tools::build::` |
| `nanovdb.tools.cuda` | `nanovdb::tools::cuda::` |
| `nanovdb.cuda` | `nanovdb::cuda::` |

## `nanovdb` (root)

### Polymorphic typed grids

A `Grid` Python base class is the type-erased parent of every typed
`<BuildT>Grid` subclass. There is one subclass per BuildT in
[`BuildTypes.def`](../../nanovdb/nanovdb/python/BuildTypes.def):

* **Scalars**: `FloatGrid`, `DoubleGrid`, `Int16Grid`, `Int32Grid`,
  `Int64Grid`, `UInt8Grid`, `UInt32Grid`
* **Vectors**: `Vec3fGrid`, `Vec3dGrid`, `Vec4fGrid`, `Vec4dGrid`,
  `Vec3u8Grid`, `Vec3u16Grid`, `RGBA8Grid`
* **Quantized** (read-only): `Fp4Grid`, `Fp8Grid`, `Fp16Grid`,
  `FpNGrid`
* **Index / mask** (read-only): `IndexGrid`, `OnIndexGrid`, `MaskGrid`,
  `BooleanGrid`
* **Point**: `PointGrid`

Each typed grid exposes (some bindings vary by BuildT category):
`gridType`, `gridClass`, `gridName`, `activeVoxelCount`,
`memUsage`, `worldBBox`, `indexBBox`, `voxelSize`, `map`, `tree`,
`getValue`, `getAccessor`, `blindDataCount`, `blindMetaData`,
`findBlindData`, `findBlindDataForSemantic`, `getBlindData`,
`isSequential`, `isBreadthFirst`, and the `leaf_values()` zero-copy
bulk extractor for BuildTs whose leaf carries a `mValues[512]`.

### Handles

* `GridHandle` — host. Methods: `grid(n=0)`, `gridCount`, `gridType(n)`,
  `gridSize(n)`, `empty`, `__bool__`, `__len__`. Construct via the
  `nanovdb.tools.create*` factories or `nanovdb.io.readGrid(s)`. The
  per-grid name is on the grid itself: `handle.grid(n).gridName()`.
* `DeviceGridHandle` — CUDA device. Methods: `grid(n=0)`,
  `deviceGrid(n=0)`, `gridCount`, plus host-side metadata mirroring
  `GridHandle`. Available only in CUDA builds.

Free functions: `splitGrids(handle)`, `mergeGrids([handles])`,
`createNodeManager(grid)`.

### Type-erased introspection

* `GridMetaData(grid)` — 768-byte view answering "what's in this
  buffer?" without knowing BuildT. Methods: `gridName`, `gridType`,
  `gridClass`, `worldBBox`, `indexBBox`, `voxelCount`,
  `activeVoxelCount`, `nodeCount(level)`, `blindDataCount`, etc.
  Also exposes a `safeCast(other_grid)` factory used by I/O helpers.

### Tree / nodes

`grid.tree()` returns the matching `<Suffix>Tree` for that grid. Each
`<Suffix>Tree` exposes `root()`, `background()`, `activeVoxelCount`,
`activeTileCount(level)`, `nodeCount(level)`, `getFirstNode(level)`,
`extrema()`. The per-node classes `NanoRoot<T>` / `NanoUpper<T>` /
`NanoLower<T>` / `NanoLeaf<T>` (bound as `<Suffix>Root` / `Upper` /
`Lower` / `Leaf`) carry origin / bbox / stats / iterators, and the
`Leaf` class adds a zero-copy 512-element `values()` view.

### NodeManager

`createNodeManager(grid)` returns a `NodeManagerHandle` whose
`mgr` property gives access to the typed `<Suffix>NodeManager` with
`leafCount()`, `lowerCount()`, `upperCount()`, `leaf(i)`, `lower(i)`,
`upper(i)`, `isLinear()`, `memUsage()`.

### VoxelBlockManager (host)

`tools.buildVoxelBlockManager(grid, log2_block_width=6, first_offset=0,
last_offset=0, n_blocks=0)` returns a `VoxelBlockManagerHandle` with:

* `firstLeafID()` — zero-copy `(blockCount,)` uint32 NumPy view.
* `jumpMap()` — zero-copy `(blockCount, jump_map_length)` uint64
  NumPy view, where `jump_map_length` is derived from the stored
  `log2_block_width` (no caller mismatch possible).
* `decodeBlock(grid, block_index)` — `(leaf_index, voxel_offset)`
  uint32 / uint16 NumPy arrays of length `BlockWidth`.
* `tools.decodeInverseMaps(grid, first_leaf_id, jump_map,
  block_first_offset, log2_block_width)` — free function form.

### Blind data

Per-grid: `blindDataCount()`, `blindMetaData(n)` returning a
`GridBlindMetaData` struct, `findBlindData(name)`,
`findBlindDataForSemantic(sem)`, `getBlindData(n)` returning a
zero-copy NumPy view typed by the blind metadata.

Enums: `GridBlindDataClass`, `GridBlindDataSemantic`.

### Point primitives

`PointAccessor` (for `PointGrid` / `PointIndexGrid` / `PointDataGrid`):
`gridPoints()`, `leafPoints(leaf_n)`, `voxelPoints(coord)`, each
returning a zero-copy NumPy view of the per-attribute arrays.

### Enums

`GridType`, `GridClass`, `Codec` (under `nanovdb.io`), `StatsMode`
(under `nanovdb.tools`), `CheckMode`, `PointType`,
`GridBlindDataClass`, `GridBlindDataSemantic`. All have a proper
`__repr__` (e.g. `repr(nanovdb.GridType.Float) == 'Float'`).

### Build / runtime probes

* `nanovdb.isCudaAvailable()` — True iff the bindings were compiled
  with `NANOVDB_USE_CUDA`.
* `nanovdb.isGpuAvailable()` — True iff a CUDA-capable GPU is
  accessible at runtime.
* `nanovdb.get_include()` — path to the C headers shipped inside the
  wheel. Use this from `setup.py` of downstream extension modules:
  ```python
  ext = Extension(..., include_dirs=[nanovdb.get_include()])
  ```

## `nanovdb.math`

Geometric primitives shared with C++:

* `Coord`, `CoordBBox` — integer voxel coordinates and bounding boxes.
* `Vec3f`, `Vec3d`, `Vec3u8`, `Vec3u16` — vector value types
  (operator overloads + `__eq__` bound).
* `Vec4f`, `Vec4d` — 4-component vectors.
* `Rgba8` — packed 32-bit colour.
* `BBoxR`, `BBoxF`, `BBoxD` — world-space bounding boxes.
* `Map` — affine index→world transform with `applyMap`,
  `applyInverseMap`, `applyJacobian`, `applyInverseJacobian`,
  `getVoxelSize`, `set(scale, translation, taper)`.
* `Mask<3>` (LeafMask), `Mask<4>` (LowerInternalNodeMask),
  `Mask<5>` (UpperInternalNodeMask).
* Samplers: `createNearestNeighborSampler(grid)`,
  `createTrilinearSampler(grid)`,
  `createTriquadraticSampler(grid)`,
  `createTricubicSampler(grid)` — return callable samplers for
  `float`, `double`, `int32`, `Vec3f` grids.

## `nanovdb.io`

File I/O for the `.nvdb` format.

* `Codec` enum: `NONE`, `ZIP`, `BLOSC`.
* `FileHeader`, `FileMetaData`, `FileGridMetaData` — header /
  per-file / per-grid metadata structs read from `.nvdb` files.
* `readGrid(path, n=0) -> GridHandle` — read a single grid by index.
* `readGrids(path) -> GridHandle` — read every grid into one handle.
* `readGridMetaData(path) -> list[FileGridMetaData]` — metadata
  inspection without loading voxels.
* `writeGrid(path, handle, codec=NONE)`, `writeGrids(path, handles,
  codec=NONE)` — write to disk.
* `hasGrid(path, name) -> bool` — check for a named grid in a file.
* Device variants: `deviceReadGrid`, `deviceReadGrids`,
  `deviceWriteGrid`, `deviceWriteGrids` (CUDA builds only).

## `nanovdb.tools`

### Primitive factories

* **Sphere / torus** (originally Phase 0): `createLevelSetSphere`,
  `createLevelSetTorus`, `createFogVolumeSphere`,
  `createFogVolumeTorus`.
* **Box / BBox / octahedron** (Phase 5a): `createLevelSetBox`,
  `createLevelSetBBox` (hollow wireframe with `thickness`),
  `createLevelSetOctahedron`, `createFogVolumeBox`,
  `createFogVolumeOctahedron`.
* **Point** (Phase 5a): `createPointSphere`, `createPointTorus`,
  `createPointBox` (return `PointDataGrid` scattered on the surface),
  `createPointScatter(srcGrid, pointsPerVoxel, ...)` (scatter into
  the active voxels of a `NanoGrid<float>` level set).
* **Callback-based** (existing): `createFloatGrid`, `createDoubleGrid`,
  `createInt32Grid`, `createVec3fGrid` — build a grid by evaluating
  a Python function at every voxel in a bounding box.

### Conversion / quantization (Phases 5b + 5c)

* `tools.createNanoGridFp4(src, sMode, cMode, ditherOn, verbose)`,
  `createNanoGridFp8(...)`, `createNanoGridFp16(...)` — fixed
  bit-width quantization. Source: `NanoGrid<float>` or
  `nanovdb.tools.build.FloatGrid`.
* `tools.createNanoGridFpN(src, oracle, sMode, cMode, ditherOn,
  verbose)` — variable bit-width. `oracle` is one of
  `tools.AbsDiff(tolerance)` (absolute error) or
  `tools.RelDiff(tolerance)` (relative error). Default tolerance
  `-1.0` means uninitialized — pass a concrete non-negative value.
* `tools.createNanoGridIndex(src, channels=0, includeStats=True,
  includeTiles=True, verbose=0)` — produce a `NanoGrid<ValueIndex>`
  giving every voxel (active or not) a uint64 sequential index.
* `tools.createNanoGridOnIndex(src, ...)` — same, only active voxels
  get indices. Used as input to `tools.buildVoxelBlockManager`.

Source set for Index / OnIndex: `NanoGrid<float | double | int32 |
Vec3f>` and `nanovdb.tools.build.Grid<float | double | int32 |
Vec3f>`.

### Stats

* `tools.<Suffix>Extrema` for every scalar / vector BuildT:
  `add(value)`, `min()`, `max()`, `__bool__`, `hasMinMax /
  hasAverage / hasStdDeviation / hasStats` static predicates.
* `tools.<Suffix>Stats` (inherits Extrema): adds `size`, `avg` /
  `mean`, `var` / `variance`, `std` / `stdDev`.
* `tools.updateGridStats(grid, mode=Default)` — polymorphic; runs
  in place on any bound NanoGrid. Special BuildTs accept `Disable`
  and `BBox` modes only (the `MinMax`/`All` arms have no arithmetic
  meaning).
* `tools.getExtrema(grid, bbox)` — per-BuildT (scalar + vector
  only) extrema over a coordinate bounding box.

### Validation + checksum

* `tools.validateGrid(handle, gridID, mode=Default, verbose=False)` —
  per-grid validation. Returns `False` (no raise) on out-of-range
  `gridID`. `CheckMode.Disable` short-circuits to True.
* `tools.validateGrids(handle, mode, verbose)` — same for the
  whole handle.
* `tools.checkGrid(grid, mode=Full)` → `(ok, error_message)`.
* `tools.isValid(grid, mode=Default, verbose=False)` — bool, also
  checks the stored checksum.
* `tools.evalChecksum(grid, mode=Default)` → `Checksum` —
  recompute without writing.
* `tools.validateChecksum(grid, mode=Default)` → `bool` — compare
  the stored checksum against a freshly computed one.
* `tools.updateChecksum(gridData, mode)` — write a fresh checksum
  into the grid's header.
* Enums: `StatsMode`, `CheckMode`.

### OpenVDB bridge (NANOVDB_USE_OPENVDB builds only)

* `tools.openToNanoVDB(base, sMode, cMode, verbose)` — convert
  any OpenVDB grid.
* `tools.nanoToOpenVDB(handle, verbose)` — convert the other way.

## `nanovdb.tools.build`

Mutable, voxel-by-voxel CPU grid builder. One typed `<Suffix>Grid`
per writable BuildT (scalars + vectors). Each grid:

* Constructor `(background, name='', gridClass=Unknown)`.
* `setValue(ijk, value)` — write and mark active.
* `setValueOn(ijk)` — mark active without changing the stored value.
* `getValue(ijk)` — read.
* `isActive(ijk)` — bool.
* `.background` property.
* `getName()`, `setName(name)`, `gridType()`, `gridClass()`,
  `setTransform(scale=1.0, translation=Vec3d(0))`, `nodeCount()`.
* `getAccessor()` → typed `<Suffix>ValueAccessor` (thread-safe read,
  non-thread-safe write, caches the last leaf / lower / upper node
  visited).
* `getWriteAccessor()` → typed `<Suffix>WriteAccessor` (thread-safe
  write — buffers into a private root, merges into the parent on
  `merge()` or destruction).
* `.to_nanovdb(sMode=Default, cMode=Default, verbose=0)` — bake
  into a host `GridHandle` of the same BuildT. Releases the GIL
  during conversion.

## `nanovdb.tools.cuda`

CUDA-accelerated kernels. Surface is unchanged from the pre-restructure
binding except that `sampleFromVoxels` migrated here from the
removed `nanovdb.math.cuda` submodule.

* `signedFloodFill(grid)` for `float` / `double` grids.
* `pointsToRGBA8Grid(tensor)` for `Rgba8` colours.
* `sampleFromVoxels(grid)` for `float` / `double` device grids.
* All CUDA primitives mirroring `nanovdb.tools` are also bound where
  the C++ template instantiates for `cuda::DeviceBuffer`.

## `nanovdb.cuda`

Currently just `DeviceBuffer` — the buffer backing
`DeviceGridHandle`. Retained from the pre-restructure binding.

---

The bindings are produced from these sources:

* [Module entry](../../nanovdb/nanovdb/python/NanoVDBModule.cc)
* [BuildT X-macro list](../../nanovdb/nanovdb/python/BuildTypes.def)
* Per-area binding files: `Py<Area>.cc` next to the module entry.

See [PythonMigration.md](PythonMigration.md) for the porting guide
from the pre-Phase-1 API.
