# NanoVDB Python Migration Guide

Phases 0–5 of the [NanoVDB Python bindings restructure](https://github.com/AcademySoftwareFoundation/openvdb/issues/2208)
deliberately broke source compatibility with the pre-restructure API
(pre-1.0 license to restructure was explicit). This guide is for users
on the old bindings; everything described as "before" worked against
the pre-Phase-1 module and everything described as "after" matches the
current `feature/nanovdb_python` surface.

There is no deprecation cycle for the old symbols — they are removed
outright. The replacements are listed below.

## 1. Polymorphic grid access replaces typed accessors

The biggest change. Every typed grid accessor on `GridHandle` and
`DeviceGridHandle` is gone; the polymorphic `grid(n)` / `deviceGrid(n)`
returns the correct typed subclass selected by `handle.gridType(n)` at
runtime.

```python
# Before — six typed accessors per handle:
handle.floatGrid(i)      # only worked if gridType(i) == Float
handle.doubleGrid(i)
handle.int32Grid(i)
handle.vec3fGrid(i)
handle.rgba8Grid(i)
handle.pointGrid(i)
# … and matching device* variants on DeviceGridHandle:
device_handle.deviceFloatGrid(i)
device_handle.deviceDoubleGrid(i)
…

# After — one polymorphic accessor each:
grid = handle.grid(i)            # NanoGrid<BuildT> for whatever gridType(i) is
dgrid = device_handle.deviceGrid(i)
```

The returned object is the same Python class you would have gotten
from the typed accessor before. The class names — `FloatGrid`,
`DoubleGrid`, `Int32Grid`, `Vec3fGrid`, etc. — are unchanged; only the
*accessor* on the handle changed.

If you genuinely need to dispatch on type from Python, use
`handle.gridType(i)` to compare against `nanovdb.GridType.*` enumerators,
or `isinstance(grid, nanovdb.FloatGrid)`.

## 2. New BuildTs you can now reach

The pre-restructure binding covered six grid types. The polymorphic
accessor returns any of the 24 the C++ library supports:

| Category | BuildTs |
| --- | --- |
| Scalar arithmetic | `float`, `double`, `int16`, `int32`, `int64`, `uint8`, `uint32` |
| Vector | `Vec3f`, `Vec3d`, `Vec4f`, `Vec4d`, `Vec3u8`, `Vec3u16`, `Rgba8` |
| Quantized (read-only) | `Fp4`, `Fp8`, `Fp16`, `FpN` (decode to `float`) |
| Index / mask | `ValueIndex`, `ValueOnIndex`, `ValueMask` |
| Point | `PointIndex` / `PointData` |

Read-only special BuildTs (`Boolean`, `Fp*`, `Index`, `OnIndex`,
`Mask`) and `Point` are now bound at parity with the scalar types for
reads — `getValue`, accessors, samplers (where applicable). They do
not support `setVoxel`; mutation must go through `tools.build.Grid<T>`
for the writable BuildTs.

## 3. `nanovdb.math.cuda` submodule is removed

`sampleFromVoxels` moved from `nanovdb.math.cuda` to
`nanovdb.tools.cuda`. The C++ side has no `math::cuda` namespace; this
aligns with the C++ layout.

```python
# Before:
sampler = nanovdb.math.cuda.sampleFromVoxels(device_grid)

# After:
sampler = nanovdb.tools.cuda.sampleFromVoxels(device_grid)
```

## 4. Pre-existing bugs fixed (observable behavior changes)

* `GridHandle.__bool__` now returns `not handle.empty()`. Before, it
  returned `None`, which made `if handle:` raise.
* `repr(nanovdb.GridType.Float)` now returns `"Float"` instead of
  nanobind's default `"<GridType.Float: 1>"`. Same for `GridClass`
  and `nanovdb.io.Codec`.

## 5. New surfaces — what to use instead of dropping to C++

Several capabilities that used to require C++ are now first-class
Python:

* **Type-erased introspection**: `nanovdb.GridMetaData(grid)` exposes
  a 768-byte view of any grid without knowing `BuildT` at compile
  time. Useful for generic loaders and viewers.

* **Blind data API**: `grid.blindDataCount()`, `grid.blindMetaData(n)`,
  `grid.findBlindData(name)`, `grid.findBlindDataForSemantic(sem)`,
  and `grid.getBlindData(n)` returning zero-copy NumPy views typed by
  the blind metadata. Enums `nanovdb.GridBlindDataClass` and
  `nanovdb.GridBlindDataSemantic` are bound.

* **`PointAccessor`**: `gridPoints()`, `leafPoints()`,
  `voxelPoints()` on `PointGrid` (and matching for `PointIndexGrid`),
  each yielding zero-copy NumPy attribute views.

* **Tree / node walking**: `grid.tree()` returns `NanoTree<T>`;
  `tree.root()`, `tree.nodeCount(level)`, `tree.getFirstLeaf()`,
  `tree.extrema()`. Per-node types (`NanoRoot<T>`, `NanoUpper<T>`,
  `NanoLower<T>`, `NanoLeaf<T>`) bound with iterators and stats.

* **NodeManager**: `nanovdb.createNodeManager(grid)` returns a
  `NodeManager` with `leaf(i)` / `lower(i)` / `upper(i)` random
  access plus `isLinear()` / `memUsage()`.

* **Bulk leaf values**: `grid.leaf_values()` returns an
  `(N_leaves, 512)` zero-copy NumPy view of every leaf's mValues for
  the BuildTs that carry one (i.e. not Fp*, Index, Mask, bool,
  Point). The single most-useful entry for analytics and ML.

* **`VoxelBlockManager`** (host): `tools.buildVoxelBlockManager(grid)`
  returns a `VoxelBlockManagerHandle` with `firstLeafID` /
  `jumpMap` zero-copy NumPy views plus `decodeBlock(grid, i)` for the
  per-block inverse maps.

* **`tools.build.Grid<T>`**: voxel-by-voxel construction in pure
  Python. Constructor `(background, name='', gridClass=Unknown)`,
  then `setValue(ijk, v)` / `setValueOn(ijk)` / `getValue(ijk)` /
  `isActive(ijk)` (or use `getAccessor()` for caching, or
  `getWriteAccessor()` for thread-safe batched writes). Call
  `.to_nanovdb(sMode, cMode, verbose)` to bake into a
  `GridHandle<HostBuffer>`. Available for every writable BuildT
  (scalar + vector).

* **Stats**: `tools.<Suffix>Extrema`, `tools.<Suffix>Stats` per
  scalar/vector BuildT; polymorphic `tools.updateGridStats(grid, mode)`
  and per-BuildT `tools.getExtrema(grid, bbox)`.

* **Validation / checksum**: `tools.validateGrid(handle, gridID, mode)`,
  `tools.checkGrid(grid, mode)` → `(ok, error)`,
  `tools.isValid(grid, mode)`, `tools.evalChecksum(grid, mode)`,
  `tools.validateChecksum(grid, mode)`.

* **Primitives**: 9 new ones on top of the original 4 —
  `createLevelSetBox`, `createLevelSetBBox`,
  `createLevelSetOctahedron`, `createFogVolumeBox`,
  `createFogVolumeOctahedron`, `createPointSphere`,
  `createPointTorus`, `createPointBox`, and `createPointScatter`.

* **Quantization + cross-type conversion**:
  `tools.createNanoGridFp4` / `Fp8` / `Fp16` accept `NanoGrid<float>`
  or `tools.build.FloatGrid`. `tools.createNanoGridFpN(src, oracle)`
  with `tools.AbsDiff(tolerance)` or `tools.RelDiff(tolerance)`
  oracles. `tools.createNanoGridIndex` / `createNanoGridOnIndex`
  accept any of `{float, double, int32, Vec3f}` source as either
  `NanoGrid` or `tools.build.Grid`.

* **Packaging**: `nanovdb.get_include()` returns the path to the C
  headers shipped inside the wheel, so downstream extension authors
  can compile against the same NanoVDB the wheel was built with.

## 6. Mechanical replacements summary

| Before | After |
| --- | --- |
| `handle.floatGrid(i)` | `handle.grid(i)` |
| `handle.doubleGrid(i)` | `handle.grid(i)` |
| `handle.int32Grid(i)` | `handle.grid(i)` |
| `handle.vec3fGrid(i)` | `handle.grid(i)` |
| `handle.rgba8Grid(i)` | `handle.grid(i)` |
| `handle.pointGrid(i)` | `handle.grid(i)` |
| `dh.deviceFloatGrid(i)` | `dh.deviceGrid(i)` |
| (… and other typed device variants) | `dh.deviceGrid(i)` |
| `nanovdb.math.cuda.sampleFromVoxels(...)` | `nanovdb.tools.cuda.sampleFromVoxels(...)` |
| `if handle: ...` (raised) | `if handle: ...` (works) |
| `repr(GridType.Float)` returned `<GridType.Float: 1>` | returns `"Float"` |

If you have a `git grep` script over your codebase, the regex
`\.\(float\|double\|int32\|vec3f\|rgba8\|point\)Grid\b` catches all
the affected typed-accessor sites, and `nanovdb\.math\.cuda` catches
the relocated `sampleFromVoxels` import.

## 7. Examples + API reference

* [Python API reference](PythonAPI.md) — narrative survey of every
  bound submodule.
* [Python examples](../../nanovdb/nanovdb/python/examples/) —
  runnable `.py` scripts covering loading + introspection, bulk
  NumPy leaf access, voxel-by-voxel grid construction, quantization,
  and validation.
