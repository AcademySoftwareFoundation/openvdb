# NanoVDB Python Examples

Runnable `.py` scripts demonstrating the NanoVDB Python bindings.
Each script is self-contained, builds its own input data, and prints
a small summary to stdout.

Run any example with:

```bash
python load_inspect.py
```

When working from the source tree, the `nanovdb` module needs to be
on `PYTHONPATH` (e.g. via the build output directory):

```bash
cd <build>/nanovdb/nanovdb/python
PYTHONPATH=. python /path/to/<example>.py
```

## Examples

| Script | What it shows |
| --- | --- |
| [`load_inspect.py`](load_inspect.py) | Polymorphic `handle.grid(n)` access, `GridMetaData` type-erased introspection, mixed-type handles via `mergeGrids`. |
| [`build_grid.py`](build_grid.py) | Voxel-by-voxel construction with `nanovdb.tools.build.FloatGrid`, the cached `ValueAccessor`, the thread-safe `WriteAccessor`, and `.to_nanovdb()` to bake into a host `GridHandle`. |
| [`bulk_leaf_numpy.py`](bulk_leaf_numpy.py) | Zero-copy `(N_leaves, 512)` NumPy view of every leaf's mValues via `grid.leaf_values()`. Includes a global-stats reduction and an in-place mutation that propagates back into the grid. Requires NumPy. |
| [`quantize.py`](quantize.py) | Quantize a `NanoGrid<float>` through `nanovdb.tools.createNanoGridFp{4,8,16,N}`. Shows fixed-width quantization with dithering and variable-width `FpN` with both `AbsDiff` and `RelDiff` oracles. |
| [`validate.py`](validate.py) | `nanovdb.tools.validateGrid` / `validateGrids`, `checkGrid`, `isValid`, and the `evalChecksum` / `validateChecksum` / `updateChecksum` round-trip. |
| [`io_roundtrip.py`](io_roundtrip.py) | `nanovdb.io` write/read round-trip over five primitives: `writeGrids` with codec fallback, `readGridMetaData`, `hasGrid`, `readGrid` by name, `splitGrids`, and zero-copy point positions via `getBlindData`. Port of `ex_write_nanovdb_grids` + `ex_read_nanovdb_sphere_accessor`. |
| [`make_funny_nanovdb.py`](make_funny_nanovdb.py) | Functor-based construction: `tools.createFloatGrid(background, name, gridClass, func, bbox)` evaluates a Python callback at every voxel. Port of `ex_make_funny_nanovdb` on a reduced domain. |
| [`make_typed_grids.py`](make_typed_grids.py) | One solid sphere per value type via `tools.build.{Float,Double,Int16,Int32,Int64,UInt32,Vec3f}Grid`, written to a single file and re-read with polymorphic `handle.grid()` dispatch. Port of `ex_make_typed_grids`. |
| [`raytrace_level_set.py`](raytrace_level_set.py) | CPU sphere-traced level-set render to PGM using `worldToIndex`, the trilinear sampler, `zeroCrossing()`, and normalized `gradient()` shading. Port of `ex_raytrace_level_set` (host path). |
| [`raytrace_fog_volume.py`](raytrace_fog_volume.py) | CPU transmittance ray-march of a fog volume to PGM using a `ReadAccessor` and `Coord.Floor` — the accessor-based sampling idiom. Port of `ex_raytrace_fog_volume` (host path). |
| [`collide_level_set.py`](collide_level_set.py) | Particles colliding with a level set: `worldToIndexF`, `tree.isActive` narrow-band test, accessor distance reads, and `sampler.gradient()` collision normals. Port of `ex_collide_level_set` (host path). |
| [`index_grid_channels.py`](index_grid_channels.py) | `tools.createNanoGridOnIndex(src, channels=1)`, `grid.valueCount()`, coordinate reads through `createChannelAccessor`, and blind-data authoring with `tools.CreateNanoGrid.addBlindData` + the writable `getBlindData` view. Extends the host half of `ex_index_grid_cuda`. Requires NumPy for the authoring section. |
| [`node_manager.py`](node_manager.py) | Linearized node iteration with `createNodeManager`: per-level counts, `leaf(i)` / `lower(i)` access, node origins, masks, and stats. Port of the host half of `ex_nodemanager_cuda`. |
| [`openvdb_interop.py`](openvdb_interop.py) | `tools.openToNanoVDB` / `nanoToOpenVDB` round-trip with accessor comparison on both sides. Self-skips unless built with `NANOVDB_USE_OPENVDB` and `openvdb` is importable. Port of `ex_openvdb_to_nanovdb_accessor`. |

Scripts that produce files write them to a fresh temporary directory
and print its path, so the source tree stays clean. The whole set is
smoke-tested by [`../test/TestExamples.py`](../test/TestExamples.py)
(ctest name `pytest_nanovdb_examples`) when the module is configured
with `NANOVDB_BUILD_PYTHON_UNITTESTS=ON`.

For full API signatures and per-argument docstrings, use Python's
`help()` on any symbol — e.g. `help(nanovdb.tools.createNanoGridFpN)`.
The bindings ship `.pyi` type stubs for IDE / type-checker support.
