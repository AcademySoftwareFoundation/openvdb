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

For full API signatures and per-argument docstrings, use Python's
`help()` on any symbol — e.g. `help(nanovdb.tools.createNanoGridFpN)`.
The bindings ship `.pyi` type stubs for IDE / type-checker support.
