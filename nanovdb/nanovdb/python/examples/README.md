# NanoVDB Python Examples

Runnable `.py` scripts demonstrating the Python surface that ships
with the [bindings restructure](../../nanovdb-python-plan.md). Each
script is self-contained, builds its own input data, and prints a
small summary to stdout.

Run any example with:

```bash
python load_inspect.py
# or, after installing the wheel:
python -m nanovdb.examples.load_inspect    # not a real entry point; just `python <file>`
```

The examples assume the `nanovdb` module is importable. When working
from the source tree, that usually means:

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
| [`quantize.py`](quantize.py) | Quantize a `NanoGrid<float>` through `tools.createNanoGridFp{4,8,16,N}`. Shows fixed-width quantization with dithering and variable-width `FpN` with both `AbsDiff` and `RelDiff` oracles. |
| [`validate.py`](validate.py) | `tools.validateGrid` / `validateGrids`, `tools.checkGrid`, `tools.isValid`, and the `evalChecksum` / `validateChecksum` / `updateChecksum` round-trip. |

## See also

* [Migration guide](../../../../doc/nanovdb/PythonMigration.md) —
  porting from the pre-Phase-1 typed-accessor API.
* [API reference](../../../../doc/nanovdb/PythonAPI.md) — narrative
  survey of every bound submodule.
