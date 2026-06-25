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

### GPU examples

These require a CUDA build of `nanovdb` and a CUDA-capable GPU; each
self-skips with a printed message when the build, GPU, or the optional
GPU-array framework it uses is unavailable.

| Script | What it shows |
| --- | --- |
| [`gpu_load_inspect.py`](gpu_load_inspect.py) | `nanovdb.io.deviceReadGrid` → `handle.deviceUpload()` → zero-copy `cupy.asarray(handle)` view of the device buffer; the host `grid(n)` vs device `deviceGrid(n)` `data_ptr()` split. Requires CuPy. |
| [`cupy_rawkernel.py`](cupy_rawkernel.py) | A `cupy.RawKernel` that `#include <nanovdb/NanoVDB.h>` (compiled with `nanovdb.cuda.compile_options()`) and reads a `const nanovdb::NanoGrid<float>*` straight from `grid.data_ptr()`. Documents the device-pointer ABI. Requires CuPy. |
| [`numba_cuda.py`](numba_cuda.py) | Adopting a NanoVDB device buffer as a zero-copy `numba.cuda` array (Numba can't parse the C++ ABI, so it operates on the raw buffer). Requires Numba. |
| [`triton_kernel.py`](triton_kernel.py) | Handing a NanoVDB device buffer to a Triton kernel via a zero-copy `torch.from_dlpack` tensor. Requires Triton + PyTorch. |

#### One level-set filter, three compute backends

[`levelset_filter.py`](levelset_filter.py) is the runnable driver for **one** GPU
level-set filter — `tools::LevelSetFilter`-style Laplacian diffusion + Godunov
renormalisation + narrow-band retrack on a `.nvdb` file, driven by the device
`VoxelBlockManager`, reading/writing either a `FloatGrid` or an `OnIndexGrid`
(SDF in a blind channel) and preserving the input style — with the per-voxel
stencil math supplied by one of **three interchangeable backends**:

```
python levelset_filter.py {rawkernel|cupy|cutile}  in.nvdb out.nvdb [outer_iters]
python levelset_filter.py {rawkernel|cupy|cutile}                  # self-test
```

The driver owns everything that isn't backend-specific (the style-detecting read,
the style-preserving write, the `dilateGrid` / `inject` / `injectPredicateToMask`
/ `pruneGrid` retrack, the outer loop). Each backend is a small `Backend` object
implementing only the three per-voxel stencils, and differs **only in how the
dense per-voxel math is computed** — a side-by-side of three GPU styles. Each
backend file also runs standalone (`python levelset_filter_<backend>.py`) as a
stencil-only smoke test (deform + renorm on a sphere, no I/O, no retrack).

| Backend | Per-voxel compute |
| --- | --- |
| [`levelset_filter_rawkernel.py`](levelset_filter_rawkernel.py) | A hand-written CUDA kernel: `decodeInverseMaps` / `computeBoxStencil` fused with the update in a `cupy.RawModule` (nvcc backend), no dense gather. Fastest, most control. Requires CuPy + nvcc. |
| [`levelset_filter_cupy.py`](levelset_filter_cupy.py) | **Kernel-free**: `gatherBoxStencil` → dense `(N, 6)` face values, then all per-voxel math as plain **CuPy** array ops. No `RawModule` / CUDA C++ / nvcc. Requires only CuPy. |
| [`levelset_filter_cutile.py`](levelset_filter_cutile.py) | The same dense arrays, with the per-voxel stencils as **NVIDIA cuTile** (`cuda.tile`) tile kernels (`ct.load` / `ct.store` over `(TILE,)` tiles). Requires CuPy + `cuda-tile`. |

For full API signatures and per-argument docstrings, use Python's
`help()` on any symbol — e.g. `help(nanovdb.tools.createNanoGridFpN)`.
The bindings ship `.pyi` type stubs for IDE / type-checker support.

## Migration notes

The CUDA device buffer and device grid handle classes now live under the
`nanovdb.cuda` submodule (mirroring the C++ `nanovdb::cuda` namespace):

- `nanovdb.DeviceBuffer` → `nanovdb.cuda.DeviceBuffer`
- `nanovdb.DeviceGridHandle` → `nanovdb.cuda.DeviceGridHandle`

Factory functions that return a device handle are unaffected — they never name
the class — so `nanovdb.io.deviceReadGrid(s)`, `nanovdb.tools.cuda.create*`,
`nanovdb.tools.cuda.pointsToRGBA8Grid`, and the `deviceUpload` /
`deviceDownload` / `deviceGrid` handle methods continue to work unchanged.

## GPU / CUDA

The CUDA surface is only present when `nanovdb` is built with CUDA. Gate any
GPU code on both checks (compiled-with-CUDA and a GPU actually present):

```python
import nanovdb
if nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable():
    ...  # nanovdb.cuda / nanovdb.tools.cuda are safe to use
```

In a non-CUDA build the `nanovdb.cuda` submodule does not exist, so guard with
`hasattr(nanovdb, "cuda")` (or the checks above) before touching it.

`nanovdb.cuda` and `nanovdb.tools.cuda` are **attributes** of the package, not
importable submodules. Use them via attribute access after `import nanovdb`
(`import nanovdb.tools.cuda` raises `ModuleNotFoundError`):

```python
import nanovdb
TC = nanovdb.tools.cuda   # ok
# import nanovdb.tools.cuda  # NOT ok
```

### Namespace layout: `nanovdb.cuda` vs `nanovdb.tools.cuda`

- **`nanovdb.cuda`** — CUDA *infrastructure* (mirrors the C++ `nanovdb::cuda`
  namespace): the buffer / handle types and device plumbing.
  `DeviceBuffer`, `DeviceGridHandle`, `UnifiedBuffer`, `UnifiedGridHandle`,
  `DeviceMesh`, `DeviceStreamMap`, `DeviceResource`, `TempDevicePool`,
  `DeviceNodeManagerHandle`, `createDeviceNodeManager`, and
  `compile_options`.
- **`nanovdb.tools.cuda`** — device *algorithms* (mirrors
  `nanovdb::tools::cuda`): point / voxel rasterizers (`pointsToGrid`,
  `voxelsTo{OnIndex,Index,RGBA8}Grid`, `pointsToRGBA8Grid`), morphology /
  topology (`dilateGrid`, `coarsenGrid`, `refineGrid`, `pruneGrid`,
  `mergeGrids`), index utilities (`indexToGrid`, `addBlindData`), in-place
  device QC (`updateGridStats`, `updateChecksum`, `evalChecksum`,
  `validateChecksum`, `isValid`), `signedFloodFill`, `sampleFromVoxels`,
  `buildVoxelBlockManager`, and the multi-GPU `Distributed*PointsToGrid`
  converters.

### Building a device grid

```python
h   = nanovdb.tools.createLevelSetSphere(radius=100.0, voxelSize=1.0)  # host
fg  = h.grid(0)                              # host FloatGrid
onh = nanovdb.tools.createOnIndexGrid(fg)    # host OnIndexGrid handle
nanovdb.io.writeGrid("sphere.nvdb", onh)
dh  = nanovdb.io.deviceReadGrid("sphere.nvdb")   # nanovdb.cuda.DeviceGridHandle
dh.deviceUpload(0, True)                          # stream=0, sync=True
dg  = dh.deviceGrid(0)                            # DEVICE grid; feed to tools.cuda.*
```

### Zero-copy interop: CAI, DLPack, and `data_ptr()`

`DeviceGridHandle`, `DeviceBuffer`, and `UnifiedBuffer` expose the whole
device buffer as a 1-D `uint8` array through both the **CUDA Array Interface
(v3)** (`__cuda_array_interface__`) and **DLPack** (`__dlpack__` /
`__dlpack_device__`). So they bridge to CuPy / PyTorch / Numba with no copy:

```python
import cupy as cp
dh.deviceUpload(0, True)
buf = cp.asarray(dh)          # zero-copy; buf.data.ptr == dh.device_ptr()
buf = cp.from_dlpack(dh)      # same, via DLPack
assert buf.nbytes == dh.size()
```

The data pointer is null and the array empty until `deviceUpload` runs.
`UnifiedGridHandle` intentionally does **not** expose the CAI / DLPack bridges
(it is the managed-memory handle returned by `Distributed*PointsToGrid`); read
its grid through `deviceGrid(n)` and the `tools.cuda.*` ops instead.

Every typed grid exposes `grid.data_ptr() -> int`. **It is a HOST pointer when
the grid came from `handle.grid(n)` and a DEVICE pointer when it came from
`handle.deviceGrid(n)`** — the grid object itself cannot tell host from device,
so provenance is the caller's responsibility. The device pointer is the base
of a `nanovdb::NanoGrid<BuildT>` in GPU memory (the same C++ ABI), which is
exactly what you pass to a custom CUDA kernel (see `cupy_rawkernel.py`).

> **Caveat — host accessors segfault on device grids.** A grid from
> `deviceGrid(n)` holds a DEVICE pointer; calling a host-side accessor on it
> (e.g. `dg.getAccessor().getValue(...)`) dereferences GPU memory on the CPU
> and **crashes the process** (SIGSEGV). Host reads are only legal on
> `grid(n)` / host grids. Feed `deviceGrid(n)` **only** to `tools.cuda.*`
> device entry points, or read its bytes via `data_ptr()` / the CAI / DLPack
> buffer from a device kernel.

### Streams

Stream arguments everywhere are **raw CUDA stream handles passed as Python
ints** (`0` == the default stream). `deviceUpload` / `deviceDownload` and the
`tools.cuda.*` ops all take a trailing `stream` int. From CuPy, pass
`stream.ptr`:

```python
s = cp.cuda.Stream(non_blocking=True)
h = nanovdb.tools.cuda.pointsToGrid(points, 1.0, s.ptr)
```

### Wrapping external memory (`from_external`)

`nanovdb.cuda.DeviceBuffer.from_external(size, gpu_ptr, cpu_ptr)` builds a
**non-owning** wrapper around memory you already allocated (e.g. a CuPy /
PyTorch buffer); it never frees, and the caller must keep the source alive. A
null `gpu_ptr` is rejected. Pair it with
`DeviceGridHandle.from_buffer(buffer)` (which **moves** the buffer, leaving it
empty, and validates the NanoVDB header) for a fully zero-copy adopt of grid
bytes you produced elsewhere.

### Managed memory for the distributed pipeline

`Distributed{Points,IndexPoints,RGBA8Points}ToGrid.getHandle(voxels)` requires
the `(N, 3)` int32 coordinate array to live in **CUDA managed (unified)**
memory and returns a `nanovdb.cuda.UnifiedGridHandle`. With CuPy, route the
allocation through the managed allocator:

```python
prev = cp.cuda.get_allocator()
cp.cuda.set_allocator(cp.cuda.malloc_managed)
try:
    voxels = cp.asarray(coords_int32)          # managed (N,3) int32
    mesh   = nanovdb.cuda.DeviceMesh()          # must outlive the converter
    conv   = nanovdb.tools.cuda.DistributedPointsToGrid(mesh, 1.0, (0, 0, 0))
    uh     = conv.getHandle(voxels)             # UnifiedGridHandle (OnIndex)
finally:
    cp.cuda.set_allocator(prev)
```

A plain (device-pool) array does not satisfy the `cuda_managed` constraint.

### Compiling custom kernels against the bundled headers

`nanovdb.cuda.compile_options(*extra)` returns the NanoVDB include flag
(`-I<headers>`) followed by any extra flags, for feeding NVRTC / a runtime CUDA
compiler so your kernel compiles against the same headers as the wheel:

```python
opts = nanovdb.cuda.compile_options("-std=c++17")
kernel = cp.RawKernel(src, "my_kernel", options=opts, backend="nvrtc")
```

The include dir is only physically present in an installed wheel; in an
in-source dev build tree the path resolves but does not exist (see
`cupy_rawkernel.py` for a `NANOVDB_INCLUDE` fallback).

### Future: pluggable memory resources (roadmap)

The device buffers above own their allocations directly. A planned evolution is
a **pluggable memory-resource** model (à la a polymorphic allocator), so device
buffers can draw from a caller-supplied pool / async / pinned allocator —
e.g. RAPIDS Memory Manager–style resources. See Mark Harris's design sketch:
<https://gist.github.com/harrism/dfacdb12d0e1502c3a4be964e92fef2c>. The
following names are **reserved** for that work and should not be used today:
`Resource`, `MemoryResource`, `AsyncResource`, `PinnedResource`,
`ResourceDeviceBuffer`, `DeviceBuffer2`, `default_resource`,
`set_default_resource`.
