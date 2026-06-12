# Topology Operators — Host-Side Port Plan

Design and execution plan for back-porting the five NanoVDB topology
operators (`DilateGrid`, `MergeGrids`, `PruneGrid`, `RefineGrid`,
`CoarsenGrid`) and their shared base class `TopologyBuilder` from the
existing CUDA-only implementation under `nanovdb::tools::cuda::` into a
CUDA-free host implementation under `nanovdb::tools::`.

---

## 1.  Motivation and Goal

The current topology operators live exclusively under
`nanovdb/tools/cuda/*.cuh` (namespace `nanovdb::tools::cuda`). They
require `nvcc` to compile and a CUDA-capable device to run, even for
workloads where a host-only path would be more appropriate (small
problem sizes, debug builds, CI environments without GPUs, downstream
projects that don't link CUDA).

The goal is to produce a parallel set of headers under
`nanovdb/tools/*.h` (namespace `nanovdb::tools`) that implement the
same operators on the host — same public surface, same correctness
guarantees, no CUDA dependency at the end of the port.

The CUDA originals are not deprecated and not modified; they remain
the high-performance path for GPU workloads.

---

## 2.  Scope

Files in scope:

| CUDA original (untouched)              | Host target (to be built up)         |
|----------------------------------------|--------------------------------------|
| `nanovdb/tools/cuda/TopologyBuilder.cuh` | `nanovdb/tools/TopologyBuilder.h`  |
| `nanovdb/tools/cuda/DilateGrid.cuh`     | `nanovdb/tools/DilateGrid.h`        |
| `nanovdb/tools/cuda/MergeGrids.cuh`     | `nanovdb/tools/MergeGrids.h`        |
| `nanovdb/tools/cuda/PruneGrid.cuh`      | `nanovdb/tools/PruneGrid.h`         |
| `nanovdb/tools/cuda/RefineGrid.cuh`     | `nanovdb/tools/RefineGrid.h`        |
| `nanovdb/tools/cuda/CoarsenGrid.cuh`    | `nanovdb/tools/CoarsenGrid.h`       |

Indirect dependency:

| `nanovdb/util/cuda/Morphology.cuh`     | host equivalents introduced as needed |

The morphology functors used by the operators (`DilateInternalNodesFunctor`,
`MergeInternalNodesFunctor`, `PruneInternalNodesFunctor`,
`RefineInternalNodesFunctor`, `CoarsenInternalNodesFunctor`, plus the
`…LeafNodesFunctor` / `…LeafMasksFunctor` siblings, `EnumerateNodesFunctor`,
and `ProcessLowerNodesFunctor`) currently live in `util/cuda/Morphology.cuh`
and use `__device__` qualifiers, shared memory, and warp-level cooperation.
The same in-place transformation discipline applies one level down (a
parallel host header is introduced under `util/` once needed by the operator
being ported).

Restrictions inherited from the CUDA implementation, preserved on the host
side:

- Only `OnIndexGrid` buildtypes are supported
  (`static_assert(BuildTraits<BuildT>::is_onindex)`).
- Input grids with tile values at any level cause the operator to throw
  ("Topological operations not supported on grids with value tiles").

---

## 3.  Architectural Decisions

### 3.1  Namespace separation, parallel files

The host and CUDA implementations live as parallel files under disjoint
namespaces:

- `nanovdb::tools::cuda::DilateGrid<BuildT>`  in  `nanovdb/tools/cuda/DilateGrid.cuh`
- `nanovdb::tools::DilateGrid<BuildT>`        in  `nanovdb/tools/DilateGrid.h`

The CUDA path is the existing convention (`nanovdb::cuda::` is reserved for
infrastructure types like `DeviceBuffer`, `UnifiedBuffer`, `TempPool`; the
`cuda` tag for tooling consistently nests as `tools::cuda::`).

`nanovdb::cuda::tools::` does **not** exist as a convention and should not
be introduced.

The inner `topology::detail` namespace appears in both implementations;
fully qualified, the two are disjoint: `tools::cuda::topology::detail::` vs
`tools::topology::detail::`. No symbol clash.

### 3.2  Why parallel files and not a single templated abstraction

Both directions are viable in principle (back-end-templated single class
vs separate `cuda::` and host implementations). We chose the parallel-files
approach because:

- The kernel decomposition differs in kind, not just degree. CUDA work is
  expressed as `dim3(N, slicesPerLowerNode, 1)` block grids with shared
  memory and warp-cooperative mask operations. CPU work is more naturally
  expressed as `util::forEach` over leaf or lower units. Forcing both
  shapes through a single class would either constrain one side or
  proliferate policy parameters.
- BBox propagation uses `expandAtomic` on CUDA (children race into parents).
  On CPU it can either stay atomic (`std::atomic`) or be reduced
  per-parent without atomics — implementation choice that is cleaner kept
  local to each back-end.
- The CUDA originals remain stable. Touching them only to introduce a
  back-end abstraction would create merge friction with upstream NanoVDB
  development.

The cost is bounded duplication of six files. That duplication is the
intentional storytelling tool — see §3.4.

### 3.3  File extension: `.h` not `.cuh`

The new host headers use `.h`, not `.cuh`. The `.cuh` convention is
informational ("this header contains `__device__` code") and is not
enforced by `nvcc` — `.h` headers can be `#include`d from `.cu`
translation units without issue. During the transition the new `.h`
files still contain CUDA tokens and must be included only from `.cu` TUs
(or other `.cuh` files), but the extension already reflects the
*intended* end state of the file.

### 3.4  In-place progressive transformation

Once the new file is created as a namespace-renamed duplicate of its
`.cuh` sibling, all subsequent commits modify it in place. Each commit
strips, replaces, or simplifies some CUDA-specific construct. The
`.cuh` originals are never modified.

This means the diff between `nanovdb/tools/cuda/DilateGrid.cuh` and
`nanovdb/tools/DilateGrid.h` *grows over time* and is the readable
record of what was removed:

```
git diff vbm-cpu-port-pre-topology..HEAD -- nanovdb/nanovdb/tools/
```

is the topology-port story at any point during the work.

We do **not** keep the in-place transformations in the working tree
uncommitted. Each meaningful step lands as a commit on `vbm-cpu-port`
so the Draft PR's CI exercises every intermediate state.

### 3.5  Branch and worktree

| | |
|---|---|
| Active branch              | `vbm-cpu-port` (existing Draft PR's branch — provides CI + upstream diff) |
| Pre-topology snapshot      | `vbm-cpu-port-pre-topology` at commit `0c8b5827` (pushed to `origin` as a fixed reference) |
| Worktree                   | `/home/esifakis/Development/openvdb-topology-cpu-port` |

Choosing to land on `vbm-cpu-port` directly (rather than a sub-branch)
keeps the topology-port work visible in the same CI and Draft PR that
covers the other CPU-port work on that branch.

`vbm-cpu-port-pre-topology` is the rollback / clean-diff baseline.

### 3.6  Validation strategy

Each operator is validated by a `_cpu` example variant that mirrors its
existing `_cuda` sibling. Both share the same OpenVDB-reference workflow:
read an input `.vdb`, convert to NanoVDB `OnIndexGrid`, run the operator,
and bit-exact compare (`bufferCheck`) the resulting grid against an
OpenVDB-computed reference also converted to `OnIndexGrid`.

| Operator      | Validating example                                            |
|---------------|---------------------------------------------------------------|
| `DilateGrid`  | `ex_dilate_nanovdb_cpu` — bufferCheck vs OpenVDB              |
| `PruneGrid`   | `ex_dilate_nanovdb_cpu` round-trip: dilate, build mask that selects original voxels, prune; bit-exact compare against the original |
| `MergeGrids`  | `ex_merge_nanovdb_cpu` — bufferCheck vs `openvdb::tools::compSum` reference |
| `RefineGrid`  | `ex_refine_nanovdb_cpu` — bufferCheck vs OpenVDB              |
| `CoarsenGrid` | `ex_coarsen_nanovdb_cpu` — bufferCheck vs OpenVDB             |

`PruneGrid` is covered indirectly via the dilate round-trip; no
standalone `_cpu` example is needed for it.

The `_cpu` examples start as namespace-renamed copies of their `_cuda`
counterparts, with source files initially in `.cu` form (still
nvcc-required, identical behavior). They progressively shed CUDA
dependencies and eventually transition to `.cpp` form (no longer
requiring nvcc) — see §4.6.

Test inputs (under `/home/esifakis/Downloads/`):
`dragon.vdb` (66M), `torus.vdb` (4.9M), `icosahedron.vdb` (1.1M),
`iss.vdb` (135M), `space.vdb` (440M). Default for single-input
operators is `dragon.vdb`; for `merge`, `dragon.vdb + torus.vdb`.

---

## 4.  Transition Phases

Each phase below is a discrete unit of work, committed independently
so that CI validates every intermediate state. Phases are sequenced to
keep correctness measurable at all times.

### 4.0  Scaffolding — *DONE*

Commit `2b75d970`. Created `nanovdb/tools/TopologyBuilder.h` and
`nanovdb/tools/MergeGrids.h` as namespace-renamed duplicates of their
`.cuh` siblings (`tools::cuda::` → `tools::`, header guards updated to
`NANOVDB_TOOLS_*_H_HAS_BEEN_INCLUDED`). Created
`nanovdb/examples/ex_merge_nanovdb_cpu/` with `.cpp` driver and `.cu`
kernels, validating with the same dragon+torus inputs and bit-exact
`bufferCheck`.

State after Phase 4.0: the new header set is a structural duplicate of
the CUDA originals, still nvcc-required, behaviorally identical.

### 4.1  Output `BufferT` default → `UnifiedBuffer`

Change the default template argument on `MergeGrids::getHandle`
(and, when the relevant operator is ported, each other operator) from
`nanovdb::cuda::DeviceBuffer` to `nanovdb::cuda::UnifiedBuffer`:

```cpp
template<typename BufferT = nanovdb::cuda::UnifiedBuffer>
GridHandle<BufferT> getHandle(const BufferT &buffer = BufferT());
```

UnifiedBuffer is a transparent superset for both sides during the
transition: `cudaMemcpyAsync` calls keep working (managed memory is a
valid target for D2H/H2D/D2D), kernel launches still write to the
allocation, and host code can read it directly without explicit copies.

Caveat noted in §5.2: concurrent host/device access to a single
managed allocation has restrictions; the existing pipeline's explicit
`cudaStreamSynchronize` between phases satisfies them.

### 4.2  Internal scratch buffers → `ScratchBufferT` alias defaulting to `UnifiedBuffer`

Introduce in `TopologyBuilder`:

```cpp
using ScratchBufferT = nanovdb::cuda::UnifiedBuffer;
```

and rewrite the scratch member types and the local
`create(...)` calls to use `ScratchBufferT` instead of
`nanovdb::cuda::DeviceBuffer`. This applies to the **device-only**
buffers identified in §6:

- Member: `mUpperMasks`, `mLowerMasks`, `mUpperOffsets`, `mLowerOffsets`,
  `mLeafOffsets`, `mVoxelOffsets`, `mLowerParents`, `mLeafParents`.
- Local in `countNodes`: `upperCountsBuffer`, `lowerCountsBuffer`,
  `leafCountsBuffer`.

The **dual-mode** buffers (`mData`, `mProcessedRoot`) are kept as
`DeviceBuffer` for the moment, since their host/device coordination via
`deviceUpload` is non-trivial and is the topic of §4.4 below.

The CUB temp pool (`mTempDevicePool`) stays on the stream-ordered
allocator path until §4.5.

API friction to budget for: `DeviceBuffer::clear(stream)` vs
`UnifiedBuffer::clear()` (no stream arg). Two options:
1. Add an `UnifiedBuffer::clear(cudaStream_t)` overload that ignores the
   stream argument (with a doc comment explaining the managed-memory
   restriction documented in §5.3).
2. Drop the stream arg at the call sites in `TopologyBuilder` when the
   alias swap happens.

Option (1) keeps the swap surgical; either is reasonable.

### 4.3  Replace lambdaKernel launches with `util::forEach`

Each `util::cuda::lambdaKernel<<<…>>>` launch in `TopologyBuilder.h` is
replaced with a `nanovdb::util::forEach`-style host loop over the same
index range. The body of each functor (which is already a plain
`__device__ void operator()(size_t, …)`) is reused as-is, dropping the
`__device__` qualifier.

Functors affected (all in `TopologyBuilder.h`):

- `BuildGridTreeRootFunctor`
- `BuildUpperNodesFunctor`
- `UpdateLeafVoxelCountsAndPrefixSumFunctor`
- `UpdateLeafVoxelOffsetsFunctor`
- `UpdateAndPropagateLeafBBoxFunctor`
- `PropagateLowerBBoxFunctor`
- `PropagateUpperBBoxFunctor`
- `UpdateRootWorldBBoxFunctor`
- `PostProcessGridTreeFunctor`

Also affected in operator headers: the `lambdaKernel`-driven loops in
`countNodes` (the upper-counts-from-lower-offsets pass) and the
`processGridTreeRoot` invocation.

BBox propagation needs an atomic story. The CUDA code uses
`expandAtomic` because children race into parents. On host we either:

- Keep it `std::atomic`-based — minimal change, ~6 int compares per
  child, negligible contention; or
- Restructure to a per-parent reduction (each parent visits its
  children) — eliminates atomics but reshapes the loop.

Either is acceptable. The first is closer to a 1-to-1 port and is the
default choice.

### 4.4  `mData` and `mProcessedRoot` — the dual-mode buffers

These two `DeviceBuffer` instances are not interchangeable with the
scratch set because they exhibit a host-write / `deviceUpload` /
device-read pattern. Under `UnifiedBuffer`:

- `deviceUpload` becomes a prefetch hint, not a memcpy.
- `data()` and `deviceData()` return the same pointer.
- The host-side `data()->nodeCount[i]` writes (in `countNodes`,
  `getBuffer`) and the device-side reads via `d_data` resolve through
  the same memory.

Under `HostBuffer` (end state):

- `deviceUpload` collapses to a no-op or is removed entirely.
- `data()` is the only accessor that makes sense.

This phase switches the two dual-mode allocations to `UnifiedBuffer`
explicitly (so the API surface in `TopologyBuilder`'s accessors
— `data()`, `deviceData()`, `hostProcessedRoot()`,
`deviceProcessedRoot()` — degrades gracefully). Once **all** stages
that exercise them are host code (after §4.3 and §4.5), the accessors
can be collapsed to a single host pointer and the `deviceUpload` calls
removed.

### 4.5  Morphology functors — host equivalents

The deepest transformation. The CUDA-side morphology functors in
`nanovdb/util/cuda/Morphology.cuh` use `__device__` qualifiers, shared
memory, and warp-level mask cooperation. The `operatorKernel<<<dim3(N,
slicesPerLowerNode, 1), MaxThreadsPerBlock>>>` launches presume a CUDA
execution model.

The host port replaces each functor used by the operator-under-port
with a host-side equivalent in a new file
`nanovdb/util/Morphology.h`. The new file is introduced incrementally,
starting with the functors used by `MergeGrids` (because that's the
first operator), then expanding as other operators are ported.

CPU strategy options (pick per functor):

- **Serial-by-block:** each CPU thread does one logical CUDA block;
  inner slice loop is sequential. Straightforward 1-to-1 port. Wins
  no parallelism over the per-slice mask cooperation but is correct
  and simple.
- **SIMD over 64-bit mask words:** reclaim some of the warp parallelism
  using the existing `nanovdb::util::Simd` infrastructure (already
  developed on `vbm-cpu-port` for the WENO stack). Higher payoff but
  larger upfront cost; defer until the serial port is green.

Default plan: serial-by-block first; SIMD-over-masks as a follow-up
optimization once the port is functionally complete.

### 4.6  CUB scans → `std::inclusive_scan`

`CALL_CUBS(DeviceScan::InclusiveSum, …)` invocations in `countNodes`
and `processLeafOffsets` map to `std::inclusive_scan` (or, when
parallelism is wanted, `std::inclusive_scan` with an execution policy
or a TBB parallel scan). The CUB temp-pool machinery
(`mTempDevicePool`, `DeviceResource`, `CALL_CUBS` macro) is deleted
along with this change.

### 4.7  Drop CUDA artifacts; transition source files `.cu` → `.cpp`

Once all stages of the operator's pipeline are host code, the
remaining work is mechanical:

1. Remove `<cub/cub.cuh>`, `<nanovdb/util/cuda/*>`, `<nanovdb/cuda/*>`
   includes from the operator's `.h`.
2. Switch the `mData`/`mProcessedRoot` allocations from `UnifiedBuffer`
   to `HostBuffer`. Collapse the host/device accessor pairs to a
   single accessor.
3. Switch the operator's `getHandle` default `BufferT` from
   `UnifiedBuffer` to `HostBuffer`.
4. In the validating `_cpu` example, fold the contents of
   `<op>_nanovdb_cpu_kernels.cu` into `<op>_nanovdb_cpu.cpp` (or
   rename `.cu` → `.cpp`).
5. The CMake `nanovdb_example` function's globbing handles `.cpp`
   sources unconditionally and only invokes nvcc when a `.cu` exists.
   So step 4 alone removes nvcc from the example's build chain.
6. Remove the `\warning … include only from .cu files …` doxygen note
   from the operator's `.h` header.

The CMake-level transition from a CUDA-requiring target to a host-only
target is the completion signal for the operator.

### 4.8  Extension across the remaining operators

After Phase 4.7 lands for `MergeGrids`, the same sequence (Phases 4.1
through 4.7) is applied to the remaining operators in this order:

1. **`CoarsenGrid`** — simplest `xxxRoot()` (each source tile maps to
   exactly one coarsened tile, no bbox-based gating, no octant
   decomposition, no neighborhood expansion). Smallest delta from
   `Merge`'s pattern.
2. **`RefineGrid`** — 8 octant tiles per source tile, bbox-gated; the
   speculative-then-cull pattern shows up but is otherwise mechanical.
3. **`DilateGrid`** — 26-neighbor expansion with bbox-proximity gating;
   also has the `NN_FACE` / `NN_FACE_EDGE` / `NN_FACE_EDGE_VERTEX`
   variants in `dilateInternalNodes` and `dilateLeafNodes`.
   Note: `NN_FACE_EDGE` at the leaf level is unimplemented in the CUDA
   version and throws — preserve this behavior on host.
4. **`PruneGrid`** — second-simplest `xxxRoot()` (same tiles as
   source, mask-based reduction at lower levels). But the public
   constructor takes a `Mask<3>*` sidecar — the leaf-mask input — which
   in the validating example is produced by the dilate round-trip
   inside `ex_dilate_nanovdb_cpu`.

Each operator inherits the morphology-functor work done for previous
operators, but typically introduces one new functor specific to that
operator (`MergeInternalNodesFunctor`, `DilateInternalNodesFunctor`,
`PruneInternalNodesFunctor`, `RefineInternalNodesFunctor`,
`CoarsenInternalNodesFunctor`, plus their leaf-side siblings). These
are added to the host `Morphology.h` as they become needed.

`TopologyBuilder.h` is touched only by the *first* operator port
(`MergeGrids`) for the structural work (Phases 4.1–4.4 land on
`TopologyBuilder.h` and `MergeGrids.h` together). Subsequent operator
ports inherit the already-host-side `TopologyBuilder.h` and need only
modify their own operator header.

---

## 5.  Key Technical Notes

### 5.1  `UnifiedBuffer` API surface — drop-in compatibility

UnifiedBuffer matches DeviceBuffer's API for the call patterns that
`TopologyBuilder` uses:

| Call pattern              | DeviceBuffer | UnifiedBuffer |
|---------------------------|--------------|---------------|
| `create(size, ref, dev, stream)` | ✓     | ✓             |
| `data()` / `deviceData()` | ✓ (different pointers) | ✓ (same pointer) |
| `deviceUpload(dev, stream, sync)` | ✓ | ✓ (prefetch)  |
| `clear(stream)`           | ✓            | **missing**   |
| `clear()`                 | —            | ✓             |

The single mismatch is `clear(stream)` (see §4.2).

### 5.2  Concurrent host/device access to managed memory

`cudaMallocManaged` allocations are well-defined for concurrent
host/device access only on Pascal+ devices with `concurrentManagedAccess`
enabled, and even then, the host cannot read/write a range while a
kernel is operating on it. The existing pipeline already has explicit
`cudaStreamSynchronize` between phases (e.g., before `getBuffer`,
before `dilateLeafNodes`), which satisfies the constraint. During the
transition, when individual stages flip from kernel to host loop, the
serial single-stream execution model preserves the invariant.

### 5.3  `cudaFreeAsync` is incompatible with `cudaMallocManaged`

Empirically verified: `cudaFreeAsync(ptr, stream)` on a
`cudaMallocManaged` allocation returns `cudaErrorNotSupported`
(101). The CUDA stream-ordered allocator is paired with `cudaMallocAsync`
only; managed memory has no stream-ordered free counterpart. Plain
`cudaFree` is the only valid deallocator for managed pointers, and it
synchronously drains all outstanding work on all streams that touched
the allocation.

This is why `UnifiedBuffer::clear()` cannot honor a stream argument
even in principle — see test artifact at `/tmp/test_freeasync_managed.cu`.

### 5.4  Build configuration gotcha — `CMAKE_CUDA_ARCHITECTURES`

NanoVDB's default `CMAKE_CUDA_ARCHITECTURES=75` (Turing) compiles
CUDA objects with only `sm_75` SASS and no PTX. On hardware that's
not sm_75 (e.g., RTX 6000 Ada is sm_89), kernel launches silently
fail; the deferred error then surfaces at the next checked CUDA call
as a misleading `cudaErrorInvalidDevice` (101).

Always configure with `-DCMAKE_CUDA_ARCHITECTURES=native` (or the
explicit arch for the target device) on systems past Turing. Document
this for any new developer touching the topology-port branch.

### 5.5  Two dual-mode `DeviceBuffer` instances, all other uses device-only

Audit findings (in `TopologyBuilder.h` and `MergeGrids.h` as of
Phase 4.0):

**Dual-mode** (host writes → `deviceUpload` → device reads):

| Buffer            | Owner            | Role                                                  |
|-------------------|------------------|-------------------------------------------------------|
| `mData`           | `TopologyBuilder` | The `Data` POD: byte offsets, node counts, etc.       |
| `mProcessedRoot`  | `TopologyBuilder` | The speculative new-topology RootNode + tile list     |

Both follow the same pattern: single-arg `create`, host fills, then
`deviceUpload(device, stream, false)`, then kernels read via
`.deviceData()`. The host copy persists throughout (host-side reads of
`hostProcessedRoot()->tileCount()` happen in
`allocateInternalMaskBuffers` and `countNodes`).

**Device-only** (allocated, then accessed only via `.deviceData()`):

- Member: `mUpperMasks`, `mLowerMasks`, `mUpperOffsets`, `mLowerOffsets`,
  `mLeafOffsets`, `mVoxelOffsets`, `mLowerParents`, `mLeafParents`.
- Local in `countNodes`: `upperCountsBuffer`, `lowerCountsBuffer`,
  `leafCountsBuffer`.
- CUB temp: `mTempDevicePool` (different type — `TempPool<DeviceResource>`,
  backed by `cudaMallocAsync`).

**Already host-only** (not `DeviceBuffer` — `nanovdb::HostBuffer`):

- In each operator's `xxxRoot()`: the `srcRootBuffer{1,2}` instances that
  receive D→H copies of input grids' RootNodes for host-side traversal.
  These need no transition; they're already pure host.

This dichotomy informs the Phase 4.2 split: device-only members move to
`UnifiedBuffer` via the `ScratchBufferT` alias as a single batch; the
two dual-mode members migrate separately in Phase 4.4 because their
accessor pairs require coordinated treatment.

---

## 6.  Build Configuration

The topology-port worktree uses:

```
cd /home/esifakis/Development/openvdb-topology-cpu-port/build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NANOVDB=ON \
      -DNANOVDB_BUILD_EXAMPLES=ON \
      -DNANOVDB_USE_CUDA=ON \
      -DNANOVDB_USE_OPENVDB=ON \
      -DCMAKE_CUDA_ARCHITECTURES=native \
      ..
```

CUDA stays on throughout the port (the in-progress `.h` files
include `cuda.h` until late phases). When the last operator transitions
to `.cpp`-sourced, the host-only target stops invoking nvcc — but
`NANOVDB_USE_CUDA=ON` remains useful for keeping the `_cuda` siblings
buildable for regression comparison.

Validation runs:

```
./ex_merge_nanovdb_cpu  /path/to/dragon.vdb /path/to/torus.vdb 3
./ex_dilate_nanovdb_cpu /path/to/dragon.vdb 3   # also exercises PruneGrid
./ex_refine_nanovdb_cpu /path/to/dragon.vdb 3
./ex_coarsen_nanovdb_cpu /path/to/dragon.vdb 3
```

Each should print `Result of <Op> check out CORRECT against reference`
and then a small number of warm-start timing lines.

---

## 7.  Current Status

### 7.1  Phase tracker

| Phase | Description                                | Status      |
|-------|--------------------------------------------|-------------|
| 4.0   | Scaffolding: TopologyBuilder.h + MergeGrids.h + ex_merge_nanovdb_cpu | **Done** (`2b75d970`) |
| 4.1   | Output `BufferT` default → `UnifiedBuffer`  | **Done** (`0179f8c3`) |
| 4.2   | Scratch buffers → `ScratchBufferT` alias    | **Done** (`003b0a59`) |
| 4.3   | lambdaKernel/operatorKernel launches → `util::forEach` | **Done** for `MergeGrids` (see 7.2) |
| 4.4   | Working buffers → `HostBuffer` (straight to host; skipped the UnifiedBuffer step) | **Done** for Merge/Dilate/Prune (see 7.3: scratch, mProcessedRoot, mData, output buffer, streams) |
| 4.5   | Morphology functors → host equivalents (MergeGrids subset) | **Done** for `MergeGrids` (see 7.2) |
| 4.6   | CUB scans → `util::inclusiveScan`           | **Done** (`858afe79`, `ed109ddd`) |
| 4.7   | Drop CUDA artifacts; `.cu` → `.cpp`         | In progress — data path CUDA-free; dead code/includes + `.cu`→`.cpp` remain (see 7.3 items 10-12) |
| 4.8   | Repeat 4.1–4.7 for the remaining operators | **In progress** — `DilateGrid` + `PruneGrid` kernel-free (see 7.7); Coarsen/Refine pending |

### 7.2  Per-method porting status (MergeGrids `getHandle` pipeline)

As of `7e5f9656`, **every compute stage of the `MergeGrids` pipeline runs on
the host.** No `lambdaKernel`/`operatorKernel` launches remain in either
`tools/MergeGrids.h` or `tools/TopologyBuilder.h`.

| Stage (call order in `getHandle`)        | Owner          | Status | Commit      | Notes |
|------------------------------------------|----------------|--------|-------------|-------|
| `mergeRoot`                              | MergeGrids     | HOST   | `2b75d970`  | `std::map` tile merge into `mProcessedRoot` (now HostBuffer); still D2H-copies the source roots (dropped in Step 4 §7.3) |
| `allocateInternalMaskBuffers`            | TopologyBuilder| alloc  | —           | No kernel; `cudaMemsetAsync` zero-fill of `mUpperMasks`/`mLowerMasks` |
| `mergeInternalNodes`                     | MergeGrids     | HOST   | `b7f41d26`  | `util::morphology::MergeInternalNodes`; OR into lower masks + `setOnAtomic` upper masks |
| `countNodes`                             | TopologyBuilder| HOST   | `858afe79`, `67e8d5c0` | `EnumerateNodes` + `util::inclusiveScan` |
| `getBuffer`                              | TopologyBuilder| alloc  | —           | No kernel; buffer alloc + `cudaMemsetAsync` + `mData.deviceUpload` |
| `processGridTreeRoot`                    | MergeGrids     | HOST   | `7e5f9656`  | host `std::memcpy` of GridData + `BuildGridTreeRootFunctor` |
| `processUpperNodes`                      | TopologyBuilder| HOST   | `729804db`  | `util::forEach` |
| `processLowerNodes`                      | TopologyBuilder| HOST   | `4438dfe9`  | `util::morphology::ProcessLowerNodes` |
| `mergeLeafNodes`                         | MergeGrids     | HOST   | `a73eb2b7`  | `util::morphology::MergeLeafNodes` (flat over source leaves) |
| `processLeafOffsets`                     | TopologyBuilder| HOST   | `ed109ddd`  | `util::forEach` ×2 + `util::inclusiveScan` |
| `processBBox`                            | TopologyBuilder| HOST   | `fe80448c`  | `util::forEach`; host `expandAtomic` (see 7.4) |
| `postProcessGridTree`                    | TopologyBuilder| HOST   | `28e0d65a`  | direct host call of `PostProcessGridTreeFunctor` |

Host morphology functions now living in `util/Morphology.h` (parallel to the
CUDA `util/cuda/Morphology.cuh`): `EnumerateNodes`, `ProcessLowerNodes`,
`MergeInternalNodes`, `MergeLeafNodes`. All take ownership of their own
`util::forEach` parallelization (word-granular or per-node), distinct from the
CUDA cooperative-block/warp structure — this divergence is by design.

### 7.3  Shared host-only cleanup (transitional plumbing, not compute)

This is the shared Phase 4.4/4.7 sweep across the three ported operators
(Merge/Dilate/Prune) and `TopologyBuilder`. The end state is no device memory at
all: all working buffers `HostBuffer`, inputs presumed host-resident, headers
compiling as pure C++ (no `cuda.h`). UnifiedBuffer was only a transitional crutch;
we migrate straight to `HostBuffer`.

**Done — `TopologyBuilder`'s own working state is now host-only:**

1. **Scratch (masks/offsets/parents) → `HostBuffer`** (`db146183`). `ScratchBufferT
   = nanovdb::HostBuffer`; `create()` drops the device/stream args; mask zero-fills
   `cudaMemsetAsync` → `std::memset`; `deviceUpperMasks()/deviceLowerMasks()`
   deleted.
2. **`mProcessedRoot` → `HostBuffer`** (`56f480b8`). It shares the `ScratchBufferT`
   policy (transient scratch); the alias is now public on `TopologyBuilder` and the
   operators alias *it* (`typename TopologyBuilder<BuildT>::ScratchBufferT`) rather
   than re-declaring. Dropped the `<op>Root` `deviceUpload` and `deviceProcessedRoot()`.
3. **`mData` → plain `Data` member** (`5fee27c3`) and **`Data::d_upperOffsets`
   removed** (`82f71cde`). No `DeviceBuffer`, no `deviceUpload`, no `deviceData()`;
   `processUpperNodes` reads `mUpperOffsets` directly. (Verified mData never had an
   intentional host/device divergence — the device copy was a pure upload target
   with no reader.) `Data` is now `{d_bufferPtr, grid..size offsets, nodeCount[3]}`
   + `get*` accessors; `data()` is a trivial `&mData` accessor, kept (used ~60×).
4. **`updateChecksum` → host** (`333c942d`, see §7.8).

**Done — output buffer, inputs, streams, timer (the whole data path is host):**

5. **`getBuffer` is backend-agnostic + CUDA-free** (`8d8993ec`): `BufferT::create(size,&pool)`
   + `std::memset(buffer.data())` + `d_bufferPtr = buffer.data()`. Works for `HostBuffer`
   (the new default) and `UnifiedBuffer` alike, no behavior change.
6. **All three operators default `getHandle` `BufferT` → `HostBuffer`** and read inputs
   host-resident: merge `8d8993ec`, dilate+prune `7eb3a506`. `getTreeData` (D2H via
   `DeviceGridTraits`) → `mSrcTreeData = srcGrid->tree()`; `mergeRoot`'s source-root D2H
   dropped (Dilate/Prune already read the source directly). Examples: all grids `HostBuffer`,
   `deviceGrid()`→`grid()`, `bufferCheck`→`memcmp`, no `deviceUpload`.
7. **Host `InjectGridMask`** (`79319066`, `util/Injection.h`): the dilate→prune round-trip's
   injection sidecar helper is now a host `util::forEach` (it was the one remaining example
   kernel). The dilate example's `mainDilateGrid` has no kernel launches.
8. **Timer → host `util::Timer`** (`dae9740a`) in all operators + examples.
9. **`cudaStream_t` plumbing ripped out** (`d5a962bf`): no async op / kernel / `cudaMemcpy*` /
   `deviceUpload` remains, so every `cudaStreamSynchronize` drained an empty stream. Dropped the
   stream param from every `TopologyBuilder`/operator method + ctor, the `mStream` members, and
   all syncs. Purely mechanical — nothing was stream-ordered.

So the operators' data path is fully host and CUDA-free in behavior; all three validate CORRECT
(merge across pairings; dilate→inject→prune round-trip at both stencils).

**Remaining (cosmetic / build-capability, not behavioral):**

10. **Dead code** — `TopologyBuilder::mNumThreads`/`numBlocks` + `mTempDevicePool` + the
    `CALL_CUBS` macro (the last things naming a now-gone `stream`, inert/unexpanded), and the
    per-operator `mNumThreads`/`numBlocks`.
11. **Include purge** — once 10 lands, drop `#include <cub/cub.cuh>`, `util/cuda/Morphology.cuh`,
    `util/cuda/DeviceGridTraits.cuh`, `util/cuda/Util.h` (`cudaCheck`), `cuda/TempPool.h`,
    `cuda/DeviceBuffer.h`, `cuda/UnifiedBuffer.h` from the four headers (and the unused cuda
    buffer includes from the example `.cpp`s). Audit each for any remaining user.
12. **`.cu` → `.cpp` (completion signal + proof).** Fold each `*_kernels.cu` into its `.cpp`;
    once no `.cu` remains and the CUDA includes are gone, CMake stops invoking nvcc for the
    example. Remove the `\warning … include only from .cu files …` doxygen notes. This compile
    under plain g++ (no `__CUDACC__`) is what *proves* the headers are CUDA-free — nvcc's host
    pass is more permissive, so this step may surface a latent dependency (a build fix, not a
    behavior change).

Note (scope): this records the host port of Merge/Dilate/Prune. CoarsenGrid/RefineGrid are still
unported (§7.7); they will be written directly in the now-established clean (HostBuffer, no-stream)
style.

### 7.4  Core-header changes (affect both host and CUDA builds)

The host port required making three atomics host-callable. These are shared
NanoVDB core headers, so they affect every consumer:

- **`util/Util.h`** (`fe80448c`): added `util::atomicMin`/`util::atomicMax`
  (`__hostdev__`, templated), following the existing `util::atomicOr`/`atomicAnd`
  precedent. Integer min/max has no direct host intrinsic, so the host paths are
  a compare-and-swap retry loop (`std::atomic_ref` for C++20,
  `__atomic_compare_exchange_n` for GCC/clang, `_InterlockedCompareExchange` for
  MSVC); the device path stays the native `::atomicMin`/`::atomicMax` intrinsic,
  so GPU codegen is unchanged.
- **`math/Math.h`** (`fe80448c`): moved `Coord`/`Coord2` `min/maxComponentAtomic`
  and `BBox::expandAtomic`/`intersectAtomic` out of `#if defined(__CUDACC__)` to
  `__hostdev__`, routing through the new util functions. Mirrors the earlier
  `Mask::setOnAtomic → util::atomicOr` migration.
- A latent missing `typename` on `NanoRoot<BuildT>::ValueType`/`FloatType` in
  `BuildGridTreeRootFunctor` surfaced once the functor became `__hostdev__`
  (the device-only compile pass had been lenient); fixed in `7e5f9656`.

### 7.5  Testing status

- **`ex_merge_nanovdb_cpu`**: prints `Result of MergeGrids check out CORRECT
  against reference` (bit-exact `bufferCheck` vs the OpenVDB-built reference)
  across `dragon+armadillo`, `armadillo+dragon`, `dragon+iss`, and `iss+space`
  pairings, and is stable over repeated runs (exercises concurrent
  `expandAtomic`/`setOnAtomic` contention under TBB).
- **Core regression** (run after the `Math.h`/`Util.h` atomic migration,
  `fe80448c`): `nanovdb_test_nanovdb` 152/152 and `nanovdb_test_cuda` 50/50
  pass. The CUDA suite includes the device-side `expandAtomic` path via the
  `DilateInjectPrune`/`RefineCoarsen`/`MergeGrids` operator tests.
  - Note: the host test writes to a `data/` directory relative to cwd; create
    it first or several I/O tests fail spuriously (unrelated to the port).
  - **Pending re-run:** the unit tests have not been re-run since the core
    `typename` fix (`7e5f9656`); that change only affects the host port header
    (`tools/TopologyBuilder.h`), not the CUDA `tools/cuda/TopologyBuilder.cuh`
    the tests exercise, but a confirming re-run is advisable.

### 7.6  Build / environment notes

- Build dir: `nanovdb/nanovdb/build/`. OpenVDB installed at `~/local`; cmake
  requires `-DOpenVDB_ROOT=~/local` explicitly (the repo-local
  `cmake/FindOpenVDB.cmake` does not auto-detect the prefix). See `CLAUDE.md`
  for the full invocation.
- `-DCMAKE_CUDA_ARCHITECTURES=120` for the Blackwell test machine (the default
  `=75` causes silent kernel failures on newer hardware).
- Unit tests require `-DNANOVDB_BUILD_UNITTESTS=ON` and
  `-DGTest_DIR=~/local/google/googletest/lib/cmake/GTest`.

### 7.7  Extension to the other operators (Phase 4.8)

`MergeGrids` is the complete worked template for the host pattern. The remaining
operators (`CoarsenGrid`, `RefineGrid`, `DilateGrid`, `PruneGrid`) reuse the
already-host `TopologyBuilder` pipeline (7.2) unchanged and need only their own
operator-specific functors ported into `util/Morphology.h`.

#### DilateGrid — fully ported (`tools/DilateGrid.h`, `ex_dilate_nanovdb_cpu`)

All four operator methods run on the host; `DilateGrid.h` has **no kernel launches and no
D2H copies**, at parity with (and in `dilateRoot`'s case ahead of) `MergeGrids.h`. The only
remaining device op is the output `mProcessedRoot.deviceUpload`, the shared Phase 4.4
dual-mode-buffer item. The TopologyBuilder pipeline it rides on is already host.

| Stage (call order in `getHandle`) | Owner | Status | Commit | Notes |
|-----------------------------------|-------|--------|--------|-------|
| `dilateRoot`           | DilateGrid | **HOST** | `38fa0259` | host `std::map` tile speculation reading source root/uppers directly from the managed grid (D2H copy dropped); only output `mProcessedRoot.deviceUpload` remains |
| `dilateInternalNodes`  | DilateGrid | **HOST** | `180b081a` | `util::morphology::DilateInternalNodes`; built on the validated host `MaskShift` |
| `processGridTreeRoot`  | DilateGrid | **HOST** | `ce255009` | host `std::memcpy` of GridData + direct `BuildGridTreeRootFunctor` call (verbatim reuse of merge's) |
| `dilateLeafNodes`      | DilateGrid | **HOST** | `01105613` | `util::morphology::DilateLeafNodes` (`NN_FACE`, `NN_FACE_EDGE_VERTEX`; `NN_FACE_EDGE` throws) |

New host pieces (validated three ways for `MaskShift`: brute-force oracle,
exhaustive single-bit sweep, and a bit-exact differential vs the device `MaskShift`):

- `util/MorphologyHelpers.h`: `util::shuffleDown`/`shuffleUp` (unmasked flat-array
  shuffles) and the host `util::morphology::MaskShift` (`6e7401eb`).
- `util/Morphology.h`: `DilateInternalNodes` (offset-mask aggregation → 27-direction
  `MaskShift` block → atomic scatter) and `DilateLeafNodes`.

`ex_dilate_nanovdb_cpu` (UnifiedBuffer source + getHandle) checks CORRECT against
`dilateActiveValues` across dragon/armadillo/iss/space for both `NN_FACE` (the
example default) and the full 26-connected `NN_FACE_EDGE_VERTEX`. It also runs the
dilate→prune round-trip (see PruneGrid below).

Remaining for DilateGrid is only the shared Phase 4.4/4.7 plumbing (dual-mode
`mData`/`mProcessedRoot` → `UnifiedBuffer`, dead-code/`cub` removal, `.cu` → `.cpp`).

Also done as part of this work: the eight host `tools::topology::detail` functor
`operator()`s in `TopologyBuilder.h` dropped their transitional `__hostdev__` (they are
host-only now that every operator's `getHandle` is kernel-free) — `517f94b1`.

#### PruneGrid — fully ported (`tools/PruneGrid.h`, via `ex_dilate_nanovdb_cpu`)

All four operator methods run on the host; `PruneGrid.h` has no kernel launches and no D2H
copies (only the shared Phase 4.4 `mProcessedRoot.deviceUpload` remains). Both functors are
thread-centric — no `MaskShift`-class complexity — so PruneGrid went in following the dilate
playbook in one pass (`e86cdee2`).

| Stage (call order in `getHandle`) | Owner | Status | Notes |
|-----------------------------------|-------|--------|-------|
| `pruneRoot`           | PruneGrid | **HOST** | duplicates all source tiles (read directly from the managed grid; pruned later if empty) |
| `pruneInternalNodes`  | PruneGrid | **HOST** | `util::morphology::PruneInternalNodes`: retain a source leaf iff its value mask intersects the sidecar; set pruned upper/lower masks via `setOnAtomic` |
| `processGridTreeRoot` | PruneGrid | **HOST** | host `std::memcpy` + direct `BuildGridTreeRootFunctor` call |
| `pruneLeafNodes`      | PruneGrid | **HOST** | `util::morphology::PruneLeafMasks`: output leaf value mask = source value mask ∩ sidecar (flat over source leaves, no atomics) |

`PruneGrid` takes a `Mask<3>*` leaf-mask sidecar marking the voxels to retain. In the example
the sidecar is the original (un-dilated) topology, injected per dilated leaf by the CUDA
`InjectGridMaskFunctor` (an example helper, not part of PruneGrid, kept a kernel for now) into a
`UnifiedBuffer` so the host operator can read it. The dilate→inject→prune round-trip recovers the
original grid bit-exact: CORRECT for both DilateGrid and PruneGrid across dragon/armadillo/iss/
space at `NN_FACE` and `NN_FACE_EDGE_VERTEX`.

#### Still to port

- `CoarsenGrid`, `RefineGrid`: their `Coarsen*`/`Refine*` internal- and leaf-node functors,
  following the `Merge*`/`Dilate*`/`Prune*` model.

All operators continue to pass bit-exact `bufferCheck` against the OpenVDB reference; the port
maintains this invariant at every commit.

### 7.8  Performance observations (MergeGrids, indicative)

Measured on the dragon+armadillo pair (build machine, sm_120), comparing
`ex_merge_nanovdb_cpu` against `ex_merge_nanovdb_cuda`. Numbers are indicative,
not a benchmark — there is meaningful run-to-run jitter (TBB scheduling,
managed-memory residency), so treat them as ballpark.

**Overall:** warm-start total ≈ **CPU 10–11 ms vs CUDA ~0.8 ms ≈ 13×**. A ~13×
host-vs-GPU gap on sparse topology work is expected and not pathological.

**Per-stage (first/cold call), the two dominant CPU stages:**

| Stage                | CPU cold | CPU warm | CUDA  |
|----------------------|---------:|---------:|------:|
| `processLowerNodes`  | ~4.9 ms  | ~4.5 ms  | 0.09 ms |
| `mergeLeafNodes`     | ~3.9 ms  | ~2.1 ms  | 0.10 ms |
| `mergeInternalNodes` | ~2.2 ms  | ~1.0 ms  | 1.25 ms |
| (all others)         | small    | small    | small |

**Key findings:**

1. **Not a one-time cold-start artifact.** The 5 warm-start iterations stay flat
   at ~10–11 ms with no decay toward the CUDA figure; a true warmup artifact
   would show iteration 1 ≫ iteration 5.

2. **Cold→warm decay splits by *which buffer* a stage touches.**
   - `mergeInternalNodes`/`mergeLeafNodes` roughly **halve** cold→warm. They read
     the **source grids**, which persist across `getHandle` calls (same
     converter), so their managed pages migrate device→host once and stay
     host-resident — first-touch cost, genuinely cold-start.
   - `processLowerNodes` **barely moves** (4.9→4.5). It works on the **output
     grid buffer**, which `getBuffer` freshly allocates and `cudaMemsetAsync`-
     zeroes *on the device* every call, so the host re-faults those pages on
     **every** call. This is a *recurring* hybrid-buffer migration cost, not
     warmup — and it is expected to drop once Phase 4.4/4.7 move the
     allocation/zero-fill host-side (no device-resident output pages to migrate).

3. **`forEach` grain size is not the lever.** A quick test bumping
   `ProcessLowerNodes`' grain from 1 → 64 (4096 word-tasks → 64 chunks) did
   **not** improve the stage (5.49 vs 4.94 ms cold, within jitter). So the cost
   is real work + page migration, not TBB task-scheduling overhead from
   `grain=1`. Conclusion: **defer any `forEach`-grain / perf tuning until after
   the Phase 4.4/4.7 buffer cleanup**, then re-measure — the cleanup is likely
   to change the hot-spot ranking and remove the migration noise that currently
   contaminates these numbers.

The host checksum swap (`333c942d`) already removed one such artifact:
`postProcessGridTree` dropped from ~1.64 ms to ~0.17 ms once it stopped invoking
the device checksum kernel on a host-written managed buffer.

#### 7.8.1  Three-way benchmark (OpenVDB CPU vs nanovdb host vs nanovdb CUDA)

Measured after the streams/round-trip removal and the HostBuffer flip (warm
steady-state for the nanovdb paths; the OpenVDB column is the example's own
ground-truth reference time). Build machine, sm_120.

| Case                    | OpenVDB CPU (ref) | nanovdb host `_cpu` | nanovdb CUDA `_cuda` |
|-------------------------|------------------:|--------------------:|---------------------:|
| Merge dragon+armadillo  |  344 ms           |  43 ms              |  0.8 ms              |
| Merge iss+space         | 1571 ms           | 122 ms              |  4.0 ms              |
| Dilate dragon (NN_FACE) | 16.7 ms           |  37 ms              |  1.0 ms              |
| Dilate space (NN_FACE)  |  106 ms           | 120 ms              |  2.9 ms              |

Grid sizes: dragon ~23M active voxels, armadillo ~23M, iss 85.7M, space 165.9M.

**Findings:**

1. **Merge: the host port is ~8× (small) to ~13× (large) *faster* than
   OpenVDB.** OpenVDB's merge is `deepCopy` + `compSum` on the `FloatGrid` (full
   grid copy + value-merge tree surgery); the nanovdb host path is a
   topology-union rebuild on the `OnIndexGrid` and is much cheaper.

2. **Dilate: roughly comparable** — ~2.2× slower on dragon, ~1.1× (par) on
   space. OpenVDB's `dilateActiveValues` is a mature in-place topology dilation;
   the nanovdb host path matches it at scale despite carrying the
   densified-scratch alloc+zero overhead (~80% of host dilate time).

3. **Compute-only, the host dilate already beats OpenVDB:** stripping the
   alloc/zero, host dilate compute is ~24 ms vs OpenVDB's 106 ms at `space`
   (~4× faster). The scratch-reuse / lazy-zero optimization (Phase 4.x) would
   therefore move host dilate from "par with OpenVDB" to "clearly faster," while
   the GPU stays another ~30–50× ahead.

**Caveats (orders of magnitude are solid, exact ratios are not):** the OpenVDB
numbers are single cold runs vs nanovdb warm steady-state; OpenVDB operates on
the `FloatGrid` (values + topology) while nanovdb operates on the `OnIndexGrid`
(topology only); and output semantics differ (OpenVDB merge yields a value
grid, nanovdb yields an index grid). The takeaway is that against the relevant
CPU baseline (OpenVDB), the host port is *faster* for merge and *on par* for
dilate — it is a legitimate CPU option, not merely a slow GPU fallback.

### 7.9  Future refactoring (TODO)

- **Share the leaf-dilation body between host and CUDA via a `__hostdev__` helper.**
  The leaf-dilation functors (`DilateLeafNodesFunctor` and the host
  `util::morphology::DilateLeafNodes`) are *thread-centric* — one thread per
  output leaf, no warp/CTA cooperation — and every primitive they use
  (`probeLeaf`, `valueMask`, `origin`, the word bit-ops) is already
  `__hostdev__`. So the per-leaf body (gather neighbor leaves, register/word
  dilation, write the output mask) could be factored into a single `__hostdev__`
  function parameterized by the nearest-neighbor stencil, called by both the host
  `util::forEach` and the CUDA `operatorKernel`. This would remove the current
  duplication and its double-maintenance cost — e.g. the `[10][3][3]` stencil
  UB fix (`01105613` host, `c199fcb3` cuda) had to be applied twice. Worth doing
  once the operator set stabilizes.
  - Note the contrast: `DilateInternalNodes` is **not** a candidate — it relies
    on warp-cooperative `MaskShift` shuffles and `cub::WarpReduce`, so its host
    and device decompositions differ in kind, not just in iteration harness.
