# CPU Port Plan for NanoVDB Level-Set Simulation

## Background and Motivation

The `Benchmark` singleton drives a GPU-accelerated level-set propagation pipeline
alongside OpenVDB.  The CUDA path (in `Benchmark.cu`) is the primary implementation
and has been validated against the OpenVDB reference.  The goal of the CPU port is to
produce a second execution path for each simulation module that runs on the host using
SIMD-vectorized C++, shares the same persistent state allocations as the CUDA path, and
can be validated side-by-side against it — without restructuring the existing host
control flow.

This document records the agreed migration plan, the rationale behind key design
choices, and the anticipated order of work.

---

## Phase 0 (Complete): UnifiedBuffer for Persistent State

The first prerequisite for a shared CPU/GPU execution model is that the persistent
simulation buffers are accessible from both host and device without explicit transfers.

**What was done:**

- `BufferT` was changed from `nanovdb::cuda::DeviceBuffer` to
  `nanovdb::cuda::UnifiedBuffer` (`cudaMallocManaged`) throughout `Benchmark.h`.
- `mHandle` (IndexGrid), `mPhi` (level-set sidecar), `mSpeed` (speed sidecar), and
  `mVBMHandle` (VoxelBlockManager) now all live in unified memory.
- Host-pinned temporaries inside `BenchmarkIO.cpp` (used for `cudaMemcpy` staging)
  and GPU-only transient allocations in `Benchmark.cu` (e.g., `pruneMaskBuffer`)
  retain `DeviceBuffer` — they are implementation details that never need to be seen
  from the other side.
- `DilateGrid::getHandle<BufferT>()` and `PruneGrid::getHandle<BufferT>()` are
  templated, so the dilated/pruned grid handles inherit the unified-memory backing
  automatically.
- `buildVoxelBlockManager<BlockWidthLog2, BufferT>()` similarly propagates `BufferT`.

**Why UnifiedBuffer and not a host copy:**
The eventual CPU kernels need to iterate over `mPhi` at full speed.  Maintaining a
synchronized host copy would double the memory budget and require explicit invalidation
on every topology change.  Unified memory lets the OS/driver handle migration
transparently; an explicit `prefetchAsync` call before each CPU kernel pass can recover
most of the performance where needed.

---

## Phase 1 (Complete): ExecutionPolicy Dispatch + CPU VBM Initialization

The second prerequisite is a dispatching mechanism that lets call sites choose the
execution path at runtime, making the dual-path visible to collaborators reading the code.

**Dispatch mechanism — `ExecutionPolicy` enum + `mPlatform`/`onCPU()`:**

```cpp
enum class ExecutionPolicy { CPU, CUDA };
```

Simulation methods that have both CPU and CUDA implementations are declared as primary
function templates parameterized on `ExecutionPolicy`:

```cpp
template<ExecutionPolicy Policy>
static void initializeVoxelBlockManager(GridHandleT&, VBMHandleT&, bool verbose = false);
```

The CUDA explicit specialization is defined in `Benchmark.cu`; the CPU explicit
specialization is defined in the new `Benchmark.cpp`.  The singleton carries a runtime
selector and a predicate:

```cpp
ExecutionPolicy mPlatform{ExecutionPolicy::CUDA};  // default: GPU path
bool onCPU() const { return mPlatform == ExecutionPolicy::CPU; }
```

**Call sites use explicit `if`/`else` dispatch:**

```cpp
if (Benchmark::getInstance().onCPU())
    Benchmark::initializeVoxelBlockManager<ExecutionPolicy::CPU>(handle, vbmHandle);
else
    Benchmark::initializeVoxelBlockManager<ExecutionPolicy::CUDA>(handle, vbmHandle);
```

This is intentionally verbose: every call site shows collaborators that a CPU
alternative exists and exactly how to select it.  The three call sites are:
- `main.cpp`: initial VBM build after grid setup
- `dilateActiveValues` (Benchmark.cu): VBM rebuild after narrow-band dilation
- `pruneNarrowBand` (Benchmark.cu): VBM rebuild after narrow-band pruning

The active path is selected at startup via the `-cpu` command-line flag, which sets
`getInstance().mPlatform = ExecutionPolicy::CPU`.

**Implementation — CPU specialization (`Benchmark.cpp`):**

Calls `nanovdb::tools::buildVoxelBlockManager<BlockWidthLog2, BufferT>(grid)` using
the host-accessible grid pointer from `gridHandle.grid<BuildT>()`.  Because the VBM
lives in unified memory, the result is immediately visible from both host and device.

**Validation — 16 frames, bit-identical output:**

Both paths were run for 16 frames (`-dt 0.125 -tmax 2`) on `taperLER.vdb` and the
output VDB files compared:

- Active voxel count, bounding box, min/max values, and compressed file size are
  identical (`file_mem_bytes: 346,233,360` for both).
- The CPU path is ~1.5× slower on the VBM rebuild steps (~20 ms vs ~13 ms), with an
  additional ~10–13 ms overhead on the subsequent CUDA kernels due to UnifiedBuffer
  page migration (grid pages pulled to CPU during VBM build, then migrated back to GPU
  for the next kernel launch).  This migration cost disappears once those kernels are
  also ported to CPU.
- Propagation kernel timing is identical between paths (~14 ms), confirming that
  operations not touching the VBM have zero overhead.

**Deferred factoring note:**
The per-leaf accumulation logic (computing `firstBlock`/`lastBlock`, backward-filling
`firstLeafID`, atomic OR into `jumpMap`) is currently duplicated between the CPU
build lambda and the GPU kernel functor.  Factoring this into a
`VoxelBlockManager<BlockWidthLog2>::accumulateLeafContribution(...)` static `__hostdev__`
member is the right long-term home, but is deferred until
`VoxelBlockManager<BlockWidth>` itself becomes `__hostdev__` accessible.

---

## Phase 2: SIMD-Vectorized Stencil Kernel (Normalization and Propagation)

The WENO5 normalization and propagation kernels are the computational heart of the
simulation and the primary beneficiaries of SIMD.  The porting strategy splits the
stencil into two functions with different vectorization profiles.

### 2a. Scalar part: `resolveWenoLeafPtrs`

```
resolveWenoLeafPtrs(grid, leaf, voxelOffset) -> WenoLeafPtrs<BuildT>
```

Performs exactly 3 `probeLeaf` calls (one per axis) and returns a struct holding
pointers to the up-to-three neighboring leaf nodes along each axis.

This function is **intentionally scalar**.  `probeLeaf` is a pointer-chasing tree
traversal with data-dependent branches; it does not vectorize and should not be forced
to.  Keeping it scalar also limits the VBM decode overhead to one decode per WENO5
stencil gather rather than per-voxel.

### 2b. SIMD part: `computeWenoStencil`

```
computeWenoStencil(leaf, voxelOffset, leafPtrs, data[19]) -> void
```

Fills the 19-element stencil index array from the leaf-pointer struct and the voxel
offset.  This is the auto-vectorization target.

**Why it vectorizes cleanly:**
- All 19 index lookups reduce to bounded-range arithmetic on the voxel offset plus
  a pointer dereference into one of at most 7 distinct leaf data arrays — no tree
  traversal inside the loop.
- The output is a plain `uint64_t[19]` (or `float[19]` after a gather), so the
  compiler sees independent stride-1 stores.

**SIMD loop structure (SIMDw = 16):**

```
decodeInverseMaps -> leafIndex[BlockWidth], voxelOffset[BlockWidth]   (768 bytes, L1-resident)

for each SIMD batch of SIMDw voxels:
    // Scalar: deduplicated probeLeaf calls
    for each axis a in {x, y, z}:
        roundedLo  = round_to_leaf_origin(coord - 3*e_a)
        roundedHi  = round_to_leaf_origin(coord + 3*e_a)
        if roundedLo != roundedCenter: ptrLo  = probeLeaf(roundedLo)   // at most 1 call
        if roundedHi != roundedCenter: ptrHi  = probeLeaf(roundedHi)   // at most 1 call
    // <= 6 probeLeaf calls per SIMDw=16 batch

    // SIMD: fill data[19][SIMDw] in a vectorized loop over i=0..SIMDw-1
    computeWenoStencil(leaf[i], voxelOffset[i], leafPtrs[i], data[][i])
```

**Auto-vectorization pragmas (clang preferred):**

```cpp
[[clang::always_inline]]
void computeWenoStencil(...)
{
    #pragma clang loop vectorize(enable) vectorize_width(16)
    for (int i = 0; i < SIMDw; ++i) { ... }
}
```

Key findings from the `simd_test/` investigation:

- `std::array<float*, 19>` output parameter style vectorizes correctly; a POD struct
  output does **not** (the compiler vectorizes the wrong dimension).
- `[[clang::always_inline]]` is necessary — inlining fragility is the main reason
  auto-vectorization silently regresses.
- `__restrict__` on output pointers eliminates aliasing uncertainty.
- A `std::experimental::simd` backend (C++23 / MSVC 19.38+) is available as a
  second implementation path via `simd_test/Simd.h` with a compile-time
  `SIMD_USE_STD_EXPERIMENTAL` switch.  This backend is portable but less tunable
  than the pragma approach; it is kept as a fallback and a cross-validation reference.

**SoA output layout at the call site:**

```cpp
float data[19][SIMDw];   // Structure of Arrays: data[stencil_point][lane]
```

After `computeWenoStencil`, each `data[k][:]` holds one stencil-point value for
all SIMDw voxels in the batch.  The WENO5 computation then operates vertically
(across lanes) with full SIMD width.

### 2c. CPU kernel wrappers

```cpp
static void normalizeLevelSet(CpuTag, GridHandleT&, VBMHandleT&, BufferT& phi,
                              ValueType background, int normCount,
                              ValueType voxelSize, bool verbose = false);

static void propagateLevelSet(CpuTag, GridHandleT&, VBMHandleT&, BufferT& phi,
                              BufferT& speed, ValueType background, ValueType dt,
                              ValueType voxelSize, bool verbose = false);
```

These iterate over VBM blocks with `std::for_each` (or TBB `parallel_for`), decode
inverse maps into stack-resident arrays (768 bytes, L1-resident), and call the SIMD
stencil gather + WENO5 compute loop described above.  TVD-RK2 is preserved: the
two-half-step structure maps directly onto two sequential passes over the VBM.

**Deferred design decision — RK2 `tempBuffer` allocation:**
The CUDA versions of `normalizeLevelSet` and `propagateLevelSet` allocate a
`BufferT tempBuffer` (currently `UnifiedBuffer`) for the RK2 intermediate.  This
buffer is used exclusively by GPU kernels and could in principle be `DeviceBuffer`.
It is kept as `UnifiedBuffer` for now so that the CPU SIMD prototype can accept the
same buffer via `data()` without an adaptation layer.

The correct long-term architecture is: CUDA path allocates `DeviceBuffer`, CPU path
allocates its own storage (stack, `std::vector`, or `UnifiedBuffer`).  Reaching that
requires templating `initializeGPUSidecarAndBackgroundValue` on buffer type:

```cpp
template<typename BufferT1>
static void initializeSidecarAndBackgroundValue(GridHandleT&, BufferT1&,
                                                ValueType background, bool verbose);
```

This refactor is deferred to Phase 2c, when the CPU kernel is being written and the
CPU allocation pattern is known.  The `TODO(cpu-port)` comments in `Benchmark.cu`
mark both call sites.

---

## Phase 3: Topological Operations (Dilation and Pruning)

Dilation and pruning modify the IndexGrid topology and are substantially more
complex than the stencil kernels.  CPU implementations are planned but deprioritized
relative to the stencil work.

**Dilation (`dilateActiveValues`):**

The GPU path calls `nanovdb::tools::cuda::DilateGrid` and then reinjects sidecar
values into new voxels via `DilateNarrowBandFunctor`.  The NanoVDB library has a
host-side dilation path (`nanovdb::tools::dilateActiveValues`) that can be used
similarly.  The sign-extrapolation sidecar injection loop is straightforwardly
parallelizable with TBB.

**Pruning (`pruneNarrowBand`):**

The GPU path builds a per-leaf prune mask in a first kernel pass, then calls
`nanovdb::tools::cuda::PruneGrid`.  The host-side equivalent involves iterating over
leaves to build the same mask, then calling the host-side prune path.

**Why last:**

Topology changes require rebuilding the VBM (Phase 1) and are gated on having a
validated stencil kernel (Phase 2) to produce a correct phi after each
dilation/normalization/pruning cycle.  They also interact with the
`DilateGrid`/`PruneGrid` API in ways that may require upstream NanoVDB changes (e.g.,
returning a `GridHandle<UnifiedBuffer>` from host-side topo ops, analogously to the
existing `getHandle<BufferT>()` templating on the CUDA paths).

---

## Validation Strategy

At each phase, CPU results are validated against the CUDA baseline using the existing
`compareOpenVDBDataToNanoVDBSidecar` infrastructure:

1. Run the CUDA path for N frames; capture `mPhi`.
2. Reset to the same initial condition; run the CPU path for N frames; capture `mPhi`.
3. Assert per-voxel max difference is within a floating-point tolerance.

Because both paths share the same `UnifiedBuffer`-backed `mPhi`, switching between
them for a single frame (to cross-check a specific step) requires only calling the
alternate-tag overload — no data movement.

---

## Summary Table

| Phase | Milestone                                  | Status      |
|-------|--------------------------------------------|-------------|
| 0     | UnifiedBuffer for persistent state         | Complete    |
| 1     | ExecutionPolicy dispatch + CPU VBM init    | Complete    |
| 2a    | `resolveWenoLeafPtrs` (scalar)             | Planned     |
| 2b    | `computeWenoStencil` SIMD + gather loop    | Planned     |
| 2c    | CPU normalization + propagation wrappers   | Planned     |
| 3     | CPU dilation + pruning (topo ops)          | Deferred    |
