# Provenance of the imported reference benchmark

Everything under this `Benchmark/` directory is a **near-verbatim copy** of the
`Benchmark/` tree from an internal NVIDIA benchmark repository.  It is checked in as a
reference baseline for the native NanoVDB renormalization work described in
`../RENORMALIZATION_DESIGN.md`.  Nothing here is built by the NanoVDB CMake
system; see "Build status" below.

Edits to the imported sources, kept to a minimum and listed here so the
remainder can be trusted as verbatim:

* the removal of source-project names from two comments in `src/main.cpp`;
* `include/Stencils.h` **deleted**, and its three include sites repointed at
  `<nanovdb/math/Stencils.h>`.  That file was a copy of the NanoVDB header
  carrying two extra static `WenoStencil` overloads, relied upon to win an
  include-guard race against upstream.  The overloads now live upstream, so the
  copy and the race are gone.  No other code was changed.

## Source

| | |
|---|---|
| Branch | `nanovdb-cpu-port` |
| Commit | `67ce36080475e3703b5116ca67e92d799662f4a4` (`67ce360`) |
| Date | 2026-06-16 |
| Subject | `Benchmark: route CPU prune mask through BufferT (drop hardcoded UnifiedBuffer)` |

## Why this branch, for CUDA specifically

Only the CUDA implementation is in scope.  `nanovdb-cpu-port` carries host-port
work as well, so the branch choice was re-verified against the other candidate,
`benchmark-add-lateral-ratio @ f8ab0ca`, restricted to `Benchmark.cu`:

**The numerics are identical.**  Diffing `f8ab0ca..67ce360` over `Benchmark.cu`
and filtering to WENO / Godunov / Euler-step lines yields changes in comments
only.  The renormalization kernel was not touched by the CPU port.

**The newer branch has CUDA-only improvements the older one lacks:**

* `21cd3a0` -- **fixes an out-of-bounds read in `DilateNarrowBandFunctor`**.  The
  old code dereferenced `newTree.getFirstNode<0>()[leafID]` and
  `newLeaf.origin()` *before* testing `leafID < leafCount`, so padding warps in
  the last block read unmapped memory.  A real CUDA bug, fixed only here.
* `f986965` -- updates to the `VoxelBlockManager` API of PR#2189, i.e. the API we
  would be building against.
* `9573926` -- Doxygen documentation of the CUDA kernels.
* Buffer-templated calls: `dilator.getHandle<BufferT>()`,
  `buildVoxelBlockManager<BlockWidthLog2, BufferT>(...)`.

**The CPU scaffolding it also carries is additive and easy to strip:**

* an `ExecutionPolicy` template parameter on every entry point
  (`dilateActiveValues<ExecutionPolicy::CUDA>` and friends);
* `src/Benchmark.cpp`, which is *entirely* `ExecutionPolicy::CPU`
  specializations;
* a `getInstance().onCPU()` branch **inside** the CUDA specialization of
  `dilateActiveValues` (`src/Benchmark.cu:145-148`), which lets a nominally-CUDA
  entry point call the host VBM builder.  Odd, and slated for removal.

**One CPU-motivated regression to be aware of.**  Commit `0b207e7` switched the
persistent state from `DeviceBuffer` to `UnifiedBuffer` *for the benefit of the
future CPU port*:

    f8ab0ca:  using BufferT = nanovdb::cuda::DeviceBuffer;
    67ce360:  using BufferT = nanovdb::cuda::UnifiedBuffer;

For a pure CUDA implementation that is a step backwards, and it is the same
issue as the TODO at `src/Benchmark.cu:441-448`.  `f8ab0ca` is the better
reference for *this one aspect*; see `../RENORMALIZATION_DESIGN.md` §7.

Net: taking `f8ab0ca` instead would have meant giving up a CUDA bug fix, the
current VBM API and the kernel documentation in order to avoid scaffolding that
deletes cleanly.  `nanovdb-cpu-port` stands as the right source.

## What matters here, and what to ignore

In scope (CUDA):

| File | Role |
|---|---|
| `src/Benchmark.cu` | **the CUDA kernels -- the subject of this effort** |
| `include/Benchmark.h` | type aliases, VBM constants, state declarations |
| `src/BenchmarkIO.cpp` | OpenVDB bridge + the comparison routines used for validation |
| `src/main.cpp` | driver; selects HJWENO5_BIAS + TVD_RK2 + normCount 3 |
| `include/LevelSetTrackerNew.h` | the OpenVDB reference implementation |

Out of scope, retained only so the snapshot stays verbatim and buildable:
`src/Benchmark.cpp` (all `ExecutionPolicy::CPU`), `CPU_PORT_PLAN.md`,
`include/LevelSetPropagate.h` and `include/VelocityExtension.h` (interface
propagation, not renormalization), `include/GridHelpers.cuh`.

## What is deliberately NOT here

**The per-stage OpenVDB/NanoVDB cross-validation call sites.**  On
`nanovdb-cpu-port` these were stripped by commits `19a213c` ("stripping away
openvdb code") and `6aebadf` ("removing all OpenVDB compute ops"), which are
ancestors of this snapshot.  The last revision that ran both stacks in tandem and
compared them after *every* pipeline stage is:

    origin/benchmark-add-lateral-ratio @ f8ab0ca728b95deda43362c858c823adb9c5e13d

In that revision `LevelSetTrackerNew<>::track()` interleaves
`tools::dilateActiveValues` / `this->normalize()` / `this->prune()` with their
`Benchmark::` counterparts and calls `compareOpenVDBDataToNanoVDBSidecar` between
stages.  Recovering those call sites is the starting point for the validation
work in `../RENORMALIZATION_DESIGN.md` §6.

**The `.vdb` models.**  `Benchmark/models/` holds `before_20M.vdb` (25 MB) and
`taperLER.vdb` (129 MB); 153 MB of binary assets do not belong in this
repository.  Fetch them from the source repository if you need to run the
imported benchmark as-is.

## Build status

This tree is **not** wired into `nanovdb/examples/CMakeLists.txt`.  It is a
self-contained application with its own `main.cpp`, its own `Makefile` and a
hard dependency on OpenVDB.  Registering it via `nanovdb_example()` would
require flattening `include/`+`src/` into the example root, since the CMake glob
is non-recursive; that is tracked as a work item rather than done silently, so
that upstream continues to configure and build unchanged.

`src/Benchmark.cu`, `src/BenchmarkIO.cpp` and `src/main.cpp` compile against
this repository's NanoVDB.  `src/Benchmark.cpp` does not: it needs
`nanovdb/tools/DilateGrid.h`, a host-port header that is not upstream.  That
predates this import and is out of scope -- only the CUDA path matters here.

To build it standalone, set `NANOVDB_ROOT` and `OPENVDB_ROOT` and run `make` in
this directory, exactly as in the source repository.
