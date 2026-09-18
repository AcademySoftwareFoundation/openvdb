# Provenance of the imported reference benchmark

Everything under this `Benchmark/` directory is a **near-verbatim copy** of the
`Benchmark/` tree from an internal NVIDIA benchmark repository.  It is checked in as a
reference baseline for the native NanoVDB renormalization work described in
`../RENORMALIZATION_DESIGN.md`.  Nothing here is built by the NanoVDB CMake
system; see "Build status" below.  The only edits to the imported sources
are the removal of source-project names from two comments in `src/main.cpp`;
no code was changed.

## Source

| | |
|---|---|
| Branch | `nanovdb-cpu-port` |
| Commit | `67ce36080475e3703b5116ca67e92d799662f4a4` (`67ce360`) |
| Date | 2026-06-16 |
| Subject | `Benchmark: route CPU prune mask through BufferT (drop hardcoded UnifiedBuffer)` |

`nanovdb-cpu-port` was chosen over the other candidate branches because it is the
most recent and structurally the most advanced: it carries the dual host/CUDA
`ExecutionPolicy` dispatch, the `BenchmarkIO.cpp` split, an out-of-bounds fix in
`DilateNarrowBandFunctor`, and the `BufferT`-templated prune mask.

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
self-contained application with its own `main.cpp`, its own `Makefile`, a hard
dependency on OpenVDB, and a local `include/Stencils.h` that deliberately
shadows `nanovdb/math/Stencils.h` (see the design document, §3).  Registering it
via `nanovdb_example()` would require flattening `include/`+`src/` into the
example root — the CMake glob is non-recursive — and resolving the shadowed
header.  Both are tracked as work items rather than done silently on import, so
that `upstream/master` continues to configure and build unchanged.

To build it standalone, set `NANOVDB_ROOT` and `OPENVDB_ROOT` and run `make` in
this directory, exactly as in the source repository.
