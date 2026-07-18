# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

OpenVDB uses CMake (minimum 3.24) and requires out-of-source builds.

**Minimal core build:**
```bash
mkdir build && cd build
cmake .. -DOPENVDB_BUILD_UNITTESTS=ON
make -j$(nproc)
```

**Using the CI build script (recommended for full builds):**
```bash
./ci/build.sh --build-type=Release \
  --components="core,test" \
  --cargs="-DOPENVDB_ABI_VERSION_NUMBER=13"
```

Component flags for `--components`: `core`, `python`, `bin`, `view`, `render`, `test`, `hou`, `axcore`, `nano`, `nanotest`

**Key CMake options:**
| Option | Default | Description |
|--------|---------|-------------|
| `OPENVDB_BUILD_CORE` | ON | Core library |
| `OPENVDB_BUILD_UNITTESTS` | OFF | Unit tests |
| `OPENVDB_BUILD_NANOVDB` | OFF | NanoVDB |
| `OPENVDB_BUILD_AX` | OFF | OpenVDB AX |
| `OPENVDB_ABI_VERSION_NUMBER` | 13 | ABI version (6–13) |
| `OPENVDB_CXX_STRICT` | OFF | Strict warnings |
| `NANOVDB_USE_CUDA` | OFF | CUDA support for NanoVDB |

## Running Tests

```bash
cd build
ctest -V                          # all tests
ctest -V -R TestGrid              # single test by name
```

To build only specific unit tests (avoids full rebuild):
```bash
cmake .. -DOPENVDB_TESTS="Grid;Tree;LeafNode"
```

Tests use Google Test (minimum 1.10). Test sources live in:
- `openvdb/openvdb/unittest/` — core library tests (`TestFoo.cc` pattern)
- `nanovdb/nanovdb/unittest/` — NanoVDB tests

## Code Architecture

### Repository Layout

```
openvdb/openvdb/        Core OpenVDB library
  tree/                 Tree node hierarchy (RootNode, InternalNode, LeafNode)
  tools/                Algorithm implementations (level sets, CSG, smoothing, etc.)
  math/                 Math primitives (Vec, Mat, Quat, Transform, BBox)
  io/                   VDB file format I/O
  points/               Point data grids
  python/               Python bindings (nanobind)
  unittest/             Unit tests

nanovdb/nanovdb/        NanoVDB — compact, GPU-friendly VDB subset
  tools/                CPU algorithms
  tools/cuda/           CUDA kernels
  examples/             Standalone example programs

openvdb_ax/openvdb_ax/  OpenVDB AX — JIT expression language for VDB operations
  ast/                  Abstract syntax tree
  codegen/              LLVM code generation
  compiler/             Compilation pipeline

openvdb_cmd/            Command-line tools (vdb_print, vdb_lod, vdb_tool, vdb_view, vdb_render)
openvdb_houdini/        Houdini plugin
openvdb_maya/           Maya plugin
cmake/                  CMake find-modules and configuration
ci/                     CI build/install scripts
```

### Core Data Model

OpenVDB uses a **B+tree-like hierarchical sparse data structure**:
- `Grid<TreeType>` — top-level container with transform and metadata
- `Tree` — composed of `RootNode → InternalNode(s) → LeafNode`
- Leaf nodes are 8×8×8 voxel blocks; internal nodes are 16³ and 32³ by default
- `ValueAccessor` caches tree traversal paths for repeated access patterns
- `GridBase` / `TypedGrid` provide the runtime-polymorphic/compile-time-typed split

### NanoVDB vs OpenVDB

NanoVDB is a read-optimized, single-allocation, GPU-portable subset of OpenVDB. It cannot be modified after construction. The `nanovdb/tools/CreateNanoGrid.h` and adjacent files handle conversion from OpenVDB grids to NanoVDB grids.

### Topology Operators — CPU Port (this branch)

The branch `vbm-cpu-port` back-ports the five CUDA-only NanoVDB topology operators (`DilateGrid`, `MergeGrids`, `PruneGrid`, `RefineGrid`, `CoarsenGrid`) from `nanovdb/tools/cuda/*.cuh` (namespace `nanovdb::tools::cuda`) to host-only headers at `nanovdb/tools/*.h` (namespace `nanovdb::tools`). The design plan is in `nanovdb/nanovdb/tools/TopologyCpuPortPlan.md`.

**Key files:**
- `nanovdb/tools/TopologyBuilder.h` — shared base class (in-progress port; still has CUDA includes during transition)
- `nanovdb/tools/MergeGrids.h` — first operator being ported
- `nanovdb/util/Morphology.h` — host-side morphology functors (parallel to `util/cuda/Morphology.cuh`)
- `nanovdb/util/PrefixSum.h` — host `inclusiveScan` used in place of CUB scans
- `nanovdb/tools/VoxelBlockManager.h` — VBM acceleration structure for active-voxel SIMT/SIMD access

**CUDA → host transition pattern:**
- Device-only scratch buffers use `ScratchBufferT = UnifiedBuffer` (transitional) → will become `HostBuffer`
- `lambdaKernel` launches → `util::forEach` calls
- CUB inclusive scans → `util::inclusiveScan`
- Morphology functors move from `util/cuda/Morphology.cuh` (`__device__`) to `util/Morphology.h` (host `inline`)
- CUDA stays on (`NANOVDB_USE_CUDA=ON`) throughout; host-only completion is signaled by renaming `.cu` → `.cpp` in examples

**Build config for this branch** (build dir: `nanovdb/nanovdb/build/`):
```bash
cd nanovdb/nanovdb/build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNANOVDB_BUILD_EXAMPLES=ON \
      -DNANOVDB_USE_CUDA=ON \
      -DNANOVDB_USE_OPENVDB=ON \
      -DNANOVDB_USE_TBB=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_PREFIX_PATH=~/local \
      -DOpenVDB_ROOT=~/local \
      ..
make -j$(nproc)
```

OpenVDB is installed at `~/local` (not `/usr/local`). The project's `cmake/FindOpenVDB.cmake` requires `-DOpenVDB_ROOT=~/local` explicitly — `CMAKE_PREFIX_PATH` alone is not enough because the finder is invoked from the repo's `cmake/` directory, not the installed `OpenVDB/` directory, so it doesn't auto-detect the prefix. Always clear `CMakeCache.txt` before re-running cmake if you change this path, as NOTFOUND results are cached.

Always use a specific `-DCMAKE_CUDA_ARCHITECTURES=` value (here `120` for Blackwell). The default `=75` (Turing) causes silent kernel failures on non-sm_75 hardware, surfacing as misleading `cudaErrorInvalidDevice`.

**Validation:**
```bash
./ex_merge_nanovdb_cpu  /path/to/dragon.vdb /path/to/torus.vdb 3
./ex_dilate_nanovdb_cpu /path/to/dragon.vdb 3
./ex_refine_nanovdb_cpu /path/to/dragon.vdb 3
./ex_coarsen_nanovdb_cpu /path/to/dragon.vdb 3
```
Each should print `Result of <Op> check out CORRECT against reference`.

**Only `OnIndexGrid` buildtypes are supported** (`static_assert(BuildTraits<BuildT>::is_onindex)`). Grids with tile values at any level cause the operator to throw.

### OpenVDB AX

AX compiles a domain-specific expression language to LLVM IR for execution over OpenVDB volumes and point grids. The pipeline is: source string → AST (`ast/`) → typed analysis → LLVM codegen (`codegen/`) → JIT execution via `compiler/`.

## C++ Standard and ABI

- Requires C++17 minimum
- ABI version is set at compile time via `OPENVDB_ABI_VERSION_NUMBER`; the current version is 13
- Headers are in `openvdb/openvdb/` and installed to `include/openvdb/`

## Dependencies

Core: Boost ≥ 1.82, TBB ≥ 2020.3, Blosc ≥ 1.17, OpenEXR/Imath ≥ 3.2, zlib ≥ 1.2.7
Tests: GTest ≥ 1.10
Python bindings: Python ≥ 3.11, nanobind ≥ 2.5.0
NanoVDB GPU: CUDA toolkit
AX: LLVM

On Linux, ASWF Docker containers (used by CI) bundle most dependencies. See `ci/install_macos.sh` and `ci/install_windows.ps1` for platform-specific setup.

## Coding Standards

Follow the style guide at https://www.openvdb.org/documentation/doxygen/codingStyle.html. Contributions require a Developer Certificate of Origin sign-off (`git commit -s`) and a CLA on file — see CONTRIBUTING.md.
