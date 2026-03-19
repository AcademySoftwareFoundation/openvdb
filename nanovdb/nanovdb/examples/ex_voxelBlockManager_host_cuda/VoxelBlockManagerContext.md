# VoxelBlockManager: Context and Design Guide

This document captures the intent, semantics, and usage patterns of the
VoxelBlockManager (VBM) acceleration structure for NanoVDB OnIndex grids.
It is intended to give a complete mental model to anyone (human or AI assistant)
working on VBM development or writing SIMT kernels that consume it.

---

## What problem does the VBM solve?

A NanoVDB `ValueOnIndex` (OnIndex) grid assigns a unique sequential integer
index to each active voxel. These indices are dense: if the grid has N active
voxels, they are numbered 1..N (the index 0 is reserved). This sequential
layout enables efficient SIMT parallelism: a GPU kernel can launch one thread
per active voxel, with thread k processing voxel k.

The challenge is *decoding*: given a voxel's sequential index, how do you find
its 3D coordinates and leaf node? The index space is flat but the tree is
hierarchical. Scanning the tree for each voxel is too expensive.

The VBM solves this by precomputing two small metadata arrays that let any
thread, given only its sequential index, quickly determine which leaf node
contains that voxel and where within the leaf it lives. This decode is done
cooperatively at thread-block granularity (one thread block per "voxel block"),
using shared memory.

---

## Core concepts

### Voxel blocks

The VBM partitions the active voxel index space into contiguous spans of
`BlockWidth` voxels called *voxel blocks*. `BlockWidth` is a compile-time
power of two (typically 128). Block `b` covers sequential indices:

    [firstOffset + b * BlockWidth,  firstOffset + (b+1) * BlockWidth - 1]

A GPU kernel processes one voxel block per thread block, with `BlockWidth`
threads. Thread `t` in block `b` is responsible for voxel index
`firstOffset + b * BlockWidth + t`.

### firstOffset and lastOffset

`firstOffset` is the base of the VBM's index range. It must satisfy
`firstOffset == 1 (mod BlockWidth)` (i.e., it is "block-aligned"). For a
single-GPU build covering the full grid, `firstOffset = 1` and
`lastOffset = activeVoxelCount()`.

In a **multi-GPU** setting, each rank owns a contiguous slice of the active
voxel index space. Rank r uses `firstOffset_r` and `lastOffset_r` to build a
VBM covering only its slice, even though all ranks hold a copy of the full
grid topology. The VBM metadata is then sized and indexed relative to each
rank's own `firstOffset`.

### blockCount

`blockCount` is the **allocated capacity** of the VBM metadata buffers. It
must be >= `ceil((lastOffset - firstOffset + 1) / BlockWidth)` but may be
larger. This allows pre-allocating a larger handle and rebuilding in-place
for a range that grows over time, without reallocating.

---

## Metadata arrays

### firstLeafID  (uint32_t[blockCount])

`firstLeafID[b]` is the index of the **first leaf node** that overlaps voxel
block `b`. A leaf overlaps block `b` if any of its active voxels fall in
`[firstOffset + b*BlockWidth, firstOffset + (b+1)*BlockWidth - 1]`.

In a sequential NanoVDB grid, leaf nodes are laid out contiguously in memory
in ascending order of their first active voxel index. So `firstLeafID[b]` is
the smallest leaf index `i` such that leaf `i` has at least one active voxel
in block `b`.

### jumpMap  (uint64_t[blockCount * JumpMapLength])

`JumpMapLength = BlockWidth / 64`. The jumpMap for block `b` is a bitfield of
`BlockWidth` bits (stored as `JumpMapLength` uint64_t words) where bit `p` is
set if and only if a new leaf node **begins** at position `p` within block `b`
(i.e., some leaf's first active voxel has sequential index
`firstOffset + b*BlockWidth + p`, and `p > 0`).

Bit 0 is never set (the leaf starting exactly at the block boundary is
recorded in `firstLeafID`, not the jumpMap).

Together, `firstLeafID[b]` and the jumpMap for block `b` enumerate all leaf
nodes that overlap block `b`: start at `firstLeafID[b]`, then count the set
bits in the jumpMap to find how many additional leaves follow.

---

## Build API

### Handle: VoxelBlockManagerHandle<BufferT>

A pure data holder. Owns two buffers (`firstLeafID` and `jumpMap`) and stores
`blockCount`, `firstOffset`, `lastOffset`. No allocation or build logic is in
the handle itself. A default-constructed handle is the canonical "null/empty"
state (`blockCount == 0`).

Accessors:
- `blockCount()`, `firstOffset()`, `lastOffset()`
- `hostFirstLeafID()` / `hostJumpMap()` -- host-side pointers
- `deviceFirstLeafID()` / `deviceJumpMap()` -- device-side pointers (only when
  BufferT has a device dual, e.g. `cuda::DeviceBuffer` or a unified buffer)

### Two-overload pattern

Both the CPU (`nanovdb::tools::`) and CUDA (`nanovdb::tools::cuda::`) build
functions follow the same two-overload pattern, mirroring the NodeManager
convention in NanoVDB:

**Allocating overload** -- returns a new, fully-constructed handle:

    // CPU
    auto handle = nanovdb::tools::buildVoxelBlockManager<BlockWidthLog2>(grid);

    // CUDA
    auto handle = nanovdb::tools::cuda::buildVoxelBlockManager<BlockWidthLog2>(d_grid, stream);

Optional parameters (with sentinel 0 meaning "derive from grid"):
- `firstOffset` -- defaults to 1
- `lastOffset`  -- defaults to `activeVoxelCount()` (CPU) or read from device
                   via `DeviceGridTraits::getActiveVoxelCount()` (CUDA)
- `nBlocks`     -- defaults to `ceil((lastOffset - firstOffset + 1) / BlockWidth)`

If `lastOffset < firstOffset` (e.g., empty grid), a default-constructed null
handle is returned immediately with no allocation attempted.

**Rebuild-in-place overload** -- takes a pre-allocated handle by reference,
zeroes the jumpMap, and recomputes both arrays. No allocation:

    // CPU
    nanovdb::tools::buildVoxelBlockManager<BlockWidthLog2>(grid, handle);

    // CUDA
    nanovdb::tools::cuda::buildVoxelBlockManager<BlockWidthLog2>(d_grid, handle, stream);

A null handle (`blockCount == 0`) is silently ignored (no-op). This overload
is the right choice for benchmarking or for rebuilding after a
topology-preserving update without paying allocation cost.

The allocating overload delegates to the rebuild overload after allocating
buffers -- no logic duplication.

---

## CPU vs GPU implementation asymmetry

### GPU: launch at lower-node granularity

A NanoVDB level-1 internal node ("lower node") has up to 4096 leaf child
slots (16^3). The GPU kernel launches one CTA per lower node (subdivided
into `SlicesPerLowerNode = 8` slices for additional parallelism), so each
thread handles approximately one leaf child slot per iteration. Threads check
the lower node's `childMask` to skip empty slots.

This grouping is chosen because:
- It naturally sizes the CTA workload (4096 slots / 8 slices / 128 threads = 4 slots per thread)
- Threads in the same warp access leaves from the same lower node, improving
  memory access locality
- The grid is `<<<dim3(lowerCount, SlicesPerLowerNode), NumThreads>>>`

The cost is wasted threads for sparse lower nodes (few active leaf children
out of 4096 slots). This is an acceptable trade-off on the GPU.

### CPU: iterate leaves directly

On the CPU there is no benefit to the lower-node grouping. The build uses
`std::for_each(std::execution::par, firstLeaf, firstLeaf + leafCount, ...)`,
iterating directly over the flat contiguous leaf array. Each task processes
exactly one leaf -- no child mask checks, no wasted iterations.

Leaf index is computed by pointer arithmetic: `&leaf - firstLeaf`.

### Why firstLeafID writes are race-free

In a sequential NanoVDB grid, leaf offset ranges are non-overlapping and
ordered. A leaf that spans from block `a` to block `a+k` (max k=3 for
BlockWidth=128 and leaves with <=512 active voxels) "backward-fills"
`firstLeafID[a+1..a+k]`. No other leaf can start in a block before `a+k`
without its offset range overlapping leaf `a`'s range -- which is impossible.
Hence at most one leaf writes each `firstLeafID[b]` entry, so the writes
are non-atomic.

The jumpMap writes, by contrast, require atomic OR because multiple leaves
from different parts of the tree can start at positions within the same block.

---

## Kernel usage pattern (SIMT consumer)

The typical VBM-powered kernel:

    __global__ void myKernel(
        NanoGrid<ValueOnIndex>* grid,
        const uint32_t* firstLeafID,
        const uint64_t* jumpMap,
        uint64_t firstOffset, uint64_t lastOffset, uint32_t nBlocks)
    {
        __shared__ uint32_t smem_leafIndex[BlockWidth];
        __shared__ uint16_t smem_voxelOffset[BlockWidth];

        int blockID = blockIdx.x;
        uint64_t blockFirstOffset = firstOffset + (uint64_t)blockID * BlockWidth;

        // Cooperative decode: all threads in the block participate
        VoxelBlockManager<BlockWidth>::decodeInverseMaps(
            grid,
            firstLeafID[blockID],
            &jumpMap[blockID * VoxelBlockManager<BlockWidth>::JumpMapLength],
            blockFirstOffset,
            smem_leafIndex,
            smem_voxelOffset);
        // smem_leafIndex[t] and smem_voxelOffset[t] now hold the leaf index
        // and intra-leaf voxel offset for thread t's voxel.
        // Entries beyond lastOffset are filled with UnusedLeafIndex / UnusedVoxelOffset.

        int t = threadIdx.x;
        uint64_t globalIndex = blockFirstOffset + t;
        if (globalIndex > lastOffset) return;
        if (smem_leafIndex[t] == VoxelBlockManager<BlockWidth>::UnusedLeafIndex) return;

        // From here: access the voxel's 3D position, values, or stencil.
        // VoxelBlockManager<BlockWidth>::computeBoxStencil(...) uses the same
        // smem arrays to look up the 27 stencil neighbor indices.
    }

    // Launch: one block per VBM block
    myKernel<<<nBlocks, BlockWidth>>>(
        d_grid,
        vbmHandle.deviceFirstLeafID(),
        vbmHandle.deviceJumpMap(),
        vbmHandle.firstOffset(),
        vbmHandle.lastOffset(),
        (uint32_t)vbmHandle.blockCount());

Key points:
- `decodeInverseMaps` must be called by ALL threads in the block
  (it uses `__syncthreads` internally). Do not call from divergent threads.
- `computeBoxStencil` does NOT synchronize and may be called per-thread.
- Voxels in the last partial block beyond `lastOffset` get sentinel values
  (`UnusedLeafIndex = 0xffffffff`, `UnusedVoxelOffset = 0xffff`); always
  guard with a `globalIndex <= lastOffset` check.

---

## Files

- `nanovdb/tools/VoxelBlockManager.h`      -- Handle class, CPU build functions
- `nanovdb/tools/cuda/VoxelBlockManager.cuh` -- VoxelBlockManager device struct,
                                                CUDA build functions
- `nanovdb/examples/ex_voxelBlockManager_host_cuda/` -- end-to-end example,
  benchmarks CPU vs GPU build time, validates CPU/GPU metadata agreement,
  validates full inverse map against grid structure
