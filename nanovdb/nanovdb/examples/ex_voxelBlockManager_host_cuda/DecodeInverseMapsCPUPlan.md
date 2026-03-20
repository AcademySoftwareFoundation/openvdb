# CPU `decodeInverseMaps` Implementation Plan

This document captures the full design for porting `VoxelBlockManager<Log2BlockWidth>::decodeInverseMaps`
to the host. It is the companion to `VoxelBlockManagerContext.md` and serves as a starting point
for implementing the function in `nanovdb/tools/VoxelBlockManager.h`.

---

## 1. Threading Model

The GPU version uses a thread block (up to 512 threads) to decode one voxel block cooperatively.
The CPU version inverts the axes:

- **Outer**: OpenMP thread parallelism over voxel blocks (one block per thread).
- **Inner**: SIMD within `decodeInverseMaps` for the single block assigned to the calling thread.

`decodeInverseMaps` itself is single-threaded. It does not use OpenMP internally. The caller
(the `buildVoxelBlockManager` CPU path) is responsible for distributing blocks across threads.

---

## 2. Outputs

Like the GPU version, the function fills two arrays for a single voxel block:

```cpp
uint32_t leafIndex[BlockWidth];   // smem_leafIndex on GPU
uint16_t voxelOffset[BlockWidth]; // smem_voxelOffset on GPU
```

Sentinel values for positions beyond the last active voxel:
```cpp
static constexpr uint32_t UnusedLeafIndex  = 0xffffffff;
static constexpr uint16_t UnusedVoxelOffset = 0xffff;
```

---

## 3. Leaf Iteration (unchanged from GPU)

Iterate leaf IDs from `firstLeafID[blockID]` through `firstLeafID[blockID] + nExtraLeaves`,
where `nExtraLeaves` is `popcount(jumpMap[blockID * JumpMapLength .. +JumpMapLength])`.

For each leaf, compute:
- `leafFirstOffset = leaf.data()->firstOffset()`
- `leafValueCount  = leaf.data()->valueCount()`  ← number of active voxels in this leaf
- `pStart = max(0, leafFirstOffset - blockFirstOffset)` — first slot in the block's output arrays
- `pEnd   = min(BlockWidth, leafFirstOffset + leafValueCount - blockFirstOffset)` — one past last slot
- `jStart = (leafFirstOffset < blockFirstOffset) ? blockFirstOffset - leafFirstOffset : 0`
  — index of first active voxel in the leaf that falls inside this block

Then:
- `leafIndex[pStart..pEnd)  = leafID`   (range fill, no scatter)
- `voxelOffset[pStart..pEnd) = leafLocalOffsets[jStart..jStart+(pEnd-pStart))`   (contiguous copy)

where `leafLocalOffsets[j]` = local offset (0..511) of the j-th active voxel in this leaf.

---

## 4. Producing `leafLocalOffsets` via Stream Compaction

`leafLocalOffsets` is the stream compaction of {0, 1, …, 511} under `valueMask`. It maps
dense index j → local offset of the j-th active voxel.

The GPU scatter `smem_voxelOffset[index - blockFirstOffset] = localOffset` is *equivalent* to
this compaction: as `localOffset` increases 0..511, the scatter destination
`index - blockFirstOffset` is non-decreasing. So the scatter is really: pack all active local
offsets in order → copy a contiguous slice.

### 4a. SIMD Word Layout: 16 × 32-bit Words

Treat `valueMask` as **16 `uint32_t` words** (not 8 `uint64_t`). Rationale:

- 16 words fill a full AVX-512 register (16 × 32-bit lanes) or two AVX2 registers (8 × 32-bit lanes).
- The multiply step in the vertical sweep (see §5) uses `VPMULLD` (AVX2, SSE4.1) for 32-bit
  multiply, which is widely available. The 64-bit equivalent `VPMULLQ` requires AVX-512DQ.
- Software popcount on 32-bit words uses AND/shift/add/multiply — all available as 32-bit SIMD
  ops in AVX2. A `#pragma omp simd` loop over 16 words auto-vectorizes without `VPOPCNTQ`.

### 4b. Vertical SIMD Sweep for Base Offsets

Compute `baseOffset[w]` = number of active voxels in words 0..w-1:

```cpp
uint32_t words[16];
// ... load valueMask as 16 x uint32_t ...
uint32_t baseOffset[16];
baseOffset[0] = 0;
for (int w = 1; w < 16; w++)
    baseOffset[w] = baseOffset[w-1] + popcount32(words[w-1]);
```

This prefix sum loop is short and can be left scalar (dependency chain); the 16 popcount calls
above it are SIMD-vectorizable.

### 4c. Vertical Sweep: Producing `leafLocalOffsets`

Outer loop over bit position k = 0..31; inner SIMD loop over all 16 words simultaneously.

For each (k, w):
- `bit = (words[w] >> k) & 1`  — is this voxel active?
- If active: `localOffset = w * 32 + k` and the dense index is
  `j = popcount32(words[w] & ((1u << k) - 1)) + baseOffset[w]`

In practice, rather than computing j by popcount on every step, maintain a running count `count[w]`
per word and increment it each time a 1-bit is encountered:

```cpp
uint32_t count[16] = { /* baseOffset[w] */ ... };

// Preallocate output (512 elements max per leaf)
uint16_t leafLocalOffsets[512];

for (int k = 0; k < 32; k++) {
    #pragma omp simd
    for (int w = 0; w < 16; w++) {
        if ((words[w] >> k) & 1u) {
            leafLocalOffsets[count[w]++] = (uint16_t)(w * 32 + k);
        }
    }
}
```

**Note:** The `#pragma omp simd` with a data-dependent conditional store does vectorize on modern
compilers (the compiler emits a masked scatter or conditional store). If the compiler balks, the
fallback is to separate the popcount prefix-sum from the scatter, using the parallel prefix
approach described in §4d.

### 4d. Parallel Prefix Compaction via `shfl_down` (Alternative / Deeper SIMD)

This approach avoids data-dependent stores entirely and is the approach validated in
`simd_test/shfl_down_test.cpp`.

**Key insight**: Define `shifts[i]` = number of zeros before position i in the bitmask =
`i - (dense_index_of_voxel_i)`. The compaction moves each active voxel at position i down by
`shifts[i]`. Decompose `shifts[i]` in binary: apply log2(BlockWidth) passes. Pass k moves
elements down by 2^k *if* bit k of `shifts[i]` is set:

```cpp
// Templated fixed-offset conditional blend
template <int Shift, int Width>
void shfl_down(uint16_t* data, const bool* move) {
    #pragma omp simd
    for (int i = 0; i < Width - Shift; i++)
        if (move[i]) data[i] = data[i + Shift];
}
```

Each pass is a **fixed-offset conditional copy** — the write index is data-independent.
Compiles to clean masked blend operations:
- **AVX-512**: `vmovdqu32` with a mask register (single instruction per pass)
- **AVX2**: double-negate blend pattern (no register-level shuffle needed)

The `move[i]` predicate for pass k is: `(shifts[i] & (1 << k)) != 0`, which itself depends on
the bitmask but can be computed upfront via popcount before the blend passes.

**Practical recommendation**: Start with the simpler vertical sweep (§4c). Fall back to `shfl_down`
if the compiler fails to vectorize the conditional store or if profiling shows it is the bottleneck.

---

## 5. Bypassing `mPrefixSum`

The leaf stores a packed 9-bit `mPrefixSum` for random access. Do **not** use it here.

For bulk sequential access over all 512 voxels, recomputing per-word popcounts from scratch via
SIMD is cheaper than unpacking the 9-bit packed fields (which requires masked shifts and is
awkward to vectorize). The vertical sweep (§4c) naturally computes exactly what is needed.

---

## 6. `leafIndex` Fill (Trivial)

```cpp
std::fill(leafIndex + pStart, leafIndex + pEnd, (uint32_t)leafID);
```

No scatter. `leafID` is constant per leaf.

---

## 7. `voxelOffset` Fill (Contiguous Copy)

```cpp
std::copy(leafLocalOffsets + jStart,
          leafLocalOffsets + jStart + (pEnd - pStart),
          voxelOffset + pStart);
```

`leafLocalOffsets` is produced once per leaf (§4) and then sliced into the output array.

---

## 8. Initialization

Before iterating over leaves, initialize sentinel values for the whole block:

```cpp
std::fill(leafIndex,   leafIndex   + BlockWidth, UnusedLeafIndex);
std::fill(voxelOffset, voxelOffset + BlockWidth, UnusedVoxelOffset);
```

---

## 9. Function Signature (Proposed)

The CPU version mirrors the GPU signature but with plain pointers (no `__device__`, no shared
memory, no sync):

```cpp
template <int BlockWidth>
template <class BuildT>
void VoxelBlockManager<Log2BlockWidth>::decodeInverseMaps(
    const NanoGrid<BuildT>* grid,
    uint32_t                blockID,
    const uint32_t*         firstLeafID,
    const uint64_t*         jumpMap,
    uint64_t                blockFirstOffset,
    uint32_t*               leafIndex,    // output, length BlockWidth
    uint16_t*               voxelOffset)  // output, length BlockWidth
```

Or as a free function in an `cpu` sub-namespace alongside `buildVoxelBlockManager` in
`VoxelBlockManager.h`.

---

## 10. Future Factoring

Once `VoxelBlockManager<Log2BlockWidth>` is annotated `__hostdev__` on all its members, the
per-leaf logic shared between the CPU and GPU builds can be factored into a `__hostdev__ static`
member (e.g., `accumulateLeafContribution(...)`) — see `project_vbm_factoring.md` in the memory
directory. The `decodeInverseMaps` CPU/GPU split is a separate concern (SIMD vs warp cooperation)
and will likely remain two implementations even after factoring.

---

## 11. Validated Findings from `simd_test/shfl_down_test.cpp`

- `#pragma omp simd` with a compile-time constant offset (`shfl_down<N>`) compiles to:
  - AVX-512: `vmovdqu32` with mask (single masked move per pass)
  - AVX2: double-negate blend pattern (compiler generates `vpblendvb` or equivalent)
- No architecture-specific intrinsics needed.
- Software `popcount32` (Hamming weight via AND/shift/add/multiply) auto-vectorizes to
  `VPMULLD` on both AVX2 and AVX-512 with `#pragma omp simd`.
- `VPOPCNTQ` (AVX-512VPOPCNTDQ) is **not** required.
