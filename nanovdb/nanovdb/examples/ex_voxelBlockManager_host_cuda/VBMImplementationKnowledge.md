# VBM Implementation Knowledge Base

This document captures non-obvious design decisions, rejected alternatives, invariants,
and performance phenomenology for the VoxelBlockManager (VBM) subsystem and its CPU
port of `decodeInverseMaps`.  It is written as dense, structured, agent-consumable
facts rather than narrative.  It complements `VoxelBlockManagerContext.md` (semantics
and API) and `DecodeInverseMapsCPUPlan.md` (algorithm design journal).

---

## 1. NanoVDB Leaf Mask Layout — Ambient Space

A `Mask<3>` stores 512 bits as 8 `uint64_t` words (`maskWords[x]`, x = 0..7).
The NanoVDB linear voxel index within a leaf is `i = x*64 + y*8 + z`, where:
- x = word index (0..7)
- y = byte index within the word (0..7) — bits 8y..8y+7 of `maskWords[x]`
- z = bit index within the byte (0..7)

**Invariant**: word x covers leaf-local positions x*64 .. x*64+63.  The bit at
position `y*8+z` within word x corresponds to leaf-local index `x*64 + y*8 + z`.

**Why this matters**: every algorithm that operates on the mask must respect this
layout.  The "fast" dimension (z, innermost) is the bit dimension within a byte.
The "slow" dimension (y) is the byte dimension within a word.  The "word" dimension
(x) is independent across the 8 words.

---

## 2. mPrefixSum Encoding

`LeafIndexBase::mPrefixSum` (a `uint64_t`) encodes 7 exclusive cumulative popcounts
at the word boundaries, packed as 7 x 9-bit fields:

    bits 9*(x-1) .. 9*(x-1)+8  =  total active voxels in words 0..x-1,  x = 1..7

- Field x-1 is the **exclusive** prefix at word x — i.e., how many active voxels
  precede word x in the leaf.
- xOffset[0] = 0 always (implicit, not stored).
- Maximum value per field: 512 (all 512 voxels in preceding words active) — fits in
  9 bits.
- **Precondition for buildMaskPrefixSums**: the caller must pass the leaf's own
  `mPrefixSum` field, not a recomputed value.  The field is set during grid
  construction and is authoritative.

---

## 3. buildMaskPrefixSums — Output Semantics

`util::buildMaskPrefixSums(mask, prefixSum, offsets[512])` produces:

    offsets[i] = number of active voxels at leaf-local positions 0..i (inclusive)

- Output is **inclusive** and **1-based**: for an active voxel at position i,
  `offsets[i]` is its 1-based rank among all active voxels in the leaf.
- Output is **leaf-local**: cross-word offsets from `mPrefixSum` are folded in,
  but the global leaf-first-offset (`leaf.data()->firstOffset()`) is NOT added.
  `offsets[i]` is in range [1, 512].
- For an **inactive** voxel at position i: `offsets[i] == offsets[i-1]` (no
  increment).  These values are valid and load-bearing for the `shfl_down`
  compaction approach (see §6).
- To recover the **exclusive** (0-based) rank of an active voxel at i:
  `offsets[i] - 1`.
- To recover the **global sequential index** of an active voxel at position i:
  `leafFirstOffset - 1 + offsets[i]`
  where `leafFirstOffset = leaf.data()->firstOffset()` is the 1-based global index
  of the leaf's first active voxel.

---

## 4. transposeByteRow — What It Does and Why

`transposeByteRow(src)` treats the low 8 bits of `src` as the first row of an 8x8
bit matrix and returns the result of transposing it:

    output.ui8[z] = (src >> z) & 1   for z = 0..7

It is the single-row specialization of `transposeBits8x8`: if only the first row of
an 8x8 bit matrix is non-zero, the full transpose reduces to this operation.

**Why it is used in buildMaskPrefixSums (Step 1 — indicator fill)**:
The algorithm needs `data[x][y].ui8[z] = indicator(x, y, z)` — a 0/1 value per
(x, y, z) triple.  The indicator for (x, y, z) is bit `y*8+z` of `maskWords[x]`,
which is bit z of byte y of `maskWords[x]`.  `transposeByteRow(maskWords[x] >> (y*8))`
extracts byte y and places bit z into byte z of the result — exactly the required
indicator layout.

**Equivalent hardware instruction**: `_pdep_u64(src & 0xFF, kSpread)` on x86, where
`kSpread = 0x0101010101010101`.  The software implementation is used for portability.

**Why not `& kSpread` (the simpler alternative)**: `(maskWords[x] >> z) & kSpread`
would extract bit z from each byte — which is what Plan #2 needed (data[x][z] layout).
Plan #1 uses data[x][y] layout, requiring the byte dimension to be the output axis,
not the input axis.  See §5 for the Plan #1 vs Plan #2 decision.

---

## 5. Plan #1 vs Plan #2 — The Rejected Alternative

**Plan #2** would have used layout `data[x][z].ui8[y]`:
- Simpler indicator fill: `data[x][z].ui64 = (maskWords[x] >> z) & kSpread`
  (just a shift and mask — no `transposeByteRow` needed).
- After the z-pass (Hillis-Steele within-uint64 over y), the output is in
  `data[x][z]` order, not `data[x][y]` order.
- **Fatal cost**: before zero-extending to uint16_t, a `transposeBytes8x8` call
  per x-slice is required to reorder from `data[x][z].ui8[y]` to the required
  linear output order.  This is ~200 instructions per call x 8 calls = ~1600
  instructions of transpose overhead, dominating the ~14-cycle z+y passes.

**Plan #1** (chosen) uses layout `data[x][y].ui8[z]`:
- Indicator fill requires `transposeByteRow` — slightly more expensive than `& kSpread`.
- After the z-pass and y-pass, the output is already in linear order (`x*64 + y*8 + z`).
- **No output transpose**: zero-extension to uint16_t is a straight vpmovzxbw over
  contiguous memory.

**Decision criterion**: Plan #1 eliminates the expensive output transpose at the cost
of a cheaper input transformation.  The output transpose (1024 bytes) is intrinsically
more expensive than the input transformation (64 bytes).

---

## 6. Compaction Approaches — Inclusive vs Exclusive, Active vs Inactive

Two approaches exist for building `leafLocalOffsets[j]` (local position of j-th active
voxel) from `offsets`:

**Scatter approach** (simple, used in decodeInverseMaps):
  For each active position i: `leafLocalOffsets[offsets[i] - 1] = i`
  Only requires `offsets[i]` at active positions.  Inactive positions are not read.

**shfl_down approach** (deeper SIMD, see DecodeInverseMapsCPUPlan §4e):
  Requires `shifts[i] = i - (offsets[i] - isActive(i))` for ALL positions, active
  and inactive.  The `move[i]` predicate for each pass depends on `shifts[i]` even
  for inactive voxels.  `buildMaskPrefixSums` correctly provides values at all 512
  positions for this purpose.

**Key invariant**: `offsets` from `buildMaskPrefixSums` is valid at all 512 positions,
not just active ones.  This is intentional and necessary for the shfl_down path.

---

## 7. decodeInverseMaps CPU — Current Implementation and Performance History

**Current implementation** (`VoxelBlockManager.h`, branch `vbm-cpu-port`):
For each leaf overlapping the block:
1. Build 513-entry exclusive prefix sum: `prefixSums[0]=0`, `buildMaskPrefixSums(..., prefixSums+1)`.
2. Compute `shifts[i] = i - prefixSums[i]` for i=0..511.
3. Run 9 shfl_down passes (Shift=1,2,4,...,256) via `shflDownSep` with ping-pong buffers.
4. Range fill `leafIndex[pStart..pEnd)` and contiguous copy from `leafLocalOffsets`.

**Performance history (2M voxels / 16384 blocks / 25% occupancy / 24 OMP threads / AVX2)**:
- Original `getValue()` loop: ~77 ms
- `buildMaskPrefixSums` + bit-scan scatter: ~65 ms (~15% improvement)
- shfl_down without vectorization (in-place, -fopenmp missing in CUDA host flags): ~250 ms
- shfl_down with proper vectorization (two-buffer __restrict__, -fopenmp fixed): ~15-20 ms

**Critical finding**: `#pragma omp simd` is silently ignored for CUDA host code unless
`-Xcompiler -fopenmp` is explicitly added.  Linking `OpenMP::OpenMP_CXX` does NOT
automatically propagate compile flags to CUDA sources via CMake.  Without -fopenmp,
the shfl_down passes compiled as scalar loops and were 4x slower than the bit-scan.

**Why shfl_down beats bit-scan with vectorization**:
- Bit-scan is inherently scalar (data-dependent BSF/BSR instruction, variable trip count).
- shfl_down's 9 passes are fixed-width, data-independent loops over 512 elements.
- With AVX2 (16 uint16_t per register), each pass takes ~32 vector ops vs ~128 scalar ops
  for the bit-scan at 25% occupancy.

---

## 8. Critical Portability Notes

**`UINT64_C` vs `UL` suffix**: always use `UINT64_C(...)` for 64-bit hex constants.
`UL` is 32 bits on MSVC/Windows.  Occurrences of `UL`-suffixed 64-bit constants in
`MorphologyHelpers.h` were corrected to `UINT64_C`.

**`__restrict__` portability**: `__restrict__` (GCC/Clang) vs `__restrict` (MSVC).
NanoVDB has no existing C++ portability macro for this.  `CNanoVDB.h` defines
`RESTRICT` but only for the C API.  A `NANOVDB_RESTRICT` macro should be added
if `__restrict__` is used in host-only headers.

**`#pragma omp simd` is safe without OpenMP**: unknown pragmas are silently ignored
by standard-conforming C++17 compilers.  All major compilers recognize the `omp`
namespace.  The pragma is present in `buildMaskPrefixSums` for portability; GCC 13.3
auto-vectorizes the loops correctly even without it.

**`#pragma omp simd` defeated by hardware POPCNT**: under `-mavx2`, GCC replaces
software Hamming-weight expressions with scalar `popcntl` and ignores the simd pragma.
The `popcount32` + `#pragma omp simd` approach in `vbm_host_cuda_kernels.cu` (sanity
bench) requires `-mno-popcnt` to vectorize, which is unsuitable for production.
`buildMaskPrefixSums` avoids this by not using popcount at all.

---

## 9. No __hostdev__ on buildMaskPrefixSums — Deliberate Decision

`buildMaskPrefixSums` is CPU-only.  A CUDA equivalent would be organized around the
32-thread warp (using `__ballot_sync`, warp-level prefix intrinsics, or cooperative
group reductions) and would look fundamentally different.  Marking it `__hostdev__`
would be misleading about intended usage and would invite incorrect porting.

The CUDA `decodeInverseMaps` already has its own highly optimized implementation.
The CPU and CUDA decode paths are expected to remain separate implementations
indefinitely.
