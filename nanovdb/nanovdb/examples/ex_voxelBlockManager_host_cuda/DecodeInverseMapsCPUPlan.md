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

### 4b. Storage Layout: `prefixCountRealigned[32][16]`

Declare a `uint32_t` array shaped as `[bitStep][lane]`:

```cpp
alignas(32) uint32_t prefixCountRealigned[/*bitStep*/32][/*lane*/16];
```

- **lane** (0..15): indexes which 32-bit word of the valueMask (one per group of 32 consecutive voxels).
- **bitStep** (0..31): indexes bit position within the word.
- `prefixCountRealigned[step][lane]` = **inclusive** prefix popcount = number of active voxels
  in positions 0..step of word `lane`.

Storage is `uint32_t` throughout to match `popcount32`'s natural precision and avoid
narrowing conversions. A full row `prefixCountRealigned[step]` is 16 × 4 = 64 bytes:
- **AVX2**: two `__m256i` registers per row (8 uint32_t each)
- **AVX-512**: one `__m512i` register per row (16 uint32_t) — the layout is designed for this upgrade

### 4c. Phase 1 — Per-Word Inclusive Prefix Counts (SIMD)

For each `step` in 0..31, compute `prefixCountRealigned[step][lane]` for all 16 lanes
simultaneously via a `#pragma omp simd` loop:

```cpp
const uint32_t* maskWords =
    reinterpret_cast<const uint32_t*>(leaf.valueMask().words());

for (int step = 0; step < 32; step++) {
    // TODO: use (uint32_t(2) << step) - 1u, NOT (1u << (step+1)) - 1u
    // The latter is UB at step=31 (shift by 32 on a 32-bit type).
    // The safe form: at step=31, (2u << 31) overflows to 0 (defined for unsigned),
    // and 0 - 1u wraps to 0xFFFFFFFF (all bits set) — correct inclusive mask.
    const uint32_t mask = (uint32_t(2) << step) - 1u;
    #pragma omp simd
    for (int lane = 0; lane < 16; lane++)
        prefixCountRealigned[step][lane] = popcount32(maskWords[lane] & mask);
}
```

At `step=31`, `mask = 0xFFFFFFFF`, so `prefixCountRealigned[31][lane] = wordPopcount[lane]`
(the full per-word active voxel count) — no separate word-popcount pass needed.

### 4d. Phase 2 — Cross-Word Prefix Sum and Global Conversion

Read the last row to get per-word counts, compute their exclusive prefix scan (scalar — short
dependency chain), then add `baseOffset[lane]` to every row in a second SIMD pass:

```cpp
// Exclusive prefix scan of the last row → baseOffset[lane]
uint32_t baseOffset[16];
baseOffset[0] = 0;
for (int lane = 1; lane < 16; lane++)
    baseOffset[lane] = baseOffset[lane-1] + prefixCountRealigned[31][lane-1];

// Add baseOffset to every row: converts per-word to global prefix counts
for (int step = 0; step < 32; step++) {
    #pragma omp simd
    for (int lane = 0; lane < 16; lane++)
        prefixCountRealigned[step][lane] += baseOffset[lane];
}
```

`baseOffset` is constant across all 32 steps for a given lane, so each row's SIMD add is a
simple lane-wise addition with no broadcast required. After this pass,
`prefixCountRealigned[step][lane]` holds the full global inclusive prefix count for voxel
`step + 32*lane` — i.e., the sequential index of that voxel within the leaf (0-based) if it
is active, counting all active voxels before it across all words.

### 4e. Parallel Prefix Compaction via `shfl_down` (Alternative / Deeper SIMD)

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

**Practical recommendation**: Start with the simpler vertical sweep (§4c/§4d). Fall back to
`shfl_down` if the compiler fails to vectorize the conditional store or if profiling shows it
is the bottleneck.

**TODO**: Investigate whether this collective SIMD prefix-popcount approach could benefit the
CUDA `decodeInverseMaps` as well. The current GPU implementation iterates all 512 voxel slots
via `getValue()` (one thread per slot cooperatively across the warp), which is already very fast
(~0.039 ms for 16384 blocks). Given that baseline, a rewrite is unlikely to be worthwhile, but
it may be worth a quick look once the CPU path is mature.

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

**Important:** `std::fill` on a `threadprivate` TLS pointer does **not** auto-vectorize to AVX2
stores even when `-mavx2` is enabled. The compiler cannot prove alignment through the TLS
indirection, so it falls back to scalar or SSE stores. Explicit AVX2 intrinsics with an
`(__m256i*)` cast are required to get `vmovdqa` and recover the expected bandwidth. On the test
machine (no AVX-512), using explicit `_mm256_store_si256` over `alignas(64)` arrays brought the
initialization cost from ~1.5 ms down to ~0.22 ms for 16384 blocks across 32 OMP threads.

The same issue will affect the `voxelOffset` range-fill and `leafIndex` range-fill in the
optimized path (§6 and §7): if the output arrays are caller-allocated (stack or TLS), `std::fill`
and `std::copy` should be replaced with explicit AVX2 stores where performance matters.

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

## 11. SIMD Codegen Experiment: `shfl_down`

The `simd_test/` directory (not checked into the repo) contained two source files and four
assembly listings produced by GCC 13.3 (`-O3 -march=avx512f` / `-O3 -march=avx2`).

### Source (both files identical except for the pragma)

```cpp
// shfl_down_test.cpp  — WITH #pragma omp simd
// shfl_down_nosimd.cpp — WITHOUT #pragma omp simd (testing auto-vectorization alone)

// Conditional blend: for j in [0, Width-Shift):
//   out[j] = (shifts[j+Shift] & Shift) ? in[j+Shift] : in[j]
// for j in [Width-Shift, Width):
//   out[j] = in[j]
template<typename T, int Shift, int Width>
void shfl_down(const T* __restrict__ in,
               const int* __restrict__ shifts,
               T* __restrict__ out)
{
#pragma omp simd   // omitted in shfl_down_nosimd.cpp
    for (int j = 0; j < Width - Shift; j++)
        out[j] = (shifts[j + Shift] & Shift) ? in[j + Shift] : in[j];

    for (int j = Width - Shift; j < Width; j++)
        out[j] = in[j];
}

// Instantiated for Shift = 1, 2, 4, 8, 16, 32, 64 with T=uint32_t, Width=128
```

### Assembly patterns observed

**AVX-512** (both files produced identical output — auto-vectorization sufficed):
```asm
; Per 16-element chunk, for Shift=S:
vpbroadcastd  S, %zmm0          ; broadcast shift constant
vpandd        S*4(%rsi), %zmm0, %zmm2   ; mask = shifts[j+S] & S
vpcmpd  $4, %zmm1, %zmm2, %k1   ; k1 = mask != 0  (take in[j+S])
vpcmpd  $0, %zmm1, %zmm2, %k2   ; k2 = mask == 0  (take in[j])
vmovdqu32  S*4(%rdi), %zmm3{%k1}{z}    ; load in[j+S] where mask != 0
vmovdqu32     (%rdi),  %zmm2{%k2}{z}   ; load in[j]   where mask == 0
vmovdqa32  %zmm3, %zmm2{%k1}           ; merge
vmovdqu32  %zmm2, (%rdx)               ; store
```
Each pass: 2 compares + 2 masked zero-loads + 1 masked merge + 1 store per 16 elements.

**AVX2** (both files produced identical output):
```asm
; Per 8-element chunk:
vpand    S*4(%rsi), %ymm1, %ymm3        ; mask = shifts[j+S] & S
vpcmpeqd %ymm0, %ymm3, %ymm3           ; ymm3 = (mask == 0) — "take in[j]" predicate
vpmaskmovd   (%rdi), %ymm3, %ymm4      ; load in[j]   where mask == 0
vpcmpeqd %ymm0, %ymm3, %ymm2           ; ymm2 = (mask != 0) — "take in[j+S]" predicate
vpmaskmovd S*4(%rdi), %ymm2, %ymm2    ; load in[j+S] where mask != 0
vpblendvb %ymm3, %ymm4, %ymm2, %ymm2  ; blend: ymm3 selects in[j], ymm2 selects in[j+S]
vmovdqu  %ymm2, (%rdx)                 ; store
```
Each pass: `vpand` + 2×`vpcmpeqd` + 2×`vpmaskmovd` + `vpblendvb` + store per 8 elements.

### Key findings

1. **`#pragma omp simd` was not needed on GCC 13.3** — the `nosimd` version auto-vectorized
   to identical output on both AVX-512 and AVX2. The pragma is still recommended for portability
   across compilers with weaker auto-vectorization. It is safe to use without guards: unknown
   pragmas are silently ignored by standard-conforming C++ compilers (C++17 §16.6), and all
   major compilers recognize the `omp` namespace even without OpenMP enabled.

2. **No architecture-specific intrinsics needed.** A single portable source compiles to optimal
   SIMD on both targets.

3. **No register-level shuffle instructions** (`vpermps`, `vpshufb`, etc.) appear anywhere. The
   fixed compile-time offset is treated as a constant address displacement — the "shuffle" is
   simply a load from `in + Shift`, which is a free addressing mode.

4. **AVX-512 is cleaner**: 5 instructions vs AVX2's 7 per chunk, and uses mask registers
   instead of `vpblendvb`.

5. **Software `popcount32`** (Hamming weight via AND/shift/add/multiply) auto-vectorizes to
   `VPMULLD` on both AVX2 and AVX-512. `VPOPCNTQ` (AVX-512VPOPCNTDQ) is **not** required.

6. **`__restrict__` is load-bearing, not just a hint.** Without it the compiler must assume
   `in` and `out` may alias, making vectorization of the loop illegal (writes to `out[j]` could
   affect subsequent reads of `in[j+Shift]`). The experiment results are only valid because
   `__restrict__` was present.

   `__restrict__` is a compiler extension, not standard C++ (`restrict` is C99 only). For
   portability a macro is needed. NanoVDB has no existing C++ macro for this — `CNanoVDB.h`
   defines `RESTRICT __restrict` but that is for the C API only. A new macro should be added:

   ```cpp
   #if defined(_MSC_VER)
   #  define NANOVDB_RESTRICT __restrict
   #else
   #  define NANOVDB_RESTRICT __restrict__
   #endif
   ```

   This matches the pattern used by `_CCCL_RESTRICT` in the bundled CCCL dependency.

---

## 12. Benchmarking Findings (ex_voxelBlockManager_host_cuda)

Measurements on the test machine (32 OMP threads, BlockWidth=128, 16384 blocks / 2M active
voxels, AVX2 but no AVX-512).

### Baseline numbers

| Path | Time per full pass |
|------|--------------------|
| GPU `decodeInverseMaps` (all blocks, `benchDecodeKernel`) | ~0.039 ms |
| CPU `decodeInverseMaps`, 32 OMP threads, unoptimized (`getValue()` loop) | ~77 ms |
| CPU initialization only (AVX2 stores, 32 threads) | ~0.22 ms |
| CPU OMP scheduling overhead (empty loop body, 16384 iterations) | ~0.002 ms |

The GPU/CPU gap is ~2000×. The `getValue()` loop accounts for essentially all of the CPU cost.

### OMP parallelism

The outer loop over blocks (`#pragma omp for schedule(static)`) parallelizes correctly — a
fill-only sanity check scaled from ~77ms (single-thread equivalent) to ~1.5ms with 32 threads
(~40×). However the full `decodeInverseMaps` showed **zero scaling** with OMP threads. This
confirms the bottleneck is memory-bandwidth or cache-thrashing in the `getValue()` traversal,
not compute: all 32 threads together saturate available bandwidth accessing leaf data, giving no
wall-time improvement over serial.

### `getValue()` is the bottleneck

`getValue(localOffset)` on a `ValueOnIndex` leaf accesses `mValueMask` and the packed
`mPrefixSum` field to compute the sequential index. It is read-only but touches leaf node data
for every one of 512 slots per leaf, for every leaf overlapping the block. The unoptimized path
is O(512 × nLeaves) memory accesses per block rather than O(64 bytes of valueMask) per leaf.
Replacing this with the prefix-array approach (§4) is the primary optimization target.

### Build flags

`-mavx2` must be passed explicitly to both the host compiler and nvcc (`-Xcompiler -mavx2`).
Without it, `std::fill` on TLS pointers generates scalar stores. The flag is set in
`examples/CMakeLists.txt` via `target_compile_options` for `ex_voxelBlockManager_host_cuda`.

### `prefix_popcnt_bench` standalone micro-benchmark

`prefix_popcnt_bench.cpp` (in the same directory) isolates the Phase 1 + Phase 2 computation —
1M blocks, each with a runtime-unknown 16-word mask generated by an LCG, single-threaded,
`prefixCountRealigned[32][16]` allocated outside the loop. Results on the test machine (AVX2,
no AVX-512, GCC 13.3, `-O3 -mavx2`):

| Implementation | Min time (1M blocks) | ns/block |
|----------------|---------------------|----------|
| Auto-vectorised (`popcount32` + `#pragma omp simd`) | ~130 ms | ~124 ns |
| Auto-vectorised with `-mno-popcnt` | ~101 ms | ~96 ns |
| Explicit AVX2 intrinsics (`vpshufb` nibble-table) | ~70 ms | ~66.5 ns |

**Key finding — `#pragma omp simd` is silently defeated by `-mavx2`.**
When hardware POPCNT is available (implied by `-mavx2` on x86), GCC replaces the `popcount32`
Hamming-weight expression with the scalar `popcntl` instruction and then runs the lane loop
scalar. The `#pragma omp simd` hint is ignored because the compiler considers scalar `popcntl`
cheaper than the vectorised software path. The result is 16 sequential `popcntl` calls per step,
not a SIMD operation across all 16 lanes.

With `-mno-popcnt`, GCC falls back to the software Hamming weight and auto-vectorizes correctly
to the 2×`__m256i` path (~96 ns). However `-mno-popcnt` is not suitable for production (it
disables hardware POPCNT throughout the TU, including places like `countOn` where it is wanted).

**Explicit `vpshufb` intrinsics** (`computePrefixPopcntAVX2` in `prefix_popcnt_bench.cpp`)
bypass this issue entirely: the nibble-table lookup uses ~10 SIMD instructions per step across
all 16 lanes, without any `popcntl` in sight. At ~66.5 ns/block this is **1.87× faster** than
the auto-vectorised baseline and is the approach to use in the optimised CPU `decodeInverseMaps`.

`vpshufb` popcount recipe (8 uint32 lanes per `__m256i`, applied twice for 16 lanes):
```cpp
// lut[i] = popcount(i), for i in 0..15, replicated in both 128-bit lanes
const __m256i lut   = _mm256_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
                                       4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
const __m256i low4  = _mm256_set1_epi8(0x0f);
const __m256i ones8 = _mm256_set1_epi8(1);
const __m256i ones16= _mm256_set1_epi16(1);

__m256i lo  = _mm256_and_si256(v, low4);                          // low nibbles
__m256i hi  = _mm256_and_si256(_mm256_srli_epi16(v, 4), low4);   // high nibbles
__m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                               _mm256_shuffle_epi8(lut, hi));     // byte popcounts
__m256i s   = _mm256_madd_epi16(_mm256_maddubs_epi16(cnt, ones8), ones16); // sum → 32-bit
```
`maskWords` (constant across all 32 steps) should be loaded into two `__m256i` registers once
before the step loop — the compiler will hoist the broadcasts of `lut`, `low4`, `ones8`, `ones16`
automatically.

---

## 13. Alternative Algorithm: Bit-Parallel Z+Y Prefix Sum

This section records an alternative algorithm under investigation for computing
`uint16_t prefixSums[512]` (exclusive linear prefix popcount per voxel) from a `Mask<3>`
(`valueMask` of a leaf node), using bit-parallel operations on the 8 × 64-bit mask words.
The algorithm is implemented and tested in `simd_test/within_word_prefix_test.cpp`.

### 13a. Data Layout

```cpp
union qword { uint64_t ui64; uint8_t ui8[8]; };
static constexpr uint64_t kSpread = 0x0101010101010101ULL;

qword data[8][8];  // indexed [z][x]
// data[z][x].ui8[y]  ↔  voxel (x, y, z), x = word index, y*8+z = bit within word
```

NanoVDB leaf linear index: `i = x*64 + y*8 + z`.  Word index = x (0..7), within-word bit
position = y\*8+z, with z as fast index (bits 0..2 of each 8-bit group) and y as slow (byte
index 0..7 within the 64-bit word).

`data[z][:]` is contiguous — 64 bytes = one cache line = two YMM registers.  This enables
`#pragma omp simd` over x in both passes below.

### 13b. Z-Pass: Indicator Fill + Running Sum

```cpp
// z=0: extract bit 0 from each byte of each word
#pragma omp simd
for (int x = 0; x < 8; x++)
    data[0][x].ui64 = maskWords[x] & kSpread;

// z=1..7: accumulate bit z from each byte, running sum over z
for (int z = 1; z < 8; z++) {
    #pragma omp simd
    for (int x = 0; x < 8; x++)
        data[z][x].ui64 = data[z-1][x].ui64 + ((maskWords[x] >> z) & kSpread);
}
```

After this pass: `data[z][x].ui8[y]` = Σ_{z'≤z} bit(x, y, z') — per-column z-prefix for
each (x, y).  Per-byte maximum = 8; fits in `uint8_t` with no inter-byte carry.  `vpaddq`
and `vpaddb` are equivalent here.

**Latency hiding**: the indicator fill `(maskWords[x] >> z) & kSpread` is independent of
`data[z-1][x]`, so the OOO engine can issue it during the 1-cycle `vpaddq` latency.  The
7-step dependency chain runs at ~1 cycle/step (throughput-bound, not latency-bound).

### 13c. Y-Pass: Hillis-Steele Prefix Scan Within uint64

```cpp
for (int z = 0; z < 8; z++) {
    #pragma omp simd
    for (int x = 0; x < 8; x++) {
        data[z][x].ui64 += data[z][x].ui64 << 8;
        data[z][x].ui64 += data[z][x].ui64 << 16;
        data[z][x].ui64 += data[z][x].ui64 << 32;
    }
}
```

`vpsllq imm8` is fully supported in AVX2 (1-cycle throughput, 1-cycle latency).  Per-byte
maximum after this pass: 64 (8 z-values × 8 y-values); still fits in `uint8_t`.  No
inter-byte carry corruption since bytes evolve independently under byte-parallel arithmetic.

After this pass: `data[z][x].ui8[y]` = **2D rectangle inclusive sum** =
Σ_{y'≤y, z'≤z} bit(x, y', z').

### 13d. Assembly Quality (GCC 13.3, -O3 -march=core-avx2)

The compiler fully unrolls both passes and keeps all intermediate values register-resident.
The z-pass processes `data[z][:]` two YMM registers at a time (x=0..3 and x=4..7), with
one spill (z=7, x=0..3 half) due to requiring all 16 YMM registers simultaneously.  The
y-pass operates directly on the register-resident z-pass results without reloading from
memory.  The only missed optimization is 16 dead stores from the z-pass that are immediately
overwritten by the y-pass.  Overall this is essentially what hand-written intrinsics would
produce.

### 13e. 2D Rectangle vs Linear Prefix (Correctness Finding)

**Key finding from `simd_test/within_word_prefix_test.cpp`**: the z+y algorithm computes a
**2D rectangle sum**, not the linear prefix sum that `getValue()` uses.

`getValue()` for `ValueOnIndex` computes: `countOn(w & ((1ULL << (y*8+z)) - 1))` = exclusive
count of set bits at positions 0..y\*8+z−1 within word x.  This is a **linear** prefix (a
staircase: all bits in rows 0..y−1 plus bits in row y up to column z).

The 2D rectangle sum Σ_{y'≤y, z'≤z} bit(x,y',z') counts only up to column z in every
preceding row, missing the "row tails" for y' < y.  Test result on 1000 random masks:
2D rectangle matches its own reference at 100% (512000/512000); linear inclusive match is
only ~26% (132806/512000), confirming the discrepancy.

First mismatch example: at (x=0, y=1, z=0), 2D rect = 2 (bits at y=0,z=0 and y=1,z=0),
linear inclusive = 7 (all 7 bits at positions 0..8 in the word).

### 13f. Rectangle→Linear Fixup

The linear inclusive prefix at (x, y, z) can be recovered from the 2D rectangle data as:

```
linear_incl(x, y, z) = data[7][x].ui8[y-1]      // all complete rows 0..y-1 (z'=0..7)
                      + data[z][x].ui8[y]          // current row y, columns 0..z
                      - data[z][x].ui8[y-1]        // subtract over-counted rectangle below
```

This simplifies to adding a y-dependent correction, expressible as a byte-parallel operation:

```cpp
for (int z = 0; z < 8; z++) {
    #pragma omp simd
    for (int x = 0; x < 8; x++)
        data[z][x].ui64 += (data[7][x].ui64 - data[z][x].ui64) >> 8;
}
```

`data[7][x].ui64` (available in registers after the y-pass) gives the full per-row popcounts
packed in bytes; the byte-shift-right-by-8 shifts row y−1's value into row y's byte lane.
This fixup is cheap — one subtract and one shift per (z, x) pair, all in-place in the
byte-packed representation.

### 13g. Cross-Word Offsets (mPrefixSum)

`LeafIndexBase::mPrefixSum` stores 7 nine-bit cumulative popcounts (the exclusive prefix
scan at word boundaries):

- bits 0–8:   Σ_{j=0}^{0} countOn(words[j])  = exclusive prefix at x=1
- bits 9–17:  Σ_{j=0}^{1} countOn(words[j])  = exclusive prefix at x=2
- ...
- bits 54–62: Σ_{j=0}^{6} countOn(words[j])  = exclusive prefix at x=7

These are available for free and must be added to `data[z][x].ui8[y]` to obtain the full
global sequential index.  However, these offsets require up to 9 bits (max value = 512),
which exceeds `uint8_t`.  Two approaches for incorporating them:

**Approach #1 — Pack offsets into a uint64 byte lane and vpaddq directly.**  This fails for
any leaf where the cross-word cumulative count exceeds 255 (i.e., more than ~255 active
voxels in the preceding words — reachable for moderately dense leaves by the 4th word).
Only viable for very sparse leaves.

**Approach #2 — Transpose to uint16_t prefixSums[8][8][8].**  Unpack the byte-packed
`data[z][x].ui8[y]` into `uint16_t prefixSums[x][y][z]` (indexed [x][y][z] = linear order),
then add the 9-bit cross-word offsets in the wider format.  Widening is safe; all values fit
in uint16_t (max = 512).  The cost is a 3D index-permutation transpose
`(z,x,y) → (x,y,z)` on 64 bytes → 128 bytes.

### 13h. Transposition Cost and Alternatives

The output transpose (approach #2) is expensive in isolation: no loop ordering gives a
unit-stride inner loop for both source and destination simultaneously, so GCC cannot
auto-vectorize it.  With explicit AVX2 intrinsics (8×8 byte matrix transpose per x-slice,
8 slices) the cost is ~200 instructions; even scalar it is ~512 operations on L1-resident
data (~400–800 cycles), dominating the ~14-cycle z+y passes.

**Bit-transpose alternative**: pre-transpose the 8 input uint64_t words (64 bytes) instead
of post-transposing 512 uint16_t values (1024 bytes).  The specific transposition that makes
the algorithm output naturally land in `[x][y][z]` memory order is: organize input as
`inputWords[y]` with bit `z*8+x` = B[x][y][z] (making y the word index, z the byte index,
and x the step variable).  Transposing 64 bytes is intrinsically cheaper than transposing
1024 bytes, and the 8×8 bit-matrix transpose per y-slice is a well-studied ~10–15 instruction
operation.

**Key tradeoff — good output order ↔ simple rectangle→linear fixup:**

With the original layout (word=x, byte=y, step=z), the 2D rectangle is over (y, z) for fixed
x, and the rectangle→linear fixup collapses to the single byte-shift expression in §13f.

With the bit-transposed layout (word=y, byte=z, step=x), the 2D rectangle is over (x, z) for
fixed y, and the "missing" terms for the linear prefix involve cross-word contributions from
all y-slices of preceding words — a significantly more complex expression that does not reduce
to a simple in-register byte operation.

No 3D transposition of the input eliminates both costs simultaneously.  The original layout
remains preferred for the simplicity of the fixup; the output transpose cost must be addressed
separately (either by tolerating it, using explicit intrinsics, or changing the consumer's
expected layout).
