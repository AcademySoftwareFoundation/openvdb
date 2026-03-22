Read the following algorithm description and use it to inform your understanding of the bit-parallel z+y prefix sum algorithm being developed in `simd_test/`. Confirm your understanding with a brief summary and indicate you are ready to assist.

---

# Bit-Parallel Z+Y Prefix Sum Algorithm

## Goal

Given the 8 words of a NanoVDB `Mask<3>` (the `valueMask` of a leaf node), compute for every voxel `(x, y, z)` with `x,y,z ∈ [0,7]` the **exclusive linear prefix popcount** — i.e. the count of active voxels whose linear index `i = x*64 + y*8 + z` is strictly less than the target voxel's index. This is exactly what `LeafData<ValueOnIndex>::getValue()` returns (minus the global `mOffset` and cross-word `mPrefixSum` terms, which are handled separately).

## Data Layout

```cpp
union qword { uint64_t ui64; uint8_t ui8[8]; };
static constexpr uint64_t kSpread = 0x0101010101010101ULL;

qword data[8][8];  // data[z][x]
// data[z][x].ui8[y]  corresponds to voxel (x, y, z)
```

The layout `data[z][x]` is chosen so that `data[z][:]` (all x at a fixed z) is contiguous in memory — 64 bytes = one cache line = two YMM registers. This enables `#pragma omp simd` over x in both the z-pass and y-pass.

NanoVDB leaf coordinates:
- Word index = x (0..7), `maskWords[x] = mask.words()[x]`
- Within a 64-bit word, bit position = `y*8 + z`, with z as the **fast** index (bits 0..2 within a byte group of 8) and y as the **slow** index (bits 3..5, i.e. byte index within the word).

## Algorithm

### Step 1: Indicator fill (z = 0)

```cpp
#pragma omp simd
for (int x = 0; x < 8; x++)
    data[0][x].ui64 = maskWords[x] & kSpread;
```

`(maskWords[x] >> 0) & kSpread` isolates bit y*8+0 of word x into byte y of the uint64.
So `data[0][x].ui8[y]` = 1 if voxel (x, y, 0) is active, else 0.

### Step 2: Z-pass (running sum over z, i.e. inclusive prefix over z for each (x,y))

```cpp
for (int z = 1; z < 8; z++) {
    #pragma omp simd
    for (int x = 0; x < 8; x++)
        data[z][x].ui64 = data[z-1][x].ui64 + ((maskWords[x] >> z) & kSpread);
}
```

After this step:
```
data[z][x].ui8[y] = Σ_{z'=0..z} bit(x, y, z')
                  = count of active voxels in the z-column at (x, y), up to z
```

The indicator fill `(maskWords[x] >> z) & kSpread` is independent of `data[z-1]`, so the OOO engine can hide the 1-cycle `vpaddq`/`vpaddb` latency by issuing the shift+AND during the preceding add's latency. The 7-step dependency chain runs at ~1 cycle/step on AVX2 (throughput-bound, not latency-bound).

Per-byte maximum after the z-pass: 8 (at most 8 active voxels per z-column). Fits in `uint8_t` with no carry between bytes. `vpaddq` and `vpaddb` are equivalent here.

### Step 3: Y-pass (Hillis-Steele prefix scan within uint64, over y-bytes)

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

`vpsllq imm8` is fully supported in AVX2 (1-cycle throughput, 1-cycle latency).
Per-byte maximum after the y-pass: 64 (8 z-columns × 8 y-rows). Fits in `uint8_t`.
No inter-byte carry corruption because bytes are independent under `uint8_t` arithmetic.

After this step:
```
data[z][x].ui8[y] = Σ_{y'=0..y, z'=0..z} bit(x, y', z')
                  = 2D rectangle inclusive sum over [0..y] × [0..z] within word x
```

## Current Status: 2D Rectangle Sum, Not Yet Linear Prefix

The z+y passes produce a **2D rectangle inclusive sum**, which differs from the **linear exclusive prefix**. The discrepancy: for voxel `(x, y, z)`, the linear prefix also counts bits in rows `y' < y` at `z' > z` (i.e. the tails of preceding y-rows beyond depth z).

The linear inclusive prefix at `(x, y, z)` can be recovered as:
```
linear_incl(x, y, z) = data[7][x].ui8[y-1]   // all complete rows 0..y-1
                      + data[z][x].ui8[y]      // current row 0..y at depth 0..z
                      - data[z][x].ui8[y-1]    // subtract over-counted rectangle below
```

i.e. `= data[7][x].ui8[y-1] + (data[z][x].ui8[y] - data[z][x].ui8[y-1])`

The correction `data[7][x].ui8[y-1]` depends only on x and y (not z), making it a single additive fixup per (x, y) pair. The algorithm extension to compute the full linear prefix from the 2D rectangle data is a pending step.

## Test File

`simd_test/within_word_prefix_test.cpp` — standalone correctness test verifying:
1. `data[z][x].ui8[y]` matches the 2D rectangle reference (PASSES)
2. `data[z][x].ui8[y]` vs the linear inclusive prefix (shows discrepancy at ~74% of positions)

Compile: `g++ -O3 -march=core-avx2 -fopenmp -std=c++17 -o within_word_prefix_test within_word_prefix_test.cpp`
