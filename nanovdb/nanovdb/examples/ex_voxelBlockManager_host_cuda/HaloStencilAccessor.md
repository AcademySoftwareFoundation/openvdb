# HaloStencilAccessor — Design Document

## §1  Motivation

`StencilAccessor` (see `StencilAccessor.md`) was measured at **537 cycles/voxel**
for the index-gathering phase alone (W=16, `Simd<uint64_t,16>`, rdtsc on i9-285K).
The root cause is structural, not a tuning issue:

- 18 taps × `Simd<uint64_t,16>` = 18 YMM registers needed simultaneously for
  `mIndices[]`, but x86 has only 16 YMM registers available.
- The compiler spills every tap slot to memory → every `mIndices[I]` write is a
  store, every subsequent read is a load.
- Additionally, the gather step (reading float values from the sidecar using the
  computed indices) is a second pass of scattered 64-bit indexed loads.

**HaloStencilAccessor** eliminates both problems by replacing the
index-buffer-then-gather pattern with a local dense halo buffer that is filled
once per center-leaf run, from which all stencil values are extracted via
sequential, gather-free SIMD operations.

---

## §2  Key idea: dense halo buffer

For a given center leaf `L`, densify the sidecar float data for `L` and its
6 axis-aligned face-neighbor leaves into a contiguous local array:

```
float buf[16][16][16]
```

The center leaf occupies positions `[R..R+7]` in each dimension (where `R` is
the stencil radius cap, see §3).  Any stencil tap at compile-time offset
`(di, dj, dk)` from center voxel `(i, j, k)` is then:

```cpp
buf[R + i + di][R + j + dj][R + k + dk]
```

This is a branch-free, uniform address expression valid for **any** tap with
`|di|, |dj|, |dk| ≤ R`.  No tree traversal, no ValueOnIndex arithmetic, no
leaf-pointer lookup occurs inside the stencil extraction loop.

---

## §3  Why R=4 and 16³

For a center leaf of 8×8×8 voxels, supporting stencils up to radius R requires
`(8 + 2R)` voxels per dimension:

| R | buffer side | buffer size | L1 resident? |
|---|-------------|-------------|--------------|
| 1 (box 3³)  | 10³  |  ~4 KB | yes |
| 3 (WENO5)   | 14³  | ~11 KB | yes |
| **4 (cap)** | **16³** | **16 KB** | **yes** |
| 5           | 18³  | ~23 KB | yes |

`R = 4` is chosen because:

1. **16 KB fits comfortably in L1** (P-core L1d = 48 KB).  The buffer stays
   L1-resident throughout the processing of an entire center-leaf run.
2. **16 = 2³ × 2 → trivially simple addressing.**  No `bi/ii` split; a flat
   `[16][16][16]` array indexed by `R + i + di` (a 4-bit quantity, 0–15).
3. **Covers both WENO5 (R=3) and any future stencil up to R=4** without
   redesign.
4. Powers-of-two strides (256, 16, 1) admit bit-shift addressing.

**Axis-aligned stencils never access corner or edge neighbor slots.**  For
WENO5, taps move in exactly one axis → only the 6 face-neighbor leaf regions
of the 16³ buffer are ever read.  Corner/edge slots may be left zero-initialized
(background value) and are never consumed.

### Buffer population

Slots that are populated from the sidecar:

| Region | Voxels | Source |
|--------|--------|--------|
| Center leaf `[R..R+7]³` | 8³ = 512 | sidecar[center leaf] |
| −x face slab `[0..R-1][R..R+7][R..R+7]` | R×8×8 = 256 | sidecar[−x neighbor] |
| +x face slab `[R+8..15][R..R+7][R..R+7]` | 256 | sidecar[+x neighbor] |
| −y, +y, −z, +z face slabs | 256 each | sidecar[respective neighbors] |

**Total sidecar reads per center-leaf run: 512 + 6×256 = 2,048 floats = 8 KB.**
Corner and edge slots are zero-initialized once at buffer allocation.

---

## §4  Run-based outer loop

Within a VBM block (128 voxels), the `leafIndex[]` array produced by
`decodeInverseMaps` is sorted (the VBM is built in leaf-traversal order).
Voxels belonging to the same center leaf therefore form **contiguous runs**.

A narrow-band block of 128 voxels sitting inside a single 8³ leaf spans at
most 1–3 distinct center leaves per block in practice.

```
for each VBM block:
    scan leafIndex[0..127] for run boundaries
    for each run (center leaf L, voxels i_start..i_end):
        populate buf[16][16][16] from sidecar   // 8 KB, amortized over run
        process all voxels in [i_start..i_end] via the slice pipeline (§6)
```

The buffer fill and the 7 neighbor-leaf pointer lookups are amortized across
all voxels in the run.

---

## §5  Stencil extraction: the fill → transpose → compact → transpose pipeline

### §5a  Why array-of-stencils [32][512] is easy to fill from buf[]

For tap `t = (di, dj, dk)`, the 512 values `stencil[t][v]` for all center-leaf
voxel positions `v = i*64 + j*8 + k` are produced by iterating over the 64
z-rows (fixed `i`, `j`; `k` = 0..7):

```
for each z-row (i, j):
    source = &buf[R+i+di][R+j+dj][R+dk]    // 8 consecutive floats in buf
    dest   = &stencil[t][i*64 + j*8]        // 8 consecutive floats in row t
    store 8 floats (one YMM)
```

Both source (L1-resident `buf`) and destination (stencil row `t`) are accessed
sequentially — **zero gathers**.  All 18 WENO5 taps fill sequentially; the 14
padding slots (see §5b) are either zero-initialized once or left unused.

### §5b  Padding taps to 32

WENO5 has 18 taps.  Padding to 32 (next power of 2) gives:

- Each voxel row in `stencil[*][32]` is exactly **128 bytes = 2 cache lines**,
  naturally aligned.
- Compaction of one row (§5d) is exactly **4 full YMM loads + 4 full YMM
  stores** — no masking, no partial registers, no scalar tail.
- `stencil[v]` address = `base + (v << 7)`: a single shift, no multiply.
- 14 padding slots are never read during WENO5 arithmetic and cost nothing.

### §5c  Layout duality and the 8×8 in-register transpose

Two layouts are needed at different pipeline stages:

| Layout | Alias | Size | Best for |
|--------|-------|------|----------|
| `float[32][N]` | SoA | 18/32 contiguous values per tap | sequential fill from buf |
| `float[N][32]` | AoS | 32 contiguous values per voxel | WENO5 arithmetic, compaction |

Converting between them: the standard **8×8 SIMD float block transpose**.

Given 8 YMM registers (one row each = 8 floats), the 8×8 transpose applies a
fixed 24-instruction register-only shuffle network
(`vunpcklps`/`vunpckhps` → `vunpcklpd`/`vunpckhpd` → `vperm2f128`)
and stores 8 result YMM registers.  All shuffles are register-to-register;
no intermediate memory is touched between the 8 loads and 8 stores.

For an M×N matrix (both M and N multiples of 8):
- Number of 8×8 block transposes: (M/8) × (N/8)
- Uses exactly 16 YMM registers (8 input + 8 output) — exact fit

### §5d  Compaction: AoS[512][32] → AoS[N_active][32]

Given `stencil[512][32]` (AoS, all leaf positions) and the `voxelOffset[]`
list of N_active active voxels (0–511, from `decodeInverseMaps`):

```
for v in 0..N_active:
    ymm0..3 = load stencil[voxelOffset[v]][0..31]   // 4 × YMM load
    store compact[v][0..31]                           // 4 × YMM store
```

Total: N_active × 8 YMM operations.  Rows are 128-byte aligned; no masking.

### §5e  Slicing: keep the working set in L1

`stencil[32][512]` = 64 KB — L2 resident.  Instead, process 4 slices of 128
voxels each:

```
stencil[32][128] = 32 × 128 × 4 = 16 KB   ← L1 resident
```

With the 16³ `buf` (16 KB) and the stencil slice (16 KB), the total working
set is **32 KB**, well within L1 (48 KB).  All fill and transpose phases stay
in L1 — no L2 traffic during data transformation.

Slice boundaries: voxel positions 0–127, 128–255, 256–383, 384–511 within the
center leaf.  Each slice is processed identically; active-voxel results are
accumulated across slices.

---

## §6  Per-slice pipeline (full detail)

For each of the 4 slices `[s*128 .. (s+1)*128 - 1]`:

```
Step 1  Fill stencil[32][128]
        ── For each tap t in 0..17:
               For each z-row (i,j) with voxels in this slice:
                   load 8 floats from buf (L1) → store to stencil[t][row] (L1)
        ── Cost: 9 KB reads + 9 KB writes, all L1.

Step 2  Transpose [32][128] → [128][32]
        ── 64 × 8×8 in-register block transposes.
        ── Cost: 16 KB L1 read + 16 KB L1 write; all shuffles register-only.

Step 3  Compact [128][32] → [N_slice][32]
        ── For each active voxel in this slice: 4 YMM loads + 4 YMM stores (L1).
        ── N_slice ≤ 128.  Cost: ≤ 16 KB L1 read + ≤ 16 KB L1 write.

Step 4  Transpose [N_slice][32] → [32][N_slice]
        ── ≤ 64 × 8×8 in-register block transposes.
        ── Cost: ≤ 16 KB L1 read + write.

Step 5  WENO5 arithmetic on stencil[32][N_slice]
        ── For each of the 18 taps: sequential load of N_slice floats (L1).
        ── ~700 FLOPs/voxel, vectorised over N_slice voxels in YMM batches.
        ── Cost: ~700 × N_slice / 32 cycles (2 FMA units × 8 floats).

Step 6  Write output
        ── N_slice scalar (or YMM-masked) stores to output sidecar.
```

---

## §7  Performance analysis

### Measured baseline (StencilAccessor, W=16)

- **8,586 TSC ticks/batch** (16 voxels) → **537 cycles/voxel**
- Gather phase only; WENO5 arithmetic not yet included.
- Root cause: register spilling of 18 × `Simd<uint64_t,16>`.

### CPU parameters (i9-285K, TSC reference clock)

| Resource | Throughput |
|----------|-----------|
| L1 read  | 64 bytes/cycle (~237 GB/s at 3.7 GHz) |
| L1 write | 32 bytes/cycle (~118 GB/s) |
| L2 read  | 32 bytes/cycle (~118 GB/s) |
| FMA peak | 32 FLOPs/cycle (2 units × 8 floats × 2 FLOPs) |

### Estimated cost per slice (128 voxels, ~32 active)

| Step | Data touched | Estimated cycles |
|------|-------------|-----------------|
| 1 — fill stencil[32][128] | 9 KB L1 write | ~288 |
| 2 — transpose [32][128]→[128][32] | 16 KB L1 r+w | ~512 |
| 3 — compact | 16 KB L1 r+w | ~512 |
| 4 — transpose compact output | ≤16 KB L1 r+w | ~256 |
| 5 — WENO5 arithmetic (32 active) | 18×32×4=2 KB L1 read | ~700 |
| **Total per slice** | | **~2,268** |

Per voxel (32 active): ~71 cycles/voxel — including full WENO5 arithmetic.

Four slices + buffer fill (~300 cycles amortised): ~9,372 cycles per 128-voxel
block → ~**73 cycles/voxel total**.

### Comparison

| Metric | StencilAccessor | HaloStencilAccessor (est.) |
|--------|-----------------|---------------------------|
| Gather phase only | 537 cycles/voxel | ~50 cycles/voxel |
| Gather + WENO5 | not measured | ~73 cycles/voxel |
| Dominant bottleneck | register spilling | L1 write bandwidth |
| Gathers in hot loop | yes (scattered 64-bit) | **none** |

**Estimated speedup over StencilAccessor: 7–10× on the gather phase;
gather + WENO5 combined comes in under the cost of gathering alone in the
old design.**

---

## §8  Future optimisations

### §8a  Fuse fill + first transpose (steps 1+2)

Fill 8 tap rows × 8 voxels into YMM registers, immediately transpose the
8×8 block and store in AoS order.  Eliminates one full pass over the 16 KB
stencil slice; saves ~512 cycles per slice.

### §8b  Fuse WENO5 arithmetic with step 4

Rather than materialising `stencil[32][N_slice]` (SoA), compute WENO5
directly from `stencil[N_slice][32]` (AoS) by loading each voxel's 32-float
row and computing vertically.  Eliminates step 4 entirely.  Effective when
N_slice is small (dense run ≤ 32 voxels).

### §8c  Software pipelining across slices

While WENO5 runs on slice `s`, fill the stencil buffer for slice `s+1`.
The two phases touch disjoint L1 regions; overlap is feasible.

### §8d  TBB parallel_for over blocks

VBM blocks are independent (grid is read-only).  Each thread owns its
block's 32 KB working set (buf + slice); no synchronisation required.
Expected 7–8× speedup across 8 P-cores.

---

## §9  Design decisions summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Stencil radius cap | R = 4 | 16³ = 16 KB, L1 resident; power-of-2 |
| Tap count padding | 18 → 32 | YMM-aligned compaction (4 full registers) |
| Dense or sparse fill | Dense (all 512 leaf positions) | Branchless; cheaper than compaction logic during fill |
| Slice size | 128 voxels (4 slices of 512) | buf(16 KB) + slice(16 KB) = 32 KB ≤ L1 |
| Transpose kernel | 8×8 in-register float block | 16 YMM registers, no memory between load/store |
| Compaction order | After first transpose | Driven by sorted voxelOffset[] from decodeInverseMaps |
| Outer loop | Run-based (by center leaf) | Amortises buffer fill over entire run |

---

## §10  Open questions

1. **Leaf-pointer resolution**: buffer fill still requires resolving 6 face-neighbor
   leaf pointers via tree traversal.  Should this reuse BatchAccessor's
   neighbor-lookup machinery, or be a standalone 6-pointer lookup?

2. **Missing neighbors**: if a face-neighbor leaf does not exist in the grid,
   the corresponding slab should be zero-filled (background = 0 for
   ValueOnIndex grids).  Confirm zero-init strategy for absent neighbors.

3. **Non-uniform active-voxel density**: some slices may have 0 active voxels
   (entire slice inactive).  Add a slice-skip predicate?

4. **Output sidecar write-back**: the `voxelOffset` of each active voxel gives
   its ValueOnIndex; use that to write the WENO5 result directly to the output
   sidecar.  Confirm index arithmetic.

5. **Tap padding slots (18..31)**: never read in WENO5 arithmetic.  Can be
   left uninitialised (no UB since never read) or zero-filled once.  Decide
   at implementation time.
