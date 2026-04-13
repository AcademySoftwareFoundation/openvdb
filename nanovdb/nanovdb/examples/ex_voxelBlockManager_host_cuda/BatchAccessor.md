# BatchAccessor — SIMD Batch Leaf-Neighborhood Cache

This document is the design reference for `BatchAccessor`, the SIMD-batch analog
of NanoVDB's `ValueAccessor`.  It captures the full design rationale developed
alongside the `ex_stencil_gather_cpu` Phase 1 prototype.

---

## 1. Motivation and Core Analogy

NanoVDB's `DefaultReadAccessor` amortizes the cost of root-to-leaf tree traversal
by caching the path for a single voxel.  When successive scalar `getValue(ijk)` calls
land in the same leaf, only the first call pays the full traversal; subsequent calls
hit the cached leaf pointer in ~6 integer instructions.

`BatchAccessor` lifts this idea one level: instead of caching the path to one leaf,
it caches the **3×3×3 neighborhood of leaf pointers** surrounding the current center
leaf.  Instead of serving one voxel per call, it serves a **SIMD batch of SIMDw
voxels** simultaneously.

| Property | Scalar `ValueAccessor` | `BatchAccessor` |
|----------|------------------------|-----------------|
| Cache unit | Path root→leaf (3 node ptrs) | 27 neighbor leaf ptrs |
| Granularity | 1 voxel per call | SIMDw voxels per call |
| Cache key | Voxel coordinate in cached leaf's bbox | `currentLeafID` (VBM ordering) |
| "Hit" condition | Next voxel in same leaf | `mProbedMask` covers needed direction |
| Eviction trigger | Implicit on any miss | Explicit: `none_of(leafMask)` |
| Guarantee of hit rate | Access-pattern dependent | Structural (VBM Morton ordering) |

The hit rate of the scalar accessor depends on the access pattern.  `BatchAccessor`'s
amortization is **structural**: the VBM groups voxels by leaf, so within any batch,
the center leaf is known in advance, and directions probed for batch k remain valid
for all subsequent batches in the same center leaf.

---

## 2. Cache State

Four pieces of state persist across batches within one center leaf:

```cpp
template<typename BuildT, int SIMDw>
class BatchAccessor {
    uint32_t       mProbedMask    = 0;    // bit d set ↔ direction d has been probed
    const LeafT*   mPtrs[27]      = {};   // canonical neighbor table; mPtrs[13] = center
    uint32_t       mCurrentLeafID;        // index of current center leaf
    nanovdb::Coord mCenterLeafCoord;      // origin of current center leaf
    // (plus a reference to the underlying grid for probeLeaf calls)
};
```

`mPtrs[27]` uses the shared 3×3×3 direction encoding from `StencilGather.md §6a`:

```
bit(dx, dy, dz) = (dx+1)*9 + (dy+1)*3 + (dz+1)     dx,dy,dz ∈ {-1, 0, +1}
```

`mPtrs[13]` (the center, `bit(0,0,0)`) always points to
`&tree.getFirstNode<0>()[mCurrentLeafID]`.  The 26 non-center entries are populated
lazily by `prefetch` calls.

**Cache advance:** when `mCurrentLeafID` changes:

```cpp
void advance(uint32_t newLeafID) {
    mCurrentLeafID  = newLeafID;
    mProbedMask     = 0;           // stale neighbor ptrs; force re-probe before use
    mCenterLeafCoord = tree.getFirstNode<0>()[newLeafID].origin();
    // mPtrs[] entries are stale but harmless; mProbedMask=0 prevents their use
}
```

---

## 3. Eviction and the `leafMask` — The Straddle Problem

This is the key structural difference from the scalar accessor.

In the scalar case, "cache miss" and "eviction" are the same event — the single voxel
is either in the cached leaf or it isn't.  In the batch case they decouple:

- **Straddle lanes**: active voxels in the batch that belong to a *later* leaf
  (`leafIndex[i] != currentLeafID`, `leafMask[i] = false`).  The cache is still valid
  for the remaining current-leaf lanes.  No eviction.
- **Eviction**: `none_of(leafMask)` — no lane in this batch belongs to the current
  leaf.  Only then does `advance()` fire.

`leafMask` is therefore the accessor's **partial-hit signal** — a concept that has
no scalar analog.  Without it, the accessor would evict prematurely on every straddle
batch, losing the cross-batch amortization that makes `mProbedMask` valuable.

The straddle lane problem is solved at the call site by masking: straddle lanes receive
a sentinel voxelOffset value (`kSentinelExpanded = expandVoxelOffset(292)`, local
coordinate (4,4,4)) that produces no false direction bits in either the plus-OR or
minus-AND reduction.  This is already implemented and verified in the Phase 1
prototype (`ex_stencil_gather_cpu`).

---

## 4. The Prefetch Insight — Extremal Taps as a Neighborhood Census

The naive "vanilla accessor" approach would issue a `probeLeaf` call on first access
for each stencil tap, lazily.  The `BatchAccessor` exploits **domain-specific
knowledge of the stencil geometry** to warm the cache with a minimal set of
strategically chosen taps — the *extremal* taps — that together constitute a complete
census of the neighborhood.

### 4a. WENO5 (Axis-Aligned, Reach R=3) — 6 Extremal Taps

For an axis-aligned stencil, only one axis can cross a leaf boundary per tap.  The
condition for needing the x+ neighbor leaf is:

```
∃ delta ∈ {1..R}  s.t.  lx + delta ≥ 8   ↔   lx ≥ 8 − R
```

The extremal tap at `+R` detects exactly `lx + R ≥ 8 ↔ lx ≥ 8 − R` — which is the
**necessary and sufficient condition** for needing x+ at all.  Any smaller delta for
the same voxel would probe the same x+ leaf if it crosses, or not cross at all.

Therefore, prefetching the 6 extremal taps covers all directions needed by any
intermediate tap:

```
prefetch<+R, 0, 0>,  prefetch<-R, 0, 0>   → x+ / x- face leaves
prefetch< 0,+R, 0>,  prefetch< 0,-R, 0>   → y+ / y- face leaves
prefetch< 0, 0,+R>,  prefetch< 0, 0,-R>   → z+ / z- face leaves
```

For WENO5 with R=3: **6 probeLeaf calls maximum** per center leaf, covering all
19 stencil taps.  This is identical to what `computeNeededDirs` computes (the carry
trick encodes all 6 thresholds simultaneously).

### 4b. 3×3×3 Box Stencil (R=1) — 8 Corner Taps

For the box stencil, a stencil tap at `(lx+dx, ly+dy, lz+dz)` where `dx,dy,dz ∈
{-1,0,+1}` can cross one, two, or three axes simultaneously (face, edge, or corner
neighbor leaf respectively).

**Claim**: the 8 corner taps `(±1, ±1, ±1)` collectively cover all 26 non-center
neighbor directions for any voxel position in the batch.

**Coverage argument**: For any voxel `(lx, ly, lz)` and any direction
`(dx, dy, dz)` that the stencil actually needs (i.e., some coordinate crosses a leaf
boundary), there exists a corner tap `(sx, sy, sz)` with `sx, sy, sz ∈ {-1, +1}`
such that when applied to this voxel it probes the **same neighbor leaf**.

Concretely, the corner tap `(-1,-1,+1)` applied to voxel `(0, 0, 4)` accesses
`(-1, -1, 5)`, which falls in the `(x−, y−)` edge leaf — the same leaf needed by
the edge tap `(-1, -1, 0)` for this voxel.  The corner tap `(-1,+1,-1)` for the
same voxel accesses `(-1, 1, 3)`, falling in the `x−` face leaf — the same leaf
needed by `(-1, 0, 0)`.

Each corner tap, applied to varying voxel positions in the batch, will probe face,
edge, or corner leaves depending on how many axes actually cross — collectively
exhausting all 26 directions across the batch.

**At most 8 probeLeaf calls** per center leaf for the full 27-point box stencil
(in practice fewer, since many corner taps land in the center leaf for interior
voxels, and `mProbedMask` prevents re-probing the same direction twice).

---

## 5. API — Three Tiers

### 5a. Core Functions

```cpp
// ── Tier 1a: warm the cache for a specific stencil offset ──────────────────
// For each active (leafMask) lane: compute which neighbor leaf the tap
// (di,dj,dk) falls in, probe it into mPtrs[] if not already in mProbedMask.
// Takes treeAcc — may call probeLeaf.
template<int di, int dj, int dk>
void prefetch(Simd<uint16_t,SIMDw> vo, LaneMask leafMask, AccT& treeAcc);

// ── Tier 1b: read from cache (cache assumed warm) ──────────────────────────
// For each active lane: compute local offset within the cached neighbor leaf,
// fetch and return the value (or index for ValueOnIndex grids).
// Does NOT take treeAcc — guaranteed not to touch the tree.
// Debug builds assert mProbedMask covers the needed direction.
template<int di, int dj, int dk>
Simd<ValueT,SIMDw> cachedGetValue(Simd<uint16_t,SIMDw> vo, LaneMask leafMask) const;

// ── Tier 2: lazy combined operation (vanilla accessor style) ───────────────
// Equivalent to prefetch<di,dj,dk> + cachedGetValue<di,dj,dk>.
// Correct without explicit prefetch management; slightly suboptimal for
// repeated calls in the same center leaf (redundant bitmask checks).
template<int di, int dj, int dk>
Simd<ValueT,SIMDw> getValue(Simd<uint16_t,SIMDw> vo, LaneMask leafMask, AccT& treeAcc);
```

The presence or absence of `treeAcc` in the signature is self-documenting:
`cachedGetValue` is the only function that can be called in a "no tree access"
context, and the compiler enforces that it doesn't get one.

### 5b. Usage Patterns

**Tier 1 — production path** (explicit prefetch, recommended for performance-critical
stencil kernels):

```cpp
// Warm the cache with the 6 WENO5 extremal taps
batchAcc.prefetch<-3, 0, 0>(vo, leafMask, treeAcc);
batchAcc.prefetch<+3, 0, 0>(vo, leafMask, treeAcc);
batchAcc.prefetch< 0,-3, 0>(vo, leafMask, treeAcc);
batchAcc.prefetch< 0,+3, 0>(vo, leafMask, treeAcc);
batchAcc.prefetch< 0, 0,-3>(vo, leafMask, treeAcc);
batchAcc.prefetch< 0, 0,+3>(vo, leafMask, treeAcc);

// All cachedGetValue calls are pure arithmetic + gather — no tree access
auto u_m3 = batchAcc.cachedGetValue<-3, 0, 0>(vo, leafMask);
auto u_m2 = batchAcc.cachedGetValue<-2, 0, 0>(vo, leafMask);
auto u_m1 = batchAcc.cachedGetValue<-1, 0, 0>(vo, leafMask);
auto u_0  = batchAcc.cachedGetValue< 0, 0, 0>(vo, leafMask);
auto u_p1 = batchAcc.cachedGetValue<+1, 0, 0>(vo, leafMask);
auto u_p2 = batchAcc.cachedGetValue<+2, 0, 0>(vo, leafMask);
auto u_p3 = batchAcc.cachedGetValue<+3, 0, 0>(vo, leafMask);
// ... y and z axes similarly

Simd<float,SIMDw> flux_x = wenoKernel(u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3);
```

**Tier 2 — prototyping path** (lazy, correct, no explicit prefetch management):

```cpp
// Identical stencil formula; each getValue probes lazily on first need
auto u_m3 = batchAcc.getValue<-3, 0, 0>(vo, leafMask, treeAcc);
auto u_m2 = batchAcc.getValue<-2, 0, 0>(vo, leafMask, treeAcc);
// ...
```

The redundant `prefetch` calls inside non-extremal `getValue` invocations reduce to
a single `mProbedMask` bitmask check and immediate return — the direction was already
probed by an earlier extremal call.

### 5c. Invariant Ordering

In Tier 1, all `prefetch` calls must precede all `cachedGetValue` calls for the same
batch.  A debug-mode RAII scope guard (`batchAcc.beginGather()` / `endGather()`) could
enforce this, but is probably overkill for a first implementation.

---

## 6. Template vs Runtime Interface

### 6a. Arguments for `<di, dj, dk>` Template Parameters

- **Compile-time direction resolution**: for `cachedGetValue<-3,0,0>`, the compiler
  proves only lx can cross, and only leftward.  The direction bit reduces to a
  compile-time choice between two constants (`mPtrs[4]` or `mPtrs[13]`); y/z
  boundary checks are eliminated entirely.
- **Dead axis elimination**: for axis-aligned taps, two of the three axis checks
  vanish at compile time.
- **VDB convention alignment**: `WenoPt<i,j,k>::idx`, `NineteenPt<i,j,k>::idx` —
  the ecosystem already addresses stencil points as compile-time named entities.
- **Structural contract**: the `prefetch`/`cachedGetValue` pairing is expressible as
  a static invariant when offsets are compile-time constants.

### 6b. When Runtime `nanovdb::Coord` Is Needed

A generic `computeStencil<StencilT>` that iterates over `StencilT::offsets` at
runtime cannot use template parameters.  A runtime overload:

```cpp
Simd<ValueT,SIMDw> getValue(nanovdb::Coord offset,
                             Simd<uint16_t,SIMDw> vo,
                             LaneMask leafMask, AccT& treeAcc);
```

dispatches through a small switch on the runtime direction bit (26 cases, easily
predicted).  The gather still dominates; the dispatch overhead is negligible.

**C++20 note**: if `nanovdb::Coord` is made a structural type, the template and
runtime interfaces unify naturally:

```cpp
template<nanovdb::Coord offset>
Simd<ValueT,SIMDw> cachedGetValue(Simd<uint16_t,SIMDw> vo, LaneMask leafMask) const;

// Called as:
batchAcc.cachedGetValue<nanovdb::Coord(-3,0,0)>(vo, leafMask);
```

### 6c. Recommendation

- **Template `<di,dj,dk>`** as the primary, idiomatic interface for all hand-written
  stencil kernels — cleaner codegen, natural fit with VDB conventions.
- **Runtime `Coord` overload** for generic stencil adapters and prototyping loops.
- Both interfaces backed by the same `mPtrs[]` / `mProbedMask` state machine.

---

## 7. AVX2 Vectorization Profile

### 7a. `prefetch<di,dj,dk>` — Crossing Detection

```
Extract lx/ly/lz from all 16 vo lanes     vpsrl / vpand  ymm (SIMD)
Compare lx+di against [0,7]               vpcmpgtd       ymm (SIMD)
Fold crossing mask to scalar bitmask      vmovmskps      ymm (SIMD)
AND with ~mProbedMask                     scalar bitmask check
If new direction needed: probeLeaf        scalar (≤1 call per prefetch for WENO5)
```

Structurally identical to the `computeNeededDirs` carry trick in the prototype
(indeed, `prefetch<di,dj,dk>` is `computeNeededDirs` specialized to a single tap).

### 7b. `cachedGetValue<di,dj,dk>` — Offset Arithmetic and Gather

```
Compute neighbor offsets for all 16 lanes:
  nx[i] = lx[i] + di,  wrapped to [0,7]  vpaddd / vpand  ymm (SIMD, constant di)
  local offset[i] = nx[i]*64 + ny[i]*8 + nz[i]          vpmadd / vpaddd ymm

Determine which lanes cross to neighbor leaf:
  crossMask = (lx < threshold)            vpcmpgtd        ymm (SIMD)

Gather values from (at most) two leaf arrays:
  centerVals   = gather(mPtrs[13]->array, offset)         vgatherdps  ymm (SIMD)
  neighborVals = gather(mPtrs[dir]->array, offset_wrapped) vgatherdps ymm (SIMD)
  result = blend(crossMask, neighborVals, centerVals)     vpblendvb   ymm (SIMD)
```

The key insight: for axis-aligned WENO5 taps, there are **at most two distinct leaf
pointers** across all 16 lanes.  This reduces the gather to two base-pointer loads
plus a predicated blend — a clean AVX2 pattern.

### 7c. Comparison to Phase 1 Prototype

The two scalar bottlenecks in the prototype are eliminated by `BatchAccessor`:

| Phase 1 bottleneck | `BatchAccessor` replacement | AVX2? |
|--------------------|----------------------------|-------|
| `expandVoxelOffset` scatter (conditional per-lane) | `cachedGetValue` offset arithmetic (uniform SIMD add) | ✓ |
| `batchPtrs` fill (pointer scatter, data-dependent) | Crossing mask + gather + blend | ✓ |
| `probeLeaf` loop | `prefetch<di,dj,dk>` (≤1 probeLeaf per call) | inherently scalar |

The WENO kernel itself (`wenoKernel(u_m3, ..., u_p3)`) operates entirely on
`Simd<float,SIMDw>` with no tree access in sight.

### 7d. Complete Per-Batch AVX2 Profile

| Operation | Instructions | Vectorized? |
|-----------|-------------|-------------|
| `activeMask` computation | `vpcmpeqd ymm ×4` + `vmovmskps ×2` | ✓ Full |
| `leafMask` computation | `vpbroadcastd` + `vpcmpeqd ymm ×2` + `vmovmskps ×2` | ✓ Full |
| `prefetch` crossing detection | `vpcmpgtd ymm` + `vmovmskps` | ✓ Full |
| `probeLeaf` (per prefetch) | scalar tree traversal | inherently scalar |
| `cachedGetValue` offset arithmetic | `vpaddd ymm` / `vpand ymm` | ✓ Full |
| `cachedGetValue` lane split | `vpcmpgtd ymm` | ✓ Full |
| `cachedGetValue` value gather | `vgatherdps ymm ×2` + `vpblendvb ymm` | ✓ Full |
| WENO kernel | `Simd<float,SIMDw>` arithmetic | ✓ Full |

---

## 8. Scoping and Lifetime

A `BatchAccessor` is scoped to **one CPU thread**, constructed once before the block
loop and reused across all batches and all blocks:

```cpp
BatchAccessor<BuildT, SIMDw> batchAcc(grid, firstLeafID[0]);

for (uint32_t bID = 0; bID < nBlocks; bID++) {
    decodeInverseMaps(..., leafIndex, voxelOffset);

    for (int b = 0; b < BlockWidth; b += SIMDw) {
        // compute activeMask, leafMask ...
        while (any_of(activeMask)) {
            if (none_of(leafMask)) {
                batchAcc.advance(++currentLeafID);
                continue;
            }
            // prefetch / cachedGetValue / kernel ...
        }
    }
}
```

**Cross-block carryover**: resetting `mProbedMask` between blocks is safe and simple.
Carrying over is also valid — consecutive blocks process spatially adjacent leaves,
so some `mPtrs[]` entries may still be correct.  In practice, resetting is recommended
(one `mProbedMask = 0` per block, negligible cost) to avoid subtle stale-pointer bugs.

---

## 9. Relationship to the Phase 1 Prototype

`ex_stencil_gather_cpu` (`stencil_gather_cpu.cpp`) implements the core cache
machinery as free functions:

| Prototype component | `BatchAccessor` equivalent |
|--------------------|-----------------------------|
| `probedMask` + `ptrs[27]` locals | `mProbedMask` + `mPtrs[27]` members |
| `computeNeededDirs(expandedVec)` | inner logic of `prefetch<di,dj,dk>` (one tap) |
| `kSentinelExpanded` broadcast | same sentinel in `prefetch` for straddle lanes |
| `probeLeaf` loop (`toProbe` bits) | `prefetch` body |
| `batchPtrs[4][SIMDw]` population | replaced by `cachedGetValue` gather + blend |
| `verifyBatchPtrs` | future: `cachedGetValue` unit test |

Phase 2 (not yet implemented): `cachedGetValue` — the actual index/value gather from
the cached leaf pointers.  The AVX2 machinery for crossing detection and offset
arithmetic is a direct extension of what is already working and verified in Phase 1.

---

## 10. Open Questions / Future Work

- **`ValueOnIndex` two-level fetch**: `cachedGetValue` returns `Simd<uint64_t,SIMDw>`
  indices for index grids; a `cachedGetValue<di,dj,dk>(channel, vo, leafMask)` overload
  dereferences through a channel pointer in one step.  Channel data layout (AoS vs SoA)
  affects gather efficiency.

- **Multi-leaf stencils (R > 4)**: the single-neighbor-per-axis assumption breaks for
  stencils with reach R > 4 (a center voxel can simultaneously need both the lo and hi
  neighbor along the same axis).  `mPtrs[27]` remains correct; only the `cachedGetValue`
  lane-split logic (currently "at most 2 leaf pointers per axis tap") needs generalization.

- **Generic stencil adapter**: a `computeStencil<StencilT>` wrapper that calls
  `getValue(StencilT::offset(n), ...)` for `n = 0..N-1` via the runtime `Coord`
  overload — correctness-first entry point for new stencil types.

- **C++20 structural `Coord`**: unify template and runtime interfaces with
  `cachedGetValue<nanovdb::Coord(-3,0,0)>(vo, leafMask)` non-type template parameter.

- **Debug-mode RAII scope guard**: enforce the prefetch-before-cachedGetValue ordering
  in debug builds without any runtime cost in release.

- **Launcher integration**: the `BatchAccessor` is a per-block, per-thread object.
  The system-level launcher (the `buildVoxelBlockManager` analogue for stencil
  computation) constructs one per worker thread and passes it into the per-block kernel.
  Design of the launcher is deferred until the per-block kernel is fully validated.
