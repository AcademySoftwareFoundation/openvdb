# BatchAccessor — SIMD Batch Leaf-Neighborhood Cache

Design reference for `BatchAccessor.h`.  Captures the full design rationale
and API contract developed alongside the `ex_stencil_gather_cpu` Phase 1 prototype.

---

## 1. Motivation and Core Analogy

NanoVDB's `DefaultReadAccessor` amortizes the cost of root-to-leaf tree traversal
by caching the path for a single voxel.  When successive scalar `getValue(ijk)` calls
land in the same leaf, only the first call pays the full traversal.

`BatchAccessor` lifts this idea one level: instead of caching the path to one leaf,
it caches the **3×3×3 neighborhood of leaf pointers** surrounding the current center
leaf.  Instead of serving one voxel per call, it serves a **SIMD batch of LaneWidth
voxels** simultaneously.

| Property | Scalar `ValueAccessor` | `BatchAccessor` |
|----------|------------------------|-----------------|
| Cache unit | Path root→leaf (3 node ptrs) | 27 neighbor leaf ptrs |
| Granularity | 1 voxel per call | LaneWidth voxels per call |
| Cache key | Voxel coordinate in cached leaf's bbox | `mCenterLeafID` |
| "Hit" condition | Next voxel in same leaf | `mProbedMask` covers needed direction |
| Eviction trigger | Implicit on any miss | Explicit: `none_of(leafMask)` |
| Hit rate guarantee | Access-pattern dependent | Structural (VBM Morton ordering) |

The hit rate of the scalar accessor depends on the access pattern.  `BatchAccessor`'s
amortization is **structural**: the VBM groups voxels by leaf, so within any batch the
center leaf is known in advance, and directions probed for batch k remain valid for all
subsequent batches in the same center leaf.

---

## 2. Template Parameters

```cpp
template<typename BuildT,
         typename ValueT       = float,
         typename VoxelOffsetT = uint16_t,
         typename LeafIDT      = uint32_t,
         typename PredicateT   = bool>
class BatchAccessor;
```

| Parameter | Scalar default | SIMD example | Role |
|-----------|---------------|--------------|------|
| `BuildT` | — | — | NanoVDB build type; determines `LeafT`, `TreeT` |
| `ValueT` | `float` | `Simd<float,W>` | Return type of `cachedGetValue` |
| `VoxelOffsetT` | `uint16_t` | `Simd<uint16_t,W>` | Compact 9-bit voxel offset within a leaf |
| `LeafIDT` | `uint32_t` | `Simd<uint32_t,W>` | Per-lane leaf ID (reserved for caller loop) |
| `PredicateT` | `bool` | `SimdMask<float,W>` | Per-lane active predicate |

For `NanoGrid<ValueOnIndex>`, use `ValueT = uint64_t` (scalar) or
`ValueT = Simd<uint64_t,W>` (SIMD).

The scalar defaults allow instantiation without a SIMD library, giving a clean
scalar path for debugging and cross-validation.

Per-lane access is provided by `nanovdb::util::simd_traits<T>` (defined in `Simd.h`),
which works for both scalar and vector types via specialisation.

---

## 3. Persistent State

Four members persist across batches within one center leaf:

```cpp
const GridT& mGrid;                            // for probeLeaf calls via mGrid.tree()
uint32_t     mCenterLeafID;                    // index of current center leaf
Coord        mCenterOrigin;                    // world-space origin of current center leaf
uint32_t     mProbedMask = (1u << 13);         // bit 13 (center) pre-set at construction
uint32_t     mNeighborLeafIDs[27];             // kNullLeafID when outside narrow band or unprobed
```

**Direction encoding** (`dir` is a `static constexpr` member):

```
dir(dx, dy, dz) = (dx+1)*9 + (dy+1)*3 + (dz+1)     dx,dy,dz ∈ {-1,0,+1}
```

`mNeighborLeafIDs[27]` is a flat array indexed by `dir(dx,dy,dz)`.
`mNeighborLeafIDs[13]` (= `dir(0,0,0)`) holds the center leaf ID.
`mNeighborLeafIDs[d] = kNullLeafID` when the neighbor lies outside the narrow band or
has not yet been probed.

```cpp
static constexpr uint32_t kNullLeafID = ~uint32_t(0);
```

**Why leaf IDs, not pointers:**  `cachedGetValue` fetches `mOffset`, `mPrefixSum`, and
`valueMask().words()[w]` for all active lanes via SIMD gathers (§8d).  The gather index
is `leaf_id × (sizeof(LeafT)/sizeof(uint64_t))`, computed once per call as a
`Simd<uint32_t,W>` multiply.  Storing IDs enables a single flat-base gather over the
contiguous leaf array; storing pointers would require per-lane pointer arithmetic that
doesn't map to `vgatherdpd` / `vpgatherqq`.  The `kNullLeafID` sentinel cleanly
replaces `nullptr` and is masked out in the gather via `where`.

**Cache advance:** when `none_of(leafMask)` fires in the outer loop:

```cpp
void advance(uint32_t newLeafID) {
    mCenterLeafID                  = newLeafID;
    mCenterOrigin                  = mGrid.tree().getFirstLeaf()[newLeafID].origin();
    for (auto& id : mNeighborLeafIDs) id = kNullLeafID;
    mNeighborLeafIDs[dir(0, 0, 0)] = newLeafID;
    mProbedMask                    = (1u << dir(0, 0, 0));
}
```

All 27 entries are reset to `kNullLeafID` on advance; `mProbedMask` is set to only
bit 13.  `toProbe = neededMask & ~mProbedMask` therefore never returns a stale index.

---

## 4. Eviction and the Straddle Problem

In a SIMD batch, "straddle lanes" are active voxels that belong to a *later* leaf
(`leafIndex[i] != mCenterLeafID`, `leafMask[i] = false`).  They do NOT trigger an
eviction — the cache is still valid for the remaining current-leaf lanes.

Eviction fires only when `none_of(leafMask)` — no lane in the batch belongs to the
current leaf.

`leafMask` is the accessor's **partial-hit signal** — a concept with no scalar analog.

Straddle lanes are given the inactive sentinel voxel offset `kInactiveVoxelOffset`
(= local coordinate (4,4,4)), which is strictly interior to the leaf and generates
no false crossing detections.  The outer `while (any_of(activeMask))` loop processes
one leaf ID per iteration, re-using the same SIMD batch:

```
while any_of(activeMask):
    leafMask = activeMask & (leafIndex_vec == mCenterLeafID)
    if none_of(leafMask):
        acc.advance(++currentLeafID)
        continue
    # prefetch + cachedGetValue for leafMask lanes only
    acc.prefetch<...>(vo, leafMask)
    acc.cachedGetValue<...>(result, vo, leafMask)   # fills leafMask lanes of result
    activeMask &= ~leafMask
# all lanes now filled; call kernel once with complete result
```

---

## 5. Center Leaf Initialisation — Eager (Constructor and advance)

`mLeafNeighbors[dir(0,0,0)]` (center) is populated **eagerly** by both the
constructor and `advance()`:

```cpp
mNeighborLeafIDs[dir(0,0,0)] = mCenterLeafID;
mProbedMask = (1u << dir(0,0,0));   // bit 13 pre-set
```

The center pointer is O(1) to compute — no `probeLeaf` traversal needed — so there
is no reason to defer it.

**Consequences:**

- `cachedGetValue<0,0,0>` (center tap) is valid immediately after construction or
  `advance()`, without any `prefetch` call.
- The SWAR `neededMask` computed inside `prefetch` never needs to include bit 13:
  crossings are detected per-axis, and a lane whose tap stays in the center leaf
  contributes `dir(0,0,0)` which is already in `mProbedMask` and filtered by
  `toProbe = neededMask & ~mProbedMask`.
- The `if (d == dir(0,0,0))` special case is removed from the probe loop: every
  direction in `toProbe` is a genuine neighbor requiring `probeLeaf`.

---

## 6. API

### 6a. Direction Helper

```cpp
static constexpr int dir(int dx, int dy, int dz);
```

### 6b. Lifecycle

```cpp
BatchAccessor(const GridT& grid, uint32_t firstLeafID);
void advance(uint32_t newLeafID);
```

### 6c. Tier 1a — `prefetch`

```cpp
template<int di, int dj, int dk>
void prefetch(VoxelOffsetT vo, PredicateT leafMask);
```

- Computes the neighbor direction for each active lane.
- Probes at most one new leaf per unique direction per call (skips directions
  already in `mProbedMask`).
- Calls `mGrid.tree().probeLeaf(coord)` directly — no `AccT` parameter.
  `ReadAccessor` is not used because `probeLeaf` only hits the LEVEL=0 leaf cache,
  which is never warm for neighbor leaves; the internal-node caches are bypassed
  entirely for `GetLeaf` operations.
- The center direction is set from `mCenterLeafID` without `probeLeaf`.

### 6d. Tier 1b — `cachedGetValue`

```cpp
template<int di, int dj, int dk>
void cachedGetValue(ValueT& result, VoxelOffsetT vo, PredicateT leafMask) const;
```

- Fills **only the `leafMask` lanes** of `result` (by reference).
- Inactive lanes are not touched — values from a previous iteration are preserved.
- This is the correct API for the straddle-aware outer loop: the caller declares
  all stencil result variables before the `while` loop, fills them progressively
  across iterations, and calls the kernel once after `activeMask` is empty.
- Requires the corresponding direction to be in `mProbedMask` (asserted in debug).
- `nullptr` leaf (outside narrow band) writes `ScalarValueT(0)`.

### 6e. Deferred

`getValue<di,dj,dk>` (lazy combined) and the runtime `nanovdb::Coord` overload
are not yet implemented.  Both are additive and straightforward once the two
primitives above are validated.

---

## 7. Prefetch Patterns

### WENO5 (R=3, axis-aligned) — 6 extremal taps

```cpp
acc.prefetch<-3, 0, 0>(vo, leafMask);
acc.prefetch<+3, 0, 0>(vo, leafMask);
acc.prefetch< 0,-3, 0>(vo, leafMask);
acc.prefetch< 0,+3, 0>(vo, leafMask);
acc.prefetch< 0, 0,-3>(vo, leafMask);
acc.prefetch< 0, 0,+3>(vo, leafMask);
// All subsequent cachedGetValue calls are pure arithmetic — no tree access.
auto u_m3 = /* ... */; acc.cachedGetValue<-3,0,0>(u_m3, vo, leafMask);
auto u_m2 = /* ... */; acc.cachedGetValue<-2,0,0>(u_m2, vo, leafMask);
// ... 19 taps total
Simd<float,W> flux_x = wenoKernel(u_m3, u_m2, u_m1, u_0, u_p1, u_p2, u_p3);
```

### Box stencil (R=1) — 8 corner taps

```cpp
for each (sx,sy,sz) in {±1}³:
    acc.prefetch<sx,sy,sz>(vo, leafMask);
// then cachedGetValue for all 27 taps
```

---

## 8. Implementation Notes

### 8a. SIMD structure of prefetch and cachedGetValue

**`prefetch` — fully SIMD for the crossing detection, scalar only for probeLeaf**

`prefetch` contains no per-lane scalar loop.  The crossing decision uses:

1. **SWAR expansion** (YMM throughout): `vpsllw`, `vpor`, `vpand` — maps the 9-bit
   voxel offset vector into the 15-bit packed form across all LaneWidth lanes.
2. **Sentinel blend**: `vpblendvb` — applies `leafMask` in one instruction.
3. **Add**: `vpaddw` — adds the compile-time `packed_tap` across all lanes.
4. **Horizontal reductions**: `vextracti128` + `vpand`/`vpor` tree → scalar `hor_and`
   / `hor_or` — unavoidable for the crossing decision, which is a single bool per axis.

Assembly-confirmed (Release, `-O3 -mavx2`, `ex_stencil_gather_cpu`):

```
vmovdqu  (%rbx,%rax,2),%ymm2       ; load vo (16 × uint16_t)
vpsllw   $0x4,%ymm2,%ymm0          ; vo << 4
vpor     %ymm2,%ymm0,%ymm0         ; vo | (vo << 4)
vpand    %ymm1,%ymm0,%ymm0         ; & 0x1C07
vpsllw   $0x2,%ymm2,%ymm1          ; vo << 2
vpand    %ymm2,%ymm1,%ymm1         ; & 0xE0
vpor     %ymm1,%ymm0,%ymm0         ; → expanded
vpblendvb %ymm1,%ymm0,%ymm6,%ymm1  ; where(leafMask, packed_lc) = expanded
vpaddw   %ymm2,%ymm1,%ymm1         ; packed_sum = packed_lc + packed_tap
vextracti128 $0x1,%ymm1,%xmm2      ; \
vpand    %xmm1,%xmm2,%xmm2         ;  | hor_and tree:
vpunpckhwd ...                      ;  | 16→8→4→2→1 lanes
vpand    ...; vpshufd ...; vpand .. ;  |
vpextrw  $0x0,%xmm1,%eax           ; / scalar hor_and
```

After the scalar crossing check, `probeLeaf` is called at most once per unique
direction per center leaf — inherently scalar tree traversal, not per-voxel.

**`cachedGetValue` — SIMD ingredient fetch, scalar value path**

The ingredient-fetch block — `mOffset`, `mPrefixSum[w]`, and `valueMask().words()[w]`
for all active lanes — is **fully SIMD** via the gather chain described in §8d.
The final value-fetch (scalar loop over `leaf->getValue(offset)`) is the remaining
work before the full SIMD index pipeline is wired in.

### 8b. No tree accessor in prefetch

NanoVDB's `ReadAccessor` is not passed to `prefetch`.  Its LEVEL=0 leaf cache is never
warm for neighbor leaves (by definition distinct from the center leaf), and its
internal-node caches are bypassed entirely when `get<GetLeaf>` misses at LEVEL=0.
`probeLeaf` is equivalent to a direct root traversal in all non-trivial cases.

### 8c. probeLeaf returns nullptr for missing neighbors

`mGrid.tree().probeLeaf(coord)` returns `nullptr` when the requested coordinate lies
outside the active narrow band.  `prefetch` stores `kNullLeafID` in
`mNeighborLeafIDs[d]` for those directions.  `cachedGetValue` detects `kNullLeafID`
and writes `ScalarValueT(0)` for those lanes, which is correct for level-set grids
(background value = 0).  The SIMD gather chain masks out `kNullLeafID` lanes via the
`valid_u32` mask before accessing any leaf data.

### 8d. SWAR direction extraction — the base-32 multiply trick

`cachedGetValue` must compute a **per-lane** neighbor direction `dir ∈ [0,26]` at
runtime, because for a fixed compile-time tap `(di, dj, dk)` different lanes can land
in different neighbor leaves (one lane may cross only the z-face; another may cross
x and z; another may stay in the center leaf).

`dir` is the mixed-radix value `dir = cz + 3·cy + 9·cx` where each carry component
`cz, cy, cx ∈ {0,1,2}` encodes {underflow, in-leaf, overflow} for the z-, y-, x-axis
respectively.  The carry components are already sitting inside the SWAR `packed_sum`
(see §8a / `prefetch` implementation) at bit positions [3:4], [8:9], [13:14].

**Step 1 — extract carry pairs into base-32 digits**

```cpp
// mask the six carry bits, right-shift by 3
// result layout: 0b 00xx 000 yy 000 zz  (three 2-bit fields, 3-bit gaps)
auto v = (packed_sum & VoxelOffsetT(0x6318u)) >> 3;
```

The 3-bit gaps are not accidental: the 5-bit SWAR groups naturally give a
**base-32 representation**.  With the `>> 3` shift, `v` is the 3-digit duotrigesimal
(base-32) number `0d cx·cy·cz`, where digit-k = the carry component for axis k.

**Step 2 — re-evaluate the same digits in base 3 via a single multiply**

```cpp
// 0d 1'3'9  =  1·32² + 3·32 + 9  =  1024 + 96 + 9  =  1129
auto dir_vec = (v * VoxelOffsetT(1129u)) >> 10;
// bits [10:14] of the product = digit-2 of v·(0d 1'3'9) = cz + 3·cy + 9·cx = dir
```

**Why digit-2 of the product equals `dir`:**

Base-32 long multiplication `(0d cx·cy·cz) × (0d 1·3·9)`:

| Digit of product | Contributions | Max value |
|---|---|---|
| 0 | 9·cz | 18 |
| 1 | 3·cz + 9·cy | 24 |
| **2** | **cz + 3·cy + 9·cx** | **26** |
| 3 | cy + 3·cx | 8 |
| 4 | cx | 2 |

Every digit sum is **< 32**, so **no carries propagate between base-32 digits**.
Digit 2 is therefore exact: it equals `cz + 3·cy + 9·cx = dir` with no contamination
from adjacent digits.  Digit-2 occupies bits [10:14] of the integer product, which is
why `>> 10` (and an optional `& 31`) extracts it.

**Overflow note:** `v` fits in `uint16_t` (max = 2 + 2·32 + 2·1024 = 2114), and
`v · 1129` reaches up to 2 386 706 — a 22-bit value that overflows `uint16_t`.
**No widening is required**, however: we extract bits [10:14] of the product, and those
bits sit entirely below bit 16.  Masking to 16 bits removes only bits 16+, leaving
bits [10:14] intact.  The `uint16_t` modular product gives the same result as the
full-width product for all valid and sentinel inputs.

**Compile-time sanity check** (all 27 valid inputs):

```cpp
for (int cx : {0,1,2}) for (int cy : {0,1,2}) for (int cz : {0,1,2}) {
    uint32_t v   = cz + 32*cy + 1024*cx;
    uint32_t dir = (v * 1129u) >> 10;
    assert(dir == unsigned(cz + 3*cy + 9*cx));
}
```

---

## 9. Relationship to Phase 1 Prototype

`ex_stencil_gather_cpu` implements the core cache machinery as free functions.

| Prototype component | `BatchAccessor` equivalent |
|--------------------|-----------------------------|
| `probedMask` + `ptrs[27]` locals | `mProbedMask` + `mLeafNeighbors[27]` members |
| `computeNeededDirs(expandedVec)` | per-lane loop inside `prefetch<di,dj,dk>` |
| `kSentinelExpanded` broadcast | sentinel applied by caller before `prefetch` |
| `probeLeaf` loop (`toProbe` bits) | `while (toProbe)` inside `prefetch` |
| `batchPtrs[4][SIMDw]` population | replaced by `cachedGetValue` |
| `verifyBatchPtrs` | future: `cachedGetValue` unit test |

---

## 10. Future Work

- **`cachedGetValue` vectorisation (Phase 2):** ingredient fetch (`mOffset`,
  `mPrefixSum[w]`, `valueMask().words()[w]`) is now fully SIMD via the gather chain
  in §8d.  Remaining: popcount `(maskWord & partial_mask)` → global value index →
  `gather_if(result, leafMask, globalValueArray, indices)` to replace the scalar
  `leaf->getValue(offset)` loop.  See `StencilGather.md §7b` for the AVX2 profile.

- **`getValue<di,dj,dk>`:** lazy combined `prefetch` + `cachedGetValue`.

- **Runtime `Coord` overload:** for generic stencil adapters iterating over an offset
  list at runtime.

- **`StencilAccessor`:** higher-level wrapper that owns the `while (any_of)` loop,
  hides straddling from the caller, and fills complete stencil result arrays.

- **Multi-leaf stencils (R > 4):** the single-neighbor-per-axis assumption in
  `cachedGetValue` holds for R ≤ 4.  Generalisation requires checking both lo and hi
  neighbors per axis.

- **C++20 structural `Coord`:** unify template and runtime interfaces via
  `cachedGetValue<nanovdb::Coord(-3,0,0)>(result, vo, leafMask)`.
