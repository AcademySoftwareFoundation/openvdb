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
const GridT& mGrid;                          // for probeLeaf calls via mGrid.tree()
uint32_t     mCenterLeafID;                  // index of current center leaf
Coord        mCenterOrigin;                  // world-space origin of current center leaf
uint32_t     mProbedMask = (1u << 13);       // bit 13 (center) pre-set at construction
const LeafT* mLeafNeighbors[27];             // [13] = center (eager); others: lazily probed
```

**Direction encoding** (`dir` is a `static constexpr` member):

```
dir(dx, dy, dz) = (dx+1)*9 + (dy+1)*3 + (dz+1)     dx,dy,dz ∈ {-1,0,+1}
```

`mLeafNeighbors[27]` is a flat array indexed by `dir(dx,dy,dz)`.
`mLeafNeighbors[13]` (= `dir(0,0,0)`) is the center leaf pointer.
`mLeafNeighbors[d]` is `nullptr` when the neighbor leaf lies outside the narrow band.

**Why pointers, not leaf IDs:**  `cachedGetValue` accesses the leaf data array for
every active lane in every batch.  Storing `const LeafT*` avoids a `base + id *
sizeof(LeafT)` multiply on every call; `nullptr` is a natural "outside narrow band"
sentinel.  `NanoVDB::ReadAccessor` uses the same approach for its cached node pointers.

**Cache advance:** when `none_of(leafMask)` fires in the outer loop:

```cpp
void advance(uint32_t newLeafID) {
    mCenterLeafID              = newLeafID;
    mCenterOrigin              = mGrid.tree().getFirstLeaf()[newLeafID].origin();
    mLeafNeighbors[dir(0,0,0)] = &mGrid.tree().getFirstLeaf()[newLeafID];
    mProbedMask                = (1u << dir(0,0,0));  // center pre-set; neighbors stale
}
```

Stale neighbor entries in `mLeafNeighbors[]` are harmless: `mProbedMask` has only
bit 13 set, so `toProbe = neededMask & ~mProbedMask` will never return a stale index.

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
mLeafNeighbors[dir(0,0,0)] = &mGrid.tree().getFirstLeaf()[mCenterLeafID];
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

### 8a. Lane loop in prefetch / cachedGetValue

The current implementation uses a scalar `for (int i = 0; i < LaneWidth; ++i)` loop
over lanes, using `simd_traits<T>::get` / `set` for per-lane access.  This is correct
for both scalar (LaneWidth=1) and SIMD (LaneWidth=W) instantiations.

`prefetch` is called at most once per direction per center leaf, so the loop is not
performance-critical.  `cachedGetValue` is in the hot path; the loop over W=16 lanes
with scalar per-lane `leaf->getValue(offset)` is a first correct implementation.
Vectorising this loop (SIMD offset arithmetic + `vgatherdps`) is the Phase 2
optimisation task described in `StencilGather.md §7b`.

### 8b. No tree accessor in prefetch

NanoVDB's `ReadAccessor` is not passed to `prefetch`.  Its LEVEL=0 leaf cache is never
warm for neighbor leaves (by definition distinct from the center leaf), and its
internal-node caches are bypassed entirely when `get<GetLeaf>` misses at LEVEL=0.
`probeLeaf` is equivalent to a direct root traversal in all non-trivial cases.

### 8c. probeLeaf returns nullptr for missing neighbors

`mGrid.tree().probeLeaf(coord)` returns `nullptr` when the requested coordinate lies
outside the active narrow band.  `cachedGetValue` checks for `nullptr` and returns
`ScalarValueT(0)`, which is correct for level-set grids (background value = 0).

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

- **`cachedGetValue` vectorisation (Phase 2):** replace per-lane scalar loop with SIMD
  offset arithmetic + `vgatherdps` × 2 + `vpblendvb` for the two-pointer case.
  See `StencilGather.md §7b` for the AVX2 profile.

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
