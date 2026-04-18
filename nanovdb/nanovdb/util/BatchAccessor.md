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
         typename PredicateT   = bool>
class BatchAccessor;
```

| Parameter | Scalar default | SIMD example | Role |
|-----------|---------------|--------------|------|
| `BuildT` | — | — | NanoVDB build type; determines `LeafT`, `TreeT` |
| `ValueT` | `float` | `Simd<float,W>` | Result type of `cachedGetValue` |
| `VoxelOffsetT` | `uint16_t` | `Simd<uint16_t,W>` | Compact 9-bit voxel offset within a leaf |
| `PredicateT` | `bool` | `SimdMask<uint32_t,W>` | Per-lane active predicate |

For `NanoGrid<ValueOnIndex>`, use `ValueT = uint64_t` (scalar) or
`ValueT = Simd<uint64_t,W>` (SIMD).

The scalar defaults allow instantiation without a SIMD library, giving a clean
scalar path for debugging and cross-validation.

Per-lane access is provided by `nanovdb::util::simd_traits<T>` (defined in `Simd.h`),
which works for both scalar and vector types via specialisation.

---

## 3. Persistent State

Members that persist across batches within one center leaf:

```cpp
const GridT&          mGrid;             // for probeLeaf calls via mGrid.tree()
uint32_t              mCenterLeafID;     // index of current center leaf
Coord                 mCenterOrigin;     // world-space origin of current center leaf
uint32_t              mProbedMask;       // bit 13 (center) pre-set at construction
uint32_t              mNeighborLeafIDs[27]; // kNullLeafID when outside narrow band or unprobed

const uint64_t* const mOffsetBase;       // &getFirstLeaf()[0].data()->mOffset
const uint64_t* const mPrefixBase;       // &getFirstLeaf()[0].data()->mPrefixSum
const uint64_t* const mMaskWordBase;     // getFirstLeaf()[0].valueMask().words()
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
`valueMask().words()[w]` for all active lanes via SIMD gathers (§8d–§8e).  The gather index
is `leaf_id × (sizeof(LeafT)/sizeof(uint64_t))`, computed as a `Simd<int64_t,W>` (see §8e).
Storing IDs enables a single flat-base gather over the contiguous leaf array; storing
pointers would require per-lane pointer arithmetic that doesn't map to `vpgatherqq`.
The `kNullLeafID` sentinel is masked out before any gather via `valid_u32` (§8e).

**Class-level base pointers:**  `mOffsetBase`, `mPrefixBase`, and `mMaskWordBase` are
`const` pointers computed once in the constructor from `getFirstLeaf()[0]`.  They are
invariant over the lifetime of the accessor (the leaf array is fixed after grid construction)
and are shared across all 18 `cachedGetValue` instantiations in a WENO5 gather, avoiding
the equivalent recomputation in every call.

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

- Fills **only the `leafMask` lanes** of `result` (by reference) via a 2-arg `where`
  directly on `result` — no intermediate copy, no write-back.
- `leafMask`-clear lanes are **not touched**: values from a previous iteration are
  preserved exactly as the caller left them.
- Additionally, lanes for which the tap voxel is inactive (outside the narrow band
  within an existing neighbor leaf) are also not written; `result` retains whatever
  default the caller initialised it to (typically 0 for a zero-initialized stencil
  buffer, matching `ValueOnIndex::getValue`'s return of 0 for inactive voxels).
- This contract suits the straddle-aware outer loop: the caller declares stencil
  result variables (zero-initialised) before the `while` loop, fills them
  progressively across iterations, and calls the kernel once after `activeMask` is empty.
- Requires the corresponding direction to be in `mProbedMask` (asserted in debug).
- `kNullLeafID` leaf (neighbor outside the narrow band entirely) also leaves `result`
  untouched, for the same reason: `maskWords = 0` → `isActive = false`.

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

**`cachedGetValue` — fully SIMD, no scalar loop**

`cachedGetValue` is fully vectorised end-to-end.  The scalar `leaf->getValue(offset)`
loop has been replaced by the gather chain described in §8e.  The result is written
directly to `result` via a 2-arg `where(isActive, result) = ...` — no intermediate
variable, no write-back copy.

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

### 8e. `cachedGetValue` gather pipeline — Steps 1–8

`cachedGetValue` recomputes `packed_sum` identically to `prefetch` (§8a), then runs
the following fully-SIMD pipeline.  All types are SIMD vectors of the indicated element
type; scalar `LaneWidth==1` degrades to plain scalar types.

```
Step 1 — d_vec  (Simd<uint16_t,W>)
    base-32 multiply trick (§8d): per-lane dir ∈ [0,26]

Step 2 — tapLeafID_u32  (Simd<uint32_t,W>)
    gather_if(tapLeafID_u32, leafMask, mNeighborLeafIDs, d_vec)
    valid_u32 = (tapLeafID_u32 != kNullLeafID)     ← effective mask for steps 3–5

Step 3 — tapLeafOffset_i64  (Simd<int64_t,W>)
    simd_cast_if<int64_t>(tapLeafOffset_i64, valid_u32, tapLeafID_u32)
    tapLeafOffset_i64 *= kStride          (kStride = sizeof(LeafT)/sizeof(uint64_t))

    Widening to int64_t is required: uint32_t * kStride overflows for large leaf
    pools (kNullLeafID = 0xFFFFFFFF).  simd_cast_if writes 0 for invalid lanes,
    keeping gather indices non-negative.  x86 vpgatherqq treats indices as signed
    int64_t, so negative values would access memory before the base pointer.

Step 4a — offsets  (Simd<uint64_t,W>)
    gather_if(offsets, valid_u32, mOffsetBase, tapLeafOffset_i64)
    → leaf->mOffset for each valid lane

Step 4b — prefixSums  (Simd<uint64_t,W>)
    gather_if(prefixSums, valid_u32, mPrefixBase, tapLeafOffset_i64)
    Extract field w from packed mPrefixSum:
        shift = (w > 0) ? (w-1)*9 : 0
        prefixSums = (w > 0) ? (prefixSums >> shift) & 511 : 0

    mPrefixSum packs 7 nine-bit prefix counts in one uint64_t:
        field w (1..7) at bits [9*(w-1) +: 9]; field 0 is defined as 0 (empty prefix).

Step 5 — maskWords  (Simd<uint64_t,W>)
    mask_idx = tapLeafOffset_i64 + simd_cast<int64_t>(wordIdx_u16)
    gather_if(maskWords, valid_u32, mMaskWordBase, mask_idx)
    → valueMask().words()[w] for each valid lane

    Heterogeneous mask: valid_u32 is SimdMask<uint32_t,W> applied to uint64_t data.
    Implemented via MaskElemT template parameter on gather_if in Simd.h.

Step 6 — dest_yz  (Simd<uint64_t,W>)
    dest_yz = ((packed_sum >> 2) & 0x38) | (packed_sum & 0x07)
    → ny_w*8 + nz_w  (6-bit intra-word bit position, range [0,63])

Step 7 — activity check + truncated maskWord
    voxelBit  = 1u64 << dest_yz
    isActive  = (maskWords & voxelBit) != 0
    truncated = maskWords & (voxelBit - 1)

    ValueOnIndex::getValue returns 0 for inactive voxels (bit not set in valueMask).
    Null-leaf lanes have maskWords=0, so isActive=false there too — no explicit
    valid_u32 guard is needed at this step.

Step 8 — fill result
    where(isActive, result) = offsets + prefixSums + popcount(truncated)

    2-arg where writes only active lanes; leafMask-clear and inactive-voxel lanes
    are untouched.
```

**popcount choice:** `popcount(Simd<uint64_t,W>)` uses a SWAR shift-and-add tree
(`popcount64` in `Simd.h`) rather than `__builtin_popcountll`.  AVX2 lacks a
64-bit lane-wise popcount (VPOPCNTQ is AVX-512DQ); `__builtin_popcountll` maps to
the scalar `popcnt` instruction, which is not vectorisable.  The SWAR tree uses only
`vpsrlq` / `vpand` / `vpaddq`, which are all AVX2-native.

### 8f. Assembly codegen — compiler × backend × ISA matrix

Flags: `-O3 -DNDEBUG -std=c++17 -fopenmp-simd -Wno-invalid-offsetof`.
ISA: `-mavx2` (base) or `-march=native` (i9-285K Arrow Lake, AVX2; no AVX-512).
Representative instantiation: `cachedGetValue<-3,0,0>` (x−3 tap, W=16), full Steps 1–8.

**Backend selection:** Simd.h auto-detects `<experimental/simd>` via `__has_include`.
`-DNANOVDB_USE_STD_SIMD` is redundant when the header is present.
Use `-DNANOVDB_NO_STD_SIMD` to force the array backend.

#### `cachedGetValue<-3,0,0>` — instruction counts

Numbers reflect the **unmasked-gather variant** (Steps 2/4a/4b changed to `gather`;
Step 5 `maskWords` kept as `gather_if`).  The `ymm`/`xmm`/`calls`/`vzup`/`vpins`
columns are from the original full measurement; `insns` and `vpgather` are
post-unmasked-gather.  `—` = not separately measured.

| Variant | ISA | insns | ymm | xmm | calls | vzup | vpgather | vpins |
|---------|-----|------:|----:|----:|------:|-----:|---------:|------:|
| GCC 13 + stdx  | avx2   |  579 | 393 | 100 |  14 |  13 |  0 |  8 |
| GCC 13 + array | avx2   | 1313 | 605 | 524 |   2 |   3 |  0 |  0 |
| Clang 18 + stdx  | avx2 |  828 | 530 | 470 |   1 |   2 |  0 | 62 |
| Clang 18 + array | avx2 | 1231 | 459 | 326 |   2 |   2 |  0 |  0 |
| GCC 13 + stdx  | native |  641 | 393 | 100 |  14 |  13 |  0 |  8 |
| GCC 13 + array | native | 1175 |  — |  — |   0 |   0 |  0 |  — |
| Clang 18 + stdx | native |  599 | 568 | 284 |   1 |   2 | **16** | 50 |
| Clang 18 + array | native | 1200 |  — |  — |   — |   — | **6** |  — |

`vpgather` breakdown (post-unmasked-gather):
- `clang18-stdx-native`: 4× `vpgatherdd` (Step 2: 16 lanes in 4×4) + 12× `vpgatherqq` (Steps 4a/4b/5: 4-wide ×4 chunks ×3) = 16 total
- `clang18-array-native`: 2× `vpgatherdd` + 4× `vpgatherqq` = 6 total

#### Before/after delta — unmasked-gather change

| Variant | ISA | insns before | insns after | Δ | vpgather before | vpgather after |
|---------|-----|------------:|------------:|--:|----------------:|---------------:|
| GCC 13 + stdx   | avx2   |  641 |  579 |  −62 | 0 |  0 |
| GCC 13 + array  | avx2   | 1320 | 1313 |   −7 | 0 |  0 |
| Clang 18 + stdx | avx2   |  795 |  828 |  +33 | 0 |  0 |
| Clang 18 + array| avx2   | 1365 | 1231 | −134 | 0 |  0 |
| GCC 13 + stdx   | native |  641 |  641 |    0 | 0 |  0 |
| GCC 13 + array  | native | 1365 | 1175 | −190 | 0 |  0 |
| Clang 18 + stdx | native |  600 |  599 |   −1 | 14 | 16 |
| Clang 18 + array| native | 1365 | 1200 | −165 |  0 |  6 |

The `clang18-stdx-avx2` regression (+33) is expected: the unmasked `gather` path
in the stdx backend emits a slightly different `where`-free code sequence that Clang
does not fold as aggressively as the original `gather_if`.  Total instruction count
is still lower than the array backend.

#### `-mavx2 -mtune=native` equivalence

On this machine (i9-285K Arrow Lake, no AVX-512), `-march=native` and
`-mavx2 -mtune=native` produce **identical hardware-gather emission** under Clang:

| Variant | flags | insns | vpgdd | vpgqq |
|---------|-------|------:|------:|------:|
| Clang 18 + stdx  | `-mavx2 -mtune=native` | 599 | 4 | 12 |
| Clang 18 + array | `-mavx2 -mtune=native` | 1219 | 2 |  4 |

The difference between `-mavx2` and `-march=native` is purely the **tuning model**,
not the ISA:
- `-mavx2`: targets `mtune=generic` — conservative gather cost model, no hardware gathers.
- `-march=native` (Clang): implies `mtune=sierraforest` — knows Arrow Lake's gather
  throughput, auto-vectorizer considers gathers profitable → emits `vpgatherqq`.
- `-march=native` (GCC): sets the ISA to sierraforest but keeps `mtune=generic` —
  same conservative behaviour as `-mavx2`.  No hardware gathers emitted by GCC even
  with `-march=native`.

GCC's stdx backend produces identical output (641 insns before / 579 after, 0 gathers)
for both `-mavx2` and `-march=native`.

#### `prefetch<-3,0,0>` — standalone vs inlined

| Variant | ISA | standalone symbol? | insns |
|---------|-----|--------------------|------:|
| GCC 13 + stdx   | any    | No — fully inlined | — |
| GCC 13 + array  | avx2   | Yes                | 260 |
| Clang 18 + stdx | any    | No — fully inlined | — |
| Clang 18 + array| avx2   | Yes                | 176 |

---

**Finding 1 — stdx backend is far superior to the array backend.**
The array backend is ≈2× larger in instruction count and degrades every `gather_if`
to a scalar lane-by-lane loop: 16 `vpextrw` to extract uint16_t direction indices, 16
conditional branches, 16 scalar uint32_t loads from `mNeighborLeafIDs`, then repeated
for each of the three uint64_t gathers (48 `vpextrq` total). In the stdx backends,
`gather_if` either maps to hardware gather instructions (Clang + native) or at worst
compact `vpinsrq` sequences (Clang + avx2). The 76 vpextr instructions (array backend)
vs 62 vpinsrb/q (stdx avx2) is telling: array is still scalar-inserting via extract,
not vectorised. The array backend also fails to inline `prefetch`.

**Finding 2 — Clang inlines all helpers; GCC emits 14 out-of-line weak stubs.**
GCC 13 emits `gather_if`, `simd_cast`, `simd_cast_if`, `where`, and `popcount` as
out-of-line COMDAT weak symbols and calls them. Each call requires `vzeroupper` on
entry (AVX ABI), yielding 13 transitions per `cachedGetValue` invocation. Clang 18
inlines all of them into a single function body except the final `popcount` call.

**Finding 3 — Hardware gathers require Clang + native tuning; unmasked gathers unlock the array backend too.**
After the unmasked-gather change, `clang18-stdx-native` emits **16** hardware gathers per `cachedGetValue`:
```
vpgatherdd  — 4× for the uint32_t tapLeafID gather   (Step 2:  4-wide × 4 = 16 lanes)
vpgatherqq  — 12× for the three uint64_t data gathers (Steps 4a/4b/5: 4-wide × 4 each)
```
`clang18-array-native` now emits **6** hardware gathers (2 vpgdd + 4 vpgqq) — the first
gathers ever seen in the array backend.  The unmasked `for (i) dst[i] = ptr[idx[i]]`
loop is the pattern Clang's auto-vectorizer converts to `vpgatherqq`; the `if (mask[i])`
conditional in `gather_if` defeated auto-vectorization for all mask types.

GCC 13 emits 0 hardware gathers even with `-march=native` — its stdx backend does not
exploit `vpgatherdd`/`vpgatherqq` for `experimental::simd` gather operations.  With
`-mavx2` alone, Clang also falls back to software gather (62 `vpinsrq/b`).

The 50 `vpinsrb` that remain in `clang18-stdx-native` are the mask-widening cost for
the one remaining heterogeneous `gather_if` (Step 5 `maskWords`): `SimdMask<uint32_t,16>`
is widened to four `SimdMask<uint64_t,4>` chunks to provide the sign-bit masks that
`vpgatherqq` expects.

**Finding 4 — `-march=native` gains nothing for GCC, in either backend.**
GCC's stdx backend produces identical output (641/579 insns, 0 gathers) for both
`-mavx2` and `-march=native`.  The array backend with `-march=native` (1175 insns,
0 gathers) also emits zero hardware gathers — even for the bare unmasked
`for (i) dst[i] = ptr[idx[i]]` loop that Clang converts to `vpgatherqq`.  GCC's
auto-vectorizer cost model treats gather instructions as unprofitable regardless of
tuning, preferring 40 `vpextrq` + 16 `vpinsrq` + 65 `vmovq` (scalar lane-by-lane)
instead.  This is a GCC backend policy, not a flag or mask-type issue.

**Finding 5 — Masking was the auto-vectorizer blocker for gathers.**
`gather_if` takes an `if (mask[i]) dst[i] = ptr[idx[i]]` shape — a conditional store.
This defeats Clang's gather auto-vectorizer for every mask element type tried (bool,
uint32_t, uint64_t).  The unmasked `gather` loop `dst[i] = ptr[idx[i]]` is the one
pattern that Clang + native tuning converts to `vpgatherqq`.  The sentinel invariant
makes the change safe: Step 2 uses `d ∈ [0,26]` (SWAR always valid); Steps 4a/4b use
`tapLeafOffset_i64 = 0` for invalid lanes (reading from base[0], the center leaf — safe
but unused); Step 5 is kept masked so that `maskWords = 0` for invalid lanes, ensuring
`isActive = false` without a cross-width mask AND.

**`popcount`** (out-of-line in all variants that reach it): 88 instructions, 85 ymm.
Fully vectorised with `vpsrlq`, `vpand`, `vpsubq`, `vpaddq`. Adding
`[[gnu::always_inline]]` to `util::popcount` in Simd.h eliminates the last remaining
out-of-line call in the Clang path and reduces GCC from 14 to 13 external calls.

**Action — `[[gnu::always_inline]]` on Simd.h helpers:**
Adding `[[gnu::always_inline]]` (or `__attribute__((always_inline))`) to `gather_if`,
`simd_cast`, `simd_cast_if`, `where`, and `popcount` in Simd.h eliminates all 13
`vzeroupper` transitions under GCC. Clang already inlines all but `popcount`; the
attribute is safe and a no-op for Clang.

**`popcount` alternative — `vpshufb`-based nibble popcount:**
The current SWAR shift-and-add tree (88 instructions, §8e) avoids the scalar `popcnt`
instruction because it is not vectorisable into `VPOPCNTQ` on AVX2.  There are two
other options worth considering:

*Scalar `popcnt` with extract/reassemble:*  `popcnt` is pipelined (Skylake+: 3-cycle
latency, 1/cycle throughput on port 1; 16 independent lanes retire in ~16 cycles).
The catch is the vector↔scalar domain crossing: extracting 16 uint64_t from 4 ymm
registers requires ~20 `vpextrq`/`vextracti128` instructions, and reassembly costs
another ~20 `vmovq`/`vpinsrq`/`vinserti128`.  Total ≈ 56 instructions — fewer than
SWAR, but the bypass latency penalty (~2 cycles per ymm→GPR crossing on Skylake)
reduces the advantage, and port 1 serialises all 16 `popcnt`s.

*`vpshufb`-based nibble popcount (recommended):*  Stays entirely in vector registers,
no domain crossing, and shrinks the body to ≈ 40 instructions:

```
lo   = v & 0x0F0F0F0F0F0F0F0F       (vpand)
hi   = (v >> 4) & 0x0F0F0F0F0F0F0F0F (vpsrlq + vpand)
bpop = vpshufb(lut, lo) + vpshufb(lut, hi)   (2× vpshufb + vpaddq)
sum  = vpsadbw(bpop, zero)            (horizontal byte-sum → 64-bit lane result)
```

`vpshufb` and `vpsadbw` use ports 0/5 and port 5 respectively — orthogonal to the
arithmetic-heavy SWAR ports — so the `vpshufb` path is also more friendly to
out-of-order overlap with surrounding code.  This is the standard compiler-generated
AVX2 popcount pattern and the likely replacement for `popcount64` in `Simd.h`.

### 8g. Cycle budget and architectural comparison

#### `cachedGetValue` critical path (Clang 18 + stdx + `-march=native`, W=16)

| Step | Work | Cumulative cycles |
|------|------|------------------:|
| 1 | SWAR expansion + base-32 multiply → `d_vec` | ~8 |
| 2 | 4× `vpgatherdd` → `tapLeafID_u32` | ~20 |
| 3 | `simd_cast_if` + ×kStride → `tapLeafOffset_i64` | ~25 |
| 4a/4b/5 | 4+4+4 `vpgatherqq` (3 independent groups, overlap in OoO) | ~41 |
| 6–8 | bitwise `dest_yz`, `maskWords & voxelBit`, popcount SWAR + `where` | **~55** |

Critical path per call: **~55 cycles** (gather-chain limited; Steps 4a/4b/5 are the
deepest dependency).

Single-core throughput reality: each call is ~600 instructions.  Arrow Lake's ROB
(~500 entries) holds less than one full call, so call-to-call OoO overlap is minimal.
Realistic single-core cost is **~80–100 cy/call**, not the ~7 cy/call that perfect 8×
OoO would imply.  For 128 elements × 18 taps = 144 calls: **~12,000–14,000 cycles
single-threaded**, or **~100 cy/element**.

#### Comparison with scalar NanoVDB `getValue(ijk)`

Naive alternative: 128 voxels × 18 taps = 2304 scalar `ReadAccessor::getValue()` calls.

| Accessor L0 cache behaviour | cy/call | 2304 calls | cy/element |
|-----------------------------|--------:|-----------:|-----------:|
| Hit (same leaf as last call) | ~22 | ~51,000 | ~400 |
| Miss, tree nodes L1-warm | ~52 | ~120,000 | ~940 |
| Miss, tree nodes cold | ~100+ | ~230,000 | ~1800 |

**BatchAccessor speedup: 4–10× depending on hit rate.**

The two sources of gain:

1. **Amortised tree traversal (dominant).** `prefetch` calls `probeLeaf` at most once
   per direction per center-leaf switch — **12 calls** for a 128-element block (6
   directions × 2 center-leaf switches) vs. up to 2304 traversals for the scalar path.
   Each saved traversal is ~25–35 cycles of pointer-chasing through root → internal →
   internal → leaf with warm L1 nodes.

2. **SIMD × 16.** The SWAR expansion, gather chain, and popcount all execute once for
   16 lanes simultaneously.  Even if the scalar accessor hit perfectly on every call,
   the SIMD path still wins by ~4× on arithmetic work alone.

The scalar hit rate depends on loop ordering.  Processing all 18 taps for one voxel
before moving to the next evicts the cached leaf on nearly every tap switch (high miss
rate).  Sweeping all 128 voxels for one tap at a time improves hit rate, but requires
18 passes over the voxel array and hurts reuse of stencil results.

#### CPU vs GPU: why the same operation inverts

On CPU (8 P-cores), the 128-element block is **compute-bound**:

- Index computation: ~12,000 cy per core
- Value fetch (512 unique floats, 32 cache lines, 8 cores competing for DDR5-5600):
  ~80–664 cycles depending on cache level and core count
- System DRAM bandwidth consumed at full parallelism: ~4.6 GB/s out of 89 GB/s
  available (~5% utilisation)

The gather chain latency is the bottleneck; bandwidth sits largely idle.  The CPU
BatchAccessor design (SIMD W=16, hardware `vpgatherqq`) directly attacks this by
compressing 16 serial gather chains into one parallel 55-cycle critical path.

On GPU the same operation becomes **bandwidth-bound**:

- An SM has hundreds of warps in flight.  When a warp stalls on a gather or arithmetic
  latency (~20–100+ cycles), the scheduler switches to another ready warp instantly.
  The entire index computation — SWAR, base-32 multiply, all gather latencies — is
  absorbed by warp switching.  Effective compute cost per thread: ~0 stall cycles.
- What remains visible to the GPU is the **global memory traffic**: fetching stencil
  float values.  With hundreds of SMs each issuing many transactions simultaneously,
  HBM bandwidth saturates quickly.
- GPU gathers are scalar-per-thread: 32 threads in a warp each doing an 8-byte load =
  32 independent transactions.  Non-contiguous addresses (stencil neighbours across
  leaves) yield uncoalesced access, amplifying bandwidth pressure.

Consequently, GPU optimisation for this workload targets **coalescing** (adjacent
threads access adjacent values) and **cache footprint** (keeping the neighbour-leaf
working set in L1/shared memory), rather than the gather-chain depth that dominates
on CPU.

### 8h. End-to-end perf: outlining, `[[gnu::flatten]]`, and W=8

§8f measured `cachedGetValue` as a standalone symbol.  This section measures the
**full WENO5 pipeline end-to-end** — `StencilAccessor::moveTo` driving 18 taps ×
128 voxels/block × 131072 blocks across 32 TBB threads — and reveals a much
larger GCC pathology that a single-function measurement cannot see.

Workload: `ex_stencil_gather_cpu 33554432 0.5` (16 M active voxels, 50% occupancy,
i9-285K Arrow Lake, 32 threads, `-O3 -march=native`).  Time is wall clock via
`nanovdb::util::Timer`; checksum-matches `LegacyStencilAccessor` in every run.

#### End-to-end latency (ns/voxel, smaller is better)

| Variant                              | GCC 13 | Clang 18 |
|--------------------------------------|-------:|---------:|
| No `flatten`                         |  7.5   |  4.3     |
| `flatten` on `BatchAccessor::{prefetch,cachedGetValue}` | 4.9 | 4.3 |
| `flatten` on `StencilAccessor::moveTo` (full transitive) | **3.7** | 4.3 |
| `LegacyStencilAccessor` reference    |  5.4   |  6.7     |

Without `flatten`, GCC's SIMD `StencilAccessor` is **39% slower than the scalar
`ReadAccessor`-based `LegacyStencilAccessor`** — the SIMD abstraction turns into a
net loss.  With `[[gnu::flatten]]` on `moveTo`, GCC becomes 33% faster than scalar
and edges out Clang.

#### Per-batch call accounting (GCC, W=16)

`moveTo` processes 16 voxels per batch.  Per-batch call count is the product of:

| Call site                           | No flatten | moveTo flatten |
|-------------------------------------|-----------:|---------------:|
| `moveTo` → `prefetchHull`, `calcTaps` |  3        |  inlined       |
| `prefetchHull` internals              | 12         |  inlined       |
| `calcTaps` → 18× `cachedGetValue` + 18× `WhereExpression::op=` | 37 | inlined |
| Inside each `cachedGetValue`: 14 outlined Simd.h helpers × 18 | 252 | inlined |
| Stack-canary / misc                   | 19         |  0             |
| **Total calls per batch**             | **~323**   | **0**          |
| **Total `vzeroupper` per batch**      | **~282**   | **1** (epilogue) |

At 16 voxels/batch, that is **~18 `vzeroupper` per voxel** without flatten.  Each
VZU is cheap (~1–2 cycles) but serves as a strong ABI barrier that defeats the
out-of-order engine's ability to overlap pre- and post-call work.  Combined with
the per-call argument marshaling of `_Fixed<16>` aggregates (128 B by reference),
the accumulated cost is the full 3.2 ns/voxel gap between the two variants.

#### Why outlining happens under GCC

Each Simd.h helper (`gather`, `gather_if`, `simd_cast`, `simd_cast_if`, `where`,
`popcount`, `WhereExpression::op=`) is an `inline` template.  With `-O3`, GCC's
inliner decides each is "too expensive to inline" once the caller
(`cachedGetValue`, ~900 B) reaches a growth-budget threshold.  It emits each
helper as a weak COMDAT and calls it.  Every such call takes `_Fixed<16>`
aggregates by reference (the parameter doesn't fit in YMM), triggering
`vzeroupper` on entry.

The same pattern propagates up: `calcTaps` (after inlining) is too big to accept
18 copies of `cachedGetValue`, so GCC outlines those too — one weak symbol per
template instantiation.  Then `StencilAccessor::moveTo` calls `calcTaps` and
`prefetchHull` across that same boundary.

Clang's inliner makes different decisions — it inlines the Simd.h helpers into
each `cachedGetValue`, keeps `cachedGetValue` outlined per-tap, and accepts the
18 calls from `calcTaps`.  Clang also emits hardware gathers under `-march=native`
(16 `vpgather` per tap, see §8f), amortising the per-call cost with faster
gather semantics.

#### Why `[[gnu::flatten]]` on `moveTo` wins

`__attribute__((flatten))` forces **every call** in the annotated function's body
to be inlined, recursively — overriding all cost heuristics.  Applied to
`StencilAccessor::moveTo`, it collapses the entire call tree (`prefetchHull`,
`calcTaps`, 18× `cachedGetValue`, 14× helpers per tap) into one monolithic
inlined body.  Observed: **0 calls, 1 `vzeroupper` (function epilogue only),
14 350 insns, 77 KB of text in a single symbol**.

Trade-offs:

- Binary size: one 77 KB function per `StencilAccessor` instantiation.  L1i is
  32 KB, but the per-batch hot path only sweeps a small fraction of the body
  linearly, so I-cache pressure is manageable.
- Debuggability: one giant symbol to step through vs 40+ small symbols.
- Compile time: GCC spends notably longer compiling a flattened `moveTo`.

#### Why `flatten` on `BatchAccessor::prefetch`/`cachedGetValue` alone is insufficient

Flattening at the BatchAccessor level inlines the 14 Simd.h helpers into each
`cachedGetValue`/`prefetch` body (so each of those becomes a clean, self-contained
~800-insn function with ≤2 residual calls — typically `WhereExpression::op=` and
the `_S_generator` stdx lambda for `popcount`).  However it leaves the 18
`cachedGetValue` call sites *themselves* outlined — `calcTaps` still pays 38
calls and 26 `vzeroupper` per batch.  Measured: 4.9 ns/voxel — halfway between
no-flatten and full-flatten.

The signal is clear: the *outer* `moveTo` → `calcTaps` → per-tap call boundary
is the dominant cost, not the inner helper-call boundary.

#### W=8 experiment (batch-width halving)

Motivation: halving the batch width reduces register pressure and spill volume,
and shifts some types from `_Fixed<W>` to `_VecBuiltin<32>` (the native
`__m256i` ABI).  Specifically at W=8:

- `Simd<uint16_t, 8>`   — 16 B, `_VecBuiltin<16>` (native XMM)
- `Simd<uint32_t, 8>`   — 32 B, `_VecBuiltin<32>` (native YMM) ✓ register-passable
- `Simd<uint64_t, 8>`   — 64 B, still `_Fixed<8>` (2× YMM aggregate, not passable)
- `Simd<int64_t, 8>`    — same as uint64

Only the `uint32_t` leaf-ID/mask vectors become register-passable; the dominant
`uint64_t` index vectors are still aggregate (half the size of the W=16
aggregate, but still stack-passed).

Measured at W=8 with full flatten:

| Metric                  | W=16    | W=8     | Δ       |
|-------------------------|--------:|--------:|--------:|
| `moveTo` text size      | 77 KB   | 34 KB   | −56%    |
| `moveTo` insns          | 14,349  | 7,182   | −50%    |
| YMM spill stores        |   469   |    67   | **−86%**|
| YMM spill loads         |   351   |   167   | −52%    |
| vpinsrq (software-gather glue) | 432 | 216 | −50%    |
| `vpgather*`             | 0       | 0       | unchanged |
| `vzeroupper`            | 1       | 1       | unchanged |
| **End-to-end (GCC)**    | **3.7 ns/vox** | 4.2 ns/vox | +0.5 |
| **End-to-end (Clang)**  | 4.3 ns/vox | 4.0 ns/vox | −0.3 |

W=8 dramatically reduces register pressure (the spill count is 86% lower).  But
GCC's end-to-end time regresses by 0.5 ns/voxel because the per-batch framing
cost (`zeroIndices<SIZE>`, `leafSlice == centerLeafID` mask compute, straddling
loop control, `prefetchHull`) is now amortised across only 8 lanes instead of
16.  The body of `moveTo` halved; the surrounding scaffolding doubled.

Clang benefits slightly (−0.3 ns/voxel), likely because its outlined
`cachedGetValue` was paying more call-frame marshaling at W=16 (4× YMM aggregate
vs 2× YMM at W=8).

**Takeaway for future design**: W=8 would become attractive if the per-batch
framing work can be amortised across multiple adjacent batches — for example,
hoisting `prefetchHull` outside the batch loop for cases where the hull mask
is invariant across several batches of the same center-leaf.

#### Findings

**F1 — GCC's default codegen for this abstraction is broken.**  Without
`flatten` or equivalent attributes, GCC emits ~323 calls / ~282 `vzeroupper`
per 16-voxel batch, making the SIMD `StencilAccessor` *slower* than the scalar
`LegacyStencilAccessor`.

**F2 — `[[gnu::flatten]]` on `StencilAccessor::moveTo` restores performance.**
One attribute, targeting the WENO5 pipeline entry point, drops GCC from 7.5 to
3.7 ns/voxel (2×) and makes GCC the fastest of the measured configurations.

**F3 — Partial flattening at `BatchAccessor::{prefetch,cachedGetValue}` is not
enough.**  The inner helper calls are eliminated but the 18 `cachedGetValue`
call sites themselves remain — 4.9 ns/voxel.

**F4 — Hardware gathers are not needed on Arrow Lake.**  GCC emits 0 `vpgather`
in all variants; Clang+native emits 16 per `cachedGetValue`.  GCC's
software-gather path (scalar loads + `vpinsrq`) nevertheless beats Clang's
hardware-gather path end-to-end (3.7 vs 4.3 ns/voxel) because the three load
ports issue the scalar gathers in parallel and the out-of-order engine hides
the latency.  §8f Finding 5 (unmasked-gather auto-vectorisation) remains
correct; it is simply not load-bearing on this microarchitecture.

**F5 — W=8 reduces spills dramatically but does not help end-to-end on GCC.**
Per-batch framing cost dominates at smaller widths.

**F6 — Clang's performance is relatively insensitive to these knobs.**
Clang inlines the Simd.h helpers regardless of `flatten`, and its outlined
`cachedGetValue` pays only moderate call overhead.  Both 4.0–4.3 ns/voxel
across all variants tested.

**Not applied.**  The codebase does not ship `[[gnu::flatten]]` by default.
StencilAccessor-style callers that require peak GCC performance may apply it
to their own hot entry point; the attribute is safe and a no-op under Clang.
This choice keeps the library's default codegen predictable and avoids forcing
a 77 KB monolithic body on callers with smaller working sets.

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

## 10. Status and Future Work

### Completed

- `prefetch<di,dj,dk>`: fully SIMD crossing detection, lazy probeLeaf.
- `cachedGetValue<di,dj,dk>`: fully SIMD end-to-end (Steps 1–8, §8e).
  Verified against scalar reference over 12M lane-checks across all 18 WENO5 taps.
- Class-level base pointers (`mOffsetBase`, `mPrefixBase`, `mMaskWordBase`).
- `simd_cast_if`, heterogeneous `gather_if`, `popcount64`/`popcount` added to `Simd.h`.
- Simd.h array-backend `Simd(const T*, element_aligned_tag)` load constructor:
  removed default argument for the tag to eliminate the `Simd(0)` null-pointer-constant
  ambiguity that breaks compilation under `-DNANOVDB_NO_STD_SIMD`.
- Full 7-variant codegen analysis (compiler × backend × ISA, §8f), including
  before/after delta for the unmasked-gather change and `-mavx2 -mtune=native`
  equivalence finding.
- **Unmasked gather (Steps 2/4a/4b):** `gather_if` replaced with `gather` using the
  sentinel invariant (d ∈ [0,26]; invalid lanes read base[0]).  Step 5 kept masked so
  `maskWords=0` for invalid lanes → `isActive=false` without cross-width mask AND.
  Verified: 12M lane-checks pass across all 18 WENO5 taps.  Unlocks hardware
  `vpgatherqq` in the array backend under Clang + native tuning.
- **End-to-end codegen analysis (§8h)**: measured the full WENO5 pipeline
  (`StencilAccessor::moveTo` × 131 K blocks × 32 threads) on i9-285K Arrow Lake.
  Established that GCC's default `-O3` outlines 14 Simd.h helpers per
  `cachedGetValue` and outlines `cachedGetValue`/`WhereExpression::op=` per tap,
  producing ~282 `vzeroupper` per 16-voxel batch and making the SIMD path
  slower than scalar `LegacyStencilAccessor`.  `[[gnu::flatten]]` on
  `StencilAccessor::moveTo` collapses the full call tree and drops GCC from
  7.5 to 3.7 ns/voxel (2×), beating Clang's 4.3 ns/voxel.  W=8 cuts spills by
  86% but regresses GCC end-to-end due to per-batch framing overhead.
  Attributes **not applied** in the shipped code; see §8h "Not applied" note.

### Remaining

- **`[[gnu::always_inline]]` on `Simd.h` helpers** (§8f) vs
  **`[[gnu::flatten]]` on StencilAccessor-style entry points** (§8h):
  two candidate approaches to restore GCC inlining.  The flatten path was
  measured end-to-end (2× speedup); the always_inline path was measured only
  on the standalone `cachedGetValue` symbol.  Decide which to ship once a
  consumer of StencilAccessor exists in the production build.

- **`vpshufb`-based `popcount` in `Simd.h`:** replace `popcount64` SWAR tree with
  nibble-LUT + `vpsadbw` pattern (§8f); reduces the out-of-line body from 88 to ≈40
  instructions and uses orthogonal execution ports.

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
