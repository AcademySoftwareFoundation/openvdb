# StencilAccessor — Design Plan

Higher-level wrapper around `BatchAccessor` that owns the straddling loop,
fills complete stencil result arrays, and presents a clean per-block API to
the WENO (or other stencil) kernel.

---

## 1. Purpose

`StencilAccessor` wraps `BatchAccessor` and owns the full stencil evaluation
for one SIMD-wide batch of voxels.  Its output is a fixed-size array of
`Simd<uint64_t, W>` — one vector per tap — containing the ValueOnIndex indices
for all W lanes simultaneously.  The caller uses these indices to fetch sidecar
data (floats, etc.) independently; no value arrays are read here.

```
input:  W voxel offsets  +  W-wide active mask  +  center-leaf context
output: Simd<uint64_t,W>  ×  N_taps
```

This separates index gathering (StencilAccessor) from value fetching (caller),
which is the right split: index gathering is the expensive irregular part;
value fetching is a straight gather from a dense sidecar array that the caller
can pipeline, prefetch, or vectorise independently.

---

## 2. Relationship to BatchAccessor

`BatchAccessor::cachedGetValue<di,dj,dk>` produces one `Simd<uint64_t,W>` for
one tap.  `StencilAccessor` calls it for every tap in the stencil and assembles
the result array.  It also owns:

- calling `prefetch<di,dj,dk>` for every direction that the batch may cross into
- the **straddling loop**: the `while (any_of(leafMask))` structure that handles
  lanes whose center leaf differs from the majority and must be processed
  separately before rejoining the batch

---

## 3. Why StencilAccessor must own the stencil — the cache invariant

`BatchAccessor`'s neighbor cache (`mNeighborLeafIDs[27]`, `mProbedMask`) is valid
only for the current center leaf.  Advancing the center leaf invalidates the cache.

This creates a hard ordering constraint: **all taps must be computed for a given
center leaf before the center leaf advances.**

If the caller drove the tap loop and called `cachedGetValue<di,dj,dk>` one tap at
a time, it could inadvertently interleave taps across a center-leaf transition,
producing silently wrong results (stale neighbor IDs).

`StencilAccessor` avoids this by:
1. Holding the complete tap list at compile time.
2. Owning the center-leaf advancement loop.
3. For each center leaf: calling `prefetch` for all needed directions, then
   `cachedGetValue` for all taps, before advancing.

The straddling case makes this constraint sharper: when some lanes cross into a new
center leaf mid-batch, `StencilAccessor` peels those lanes off, runs the **full
stencil** for the new center leaf on the peeled subset, then recombines — all before
yielding the complete result array to the caller.  This is only possible because
the full tap list is known upfront.

---

## 4. Compile-time stencil description — `StencilT`

The stencil is encoded in a `StencilT` policy class passed as a template argument
to `StencilAccessor`.  It carries two compile-time sets:

### 4a. Tap set

An ordered, sized list of `(di, dj, dk)` offsets.  `SIZE` determines the number
of output `Simd<uint64_t,W>` vectors; the index of each tap in the list is its
slot in the output array, so the caller knows which slot corresponds to which offset.

### 4b. Prefetch hull

A list of **actual tap offsets** — not normalized `{-1,0,1}³` leaf directions —
that `StencilAccessor` calls `prefetch<di,dj,dk>` on before evaluating any tap.

The hull is the **minimal set of extreme taps** such that prefetching them
guarantees every `cachedGetValue` call for every stencil tap will find its
neighbor leaf already cached.

**Why extreme taps suffice — the monotonicity argument:**

`prefetch<di,dj,dk>` computes, for each lane, which neighbor-leaf direction it
crosses into (encoded as the carry triple `(cx,cy,cz) ∈ {under,in,over}³` from
the SWAR expansion).  A crossing in the −x direction occurs when `x + di < 0`,
i.e., when `x < |di|`.  For a more extreme tap `hi` with `|hi| ≥ |di|` and the
same sign, `x < |di| ⟹ x < |hi|` — so any lane that the intermediate tap would
cause to cross is **also** detected by the extreme tap.  The converse is not true
(the extreme tap may probe a neighbor that the intermediate tap would not reach),
but that is safe: a conservative probe wastes at most one `probeLeaf` call with
no correctness impact.

**WENO5 (axis-aligned taps, radius 3):**
Lanes can never simultaneously cross two axis boundaries for a single tap, so
edge and corner leaf neighbors are unreachable.  The 6 axis-extremal taps are
sufficient:

```
hull = { {-3,0,0}, {3,0,0},
         {0,-3,0}, {0,3,0},
         {0,0,-3}, {0,0,3} }
```

**3×3×3 box stencil (includes diagonal taps):**
A lane at `(x=0, y=0, z=0)` with tap `(-1,-1,-1)` crosses all three axes at
once, reaching the `(-1,-1,-1)` corner leaf neighbor.  The 8 corner taps
`{(±1,±1,±1)}` form the hull: each corner tap, across all lane positions,
generates crossings in every combination of axes within its sign octant,
covering all 26 neighbor directions (faces, edges, and corners).

**General rule:** the hull = the **sign-octant convex hull vertices** of the
tap set.  For axis-aligned stencils these are the axis extremes; for stencils
with diagonal taps these are the corners of the tap set's bounding box in each
octant.

The hull is **provided explicitly** rather than derived automatically — it is a
one-time design-time decision per stencil type, and it avoids compile-time logic
that would need to reason about leaf size vs. tap radius.

### 4c. Sketch of `StencilT` concept

```cpp
// WENO5 3D stencil: 18 axis-aligned taps, radius 3, hull = 6 extremal taps
struct Weno5Stencil {
    static constexpr int SIZE = 18;

    // ordered tap list: output slot i ↔ taps[i]
    static constexpr nanovdb::Coord taps[SIZE] = {
        {-3,0,0}, {-2,0,0}, {-1,0,0}, {1,0,0}, {2,0,0}, {3,0,0},
        {0,-3,0}, {0,-2,0}, {0,-1,0}, {0,1,0}, {0,2,0}, {0,3,0},
        {0,0,-3}, {0,0,-2}, {0,0,-1}, {0,0,1}, {0,0,2}, {0,0,3},
    };

    // prefetch hull: 6 extremal taps cover all 18
    static constexpr int HULL_SIZE = 6;
    static constexpr nanovdb::Coord hull[HULL_SIZE] = {
        {-3,0,0}, {3,0,0},
        {0,-3,0}, {0,3,0},
        {0,0,-3}, {0,0,3},
    };
};
```

The exact representation (constexpr arrays, parameter packs, index sequences) is
to be refined.  The conceptual contract is fixed: `StencilT` exposes `SIZE`,
an indexed tap list, `HULL_SIZE`, and an indexed hull list — all at compile time.

---

## 5. Template parameters and type aliases

```cpp
template<typename BuildT, int W, typename StencilT>
class StencilAccessor {

    // Scalar/SIMD split — explicit conditional, not Simd<T,1> degeneracy.
    // Matches the convention BatchAccessor already uses for its own template params.
    using IndexVec  = std::conditional_t<W == 1, uint64_t,  Simd<uint64_t, W>>;
    using OffsetVec = std::conditional_t<W == 1, uint16_t,  Simd<uint16_t, W>>;
    using LeafIdVec = std::conditional_t<W == 1, uint32_t,  Simd<uint32_t, W>>;

    // Two distinct mask types — they differ in element width and in role:
    //
    //   LeafMaskVec  — mask over leafIndex[] (uint32_t) comparisons.
    //                  Used internally in the straddling loop and passed to
    //                  BatchAccessor::prefetch / cachedGetValue.
    //
    //   IndexMaskVec — mask over mIndices[] (uint64_t) values.
    //                  Returned by moveTo so the caller can gate reads from
    //                  Simd<uint64_t,W> stencil result vectors.
    //
    // In the underlying bitmask representation both are W-bit masks; the type
    // distinction exists for semantic correctness when blending or gating on
    // 64-bit vs 32-bit SIMD data.  A widening reinterpret is needed when
    // converting the initial LeafMaskVec activeMask to the IndexMaskVec return.
    using LeafMaskVec  = std::conditional_t<W == 1, bool, SimdMask<uint32_t, W>>;
    using IndexMaskVec = std::conditional_t<W == 1, bool, SimdMask<uint64_t, W>>;

    // BatchAccessor is parameterised with LeafMaskVec because prefetch() and
    // cachedGetValue() operate in the leaf-ID (uint32_t) domain.
    using BatchAcc  = std::conditional_t<W == 1,
                          BatchAccessor<BuildT, uint64_t, uint16_t, bool>,
                          BatchAccessor<BuildT, IndexVec, OffsetVec, LeafMaskVec>>;

    static constexpr int SIZE      = std::tuple_size_v<typename StencilT::Taps>;
    static constexpr int HULL_SIZE = std::tuple_size_v<typename StencilT::Hull>;
};
```

W=1 gives a fully scalar `BatchAccessor` underneath with plain scalar `mIndices` —
a clean debug and cross-validation path identical in logic to the SIMD path.

---

## 6. Internal state

```cpp
BatchAcc  mBatch;        // owns neighbor-leaf cache, mCenterLeafID, and cachedGetValue
IndexVec  mIndices[SIZE]; // one SIMD vector (or scalar) per tap — output store
```

**`mBatch`** — the embedded `BatchAccessor`.  It is the **single source of truth**
for the current center leaf ID.  `BatchAccessor` exposes a `centerLeafID()` getter
so `StencilAccessor::moveTo` can read it for the `leafSlice == currentLeafID`
comparison without maintaining a redundant copy.  `StencilAccessor` drives
advancement by calling `mBatch.advance(newLeafID)`.

`StencilAccessor` has **no separate `mCurrentLeafID` member** — having both
`mBatch.mCenterLeafID` and a local copy would be redundant state that can get
out of sync.

**`mIndices`** — accumulation buffer filled by `moveTo`.  At the **top of each
`moveTo` call**, all `SIZE` vectors are zeroed.  Index 0 is the NanoVDB
IndexGrid "not found / background" sentinel, so inactive lanes (those not set
in the returned `IndexMaskVec`) yield a well-defined background index rather
than stale data.  Active lanes are then written by the straddling loop via
`where`-blend; in the straddling case the blend ensures majority-leaf results
are not overwritten when minority-leaf lanes are processed.

**Stack footprint:** for WENO5, W=16: 18 × 16 × 8 bytes = **2.25 KB**.
Acceptable for a stack-local object within a VBM block kernel; would need care
if embedded in a larger persistent structure.

---

## 7. Construction and leaf-ID monotonicity

```cpp
StencilAccessor(const GridT& grid, uint32_t firstLeafID, uint32_t nExtraLeaves)
    : mBatch(grid, firstLeafID)
#ifndef NDEBUG
    , mNExtraLeaves(nExtraLeaves)
#endif
{}
```

Constructed once per VBM block.  `firstLeafID = vbmHandle.hostFirstLeafID()[blockID]`
is the correct starting center leaf — the VBM block begins there by definition.

`nExtraLeaves` is the number of distinct center-leaf advances the straddling loop
may make across the entire block (computed from the jumpMap by the caller).  It is
used only as a debug-mode assert bound; it is not needed for correctness.  Once the
implementation is vetted, remove the `#ifndef NDEBUG` member, the assert in `moveTo`,
and the constructor parameter — four targeted deletions with no restructuring.

**Leaf-ID monotonicity invariant:**  The VBM assigns leaf IDs in Morton order.
Within a block, `leafIndex[0..BlockWidth-1]` is **non-decreasing**: as the voxel
index advances, the leaf IDs can only stay the same or increase — never decrease.

This invariant is load-bearing for the straddling loop:

- `advance(centerLeafID() + 1)` is always correct: once all lanes for leaf N are
  consumed from the current batch, no future batch will ever contain a lane for
  leaf N.  A simple increment is sufficient; no backward search is needed.
- The `while (any_of(activeMask))` loop is guaranteed to terminate: each iteration
  either removes lanes from `activeMask` (progress toward `none_of`) or increments
  the center leaf (progress toward the end of the block).  At most `nLeaves`
  center-leaf advances occur per batch; typically zero or one.
- The `BatchAccessor` neighbor cache is never invalidated "in reverse" — its
  monotonic advance matches the monotonic leaf-ID layout.

The instance persists for the entire block (across all `moveTo` calls) and is
destroyed when the block loop advances to the next block.

---

## 8. `moveTo` — signature and body

### 8a. Signature

```cpp
IndexMaskVec moveTo(const uint32_t* leafIndex,    // ptr to leafIndex[batchStart]
                    const uint16_t* voxelOffset); // ptr to voxelOffset[batchStart]
```

Takes raw pointers into the block's decoded inverse-map arrays at the current
batch offset.  Returns the **initial** active-lane mask — `(leafSlice !=
UnusedLeafIndex)` computed before the straddling loop — converted from
`LeafMaskVec` (uint32_t domain) to `IndexMaskVec` (uint64_t domain).

The returned mask has two simultaneous readings:
- **Validity**: lane `i` held a real voxel (not a padding sentinel).
- **Usability**: `mIndices[k][i]` contains a valid stencil index for lane `i`.

They are the same predicate because active lanes are written by `cachedGetValue`
and inactive lanes hold 0 (zeroed at the top of `moveTo`).  The straddling loop
drains `activeMask` to zero internally; the initial mask is saved separately and
returned so the caller always receives a meaningful result.

### 8b. Straddling loop body

Mirrors the `while (any_of(activeMask))` loop from
`ex_stencil_gather_cpu/stencil_gather_cpu.cpp` (lines 698–789):

```
moveTo(leafIndex*, voxelOffset*):

    // Zero all tap slots — inactive lanes will hold index 0 (NanoVDB background).
    for I in [0, SIZE): mIndices[I] = IndexVec(0)

    leafSlice  ← load W values from leafIndex   (LeafIdVec)
    voVec      ← load W values from voxelOffset (OffsetVec)
    activeMask ← (leafSlice != UnusedLeafIndex) as LeafMaskVec

    // Save initial mask before the drain loop; this is what we return.
    resultMask ← widen(activeMask) as IndexMaskVec

    if none_of(activeMask): return resultMask   // entire batch inactive

    // Debug-only advance counter — see §7 for removal instructions.
    #ifndef NDEBUG
        uint32_t nAdvances = 0
    #endif

    while any_of(activeMask):

        leafMask ← activeMask & (leafSlice == LeafIdVec(mBatch.centerLeafID()))

        if none_of(leafMask):
            // No lanes for this leaf — advance to next, assert bound.
            mBatch.advance(mBatch.centerLeafID() + 1)
            NANOVDB_ASSERT(++nAdvances <= mNExtraLeaves)
            continue

        // Prefetch hull (compile-time fold over StencilT::Hull)
        for each HullPoint H in StencilT::Hull:
            mBatch.prefetch<H::di, H::dj, H::dk>(voVec, leafMask)

        // Compute all taps and blend into mIndices
        for each tap I in [0, SIZE):
            using P = tuple_element_t<I, StencilT::Taps>
            tmp ← IndexVec(0)
            mBatch.cachedGetValue<P::di, P::dj, P::dk>(tmp, voVec, leafMask)
            where(leafMask, mIndices[I]) = tmp    // blend: preserve other lanes

        activeMask &= !leafMask    // remove processed lanes

    return resultMask
```

The `where`-blend is essential for correctness in straddling batches: lanes
belonging to a second center leaf must not overwrite results already written
for the first center leaf in the same `mIndices` slot.

Note: `leafMask` is `LeafMaskVec` (uint32_t domain) while `mIndices[I]` is
`IndexVec` (uint64_t).  The `where`-blend requires either a widening cast of
`leafMask` to `IndexMaskVec`, or a `where` overload in Simd.h that accepts
cross-width masks.  Since both are W-bit masks, this is a bitmask reinterpret
with no data movement.

### 8c. Hull and tap loops as compile-time folds

Both loops expand to zero-overhead compile-time instantiations:

```cpp
// Hull prefetch fold
[this, &voVec, &leafMask]<size_t... Is>(std::index_sequence<Is...>) {
    using Hull = typename StencilT::Hull;
    (mBatch.prefetch<
        std::tuple_element_t<Is, Hull>::di,
        std::tuple_element_t<Is, Hull>::dj,
        std::tuple_element_t<Is, Hull>::dk
     >(voVec, leafMask), ...);
}(std::make_index_sequence<HULL_SIZE>{});

// Tap cachedGetValue fold
[this, &voVec, &leafMask]<size_t... Is>(std::index_sequence<Is...>) {
    using Taps = typename StencilT::Taps;
    (blendOneTap<Is>(voVec, leafMask), ...);
}(std::make_index_sequence<SIZE>{});
```

where `blendOneTap<I>` calls `cachedGetValue<P::di, P::dj, P::dk>` into a
temporary and then `where`-blends into `mIndices[I]`.

---

## 9. `getValue<DI,DJ,DK>()` — tap access by coordinate

```cpp
template<int DI, int DJ, int DK>
const IndexVec& getValue() const {
    constexpr int I = findIndex<typename StencilT::Taps, DI, DJ, DK>(
        std::make_index_sequence<SIZE>{});
    static_assert(I >= 0, "StencilAccessor::getValue: tap not in stencil");
    return mIndices[I];
}
```

**Inverse map** (`findIndex`): a `constexpr` fold over all `SIZE` taps, comparing
`(di,dj,dk)` against each `StencilPoint`.  O(N) compile-time evaluations —
negligible for realistic stencil sizes.  Resolved entirely at compile time; the
resulting `I` is a compile-time constant used as an array index.

**`static_assert`**: catches invalid tap coordinates at compile time with a clear
message.  Same safety guarantee as OpenVDB stencil's bounds check.

**Lifetime**: the returned reference is valid only until the next `moveTo` call.
The caller must not cache the reference across batches.

**Indexed access** — for kernels that iterate over all taps generically:

```cpp
const IndexVec& operator[](int i) const { return mIndices[i]; }
```

Public, no bounds check in release.  Same lifetime caveat as `getValue`.

---

## 10. Caller-side usage pattern

```cpp
// Construct once per VBM block
StencilAccessor<BuildT, W, Weno5Stencil> stencil(grid, vbm.firstLeafID(blockID), nExtraLeaves);

for (int b = 0; b < nBatches; ++b) {
    auto active = stencil.moveTo(leafIndex + b*W, voxelOffset + b*W);
    if (util::none_of(active)) continue;

    // Access by coordinate — compile-time slot resolution
    auto idx_m3 = stencil.getValue<-3, 0, 0>();  // Simd<uint64_t,W>
    auto idx_m2 = stencil.getValue<-2, 0, 0>();
    // ... feed into WENO kernel alongside sidecar value fetches
}
// stencil destroyed here (end of block scope)
```

---

## 11. Ownership summary

| Concern | Owner |
|---------|-------|
| Neighbor-leaf cache (`mNeighborLeafIDs[27]`, `mProbedMask`) | `BatchAccessor` |
| Cache population | `BatchAccessor::prefetch` (called by `StencilAccessor`) |
| Cache invalidation | `BatchAccessor` constructor + `advance()` — both clear `mProbedMask` and set `mCenterLeafID`; neither rebuilds the cache |
| `cachedGetValue<di,dj,dk>` | `BatchAccessor` (called by `StencilAccessor`) |
| `advance(newLeafID)` | `BatchAccessor` — this is the only legitimate setter for `mCenterLeafID`; no raw setter exists (would bypass cache invalidation) |
| `mCenterLeafID` read access | `BatchAccessor::centerLeafID()` getter — exposed to `StencilAccessor`; no external setter |
| `leafMask` computation | `StencilAccessor` (derived inside `moveTo`) |
| Straddling loop | `StencilAccessor` |
| Hull prefetch sequencing | `StencilAccessor` |
| Tap fold and `where`-blend | `StencilAccessor` |
| `mIndices[SIZE]` storage and zeroing | `StencilAccessor` (zeroed at top of each `moveTo`) |
| `nExtraLeaves` debug bound | `StencilAccessor` (`#ifndef NDEBUG` member; removable) |
| Center-leaf lifetime (block scope) | Caller |

---

## 12. Design decisions (all resolved)

1. **`moveTo` return type — `IndexMaskVec` by value.**
   The initial `activeMask = (leafSlice != UnusedLeafIndex)` is saved before the
   straddling loop drains it to zero, widened from `LeafMaskVec` (uint32_t) to
   `IndexMaskVec` (uint64_t), and returned.  This gives the caller a mask that is
   semantically aligned with the uint64_t `mIndices` data.  The returned mask has
   two simultaneous readings: which lanes held valid voxels (not padding sentinels),
   and which lanes of `mIndices[k]` contain valid stencil indices.  These are the
   same predicate.  No member copy is kept — the mask is consumed at the call site.

2. **Inactive-lane `mIndices` values — zeroed at top of `moveTo`.**
   `mIndices[0..SIZE-1]` is set to `IndexVec(0)` at the start of every `moveTo`
   call.  Index 0 is the NanoVDB IndexGrid "not found / background" sentinel, so
   inactive lanes yield a well-defined background index rather than stale data.
   The cost is `SIZE` × W zero-writes per call (~36 YMM stores for WENO5 W=16),
   which is negligible.

3. **`operator[]` — public, const-ref, no bounds check.**
   ```cpp
   const IndexVec& operator[](int i) const { return mIndices[i]; }
   ```
   For kernels that iterate over all taps generically.  Same lifetime as
   `getValue`: valid only until the next `moveTo` call.

4. **`StencilT` representation — `std::tuple<StencilPoint<...>...>` for both
   `Taps` and `Hull`.**
   The compile-time fold in §8c requires `std::tuple_element_t<I, Taps>::di` to
   be a compile-time constant.  This is clean with a tuple-of-types but not with a
   constexpr array indexed by a template parameter.  `std::tuple_size_v` and
   `std::tuple_element_t` are the sole introspection mechanisms needed.

5. **`BatchAccessor::centerLeafID()` getter — add; no raw setter.**
   ```cpp
   uint32_t centerLeafID() const { return mCenterLeafID; }
   ```
   The only change required in `BatchAccessor`.  No raw setter: `advance()` is
   the sole legitimate state transition for `mCenterLeafID`.  Both the constructor
   and `advance()` only **invalidate** the cache (clear `mProbedMask`); they do
   not rebuild it.  Cache population is entirely the caller's responsibility via
   `prefetch()`, called by `StencilAccessor` inside the straddling loop.

6. **`nExtraLeaves` — kept as a removable debug sanity check.**
   Passed to the constructor, stored as `#ifndef NDEBUG uint32_t mNExtraLeaves`
   member, asserted against a local `nAdvances` counter on each `advance()` call
   inside `moveTo`.  Termination is guaranteed by the VBM monotonicity invariant
   (§7) without this bound; the bound is belt-and-suspenders only.  To remove once
   vetted: delete the `#ifndef NDEBUG` member block, the assert line, and the
   `nExtraLeaves` constructor parameter — four targeted deletions.
