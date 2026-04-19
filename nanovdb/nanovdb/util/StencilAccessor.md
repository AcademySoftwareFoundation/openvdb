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

### 8.1 Hybrid SIMD → scalar-tail design and public API

`StencilAccessor` uses the hybrid design documented in `BatchAccessor.md`
§8i.  The straddling loop in `moveTo` and the SWAR / direction-extraction
portion of each tap are SIMD; `BatchAccessor::cachedGetValue` then harvests
per-lane direction and local-offset values into stack C arrays and runs a
scalar loop calling `leaf.getValue(offset)`.  Each tap writes directly into
`mIndices[I][0..W-1]` — one scalar `mov` per active lane, no
mask-bool round-trip.

#### Public API is Simd-free

| Member | Type |
|--------|:-----|
| `mIndices` (public) | `alignas(64) uint64_t[SIZE][W]` — results buffer, populated by `moveTo()` |
| `moveTo(leafIndex*, voxelOffset*)` | returns `void` |
| `tapIndex<DI,DJ,DK>()` (static constexpr) | `int` — compile-time tap slot lookup |
| `size()` (static constexpr) | `int` |

Callers consume `mIndices` directly.  Active-lane information comes from
`leafIndex[i] != UnusedLeafIndex` — the same sentinel that `decodeInverseMaps`
produces.  No `SimdMask<>` or `Simd<>` appears in the API.

```cpp
stencilAcc.moveTo(leafIndex + bs, voxelOffset + bs);
for (int i = 0; i < W; ++i) {
    if (leafIndex[bs + i] == CPUVBM::UnusedLeafIndex) continue;
    // named-tap access (compile-time, reorder-safe):
    uint64_t idx_xm3 = stencilAcc.mIndices[SAccT::tapIndex<-3,0,0>()][i];
    // iteration:
    for (int k = 0; k < SAccT::size(); ++k)
        consume(stencilAcc.mIndices[k][i]);
    // SIMD load of tap row using caller's own backend:
    auto row = nanovdb::util::Simd<uint64_t,W>(stencilAcc.mIndices[k],
                                               nanovdb::util::element_aligned);
}
```

#### Layout is ABI

`mIndices[SIZE][W]` row-major is part of the contract.  Changing it (for
example to `[W][SIZE]` or to a SIMD aggregate) is a breaking change.  The
choice matches how the scalar tail produces the data, so "what's written"
and "what's read" share a single authoritative layout.

#### GCC codegen (short version)

With the hybrid in place, neither compiler needs `[[gnu::flatten]]` to
reach reasonable performance.  Measured at 32 M ambient voxels / 50% / 32
threads on i9-285K Arrow Lake: GCC 5.1 ns/voxel, Clang 4.9 ns/voxel —
both beat the scalar `LegacyStencilAccessor` oracle (5.5 GCC, 6.7 Clang).
Adding `flatten` on `moveTo` closes the compiler gap to ~4.8 ns/voxel on
both; the 0.3 ns/voxel gain is not worth the 77 KB monolithic body for
default builds.  Consumers that need peak GCC performance can still
annotate their own entry point.  See `BatchAccessor.md` §8i for the full
perf matrix and the analysis of which operations were kept SIMD vs
scalarized.

### 8.2 What actually bottlenecks the CPU path — `valueMask.isOn` mispredicts

A PMU-counter investigation (`BatchAccessor.md` §8j) replaced several rounds
of structural reasoning with hardware measurements.  It refutes two
hypotheses that had shaped earlier design discussions and identifies the
one lever that dominates CPU performance.

#### What we measured

On a single P-core of an i9-285K (Arrow Lake, 8 P + 16 E, GCC 13 at
`-O3 -march=native`, 32 M-voxel / 50 %-occupancy workload), comparing
per-variant PMU counters for every benchmarking pass exposed via
`ex_stencil_gather_cpu --pass=<name>`:

| Variant | ns/voxel | IPC | branch-miss | L1 miss |
|---------|---------:|----:|------------:|--------:|
| Degenerate (18 × (0,0,0), CSE'd) | 29.0 | 4.02 | **0.75 %** | 0.41 % |
| center-hit × 18 (Legacy, same-leaf, `cmov`'d) | 19.0 | 4.80 | **0.84 %** | 0.47 % |
| InLeaf (hybrid, 18 distinct same-leaf, no CSE) | 76.6 | 1.45 | **9.87 %** | 0.68 % |
| Stencil (hybrid WENO5 cross-leaf) | 96.9 | 1.53 | **8.75 %** | 0.46 % |
| Legacy (WENO5, full tree walks) | 99.2 | 1.98 | **8.85 %** | 0.40 % |

#### The two big findings

1. **L1-dcache miss rates are flat across all variants (~0.4–0.7 %).**
   Multi-leaf L1 pressure — the earlier narrative for why cross-leaf taps
   cost so much — is **not a factor** on this workload.  The neighbour
   leaves' `mValueMask` / `mPrefixSum` data stays L1-resident throughout a
   VBM block.

2. **Branch-miss rates split cleanly into two groups**, and the split is
   not along tree-walk lines.  InLeaf has *no* tree walks (it wraps taps
   mod 8 to the centre leaf by construction) but still lands in the "bad"
   group at 9.87 % — higher than Legacy.  The common factor is the
   **`valueMask.isOn(offset)` conditional** inside
   `LeafNode<ValueOnIndex>::getValue(offset)`:

   ```cpp
   if (!(w & mask)) return 0;   // data-dependent, ~50/50 outcome, unpredictable
   ```

   Every per-tap leaf lookup in the "bad" group — the hybrid's scalar tail,
   Legacy's `legacyAcc[k]`, InLeaf's `cachedGetValueInLeaf` — routes through
   this branch.  Degenerate escapes it via CSE (18 identical taps collapse
   to 1 evaluation).  center-hit escapes it because GCC's inliner in that
   tight loop emits the guarded return as a branchless `cmov` — an
   optimiser accident, not a general property.

#### Branchless experiment

A `legacy-branchless` variant that replaces `leaf.getValue(offset)` with
the unconditional formula inlined at the call site (see §8j.5) recovers a
**3× speedup on Legacy**: from 5.6 ns/voxel to 2.0 ns/voxel at 32 threads,
IPC from 1.98 to 4.29, branch-miss rate from 8.07 % to 1.67 %.  The
tree-walk machinery (`acc.probeLeaf()`) is preserved in that variant; the
only thing removed is the single `isOn` branch per tap.  That single
change accounts for ~65 % of Legacy's total wall-clock time.

#### Revised attribution of Legacy's 5.4 ns/voxel

| Component | ns/voxel |
|-----------|---------:|
| Framing (`decodeInverseMaps`, loop, anti-DCE) | 0.25 |
| Leaf-local `getValue` work (loads + `popcnt`) | 0.75 |
| **`valueMask.isOn` branch mispredicts** (~24/voxel × ~15 cy) | **~3.6** |
| Tree walk vs 27-leaf cache differential | ~0.3 |
| Multi-leaf L1 pressure | ~0 |
| **Total** | **~5.4** |

Earlier versions of this section attributed the bulk of Legacy's cost to
tree-walk pointer chases and multi-leaf L1 traffic; both turned out to
be minor.  The hybrid `StencilAccessor` matches Legacy (~5.1 ns/voxel)
because both pay the same dominant `isOn` mispredict cost.

#### Consequence for architectural decisions

- **The shipped hybrid design is the right API choice** (Simd-free public
  surface, compiler-portable) but its wall-clock edge over Legacy is
  marginal (~0.3 ns/voxel), not the ~3 ns/voxel originally implied.
- **The cheap architectural win was a branchless reformulation of
  `LeafData<ValueOnIndex>::getValue`**: shipped as the default body of
  `getValue` in `NanoVDB.h` (see `BatchAccessor.md` §8k), gated by
  `NANOVDB_USE_BRANCHY_GETVALUE` to restore the old branchy form.
  End-to-end 1.4× on realistic narrow-band workloads, 2.8× on
  random-access.
- **HaloStencilAccessor's value proposition is validated but narrower**:
  its precomputed uint64 index buffer naturally eliminates `isOn`
  branches by never evaluating them.  Now that the branchless
  `getValue` captures the same win cheaply, the halo's remaining
  advantage is "zero per-tap work at query time" rather than "avoids
  the isOn mispredict storm."  Worth building for the absolute-perf
  cases; less urgent than previously framed.

See `BatchAccessor.md` §8j for the original measurement matrix and
correction log (§8g/§8h/§8i), and `BatchAccessor.md` §8k for the
follow-on that made `getValue` branchless-by-default, added the
narrow-band validation benchmark (`ex_narrowband_stencil_cpu`), and
the leaf-only `ReadAccessor<BuildT, 0, -1, -1>` finding.

---

## 9. `tapIndex<DI,DJ,DK>()` — compile-time slot lookup, `mIndices[][]` access

> **API evolution.**  Earlier drafts of this document described a
> `getValue<DI,DJ,DK>() const → const IndexVec&` member and an
> `operator[](int) → const IndexVec&` accessor.  Both were removed in the
> hybrid refactor (§8.1).  The results buffer is now a plain public 2D
> C array; callers pick their own access pattern.  The change aligns with
> the hybrid's Simd-free public API — no `Simd<>` or `SimdMask<>` type
> appears in the class's public interface.

```cpp
// Storage — public, part of the ABI:
alignas(64) uint64_t mIndices[SIZE][W];

// Compile-time slot lookup (reorder-safe, zero runtime cost):
template<int DI, int DJ, int DK>
static constexpr int tapIndex() {
    constexpr int I = detail::findIndex<typename StencilT::Taps, DI, DJ, DK>(
        std::make_index_sequence<SIZE>{});
    static_assert(I >= 0, "StencilAccessor::tapIndex: tap not in stencil");
    return I;
}

// Iteration bound:
static constexpr int size() { return SIZE; }
```

**Inverse map** (`detail::findIndex`): a `constexpr` fold over all `SIZE`
taps, comparing `(DI,DJ,DK)` against each `StencilPoint`.  O(N) compile-time
evaluations — negligible for realistic stencil sizes.  Resolved entirely at
compile time; `tapIndex<-3,0,0>()` compiles to an integer literal.

**`static_assert`**: catches invalid tap coordinates at compile time with a
clear message.  Same safety guarantee as OpenVDB stencil's bounds check.

**Lifetime**: `mIndices` is valid only until the next `moveTo` call.  The
caller must not cache references across batches.

**Why expose `mIndices` directly** (rather than a method that returns it):
the results buffer is plain data — no lazy work, no layout translation, no
invariants to enforce.  Hiding it behind an accessor would pretend
otherwise.  Direct access also lets callers choose their SIMD load pattern
(or scalar iteration) without our API imposing one.

---

## 10. Caller-side usage pattern

```cpp
// Construct once per VBM block.
StencilAccessor<BuildT, W, Weno5Stencil> stencil(
    grid, vbm.firstLeafID(blockID), nExtraLeaves);

// Active-lane information comes from decodeInverseMaps's UnusedLeafIndex
// sentinel — the same source that StencilAccessor uses internally.
for (int bs = 0; bs < BlockWidth; bs += W) {
    stencil.moveTo(leafIndex + bs, voxelOffset + bs);  // returns void

    // Option A: scalar iteration across lanes and taps.
    for (int i = 0; i < W; ++i) {
        if (leafIndex[bs + i] == UnusedLeafIndex) continue;
        for (int k = 0; k < StencilAccessor::size(); ++k) {
            consume(stencil.mIndices[k][i]);  // uint64_t
        }
    }

    // Option B: SIMD load of an entire tap row (caller picks backend/width).
    auto row_m3 = util::Simd<uint64_t, W>(
        stencil.mIndices[stencilAccT::tapIndex<-3, 0, 0>()],
        util::element_aligned);

    // Option C: compile-time named tap access for a handful of taps.
    const uint64_t& xm3 = stencil.mIndices[stencilAccT::tapIndex<-3,0,0>()][i];
}
// stencil destroyed here (end of block scope)
```

No `Simd<>` or `SimdMask<>` types appear in the public API.  The caller
uses its own SIMD backend (or none) to consume `mIndices`.

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
| Tap fold (writes directly into `mIndices[Is]`) | `StencilAccessor::calcTaps` |
| `mIndices[SIZE][W]` storage and zeroing | `StencilAccessor` (public member; `std::memset` at top of each `moveTo`) |
| `nExtraLeaves` debug bound | `StencilAccessor` (`#ifndef NDEBUG` member; removable) |
| Center-leaf lifetime (block scope) | Caller |

---

## 12. Design decisions (all resolved)

> **Evolution.**  Decisions 1 and 3 below have been superseded by the
> hybrid refactor (§8.1): `moveTo` now returns `void`, and `operator[]` /
> `getValue<>()` were removed in favour of public `mIndices` access +
> `tapIndex<>()`.  The original rationales are preserved for historical
> context; the current API is §9's.

1. **`moveTo` return type — ~~`IndexMaskVec` by value~~ `void` (revised §8.1).**
   *Original rationale:* The initial
   `activeMask = (leafSlice != UnusedLeafIndex)` was saved before the
   straddling loop drains it to zero, widened from `LeafMaskVec` (uint32_t)
   to `IndexMaskVec` (uint64_t), and returned.  This gave the caller a mask
   semantically aligned with the uint64_t `mIndices` data.
   *Revised:* `moveTo` now returns `void`.  The active-lane information is
   redundant: callers already have `leafIndex[]` from `decodeInverseMaps`
   and the same `UnusedLeafIndex` sentinel that `StencilAccessor` uses
   internally.  Returning the mask duplicated state and forced a
   heterogeneous `SimdMask<uint32_t>` → `SimdMask<uint64_t>` widening with
   a boolean round-trip (§8h) — all for zero information gain.  Removing
   it also eliminated the last `SimdMask<>` type from the public API.

2. **Inactive-lane `mIndices` values — zeroed at top of `moveTo`.**
   `mIndices` is set to zero (via `std::memset`) at the start of every
   `moveTo` call.  Index 0 is the NanoVDB IndexGrid "not found / background"
   sentinel, so inactive lanes yield a well-defined background index rather
   than stale data.  The cost is a single `memset` of `SIZE * W * 8` bytes
   per call (2304 B for WENO5 W=16), which stays in L1 and pipelines under
   other work.

3. **~~`operator[]` — public, const-ref, no bounds check~~ removed (revised §9).**
   *Original:* `const IndexVec& operator[](int i) const { return mIndices[i]; }`
   for kernels that iterate over all taps generically.
   *Revised:* `mIndices` is now a public member (§9); direct indexing
   replaces both `operator[](int)` and `getValue<DI,DJ,DK>()`.  Named-tap
   access is via the `tapIndex<DI,DJ,DK>()` static constexpr slot lookup.
   This change is consistent with the hybrid's Simd-free public API — no
   method can now return a `Simd<>` or `SimdMask<>` reference.

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
