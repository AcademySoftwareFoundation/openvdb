# Native NanoVDB Level-Set Renormalization — Design Notes

**Status:** design settled on the points below; no implementation written yet.
The `Benchmark/` subtree is a verbatim reference import (see `Benchmark/PROVENANCE.md`).

**Branch:** `levelset-renormalization`, based on ASWF `master`.

### Picking this up cold

This document is meant to be sufficient on its own.  Read in this order:

* **§3** -- the settled scope.  Five decisions, each with its rationale; treat
  these as fixed unless explicitly revisited.
* **§4** -- the open-question register.  Everything not yet decided, with enough
  context to decide it.  This is the to-do list.
* **§5** -- what NanoVDB already provides versus the delta the reference carries.  Start
  here to size the work.
* **§8** -- proposed order of work.
* **§9** -- `file:line` map into the imported reference code.

Landed on this branch so far: the reference import (`Benchmark/`), this document, and
`nanovdb/math/FiniteDifference.h` (§3.4).  Nothing else exists yet.

**Scope reminder:** CUDA only.  A host path is explicitly *not* in scope at this
stage, notwithstanding the `ExecutionPolicy::CPU` code present in the reference
import.

---

## 1. Objective

Provide a native NanoVDB implementation of **level-set renormalization**
(reinitialization / redistancing): restoring the Eikonal property `|grad phi| = 1`
to a narrow-band level set that has drifted away from being a true signed
distance function.

The starting point is the CUDA implementation in the reference benchmark
(imported under `Benchmark/`), which today lives outside NanoVDB and shadows one of
its headers.  The goal is to promote that code into NanoVDB proper, as a
first-class tool, with the scope restrictions recorded in §3.

This is strictly a **repair** operator on an existing level set.  It does not
move the interface.  Interface motion under a speed function is a separate
concern (`Benchmark/include/LevelSetPropagate.h`) and is **out of scope** here, even
though the imported tree contains it.

## 2. The operator

Renormalization solves, in pseudo-time `tau`, to steady state:

    d(phi)/d(tau) = S(phi_0) * (1 - |grad phi|)

The zero isocontour is a fixed point; everything else relaxes toward unit
gradient magnitude.  The discretization inherited from OpenVDB and the reference is:

* `S` in the smoothed Peng et al. form `phi_0 / sqrt(phi_0^2 + |grad phi|^2)`
  (self-scaling, rather than a fixed epsilon);
* `|grad phi|^2` by **WENO5 upwinding + Godunov's scheme**, evaluated in index
  space (`invDx2 = dx2 = 1`), with the physical scale reintroduced through
  `mInvDx` in the update;
* **TVD-RK2** in pseudo-time, as a pair of Euler stages combined by the
  `<Numerator, Denominator>` template pair (`alpha = N/D`, `beta = 1 - alpha`);
* `dt = 0.9 * dx`, matching OpenVDB's `TVD_RK2` constant.

The per-voxel update is identical in the two implementations:

    ValueType v = phi0 / ( Sqrt(Pow2(phi0) + normSqGradPhi) + Tolerance );
    v = phi0 - mDt * v * (Sqrt(normSqGradPhi) * mInvDx - 1.0f);
    result[n] = Nominator ? alpha * phi[n] + beta * v : v;

* OpenVDB: `openvdb/openvdb/tools/LevelSetTracker.h:637-642`
* Reference/CUDA: `Benchmark/src/Benchmark.cu:347-355`

### Relationship to OpenVDB

`Benchmark/include/LevelSetTrackerNew.h` is a fork of
`openvdb/openvdb/tools/LevelSetTracker.h` with NanoVDB calls interleaved into
`track()`.  Upstream OpenVDB's `track()` is the standard narrow-band maintenance
cycle (`LevelSetTracker.h:305-316`):

    dilateActiveValues(..., NN_FACE)   // grow the band
    normalize()                        // renormalize (this operator)
    prune()                            // drop voxels outside the band

Renormalization proper is the middle step and is **topology-preserving**.

OpenVDB also offers `tools/FastSweeping.h` (same Eikonal equation by parallel
fast sweeping) and `tools/LevelSetRebuild.h` (mesh + re-rasterize).  Both are
alternative routes to the same objective and are out of scope.

## 3. Scope decisions (settled)

### 3.1 IndexGrid + sidecar only

The operator targets a **`ValueOnIndex` IndexGrid with a separate value
sidecar**, not a `NanoGrid<float>`.  This matches the reference implementation and is
the representation the rest of the pipeline already speaks: `DilateGrid` and
`PruneGrid` natively emit and consume IndexGrids, and a TVD-RK temporary is a
plain `ValueT*` allocation rather than a second grid.

No `FloatGrid` overload is planned.

Consequences:

* `BuildT` (topology encoding, fixed at `ValueOnIndex`) and `ValueT` (sidecar
  element type) become **independent** template parameters.  The reference code
  currently uses `WenoStencil<nanovdb::FloatGrid>` purely as a carrier for
  `SIZE` and the `WenoPt<>` index map (`Benchmark/src/Benchmark.cu:335`), which is the
  awkwardness that the `WenoStencil: ValueType-templated` work on the
  `vbm-cpu-port` branch (`9b2ef25f9`) was addressing.  That refactor is on the
  critical path.
* **The slot-0 convention is part of the published contract**, not an internal
  trick: index `0` means "no value", sidecar element `0` holds the background,
  and every kernel depends on it.  It must be documented in the header.
* **Sidecar re-indexing is ours to own.**  Dilation and pruning renumber every
  active voxel, so a sidecar is not merely resized across a topology change —
  it must be scattered through the old-to-new index mapping
  (`nanovdb/util/cuda/Injection.cuh`).  A sidecar is meaningless without the
  handle that indexes it; consider binding them in one small owning type rather
  than passing two loose arguments, since "this buffer matches that grid's
  numbering" is otherwise an invariant that can be violated silently.

### 3.2 Pure narrow-band: no background sign information

We assume knowledge of `phi` **only on active voxels**.  We do **not** presume
any sign information on inactive voxels.

This is a real departure from OpenVDB.  A `FloatGrid` level set can carry sign
information beyond the active set, because *tile values* exist at grid locations
outside the narrow band, and OpenVDB's stencil reads them through its accessor
to obtain a correctly signed `+/- background` at the band edge.  A `ValueOnIndex`
grid has no value at all for an inactive voxel, and we are choosing not to
reintroduce one.

Instead we depend on the **sign-extrapolation heuristic** already used by the reference: a
missing neighbor reads as `background * Sign(<nearer known value>)` — i.e.
truncation at the narrow-band half-width, carrying the sign of the nearest
in-band voxel.

Rationale: a narrow-band level set that carries sign information only where it
carries distance information is self-consistent, and the operator then cannot
silently degrade when out-of-band information is stale or absent — no operator
in this pipeline maintains it.

This also settles the `ValueIndex` question: numbering inactive in-leaf voxels
would only be worth it in order to store signed backgrounds, which this decision
rules out.  **`ValueOnIndex`, active voxels only.**

Consequences:

* No `changeLevelSetBackground`, no `TrimMode` (trimming reduces to the
  `|phi| >= background` prune already implemented), no interior/exterior tile
  bookkeeping.  The API shrinks accordingly.
* The validation posture inverts — see §6.

### 3.3 The heuristic is currently two rules; it must become one specification

The same question ("what sign does a voxel just outside the known band have?")
is answered twice, with different tie-breaking:

1. **Stencil gather** (`Benchmark/src/Benchmark.cu:328-345`): a missing neighbor takes
   `background * Sign(next-inner ring value)`, cascading outward 1 -> 2 -> 3.
   This is **order-dependent** — ring 2 consumes the already-repaired ring 1 —
   so any refactor must preserve the sequencing or it silently changes results.
2. **Dilation seeding** (`Benchmark/src/Benchmark.cu:108-116`): a newly activated voxel
   takes `background * Sign(first active old face-neighbor)` under a fixed
   six-way priority order.

In a library these should be one documented rule with one implementation.  It is
also worth deciding deliberately between first-hit priority and something
sign-symmetric (majority / nearest-magnitude vote): first-hit makes the result
depend on traversal order, which is fine if documented and bad if discovered
later.

Note that rule 2 is emulating OpenVDB behaviour rather than inventing it: in
OpenVDB, inactive voxels inside an allocated leaf already hold correctly signed
`+/- background`, so dilation activates voxels that are already correct.

### 3.4 One implemented scheme pair; the rest are hooks that throw

The API keeps the **full** scheme matrix visible, but only
**HJ-WENO5 + TVD-RK2** is implemented.  Every other requested combination
throws a "not implemented" error at the host-side dispatch.

This is deliberate: the enum, the dispatch skeleton and the error arms are cheap
now and make adding a scheme later a pure addition rather than an API change.
On GPU it is also *free* -- an unimplemented arm never instantiates a kernel, so
the hooks cost nothing in compile time or code size.  (Compare: every
*implemented* pair is a distinct kernel instantiation, which is the real reason
not to open the matrix speculatively.)

`throw std::runtime_error` is the established NanoVDB convention for host-side
tool errors (`tools/cuda/PointsToGrid.cuh`, `tools/cuda/DilateGrid.cuh`, and
most other tool headers), so the unimplemented arms should use it rather than
inventing a mechanism.

#### The enums have to be defined in NanoVDB

OpenVDB owns the scheme type tags, in `openvdb/openvdb/math/FiniteDifference.h`:

    enum BiasedGradientScheme {          // :164
        UNKNOWN_BIAS = -1, FIRST_BIAS = 0, SECOND_BIAS, THIRD_BIAS,
        WENO5_BIAS, HJWENO5_BIAS };

    enum TemporalIntegrationScheme {     // :233
        UNKNOWN_TIS = -1, TVD_RK1, TVD_RK2, TVD_RK3 };

**NanoVDB has no equivalent -- a grep for `BiasedGradientScheme`,
`TemporalIntegrationScheme`, `TVD_RK`, `WENO5_BIAS` or `DScheme` across all of
`nanovdb/` returns nothing.**  And we cannot simply reuse OpenVDB's: NanoVDB is
a standalone header library in which OpenVDB is an *optional* dependency
(`NANOVDB_USE_OPENVDB`), pulled in only by the conversion utilities
(`CreateNanoGrid.h`, `NanoToOpenVDB.h`).  A core tool header must not include
OpenVDB.

So we declare our own, in **`nanovdb/math/FiniteDifference.h`** (new header,
namespace `nanovdb::math`, registered in `nanovdb/CMakeLists.txt`).  Decided:

* **Declare only the schemes we implement, plus the `UNKNOWN` sentinels.**  Not
  the full OpenVDB set.  An unsupported scheme then cannot be *named* at all,
  which is a stronger guarantee than accepting it and throwing -- the error moves
  from run time to compile time.  The §3.4 "hooks that throw" arms therefore only
  need to cover `UNKNOWN_*` and out-of-range integers arriving from a cast.
* **Pin the values explicitly to their OpenVDB numbers**, which OpenVDB assigns
  implicitly:

      UNKNOWN_BIAS = -1   FIRST_BIAS = 0  SECOND_BIAS = 1  THIRD_BIAS = 2
      WENO5_BIAS   =  3   HJWENO5_BIAS = 4
      UNKNOWN_TIS  = -1   TVD_RK1    = 0  TVD_RK2     = 1  TVD_RK3     = 2

  so NanoVDB declares `UNKNOWN_BIAS = -1`, **`HJWENO5_BIAS = 4`**,
  `UNKNOWN_TIS = -1`, **`TVD_RK2 = 1`**.  The gaps in the numbering are
  intentional; adding a scheme later means adding its label *at its OpenVDB
  value*, never renumbering an existing one.  Matching numbers make translation
  across the boundary a `static_cast` rather than a mapping table, and let a
  `static_assert` catch drift.
* **`HJWENO5_BIAS` (4), not `WENO5_BIAS` (3).**  Worth stating loudly because the
  two are easy to conflate in conversation but are different operators (§6), and
  picking the wrong one silently changes the constant.
* Mirror only the two user-facing enums.  OpenVDB's lower-level `DScheme`
  (`FD_HJWENO5`, `BD_WENO5`, ...) is an implementation layer that NanoVDB does
  not need: `WenoStencil` bakes in the HJ form (§6).
* Use **unscoped** enums, following NanoVDB's own precedent for an
  operator-selection enum used as a template argument,
  `nanovdb::tools::morphology::NearestNeighbors`
  (`nanovdb/util/MorphologyHelpers.h:22`).  `nanovdb::math` keeps call sites
  reading the same as OpenVDB's.

Consequence accepted: OpenVDB -> NanoVDB translation is now *partial* rather than
total -- only `HJWENO5_BIAS` and `TVD_RK2` have NanoVDB counterparts.  The
boundary shim must reject the rest rather than cast blindly.

Open: whether the pair is carried as **runtime enums** (mirroring OpenVDB's
`State` + `normalize -> normalize1<S> -> normalize2<S,T>` dispatch ladder, which
exists precisely to turn runtime enums into template arguments) or as
**compile-time tags** with a thin runtime shim at the boundary.  The runtime-enum
form is friendlier to a host-side caller; the tag form avoids generating the
dispatch ladder at all while only one pair is live.

### 3.5 The tracker owns all of its state

**Decision: the tracker owns the grid handle, the value sidecar, the
time-stepping scratch, and the VoxelBlockManager.  None of the four is injected
from outside.**

#### Why this is not the OpenVDB arrangement

`LevelSetTracker` holds `GridT& mGrid` plus a `LeafManager*`, and that suffices
because of three properties NanoVDB does not have:

1. **Topology mutation is in-place**, so grid *identity* is stable across
   `dilate`/`prune`.  A NanoVDB grid is an immutable linear buffer; a topology
   change produces a *new* buffer, and the previous device pointer is garbage,
   not merely stale.
2. **Values live inside the tree**, so data follows topology automatically.  Our
   sidecar does not merely need resizing: dilation and pruning renumber every
   active voxel, so its contents must be *scattered through the old-to-new index
   map* (`nanovdb/util/cuda/Injection.cuh`).
3. **RK temporaries come from `LeafManager::rebuildAuxBuffers(n)`**, which owns
   the shadow buffers and keeps them alive.  We have no analogue.

So in place of one stable reference we have **four artifacts keyed to a single
active-voxel numbering** -- grid handle, sidecar, scratch, VBM -- that must be
replaced *atomically* on any topology change.  Every one of them produces
silently wrong answers rather than crashing when it falls out of sync.  That is
the whole reason ownership is centralized.

#### Shape

    template<typename BuildT, typename ValueT, typename BufferT>
    class LevelSetRenormalizer {           // name TBD, see §4
        GridHandle<BufferT>              mGrid;       // owned; replaced on topology change
        BufferT                          mPhi;        // owned sidecar; slot 0 == background
        BufferT                          mScratch;    // RK temporaries; cached across calls
        VoxelBlockManagerHandle<BufferT> mVBM;        // owned, derived, never injected
        ValueT                           mBackground; // no home in the grid; see below
    };

Construction moves a grid handle and sidecar in; `release()` moves them back
out.  Ownership then is structural rather than documented, and the four-way
invariant cannot be violated from outside.

#### Why move-in/move-out rather than references

The reference form -- `LevelSetRenormalizer(GridHandle<BufferT>& grid,
BufferT& phi)` -- is closer to OpenVDB and was rejected.  It reseats the
caller's handle behind their back, and unlike OpenVDB the caller's cached
`deviceGrid<BuildT>()` pointer is then **dangling, not stale**.  The trap only
springs once topology operations land, i.e. after the API has hardened, so it
has to be avoided up front.

#### Why the VBM specifically must never be injected

* **`Log2BlockWidth` is not part of the handle type.**
  `VoxelBlockManagerHandle` is `template<typename BufferT>` only
  (`nanovdb/tools/VoxelBlockManager.h:94`), while the block width lives on
  `VoxelBlockManagerBase<Log2BlockWidth>` (`:229`).  A VBM built with
  `buildVoxelBlockManager<6, BufferT>` and consumed by a kernel calling
  `VoxelBlockManager<7>::decodeInverseMaps` type-checks, compiles, links, and
  yields garbage.  An injected VBM is an unverifiable precondition.  This
  argument alone is decisive.
* **Cache invalidation is only decidable under ownership.**  The VBM is pure
  derived state, stale iff topology changed.  If the tracker owns the grid,
  nothing else *can* change topology, so it knows its cached VBM is valid.  With
  an external grid it must either rebuild unconditionally or trust the caller.
* **The handle is already move-only** (copy constructor and copy assignment are
  `= delete`, `:119-120`), so the library itself expects single ownership.

The one real counter-argument is amortization: `dilate -> renormalize -> prune`
under separate owners rebuilds the VBM three times.  That should be answered
*inside* a future `track()` composite -- which owns one VBM and passes it to its
internal steps -- rather than by exposing injection in the public API.  For the
renormalize-only scope the cost is zero anyway: renormalization is
topology-preserving, so it is one build per call regardless.

#### The scratch buffer is tracker state, not a local

The reference allocates it fresh inside every call (`Benchmark/src/Benchmark.cu:450-451`): a full
allocation plus `initializeGPUSidecarAndBackgroundValue`, which performs a D2H
read of `valueCount` *and* a one-element H2D memcpy -- two synchronizations per
call, in a per-frame function.  Held on the tracker it is reallocated only when
`valueCount` changes.

The *number* of temporaries is a function of the temporal scheme, not a
constant: OpenVDB uses `rebuildAuxBuffers(TemporalScheme == TVD_RK3 ? 2 : 1)`.
Derive it at compile time from the scheme so that filling in the TVD-RK3 hook
later cannot silently under-allocate.

#### What has to live on the tracker because the grid cannot hold it

`BuildToValueMap<ValueOnIndex>::Type` is `uint64_t`
(`nanovdb/NanoVDB.h:539-542`), so an IndexGrid's "background" is an *index*, not
a distance.  OpenVDB reads `mGrid->background()`; we must carry the float
background and the half-width as tracker state.  `dx` is recoverable from the
grid's map and need not be stored.

Because every sidecar reallocation must re-establish the `slot 0 == background`
contract (§3.1), that too becomes automatic under ownership and an extra
invariant for the caller to break otherwise.

#### Renormalization alone never replaces anything

Tracing the RK2 ping-pong (`Benchmark/src/Benchmark.cu:370-429`): `euler01` reads phi
and writes scratch; `euler12` reads scratch for the stencil and phi for the
convex combination, and writes phi.  Each pair therefore ends with the result
back in phi, and grid, sidecar and VBM are all stable across the entire call.

The replacement machinery is exercised only once `track()` lands.  This is an
argument for settling ownership *now* rather than shipping a reference-based API
and breaking it later -- but it also means the renormalizer can be built and
validated before any replacement logic exists.

## 4. Open questions (resumption register)

Everything not yet decided, with enough context to decide it.  Rough priority
order: A-blockers shape the API, B-items are implementation choices, C-items are
deferrable.

### A1. Scope -- renormalization only, or all of `track()`?

Renormalization is topology-preserving: one kernel pair, one scratch buffer, no
handle churn, nothing replaced (§3.5).  `track()` adds dilate -> prune, a VBM
rebuild per topology change, sidecar re-indexing, and grid-handle replacement.

Leaning: ship renormalization first as a self-contained unit; `track()` as the
natural follow-on.  Not yet confirmed.

### A2. Dispatch form -- runtime enums or compile-time tags?

§3.4 settles *which* schemes exist; it does not settle how the selection is
carried.  Two forms:

* **Runtime enums**, mirroring OpenVDB's `State` plus the
  `normalize -> normalize1<SpatialScheme> -> normalize2<...,TemporalScheme>`
  ladder, which exists precisely to turn runtime enums into template arguments.
  Friendlier to a host-side caller, familiar to OpenVDB users.
* **Compile-time tags** with a thin runtime shim only at the API boundary.
  Avoids generating the ladder at all while only one pair is live.

Leaning: runtime enums, for familiarity.  Note the ladder is nearly degenerate
while only one pair exists.

### A3. Naming and placement

Working names used in this document, none of them decided:

* Header: `nanovdb/tools/cuda/LevelSetRenormalize.cuh`?  `LevelSetTracker.cuh`
  reads better if A1 goes the `track()` way, but would then collide conceptually
  with OpenVDB's class of the same name.
* Class: `LevelSetRenormalizer`?  Existing NanoVDB tools are noun-ish operator
  classes (`DilateGrid`, `PruneGrid`, `MergeGrids`, `PointsToGrid`), which
  argues for something like `RenormalizeLevelSet`.
* Free-function entry point alongside the class, as `DilateGrid` does?

Note `nanovdb/tools/` currently has **no** `LevelSet*` header at all; the
nearest relative is `tools/cuda/SignedFloodFill.cuh`.

### A4. Should `{handle, channels}` be a named reusable type?

The "grid handle plus one or more value sidecars, re-indexed together on
topology change" pairing recurs throughout this pipeline -- phi, speed, and any
future extended field -- and propagation would want the same thing.  It is the
natural home for the slot-0 contract (§3.1) and the `Injection` scatter.

Broader than renormalization, so possibly out of scope; but if it is going to
exist, the tracker should be built on it rather than retrofitted.  Open.

### B1. Unify the sign-extrapolation rule (§3.3)

Two implementations of one semantic today.  Needs to become one specification
plus one implementation.  Sub-decision: first-hit priority (current, traversal-
order dependent) versus something sign-symmetric such as a majority or
nearest-magnitude vote.

### B2. Where does `Log2BlockWidth` live?

Currently `Benchmark::BlockWidthLog2 = 7` (128 active voxels per CUDA block), a
constant of the benchmark rather than of the data.  Options: a template
parameter of the tracker defaulting to 7; a fixed implementation detail; or
tuned per-kernel.  Note §3.5: because the value is *not* encoded in
`VoxelBlockManagerHandle`, whatever is chosen must be enforced by construction
rather than by convention.

### B3. Stream and buffer discipline

The reference hardcodes the default stream and `UnifiedBuffer` (§7).  The promoted code
should take an explicit `cudaStream_t` and be templated on `BufferT`, with
`cuda::Buffer<T>` (upstream `0ab0a81e7`) for device-only scratch.  Mechanical,
but decide the signatures before writing kernels rather than after.

### B4. Reuse the existing `gatherIndices` refactor?

The 19-point gather is currently open-coded in the functor
(`Benchmark/src/Benchmark.cu:286-327`).  It has already been solved once, on this
repository's `origin/vbm-cpu-port` branch, as
`158e3df53 "WenoStencil: absorb gather as static gatherIndices(); drop
LegacyStencilAccessor"`, together with `9b2ef25f9` making `WenoStencil`
`ValueType`-templated (which §3.1 puts on the critical path anyway).

Open: cherry-pick those two commits onto this branch, or re-derive.  Related:
`vbm-cpu-port` is 157 commits ahead of the fork's `master` and un-upstreamed, so
cherry-picking narrowly is probably right.

### B5. Validation cases

§6 sets the posture; the concrete cases are not chosen.  Needed: analytic SDFs
where the exact answer is known (sphere, torus, a thin sheet to exercise the
band-edge extrapolation), plus recovery of the OpenVDB tandem harness from
the source repository at `f8ab0ca`.  Also undecided: unit test in `nanovdb/unittest`
versus example-only.

### B6. Wiring the example into CMake

The imported `Benchmark/` tree is deliberately unregistered (`Benchmark/PROVENANCE.md`).
The eventual `ex_renormalize_levelset_cuda` proper needs a flat file layout --
`nanovdb_example()`'s glob is non-recursive -- and must resolve the shadowed
`Stencils.h` (already removed; see §5).

### C1. The WENO5 constants (§6)

`scale2` (100x epsilon discrepancy) and `RealT` (float versus double weights).
Secondary: they do not gate the design, but both defaults should be chosen
deliberately rather than inherited, and the apportionment experiment described
in §6 should be run before NanoVDB's `WENO5` is blessed as the reference.

### C2. Apportion the 5e-3

The experiment itself: build the reference harness with NanoVDB's `WENO5` instantiated
at `RealT = double`, and separately with `scale2 = 0.01` / `dx_world^2` instead
of `1.f`, to see how much of the discrepancy is epsilon and how much is
precision.

### Settled elsewhere, recorded here so they are not reopened

* IndexGrid + sidecar only, `ValueOnIndex` -- §3.1
* Pure narrow-band, no tile sign information -- §3.2
* HJ-WENO5 + TVD-RK2 only, other schemes unnameable -- §3.4,
  `nanovdb/math/FiniteDifference.h`
* The tracker owns grid, sidecar, scratch and VBM -- §3.5
* CUDA only; no host path at this stage
* Interface propagation and velocity extension are out of scope entirely

## 5. Dependency inventory

### Already upstream — no work needed

Everything `Benchmark/src/Benchmark.cu` includes is satisfied by ASWF `master`:

| Header | Provides |
|---|---|
| `nanovdb/tools/VoxelBlockManager.h`, `tools/cuda/VoxelBlockManager.cuh` | block -> CUDA-block mapping, `decodeInverseMaps` |
| `nanovdb/tools/cuda/DilateGrid.cuh` | topological dilation of an IndexGrid |
| `nanovdb/tools/cuda/PruneGrid.cuh` | topological pruning of an IndexGrid |
| `nanovdb/util/cuda/Injection.cuh` | sidecar scatter across a topology change |
| `nanovdb/math/Stencils.h` | `WenoStencil`, `WenoPt<>`, `WENO5`, `GodunovsNormSqrd` |
| `nanovdb/util/cuda/Util.h` | `dynamicSharedMemoryLauncher`, `operatorKernel` |

`nanovdb/tools/` has **no** `LevelSet*` header today; the nearest relative is
`tools/cuda/SignedFloodFill.cuh`.

### The actual delta the reference carries

1. ~~**`Benchmark/include/Stencils.h`**~~ — **DONE.**  Two **static, array-based**
   overloads, `WenoStencil::normSqGrad(const ValueType* v, invDx2, dx2, iso)` and
   `WenoStencil::gradient(const ValueType* v, inv2Dx)`, now live in
   `nanovdb/math/Stencils.h`; the member forms delegate to them, so the WENO
   formula exists in one place.  They exist because a GPU kernel gathers the
   stencil into registers itself and cannot use the stencil object's accessor.
   The reference's shadowed copy — and the include-guard race it relied on
   (source commit `f8ab0ca`) — have been deleted.
2. **The 19-point gather** (`Benchmark/src/Benchmark.cu:286-327`) — the `leafPtrs[3][3]`
   + octal-offset block.  The only genuinely new GPU machinery.  Note this has
   been solved once already: `vbm-cpu-port` commit `158e3df53`
   ("WenoStencil: absorb gather as static gatherIndices(); drop
   LegacyStencilAccessor").
3. **The narrow-band boundary rule** (`Benchmark/src/Benchmark.cu:328-345`), per §3.2/§3.3.
4. **The Euler step** (`Benchmark/src/Benchmark.cu:347-355`) — already line-for-line
   identical to OpenVDB.

### Related un-upstreamed work

`origin/vbm-cpu-port` in this repository (157 commits ahead of the fork's
`master`) carries host counterparts and refactors that overlap heavily with this
effort: `tools/DilateGrid.h`, `util/Morphology.h`, `util/Injection.h`, the
`ValueType`-templated `WenoStencil` with `gatherIndices()`, `util/Simd.h`,
`util/BatchAccessor.h`.  Whether to build on that branch or cherry-pick from it
is an open question; this branch currently takes the second route (based on ASWF
`master`, importing nothing but the reference tree).

## 6. Validation strategy

Because of §3.2, **OpenVDB is no longer a bit-exact oracle.**  It becomes a
reference that should agree wherever both are well-defined.  OpenVDB reads
`+/- background` from tile values at the band edge; we reconstruct the sign.
Those coincide wherever the band is locally thick and single-signed, and
legitimately differ on thin or under-resolved features.  A residual discrepancy
against OpenVDB is therefore an expected property, not a defect to chase.

This implies two tiers:

1. **Analytic ground truth** — sphere, torus, a thin sheet — where the exact SDF
   is known and the band-edge cases can be measured rather than compared.
2. **OpenVDB cross-check** — the existing harness.  The tandem call sites were
   stripped from the imported snapshot; recover them from
   the source repository at `f8ab0ca:Benchmark/include/LevelSetTrackerNew.h:340-372`, which
   interleaves both stacks and calls `compareOpenVDBDataToNanoVDBSidecar` after
   each stage.

### Resolved: the scheme the reference actually uses

`Benchmark/src/main.cpp:113-117` selects, for both the propagator and the tracker:

    setSpatialScheme(HJWENO5_BIAS)
    setTemporalScheme(TVD_RK2)
    setNormCount(3)

So the target configuration is **HJ-WENO5 in space, TVD-RK2 in pseudo-time,
three renormalization sweeps per call**.  (`LevelSetTrackerNew::State` defaults
to `HJWENO5_BIAS` + `TVD_RK1` + `normCount = LEVEL_SET_HALF_WIDTH`; `main.cpp`
overrides the temporal scheme and the count.)

This matters because OpenVDB distinguishes two fifth-order WENO variants, and
they are genuinely different operators
(`openvdb/openvdb/math/FiniteDifference.h:1079`, `:1178`):

    FD_WENO5    WENO5(xp3,xp2,xp1,xp0,xm1) - WENO5(xp2,xp1,xp0,xm1,xm2)
                  -- WENO reconstruction of the function, then differenced
    FD_HJWENO5  WENO5(xp3-xp2, xp2-xp1, xp1-xp0, xp0-xm1, xm1-xm2)
                  -- WENO applied to the divided differences (Hamilton-Jacobi form)

NanoVDB's `WenoStencil::normSqGrad` feeds `WENO5` with *differences*
(`nanovdb/math/Stencils.h:650-656`), i.e. it implements **HJ-WENO5**.  The two
stacks therefore agree on the scheme.  A scheme mismatch is *not* the
explanation for the tolerance gap below.

### Resolved: why OpenVDB and NanoVDB disagree at 5e-3

> **Priority: secondary.**  These are constant/precision discrepancies, not
> algorithmic ones.  They are recorded here so they are not rediscovered, and
> so that the two choices they force are made deliberately -- but they do not
> gate the high-level design, and nothing below should be read as a blocker.

The tandem harness uses two tolerances (`LevelSetTrackerNew.h:351-355` in the
`f8ab0ca` revision): `5e-5` with `USE_NANOVDB_IMPLEMENTATION_FOR_NORMGRAD`
(OpenVDB forced to use NanoVDB's `normSqGrad`) and `5e-3` without it.  Since the
schemes match, the residual has to come from the shared `WENO5` kernel.  The
algebra is identical line for line; **two things differ**:

**1. The regularization epsilon differs by 100x.**  Both compute
`eps = 1e-6 * scale2`, but:

* OpenVDB's `WENO5` defaults to `scale2 = 0.01f`
  (`openvdb/math/FiniteDifference.h:304`), and `D1<FD_HJWENO5>::difference`
  never passes one -- so `eps = 1e-8`, unconditionally, independent of `dx`.
* NanoVDB's `WENO5` defaults to `scale2 = 1.0` and carries the comment
  "openvdb uses scale2 = 0.01" (`nanovdb/math/Stencils.h:42`), so the divergence
  is deliberate and known.  Its `normSqGrad` member passes `mDx2` (= dx^2), but
  the reference's static overload is called as `normSqGrad(stencil, 1.f, 1.f)`
  (`Benchmark/src/Benchmark.cu:358`) -- so `eps = 1e-6`.

This is not cosmetic.  `eps` sets the floor of the WENO smoothness indicators
`beta`, and the ratio `eps/beta` decides how far the nonlinear weights drift from
the linear optimal weights.  For the HJ form the arguments are *differences* of
phi across one voxel, so with `|grad phi| ~ 1` they are `O(dx_world)` and
`beta = O(dx_world^2)`.  Scaling `eps` by `dx^2` (what NanoVDB's member version
does) is therefore dimensionally consistent for HJ-WENO, and OpenVDB's fixed
`1e-8` is a dx-independent magic number -- but **the reference's `dx2 = 1.f` gets neither**.
If `dx_world^2` approaches `1e-6`, `eps` starts to dominate `beta` and the scheme
degenerates toward plain fifth-order linear upwinding, silently losing the WENO
non-oscillatory property exactly where it is needed.

**2. The working precision differs.**  OpenVDB computes `C`, `eps`, `A1..A3` and
the final combination in **`double`** (`FiniteDifference.h:306-320`), then casts
down.  NanoVDB is templated: `WenoStencil<GridT, RealT = typename
GridT::ValueType>` (`nanovdb/math/Stencils.h:614`), so `WenoStencil<FloatGrid>`
does the entire weight computation in **`float`**.  The `A_k` are reciprocals of
fourth powers of small quantities -- the worst possible shape for single
precision.

Both hypotheses are cheaply testable in isolation (build the reference harness with
NanoVDB's `WENO5` instantiated at `RealT = double`; separately, pass
`scale2 = 0.01` / `dx_world^2` instead of `1.f`).  Doing so should say how much of
the `5e-3` is epsilon and how much is precision.

Decisions this forces before we bless NanoVDB's version as the reference:

* **What should `scale2` be?**  Fixed `0.01` (bug-compatible with OpenVDB),
  `dx^2` (dimensionally consistent for HJ-WENO, what NanoVDB's member version
  already does), or caller-supplied with no default?
* **What should `RealT` be?**  Defaulting it to `ValueType` means a `float` grid
  silently gets `float` WENO weights.  Defaulting to `double` for `float` grids
  costs registers in a GPU kernel but matches OpenVDB.  This should be an
  explicit choice, not an inherited default.

## 7. Cleanup required during promotion

Inherited from the reference and not to be carried into NanoVDB as-is:

* **Hardcoded default stream** — every launch passes `cudaStream_t(0)` and the
  driver ends in `cudaStreamSynchronize(0)` (`Benchmark/src/Benchmark.cu:459`).
  Upstream convention wants an explicit `cudaStream_t` parameter.
* **Hardcoded `UnifiedBuffer`**, including the RK temporary that the CUDA path
  never touches from the host — conceded in a TODO at
  `Benchmark/src/Benchmark.cu:441-448`.  `cuda::Buffer<T>` landed upstream in `0ab0a81e7`.
  Note this is a *regression introduced by the CPU port*: the earlier
  `benchmark-add-lateral-ratio @ f8ab0ca` revision used
  `BufferT = nanovdb::cuda::DeviceBuffer`, and `0b207e7` switched it to
  `UnifiedBuffer` purely so a future host path could share the allocation.  For
  a CUDA-only implementation, `f8ab0ca` is the better reference on this one
  point.  See `Benchmark/PROVENANCE.md`.
* **CPU scaffolding inside the CUDA path.**  Since only CUDA is in scope, the
  `ExecutionPolicy` template parameter comes off every entry point, and
  `Benchmark/src/Benchmark.cpp` (entirely `ExecutionPolicy::CPU`) has no counterpart in
  the promoted code.  Most pointedly, the CUDA specializations of
  `dilateActiveValues` and `pruneNarrowBand` contain a `getInstance().onCPU()`
  branch that can call the *host* VBM builder (`Benchmark/src/Benchmark.cu:144`,
  `:534`) -- a nominally-CUDA entry point with a runtime escape into host code.
* **Misnamed axis variables** in the dilate seeder (`Benchmark/src/Benchmark.cu:110-115`):
  `oldIdx_pX` is `offsetBy(0,0,1)` (+Z), `oldIdx_mZ` is `offsetBy(-1,0,0)` (-X),
  etc.  Behaviour is unaffected — it is an unordered scan of all six face
  neighbors — but the names are actively misleading.
* **No terminal `else`** in that same chain.  An `NN_FACE`-dilated voxel is
  guaranteed at least one active old face-neighbor, so it cannot miss today, but
  in library code that invariant deserves an assert rather than an
  uninitialized sidecar slot.
* **Singleton state.**  `Benchmark` is a singleton so that NanoVDB calls could be
  injected into an existing OpenVDB control flow without restructuring it.  A
  NanoVDB tool must not be.
* **`gridHandle.reset()` / `phiBuffer.clear()` before move-assign**
  (`Benchmark/src/Benchmark.cu:164-166`), worked around with
  "TODO: Remove after memory leak in move constructor is fixed".  Needs to be
  diagnosed rather than inherited.

## 8. Proposed order of work

1. ~~**Upstream the two static `WenoStencil` overloads**~~ — **DONE** (§5).
2. **Settle B1** -- write the boundary rule down as one specification.
3. **Settle A3 and A1** -- name and scope, since they fix the header's shape.
4. **The renormalizer itself**: IndexGrid + sidecar, owning grid/sidecar/scratch/
   VBM (§3.5), explicit stream, templated `BufferT`, HJ-WENO5 + TVD-RK2 only.
   Resolve B4 (reuse `gatherIndices` or re-derive) on the way in.
5. **`ex_renormalize_levelset_cuda` proper** (B6): analytic ground-truth cases
   plus the recovered OpenVDB cross-check.
6. **Unit test** in `nanovdb/unittest` (B5).
7. **Then, separately:** `track()` (dilate -> renormalize -> prune), which is
   where §3.5's replacement machinery first gets exercised.

Deferred indefinitely unless scope changes: the host path, interface
propagation, velocity extension, and the remaining scheme combinations.

## 9. Reference map

| What | Where |
|---|---|
| CUDA renormalization functor | `Benchmark/src/Benchmark.cu:245` |
| — VBM block decode | `Benchmark/src/Benchmark.cu:283` |
| — 19-point stencil gather | `Benchmark/src/Benchmark.cu:286-327` |
| — sign-extrapolation cascade | `Benchmark/src/Benchmark.cu:328-345` |
| — Euler step | `Benchmark/src/Benchmark.cu:347-355` |
| RK2 driver | `Benchmark/src/Benchmark.cu:436` |
| CUDA dilation + sidecar seeding | `Benchmark/src/Benchmark.cu:80`, `:108-116` |
| CUDA narrow-band prune | `Benchmark/src/Benchmark.cu:474` |
| Static WENO overloads (upstreamed) | `nanovdb/math/Stencils.h`, `WenoStencil` |
| Host `ExecutionPolicy::CPU` paths | `Benchmark/src/Benchmark.cpp` |
| OpenVDB/NanoVDB bridge + comparisons | `Benchmark/src/BenchmarkIO.cpp` |
| OpenVDB reference tracker (forked) | `Benchmark/include/LevelSetTrackerNew.h` |
| OpenVDB upstream `track()` | `openvdb/openvdb/tools/LevelSetTracker.h:305-316` |
| OpenVDB upstream Euler step | `openvdb/openvdb/tools/LevelSetTracker.h:637-642` |
| OpenVDB upstream `dt` constants | `openvdb/openvdb/tools/LevelSetTracker.h:521-522` |
| Out of scope: propagation | `Benchmark/include/LevelSetPropagate.h`, `Benchmark/include/VelocityExtension.h` |
