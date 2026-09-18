# Native NanoVDB Level-Set Renormalization — Design Notes

**Status:** design in progress.  Nothing in this directory is built yet; the
`Benchmark/` subtree is a verbatim reference import (see `Benchmark/PROVENANCE.md`).

**Branch:** `levelset-renormalization`, based on ASWF `master`.

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

## 4. Scope questions still open

* **Renormalization only, or all of `track()`?**  Renormalization is
  topology-preserving: one kernel, one temporary sidecar, no handle churn.
  `track()` drags in dilate -> prune, a VBM rebuild after every topology change,
  and grid-handle replacement.  Shipping renormalization first is a clean unit;
  `track()` is the natural follow-on.
* **How much of OpenVDB's feature matrix?**  Upstream has 5 spatial x 3 temporal
  schemes, masked `normalize(&mask)`, `TrimMode`, `dilate`/`erode`/`resize`.  the reference
  implements exactly one point of that space: WENO5 + TVD-RK2, unmasked, no
  trimming (its copy of the masked branch throws).  Mirror the
  `State`/scheme-dispatch API with one instantiation, or ship a narrow free
  function?
* **Host path in scope?**  Precedent in this tree (`DilateGrid.h` /
  `DilateGrid.cuh`) says host+device.  the reference's `ExecutionPolicy` templating was
  built for exactly that, and `normalizeLevelSet<ExecutionPolicy::CPU>` exists
  (`Benchmark/src/Benchmark.cpp:224`) — but the host WENO kernel itself is the
  unfinished part (`Benchmark/CPU_PORT_PLAN.md` Phase 2).
* **Where does the VBM live?**  Presumably owned by the operator and rebuilt on
  topology change.  `BlockWidthLog2 = 7` (128 active voxels per CUDA block) is
  currently a `Benchmark` constant.

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

1. **`Benchmark/include/Stencils.h`** — 30 diff lines against `nanovdb/math/Stencils.h`:
   two **static, array-based** overloads,
   `WenoStencil::normSqGrad(const ValueType* v, invDx2, dx2, iso)`
   (`Benchmark/include/Stencils.h:665`) and
   `WenoStencil::gradient(const ValueType* v, inv2Dx)`
   (`Benchmark/include/Stencils.h:710`).  They exist because a GPU kernel gathers the
   stencil into registers itself and cannot use the stencil object's accessor.
   This is the entire reason the reference shadows the upstream header (source commit
   `f8ab0ca`, "include local Stencils.h before upstream to win guard race").
   **Upstreaming these two functions dissolves the hack** and is the natural
   first commit — small, independently useful, no design commitments.
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

### Open item inherited from the reference

The tandem harness uses two different tolerances
(`LevelSetTrackerNew.h:351-355` in the `f8ab0ca` revision):

* `5e-5` with `USE_NANOVDB_IMPLEMENTATION_FOR_NORMGRAD` — OpenVDB forced to use
  NanoVDB's `WenoStencil::normSqGrad`;
* `5e-3` without it — OpenVDB using its own `ISGradientNormSqrd<WENO5_BIAS>`.

Both are nominally the same WENO5 + Godunov scheme, so `5e-3` is larger than
floating-point reassociation alone should explain.  Encouragingly, this also
bounds the *boundary-condition* contribution at `5e-5` on the reference model — i.e.
the sign-extrapolation heuristic agrees closely with OpenVDB's tile-value
treatment there.  But if NanoVDB's WENO is to be the reference implementation,
we should understand which of the two is right before blessing it.

## 7. Cleanup required during promotion

Inherited from the reference and not to be carried into NanoVDB as-is:

* **Hardcoded default stream** — every launch passes `cudaStream_t(0)` and the
  driver ends in `cudaStreamSynchronize(0)` (`Benchmark/src/Benchmark.cu:459`).
  Upstream convention wants an explicit `cudaStream_t` parameter.
* **Hardcoded `UnifiedBuffer`**, including the RK temporary that the CUDA path
  never touches from the host — conceded in a TODO at
  `Benchmark/src/Benchmark.cu:441-448`.  `cuda::Buffer<T>` landed upstream in `0ab0a81e7`.
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

1. Upstream the two static `WenoStencil` overloads into `nanovdb/math/Stencils.h`;
   drop the shadowed header.
2. Settle §3.3 — write the boundary rule down as one specification.
3. `nanovdb/tools/cuda/LevelSetRenormalize.cuh` (name TBD): renormalization only,
   IndexGrid + sidecar, explicit stream, templated `BufferT`, owned VBM.
4. `ex_renormalize_levelset_cuda` proper: analytic ground-truth cases plus the
   OpenVDB cross-check.
5. Unit test in `nanovdb/unittest`.
6. Then, separately: `track()` (dilate -> renormalize -> prune) and the host path.

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
| Static WENO overloads (the delta) | `Benchmark/include/Stencils.h:665`, `:710` |
| Host `ExecutionPolicy::CPU` paths | `Benchmark/src/Benchmark.cpp` |
| OpenVDB/NanoVDB bridge + comparisons | `Benchmark/src/BenchmarkIO.cpp` |
| OpenVDB reference tracker (forked) | `Benchmark/include/LevelSetTrackerNew.h` |
| OpenVDB upstream `track()` | `openvdb/openvdb/tools/LevelSetTracker.h:305-316` |
| OpenVDB upstream Euler step | `openvdb/openvdb/tools/LevelSetTracker.h:637-642` |
| OpenVDB upstream `dt` constants | `openvdb/openvdb/tools/LevelSetTracker.h:521-522` |
| Out of scope: propagation | `Benchmark/include/LevelSetPropagate.h`, `Benchmark/include/VelocityExtension.h` |
