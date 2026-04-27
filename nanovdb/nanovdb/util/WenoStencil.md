# WenoStencil — Single-Source Scalar/SIMD WENO5 Stencil Value Container

Design reference for `nanovdb/nanovdb/util/WenoStencil.h`.  Captures the
rationale behind the templated-on-lane-width class, the out-of-band
extrapolation algorithm, the Godunov norm-square-gradient method, and the
relationship to the broader Phase-2 pipeline sketched in
`BatchAccessor.md §11`.

---

## 1.  Motivation

The WENO5 CPU pipeline (`BatchAccessor.md §11`) assembles, per voxel
batch, a 19-tap value matrix with per-tap activity flags:

```
float  values[Ntaps][W]     -- real sidecar value, or background for OOB lanes
bool   isActive[Ntaps][W]   -- true iff the tap voxel is in the narrow band
```

Downstream phases are (1) **extrapolation** to repair out-of-band lanes
with sign-corrected background, and (2) **WENO5 arithmetic** (the fifth-
order upwind Godunov norm-square-gradient) to produce a per-voxel
`|∇φ|²` scalar.

`WenoStencil<W>` encapsulates the compute state and both operations:

1. Storage of the 19-tap × W-lane Simd values + activity masks.
2. `extrapolate()` — ascending-|Δ| cascade, hardcoded tap pairs.
3. `normSqGrad()` — six axial WENO5 reconstructions + Godunov combinator.
4. A single-source pattern that keeps GPU (W=1, per-thread) code
   textually identical to CPU SIMD (W>1) code — only the compile-time
   lane width changes.

---

## 2.  Single-source scalar/SIMD design

### 2.1  `Simd<T, W>` as the storage type

Storage is `Simd<float, W>` / `SimdMask<float, W>` directly — *not* raw
`float[W]` / `bool[W]` arrays:

```cpp
template<int W = 1>
class WenoStencil
{
public:
    using FloatV = util::Simd    <float, W>;
    using MaskV  = util::SimdMask<float, W>;

    FloatV values  [SIZE];
    MaskV  isActive[SIZE];

    float mDx2{1.f};       // dx²      — scalar, broadcast on use
    float mInvDx2{1.f};    // 1 / dx²  — scalar, broadcast on use
    /* ... extrapolate(), normSqGrad() ... */
};
```

At W=1 the Simd types collapse to plain `float` / `bool` under the
array backend.  At W>1 they are the backend-native SIMD type (stdx or
array wrapper).  Memory layout at W>1 is identical to `float[SIZE][W]`
directly, so there is no storage-cost penalty — just a cleaner
compute-side type.

### 2.2  Why first-class Simd storage (vs raw C arrays + `addr()` bridge)

An earlier version used `std::conditional_t<W==1, float, float[W]>` as
the element type, with an `addr()` helper to normalize the W=1 scalar-
reference vs W>1 array-decay at every SIMD load/store site.  That
design was rejected in favour of Simd-typed storage for three reasons:

- **W=1 ceremony**.  Approach 1 forced the scalar case to read
  `FloatV val(addr(mValues[k]), element_aligned)` — loading a
  `Simd<float, 1>` from a scalar reference.  On the CUDA per-thread
  path (where the scalar case is the *production* pipeline, not a
  degenerate convenience) this ceremony survives into every
  `__hostdev__` method that reads the stencil.  Under Simd-typed
  storage, W=1 reads `FloatV val = values[k]` — pure scalar code.

- **Arithmetic boundary ceremony**.  Approach 1 made `extrapolate()`
  and a prospective `normSqGrad()` bracket every read/write with an
  explicit Simd load or store.  With Simd-typed storage, the
  arithmetic reads as if scalar (`values[k] = util::where(...)`) and
  the load/store boundary moves out to the caller where the Simd
  values meet raw fill-side buffers.

- **Symmetric, explicit boundary placement**.  The caller already owns
  the fill-side scalar-scatter loop (sidecar `sidecar[idx]` gathers are
  inherently scalar-indexed per lane).  Making the array→Simd
  conversion an explicit caller-side step (`FloatV(raw_values[k],
  element_aligned)`) preserves that ownership — the arithmetic class
  doesn't care where its data came from.

### 2.3  Scalar runtime constants, broadcast on use

`mDx2` and `mInvDx2` stay plain `float` at every W.  They are
broadcast to `FloatV` at the point of use inside `normSqGrad()`:

```cpp
return FloatV(mInvDx2) * detail::GodunovsNormSqrd<FloatV, MaskV>(...);
```

`vbroadcastss` is free on x86 (folds into the FMA consumer); identity
at W=1.  Storing these as `Simd<float, W>` instead would cost 64
bytes × 2 of storage and hold two YMM registers across the entire
kernel lifetime for no benefit.

### 2.4  Caller-owned fill-side buffers

The class has **no** fill-side storage — no `mValues`/`mIsActive` raw C
arrays.  Callers own whatever shape of raw data is natural for them.
For the CPU SIMD case that's typically a pair of stack-local
`alignas(64)` C arrays sized `[SIZE][W]`; for the CUDA per-thread case
no intermediate buffer is needed at all.

This preserves the arithmetic class's purity and gives callers flex-
ibility — a different Phase-2 path (e.g. a future hardware-gather
fill) can populate the stencil using whatever pattern fits, without
the class having to expose a "fill API" that bakes in one shape.

---

## 3.  Extrapolation semantics

### 3.1  The out-of-band problem

For a narrow-band SDF, only the center tap `<0,0,0>` is guaranteed to
be in the active narrow band.  Every other tap may land outside the
band for some lanes — for those lanes the sidecar fill writes
`values[k][lane] = sidecar[0]` (magnitude of the background, since the
sidecar builder pre-sets slot 0 to `floatGrid.background()`) and
`isActive[k][lane] = false`.

Applying WENO5 arithmetic directly to the unsigned `|background|`
magnitude produces wrong gradients at the band boundary: the
reconstructed field would not track the sign of the underlying signed-
distance function.  The standard fix is to **extrapolate from the next
inner tap's sign**:

```
if (!isActive[k][lane])
    values[k][lane] = copysign(|background|, values[innerTap][lane])
```

The `|background|` magnitude is preserved; the sign is copied from
whichever "inner" tap (one step closer to center along the same axis)
best represents which side of the surface the lane belongs to.

### 3.2  Inner-tap cascade — ascending |Δ| order

| Outer tap |Δ| | Inner tap (source of sign) |
|---|---|
| `<±1,0,0>`, `<0,±1,0>`, `<0,0,±1>` | center `<0,0,0>` |
| `<±2,0,0>`, `<0,±2,0>`, `<0,0,±2>` | `<±1,0,0>`, `<0,±1,0>`, `<0,0,±1>` |
| `<±3,0,0>`, `<0,±3,0>`, `<0,0,±3>` | `<±2,0,0>`, `<0,±2,0>`, `<0,0,±2>` |

Processing taps in ascending-|Δ| order guarantees the inner tap is
already resolved (real value or previously extrapolated) when the
outer tap is processed.  Sign propagation through a |Δ|=1 → |Δ|=2 →
|Δ|=3 chain is transitive — no special casing.

### 3.3  The `kPairs[]` table

The inner-tap relationship is WENO5-specific and hardcoded as a static
table inside the class:

```cpp
static constexpr int kPairs[18][2] = {
    // |Δ|=1 (inner = center, idx 0)
    {3,0},{4,0},{9,0},{10,0},{15,0},{16,0},
    // |Δ|=2 (inner = |Δ|=1 on same axis)
    {2,3},{5,4},{8,9},{11,10},{14,15},{17,16},
    // |Δ|=3 (inner = |Δ|=2 on same axis)
    {1,2},{6,5},{7,8},{12,11},{13,14},{18,17}
};
```

Indices match the `WenoStencil<W>::Taps` tuple defined in
`WenoStencil.h` (same ordering as `WenoPt<i,j,k>::idx` in
`nanovdb/math/Stencils.h`).  Center tap (idx 0) is not processed —
assumed always in-band.

**Why hardcoded, not template-derived:**  a generic scheme would walk
`Taps` at compile time and derive inner-tap indices from |Δ| and axis
alignment.  For a single stencil the table is 18 entries, reads
directly, and makes the cascade ordering self-documenting.  Worth
revisiting if we add Weno7 or other axis-aligned WENO variants.

### 3.4  `extrapolate()` implementation

```cpp
template<int W>
void WenoStencil<W>::extrapolate(float absBackground)
{
    const FloatV absBg(absBackground);
    const FloatV zero (0.f);

    for (int p = 0; p < kNumPairs; ++p) {
        const int k      = kPairs[p][0];
        const int kInner = kPairs[p][1];

        // copysign(absBg, inner): +absBg if inner >= 0, else -absBg.
        const MaskV  isNegInner = zero > values[kInner];
        const FloatV extrap     = util::where(isNegInner, -absBg, absBg);

        // Active lanes keep their own value; inactive take the extrapolated sign-corrected background.
        values[k] = util::where(isActive[k], values[k], extrap);
    }
}
```

No `addr()`.  No `element_aligned`.  Reads `values[]` as Simd,
operates as Simd, writes Simd — the kernel body never drops to the
underlying scalar/array representation.

**Per-pair cost (W=16, AVX2):**

| Op | Cycles (est.) |
|----|--------------:|
| compare `zero > values[kInner]` | 1 |
| `where(isNegInner, -absBg, absBg)` (vblendvps) | 1 |
| `where(isActive[k], values[k], extrap)` (mask convert + vblendvps) | 1–2 |
| **≈ 4 cycles / pair** (values[] register-resident) |

Total: 18 pairs × ~4 cycles = ~72 cycles per call — lower than the
Approach-1 estimate of ~7 cycles/pair, because we no longer do the
explicit per-pair load.  The Simd values live in YMM registers across
the pair loop.  Measured end-to-end cost (sidecar-stencil-extrap minus
sidecar-stencil): ~0.14–0.19 ns/voxel on 24 threads.

---

## 4.  Godunov norm-square-gradient

### 4.1  Semantics — tracking the ground-truth scalar

`nanovdb::math::WenoStencil<GridT>::normSqGrad(isoValue)` in
`nanovdb/math/Stencils.h` is the ground-truth scalar reference.  Its
body:

```cpp
const ValueType* v = mValues;
const RealT
    dP_xm = WENO5<RealT>(v[2]-v[1], v[3]-v[2], v[0]-v[3], v[4]-v[0], v[5]-v[4], mDx2),
    dP_xp = WENO5<RealT>(v[6]-v[5], v[5]-v[4], v[4]-v[0], v[0]-v[3], v[3]-v[2], mDx2),
    dP_ym = ..., dP_yp = ..., dP_zm = ..., dP_zp = ...;
return mInvDx2 * GodunovsNormSqrd(v[0] > isoValue, dP_xm, dP_xp, dP_ym, dP_yp, dP_zm, dP_zp);
```

`WenoStencil<W>::normSqGrad(iso)` is a line-for-line transliteration of
the same body, with three adaptations:

1. `v = values` (Simd-typed storage, not `mValues` scalar array).
2. Local `dP_*` are `FloatV` rather than `RealT`.
3. The final `mInvDx2 * ...` multiplication broadcasts the scalar
   `mInvDx2` to `FloatV` (via `FloatV(mInvDx2)`); at W=1 this is a
   no-op.

### 4.2  `WENO5<T>` — generic over scalar and Simd

The six axial reconstructions are driven by a single free-function
template `nanovdb::detail::WENO5<T, RealT=T>` that mirrors
`nanovdb::math::WENO5<ValueType, RealT>` exactly (Shu ICASE
smoothness indicators, 0.1/0.6/0.3 linear weights, static_cast at the
end replaced by the trailing division).  Structure is the same; only
the literal constants are wrapped in `RealT(...)` constructors to
broadcast at W>1.  Lives in `nanovdb::detail` to keep the naming
convention close to the ground-truth without colliding with the
existing `nanovdb::math::WENO5`.

### 4.3  `GodunovsNormSqrd<T, MaskT>` — `where`-based, no control flow

The scalar ground-truth has a runtime `if (isOutside) { … } else { … }`.
The generic-T version computes both branches unconditionally and
blends via `util::where`:

```cpp
template<typename T, typename MaskT>
inline T GodunovsNormSqrd(MaskT isOutside,
                          T dP_xm, T dP_xp, T dP_ym, T dP_yp, T dP_zm, T dP_zp)
{
    const T zero(0.f);
    const T outside = max(Pow2(max(dP_xm, zero)), Pow2(min(dP_xp, zero)))   // (dP/dx)²
                    + max(Pow2(max(dP_ym, zero)), Pow2(min(dP_yp, zero)))
                    + max(Pow2(max(dP_zm, zero)), Pow2(min(dP_zp, zero)));
    const T inside  = max(Pow2(min(dP_xm, zero)), Pow2(max(dP_xp, zero)))
                    + max(Pow2(min(dP_ym, zero)), Pow2(max(dP_yp, zero)))
                    + max(Pow2(min(dP_zm, zero)), Pow2(max(dP_zp, zero)));
    return where(isOutside, outside, inside);
}
```

At `T=float, MaskT=bool` this compiles to scalar code with both
branches speculatively evaluated — slightly slower than the
ground-truth's branchy form for scalar workloads in isolation, but
identical correctness.  At `T=Simd<float,W>, MaskT=SimdMask<float,W>`
the branches are unconditional SIMD compute plus a `vblendvps` —
no lane-divergent branches, no scalarisation.

Per-lane cost of the full `normSqGrad`:

| Phase | Ops |
|-------|-----|
| 6× axial WENO5 | ~60 mul/add/fma + 6× reciprocals (the `0.N / Pow2(…)` terms) |
| Godunov: 12× max/min + 12× mul + 5× add, both branches | ~29 ops |
| Blend + final multiply by FloatV(mInvDx2) | 2 ops |

Roughly ~100 arithmetic ops per voxel per `normSqGrad` call.  At W=16
AVX2 that's ~100 / 16 ≈ 6.3 cycles/voxel × some FMA-throughput factor
— call it 2 ns/voxel single-threaded, 0.1 ns/voxel on 24 threads.
(To be validated by measurement; see §7.1 Future work.)

---

## 5.  API usage

### 5.1  CPU SIMD — caller-owned raw buffers, explicit load

```cpp
// Caller owns its scalar-scatter target.
alignas(64) float raw_values[SIZE][W];
alignas(64) bool  raw_active[SIZE][W];

nanovdb::WenoStencil<W> stencil(dx);

// Fill — pure scalar stores, guaranteed fast codegen on all backends.
for (int k = 0; k < SIZE; ++k) {
    for (int i = 0; i < W; ++i) {
        const uint64_t idx = /* sidecar index for tap k, lane i */;
        raw_values[k][i] = sidecar[idx];
        raw_active[k][i] = (idx != 0);
    }
}

// Bridge — one SIMD load per tap.
for (int k = 0; k < SIZE; ++k) {
    stencil.values  [k] = FloatV(raw_values[k], util::element_aligned);
    stencil.isActive[k] = MaskV (raw_active[k], util::element_aligned);
}

// Arithmetic — reads/writes stencil.values[] as Simd in place.
stencil.extrapolate(std::abs(sidecar[0]));
FloatV normSq = stencil.normSqGrad(/* iso = */ 0.f);

// Simd → scalar bridge at the output side if downstream consumers are scalar.
alignas(64) float normSq_lanes[W];
util::store(normSq, normSq_lanes, util::element_aligned);
```

### 5.2  CUDA scalar — no intermediate buffer

```cpp
nanovdb::WenoStencil<1> stencil(dx);
for (int k = 0; k < SIZE; ++k) {
    const uint64_t idx = gather_index_for_tap(k);
    stencil.values  [k] = sidecar[idx];
    stencil.isActive[k] = (idx != 0);
}

stencil.extrapolate(fabsf(sidecar[0]));
float normSq = stencil.normSqGrad();
```

`FloatV` is `float` at W=1; direct scalar assignment.  `MaskV` is
`bool`.  No raw buffers, no `element_aligned`, no load loops — the
per-thread path reads as pure scalar arithmetic.

### 5.3  Compile-time named-tap access

```cpp
constexpr int ctr = WenoStencil<W>::tapIndex<0, 0, 0>();
FloatV centerValue = stencil.values[ctr];

constexpr int xm3 = WenoStencil<W>::tapIndex<-3, 0, 0>();
FloatV xm3Value   = stencil.values[xm3];
```

`tapIndex<DI,DJ,DK>()` forwards to a private static `findTap` helper
inside `WenoStencil<W>`, static-asserting at compile time that the
requested tap exists in the `Taps` tuple.

---

## 6.  Ownership boundaries

```
┌───────────────────────────────────────────────────────────────────┐
│ Caller                                                            │
│   alignas(64) float raw_values[SIZE][W];    ← fill-side buffer    │
│   alignas(64) bool  raw_active[SIZE][W];                          │
│                                                                   │
│   <scalar scatter fill>                                           │
│                                                                   │
│   for k: stencil.values[k]   = FloatV(raw_values[k], ...);        │
│          stencil.isActive[k] = MaskV (raw_active[k], ...);        │
│   ═══════════════════════════════════════════════════ Simd border │
├───────────────────────────────────────────────────────────────────┤
│ WenoStencil<W>                                                    │
│   FloatV values  [19];   MaskV isActive[19];                      │
│   float  mDx2, mInvDx2;                                           │
│   extrapolate() / normSqGrad() — Simd-in / Simd-out, pure compute │
├───────────────────────────────────────────────────────────────────┤
│ Caller                                                            │
│   ═══════════════════════════════════════════════════ Simd border │
│   util::store(normSq, normSq_lanes, util::element_aligned);       │
│   <per-lane scalar write to output sidecar>                       │
└───────────────────────────────────────────────────────────────────┘
```

Array↔Simd bridges exist only at the two explicit boundaries where
scalar-indexed I/O meets SIMD-parallel compute.  Inside `WenoStencil`
everything is Simd; outside the class the caller chooses whatever
scalar pattern fits its source/sink.

---

## 7.  Future work

### 7.1  Measurement — lock in the perf numbers

Reconstruct()-path (normSqGrad) cost hasn't been measured yet.  Next
step: add a `sidecar-stencil-normsqgrad` benchmark pass in
`ex_narrowband_stencil_cpu` to drive normSqGrad to completion on
taperLER.vdb; compare against `sidecar-stencil-extrap` (which writes
the tap-sum instead of normSqGrad) to isolate the Phase-3 arithmetic
cost.

### 7.2  Alternative stencils

If/when Weno7 or a non-axis-aligned stencil is needed, the class
would specialise on a stencil-policy template parameter rather than
hardcoding the 19-tap WENO5 shape:

```cpp
template<typename StencilPolicy, int W>
class AxisAlignedStencil { /* derive kPairs at compile time */ };
```

The `kPairs` table would be generated from `StencilPolicy::Taps` via a
constexpr pass that finds, for each tap, the same-axis neighbour with
|Δ| = |tap.Δ| − 1.  Not needed until a second axis-aligned stencil
exists.

---

## 8.  Relationship to other design docs

- **`BatchAccessor.md §11`** — the broader Phase-2/3 pipeline plan
  (VBM decode → sidecar assembly → extrapolation → WENO arithmetic
  → write-back).  `WenoStencil<W>` implements the extrapolation and
  (now) the WENO arithmetic steps; the storage carries data across
  from sidecar-assembly.
- **`StencilAccessor.md`** — Phase-1 accessor (batched uint64 index
  gather).  `StencilAccessor` fills `mIndices[SIZE][W]`; callers
  consume those indices (via `sidecar[idx]` in their fill loops) and
  populate `WenoStencil<W>::values[]` / `isActive[]`.
- **`nanovdb/math/Stencils.h`** — the scalar ground-truth for WENO5
  and Godunov.  `WenoStencil<W>::normSqGrad()` is a line-for-line
  transliteration of `nanovdb::math::WenoStencil<GridT>::normSqGrad()`
  to generic-T form.
