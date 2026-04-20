# WenoStencil — Single-Source Scalar/SIMD WENO5 Stencil Value Container

Design reference for `nanovdb/nanovdb/util/WenoStencil.h`.  Captures the
rationale behind the templated-on-lane-width class, the out-of-band
extrapolation algorithm, and the relationship to the broader Phase-2
pipeline sketched in `BatchAccessor.md §11`.

---

## 1.  Motivation

The WENO5 CPU pipeline (`BatchAccessor.md §11`) produces a per-batch
matrix of 19 tap values per voxel:

```
float  values[Ntaps][W]     -- real sidecar value, or background for OOB lanes
bool   isActive[Ntaps][W]   -- true iff the tap voxel is in the narrow band
```

The next pipeline phase consumes `values[tap][lane]` as 19 `Simd<float, W>`
rows for the WENO5 reconstruction arithmetic.  But before arithmetic,
out-of-band lanes must be repaired so that `values[tap][lane]` holds a
sensible float for every lane, not just the in-band ones.  That repair
is the **extrapolation** step.

`WenoStencil<W>` encapsulates:

1. Storage for the 19-tap × W-lane data + activity flags.
2. The extrapolation algorithm (ascending-|Δ| cascade, hardcoded tap pairs).
3. A pattern that keeps GPU (W=1) code textually identical to CPU SIMD
   (W>1) code — only the storage shapes and the Simd.h backend differ.

---

## 2.  Single-source scalar/SIMD design

### 2.1  The conditional_t storage trick

```cpp
template<int W = 1>
class WenoStencil
{
public:
    using ValueT = std::conditional_t<W == 1, float, float[W]>;
    using PredT  = std::conditional_t<W == 1, bool,  bool[W]>;

    alignas(64) ValueT mValues  [SIZE];
    alignas(64) PredT  mIsActive[SIZE];
    /* ... */
};
```

| W | `mValues` expands to | `mIsActive` expands to | Intended use |
|---|---|---|---|
| 1    | `float mValues[19]`         | `bool mIsActive[19]`        | GPU thread-per-voxel — 19 scalar registers |
| 16   | `float mValues[19][16]`     | `bool mIsActive[19][16]`    | CPU SIMD batch — 19 YMM-tiles in L1 |

Same declaration syntax, different expansion.  Memory layout at W>1 is
identical to writing `float mValues[SIZE][W]` directly — no performance
difference, just a cleaner scalar case at W=1.

### 2.2  Why the same source compiles to good scalar and SIMD code

`extrapolate()`'s body uses only `nanovdb::util::Simd<float, W>` primitives:

- load / store (ctor + free `store`)
- `operator>` (produces `SimdMask<float, W>`)
- unary `operator-`
- `where(mask, a, b)` — 3-arg blend

All of these exist in Simd.h's scalar degenerate (`fixed_size<1>` in the
stdx backend, 1-element array in the array backend).  At W=1 the
compiler inlines and collapses every operation to a plain scalar
instruction.  **One source body, two target ISAs, no `if constexpr`
branches on W inside the algorithm.**

---

## 3.  The `addr()` bridge

One asymmetry survives the `conditional_t` unification: at W=1,
`mValues[k]` is a scalar `float` value (not an array), so it does not
decay to `float*` — `Simd<float,1>::load(mValues[k], flags)` wouldn't
type-check.  At W>1, `mValues[k]` is a `float[W]` array and decays
naturally.

A private `addr()` helper papers over this in one place:

```cpp
static constexpr float* addr(ValueT& v) noexcept {
    if constexpr (W == 1) return &v; else return v;
}
```

Callers (inside `extrapolate()`) always write:

```cpp
FloatV val(addr(mValues[k]), element_aligned);
```

and get a uniform expression that works at any W.  There are four
overloads (`ValueT&` / `const ValueT&` / `PredT&` / `const PredT&`);
all are `constexpr` and compile to a no-op at W>1 and a trivial address
fetch at W=1.

**Alternative considered (rejected):** overloading `Simd<T,W>::load`
to accept `T&` at W=1.  Blocked by the stdx backend's type-alias
representation (`using Simd = stdx::fixed_size_simd`) — we can't add
member ctors to an alias.  The equivalent free-function workaround
turned out no shorter than the `addr()` helper and would have forced
Simd.h churn for a benefit scoped to one class.  See the Stage-3 design
exchange for the full discussion.

---

## 4.  Extrapolation semantics

### 4.1  The out-of-band problem

For a narrow-band SDF, only the center tap `<0,0,0>` is guaranteed to
be in the active narrow band.  Every other tap may land outside the
band for some lanes of a batch — for those lanes, `idx == 0` in the
sidecar fill, so `values[k][lane] = sidecar[0] = |background|` and
`isActive[k][lane] = false`.

Applying WENO5 arithmetic directly to the `|background|` magnitude
produces wrong gradients at the band boundary: the reconstructed field
would not track the sign of the underlying signed distance function.
The standard fix is to **extrapolate from the next inner tap's sign**:

```
if (!isActive[k][lane])
    values[k][lane] = copysign(|background|, values[innerTap][lane])
```

The `|background|` magnitude is preserved; the sign is copied from
whichever "inner" tap (one step closer to center along the same axis)
best represents which side of the surface this lane belongs to.

### 4.2  Inner-tap cascade — ascending |Δ| order

| Outer tap |Δ| | Inner tap (source of sign) |
|---|---|
| `<±1,0,0>`, `<0,±1,0>`, `<0,0,±1>` | center `<0,0,0>` |
| `<±2,0,0>`, `<0,±2,0>`, `<0,0,±2>` | `<±1,0,0>`, `<0,±1,0>`, `<0,0,±1>` |
| `<±3,0,0>`, `<0,±3,0>`, `<0,0,±3>` | `<±2,0,0>`, `<0,±2,0>`, `<0,0,±2>` |

Processing taps in ascending-|Δ| order guarantees the inner tap is
already resolved (real value or previously extrapolated) when the outer
tap is processed.  Sign propagation through a |Δ|=1 → |Δ|=2 → |Δ|=3
chain is automatic — no special casing.

### 4.3  The `kPairs[]` table

The inner-tap relationship is `Weno5Stencil`-specific and hardcoded as
a static table inside the class:

```cpp
static constexpr int kNumPairs = 18;
static constexpr int kPairs[kNumPairs][2] = {
    // |Δ|=1 (inner = center, idx 0)
    {3,0},{4,0},{9,0},{10,0},{15,0},{16,0},
    // |Δ|=2 (inner = |Δ|=1 on same axis)
    {2,3},{5,4},{8,9},{11,10},{14,15},{17,16},
    // |Δ|=3 (inner = |Δ|=2 on same axis)
    {1,2},{6,5},{7,8},{12,11},{13,14},{18,17}
};
```

Indices match the tuple ordering in `Weno5Stencil::Taps`
(`StencilAccessor.h`).  Center tap (idx 0) is not processed — assumed
always in-band.

**Why hardcoded, not template-derived:**  a generic scheme would walk
`Weno5Stencil::Taps` at compile time and derive inner-tap indices from
|Δ| and axis alignment.  For a single stencil (Weno5) this is
over-engineering: the table is 18 entries, reads directly, and makes
the cascade ordering self-documenting.  Worth revisiting if we add
Weno7 or other axis-aligned WENO variants.

---

## 5.  Extrapolate — implementation

```cpp
template<int W>
void WenoStencil<W>::extrapolate(float absBackground)
{
    using FloatV = nanovdb::util::Simd    <float, W>;
    using MaskV  = nanovdb::util::SimdMask<float, W>;

    const FloatV absBg(absBackground);
    const FloatV zero (0.0f);

    for (int p = 0; p < kNumPairs; ++p) {
        const int k      = kPairs[p][0];
        const int kInner = kPairs[p][1];

        const MaskV  active(addr(mIsActive[k]),      element_aligned);
        const FloatV val   (addr(mValues  [k]),      element_aligned);
        const FloatV inner (addr(mValues  [kInner]), element_aligned);

        // copysign(absBg, inner): +absBg if inner >= 0, else -absBg.
        const MaskV  isNegInner = zero > inner;
        const FloatV extrap     = where(isNegInner, -absBg, absBg);

        // Active lanes keep `val`; inactive lanes take `extrap`.
        const FloatV result = where(active, val, extrap);
        store(result, addr(mValues[k]), element_aligned);
    }
}
```

**Per-pair cost (W=16, AVX2):**

| Op | Cycles (est.) |
|----|--------------:|
| 3× load (mIsActive, mValues[k], mValues[kInner]) | 3 |
| `0 > inner` (vcmpltps + sign mask) | 1 |
| `where(isNegInner, -absBg, absBg)` (vblendvps) | 1 |
| `where(active, val, extrap)` (mask convert + vblendvps) | 1–2 |
| 1× store (mValues[k]) | 1 |
| **≈ 7 cycles / pair** |

Total: 18 pairs × 7 cycles = ~126 cycles per call.  Amortised over
W=16 lanes gives ~8 cycles/voxel, or ~2 ns/voxel on a 4 GHz core.

Measured overhead in `sidecar-stencil-extrap` pass on taperLER.vdb
(24 threads): **+4.5 ms / 31.8M voxels = 0.14 ns/voxel** end-to-end —
lines up with the per-core estimate divided by thread count (24×
speedup ≈ 2 / 24 ≈ 0.083 ns; measurement includes framing overhead).

**Skipping active lanes:**  the algorithm reads and computes for every
lane regardless of `isActive`.  For active lanes, `extrap` is computed
but then discarded by the final `where`.  This wasted work is cheaper
than a predicated-store alternative because:

- The SIMD blend is one instruction (`vblendvps`).
- Per-lane branching would serialize the batch.
- Active-fraction is high (~90% on narrow-band SDFs), so masked
  computation saves little even in the best case.

---

## 6.  API usage

### 6.1  Filling the stencil from sidecar indices

```cpp
WenoStencil<SIMDw> stencil;
StencilAccessor<BuildT, SIMDw, Weno5Stencil> acc(grid, ...);
acc.moveTo(leafIndex + batchStart, voxelOffset + batchStart);

for (int k = 0; k < WenoStencil<SIMDw>::size(); ++k) {
    for (int i = 0; i < SIMDw; ++i) {
        const uint64_t idx = acc.mIndices[k][i];
        stencil.mValues  [k][i] = sidecar[idx];     // sidecar[0] = background
        stencil.mIsActive[k][i] = (idx != 0);
    }
}
```

At W=1 (GPU per-thread) the same body would just drop the `[i]` index:

```cpp
stencil.mValues  [k] = sidecar[idx];
stencil.mIsActive[k] = (idx != 0);
```

### 6.2  Extrapolating

```cpp
stencil.extrapolate(std::abs(floatGrid.background()));
```

After this call, every `stencil.mValues[k][i]` holds either the real
sidecar value (for active lanes) or a sign-corrected `|background|`
(for inactive lanes).  `mIsActive[]` is no longer needed downstream.

### 6.3  Compile-time named-tap access

```cpp
constexpr int ctr = WenoStencil<SIMDw>::tapIndex<0, 0, 0>();
float centerValue = stencil.mValues[ctr][i];

constexpr int xm3 = WenoStencil<SIMDw>::tapIndex<-3, 0, 0>();
// ... etc, for WENO5 arithmetic
```

`tapIndex<DI,DJ,DK>()` forwards to `detail::findIndex` (shared with
`StencilAccessor`), static-asserting at compile time that the requested
tap exists in the Weno5Stencil::Taps tuple.

---

## 7.  Future work

### 7.1  WENO5 reconstruction method

The class's second substantive operation (not yet implemented) will be
the WENO5 arithmetic itself — a compile-time fold over the 19 tap
rows, producing three `Simd<float, W>` fluxes (one per axis) from the
fully-resolved `mValues` matrix.  Natural signature:

```cpp
struct Weno5Flux { FloatV dx, dy, dz; };
Weno5Flux reconstruct() const;
```

Adopting the same single-source structure: at W=1 the fluxes collapse
to scalars; at W>1 they are SIMD vectors.

### 7.2  Consolidate the Weno5Stencil policy

Currently `Weno5Stencil` (the tap-tuple policy struct) lives in
`StencilAccessor.h` and is shared with `WenoStencil<W>` via
`using Taps = Weno5Stencil::Taps`.  The policy is arguably a
Weno-specific definition and could move into `WenoStencil.h`;
`StencilAccessor.h` would then `#include <.../WenoStencil.h>` for the
policy.  Left as-is for this pass to minimise Stage-3 churn.

### 7.3  Alternative stencils

If/when Weno7 or a non-axis-aligned stencil is needed, the class would
specialise on a stencil-policy template parameter rather than hardcode
`Weno5Stencil`:

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

- **`BatchAccessor.md §11`** — the broader Phase-2 pipeline plan
  (VBM decode → sidecar assembly → extrapolation → WENO arithmetic →
  write-back).  WenoStencil<W> implements the "extrapolation" step and
  provides the storage that carries data from "sidecar assembly" into
  the future "WENO arithmetic" step.
- **`StencilAccessor.md`** — Phase-1 accessor (batched uint64 index
  gather).  StencilAccessor fills `mIndices[SIZE][W]`; WenoStencil
  consumes those indices (via `sidecar[idx]` in user code) and owns
  the per-lane float result.
- **`HaloStencilAccessor.md`** — speculative alternative that
  precomputes a dense float halo buffer; if that path is pursued,
  WenoStencil<W> would fill from the halo instead of from sidecar
  indices.  The extrapolation algorithm here transfers unchanged.
