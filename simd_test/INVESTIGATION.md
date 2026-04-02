# liftToSimd Vectorization Investigation

This document captures the design idea, experiments, findings, and open questions
from an in-progress investigation into auto-vectorizing a scalar stencil kernel
through a generic SIMD lifting abstraction.  It is written as a reference for
resuming the investigation in a future session.

---

## 1. Motivation

The VoxelBlockManager CPU port (branch `vbm-cpu-port`) processes voxels in batches
of `SIMDw = 16` (one AVX2 register width of uint16_t).  For each batch the same
stencil computation is applied to every lane.  The goal is to write the stencil
physics **once** as a scalar, `__hostdev__`-compatible lambda (usable unmodified on
the GPU), and automatically derive an auto-vectorized CPU batch kernel from it.

---

## 2. The `liftToSimd` Pattern

### 2a. Core Idea

A scalar kernel has signature:

```cpp
ScalarTupleOut kernel(ScalarTupleIn);
```

where `ScalarTupleIn = std::tuple<T0, T1, ..., TN>` and `ScalarTupleOut = std::tuple<U0, ...>`.

The SIMD version replaces every `T` in the tuple with `std::array<T, W>`, giving an
SoA (struct-of-arrays) layout:

```cpp
SimdTupleIn  = std::tuple<std::array<T0,W>, std::array<T1,W>, ..., std::array<TN,W>>
SimdTupleOut = std::tuple<std::array<U0,W>, ...>
```

`liftToSimd<W>(kernel)` returns a lambda that loops over lanes 0..W-1, extracts the
i-th element from each input array (forming a `ScalarTupleIn`), calls `kernel`, and
writes the result back into the i-th slot of each output array.  The loop is the
auto-vectorization target.

### 2b. Infrastructure

```cpp
// ToSimdTuple<tuple<Ts...>, W>::type = tuple<array<Ts,W>...>
template<typename TupleT, int W> struct ToSimdTuple;

// extractSlice: return tuple of the i-th elements from a tuple-of-arrays
template<typename SimdTupleT, std::size_t... Is>
auto extractSlice(const SimdTupleT& t, int i, std::index_sequence<Is...>);

// storeSlice: write a scalar tuple into the i-th slot of a SIMD tuple
template<typename SimdTupleT, typename ScalarTupleT, std::size_t... Is>
void storeSlice(SimdTupleT& t, int i, const ScalarTupleT& s, std::index_sequence<Is...>);

template<int W, typename ScalarFn>
auto liftToSimd(ScalarFn f) {
    return [f](const auto& simdIn, auto& simdOut) {
        constexpr auto inSize  = std::tuple_size_v<std::decay_t<decltype(simdIn)>>;
        constexpr auto outSize = std::tuple_size_v<std::decay_t<decltype(simdOut)>>;
        for (int i = 0; i < W; i++) {
            auto scalarIn  = extractSlice(simdIn,  i, std::make_index_sequence<inSize>{});
            auto scalarOut = f(scalarIn);
            storeSlice(simdOut, i, scalarOut, std::make_index_sequence<outSize>{});
        }
    };
}
```

### 2c. Key Requirement: `__attribute__((noinline))` Wrapper

The vectorization loop must live inside a `__attribute__((noinline))` function.
Without this, GCC constant-folds the entire computation (because the test's input
data is compile-time computable) and emits no packed instructions at all, making it
appear that vectorization failed when it actually never ran.

---

## 3. Kernel Under Test: WENO5 `normSqGrad`

The scalar kernel computes the Godunov upwind norm-squared gradient using WENO5
differences, matching `WenoStencil::normSqGrad` from `nanovdb/math/Stencils.h`.

**Inputs**: 19 floats `v0..v18` representing the center voxel and ±3 neighbors along
each axis (same layout as `WenoPt<i,j,k>::idx`).

**Computation**:
1. Six WENO5 calls → six upwind differences `dP_xm, dP_xp, dP_ym, dP_yp, dP_zm, dP_zp`
2. `godunovsNormSqrd(v0 > isoValue, dP_xm, ..., dP_zp)` → Godunov norm-squared
3. Scale by `invDx2`

**Type aliases**:
```cpp
using WenoIn  = std::tuple<float,float,...>;  // 19 floats
using WenoOut = std::tuple<float>;
constexpr int W = 16;
using WenoSimdIn  = ToSimdTuple<WenoIn,  W>::type;
using WenoSimdOut = ToSimdTuple<WenoOut, W>::type;
```

---

## 4. Vectorization Experiments and Findings

Compiled with: `g++ -O3 -march=native -std=c++17 -fopt-info-vec-missed`
Platform: x86-64, GCC 13, AVX2.

### Experiment 1 — Simple Laplacian (baseline)

Kernel: `v0 - 6*v1 + v2 + ...` (7-point stencil, pure arithmetic).
**Result: VECTORIZES.**  Emits `ymm`-width `vfmadd*ps`, `vaddps`.

### Experiment 2 — Six WENO5 calls, sum all six dP terms

Kernel computes `dP_xm + dP_xp + ... + dP_zp` (no `godunovsNormSqrd`).
Each `weno5` call is ~12 fmas and 2 divisions — complex but purely arithmetic.
**Result: VECTORIZES.**  Confirms the WENO5 computation itself is not the blocker.

### Experiment 3 — Full `normSqGrad` with `v0 > isoValue`

Adds `godunovsNormSqrd(v0 > isoValue, ...)` to the pipeline.
GCC reports: `not vectorized: control flow in loop.` (loop line 41)
**Result: DOES NOT VECTORIZE.**

### Experiment 4 — Constant `true` instead of `v0 > isoValue`

Replace `v0 > isoValue` with compile-time `true` to rule out the comparison as the
blocker.
GCC still reports: `not vectorized: control flow in loop.`
**Result: DOES NOT VECTORIZE.**
Conclusion: the issue is inside `godunovsNormSqrd`, not at the call site.

### Experiment 5 — Replace `bool isOutside` with `float sign`, use `fmaxf`

Reformulated `godunovsNormSqrd` to take `float sign` (+1.f/-1.f) and use `fmaxf`
instead of `std::max` and ternary operators:
```cpp
float xm = fmaxf( sign * dP_xm, 0.f); xm *= xm;
```
GCC now reports a different error for the same loop:
```
not vectorized: no vectype for stmt:
    MEM <const struct array> [(const float &)simdIn_5(D)]._M_elems[_67]
    scalar_type: const float
```
**Result: DOES NOT VECTORIZE — but for a different reason.**
The `control flow` blocker is gone.  A new blocker appears: GCC's vectorizer cannot
find a vector type for the struct member access through `std::tuple`'s
implementation-detail inheritance chain (`_Tuple_impl`, `_Head_base`).  The
`(const float &)simdIn_5(D)` in the GIMPLE indicates the vectorizer is seeing the
parameter reference cast through the tuple internals and cannot determine the memory
access is stride-1.

---

## 5. Current Blockers (in priority order)

### Blocker A: `std::tuple` struct indirection

`std::get<k>(simdIn)[i]` for fixed k, varying i, is a stride-1 access into one of
the 19 contiguous `std::array<float,16>` members of the tuple.  GCC's vectorizer
fails to prove this because `std::tuple` in libstdc++ uses recursive inheritance
(`_Tuple_impl<N, Ts...> : _Tuple_impl<N+1, Ts...>, _Head_base<N, T>`), and the
GIMPLE representation of member access through that chain is too opaque for the
vectorizer's alias analysis.

**Hypothesis**: caching `.data()` pointers for each tuple element outside the hot
loop — so the loop only sees `inPtrs[k][i]` (simple indirect load) — may allow the
vectorizer to prove stride-1 access.  This would require reworking `extractSlice`
and `storeSlice` to operate on pointer arrays rather than going through `std::get`.

### Blocker B: `std::max` / ternary in `godunovsNormSqrd`

Even before reaching the struct-access issue, the ternary-based `std::max(a, b)` in
`godunovsNormSqrd` generates control flow IR that blocks vectorization.  Using
`fmaxf` (which maps to a hardware `maxss`/`maxps` instruction) removes this blocker.
The current file keeps `std::max` / `bool isOutside` for readability and correctness;
any vectorization-capable reformulation will need `fmaxf` or equivalent.

---

## 6. Proposed Next Steps

### Step 1 — Pointer-cache approach in `liftToSimd`

Before the hot loop, extract pointers to each tuple element's underlying array:

```cpp
// Before loop:
using ElemT = float;  // known for homogeneous tuples
const ElemT* inPtrs[inSize];
apply to index_sequence: inPtrs[Is] = std::get<Is>(simdIn).data();

// In loop body:
// access: inPtrs[k][i] — provably stride-1 for fixed k, varying i
```

Re-run `fopt-info-vec-missed` to see if the struct-access blocker disappears.

### Step 2 — Combine with `fmaxf` in `godunovsNormSqrd`

Once the struct-access blocker is cleared, reinstate the `fmaxf` / `float sign`
formulation and check whether the full `normSqGrad` kernel vectorizes.

### Step 3 — Clang comparison

Compile the same test with `clang++ -O3 -march=native -Rpass=loop-vectorize` to
determine whether this is a GCC-specific limitation or a fundamental IR issue.
Clang's vectorizer handles struct member accesses differently and may succeed where
GCC fails.

### Step 4 — Restore `v0 > isoValue`

Once the constant-sign version vectorizes, replace `1.f` with
`(v0 > isoValue) ? 1.f : -1.f` at the call site.  This introduces a VCMPPS+BLENDVPS
at the call site but no branching inside the arithmetic, which the vectorizer should
handle as a blend.

### Step 5 — Consider alternative abstraction: `const ValueType*` kernel

`StencilGather.md §4a` already specifies the kernel lambda signature as
`std::array<ValueType,K>(const ValueType* u)` (raw pointer, not tuple).
If the tuple path proves too resistant to auto-vectorization, the SIMD lift can be
reformulated over flat `float[N][W]` SoA arrays instead.  The `liftToSimd` idea
survives — the tuple input/output types would be replaced by flat arrays — but the
scalar lambda signature changes slightly.

---

## 7. File Reference

| File | Purpose |
|------|---------|
| `simd_test/lift_test.cpp` | Self-contained test: `liftToSimd` infrastructure + WENO5 normSqGrad kernel |
| `nanovdb/nanovdb/math/Stencils.h` | Original `weno5`, `GodunovsNormSqrd`, `WenoStencil::normSqGrad` |
| `nanovdb/nanovdb/examples/ex_voxelBlockManager_host_cuda/StencilGather.md` | Per-block stencil gather design doc (kernel lambda spec, CPU batch strategy) |
| `nanovdb/nanovdb/tools/VoxelBlockManager.h` | CPU VBM implementation |

Build command:
```sh
g++ -O3 -march=native -std=c++17 -fopt-info-vec-missed -o lift_test lift_test.cpp
```
