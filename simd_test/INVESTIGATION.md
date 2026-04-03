# liftToSimd / Generic-T SIMD Vectorization Investigation

This document captures the design ideas, experiments, findings, and open questions
from an in-progress investigation into auto-vectorizing a scalar stencil kernel
for the VoxelBlockManager CPU port.  Written as a reference for resuming the
investigation in a future session.

---

## 1. Motivation

The VoxelBlockManager CPU port (branch `vbm-cpu-port`) processes voxels in batches
of `SIMDw = 16` (one AVX2 register width of uint16_t).  For each batch the same
stencil computation is applied to every lane.  The goal is to write the stencil
physics **once** as a scalar, `__hostdev__`-compatible function (usable unmodified on
the GPU), and automatically derive an auto-vectorized CPU batch kernel from it.

---

## 2. Approach A: `liftToSimd` Pattern (superseded)

### Core Idea

A scalar kernel with signature `ScalarTupleOut kernel(ScalarTupleIn)` is lifted to
W lanes by replacing every `T` in the tuple types with `std::array<T, W>` (SoA
layout).  A W-iteration loop extracts the i-th element from each input array, calls
the scalar kernel, and stores results back.  This loop is the auto-vectorization
target.

```cpp
template<int W, typename ScalarFn>
auto liftToSimd(ScalarFn f) {
    return [f](const auto& simdIn, auto& simdOut) {
        for (int i = 0; i < W; i++) {
            auto scalarIn  = extractSlice(simdIn,  i, ...);
            auto scalarOut = f(scalarIn);
            storeSlice(simdOut, i, scalarOut, ...);
        }
    };
}
```

### Outcome

Clang 18 vectorizes the unmodified kernel (with `std::max` and `bool isOutside`)
producing a full ymm path with a runtime alias check.  GCC 13 does not vectorize in
any attempted form (see §5).

### Why Superseded

1. The scalar kernel takes a tuple, not individual arguments, and cannot be
   templated on `T` directly — it is a separate code path from the GPU kernel.
2. Vectorization relies entirely on the auto-vectorizer seeing through the tuple
   extraction loop, which GCC cannot do.

---

## 3. Approach B: Generic-T Pattern (current)

### Core Idea

Write the kernel **once** as a template on its value type `T`:

- `T = float` → scalar path, `__hostdev__`-compatible, used on GPU per-thread
- `T = Simd<float, W>` → W-wide SIMD path, used on CPU per-batch

All arithmetic operators, `min`, `max`, and `where` are overloaded for both `float`
and `Simd<float, W>`, so the same source compiles correctly for both contexts with
zero `#ifdef`.

### `where()` — the key primitive

`bool isOutside ? a : b` cannot be used with a SIMD mask.  `where(mask, a, b)`
replaces it:

```cpp
// Scalar (T=float): plain ternary — GPU path
template<typename T> T where(bool mask, T a, T b) { return mask ? a : b; }

// SIMD (T=Simd<float,W>): lane-wise blend → VBLENDVPS, no branch
template<typename T, int W>
Simd<T,W> where(SimdMask<T,W> mask, Simd<T,W> a, Simd<T,W> b);
```

`v0 > T(isoValue)` deduces to `bool` when `T=float` and `SimdMask<float,W>` when
`T=Simd<float,W>`, so the `where()` call resolves correctly in both cases.

### Class hierarchy

`WENO5<T>` and `GodunovsNormSqrd<T, MaskT>` are free functions in `StencilKernel.h`,
mirroring their counterparts in `Stencils.h`.  The stencil data and compute methods
live in a two-level class hierarchy:

```
BaseStencilKernel<T, SIZE>      mValues[SIZE], mDx2, mInvDx2 — pure data
         |
WenoStencilKernel<T>            normSqGrad(), ... — pure compute
```

No grid coupling, no accessor, no `moveTo()`.  The VBM gather populates `mValues`
directly; `normSqGrad()` is then called on the populated kernel object.

### GPU / CPU call sites

```cpp
// GPU: one thread, scalar — fill from per-thread stencil gather
WenoStencilKernel<float> sk(dx);
for (int n = 0; n < 19; n++) sk[n] = gathered_scalar_values[n];
float result = sk.normSqGrad(isoValue);

// CPU: one batch, SIMD — fill from VBM batch gather
WenoStencilKernel<Simd<float,16>> sk(dx);
for (int n = 0; n < 19; n++) sk[n] = gathered_simd_values[n];
Simd<float,16> result = sk.normSqGrad(isoValue);
```

### Relationship to legacy WenoStencil

The existing `BaseStencil<Derived, SIZE, GridT>` / `WenoStencil<GridT>` hierarchy in
`Stencils.h` couples data storage to a grid accessor and a `moveTo()` cursor — a
sequential, single-threaded API incompatible with VBM batch processing.  The kernel
hierarchy is designed as its eventual replacement.  During transition, the legacy
classes can simply derive from the kernel classes to inherit the compute methods
without disruption.

NVCC's demand-driven template instantiation ensures `WenoStencilKernel<Simd<float,W>>`
is never compiled for device.

---

## 4. `nanovdb::util::Simd<T, W>` — two backends

`simd_test/Simd.h` (destined for `nanovdb/util/`) provides `Simd<T,W>`,
`SimdMask<T,W>`, `min`, `max`, and `where` with two interchangeable implementations
selected automatically at compile time.  Suppress Backend A with
`-DNANOVDB_NO_STD_SIMD` to force the fallback.

### Backend A: `std::experimental::simd` (C++26 / Parallelism TS v2)

Activated when `<experimental/simd>` is available,
`__cpp_lib_experimental_parallel_simd` is defined, and `NANOVDB_NO_STD_SIMD` is not
set.

`Simd<T,W>` and `SimdMask<T,W>` are **pure type aliases** for
`std::experimental::fixed_size_simd<T,W>` and
`std::experimental::fixed_size_simd_mask<T,W>`.  All arithmetic delegates to the
standard types; the compiler emits native vector instructions without relying on the
auto-vectorizer.

The TS v2 `where(mask, v)` is a 2-arg masked-assignment proxy, not a 3-arg select.
A thin free function adapts it:
```cpp
template<typename T, int W>
Simd<T,W> where(SimdMask<T,W> mask, Simd<T,W> a, Simd<T,W> b) {
    auto result = b;
    stdx::where(mask, result) = a;
    return result;
}
```

### Backend B: `std::array` (default, C++17)

`Simd<T,W>` wraps `std::array<T,W>` with element-wise operator loops.
`__hostdev__`-annotated throughout for CUDA compatibility.

**GCC vectorization note**: GCC's failure to auto-vectorize in §5 was specific to
Approach A's outer-lane loop pattern, where GCC could not see through `std::tuple`
struct indirection in GIMPLE.  Backend B's element-wise operator loops (e.g.
`for (int i = 0; i < W; i++) r[i] = a[i] + b[i]`) are a completely different target
— fixed-count, no struct indirection — and GCC does auto-vectorize them when used
with the Generic-T kernel class hierarchy (see §6).

### element_aligned_tag — portable load/store descriptor

`nanovdb::util::element_aligned_tag` and `nanovdb::util::element_aligned` are always
present.  In Backend A they alias `stdx::element_aligned_tag` (same type the stdx
constructors expect); in Backend B they are a standalone dummy struct (ignored).
This makes the load constructor `Simd(const T*, element_aligned)` portable across
both backends and forward-compatible with `std::simd`.

### C++26 migration path

When `std::simd` lands in `<simd>`, migration is a one-line change: replace the
`stdx` detection block with `#if __cpp_lib_simd` and `std::experimental` with `std`.
The kernel source, `element_aligned_tag`, and all call sites are unchanged.

---

## 5. Vectorization Experiments and Findings (Approach A)

Platform: x86-64, AVX2, Ubuntu.  GCC 13.  Clang 18.
Base flags: `-O3 -march=native -std=c++17`

> **Warning — GCC false positive diagnostics**: `-fopt-info-vec-missed` / `-fopt-info-vec`
> can report `optimized: loop vectorized using 32 byte vectors` for code *outside* the
> hot loop.  Assembly inspection is the only ground truth — always verify with
> `grep -c 'ymm'` and confirm the instructions fall inside the target function.

| Experiment | Kernel | GCC | Clang |
|---|---|---|---|
| 1 | Simple Laplacian (pure arithmetic) | Yes | Yes |
| 2 | WENO5 sum, no conditionals | Yes | Yes |
| 3 | Full `normSqGrad`, `bool isOutside` | **No** (control flow) | **Yes** |
| 4 | Same, `isOutside` = constant `true` | No (control flow in `std::max`) | Yes |
| 5 | `fmaxf` + `float sign` | No (struct-access blocker) | Yes |
| 6 | `fmaxf` + `-ffinite-math-only` | No (false positive diagnostic) | Yes |
| 7 | `__attribute__((optimize("finite-math-only")))` | No (doesn't propagate) | Yes |
| 8 | `__builtin_fmaxf` + `float sign` | No (struct-access blocker) | Yes |
| 9 | Pointer-cache + `__builtin_fmaxf` | No (call-clobbers-memory) | Yes |
| 10 | Flat `float[N][W]` arrays | No (gather stride) | n/a |

**Conclusion for Approach A**: GCC 13 cannot auto-vectorize the `liftToSimd` pattern
in any attempted form.  The root cause is GCC's inability to see through `std::tuple`'s
recursive-inheritance struct layout in GIMPLE — not a limitation of Backend B per se.

---

## 6. Vectorization Results (Approach B, assembly-verified)

GCC 13, AVX2, `-O3 -march=native -std=c++17`.  ymm counts per function (assembly-inspected).

### Backend A (`std::experimental::simd`, auto-detected)

| Function | ymm instructions |
|---|---|
| `WenoStencilKernel::normSqGrad` | 945 (WENO5 inlined ×6) |
| `GodunovsNormSqrd` | 289 (out-of-line) |
| `min` / `max` | 10 each |
| `runSimdNormSqGrad` (test wrapper) | 0 (call shell only) |
| **Total** | **1267** |

### Backend B (`std::array`, forced with `-DNANOVDB_NO_STD_SIMD`)

| Function | ymm instructions |
|---|---|
| `WenoStencilKernel::normSqGrad` | 365 |
| `WENO5` | 137 (out-of-line) |
| `GodunovsNormSqrd` | 117 (out-of-line) |
| **Total** | **619** |

Both backends pass all 16 lanes.  Backend B vectorizes via GCC's auto-vectorizer on
the fixed-count element-wise operator loops — the struct-access limitation from
Approach A does not apply here.

Key instructions in both paths: `vfmadd*ps`, `vsubps`, `vmulps`, `vmaxps`,
`vminps`, `vblendvps`, `vcmpnltps`.

---

## 7. Open Questions / Next Steps

- **Benchmarking**: Throughput of the vectorized path vs. scalar not yet measured on
  representative VBM data.
- **Integration**: Move `Simd.h` to `nanovdb/util/Simd.h`; move `StencilKernel.h`
  to `nanovdb/math/`; have legacy `WenoStencil` derive from `WenoStencilKernel`
  during transition, then retire it.
- **`<simd>` header**: Clang 18 provides `<experimental/simd>` but not `<simd>`.
  Once `<simd>` is available, the detection guard simplifies to `#if __cpp_lib_simd`.
- **Clang assembly verification**: Clang not yet installed on this machine.  Previous
  results (691 ymm flat in hot function, free-function version) predate the
  class-based refactor; re-verification pending.

---

## 8. File Reference

| File | Purpose |
|------|---------|
| `simd_test/Simd.h` | `nanovdb::util::Simd<T,W>` — two backends, auto-detected (prototype for `nanovdb/util/`) |
| `simd_test/StencilKernel.h` | `BaseStencilKernel<T,SIZE>`, `WenoStencilKernel<T>`, `WENO5<T>`, `GodunovsNormSqrd<T,MaskT>` (prototype for `nanovdb/math/`) |
| `simd_test/lift_test.cpp` | Correctness test: SIMD vs scalar reference via `WenoStencilKernel` |
| `nanovdb/nanovdb/math/Stencils.h` | Original scalar `WENO5`, `GodunovsNormSqrd`, `WenoStencil::normSqGrad` |
| `nanovdb/nanovdb/examples/ex_voxelBlockManager_host_cuda/StencilGather.md` | Per-block stencil gather design doc |
| `nanovdb/nanovdb/tools/VoxelBlockManager.h` | CPU VBM implementation |

Build commands:
```sh
# GCC, Backend A (std::experimental::simd, auto-detected):
g++ -O3 -march=native -std=c++17 -o lift_test lift_test.cpp

# GCC, Backend B (std::array, forced):
g++ -O3 -march=native -std=c++17 -DNANOVDB_NO_STD_SIMD -o lift_test lift_test.cpp

# Clang, Backend A (std::experimental::simd, C++26):
clang++-18 -O3 -march=native -std=c++26 \
  -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
  -o lift_test lift_test.cpp

# Clang, Backend B (std::array, C++17 or forced):
clang++-18 -O3 -march=native -std=c++17 \
  -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
  -o lift_test lift_test.cpp
```
