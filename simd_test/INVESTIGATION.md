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
any attempted form (see §4).

### Why Superseded

The input/output types are `std::tuple<const std::array<T,W>&, ...>` — reference
tuples pointing into existing SoA buffers.  While correct, the design has two
limitations:

1. The scalar kernel is a separate code path from the GPU kernel — it takes a
   tuple, not individual arguments, and cannot be templated on `T` directly.
2. Vectorization relies entirely on the auto-vectorizer seeing through the tuple
   extraction loop, which GCC cannot do.

---

## 3. Approach B: Generic-T Pattern (current)

### Core Idea

Instead of lifting a scalar kernel into SIMD, write the kernel **once** as a
template on its value type `T`:

```cpp
template<typename T>
T normSqGrad(T v0, T v1, ..., T v18, float dx2, float invDx2, float isoValue);
```

- `T = float` → scalar path, `__hostdev__`-compatible, used on GPU per-thread
- `T = Simd<float, W>` → W-wide SIMD path, used on CPU per-batch

All arithmetic operators, `min`, `max`, and `where` are overloaded for both `float`
and `Simd<float, W>`, so the same source compiles correctly for both contexts with
zero `#ifdef`.

### `where()` — the key primitive

The `bool isOutside ? a : b` ternary cannot be used with a SIMD mask.  `where(mask,
a, b)` replaces it:

```cpp
// Scalar overload (T=float): plain ternary
template<typename T> T where(bool mask, T a, T b) { return mask ? a : b; }

// SIMD overload (T=Simd<float,W>): lane-wise blend → VBLENDVPS, no branch
template<typename T, int W>
Simd<T,W> where(SimdMask<T,W> mask, Simd<T,W> a, Simd<T,W> b);
```

`v0 > T(isoValue)` deduces to `bool` when `T=float` and `SimdMask<float,W>` when
`T=Simd<float,W>`, so the call to `where()` resolves correctly in both cases.

### `nanovdb::util::Simd<T, W>`

A minimal header-only library (`simd_test/Simd.h`, destined for `nanovdb/util/`)
providing:

| Component | Purpose |
|-----------|---------|
| `Simd<T, W>` | W-wide vector backed by `std::array<T, W>`; broadcast constructor, `operator[]`, `store()`, arithmetic operators |
| `SimdMask<T, W>` | Lane-wise boolean result of comparisons |
| `min`, `max` | Lane-wise min/max; scalar overloads for GPU path |
| `where` | Lane-wise blend; scalar overload for GPU path |
| Mixed `T op Simd` / `Simd op T` overloads | Enable `2.f * simd_val` etc. without requiring implicit conversions in template deduction |

~150 lines total.  `__hostdev__`-annotated throughout (macro-guarded for non-CUDA
builds).  Mirrors C++26 `std::simd` naming deliberately — migration is a typedef.

### Kernel structure

```cpp
template<typename T, typename MaskT>
T godunovsNormSqrd(MaskT isOutside,
                   T dP_xm, T dP_xp, T dP_ym, T dP_yp, T dP_zm, T dP_zp)
{
    const T zero(0.f);
    T outside = max(max(dP_xm,zero)*max(dP_xm,zero), min(dP_xp,zero)*min(dP_xp,zero))
              + ...;  // y, z
    T inside  = max(min(dP_xm,zero)*min(dP_xm,zero), max(dP_xp,zero)*max(dP_xp,zero))
              + ...;  // y, z
    return where(isOutside, outside, inside);
}

template<typename T>
T normSqGrad(T v0, T v1, ..., T v18, float dx2, float invDx2, float isoValue)
{
    const T dP_xm = weno5<T>(...), dP_xp = weno5<T>(...);
    const T dP_ym = weno5<T>(...), dP_yp = weno5<T>(...);
    const T dP_zm = weno5<T>(...), dP_zp = weno5<T>(...);
    return invDx2 * godunovsNormSqrd(v0 > T(isoValue),
                                     dP_xm, dP_xp, dP_ym, dP_yp, dP_zm, dP_zp);
}
```

This is structurally identical to `WenoStencil::normSqGrad` in `Stencils.h`.

### GPU / CPU call sites

```cpp
// GPU: one thread per voxel, scalar instantiation
float result = normSqGrad<float>(v[0], v[1], ..., v[18], dx2, invDx2, iso);

// CPU: one call per batch of W voxels, SIMD instantiation
using FloatSimd = nanovdb::util::Simd<float, 16>;
FloatSimd result = normSqGrad<FloatSimd>(sv[0], sv[1], ..., sv[18], dx2, invDx2, iso);
```

NVCC's demand-driven template instantiation ensures `normSqGrad<FloatSimd>` is
never compiled for device — it is only instantiated in host code.

---

## 4. Vectorization Experiments and Findings (Approach A)

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
in any attempted form.  Clang 18 vectorizes the unmodified original.

---

## 5. Vectorization Results (Approach B, assembly-verified)

Compiled with:
```sh
clang++-18 -O3 -march=native -std=c++17 \
  -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
  -o lift_test lift_test.cpp
```

**Clang**: 964 total ymm instructions; 691 inside `runSimdNormSqGrad` +
`normSqGrad<Simd<float,16>>`.  Key instructions: `vfmadd*ps`, `vsubps`, `vmulps`,
`vmaxps`, `vminps`, `vblendvps`, `vcmpnltps`.  Two separate instantiations confirmed
in the symbol table: `normSqGrad<Simd<float,16>>` and `normSqGrad<float>`.

**GCC**: Correct results but does not vectorize the per-operator loops inside
`Simd<T,W>` ("more than one data ref in stmt", "return slot optimization" on
`weno5` calls).  Same class of struct-access limitation as Approach A.

All 16 lanes produce correct results vs. the scalar (`T=float`) reference on both
compilers.

---

## 6. Open Questions / Next Steps

- **GCC support**: The per-operator loops in `Simd<T,W>` (e.g., `operator+`) are
  simple W-iteration loops over `std::array` members.  GCC's "return slot
  optimization" diagnostic on `weno5` calls suggests it cannot treat the `Simd`
  return values as local registers.  Explicit intrinsics (AVX2 `__m256`) in
  `Simd<float,8>` would guarantee GCC vectorization but require architecture-specific
  specializations.
- **Benchmarking**: Throughput of the vectorized Clang path vs. scalar not yet
  measured on representative VBM data.
- **Integration**: `Simd.h` to be moved to `nanovdb/util/Simd.h`; `weno5`,
  `godunovsNormSqrd`, `normSqGrad` to be templated in `nanovdb/math/Stencils.h`.
- **C++26 migration**: Once `std::simd` is available, `nanovdb::util::Simd<T,W>`
  can be replaced with `std::fixed_size_simd<T,W>` — the kernel source is unchanged.

---

## 7. File Reference

| File | Purpose |
|------|---------|
| `simd_test/Simd.h` | Minimal `nanovdb::util::Simd<T,W>` library (prototype; destined for `nanovdb/util/`) |
| `simd_test/lift_test.cpp` | Test: templated `weno5`, `godunovsNormSqrd`, `normSqGrad`; correctness check vs. scalar reference |
| `nanovdb/nanovdb/math/Stencils.h` | Original `weno5`, `GodunovsNormSqrd`, `WenoStencil::normSqGrad` |
| `nanovdb/nanovdb/examples/ex_voxelBlockManager_host_cuda/StencilGather.md` | Per-block stencil gather design doc |
| `nanovdb/nanovdb/tools/VoxelBlockManager.h` | CPU VBM implementation |

Build commands:
```sh
# Clang (vectorizes):
clang++-18 -O3 -march=native -std=c++17 \
  -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
  -o lift_test lift_test.cpp

# GCC (correct results, does not vectorize):
g++ -O3 -march=native -std=c++17 -fopt-info-vec-missed -o lift_test lift_test.cpp
```
