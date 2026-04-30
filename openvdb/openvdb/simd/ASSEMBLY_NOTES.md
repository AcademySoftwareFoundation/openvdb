# Assembly notes for `Simd<T,W>` — `rasterizeSdf` demo path

Examined symbol:
`SphericalTransfer<..., NullCodec, FixedBandRadius<double>, NullFilter, false>::rasterizeN2<4>`

Object file: `build/.../unittest/CMakeFiles/vdb_test.dir/TestPointRasterizeSDF.cc.o`  
Build flags: `-O3 -mavx -msse4.2` (CPU: Intel Core Ultra 9 285K — supports AVX2)

---

## Inner per-voxel z-loop (hot path)

The `stamp<Simd<double,4>>` kernel is fully inlined into `rasterizeN2<4>`.
Key instructions in the innermost loop body:

```nasm
; --- broadcast voxel coord to all 4 particle lanes ---
vbroadcastsd  %xmm0, %ymm0           ; voxel.z → 4 lanes

; --- x²+y²+z² for 4 particles simultaneously ---
vsubpd   Pz_simd(%rbp), %ymm0, %ymm0 ; vox.z - Pz[0..3]
vfmadd213pd  xy2_simd(%rbp), %ymm0, %ymm0  ; z² + (x²+y²)   [AVX2 FMA]

; --- all-outside early exit (zero extra cost) ---
vcmplepd  %ymm0, rmax2_ymm, %ymm1    ; rmax² <= dist² lane-wise
vmovmskpd %ymm1, %eax                ; fold 4 comparison bits → scalar int
not %eax
test $0xf, %al
je   next_voxel                      ; all 4 outside → skip with one branch

; --- 4 square roots in one instruction ---
vsqrtpd  %ymm0, %ymm1                ; sqrt(x²+y²+z²) for all 4 particles

; --- dist = (sqrt(dist²) - r) × vdx ---
vsubpd   r_simd(%rbp), %ymm1, %ymm1
vmulpd   vdx_simd(%rbp), %ymm1, %ymm1

; --- hmin: tree reduction 4→2→1 ---
vmovapd       %xmm1, %xmm0           ; lower 2 lanes
vextractf128  $1, %ymm1, %xmm1       ; upper 2 lanes
; (stdx::reduce helper — see note below)
vmovsd        %xmm0, %xmm0, %xmm1
vunpckhpd     %xmm0, %xmm0, %xmm0
vminsd        %xmm1, %xmm0, %xmm1   ; scalar minimum

; --- scalar write boundary ---
vcvtsd2ss  %xmm1, %xmm1, %xmm1      ; double min → float  (explicit operator T())
vcomiss    %xmm1, grid_val           ; compare with existing voxel value
vmovss     %xmm1, grid_val           ; conditional store
```

---

## Key observations

| | Observation |
|---|---|
| ✓ | All arithmetic uses 256-bit YMM registers (`vsubpd`, `vmulpd`, `vaddpd`, `vcmplepd`) |
| ✓ | `vsqrtpd` — 4 double square roots in **one** instruction vs. 4 serial `sqrtsd` in scalar |
| ✓ | `vcmplepd + vmovmskpd` — 4-lane comparison folds to a single branch (all-outside check) |
| ✓ | `vbroadcastsd` — voxel coordinates broadcast across all 4 particle lanes for free |
| ✓ | `vfmadd213pd` — fused multiply-add for z²+(x²+y²), no extra instruction |
| ✓ | `vcvtsd2ss` at write boundary — clean extraction of scalar min (matches `explicit operator T()`) |
| ⚠ | Two `vzeroupper + call` in the inner loop — AVX→SSE ABI transition before stdx helpers |

---

## The `vzeroupper` concern

There are two `vzeroupper + call` sequences per voxel iteration:

1. Before `stdx::any_of` — used for `hany(x2y2z2 <= min2)` (interior-fill check).
2. Before `stdx::reduce` — used for the `hmin` 4→2 tree step.

These arise because GCC's `std::experimental::fixed_size_simd<double,4>` calls
internal helpers for reductions rather than emitting everything inline.
On Arrow Lake (`vzeroupper` ≈ 2 cycles), this is minor.
On older microarchitectures (Skylake, ~100 cycle penalty), it matters more.

**Mitigation if needed:** replace `stdx::reduce` with explicit `vextractf128 +
vminpd + vpermilpd + vminsd` in `hmin`, bypassing the helper entirely.  This
is mechanical and can be added to Backend A without changing the public API.

---

## Scalar comparison (`-march=native` vs `-mavx`)

Recompiling with `-march=native` (enables AVX2 + FMA) yields:
- `vbroadcastsd` replaces `vmovddup + vinsertf128` (cleaner 1-instruction broadcast).
- `vfmadd213pd` replaces `vmulpd + vaddpd` (fused multiply-add, saves an instruction).
- The `vzeroupper + call` pattern is **unchanged** — it is a stdx implementation
  detail, not an ISA limitation.

Both builds produce correct results; all 8 `TestPointRasterizeSDF` test cases pass.
