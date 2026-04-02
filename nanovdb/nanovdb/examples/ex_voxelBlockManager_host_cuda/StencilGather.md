# Per-Block Stencil Gather

This document is the design and planning reference for the per-block stencil gather
kernel — the operation that, given a built VBM, a block ID, and a user-supplied kernel
lambda, gathers stencil neighbor values for all active voxels in the block and
produces a per-voxel output array.  It is written as dense, agent-consumable facts
and design decisions.

The WENO5 19-point stencil (±3 along each axis independently) is the motivating
instantiation, but the architecture is stencil-agnostic.  The stencil shape enters
as a compile-time template parameter governing the number of neighbor slots, the
neighbor leaf resolution logic, and the value fetch.  The user supplies a scalar
kernel lambda that operates on the gathered values and produces the output.

---

## 1. Scope and Place in the Architecture

The stencil gather sits at the **second level** of the two-level VBM parallelism
hierarchy:

| Level | Operation | Parallelism |
|-------|-----------|-------------|
| System | `buildVoxelBlockManager` | Threading (TBB/CUDA grid) over blocks |
| Per-block | `decodeInverseMaps` + stencil gather | SIMD/SIMT within one block |

The stencil gather:
- **Assumes** the VBM has already been built (`firstLeafID[]`, `jumpMap[]` populated).
- **Is called** once per voxel block, from one CPU thread or one CUDA CTA.
- **Uses** no inter-block communication and holds no state beyond its call.
- **Is not** responsible for launching threads or distributing work across blocks.
  That is the caller's responsibility (a future launcher, analogous to
  `buildVoxelBlockManager`).

---

## 2. Per-Block Execution Model

Within one call (one CPU thread / one CUDA CTA):

1. **Decode inverse maps** into block-local storage:
   - GPU: `smem_leafIndex[BlockWidth]` / `smem_voxelOffset[BlockWidth]` in shared
     memory, filled cooperatively by the CTA via `decodeInverseMaps`.
   - CPU: `leafIndex[BlockWidth]` / `voxelOffset[BlockWidth]` on the stack
     (cache-resident), filled by a single call to `decodeInverseMaps`.
2. **Loop over active voxels** in the block (positions where
   `leafIndex[p] != UnusedLeafIndex`).
3. **For each active voxel**: resolve the neighbor leaf pointers, fetch the N
   neighbor values into a local array, invoke the kernel lambda, and write the
   output.

**Key invariant on intermediate storage**: `leafIndex`, `voxelOffset`, the
neighbor-leaf-pointer structs, and the per-voxel value arrays are all scratch only.
They do not persist beyond this per-block call.  The kernel output array is the
only output.

---

## 3. Stencil Type as Template Parameter

### 3a. What the Infrastructure Needs

The gather infrastructure iterates over stencil slots `n = 0 .. N-1` and for each
needs to know the Cartesian offset `(Δx, Δy, Δz)` to look up.  The pipeline is:

```
for n in 0..N-1:
    values[n] = grid.getValue(center + StencilT::offset(n))
```

This requires **index → offsets** direction: given slot index n, return `(Δx, Δy, Δz)`.

The existing `WenoPt<i,j,k>::idx` (NanoVDB) and `NineteenPt<i,j,k>::idx` (OpenVDB)
go in the **opposite** direction (offsets → index) and are primarily useful to the
user writing the kernel lambda (addressing a specific neighbor by name).  They are
not directly usable by the infrastructure's gather loop.

The stencil type must therefore expose a compile-time offset table:

```cpp
// For each slot n in [0, N), the Cartesian offset
static constexpr std::array<std::array<int,3>, N> offsets;
// or equivalently a static constexpr accessor:
static constexpr std::array<int,3> offset(int n);
```

### 3b. Relationship to BaseStencil / WenoStencil

`nanovdb::math::BaseStencil<Derived, SIZE, GridT>` and `WenoStencil` couple the
stencil geometry to a grid accessor (`mAcc`) via `init()` / `moveTo()`.  This coupling
is incompatible with the VBM batch gather, where the infrastructure owns the value
lookup.

What is reusable from the existing design:
- `SIZE` / `static constexpr int SIZE` — directly useful.
- `WenoPt<i,j,k>::idx` / `pos<i,j,k>()` — useful to the *user's kernel lambda*
  for addressing neighbors by name, but not to the gather loop itself.

The stencil type for our template parameter is a **geometry-only descriptor** — no
accessor, no stored values.  It could be a thin wrapper around the existing types,
or a new family of types alongside them.

### 3c. Stencil Characteristics

- **N** (`SIZE`): number of points including center.
- **Offset table**: compile-time mapping from slot index → `(Δx, Δy, Δz)`.
- **Reach R**: `max |Δ|` over all axes and all slots.  Governs neighbor leaf
  resolution (see §5).

For WENO5: N=19, R=3, offsets derived from `WenoPt` specializations.

---

## 4. Kernel Lambda and Output Type

### 4a. Kernel Lambda Signature

The user supplies a kernel lambda with signature:

```cpp
std::array<ValueType, K> kernel(const ValueType* u);
```

where `u[n]` is the grid value at stencil slot `n` (i.e. `u[0]` is the center,
`u[WenoPt<1,0,0>::idx]` is the +x neighbor for WENO5, etc.).  The lambda is
completely unaware of indices, leaf pointers, or SIMD lanes.

Example — Laplacian (K=1):
```cpp
auto laplacian = [](const float* u) -> std::array<float, 1> {
    return { -6.f*u[0] + u[GradPt<1,0,0>::idx] + u[GradPt<-1,0,0>::idx]
                       + u[GradPt<0,1,0>::idx] + u[GradPt<0,-1,0>::idx]
                       + u[GradPt<0,0,1>::idx] + u[GradPt<0,0,-1>::idx] };
};
```

Example — gradient (K=3):
```cpp
auto grad = [](const float* u) -> std::array<float, 3> {
    return { 0.5f*(u[GradPt<1,0,0>::idx] - u[GradPt<-1,0,0>::idx]),
             0.5f*(u[GradPt<0,1,0>::idx] - u[GradPt<0,-1,0>::idx]),
             0.5f*(u[GradPt<0,0,1>::idx] - u[GradPt<0,0,-1>::idx]) };
};
```

### 4b. Output Type: std::array<ValueType, K>

The output is always `std::array<ValueType, K>` — homogeneous in type.  K=1
degenerates naturally to the scalar case without special-casing.

Heterogeneous output (e.g. `std::tuple`) is not needed for the typical PDE/level-set
workload: Laplacian (K=1), gradient (K=3), WENO upwind differences (K=6), curvature
components (K=2) are all uniform in type.  A tuple would also defeat auto-vectorization.

### 4c. Output Buffer Layout

The per-block output is stored in SoA layout:

```
results[k][BlockWidth]   for k = 0 .. K-1
```

Each channel `k` is a contiguous array of `ValueType` across all BlockWidth voxel
positions, mapping cleanly to K independent SIMD registers.  AoS layout
(`results[BlockWidth][K]`) would interleave channels and defeat SIMD.

K is either deduced from the lambda's return type or supplied as an explicit template
parameter.

---

## 5. Neighbor Leaf Resolution

### 5a. How Many Leaf Neighbors Per Axis

A leaf covers 8 positions along each axis.  For a stencil with reach R, a voxel at
leaf-local position p along one axis needs neighbors at p-R .. p+R.  The number of
distinct leaves touched along that axis depends on where p falls within the leaf:

- For R ≤ 3 (e.g. WENO5): at most **one** neighbor leaf per axis (either lo or hi,
  never both simultaneously, for any p in [0,7]).  This is because the worst case
  (p=0, reach=3) reaches p-3 = -3 (one leaf back) but p+3 = 3 (still in the same
  leaf).
- For R > 4: a center voxel near the middle of a leaf can require neighbors in both
  the lo and the hi neighboring leaf along the same axis simultaneously.

The current `resolveLeafPtrs` design (`ptrs[axis][0..2]`: lo/center/hi) is correct
for R ≤ 3.  A more general design would use `ptrs[axis][0..K]` where K = number of
neighbor leaves per axis.

### 5b. resolveLeafPtrs — Design

```
resolveLeafPtrs(grid, leaf, voxelOffset) → StencilLeafPtrs
```

- Performs the minimum number of `probeLeaf` calls required by the stencil shape.
- For WENO5 (R=3): exactly **3 probeLeaf calls total** (one per axis), since at most
  one neighbor leaf is needed per axis.
- Returns a `StencilLeafPtrs` struct whose layout is stencil-specific (see §5).
- Intentionally scalar: `probeLeaf` is pointer-chasing and not vectorizable.

### 5c. computeStencil — Design

```
computeStencil(leaf, voxelOffset, leafPtrs, data[N])
```

- Fills `data[N]` with global sequential indices for all N stencil points.
- Caller must zero-initialize `data[]`; entries for out-of-narrow-band neighbors
  remain 0.
- Uses the stencil's index mapping (e.g. `WenoPt<i,j,k>::idx`) throughout —
  never hardcoded integers.
- This is the auto-vectorization target for the CPU port (see §6).

---

## 6. StencilLeafPtrs Struct

Unified template parameterized on build type and leaf pointer type, enabling both
scalar (GPU) and batched (CPU) instantiations from one definition:

```cpp
template<typename BuildT, typename LeafPtrT>
struct StencilLeafPtrs {
    LeafPtrT ptrs[3][3];  // [axis][slot]: slot 0=lo, 1=center, 2=hi
};
```

- **GPU** (scalar per thread): `LeafPtrT = const NanoLeaf<BuildT>*`
- **CPU batch** (SIMDw lanes): `LeafPtrT = std::array<const NanoLeaf<BuildT>*, SIMDw>`

The `ptrs[3][3]` shape is correct for stencils with R ≤ 3.  Larger stencils would
require a different slot count.

The current GPU draft in `VoxelBlockManager.cuh` uses an unparameterized
`WenoLeafPtrs<BuildT>` (GPU-only, WENO5-specific).  Generalizing to
`StencilLeafPtrs<BuildT, LeafPtrT>` is a prerequisite for the CPU implementation
and for supporting additional stencil shapes.

---

## 7. GPU Inner Loop (Current Draft)

After `decodeInverseMaps`, each thread with `smem_leafIndex[tID] != UnusedLeafIndex`:

```cpp
const auto& leaf = tree.getFirstNode<0>()[smem_leafIndex[tID]];
const uint16_t vo = smem_voxelOffset[tID];

uint64_t stencilData[N] = {};
auto leafPtrs = VBM::resolveLeafPtrs(grid, leaf, vo);
VBM::computeStencil(leaf, vo, leafPtrs, stencilData);
```

No synchronization needed between decode and stencil steps beyond the `__syncthreads()`
already inside `decodeInverseMaps`.  `resolveLeafPtrs` and `computeStencil` are both
per-thread and divergence-safe.

---

## 8. CPU Inner Loop

### 8a. SIMD Batch Width

Process voxels in batches of `SIMDw = 16`.  With AVX2 (16 × uint16_t per register),
each batch maps to one SIMD register width for `voxelOffset`.

### 8b. probeLeaf Deduplication

Within a batch of SIMDw=16 voxels, the neighbor coordinate along each axis (rounded
to leaf granularity) takes at most **2 distinct values** per axis.  The result of each
`probeLeaf` call is broadcast to the lanes sharing that neighbor coordinate.

For a stencil with R ≤ 3: ≤ 2 `probeLeaf` calls per axis × 3 axes =
**≤ 6 `probeLeaf` calls per batch** (vs up to 3×SIMDw for naive per-voxel approach).

The deduplication bound depends on both SIMDw and leaf size (8).  For larger SIMDw
or larger R, more distinct neighbor coordinates can appear per batch.

### 8c. computeStencil Vectorization

The outer loop over lanes (i = 0 .. SIMDw-1) calls `computeStencil` once per lane
with output into a SoA `stencilData[N][SIMDw]` array.  Auto-vectorization strategy:

- `[[clang::always_inline]]` on `computeStencil`.
- `__restrict__` on output pointers.
- `#pragma clang loop vectorize(enable) vectorize_width(16)` on the outer lane loop.
- Output via `std::array<uint64_t*, N>` (proven to vectorize; POD struct output
  vectorizes the wrong dimension).

---

## 9. Open Questions / Deferred Decisions

- **Launcher design**: the system-level wrapper that dispatches per-block calls
  (the `buildVoxelBlockManager` analogue for the stencil gather).  Deferred until
  the per-block kernel is validated.

- **Stencil type definition**: the geometry-only stencil descriptor (§3) needs a
  concrete C++ form — whether a new family of types, a thin wrapper around existing
  `BaseStencil` specializations, or a standalone `constexpr` struct.  The offset table
  representation (`std::array<std::array<int,3>, N>` vs a static `constexpr` accessor
  function) is also TBD.

- **K deduction vs explicit parameter**: whether K (output count) is deduced from the
  lambda's return type via `decltype` / CTAD, or supplied as an explicit template
  parameter alongside the stencil type.

- **CPU `resolveLeafPtrs` batch function**: the per-batch deduplication logic (§8b)
  needs its own function, separate from the GPU scalar `resolveLeafPtrs`.  Signature
  and deduplication algorithm TBD.

- **Generalizing beyond R ≤ 3**: the `ptrs[3][3]` struct and single-neighbor-per-axis
  assumption are baked into the current design.  Any stencil with R > 4 would require
  revisiting §5a and §6.
