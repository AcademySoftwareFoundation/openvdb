# Per-Block Stencil Gather

This document is the design and planning reference for the per-block stencil gather
kernel — the operation that, given a built VBM and a block ID, computes the neighbor
index sets for all active voxels in that block under a given stencil shape.  It is
written as dense, agent-consumable facts and design decisions.

The WENO5 19-point stencil (±3 along each axis independently) is the motivating
instantiation, but the architecture is stencil-agnostic.  The stencil shape enters
only as a compile-time parameter governing the number of output slots and the
neighbor leaf resolution logic.

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
3. **For each active voxel**: resolve the neighbor leaf pointers for the stencil
   shape, then fill the stencil's N-entry index array.

**Key invariant on intermediate storage**: `leafIndex` and `voxelOffset` are scratch
only.  They do not persist beyond this per-block call, and neither do any intermediate
neighbor-leaf-pointer structures.  The stencil index arrays are the outputs.

---

## 3. Stencil Parameterization

A stencil is characterized by:

- **N**: number of points (including center).
- **Point set**: compile-time set of relative (Δx, Δy, Δz) offsets, with a defined
  mapping from each offset to an index in [0, N).
- **Reach R**: max |Δ| along any axis.  Governs how many distinct neighbor leaves
  per axis must be resolved (see §4).

For the WENO5 stencil: N=19, reach R=3, point set = {0} ∪ {±1,±2,±3 along each
axis independently}, index mapping = `WenoPt<i,j,k>::idx`.

The index mapping convention is stencil-specific and must be documented per stencil.
In particular, `WenoPt<i,j,k>::idx` (NanoVDB) is inconsistent with `NineteenPt<i,j,k>::idx`
(OpenVDB) and must not be cross-used.

---

## 4. Neighbor Leaf Resolution

### 4a. How Many Leaf Neighbors Per Axis

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

### 4b. resolveLeafPtrs — Design

```
resolveLeafPtrs(grid, leaf, voxelOffset) → StencilLeafPtrs
```

- Performs the minimum number of `probeLeaf` calls required by the stencil shape.
- For WENO5 (R=3): exactly **3 probeLeaf calls total** (one per axis), since at most
  one neighbor leaf is needed per axis.
- Returns a `StencilLeafPtrs` struct whose layout is stencil-specific (see §5).
- Intentionally scalar: `probeLeaf` is pointer-chasing and not vectorizable.

### 4c. computeStencil — Design

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

## 5. StencilLeafPtrs Struct

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

## 6. GPU Inner Loop (Current Draft)

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

## 7. CPU Inner Loop

### 7a. SIMD Batch Width

Process voxels in batches of `SIMDw = 16`.  With AVX2 (16 × uint16_t per register),
each batch maps to one SIMD register width for `voxelOffset`.

### 7b. probeLeaf Deduplication

Within a batch of SIMDw=16 voxels, the neighbor coordinate along each axis (rounded
to leaf granularity) takes at most **2 distinct values** per axis.  The result of each
`probeLeaf` call is broadcast to the lanes sharing that neighbor coordinate.

For a stencil with R ≤ 3: ≤ 2 `probeLeaf` calls per axis × 3 axes =
**≤ 6 `probeLeaf` calls per batch** (vs up to 3×SIMDw for naive per-voxel approach).

The deduplication bound depends on both SIMDw and leaf size (8).  For larger SIMDw
or larger R, more distinct neighbor coordinates can appear per batch.

### 7c. computeStencil Vectorization

The outer loop over lanes (i = 0 .. SIMDw-1) calls `computeStencil` once per lane
with output into a SoA `stencilData[N][SIMDw]` array.  Auto-vectorization strategy:

- `[[clang::always_inline]]` on `computeStencil`.
- `__restrict__` on output pointers.
- `#pragma clang loop vectorize(enable) vectorize_width(16)` on the outer lane loop.
- Output via `std::array<uint64_t*, N>` (proven to vectorize; POD struct output
  vectorizes the wrong dimension).

---

## 8. Open Questions / Deferred Decisions

- **Launcher design**: the system-level wrapper that dispatches per-block calls
  (the `buildVoxelBlockManager` analogue for the stencil gather).  Deferred until
  the per-block kernel is validated.

- **Index → value conversion**: `stencilData[N]` currently holds global sequential
  indices.  The PDE consumer wants `float` values.  Whether the index-to-value lookup
  (`grid->tree().getValue(idx)`) happens inside or outside this kernel is TBD.

- **CPU `resolveLeafPtrs` batch function**: the per-batch deduplication logic (§7b)
  needs its own function, separate from the GPU scalar `resolveLeafPtrs`.  Signature
  and deduplication algorithm TBD.

- **Generalizing beyond R ≤ 3**: the `ptrs[3][3]` struct and single-neighbor-per-axis
  assumption are baked into the current design.  Any stencil with R > 4 would require
  revisiting §4a and §5.
