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

## 6. Neighbor Direction Encoding and Leaf Pointer Tables

### 6a. Shared 3×3×3 Bit Encoding

All stencil types use the same flat bit encoding for neighbor directions, based on
the 3×3×3 cube of immediately adjacent leaves:

```
bit(dx, dy, dz) = (dx+1)*9 + (dy+1)*3 + (dz+1)
```

where `(dx, dy, dz) ∈ {-1, 0, +1}³`.  This yields 27 bits total, fitting in a
`uint32_t`.  Bit 13 is the center `(0,0,0)` — always implicit, never probed.

```
neighborCoord(centerCoord, bit):
    dx = bit/9 - 1,  dy = (bit/3)%3 - 1,  dz = bit%3 - 1
    return centerCoord + Coord(dx*8, dy*8, dz*8)   // leaf origin offset
```

The six WENO5 face-neighbor bits are a strict subset of the 27:

| Direction | (dx,dy,dz) | bit |
|-----------|-----------|-----|
| x-lo      | (-1, 0, 0) | **4**  |
| y-lo      | ( 0,-1, 0) | **10** |
| z-lo      | ( 0, 0,-1) | **12** |
| z-hi      | ( 0, 0,+1) | **14** |
| y-hi      | ( 0,+1, 0) | **16** |
| x-hi      | (+1, 0, 0) | **22** |

For the box stencil, all 26 non-center bits may be set.  The encoding is identical;
only the set of active bits differs.

### 6b. Common Per-Leaf Canonical State (CPU)

State that persists across batches within one center leaf:

```cpp
uint32_t      probedMask = 0;    // bit d set ↔ direction d has been probed this leaf
const LeafT*  ptrs[27]   = {};   // canonical neighbor table; ptrs[13] unused (center)
Coord         centerLeafCoord;
```

`ptrs[]` is populated lazily by the `probeLeaf` loop (§8d).  For WENO5, only the
six face-direction entries (bits 4,10,12,14,16,22) are ever non-null; the 21
edge/corner entries remain null throughout.

### 6c. Stencil-Specific Per-Batch SIMD Table

After the probeLeaf loop fills `ptrs[27]`, the relevant entries are broadcast into a
per-lane SIMD table whose layout is stencil-specific:

**WENO5 — `batchPtrs[4][SIMDw]`** (center + one per axis):
- `[0][i]` — center leaf (uniform broadcast of `&currentLeaf`)
- `[1][i]` — x-axis neighbor: `ptrs[4]` if `lx < R`, `ptrs[22]` if `lx >= 8-R`, else `nullptr`
- `[2][i]` — y-axis neighbor: `ptrs[10]` / `ptrs[16]` / `nullptr`
- `[3][i]` — z-axis neighbor: `ptrs[12]` / `ptrs[14]` / `nullptr`

The broadcast is masked: a scalar `ptrs[bit]` value is written into lane `i` under
the condition that lane `i`'s local coordinate requires that direction.  The
lo/hi decision is encoded in the ptr value itself — `computeStencil` does not need
to distinguish lo from hi at index-computation time.

**Box stencil — `batchPtrs[3][3][3][SIMDw]`**: the full 27-entry cube, per lane.
Population follows the same masked-broadcast pattern, driven by each lane's
`(lx, ly, lz)` relative to leaf boundaries.

This compaction is the step that bridges the shared scalar probeLeaf machinery (§8d)
and the SIMD stencil index computation (§8g).

### 6d. GPU Scalar Design (Unchanged)

The GPU per-thread design uses `ptrs[3][3]` (axis × {lo, center, hi}) and probes
all needed directions unconditionally on entry — acceptable because each GPU thread
handles one voxel and the probe count is bounded by 3.  The GPU design does not
use `probedMask` or the 27-bit encoding.  Both CPU and GPU designs resolve neighbor
leaves via `probeLeaf`; the machinery diverges only in batch vs. scalar granularity.

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

### 8b. Scan-Order Coherence and Expected probeLeaf Count

NanoVDB linearizes active voxels **z-fast, y-medium, x-slow** (offset = x×64 + y×8 + z).
This means consecutive sequential active voxels vary z fastest and x slowest.  The
expected intra-leaf distribution across a batch of SIMDw=16 at ~50% leaf density
(~256 active voxels per leaf):

- 16 active voxels span ~32 scan positions → one fixed intra-leaf **x** value,
  ~4 consecutive **y** values, and all 8 **z** values covered.

This axis asymmetry determines the expected number of **unique** probeLeaf calls
per batch after deduplication:

| Axis | Reason | Expected unique probes |
|------|--------|----------------------|
| x    | All 16 voxels at same intra-leaf x; need lo or hi but not both | **≈ 0.75** |
| y    | Spans ~4 y values; may straddle lo/hi boundary | **≈ 1.2** |
| z    | All 8 z values present; always needs both z-lo and z-hi | **≈ 2** (deterministic) |

**Total expected unique probeLeaf calls per batch: ~4** (well below the theoretical
maximum of 6).

For stencils with R ≤ 3 (WENO5), a voxel at intra-leaf position p needs the
lo neighbor when p < R and the hi neighbor when p > (LeafDim - 1 - R).
For R=3, LeafDim=8: lo needed for p ∈ {0,1,2}, hi for p ∈ {5,6,7}.

At lower leaf densities the batch spans more leaves and the expected count rises
toward 6; at higher densities it falls toward 2 (x and y each converge to 1, z stays 2).

### 8c. ReadAccessor: Cache Behavior for probeLeaf

The NanoVDB `DefaultReadAccessor<BuildT>` (`ReadAccessor<BuildT, 0, 1, 2>`) stores
three independent single-slot caches: one per tree level (leaf/lower/upper).  The
`get<OpT>` dispatch checks **only the cache at `OpT::LEVEL`**, as an `if constexpr`
chain:

```cpp
if constexpr(OpT::LEVEL <= 0) {
    if (isCached<LeafT>(ijk)) return leaf->getAndCache<OpT>(...);  // leaf hit
} else if constexpr(OpT::LEVEL <= 1) { ... }   // compiled away for GetLeaf
  else if constexpr(OpT::LEVEL <= 2) { ... }   // compiled away for GetLeaf
return mRoot->getAndCache<OpT>(ijk, *this);    // leaf miss → full traversal
```

For `GetLeaf` (LEVEL=0), the compiled code is exactly two paths:

- **Leaf cache hit**: `isCached<LeafT>` check (3 masked comparisons) + return
  `mNode[0]`.  No memory loads beyond the accessor struct.  Cost: ~6 integer
  instructions, essentially free.

- **Leaf cache miss**: falls directly to `mRoot->getAndCache<GetLeaf>` — a **full
  root-to-leaf traversal**, identical in cost to `tree.probeLeaf(ijk)`.  The lower
  and upper node caches (`mNode[1]`, `mNode[2]`) are **not consulted** for LEVEL=0
  operations; they are populated as a side effect of the traversal but never read
  back for subsequent `get<GetLeaf>` calls.

This is a deliberate NanoVDB design choice (simpler code, better GPU SIMT behavior).
It differs from OpenVDB's `ValueAccessor3`, which does check lower/upper caches on
a leaf miss and can short-circuit traversal from a cached lower node.

**Implications for probeLeaf in the stencil gather:**

The ReadAccessor only helps `probeLeaf` when consecutive calls land in the **same
leaf**.  For calls targeting different leaves — even adjacent leaves in the same
lower node — it is a full root traversal each time.

**Accessor granularity:** Use **one `DefaultReadAccessor<BuildT>` per CPU thread**,
constructed once before the block loop and reused across all blocks and all axes.
Per-axis accessors would each pay a cold traversal for their first probe in a batch,
losing the cross-axis leaf-cache sharing (in the typical single-leaf batch, one probe
warms the leaf and all subsequent probes across any axis that happen to need the same
leaf get the hit for free).  Per-block construction discards carryover between
consecutive blocks, which is wasteful since consecutive blocks process spatially
adjacent leaves.

### 8d. Neighbor Leaf Resolution — Lazy Probe with Per-Leaf Cache

**Why not unconditional probing:** an alternative design probes all `NUM_DIRS`
neighbor directions when the center leaf changes, caching the full pointer table
upfront.  For WENO5 (6 face-neighbor directions) this is only marginally wasteful.
For the box stencil (26 directions: 6 faces + 12 edges + 8 corners), most batches
are interior and never touch edge or corner leaves; unconditionally probing all 26
would waste ~15–20 probeLeaf calls per center leaf.

**Why not naive per-voxel accessor use:** calling `acc.probeLeaf` for every lane
without deduplication causes leaf-cache thrashing at every y-row boundary (the cache
alternates between z-lo and z-hi at each transition).  For 4 y-rows per batch, the
z-direction alone produces ~8 full traversals instead of 2.  Not recommended.

**Design: lazy probe with per-leaf `probedMask`.**

State that persists across all batches within the same center leaf (see §6b):

```cpp
uint32_t      probedMask = 0;    // 27-bit; bit = (dx+1)*9 + (dy+1)*3 + (dz+1)
const LeafT*  ptrs[27]   = {};   // canonical neighbor table (§6a); center implicit
Coord         centerLeafCoord;
```

Per-batch logic — Phase 1 (probeLeaf):

```cpp
uint32_t neededMask = computeNeededDirs(voxelOffset_batch, laneMask);  // §8e
uint32_t toProbe    = neededMask & ~probedMask;   // needed AND not yet cached

while (toProbe) {
    int d = __builtin_ctz(toProbe);               // position of lowest set bit
    ptrs[d] = acc.get<GetLeaf>(neighborCoord(centerLeafCoord, d));
    probedMask |= (1u << d);
    toProbe    &= toProbe - 1;                    // clear lowest set bit
}
```

Per-batch — Phase 2 (populate stencil-specific `batchPtrs` from `ptrs[27]`):

```cpp
// WENO5 example:
const LeafT* batchPtrs[4][SIMDw];
for (int i = 0; i < SIMDw; i++) batchPtrs[0][i] = &currentLeaf;
for (int i = 0; i < SIMDw; i++) {
    int lx = voxelOffset[b+i] >> 6;
    batchPtrs[1][i] = (lx < R) ? ptrs[4] : (lx >= 8-R) ? ptrs[22] : nullptr;
    int ly = (voxelOffset[b+i] >> 3) & 7;
    batchPtrs[2][i] = (ly < R) ? ptrs[10] : (ly >= 8-R) ? ptrs[16] : nullptr;
    int lz = voxelOffset[b+i] & 7;
    batchPtrs[3][i] = (lz < R) ? ptrs[12] : (lz >= 8-R) ? ptrs[14] : nullptr;
}
```

Then `computeStencil(batchPtrs, voxelOffset + b, data + b)` (§8g).

On center leaf advance (`currentLeafID++`):

```cpp
probedMask    = 0;
centerLeafCoord = tree.getFirstNode<0>()[currentLeafID].origin();
// stale ptrs[] entries are harmless; probedMask=0 guarantees re-probe before use
```

`probedMask` persists across batch boundaries.  A direction probed during batch k
is not re-probed during batch k+1 if the center leaf has not changed.  Total
probeLeaf calls per center leaf = number of distinct directions needed across all
batches in that leaf, always ≤ 26 (≤ 6 for WENO5).

**Where the ReadAccessor genuinely earns its keep:** the `getValue` calls inside
`computeStencil` that fetch N stencil values per voxel.  Many of these land in the
same leaf repeatedly.  One accessor per thread, reused across the entire block loop,
accumulates leaf-cache hits throughout the computation.

### 8e. `computeNeededDirs` — Stencil-Specific Batch Probe Mask

```cpp
uint32_t computeNeededDirs(const uint16_t* voxelOffset, uint32_t laneMask);
```

Inspects the `voxelOffset` values of active lanes and returns a bitmask of directions
whose neighbor leaf is accessed by at least one lane.  This is purely arithmetic on
voxelOffsets — no tree access, no probeLeaf.

Direction bits use the shared 3×3×3 encoding from §6a:
`bit(dx,dy,dz) = (dx+1)*9 + (dy+1)*3 + (dz+1)`.

**Direction encoding (WENO5, 6 active bits out of 27):**

| Bit | Direction | (dx,dy,dz) | Condition (per lane)              |
|-----|-----------|-----------|-----------------------------------|
| 4   | x-lo      | (-1,0,0)  | `(vo >> 6) < R`                   |
| 10  | y-lo      | (0,-1,0)  | `((vo >> 3) & 0x7) < R`           |
| 12  | z-lo      | (0,0,-1)  | `(vo & 0x7) < R`                  |
| 14  | z-hi      | (0,0,+1)  | `(vo & 0x7) >= (8 - R)`           |
| 16  | y-hi      | (0,+1,0)  | `((vo >> 3) & 0x7) >= (8 - R)`    |
| 22  | x-hi      | (+1,0,0)  | `(vo >> 6) >= (8 - R)`            |

For WENO5 (R=3): z-lo when lz ∈ {0,1,2}, z-hi when lz ∈ {5,6,7}.  Each condition
is a threshold comparison across SIMDw lanes, folded with an `any()` reduction.
The remaining 21 bits of `neededMask` are always zero for WENO5.

**Box stencil (R=1, up to 26 active bits):** face directions use the same six bit
positions with thresholds 0 and 7.  Edge directions (e.g. (-1,-1,0) → bit 1)
require a pairwise AND: `any(lx == 0 && ly == 0)`.  Corner directions require all
three simultaneously.  Same mechanism throughout — all pure arithmetic on
voxelOffsets, same `uint32_t` mask type.

`computeNeededDirs` is the only function that encodes knowledge of the stencil's
reach R and how offsets map to neighbor leaves.  It is written once per stencil
shape and is small (≤ 20 SIMD instructions for WENO5).

### 8f. CPU Block-Level Loop Structure

**Block dispatch using `nExtraLeaves`.**

`nExtraLeaves` is the popcount of the entire block's `jumpMap` — already computed
inside `decodeInverseMaps` as the loop bound for the leaf-iteration pass:

```cpp
int nExtraLeaves = 0;
for (int i = 0; i < JumpMapLength; i++)
    nExtraLeaves += util::countOn(jumpMap[i]);
```

`nExtraLeaves + 1` equals the total number of center leaves touched within this
block.  This value is a natural block-level dispatch condition:

- `nExtraLeaves == 0`: entire block is single-leaf.  No `currentLeafID` advances,
  no straddle batches.  Can specialize the inner loop to eliminate dead branches.
- `nExtraLeaves >= 1`: at least one leaf transition.  At most `nExtraLeaves` straddle
  batches exist; all other batches are single-leaf.

**Loop skeleton (general path):**

```cpp
uint32_t currentLeafID   = firstLeafID;
uint32_t probedMask      = 0;
const LeafT* ptrs[NUM_DIRS] = {};
Coord centerLeafCoord    = tree.getFirstNode<0>()[currentLeafID].origin();

for (int b = 0; b < BlockWidth; b += SIMDw) {
    uint32_t activeMask = non_sentinel_mask(leafIndex + b);
    if (!activeMask) continue;

    while (activeMask) {
        // Which lanes belong to the current center leaf?
        uint32_t leafMask = lanes_equal(leafIndex + b, currentLeafID) & activeMask;

        if (!leafMask) {
            // No lanes match: advance to the next leaf
            currentLeafID++;
            probedMask    = 0;
            centerLeafCoord = tree.getFirstNode<0>()[currentLeafID].origin();
            continue;
        }

        // Probe any newly needed neighbors
        uint32_t neededMask = computeNeededDirs(voxelOffset + b, leafMask);
        uint32_t toProbe    = neededMask & ~probedMask;
        while (toProbe) {
            int d = __builtin_ctz(toProbe);
            ptrs[d]     = acc.get<GetLeaf>(neighborCoord(centerLeafCoord, d));
            probedMask |= (1u << d);
            toProbe    &= toProbe - 1;
        }

        computeStencil(leafMask, ptrs, voxelOffset + b, data + b);
        activeMask &= ~leafMask;
    }
}
```

**Key invariants:**
- `currentLeafID` is monotonically non-decreasing across the entire block; it
  advances at most `nExtraLeaves` times.
- `probedMask` is reset only when `currentLeafID` changes — not on every batch.
  Directions probed in earlier batches stay cached.
- For single-leaf blocks, the `if (!leafMask)` branch is dead, `currentLeafID`
  never changes, and `probedMask` accumulates across all batches in the block.
- For straddle batches, the `while (activeMask)` iterates twice (once per leaf
  present in the batch), each time consuming its subset of lanes.

### 8g. computeStencil Vectorization

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

- **`ptrs[]` layout — GPU vs CPU divergence**: the GPU design keeps `ptrs[3][3]`
  (axis × {lo,center,hi}), probing unconditionally per thread.  The CPU design uses
  the canonical `ptrs[27]` + `probedMask` (§6b) as common infrastructure, then
  populates a stencil-specific `batchPtrs` (§6c).  These two designs are intentionally
  separate; no unification is needed.

- **`nExtraLeaves` surfacing**: recomputed cheaply from the block's jumpMap after
  `decodeInverseMaps` returns (popcount loop, same as the internal loop bound).
  `decodeInverseMaps` API is not modified — avoids CPU/GPU asymmetry.

- **Prototype — immediate next step**: `stencil_gather_cpu.cpp` in
  `ex_voxelBlockManager_host_cuda/`.  Scope:
  - Generate domain with `generateDomain` (reuse from `vbm_host_cuda.cpp`).
  - Build VBM.  Iterate over blocks; call `decodeInverseMaps` per block.
  - For each batch: run the full §8d probeLeaf + `batchPtrs[4][SIMDw]` population.
  - Verification only (no `computeStencil`): for each active lane, walk all 19
    WENO5 offsets, check that `batchPtrs[axis][i]` matches a direct `probeLeaf`
    reference for every neighbor that crosses a leaf boundary.
  - Use WENO5 stencil directly (not the simpler 7-pt Laplacian — WENO5 exercises
    R=3 boundary conditions and all six face directions).

- **Generalizing beyond R ≤ 3**: the single-neighbor-per-axis assumption is baked
  into the current design.  Any stencil with R > 4 would require revisiting §5a and §6.
