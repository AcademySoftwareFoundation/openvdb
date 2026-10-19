# PartitionGrid — partitioning an index grid by a sidecar

> **Status: design discussion, nothing implemented.** This records the reasoning behind a proposed
> `PartitionGrid` topology operator and the decisions taken so far. Both the binary (§3-6) and n-ary
> (§7) cases are covered; the open items are in §8.
>
> The operator is a general `tools/cuda` primitive — a sibling of `PruneGrid` / `DilateGrid` /
> `MergeGrids` — not SDF-specific. It is documented here because this is where the design notes for
> the surrounding work live; it can move next to the code when it is upstreamed.
>
> Copied unchanged from the `mesh-to-sdf` branch (`nanovdb/nanovdb/examples/ex_mesh_to_sdf_cuda/`). The
> documents and code it refers to (`MeshToSDF`, `InclusionSigningDesign.md`, `BENCHMARK.md`,
> `MeshToSDFDevelopmentPlan.md`, `ConnectedComponents.cuh`) live on that branch, not on this one.

---

## 1. Why

`MeshToSDF` already performs this operation, one output at a time. `signSurface()` builds a
`Mask<3>` retain mask over the **whole** rasterized band, runs a complete `PruneGrid`, and re-indexes
the UDF and nearest-triangle sidecars onto the result — then does it again for the next surface. For
`N` closed surfaces that is `N` sequential full passes over the entire band that happen to produce
`N` grids, rather than one pass that partitions.

`InclusionSigningDesign.md` (M1) parked exactly this: prototype with per-label `PruneGrid`, then
"replace that with a dedicated **batched partition primitive** (label → N index grids in one pass) as
a later optimization — a sibling of the existing prune / dilate / merge topology ops."

`BENCHMARK.md` §5-6 measures what that costs today: the carve is 70-74% of the per-surface work at a
flat ~2.2 ms each, and because each carve prunes the whole band there is an `O(N × voxels)` term with
a small (~0.03 ns/voxel) coefficient.

## 2. Two entry points

The binary and n-ary cases take **different inputs**, so they are separate entry points that share a
name rather than one generalized call:

| | input | outputs |
|---|---|---|
| **binary** | a `Mask<3>` sidecar, one per source leaf — exactly `PruneGrid`'s input | 2 |
| **n-ary** | a label sidecar indexed by `leaf.getValue(n)` | N |

The n-ary case is built **on top of** the binary one by recursive bisection (§7); it introduces no
device kernels of its own.

### 2a. Why not "SplitGrids"

`nanovdb::splitGrids()` already exists in `GridHandle.h` and means something entirely different:
splitting a *multi-grid handle* into single-grid handles. Its CUDA counterparts
`cuda::splitGridHandles` / `cuda::mergeGridHandles` are exactly what a caller would use to
post-process this operator's output, so having both names in play would be actively confusing.
`PartitionGrid` sits naturally beside `PruneGrid` / `DilateGrid` / `MergeGrids` and matches the
language the surrounding design notes already use.

### 2b. Return types

**Decided.**

| | returns |
|---|---|
| **binary** | `std::array<GridHandle<BufferT>, 2>` |
| **n-ary** | `std::vector<GridHandle<BufferT>>` |

Single-grid handles in both cases. `std::array` rather than `std::pair` for the binary case because
`std::array` and `std::vector` share `operator[]`, `size()` and iteration, so generalising from
binary to n-ary does not force callers to rewrite access sites — whereas `.first`/`.second` would.
Positional access also reads better given that §3a *defines* index 0 as "whatever `PruneGrid` would
have returned"; that is positional information, not a first/second distinction. `std::array` over a
length-2 `std::vector` because it encodes "exactly 2" in the type and avoids a container heap
allocation, which the n-ary recursion would otherwise pay N−1 times.

`GridHandle` supports this: default-constructible (`GridHandle.h:62`), move-constructible and
move-assignable, copy deleted. Accepted caveat: `std::array<T,2>` and `std::vector<T>` remain
distinct types, so code generic over both needs a template parameter (`std::span` is C++20).
Interface-level uniformity is judged sufficient, and the recursion — the main internal consumer of
the binary entry point — wants exactly-2 regardless.

Single-grid handles are what the operator produces *natively*: each
`TopologyBuilder::getBuffer()` allocates its own buffer, so the container costs nothing extra. The
alternative — one multi-grid `GridHandle` — would force a `cuda::mergeGridHandles` pass, a full copy
of the entire output (≈V), on every caller including those that never need the grids adjacent
(`MeshToSDF` indexes per-surface and does not). Callers who *do* want contiguity get it in one
documented, stream-aware call. For the n-ary case this also matches `cuda::splitGridHandles`, which
returns exactly that type.

The cost is accepted knowingly: N separate device allocations, plus the fixed
`GridData`/`TreeData`/`RootData` overhead carried by every output grid. At very large N that overhead
can exceed the payload, and neither return type fixes it — that would require the builders to write
into a single pre-sized buffer, avoiding both the copy and the N allocations. That is invasive
(`getBuffer()` allocates for itself today) and is a further argument for the §7e cutoff: capping the
number of outputs is a cheaper remedy than restructuring allocation.

## 3. Binary — semantics

Given source grid `S` and leaf-mask sidecar `m`:

```
Grid0 = S.valueMask & m
Grid1 = S.valueMask & ~m
```

An exact partition of the active set: disjoint, covering, nothing lost or duplicated. The two output
active counts sum to the source's.

### 3a. The governing constraint — symmetry with PruneGrid

**`PartitionGrid(S, m)[0]` must be identical to `PruneGrid(S, m)`.** The same mask passed to either
operator produces the same first result, which makes `PartitionGrid` a strict refinement of `PruneGrid`
— "the same thing, but hand me the discards too" — and lets a caller move between them without
rethinking their mask.

This was chosen over aligning the binary index convention with the n-ary one (where `grid[i]`
naturally means `label == i`, which would want `Grid0` to be the mask-*clear* side). Since the n-ary
case takes a different input entirely, there is no shared convention for it to contradict.

### 3b. Inherited mask semantics

Confirmed against the code, and unchanged by this design:

- **The mask is a *retain* mask.** Set bit = keep (`PruneGrid.cuh:45`).
- **It is intersected with the value mask, not required to be a subset of it.** Both stages compute
  the same `&`: topology via `srcLeaf.valueMask().words()[w] & leafMask.words()[w]`
  (`PruneInternalNodesFunctor`), values via `dstValueMask = srcValueMask & srcLeafMask`
  (`PruneLeafMasksFunctor`). Because it is the same predicate at both levels they cannot disagree —
  a materialized leaf can never come out with an empty value mask.
- **The don't-care region survives complementation.** A bit set on an *inactive* voxel is excluded
  from `Grid0` by the AND with `valueMask`, and from `Grid1` too, since that side also ANDs with
  `valueMask`. So positions outside the value mask remain genuinely don't-care on *both* outputs.
  In practice masks derived from a label sidecar are subsets anyway, since `leaf.getValue(n)` is
  nonzero only for active voxels.

## 4. Binary — implementation shape

Of `PruneGrid::getHandle()`'s nine stages, only **two** consult the mask:

| stage | mask-dependent? | per-output? |
|---|---|---|
| `pruneRoot()` | **no** | **no — shared** |
| `allocateInternalMaskBuffers()` | no | yes |
| `pruneInternalNodes()` | **yes** — topology predicate | fused |
| `countNodes()` | no | yes |
| `getBuffer()` | no | yes |
| `processGridTreeRoot()` | no | yes |
| `processUpperNodes()` / `processLowerNodes()` | no | yes |
| `pruneLeafNodes()` | **yes** — value predicate | fused |
| `processBBox()` / `postProcessGridTree()` | no | yes |

So the change is: **two `TopologyBuilder`s, two predicates, one shared root.**

### 4a. Hoist the root computation

`pruneRoot()` is mask-independent — it speculatively keeps every source tile and lets later stages
drop the empties — so both outputs want a byte-identical result. It is also the only **serial** stage:
a `cudaStreamSynchronize`, a D2H copy of the root table, a host `std::map` build, and an upload.
Today the SDF carve pays it once per surface.

Hoist the **host** work, not the buffer: build the `prunedTiles` map once, then give each builder its
own small device buffer from that one tile list. Two uploads of a few KB is noise; the serial host
stage happens once, and `TopologyBuilder` needs no changes.

### 4b. Duplicate `mProcessedRoot`, do not share it

`mProcessedRoot` is an owned `DeviceBuffer` member of `TopologyBuilder`, and `processLowerNodes()`
calls `mProcessedRoot.clear(stream)` on it (`TopologyBuilder.cuh:456`) — the builder both owns and
frees it mid-pipeline. Sharing one buffer between two builders would either double-free or require
changing an ownership contract across all of `TopologyBuilder`'s consumers, for the benefit of one
new operator. Giving each builder its own copy resolves the lifetime coupling outright.

### 4c. The fused predicate kernels

Because the two predicates are exact complements, the fused kernel is barely more expensive than the
current one — one read of the source leaf yields both answers:

```
a = valueMask & mask          // Grid0
b = valueMask ^ a             // Grid1, i.e. valueMask & ~mask
retain0 = any(a)   retain1 = any(b)
```

### 4d. Expected payoff, honestly bounded

Fusing halves the two source-proportional passes and removes the duplicated root work. It does
**not** halve the operator: stages 4-9 are *output*-proportional, and the two outputs together are
about the size of the source, so those cost the same either way. The win is a fraction, not 2x.

## 5. Binary — validation

The binary case needs no hand-written CPU oracle — the reference implementation already exists and is
already trusted:

- `PartitionGrid(S, m)[0]` must be bit-identical to `PruneGrid(S, m)`
- `PartitionGrid(S, m)[1]` must be bit-identical to `PruneGrid(S, ~m)`

This targets precisely the class of bug a one-pass rewrite risks: the topology stage and the value
stage disagreeing about which voxels survived.

## 6. Binary — the main risk: empty outputs

A binary split produces an empty side whenever the mask is all-on or all-off. What `PruneGrid` does
only in a pathological case, `PartitionGrid` will do **routinely** — and the n-ary recursion in §7 makes
it more routine still, since every leaf of the split tree terminates on a single-label subtree.

Every guard in `PruneGrid.cuh` tests the *source* (`mSrcTreeData.mVoxelCount`,
`mNodeCount[0]`); nothing tests whether the retained set came out empty.
`MeshToSDFDevelopmentPlan.md` records an all-zero retain mask segfaulting in `processLowerNodes`, and
nothing in the current code appears to have fixed it.

**This is unknown rather than hard** — a short experiment settles what actually happens today. But it
should be settled before the contract in §8 is written, and the empty path must work before any of
the rest is worth anything.

## 7. N-ary — recursive bisection

### 7a. Shape

Split the label range in two by voxel count, build a mask (or predicate) from that pivot, call the
binary operator, and recurse on each side until a subtree holds a single label. The binary op is the
workhorse; the n-ary op is orchestration and contributes no device kernels of its own.

**The strongest argument for this over a direct N-way pass is memory, not code reuse.** A single
N-way pass would need N live `TopologyBuilder`s — each with upper/lower mask buffers sized by root
tile count — all resident at once. At anything like the hairball's 333,703 pruned components that is
fatal. Bisection holds O(depth) intermediates instead.

### 7b. Choosing the pivot — one histogram, reused everywhere

No sort is required. `ConnectedComponents::getVoxelLabelsAndCount()` already returns **dense ids in
`[0, N)`** (`ConnectedComponents.cuh:154-159`), so the per-label voxel counts needed for a balanced
pivot are a histogram into N bins plus a prefix sum — not a radix sort of V elements.

Compute it **once**. Splitting never changes a voxel's label, only which grid holds it, so the global
prefix-count array stays valid for the entire recursion. The pivot for a label range `[lo, hi)` is
then a binary search in that array, i.e. O(log N) of host arithmetic per node. The predicate is
`label < pivot`, which keeps every split a contiguous range test.

### 7c. What the split tree is actually optimizing

Total voxel work is the sum over internal nodes of the voxels in that node's subtree, which equals

```
work = Σ c_i · d_i        c_i = label i's voxel count,  d_i = its depth in the split tree
```

That is the **weighted external path length** — Huffman's objective. Minimized, it is `≈ V·H`, where
H is the entropy of the label distribution in bits, and `H ≤ log₂N` with equality only when the
counts are uniform.

This is why the pivot balances **voxel count** rather than label count, and it also settles an
apparent tension: minimizing depth and minimizing work are *aligned*, not opposed.

| distribution | optimal tree | cost | what median-split produces |
|---|---|---|---|
| uniform | balanced | `V·log₂N` | balanced |
| geometric (V/2, V/4, …) | peel one at a time | `~2V` | peel chain |

Splitting at the voxel-count median is Shannon-Fano, the standard top-down greedy approximation to
Huffman, and it adapts to both regimes automatically. Balanced is the *worst* case of this scheme,
not its typical one.

> Refinement, filed as a note rather than a plan: building a true Huffman tree on the host (N is
> small and the histogram is already there) and then **relabelling by DFS order of that tree** makes
> every Huffman subtree a contiguous id range, so splits stay `label < pivot` while achieving the
> entropy bound exactly. Shannon-Fano is typically within a few percent, so this is unlikely to be
> worth the relabelling pass.

### 7d. The distribution we expect

A polygon soup plausibly yields one or two large components plus a tail of small stragglers. That is
near the **best** case here, not a problem case: entropy is ~1-1.5 bits, so work lands near `1.5V`.
With A = 0.5V, B = 0.48V and 0.02V spread over k stragglers, the recursion peels A (work `V`), then
B (work `0.5V`), then chews the tail at `0.02V·log₂k` — negligible, because each level costs only
what remains in that subtree. Against today's `N·V`, that is roughly 15x at k = 20.

### 7e. The cost that does not shrink

There are always exactly **N−1 splits**, whatever the distribution. Each pays the serial root
computation, a `cudaStreamSynchronize`, and two full builder pipelines. On straggler subtrees those
splits touch almost no voxels, so they are pure launch overhead — and `BENCHMARK.md` §6 warns that
below ~100M voxels the carve is already "allocations, kernel launches and stream synchronizations,
not arithmetic" at a flat ~2.2 ms. Twenty stragglers could cost more than the two components holding
98% of the data.

The natural remedy is a **small-subtree cutoff**: below some voxel threshold, stop recursing and hand
the remaining labels to one direct N-way pass. The memory objection of §7a does not apply there, since
k is small and the data is a few percent. This caps the split count near `2 × (number of large
components)`. The cost is a second code path, which is a genuine trade against §7f — see §8.

### 7f. Keeping the layering honest

The binary op takes a `Mask<3>`; the recursion has a label sidecar and a pivot. Materializing a mask
at every node costs an extra kernel and a `leafCount × 64 B` allocation per split — roughly a second
`V·H` of work, which the "pure orchestration" framing hides.

The fix that preserves both properties: **template the binary op on a device predicate functor.** The
public entry point stays the `Mask<3>` one required by §3a; the recursion passes a `label < pivot`
functor. Same builders, same fused kernel — the predicate is already the only mask-dependent thing in
`pruneInternalNodes` / `pruneLeafNodes`.

### 7g. Validation

Binary has a free oracle (§5), so n-ary only needs cheap composition invariants: the N outputs' active
counts sum to the source's, and every voxel of output *i* carries a label in *i*'s range. The topology
machinery does not need re-verifying at this layer.

## 8. Open items

1. **What "empty" means.** A valid empty grid, a null handle, or a precondition violation. Blocked on
   the experiment in §6, and made urgent by §7 (every leaf of the split tree terminates on a
   single-label subtree).
2. **Sidecar carrying.** Who re-indexes UDF / nearest-triangle / label sidecars through the
   recursion. Injecting all of them at every level costs `log N` passes per sidecar; the alternative
   is to carry a single `uint32_t` *source slot* sidecar down and gather every real sidecar once at
   the leaves.
3. **Whether the §7e cutoff exists.** It bounds the fixed cost but adds a second code path, cutting
   against §7f's modularity.
4. **Which consumer we are designing for.** Every real mesh in `BENCHMARK.md` §3 has N = 1 for the
   surface partition, so the carve never runs today. The large counts on record (dragon 40, hairball
   333,703) are from the *pruned* CC, a different labelling that does not drive the carve. The
   straggler distribution of §7d is expected but, so far, unmeasured.

## Update 2026-10-19: binary implemented

### Empty outputs (§6, §8.1): an upstream bug, now fixed

The experiment of §6 ran `PruneGrid` on a non-empty source with an all-off retain mask. It failed with
an illegal memory access in `TopologyBuilder::processUpperNodes`; all-on and partial masks were fine,
and an empty source is not affected.

Cause: the builder starts from a root table holding every source root tile (`pruneRoot` copies them
all and lets later stages drop the unused ones), so with no voxel retained the table still has tiles
while the output has no upper node. `getBuffer` hands the device the table that maps each of those
tiles to its output upper node only when the output has one, but `processUpperNodes` decided whether to
run from the table's tile count, launched anyway, and read the missing map. `processLowerNodes` uses
the same tile-count test but reads the map directly, which is all zeros then, so it does no harm.

Fix (`f68e5aa6c`): `processUpperNodes` now runs only when the output has an upper node
(`nodeCount[2]`), the same test as `getBuffer`. Behavior changes only where it used to crash, so no
other consumer of the builder is affected. `PruneGrid` now returns a valid empty grid (no nodes, no
active voxel, empty bbox), and so does an empty side of `PartitionGrid`. This settles §8.1: an empty
output is a valid empty grid. `TestNanoVDBCUDA.PruneToEmpty_ValueOnIndex` covers it and fails without
the fix.

### Binary `PartitionGrid` (`51b54da7b`)

- `tools/cuda/PartitionGrid.cuh`: `PartitionGrid<BuildT, ResourceT>(srcGrid, retainMask, stream,
  resource)`, the same constructor as `PruneGrid`, and `getHandles<BufferT>()` returning
  `std::array<GridHandle<BufferT>, 2>` (§2b). `BufferT` defaults to the single-space
  `cuda::Buffer<std::byte, cuda::DeviceResource>`.
- Two `TopologyBuilder`s, run through `PruneGrid`'s stages. The source root table is read and sorted
  once on the host and copied into each builder (§4a). Each builder owns its own copy, so §4b's
  conclusion holds; its rationale is out of date, since the builder now keeps the root in pinned host
  staging plus a device copy (`allocateProcessedRoot` / `uploadProcessedRoot`) instead of a
  `DeviceBuffer` it frees mid-pipeline.
- The two mask-dependent kernels each read a source leaf once and write both outputs (§4c):
  `PartitionInternalNodesFunctor` and `PartitionLeafMasksFunctor`, next to their `PruneGrid`
  counterparts in `util/cuda/Morphology.cuh`.
- Validation as in §5: `TestNanoVDBCUDA.PartitionGrid_ValueOnIndex` checks that output 0 is
  byte-identical to `PruneGrid(m)` and output 1 to `PruneGrid(~m)`, and that their active counts add up
  to the source's, for a mixed mask and for all-on and all-off masks.

### Measured payoff (§4d)

Sphere-shell index grids (|r - R| < 3, centered at the origin, so 8 root tiles), R = 64 to 1024 (0.3M to
79M voxels), with a half-space mask (most leaves on one side), a checkerboard mask (every leaf split) and
an all-on mask. One `PartitionGrid` takes 0.68 to 0.90 of the time of `PruneGrid(m)` plus
`PruneGrid(~m)`, about 0.79 in most cases; the outputs are byte-identical in every case.

As §4d expected, the gain is a fraction. Only the host root table and the reads of the source leaves are
shared; the builder's other work is per output, and much of it scales with the number of root tiles
rather than voxels (the dense lower-node masks take 16 MB per root tile per builder to allocate and
clear, plus node counting and the lower-node kernel), which is why one `PruneGrid` costs about the same
from 0.3M to 20M voxels. With the checkerboard mask every source leaf appears in both outputs, so the
output-proportional stages cost about twice the source, and the ratio is worst (0.90 at R = 1024); the
half-space mask gains most (0.68 at R = 1024). Cutting the per-output fixed cost would mean changing
`TopologyBuilder` itself, which every topology operator shares.

The scratch memory is about twice `PruneGrid`'s, for the same reason (two sets of dense masks); the
class documents it.

### Not done yet

- The n-ary entry point (§7), including the predicate-templated binary core (§7f): deferred until the
  open items of §8.2 to §8.4 are settled.
- The example in this directory.
