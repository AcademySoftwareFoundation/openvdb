# PropagateSign design notes

Working design notes for `nanovdb/nanovdb/tools/cuda/PropagateSign.cuh` (PR 2361), kept next to the
planned example while the PR is in draft. To be removed before the PR leaves draft.

## Model

- **Active voxels** form the narrow band. Each has a value index, and so a sign in `deviceSigns`:
  - *negative* (inside): `sign == -1`;
  - *nonnegative* (outside): any other value. Zero is lumped in here.
- **Inactive voxels** are every other (i,j,k). Each has a ground-truth label, inside or outside, whether
  or not it is stored at the finest resolution. It can be represented as:
  1. an inactive voxel in an existing leaf (leaf mask bit);
  2. a childless slot of a lower node, i.e. an 8^3 block (lower mask bit);
  3. a childless slot of an upper node, i.e. a 128^3 block (upper mask bit);
  4. a root value tile or absent root space, i.e. a 4096^3 block (root-tile origin list).
- The grid is assumed to have no active tiles: every active value is a leaf voxel.
- **Reference labeling.** The correct labels are those of a uniform flood fill over all of space at voxel
  resolution: start from the negative active voxels, treat nonnegative active voxels as walls, and spread
  across faces through inactive (and negative active) voxels. Inactive voxels reached are inside; all
  others are outside. Consequently, two face-adjacent inactive voxels always share a label. The
  hierarchical algorithm is correct iff it reproduces this labeling.

## Preconditions (contract with the caller)

None is checked by the operator; all must be guaranteed by how the input is built.

1. **No empty nodes.** Every leaf contains at least one active voxel, every lower node has at least one
   leaf child, and every upper node has at least one lower child. (With no active tiles, this is the same
   as: every node has at least one active voxel below it, i.e. the tree is pruned.)

   *If a leaf is empty:* its mask stays all zeros: the leaf fill only seeds from the leaf's own active
   voxels, and no later stage writes leaf masks. In its parent lower node the leaf is a child slot, so it
   is also a wall for the lower fill. An empty leaf therefore acts as a solid block of outside, even when
   it lies entirely inside.

   *If a lower (or upper) node is empty:* its childless slots form one region that borders no child, so
   it can only be seeded by finer nodes of *neighboring* nodes across its faces. If those neighbors offer
   only childless slots, nothing seeds it; its parent treats it as a wall, so its slots stay unlabeled.
   `needsRootPropagation` (`mRootMask.size() > mUpperInteriorMasks.size()`) also assumes every upper node
   lies inside the active bbox, which holds when no upper node is empty.

   *Which builders guarantee it:*
   - `TopologyBuilder` alone does **not**. It drops upper nodes with no lower child (`countNodes`), but
     creates a lower node for every bit set in the densified upper masks, even when that node's leaf mask
     is all zeros (`EnumerateNodesFunctor`, `util/cuda/Morphology.cuh`), and creates a leaf for every bit
     set in the densified lower masks without checking that its voxel mask, filled in later by the caller,
     ends up non-empty. Whether a given caller (`DilateGrid`, `RefineGrid`, ...) can produce empty nodes
     depends on how it sets those masks.
   - **`PruneGrid` output does.** `PruneInternalNodesFunctor` keeps a leaf only if its active mask
     intersected with the retain mask is non-empty, and only then sets the leaf's bit in its lower mask
     and the lower node's bit in its upper mask. `PruneLeafMasksFunctor` sets the output leaf mask to that
     same non-empty intersection. So every leaf has an active voxel, every lower node has a leaf, and (by
     `TopologyBuilder`'s rule) every upper node has a lower node.
   - `MeshToGrid` ends with exactly such a `PruneGrid` pass ("Pruning empty leaves", with an all-ones
     retain mask), so narrow bands from `MeshToGrid` satisfy precondition 1.
   - Any `TopologyBuilder`-based output has only child tiles at the root (`root.mTableSize =
     nodeCount[2]`, each tile `setChild`), so it never contains root value tiles.

2. **Two-sided shielding.** Over face-connected paths (which may pass through any voxels):
   - (a) any path from an inactive *outside* voxel to a negative active voxel transits through at least
     one nonnegative active voxel;
   - (b) any path from an inactive *inside* voxel to a nonnegative active voxel transits through at least
     one negative active voxel.

   That is, each inactive region is walled off by active voxels of its own sign. Under the reference
   labeling (adjacent inactive voxels share a label) these reduce to local conditions: no negative active
   voxel is face-adjacent to an inactive outside voxel, and no nonnegative active voxel is face-adjacent
   to an inactive inside voxel. A symmetric narrow band (at least one active voxel on each side of the
   zero crossing) satisfies both.

   *If (a) is violated:* the flood leaks into the exterior and spreads through it, up through the
   coarser levels, possibly marking large parts of the outside as inside. (This matters if the labels are
   meant to match the geometry; under the reference labeling it is what makes the two agree.)

   *If (b) is violated:* inside voxels can be missed. Example: a leaf with a full plane of nonnegative
   active voxels at local x = 4, outside for x < 4, inactive inside for x > 4, reached only through the
   leaf's +x face. The leaf has no negative voxels, so a leaf-local fill leaves x > 4 unmarked.

   Neither side can be checked by the operator, since both refer to the labels of inactive voxels,
   which are what the operator computes.

3. **The unbounded region is outside.** No negative active voxel is face-connected, through inactive
   voxels, to the inactive space outside the bbox of the active voxels (this is 2(a) with "infinity"
   counted as outside). Then every inside voxel lies within the active bbox: anything outside it is
   inactive and connected to infinity. This justifies restricting the root flood to the root tiles that
   cover the active bbox, and it means an inside root region never touches the boundary of that array.

## Results

- **Leaf locality.** Under preconditions 1 and 2(b), flood filling each leaf in isolation (seeds: its
  negative active voxels; walls: its nonnegative active voxels) gives exactly the reference labels of
  the leaf's inactive voxels. No leaf-to-leaf propagation is needed.

  *Sound:* a path inside the leaf is a path in space, so anything reached locally is reached globally.

  *Complete:* let v be an inactive inside voxel of leaf L, and C its component in L after removing the
  nonnegative active voxels. If C contains a negative active voxel, v is reached. Otherwise C is all
  inactive, hence all inside, and every voxel of L adjacent to C but not in C is nonnegative active,
  which 2(b) forbids. So nothing in L borders C; since the leaf is connected, C = L, so L has no active
  voxels, contradicting precondition 1.

- **Leaf sweep bound.** Under preconditions 1 and 2(b), a Jacobi leaf fill needs at most 21 sweeps that
  turn voxels on, plus one that confirms convergence (22 loop iterations).

  An inactive inside voxel is reached after exactly its Manhattan distance to the nearest negative active
  voxel of its leaf: walls (nonnegative active voxels) never border the inside region, since 2(b) means
  any wall facing it is lined with negative active voxels, which are seeds at sweep 0. So no wall forces
  a detour, and no serpentine path is possible. The maximum Manhattan distance in an 8^3 leaf is
  7 + 7 + 7 = 21, reached by a single negative active voxel at (0,0,0) with the other 511 voxels inactive
  and inside (consistent with all preconditions).

  Without 2(b), walls could force serpentine paths of hundreds of voxels; the only hard bound would be
  the 511 inactive voxels, which is what the current code's `MaxPropagationSweeps = 512` covers.

- **Hierarchy completeness.** Under preconditions 1-3 (and no active tiles, no root value tiles; see
  open points), propagating only upward, level by level (leaf -> lower -> upper -> root), labels every
  inside childless slot. No downward or same-level propagation is needed.

  *Lemma (inside faces).* If a child node C is face-adjacent to an inside childless slot s, every voxel
  on C's face toward s is inside: an inactive face voxel touches an inactive inside voxel of s, so it
  shares the label; an active face voxel must be negative, by 2(b).

  *Lower slots.* Let K be a connected set of inside childless slots of a lower node N (child slots
  are walls). N has a leaf (precondition 1), so K is not all of N; N's slot grid is connected, so some
  slot bordering K is outside K, i.e. holds a leaf L. By the lemma, L's face toward K is all inside, and
  by leaf locality L's mask has those bits on. `FaceSeedFunctor<0>` probes the block across that face,
  which is a slot of K, and seeds it. The lower fill spreads from that seed over all of K.

  *Upper slots.* Same argument: K borders a lower child N of U, and N's face toward K is all inside.
  That face is a 16x16 grid of 8^3 patches. A leaf patch seeds K directly (its `FaceSeedFunctor<0>`
  probe returns level 2); a childless lower-slot patch is inside, so it was already labeled at the lower
  level, and `FaceSeedFunctor<1>` seeds K from it. This is why leaves seed every coarser level directly:
  `FaceSeedFunctor<1>` skips N's child slots, so a lower face covered entirely by leaves would otherwise
  seed nothing.

  *Root.* Same argument one level up. An inside root region K borders some upper node U (precondition 3:
  K never reaches the boundary of the root array, so it is not all of it); U's face toward K is all
  inside, and each of its patches (childless upper slot, childless lower slot, or leaf) seeds K in
  stage 6, 4 or 2 respectively.

  *Soundness.* A slot is seeded only when it touches an inside voxel, and the slots joined during a fill
  share faces, so every labeled slot is truly inside.

  *Stage order.* Each level's seeds come only from finer levels that are already final, and each fill
  runs after all its seeds have been written (seeds in stages 2 and 4, upper fill in stage 5, root fill in
  stage 7).

## Unified node-local propagator (proposed)

### Current code

There are three separate fill implementations:

| | threads | state | updates | seeds | walls |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `LeafPropagationFunctor` | 1 per voxel (512) | 1 byte/voxel, shared | Jacobi (barrier between read and write) | active, sign -1 | active, sign >= 0 |
| `InternalPropagationFunctor<1/2>` | 512, each looping over 8 or 64 slots | bitmask words, shared, `setOnAtomic` | asynchronous (no barrier within a sweep) | slots seeded by `FaceSeedFunctor` | `childMask()` |
| `RootPropagationFunctor` | 1 block of 256 | 1 byte/position, global | asynchronous, plain byte stores | slots seeded by `FaceSeedFunctor` | `RootWallFunctor` (upper children) |

The asynchronous fills still converge to the same fixed point (updates only turn bits on, so every
order reaches the same reachable set), and their stopping test is sound (a sweep that writes nothing
reads a consistent state). Only the sweep count is nondeterministic. But the inconsistency with the
leaf fill, the atomics, and the formal races (plain loads of words updated with `atomicOr`; plain byte
stores at the root) are avoidable.

### Proposal

The leaf and internal fills solve the same problem: given `Mask<Log2Dim>`-shaped seeds and walls, compute
everything reachable from the seeds without crossing a wall. Only the setup differs per level. So:

- **One fill, templated on `Log2Dim`**, shared by leaf (3), lower (4) and upper (5) nodes.
- **State: one byte per slot in shared memory** (inside / not inside): 512 B for a leaf, 4 KB for a lower
  node, 32 KB for an upper node. This is the only shared array.
- **Thread per slot.** At most 1024 threads fit in a block, so internal nodes cannot be one thread per
  slot; each thread owns a fixed set of slots (1, 8 or 64), strided by the block size.
- **Walls stay in registers.** A slot's update needs only its own wall bit and whether any neighbor is
  inside; it never asks whether a neighbor is a wall. So a thread holds the wall bits of the slots it owns
  in one `uint64_t` register (leaf: a single flag, as `isActive` today; lower/upper: read straight from
  `childMask().words()`).
- **Deterministic Jacobi sweeps.** Each thread records the decisions for its slots in a `uint64_t`
  register, then a barrier, then it writes the bytes it owns. No atomics, since every byte has a single
  owner. Stop on a sweep with no change, as today.
- **Output** packed into the `Mask` words with `__ballot_sync`, as the leaf does today.
- **Per-level setup**, done in the same launch with the same thread-to-slot mapping:
  - leaf: thread n reads `signs[leaf.getValue(n)]` and sets its seed byte and wall flag directly (no
    separate pass to build masks);
  - lower/upper: seed bytes from the bits `FaceSeedFunctor` set, wall bits from `childMask()`.

The root keeps its own fill: it is a dense array of arbitrary size, not a `Mask<n>`. It could adopt the
same Jacobi-with-register-decisions pattern to remove its races.

### Alternative considered: word-parallel bitwise fill

Each thread owns whole 64-bit mask words and spreads them with shifts: since `n = (x << 2L) | (y << L) | z`,
a word is a contiguous slab (`Mask<3>`: word = x, ±y = shift by 8; `Mask<4>`: word = 4x + y/4, ±y = shift
by 16 with carry from word ±1, ±x = word ±4; `Mask<5>`: word = 16x + y/2, ±y = shift by 32 with carry,
±x = word ±16; ±z = shift by 1 with the wrap masked out). One sweep is `new = (I | spread(I)) & ~walls`,
with 64 slots per instruction, no atomics, deterministic.

Not pursued for now:
- the fills are probably not the bottleneck: the leaf stage is likely dominated by the per-voxel global
  reads `leaf.getValue(n)` / `signs[...]`, and lower/upper nodes are few;
- a leaf has only 8 words, so leaves would need packing several per warp or block, while the sign lookup
  is inherently per voxel: setup and fill would need different thread mappings and an extra mask-building
  step;
- more complex shift/carry logic per level.

Measure first (`setVerbose(1)` times each stage). If the lower/upper fills ever matter, the word-parallel
fill can replace the shared fill function without changing its interface.

## Open points

- **Bug: root value tiles are never seeded.** `probeChildlessSlot` returns 0 for a root tile that is not
  a child, so no face seeds it; but `RootWallFunctor` does not treat it as a wall either. An inside value
  tile is then labeled only if the root fill reaches it from another seeded position; one that can only
  be seeded directly by an upper face (e.g. wedged between upper nodes) stays unlabeled. Fix: return 3
  for any non-child root tile; the level-3 branch of `seedAcrossFace` already derives the array position
  from `ijk`. Latent for `TopologyBuilder`-based inputs, which have no root value tiles.
- **No active tiles** is still an unchecked assumption: the internal fills treat every childless slot as
  passable inactive space.
- **Sweep counts at the coarser levels.** In lower and upper nodes the walls are child slots, which are
  not lined with seeds (seeds arrive only through faces from finer levels), so the leaf sweep-bound
  argument does not carry over. Without detours a Jacobi fill needs up to 45 (lower) or 93 (upper)
  sweeps; detours around children could add more.

## Code-level review findings

Smaller items noted against `PropagateSign.cuh` as of commit `668691585`:

- **Sign convention docs.** The constructor's `@param deviceSigns` says "-1 inside and +1 outside", but
  the code tests `== int8_t(-1)`, so any other value (including 0) is outside. Document it as "-1 for
  inside (negative), anything else for outside (nonnegative)", and that mapping zero to nonnegative is
  the caller's job.
- **Preconditions undocumented.** None of preconditions 1-3 appear in the header; the class `@details`
  should state them as the caller contract.
- **Leaf sweep limit.** `MaxPropagationSweeps = 512` (leaf) could be 22 under the preconditions; the
  `NANOVDB_ASSERT` after the loop would then also catch inputs whose detours violate 2(b).
- **Stale comments.**
  - The leaf-fill comment weighs a scanline fill against Jacobi for "long corridors"; under the
    preconditions corridors cannot occur inside a leaf.
  - The stage 3/5 section comment says the lower/upper fills use "sweeps like those of the leaf fill",
    but the leaf fill is Jacobi (barrier between read and write) while the lower/upper and root fills are
    asynchronous (no barrier within a sweep).
- **Formal races.** The asynchronous fills read words that other threads update in the same sweep
  (plain loads against `setOnAtomic`; plain byte loads/stores in the root fill). Benign, since every
  racing writer stores 1, but avoidable with the proposed unified fill.
- **Open `todo`s:** naming of the interior / invert masks, `probeChildlessSlot`'s name, and renaming
  `signs` to `signSidecar`.
- **Packaging.**
  - `tools/cuda/PropagateSign.cuh` is not in the install list in `nanovdb/nanovdb/CMakeLists.txt` (unlike
    `tools/cuda/SignedFloodFill.cuh`), so it will not be installed.
  - No unit test. A GPU regression test from the earlier draft exists
    (`examples/propagate_sign_test/propagate_sign_test.cu` in commit `4e2e13a41` on `sifakis/mesh-to-sdf`).
  - No example yet; this directory is reserved for it (needs a `nanovdb_example(NAME ...)` entry).
