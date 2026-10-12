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

1. **No empty leaves.** Every leaf in the input tree contains at least one active voxel.

   *If violated:* an empty leaf's mask stays all zeros: the leaf fill only seeds from the leaf's own
   active voxels, and no later stage writes leaf masks. In its parent lower node the leaf is a child
   slot, so it is also a wall for the lower fill. An empty leaf therefore acts as a solid block of
   outside, even when it lies entirely inside.

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

- **Coarser levels.** Leaf locality justifies the leaf stage. Still to examine: whether the upward-only
  seeding (leaf faces -> lower/upper/root slots, lower faces -> upper/root, upper faces -> root) and the
  per-node fills at the lower and upper levels reproduce the reference labels for childless slots.
- **Sweep counts at the coarser levels.** In lower and upper nodes the walls are child slots, which are
  not lined with seeds (seeds arrive only through faces from finer levels), so the leaf sweep-bound
  argument does not carry over. Without detours a Jacobi fill needs up to 45 (lower) or 93 (upper)
  sweeps; detours around children could add more.
