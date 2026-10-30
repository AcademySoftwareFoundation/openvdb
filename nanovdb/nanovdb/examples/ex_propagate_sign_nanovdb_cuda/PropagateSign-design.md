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
- Every active value is a leaf voxel, and the root holds only child tiles (precondition 4). Childless
  slots of lower and upper nodes are still inactive tiles in NanoVDB's sense (in a `ValueOnIndex` grid
  their value is index 0, the background); those are exactly the inactive blocks of items 2 and 3.
- **Reference labeling.** The correct labels are those of a uniform flood fill over all of space at voxel
  resolution: start from the negative active voxels, treat nonnegative active voxels as walls, and spread
  across faces through inactive (and negative active) voxels. Inactive voxels reached are inside; all
  others are outside. Consequently, two face-adjacent inactive voxels always share a label. The
  hierarchical algorithm is correct iff it reproduces this labeling.

## Preconditions (contract with the caller)

Preconditions 1-3 are not checked by the operator; they must be guaranteed by how the input is built.
Precondition 4 is checked on the host since 2026-10-30 (see the update at the end of these notes).

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
   lies inside the active bbox, which holds when no upper node is empty. (The sparse root proposal at the
   end of these notes removes both `needsRootPropagation` and the dense root array.)

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

4. **Active values are voxels only, and the root holds only child tiles.**
   - (a) No active tiles at any level: every active domain is a leaf voxel. *If violated:* an active tile
     carries a sign that the code never reads; at the lower/upper levels it is a childless slot and at the
     root a non-child tile, so the fills treat it as passable inactive space. A nonnegative active tile
     then fails to block the flood, breaking 2(a).
   - (b) Every root tile is a child tile (no root value tiles, active or inactive). *If violated:* see the
     root value-tile seeding gap under Open points.

   Both hold for any `TopologyBuilder`-based output (including `PruneGrid` and `MeshToGrid`), whose root
   has only child tiles and whose active values are leaf voxels.

   *Checked* (since 2026-10-30): `initializeMasks()` throws if any `mTileCount` is nonzero (a) or if the
   root table has more entries than there are upper nodes (b).

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
  the 511 inactive voxels, which is what the current code's backstop (`SlotCount + 1` sweeps) covers.

- **Hierarchy completeness.** Under preconditions 1-4, propagating only upward, level by level
  (leaf -> lower -> upper -> root), labels every inside childless slot. No downward or same-level
  propagation is needed. (Code names below are as of `700ebd682`: `BitFloodFunctor<BuildT, Level>` fills
  each node and then seeds across its faces with `seedAcrossFace<BuildT, Level>`; stages 1-3 are the
  leaf, lower and upper kernels, stage 4 the root fill.)

  *Lemma (inside faces).* If a child node C is face-adjacent to an inside childless slot s, every voxel
  on C's face toward s is inside: an inactive face voxel touches an inactive inside voxel of s, so it
  shares the label; an active face voxel must be negative, by 2(b).

  *Lower slots.* Let K be a connected set of inside childless slots of a lower node N (child slots
  are walls). N has a leaf (precondition 1), so K is not all of N; N's slot grid is connected, so some
  slot bordering K is outside K, i.e. holds a leaf L. By the lemma, L's face toward K is all inside, and
  by leaf locality L's mask has those bits on. The leaf kernel's `seedAcrossFace<BuildT, 0>` probes the
  block across that face, which is a slot of K, and seeds it. The lower fill spreads from that seed over
  all of K.

  *Upper slots.* Same argument: K borders a lower child N of U, and N's face toward K is all inside.
  That face is a 16x16 grid of 8^3 patches. A leaf patch seeds K directly (its level-0 probe returns
  level 2); a childless lower-slot patch is inside, so it was already labeled at the lower level, and the
  lower kernel seeds K from it. This is why leaves seed every coarser level directly: a lower node's face
  test only sees its childless slots (its interior mask is restricted to them), so a lower face covered
  entirely by leaves would otherwise seed nothing.

  *Root.* Same argument one level up. An inside root region K borders some upper node U (precondition 3:
  K never reaches the boundary of the root array, so it is not all of it); U's face toward K is all
  inside, and each of its patches (childless upper slot, childless lower slot, or leaf) seeds K from the
  upper, lower or leaf kernel respectively.

  *Soundness.* A slot is seeded only when it touches an inside voxel, and the slots joined during a fill
  share faces, so every labeled slot is truly inside.

  *Stage order.* Each level's seeds come only from finer levels that are already final, and each fill
  runs after all its seeds have been written: the leaf kernel (stage 1) seeds lower, upper and root; the
  lower kernel (stage 2) fills, then seeds upper and root; the upper kernel (stage 3) fills, then seeds
  the root; the root fill is stage 4. A kernel only seeds strictly coarser levels, so it never writes a
  mask that another warp of the same launch is filling.

## Node fills

### Current code (`700ebd682`)

One template, `BitFloodFunctor<BuildT, Level>`, fills leaf (0), lower (1) and upper (2) nodes, one warp per
node and 4 nodes per block. The warp holds the node's mask in registers as a `WarpSlicedMask<Log2Dim>`:
lane x holds x-slice x, i.e. `WordsPerSlice` = 1, 4 or 16 consecutive words of the `Mask` (y major, z
minor within the slice). Each sweep is `interior | (open & interior.faceNeighbors())`:

- z neighbors: 1-bit shifts within each word, masked so they do not cross from one y to the next;
- y neighbors: shifts by `Dim` bits, carrying the edge y in from the neighboring word of the slice;
- x neighbors: `__shfl_up_sync` / `__shfl_down_sync` from lanes x - 1 and x + 1;
- stop: `__any_sync` reports that no lane's words changed.

The leaf's seeds are built with `fromSlots`: 16 `__ballot_sync`s over all 32 lanes, each testing
`signs[getValue(n)] == -1` for 32 consecutive voxels. After storing its mask, the warp votes which faces
hold an inside bit (`facesWithBits`), and lanes 0-5 each call `seedAcrossFace` for one face, so seeding
runs in the same kernel and the pipeline has four stages (leaf, lower, upper, root). The root fill is
unchanged: a single block, asynchronous sweeps over a dense byte array.

Properties: deterministic Jacobi at every node level, no atomics or races inside the node fills, and
5-9x faster `propagate()` on MeshToSDF output for 10 meshes, with identical masks (per the commit
message). Lane use: the sign gather uses all 32 lanes; the sweeps use `Dim` lanes, i.e. 8 (leaf),
16 (lower) or 32 (upper), so leaf and lower sweeps run at 25% and 50% lane utilization.

Superseded earlier ideas: a byte-per-slot shared-memory fill with one thread per slot (proposed in this
review before `700ebd682`), and a word-parallel fill, which `700ebd682` effectively implemented.

### Proposal: a thread-agnostic `Mask2D` with `MaskShift` as the shift vocabulary

**Motivation.** `WarpSlicedMask` exposes the thread structure in the type: its data is a plain
(y, z) slice (its only non-static member is `uint64_t words[WordsPerSlice]`, so it is trivial and
standard-layout), but its member functions read `threadIdx.x` (`lane()`, `ownsSlice()`), so even its
simple operations depend on which lane runs them (`~` keeps lanes `>= Dim` empty). By contrast,
`util::morphology::cuda::MaskShift<di, dj, dk>` (DilateGrid) takes a plain `Mask<4>`; the lane mapping
lives inside the function, computed from `threadIdx.x` and discarded. The proposal applies the same
separation here: a plain data type for one slice, plus warp-level functions that coordinate slices.
It also reuses `MaskShift`, a concept the library already has and DilateGrid has exercised, instead of
introducing new vocabulary such as "face neighbors" or "in-plane neighbors".

**`Mask2D<Log2Dim>` (in `nanovdb/util/`, `__hostdev__`).** One x-slice of a `Mask<Log2Dim>`: the
Dim x Dim plane of slots (y, z), in the mask's own layout (slot (y, z) is bit y * Dim + z of the slice;
the Dim slots of one y are contiguous bits, and every 64-bit word holds whole y's). A plain aggregate
(`Mask2D m{}` is all off) with no notion of threads and no intrinsics. Its interface mirrors `Mask`:

- `load(const Mask<L>&, int x)` / `store(Mask<L>&, int x)`: the slice index is explicit;
- `isOn()`, `isOff()`, `setOn()`, `setOff()`, `toggle()`, `==`, `!=`, `&=`, `|=`, as in `Mask`;
- value-returning `&` and `|` in addition, for slices held in registers (`Mask` has only the compound
  forms).

Probing which edges of a slice hold bits is algorithm-specific (it serves the face seeding) and has no
counterpart on `Mask`, so it is a free function on the algorithm side, not a member.

**`MaskShift<dj, dk, combine>` on `Mask2D` (same file).** `out(y, z) |= in(y + dj, z + dk)` (or `=` if
`!combine`), with zeros from outside the slice: the same convention as the 3D `MaskShift`. Plain
per-thread code; any offset with `|dj|, |dk| < Dim`; `in` and `out` may alias. A z shift moves each
word by `dk` bits and masks off what crossed from one y into the next; a y shift moves whole y's, i.e.
whole words plus a carried remainder, and needs no masking.

**`detail::zShiftBits<Log2Dim, dk>`.** The bits of a word, in every y of the word, whose slot z has a
source under a z shift by dk (0 <= z + dk < Dim): the slots `MaskShift` does not zero-fill. A single
compile-time expression: one y's worth, `(yBits >> max(dk, 0)) & (yBits << max(-dk, 0))`, replicated
to every y of the word by multiplying with `~0 / yBits` (exact, since Dim divides 64). The slice's z
edges are its complements for unit shifts: z = 0 is `~zShiftBits<L, -1>`, z = Dim - 1 is
`~zShiftBits<L, +1>`.

**Warp layer (in `util/cuda/MorphologyHelpers.cuh`, next to the existing `MaskShift`).** Everything that
crosses lanes, documented as warp collectives:

- the x part of a shift: shuffles by `di` lanes within groups of `Dim` lanes (`width = Dim`), with an
  explicit mapping `x = lane % Dim`, `node = lane / Dim`; together with the 2D shift this gives a 3D
  `MaskShift<di, dj, dk>` over a warp of slices, and the fill's face pass is the OR of the six unit shifts;
- votes: any-differs (stopping test) and face flags (x faces from slices 0 and Dim - 1 being non-empty,
  y/z faces from the edge probe of every slice);
- building slices from per-slot predicates with ballots (today's `fromSlots`).

Lanes without a slice are ignored by the votes and stores; the rule lives with the code that defines
the mapping instead of in the data type.

**Payoffs.**
- `PropagateSign.cuh` loses `repeatBits` and `WarpSlicedMask` (about 160 of ~710 lines) and keeps only
  the algorithm.
- Packing comes almost for free: with `width = Dim` shuffles and per-group votes, a warp can fill 4
  leaves or 2 lower nodes, recovering the idle lanes (the stopping vote can stay warp-wide; converged
  nodes just run no-op sweeps).
- `Mask2D` and the 2D `MaskShift` run on the host, so they can be tested exhaustively on the CPU
  (`nanovdb_test_nanovdb`), and a CPU reference or CPU implementation of the fill can reuse them (the warp
  layer's "slice at x +/- d" becomes `slices[x +/- d]`).
- One shift vocabulary across DilateGrid and PropagateSign; ConnectedComponents already builds masks
  from ballots (`ConnectedComponents.cuh:420-440`), a candidate for the shared slice builder.

**Prototype (validated).** The code below was compiled and tested outside the repo: g++ (`-O2` and `-O0`,
`-Wall -Wextra -Werror`) and nvcc 13.3 host code, all with 0 mismatches against slot-by-slot references
for every (dj, dk) in range at all three sizes (combine and in-place overwrite), `zShiftBits` for every
dk, the edge probe, the `Mask`-style members, and `load`/`store` (slot (x, y, z) of the `Mask` is slot
(y, z) of slice x). A device kernel doing one fill step (four in-plane shifts, `s | (open & n)`, the edge
probe) compiles for sm_120 with all warnings as errors, no spills, 20 / 33 / 40 registers for `Mask<3>` /
`<4>` / `<5>`.

```cpp
namespace nanovdb::util {

/// @brief One x-slice of a Mask<Log2Dim>: the Dim x Dim plane of slots (x, y, z) with x fixed, in the
///        mask's own layout. Slot (y, z) is bit y * Dim + z of the slice, so the Dim slots of one y are
///        contiguous bits, and every 64-bit word holds whole y's. A plain aggregate with no notion of
///        threads: Mask2D m{} is all off.
template<int Log2Dim>
struct Mask2D
{
    static_assert(Log2Dim >= 3 && Log2Dim <= 5, "every 64-bit word must hold whole y's of the slice");
    static constexpr int Dim       = 1 << Log2Dim;   // slots per axis: 8, 16, 32
    static constexpr int WordCount = Dim * Dim / 64; // 1, 4, 16

    uint64_t words[WordCount];

    /// @brief Slice @a x of @a mask, and its inverse.
    __hostdev__ static Mask2D load(const Mask<Log2Dim>& mask, int x)
    {
        Mask2D s;
        for (int j = 0; j < WordCount; ++j) s.words[j] = mask.words()[x * WordCount + j];
        return s;
    }
    __hostdev__ void store(Mask<Log2Dim>& mask, int x) const
    {
        for (int j = 0; j < WordCount; ++j) mask.words()[x * WordCount + j] = words[j];
    }

    // Whole-slice queries and updates, named as in Mask.
    __hostdev__ bool isOn() const  { uint64_t a = ~uint64_t(0); for (int j = 0; j < WordCount; ++j) a &= words[j]; return a == ~uint64_t(0); }
    __hostdev__ bool isOff() const { uint64_t a = 0; for (int j = 0; j < WordCount; ++j) a |= words[j]; return a == 0; }
    __hostdev__ void setOn()       { for (int j = 0; j < WordCount; ++j) words[j] = ~uint64_t(0); }
    __hostdev__ void setOff()      { for (int j = 0; j < WordCount; ++j) words[j] = 0; }
    __hostdev__ void toggle()      { for (int j = 0; j < WordCount; ++j) words[j] = ~words[j]; }

    __hostdev__ bool operator==(const Mask2D& o) const { bool eq = true; for (int j = 0; j < WordCount; ++j) eq &= words[j] == o.words[j]; return eq; }
    __hostdev__ bool operator!=(const Mask2D& o) const { return !(*this == o); }
    __hostdev__ Mask2D& operator&=(const Mask2D& o)    { for (int j = 0; j < WordCount; ++j) words[j] &= o.words[j]; return *this; }
    __hostdev__ Mask2D& operator|=(const Mask2D& o)    { for (int j = 0; j < WordCount; ++j) words[j] |= o.words[j]; return *this; }

    // Value-returning forms, for slices held in registers (Mask has only the compound forms).
    __hostdev__ Mask2D operator&(const Mask2D& o) const { Mask2D r = *this; return r &= o; }
    __hostdev__ Mask2D operator|(const Mask2D& o) const { Mask2D r = *this; return r |= o; }
}; // Mask2D<Log2Dim>

namespace detail {

/// @brief The bits of one y of a Mask2D<Log2Dim> word (its Dim low bits).
template<int Log2Dim>
inline constexpr uint64_t yBits = (uint64_t(1) << (1 << Log2Dim)) - 1;

/// @brief The bits of a Mask2D<Log2Dim> word, in every y of the word, whose slot z has a source under
///        a z shift by dk, i.e. 0 <= z + dk < Dim: the slots that MaskShift<dj, dk> does not zero-fill.
template<int Log2Dim, int dk>
inline constexpr uint64_t zShiftBits =
    ((yBits<Log2Dim> >> (dk > 0 ? dk : 0)) & (yBits<Log2Dim> << (dk < 0 ? -dk : 0))) // one y: z in [-dk, Dim - dk)
    * (~uint64_t(0) / yBits<Log2Dim>);                                               // repeated for every y

} // namespace detail

/// @brief out(y, z) |= in(y + dj, z + dk), or = if !combine, with zeros from outside the slice: the 2D,
///        per-thread counterpart of util::morphology::cuda::MaskShift. @a in and @a out may be the same slice.
template<int dj, int dk, bool combine = true, int Log2Dim>
__hostdev__ void MaskShift(const Mask2D<Log2Dim>& in, Mask2D<Log2Dim>& out)
{
    using SliceT = Mask2D<Log2Dim>;
    constexpr int Dim = SliceT::Dim, WordCount = SliceT::WordCount;
    static_assert(dj > -Dim && dj < Dim && dk > -Dim && dk < Dim, "the shift must stay within the slice");

    // z: shift every word, then clear the slots whose source lies outside the slice (they received bits
    // from the neighboring y instead).
    uint64_t t[WordCount];
    for (int j = 0; j < WordCount; ++j) {
        if constexpr (dk > 0)      t[j] = (in.words[j] >>  dk) & detail::zShiftBits<Log2Dim, dk>;
        else if constexpr (dk < 0) t[j] = (in.words[j] << -dk) & detail::zShiftBits<Log2Dim, dk>;
        else                       t[j] = in.words[j];
    }

    // y: whole y's, so the slice moves as one (64 * WordCount)-bit integer by |dj| * Dim bits,
    // i.e. Q whole words plus R bits carried in from the next word over.
    constexpr int Q = (dj < 0 ? -dj : dj) * Dim / 64;
    constexpr int R = (dj < 0 ? -dj : dj) * Dim % 64;
    SliceT r;
    for (int i = 0; i < WordCount; ++i) {
        uint64_t w = 0;
        if constexpr (dj >= 0) { // toward lower bits: out bit b = t bit b + |dj| * Dim
            if (i + Q < WordCount) w = t[i + Q] >> R;
            if constexpr (R != 0) { if (i + Q + 1 < WordCount) w |= t[i + Q + 1] << (64 - R); }
        } else {                 // toward higher bits: out bit b = t bit b - |dj| * Dim
            if (i - Q >= 0) w = t[i - Q] << R;
            if constexpr (R != 0) { if (i - Q - 1 >= 0) w |= t[i - Q - 1] >> (64 - R); }
        }
        r.words[i] = w;
    }
    for (int i = 0; i < WordCount; ++i) out.words[i] = combine ? (out.words[i] | r.words[i]) : r.words[i];
}

} // namespace nanovdb::util
```

The algorithm-side edge probe (would live in `PropagateSign.cuh`):

```cpp
/// @brief Bits 0-3: whether slice @a s has an on slot with y = 0, y = Dim - 1, z = 0 or z = Dim - 1.
///        The z edges are the slots a unit z shift zero-fills: z = 0 has no source under dk = -1, and
///        z = Dim - 1 none under dk = +1.
template<int Log2Dim>
__hostdev__ uint32_t edgesWithBits(const nanovdb::util::Mask2D<Log2Dim>& s)
{
    using namespace nanovdb::util;
    constexpr int      Dim     = Mask2D<Log2Dim>::Dim, WordCount = Mask2D<Log2Dim>::WordCount;
    constexpr uint64_t FirstY  = detail::yBits<Log2Dim>;               // y = 0: the low Dim bits of the first word
    constexpr uint64_t LastY   = detail::yBits<Log2Dim> << (64 - Dim); // y = Dim - 1: the high Dim bits of the last word
    constexpr uint64_t FirstZ  = ~detail::zShiftBits<Log2Dim, -1>;     // z = 0, in every y of a word
    constexpr uint64_t LastZ   = ~detail::zShiftBits<Log2Dim, +1>;     // z = Dim - 1, in every y of a word
    uint64_t firstZ = 0, lastZ = 0;
    for (int j = 0; j < WordCount; ++j) { firstZ |= s.words[j] & FirstZ; lastZ |= s.words[j] & LastZ; }
    return uint32_t((s.words[0] & FirstY) != 0) | uint32_t((s.words[WordCount - 1] & LastY) != 0) << 1 |
           uint32_t(firstZ != 0) << 2 | uint32_t(lastZ != 0) << 3;
}
```

**Still open.**
- The warp layer is not prototyped yet (x shuffles with `width = Dim`, votes, ballot-built slices).
- Naming: overload `MaskShift` (the argument types disambiguate `MaskShift<dj, dk>` on `Mask2D` from
  `MaskShift<di, dj, dk>` on `Mask<4>`) or call it `MaskShift2D`; leaning towards the overload.
- Scope: `Mask2D` changes shared utilities, so it may be better as a separate small PR (with its CPU unit
  test) that 2361 builds on, or a separate commit within 2361 while it is a draft.
- Later: express DilateGrid's `Mask<4>` `MaskShift` through the same pieces, or at least cross-reference
  the two (its layout spreads `Mask<4>` over all 32 lanes, 2 words each; `Mask2D` uses one slice per lane).

## Open points

- **Root value tiles are never seeded (excluded by precondition 4(b)).** `probeChildlessSlot` returns 0
  for a root tile that is not a child, so no face seeds it; but `RootWallFunctor` does not treat it as a
  wall either. An inside value tile is then labeled only if the root fill reaches it from another seeded
  position; one that can only be seeded directly by an upper face (e.g. a single value tile enclosed by
  upper nodes) stays unlabeled, although absent space in the same place would be labeled. With 4(b) in
  the contract this branch is unreachable: replace it with an assert (or, to support root value tiles
  later, return 3 for any non-child root tile; the level-3 branch of `seedAcrossFace` already derives
  the array position from `ijk`).
- **Sweep counts at the coarser levels.** In lower and upper nodes the walls are child slots, which are
  not lined with seeds (seeds arrive only through faces from finer levels), so the leaf sweep-bound
  argument does not carry over. Without detours a Jacobi fill needs up to 45 (lower) or 93 (upper)
  sweeps; detours around children could add more.

## Code-level review findings

Smaller items, first noted against `668691585` and re-checked against `700ebd682`:

- **Sign convention docs.** The constructor's `@param deviceSigns` says "-1 inside and +1 outside", but
  the code tests `== int8_t(-1)`, so any other value (including 0) is outside. Document it as "-1 for
  inside (negative), anything else for outside (nonnegative)", and that mapping zero to nonnegative is
  the caller's job. Still open.
- **Preconditions undocumented.** None of preconditions 1-4 appear in the header; the class `@details`
  should state them as the caller contract. Still open.
- **Leaf sweep limit.** The backstop is `SlotCount + 1` sweeps (513 for a leaf; `MaxPropagationSweeps = 512`
  before `700ebd682`). It could be 22 for leaves under the preconditions; the `NANOVDB_ASSERT` after the
  loop would then also catch inputs whose detours violate 2(b).
- **Stale comment.** The stage 1-3 comment still weighs a scanline fill against Jacobi for "long
  corridors"; under the preconditions corridors cannot occur inside a leaf. (The earlier mismatch, where
  the lower/upper fills were asynchronous while the comment called them "like the leaf fill", is resolved:
  all node fills are Jacobi now.)
- **Formal races.** Resolved for the node fills by `700ebd682`. The root fill still reads and writes plain
  bytes that other threads update in the same sweep; benign (every racing writer stores 1), but avoidable.
- **Register pressure.** For upper nodes a sweep keeps several 16-word `WarpSlicedMask`s live; check for
  spills with `-Xptxas -v`.
- **Not compiled anywhere.** No unit test, example or installed header includes `PropagateSign.cuh`, so
  neither CI nor `NANOVDB_CUDA_WERROR` sees it.
- **Lane utilization.** The leaf and lower sweeps use 8 and 16 of 32 lanes; packing 4 leaves or 2 lower
  nodes per warp would recover them (see the `Mask2D` proposal).
- **Cross-reference `MaskShift`.** `WarpSlicedMask::faceNeighbors()` is the OR of six unit
  `MaskShift`s; at least point to `util/cuda/MorphologyHelpers.cuh` (see the `Mask2D` proposal).
- **Open `todo`s:** naming of the interior / invert masks, `probeChildlessSlot`'s name, and renaming
  `signs` to `signSidecar`.
- **Packaging.**
  - `tools/cuda/PropagateSign.cuh` is not in the install list in `nanovdb/nanovdb/CMakeLists.txt` (unlike
    `tools/cuda/SignedFloodFill.cuh`), so it will not be installed.
  - No unit test. A GPU regression test from the earlier draft exists
    (`examples/propagate_sign_test/propagate_sign_test.cu` in commit `4e2e13a41` on `sifakis/mesh-to-sdf`).
  - No example yet; this directory is reserved for it (needs a `nanovdb_example(NAME ...)` entry).

## Update 2026-10-18: `Mask2D` adopted, review findings addressed

### Code changes

- **`Mask2D` and the 2D `MaskShift`** are in `util/MorphologyHelpers.h` as in the proposal above, in
  `nanovdb::util::morphology` (next to the existing `MaskShift`) rather than `nanovdb::util`, with
  `detail::yBits` and `detail::zShiftBits`. `MaskShift` is overloaded, not renamed.
- **Warp layer: one function.** `util/cuda/MorphologyHelpers.cuh` adds a 3D
  `MaskShift<di, dj, dk, combine>` on `Mask2D` slices held by a warp, one node per warp: lane x holds
  x-slice x, and lanes x >= Dim hold empty slices. The x part is a `__shfl_up_sync` / `__shfl_down_sync`
  by `di` lanes; a lane whose source lies outside the node, or that has no slice, takes zeros (a shuffle
  from outside the warp returns the caller's own value, so this guard is needed). y and z use the 2D
  `MaskShift` within the lane. The two votes stay inline in `BitFloodFunctor`, since each is one line:
  `__any_sync` to stop, and an OR of the face sets over the 32 lanes by xor-shuffles
  (`__reduce_or_sync` would need sm_80).
- **`PropagateSign.cuh`.** `repeatBits` and `WarpSlicedMask` are gone. `BitFloodFunctor` fills a node
  with six unit `MaskShift`s per sweep. Leaf seeds come from `leafSeedSlice(leaf, signs)` (the former
  `fromSlots`, leaf only): each `__ballot_sync` tests 32 consecutive voxels, half of one slice, and the
  lane holding that slice keeps the answers. The edge probe and the x faces are one function,
  `facesWithOnSlots(slice, x)`, which returns a set of `Face` bits directly.
- **`probeChildlessSlot` -> `probeTerminalSlot`,** which returns a `TerminalSlot {level, nodeID, slot}`:
  the slot that holds the coordinate's value. `level` is the `LEVEL` of the node holding it
  (`NanoLeaf`/`NanoLower`/`NanoUpper<BuildT>::LEVEL`, or `NanoRoot`'s for an absent root-tile position)
  instead of the literals 0-3.
- **Root value tiles** (precondition 4(b)) now trip a `NANOVDB_ASSERT` in both places that meet them,
  `probeTerminalSlot` and `RootWallFunctor`, so a debug build catches any such tile inside the root-tile
  array. Release behavior is unchanged: such a tile is neither seeded nor a wall (see the open point
  above).
- **Sweep bounds as asserts.** Every fill still loops until a sweep changes nothing (at most
  `SlotCount + 1` sweeps for a node, `total + 1` for the root array), so release builds converge on any
  input. A `NANOVDB_ASSERT` after the loop checks the bound that holds under preconditions 1 and 2:
  22, 45 and 93 sweeps for leaf, lower and upper nodes, and (W - 1) + (H - 1) + (D - 1) for a W x H x D
  root-tile array (argument below).
- **Root fill.** Jacobi sweeps over two byte arrays (read one, write the other), with
  `__syncthreads_or` as the stopping vote, so it no longer reads and writes the same bytes in one sweep.
- **Scratch release.** `propagate()` destroys the root-tile array (`mRootMask`) before returning, ordered
  on the stream, so a returned operator holds no device memory and the resource only has to outlive the
  returned buffers, as ConnectedComponents now does.
- **Launches** use `util::cuda::blocksPerGrid(count * 32, MaxThreadsPerBlock)`; `NodesPerBlock` stays 4.
  Blocks have no meaning in these kernels (no shared memory, no `__syncthreads`), only for scheduling: on
  the meshes below, 1 node per block was 25% slower in the leaf stage (median), 2 and 4 the same, and 8
  and 16 up to 6% and 13% slower on the largest cases.
- **Docs.** The class `@details` lists preconditions 1-4 briefly, and a todo to handle inputs that break
  them (e.g. root value tiles from OpenVDB level sets converted to NanoVDB). `deviceSigns`: -1 inside,
  any other value (including 0) outside. The stage comment no longer weighs a scanline fill. The root
  comments say that seeds come from leaf, lower and upper faces (each kernel seeds every coarser level),
  and use one term for a cell of the root array: a root-tile position, absent if the root has no tile
  there.

Verified: the masks are identical to those of the `700ebd682` fill and of the earlier byte-per-slot fill
on 29 MeshToSDF cases (10 meshes at 512, 1024, 2048; hairball at 512 and 1024), in release and in debug
builds, with no assert; the regression test passes 14/14 in release. In a debug build its robustness
cases, which break precondition 2 on purpose, trip the new sweep assert. Speed at the leaf stage and for
`propagate()` is the same as `700ebd682` (median ratio 1.01, measured before the root changes, which do
not touch the node fills).

### Sweep bound under preconditions 1 and 2 (closes "Sweep counts at the coarser levels")

Take an inside open slot v and the nearest wall to it: in a leaf the nearest active voxel, in a lower or
upper node the nearest child slot, at the root the nearest position with an upper child. One exists by
precondition 1. On a straight (Manhattan-shortest) walk from v to that wall, every slot before the wall
is open (a wall there would be nearer) and inside (open slots connected to v share its label). So the
walk ends on a seed:

- in a leaf, the wall itself: an active voxel next to an inactive inside voxel has sign -1 by 2(b);
- in a lower or upper node, or the root array, the slot just before the wall: it is inside and touches
  the child, so by the inside-faces lemma the child's face toward it is all inside and seeds it.

No wall forces a detour, so the fill reaches v after its Manhattan distance to that seed: at most
3 * (Dim - 1) = 21 sweeps turn voxels on in a leaf and one fewer in a lower or upper node (44, 92),
plus one sweep that changes nothing: 22, 45, 93. For the root array the same count gives
(W - 1) + (H - 1) + (D - 1). The open point above expected detours around children; with 2(b) they
cannot happen. Measured on the regression test inputs: leaves with a single active voxel in a corner
take exactly 22 sweeps, and a lower node with a single leaf child exactly 45; the robustness cases that
break 2(b) take up to 27 (leaf) and 46 (lower). To be confirmed.

### Not done from the proposal, and why

- **Packing several nodes per warp** (4 leaves or 2 lower nodes, shuffles with `width = Dim`, so no lane
  sits idle) was implemented and measured, and was about 8% slower in the leaf stage (median over the 29
  cases). One node per warp is simpler and as fast as `700ebd682`, so it stays. Lower and upper fills take
  about 0.01 ms either way. A likely cause, not verified: the packed seed gather reads 4 runs of 8
  voxels per ballot instead of 32 consecutive voxels. The packed version's higher register count was
  ruled out as the cause.
- **A shared slice builder with ConnectedComponents.** Not done: there the leaf mask fill runs one block
  of 512 threads per leaf and writes each warp's ballot into a `Mask<3>` in shared memory, while here
  one warp holds a node as slices in registers. The common part is a single ballot, so the builder is a
  leaf-only function in `PropagateSign.cuh`, its only user.
- **Votes as warp-layer functions:** kept inline (see above).
- **A separate small PR for `Mask2D`:** not split out; it is part of 2361 for now.
- **CPU unit test for `Mask2D` and the 2D `MaskShift`, and a GPU test of the 3D `MaskShift`:** only
  outside the repo so far; to be added with the PropagateSign unit test.
- **DilateGrid's `Mask<4>` `MaskShift` through the same pieces:** not done.

### Still open

- Review findings above: "Not compiled anywhere" and "Packaging" (install list, unit test, example);
  todos for the mask naming and `signs` -> `signSidecar`.
- Formal races: the seeds that `seedAcrossFace` writes into the root-tile array are plain byte stores
  from many warps (all store 1), unlike the atomic bit sets in lower and upper masks.
- Inputs that break the preconditions, e.g. root value tiles in OpenVDB level sets converted to NanoVDB
  (`signedFloodFill` and `pruneLevelSet` can add inside tiles at the root for large objects).
- The future unit test should include, as ConnectedComponents' does, a run on a non-blocking stream, a
  check that the outputs stay valid after the operator is destroyed, and an injected resource that sees
  every allocation.

## Proposal 2026-10-29: sparse root propagation (replaces the dense root-tile array)

Agreed in review, for implementation. It replaces `mRootMask`, `mRootTileMin`, `mRootTileDims`,
`needsRootPropagation`, `RootWallFunctor` and `RootPropagationFunctor` with per-entry face flags and a
sparse expansion over root keys. The expansion code is not pretty (sorts, sentinels, set operations),
but it is contained in one function, the new stage 4; stages 1-3 change by one store.

### What is wrong with the dense root-tile array

`initializeMasks()` takes the index bbox of the active voxels (the root's `mBBox`), quantizes it to
root-tile coordinates (`>> NanoUpper::TOTAL`, i.e. 4096-voxel tiles), and allocates and zeroes one byte
per root-tile position in that box (`mRootMask`). Stages 1-3 write a 1 into it for every absent position
they seed; stage 4 marks the positions that hold an upper node as walls, floods the rest with Jacobi
sweeps in a single block, copies the box to the host and collects the 1s into `mRootInteriorTileOrigins`.
This is correct under preconditions 1-4. Not padding the box is also correct: the flood only spreads
interior, and by precondition 3 nothing outside the active bbox is interior. The problems are elsewhere:

1. **`needsRootPropagation` is decided before anything is known.** It is computed before stage 1, as
   `mRootMask.size() > mUpperInteriorMasks.size()`: the box has more positions than there are upper nodes.
   Every upper node holds an active voxel (precondition 1), so the upper nodes occupy `upperCount`
   distinct positions of the box, and the test says "the box has at least one absent position". That is
   a capacity fact, not a determination: whether interior root tiles have to be *introduced* is an output
   of the hierarchy, known only once stage 3 has seeded (or not seeded) the root. The test is true for
   almost every curved band. Example: a band that touches all eight 4096^3 octants around the origin
   except [-4096, -1]^3, which is fully exterior: the box is 2 x 2 x 2 = 8 positions, `upperCount` is 7,
   so the test is true, but no face ever seeds the missing octant. Stage 4 then runs for nothing (walls
   and scratch allocations, a wall kernel, a flood that stops after one sweep, a device-to-host copy and
   a sync, a host scan). It only skips stage 4 when the box has no hole at all (a band inside one root
   tile, or one that fills all eight octants), and in exactly those cases a test made after stage 3 would
   skip it too, so the early test buys nothing.
2. **The array is sized by the extent of the band, not by the interior, and allocated up front.**
   Scattered inputs pay for the space between their pieces. Two tiny closed objects (valid input, which
   seed nothing at the root) at (-1M, -1M, -1M) and (+1M, +1M, +1M): 1M = 2^20 voxels = 256 tiles, so the
   box is 512^3 = 134M positions, 128 MB allocated and zeroed (384 MB with stage 4's `walls` and `next`,
   which the early test does not avoid). At +/-2M it is 1024^3 (1 GB per array); at +/-4M, 2048^3 = 8.6G
   positions exceeds `MaxRootPositionCount` (`INT_MAX - 1`) and `PropagateSign` throws on valid input.
3. **The flood scans the whole box every sweep**, exterior and walls included, in one block.
4. **It trusts the root's `mBBox`.** A grid whose bbox was not computed (e.g. built with stats and bbox
   disabled) or is stale gives a wrong box: an empty box silently skips the root stage, a small one drops
   interior positions and walls.
5. **Formal race:** the seeds are plain byte stores from many warps (all store 1); see "Still open".

At realistic sizes the dense map is fine (+/-256K voxels is 128^3 = 2M positions, 2 MB per array), and
for a closed band the box is tiny (see the cost argument below). The proposal is about not paying for
extent at all, and about deciding the root stage from what the hierarchy actually delivered.

### Facts the proposal relies on

All under preconditions 1-4.

- **F1. Seeds land only on absent positions, and a seeded position is interior.** `seedAcrossFace` writes
  a root seed only when `probeTerminalSlot` stops at the root, i.e. the root table has no entry there.
  The seeding node's face toward it is inside and has no active voxel across it, so by the reference
  labeling (face-adjacent inactive voxels share a label) the absent tile is interior.
- **F2. Every root seed crosses a face of a root-table entry.** A seeding node (leaf, lower or upper) lies
  in some upper node U, and the neighbor across its face lies outside U (it is in an absent tile). So the
  node's face lies on U's face, and the seeded tile is U's face neighbor across that same face. Hence all
  root seeds are recorded exactly by 6 bits per root-table entry: "the absent tile across my face f is
  interior".
- **F3. Face-adjacent absent tiles share a label.** Neither contains an active voxel, so the shared face
  has inactive voxels on both sides. In particular, every absent neighbor of an interior absent tile is
  interior, unconditionally: no signs or masks need to be consulted at the root level.
- **F4. Seeding is complete at the root** ("Hierarchy completeness", *Root*, above): every connected set of
  interior absent tiles borders an upper node whose face toward it is all inside and seeds it. So interior
  root tiles exist if and only if stages 1-3 set at least one flag.
- **F5. The interior is enclosed, within the hull of the root table.** By precondition 3 every interior
  tile lies in the root-tile box of the active voxels. Under preconditions 1 and 4 every root entry is an
  upper node with an active voxel and every active voxel lies in an entry, so that box is exactly the
  bounding box of the root-table entries' tile coordinates (the "hull"). It can be computed from the table
  itself, with no reliance on `mBBox`.

### Data and stages 1-3

```cpp
// One word per root-table entry; bit f (Face: minusX, plusX, minusY, plusY, minusZ, plusZ) is set if the
// absent root tile across face f of this entry is interior.
BufferT<uint32_t> mRootFaceSeeds; // size = root table size (= upperCount under precondition 4), zeroed
```

- Allocated and zeroed in `initializeMasks()`, in place of `mRootMask`. Size: the root table, typically
  tens to a few thousand entries, independent of the extent.
- In the root branch of `seedAcrossFace` (today: `rootMask[rootTileIndex(p, tileDims)] = 1`), set
  `atomicOr(&rootFaceSeeds[entry], 1u << face)` instead. `face` is the face the node is crossing (F2:
  the same face of the enclosing entry). Different nodes of one entry set different bits of the same
  word, so the store must be atomic (CUDA has no byte `atomicOr`, hence a 32-bit word per entry). This
  also closes the formal-race item.
- `entry` is the index of the root-table entry enclosing the seeding node:
  `root.probeTile(origin) - root.tile(0)` with the seeding node's origin. NanoVDB's root `probe` is a
  linear scan of the table, as in `probeTerminalSlot` today; it runs only for root seeds. (Using the
  upper node's id instead is possible if upper nodes are laid out in table order; that holds for
  `TopologyBuilder` output as far as we know but is not guaranteed by NanoVDB, so it would need checking.)
- `mRootTileMin`, `mRootTileDims`, `rootTilePosition`, `rootTileIndex`, `MaxRootPositionCount` and the
  bounds check in `seedAcrossFace` go away.

### Stage 4: decide, then expand

**Decide.** After stage 3, copy `mRootFaceSeeds` to the host (a few KB at most) and test whether any bit
is set. If none is, there is no interior root tile (F4): stage 4 ends, `mRootInteriorTileOrigins` stays
empty, and no further scratch is allocated. This is the overwhelmingly common case; its only costs are
the flag array and this copy, whose sync takes the place of `propagate()`'s final
`cudaStreamSynchronize` (which then finds nothing left to wait for). The readback *is* the test; no
separate device flag or reduction is needed.

**Expand.** Otherwise, grow the set of root tiles in rounds, all on the GPU:

- *Round 1.* Every existing entry emits its own key plus, for each set flag, the key of its neighbor
  across that face; unused slots get a sentinel (e.g. `UINT64_MAX`, which sorts last). So 7 slots per
  entry (5 in the 2D example below). Compact the sentinels away, sort, remove duplicates (two entries can
  flag the same tile). The result is the existing entries plus the seeds, sorted and unique; this is the
  new candidate table. By F1, every key that is not an original entry is a new interior tile.
- *Flags are consumed.* After an entry has emitted, its flags are cleared to 0. A tile that enters the
  table in round k enters with **all six flags set** (`0x3F`): by F3 every absent neighbor of it is
  interior, and a neighbor that already exists is merged away by the deduplication. Nothing is tested
  per face (a new tile has no nodes and no voxels whose faces could be tested).
- *Round k + 1* is the same operation on the new table: every entry emits itself plus its flagged
  neighbors (only the tiles added in round k have flags), then compact, sort, deduplicate.
- *Stop* when a round adds no tile, i.e. no entry has a flag set.
- *Output:* the final table minus the original entries, i.e. all tiles added in some round. These are
  exactly the interior absent tiles: sound by F1 and F3 (every added tile is interior), complete by F4
  and F3 (every interior component is seeded, and the rounds spread through it).

**Refinement (same result, less work).** Do not re-emit and re-sort the whole table each round.
- Round 1: the original entries need not emit themselves. Emit only the flagged neighbors (6 slots per
  entry, mostly sentinels), compact, sort, unique: these are the seeds, absent by construction (F1), so
  no comparison with the table is needed. They are the first frontier; merge them into the table.
- Round k + 1: every frontier tile emits all 6 neighbors (no sentinels at all); sort and unique those;
  `set_difference` against the table (sorted) gives the new tiles, i.e. the next frontier; `merge` them
  into the table.
- Per round this costs about 6 x |frontier| keys plus a linear merge, instead of re-sorting the table.

**Keys.** Use one key for every list, with an inverse, e.g. the biased encoding of
`topology::detail::tileSortKey` (`TopologyBuilder.cuh`: tile coordinates biased by 2^19 per axis, x high,
y middle, z low, so tiles sort by coordinate with negatives first, as `TopologyBuilder` lays out the
table), plus a decoder. Do not use the key stored in the tile (`RootData::CoordToKey`), whose fields
are unbiased two's complement, and do not assume the table is already sorted in any order: build the
original key list from the tiles' origins and sort it (table-size, cheap). Neighbors are computed on
tile coordinates (decode, +/-1 on one axis, check the hull, encode), never by key arithmetic.

**Guard.** Under the preconditions the rounds terminate on their own (F5: the interior is enclosed by
table entries). If the input breaks precondition 2 or 3, wrongly seeded tiles can belong to the
unbounded exterior and the rounds would never stop. So every added tile is checked against the hull of
the original table (F5): a tile outside it means a broken precondition: assert in debug, and stop (or
throw) in release. This keeps termination guaranteed; it does not detect every broken input (a wrong
seed in an enclosed exterior pocket fills that pocket and stops inside the hull). Checking the hull
before encoding also keeps every key within the representable range.

**Syncs.** The host needs each round's frontier size to size the next round, so one small readback per
round (e.g. the count that `cub::DeviceSelect` writes). Building blocks: `cub::DeviceSelect::If`
(sentinels), `cub::DeviceRadixSort::SortKeys`, `cub::DeviceSelect::Unique`, and a merge and a set
difference on sorted keys (`thrust::merge` / `thrust::set_difference`, or their cub counterparts).

**Output format.** `mRootInteriorTileOrigins` is the decoded list of added tiles. Since the rounds produce
them sorted, return them sorted (or return the sorted keys as well): the sanitized level-set convention
inserts interior root tiles into the root table, and a sorted list is exactly what that insertion
needs.

### Example in 2D

Root-tile coordinates (x right, y up); 2D faces are -x, +x, -y, +y (4 flags, 5 slots per entry in round
1 of the plain scheme). `[ ]` is a root-table entry (an upper node holding part of the band); an arrow
inside marks a set seed flag and points across that face. Letters are absent interior tiles; `.` is
absent exterior.

```
            x=0   x=1   x=2   x=3   x=4   x=5
    y=4     [ ]   [v]   [v]   [v]   [ ]    .
    y=3     [>]    a     b     c    [<]    .
    y=2     [>]    d     e     f    [<]   [ ]
    y=1     [>]    g     h     i    [<]   [ ]
    y=0     [ ]   [^]   [^]   [^]   [ ]   [ ]
```

- The table has 19 entries. The 12 that face the hole have one flag each; the four ring corners touch
  the hole only diagonally and have none. (4,3) and (4,4) face the exterior tiles (5,3) and (5,4), and
  (5,2) faces (5,3): no flags, since those faces are outside.
- Today's code: the box is the hull, 6 x 5 = 30 positions, against 19 upper nodes, so
  `needsRootPropagation` is true. It would still be true with the hole filled in by band tiles
  (30 > 28), because of the exterior notch (5,3), (5,4): the false positive above, in 2D.
- *Round 1* emits 12 neighbor keys: a, b, c, d, f, g, h, i, where a, c, g and i are each emitted twice
  (a by (0,3) and (1,4), and so on). After sort and unique, 8 new tiles; the table has 27 entries. e is
  not seeded: all four of its neighbors are absent, so no band tile touches it.
- *Round 2*: the 8 new tiles emit their 4 neighbors each (32 keys). After sort and unique, everything but
  e is already in the table (band tiles or round-1 tiles): one new tile, e; 28 entries.
- *Round 3*: e emits b, d, f and h, all in the table. Nothing is added: stop.
- Output: a-i (9 tiles). The exterior notch was never touched. Every added tile is inside the hull
  x in [0, 5], y in [0, 4].

The flattened candidate table, as drawn on the whiteboard (flags after each round):

| Entries | After round 1 | After round 2 |
| :-- | :-- | :-- |
| the 12 band tiles that had a flag | consumed (0) | 0 |
| the 7 band tiles that never had one (ring corners, (5,0)-(5,2)) | 0 | 0 |
| a, b, c, d, f, g, h, i (added in round 1) | all faces | consumed (0) |
| e (added in round 2) | (not yet in the table) | all faces |

The guard in this picture: if the ring had a gap at (4,3) and the hole were still seeded, round 2 would
add (4,3), round 3 (5,3), and round 4 (5,4) and (6,3); (6,3) is outside the hull, so the expansion stops
and asserts.

### Cost

- **Common case (no interior root tile):** one allocation of table-size words (bytes to kilobytes), one
  copy of it to the host, no other scratch, no root kernels. Today: an allocation the size of the band's
  box, plus a stage 4 that the early test fails to skip most of the time.
- **With interior root tiles:** the number of rounds is the depth of the interior in tiles (about its
  radius), one small sync each; the total work is about the number of interior tiles, times 6, plus the
  merges.
- **The linear number of rounds is not a concern.** A deep cavity needs a closed band around it. For a
  cavity D tiles across, the band covers about D^2 tiles, each holding about 4096^2 x (band width)
  active voxels; at D = 8 that is already 10^9-10^10 active voxels, around what fits on a GPU, and the
  band cost O(D^2) work (D = diameter) with a far larger constant to build. So there are at most a
  handful of rounds (D / 2 <= 4-8) and well under a millisecond of syncs, and the root work (O(D^3) tile
  operations, a few hundred keys) is negligible next to the band's O(D^2 x 4096^2) voxels (ratio about
  D / 1.7 x 10^7).
- Today's Jacobi flood also needs a number of sweeps proportional to the depth, but each sweep scans the
  whole box; the rounds only touch the frontier.

### Alternatives considered

- **Keep the dense array, decide after stage 3.** Fixes the false positive of `needsRootPropagation`, not
  the extent problem (allocation, `INT_MAX` throw).
- **Dense array sized by the seeds.** Record seeds as face bits per upper node (F2), and after stage 3
  allocate the dense array over the bbox of the seeds only. Correct because an interior component's bbox
  equals the bbox of its seeds: a position of the component that is extreme along +x has a +x neighbor
  that is neither in the component (it is extreme) nor exterior absent (F3), so it is an upper node, whose
  face toward it is all inside (lemma, 2(b)), so the position is seeded; likewise for the other five
  directions. Sized by the interior, but still a dense flood over a box.
- **Host BFS.** Copy the flags and the table keys to the host and expand with a hash set. Simpler code,
  fine at these sizes; the GPU version was preferred so that only a size per round crosses to the host.
- **Count, scan, then emit** (as in MeshToUDF), to allocate exactly the emitted keys instead of 7 (or 6)
  slots per entry. Not worth it: the expansion per entry is a fixed bound of 6 over a table-size array
  (56 bytes per entry with 8-byte keys), and counting adds a scan and a sync per round to save a few
  kilobytes. MeshToUDF needs it because its per-triangle output is large and unbounded. With the
  refinement, only round 1 has sentinels at all.

### What this changes elsewhere in these notes

- Precondition 3's "restricting the root flood to the root tiles that cover the active bbox" becomes the
  guard (F5), and the hull is computed from the table rather than read from `mBBox`.
- The root sweep-bound assert ((W - 1) + (H - 1) + (D - 1)) goes away with the Jacobi root flood; the
  guard bounds the rounds instead.
- The "Formal races" item under "Still open" is closed by the atomic flag store.
- Root value tiles (precondition 4(b)) are unchanged: such a tile is an original entry, so it is never
  added (and `probeTerminalSlot` still asserts on it).

## Update 2026-10-30: SDF input, invert masks, sparse root implemented, precondition 4 checked

This supersedes the earlier sections where they disagree: `deviceSigns` and its -1/+1 tags, "interior"
masks, `mRootMask` and the dense root flood, `TerminalSlot`/`probeTerminalSlot`, and `signing::detail`.

### Interface (`fab481195`, `7822d9523`)

- **Input** is the SDF sidecar itself, `const ValueT* deviceSdf` with `ValueT` floating point (float or
  double, `static_assert`), not an int8 sign or tag array. Inside means a negative value; zero is outside.
- **`BuildT` must be `ValueOnIndex`** (`static_assert` on `BuildTraits<BuildT>::is_onindex`). An inactive
  voxel has value index 0, so `leafSeedSlice` reads one `getValue(n)` per voxel and tests `index != 0`
  before reading `sdf[index]`.
- **Outputs are invert masks:** a bit is on for an inactive element (voxel, childless slot, absent root
  tile) whose value is `-sdf[0]`. Leaf masks hold only inactive voxels (`interior & open`), while face
  seeding still uses the whole interior. Complementing a level set means negating the sidecar, slot 0
  included, and leaves the masks unchanged. Invert is not the same as bounded: in a hollow ball the
  cavity is bounded but has the background's sign, so its bits are off.
- **`propagate(leafMasks, lowerMasks, upperMasks)`** fills caller-allocated masks, which need not be
  initialized, and returns `{root-tile origins, hasInvertRootTiles}`, the origins sorted by (x, y, z).
- The detail namespace is `floodfill::detail`.

### Sparse root propagation as implemented (`7822d9523`)

It follows the 2026-10-29 proposal in the form of the refinement: round 1 emits only the flagged
neighbors of the table entries, and each later round emits all six neighbors of the tiles the previous
round added (the plain scheme's "flags are consumed, new tiles enter with all faces set"). Deviations:

- **No `set_difference`:** each sorted, unique candidate is binary-searched in the sorted known tiles,
  and `cub::DeviceSelect::Flagged` keeps the new ones; merges use `cub::DeviceMerge::MergeKeys`
  (thrust's scratch would bypass `ResourceT`). Round 1 filters against the known tiles too, which F1
  makes unnecessary, so that every round runs the same loop body.
- **Two sorted lists,** `known` and `added`, each merged every round; the output is `added`.
- **The hull** is computed on the host from a copy of the entries' keys (one more sync, on the expansion
  path only).
- **The guard is weaker than proposed:** a neighbor outside the hull is dropped and sets a flag that only
  a `NANOVDB_ASSERT` checks after the loop, so a release build returns a wrong (though bounded) result
  without notice. See Still open.
- **`seedAcrossFace` walks root -> upper -> lower itself** and seeds where the walk ends; `TerminalSlot`
  and `probeTerminalSlot` are gone. A root seed sets bit `face` of the seeding node's own root-table
  entry, found with a second `probeTile` (on the node's own origin, so it always succeeds; asserted).

### Review pass 2026-10-30

- **Precondition 4 is checked on the host,** in `initializeMasks()`: it throws if any `mTileCount` is
  nonzero (active tiles) or if the root table has more entries than there are upper nodes (root value
  tiles, active or not; `mTileCount` misses the inactive ones). `seedAcrossFace` then only asserts
  `tile->isChild()`. Without the host check, release builds would be unsafe there: `getChild` on a
  value tile returns the root itself, and the walk would write outside `upperMasks`.
- **`FaceSet` dropped:** a set of faces is a plain `uint32_t` (bit f for face f). `atomicOr` and the
  warp shuffles fix the width at 32 bits anyway.
- **Comments:** `leafSeedSlice` and `BitFloodFunctor` are documented as warp-collective, `hasSlice`
  says which lanes hold slices (8, 16, 32), the flood update cites precondition 2, the sweep bound is
  stated as the node's Manhattan diameter, and the same-level returns in `seedAcrossFace` say why
  nothing is seeded (a node of our own level fills itself).

### Still open

- **Packaging:** no target includes the header, so CI never compiles it; it is not in
  `NANOVDB_INCLUDE_TOOLS_CUDA_FILES`; there is no test or example. Planned test: a gtest in
  `nanovdb_test_openvdb` with OpenVDB level sets (sphere, torus, CSG hollow shell, two distant spheres,
  and a band enclosing a cavity several root tiles wide so that stage 4 expands), stripped to the
  narrow band and compared with OpenVDB's inactive signs; plus a run on a non-blocking stream, outputs
  that outlive the operator, and an injected resource that sees every allocation.
- **`pendingchanges/propagatesign.txt`** still describes the sign-array interface.
- **Root guard in release:** throw (or stop) when a neighbor leaves the hull; the flag is already read
  back every round.
- **Optional: check `seedAcrossFace` with a host simulator on synthetic trees.** Build trees on the host
  whose nodes sit at every kind of boundary: within one lower node, across lower, upper and root-tile
  boundaries, at negative coordinates, and next to absent root tiles. For every (node, face), compute
  the expected destination by brute force from the node sets: a root face bit of the node's entry if no
  upper node contains the block across, else an upper slot if that upper node has no lower child there,
  else a lower slot if that lower node has no leaf there, else nothing (a node of the same level). Run a
  kernel that calls `seedAcrossFace` for every (node, face) and compare masks and face bits exactly.
  This tests the walk and its indexing (`getFirstNode` offsets, `CoordToOffset`, the entry index)
  independently of the fills.
- Preconditions 1-3 remain unchecked. Root value tiles from OpenVDB level sets are now rejected, not
  handled. Background-relative seeding (precondition 3 requires `sdf[0] > 0`) is deferred.
- **Closed from the earlier "Still open" lists:** formal races (root seeds are `atomicOr`), the
  `signs` -> `signSidecar` todo (the input is the SDF sidecar), and the mask naming (invert masks).
