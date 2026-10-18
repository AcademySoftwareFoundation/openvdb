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

4. **Active values are voxels only, and the root holds only child tiles.**
   - (a) No active tiles at any level: every active domain is a leaf voxel. *If violated:* an active tile
     carries a sign that the code never reads; at the lower/upper levels it is a childless slot and at the
     root a non-child tile, so the fills treat it as passable inactive space. A nonnegative active tile
     then fails to block the flood, breaking 2(a).
   - (b) Every root tile is a child tile (no root value tiles, active or inactive). *If violated:* see the
     root value-tile seeding gap under Open points.

   Both hold for any `TopologyBuilder`-based output (including `PruneGrid` and `MeshToGrid`), whose root
   has only child tiles and whose active values are leaf voxels.

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
