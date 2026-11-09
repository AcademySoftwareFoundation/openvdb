# MeshToSDF optimization candidates

Changes worth applying later, for speed or for peak device memory. Each item says what changes, what it was
measured to gain and cost, and what it departs from. Peak memory is given in bytes per active voxel of the
rasterized grid, as reported by the stream-ordered memory pool, without the mesh itself.

| # | item | gain | cost | status |
|---|---|---|---|---|
| S1 | [ConnectedComponents](#s1-connectedcomponents) | whole call 1.9x to 3.5x, same labels | changes `ConnectedComponents.cuh`, a separate PR | parked branch |
| M1 | [Barrier votes](#m1-barrier-votes) | hairball peak 13.25 -> 10.04 B/voxel, max active voxels 2.44B -> 3.26B | +85 ms at res 3408, about twice the barrier code | scratch only, bit-identical |

## Speed

### S1. ConnectedComponents

`MeshToSDF` runs `ConnectedComponents` twice, on the grid (surfaces) and on the barrier-free grid. On hairball at
res 3904 the two calls take 488 ms of a 1736 ms stage sum (RTX 5090, build that synchronizes after each stage).
Three changes, one per stage of `ConnectedComponents`, keep the labels identical:

- **Leaf stage: union-find over z-runs instead of voxels.** A z-run is a maximal stretch of active voxels along z in
  one row (x, y) of a leaf; two runs are adjacent when they lie in neighbouring rows and their bits overlap. The
  schedule stays `LeafUnionFind`'s (parent-connect, then parent-root-connect, O(log n) rounds), run by one warp per
  leaf instead of one block. Runs are numbered by their first voxel offset, so a component's root still holds its
  smallest voxel offset and the global numbering does not change.
- **Cross-leaf stage: unite where a touching pair is found.** The current code counts the touching component pairs
  of each leaf, scans the counts, reads the total back on the host, writes an edge list and unites it in a separate
  pass. Calling the same lock-free `unite` where the pair is found needs none of these; the union is
  order-independent (a class's smallest slot stays its root), so the representatives are the same.
- **Label stage: one warp per leaf scatters each component's label.** It reads each component's label once and
  writes it at every voxel of the component, instead of one thread per voxel searching the leaf's components. Only
  slot 0 of the label sidecar is initialized, since every other slot is an active voxel.

Measured on an RTX PRO 3000 Blackwell laptop GPU at its 60 W power limit (2026-11-03), whole call, median of 3 runs:
1.92x to 3.46x on 8 inputs. On hairball (voxel 0.004, half-width 2, 561M voxels) it went from 272 to 92 ms: leaf
stage 155.6 -> 36.6 ms, cross-leaf stage 64.6 -> 38.3 ms, labels 41.5 -> 10.0 ms. The implementation is
`ConnectedComponentsOpt.cuh` on branch `cc-optimization` of `sifakis/openvdb` (commit 98986ebce), with its benchmark
and notes in `nanovdb/nanovdb/examples/ex_connected_components_opt/`.

## Peak memory

### M1. Barrier votes

**Where the peak is now.** For a mesh with one surface the peak is the barrier-free `ConnectedComponents` call
(stage 3c). At that point the device holds:

| buffer | B/voxel (hairball) |
|---|---:|
| UDF, which becomes the output distances | 4.00 |
| nearest-triangle index from `MeshToGrid` | 4.00 |
| barrier-free component labels (output of `ConnectedComponents`) | 3.25 |
| `ConnectedComponents` scratch | 0.73 |
| grid | 0.64 |
| barrier-free grid (input of `ConnectedComponents`) | 0.63 |
| total | 13.25 |

Everything except the triangle index is an output or an input/output of `ConnectedComponents`. The triangle index
is alive only because the barrier heuristic (`SignBarrierFunctor`) reads it, and the heuristic runs after the
barrier-free components, since it needs the signs of the non-barrier voxels.

**The change.** The heuristic decides a barrier voxel q: q is exterior if some voxel n among its 27 neighbours is
an anchor (exterior, at least `AnchorMinDistance` voxels from the signed surface, with a nearest triangle) and the
anchor's geometry places q on its own side (`isExteriorBy`). Only "exterior" needs the signs; the distance test,
the nearest triangle and the side test need only the UDF and the triangle index. So the heuristic is split in two:

1. **Votes, right after `MeshToGrid`.** For each barrier voxel q of the rasterized grid, a 27-bit mask: bit
   `(dx+1)*9 + (dy+1)*3 + (dz+1)` is on when the neighbour at offset (dx, dy, dz) passes the sign-free anchor
   conditions and its geometry places q outside. The triangle index is then released, before the surface
   components.
2. **Apply, after `signNonBarrierComponents`.** q is exterior if any neighbour whose bit is on is a voxel of the
   current surface's grid and positive; otherwise its distance is negated. An anchor is never a barrier voxel
   (`AnchorMinDistance` 0.866 > `BarrierHalfWidth` 0.501), so this kernel reads only signs it does not write, and
   negates in place without the uint8 mark array the current code allocates.

The votes are stored only for barrier voxels (about 19% of the voxels on hairball), one `uint32_t` each, packed
per leaf: a count of the barrier voxels per leaf, an inclusive scan into per-leaf offsets, and q's vote at
`offset[leaf] + rank`, where rank is the number of barrier voxels before q in its leaf. That is 0.79 B/voxel
instead of the 4 B/voxel index. With several surfaces the votes are computed once on the whole grid; the apply
step ignores neighbours outside the current surface's grid, as signing on that grid alone does, and the
per-surface copies of the triangle index are no longer needed.

**Measured.** NVIDIA GeForce RTX 5090 (32 GB), hairball (2.88M triangles), band 3, default memory pool,
2026-11-09. Peaks and stage times come from a build that synchronizes after each stage (median of 3 runs).
`MeshToSDF` times are medians of 5 runs of the normal build, with and without votes in the same run.

| | without votes (current) | with votes |
|---|---:|---:|
| peak at res 3408 | 13.25 B/voxel (stage 3c) | 10.04 B/voxel (stage 3c) |
| largest resolution that fits | 3904 | 4480 |
| active voxels there | 2.44B | 3.26B |
| `MeshToSDF` at res 3408 | 1381 ms | 1467 ms |
| barrier stages at res 3408 | 154 ms | 170 ms votes + 91 ms apply |

The output was bit-identical to the current code on 73 meshes at resolutions 512 and 1024 (UDF, triangle index,
leaves, and `MeshToSDF` output for both barrier signings and iso 0 and 1). On a laptop RTX PRO 3000 at res 2048,
the meshes with several surfaces peaked at 2.29 GB (oilrig) and 1.20 GB (bmw) with votes; the current code was not
measured on them.

**Costs.**

- About twice the barrier code (116 -> 229 lines): a count kernel, a scan, a host readback of the vote count, the
  votes kernel, the apply kernel and a block-wide rank helper.
- The rule is split across two kernels far apart in the pipeline, so a reader has to put it back together.
- The apply kernel finds q's vote by counting the barrier voxels before q again. Count, votes and apply must apply
  the same barrier test to the same UDF magnitudes; only the signs change in between. Routing all three through one
  helper would make that hold by construction.
- Alternatives for finding q's vote, measured the same way in a separate run in which the recount version also took
  1467 ms: a per-leaf barrier mask and `Mask::countOn(n)`, as a `ValueOnIndex` leaf finds its values
  (10.34 B/voxel, 3.16B voxels, 1474 ms); or a grid of only the barrier voxels from `PruneGrid` with the votes as
  its sidecar (10.50 B/voxel, 3.11B voxels, 1502 ms). Both are easier to read and cost some of the gain.

**Departs from.** `MeshToSDF_PipelinePlan.md`, step 5: the heuristic would no longer consume the triangle index
and the signed non-barrier neighbours in one pass.

**Reference implementation.** Outside the repository, in the scratch tree
`learning/work/bench/meshtosdf_peakmem/src_s4r` (votes with recount), `src_s4m` (mask) and `src_s4g` (barrier
grid); the measurements are in `r5090/mem_s4`, `r5090/total_s4.txt`, `r5090/mem_nv` and `r5090/total_nv.txt`.
