# vdb_bench

Times the core library tools that read the neighbors of a voxel, so two builds
of OpenVDB can be compared on your hardware. The same source builds against
any build of the library. Every row calls the public API only, and prints a
hash of its result, so a build that is faster but gives a different answer is
reported.

## Build

Build OpenVDB twice, once for each library to compare (for example master and a
branch), with the same compiler, flags and thread library:

    cmake -DOPENVDB_BUILD_VDB_BENCH=ON -DCMAKE_BUILD_TYPE=Release ...

`vdb_bench` is built next to the other command line tools. Use a Release build.
On an Intel machine affected by the jump conditional code erratum, results of
tight loops can move by tens of percent between builds. If so, build both with
`-Wa,-mbranches-within-32B-boundaries`.

## Run

Run it once for each build. The first run saves its rows, the second compares
with them:

    vdb_bench all --dragon-vdb ~/vdbs/dragon.vdb --save before.csv
    vdb_bench all --dragon-vdb ~/vdbs/dragon.vdb --save after.csv --baseline before.csv

`all` runs every group twice at 8 threads and writes a summary to standard
output at the end (`--summary FILE` writes it to a file). A full run takes
about 30 minutes. Without `--baseline` the summary lists the time of each tool.
With it, the summary lists the speedup of each tool over the baseline. The
exit code is 1 if a result differs from the baseline.

`--dragon-vdb` is the path of `dragon.vdb`, one of the sample models on
https://www.openvdb.org/download/. Without it the meshing rows that use the
dragon are skipped. Every other grid is made by the benchmark.

Run one group, or a few rows of it, while you work:

    vdb_bench levelset --threads 8 --only normalize

| Group | Tools |
|---|---|
| levelset | LevelSetTracker, LevelSetAdvection, LevelSetMorphing, LevelSetFilter |
| analysis | LevelSetMeasure, FastSweeping, Diagnose, opStatistics |
| operators | gradient, laplacian, curl, divergence, cpt and more |
| filter | tools::Filter |
| poisson | the Laplacian matrix and the solver |
| sampling | resampleToMatch, MultiResGrid, VolumeAdvection, points sampling |
| meshing | volumeToMesh, meshToVolume, level set utilities |
| accessor | one read pattern per tool, ValueAccessor and NeighborAccessor |
| all | every group above |

## Reading the results

- The speedup is the time of the baseline divided by the time of this run.
  More than 1.0x is faster. A tool is the geometric mean of its rows.
- A speedup within the noise of the machine is written as 1.0x. The noise of a
  group is how much its rows move between the two passes.
- Rows under about 20 ms move 10% to 50% between runs on a shared machine.
  Use an idle machine, and look at the Notes at the end of the summary for rows
  to run again.
- A row whose result differs from the baseline is left out of the speedups and
  listed. Rows with active tiles in the statistics tools are expected to differ
  and are left out without a note.
- `resampleToMatch` rows use a coarse hash, since tasks that overlap write in
  no fixed order.

## The accessor group

The `accessor` group answers questions about NeighborAccessor itself. Each tool
is a read pattern on the active voxels of a sphere, one row per implementation:

| Tool | What it shows |
|---|---|
| read center | cost of a read in the bound region, with a leaf iterator and with `getLeafValueUnsafe` |
| read 6 faces | the stencil of a Laplacian, with an iterator, with an offset, with offsets known at run time, and with a new accessor for every leaf |
| read 8 corners | the farthest of the 26 neighbor regions |
| read far, one region | reads two leaf widths away, always in the same region |
| read far, alternating | the same, but alternating between two regions, so the cache of far reads misses |

The `ValueAccessor` row of each tool is the cost of the same reads without a
NeighborAccessor. The summary prints the time of each row relative to the
ValueAccessor row, which does not depend on the machine. The group is useful
across builds too: build the library with a change to NeighborAccessor and
compare the rows of the two builds with `--baseline`.
