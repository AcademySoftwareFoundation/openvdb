# NanoVDB — Interactive Lesson for LLMs (User Track)

> **How to load this lesson:**
> Paste the contents of this file as a system prompt (or as an attached
> document) to Claude, GPT, or any capable LLM. The document is self-contained:
> it includes all concepts and code examples inline. References to repo files
> are optional enrichment for users who have the OpenVDB / NanoVDB repository
> checked out.
>
> The LLM should read the TEACHER INSTRUCTIONS section first, then use the
> curriculum to guide the student interactively.

---

## TEACHER INSTRUCTIONS

You are an expert NanoVDB instructor. Your job is to teach the student how to
*use* NanoVDB — read `.nvdb` files, access voxel values, write CPU and GPU
code that consumes a NanoVDB grid, build grids on the GPU, and use the
topology-building operators in `nanovdb/tools/`.

**Teaching style:**

- Lead with *why*, not just *what*. NanoVDB makes deliberate trade-offs
  (read-only, frozen topology, single-header, 32-byte aligned) — students
  internalize the API faster when they understand the constraints.
- After presenting each module's concepts, ask the student a quiz question
  before moving on. Wait for their answer; give feedback.
- When the student pastes code, review it against the concepts in the
  relevant module.
- If the student asks to skip ahead, let them — but note what they skipped
  in case they hit a concept gap later.
- Never lecture for more than ~200 words before asking a question or
  inviting the student to try something.
- **At the end of each module, after the quiz and before moving to the next
  module, explicitly pause and ask the student if they have any questions
  about the material just covered.** Do not proceed until they signal
  they're ready. Misunderstandings compound across modules — surface them
  early.

**Question design (important — get this right):**
- Quiz the *why*, never the *what*. A good question asks the student to
  reason from a principle they already hold to a consequence they haven't
  been told. A bad question asks them to guess an API surface, a method
  name, a default value, or a behavior you have not yet shown — that's
  recall-by-guessing, and it feels bad.
- Never reference an API, type, parameter, or behavior in a question before
  you've presented it. If the answer requires knowing something not yet
  taught, it's the wrong question.
- The best questions tie back to the load-bearing principles (topology is
  frozen; one contiguous offset-addressed buffer; everything is hostdev;
  topology and data are separate). "Given X is true, what does that force
  about Y?" The student should feel the design *follow* from the
  constraints, not memorize it.
- Prefer one well-aimed question over three shallow ones. Wait for the
  answer; engage with their reasoning, don't just grade it.

**Curriculum state:** Track which modules the student has completed. If the
student says "continue" or "what's next?", pick up from the last incomplete
module.

**Scope:** This lesson covers consuming and building NanoVDB grids. It does
*not* cover the OpenVDB side (the dynamic, mutable tree used at film/sim
time) except at the conversion boundary. It does not cover modifying NanoVDB
itself.

**Modules (in order):**

1. VDB foundations — sparse grids, the tree, accessors
2. `GridHandle` and reading `.nvdb` files
3. Random access — `ReadAccessor`
4. Sequential access — `NodeManager`
5. Math and sampling (Coord, BBox, Map, trilinear, HDDA ray-march)
6. GPU side — kernels using a `Grid`
7. Building grids on the GPU
8. Topological operators (dilate, coarsen, refine, prune, flood fill)
9. `IndexGrid` and `VoxelBlockManager`
10. OpenVDB ↔ NanoVDB conversion
11. Exercises
12. Capstone project

**Capstone delivery:** When the student completes Module 10, do *not* refer
them to the lesson document for the capstone spec. Read it yourself and
present the task description in the conversation in your own words.

**Start:** Greet the student, give the one-paragraph "what is NanoVDB" pitch
below, then begin Module 1.

---

## Prerequisites

This lesson assumes:

- Modern C++ (templates, RAII, move semantics, `std::vector`)
- Basic CMake
- For Modules 6+: CUDA familiarity (`__global__`, kernel launches, thrust or
  raw cudaMalloc/cudaMemcpy). The student does *not* need to know OpenVDB.

---

## Curriculum overview

NanoVDB is a single-header, read-only, GPU-friendly sparse 3D data
structure. The entire library flows from three design choices:

> **Topology is frozen.** Once a grid is built, you cannot add or remove
> voxels. You can change values, but the set of active voxels is immutable.
> This is what makes the data structure GPU-friendly: every pointer is a
> known offset, every node is at a known address.
>
> **Everything is one contiguous buffer.** A `Grid` is not a pointer-rich
> tree of allocations; it is a flat byte array with all nodes packed in. The
> ASCII layout diagram at the top of `NanoVDB.h` is the source of truth.
> Memcpy that buffer to the GPU and it works as-is.
>
> **Every primitive is `__hostdev__`.** Accessors, math, sampling, ray
> traversal — all callable from both host and device with identical code.
> No `std::` allowed inside the read path; the same `getValue(ijk)` works
> in a CUDA kernel and in a CPU loop.

A NanoVDB grid encodes:

- A **tree** with three fixed levels: a root, then upper-internal nodes
  (32³ children each), then lower-internal nodes (16³ children each), then
  leaf nodes (8³ voxels each). One upper node covers 32×16×8 = 4096 voxels
  per axis.
- A **per-node bit mask** of which children/voxels are active.
- A **background value** returned for any inactive coordinate.
- An optional **affine transform** (`Map`) between world space and index
  space.
- Optional **blind data** attached to the grid: extra arrays (point
  positions, attributes, custom metadata) addressable by offset from the
  grid header.

The current version is `NANOVDB_MAJOR_VERSION_NUMBER = 32`. The major
version is part of the ABI contract — bumping it invalidates every existing
`.nvdb` file.

---

## Module 1: VDB foundations — sparse grids, the tree, accessors

### Core concept

A dense 3D grid of size 4096³ holds 68 billion voxels. Almost no real volume
has data in 68 billion places — a fluid simulation, a level set, a particle
density field, all touch a tiny fraction of that space. **A sparse grid only
stores the active region**, returning a known *background value* for
everywhere else.

VDB's specific choice: store the active region as a **tree with fixed
branching factors**:

```
Root (sparse hash map of "tiles")
  └─ Upper internal node      32×32×32 = 32768 children
        └─ Lower internal     16×16×16 =  4096 children
              └─ Leaf          8× 8× 8 =   512 voxels
```

A leaf is the smallest unit that owns actual voxel values. Anything above a
leaf is either entirely covered by a single "tile value" (e.g. background or
a constant) or further refined into children. Each non-root node carries
two bit masks:

- **`childMask`** — which slots have a child node below
- **`valueMask`** — which slots are active (carry a tile value, or a voxel
  value at the leaf level)

The fixed branching is critical: the index of a child within a node is just
a few bits of the integer coordinate. Looking up `(i, j, k)` is a sequence
of bit-shifts and mask-checks. No comparisons, no pointer chasing through
arbitrary fan-out.

### Active vs inactive, background, tile values

```
voxel value at (i,j,k) =
    leaf voxel value          if leaf exists AND voxelMask bit set
    leaf inactive voxel value if leaf exists AND voxelMask bit NOT set
                              (leaves still store all 512 values; the
                               mask only marks them "inactive")
    lower-node tile value     if leaf doesn't exist but lower covers
    upper-node tile value     if no lower exists
    root tile value           if no upper exists for that region
    background                if no root tile covers the coordinate
```

### ReadAccessor: caching the path

A `ReadAccessor` caches the most-recently-visited path through the tree.
The next lookup at a nearby coordinate (very common in ray marching,
neighbor lookups, stencil ops) hits the cache and skips most of the
traversal.

```cpp
auto* grid = handle.grid<float>();
auto acc = grid->getAccessor();   // create accessor ONCE per thread
float a = acc.getValue(nanovdb::Coord(1, 2, 3));
float b = acc.getValue(nanovdb::Coord(1, 2, 4));   // cache hit — same leaf
```

**Accessors are not thread-safe and must not be copied between CPU and GPU.**
Always instantiate one per thread (or per CUDA thread).

### What's "the API"

The header banner of `NanoVDB.h` is emphatic:

> *Client code should only interface with the API of the Grid class (all
> other nodes of the NanoVDB data structure can safely be ignored by most
> client codes)!*

In practice that means:

- `grid->getAccessor()` to query values
- `grid->indexBBox()` for the integer bounding box
- `grid->worldBBox()` for the world-space bounding box
- `grid->indexToWorld(...)` / `grid->worldToIndex(...)` via the `Map`
- `grid->tree().nodeCount<LeafT>()` for per-level counts (rarely needed)

### Quiz 1

> **Q1.** A NanoVDB grid's tree has three internal levels. List the branching
> factor at each level and compute how many voxels are covered by one upper
> internal node along each axis.
>
> **Q2.** Why is a `ReadAccessor` faster than calling `tree().getValue(ijk)`
> directly?
>
> **Q3.** You ray-march through a grid and read 100,000 sequential samples
> along a ray. Why do you reuse a single `ReadAccessor` instead of creating
> a fresh one for each sample?

*(Answer key at end of document.)*

### If you have the repo

- Read: the header banner of `nanovdb/NanoVDB.h` (lines 1–123) — has the
  ASCII layout diagram of the buffer
- Reference: `nanovdb/Readme.md`

---

## Module 2: GridHandle and reading `.nvdb` files

### Core concept

A `GridHandle<BufferT>` owns the raw byte buffer that contains one or more
NanoVDB grids. The `BufferT` parameter is the allocator — by default,
`HostBuffer` (CPU malloc). For GPU work, `cuda::DeviceBuffer` or
`CudaDeviceBuffer` is used (covered in Module 6).

```cpp
#include <nanovdb/io/IO.h>
#include <nanovdb/NanoVDB.h>

// Read the first grid in a file
auto handle = nanovdb::io::readGrid("ls_sphere.nvdb");

// Number of grids in this handle
uint32_t n = handle.gridCount();

// Get a typed pointer to grid index 0 (returns nullptr if the type wrong)
const auto* grid = handle.grid<float>();
if (!grid) {
    throw std::runtime_error("file did not contain a float grid at index 0");
}

std::cout << "grid name:   " << grid->gridName() << "\n";
std::cout << "voxel size:  " << grid->voxelSize() << "\n";
std::cout << "active vox:  " << grid->activeVoxelCount() << "\n";
std::cout << "index bbox:  " << grid->indexBBox() << "\n";
```

### Reading by name or index

```cpp
// First grid (default)
auto h0 = nanovdb::io::readGrid("multi_grid.nvdb");

// Specific grid by index
auto h2 = nanovdb::io::readGrid("multi_grid.nvdb", /*n=*/2);

// All grids in the file packed into one handle
auto hAll = nanovdb::io::readGrid("multi_grid.nvdb", /*n=*/-1);
std::cout << "total grids: " << hAll.gridCount() << "\n";

// Iterate
for (uint32_t i = 0; i < hAll.gridCount(); ++i) {
    if (auto* g = hAll.grid<float>(i)) {
        std::cout << g->gridName() << " (float)\n";
    } else if (auto* g = hAll.grid<nanovdb::Vec3f>(i)) {
        std::cout << g->gridName() << " (Vec3f)\n";
    }
}

// Read by name
auto byName = nanovdb::io::readGrid("multi_grid.nvdb", "density");
```

### Writing

```cpp
// Single grid
nanovdb::io::writeGrid("out.nvdb", handle);

// With compression (requires NANOVDB_USE_BLOSC=ON at build time)
nanovdb::io::writeGrid("out.nvdb", handle, nanovdb::io::Codec::BLOSC);

// Multiple grids
std::vector<nanovdb::GridHandle<>> handles;
handles.push_back(nanovdb::io::readGrid("density.nvdb"));
handles.push_back(nanovdb::io::readGrid("velocity.nvdb"));
nanovdb::io::writeGrids("scene.nvdb", handles);
```

### Build types — what `grid<T>()` accepts

A NanoVDB grid is templated on a *build type*. Common ones:

| Build type        | What it stores                                     |
|-------------------|----------------------------------------------------|
| `float` / `double`| Scalar floating-point voxels (SDFs, density)       |
| `int32_t`         | Integer voxels (segmentation, indices)             |
| `Vec3f` / `Vec3d` | 3-vector voxels (velocity, color, normals)         |
| `bool`            | Active/inactive only (no value)                    |
| `ValueIndex`      | Each active voxel maps to a sequential index       |
| `ValueOnIndex`    | Sequential index over *active* voxels only         |
| `ValueMask`       | Pure topology — no values stored                   |
| `Half`, `Fp4/8/16`| Quantized scalar floats                            |
| `Point`           | Voxels carrying point counts (for point grids)     |

`ValueIndex` / `ValueOnIndex` are the basis of *IndexGrids* — covered in
Module 9.

`handle.grid<T>()` returns `nullptr` if T does not match what's in the
buffer, so always check.

### Quiz 2

> **Q1.** What does `handle.grid<float>(2)` return if grid #2 in the file is
> a `Vec3f` grid?
>
> **Q2.** A `.nvdb` file contains 3 grids (`density`, `temperature`,
> `velocity`). Write the code to load just the velocity grid by name.
>
> **Q3.** What's the difference between `readGrid(file, -1)` and
> `readGrids(file)`?

*(Answer key at end of document.)*

### If you have the repo

- Read: `nanovdb/io/IO.h` lines 1–120 for the public API
- Read: `nanovdb/cmd/print/nanovdb_print.cc` for a worked example
  inspecting all grids in a file
- Reference tool: `nanovdb_print my_file.nvdb` — prints per-grid metadata

---

## Module 3: Random access — `ReadAccessor`

### Core concept

The most common NanoVDB operation is "give me the value at integer
coordinate `(i, j, k)`". A `ReadAccessor` is the fast way to do that, and
it's how you do it on both CPU and GPU.

```cpp
const auto* grid = handle.grid<float>();
auto acc = grid->getAccessor();

float v = acc.getValue(nanovdb::Coord(10, 20, 30));
bool  on = acc.isActive(nanovdb::Coord(10, 20, 30));
```

### Coord vs world-space

`Coord` is a signed 32-bit integer 3-vector representing an *index-space*
coordinate. To query at a world-space point, convert first via the grid's
`Map`:

```cpp
nanovdb::Vec3f worldPt(1.25f, -0.7f, 3.1f);
nanovdb::Coord ijk = nanovdb::Coord::Floor(grid->worldToIndex(worldPt));
float v = acc.getValue(ijk);
```

(Use `Module 5: Math` for the sampling story when you need interpolation
rather than nearest-voxel.)

### Active vs background

```cpp
auto acc = grid->getAccessor();
const float bg = grid->tree().background();

for (auto& c : queries) {
    if (acc.isActive(c)) {
        // ... use the actual stored value
        float v = acc.getValue(c);
    } else {
        // ... or skip entirely — value would be `bg`
    }
}
```

For SDF-style grids, the *value* at an inactive voxel is meaningful too —
it's the signed-distance approximation in the inside/outside region. Don't
assume "inactive" means "ignore" without knowing what kind of grid you
have.

### `probeValue` / `probeLeaf` — value *and* state in one descent

`getValue(ijk)` answers "what value?" and `isActive(ijk)` answers "is it
on?" — but calling both walks the tree twice, and neither tells you whether
a *leaf even exists* for that coordinate. The accessor's `probe*` methods
collapse the question:

```cpp
auto acc = grid->getAccessor();
nanovdb::Coord ijk(10, 20, 30);

// probeValue: fills `v`, returns whether the voxel is ACTIVE — one descent.
float v;
bool on = acc.probeValue(ijk, v);   // v is always set; `on` is the state

// probeLeaf: the LEAF POINTER covering ijk, or nullptr if no leaf exists.
const nanovdb::NanoLeaf<float>* leaf = acc.probeLeaf(ijk);
if (leaf) {
    // a real leaf covers ijk — values here are voxel-resolution
} else {
    // ijk is covered by a tile (or background) — no per-voxel data here
}
```

Why this matters: Quiz 3 (Q3) noted that `getValue` returns the background
value in two distinct situations — "no covering node" and
"covered-but-equals-background". `getValue` alone can't tell them apart.
`probeValue` disambiguates the **active/inactive** axis (its bool), and
`probeLeaf` disambiguates the **leaf-exists** axis (nullptr or not). Together
they answer "did I land on a real voxel, an inactive voxel, or empty space?"
without re-traversing.

### Caching, thread safety, GPU

```cpp
// CPU: one accessor per thread
#pragma omp parallel
{
    auto myAcc = grid->getAccessor();   // local; not shared
    #pragma omp for
    for (size_t i = 0; i < N; ++i)
        out[i] = myAcc.getValue(coords[i]);
}

// GPU: one accessor per CUDA thread, constructed in the kernel
__global__ void kernel(const nanovdb::NanoGrid<float>* grid,
                       const nanovdb::Coord* coords,
                       float* out, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    auto acc = grid->getAccessor();      // local per-thread
    out[tid] = acc.getValue(coords[tid]);
}
```

The accessor is small (a few pointers and a `Coord`). Constructing one per
thread is essentially free.

### Common pitfall: shared accessor across threads

```cpp no-compile
// WRONG: race on the cached path
auto acc = grid->getAccessor();
std::for_each(std::execution::par, ..., [&](Coord c) { acc.getValue(c); });

// RIGHT: thread-local accessor
std::for_each(std::execution::par, ..., [&](Coord c) {
    auto myAcc = grid->getAccessor();   // local
    myAcc.getValue(c);
});
```

### Quiz 3

> **Q1.** You want to check whether 1,000 world-space points fall inside the
> active region of a level set grid. Outline the steps and write the inner
> loop body.
>
> **Q2.** Why is it incorrect to share one `ReadAccessor` across threads,
> even if all threads only call `getValue` (a "read-only" operation)?
>
> **Q3.** `acc.getValue(ijk)` returns the background value in two distinct
> situations. Name both. (These are defined behaviors, not bugs — but they
> matter because "I got back the background" doesn't uniquely identify
> what happened in the tree traversal.)

*(Answer key at end of document.)*

### If you have the repo

- Read: `nanovdb/NanoVDB.h`, search for `class ReadAccessor` for the public
  interface
- Run: `nanovdb/examples/ex_read_nanovdb_sphere/` — minimal accessor usage

---

## Module 4: Sequential access — `NodeManager`

### Core concept

`ReadAccessor` is for *random* access at arbitrary coordinates. Sometimes
you want the opposite: visit *every* active node in linear order — every
leaf, every lower-internal, every upper-internal — for things like:

- Computing per-leaf statistics (min/max/avg)
- Launching one CUDA thread per active voxel
- Building a parallel reduction over the tree
- Iterating leaf-by-leaf for IO

`NodeManager` is a small acceleration structure that holds a linear array
of pointers to every node at every level.

```cpp
#include <nanovdb/NodeManager.h>

const auto* grid = handle.grid<float>();
auto nodeMgr = nanovdb::createNodeManager(*grid);
const auto* mgr = nodeMgr.mgr<float>();

uint64_t leafN  = mgr->leafCount();
uint64_t lowerN = mgr->lowerCount();
uint64_t upperN = mgr->upperCount();

// Iterate every leaf
for (uint64_t i = 0; i < leafN; ++i) {
    const auto& leaf = mgr->leaf(i);
    for (auto it = leaf.beginValueOn(); it; ++it) {
        float v = *it;
        nanovdb::Coord c = it.getCoord();
        // ... do something with active voxel
    }
}
```

### When to use which

| You want to...                                                  | Use            |
|-----------------------------------------------------------------|----------------|
| Look up a value at a known coordinate                           | `ReadAccessor` |
| Visit every active voxel (or every leaf)                        | `NodeManager`  |
| Launch one GPU thread per active voxel                          | `NodeManager`  |
| Do neighbor lookups along a ray or stencil                      | `ReadAccessor` |
| Compute per-leaf statistics over all leaves                     | `NodeManager`  |

### Things worth knowing

**Iteration order.** Nodes are arranged *breadth-first per level* in memory:
all leaves contiguous, then all lowers, then all uppers. Within a single
level, the order is the build-time tree-traversal order (root's tile-map
iteration → each upper's tile-array iteration → each lower's tile-array
iteration). It is **not** strictly spatially coherent — adjacent
`mgr->leaf(i)` and `mgr->leaf(i+1)` may not be spatial neighbors. If you
need spatial locality, build it yourself (e.g. sort by `leaf.origin()`).

**One manager covers every level.** `NodeManager<BuildT>` is templated on
the build (value) type, *not* the level. The same instance exposes
`leaf(i)` / `lower(i)` / `upper(i)` and `leafCount()` / `lowerCount()` /
`upperCount()`. No need for a separate manager per level — you build once
and iterate at whatever level your algorithm wants:

```cpp
auto nmh = nanovdb::createNodeManager(*grid);
const auto* mgr = nmh.mgr<float>();

// Per-lower-tile statistics
for (uint64_t i = 0; i < mgr->lowerCount(); ++i) {
    const auto& low = mgr->lower(i);
    // ... low is a NanoLower<float> with 16^3 children/tiles
}
```

**`mgr<T>()` vs `deviceMgr<T>()`.** Two buffer types in current NanoVDB:

| `BufferT`             | Address space(s) | `mgr<T>()` returns | `deviceMgr<T>()` |
|-----------------------|------------------|--------------------|-------------------|
| `HostBuffer`          | host only        | host pointer       | not available     |
| `cuda::DeviceBuffer`  | host **and** device (always dual; `hasDeviceDual = true`) | host pointer | device pointer |

`cuda::DeviceBuffer` is always dual: the host side holds the canonical
bytes and `deviceUpload()` / `deviceDownload()` move them across the
PCIe boundary explicitly. So `mgr<T>()` gives you the host-side
NodeManager (for inspection or per-leaf algorithms running on the CPU),
and `deviceMgr<T>()` gives you the device-side pointer you pass into a
kernel. **Common mistake:** passing the host pointer into a kernel by
accident because both methods are available — always `deviceMgr` for
kernel args.

<!-- teachme:stale-ok — the renamed header below is shown deliberately as a counter-example -->
> Heads up: older docs and examples (e.g. `HelloWorld.md`) reference
> `#include <nanovdb/util/cuda/CudaDeviceBuffer.h>` and the type
> `CudaDeviceBuffer`. That header was renamed; current code is
> `#include <nanovdb/cuda/DeviceBuffer.h>` and `cuda::DeviceBuffer`.
> `CudaDeviceBuffer` survives as a deprecated alias
> (`DeviceBuffer.h:404`) for source compatibility.

### GPU NodeManager

The CUDA variant is in `nanovdb/cuda/NodeManager.cuh`. Same idea, but the
manager itself lives in device memory:

```cpp
#include <nanovdb/cuda/NodeManager.cuh>

auto deviceHandle = handle.copy<nanovdb::cuda::DeviceBuffer>();
deviceHandle.deviceUpload();                               // push bytes to device!
const auto* dGrid = deviceHandle.deviceGrid<float>();

auto dNodeMgr = nanovdb::cuda::createNodeManager(dGrid);   // device-side
const auto* dMgr = dNodeMgr.template deviceMgr<float>();

// Launch one block per leaf (your kernel takes the device grid + manager):
//   myKernel<<<dMgr->leafCount(), 64>>>(dGrid, dMgr);
```

### Quiz 4

> **Q1.** You want to compute the per-leaf maximum voxel value. Would you
> use `ReadAccessor` or `NodeManager`? Why?
>
> **Q2.** Your grid has 10,000 leaves, each with ≈100 active voxels. Roughly
> how many active voxels total? How many threads would you launch in a
> "one thread per active voxel" GPU kernel?
>
> **Q3.** What does `mgr->upperCount()` return if a grid contains exactly
> one active voxel?

*(Answer key at end of document.)*

### If you have the repo

- Read: `nanovdb/NodeManager.h` (CPU)
- Read: `nanovdb/cuda/NodeManager.cuh` (GPU)
- Run: `nanovdb/examples/ex_nodemanager_cuda/`

---

## Module 5: Math and sampling

### Core concept

Everything in `nanovdb/math/*.h` is `__hostdev__` — same code, host and
device. These are the primitives that show up in every NanoVDB algorithm:

- `Coord` — signed integer 3-vector
- `Vec3<T>` / `Vec3f` / `Vec3d` — floating-point 3-vector
- `BBox<T>` — axis-aligned bounding box
- `Map` — affine world ↔ index transform (stored inside the grid)
- `Ray<T>` — ray, with intersection helpers
- `HDDA` — hierarchical DDA, fast ray traversal of the tree
- `SampleFromVoxels` — trilinear / triquadratic / tricubic interpolation
- `Stencils` — finite-difference stencils (gradient, curvature)
- `DitherLUT` — dither tables for quantized grids

### Coord, BBox, Map

```cpp
nanovdb::Coord a(1, 2, 3);
nanovdb::Coord b = a + nanovdb::Coord(1, 0, 0);   // (2, 2, 3)
auto floored = nanovdb::Coord::Floor(nanovdb::Vec3f(1.7f, 2.1f, 3.9f));
// (1, 2, 3)

auto ibb = grid->indexBBox();                    // CoordBBox in index space
auto wbb = grid->worldBBox();                    // BBox<Vec3d> in world space

nanovdb::Vec3f w(2.5f, 1.2f, 0.0f);
auto ijkF = grid->worldToIndex(w);               // floating-point ijk
auto ijk  = nanovdb::Coord::Floor(ijkF);         // nearest integer voxel
```

### Trilinear sampling

```cpp
#include <nanovdb/math/SampleFromVoxels.h>

const auto* grid = handle.grid<float>();
auto acc = grid->getAccessor();

// Build a sampler — second template arg = polynomial order (1 = trilinear)
auto sampler = nanovdb::math::createSampler<1>(acc);

nanovdb::Vec3f worldPt(1.25f, -0.7f, 3.1f);
auto idxPt = grid->worldToIndex(worldPt);
float v = sampler(idxPt);     // trilinear sample
```

Higher orders are available: `createSampler<0>` (nearest, `NearestNeighborSampler`),
`<1>` (trilinear, `TrilinearSampler` — 2³ = 8 voxels read), `<2>`
(triquadratic, `TriquadraticSampler` — 3³ = 27), `<3>` (tricubic,
`TricubicSampler` — 4³ = 64). The naming matches the **polynomial degree**:
degree-1 / 2 / 3 across each axis.

### Ray marching with HDDA

```cpp
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>

auto acc = grid->getAccessor();

nanovdb::math::Ray<float> wRay(/*eye=*/{0, 0, -5}, /*dir=*/{0, 0, 1});
auto iRay = wRay.worldToIndexF(*grid);            // ray in index space

// Idiomatic level-set hit: ZeroCrossing drives an HDDA internally and stops
// at the first sign change. (This is what ex_raytrace_level_set uses.)
nanovdb::Coord ijk;
float v, t;
if (nanovdb::math::ZeroCrossing(iRay, acc, ijk, v, t)) {
    // hit at index-space voxel `ijk`; SDF value `v`; ray parameter `t`
}
```

Driving the HDDA yourself, one level down: its constructor takes the ray
**and** the starting march dimension from `getDim` (1 = per-voxel, or
8 / 128 / 4096 to skip an empty leaf / lower / upper node):

```cpp
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>

auto acc = grid->getAccessor();
nanovdb::math::Ray<float> wRay({0, 0, -5}, {0, 0, 1});
auto iRay = wRay.worldToIndexF(*grid);

nanovdb::Coord ijk(0, 0, 0);
nanovdb::math::HDDA<decltype(iRay), nanovdb::Coord>
    hdda(iRay, acc.getDim(ijk, iRay));
while (hdda.step()) {
    if (acc.isActive(hdda.voxel())) break;        // first active voxel
}
```

**How it actually works.** The "DDA" half is the classic Amanatides–Woo
ray-stepping math: parameterize the ray by `t`, track per-axis `tMax`
values (parameter at which the ray crosses the next axis-aligned plane at
the current stride), advance to the smallest `tMax`, repeat. The "H" half
is what makes it sparse-friendly: on each step HDDA asks the accessor
`acc.getDim(ijk, ray)` for the **stride** (the edge length, in voxels, of
the region it can skip) at the current location. The menu of strides is the
node edge lengths — 1 (voxel), 8 (leaf), 128 (lower), 4096 (upper). The
exact value: inside an active leaf you get 1 (step voxel-by-voxel); at a
*tile* (a node slot with no child below) you get the **child** node's edge
length — 8 at a lower-node tile, 128 at an upper-node tile, 4096 at a root
tile; and you get a node's **own** edge length only when the whole node is
flagged skippable. So a ray crossing empty space lands at the next
non-empty region's face in one step instead of visiting every voxel. Descent is implicit:
the next `getDim` call at the new `ijk` returns a smaller stride once the
ray enters more-populated regions. This is also where the multi-level
accessor cache (Module 1) earns its keep — the upper/lower/leaf slots
are populated during descent, so subsequent `getDim` queries on a ray
still inside the same upper can skip the root walk.

### Stencils

```cpp
#include <nanovdb/math/Stencils.h>

// GradStencil is templated on the GRID type (not a pointer): pass NanoGrid<T>.
nanovdb::math::GradStencil<nanovdb::NanoGrid<float>> stencil(*grid);
stencil.moveTo(nanovdb::Coord(10, 20, 30));
nanovdb::Vec3f gradient = stencil.gradient();
```

Stencils are useful for computing gradients of SDFs (for surface normals)
or curl/divergence of velocity grids.

### Quiz 5

> **Q1.** What does `__hostdev__` mean? Why are all of `nanovdb/math/*.h`
> tagged with it?
>
> **Q2.** You have a grid with `voxelSize() = 0.1`. A world-space point is at
> `(1.25, -0.7, 3.1)`. What's the corresponding nearest-integer index
> coordinate? (Assume identity origin.)
>
> **Q3.** Why is HDDA dramatically faster than naive 1-voxel-at-a-time
> ray marching for a sparse grid?

*(Answer key at end of document.)*

### If you have the repo

- Read: `nanovdb/math/Math.h` — Coord, BBox, Vec3, Map basics
- Read: `nanovdb/math/HDDA.h` — ray traversal
- Read: `nanovdb/math/SampleFromVoxels.h` — sampling
- Run: `nanovdb/examples/ex_raytrace_*/` — multiple HDDA examples

---

## Module 6: GPU side — kernels using a `Grid`

### Core concept

A NanoVDB grid is one contiguous buffer. To use it on the GPU you
`cudaMemcpy` (or equivalent) that buffer to device memory, get a device
pointer to the grid, and pass it as a kernel argument. Everything
inside — accessors, sampling, HDDA — is `__hostdev__` and works
identically on device.

`GridHandle<BufferT>` is the bookkeeping wrapper for this. The
`BufferT = HostBuffer` default lives on the CPU; `BufferT = DeviceBuffer`
lives on the GPU; `BufferT = CudaDeviceBuffer` is a dual-buffer that owns
both copies.

### Moving a host handle to the GPU

```cpp
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/io/IO.h>

// Load on host
auto hostHandle = nanovdb::io::readGrid("density.nvdb");

// Deep copy to a (dual host+device) buffer, then upload to the device.
// copy<DeviceBuffer>() fills only the HOST side; deviceUpload() pushes the
// bytes to the GPU. Skip the upload and deviceGrid() returns nullptr.
auto devHandle = hostHandle.copy<nanovdb::cuda::DeviceBuffer>();
devHandle.deviceUpload();

const auto* dGrid = devHandle.deviceGrid<float>();
if (!dGrid) throw std::runtime_error("device grid null (did you deviceUpload?)");
```

`deviceGrid<T>()` returns a `const NanoGrid<T>*` that points into device
memory. You can pass that pointer straight to a kernel.

### A minimal kernel

```cpp
__global__ void sumAtCoords(const nanovdb::NanoGrid<float>* grid,
                            const nanovdb::Coord* coords,
                            float* outSum, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    auto acc = grid->getAccessor();         // local
    float v = acc.getValue(coords[tid]);
    atomicAdd(outSum, v);
}

// Launch
sumAtCoords<<<(N + 127) / 128, 128>>>(dGrid, dCoords, dOut, N);
```

### Sampling on the device

`createSampler<1>(acc)` works identically on host and device. The standard
GPU pattern: one thread per query point, each with its own accessor and
sampler.

```cpp
__global__ void sampleAtPoints(const nanovdb::NanoGrid<float>* grid,
                               const nanovdb::Vec3f* worldPts,
                               float* outVals, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    auto acc = grid->getAccessor();
    auto sampler = nanovdb::math::createSampler<1>(acc);
    auto idxPt = grid->worldToIndex(worldPts[tid]);
    outVals[tid] = sampler(idxPt);
}
```

### CMake integration

```cmake
find_package(nanovdb CONFIG REQUIRED)

add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE nanovdb::nanovdb)
```

**Don't turn on `CUDA_SEPARABLE_COMPILATION` (RDC) by default.** NanoVDB is
header-only; every `__hostdev__` template is instantiated in the calling
`.cu` file, so cross-TU device linkage isn't needed. RDC has real
performance costs (less aggressive cross-TU inlining, more conservative
register allocation, more device-link relocations, requires opt-in for
device-LTO). Only enable it if device code in one TU calls a `__device__`
function defined in another TU — dynamic parallelism, separately-compiled
CUDA libraries, or shared cross-TU `__device__` helpers.

### Quiz 6

> **Q1.** Why is it safe to pass a `const NanoGrid<float>*` from device
> memory directly to a `__global__` kernel, but unsafe to pass a host-side
> `std::vector<float>` pointer?
>
> **Q2.** A `GridHandle<HostBuffer>` is on the CPU. Write the line that
> moves it to a `GridHandle<cuda::DeviceBuffer>` on the GPU.
>
> **Q3.** Inside a CUDA kernel, you call `acc.getValue(ijk)`. Where does
> the accessor's cache live? (Hint: which memory space?)

*(Answer key at end of document.)*

### If you have the repo

- Read: `nanovdb/cuda/DeviceBuffer.h`, `nanovdb/cuda/GridHandle.cuh`
- Run: `nanovdb/examples/ex_make_custom_nanovdb_cuda/`
- Run: `nanovdb/examples/ex_raytrace_level_set/`

---

## Module 7: Building grids on the GPU

### Core concept

For interactive workloads (point clouds, mesh-derived grids, simulation
output) you don't want to go OpenVDB → host → device every frame. You want
to build a NanoVDB grid directly from device-side data.

`nanovdb/tools/cuda/` contains the GPU builders:

- `PointsToGrid.cuh` — point cloud → grid (e.g. for splatting)
- `MeshToGrid.cuh` — triangle mesh → IndexGrid (recent: PR #2178)
- `IndexToGrid.cuh` — `IndexGrid` → value grid
- `DistributedPointsToGrid.cuh` — multi-GPU points → grid
- `AddBlindData.cuh` — attach blind data

There's also `nanovdb/tools/GridBuilder.h` (CPU) for fine-grained
construction.

### PointsToGrid

`PointsToGrid<BuildT, ResourceT>` is a *class*, configured then run via
`getHandle(points, count)`. Build type `Point` produces a grid whose
voxels carry indices into a blind-data array of vertex positions; other
build types produce only the topology.

```cpp
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/cuda/DeviceBuffer.h>

// Device array of point positions (Vec3f or Vec3d), already populated.
// Non-const: the Point build type copies these into the grid's blind data.
nanovdb::Vec3f* dPoints;
size_t nPts;

// Construct with a voxel size (and optional translation, stream)
nanovdb::tools::cuda::PointsToGrid<nanovdb::Point> converter(
    /*voxelSize=*/0.05);
converter.setGridName("particles");                          // optional

// Run it on the device data
auto devHandle = converter.getHandle(dPoints, nPts);          // GridHandle<DeviceBuffer>
const auto* dGrid = devHandle.deviceGrid<nanovdb::Point>();

// Alternative ctor: target max points-per-voxel instead of voxel size.
// PointsToGrid<Point>(maxPointsPerVoxel=8, tolerance=1, maxIterations=10);
```

### MeshToGrid

`MeshToGrid<BuildT>` is also a class. Vertex positions are `Vec3f` on the
device, triangle indices are `Vec3i` (three vertex indices per triangle).
The result is an `OnIndexGrid` covering a narrow band around the
mesh surface; pair it with the optional `getHandleAndUDF()` overload if
you also need per-active-voxel unsigned-distance values.

```cpp
#include <nanovdb/tools/cuda/MeshToGrid.cuh>

const nanovdb::Vec3f* dVerts;   // [vertN] vertex positions
const nanovdb::Vec3i* dTris;    // [triN]  triangle index triples
size_t vertN, triN;

double voxelSize = 0.01;
nanovdb::tools::cuda::MeshToGrid<nanovdb::ValueOnIndex> mesher(
    dVerts, vertN, dTris, triN,
    nanovdb::Map(voxelSize));   // affine map from voxel size
mesher.setNarrowBandWidth(3.f);

auto devHandle = mesher.getHandle();                       // GridHandle<DeviceBuffer>
const auto* dGrid = devHandle.deviceGrid<nanovdb::ValueOnIndex>();

// If you also want a sidecar buffer of per-active-voxel UDF values:
auto [gridH, udfBuffer] = mesher.getHandleAndUDF();        // pair
```

This is the GPU equivalent of OpenVDB's mesh-to-volume; useful for
on-the-fly collision grids or rendering acceleration.

### Host-side construction with `tools::build::Grid`

The host-side builder lives in `nanovdb::tools::build::Grid<BuildT>` and
its associated accessor types. After populating the build-tree, you bake
it into a packed NanoVDB grid via `createNanoGrid`.

```cpp
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>

nanovdb::tools::build::Grid<float> grid(/*background=*/0.0f,
                                        /*name=*/"ls_sphere",
                                        nanovdb::GridClass::LevelSet);
grid.setTransform(/*scale=*/0.1);                          // voxel size

auto acc = grid.getAccessor();                             // not thread-safe
for (int z = -5; z < 5; ++z)
    for (int y = -5; y < 5; ++y)
        for (int x = -5; x < 5; ++x)
            acc.setValue(nanovdb::Coord(x, y, z),
                         std::sqrt(float(x*x + y*y + z*z)));

// Or, for multi-threaded writes:
//   auto wAcc = grid.tree().getWriteAccessor();           // thread-safe writer

// Bake the build-tree into a packed NanoVDB grid
auto handle = nanovdb::tools::createNanoGrid(grid);        // GridHandle<HostBuffer>
```

`CreatePrimitives.h` has helpers for common test grids:
`createLevelSetSphere`, `createLevelSetTorus`, `createLevelSetBox`, etc.

### Quiz 7

> **Q1.** Why might you prefer `pointsToGrid` over loading an OpenVDB grid
> for an interactive application that receives a new point cloud every
> frame?
>
> **Q2.** `MeshToGrid` builds an `OnIndexGrid`. What's a build type and why
> is `OnIndexGrid` a sensible default for a narrow-band-around-the-surface
> representation?
>
> **Q3.** Look at the `GridBuilder` example above. Could the inner loop run
> in parallel as written?

*(Answer key at end of document.)*

### If you have the repo

- Read: `nanovdb/tools/cuda/PointsToGrid.cuh` — public API at top
- Read: `nanovdb/tools/cuda/MeshToGrid.cuh`
- Read: `nanovdb/tools/GridBuilder.h`
- Run: `nanovdb/examples/ex_make_custom_nanovdb_cuda/`

---

## Module 8: Reshaping grids — topology ops, data re-homing, SDF sign-fill

### The organizing dichotomy

Once you have a grid you often want to change it. Every tool in this corner
of `nanovdb/tools/cuda/` falls into one of two kinds, and the kind predicts
its whole signature:

| | **Topology operators** | **Data operators** |
|---|---|---|
| Change | *which voxels exist* | *values of existing voxels* |
| Examples | `DilateGrid`, `CoarsenGrid`, `RefineGrid`, `PruneGrid`, `MergeGrids` | `signedFloodFill` |
| Output | a **new** grid (`getHandle()`) | **in place** |
| Build type | **`ValueOnIndex` only** | the value type (e.g. `float`) |
| Why | topology is frozen → can't grow in place; new voxels have no data to invent, so they live on `ValueOnIndex` where the "value" is just an index | values live in the grid → needs a real value type; counts don't change → no rebuild, safe in place |

Ask "active set or values?" of any tool here and you can predict the rest.
The two halves aren't independent, though: a topology op renumbers the
active-voxel indices, so your data has to be **re-homed** afterward — that's
section 8b.

### 8a — Topology operators

Like `PointsToGrid` / `MeshToGrid`, these are **builder classes** built on
the shared `TopologyBuilder`, which `static_assert`s
`BuildTraits<BuildT>::is_onindex` — so they take `ValueOnIndex` only.
Construct from a device grid pointer, get a new handle:

```cpp
#include <nanovdb/tools/cuda/DilateGrid.cuh>

auto* idxGrid = devH.deviceGrid<nanovdb::ValueOnIndex>();
nanovdb::tools::cuda::DilateGrid<nanovdb::ValueOnIndex> dilater(idxGrid);
dilater.setOperation(nanovdb::tools::morphology::NN_FACE_EDGE_VERTEX);
// NN_FACE (6) | NN_FACE_EDGE (18) | NN_FACE_EDGE_VERTEX (26-connected)
auto dilatedH = dilater.getHandle();          // GridHandle<cuda::DeviceBuffer>
```

Dilate exists because a radius-`r` stencil or sparse convolution reads
neighbors *outside* the active set; pre-dilating makes those neighbors
exist. The rest of the family:

```cpp
#include <nanovdb/tools/cuda/CoarsenGrid.cuh>
#include <nanovdb/tools/cuda/RefineGrid.cuh>

using OnIdx = nanovdb::ValueOnIndex;
nanovdb::tools::cuda::CoarsenGrid<OnIdx> coarsener(devH.deviceGrid<OnIdx>());
auto coarseH = coarsener.getHandle();   // voxel size ×2, ~1/8 the voxels

nanovdb::tools::cuda::RefineGrid<OnIdx> refiner(coarseH.deviceGrid<OnIdx>());
auto fineH = refiner.getHandle();       // voxel size ÷2, up to 8× the voxels
```

`PruneGrid` is the one with a different constructor: it takes a per-leaf
`Mask<3>*` whose set bits mark the voxels to **keep** (one mask per leaf).

```cpp
#include <nanovdb/tools/cuda/PruneGrid.cuh>

const nanovdb::Mask<3>* dLeafMask = nullptr;   // device array, one Mask per leaf
nanovdb::tools::cuda::PruneGrid<nanovdb::ValueOnIndex>
    pruner(devH.deviceGrid<nanovdb::ValueOnIndex>(), dLeafMask);
auto prunedH = pruner.getHandle();
```

`MergeGrids<OnIdx>(gridA, gridB)` unions two topologies. CPU morphology
equivalents live in `nanovdb/util/MorphologyHelpers.h`.

### 8b — Re-homing data after a topology change (`Injection.cuh`)

A topology op gives you a new grid with a **renumbered** index enumeration.
Your old sidecar array (length = old active count) no longer lines up with
it. `nanovdb/util/cuda/Injection.cuh` is the tool that moves your data
across — and tells you which voxels are new.

Two functors do the work, launched through the kernel wrappers in
`util/cuda/Util.h`:

- `InjectGridDataFunctor<BuildT, ValueT>` (via `operatorKernel`) copies the
  sidecar of the source into the sidecar of the destination, *only* where
  the two grids' active voxels overlap. New voxels are left untouched.
- `InjectGridMaskFunctor<BuildT>` (via `lambdaKernel`) writes a per-leaf
  `Mask<3>` of the overlap (`srcLeaf.valueMask & dstLeaf.valueMask`) — so
  the **new** voxels are `dstLeaf.valueMask AND NOT overlap`.

```cpp
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/util/cuda/Injection.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Util.h>

using OnIdx = nanovdb::ValueOnIndex;
const auto* srcGrid = devH.deviceGrid<OnIdx>();
auto dilatedH = nanovdb::tools::cuda::DilateGrid<OnIdx>(srcGrid).getHandle();
const auto* dstGrid = dilatedH.deviceGrid<OnIdx>();

// 1) Move overlapping values: oldSidecar -> newSidecar (sized to new count).
const float* oldSidecar = nullptr;   // your old per-active-voxel data
float*       newSidecar = nullptr;   // fresh alloc, size = new activeVoxelCount(+1)
auto srcLeaves = nanovdb::util::cuda::DeviceGridTraits<OnIdx>::getTreeData(srcGrid).mNodeCount[0];
nanovdb::util::cuda::operatorKernel<
    nanovdb::util::cuda::InjectGridDataFunctor<OnIdx, float>>
    <<<srcLeaves, 256>>>(srcGrid, dstGrid, oldSidecar, newSidecar);

// 2) Find the new voxels: per-leaf overlap mask; new = dstMask & ~overlap.
auto dstLeaves = nanovdb::util::cuda::DeviceGridTraits<OnIdx>::getTreeData(dstGrid).mNodeCount[0];
auto maskBuf = nanovdb::cuda::DeviceBuffer::create(
    dstLeaves * sizeof(nanovdb::Mask<3>), nullptr, false);
auto* overlap = static_cast<nanovdb::Mask<3>*>(maskBuf.deviceData());
unsigned nthreads = 128, nblocks = (dstLeaves + nthreads - 1) / nthreads;
nanovdb::util::cuda::lambdaKernel<<<nblocks, nthreads>>>(dstLeaves,
    nanovdb::util::cuda::InjectGridMaskFunctor<OnIdx>(),
    srcGrid, dstGrid, overlap);
// Now initialize newSidecar at the new voxels: their index is dstGrid's
// getValue(ijk), enumerated where overlap is off.
```

A voxel's sidecar index *is* what a `ValueOnIndex` grid stores —
`getValue(ijk)` returns it (mechanically: leaf `firstOffset()` + the
voxel's rank within the leaf). So once you know which voxels are new, you
know exactly which `newSidecar` slots to fill.

**Simpler alternative:** pre-fill `newSidecar` with a sentinel (NaN, −1)
before step 1. Injection only writes the overlap, so any slot still holding
the sentinel afterward is a new voxel — one scan finds them, no mask pass.

(The verified end-to-end version of this — dilate, inject mask, prune back
to the original — is the `DilateInjectPrune_ValueOnIndex` unit test.)

### 8c — A data operator: `signedFloodFill`

The odd one out, and the cleanest illustration of the dichotomy. A
level-set SDF only stores meaningful distances in the narrow band; the far
interior/exterior are inactive tiles with no sign yet. `signedFloodFill`
propagates the sign outward — walking the tree and setting each inactive
tile to ±background by inside/outside — so queries beyond the band get the
right sign (needed for CSG, collision, and the `ZeroCrossing` test).

```cpp
#include <nanovdb/tools/cuda/SignedFloodFill.cuh>

// In place on the *non-const* device grid; float, not ValueOnIndex.
nanovdb::tools::cuda::signedFloodFill(devH.deviceGrid<float>());
```

It writes *values* (signs), not topology — which is why it needs `float`
and why it's safe in place (no counts change, no offset invalidation).

Because it's `float`-only, you can't run it on a `ValueOnIndex` grid
directly. If your distances live in a sidecar, bake them into a value grid
first with `indexToGrid<float>` (Module 9), then flood-fill. (Note:
flood-fill *propagates* the band's existing signs into the inactive tiles
— it can't sign an unsigned distance field for you.)

### 8d — Stats are cached; editing values makes them STALE

Every node in the tree caches summary statistics of the voxels beneath it:
min, max, average, standard deviation, and an active-voxel bounding box.
HDDA space-skipping (Module 5) and ray-tracing lean on the cached **bbox**
to decide a whole node is empty and skip it in one step. The min/max caches
feed CSG and range queries.

The catch: if you **mutate voxel values in place** — `signedFloodFill`, a
custom kernel that overwrites a leaf, anything that changes values without
rebuilding — those caches do **not** update themselves. They go *stale*: a
node still advertises an old bbox/min/max that no longer matches its
voxels. Stale stats silently produce wrong renders (HDDA skips a node that
is actually occupied) and wrong range queries. **After any manual value
edit, recompute the stats.**

```cpp global
#include <nanovdb/tools/GridStats.h>

static void refreshStats(nanovdb::NanoGrid<float>* mutableGrid)
{
    // Recompute bbox + min/max + average + stddev for every node.
    nanovdb::tools::updateGridStats(mutableGrid, nanovdb::tools::StatsMode::All);
    // Cheaper modes: ::BBox (just bboxes — enough to fix HDDA skipping),
    // ::MinMax (bbox + extrema), ::Disable (skip), ::Default (== All).
}
```

`StatsMode` is the dial: `BBox` only fixes the topology bbox (the minimum to
keep HDDA correct), `MinMax` adds extrema, `All` (the default) adds average
and standard deviation.

To query the extrema of an arbitrary index-space box *without* relying on
the cache, `getExtrema` walks the voxels directly:

```cpp
#include <nanovdb/tools/GridStats.h>

nanovdb::CoordBBox box(nanovdb::Coord(-10), nanovdb::Coord(10));
auto ex = nanovdb::tools::getExtrema(*grid, box);
float lo = ex.min(), hi = ex.max();
```

On the GPU the same recompute lives in `tools/cuda/GridStats.cuh` and takes
a device grid pointer plus an optional stream:

```cpp
#include <nanovdb/tools/cuda/GridStats.cuh>

// dGrid here is a non-const device pointer to the grid you just edited.
nanovdb::NanoGrid<float>* dMutable = nullptr;   // your device grid
nanovdb::tools::cuda::updateGridStats(dMutable, nanovdb::tools::StatsMode::All);
```

So `signedFloodFill` (8c) and `updateGridStats` are a natural pair: flood
the signs, then refresh the stats so the freshly-signed tiles don't break
the next ray-march.

### 8e — Validation & integrity (certify a grid)

A NanoVDB grid is one offset-addressed buffer (Module 1): a bad write, a
truncated file, a half-finished device transfer, or a stats refresh you
forgot can leave it self-inconsistent. Two tools certify it.

**Checksum** (`tools/GridChecksum.h`) — a hash over the buffer, stored *in*
the grid header. `evalChecksum` computes it, `validateChecksum` compares the
stored one against a fresh recompute (cheap detection of corruption /
truncation), and `updateChecksum` rewrites it after a legitimate edit.

```cpp
#include <nanovdb/tools/GridChecksum.h>

// Detect corruption: stored checksum vs. fresh recompute.
bool ok = nanovdb::tools::validateChecksum(grid, nanovdb::CheckMode::Full);

// After a legitimate in-place edit, refresh the stored checksum so future
// validation passes. (Needs a non-const grid.)
nanovdb::NanoGrid<float>* mutableGrid = nullptr;   // your edited grid
nanovdb::tools::updateChecksum(mutableGrid, nanovdb::CheckMode::Full);
```

`CheckMode` trades speed for coverage: `Disable`/`Empty` (none),
`Half`/`Partial`/`Default` (fast, hashes grid+tree+root only), `Full`
(slow, hashes the node and blind-data blocks too).

**Validator** (`tools/GridValidator.h`) — deeper structural checks (grid
type/class consistency, node layout, *and* a checksum check). `isValid`
returns a bool; `checkGrid` writes a human-readable reason into a buffer.

```cpp
#include <nanovdb/tools/GridValidator.h>

if (!nanovdb::tools::isValid(grid, nanovdb::CheckMode::Full, /*verbose=*/true)) {
    // grid failed validation — bad type, bad layout, or checksum mismatch
}

// Or get the specific reason as a string (StrLen is the max message length;
// CheckMode is a scoped enum, so cast it for the array bound):
char err[(uint32_t)nanovdb::CheckMode::StrLen] = {};
nanovdb::tools::checkGrid(grid, err, nanovdb::CheckMode::Full);
// err is "" if the grid is valid, otherwise the first failure reason
```

Use case: certify right after `readGrid` (catch a bad file), after a
host→device `copy` (catch a truncated transfer — GPU variants live in
`tools/cuda/GridChecksum.cuh` and `tools/cuda/GridValidator.cuh`), and after
any build or in-place edit (then `updateChecksum` so the stored hash stays
honest). The CLI `nanovdb_validate` wraps the same checks.

### Quiz 8

> **Q1.** You ran `CoarsenGrid` on an `OnIndexGrid` that had a per-voxel
> color sidecar. The operator hands you back a new grid but does **not**
> fill in colors for the coarsened voxels. Why is it *right* for it to
> refuse — what does it not know that you do?
>
> **Q2.** `signedFloodFill` mutates a `float` grid in place, while every
> topology operator returns a new grid. Reason from "what each one changes"
> to *why* the data operator can get away with in-place mutation but the
> topology operators cannot.

*(Answer key at end of document.)*

### If you have the repo

- Read: `nanovdb/tools/cuda/DilateGrid.cuh`, `CoarsenGrid.cuh`,
  `RefineGrid.cuh`, `PruneGrid.cuh`, `SignedFloodFill.cuh`
- Read: `nanovdb/util/cuda/Injection.cuh` — the re-homing functors
- Read: `nanovdb/unittest/TestNanoVDB.cu` → `DilateInjectPrune_ValueOnIndex`
  for the full dilate → inject → prune workflow
- Read: `nanovdb/util/MorphologyHelpers.h` (CPU)

---

## Module 9: `IndexGrid` and `VoxelBlockManager`

### Core concept — IndexGrid

A regular NanoVDB grid stores a value at every active voxel (a float, a
Vec3f, etc.) — values live inside the grid buffer. An **IndexGrid** instead
stores an *index*: each active voxel knows its sequential position in a
separate, externally-managed value array.

```
Regular grid:                  IndexGrid + external array:
[GridData ... LeafData(values)]    [GridData ... LeafData(uint64 indices)] + [values...]
```

Build types:

- `ValueIndex` — each *voxel slot* (active or not) carries a sequential
  index. Indices go 0, 1, 2, ... over the full voxel slots.
- `ValueOnIndex` — only *active* voxels carry sequential indices. Indices
  go 0, 1, 2, ... over actives, skipping inactives.

### Why IndexGrid

- **Memory**: store huge multi-channel attribute data outside the grid
  buffer once, share one topology across many attribute arrays.
- **GPU update**: change attribute values without rebuilding the grid.
- **Custom types**: attributes can be anything — `float3`, `int32_t`,
  custom structs, multi-channel — not constrained to NanoVDB's build types.

You obtain the device grid with
`devHandle.deviceGrid<nanovdb::ValueOnIndex>()` and read it through a normal
accessor — but the "value" you get back is an index into your external
attribute array:

```cpp
// dValues[i] is the per-active-voxel attribute, sequentially indexed
// (length = active voxel count, plus a background slot at index 0).
__global__ void readActiveValues(
    const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* g,
    const float* dValues,
    nanovdb::Coord c,
    float* out)
{
    auto a = g->getAccessor();
    uint64_t idx = a.getValue(c);   // ValueOnIndex: this *is* the value
    *out = dValues[idx];
}
```

### `ChannelAccessor` — the typed way to read the sidecar

The manual `getValue(ijk)` → `array[idx]` above works, but it's two steps
you have to keep in sync, and nothing stops you from indexing the wrong
array. `ChannelAccessor<ChannelT, IndexT>` (in `NanoVDB.h`) wraps the index
grid *and* its value array into one object whose `getValue(ijk)` returns the
**attribute itself**, not the index. It does the index lookup and the array
fetch in one call.

```cpp
// IndexT defaults to ValueIndex, so spell out ValueOnIndex explicitly.
__global__ void readChannel(
    const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* g,
    float* dValues,                 // external sidecar (the "channel")
    nanovdb::Coord c,
    float* out)
{
    // Construct from the grid + an external channel pointer.
    nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex> ca(*g, dValues);
    *out = ca.getValue(c);          // attribute value, NOT the index
    // ca(c) is the same; ca.getIndex(c) still gives the raw index if needed.
}
```

If the channel is stored as the grid's *blind data* (e.g. a `ChannelArray`
segment — Module 10), construct from a channel ID instead and it pulls the
pointer out of the grid for you:

```cpp no-compile
nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex> ca(*g, /*channelID=*/0u);
if (!ca) { /* channel 0 absent or wrong type */ }
float v = ca.getValue(c);
```

Prefer `ChannelAccessor` over the manual two-step whenever the attribute is
a single scalar/vector type: it keeps the index lookup and the array fetch
in one typed call, and `probeValue(ijk, v)` works too (state + value).

### Baking a sidecar back into a value grid — `indexToGrid`

The split has a round-trip partner. When you want the values to live *in*
the grid again — e.g. to hand it to a tool that needs a real value type,
like `signedFloodFill` (`float` only) — `indexToGrid<DstBuildT>` walks the
index grid and writes `sidecar[voxelIndex]` into each voxel of a new
`NanoGrid<DstBuildT>`:

```cpp
#include <nanovdb/tools/cuda/IndexToGrid.cuh>

const auto* idxGrid = devHandle.deviceGrid<nanovdb::ValueOnIndex>();
const float* dValues = nullptr;   // your per-active-voxel sidecar

auto floatH = nanovdb::tools::cuda::indexToGrid<float>(idxGrid, dValues);
const auto* floatGrid = floatH.deviceGrid<float>();   // values now in-grid
```

So `ValueOnIndex` + sidecar and `NanoGrid<float>` are two views of the same
data: keep values external and index *in* (cheap to share one topology
across many attribute arrays, cheap to update), or bake them *out* into a
self-contained value grid (needed by value-typed tools). `indexToGrid` is
the bake-out direction.

### VoxelBlockManager

`VoxelBlockManager` (added July 2025; `nanovdb/tools/VoxelBlockManager.h`
and `nanovdb/tools/cuda/VoxelBlockManager.cuh`) is an acceleration
structure for **SIMT-parallel iteration over the active voxels of an
OnIndexGrid in fixed-size blocks**.

The use case: "I have N active voxels and I want to launch one warp per K
consecutive active voxels and have each thread pull out *its* voxel's
attribute, independent of how the voxels are scattered across leaves."

Components:

- **`VoxelBlockManagerHandle<BufferT>`** — owns metadata: a `firstLeafID`
  array (one per block) and a `jumpMap` (one bitmask per block).
- **`buildVoxelBlockManager<Log2BlockWidth>(grid)`** — constructs the
  metadata from a `NanoGrid<ValueOnIndex>` and returns a handle. The
  template arg is log2 of the active voxels per block and must be ≥ 6
  (≥ 64/block — one 64-bit jumpMap word per block).
- **`VoxelBlockManager`** — given a sequential active-voxel index, returns
  the leaf ID and the in-leaf voxel offset. One call per block, from a
  parallel loop.
- **`nanovdb::util::shuffleDownMask`** — the SIMD/SIMT primitive used in
  the decode (a candidate to graduate to `util/Algo.h`).

```cpp
#include <nanovdb/tools/VoxelBlockManager.h>

// Build VBM metadata from an OnIndexGrid (host pointer here; a CUDA variant
// lives in tools/cuda/VoxelBlockManager.cuh). Log2BlockWidth=6 → 64 active
// voxels per block (must be >= 6).
const auto* idxGrid = handle.grid<nanovdb::ValueOnIndex>();
auto vbmH = nanovdb::tools::buildVoxelBlockManager</*Log2BlockWidth=*/6>(idxGrid);
```

### A stencil kernel on the GPU with VBM

The point of VBM is balanced per-active-voxel work. Here is a real box-stencil
kernel: one CUDA block per VBM block, `BlockWidth` threads (one active voxel
each), gathering the 27-point neighborhood — the `ValueOnIndex` index of each
neighbor (0 = inactive) — for every active voxel. The two `VBM` statics do the
heavy lifting: `decodeInverseMaps` turns this block's sequential indices into
per-voxel `(leafIndex, in-leaf offset)`, and `computeBoxStencil` gathers the
neighbors from there. (This mirrors the `testVoxelBlockManager` unit test.)

```cpp global
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>

using OnIdx = nanovdb::ValueOnIndex;
static constexpr int Log2BlockWidth = 7;          // 2^7 = 128 active voxels/block
using VBM = nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>;

__global__ void boxStencilKernel(nanovdb::NanoGrid<OnIdx>* grid,
                                 uint32_t* firstLeafID,
                                 uint64_t* jumpMap,
                                 uint64_t* outNeighbors)  // [nBlocks*BlockWidth][27]
{
    constexpr int BW  = VBM::BlockWidth;
    constexpr int JML = VBM::JumpMapLength;
    const int bID = blockIdx.x;

    // Decode this block's BW active voxels (sequential indices starting at
    // 1 + bID*BW) into their physical (leaf index, in-leaf voxel offset).
    __shared__ uint32_t leafIndex[BW];
    __shared__ uint16_t voxelOffset[BW];
    VBM::decodeInverseMaps(grid, firstLeafID[bID], jumpMap + JML * bID,
                           1 + bID * BW, leafIndex, voxelOffset);

    // Gather the 27 neighbor indices for each voxel of the block.
    uint64_t neighbors[27] = {};
    VBM::computeBoxStencil(grid, leafIndex, voxelOffset, neighbors);
    __syncthreads();

    uint64_t* out = outNeighbors + 27 * BW * bID + 27 * threadIdx.x;
    for (int i = 0; i < 27; ++i) out[i] = neighbors[i];
}

// Host side: build the VBM, launch one block per VBM block.
static void launchBoxStencil()
{
    auto* grid = devHandle.deviceGrid<OnIdx>();
    auto vbmH  = nanovdb::tools::cuda::buildVoxelBlockManager<Log2BlockWidth>(grid);

    // Active-voxel count via the device tree data; one VBM block per BlockWidth.
    auto td = nanovdb::util::cuda::DeviceGridTraits<OnIdx>::getTreeData(grid);
    const std::size_t nBlocks =
        (td.mVoxelCount + VBM::BlockWidth - 1) >> Log2BlockWidth;

    uint64_t* outNeighbors = nullptr;  // device: nBlocks*BlockWidth*27 uint64
    boxStencilKernel<<<nBlocks, VBM::BlockWidth>>>(
        grid, vbmH.deviceFirstLeafID(), vbmH.deviceJumpMap(), outNeighbors);
}
```

Every block processes exactly `BlockWidth` active voxels regardless of how
they scatter across leaves — that's the balance VBM buys you. For the full
verified version (and the index-vs-accessor cross-check), see
`unittest/TestNanoVDB.cu` → `testVoxelBlockManager`.

### When to reach for VBM

- You're computing something per-active-voxel and you care about warp
  coherence (each warp processes K consecutive active voxels).
- The attribute lives in an external dense array (length = active voxel
  count) and you need fast index → (leaf, offset) decoding.
- You're not just doing `NodeManager` leaf-by-leaf — you want sub-leaf
  block granularity with predictable work per warp.

For "I just want to launch one thread per active voxel and don't care
about block locality", plain `NodeManager` is simpler.

### Quiz 9

> **Q1.** What's the difference between `ValueIndex` and `ValueOnIndex` as
> build types?
>
> **Q2.** A grid has 1M active voxels and you want to attach a 12-channel
> float attribute (48 bytes per voxel). Compare two storage strategies:
> (a) a `NanoGrid<float[12]>` (if it existed); (b) a `NanoGrid<ValueOnIndex>`
> + `float values[1M][12]`. Which makes it easier to update one channel
> without touching the others?
>
> **Q3.** When would you choose `VoxelBlockManager` over `NodeManager`?

*(Answer key at end of document.)*

### If you have the repo

- Read: `nanovdb/tools/CreateNanoGrid.h` for the `ValueOnIndex` /
  `ValueIndex` conversion path (how an IndexGrid gets built)
- Read: `nanovdb/tools/VoxelBlockManager.h` header banner
- Read: `nanovdb/tools/cuda/VoxelBlockManager.cuh`

---

## Module 10: OpenVDB ↔ NanoVDB conversion

### Core concept

NanoVDB doesn't include OpenVDB. The only file in NanoVDB that *can*
depend on OpenVDB is `tools/CreateNanoGrid.h`, gated behind
`NANOVDB_USE_OPENVDB`. Everything else compiles without OpenVDB present.

### OpenVDB → NanoVDB

```cpp no-compile
#include <openvdb/openvdb.h>
#include <nanovdb/tools/CreateNanoGrid.h>

openvdb::initialize();
openvdb::FloatGrid::Ptr ovGrid = ...;   // your OpenVDB grid

auto handle = nanovdb::tools::createNanoGrid<openvdb::FloatGrid, float>(*ovGrid);

// handle is a GridHandle<HostBuffer> with a NanoGrid<float> inside
nanovdb::io::writeGrid("out.nvdb", handle);
```

The two template parameters:

1. **Source OpenVDB grid type** (`openvdb::FloatGrid`, `Vec3fGrid`, ...)
2. **Destination NanoVDB build type** (`float`, `nanovdb::ValueOnIndex`,
   `nanovdb::Half`, ...)

This is also where type conversion happens — you can take an OpenVDB
`FloatGrid` and write a NanoVDB `Half` grid (16-bit float) to save space,
or an `OnIndexGrid` to set up for VBM-style attribute storage.

### NanoVDB → OpenVDB (round-trip / inspection)

```cpp openvdb
#include <nanovdb/tools/NanoToOpenVDB.h>

auto ovGrid = nanovdb::tools::nanoToOpenVDB<float>(handle);
// ovGrid is an openvdb::FloatGrid::Ptr
```

Useful for validation, for hybrid workflows (NanoVDB on GPU, OpenVDB on
CPU for offline simulation), or for debugging.

### BlindData — attaching arbitrary metadata

A grid can have one or more *blind data* segments — arbitrary byte arrays
addressed by offset from the grid header. NanoVDB doesn't interpret them;
they're for the application.

Common uses:

- Per-voxel point positions (for `Point` grids)
- Per-voxel attribute arrays (for IndexGrid use)
- Application-specific metadata (timestamps, source filenames, etc.)

#### Reading blind data (host or device)

Each segment is described by a `GridBlindMetaData` record. The grid exposes
`blindDataCount()`, `blindMetaData(i)` (the record), and the typed
`getBlindData<T>(i)` (the actual array — returns `nullptr` if `T` doesn't
match the stored type). Useful metadata fields: `mValueCount` (number of
values), `mValueSize` (bytes each), `mDataType`, `mSemantic`, `mDataClass`,
`mName`.

```cpp
// Inspect every blind-data segment on a grid.
for (uint32_t i = 0; i < grid->blindDataCount(); ++i) {
    const nanovdb::GridBlindMetaData& meta = grid->blindMetaData(i);
    // meta.mValueCount, meta.mValueSize, meta.mName, meta.mSemantic, ...
}

// Find a segment by semantic, then read it typed.
int id = grid->findBlindDataForSemantic(
    nanovdb::GridBlindDataSemantic::PointPosition);
if (id >= 0) {
    const nanovdb::Vec3f* positions = grid->getBlindData<nanovdb::Vec3f>(id);
    // positions[0 .. blindMetaData(id).mValueCount-1]
}
// Or by name: int id = grid->findBlindData("my_attr");
```

`GridBlindDataSemantic` enumerates known meanings (`PointPosition`,
`PointColor`, `PointNormal`, `PointVelocity`, `WorldCoords`, ...);
`GridBlindDataClass` describes the role (`AttributeArray`, `ChannelArray`,
`IndexArray`, ...). Reading is `__hostdev__`, so the same loop works in a
kernel.

#### Attaching blind data on the GPU — `AddBlindData.cuh`

Topology is frozen, but a grid's *buffer* can be reallocated with extra
bytes appended. `addBlindData` (in `tools/cuda/AddBlindData.cuh`) takes a
device grid plus a device array, and returns a **new** `GridHandle` whose
grid has the array appended as a blind-data segment (it also fixes up the
checksum for you):

```cpp
#include <nanovdb/tools/cuda/AddBlindData.cuh>

// Attach a per-active-voxel float channel to an existing device grid.
const float* dChannel = nullptr;       // device array, length = valueCount
uint64_t     valueCount = 0;

auto newHandle = nanovdb::tools::cuda::addBlindData(
    dGrid, dChannel, valueCount,
    nanovdb::GridBlindDataClass::ChannelArray,
    nanovdb::GridBlindDataSemantic::Unknown,
    "my_channel");
const auto* dGridWithData = newHandle.deviceGrid<float>();
```

This is exactly how you'd give an `IndexGrid` an *internal* channel that
`ChannelAccessor`'s channel-ID constructor (Module 9) can then read back —
attach the sidecar as `ChannelArray` blind data, then read it via
`ChannelAccessor(grid, channelID)`.

### Quiz 10

> **Q1.** You want to convert an `openvdb::FloatGrid` to a NanoVDB
> `Half` (16-bit float) to halve the file size. Write the conversion call.
>
> **Q2.** Why does NanoVDB compile cleanly without OpenVDB present?
>
> **Q3.** What is "blind data" and what does NanoVDB do with the bytes?

*(Answer key at end of document.)*

### If you have the repo

- Read: `nanovdb/tools/CreateNanoGrid.h` — main entry point with usage
  examples in the comment block
- Read: `nanovdb/tools/NanoToOpenVDB.h` — reverse direction
- Run: `nanovdb/examples/ex_openvdb_to_nanovdb/`

---

## Exercises

Three exercises bridge the gap between module quizzes and the capstone.
Present them in order. Based on how the student does, adjust the capstone
guidance: if they breeze through all three, give only the spec; if they
struggle with Exercise 3, offer skeleton code.

---

### Exercise 1: Read and print

Load a `.nvdb` file (have the student supply one, or use a primitive built
on the fly with `nanovdb::tools::createLevelSetSphere<float>(/*radius=*/10.0,
/*center=*/{0,0,0}, /*voxelSize=*/0.1)`). Print:

- Grid name and grid class
- Active voxel count
- Index-space bounding box
- World-space bounding box
- Background value
- Voxel value at the center of the index-space bbox (use a `ReadAccessor`)

*Tests:* `GridHandle`, `grid<T>()`, accessor basics.

---

### Exercise 2: Trilinear sampling at random world-space points

Load (or build) the same sphere grid. Generate 1,000 random world-space
points in the world-space bbox. Sample the grid at each point with
trilinear interpolation. Print the mean and standard deviation of the
sampled values.

*Tests:* `worldBBox()`, `worldToIndex()`, `createSampler<1>`, `ReadAccessor`.

---

### Exercise 3: HDDA ray-march

Pick a single ray from the world-space origin going through the center of
the sphere. Use HDDA to find the first active voxel along the ray. Print
the world-space hit position and the voxel value at the hit.

*Tests:* `Ray`, `HDDA`, accessor, world ↔ index transforms.

**After Exercise 3**, ask the student: *"How would you parallelize this
to render an image?"* This sets up the capstone.

---

## Capstone: GPU level-set ray-march renderer

The student should build a small CUDA program that:

1. Loads an SDF `.nvdb` file from disk (a level-set sphere or any other
   level set the student has handy).
2. Moves the grid to the GPU with
   `handle.copy<nanovdb::cuda::DeviceBuffer>()`.
3. For a `W × H` image, launches one thread per pixel.
4. Each thread constructs a primary ray, ray-marches the grid with HDDA
   until it hits the level-set zero crossing (sign change in the SDF
   value), and stores either:
   - the world-space hit depth (for a depth image), or
   - a Lambertian shaded color using the SDF gradient as the surface
     normal (gradients via `Stencils.h`).
5. Writes the image to disk as PPM (simplest format — three bytes per
   pixel after a small ASCII header).

### Instructor hints

If the student gets stuck, suggest in this order:

1. *"Get a single ray working on the CPU first."* Sequential ray-march,
   one pixel; print the hit. Then parallelize.
2. *"Use `nanovdb::tools::createLevelSetSphere` to build the SDF
   in-memory."* Skips the file-format hassle.
3. *"Cast the ray in world space, convert to index space inside the
   kernel."* The `Ray::worldToIndexF(grid)` method handles this.
4. *"For shading: surface normal = normalize(`grad(sdf)`) at the hit."*
   `GradStencil<>` from `Stencils.h`.

### What "done" looks like

A `512 × 512` rendering of a sphere with diffuse shading. Discuss
performance: how many rays per second? Where is the time going?

This caps the whole user lesson: file IO, host → device transfer, GPU
kernels, ray traversal, sampling, stencils.

---

## Quiz answer key

**Quiz 1**
- **Q1.** Upper internal: 32³ children; lower internal: 16³ children; leaf:
  8³ voxels. One upper covers 32 × 16 × 8 = 4096 voxels per axis.
- **Q2.** `ReadAccessor` caches the path from root → upper → lower → leaf
  for the most recent query. The next query at a nearby coordinate (same
  leaf, or same lower) hits the cache and skips most of the tree
  traversal. `tree().getValue(ijk)` always traverses from the root.
- **Q3.** Sequential samples along a ray almost always share a leaf or a
  lower-internal node with the previous sample, so the accessor's cache
  hit rate is very high. Creating a fresh accessor each call throws that
  cache away.

**Quiz 2**
- **Q1.** `nullptr`. `grid<float>()` returns null if the requested type
  doesn't match the actual grid's value type. Always check.
- **Q2.** `auto h = nanovdb::io::readGrid("file.nvdb", "velocity");`
- **Q3.** `readGrid(file, -1)` returns *one* `GridHandle` containing all
  grids packed into a single buffer. `readGrids(file)` returns a
  `std::vector<GridHandle>` with one handle per *segment* in the file (and
  each segment may itself contain multiple grids). For files written with
  `writeGrid`/`writeGrids` and a single segment, the two are nearly
  equivalent.

**Quiz 3**
- **Q1.** For each world-space point:
  1. Convert to index space via `grid->worldToIndex(p)`.
  2. Round/floor to a `Coord`.
  3. Call `acc.isActive(ijk)`.
  ```cpp
  auto acc = grid->getAccessor();
  int hits = 0;
  for (auto& w : pts) {
      auto ijk = nanovdb::Coord::Floor(grid->worldToIndex(w));
      if (acc.isActive(ijk)) ++hits;
  }
  ```
- **Q2.** The cached path is *mutated* on each lookup: when a query lands
  in a new leaf, the cache is updated to point at that leaf. Concurrent
  threads racing on those writes leads to torn pointers and incorrect
  cache state. The accessor only *reads* the underlying grid, but its
  *cache* is read-write per call.
- **Q3.** (1) The tree traversal finds **no covering node or tile** for
  `ijk` at any level — the root has no tile whose key covers the region,
  so descent terminates at the root and returns `mBackground`. The
  coordinate is "way outside" everything the grid stores. (2) The
  traversal *succeeds* and returns a value that **happens to equal
  background** — either a tile value (root / upper / lower) set to
  background by flood-fill or builder defaults, or an inactive leaf
  voxel whose stored value equals background. Both look identical to
  the caller — disambiguate with `isActive(ijk)` or `probeValue(ijk, v)`
  (which returns a bool + value).
  Special case: for `OnIndexGrid`, background is the sentinel "no entry
  in the external value array" (typically 0), so querying an inactive
  coordinate returns the sentinel by design — a planned instance of (2).

**Quiz 4**
- **Q1.** `NodeManager`. You want to iterate every leaf — that's exactly
  what NodeManager gives you a linear array of. `ReadAccessor` is for
  arbitrary-coordinate lookups.
- **Q2.** 1,000,000 active voxels. A typical kernel launch:
  `<<<(1000000 + 127) / 128, 128>>>` = 7,813 blocks of 128 threads.
- **Q3.** At least 1 (there must be at least one upper covering it),
  and exactly 1 if the active voxel's only ancestry path doesn't share
  uppers with anything else.

**Quiz 5**
- **Q1.** `__hostdev__` is a macro that expands to `__host__ __device__`
  when compiled by `nvcc`, and to nothing otherwise. It marks a function
  as callable from both CPU and CUDA device code. NanoVDB's math is used
  inside both CPU loops and `__global__` kernels, so every primitive is
  tagged this way.
- **Q2.** `worldToIndex(p)` = `p / voxelSize` (when origin is zero) =
  `(12.5, -7.0, 31.0)`. Floored: `(12, -7, 31)`.
- **Q3.** Naive ray-march samples every voxel along the ray. HDDA tests
  the current tree level for emptiness and *steps the ray to the next
  non-empty node at that level*. In a sparse grid, ray segments that pass
  through empty regions of upper-internal extent (4096 voxels per axis)
  skip ~4096 voxels in a single step.

**Quiz 6**
- **Q1.** Device kernels can only read memory in the device address
  space. A `NanoGrid<float>*` returned by `deviceGrid<>()` already points
  into device memory. A host `std::vector` lives in host memory; passing
  its `.data()` pointer to a kernel results in an illegal memory access.
- **Q2.** `auto devHandle = hostHandle.copy<nanovdb::cuda::DeviceBuffer>();`
- **Q3.** In registers (or, if the compiler spills, in local memory —
  which on NVIDIA is per-thread stack in global memory backed by L1/L2
  cache). The accessor object itself is constructed on the kernel's
  stack, so its cache fields live in the thread's stack frame, not in
  shared memory or globally.

**Quiz 7**
- **Q1.** OpenVDB grids are CPU-side mutable trees with allocations per
  node. Building one per frame, then converting to NanoVDB on the CPU,
  then copying to the GPU, costs at minimum a round-trip across the PCIe
  bus and several allocator passes. `pointsToGrid` does it in one GPU
  pipeline.
- **Q2.** A "build type" is the type parameter of `NanoGrid<T>` —
  determining what's stored per voxel. `ValueOnIndex` is sensible for a
  narrow band because: (a) the value at each active voxel is just an
  index into an external array, so you can attach SDF distances *and*
  other attributes (gradients, masks) to the same topology; (b) memory
  for the dense float array isn't allocated for inactive voxels.
- **Q3.** Yes — `GridBuilder`'s accessor is thread-local-friendly *if* you
  set up the threading yourself and each thread has its own accessor.
  But topology changes need synchronization; the canonical safe pattern
  is parallel-by-leaf, not parallel-by-voxel.

**Quiz 8**
- **Q1.** Coarsening merges 2³ source voxels into one output voxel — but
  what color should the merged voxel get? Average the eight? Pick one?
  Weight by occupancy? That's an *application* decision (and depends on
  what the channel means — averaging makes sense for color, not for a
  material ID). The operator only knows topology, not the semantics of
  your sidecar, so it correctly refuses to invent values and leaves the
  re-homing to you (via `Injection.cuh` + your own reduction over the
  merged voxels). Topology and data are separate; the operator owns only
  the first.
- **Q2.** A topology operator *changes which voxels exist*, so node counts
  at the leaf/lower/upper levels change → the contiguous buffer's offsets
  shift → the whole layout must be rebuilt (Module 7's offset
  invalidation). It can't edit in place; it returns a new grid.
  `signedFloodFill` *changes only values* (tile signs) — the active set,
  counts, and offsets are all unchanged, so it's a fixed-size overwrite of
  existing slots, which is safe to do in place.

**Quiz 9**
- **Q1.** `ValueIndex` assigns a sequential index to *every voxel slot*
  in every active leaf (active and inactive). `ValueOnIndex` assigns
  indices *only to active voxels*. For "I have one attribute per active
  voxel," `ValueOnIndex` is the natural choice — fewer indices, denser
  external array.
- **Q2.** (b) Strategy (b) is easier to update one channel: the external
  array has channels as the last dimension (or as separate arrays), so
  channel 7 is a contiguous slice. With (a), changing channel 7 means
  rewriting every leaf's per-voxel data, and the whole grid buffer is
  one allocation.
- **Q3.** When you want SIMT-friendly block-coherent iteration over
  active voxels — every warp processes K consecutive active voxels —
  and the attribute lives in an external dense array indexed by
  sequential active-voxel position. `NodeManager` gives you per-leaf
  iteration, which is coarser; VBM gives you per-block (smaller-than-leaf)
  granularity with predictable per-warp work.

**Quiz 10**
- **Q1.** `auto h = nanovdb::tools::createNanoGrid<openvdb::FloatGrid,`
  ` nanovdb::Half>(*ovGrid);`
- **Q2.** Because `NanoVDB.h` and the rest of the read path have no
  `#include <openvdb/...>`. Only `tools/CreateNanoGrid.h` and
  `tools/NanoToOpenVDB.h` reach into OpenVDB, and those are
  gated behind `NANOVDB_USE_OPENVDB`. The grid format itself is
  defined purely in NanoVDB's own headers.
- **Q3.** Blind data is a contiguous byte array attached to the grid
  buffer at a known offset, with associated metadata describing what
  it is. NanoVDB stores it but doesn't interpret the bytes — that's
  the application's job. Common use is per-voxel attributes for
  IndexGrid-style storage, or attached point positions for `Point`
  grids.

---

## Reference: Key APIs at a glance

| Task                                   | API                                                                   |
|----------------------------------------|-----------------------------------------------------------------------|
| Read grid from file                    | `nanovdb::io::readGrid(filename, n=0)`                                |
| Read all grids                         | `nanovdb::io::readGrid(filename, -1)` or `readGrids(filename)`        |
| Write grid to file                     | `nanovdb::io::writeGrid(filename, handle, codec)`                     |
| Number of grids in handle              | `handle.gridCount()`                                                  |
| Typed access to grid n                 | `handle.grid<ValueT>(n)` (nullptr if mismatch)                        |
| Active voxel count                     | `grid->activeVoxelCount()`                                            |
| Index-space bbox                       | `grid->indexBBox()`                                                   |
| World-space bbox                       | `grid->worldBBox()`                                                   |
| Voxel size                             | `grid->voxelSize()`                                                   |
| World → index                          | `grid->worldToIndex(worldPt)`                                         |
| Index → world                          | `grid->indexToWorld(idxPt)`                                           |
| Random access                          | `grid->getAccessor()`, `acc.getValue(ijk)`, `acc.isActive(ijk)`       |
| Value + state in one descent           | `acc.probeValue(ijk, v)` (bool), `acc.probeLeaf(ijk)` (nullptr if none)|
| Sequential access (CPU)                | `auto h = createNodeManager(*grid); h.mgr<T>()->leaf(i)`              |
| Sequential access (GPU)                | `nanovdb::cuda::createNodeManager(dGrid)`                             |
| Sampling                               | `auto s = nanovdb::math::createSampler<order>(acc); s(idxPt)`         |
| Ray-march                              | `nanovdb::math::HDDA<RayT, Coord> hdda(ray); while (hdda.step()) ...` |
| Move handle to GPU                     | `handle.copy<nanovdb::cuda::DeviceBuffer>()`                          |
| Device grid pointer                    | `devHandle.deviceGrid<T>()`                                           |
| Points → grid (GPU)                    | `nanovdb::tools::cuda::pointsToGrid(dPts, n, voxelSize)`              |
| Mesh → grid (GPU)                      | `nanovdb::tools::cuda::MeshToGrid<ValueOnIndex>(dV,vN,dT,tN,Map(vs)).getHandle()`     |
| Build grid (CPU)                       | `nanovdb::tools::build::Grid<T> g(bg); g.getAccessor().setValue(...)`     |
| Level-set sphere primitive             | `nanovdb::tools::createLevelSetSphere<T>(radius, center, voxelSize)`  |
| Dilate                                 | `nanovdb::tools::cuda::DilateGrid<ValueOnIndex>(dGrid).getHandle()`                 |
| Coarsen                                | `nanovdb::tools::cuda::CoarsenGrid<ValueOnIndex>(dGrid).getHandle()`                        |
| Refine                                 | `nanovdb::tools::cuda::RefineGrid<ValueOnIndex>(dGrid).getHandle()`                         |
| Prune                                  | `nanovdb::tools::cuda::PruneGrid<ValueOnIndex>(dGrid, leafMask).getHandle()`                          |
| Signed flood fill                      | `nanovdb::tools::cuda::signedFloodFill(devHandle.deviceGrid<float>())`                    |
| Recompute node stats (after edit)      | `nanovdb::tools::updateGridStats(mutableGrid, StatsMode::All)`        |
| Extrema over a box                     | `nanovdb::tools::getExtrema(*grid, bbox)` → `.min()` / `.max()`       |
| Validate checksum                      | `nanovdb::tools::validateChecksum(grid, CheckMode::Full)`            |
| Refresh checksum (after edit)          | `nanovdb::tools::updateChecksum(mutableGrid, CheckMode::Full)`       |
| Validate grid                          | `nanovdb::tools::isValid(grid, CheckMode::Full, verbose)`            |
| Read sidecar via ChannelAccessor       | `nanovdb::ChannelAccessor<T, ValueOnIndex>(*g, arr); ca.getValue(c)`  |
| Blind-data count / record              | `grid->blindDataCount()`, `grid->blindMetaData(i)`                    |
| Typed blind-data array                 | `grid->getBlindData<T>(i)` (nullptr if T mismatch)                    |
| Attach blind data (GPU)                | `nanovdb::tools::cuda::addBlindData(dGrid, dArr, count, class, sem)`  |
| Build VBM                              | `nanovdb::tools::buildVoxelBlockManager<Log2BlockWidth>(idxGrid)`                       |
| OpenVDB → NanoVDB                      | `nanovdb::tools::createNanoGrid<SrcGrid, DstBuildT>(srcGrid)`         |
| NanoVDB → OpenVDB                      | `nanovdb::tools::nanoToOpenVDB<ValueT>(handle)`                       |

---

*This lesson lives with the NanoVDB docs, at
`doc/nanovdb/TEACHME/nanovdb_user_lesson.md`. Pair it with
`nanovdb_user_cheatsheet.md` for a quick reference while working.*
