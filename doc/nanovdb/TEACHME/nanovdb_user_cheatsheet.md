# NanoVDB User Cheat Sheet

A one-page reference for the APIs covered in `nanovdb_user_lesson.md`. Keep
this open while working through exercises.

---

## In one paragraph

NanoVDB is a single-header, read-only, GPU-friendly sparse 3D data
structure. The whole grid (one or more) lives in one contiguous,
32-byte-aligned byte buffer. `cudaMemcpy` it to the device and it works
as-is. Topology is *frozen* after construction; values can change, the set
of active voxels cannot. Every primitive is `__hostdev__` (callable from
both host and device with identical code).

---

## Includes & CMake

```cpp openvdb
#include <nanovdb/NanoVDB.h>               // grid, accessor, math
#include <nanovdb/GridHandle.h>            // owns the buffer
#include <nanovdb/HostBuffer.h>            // CPU allocator
#include <nanovdb/NodeManager.h>           // sequential iteration
#include <nanovdb/io/IO.h>                 // file IO
#include <nanovdb/math/HDDA.h>             // ray-march
#include <nanovdb/math/SampleFromVoxels.h> // sampling

// GPU
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/cuda/NodeManager.cuh>

// Builders & operators (GPU)
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/cuda/CoarsenGrid.cuh>
#include <nanovdb/tools/cuda/RefineGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/tools/cuda/SignedFloodFill.cuh>

// CPU builders & primitives
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreatePrimitives.h>

// IndexGrid / VBM
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

// Stats, validation, blind data
#include <nanovdb/tools/GridStats.h>
#include <nanovdb/tools/GridChecksum.h>
#include <nanovdb/tools/GridValidator.h>
#include <nanovdb/tools/cuda/AddBlindData.cuh>

// OpenVDB bridge (gated behind NANOVDB_USE_OPENVDB)
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/NanoToOpenVDB.h>
```

```cmake
find_package(nanovdb CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE nanovdb::nanovdb)
```

---

## Build types (`NanoGrid<T>`)

| `T`               | Stored per voxel                          |
|-------------------|-------------------------------------------|
| `float`, `double` | Scalar (SDF, density)                     |
| `int32_t`, `int64_t` | Integer                                |
| `Vec3f`, `Vec3d`  | 3-vector (velocity, color)                |
| `bool`            | Active state only                         |
| `ValueIndex`      | Sequential index over all voxels          |
| `ValueOnIndex`    | Sequential index over **active** voxels   |
| `ValueMask`       | Topology only (no per-voxel values)       |
| `Half`            | 16-bit float                              |
| `Fp4`/`Fp8`/`Fp16`/`FpN` | Quantized float (4/8/16 bits)      |
| `Point`           | Point counts (for point grids)            |

---

## File IO

```cpp
namespace io = nanovdb::io;

auto h  = io::readGrid("file.nvdb");                 // first grid
auto h2 = io::readGrid("file.nvdb", 2);              // grid #2 (0-based)
auto hN = io::readGrid("file.nvdb", "density");      // by name
auto hAll = io::readGrid("file.nvdb", -1);           // all → one handle
auto v  = io::readGrids("file.nvdb");                // vector<handle>, one per segment

io::writeGrid("out.nvdb", h);                        // single
io::writeGrid("out.nvdb", h, io::Codec::BLOSC);      // compressed
io::writeGrids("out.nvdb", handles);                 // many
```

---

## Inspecting a grid

```cpp
const auto* grid = handle.grid<float>();             // nullptr if T mismatch
if (!grid) throw std::runtime_error("type mismatch");

grid->gridName();           // const char*
grid->gridClass();          // GridClass enum (LevelSet, FogVolume, etc.)
grid->gridType();           // GridType enum (Float, Vec3f, OnIndex, ...)
grid->voxelSize();          // Vec3d
grid->activeVoxelCount();   // uint64_t
grid->indexBBox();          // CoordBBox
grid->worldBBox();          // BBox<Vec3d>
grid->tree().background();  // T
grid->worldToIndex(p);      // Vec3<T>, p is world-space
grid->indexToWorld(p);      // Vec3<T>, p is index-space (fractional)
```

---

## Random access (ReadAccessor)

```cpp
// One accessor per thread; not thread-safe; never copy between CPU & GPU
auto acc = grid->getAccessor();
float v   = acc.getValue(nanovdb::Coord(1, 2, 3));
bool  on  = acc.isActive(nanovdb::Coord(1, 2, 3));
const auto* leaf = acc.getNode<nanovdb::NanoLeaf<float>>();  // cached leaf ptr, if needed

// probe* = value/state/leaf-existence in ONE descent (disambiguates the
// two "got background" cases: inactive-vs-no-node)
float pv;
bool active = acc.probeValue(nanovdb::Coord(1,2,3), pv);     // pv set; ret = state
const nanovdb::NanoLeaf<float>* lf = acc.probeLeaf(nanovdb::Coord(1,2,3)); // nullptr if no leaf
```

---

## Sequential access (NodeManager)

```cpp
#include <nanovdb/NodeManager.h>

auto nmh = nanovdb::createNodeManager(*grid);
const auto* mgr = nmh.mgr<float>();

auto leafN  = mgr->leafCount();
auto lowerN = mgr->lowerCount();
auto upperN = mgr->upperCount();

for (uint64_t i = 0; i < leafN; ++i) {
    const auto& leaf = mgr->leaf(i);
    for (auto it = leaf.beginValueOn(); it; ++it) {
        float v = *it;
        auto c = it.getCoord();
    }
}
```

GPU. Pass the device-side `NodeManager*` to the kernel and index leaves by
block, voxels within a leaf by thread:

```cpp no-compile
#include <nanovdb/cuda/NodeManager.cuh>

auto dnmh = nanovdb::cuda::createNodeManager<float>(dGrid);
const auto* dmgr = dnmh.template deviceMgr<float>();   // device pointer

// One block per leaf; 64 threads cover the 512 voxels in each leaf.
perLeafKernel<<<dmgr->leafCount(), 64>>>(dmgr);   // kernel defined just below
```

The kernel:

```cpp
__global__ void perLeafKernel(const nanovdb::NodeManager<float>* mgr)
{
    using LeafT = nanovdb::NanoLeaf<float>;
    const uint32_t leafIdx = blockIdx.x;
    if (leafIdx >= mgr->leafCount()) return;

    const LeafT& leaf = mgr->leaf(leafIdx);

    // 512 voxels per leaf (LeafT::SIZE == 1<<3*LOG2DIM == 8^3).
    for (uint32_t n = threadIdx.x; n < LeafT::SIZE; n += blockDim.x) {
        if (!leaf.isActive(n)) continue;
        float          v = leaf.getValue(n);
        nanovdb::Coord c = leaf.offsetToGlobalCoord(n);
        // ... do something with (v, c)
    }
}
```

Same methods exist on `lower(i)` and `upper(i)` (internal nodes) — you'll
mostly want `leaf(i)` since that's where actual voxel values live.

Indexed leaf-voxel API (callable from `__hostdev__` code):

| Call                          | Returns                                   |
|-------------------------------|-------------------------------------------|
| `leaf.getValue(n)`            | voxel value at offset `n` (0..511)        |
| `leaf.isActive(n)`            | active state at offset `n`                |
| `leaf.offsetToGlobalCoord(n)` | the global `Coord` for offset `n`         |
| `leaf.origin()`               | the leaf's `(8,8,8)`-aligned origin Coord |
| `LeafT::SIZE`                 | `512`, voxels per leaf                    |

---

## Math primitives (all `__hostdev__`)

```cpp
nanovdb::Coord c(1, 2, 3);
auto c2 = nanovdb::Coord::Floor(nanovdb::Vec3f(1.7f, 2.1f, 3.9f)); // (1, 2, 3)

nanovdb::Vec3f v(1, 2, 3);
auto d = v.dot(v);            // 14
auto n = v.length();          // sqrt(14)
auto u = v.normalize();

nanovdb::math::BBox<nanovdb::Vec3f> b(nanovdb::Vec3f(0,0,0),
                                      nanovdb::Vec3f(10,10,10));
b.expand(nanovdb::Vec3f(5,5,5));
b.isInside(nanovdb::Vec3f(1,1,1));
```

---

## Sampling

```cpp
#include <nanovdb/math/SampleFromVoxels.h>

auto acc = grid->getAccessor();
auto sampler = nanovdb::math::createSampler<1>(acc);  // 0=nearest, 1=trilinear, 2=triquadratic, 3=tricubic
auto idxPt = grid->worldToIndex(worldPt);
float v = sampler(idxPt);
```

---

## HDDA ray-march

```cpp
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>

auto acc = grid->getAccessor();
nanovdb::math::Ray<float> wRay({0,0,-5}, {0,0,1});
auto iRay = wRay.worldToIndexF(*grid);

// High-level: ZeroCrossing drives an HDDA and stops at the first sign change.
nanovdb::Coord ijk;
float v, t;
if (nanovdb::math::ZeroCrossing(iRay, acc, ijk, v, t)) { /* hit at ijk */ }

// Low-level: HDDA ctor takes the ray AND the start dimension from getDim.
nanovdb::Coord start(0, 0, 0);
nanovdb::math::HDDA<decltype(iRay), nanovdb::Coord> hdda(iRay, acc.getDim(start, iRay));
while (hdda.step()) {
    if (acc.isActive(hdda.voxel())) break;   // first active voxel
}
```

---

## GPU: transfer + kernel

```cpp
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/GridHandle.cuh>

auto devH = hostH.copy<nanovdb::cuda::DeviceBuffer>();
devH.deviceUpload();                       // REQUIRED: push to device, else deviceGrid()==nullptr
const auto* dGrid = devH.deviceGrid<float>();

__global__ void k(const nanovdb::NanoGrid<float>* g,
                  const nanovdb::Coord* coords, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    auto acc = g->getAccessor();
    out[tid] = acc.getValue(coords[tid]);
}

k<<<(N+127)/128, 128>>>(dGrid, dCoords, dOut, N);
```

---

## GPU builders

```cpp
// Points → grid (builder-pattern class, not a free function)
nanovdb::tools::cuda::PointsToGrid<nanovdb::Point> p2g(/*voxelSize=*/0.05);
auto ptsH = p2g.getHandle(dPoints, nPts);
const auto* ptsGrid = ptsH.deviceGrid<nanovdb::Point>();

// Mesh → OnIndexGrid (narrow-band). Verts are Vec3f, triangles Vec3i.
const nanovdb::Vec3f* dVerts; size_t vertN;
const nanovdb::Vec3i* dTris;  size_t triN;
nanovdb::tools::cuda::MeshToGrid<nanovdb::ValueOnIndex> m2g(
    dVerts, vertN, dTris, triN, nanovdb::Map(voxelSize));
m2g.setNarrowBandWidth(3.f);
auto meshH = m2g.getHandle();
const auto* meshGrid = meshH.deviceGrid<nanovdb::ValueOnIndex>();

// MeshToGrid + per-active-voxel UDF sidecar
auto [gridH, udfBuf] = m2g.getHandleAndUDF();
```

## CPU builder

```cpp
// Build a tree on the host, then bake into a packed NanoVDB grid.
nanovdb::tools::build::Grid<float> g(/*background=*/0.0f,
                                     "my_grid", nanovdb::GridClass::FogVolume);
g.setTransform(/*scale=*/0.1);                   // voxel size
auto acc = g.getAccessor();                      // not thread-safe
acc.setValue(nanovdb::Coord(0, 0, 0), 1.0f);
auto h = nanovdb::tools::createNanoGrid(g);      // GridHandle<HostBuffer>
```

## Primitive helpers

```cpp
auto h = nanovdb::tools::createLevelSetSphere<float>(
    /*radius=*/10.0, /*center=*/{0,0,0}, /*voxelSize=*/0.1);
// also: createLevelSetTorus, createLevelSetBox, createFogVolumeSphere, ...
```

---

## Topological operators (GPU)

```cpp
// Builder classes (NOT free functions); ValueOnIndex grids only. Construct
// from a device grid pointer, getHandle() returns a new GridHandle.
using OnIdx = nanovdb::ValueOnIndex;
auto* g = devH.deviceGrid<OnIdx>();

auto dilatedH = nanovdb::tools::cuda::DilateGrid<OnIdx>(g).getHandle();
auto coarseH  = nanovdb::tools::cuda::CoarsenGrid<OnIdx>(g).getHandle();
auto fineH    = nanovdb::tools::cuda::RefineGrid<OnIdx>(g).getHandle();
const nanovdb::Mask<3>* leafMask = nullptr;   // PruneGrid needs a per-leaf mask
auto prunedH  = nanovdb::tools::cuda::PruneGrid<OnIdx>(g, leafMask).getHandle();
auto mergedH  = nanovdb::tools::cuda::MergeGrids<OnIdx>(g, g).getHandle();

// signedFloodFill is the exception: a free function, in place, on a scalar
// SDF grid (not OnIndex).
nanovdb::tools::cuda::signedFloodFill(devH.deviceGrid<float>());
```

CPU morphology equivalents live under `nanovdb/util/MorphologyHelpers.h`.

## Re-home sidecar data after a topology change (Injection.cuh)

A topology op renumbers the active-voxel indices, so the old sidecar no
longer matches. Move values across, and find the new voxels:

```cpp
#include <nanovdb/util/cuda/Injection.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Util.h>

using OnIdx = nanovdb::ValueOnIndex;
const auto* srcGrid = devH.deviceGrid<OnIdx>();      // old grid
const auto* dstGrid = devHandle.deviceGrid<OnIdx>(); // new (e.g. dilated) grid
const float* oldSidecar = nullptr;
float*       newSidecar = nullptr;                   // size = new activeVoxelCount(+1)

// Copy overlapping values (non-overlap dst voxels left untouched):
auto srcLeaves = nanovdb::util::cuda::DeviceGridTraits<OnIdx>::getTreeData(srcGrid).mNodeCount[0];
nanovdb::util::cuda::operatorKernel<
    nanovdb::util::cuda::InjectGridDataFunctor<OnIdx, float>>
    <<<srcLeaves, 256>>>(srcGrid, dstGrid, oldSidecar, newSidecar);

// Per-leaf overlap mask (new voxels = dstMask & ~overlap):
auto dstLeaves = nanovdb::util::cuda::DeviceGridTraits<OnIdx>::getTreeData(dstGrid).mNodeCount[0];
auto maskBuf = nanovdb::cuda::DeviceBuffer::create(dstLeaves*sizeof(nanovdb::Mask<3>), nullptr, false);
auto* overlap = static_cast<nanovdb::Mask<3>*>(maskBuf.deviceData());
unsigned nt = 128, nb = (dstLeaves + nt - 1) / nt;
nanovdb::util::cuda::lambdaKernel<<<nb, nt>>>(dstLeaves,
    nanovdb::util::cuda::InjectGridMaskFunctor<OnIdx>(), srcGrid, dstGrid, overlap);
```

Simpler: pre-fill `newSidecar` with a sentinel; slots still holding it after
the data inject are the new voxels.

---

## Stats, validation & integrity

Nodes cache bbox/min/max/avg per node. Mutating values in place leaves them
**stale** — which silently breaks HDDA space-skipping and ray-tracing.
Recompute after any manual edit.

```cpp
#include <nanovdb/tools/GridStats.h>          // CPU
// #include <nanovdb/tools/cuda/GridStats.cuh> // GPU (device grid + stream)

nanovdb::NanoGrid<float>* mutableGrid = nullptr;   // non-const: it gets written
nanovdb::tools::updateGridStats(mutableGrid, nanovdb::tools::StatsMode::All);
// StatsMode: Disable | BBox | MinMax | All(=Default)

// Extrema over an index-space box, computed directly (no cache):
auto ex = nanovdb::tools::getExtrema(*grid, nanovdb::CoordBBox(nanovdb::Coord(-10), nanovdb::Coord(10)));
float lo = ex.min(), hi = ex.max();
```

Certify a grid after load / build / device transfer:

```cpp
#include <nanovdb/tools/GridChecksum.h>
#include <nanovdb/tools/GridValidator.h>

nanovdb::NanoGrid<float>* mutableGrid = nullptr;   // non-const: edited grid

bool good = nanovdb::tools::validateChecksum(grid, nanovdb::CheckMode::Full);
nanovdb::tools::updateChecksum(mutableGrid, nanovdb::CheckMode::Full); // after a legit edit

bool valid = nanovdb::tools::isValid(grid, nanovdb::CheckMode::Full, /*verbose=*/false);
char err[(uint32_t)nanovdb::CheckMode::StrLen] = {};   // CheckMode is scoped → cast
nanovdb::tools::checkGrid(grid, err, nanovdb::CheckMode::Full);   // err="" if valid
// CheckMode: Disable/Empty | Half/Partial/Default | Full
// GPU variants: tools/cuda/GridChecksum.cuh, tools/cuda/GridValidator.cuh
```

---

## IndexGrid & VBM

```cpp no-compile
// Build a ValueOnIndex grid from an OpenVDB grid
auto h = nanovdb::tools::createNanoGrid<openvdb::FloatGrid,
                                        nanovdb::ValueOnIndex>(*ovGrid);

// Or from a mesh on the GPU (already returns OnIndexGrid)
nanovdb::tools::cuda::MeshToGrid<nanovdb::ValueOnIndex> mesher(dV, vN, dT, tN, nanovdb::Map(vs));
auto devH = mesher.getHandle();
const auto* g = devH.deviceGrid<nanovdb::ValueOnIndex>();

// External attribute array — same length as active voxel count
size_t N = g->activeVoxelCount();
thrust::device_vector<float> attr(N);

// Read attribute at a coordinate
auto acc = g->getAccessor();
uint64_t idx = acc.getValue(c);       // index into attr
float a = attr[idx];

// VBM: SIMT-friendly block iteration (Log2BlockWidth >= 6, i.e. >=64/block)
auto vbmH = nanovdb::tools::buildVoxelBlockManager<6>(g);
// see tools/cuda/VoxelBlockManager.cuh for the GPU decode pattern
```

ChannelAccessor — read the sidecar by attribute, not by index (one typed
call instead of getValue(ijk) → array[idx]). Note: IndexT defaults to
`ValueIndex`, so spell out `ValueOnIndex`:

```cpp
__global__ void readChan(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* g,
                         float* dValues, nanovdb::Coord c, float* out) {
    nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex> ca(*g, dValues);
    *out = ca.getValue(c);          // attribute value (not the index)
    // ca(c) same; ca.getIndex(c) for the raw index; ca.probeValue(c, v) for state+value
    // ctor (*g, channelID) instead reads an internal blind-data channel
}
```

Bake a sidecar back into a value grid (round-trip partner of the split;
needed by value-typed tools like `signedFloodFill`):

```cpp
#include <nanovdb/tools/cuda/IndexToGrid.cuh>

const auto* idxGrid = devH.deviceGrid<nanovdb::ValueOnIndex>();
const float* dValues = nullptr;                       // per-active-voxel sidecar
auto floatH = nanovdb::tools::cuda::indexToGrid<float>(idxGrid, dValues);
const auto* floatGrid = floatH.deviceGrid<float>();   // values now in-grid
```

---

## OpenVDB ↔ NanoVDB

```cpp openvdb
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/NanoToOpenVDB.h>

// OpenVDB → NanoVDB (optionally with type change)
auto h = nanovdb::tools::createNanoGrid<openvdb::FloatGrid, float>(*ovGrid);
auto h_half = nanovdb::tools::createNanoGrid<openvdb::FloatGrid,
                                             nanovdb::Half>(*ovGrid);

// NanoVDB → OpenVDB
auto ovGrid = nanovdb::tools::nanoToOpenVDB<float>(handle);
```

---

## Blind data (attached arrays)

Read (host or device — `__hostdev__`):

```cpp
uint32_t n = grid->blindDataCount();
const nanovdb::GridBlindMetaData& meta = grid->blindMetaData(0);
// meta.mValueCount, mValueSize, mDataType, mSemantic, mDataClass, mName

int id = grid->findBlindDataForSemantic(nanovdb::GridBlindDataSemantic::PointPosition);
// or: int id = grid->findBlindData("name");
const nanovdb::Vec3f* pos = grid->getBlindData<nanovdb::Vec3f>(id);  // nullptr if T mismatch
```

Attach on the GPU (returns a NEW handle with the array appended; fixes the
checksum):

```cpp
#include <nanovdb/tools/cuda/AddBlindData.cuh>

const float* dChannel = nullptr; uint64_t valueCount = 0;
auto h = nanovdb::tools::cuda::addBlindData(
    dGrid, dChannel, valueCount,
    nanovdb::GridBlindDataClass::ChannelArray,
    nanovdb::GridBlindDataSemantic::Unknown, "my_channel");
const auto* g = h.deviceGrid<float>();
```

---

## Common idioms

```cpp
// Iterate active voxels in a grid (CPU, leaf-by-leaf)
auto nmh = nanovdb::createNodeManager(*grid);
auto* mgr = nmh.mgr<float>();
for (uint64_t L = 0; L < mgr->leafCount(); ++L) {
    const auto& leaf = mgr->leaf(L);
    for (auto it = leaf.beginValueOn(); it; ++it) {
        // *it is the value, it.getCoord() is the ijk
    }
}

// Sample a grid at a list of world-space points (pts is a std::vector<Vec3f>)
auto sampler = nanovdb::math::createSampler<1>(acc);
std::vector<float> sampled;
for (const auto& w : pts)
    sampled.push_back(sampler(grid->worldToIndex(w)));

// Save a level-set sphere to disk
auto h = nanovdb::tools::createLevelSetSphere<float>(10.0, {0,0,0}, 0.1);
nanovdb::io::writeGrid("sphere.nvdb", h);
```

---

## Pitfalls

- **Type mismatch returns nullptr.** Always check `handle.grid<T>()`.
- **Accessors are not thread-safe.** One per thread/warp.
- **Accessors don't cross CPU/GPU.** Build inside the kernel.
- **Topology is frozen.** To "add a voxel" you must rebuild the grid.
- **32-byte alignment.** Don't allocate the buffer yourself with
  `new char[N]` — use `HostBuffer` or `DeviceBuffer`, which guarantee
  alignment.
- **No runtime `std::` in `__hostdev__` code.** Containers, iostreams,
  allocators, `std::sort` etc. won't work on device. For type traits and
  compile-time helpers, NanoVDB uses its own `nanovdb::util::is_same` /
  `util::enable_if` / `util::conditional` / `util::remove_const` (in
  `util/Util.h`) — these compile under C++11 across compilers and don't
  need `--expt-relaxed-constexpr`. Where NanoVDB needs more, it pulls from
  CCCL's `cuda::std::` (e.g. `cuda::std::numeric_limits` in `math/Math.h`).
  Plain `std::` compile-time facilities work with relaxed constexpr but
  NanoVDB's own headers avoid them for portability.
- **HDDA is essential for sparse ray-march.** Without it, you're stepping
  one voxel at a time through empty space.

---

## Tools / binaries

| Tool                | Purpose                                   |
|---------------------|-------------------------------------------|
| `nanovdb_print`     | Print metadata for grids in a `.nvdb`     |
| `nanovdb_convert`   | OpenVDB ↔ NanoVDB on the CLI              |
| `nanovdb_validate`  | ABI / format validation                   |

Run with `--help` for options.
