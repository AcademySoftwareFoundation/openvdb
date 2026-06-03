// fixture.cuh — common headers + stand-in "given" symbols for TEACHME
// code-block compilation.
//
// Lesson fragments assume context: "you have a grid", "you have device
// points", etc. The harness compiles each block (compile-only, never linked)
// with this header in scope, so those assumed symbols need declarations with
// the right *types* — enough for the compiler to type-check member calls and
// catch API drift. They are declared `extern` (no definitions): a block that
// declares its own local `grid`/`acc`/etc. simply shadows the extern, and
// since we never link, the unresolved externs cost nothing.
//
// Grow this file as the tagging pass surfaces more assumed names. Keep the
// types matching how the lesson uses each symbol.

#pragma once

// ---- common headers (header guards make re-includes in blocks harmless) ----
// Core read path + IO + math + CUDA.
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NodeManager.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/math/Stencils.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/cuda/NodeManager.cuh>
// Commonly-assumed "tools" headers — ambient so terse fragments compile
// without restating the include. (OpenVDB-conversion headers are NOT here;
// blocks that need them are tagged `openvdb` and include them explicitly.)
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/cuda/CoarsenGrid.cuh>
#include <nanovdb/tools/cuda/RefineGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/tools/cuda/SignedFloodFill.cuh>

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

// ---- stand-in "given" symbols (extern: declared, never defined) ----
// Handles
extern nanovdb::GridHandle<nanovdb::HostBuffer>        handle;
extern nanovdb::GridHandle<nanovdb::HostBuffer>        hostH;
extern nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> devH;
extern nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> devHandle;
extern std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> handles;

// Grids (host + device pointers into a float grid)
extern const nanovdb::NanoGrid<float>* grid;
extern const nanovdb::NanoGrid<float>* dGrid;

// A read accessor + sampler over the float grid (fragments often assume these
// already exist from a prior step).
extern nanovdb::DefaultReadAccessor<float> acc;

// Device point / coord / output buffers used by kernel launches.
// dPoints is non-const: PointsToGrid<Point> writes input points into the
// grid's blind data via getPoint<element_type>(...) = ..., which requires a
// non-const element type.
extern nanovdb::Vec3f*       dPoints;
extern std::size_t           nPts;
extern const nanovdb::Vec3f* dPts;
extern const nanovdb::Vec3f* worldPts;
extern const nanovdb::Coord* dCoords;
extern const nanovdb::Coord* coords;
extern float*                dOut;
extern float*                out;
extern int                   N;

// Host-side query containers + single query points
extern std::vector<nanovdb::Vec3f> pts;
extern std::vector<nanovdb::Coord> queries;
extern nanovdb::Vec3f worldPt;   // a single world-space query point
extern nanovdb::Vec3f p;         // generic world-space point

// Misc scalars
extern double voxelSize;
