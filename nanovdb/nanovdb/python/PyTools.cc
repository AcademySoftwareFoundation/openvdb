// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyTools.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>
#ifdef NANOVDB_USE_CUDA
#include <nanovdb/cuda/DeviceBuffer.h>
#endif

#include "PyGridStats.h"
#include "PyCreateNanoGrid.h"
#include "PyPrimitives.h"
#include "PyGridChecksum.h"
#include "PyGridValidator.h"
#include "PyNanoToOpenVDB.h"
#include "PyVoxelBlockManager.h"  // for defineDeviceVoxelBlockManager (CUDA)
#ifdef NANOVDB_USE_CUDA
#include "cuda/PyPointsToGrid.h"
#include "cuda/PySampleFromVoxels.h"
#include "cuda/PySignedFloodFill.h"
#endif

namespace nb = nanobind;
using namespace nanovdb;

namespace pynanovdb {

void defineToolsModule(nb::module_& m)
{
    defineUpdateChecksum(m);
    defineValidateGrids<HostBuffer>(m);

    defineStatsMode(m);

    defineGridStatsModule(m);
    defineGridValidatorModule(m);
    defineEvalChecksumModule(m);

    defineCreateNanoGridConversions(m);

    definePrimitives<HostBuffer>(m);

#define NANOVDB_PY_FOR_EACH_SAMPLEABLE_BUILDT(T, Suffix) \
    defineCreateNanoGrid<T>(m, "create" #Suffix "Grid");
#include "BuildTypes.def"

#ifdef NANOVDB_USE_OPENVDB
    defineOpenToNanoVDB<HostBuffer>(m);
    defineNanoToOpenVDB<HostBuffer>(m);
#endif

#ifdef NANOVDB_USE_CUDA
    nb::module_ cudaModule = m.def_submodule("cuda");
    cudaModule.doc() = "A submodule that implements CUDA-accelerated tools";

    defineValidateGrids<cuda::DeviceBuffer>(m);

    definePrimitives<cuda::DeviceBuffer>(cudaModule);

    defineSignedFloodFill<float>(cudaModule, "signedFloodFill");
    defineSignedFloodFill<double>(cudaModule, "signedFloodFill");

    // Coordinate-input (index-space int32 (N,3)) -> grid. The legacy Rgba8
    // entry keeps its original Python name; OnIndex/Index get descriptive
    // names. (Point is excluded here -- see PyPointsToGrid.cu -- and is built
    // from world positions via pointsToGrid below.)
    defineVoxelsToGrid<math::Rgba8>(cudaModule, "pointsToRGBA8Grid");
    defineVoxelsToGrid<math::Rgba8>(cudaModule, "voxelsToRGBA8Grid");
    defineVoxelsToGrid<ValueOnIndex>(cudaModule, "voxelsToOnIndexGrid");
    defineVoxelsToGrid<ValueIndex>(cudaModule, "voxelsToIndexGrid");

    // World-position-input ((N,3) float OR double) -> NanoGrid<Point>. Both
    // scalar precisions are bound under the same Python name; nanobind picks
    // the overload that matches the input tensor dtype.
    definePointsToGrid<float>(cudaModule, "pointsToGrid");
    definePointsToGrid<double>(cudaModule, "pointsToGrid");

    defineSampleFromVoxels<float>(cudaModule, "sampleFromVoxels");
    defineSampleFromVoxels<double>(cudaModule, "sampleFromVoxels");

    // Device VoxelBlockManager (nanovdb::tools::cuda) on nanovdb.tools.cuda.
    defineDeviceVoxelBlockManager(cudaModule);
#endif
}

} // namespace pynanovdb
