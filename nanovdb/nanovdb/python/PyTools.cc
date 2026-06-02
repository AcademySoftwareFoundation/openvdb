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
#include "cuda/PyDistributedPointsToGrid.h"
#include "cuda/PySampleFromVoxels.h"
#include "cuda/PySignedFloodFill.h"
#include "cuda/PyDilateGrid.h"
#include "cuda/PyCoarsenGrid.h"
#include "cuda/PyRefineGrid.h"
#include "cuda/PyPruneGrid.h"
#include "cuda/PyMergeGrids.h"
#include "cuda/PyInjectData.h"
#include "cuda/PyIndexToGrid.h"
#include "cuda/PyAddBlindData.h"
#include "cuda/PyDeviceGridStats.h"
#include "cuda/PyDeviceGridValidator.h"
#include "cuda/PyDeviceGridChecksum.h"
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

    // Multi-GPU voxel-coordinate -> grid builder (nanovdb::tools::cuda::
    // DistributedPointsToGrid). Distributes an (N, 3) int32 unified-memory
    // array of index-space voxel coordinates over a DeviceMesh. Bound for the
    // index / Rgba8 build types (matching the voxelsTo*Grid set); Point is
    // excluded (its coords must be world-space Vec3, not int32 Coord).
    defineDistributedPointsToGrid<ValueOnIndex>(cudaModule, "DistributedPointsToGrid");
    defineDistributedPointsToGrid<ValueIndex>(cudaModule, "DistributedIndexPointsToGrid");
    defineDistributedPointsToGrid<math::Rgba8>(cudaModule, "DistributedRGBA8PointsToGrid");

    defineSampleFromVoxels<float>(cudaModule, "sampleFromVoxels");
    defineSampleFromVoxels<double>(cudaModule, "sampleFromVoxels");

    // Topological/morphological ops on OnIndex grids (nanovdb::tools::cuda).
    // Each is constrained to OnIndex build types by TopologyBuilder's
    // is_onindex static_assert, so only ValueOnIndex is instantiated.
    defineDilateGrid<ValueOnIndex>(cudaModule, "dilateGrid");
    defineCoarsenGrid<ValueOnIndex>(cudaModule, "coarsenGrid");
    defineRefineGrid<ValueOnIndex>(cudaModule, "refineGrid");
    definePruneGrid<ValueOnIndex>(cudaModule, "pruneGrid");
    defineMergeGrids<ValueOnIndex>(cudaModule, "mergeGrids");

    // Sidecar value transfer across a topology change, and the predicate->mask
    // helper that feeds pruneGrid (nanovdb::util::cuda::Inject* functors).
    defineInject<float>(cudaModule, "inject");
    defineInject<double>(cudaModule, "inject");
    defineInjectPredicateToMask(cudaModule, "injectPredicateToMask");

    // Device VoxelBlockManager (nanovdb::tools::cuda) on nanovdb.tools.cuda.
    defineDeviceVoxelBlockManager(cudaModule);

    // IndexGrid -> regular Grid (nanovdb::tools::cuda::indexToGrid). Source is
    // an index grid (ValueIndex / ValueOnIndex); destination value type is a
    // non-special type (float / double scalar, or Vec3f / Vec3d). All register
    // under "indexToGrid"; nanobind disambiguates on the source grid class and
    // the values ndarray dtype/shape.
    defineIndexToGridScalar<float, ValueIndex>(cudaModule, "indexToGrid");
    defineIndexToGridScalar<float, ValueOnIndex>(cudaModule, "indexToGrid");
    defineIndexToGridScalar<double, ValueIndex>(cudaModule, "indexToGrid");
    defineIndexToGridScalar<double, ValueOnIndex>(cudaModule, "indexToGrid");
    defineIndexToGridVec3<Vec3f, ValueIndex>(cudaModule, "indexToGrid");
    defineIndexToGridVec3<Vec3f, ValueOnIndex>(cudaModule, "indexToGrid");
    defineIndexToGridVec3<Vec3d, ValueIndex>(cudaModule, "indexToGrid");
    defineIndexToGridVec3<Vec3d, ValueOnIndex>(cudaModule, "indexToGrid");

    // Append blind data to a device grid (nanovdb::tools::cuda::addBlindData).
    // Registered under "addBlindData" for a set of (grid BuildT, blind-data
    // element type) combinations; nanobind disambiguates on the grid class and
    // the blindData ndarray dtype.
    defineAddBlindData<float, float>(cudaModule, "addBlindData");
    defineAddBlindData<float, double>(cudaModule, "addBlindData");
    defineAddBlindData<float, uint32_t>(cudaModule, "addBlindData");
    defineAddBlindData<double, float>(cudaModule, "addBlindData");
    defineAddBlindData<double, double>(cudaModule, "addBlindData");
    defineAddBlindData<double, uint32_t>(cudaModule, "addBlindData");
    defineAddBlindData<ValueOnIndex, float>(cudaModule, "addBlindData");
    defineAddBlindData<ValueOnIndex, double>(cudaModule, "addBlindData");
    defineAddBlindData<ValueOnIndex, uint32_t>(cudaModule, "addBlindData");
    defineAddBlindData<ValueIndex, float>(cudaModule, "addBlindData");
    defineAddBlindData<ValueIndex, double>(cudaModule, "addBlindData");
    defineAddBlindData<ValueIndex, uint32_t>(cudaModule, "addBlindData");

    // Device quality-control tools (mirror the host tools.* names on
    // tools.cuda). updateGridStats covers scalar/vector/bool grids; isValid
    // and the checksum entries cover the full callNanoGrid BuildT set.
    defineDeviceUpdateGridStats<float>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<double>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<int16_t>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<int32_t>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<int64_t>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<uint8_t>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<uint32_t>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<Vec3f>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<Vec3d>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<Vec4f>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<Vec4d>(cudaModule, "updateGridStats");
    defineDeviceUpdateGridStats<bool>(cudaModule, "updateGridStats");

    defineDeviceIsValid<float>(cudaModule, "isValid");
    defineDeviceIsValid<double>(cudaModule, "isValid");
    defineDeviceIsValid<int16_t>(cudaModule, "isValid");
    defineDeviceIsValid<int32_t>(cudaModule, "isValid");
    defineDeviceIsValid<int64_t>(cudaModule, "isValid");
    defineDeviceIsValid<uint8_t>(cudaModule, "isValid");
    defineDeviceIsValid<uint32_t>(cudaModule, "isValid");
    defineDeviceIsValid<Vec3f>(cudaModule, "isValid");
    defineDeviceIsValid<Vec3d>(cudaModule, "isValid");
    defineDeviceIsValid<Vec4f>(cudaModule, "isValid");
    defineDeviceIsValid<Vec4d>(cudaModule, "isValid");
    defineDeviceIsValid<math::Rgba8>(cudaModule, "isValid");
    defineDeviceIsValid<bool>(cudaModule, "isValid");
    defineDeviceIsValid<Fp4>(cudaModule, "isValid");
    defineDeviceIsValid<Fp8>(cudaModule, "isValid");
    defineDeviceIsValid<Fp16>(cudaModule, "isValid");
    defineDeviceIsValid<FpN>(cudaModule, "isValid");
    defineDeviceIsValid<ValueIndex>(cudaModule, "isValid");
    defineDeviceIsValid<ValueOnIndex>(cudaModule, "isValid");
    defineDeviceIsValid<ValueMask>(cudaModule, "isValid");

    defineDeviceGridChecksum<float>(cudaModule);
    defineDeviceGridChecksum<double>(cudaModule);
    defineDeviceGridChecksum<int16_t>(cudaModule);
    defineDeviceGridChecksum<int32_t>(cudaModule);
    defineDeviceGridChecksum<int64_t>(cudaModule);
    defineDeviceGridChecksum<uint8_t>(cudaModule);
    defineDeviceGridChecksum<uint32_t>(cudaModule);
    defineDeviceGridChecksum<Vec3f>(cudaModule);
    defineDeviceGridChecksum<Vec3d>(cudaModule);
    defineDeviceGridChecksum<Vec4f>(cudaModule);
    defineDeviceGridChecksum<Vec4d>(cudaModule);
    defineDeviceGridChecksum<math::Rgba8>(cudaModule);
    defineDeviceGridChecksum<bool>(cudaModule);
    defineDeviceGridChecksum<Fp4>(cudaModule);
    defineDeviceGridChecksum<Fp8>(cudaModule);
    defineDeviceGridChecksum<Fp16>(cudaModule);
    defineDeviceGridChecksum<FpN>(cudaModule);
    defineDeviceGridChecksum<ValueIndex>(cudaModule);
    defineDeviceGridChecksum<ValueOnIndex>(cudaModule);
    defineDeviceGridChecksum<ValueMask>(cudaModule);
#endif
}

} // namespace pynanovdb
