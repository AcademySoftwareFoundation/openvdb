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
#ifdef NANOVDB_USE_CUDA
#include "cuda/PyPointsToGrid.h"
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

    definePrimitives<HostBuffer>(m);

    defineCreateNanoGrid<float>(m, "createFloatGrid");
    defineCreateNanoGrid<double>(m, "createDoubleGrid");
    defineCreateNanoGrid<int32_t>(m, "createInt32Grid");
    defineCreateNanoGrid<Vec3f>(m, "createVec3fGrid");

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

    definePointsToGrid<math::Rgba8>(cudaModule, "pointsToRGBA8Grid");
#endif
}

} // namespace pynanovdb
