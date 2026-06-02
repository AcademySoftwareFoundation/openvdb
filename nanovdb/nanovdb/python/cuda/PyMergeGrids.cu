// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyMergeGrids.h"

#include <cstdint>

#include <nanovdb/tools/cuda/MergeGrids.cuh>

namespace nb = nanobind;
using namespace nb::literals;
// NOTE: deliberately NOT `using namespace nanovdb;`. These tools instantiate
// CUB DeviceScan, whose nvcc-generated host stub references unqualified
// `cuda::std::...`; with `nanovdb::cuda` in scope that becomes ambiguous and
// fails to compile. Fully qualify nanovdb:: instead (matches PyPointsToGrid.cu).

namespace pynanovdb {

template<typename BuildT> void defineMergeGrids(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nanovdb::NanoGrid<BuildT>* d_grid1,
           nanovdb::NanoGrid<BuildT>* d_grid2,
           uintptr_t                  stream) {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            // MergeGrids::getHandle launches kernels and synchronizes the
            // stream; pure CUDA touching no Python objects, so release the GIL.
            nb::gil_scoped_release release;
            nanovdb::tools::cuda::MergeGrids<BuildT> merger(d_grid1, d_grid2, s);
            return merger.getHandle();
        },
        "d_grid1"_a,
        "d_grid2"_a,
        "stream"_a = 0,
        "Topologically merge (active-mask union) two device OnIndex grids and "
        "return a fresh device GridHandle of the union. The operation is "
        "strictly binary; chain calls to union more than two grids. Output "
        "metadata is taken from d_grid1. stream is a raw CUDA stream handle "
        "(Python int; 0 = default stream).");
}

template void defineMergeGrids<nanovdb::ValueOnIndex>(nb::module_&, const char*);

} // namespace pynanovdb
