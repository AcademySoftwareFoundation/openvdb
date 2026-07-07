// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyCoarsenGrid.h"

#include <cstdint>

#include <nanovdb/tools/cuda/CoarsenGrid.cuh>

namespace nb = nanobind;
using namespace nb::literals;
// NOTE: deliberately NOT `using namespace nanovdb;`. These tools instantiate
// CUB DeviceScan, whose nvcc-generated host stub references unqualified
// `cuda::std::...`; with `nanovdb::cuda` in scope that becomes ambiguous and
// fails to compile. Fully qualify nanovdb:: instead (matches PyPointsToGrid.cu).

namespace pynanovdb {

template<typename BuildT> void defineCoarsenGrid(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nanovdb::NanoGrid<BuildT>* d_grid, uintptr_t stream) {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            // CoarsenGrid::getHandle launches kernels and synchronizes the
            // stream; pure CUDA touching no Python objects, so release the GIL.
            nb::gil_scoped_release release;
            nanovdb::tools::cuda::CoarsenGrid<BuildT> coarsener(d_grid, s);
            return coarsener.getHandle();
        },
        "d_grid"_a,
        "stream"_a = 0,
        "Topologically coarsen (2x downsample) a device OnIndex grid and return "
        "a fresh device GridHandle of the coarsened grid. stream is a raw CUDA "
        "stream handle (Python int; 0 = default stream).");
}

template void defineCoarsenGrid<nanovdb::ValueOnIndex>(nb::module_&, const char*);

} // namespace pynanovdb
