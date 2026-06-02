// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyDilateGrid.h"

#include <cstdint>

#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/util/MorphologyHelpers.h>

namespace nb = nanobind;
using namespace nb::literals;
// NOTE: deliberately NOT `using namespace nanovdb;`. These tools instantiate
// CUB DeviceScan, whose nvcc-generated host stub references unqualified
// `cuda::std::...`; with `nanovdb::cuda` in scope that becomes ambiguous and
// fails to compile. Fully qualify nanovdb:: instead (matches PyPointsToGrid.cu).

namespace pynanovdb {

template<typename BuildT> void defineDilateGrid(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nanovdb::NanoGrid<BuildT>* d_grid, int op, uintptr_t stream) {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            // DilateGrid::getHandle launches kernels and synchronizes the
            // stream; pure CUDA touching no Python objects, so release the GIL.
            nb::gil_scoped_release release;
            nanovdb::tools::cuda::DilateGrid<BuildT> dilator(d_grid, s);
            dilator.setOperation(static_cast<nanovdb::tools::morphology::NearestNeighbors>(op));
            return dilator.getHandle();
        },
        "d_grid"_a,
        "op"_a = static_cast<int>(nanovdb::tools::morphology::NN_FACE_EDGE_VERTEX),
        "stream"_a = 0,
        "Morphologically dilate a device OnIndex grid and return a fresh device "
        "GridHandle of the dilated grid. op is a nearest-neighbor stencil: 6 "
        "(NN_FACE) or 26 (NN_FACE_EDGE_VERTEX). NN_FACE_EDGE (18) is accepted by "
        "the C++ setter but is not implemented and raises at getHandle time. "
        "stream is a raw CUDA stream handle (Python int; 0 = default stream).");
}

template void defineDilateGrid<nanovdb::ValueOnIndex>(nb::module_&, const char*);

} // namespace pynanovdb
