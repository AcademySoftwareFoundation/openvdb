// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PySignedFloodFill.h"

#include <cstdint>

#include <nanovdb/tools/cuda/SignedFloodFill.cuh>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

template<typename BuildT> void defineSignedFloodFill(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](NanoGrid<BuildT>* d_grid, bool verbose, uintptr_t stream) {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            // signedFloodFill launches kernels and synchronizes the stream;
            // pure CUDA touching no Python objects, so release the GIL.
            nb::gil_scoped_release release;
            tools::cuda::signedFloodFill(d_grid, verbose, s);
        },
        "d_grid"_a,
        "verbose"_a = false,
        "stream"_a = 0,
        "Perform a signed flood fill on a device float/double grid in place. "
        "stream is a raw CUDA stream handle (Python int; 0 = default stream).");
}

template void defineSignedFloodFill<float>(nb::module_&, const char*);
template void defineSignedFloodFill<double>(nb::module_&, const char*);

} // namespace pynanovdb
