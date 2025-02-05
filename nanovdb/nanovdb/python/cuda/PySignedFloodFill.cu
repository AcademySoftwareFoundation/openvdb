// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PySignedFloodFill.h"

#include <nanovdb/tools/cuda/SignedFloodFill.cuh>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

template<typename BuildT> void defineSignedFloodFill(nb::module_& m, const char* name)
{
    m.def(
        name, [](NanoGrid<BuildT>* d_grid, bool verbose) { return tools::cuda::signedFloodFill(d_grid, verbose); }, "d_grid"_a, "verbose"_a = false);
}

template void defineSignedFloodFill<float>(nb::module_&, const char*);
template void defineSignedFloodFill<double>(nb::module_&, const char*);

} // namespace pynanovdb
