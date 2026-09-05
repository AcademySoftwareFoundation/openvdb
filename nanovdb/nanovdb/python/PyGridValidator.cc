// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyGridValidator.h"

#include <nanovdb/GridHandle.h>
#include <nanovdb/tools/GridValidator.h>
#ifdef NANOVDB_USE_CUDA
#include <nanovdb/cuda/DeviceBuffer.h>
#endif

#include <nanobind/operators.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

template<typename BufferT> void defineValidateGrids(nb::module_& m)
{
    m.def("validateGrids", &tools::validateGrids<GridHandle<BufferT>>, "handle"_a, "mode"_a, "verbose"_a);
}

template void defineValidateGrids<HostBuffer>(nb::module_&);
#ifdef NANOVDB_USE_CUDA
template void defineValidateGrids<cuda::DeviceBuffer>(nb::module_&);
#endif

} // namespace pynanovdb
