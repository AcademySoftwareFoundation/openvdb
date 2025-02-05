// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyNanoToOpenVDB.h"

#include <nanovdb/HostBuffer.h>
#include <nanobind/operators.h>
#ifdef NANOVDB_USE_OPENVDB
#include <nanovdb/tools/NanoToOpenVDB.h>
#include <nanobind/stl/shared_ptr.h>
#endif

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

template<typename BufferT> void defineNanoToOpenVDB(nb::module_& m)
{
#ifdef NANOVDB_USE_OPENVDB
    // Wrap nanoToOpenVDB into a lambda to workaround an MSVC compiler bug
    m.def(
        "nanoToOpenVDB", [](GridHandle<BufferT>& handle, int verbose, uint32_t n){
            return tools::nanoToOpenVDB<BufferT>(handle, verbose, n);
        }, "handle"_a, "verbose"_a = 0, "n"_a = 0);
#endif
}

template void defineNanoToOpenVDB<HostBuffer>(nb::module_&);

} // namespace pynanovdb
