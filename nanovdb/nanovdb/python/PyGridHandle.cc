// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyGridHandle.h"
#include <nanobind/ndarray.h>

#include <iostream>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

void defineHostGridHandle(nb::module_& m)
{
    using BufferT = HostBuffer;
    defineGridHandle<BufferT>(m, "GridHandle")
        .def(
            "__init__",
            [](GridHandle<BufferT>& handle, nb::ndarray<uint32_t, nb::ndim<1>, nb::device::cpu> t) {
                auto buffer = BufferT::createFull(t.size() * sizeof(uint32_t), t.data());
                new (&handle) GridHandle<BufferT>(std::move(buffer));
            },
            "t"_a.noconvert());
}

} // namespace pynanovdb
