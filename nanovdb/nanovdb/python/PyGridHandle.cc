// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyGridHandle.h"
#include "PyValidate.h"
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
                requireAlignedBuffer(t.data(), "GridHandle", "t");
                auto buffer = BufferT::createFull(t.size() * sizeof(uint32_t), t.data());
                new (&handle) GridHandle<BufferT>(std::move(buffer));
            },
            "t"_a.noconvert(),
            // The HostBuffer is non-owning, so the handle must keep the
            // source array alive for as long as it exists.
            nb::keep_alive<1, 2>(),
            "Construct a GridHandle that wraps an existing 1-D uint32 array "
            "holding one or more serialized NanoVDB grids. The handle does not "
            "copy or own the memory: it keeps the array alive for its own "
            "lifetime and reads the grids in place. The array's data pointer "
            "must be aligned to NANOVDB_DATA_ALIGNMENT (32 bytes); a plain "
            "NumPy allocation is usually only 16-byte aligned and raises "
            "ValueError. Raises RuntimeError if the bytes do not form a valid "
            "grid.");
    defineGridHandleUtilities<BufferT>(m);
}

} // namespace pynanovdb
