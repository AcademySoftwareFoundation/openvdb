// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyHostBuffer.h"

#include <nanovdb/HostBuffer.h>

namespace nb = nanobind;
using namespace nanovdb;

namespace pynanovdb {

void defineHostBuffer(nb::module_& m)
{
    nb::class_<HostBuffer>(m, "HostBuffer",
        "Default host-side buffer used to back a GridHandle. Memory is "
        "owned by this buffer and freed when the handle (and therefore "
        "the buffer) is destroyed.");
}

} // namespace pynanovdb
