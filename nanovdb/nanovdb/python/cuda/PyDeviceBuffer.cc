// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyDeviceBuffer.h"

#include <nanovdb/cuda/DeviceBuffer.h>

namespace nb = nanobind;
using namespace nanovdb;

namespace pynanovdb {

void defineDeviceBuffer(nb::module_& m)
{
    nb::class_<cuda::DeviceBuffer>(m, "DeviceBuffer",
        "CUDA device-side buffer used to back a DeviceGridHandle. Holds a "
        "host mirror and a device pointer; deviceUpload / deviceDownload on "
        "the handle move bytes between the two.");
}

} // namespace pynanovdb

#endif
