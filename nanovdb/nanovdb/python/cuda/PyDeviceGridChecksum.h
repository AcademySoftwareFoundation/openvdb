// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYDEVICEGRIDCHECKSUM_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYDEVICEGRIDCHECKSUM_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

// Bind the device checksum entry points for one grid BuildT. Each registers
// nanovdb.tools.cuda.evalChecksum / validateChecksum / updateChecksum as an
// overload taking a (device) NanoGrid<BuildT>* reinterpreted as GridData*.
// nanobind disambiguates the overloads on the device grid class.
template<typename BuildT>
void defineDeviceGridChecksum(nb::module_& m);

} // namespace pynanovdb

#endif
