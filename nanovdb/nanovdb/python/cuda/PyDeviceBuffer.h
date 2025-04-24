// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYDEVICEBUFFER_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYDEVICEBUFFER_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

#ifdef NANOVDB_USE_CUDA
void defineDeviceBuffer(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
