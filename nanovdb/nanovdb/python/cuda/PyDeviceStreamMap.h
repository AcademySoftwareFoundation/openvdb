// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYDEVICESTREAMMAP_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYDEVICESTREAMMAP_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

#ifdef NANOVDB_USE_CUDA
/// @brief Register nanovdb::cuda::DeviceStreamMap and its DeviceType enum on
///        the nanovdb.cuda submodule. DeviceStreamMap is a std::map<int,
///        cudaStream_t> that, on construction, filters the available devices by
///        a DeviceType policy and creates one stream per surviving device.
/// @note  DeviceStreamMap.h defines its (non-template) ctor/dtor in the header
///        without `inline`, so it must be included in exactly ONE translation
///        unit — this one.
void defineDeviceStreamMap(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
