// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYCREATENANOGRID_HAS_BEEN_INCLUDED
#define NANOVDB_PYCREATENANOGRID_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BuildT> void defineCreateNanoGrid(nb::module_& m, const char* name);

#ifdef NANOVDB_USE_OPENVDB
template<typename BufferT> void defineOpenToNanoVDB(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
