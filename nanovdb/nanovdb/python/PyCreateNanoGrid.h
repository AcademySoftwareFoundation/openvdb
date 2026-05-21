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

/// @brief Bind the AbsDiff / RelDiff quantization oracle classes and the
///        polymorphic createNanoGridFp4 / Fp8 / Fp16 / FpN / Index / OnIndex
///        free functions on the nanovdb.tools submodule. Sources accepted
///        include both NanoGrid<SrcBuildT> and tools::build::Grid<SrcBuildT>
///        for every scalar/vector BuildT in BuildTypes.def that the
///        underlying tools::createNanoGrid template supports.
void defineCreateNanoGridConversions(nb::module_& toolsModule);

} // namespace pynanovdb

#endif
