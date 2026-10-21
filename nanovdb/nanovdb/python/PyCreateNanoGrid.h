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
///        include both NanoGrid<SrcBuildT> and tools::build::Grid<SrcBuildT>:
///        the quantized createNanoGridFp* / FpN paths accept float only
///        (C++ Fp{4,8,16,N}::preProcess static-asserts SrcValueT == float);
///        the createNanoGridIndex / OnIndex paths accept float, double,
///        int32_t, and Vec3f sources. Additional source BuildTs can be
///        added by extending the explicit try-each-SrcBuildT chains in
///        createNanoGridFpX / FpNImpl / createIndexImpl. Also binds the
///        tools.CreateNanoGrid converter class, which mirrors the C++
///        tools::CreateNanoGrid and adds blind-data authoring via
///        addBlindData() (same source BuildT set as the index paths).
void defineCreateNanoGridConversions(nb::module_& toolsModule);

} // namespace pynanovdb

#endif
