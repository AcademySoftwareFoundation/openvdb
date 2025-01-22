// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyCreateNanoGrid.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>
#ifdef NANOVDB_USE_OPENVDB
#include <nanobind/stl/shared_ptr.h>
#endif

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

namespace {

template<typename BuildT>
GridHandle<HostBuffer> createNanoGridFromFunc(const BuildT&                                       background,
                                              const std::string&                                  name,
                                              GridClass                                           gridClass,
                                              const std::function<BuildT(const nanovdb::Coord&)>& func,
                                              const CoordBBox&                                    bbox)
{
    nanovdb::tools::build::Grid<BuildT> srcGrid(background, name, gridClass);
    srcGrid(func, bbox);
    return nanovdb::tools::createNanoGrid(srcGrid);
}

} // namespace

template<typename BuildT> void defineCreateNanoGrid(nb::module_& m, const char* name)
{
    m.def(name, &createNanoGridFromFunc<BuildT>, nb::call_guard<nb::gil_scoped_release>(), "background"_a, "name"_a, "gridClass"_a, "func"_a, "bbox"_a);
}

template<typename BufferT> void defineOpenToNanoVDB(nb::module_& m)
{
#ifdef NANOVDB_USE_OPENVDB
    m.def("openToNanoVDB", &tools::openToNanoVDB<BufferT>, "base"_a, "sMode"_a = tools::StatsMode::Default, "cMode"_a = CheckMode::Default, "verbose"_a = 0);
#endif
}

template void defineCreateNanoGrid<float>(nb::module_&, const char*);
template void defineCreateNanoGrid<double>(nb::module_&, const char*);
template void defineCreateNanoGrid<int32_t>(nb::module_&, const char*);
template void defineCreateNanoGrid<Vec3f>(nb::module_&, const char*);

template void defineOpenToNanoVDB<HostBuffer>(nb::module_&);

} // namespace pynanovdb
