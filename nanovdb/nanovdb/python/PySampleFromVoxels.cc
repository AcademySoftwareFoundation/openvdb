// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PySampleFromVoxels.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/SampleFromVoxels.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

namespace {

template<typename TreeT, int Order> void defineSampleFromVoxels(nb::module_& m, const char* name)
{
    using CoordT = typename TreeT::CoordType;
    nb::class_<math::SampleFromVoxels<TreeT, Order, false>>(m, name)
        .def(
            "__call__", [](const math::SampleFromVoxels<TreeT, Order, false>& sampler, const CoordT& ijk) { return sampler(ijk); }, nb::is_operator(), "ijk"_a)
        .def(
            "__call__", [](const math::SampleFromVoxels<TreeT, Order, false>& sampler, const Vec3f& xyz) { return sampler(xyz); }, nb::is_operator(), "xyz"_a)
        .def(
            "__call__", [](const math::SampleFromVoxels<TreeT, Order, false>& sampler, const Vec3d& xyz) { return sampler(xyz); }, nb::is_operator(), "xyz"_a);
}

template<typename TreeT, int Order> void defineCreateSampler(nb::module_& m, const char* name)
{
    m.def(
        name, [](const Grid<TreeT>& grid) { return math::createSampler<Order, TreeT, false>(grid.tree()); }, "grid"_a);
}

} // namespace

template<typename BuildT> void defineNearestNeighborSampler(nb::module_& m, const char* name)
{
    using TreeT = NanoTree<BuildT>;

    defineSampleFromVoxels<TreeT, 0>(m, name);
    defineCreateSampler<TreeT, 0>(m, "createNearestNeighborSampler");
}

template<typename BuildT> void defineTrilinearSampler(nb::module_& m, const char* name)
{
    using TreeT = NanoTree<BuildT>;

    defineSampleFromVoxels<TreeT, 1>(m, name);
    defineCreateSampler<TreeT, 1>(m, "createTrilinearSampler");
}

template<typename BuildT> void defineTriquadraticSampler(nb::module_& m, const char* name)
{
    using TreeT = NanoTree<BuildT>;

    defineSampleFromVoxels<TreeT, 2>(m, name);
    defineCreateSampler<TreeT, 2>(m, "createTriquadraticSampler");
}

template<typename BuildT> void defineTricubicSampler(nb::module_& m, const char* name)
{
    using TreeT = NanoTree<BuildT>;

    defineSampleFromVoxels<TreeT, 3>(m, name);
    defineCreateSampler<TreeT, 3>(m, "createTricubicSampler");
}

template void defineNearestNeighborSampler<float>(nb::module_&, const char*);
template void defineTrilinearSampler<float>(nb::module_&, const char*);
template void defineTriquadraticSampler<float>(nb::module_&, const char*);
template void defineTricubicSampler<float>(nb::module_&, const char*);

template void defineNearestNeighborSampler<double>(nb::module_&, const char*);
template void defineTrilinearSampler<double>(nb::module_&, const char*);
template void defineTriquadraticSampler<double>(nb::module_&, const char*);
template void defineTricubicSampler<double>(nb::module_&, const char*);

template void defineNearestNeighborSampler<int32_t>(nb::module_&, const char*);
template void defineTrilinearSampler<int32_t>(nb::module_&, const char*);
template void defineTriquadraticSampler<int32_t>(nb::module_&, const char*);
template void defineTricubicSampler<int32_t>(nb::module_&, const char*);

template void defineNearestNeighborSampler<Vec3f>(nb::module_&, const char*);
template void defineTrilinearSampler<Vec3f>(nb::module_&, const char*);
template void defineTriquadraticSampler<Vec3f>(nb::module_&, const char*);
template void defineTricubicSampler<Vec3f>(nb::module_&, const char*);

} // namespace pynanovdb
