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
    using CoordT   = typename TreeT::CoordType;
    using ValueT   = typename TreeT::ValueType;
    using SamplerT = math::SampleFromVoxels<TreeT, Order, false>;
    auto cls = nb::class_<SamplerT>(m, name,
        "Callable sampler that reconstructs a grid value at an arbitrary "
        "index-space position. Build via the matching create*Sampler() factory.")
        .def(
            "__call__", [](const SamplerT& sampler, const CoordT& ijk) { return sampler(ijk); }, nb::is_operator(), "ijk"_a,
            "Sample the grid at integer voxel coordinate ijk.")
        .def(
            "__call__", [](const SamplerT& sampler, const Vec3f& xyz) { return sampler(xyz); }, nb::is_operator(), "xyz"_a,
            "Sample the grid at fractional index-space position xyz.")
        .def(
            "__call__", [](const SamplerT& sampler, const Vec3d& xyz) { return sampler(xyz); }, nb::is_operator(), "xyz"_a,
            "Sample the grid at fractional index-space position xyz (double).");
    // gradient() exists on the trilinear sampler only, zeroCrossing() on the
    // trilinear and triquadratic samplers, and both static_assert a
    // floating-point ValueT in C++ — mirror that gating here.
    if constexpr (Order == 1 && util::is_floating_point<ValueT>::value) {
        cls.def(
               "gradient", [](const SamplerT& sampler, const Vec3f& xyz) { return sampler.gradient(xyz); }, "xyz"_a,
               "Return the index-space gradient of the trilinear reconstruction "
               "at fractional index-space position xyz. Use "
               "grid.indexToWorldGrad() to move it to world space.")
            .def(
               "gradient", [](const SamplerT& sampler, const Vec3d& xyz) { return sampler.gradient(xyz); }, "xyz"_a,
               "Return the index-space gradient at fractional index-space "
               "position xyz (double).");
    }
    if constexpr ((Order == 1 || Order == 2) && util::is_floating_point<ValueT>::value) {
        cls.def(
               "zeroCrossing", [](const SamplerT& sampler, const Vec3f& xyz) { return sampler.zeroCrossing(xyz); }, "xyz"_a,
               "True iff the reconstruction stencil at fractional index-space "
               "position xyz straddles the zero iso-surface.")
            .def(
               "zeroCrossing", [](const SamplerT& sampler, const Vec3d& xyz) { return sampler.zeroCrossing(xyz); }, "xyz"_a,
               "True iff the reconstruction stencil at fractional index-space "
               "position xyz (double) straddles the zero iso-surface.");
    }
}

template<typename TreeT, int Order> void defineCreateSampler(nb::module_& m, const char* name)
{
    m.def(
        name, [](const Grid<TreeT>& grid) { return math::createSampler<Order, TreeT, false>(grid.tree()); }, "grid"_a,
        "Build a sampler of the matching order that reads values from the "
        "given grid's tree.");
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

#define NANOVDB_PY_FOR_EACH_SAMPLEABLE_BUILDT(T, Suffix)            \
    template void defineNearestNeighborSampler<T>(nb::module_&, const char*); \
    template void defineTrilinearSampler<T>(nb::module_&, const char*);       \
    template void defineTriquadraticSampler<T>(nb::module_&, const char*);    \
    template void defineTricubicSampler<T>(nb::module_&, const char*);
#include "BuildTypes.def"

} // namespace pynanovdb
