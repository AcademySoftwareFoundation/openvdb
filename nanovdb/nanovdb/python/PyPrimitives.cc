// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyPrimitives.h"

#include <nanobind/stl/string.h>

#include <nanovdb/tools/CreatePrimitives.h>
#ifdef NANOVDB_USE_CUDA
#include <nanovdb/cuda/DeviceBuffer.h>
#endif

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

namespace {

template<typename BufferT>
GridHandle<BufferT> createLevelSetSphere(GridType           gridType,
                                         double             radius,
                                         const Vec3d&       center,
                                         double             voxelSize,
                                         double             halfWidth,
                                         const Vec3d&       origin,
                                         const std::string& name,
                                         tools::StatsMode   sMode,
                                         CheckMode          cMode,
                                         const BufferT&     buffer)
{
    switch (gridType) {
    case GridType::Float: return createLevelSetSphere<float, BufferT>(radius, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    case GridType::Double: return createLevelSetSphere<double, BufferT>(radius, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    default: {
        std::stringstream ss;
        // ss << "Cannot createLevelSetSphere for grid of type \"" << toStr(gridType);
        throw std::runtime_error(ss.str() + "\"");
    }
    }
}

template<typename BufferT>
GridHandle<BufferT> createLevelSetTorus(GridType           gridType,
                                        double             majorRadius,
                                        double             minorRadius,
                                        const Vec3d&       center,
                                        double             voxelSize,
                                        double             halfWidth,
                                        const Vec3d&       origin,
                                        const std::string& name,
                                        tools::StatsMode   sMode,
                                        CheckMode          cMode,
                                        const BufferT&     buffer)
{
    switch (gridType) {
    case GridType::Float:
        return createLevelSetTorus<float, BufferT>(majorRadius, minorRadius, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    case GridType::Double:
        return createLevelSetTorus<double, BufferT>(majorRadius, minorRadius, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    default: {
        std::stringstream ss;
        // ss << "Cannot createLevelSetTorus for grid of type \"" << toStr(gridType);
        throw std::runtime_error(ss.str() + "\"");
    }
    }
}

template<typename BufferT>
GridHandle<BufferT> createFogVolumeSphere(GridType           gridType,
                                          double             radius,
                                          const Vec3d&       center,
                                          double             voxelSize,
                                          double             halfWidth,
                                          const Vec3d&       origin,
                                          const std::string& name,
                                          tools::StatsMode   sMode,
                                          CheckMode          cMode,
                                          const BufferT&     buffer)
{
    switch (gridType) {
    case GridType::Float: return createFogVolumeSphere<float, BufferT>(radius, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    case GridType::Double: return createFogVolumeSphere<double, BufferT>(radius, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    default: {
        std::stringstream ss;
        // ss << "Cannot createFogVolumeSphere for grid of type \"" << toStr(gridType);
        throw std::runtime_error(ss.str() + "\"");
    }
    }
}

template<typename BufferT>
GridHandle<BufferT> createFogVolumeTorus(GridType           gridType,
                                         double             majorRadius,
                                         double             minorRadius,
                                         const Vec3d&       center,
                                         double             voxelSize,
                                         double             halfWidth,
                                         const Vec3d&       origin,
                                         const std::string& name,
                                         tools::StatsMode   sMode,
                                         CheckMode          cMode,
                                         const BufferT&     buffer)
{
    switch (gridType) {
    case GridType::Float:
        return createFogVolumeTorus<float, BufferT>(majorRadius, minorRadius, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    case GridType::Double:
        return createFogVolumeTorus<double, BufferT>(majorRadius, minorRadius, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    default: {
        std::stringstream ss;
        // ss << "Cannot createFogVolumeTorus for grid of type \"" << toStr(gridType);
        throw std::runtime_error(ss.str() + "\"");
    }
    }
}

} // namespace

template<typename BufferT> void definePrimitives(nb::module_& m)
{
    m.def("createLevelSetSphere",
          nb::overload_cast<GridType, double, const Vec3d&, double, double, const Vec3d&, const std::string&, tools::StatsMode, CheckMode, const BufferT&>(
              &createLevelSetSphere<BufferT>),
          "gridType"_a = GridType::Float,
          "radius"_a = 100.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "halfWidth"_a = 3.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "sphere_ls",
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT());

    m.def("createLevelSetTorus",
          nb::overload_cast<GridType,
                            double,
                            double,
                            const Vec3d&,
                            double,
                            double,
                            const Vec3d&,
                            const std::string&,
                            tools::StatsMode,
                            CheckMode,
                            const BufferT&>(&createLevelSetTorus<BufferT>),
          "gridType"_a = GridType::Float,
          "majorRadius"_a = 100.0,
          "minorRadius"_a = 50.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "halfWidth"_a = 3.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "torus_ls",
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT());

    m.def("createFogVolumeSphere",
          nb::overload_cast<GridType, double, const Vec3d&, double, double, const Vec3d&, const std::string&, tools::StatsMode, CheckMode, const BufferT&>(
              &createFogVolumeSphere<BufferT>),
          "gridType"_a = GridType::Float,
          "radius"_a = 100.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "halfWidth"_a = 3.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "sphere_fog",
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT());

    m.def("createFogVolumeTorus",
          nb::overload_cast<GridType,
                            double,
                            double,
                            const Vec3d&,
                            double,
                            double,
                            const Vec3d&,
                            const std::string&,
                            tools::StatsMode,
                            CheckMode,
                            const BufferT&>(&createFogVolumeTorus<BufferT>),
          "gridType"_a = GridType::Float,
          "majorRadius"_a = 100.0,
          "minorRadius"_a = 50.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "halfWidth"_a = 3.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "torus_fog",
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT());
}

template void definePrimitives<HostBuffer>(nb::module_&);
#ifdef NANOVDB_USE_CUDA
template void definePrimitives<cuda::DeviceBuffer>(nb::module_&);
#endif

} // namespace pynanovdb
