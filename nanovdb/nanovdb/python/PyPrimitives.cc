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

// ---------- New primitives ----------
// Same float/double switch pattern as the four existing primitives above.
// The C++ templates also accept Fp4/Fp8/Fp16/FpN; those flavors live on
// the createNanoGrid path with explicit oracle and dither parameters
// rather than being expressed as primitive overloads here.

template<typename BufferT>
GridHandle<BufferT> createLevelSetBox(GridType           gridType,
                                      double             width,
                                      double             height,
                                      double             depth,
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
        return tools::createLevelSetBox<float, BufferT>(
            width, height, depth, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    case GridType::Double:
        return tools::createLevelSetBox<double, BufferT>(
            width, height, depth, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    default:
        throw std::runtime_error(
            "createLevelSetBox: only float and double grid types are supported");
    }
}

template<typename BufferT>
GridHandle<BufferT> createLevelSetBBox(GridType           gridType,
                                       double             width,
                                       double             height,
                                       double             depth,
                                       double             thickness,
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
        return tools::createLevelSetBBox<float, BufferT>(
            width, height, depth, thickness, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    case GridType::Double:
        return tools::createLevelSetBBox<double, BufferT>(
            width, height, depth, thickness, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    default:
        throw std::runtime_error(
            "createLevelSetBBox: only float and double grid types are supported");
    }
}

template<typename BufferT>
GridHandle<BufferT> createLevelSetOctahedron(GridType           gridType,
                                             double             scale,
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
        return tools::createLevelSetOctahedron<float, BufferT>(
            scale, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    case GridType::Double:
        return tools::createLevelSetOctahedron<double, BufferT>(
            scale, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    default:
        throw std::runtime_error(
            "createLevelSetOctahedron: only float and double grid types are supported");
    }
}

template<typename BufferT>
GridHandle<BufferT> createFogVolumeBox(GridType           gridType,
                                       double             width,
                                       double             height,
                                       double             depth,
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
        return tools::createFogVolumeBox<float, BufferT>(
            width, height, depth, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    case GridType::Double:
        return tools::createFogVolumeBox<double, BufferT>(
            width, height, depth, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    default:
        throw std::runtime_error(
            "createFogVolumeBox: only float and double grid types are supported");
    }
}

template<typename BufferT>
GridHandle<BufferT> createFogVolumeOctahedron(GridType           gridType,
                                              double             scale,
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
        return tools::createFogVolumeOctahedron<float, BufferT>(
            scale, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    case GridType::Double:
        return tools::createFogVolumeOctahedron<double, BufferT>(
            scale, center, voxelSize, halfWidth, origin, name, sMode, cMode, buffer);
    default:
        throw std::runtime_error(
            "createFogVolumeOctahedron: only float and double grid types are supported");
    }
}

// Point primitives. The result is always a PointDataGrid (uint32 storage),
// so unlike the level-set / fog-volume primitives there's no value-type
// dispatch worth exposing — the intermediate level-set's precision is
// not user-controllable in this binding. The C++ template also accepts
// BuildT=double, but that path segfaults during scatter at least with the
// current C++ implementation, so the binding stays on the float-only
// instantiation that's exercised by the C++ unit tests.
template<typename BufferT>
GridHandle<BufferT> createPointSphere(int                pointsPerVoxel,
                                      double             radius,
                                      const Vec3d&       center,
                                      double             voxelSize,
                                      const Vec3d&       origin,
                                      const std::string& name,
                                      CheckMode          mode,
                                      const BufferT&     buffer)
{
    return tools::createPointSphere<float, BufferT>(
        pointsPerVoxel, radius, center, voxelSize, origin, name, mode, buffer);
}

template<typename BufferT>
GridHandle<BufferT> createPointTorus(int                pointsPerVoxel,
                                     double             majorRadius,
                                     double             minorRadius,
                                     const Vec3d&       center,
                                     double             voxelSize,
                                     const Vec3d&       origin,
                                     const std::string& name,
                                     CheckMode          cMode,
                                     const BufferT&     buffer)
{
    return tools::createPointTorus<float, BufferT>(
        pointsPerVoxel, majorRadius, minorRadius, center, voxelSize, origin, name, cMode, buffer);
}

template<typename BufferT>
GridHandle<BufferT> createPointBox(int                pointsPerVoxel,
                                   double             width,
                                   double             height,
                                   double             depth,
                                   const Vec3d&       center,
                                   double             voxelSize,
                                   const Vec3d&       origin,
                                   const std::string& name,
                                   CheckMode          mode,
                                   const BufferT&     buffer)
{
    return tools::createPointBox<float, BufferT>(
        pointsPerVoxel, width, height, depth, center, voxelSize, origin, name, mode, buffer);
}

// createPointScatter takes an existing level set as its source. We bind
// the float source variant — the C++ template also accepts double, but
// the existing primitives and tests use float, and the source grid is
// the runtime-typed nanovdb::NanoGrid<float>, so a single overload keeps
// the Python surface simple.
template<typename BufferT>
GridHandle<BufferT> createPointScatter(const NanoGrid<float>& srcGrid,
                                       int                    pointsPerVoxel,
                                       const std::string&     name,
                                       CheckMode              mode,
                                       const BufferT&         buffer)
{
    return tools::createPointScatter<float, BufferT>(
        srcGrid, pointsPerVoxel, name, mode, buffer);
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
          "buffer"_a = BufferT(),
          "Narrow-band level set of a sphere of the given radius and center.");

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
          "buffer"_a = BufferT(),
          "Narrow-band level set of a torus with the given major and minor radii.");

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
          "buffer"_a = BufferT(),
          "Sparse fog volume of a sphere of the given radius and center.");

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
          "buffer"_a = BufferT(),
          "Sparse fog volume of a torus with the given major and minor radii.");

    // ---------- Level-set / fog-volume box / bbox / octahedron primitives ----
    m.def("createLevelSetBox", &createLevelSetBox<BufferT>,
          "gridType"_a = GridType::Float,
          "width"_a = 40.0,
          "height"_a = 60.0,
          "depth"_a = 100.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "halfWidth"_a = 3.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "box_ls",
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT(),
          "Narrow-band level set of an axis-aligned box.");

    m.def("createLevelSetBBox", &createLevelSetBBox<BufferT>,
          "gridType"_a = GridType::Float,
          "width"_a = 40.0,
          "height"_a = 60.0,
          "depth"_a = 100.0,
          "thickness"_a = 10.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "halfWidth"_a = 3.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "bbox_ls",
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT(),
          "Narrow-band level set of a hollow box wireframe (BBox = bounding "
          "box edges with the given thickness).");

    m.def("createLevelSetOctahedron", &createLevelSetOctahedron<BufferT>,
          "gridType"_a = GridType::Float,
          "scale"_a = 100.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "halfWidth"_a = 3.0,
          "origin"_a = Vec3d(0.0),
          // Default name spells the shape correctly even though the
          // upstream C++ default still carries the historical
          // "octadedron_ls" typo. Callers can override either way.
          "name"_a = "octahedron_ls",
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT(),
          "Narrow-band level set of an octahedron.");

    m.def("createFogVolumeBox", &createFogVolumeBox<BufferT>,
          "gridType"_a = GridType::Float,
          "width"_a = 40.0,
          "height"_a = 60.0,
          "depth"_a = 100.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "halfWidth"_a = 3.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "box_fog",
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT(),
          "Sparse fog volume of a box (exterior 0/inactive, interior active "
          "with values smoothly varying from 0 at the surface to 1 inside).");

    m.def("createFogVolumeOctahedron", &createFogVolumeOctahedron<BufferT>,
          "gridType"_a = GridType::Float,
          "scale"_a = 100.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "halfWidth"_a = 3.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "octahedron_fog",
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT(),
          "Sparse fog volume of an octahedron.");

    // ---------- Point primitives ----------
    m.def("createPointSphere", &createPointSphere<BufferT>,
          "pointsPerVoxel"_a = 1,
          "radius"_a = 100.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "sphere_points",
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT(),
          "PointDataGrid of points scattered on the surface of a sphere. "
          "The output grid is always a UInt32 PointDataGrid; the "
          "intermediate level-set's value type is hard-coded to float.");

    m.def("createPointTorus", &createPointTorus<BufferT>,
          "pointsPerVoxel"_a = 1,
          "majorRadius"_a = 100.0,
          "minorRadius"_a = 50.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "torus_points",
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT(),
          "PointDataGrid of points scattered on the surface of a torus. "
          "Always returns a UInt32 PointDataGrid.");

    m.def("createPointBox", &createPointBox<BufferT>,
          "pointsPerVoxel"_a = 1,
          "width"_a = 40.0,
          "height"_a = 60.0,
          "depth"_a = 100.0,
          "center"_a = Vec3d(0.0),
          "voxelSize"_a = 1.0,
          "origin"_a = Vec3d(0.0),
          "name"_a = "box_points",
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT(),
          "PointDataGrid of points scattered on the surface of a box. "
          "Always returns a UInt32 PointDataGrid.");

    m.def("createPointScatter", &createPointScatter<BufferT>,
          "srcGrid"_a,
          "pointsPerVoxel"_a = 1,
          "name"_a = "point_scatter",
          "cMode"_a = CheckMode::Default,
          "buffer"_a = BufferT(),
          "Scatter a PointDataGrid into the active voxels of a "
          "NanoGrid<float> level set. The source grid must satisfy "
          "srcGrid.isLevelSet() and have an active bounding box; "
          "non-level-set sources (e.g. fog volumes) raise RuntimeError. "
          "Point coordinates are stored as blind data in world space.");
}

template void definePrimitives<HostBuffer>(nb::module_&);
#ifdef NANOVDB_USE_CUDA
template void definePrimitives<cuda::DeviceBuffer>(nb::module_&);
#endif

} // namespace pynanovdb
