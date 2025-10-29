// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h> // for __repr__
#ifdef NANOVDB_USE_CUDA
#include <nanovdb/cuda/DeviceBuffer.h>
#endif

#include <sstream>

#include "cuda/PyDeviceBuffer.h"
#include "PyGridHandle.h"
#include "PyHostBuffer.h"
#include "PyIO.h"
#include "PyMath.h"
#include "PyTools.h"
#include "PyGridChecksum.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

void defineVersion(nb::module_& m)
{
    nb::class_<Version>(m, "Version", "Bit-compacted representation of all three version numbers")
        .def(nb::init<>())
        .def(nb::init<uint32_t>(), "data"_a)
        .def(nb::init<uint32_t, uint32_t, uint32_t>(), "major"_a, "minor"_a, "patch"_a)
        .def(nb::self == nb::self, "rhs"_a)
        .def(nb::self < nb::self, "rhs"_a)
        .def(nb::self <= nb::self, "rhs"_a)
        .def(nb::self > nb::self, "rhs"_a)
        .def(nb::self >= nb::self, "rhs"_a)
        .def("id", &Version::id)
        .def("getMajor", &Version::getMajor)
        .def("getMinor", &Version::getMinor)
        .def("getPatch", &Version::getPatch)
        .def("age", &Version::age)
        .def("__repr__", [](const Version& version) {
            char str[strlen<Version>()];
            toStr(str, version);
            return std::string(str);
        });
}

void definePointTypes(nb::module_& m)
{
    nb::enum_<PointType>(m, "PointType")
        .value("Disable", PointType::Disable)
        .value("PointID", PointType::PointID)
        .value("World64", PointType::World64)
        .value("World32", PointType::World32)
        .value("Grid64", PointType::Grid64)
        .value("Grid32", PointType::Grid32)
        .value("Voxel32", PointType::Voxel32)
        .value("Voxel16", PointType::Voxel16)
        .value("Voxel8", PointType::Voxel8)
        .value("Default", PointType::Default)
        .value("End", PointType::End);
}

template<uint32_t LOG2DIM> void defineMask(nb::module_& m, const char* name, const char* doc)
{
    nb::class_<Mask<LOG2DIM>>(m, name, doc)
        .def_static("memUsage", &Mask<LOG2DIM>::memUsage)
        .def_static("bitCount", &Mask<LOG2DIM>::bitCount)
        .def_static("wordCount", &Mask<LOG2DIM>::wordCount)
        .def("countOn", nb::overload_cast<>(&Mask<LOG2DIM>::countOn, nb::const_))
        .def("countOn", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::countOn, nb::const_), "i"_a)
        .def(nb::init<>())
        .def(nb::init<bool>(), "on"_a)
        .def(nb::init<const Mask<LOG2DIM>>(), "other"_a)
        .def(nb::self == nb::self, "other"_a)
        .def(nb::self != nb::self, "other"_a)
        .def("isOn", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::isOn, nb::const_), "n"_a)
        .def("isOff", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::isOff, nb::const_), "n"_a)
        .def("isOn", nb::overload_cast<>(&Mask<LOG2DIM>::isOn, nb::const_))
        .def("isOff", nb::overload_cast<>(&Mask<LOG2DIM>::isOff, nb::const_))
        .def("setOn", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::setOn), "n"_a)
        .def("setOff", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::setOff), "n"_a)
        .def("set", nb::overload_cast<uint32_t, bool>(&Mask<LOG2DIM>::set), "n"_a, "on"_a)
        .def("setOn", nb::overload_cast<>(&Mask<LOG2DIM>::setOn))
        .def("setOff", nb::overload_cast<>(&Mask<LOG2DIM>::setOff))
        .def("set", nb::overload_cast<bool>(&Mask<LOG2DIM>::set), "on"_a)
        .def("toggle", nb::overload_cast<>(&Mask<LOG2DIM>::toggle))
        .def("toggle", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::toggle), "n"_a)
        .def(
            "__iand__", [](Mask<LOG2DIM>& a, const Mask<LOG2DIM>& b) { return a &= b; }, nb::is_operator(), "other"_a)
        .def(
            "__ior__", [](Mask<LOG2DIM>& a, const Mask<LOG2DIM>& b) { return a |= b; }, nb::is_operator(), "other"_a)
        .def(
            "__isub__", [](Mask<LOG2DIM>& a, const Mask<LOG2DIM>& b) { return a -= b; }, nb::is_operator(), "other"_a)
        .def(
            "__ixor__", [](Mask<LOG2DIM>& a, const Mask<LOG2DIM>& b) { return a ^= b; }, nb::is_operator(), "other"_a)
        .def("findFirstOn", &Mask<LOG2DIM>::template findFirst<true>)
        .def("findFirstOff", &Mask<LOG2DIM>::template findFirst<false>)
        .def("findNextOn", &Mask<LOG2DIM>::template findNext<true>, "start"_a)
        .def("findNextOff", &Mask<LOG2DIM>::template findNext<false>, "start"_a)
        .def("findPrevOn", &Mask<LOG2DIM>::template findPrev<true>, "start"_a)
        .def("findPrevOff", &Mask<LOG2DIM>::template findPrev<false>, "start"_a);
}

void defineMap(nb::module_& m)
{
    nb::class_<Map>(m, "Map", "Defines an affine transform and its inverse represented as a 3x3 matrix and a vec3 translation")
        .def(nb::init<>())
        .def(nb::init<double, const Vec3d&>(), "s"_a, "t"_a = Vec3d(0.))
        .def("set", nb::overload_cast<double, const Vec3f&, double>(&Map::template set<Vec3f>), "scale"_a, "translation"_a, "taper"_a = 1.)
        .def("set", nb::overload_cast<double, const Vec3d&, double>(&Map::template set<Vec3d>), "scale"_a, "translation"_a, "taper"_a = 1.)
        .def("applyMap", nb::overload_cast<const Vec3f&>(&Map::template applyMap<Vec3f>, nb::const_), "ijk"_a)
        .def("applyMap", nb::overload_cast<const Vec3d&>(&Map::template applyMap<Vec3d>, nb::const_), "ijk"_a)
        .def("applyMapF", nb::overload_cast<const Vec3f&>(&Map::template applyMapF<Vec3f>, nb::const_), "ijk"_a)
        .def("applyMapF", nb::overload_cast<const Vec3d&>(&Map::template applyMapF<Vec3d>, nb::const_), "ijk"_a)
        .def("applyJacobian", nb::overload_cast<const Vec3f&>(&Map::template applyJacobian<Vec3f>, nb::const_), "ijk"_a)
        .def("applyJacobian", nb::overload_cast<const Vec3d&>(&Map::template applyJacobian<Vec3d>, nb::const_), "ijk"_a)
        .def("applyJacobianF", nb::overload_cast<const Vec3f&>(&Map::template applyJacobianF<Vec3f>, nb::const_), "ijk"_a)
        .def("applyJacobianF", nb::overload_cast<const Vec3d&>(&Map::template applyJacobianF<Vec3d>, nb::const_), "ijk"_a)
        .def("applyInverseMap", nb::overload_cast<const Vec3f&>(&Map::template applyInverseMap<Vec3f>, nb::const_), "xyz"_a)
        .def("applyInverseMap", nb::overload_cast<const Vec3d&>(&Map::template applyInverseMap<Vec3d>, nb::const_), "xyz"_a)
        .def("applyInverseMapF", nb::overload_cast<const Vec3f&>(&Map::template applyInverseMapF<Vec3f>, nb::const_), "xyz"_a)
        .def("applyInverseMapF", nb::overload_cast<const Vec3d&>(&Map::template applyInverseMapF<Vec3d>, nb::const_), "xyz"_a)
        .def("applyInverseJacobian", nb::overload_cast<const Vec3f&>(&Map::template applyInverseJacobian<Vec3f>, nb::const_), "xyz"_a)
        .def("applyInverseJacobian", nb::overload_cast<const Vec3d&>(&Map::template applyInverseJacobian<Vec3d>, nb::const_), "xyz"_a)
        .def("applyInverseJacobianF", nb::overload_cast<const Vec3f&>(&Map::template applyInverseJacobianF<Vec3f>, nb::const_), "xyz"_a)
        .def("applyInverseJacobianF", nb::overload_cast<const Vec3d&>(&Map::template applyInverseJacobianF<Vec3d>, nb::const_), "xyz"_a)
        .def("applyIJT", nb::overload_cast<const Vec3f&>(&Map::template applyIJT<Vec3f>, nb::const_), "xyz"_a)
        .def("applyIJT", nb::overload_cast<const Vec3d&>(&Map::template applyIJT<Vec3d>, nb::const_), "xyz"_a)
        .def("applyIJTF", nb::overload_cast<const Vec3f&>(&Map::template applyIJTF<Vec3f>, nb::const_), "xyz"_a)
        .def("applyIJTF", nb::overload_cast<const Vec3d&>(&Map::template applyIJTF<Vec3d>, nb::const_), "xyz"_a)
        .def("getVoxelSize", &Map::getVoxelSize);
}

void defineGridData(nb::module_& m)
{
    nb::class_<GridData>(m, "GridData")
        .def("isValid", &GridData::isValid)
        .def("setMinMaxOn", &GridData::setMinMaxOn, "on"_a = true)
        .def("setBBoxOn", &GridData::setBBoxOn, "on"_a = true)
        .def("setLongGridNameOn", &GridData::setLongGridNameOn, "on"_a = true)
        .def("setAverageOn", &GridData::setAverageOn, "on"_a = true)
        .def("setStdDeviationOn", &GridData::setStdDeviationOn, "on"_a = true)
        .def("setGridName", &GridData::setGridName, "src"_a)
        .def("applyMap", nb::overload_cast<const Vec3f&>(&GridData::template applyMap<Vec3f>, nb::const_), "xyz"_a)
        .def("applyMap", nb::overload_cast<const Vec3d&>(&GridData::template applyMap<Vec3d>, nb::const_), "xyz"_a)
        .def("applyMapF", nb::overload_cast<const Vec3f&>(&GridData::template applyMapF<Vec3f>, nb::const_), "xyz"_a)
        .def("applyMapF", nb::overload_cast<const Vec3d&>(&GridData::template applyMapF<Vec3d>, nb::const_), "xyz"_a)
        .def("applyJacobian", nb::overload_cast<const Vec3f&>(&GridData::template applyJacobian<Vec3f>, nb::const_), "xyz"_a)
        .def("applyJacobian", nb::overload_cast<const Vec3d&>(&GridData::template applyJacobian<Vec3d>, nb::const_), "xyz"_a)
        .def("applyJacobianF", nb::overload_cast<const Vec3f&>(&GridData::template applyJacobianF<Vec3f>, nb::const_), "xyz"_a)
        .def("applyJacobianF", nb::overload_cast<const Vec3d&>(&GridData::template applyJacobianF<Vec3d>, nb::const_), "xyz"_a)
        .def("applyInverseMap", nb::overload_cast<const Vec3f&>(&GridData::template applyInverseMap<Vec3f>, nb::const_), "xyz"_a)
        .def("applyInverseMap", nb::overload_cast<const Vec3d&>(&GridData::template applyInverseMap<Vec3d>, nb::const_), "xyz"_a)
        .def("applyInverseMapF", nb::overload_cast<const Vec3f&>(&GridData::template applyInverseMapF<Vec3f>, nb::const_), "xyz"_a)
        .def("applyInverseMapF", nb::overload_cast<const Vec3d&>(&GridData::template applyInverseMapF<Vec3d>, nb::const_), "xyz"_a)
        .def("applyInverseJacobian", nb::overload_cast<const Vec3f&>(&GridData::template applyInverseJacobian<Vec3f>, nb::const_), "xyz"_a)
        .def("applyInverseJacobian", nb::overload_cast<const Vec3d&>(&GridData::template applyInverseJacobian<Vec3d>, nb::const_), "xyz"_a)
        .def("applyInverseJacobianF", nb::overload_cast<const Vec3f&>(&GridData::template applyInverseJacobianF<Vec3f>, nb::const_), "xyz"_a)
        .def("applyInverseJacobianF", nb::overload_cast<const Vec3d&>(&GridData::template applyInverseJacobianF<Vec3d>, nb::const_), "xyz"_a)
        .def("applyIJT", nb::overload_cast<const Vec3f&>(&GridData::template applyIJT<Vec3f>, nb::const_), "xyz"_a)
        .def("applyIJT", nb::overload_cast<const Vec3d&>(&GridData::template applyIJT<Vec3d>, nb::const_), "xyz"_a)
        .def("applyIJTF", nb::overload_cast<const Vec3f&>(&GridData::template applyIJTF<Vec3f>, nb::const_), "xyz"_a)
        .def("applyIJTF", nb::overload_cast<const Vec3d&>(&GridData::template applyIJTF<Vec3d>, nb::const_), "xyz"_a)
        .def("gridName", &GridData::gridName)
        .def("memUsage", &GridData::memUsage)
        .def("worldBBox", &GridData::worldBBox)
        .def("indexBBox", &GridData::indexBBox)
        .def("isEmpty", &GridData::isEmpty);
}

template<typename BuildT> void defineGrid(nb::module_& m, const char* name)
{
    nb::class_<NanoGrid<BuildT>, GridData>(m, name)
        .def("version", &NanoGrid<BuildT>::version)
        .def("memUsage", &NanoGrid<BuildT>::memUsage)
        .def("gridSize", &NanoGrid<BuildT>::gridSize)
        .def("gridIndex", &NanoGrid<BuildT>::gridIndex)
        .def("gridCount", &NanoGrid<BuildT>::gridCount)
        .def("getAccessor", &NanoGrid<BuildT>::getAccessor)
        .def("voxelSize", &NanoGrid<BuildT>::voxelSize)
        .def("map", &NanoGrid<BuildT>::map)
        .def("worldBBox", &NanoGrid<BuildT>::worldBBox)
        .def("indexBBox", &NanoGrid<BuildT>::indexBBox)
        .def("activeVoxelCount", &NanoGrid<BuildT>::activeVoxelCount)
        .def("isValid", &NanoGrid<BuildT>::isValid)
        .def("gridType", &NanoGrid<BuildT>::gridType)
        .def("gridClass", &NanoGrid<BuildT>::gridClass)
        .def("isLevelSet", &NanoGrid<BuildT>::isLevelSet)
        .def("isFogVolume", &NanoGrid<BuildT>::isFogVolume)
        .def("isStaggered", &NanoGrid<BuildT>::isStaggered)
        .def("isPointIndex", &NanoGrid<BuildT>::isPointIndex)
        .def("isGridIndex", &NanoGrid<BuildT>::isGridIndex)
        .def("isPointData", &NanoGrid<BuildT>::isPointData)
        .def("isMask", &NanoGrid<BuildT>::isMask)
        .def("isUnknown", &NanoGrid<BuildT>::isUnknown)
        .def("hasMinMax", &NanoGrid<BuildT>::hasMinMax)
        .def("hasBBox", &NanoGrid<BuildT>::hasBBox)
        .def("hasLongGridName", &NanoGrid<BuildT>::hasLongGridName)
        .def("hasAverage", &NanoGrid<BuildT>::hasAverage)
        .def("hasStdDeviation", &NanoGrid<BuildT>::hasStdDeviation)
        .def("isBreadthFirst", &NanoGrid<BuildT>::isBreadthFirst)
        // .def("isLexicographic", &NanoGrid<BuildT>::isLexicographic)
        .def("isSequential", [](const NanoGrid<BuildT>& grid) { return grid.isSequential(); })
        .def("gridName", &NanoGrid<BuildT>::gridName)
        .def("shortGridName", &NanoGrid<BuildT>::shortGridName)
        .def("checksum", &NanoGrid<BuildT>::checksum)
        .def("isEmpty", &NanoGrid<BuildT>::isEmpty);
}

template<typename BuildT> nb::class_<DefaultReadAccessor<BuildT>> defineAccessor(nb::module_& m, const char* name)
{
    using ValueType = typename DefaultReadAccessor<BuildT>::ValueType;
    using CoordType = typename DefaultReadAccessor<BuildT>::CoordType;

    nb::class_<DefaultReadAccessor<BuildT>> accessor(m, name);
    accessor.def(nb::init<const NanoGrid<BuildT>&>(), "grid"_a)
        .def("getValue", nb::overload_cast<const CoordType&>(&DefaultReadAccessor<BuildT>::getValue, nb::const_), "ijk"_a)
        .def("getValue", nb::overload_cast<int, int, int>(&DefaultReadAccessor<BuildT>::getValue, nb::const_), "i"_a, "j"_a, "k"_a)
        .def(
            "__call__", [](const DefaultReadAccessor<BuildT>& accessor, const CoordType& ijk) { return accessor.getValue(ijk); }, nb::is_operator(), "ijk"_a)
        .def(
            "__call__",
            [](const DefaultReadAccessor<BuildT>& accessor, int i, int j, int k) { return accessor.getValue(i, j, k); },
            nb::is_operator(),
            "i"_a,
            "j"_a,
            "k"_a)
        .def("isActive", &DefaultReadAccessor<BuildT>::isActive, "ijk"_a)
        .def(
            "probeValue",
            [](const DefaultReadAccessor<BuildT>& accessor, const CoordType& ijk) {
                ValueType v;
                bool      isOn = accessor.probeValue(ijk, v);
                return std::make_tuple(v, isOn);
            },
            "ijk"_a);
    return accessor;
}

template<typename BuildT> void defineScalarAccessor(nb::module_& m, const char* name)
{
    using ValueType = typename DefaultReadAccessor<BuildT>::ValueType;
    using CoordType = typename DefaultReadAccessor<BuildT>::CoordType;

    defineAccessor<BuildT>(m, name)
        .def("getNodeInfo", &DefaultReadAccessor<BuildT>::getNodeInfo, "ijk"_a)
        .def(
            "setVoxel",
            [](DefaultReadAccessor<BuildT>& accessor, const CoordType& ijk, const ValueType& v) {
                using OpT = SetVoxel<BuildT>;
                accessor.template set<OpT>(ijk, v);
            },
            "ijk"_a,
            "v"_a);
}

template<typename BuildT> void defineVectorAccessor(nb::module_& m, const char* name)
{
    using ValueType = typename DefaultReadAccessor<BuildT>::ValueType;
    using CoordType = typename DefaultReadAccessor<BuildT>::CoordType;

    defineAccessor<BuildT>(m, name).def(
        "setVoxel",
        [](DefaultReadAccessor<BuildT>& accessor, const CoordType& ijk, const ValueType& v) {
            using OpT = SetVoxel<BuildT>;
            accessor.template set<OpT>(ijk, v);
        },
        "ijk"_a,
        "v"_a);
}

template<typename BuildT> void defineNodeInfo(nb::module_& m, const char* name)
{
    nb::class_<typename GetNodeInfo<BuildT>::NodeInfo>(m, name)
        .def_ro("level", &GetNodeInfo<BuildT>::NodeInfo::level)
        .def_ro("dim", &GetNodeInfo<BuildT>::NodeInfo::dim)
        .def_ro("minimum", &GetNodeInfo<BuildT>::NodeInfo::minimum)
        .def_ro("maximum", &GetNodeInfo<BuildT>::NodeInfo::maximum)
        .def_ro("average", &GetNodeInfo<BuildT>::NodeInfo::average)
        .def_ro("stdDevi", &GetNodeInfo<BuildT>::NodeInfo::stdDevi)
        .def_ro("bbox", &GetNodeInfo<BuildT>::NodeInfo::bbox);
}

bool isCudaAvailable()
{
#ifdef NANOVDB_USE_CUDA
    return true;
#else
    return false;
#endif
}

NB_MODULE(nanovdb, m)
{
    m.doc() = "Python module for NanoVDB";
    m.def("isCudaAvailable", &isCudaAvailable, "Returns whether or not the module was compiled with CUDA support");

    nb::enum_<GridType>(m, "GridType")
        .value("Unknown", GridType::Unknown)
        .value("Float", GridType::Float)
        .value("Double", GridType::Double)
        .value("Int16", GridType::Int16)
        .value("Int32", GridType::Int32)
        .value("Int64", GridType::Int64)
        .value("Vec3f", GridType::Vec3f)
        .value("Vec3d", GridType::Vec3d)
        .value("Mask", GridType::Mask)
        .value("Half", GridType::Half)
        .value("UInt32", GridType::UInt32)
        .value("Boolean", GridType::Boolean)
        .value("RGBA8", GridType::RGBA8)
        .value("Fp4", GridType::Fp4)
        .value("Fp8", GridType::Fp8)
        .value("Fp16", GridType::Fp16)
        .value("FpN", GridType::FpN)
        .value("Vec4f", GridType::Vec4f)
        .value("Vec4d", GridType::Vec4d)
        .value("Index", GridType::Index)
        .value("OnIndex", GridType::OnIndex)
        .value("PointIndex", GridType::PointIndex)
        .value("Vec3u8", GridType::Vec3u8)
        .value("Vec3u16", GridType::Vec3u16)
        .value("End", GridType::End)
        .export_values();
        // .def("__repr__", [](const GridType& gridType) {
        //     char str[strlen<GridType>()];
        //     toStr(str, gridType);
        //     return std::string(str);
        // });

    nb::enum_<GridClass>(m, "GridClass")
        .value("Unknown", GridClass::Unknown)
        .value("LevelSet", GridClass::LevelSet)
        .value("FogVolume", GridClass::FogVolume)
        .value("Staggered", GridClass::Staggered)
        .value("PointIndex", GridClass::PointIndex)
        .value("PointData", GridClass::PointData)
        .value("Topology", GridClass::Topology)
        .value("VoxelVolume", GridClass::VoxelVolume)
        .value("IndexGrid", GridClass::IndexGrid)
        .value("TensorGrid", GridClass::TensorGrid)
        .value("End", GridClass::End)
        .export_values();
        // .def("__repr__", [](const GridClass& gridClass) {
        //     char str[strlen<GridClass>()];
        //     toStr(str, gridClass);
        //     return std::string(str);
        // });

    defineVersion(m);

    nb::module_ mathModule = m.def_submodule("math");
    mathModule.doc() = "A submodule that implements math functions";
    defineMathModule(mathModule);

    definePointTypes(m);

    defineMask<5>(m, "UpperInternalNodeMask", "Bit-mask for encoding active states of upper internal nodes");
    defineMask<4>(m, "LowerInternalNodeMask", "Bit-mask for encoding active states of lower internal nodes");
    defineMask<3>(m, "LeafMask", "Bit-mask for encoding active states of leaf nodes");

    defineMap(m);

    defineGridData(m);

    defineGrid<float>(m, "FloatGrid");
    defineScalarAccessor<float>(m, "FloatReadAccessor");
    defineNodeInfo<float>(m, "FloatNodeInfo");

    defineGrid<double>(m, "DoubleGrid");
    defineScalarAccessor<double>(m, "DoubleReadAccessor");
    defineNodeInfo<double>(m, "DoubleNodeInfo");

    defineGrid<int32_t>(m, "Int32Grid");
    defineScalarAccessor<int32_t>(m, "Int32ReadAccessor");
    defineNodeInfo<int32_t>(m, "Int32NodeInfo");

    defineGrid<Vec3f>(m, "Vec3fGrid");
    defineVectorAccessor<Vec3f>(m, "Vec3fReadVectorAccessor");

    defineGrid<math::Rgba8>(m, "RGBA8Grid");
    defineVectorAccessor<math::Rgba8>(m, "RGBA8ReadAccessor");

    defineGrid<Point>(m, "PointGrid");
    defineAccessor<Point>(m, "PointReadAccessor");

    defineHostBuffer(m);
    defineHostGridHandle(m);

    defineCheckMode(m);
    defineChecksum(m);

#ifdef NANOVDB_USE_CUDA
    defineDeviceBuffer(m);
    defineDeviceGridHandle(m);
#endif

    nb::module_ toolsModule = m.def_submodule("tools");
    toolsModule.doc() = "A submodule that implements tools for NanoVDB grids";
    defineToolsModule(toolsModule);

    nb::module_ ioModule = m.def_submodule("io");
    ioModule.doc() = "A submodule that implements I/O functionality for NanoVDB grids";
    defineIOModule(ioModule);
}

} // namespace pynanovdb
