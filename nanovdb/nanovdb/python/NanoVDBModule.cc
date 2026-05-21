// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
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
#include "PyBuildGrid.h"
#include "PyGridHandle.h"
#include "PyHostBuffer.h"
#include "PyIO.h"
#include "PyMath.h"
#include "PyTools.h"
#include "PyTree.h"
#include "PyVoxelBlockManager.h"
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

// Forward declaration — body lives below defineGridBlindData() so it can
// reference the enum / class bindings registered there.
static nb::object pyGetBlindData(nb::handle py_grid, uint32_t n);

// Type-erased Grid base class. nb::class_ is bound to nanovdb::GridData
// (the 672 B POD prefix present at the start of every NanoGrid<BuildT>),
// but the Python name is "Grid" to match the C++ class hierarchy where
// Grid<TreeT> (a.k.a. NanoGrid<BuildT>) is the user-facing type and
// GridData is the implementation-detail POD.
//
// Every BuildT-independent method lives here. The lambdas below read public
// data members on GridData directly because the accessor methods named
// version()/gridType()/isLevelSet()/etc. are defined on Grid<TreeT>, not on
// GridData itself — but they all just forward to a GridData data member,
// so the same value is reachable from the base.
void defineGrid(nb::module_& m)
{
    nb::class_<GridData>(m, "Grid")
        // Validation and flag mutators (already member functions on GridData).
        .def("isValid", &GridData::isValid)
        .def("setMinMaxOn", &GridData::setMinMaxOn, "on"_a = true)
        .def("setBBoxOn", &GridData::setBBoxOn, "on"_a = true)
        .def("setLongGridNameOn", &GridData::setLongGridNameOn, "on"_a = true)
        .def("setAverageOn", &GridData::setAverageOn, "on"_a = true)
        .def("setStdDeviationOn", &GridData::setStdDeviationOn, "on"_a = true)
        .def("setGridName", &GridData::setGridName, "src"_a)
        // Affine transforms (already member functions on GridData).
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
        // Strings, geometry, layout (already member functions on GridData).
        .def("gridName", &GridData::gridName)
        .def("memUsage", &GridData::memUsage)
        .def("worldBBox", &GridData::worldBBox)
        .def("indexBBox", &GridData::indexBBox)
        .def("isEmpty", &GridData::isEmpty)
        // Lifted from Grid<TreeT> via direct data-member access.
        .def("version",         [](const GridData& g) { return g.mVersion; })
        .def("gridSize",        [](const GridData& g) { return g.mGridSize; })
        .def("gridIndex",       [](const GridData& g) { return g.mGridIndex; })
        .def("gridCount",       [](const GridData& g) { return g.mGridCount; })
        .def("voxelSize",       [](const GridData& g) -> const Vec3d& { return g.mVoxelSize; },
             nb::rv_policy::reference_internal)
        .def("map",             [](const GridData& g) -> const Map& { return g.mMap; },
             nb::rv_policy::reference_internal)
        .def("gridType",  [](const GridData& g) { return g.mGridType; })
        .def("gridClass", [](const GridData& g) { return g.mGridClass; })
        .def("checksum",  [](const GridData& g) { return g.mChecksum; })
        .def("isLevelSet",   [](const GridData& g) { return g.mGridClass == GridClass::LevelSet; })
        .def("isFogVolume",  [](const GridData& g) { return g.mGridClass == GridClass::FogVolume; })
        .def("isStaggered",  [](const GridData& g) { return g.mGridClass == GridClass::Staggered; })
        .def("isPointIndex",
             [](const GridData& g) { return g.mGridClass == GridClass::PointIndex; })
        .def("isGridIndex",  [](const GridData& g) { return g.mGridClass == GridClass::IndexGrid; })
        .def("isPointData",  [](const GridData& g) { return g.mGridClass == GridClass::PointData; })
        .def("isMask",       [](const GridData& g) { return g.mGridClass == GridClass::Topology; })
        .def("isUnknown",    [](const GridData& g) { return g.mGridClass == GridClass::Unknown; })
        .def("hasMinMax", [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::HasMinMax); })
        .def("hasBBox",   [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::HasBBox); })
        .def("hasLongGridName",
             [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::HasLongGridName); })
        .def("hasAverage",
             [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::HasAverage); })
        .def("hasStdDeviation",
             [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::HasStdDeviation); })
        .def("isBreadthFirst",
             [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::IsBreadthFirst); })
        .def("shortGridName", [](const GridData& g) { return std::string(g.mGridName); })
        // Blind data — exposes the sidecar channels that PointGrid and
        // OnIndexGrid use to carry their actual values, colors, normals, IDs,
        // etc. blindMetaData(n) returns the descriptor; getBlindData(n)
        // returns a zero-copy NumPy view onto the underlying bytes typed by
        // mDataType (Float -> float32 ndarray, Vec3f -> (N, 3) float32, etc.;
        // unrecognized types fall back to a flat uint8 byte view).
        .def("blindDataCount",          [](const GridData& g) { return g.mBlindMetadataCount; })
        .def("blindMetaData",
             [](const GridData& g, uint32_t n) -> const GridBlindMetaData* {
                 return n < g.mBlindMetadataCount ? g.blindMetaData(n) : nullptr;
             },
             nb::rv_policy::reference_internal, "n"_a)
        .def("findBlindData",           [](const GridData& g, const std::string& name) -> int {
            for (uint32_t i = 0; i < g.mBlindMetadataCount; ++i) {
                const auto* meta = g.blindMetaData(i);
                if (std::strncmp(meta->mName, name.c_str(), GridBlindMetaData::MaxNameSize) == 0)
                    return static_cast<int>(i);
            }
            return -1;
        }, "name"_a)
        .def("findBlindDataForSemantic", [](const GridData& g, GridBlindDataSemantic sem) -> int {
            for (uint32_t i = 0; i < g.mBlindMetadataCount; ++i) {
                if (g.blindMetaData(i)->mSemantic == sem)
                    return static_cast<int>(i);
            }
            return -1;
        }, "semantic"_a)
        .def("getBlindData", &pyGetBlindData, "n"_a,
             nb::keep_alive<0, 1>(),
             "Return a zero-copy NumPy view of the n-th blind data channel, "
             "or None if n is out of range. dtype and shape are derived from "
             "the channel's mDataType / mValueCount. The view keeps the grid "
             "alive (and therefore the GridHandle that owns the buffer).");
}

// BuildT-dependent slice of the typed grid Python class. Inherits the
// type-erased Grid base bound by defineGrid() above — anything that doesn't
// need to know BuildT lives there, not here.
template<typename BuildT> void defineNanoGrid(nb::module_& m, const char* name)
{
    auto cls = nb::class_<NanoGrid<BuildT>, GridData>(m, name)
        .def("getAccessor", &NanoGrid<BuildT>::getAccessor)
        .def("activeVoxelCount", &NanoGrid<BuildT>::activeVoxelCount)
        .def("isSequential", [](const NanoGrid<BuildT>& grid) { return grid.isSequential(); })
        .def("tree",
             nb::overload_cast<>(&NanoGrid<BuildT>::tree, nb::const_),
             nb::rv_policy::reference_internal,
             "Return the tree associated with this grid. Lifetime is "
             "anchored to the grid (and therefore to the GridHandle).");
    // Add leaf_values() only for BuildTs whose LeafData carries T mValues[512].
    PyLeafValuesBinder<BuildT>::apply(cls);
}

void defineGridBlindData(nb::module_& m)
{
    nb::enum_<GridBlindDataClass>(m, "GridBlindDataClass")
        .value("Unknown",        GridBlindDataClass::Unknown)
        .value("IndexArray",     GridBlindDataClass::IndexArray)
        .value("AttributeArray", GridBlindDataClass::AttributeArray)
        .value("GridName",       GridBlindDataClass::GridName)
        .value("ChannelArray",   GridBlindDataClass::ChannelArray)
        .value("End",            GridBlindDataClass::End)
        .export_values();

    nb::enum_<GridBlindDataSemantic>(m, "GridBlindDataSemantic")
        .value("Unknown",       GridBlindDataSemantic::Unknown)
        .value("PointPosition", GridBlindDataSemantic::PointPosition)
        .value("PointColor",    GridBlindDataSemantic::PointColor)
        .value("PointNormal",   GridBlindDataSemantic::PointNormal)
        .value("PointRadius",   GridBlindDataSemantic::PointRadius)
        .value("PointVelocity", GridBlindDataSemantic::PointVelocity)
        .value("PointId",       GridBlindDataSemantic::PointId)
        .value("WorldCoords",   GridBlindDataSemantic::WorldCoords)
        .value("GridCoords",    GridBlindDataSemantic::GridCoords)
        .value("VoxelCoords",   GridBlindDataSemantic::VoxelCoords)
        .value("LevelSet",      GridBlindDataSemantic::LevelSet)
        .value("FogVolume",     GridBlindDataSemantic::FogVolume)
        .value("Staggered",     GridBlindDataSemantic::Staggered)
        .value("End",           GridBlindDataSemantic::End)
        .export_values();

    nb::class_<GridBlindMetaData>(m, "GridBlindMetaData",
                                  "Sidecar metadata for one blind-data channel attached to a Grid.")
        .def_ro("valueCount", &GridBlindMetaData::mValueCount)
        .def_ro("valueSize",  &GridBlindMetaData::mValueSize)
        .def_ro("semantic",   &GridBlindMetaData::mSemantic)
        .def_ro("dataClass",  &GridBlindMetaData::mDataClass)
        .def_ro("dataType",   &GridBlindMetaData::mDataType)
        .def("name",  [](const GridBlindMetaData& m) { return std::string(m.mName); })
        .def("isValid", &GridBlindMetaData::isValid)
        .def("blindDataSize", &GridBlindMetaData::blindDataSize);
}

// Resolve a blind-data channel into a zero-copy NumPy view. The dtype and
// shape are derived from the GridBlindMetaData's mDataType / mValueSize.
// For unrecognized types we fall back to a flat uint8 byte view so callers
// can still copy out the raw bytes. Falls back to None on a count mismatch
// between mValueSize and the GridType-implied stride.
static nb::object pyGetBlindData(nb::handle py_grid, uint32_t n)
{
    const auto& grid = nb::cast<const GridData&>(py_grid);
    if (n >= grid.mBlindMetadataCount) return nb::none();
    const auto* meta = grid.blindMetaData(n);
    void* data = const_cast<void*>(static_cast<const void*>(
        util::PtrAdd<uint8_t>(meta, meta->mDataOffset)));
    const size_t count = static_cast<size_t>(meta->mValueCount);
    const uint32_t valueSize = meta->mValueSize;

    auto make1D = [&](void* p, size_t n_elems, auto sentinel) -> nb::object {
        using T = decltype(sentinel);
        size_t shape[1] = {n_elems};
        return nb::cast(nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>(
                            static_cast<T*>(p), 1, shape, py_grid),
                        nb::rv_policy::reference);
    };
    auto make2D = [&](void* p, size_t n_outer, size_t n_inner, auto sentinel) -> nb::object {
        using T = decltype(sentinel);
        size_t shape[2] = {n_outer, n_inner};
        return nb::cast(nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>(
                            static_cast<T*>(p), 2, shape, py_grid),
                        nb::rv_policy::reference);
    };
    // Raw byte view fallback. Used either when the data type is unknown OR
    // when the recorded mValueSize doesn't match the stride implied by
    // mDataType — in that case constructing a typed ndarray would overrun the
    // underlying blind-data region. mValueCount * mValueSize is by definition
    // the actual byte extent of the channel, so this is always safe.
    auto raw = [&]() -> nb::object {
        return make1D(data, count * valueSize, uint8_t{});
    };

    switch (meta->mDataType) {
        case GridType::Float:
            return valueSize == sizeof(float) ? make1D(data, count, float{}) : raw();
        case GridType::Double:
            return valueSize == sizeof(double) ? make1D(data, count, double{}) : raw();
        case GridType::Int16:
            return valueSize == sizeof(int16_t) ? make1D(data, count, int16_t{}) : raw();
        case GridType::Int32:
            return valueSize == sizeof(int32_t) ? make1D(data, count, int32_t{}) : raw();
        case GridType::Int64:
            return valueSize == sizeof(int64_t) ? make1D(data, count, int64_t{}) : raw();
        case GridType::UInt8:
            return valueSize == sizeof(uint8_t) ? make1D(data, count, uint8_t{}) : raw();
        case GridType::UInt32:
            return valueSize == sizeof(uint32_t) ? make1D(data, count, uint32_t{}) : raw();
        case GridType::Vec3f:
            return valueSize == 3 * sizeof(float) ? make2D(data, count, 3, float{}) : raw();
        case GridType::Vec3d:
            return valueSize == 3 * sizeof(double) ? make2D(data, count, 3, double{}) : raw();
        case GridType::Vec4f:
            return valueSize == 4 * sizeof(float) ? make2D(data, count, 4, float{}) : raw();
        case GridType::Vec4d:
            return valueSize == 4 * sizeof(double) ? make2D(data, count, 4, double{}) : raw();
        case GridType::Vec3u8:
            return valueSize == 3 * sizeof(uint8_t) ? make2D(data, count, 3, uint8_t{}) : raw();
        case GridType::Vec3u16:
            return valueSize == 3 * sizeof(uint16_t) ? make2D(data, count, 3, uint16_t{}) : raw();
        case GridType::RGBA8:
            return valueSize == 4 * sizeof(uint8_t) ? make2D(data, count, 4, uint8_t{}) : raw();
        default:
            return raw();
    }
}

// PointAccessor — exposes the per-voxel point attributes that PointGrid
// carries as blind data. PointIndex grids store uint32 voxel indices;
// PointData grids store Vec3f positions. Constructor asserts the grid is
// the right shape; in Python an exception is the result of a mismatch.
//
// gridPoints() / leafPoints(ijk) / voxelPoints(ijk) all return a zero-copy
// NumPy view onto the underlying blind-data buffer, sliced to just the
// range associated with the call. Lifetime is anchored to the accessor.
template<typename AttT, int Dim>
struct PyPointAccessorTraits;
template<> struct PyPointAccessorTraits<uint32_t, 1> { using Scalar = uint32_t; };
template<> struct PyPointAccessorTraits<Vec3f, 2>    { using Scalar = float; };

template<typename AttT>
static nb::object pyPointsToNdarray(nb::handle py_self,
                                    const AttT* begin,
                                    uint64_t count);

template<>
nb::object pyPointsToNdarray<uint32_t>(nb::handle py_self,
                                       const uint32_t* begin,
                                       uint64_t count)
{
    size_t shape[1] = {static_cast<size_t>(count)};
    return nb::cast(
        nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>(
            const_cast<uint32_t*>(begin), 1, shape, py_self),
        nb::rv_policy::reference);
}

template<>
nb::object pyPointsToNdarray<Vec3f>(nb::handle py_self,
                                    const Vec3f* begin,
                                    uint64_t count)
{
    size_t shape[2] = {static_cast<size_t>(count), 3};
    return nb::cast(
        nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>(
            reinterpret_cast<float*>(const_cast<Vec3f*>(begin)), 2, shape, py_self),
        nb::rv_policy::reference);
}

template<typename AttT> void definePointAccessor(nb::module_& m, const char* name)
{
    using PA = PointAccessor<AttT, Point>;
    nb::class_<PA>(m, name,
                   "Per-voxel access to the point attributes carried as blind "
                   "data on a PointGrid. gridPoints / leafPoints / voxelPoints "
                   "return zero-copy NumPy views.")
        .def(nb::init<const NanoGrid<Point>&>(), "grid"_a, nb::keep_alive<1, 2>())
        .def("__bool__", [](const PA& a) { return bool(a); })
        .def("grid", &PA::grid, nb::rv_policy::reference_internal)
        .def("gridPoints", [](nb::handle py_self) -> nb::object {
            auto& acc = nb::cast<PA&>(py_self);
            const AttT* begin = nullptr;
            const AttT* end = nullptr;
            uint64_t count = acc.gridPoints(begin, end);
            if (begin == nullptr || count == 0) return nb::none();
            return pyPointsToNdarray<AttT>(py_self, begin, count);
        }, nb::keep_alive<0, 1>(),
           "Return all point attributes in the grid as a single NumPy view. "
           "The view keeps this accessor alive.")
        .def("leafPoints", [](nb::handle py_self, const Coord& ijk) -> nb::object {
            auto& acc = nb::cast<PA&>(py_self);
            const AttT* begin = nullptr;
            const AttT* end = nullptr;
            uint64_t count = acc.leafPoints(ijk, begin, end);
            if (begin == nullptr || count == 0) return nb::none();
            return pyPointsToNdarray<AttT>(py_self, begin, count);
        }, "ijk"_a, nb::keep_alive<0, 1>(),
           "Return the point attributes contained within the leaf node "
           "covering ijk, or None if no leaf is present. The view keeps "
           "this accessor alive.")
        .def("voxelPoints", [](nb::handle py_self, const Coord& ijk) -> nb::object {
            auto& acc = nb::cast<PA&>(py_self);
            const AttT* begin = nullptr;
            const AttT* end = nullptr;
            uint64_t count = acc.voxelPoints(ijk, begin, end);
            if (begin == nullptr || count == 0) return nb::none();
            return pyPointsToNdarray<AttT>(py_self, begin, count);
        }, "ijk"_a, nb::keep_alive<0, 1>(),
           "Return the point attributes at the specific voxel ijk, or None "
           "if the voxel is inactive / empty. The view keeps this accessor "
           "alive.");
}

// Type-erased grid introspector. Mirrors nanovdb::GridMetaData (768B) and
// answers "what's in this buffer?" questions without needing to know
// BuildT. Construct from a Grid (which is the Python-side GridData); all
// queries below are flat data-member reads with no tree traversal.
void defineGridMetaData(nb::module_& m)
{
    // Constructing GridMetaData calls into nanovdb::GridMetaData::safeCast
    // which has a NANOVDB_ASSERT(gridData && gridData->isValid()). nanobind
    // already rejects Python None at the type-check level (None can't bind to
    // const GridData*), but a Grid wrapping a corrupted / partially-formed
    // buffer would still abort debug builds and undefined-behave in release.
    // Guard explicitly: validate first, raise nb::value_error on bad input.
    nb::class_<GridMetaData>(m, "GridMetaData",
                             "Type-erased introspector. Mirrors FileMetaData "
                             "but reads from an in-memory grid header.")
        .def("__init__",
             [](GridMetaData* self, const GridData* gd) {
                 if (gd == nullptr) {
                     throw nb::value_error("GridMetaData: grid must not be None");
                 }
                 if (!gd->isValid()) {
                     throw nb::value_error("GridMetaData: grid header is invalid "
                                           "(bad magic, version, or class/type tags)");
                 }
                 new (self) GridMetaData(gd);
             }, "grid"_a)
        .def_static("safeCast",
                    [](const GridData* gd) {
                        // Mirror the spirit of NanoVDB's static safeCast: "is
                        // it safe to cast this gridData to a GridMetaData?".
                        // null and invalid grids are by definition not safe;
                        // return False rather than dereference.
                        if (gd == nullptr || !gd->isValid()) return false;
                        return GridMetaData::safeCast(gd);
                    }, "grid"_a)
        .def("isValid", &GridMetaData::isValid)
        .def("gridType", &GridMetaData::gridType)
        .def("gridClass", &GridMetaData::gridClass)
        .def("isLevelSet", &GridMetaData::isLevelSet)
        .def("isFogVolume", &GridMetaData::isFogVolume)
        .def("isStaggered", &GridMetaData::isStaggered)
        .def("isPointIndex", &GridMetaData::isPointIndex)
        .def("isGridIndex", &GridMetaData::isGridIndex)
        .def("isPointData", &GridMetaData::isPointData)
        .def("isMask", &GridMetaData::isMask)
        .def("isUnknown", &GridMetaData::isUnknown)
        .def("hasMinMax", &GridMetaData::hasMinMax)
        .def("hasBBox", &GridMetaData::hasBBox)
        .def("hasLongGridName", &GridMetaData::hasLongGridName)
        .def("hasAverage", &GridMetaData::hasAverage)
        .def("hasStdDeviation", &GridMetaData::hasStdDeviation)
        .def("isBreadthFirst", &GridMetaData::isBreadthFirst)
        .def("gridSize", &GridMetaData::gridSize)
        .def("gridIndex", &GridMetaData::gridIndex)
        .def("gridCount", &GridMetaData::gridCount)
        .def("shortGridName", [](const GridMetaData& m) { return std::string(m.shortGridName()); })
        .def("map", &GridMetaData::map, nb::rv_policy::reference_internal)
        .def("worldBBox", &GridMetaData::worldBBox, nb::rv_policy::reference_internal)
        .def("indexBBox", &GridMetaData::indexBBox, nb::rv_policy::reference_internal)
        .def("voxelSize", &GridMetaData::voxelSize)
        .def("blindDataCount", &GridMetaData::blindDataCount)
        .def("activeVoxelCount", &GridMetaData::activeVoxelCount)
        .def("activeTileCount", &GridMetaData::activeTileCount, "level"_a)
        .def("nodeCount", &GridMetaData::nodeCount, "level"_a)
        .def("checksum", &GridMetaData::checksum, nb::rv_policy::reference_internal)
        .def("rootTableSize", &GridMetaData::rootTableSize)
        .def("isEmpty", &GridMetaData::isEmpty)
        .def("version", &GridMetaData::version);
}

template<typename BuildT> nb::class_<DefaultReadAccessor<BuildT>> defineAccessor(nb::module_& m, const char* name)
{
    // Use the decoded value type (nanovdb::BuildToValueMap<BuildT>::Type)
    // rather than DefaultReadAccessor<BuildT>::ValueType. For ordinary types
    // (float/double/Int*/Vec*) the two are identical, but for Half / Fp* the
    // accessor decodes to float on read, for ValueIndex/OnIndex it returns
    // uint64, and for ValueMask / bool it returns bool. The probeValue out-
    // parameter and the Python return type both want the decoded form.
    using ValueType = typename nanovdb::BuildToValueMap<BuildT>::Type;
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

bool isGpuAvailable()
{
#ifdef NANOVDB_USE_CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return deviceCount > 0;
#else
    return false;
#endif
}

NB_MODULE(nanovdb, m)
{
    m.doc() = "Python module for NanoVDB";
    m.def("isCudaAvailable", &isCudaAvailable, "Returns whether or not the module was compiled with CUDA support");
    m.def("isGpuAvailable", &isGpuAvailable, "Returns whether a CUDA-capable GPU is available at runtime");

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
        .value("UInt8", GridType::UInt8)
        .value("End", GridType::End)
        .export_values()
        .def("__repr__", [](const GridType& gridType) {
            char str[strlen<GridType>()];
            toStr(str, gridType);
            return std::string(str);
        });

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
        .export_values()
        .def("__repr__", [](const GridClass& gridClass) {
            char str[strlen<GridClass>()];
            toStr(str, gridClass);
            return std::string(str);
        });

    defineVersion(m);

    nb::module_ mathModule = m.def_submodule("math");
    mathModule.doc() = "A submodule that implements math functions";
    defineMathModule(mathModule);

    definePointTypes(m);

    defineMask<5>(m, "UpperInternalNodeMask", "Bit-mask for encoding active states of upper internal nodes");
    defineMask<4>(m, "LowerInternalNodeMask", "Bit-mask for encoding active states of lower internal nodes");
    defineMask<3>(m, "LeafMask", "Bit-mask for encoding active states of leaf nodes");

    defineMap(m);

    // CheckMode + Checksum must be bound before defineGrid() because
    // Grid.checksum() returns Checksum by value.
    defineCheckMode(m);
    defineChecksum(m);

    // GridBlindData enums + GridBlindMetaData class — must be bound before
    // defineGrid() because Grid.blindMetaData()/findBlindDataForSemantic()
    // use them in their signatures.
    defineGridBlindData(m);
    defineGrid(m);
    defineGridMetaData(m);

    // Tree / node bindings must come BEFORE defineNanoGrid because
    // NanoGrid<T>.tree() returns NanoTree<T> (registered here) by const
    // reference. Per-BuildT, register Leaf, Lower, Upper, Root, Tree in
    // child->parent order so each return type is registered before the
    // method binding that returns it.
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum) \
    defineNanoLeaf<T>(m,  #Suffix "Leaf");                         \
    defineNanoLower<T>(m, #Suffix "Lower");                        \
    defineNanoUpper<T>(m, #Suffix "Upper");                        \
    defineNanoRoot<T>(m,  #Suffix "Root");                         \
    defineNanoTree<T>(m,  #Suffix "Tree");                         \
    defineNodeManager<T>(m, #Suffix "NodeManager");
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
    defineNanoLeaf<T>(m,  #Suffix "Leaf");                         \
    defineNanoLower<T>(m, #Suffix "Lower");                        \
    defineNanoUpper<T>(m, #Suffix "Upper");                        \
    defineNanoRoot<T>(m,  #Suffix "Root");                         \
    defineNanoTree<T>(m,  #Suffix "Tree");                         \
    defineNodeManager<T>(m, #Suffix "NodeManager");
#define NANOVDB_PY_FOR_EACH_POINT_BUILDT(T, Suffix, GridTypeEnum)  \
    defineNanoLeaf<T>(m,  #Suffix "Leaf");                         \
    defineNanoLower<T>(m, #Suffix "Lower");                        \
    defineNanoUpper<T>(m, #Suffix "Upper");                        \
    defineNanoRoot<T>(m,  #Suffix "Root");                         \
    defineNanoTree<T>(m,  #Suffix "Tree");                         \
    defineNodeManager<T>(m, #Suffix "NodeManager");
#define NANOVDB_PY_FOR_EACH_READONLY_BUILDT(T, Suffix, GridTypeEnum) \
    defineNanoLeaf<T>(m,  #Suffix "Leaf");                         \
    defineNanoLower<T>(m, #Suffix "Lower");                        \
    defineNanoUpper<T>(m, #Suffix "Upper");                        \
    defineNanoRoot<T>(m,  #Suffix "Root");                         \
    defineNanoTree<T>(m,  #Suffix "Tree");                         \
    defineNodeManager<T>(m, #Suffix "NodeManager");
#include "BuildTypes.def"

    // Now bind the per-BuildT NanoGrid + accessors (tree() return type now
    // registered above).
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum) \
    defineNanoGrid<T>(m, #Suffix "Grid");                      \
    defineScalarAccessor<T>(m, #Suffix "ReadAccessor");        \
    defineNodeInfo<T>(m, #Suffix "NodeInfo");
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
    defineNanoGrid<T>(m, #Suffix "Grid");                      \
    defineVectorAccessor<T>(m, AccessorName);
#define NANOVDB_PY_FOR_EACH_POINT_BUILDT(T, Suffix, GridTypeEnum) \
    defineNanoGrid<T>(m, #Suffix "Grid");                      \
    defineAccessor<T>(m, #Suffix "ReadAccessor");
#define NANOVDB_PY_FOR_EACH_READONLY_BUILDT(T, Suffix, GridTypeEnum) \
    defineNanoGrid<T>(m, #Suffix "Grid");                      \
    defineAccessor<T>(m, #Suffix "ReadAccessor");
#include "BuildTypes.def"

    // Host-side NodeManagerHandle + module-scope createNodeManager.
    defineNodeManagerHandle(m);
    defineCreateNodeManager(m);

    // PointAccessor variants — PointIndex grids carry uint32 indices,
    // PointData grids carry Vec3f positions.
    definePointAccessor<uint32_t>(m, "PointIndexAccessor");
    definePointAccessor<Vec3f>(m,    "PointDataAccessor");

    defineHostBuffer(m);
    defineHostGridHandle(m);

#ifdef NANOVDB_USE_CUDA
    defineDeviceBuffer(m);
    defineDeviceGridHandle(m);
#endif

    nb::module_ toolsModule = m.def_submodule("tools");
    toolsModule.doc() = "A submodule that implements tools for NanoVDB grids";
    defineToolsModule(toolsModule);
    defineVoxelBlockManagerModule(toolsModule);
    defineBuildGridModule(toolsModule);

    nb::module_ ioModule = m.def_submodule("io");
    ioModule.doc() = "A submodule that implements I/O functionality for NanoVDB grids";
    defineIOModule(ioModule);
}

} // namespace pynanovdb
