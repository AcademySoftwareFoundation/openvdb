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
        .def(nb::init<>(),
             "Construct a default-initialized Version matching the running NanoVDB build.")
        .def(nb::init<uint32_t>(), "data"_a,
             "Construct a Version from a raw bit-packed uint32 value.")
        .def(nb::init<uint32_t, uint32_t, uint32_t>(), "major"_a, "minor"_a, "patch"_a,
             "Construct a Version from explicit major, minor, and patch numbers.")
        .def(nb::self == nb::self, "rhs"_a,
             "Equality of major, minor and patch numbers.")
        .def(nb::self < nb::self, "rhs"_a,
             "Lexicographic less-than comparison over (major, minor, patch).")
        .def(nb::self <= nb::self, "rhs"_a,
             "Lexicographic less-than-or-equal comparison.")
        .def(nb::self > nb::self, "rhs"_a,
             "Lexicographic greater-than comparison.")
        .def(nb::self >= nb::self, "rhs"_a,
             "Lexicographic greater-than-or-equal comparison.")
        .def("id", &Version::id,
             "Return the bit-packed uint32 representation of this version.")
        .def("getMajor", &Version::getMajor,
             "Major version number.")
        .def("getMinor", &Version::getMinor,
             "Minor version number.")
        .def("getPatch", &Version::getPatch,
             "Patch version number.")
        .def("age", &Version::age,
             "Numeric age of this version relative to the running NanoVDB build.")
        .def("__repr__", [](const Version& version) {
            char str[strlen<Version>()];
            toStr(str, version);
            return std::string(str);
        });
}

void definePointTypes(nb::module_& m)
{
    nb::enum_<PointType>(m, "PointType",
        "Encoding selector for point attributes attached to a PointGrid. "
        "Controls the bit width and frame (world / grid / voxel) used when "
        "rasterising points to blind data.")
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
        .def_static("memUsage", &Mask<LOG2DIM>::memUsage,
                    "Byte size of a Mask instance.")
        .def_static("bitCount", &Mask<LOG2DIM>::bitCount,
                    "Total number of bits this mask can store.")
        .def_static("wordCount", &Mask<LOG2DIM>::wordCount,
                    "Number of 64-bit words used to back this mask.")
        .def("countOn", nb::overload_cast<>(&Mask<LOG2DIM>::countOn, nb::const_),
             "Number of bits currently set in the mask.")
        .def("countOn", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::countOn, nb::const_), "i"_a,
             "Number of bits currently set in the prefix [0, i).")
        .def(nb::init<>(),
             "Construct an all-off mask.")
        .def(nb::init<bool>(), "on"_a,
             "Construct a mask with every bit set to on.")
        .def(nb::init<const Mask<LOG2DIM>>(), "other"_a,
             "Copy-construct from another Mask.")
        .def(nb::self == nb::self, "other"_a,
             "Bit-for-bit equality with another Mask.")
        .def(nb::self != nb::self, "other"_a,
             "Bit-for-bit inequality with another Mask.")
        .def("isOn", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::isOn, nb::const_), "n"_a,
             "True iff bit n is set.")
        .def("isOff", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::isOff, nb::const_), "n"_a,
             "True iff bit n is cleared.")
        .def("isOn", nb::overload_cast<>(&Mask<LOG2DIM>::isOn, nb::const_),
             "True iff every bit is set.")
        .def("isOff", nb::overload_cast<>(&Mask<LOG2DIM>::isOff, nb::const_),
             "True iff every bit is cleared.")
        .def("setOn", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::setOn), "n"_a,
             "Set bit n.")
        .def("setOff", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::setOff), "n"_a,
             "Clear bit n.")
        .def("set", nb::overload_cast<uint32_t, bool>(&Mask<LOG2DIM>::set), "n"_a, "on"_a,
             "Assign bit n to the given on/off state.")
        .def("setOn", nb::overload_cast<>(&Mask<LOG2DIM>::setOn),
             "Set every bit.")
        .def("setOff", nb::overload_cast<>(&Mask<LOG2DIM>::setOff),
             "Clear every bit.")
        .def("set", nb::overload_cast<bool>(&Mask<LOG2DIM>::set), "on"_a,
             "Set every bit to the same on/off state.")
        .def("toggle", nb::overload_cast<>(&Mask<LOG2DIM>::toggle),
             "Flip every bit.")
        .def("toggle", nb::overload_cast<uint32_t>(&Mask<LOG2DIM>::toggle), "n"_a,
             "Flip bit n.")
        .def(
            "__iand__", [](Mask<LOG2DIM>& a, const Mask<LOG2DIM>& b) { return a &= b; }, nb::is_operator(), "other"_a,
            "In-place bitwise AND with another Mask.")
        .def(
            "__ior__", [](Mask<LOG2DIM>& a, const Mask<LOG2DIM>& b) { return a |= b; }, nb::is_operator(), "other"_a,
            "In-place bitwise OR with another Mask.")
        .def(
            "__isub__", [](Mask<LOG2DIM>& a, const Mask<LOG2DIM>& b) { return a -= b; }, nb::is_operator(), "other"_a,
            "In-place bitwise difference (clear every bit that is set in other).")
        .def(
            "__ixor__", [](Mask<LOG2DIM>& a, const Mask<LOG2DIM>& b) { return a ^= b; }, nb::is_operator(), "other"_a,
            "In-place bitwise XOR with another Mask.")
        .def("findFirstOn", &Mask<LOG2DIM>::template findFirst<true>,
             "Index of the first set bit, or bitCount() if every bit is clear.")
        .def("findFirstOff", &Mask<LOG2DIM>::template findFirst<false>,
             "Index of the first clear bit, or bitCount() if every bit is set.")
        .def("findNextOn", &Mask<LOG2DIM>::template findNext<true>, "start"_a,
             "Index of the first set bit at or after start.")
        .def("findNextOff", &Mask<LOG2DIM>::template findNext<false>, "start"_a,
             "Index of the first clear bit at or after start.")
        .def("findPrevOn", &Mask<LOG2DIM>::template findPrev<true>, "start"_a,
             "Index of the first set bit at or before start.")
        .def("findPrevOff", &Mask<LOG2DIM>::template findPrev<false>, "start"_a,
             "Index of the first clear bit at or before start.");
}

void defineMap(nb::module_& m)
{
    nb::class_<Map>(m, "Map", "Defines an affine transform and its inverse represented as a 3x3 matrix and a vec3 translation")
        .def(nb::init<>(),
             "Construct an identity Map (uniform unit scale, zero translation).")
        .def(nb::init<double, const Vec3d&>(), "s"_a, "t"_a = Vec3d(0.),
             "Construct a Map with uniform scale s and translation t.")
        .def("set", nb::overload_cast<double, const Vec3f&, double>(&Map::template set<Vec3f>), "scale"_a, "translation"_a, "taper"_a = 1.,
             "Rebuild this Map from a uniform scale, translation and optional frustum taper.")
        .def("set", nb::overload_cast<double, const Vec3d&, double>(&Map::template set<Vec3d>), "scale"_a, "translation"_a, "taper"_a = 1.,
             "Rebuild this Map from a uniform scale, translation and optional frustum taper.")
        .def("applyMap", nb::overload_cast<const Vec3f&>(&Map::template applyMap<Vec3f>, nb::const_), "ijk"_a,
             "Transform an index-space point to world space using 64-bit math; returns a vector of the same dtype as the input.")
        .def("applyMap", nb::overload_cast<const Vec3d&>(&Map::template applyMap<Vec3d>, nb::const_), "ijk"_a,
             "Transform an index-space point to world space using 64-bit math; returns a vector of the same dtype as the input.")
        .def("applyMapF", nb::overload_cast<const Vec3f&>(&Map::template applyMapF<Vec3f>, nb::const_), "ijk"_a,
             "Transform an index-space point to world space using 32-bit math; returns a vector of the same dtype as the input.")
        .def("applyMapF", nb::overload_cast<const Vec3d&>(&Map::template applyMapF<Vec3d>, nb::const_), "ijk"_a,
             "Transform an index-space point to world space using 32-bit math; returns a vector of the same dtype as the input.")
        .def("applyJacobian", nb::overload_cast<const Vec3f&>(&Map::template applyJacobian<Vec3f>, nb::const_), "ijk"_a,
             "Apply the linear (Jacobian) part of the transform using 64-bit math, ignoring translation.")
        .def("applyJacobian", nb::overload_cast<const Vec3d&>(&Map::template applyJacobian<Vec3d>, nb::const_), "ijk"_a,
             "Apply the linear (Jacobian) part of the transform using 64-bit math, ignoring translation.")
        .def("applyJacobianF", nb::overload_cast<const Vec3f&>(&Map::template applyJacobianF<Vec3f>, nb::const_), "ijk"_a,
             "Apply the linear (Jacobian) part of the transform using 32-bit math, ignoring translation.")
        .def("applyJacobianF", nb::overload_cast<const Vec3d&>(&Map::template applyJacobianF<Vec3d>, nb::const_), "ijk"_a,
             "Apply the linear (Jacobian) part of the transform using 32-bit math, ignoring translation.")
        .def("applyInverseMap", nb::overload_cast<const Vec3f&>(&Map::template applyInverseMap<Vec3f>, nb::const_), "xyz"_a,
             "Transform a world-space point back to index space using 64-bit math; returns a vector of the same dtype as the input.")
        .def("applyInverseMap", nb::overload_cast<const Vec3d&>(&Map::template applyInverseMap<Vec3d>, nb::const_), "xyz"_a,
             "Transform a world-space point back to index space using 64-bit math; returns a vector of the same dtype as the input.")
        .def("applyInverseMapF", nb::overload_cast<const Vec3f&>(&Map::template applyInverseMapF<Vec3f>, nb::const_), "xyz"_a,
             "Transform a world-space point back to index space using 32-bit math; returns a vector of the same dtype as the input.")
        .def("applyInverseMapF", nb::overload_cast<const Vec3d&>(&Map::template applyInverseMapF<Vec3d>, nb::const_), "xyz"_a,
             "Transform a world-space point back to index space using 32-bit math; returns a vector of the same dtype as the input.")
        .def("applyInverseJacobian", nb::overload_cast<const Vec3f&>(&Map::template applyInverseJacobian<Vec3f>, nb::const_), "xyz"_a,
             "Apply the inverse linear (Jacobian) part of the transform using 64-bit math.")
        .def("applyInverseJacobian", nb::overload_cast<const Vec3d&>(&Map::template applyInverseJacobian<Vec3d>, nb::const_), "xyz"_a,
             "Apply the inverse linear (Jacobian) part of the transform using 64-bit math.")
        .def("applyInverseJacobianF", nb::overload_cast<const Vec3f&>(&Map::template applyInverseJacobianF<Vec3f>, nb::const_), "xyz"_a,
             "Apply the inverse linear (Jacobian) part using 32-bit math.")
        .def("applyInverseJacobianF", nb::overload_cast<const Vec3d&>(&Map::template applyInverseJacobianF<Vec3d>, nb::const_), "xyz"_a,
             "Apply the inverse linear (Jacobian) part using 32-bit math.")
        .def("applyIJT", nb::overload_cast<const Vec3f&>(&Map::template applyIJT<Vec3f>, nb::const_), "xyz"_a,
             "Apply the inverse-Jacobian-transpose used for transforming normals.")
        .def("applyIJT", nb::overload_cast<const Vec3d&>(&Map::template applyIJT<Vec3d>, nb::const_), "xyz"_a,
             "Apply the inverse-Jacobian-transpose used for transforming normals.")
        .def("applyIJTF", nb::overload_cast<const Vec3f&>(&Map::template applyIJTF<Vec3f>, nb::const_), "xyz"_a,
             "Apply the inverse-Jacobian-transpose in single precision.")
        .def("applyIJTF", nb::overload_cast<const Vec3d&>(&Map::template applyIJTF<Vec3d>, nb::const_), "xyz"_a,
             "Apply the inverse-Jacobian-transpose in single precision.")
        .def("getVoxelSize", &Map::getVoxelSize,
             "World-space size of a single voxel implied by this Map.");
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
    nb::class_<GridData>(m, "Grid",
        "Type-erased base for every NanoVDB grid. Carries header fields, "
        "the affine transform, grid flags and blind-data accessors. Concrete "
        "BuildT-typed subclasses (FloatGrid, Vec3fGrid, ...) add tree access "
        "and per-voxel queries.")
        // Validation and flag mutators (already member functions on GridData).
        .def("isValid", &GridData::isValid,
             "True iff the grid header looks consistent (magic / version / class tags).")
        .def("setMinMaxOn", &GridData::setMinMaxOn, "on"_a = true,
             "Toggle the HasMinMax grid flag.")
        .def("setBBoxOn", &GridData::setBBoxOn, "on"_a = true,
             "Toggle the HasBBox grid flag.")
        .def("setLongGridNameOn", &GridData::setLongGridNameOn, "on"_a = true,
             "Toggle the HasLongGridName grid flag.")
        .def("setAverageOn", &GridData::setAverageOn, "on"_a = true,
             "Toggle the HasAverage grid flag.")
        .def("setStdDeviationOn", &GridData::setStdDeviationOn, "on"_a = true,
             "Toggle the HasStdDeviation grid flag.")
        .def("setGridName", &GridData::setGridName, "src"_a,
             "Overwrite the grid's short name (truncated to the in-header buffer).")
        // Affine transforms (already member functions on GridData).
        .def("applyMap", nb::overload_cast<const Vec3f&>(&GridData::template applyMap<Vec3f>, nb::const_), "xyz"_a,
             "Transform an index-space point to world space using this grid's Map.")
        .def("applyMap", nb::overload_cast<const Vec3d&>(&GridData::template applyMap<Vec3d>, nb::const_), "xyz"_a,
             "Transform an index-space point to world space using this grid's Map.")
        .def("applyMapF", nb::overload_cast<const Vec3f&>(&GridData::template applyMapF<Vec3f>, nb::const_), "xyz"_a,
             "Transform an index-space point to world space in single precision.")
        .def("applyMapF", nb::overload_cast<const Vec3d&>(&GridData::template applyMapF<Vec3d>, nb::const_), "xyz"_a,
             "Transform an index-space point to world space in single precision.")
        .def("applyJacobian", nb::overload_cast<const Vec3f&>(&GridData::template applyJacobian<Vec3f>, nb::const_), "xyz"_a,
             "Apply the linear (Jacobian) part of the transform using 64-bit math, ignoring translation.")
        .def("applyJacobian", nb::overload_cast<const Vec3d&>(&GridData::template applyJacobian<Vec3d>, nb::const_), "xyz"_a,
             "Apply the linear (Jacobian) part of the transform using 64-bit math, ignoring translation.")
        .def("applyJacobianF", nb::overload_cast<const Vec3f&>(&GridData::template applyJacobianF<Vec3f>, nb::const_), "xyz"_a,
             "Apply the linear part in single precision.")
        .def("applyJacobianF", nb::overload_cast<const Vec3d&>(&GridData::template applyJacobianF<Vec3d>, nb::const_), "xyz"_a,
             "Apply the linear part in single precision.")
        .def("applyInverseMap", nb::overload_cast<const Vec3f&>(&GridData::template applyInverseMap<Vec3f>, nb::const_), "xyz"_a,
             "Transform a world-space point back to index space.")
        .def("applyInverseMap", nb::overload_cast<const Vec3d&>(&GridData::template applyInverseMap<Vec3d>, nb::const_), "xyz"_a,
             "Transform a world-space point back to index space.")
        .def("applyInverseMapF", nb::overload_cast<const Vec3f&>(&GridData::template applyInverseMapF<Vec3f>, nb::const_), "xyz"_a,
             "Transform a world-space point back to index space in single precision.")
        .def("applyInverseMapF", nb::overload_cast<const Vec3d&>(&GridData::template applyInverseMapF<Vec3d>, nb::const_), "xyz"_a,
             "Transform a world-space point back to index space in single precision.")
        .def("applyInverseJacobian", nb::overload_cast<const Vec3f&>(&GridData::template applyInverseJacobian<Vec3f>, nb::const_), "xyz"_a,
             "Apply the inverse linear (Jacobian) part of the transform using 64-bit math.")
        .def("applyInverseJacobian", nb::overload_cast<const Vec3d&>(&GridData::template applyInverseJacobian<Vec3d>, nb::const_), "xyz"_a,
             "Apply the inverse linear (Jacobian) part of the transform using 64-bit math.")
        .def("applyInverseJacobianF", nb::overload_cast<const Vec3f&>(&GridData::template applyInverseJacobianF<Vec3f>, nb::const_), "xyz"_a,
             "Apply the inverse linear (Jacobian) part using 32-bit math.")
        .def("applyInverseJacobianF", nb::overload_cast<const Vec3d&>(&GridData::template applyInverseJacobianF<Vec3d>, nb::const_), "xyz"_a,
             "Apply the inverse linear (Jacobian) part using 32-bit math.")
        .def("applyIJT", nb::overload_cast<const Vec3f&>(&GridData::template applyIJT<Vec3f>, nb::const_), "xyz"_a,
             "Apply the inverse-Jacobian-transpose used for transforming normals.")
        .def("applyIJT", nb::overload_cast<const Vec3d&>(&GridData::template applyIJT<Vec3d>, nb::const_), "xyz"_a,
             "Apply the inverse-Jacobian-transpose used for transforming normals.")
        .def("applyIJTF", nb::overload_cast<const Vec3f&>(&GridData::template applyIJTF<Vec3f>, nb::const_), "xyz"_a,
             "Apply the inverse-Jacobian-transpose in single precision.")
        .def("applyIJTF", nb::overload_cast<const Vec3d&>(&GridData::template applyIJTF<Vec3d>, nb::const_), "xyz"_a,
             "Apply the inverse-Jacobian-transpose in single precision.")
        // Strings, geometry, layout (already member functions on GridData).
        .def("gridName", &GridData::gridName,
             "Full grid name as a C string. Reads the long-form name from "
             "blind data when the HasLongGridName flag is set, falling "
             "back to the in-header buffer otherwise. Use shortGridName() "
             "if you specifically want the truncated 255-byte header copy.")
        .def("memUsage", &GridData::memUsage,
             "Byte size of this grid header.")
        .def("worldBBox", &GridData::worldBBox,
             "World-space bounding box of active voxels.")
        .def("indexBBox", &GridData::indexBBox,
             "Index-space bounding box of active voxels.")
        .def("isEmpty", &GridData::isEmpty,
             "True iff this grid has no active voxels.")
        // Lifted from Grid<TreeT> via direct data-member access.
        .def("version",         [](const GridData& g) { return g.mVersion; },
             "NanoVDB Version stored in this grid's header.")
        .def("gridSize",        [](const GridData& g) { return g.mGridSize; },
             "Total byte size of this grid (header + tree + blind data).")
        .def("gridIndex",       [](const GridData& g) { return g.mGridIndex; },
             "Zero-based index of this grid within its parent GridHandle.")
        .def("gridCount",       [](const GridData& g) { return g.mGridCount; },
             "Number of grids stored alongside this one in the parent GridHandle.")
        .def("voxelSize",       [](const GridData& g) -> const Vec3d& { return g.mVoxelSize; },
             nb::rv_policy::reference_internal,
             "World-space voxel size as a Vec3d.")
        .def("map",             [](const GridData& g) -> const Map& { return g.mMap; },
             nb::rv_policy::reference_internal,
             "Affine index-to-world Map associated with this grid.")
        .def("gridType",  [](const GridData& g) { return g.mGridType; },
             "GridType enumerator naming the BuildT carried by this grid.")
        .def("gridClass", [](const GridData& g) { return g.mGridClass; },
             "GridClass enumerator (LevelSet, FogVolume, ...).")
        .def("checksum",  [](const GridData& g) { return g.mChecksum; },
             "Checksum stored in the grid header; compare against tools.evalChecksum.")
        .def("isLevelSet",   [](const GridData& g) { return g.mGridClass == GridClass::LevelSet; },
             "True iff this grid's class is LevelSet.")
        .def("isFogVolume",  [](const GridData& g) { return g.mGridClass == GridClass::FogVolume; },
             "True iff this grid's class is FogVolume.")
        .def("isStaggered",  [](const GridData& g) { return g.mGridClass == GridClass::Staggered; },
             "True iff this grid's class is Staggered.")
        .def("isPointIndex",
             [](const GridData& g) { return g.mGridClass == GridClass::PointIndex; },
             "True iff this grid's class is PointIndex.")
        .def("isGridIndex",  [](const GridData& g) { return g.mGridClass == GridClass::IndexGrid; },
             "True iff this grid's class is IndexGrid.")
        .def("isPointData",  [](const GridData& g) { return g.mGridClass == GridClass::PointData; },
             "True iff this grid's class is PointData.")
        .def("isMask",       [](const GridData& g) { return g.mGridClass == GridClass::Topology; },
             "True iff this grid's class is Topology.")
        .def("isUnknown",    [](const GridData& g) { return g.mGridClass == GridClass::Unknown; },
             "True iff this grid's class is Unknown.")
        .def("hasMinMax", [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::HasMinMax); },
             "True iff per-node min/max stats are stored in this grid.")
        .def("hasBBox",   [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::HasBBox); },
             "True iff per-node bbox stats are stored in this grid.")
        .def("hasLongGridName",
             [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::HasLongGridName); },
             "True iff the grid's name was too long to fit in the header buffer.")
        .def("hasAverage",
             [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::HasAverage); },
             "True iff per-node average stats are stored in this grid.")
        .def("hasStdDeviation",
             [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::HasStdDeviation); },
             "True iff per-node standard-deviation stats are stored in this grid.")
        .def("isBreadthFirst",
             [](const GridData& g) { return g.mFlags.isMaskOn(GridFlags::IsBreadthFirst); },
             "True iff this grid's nodes are laid out in breadth-first order.")
        .def("shortGridName", [](const GridData& g) { return std::string(g.mGridName); },
             "Short in-header copy of the grid name as a Python string.")
        // Blind data — exposes the sidecar channels that PointGrid and
        // OnIndexGrid use to carry their actual values, colors, normals, IDs,
        // etc. blindMetaData(n) returns the descriptor; getBlindData(n)
        // returns a zero-copy NumPy view onto the underlying bytes typed by
        // mDataType (Float -> float32 ndarray, Vec3f -> (N, 3) float32, etc.;
        // unrecognized types fall back to a flat uint8 byte view).
        .def("blindDataCount",          [](const GridData& g) { return g.mBlindMetadataCount; },
             "Number of blind-data channels attached to this grid.")
        .def("blindMetaData",
             [](const GridData& g, uint32_t n) -> const GridBlindMetaData* {
                 return n < g.mBlindMetadataCount ? g.blindMetaData(n) : nullptr;
             },
             nb::rv_policy::reference_internal, "n"_a,
             "GridBlindMetaData descriptor for the n-th blind-data channel, or None if n is out of range.")
        .def("findBlindData",           [](const GridData& g, const std::string& name) -> int {
            for (uint32_t i = 0; i < g.mBlindMetadataCount; ++i) {
                const auto* meta = g.blindMetaData(i);
                if (std::strncmp(meta->mName, name.c_str(), GridBlindMetaData::MaxNameSize) == 0)
                    return static_cast<int>(i);
            }
            return -1;
        }, "name"_a,
        "Index of the blind-data channel whose name matches name, or -1 if none.")
        .def("findBlindDataForSemantic", [](const GridData& g, GridBlindDataSemantic sem) -> int {
            for (uint32_t i = 0; i < g.mBlindMetadataCount; ++i) {
                if (g.blindMetaData(i)->mSemantic == sem)
                    return static_cast<int>(i);
            }
            return -1;
        }, "semantic"_a,
        "Index of the blind-data channel whose semantic matches the given enum, or -1 if none.")
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
    auto cls = nb::class_<NanoGrid<BuildT>, GridData>(m, name,
        "BuildT-typed NanoVDB grid. Inherits the type-erased Grid base for "
        "header / transform / blind data, and adds the typed tree and "
        "accessor API for the underlying value type.")
        .def("getAccessor", &NanoGrid<BuildT>::getAccessor,
             "Return a DefaultReadAccessor caching the most recently visited path.")
        .def("activeVoxelCount", &NanoGrid<BuildT>::activeVoxelCount,
             "Total number of active voxels in this grid. See nanovdb::Grid::activeVoxelCount in NanoVDB.h.")
        .def("isSequential", [](const NanoGrid<BuildT>& grid) { return grid.isSequential(); },
             "True iff this grid's nodes are laid out sequentially (per-level contiguous).")
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
    nb::enum_<GridBlindDataClass>(m, "GridBlindDataClass",
        "Coarse classifier for a blind-data channel (index, attribute, "
        "channel, ...). Pairs with GridBlindDataSemantic to describe what "
        "the channel actually carries.")
        .value("Unknown",        GridBlindDataClass::Unknown)
        .value("IndexArray",     GridBlindDataClass::IndexArray)
        .value("AttributeArray", GridBlindDataClass::AttributeArray)
        .value("GridName",       GridBlindDataClass::GridName)
        .value("ChannelArray",   GridBlindDataClass::ChannelArray)
        .value("End",            GridBlindDataClass::End)
        .export_values();

    nb::enum_<GridBlindDataSemantic>(m, "GridBlindDataSemantic",
        "Fine-grained role of a blind-data channel: PointPosition, "
        "PointColor, PointNormal, ... Used by find* helpers on Grid.")
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
        .def_ro("valueCount", &GridBlindMetaData::mValueCount,
                "Number of values stored in this channel.")
        .def_ro("valueSize",  &GridBlindMetaData::mValueSize,
                "Byte size of a single value in this channel.")
        .def_ro("semantic",   &GridBlindMetaData::mSemantic,
                "GridBlindDataSemantic describing what this channel carries.")
        .def_ro("dataClass",  &GridBlindMetaData::mDataClass,
                "GridBlindDataClass coarsely classifying this channel.")
        .def_ro("dataType",   &GridBlindMetaData::mDataType,
                "GridType enumerator giving the dtype of a single value.")
        .def("name",  [](const GridBlindMetaData& m) { return std::string(m.mName); },
             "Name of this blind-data channel as a Python string.")
        .def("isValid", &GridBlindMetaData::isValid,
             "True iff this descriptor's class/semantic/type combination looks consistent.")
        .def("blindDataSize", &GridBlindMetaData::blindDataSize,
             "Total byte size of this channel (valueCount * valueSize, padded).");
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
        .def(nb::init<const NanoGrid<Point>&>(), "grid"_a, nb::keep_alive<1, 2>(),
             "Construct a PointAccessor bound to the given PointGrid.")
        .def("__bool__", [](const PA& a) { return bool(a); },
             "True iff this accessor is bound to a valid PointGrid.")
        .def("grid", &PA::grid, nb::rv_policy::reference_internal,
             "Return the PointGrid this accessor is bound to.")
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
             }, "grid"_a,
             "Construct from a Grid. Raises ValueError if the grid is None or has an invalid header.")
        .def_static("safeCast",
                    [](const GridData* gd) {
                        // Mirror the spirit of NanoVDB's static safeCast: "is
                        // it safe to cast this gridData to a GridMetaData?".
                        // null and invalid grids are by definition not safe;
                        // return False rather than dereference.
                        if (gd == nullptr || !gd->isValid()) return false;
                        return GridMetaData::safeCast(gd);
                    }, "grid"_a,
                    "True iff the given grid header is well-formed enough to wrap in a GridMetaData.")
        .def("isValid", &GridMetaData::isValid,
             "True iff the wrapped header looks consistent.")
        .def("gridType", &GridMetaData::gridType,
             "GridType enumerator naming the BuildT of the wrapped grid.")
        .def("gridClass", &GridMetaData::gridClass,
             "GridClass enumerator (LevelSet, FogVolume, ...).")
        .def("isLevelSet", &GridMetaData::isLevelSet,
             "True iff the wrapped grid's class is LevelSet.")
        .def("isFogVolume", &GridMetaData::isFogVolume,
             "True iff the wrapped grid's class is FogVolume.")
        .def("isStaggered", &GridMetaData::isStaggered,
             "True iff the wrapped grid's class is Staggered.")
        .def("isPointIndex", &GridMetaData::isPointIndex,
             "True iff the wrapped grid's class is PointIndex.")
        .def("isGridIndex", &GridMetaData::isGridIndex,
             "True iff the wrapped grid's class is IndexGrid.")
        .def("isPointData", &GridMetaData::isPointData,
             "True iff the wrapped grid's class is PointData.")
        .def("isMask", &GridMetaData::isMask,
             "True iff the wrapped grid's class is Topology.")
        .def("isUnknown", &GridMetaData::isUnknown,
             "True iff the wrapped grid's class is Unknown.")
        .def("hasMinMax", &GridMetaData::hasMinMax,
             "True iff per-node min/max stats are stored in the wrapped grid.")
        .def("hasBBox", &GridMetaData::hasBBox,
             "True iff per-node bbox stats are stored in the wrapped grid.")
        .def("hasLongGridName", &GridMetaData::hasLongGridName,
             "True iff the wrapped grid's name was too long for the header buffer.")
        .def("hasAverage", &GridMetaData::hasAverage,
             "True iff per-node average stats are stored in the wrapped grid.")
        .def("hasStdDeviation", &GridMetaData::hasStdDeviation,
             "True iff per-node standard-deviation stats are stored in the wrapped grid.")
        .def("isBreadthFirst", &GridMetaData::isBreadthFirst,
             "True iff the wrapped grid is laid out breadth-first.")
        .def("gridSize", &GridMetaData::gridSize,
             "Total byte size of the wrapped grid.")
        .def("gridIndex", &GridMetaData::gridIndex,
             "Index of this grid within its parent GridHandle.")
        .def("gridCount", &GridMetaData::gridCount,
             "Number of grids in the parent GridHandle.")
        .def("shortGridName", [](const GridMetaData& m) { return std::string(m.shortGridName()); },
             "Short in-header copy of the grid name as a Python string.")
        .def("map", &GridMetaData::map, nb::rv_policy::reference_internal,
             "Affine index-to-world Map of the wrapped grid.")
        .def("worldBBox", &GridMetaData::worldBBox, nb::rv_policy::reference_internal,
             "World-space bounding box of the wrapped grid's active voxels.")
        .def("indexBBox", &GridMetaData::indexBBox, nb::rv_policy::reference_internal,
             "Index-space bounding box of the wrapped grid's active voxels.")
        .def("voxelSize", &GridMetaData::voxelSize,
             "World-space voxel size of the wrapped grid.")
        .def("blindDataCount", &GridMetaData::blindDataCount,
             "Number of blind-data channels attached to the wrapped grid.")
        .def("activeVoxelCount", &GridMetaData::activeVoxelCount,
             "Total active voxel count of the wrapped grid.")
        .def("activeTileCount", &GridMetaData::activeTileCount, "level"_a,
             "Number of active tiles at the given tree level (1=lower, 2=upper, 3=root).")
        .def("nodeCount", &GridMetaData::nodeCount, "level"_a,
             "Number of nodes at the given tree level (0=leaf, 1=lower, 2=upper).")
        .def("checksum", &GridMetaData::checksum, nb::rv_policy::reference_internal,
             "Checksum stored in the wrapped grid's header.")
        .def("rootTableSize", &GridMetaData::rootTableSize,
             "Number of entries in the wrapped grid's root tile table.")
        .def("isEmpty", &GridMetaData::isEmpty,
             "True iff the wrapped grid has no active voxels.")
        .def("version", &GridMetaData::version,
             "NanoVDB Version stored in the wrapped grid's header.");
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

    nb::class_<DefaultReadAccessor<BuildT>> accessor(m, name,
        "Read accessor that caches the most recently visited tree path for "
        "fast neighbor lookups. Construct from a typed grid.");
    accessor.def(nb::init<const NanoGrid<BuildT>&>(), "grid"_a,
                 "Construct an accessor bound to the given grid.")
        .def("getValue", nb::overload_cast<const CoordType&>(&DefaultReadAccessor<BuildT>::getValue, nb::const_), "ijk"_a,
             "Return the grid's value at the integer Coord ijk.")
        .def("getValue", nb::overload_cast<int, int, int>(&DefaultReadAccessor<BuildT>::getValue, nb::const_), "i"_a, "j"_a, "k"_a,
             "Return the grid's value at the integer voxel (i, j, k).")
        .def(
            "__call__", [](const DefaultReadAccessor<BuildT>& accessor, const CoordType& ijk) { return accessor.getValue(ijk); }, nb::is_operator(), "ijk"_a,
            "Operator form of getValue(ijk).")
        .def(
            "__call__",
            [](const DefaultReadAccessor<BuildT>& accessor, int i, int j, int k) { return accessor.getValue(i, j, k); },
            nb::is_operator(),
            "i"_a,
            "j"_a,
            "k"_a,
            "Operator form of getValue(i, j, k).")
        .def("isActive", &DefaultReadAccessor<BuildT>::isActive, "ijk"_a,
             "True iff the voxel at ijk is active.")
        .def(
            "probeValue",
            [](const DefaultReadAccessor<BuildT>& accessor, const CoordType& ijk) {
                ValueType v;
                bool      isOn = accessor.probeValue(ijk, v);
                return std::make_tuple(v, isOn);
            },
            "ijk"_a,
            "Return (value, isActive) for the voxel at ijk in a single tree traversal.");
    return accessor;
}

template<typename BuildT> void defineScalarAccessor(nb::module_& m, const char* name)
{
    using ValueType = typename DefaultReadAccessor<BuildT>::ValueType;
    using CoordType = typename DefaultReadAccessor<BuildT>::CoordType;

    defineAccessor<BuildT>(m, name)
        .def("getNodeInfo", &DefaultReadAccessor<BuildT>::getNodeInfo, "ijk"_a,
             "Return a NodeInfo describing the deepest tree node covering ijk.")
        .def(
            "setVoxel",
            [](DefaultReadAccessor<BuildT>& accessor, const CoordType& ijk, const ValueType& v) {
                using OpT = SetVoxel<BuildT>;
                accessor.template set<OpT>(ijk, v);
            },
            "ijk"_a,
            "v"_a,
            "Set the value at ijk to v and mark the voxel active.");
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
        "v"_a,
        "Set the vector value at ijk to v and mark the voxel active.");
}

template<typename BuildT> void defineNodeInfo(nb::module_& m, const char* name)
{
    nb::class_<typename GetNodeInfo<BuildT>::NodeInfo>(m, name,
        "Descriptor of the deepest tree node covering a queried voxel. "
        "Returned by ReadAccessor.getNodeInfo().")
        .def_ro("level", &GetNodeInfo<BuildT>::NodeInfo::level,
                "Tree level of the node (0=leaf, 1=lower, 2=upper, 3=root).")
        .def_ro("dim", &GetNodeInfo<BuildT>::NodeInfo::dim,
                "Side length of the node's covered region in voxels.")
        .def_ro("minimum", &GetNodeInfo<BuildT>::NodeInfo::minimum,
                "Minimum active value within this node.")
        .def_ro("maximum", &GetNodeInfo<BuildT>::NodeInfo::maximum,
                "Maximum active value within this node.")
        .def_ro("average", &GetNodeInfo<BuildT>::NodeInfo::average,
                "Average of active values within this node.")
        .def_ro("stdDevi", &GetNodeInfo<BuildT>::NodeInfo::stdDevi,
                "Standard deviation of active values within this node.")
        .def_ro("bbox", &GetNodeInfo<BuildT>::NodeInfo::bbox,
                "Index-space bounding box of this node's covered region.");
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

    nb::enum_<GridType>(m, "GridType",
        "Enumerator naming every BuildT a NanoVDB grid can carry. Used by "
        "handle.gridType(n) and as the dtype field on GridBlindMetaData.")
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

    nb::enum_<GridClass>(m, "GridClass",
        "Semantic class of a grid (LevelSet, FogVolume, PointIndex, ...). "
        "Independent of GridType / BuildT; consumed by tools that special-"
        "case sign / point / index grids.")
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
