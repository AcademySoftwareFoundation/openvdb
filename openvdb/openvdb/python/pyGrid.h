// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file pyGrid.h
/// @author Peter Cucka
/// @brief nanobind wrapper for openvdb::Grid

#ifndef OPENVDB_PYGRID_HAS_BEEN_INCLUDED
#define OPENVDB_PYGRID_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#ifdef PY_OPENVDB_USE_NUMPY
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h> // for tools::volumeToMesh()
#endif
#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/math/Math.h> // for math::isExactlyEqual()
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Count.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/SignedFloodFill.h>
#include "pyutil.h"
#include "pyTypeCasters.h"
#include "pyAccessor.h" // for pyAccessor::AccessorWrap
#include <algorithm> // for std::max()
#include <cstring> // for memcpy()
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <variant>

namespace nb = nanobind;

#ifdef __clang__
// This is a private header, so it's OK to include a "using namespace" directive.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wheader-hygiene"
#endif

using namespace openvdb::OPENVDB_VERSION_NAME;

#ifdef __clang__
#pragma clang diagnostic pop
#endif


namespace pyGrid {


template<typename GridType>
inline typename GridType::Ptr
copyGrid(GridType& grid)
{
    return grid.copy();
}


template<typename GridType>
inline bool
sharesWith(const GridType& grid, nb::object other)
{
    if (nb::isinstance<GridType>(other)) {
        typename GridType::ConstPtr otherGrid = nb::cast<typename GridType::Ptr>(other);
        return (&otherGrid->tree() == &grid.tree());
    }
    return false;
}


////////////////////////////////////////


template<typename GridType>
inline std::string
getValueType()
{
    return pyutil::GridTraits<GridType>::valueTypeName();
}


template<typename GridType>
inline typename GridType::ValueType
getZeroValue()
{
    return openvdb::zeroVal<typename GridType::ValueType>();
}


template<typename GridType>
inline typename GridType::ValueType
getOneValue()
{
    using ValueT = typename GridType::ValueType;
    return ValueT(openvdb::zeroVal<ValueT>() + 1);
}


template<typename GridType>
inline typename GridType::ValueType
getGridBackground(const GridType& grid)
{
    return grid.background();
}


template<typename GridType>
inline void
setGridBackground(GridType& grid, const typename GridType::ValueType& background)
{
    tools::changeBackground(grid.tree(), background);
}

////////////////////////////////////////


// Helper class to construct a pyAccessor::AccessorWrap for a given grid,
// permitting partial specialization for const vs. non-const grids
template<typename GridType>
struct AccessorHelper
{
    using Wrapper = typename pyAccessor::AccessorWrap<GridType>;
    static Wrapper wrap(typename GridType::Ptr grid)
    {
        if (!grid) {
            throw nb::value_error("null grid");
        }
        return Wrapper(grid);
    }
};

// Specialization for const grids
template<typename GridType>
struct AccessorHelper<const GridType>
{
    using Wrapper = typename pyAccessor::AccessorWrap<const GridType>;
    static Wrapper wrap(typename GridType::ConstPtr grid)
    {
        if (!grid) {
            throw nb::value_error("null grid");
        }
        return Wrapper(grid);
    }
};


/// Return a non-const accessor (wrapped in a pyAccessor::AccessorWrap) for the given grid.
template<typename GridType>
inline typename AccessorHelper<GridType>::Wrapper
getAccessor(typename GridType::Ptr grid)
{
    return AccessorHelper<GridType>::wrap(grid);
}

/// @brief Return a const accessor (wrapped in a pyAccessor::AccessorWrap) for the given grid.
/// @internal Note that the grid pointer is non-const, even though the grid is
/// treated as const.  This is because we don't expose a const grid type in Python.
template<typename GridType>
inline typename AccessorHelper<const GridType>::Wrapper
getConstAccessor(typename GridType::Ptr grid)
{
    return AccessorHelper<const GridType>::wrap(grid);
}


////////////////////////////////////////


template<typename GridType>
std::tuple<Coord, Coord>
evalLeafBoundingBox(const GridType& grid)
{
    CoordBBox bbox;
    grid.tree().evalLeafBoundingBox(bbox);
    return std::make_tuple(bbox.min(), bbox.max());
}


template<typename GridType>
inline Coord
evalLeafDim(const GridType& grid)
{
    Coord dim;
    grid.tree().evalLeafDim(dim);
    return dim;
}


template<typename GridType>
inline std::vector<Index>
getNodeLog2Dims(const GridType& grid)
{
    std::vector<Index> dims;
    grid.tree().getNodeLog2Dims(dims);
    return dims;
}


template<typename GridType>
inline Index64
treeDepth(const GridType& grid)
{
    return grid.tree().treeDepth();
}


template<typename GridType>
inline Index64
leafCount(const GridType& grid)
{
    return grid.tree().leafCount();
}


template<typename GridType>
inline Index64
nonLeafCount(const GridType& grid)
{
    return grid.tree().nonLeafCount();
}


template<typename GridType>
inline Index64
activeLeafVoxelCount(const GridType& grid)
{
    return grid.tree().activeLeafVoxelCount();
}


template<typename GridType>
inline std::tuple<typename GridType::ValueType, typename GridType::ValueType>
evalMinMax(const GridType& grid)
{
    const math::MinMax<typename GridType::ValueType> extrema = tools::minMax(grid.tree());
    return std::make_tuple(extrema.min(), extrema.max());
}


template<typename GridType>
inline std::tuple<Coord, Coord>
getIndexRange(const GridType& grid)
{
    CoordBBox bbox;
    grid.tree().getIndexRange(bbox);
    return std::make_tuple(bbox.min(), bbox.max());
}


//template<typename GridType>
//inline void
//expandIndexRange(GridType& grid, const Coord& xyz)
//{
//    grid.tree().expand(xyz);
//}


////////////////////////////////////////

template<typename GridType>
inline void
prune(GridType& grid, typename GridType::ValueType tol)
{
    tools::prune(grid.tree(), tol);
}


template<typename GridType>
inline void
pruneInactive(GridType& grid, const std::optional<typename GridType::ValueType>& value)
{
    if (value)
        tools::pruneInactiveWithValue(grid.tree(), *value);
    else
        tools::pruneInactive(grid.tree());
}


template<typename GridType>
inline void
fill(GridType& grid, const Coord& bmin, const Coord& bmax, const typename GridType::ValueType& value, bool active)
{
    grid.fill(CoordBBox(bmin, bmax), value, active);
}


template<typename GridType>
inline void
signedFloodFill(GridType& grid)
{
    tools::signedFloodFill(grid.tree());
}


////////////////////////////////////////


#ifndef PY_OPENVDB_USE_NUMPY

template<typename GridType>
inline void
copyFromArrayScalar(GridType&, const nb::object&, nb::object, nb::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw nb::python_error();
}

template<typename GridType>
inline void
copyFromArrayVector(GridType&, const nb::object&, nb::object, nb::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw nb::python_error();
}

template<typename GridType>
inline void
copyToArrayScalar(GridType&, const nb::object&, nb::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw nb::python_error();
}

template<typename GridType>
inline void
copyToArrayVector(GridType&, const nb::object&, nb::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw nb::python_error();
}

#else // if defined(PY_OPENVDB_USE_NUMPY)

template<typename GridType>
inline void
copyFromArrayScalar(GridType& grid, nb::ndarray<nb::numpy> array, const Coord& origin, const typename GridType::ValueType& tolerance)
{
    if (array.ndim() != 3) {
        std::stringstream ss;
        ss << "Expected array with ndim = 3, found array with ndim = " << array.ndim();
        throw nb::value_error(ss.str().c_str());
    }

    // Compute the bounding box of the region of the grid that is to be copied from or to.
    // origin specifies the coordinates (i, j, k) of the voxel at which to start populating data.
    // Voxel (i, j, k) will correspond to array element (0, 0, 0).
    CoordBBox bbox(origin, origin + Coord(static_cast<Int32>(array.shape(0)), static_cast<Int32>(array.shape(1)), static_cast<Int32>(array.shape(2))) - Coord(1));
    if (array.dtype() == nb::dtype<float>()) {
        tools::Dense<float> valArray(bbox, reinterpret_cast<float*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<double>()) {
        tools::Dense<double> valArray(bbox, reinterpret_cast<double*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<bool>()) {
        tools::Dense<bool> valArray(bbox, reinterpret_cast<bool*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<Int16>()) {
        tools::Dense<Int16> valArray(bbox, reinterpret_cast<Int16*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<Int32>()) {
        tools::Dense<Int32> valArray(bbox, reinterpret_cast<Int32*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<Int64>()) {
        tools::Dense<Int64> valArray(bbox, reinterpret_cast<Int64*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<Index32>()) {
        tools::Dense<Index32> valArray(bbox, reinterpret_cast<Index32*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<Index64>()) {
        tools::Dense<Index64> valArray(bbox, reinterpret_cast<Index64*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else {
        throw openvdb::TypeError();
    }
}

template<typename GridType>
inline void
copyFromArrayVector(GridType& grid, nb::ndarray<nb::numpy> array, const Coord& origin, const typename GridType::ValueType& tolerance)
{
    if (array.ndim() != 4) {
        std::stringstream ss;
        ss << "Expected array with ndim = 4, found array with ndim = " << array.ndim();
        throw nb::value_error(ss.str().c_str());
    }

    // Compute the bounding box of the region of the grid that is to be copied from or to.
    // origin specifies the coordinates (i, j, k) of the voxel at which to start populating data.
    // Voxel (i, j, k) will correspond to array element (0, 0, 0).
    CoordBBox bbox(origin, origin + Coord(static_cast<Int32>(array.shape(0)), static_cast<Int32>(array.shape(1)), static_cast<Int32>(array.shape(2))) - Coord(1));
    if (array.dtype() == nb::dtype<float>()) {
        tools::Dense<math::Vec3<float>> valArray(bbox, reinterpret_cast<math::Vec3<float>*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<double>()) {
        tools::Dense<math::Vec3<double>> valArray(bbox, reinterpret_cast<math::Vec3<double>*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<bool>()) {
        tools::Dense<math::Vec3<bool>> valArray(bbox, reinterpret_cast<math::Vec3<bool>*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<Int16>()) {
        tools::Dense<math::Vec3<Int16>> valArray(bbox, reinterpret_cast<math::Vec3<Int16>*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<Int32>()) {
        tools::Dense<math::Vec3<Int32>> valArray(bbox, reinterpret_cast<math::Vec3<Int32>*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<Int64>()) {
        tools::Dense<math::Vec3<Int64>> valArray(bbox, reinterpret_cast<math::Vec3<Int64>*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<Index32>()) {
        tools::Dense<math::Vec3<Index32>> valArray(bbox, reinterpret_cast<math::Vec3<Index32>*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else if (array.dtype() == nb::dtype<Index64>()) {
        tools::Dense<math::Vec3<Index64>> valArray(bbox, reinterpret_cast<math::Vec3<Index64>*>(array.data()));
        tools::copyFromDense(valArray, grid, tolerance);
    } else {
        throw openvdb::TypeError();
    }
}

template<typename GridType>
inline void
copyToArrayScalar(GridType& grid, nb::ndarray<nb::numpy> array, const Coord& origin)
{
    if (array.ndim() != 3) {
        std::stringstream ss;
        ss << "Expected array with ndim = 3, found array with ndim = " << array.ndim();
        throw nb::value_error(ss.str().c_str());
    }

    // Compute the bounding box of the region of the grid that is to be copied from or to.
    // origin specifies the coordinates (i, j, k) of the voxel at which to start populating data.
    // Voxel (i, j, k) will correspond to array element (0, 0, 0).
    CoordBBox bbox(origin, origin + Coord(static_cast<Int32>(array.shape(0)), static_cast<Int32>(array.shape(1)), static_cast<Int32>(array.shape(2))) - Coord(1));
    if (array.dtype() == nb::dtype<float>()) {
        tools::Dense<float> valArray(bbox, reinterpret_cast<float*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<double>()) {
        tools::Dense<double> valArray(bbox, reinterpret_cast<double*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<bool>()) {
        tools::Dense<bool> valArray(bbox, reinterpret_cast<bool*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<Int16>()) {
        tools::Dense<Int16> valArray(bbox, reinterpret_cast<Int16*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<Int32>()) {
        tools::Dense<Int32> valArray(bbox, reinterpret_cast<Int32*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<Int64>()) {
        tools::Dense<Int64> valArray(bbox, reinterpret_cast<Int64*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<Index32>()) {
        tools::Dense<Index32> valArray(bbox, reinterpret_cast<Index32*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<Index64>()) {
        tools::Dense<Index64> valArray(bbox, reinterpret_cast<Index64*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else {
        throw openvdb::TypeError();
    }
}

template<typename GridType>
inline void
copyToArrayVector(GridType& grid, nb::ndarray<nb::numpy> array, const Coord& origin)
{
    if (array.ndim() != 4) {
        std::stringstream ss;
        ss << "Expected array with ndim = 4, found array with ndim = " << array.ndim();
        throw nb::value_error(ss.str().c_str());
    }

    // Compute the bounding box of the region of the grid that is to be copied from or to.
    // origin specifies the coordinates (i, j, k) of the voxel at which to start populating data.
    // Voxel (i, j, k) will correspond to array element (0, 0, 0).
    CoordBBox bbox(origin, origin + Coord(static_cast<Int32>(array.shape(0)), static_cast<Int32>(array.shape(1)), static_cast<Int32>(array.shape(2))) - Coord(1));
    if (array.dtype() == nb::dtype<float>()) {
        tools::Dense<math::Vec3<float>> valArray(bbox, reinterpret_cast<math::Vec3<float>*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<double>()) {
        tools::Dense<math::Vec3<double>> valArray(bbox, reinterpret_cast<math::Vec3<double>*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<bool>()) {
        tools::Dense<math::Vec3<bool>> valArray(bbox, reinterpret_cast<math::Vec3<bool>*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<Int16>()) {
        tools::Dense<math::Vec3<Int16>> valArray(bbox, reinterpret_cast<math::Vec3<Int16>*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<Int32>()) {
        tools::Dense<math::Vec3<Int32>> valArray(bbox, reinterpret_cast<math::Vec3<Int32>*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<Int64>()) {
        tools::Dense<math::Vec3<Int64>> valArray(bbox, reinterpret_cast<math::Vec3<Int64>*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<Index32>()) {
        tools::Dense<math::Vec3<Index32>> valArray(bbox, reinterpret_cast<math::Vec3<Index32>*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else if (array.dtype() == nb::dtype<Index64>()) {
        tools::Dense<math::Vec3<Index64>> valArray(bbox, reinterpret_cast<math::Vec3<Index64>*>(array.data()));
        tools::copyToDense(grid, valArray);
    } else {
        throw openvdb::TypeError();
    }
}

#endif // defined(PY_OPENVDB_USE_NUMPY)


////////////////////////////////////////


#ifndef PY_OPENVDB_USE_NUMPY

template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(nb::object, nb::object, nb::object, nb::object, nb::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw nb::python_error();
    return typename GridType::Ptr();
}

template<typename GridType>
inline typename GridType::Ptr
meshToSignedDistanceField(nb::object, nb::object, nb::object, nb::object, nb::object, nb::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw nb::python_error();
    return typename GridType::Ptr();
}

template<typename GridType>
inline nb::object
volumeToQuadMesh(const GridType&, nb::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw nb::python_error();
    return nb::object();
}

template<typename GridType>
inline nb::object
volumeToMesh(const GridType&, nb::object, nb::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw nb::python_error();
    return nb::object();
}

#else // if defined(PY_OPENVDB_USE_NUMPY)

/// @brief Given NumPy arrays of points, triangle indices, and quad indices,
/// call tools::meshToLevelSet() to generate a level set grid.
template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
        nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu> pointsObj,
        std::optional<nb::ndarray<Index32, nb::shape<-1, 3>, nb::device::cpu>>& trianglesObj,
        std::optional<nb::ndarray<Index32, nb::shape<-1, 4>, nb::device::cpu>>& quadsObj,
        math::Transform::Ptr xform, float halfWidth)
{
    // Extract the list of mesh vertices from the arguments to this method.
    std::vector<Vec3s> points(pointsObj.shape(0));
    // Copy values from the array to the vector.
    for (size_t i = 0; i < pointsObj.shape(0); ++i)
        points[i] = Vec3s(pointsObj(i, 0), pointsObj(i, 1), pointsObj(i, 2));

    // Extract the list of triangle indices from the arguments to this method.
    std::vector<Vec3I> triangles;
    if (trianglesObj) {
        triangles.resize(trianglesObj->shape(0));
         for (size_t i = 0; i < trianglesObj->shape(0); ++i)
             triangles[i] = Vec3I((*trianglesObj)(i, 0), (*trianglesObj)(i, 1), (*trianglesObj)(i, 2));
    }

    // Extract the list of quad indices from the arguments to this method.
     std::vector<Vec4I> quads;
    if (quadsObj) {
        quads.resize(quadsObj->shape(0));
         for (size_t i = 0; i < quadsObj->shape(0); ++i)
             quads[i] = Vec4I((*quadsObj)(i, 0), (*quadsObj)(i, 1), (*quadsObj)(i, 2), (*quadsObj)(i, 3));
    }

    // Generate and return a level set grid.
    if (xform) {
        return tools::meshToLevelSet<GridType>(*xform, points, triangles, quads, halfWidth);
    }
    else {
        math::Transform::Ptr identity = math::Transform::createLinearTransform();
        return tools::meshToLevelSet<GridType>(*identity, points, triangles, quads, halfWidth);
    }
}

template<typename GridType>
inline typename GridType::Ptr
meshToSignedDistanceField(
        nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu> pointsObj,
        std::optional<nb::ndarray<Index32, nb::shape<-1, 3>, nb::device::cpu>>& trianglesObj,
        std::optional<nb::ndarray<Index32, nb::shape<-1, 4>, nb::device::cpu>>& quadsObj,
        math::Transform::Ptr xform, float exBandWidth, float inBandWidth)
{
    // Extract the list of mesh vertices from the arguments to this method.
    std::vector<Vec3s> points(pointsObj.shape(0));
    // Copy values from the array to the vector.
    for (size_t i = 0; i < pointsObj.shape(0); ++i)
        points[i] = Vec3s(pointsObj(i, 0), pointsObj(i, 1), pointsObj(i, 2));

    // Extract the list of triangle indices from the arguments to this method.
    std::vector<Vec3I> triangles;
    if (trianglesObj) {
        triangles.resize(trianglesObj->shape(0));
         for (size_t i = 0; i < trianglesObj->shape(0); ++i)
             triangles[i] = Vec3I((*trianglesObj)(i, 0), (*trianglesObj)(i, 1), (*trianglesObj)(i, 2));
    }

    // Extract the list of quad indices from the arguments to this method.
     std::vector<Vec4I> quads;
    if (quadsObj) {
        quads.resize(quadsObj->shape(0));
         for (size_t i = 0; i < quadsObj->shape(0); ++i)
             quads[i] = Vec4I((*quadsObj)(i, 0), (*quadsObj)(i, 1), (*quadsObj)(i, 2), (*quadsObj)(i, 3));
    }

    // Generate and return a level set grid.
    if (xform) {
        return tools::meshToSignedDistanceField<GridType>(*xform, points, triangles, quads, exBandWidth, inBandWidth);
    }
    else {
        math::Transform::Ptr identity = math::Transform::createLinearTransform();
        return tools::meshToSignedDistanceField<GridType>(*identity, points, triangles, quads, exBandWidth, inBandWidth);
    }
}

template<typename GridType, typename std::enable_if_t<!std::is_scalar<typename GridType::ValueType>::value>* = nullptr>
inline std::tuple<nb::ndarray<nb::numpy, float>, nb::ndarray<nb::numpy, Index32> >
volumeToQuadMesh(const GridType&, double)
{
    OPENVDB_THROW(TypeError, "volume to mesh conversion is supported only for scalar grids");
}

template<typename GridType, typename std::enable_if_t<std::is_scalar<typename GridType::ValueType>::value>* = nullptr>
inline std::tuple<nb::ndarray<nb::numpy, float>, nb::ndarray<nb::numpy, Index32> >
volumeToQuadMesh(const GridType& grid, double isovalue)
{
    // Mesh the input grid and populate lists of mesh vertices and face vertex indices.
    auto points = new std::vector<Vec3s>();
    auto quads = new std::vector<Vec4I>();
    tools::volumeToMesh(grid, *points, *quads, isovalue);

    nb::capsule pointsDeleter(points, [](void* p) noexcept {
        delete (std::vector<Vec3s>*) p;
    });
    nb::ndarray<nb::numpy, float> pointArray(points->data(), {points->size(), 3}, pointsDeleter, {3, 1});

    nb::capsule quadsDeleter(quads, [](void* p) noexcept {
        delete (std::vector<Vec4I>*) p;
    });
    nb::ndarray<nb::numpy, Index32> quadArray(quads->data(), {quads->size(), 4}, quadsDeleter, {4, 1});

    return std::make_tuple(pointArray, quadArray);
}

template<typename GridType, typename std::enable_if_t<!std::is_scalar<typename GridType::ValueType>::value>* = nullptr>
inline std::tuple<nb::ndarray<nb::numpy, float>, nb::ndarray<nb::numpy, Index32>, nb::ndarray<nb::numpy, Index32> >
volumeToMesh(const GridType&, double, double)
{
    OPENVDB_THROW(TypeError, "volume to mesh conversion is supported only for scalar grids");
}

template<typename GridType, typename std::enable_if_t<std::is_scalar<typename GridType::ValueType>::value>* = nullptr>
inline std::tuple<nb::ndarray<nb::numpy, float>, nb::ndarray<nb::numpy, Index32>, nb::ndarray<nb::numpy, Index32> >
volumeToMesh(const GridType& grid, double isovalue, double adaptivity)
{
    // Mesh the input grid and populate lists of mesh vertices and face vertex indices.
    auto points = new std::vector<Vec3s>();
    auto triangles = new std::vector<Vec3I>();
    auto quads = new std::vector<Vec4I>();
    tools::volumeToMesh(grid, *points, *triangles, *quads, isovalue, adaptivity);

    nb::capsule pointsDeleter(points, [](void* p) noexcept {
        delete (std::vector<Vec3s>*) p;
    });
    nb::ndarray<nb::numpy, float> pointArray(points->data(), {points->size(), 3}, pointsDeleter, {3, 1});

    nb::capsule trianglesDeleter(triangles, [](void* p) noexcept {
        delete (std::vector<Vec3I>*) p;
    });
    nb::ndarray<nb::numpy, Index32> triangleArray(triangles->data(), {triangles->size(), 3}, trianglesDeleter, {3, 1});

    nb::capsule quadsDeleter(quads, [](void* p) noexcept {
        delete (std::vector<Vec4I>*) p;
    });
    nb::ndarray<nb::numpy, Index32> quadArray(quads->data(), {quads->size(), 4}, quadsDeleter, {4, 1});

    return std::make_tuple(pointArray, triangleArray, quadArray);
}

#endif // defined(PY_OPENVDB_USE_NUMPY)


////////////////////////////////////////


template<typename GridType, typename IterType>
inline void
applyMap(const char* methodName, GridType& grid, nb::object funcObj)
{
    using ValueT = typename GridType::ValueType;

    for (IterType it = grid.tree().template begin<IterType>(); it; ++it) {
        // Evaluate the functor.
        nb::object result = funcObj(*it);

        // Verify that the result is of type GridType::ValueType.
        try {
            nb::cast<ValueT>(result);
        } catch (nb::cast_error&) {
            std::ostringstream os;
            os << "expected callable argument to ";
            os << pyutil::GridTraits<GridType>::name();
            os << "." << methodName << "() to return ";
            os << openvdb::typeNameAsString<ValueT>();
            os << ", found " << pyutil::className(result);
            throw nb::type_error(os.str().c_str());
        }

        it.setValue(nb::cast<ValueT>(result));
    }
}


template<typename GridType>
inline void
mapOn(GridType& grid, nb::object funcObj)
{
    applyMap<GridType, typename GridType::ValueOnIter>("mapOn", grid, funcObj);
}


template<typename GridType>
inline void
mapOff(GridType& grid, nb::object funcObj)
{
    applyMap<GridType, typename GridType::ValueOffIter>("mapOff", grid, funcObj);
}


template<typename GridType>
inline void
mapAll(GridType& grid, nb::object funcObj)
{
    applyMap<GridType, typename GridType::ValueAllIter>("mapAll", grid, funcObj);
}


////////////////////////////////////////


template<typename GridType>
struct TreeCombineOp
{
    using TreeT = typename GridType::TreeType;
    using ValueT = typename GridType::ValueType;

    TreeCombineOp(const std::function<typename GridType::ValueType(typename GridType::ValueType, typename GridType::ValueType)>& _op): op(_op) {}
    void operator()(const ValueT& a, const ValueT& b, ValueT& result)
    {
        result = op(a, b);
    }
    const std::function<typename GridType::ValueType(typename GridType::ValueType, typename GridType::ValueType)>& op;
};


template<typename GridType>
inline void
combine(GridType& grid, GridType& otherGrid, const std::function<typename GridType::ValueType(typename GridType::ValueType, typename GridType::ValueType)>& func)
{
    TreeCombineOp<GridType> op(func);
    grid.tree().combine(otherGrid.tree(), op, /*prune=*/true);
}


////////////////////////////////////////


template<typename GridType>
inline typename GridType::Ptr
createLevelSetSphere(float radius, const openvdb::Vec3f& center, float voxelSize, float halfWidth)
{
    return tools::createLevelSetSphere<GridType>(radius, center, voxelSize, halfWidth);
}


////////////////////////////////////////


template<typename GridT, typename IterT> class IterWrap; // forward declaration

//
// Type traits for various iterators
//
template<typename GridT, typename IterT> struct IterTraits
{
    // IterT    the type of the iterator
    // name()   function returning the base name of the iterator type (e.g., "ValueOffIter")
    // descr()  function returning a string describing the iterator
    // begin()  function returning a begin iterator for a given grid
};

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueOnCIter>
{
    using IterT = typename GridT::ValueOnCIter;
    static std::string name() { return "ValueOnCIter"; }
    static std::string descr()
    {
        return std::string("Read-only iterator over the active values (tile and voxel)\nof a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<const GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<const GridT, IterT>(g, g->cbeginValueOn());
    }
}; // IterTraits<ValueOnCIter>

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueOffCIter>
{
    using IterT = typename GridT::ValueOffCIter;
    static std::string name() { return "ValueOffCIter"; }
    static std::string descr()
    {
        return std::string("Read-only iterator over the inactive values (tile and voxel)\nof a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<const GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<const GridT, IterT>(g, g->cbeginValueOff());
    }
}; // IterTraits<ValueOffCIter>

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueAllCIter>
{
    using IterT = typename GridT::ValueAllCIter;
    static std::string name() { return "ValueAllCIter"; }
    static std::string descr()
    {
        return std::string("Read-only iterator over all tile and voxel values of a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<const GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<const GridT, IterT>(g, g->cbeginValueAll());
    }
}; // IterTraits<ValueAllCIter>

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueOnIter>
{
    using IterT = typename GridT::ValueOnIter;
    static std::string name() { return "ValueOnIter"; }
    static std::string descr()
    {
        return std::string("Read/write iterator over the active values (tile and voxel)\nof a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<GridT, IterT>(g, g->beginValueOn());
    }
}; // IterTraits<ValueOnIter>

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueOffIter>
{
    using IterT = typename GridT::ValueOffIter;
    static std::string name() { return "ValueOffIter"; }
    static std::string descr()
    {
        return std::string("Read/write iterator over the inactive values (tile and voxel)\nof a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<GridT, IterT>(g, g->beginValueOff());
    }
}; // IterTraits<ValueOffIter>

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueAllIter>
{
    using IterT = typename GridT::ValueAllIter;
    static std::string name() { return "ValueAllIter"; }
    static std::string descr()
    {
        return std::string("Read/write iterator over all tile and voxel values of a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<GridT, IterT>(g, g->beginValueAll());
    }
}; // IterTraits<ValueAllIter>


////////////////////////////////////////


// Helper class to modify a grid through a non-const iterator
template<typename GridT, typename IterT>
struct IterItemSetter
{
    using ValueT = typename GridT::ValueType;
    static void setValue(const IterT& iter, const ValueT& val) { iter.setValue(val); }
    static void setActive(const IterT& iter, bool on) { iter.setActiveState(on); }
};

// Partial specialization for const iterators
template<typename GridT, typename IterT>
struct IterItemSetter<const GridT, IterT>
{
    using ValueT = typename GridT::ValueType;
    static void setValue(const IterT&, const ValueT&)
    {
        throw nb::attribute_error("can't set attribute 'value'");
    }
    static void setActive(const IterT&, bool /*on*/)
    {
        throw nb::attribute_error("can't set attribute 'active'");
    }
};


/// @brief Value returned by the next() method of a grid's value iterator
/// @details This class allows both dictionary-style (e.g., items["depth"]) and
/// attribute access (e.g., items.depth) to the items returned by an iterator.
/// @todo Create a reusable base class for "named dicts" like this?
template<typename _GridT, typename _IterT>
class IterValueProxy
{
public:
    using GridT = _GridT;
    using IterT = _IterT;
    using ValueT = typename GridT::ValueType;
    using SetterT = IterItemSetter<GridT, IterT>;

    IterValueProxy(typename GridT::ConstPtr grid, const IterT& iter): mGrid(grid), mIter(iter) {}

    IterValueProxy copy() const { return *this; }

    typename GridT::ConstPtr parent() const { return mGrid; }

    ValueT getValue() const { return *mIter; }
    bool getActive() const { return mIter.isValueOn(); }
    Index getDepth() const { return mIter.getDepth(); }
    Coord getBBoxMin() const { return mIter.getBoundingBox().min(); }
    Coord getBBoxMax() const { return mIter.getBoundingBox().max(); }
    Index64 getVoxelCount() const { return mIter.getVoxelCount(); }

    void setValue(const ValueT& val) { SetterT::setValue(mIter, val); }
    void setActive(bool on) { SetterT::setActive(mIter, on); }

    /// Return this dictionary's keys as a list of C strings.
    static const char* const * keys()
    {
        static const char* const sKeys[] = {
            "value", "active", "depth", "min", "max", "count", nullptr
        };
        return sKeys;
    }

    /// Return @c true if the given string is a valid key.
    static bool hasKey(const std::string& key)
    {
        for (int i = 0; keys()[i] != nullptr; ++i) {
            if (key == keys()[i]) return true;
        }
        return false;
    }

    /// Return this dictionary's keys as a Python list of Python strings.
    static std::vector<std::string> getKeys()
    {
        std::vector<std::string> keyList;
        for (int i = 0; keys()[i] != nullptr; ++i)
            keyList.push_back(keys()[i]);
        return keyList;
    }

    /// @brief Return the value for the given key.
    /// @throw KeyError if the key is invalid
    nb::object getItem(nb::object keyObj) const
    {
        if (nb::isinstance<std::string>(keyObj)) {
            const std::string key = nb::cast<std::string>(keyObj);
            if (key == "value") return nb::cast(this->getValue());
            else if (key == "active") return nb::cast(this->getActive());
            else if (key == "depth") return nb::cast(this->getDepth());
            else if (key == "min") return nb::cast(this->getBBoxMin());
            else if (key == "max") return nb::cast(this->getBBoxMax());
            else if (key == "count") return nb::cast(this->getVoxelCount());
        }
        throw nb::key_error(nb::cast<std::string>(keyObj.attr("__repr__")()).c_str());
        return nb::object();
    }

    /// @brief Set the value for the given key.
    /// @throw KeyError if the key is invalid
    /// @throw AttributeError if the key refers to a read-only item
    void setItem(nb::object keyObj, nb::object valObj)
    {
        if (nb::isinstance<std::string>(keyObj)) {
            const std::string key = nb::cast<std::string>(keyObj);
            if (key == "value") {
                this->setValue(nb::cast<ValueT>(valObj)); return;
            } else if (key == "active") {
                this->setActive(nb::cast<bool>(valObj)); return;
            } else if (this->hasKey(key)) {
                std::ostringstream os;
                os << "can't set attribute '";
                os << nb::cast<std::string>(keyObj.attr("__repr__")());
                os << "'";
                throw nb::attribute_error(os.str().c_str());
            }
        }
        throw nb::key_error(nb::cast<std::string>(keyObj.attr("__repr__")()).c_str());
    }

    bool operator==(const IterValueProxy& other) const
    {
        return (other.getActive() == this->getActive()
            && other.getDepth() == this->getDepth()
            && math::isExactlyEqual(other.getValue(), this->getValue())
            && other.getBBoxMin() == this->getBBoxMin()
            && other.getBBoxMax() == this->getBBoxMax()
            && other.getVoxelCount() == this->getVoxelCount());
    }
    bool operator!=(const IterValueProxy& other) const { return !(*this == other); }

    /// Print this dictionary to a stream.
    std::ostream& put(std::ostream& os) const
    {
        // valuesAsStrings = ["%s: %s" % key, repr(this[key]) for key in this.keys()]
        nb::list valuesAsStrings;
        for (int i = 0; this->keys()[i] != nullptr; ++i) {
            nb::str
                key(this->keys()[i]),
                val(this->getItem(key).attr("__repr__")());
            valuesAsStrings.append(nb::str("'%s': %s").format(nb::make_tuple(key, val)));
        }
        // print ", ".join(valuesAsStrings)
        nb::object joined = nb::str(", ").attr("join")(valuesAsStrings);
        std::string s = nb::cast<std::string>(joined);
        os << "{" << s << "}";
        return os;
    }
    /// Return a string describing this dictionary.
    std::string info() const { std::ostringstream os; os << *this; return os.str(); }

private:
    // To keep the iterator's grid from being deleted (leaving the iterator dangling),
    // store a shared pointer to the grid.
    const typename GridT::ConstPtr mGrid;
    const IterT mIter; // the iterator may not be incremented
}; // class IterValueProxy


template<typename GridT, typename IterT>
inline std::ostream&
operator<<(std::ostream& os, const IterValueProxy<GridT, IterT>& iv) { return iv.put(os); }


////////////////////////////////////////


/// Wrapper for a grid's value iterator classes
template<typename _GridT, typename _IterT>
class IterWrap
{
public:
    using GridT = _GridT;
    using IterT = _IterT;
    using ValueT = typename GridT::ValueType;
    using IterValueProxyT = IterValueProxy<GridT, IterT>;
    using Traits = IterTraits<GridT, IterT>;

    IterWrap(typename GridT::ConstPtr grid, const IterT& iter): mGrid(grid), mIter(iter) {}

    typename GridT::ConstPtr parent() const { return mGrid; }

    /// Return an IterValueProxy for the current iterator position.
    IterValueProxyT next()
    {
        if (!mIter) {
            throw nb::stop_iteration("no more values");
        }
        IterValueProxyT result(mGrid, mIter);
        ++mIter;
        return result;
    }

    static nb::object returnSelf(const nb::object& obj) { return obj; }

    /// @brief Define a Python wrapper class for this C++ class and another for
    /// the IterValueProxy class returned by iterators of this type.
    static void wrap(nb::module_ m)
    {
        const std::string
            gridClassName = pyutil::GridTraits<typename std::remove_const<GridT>::type>::name(),
            iterClassName = gridClassName + Traits::name(),
            valueClassName = gridClassName + Traits::name() + "Value";

        nb::class_<IterWrap>(m,
            iterClassName.c_str(),
            /*docstring=*/Traits::descr().c_str())

            .def_prop_ro("parent", &IterWrap::parent,
                ("the " + gridClassName + " over which to iterate").c_str())

            .def("next", &IterWrap::next)
            .def("__next__", &IterWrap::next)
            .def("__iter__", &returnSelf);

        nb::class_<IterValueProxyT>(m,
            valueClassName.c_str(),
            /*docstring=*/("Proxy for a tile or voxel value in a " + gridClassName).c_str())

            .def("copy", &IterValueProxyT::copy,
                "Return a shallow copy of this value, i.e., one that shares its data with the original.")

            .def_prop_ro("parent", &IterValueProxyT::parent,
                ("the " + gridClassName + " to which this value belongs").c_str())

            .def("__str__", &IterValueProxyT::info)
            .def("__repr__", &IterValueProxyT::info)

            .def("__eq__", &IterValueProxyT::operator==)
            .def("__ne__", &IterValueProxyT::operator!=)

            .def_prop_rw("value", &IterValueProxyT::getValue, &IterValueProxyT::setValue,
                "value of this tile or voxel")
            .def_prop_rw("active", &IterValueProxyT::getActive, &IterValueProxyT::setActive,
                "active state of this tile or voxel")
            .def_prop_ro("depth", &IterValueProxyT::getDepth,
                "tree depth at which this value is stored")
            .def_prop_ro("min", &IterValueProxyT::getBBoxMin,
                "lower bound of the axis-aligned bounding box of this tile or voxel")
            .def_prop_ro("max", &IterValueProxyT::getBBoxMax,
                "upper bound of the axis-aligned bounding box of this tile or voxel")
            .def_prop_ro("count", &IterValueProxyT::getVoxelCount,
                "number of voxels spanned by this value")

            .def_static("keys", &IterValueProxyT::getKeys,
                "Return a list of keys for this tile or voxel.")
            .def_static("__contains__", &IterValueProxyT::hasKey,
                "Return True if the given key exists.")
            .def("__getitem__", &IterValueProxyT::getItem,
                "Return the value of the item with the given key.")
            .def("__setitem__", &IterValueProxyT::setItem,
                "Set the value of the item with the given key.");
    }

private:
    // To keep this iterator's grid from being deleted, leaving the iterator dangling,
    // store a shared pointer to the grid.
    const typename GridT::ConstPtr mGrid;
    IterT mIter;
}; // class IterWrap


////////////////////////////////////////


template<typename GridType>
struct PickleSuite
{
    /// Return a tuple representing the state of the given Grid.
    static std::tuple<nb::bytes> getState(const typename GridType::Ptr& grid)
    {
        // Serialize the Grid to a string.
        std::ostringstream ostr(std::ios_base::binary);
        {
            openvdb::io::Stream strm(ostr);
            strm.setGridStatsMetadataEnabled(false);
            strm.write(openvdb::GridPtrVec(1, grid));
        }

        // Construct a state tuple for the serialized Grid.
        // Convert the byte string to a "bytes" sequence.
        const std::string s = ostr.str();
        nb::bytes bytesObj(s.c_str(), s.length());
        return std::make_tuple(bytesObj);
    }

    /// Restore the given Grid to a saved state.
    static void setState(GridType& grid, std::tuple<nb::bytes> state)
    {
        nb::bytes bytesObj = std::get<0>(state);
        std::string serialized(bytesObj.c_str(), bytesObj.c_str() + bytesObj.size());

        // Restore the internal state of the C++ object.
        GridPtrVecPtr grids;
        {
            std::istringstream istr(serialized, std::ios_base::binary);
            io::Stream strm(istr);
            grids = strm.getGrids(); // (note: file-level metadata is ignored)
        }
        if (grids && !grids->empty()) {
            if (typename GridType::Ptr savedGrid = gridPtrCast<GridType>((*grids)[0])) {
                new (&grid) GridType(*savedGrid);
                return;
            }
        }

        new (&grid) GridType();
    }
}; // struct PickleSuite


////////////////////////////////////////


/// Create a Python wrapper for a particular template instantiation of Grid.
template<typename GridType>
inline nb::class_<GridType, GridBase>
exportGrid(nb::module_ m)
{
    using ValueT = typename GridType::ValueType;
    using Traits = pyutil::GridTraits<GridType>;

    using ValueOnCIterT = typename GridType::ValueOnCIter;
    using ValueOffCIterT = typename GridType::ValueOffCIter;
    using ValueAllCIterT = typename GridType::ValueAllCIter;
    using ValueOnIterT = typename GridType::ValueOnIter;
    using ValueOffIterT = typename GridType::ValueOffIter;
    using ValueAllIterT = typename GridType::ValueAllIter;

    const std::string pyGridTypeName = Traits::name();
    std::stringstream docstream;
    docstream << "Initialize with a background value of " << pyGrid::getZeroValue<GridType>() << ".";
    std::string docstring = docstream.str();

    // Define the Grid wrapper class and make it the current scope.
    nb::class_<GridType, GridBase> typedGridClass(m,
        /*classname=*/(Traits::name()),
        /*docstring=*/(Traits::descr()).c_str());
    typedGridClass
        .def(nb::init<>(), docstring.c_str())
        .def(nb::init<const ValueT&>(), nb::arg("background"),
            "Initialize with the given background value.")

        .def("copy", &pyGrid::copyGrid<GridType>,
            "Return a shallow copy of this grid, i.e., a grid that shares its voxel data with this grid.")
        .def("deepCopy", &GridType::deepCopy,
            "Return a deep copy of this grid.")

        .def("__getstate__", &PickleSuite<GridType>::getState)
        .def("__setstate__", &PickleSuite<GridType>::setState)

        .def("sharesWith", &pyGrid::sharesWith<GridType>,
            "Return whether or not this grid shares its voxel data with the given grid.")

        .def_prop_ro_static("valueTypeName", [](const nb::object&) { return pyGrid::getValueType<GridType>(); }, "Name of this grid's value type")
        .def_prop_ro_static("zeroValue", [](const nb::object&) { return pyGrid::getZeroValue<GridType>(); }, "Zero, as expressed in this grid's value type")
        .def_prop_ro_static("oneValue", [](const nb::object&) { return pyGrid::getOneValue<GridType>(); }, "One, as expressed in this grid's value type")
        /// @todo Is Grid.typeName ever needed?
        //.def_prop_rw_static("typeName", &GridType::gridType, "Name of this grid's type")

        .def_prop_rw("background",
            &pyGrid::getGridBackground<GridType>, &pyGrid::setGridBackground<GridType>,
            "value of this grid's background voxels")

        .def("getAccessor", &pyGrid::getAccessor<GridType>,
            "Return an accessor that provides random read and write access to this grid's voxels.")
        .def("getConstAccessor", &pyGrid::getConstAccessor<GridType>,
            "Return an accessor that provides random read-only access to this grid's voxels.")

        //
        // Statistics
        //
        .def("evalLeafBoundingBox", &pyGrid::evalLeafBoundingBox<GridType>,
            "Return the coordinates of opposite corners of the axis-aligned bounding box of all leaf nodes.")
        .def("evalLeafDim", &pyGrid::evalLeafDim<GridType>,
            "Return the dimensions of the axis-aligned bounding box of all leaf nodes.")

        .def_prop_ro("treeDepth", &pyGrid::treeDepth<GridType>,
            "depth of this grid's tree from root node to leaf node")
        .def("nodeLog2Dims", &pyGrid::getNodeLog2Dims<GridType>,
            "list of Log2Dims of the nodes of this grid's tree in order from root to leaf")

        .def("leafCount", &pyGrid::leafCount<GridType>,
            "Return the number of leaf nodes in this grid's tree.")
        .def("nonLeafCount", &pyGrid::nonLeafCount<GridType>,
            "Return the number of non-leaf nodes in this grid's tree.")

        .def("activeLeafVoxelCount", &pyGrid::activeLeafVoxelCount<GridType>,
            "Return the number of active voxels that are stored in the leaf nodes of this grid's tree.")

        .def("evalMinMax", &pyGrid::evalMinMax<GridType>,
            "Return the minimum and maximum active values in this grid.")

        .def("getIndexRange", &pyGrid::getIndexRange<GridType>,
            "Return the minimum and maximum coordinates that are represented in this grid.  These might include background voxels.")
        //.def("expand", &pyGrid::expandIndexRange<GridType>,
        //    nb::arg("xyz"),
        //    "Expand this grid's index range to include the given coordinates.")

        //
        // Tools
        //
        .def("fill", &pyGrid::fill<GridType>,
            nb::arg("min"), nb::arg("max"), nb::arg("value"), nb::arg("active")=true,
            "Set all voxels within a given axis-aligned box to a constant value (either active or inactive).")
        .def("signedFloodFill", &pyGrid::signedFloodFill<GridType>,
            "Propagate the sign from a narrow-band level set into inactive voxels and tiles.")
        .def("convertToQuads",
            &pyGrid::volumeToQuadMesh<GridType>,
            nb::arg("isovalue")=0,
            "Uniformly mesh a scalar grid that has a continuous isosurface\n"
            "at the given isovalue.  Return a NumPy array of world-space\n"
            "points and a NumPy array of 4-tuples of point indices, which\n"
            "specify the vertices of the quadrilaterals that form the mesh.")
        .def("convertToPolygons",
            &pyGrid::volumeToMesh<GridType>,
            nb::arg("isovalue")=0, nb::arg("adaptivity")=0,
            "Adaptively mesh a scalar grid that has a continuous isosurface\n"
            "at the given isovalue.  Return a NumPy array of world-space\n"
            "points and NumPy arrays of 3- and 4-tuples of point indices,\n"
            "which specify the vertices of the triangles and quadrilaterals\n"
            "that form the mesh.  Adaptivity can vary from 0 to 1, where 0\n"
            "produces a high-polygon-count mesh that closely approximates\n"
            "the isosurface, and 1 produces a lower-polygon-count mesh\n"
            "with some loss of surface detail.")
        .def_static("createLevelSetFromPolygons",
            &pyGrid::meshToLevelSet<GridType>,
            nb::arg("points"),
#ifdef PY_OPENVDB_USE_NUMPY
            nb::arg("triangles")=nb::none(),
            nb::arg("quads")=nb::none(),
#else
            nb::arg("triangles")=std::vector<Index32>(),
            nb::arg("quads")=std::vector<Index32>(),
#endif
            nb::arg("transform")=openvdb::math::Transform(),
            nb::arg("halfWidth")=openvdb::LEVEL_SET_HALF_WIDTH,
             "Convert a triangle and/or quad mesh to a narrow-band level set volume.\n"
             "The mesh must form a closed surface, but the surface need not be\n"
             "manifold and may have self intersections and degenerate faces.\n"
             "The mesh is described by a NumPy array of world-space points\n"
             "and NumPy arrays of 3- and 4-tuples of point indices that specify\n"
             "the vertices of the triangles and quadrilaterals that form the mesh.\n"
             "Either the triangle or the quad array may be empty or None.\n"
             "The resulting volume will have the given transform (or the identity\n"
             "transform if no transform is given) and a narrow band width of\n"
             "2 x halfWidth voxels.")
        .def_static("createLevelSetFromPolygons",
            &pyGrid::meshToSignedDistanceField<GridType>,
            nb::arg("points"),
#ifdef PY_OPENVDB_USE_NUMPY
            nb::arg("triangles")=nb::none(),
            nb::arg("quads")=nb::none(),
#else
            nb::arg("triangles")=std::vector<Index32>(),
            nb::arg("quads")=std::vector<Index32>(),
#endif
            nb::arg("transform")=openvdb::math::Transform(),
            nb::arg("exBandWidth")=openvdb::LEVEL_SET_HALF_WIDTH,
            nb::arg("inBandWidth")=openvdb::LEVEL_SET_HALF_WIDTH,
             "Convert a triangle and/or quad mesh to a narrow-band level set volume.\n"
             "The mesh must form a closed surface, but the surface need not be\n"
             "manifold and may have self intersections and degenerate faces.\n"
             "The mesh is described by a NumPy array of world-space points\n"
             "and NumPy arrays of 3- and 4-tuples of point indices that specify\n"
             "the vertices of the triangles and quadrilaterals that form the mesh.\n"
             "Either the triangle or the quad array may be empty or None.\n"
             "The resulting volume will have the given transform (or the identity\n"
             "transform if no transform is given) and a narrow band width of\n"
             "exBandWidth exterior voxels and inBandWidth interior voxels.")


        .def("prune", &pyGrid::prune<GridType>,
            nb::arg("tolerance") = 0,
            "Remove nodes whose values all have the same active state and are equal to within a given tolerance.")
        .def("pruneInactive", &pyGrid::pruneInactive<GridType>,
            nb::arg("value") = nb::none(),
            "Remove nodes whose values are all inactive and replace them with background tiles.")

        .def("merge", &GridType::merge,
            "Move child nodes from the other grid into this grid wherever\n"
            "those nodes correspond to constant-value tiles in this grid,\n"
            "and replace leaf-level inactive voxels in this grid with\n"
            "corresponding voxels in the other grid that are active.\n\n"
            "Note: this operation always empties the other grid.")

        .def("mapOn", &pyGrid::mapOn<GridType>,
            nb::arg("function"),
            "Iterate over all the active (\"on\") values (tile and voxel)\n"
            "of this grid and replace each value with function(value).\n\n"
            "Example: grid.mapOn(lambda x: x * 2 if x < 0.5 else x)")

        .def("mapOff", &pyGrid::mapOff<GridType>,
            nb::arg("function"),
            "Iterate over all the inactive (\"off\") values (tile and voxel)\n"
            "of this grid and replace each value with function(value).\n\n"
            "Example: grid.mapOff(lambda x: x * 2 if x < 0.5 else x)")

        .def("mapAll", &pyGrid::mapAll<GridType>,
            nb::arg("function"),
            "Iterate over all values (tile and voxel) of this grid\n"
            "and replace each value with function(value).\n\n"
            "Example: grid.mapAll(lambda x: x * 2 if x < 0.5 else x)")

        .def("combine", &pyGrid::combine<GridType>,
            nb::arg("grid"), nb::arg("function"),
            "Compute function(self, other) over all corresponding pairs\n"
            "of values (tile or voxel) of this grid and the other grid\n"
            "and store the result in this grid.\n\n"
            "Note: this operation always empties the other grid.\n\n"
            "Example: grid.combine(otherGrid, lambda a, b: min(a, b))")

        //
        // Iterators
        //
        .def("citerOnValues", &pyGrid::IterTraits<GridType, ValueOnCIterT>::begin,
            "Return a read-only iterator over this grid's active tile and voxel values.")
        .def("citerOffValues", &pyGrid::IterTraits<GridType, ValueOffCIterT>::begin,
            "Return a read-only iterator over this grid's inactive tile and voxel values.")
        .def("citerAllValues", &pyGrid::IterTraits<GridType, ValueAllCIterT>::begin,
            "Return a read-only iterator over all of this grid's tile and voxel values.")

        .def("iterOnValues", &pyGrid::IterTraits<GridType, ValueOnIterT>::begin,
            "Return a read/write iterator over this grid's active tile and voxel values.")
        .def("iterOffValues", &pyGrid::IterTraits<GridType, ValueOffIterT>::begin,
            "Return a read/write iterator over this grid's inactive tile and voxel values.")
        .def("iterAllValues", &pyGrid::IterTraits<GridType, ValueAllIterT>::begin,
            "Return a read/write iterator over all of this grid's tile and voxel values.");

    // Wrap const and non-const value accessors and expose them
    // as nested classes of the Grid class.
    pyAccessor::AccessorWrap<const GridType>::wrap(m);
    pyAccessor::AccessorWrap<GridType>::wrap(m);

    // Wrap tree value iterators and expose them as nested classes of the Grid class.
    IterWrap<const GridType, ValueOnCIterT>::wrap(m);
    IterWrap<const GridType, ValueOffCIterT>::wrap(m);
    IterWrap<const GridType, ValueAllCIterT>::wrap(m);
    IterWrap<GridType, ValueOnIterT>::wrap(m);
    IterWrap<GridType, ValueOffIterT>::wrap(m);
    IterWrap<GridType, ValueAllIterT>::wrap(m);

    // Add the Python type object for this grid type to the module-level list.
    nb::cast<nb::list>(m.attr("GridTypes")).append(m.attr(Traits::name()));

    return typedGridClass;
}

template<typename GridType>
inline void
exportScalarGrid(nb::module_ m)
{
    exportGrid<GridType>(m)
        .def("copyFromArray", &pyGrid::copyFromArrayScalar<GridType>,
           nb::arg("array").noconvert(), nb::arg("ijk")=Coord(0,0,0),
                nb::arg("tolerance")=pyGrid::getZeroValue<GridType>(),
           "Populate this grid, starting at voxel (i, j, k), with values\n"
           "from a three-dimensional array.  Mark voxels as inactive\n"
           "if and only if their values are equal to this grid's\n"
           "background value within the given tolerance.")
        .def("copyToArray", &pyGrid::copyToArrayScalar<GridType>,
            nb::arg("array").noconvert(), nb::arg("ijk")=Coord(0,0,0),
            "Populate a three-dimensional array with values\n"
            "from this grid, starting at voxel (i, j, k).");
}

template<typename GridType>
inline void
exportVectorGrid(nb::module_ m)
{
    exportGrid<GridType>(m)
        .def("copyFromArray", &pyGrid::copyFromArrayVector<GridType>,
           nb::arg("array").noconvert(), nb::arg("ijk")=Coord(0,0,0),
                nb::arg("tolerance")=pyGrid::getZeroValue<GridType>(),
           "Populate this grid, starting at voxel (i, j, k), with values\n"
           "from a four-dimensional array.  Mark voxels as inactive\n"
           "if and only if their values are equal to this grid's\n"
           "background value within the given tolerance.")
        .def("copyToArray", &pyGrid::copyToArrayVector<GridType>,
            nb::arg("array").noconvert(), nb::arg("ijk")=Coord(0,0,0),
            "Populate a four-dimensional array with values\n"
            "from this grid, starting at voxel (i, j, k).");
}

} // namespace pyGrid

#endif // OPENVDB_PYGRID_HAS_BEEN_INCLUDED
