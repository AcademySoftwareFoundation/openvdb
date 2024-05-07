// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file pyGrid.h
/// @author Peter Cucka
/// @brief pybind11 wrapper for openvdb::Grid

#ifndef OPENVDB_PYGRID_HAS_BEEN_INCLUDED
#define OPENVDB_PYGRID_HAS_BEEN_INCLUDED

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#ifdef PY_OPENVDB_USE_NUMPY
#include <pybind11/numpy.h>
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

namespace py = pybind11;

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
sharesWith(const GridType& grid, py::object other)
{
    if (py::isinstance<GridType>(other)) {
        typename GridType::ConstPtr otherGrid = py::cast<typename GridType::Ptr>(other);
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
            throw py::value_error("null grid");
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
            throw py::value_error("null grid");
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
inline Index
treeDepth(const GridType& grid)
{
    return grid.tree().treeDepth();
}


template<typename GridType>
inline Index32
leafCount(const GridType& grid)
{
    return grid.tree().leafCount();
}


template<typename GridType>
inline Index32
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
copyFromArray(GridType&, const py::object&, py::object, py::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw py::error_already_set();
}

template<typename GridType>
inline void
copyToArray(GridType&, const py::object&, py::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw py::error_already_set();
}

#else // if defined(PY_OPENVDB_USE_NUMPY)

using ArrayDimVec = std::vector<ssize_t>;

// ID numbers for supported value types
enum class DtId { NONE, FLOAT, DOUBLE, BOOL, INT16, INT32, INT64, UINT32, UINT64/*, HALF*/ };

template<DtId TypeId> struct NumPyToCpp {};
template<> struct NumPyToCpp<DtId::FLOAT>  { using type = float; };
template<> struct NumPyToCpp<DtId::DOUBLE> { using type = double; };
template<> struct NumPyToCpp<DtId::BOOL>   { using type = bool; };
template<> struct NumPyToCpp<DtId::INT16>  { using type = Int16; };
template<> struct NumPyToCpp<DtId::INT32>  { using type = Int32; };
template<> struct NumPyToCpp<DtId::INT64>  { using type = Int64; };
template<> struct NumPyToCpp<DtId::UINT32> { using type = Index32; };
template<> struct NumPyToCpp<DtId::UINT64> { using type = Index64; };
//template<> struct NumPyToCpp<DtId::HALF>   { using type = math::half; };


#if 0
template<typename T> struct CppToNumPy { static const DtId typeId = DtId::NONE; };
template<> struct CppToNumPy<float>    { static const DtId typeId = DtId::FLOAT; };
template<> struct CppToNumPy<double>   { static const DtId typeId = DtId::DOUBLE; };
template<> struct CppToNumPy<bool>     { static const DtId typeId = DtId::BOOL; };
template<> struct CppToNumPy<Int16>    { static const DtId typeId = DtId::INT16; };
template<> struct CppToNumPy<Int32>    { static const DtId typeId = DtId::INT32; };
template<> struct CppToNumPy<Int64>    { static const DtId typeId = DtId::INT64; };
template<> struct CppToNumPy<Index32>  { static const DtId typeId = DtId::UINT32; };
template<> struct CppToNumPy<Index64>  { static const DtId typeId = DtId::UINT64; };
//template<> struct CppToNumPy<math::half>     { static const DtId typeId = DtId::HALF; };
#endif


// Return the ID number of the given NumPy array's data type.
/// @todo Revisit this if and when py::numpy::dtype ever provides a type number accessor.
inline DtId
arrayTypeId(const py::array& array)
{
    const auto dtype = array.dtype();
    if (dtype.is(py::dtype::of<float>())) return DtId::FLOAT;
    if (dtype.is(py::dtype::of<double>())) return DtId::DOUBLE;
    if (dtype.is(py::dtype::of<bool>())) return DtId::BOOL;
    if (dtype.is(py::dtype::of<Int16>())) return DtId::INT16;
    if (dtype.is(py::dtype::of<Int32>())) return DtId::INT32;
    if (dtype.is(py::dtype::of<Int64>())) return DtId::INT64;
    if (dtype.is(py::dtype::of<Index32>())) return DtId::UINT32;
    if (dtype.is(py::dtype::of<Index64>())) return DtId::UINT64;
    //if (dtype.is(py::dtype::of<math::half>())) return DtId::HALF;
    throw openvdb::TypeError{};
}


// Return a string description of the given NumPy array's data type.
inline std::string
arrayTypeName(const py::array& array)
{
    return py::str(array.dtype());
}


// Return the dimensions of the given NumPy array.
inline ArrayDimVec
arrayDimensions(const py::array& array)
{
    ArrayDimVec dims;
    for (size_t i = 0, N = array.ndim(); i < N; ++i) {
        dims.push_back(array.shape(i));
    }
    return dims;
}



// Abstract base class for helper classes that copy data between
// NumPy arrays of various types and grids of various types
template<typename GridType>
class CopyOpBase
{
public:
    using ValueT = typename GridType::ValueType;

    CopyOpBase(bool toGrid, GridType& grid, py::array array,
        const Coord& origin, const typename GridType::ValueType& tolerance)
        : mToGrid(toGrid)
        , mGrid(&grid)
    {
        mArray = array.mutable_data();
        mArrayTypeName = arrayTypeName(array);
        mArrayTypeId = arrayTypeId(array);
        mArrayDims = arrayDimensions(array);

        mTolerance = tolerance;

        // Compute the bounding box of the region of the grid that is to be copied from or to.
        // origin specifies the coordinates (i, j, k) of the voxel at which to start populating data.
        // Voxel (i, j, k) will correspond to array element (0, 0, 0).
        Coord bboxMax = origin;
        for (size_t n = 0, N = std::min<size_t>(mArrayDims.size(), 3); n < N; ++n) {
            bboxMax[n] += int(mArrayDims[n]) - 1;
        }
        mBBox.reset(origin, bboxMax);
    }
    virtual ~CopyOpBase() {}

    void operator()() const
    {
        try {
            if (mToGrid) {
                copyFromArray(); // copy data from the array to the grid
            } else {
                copyToArray(); // copy data from the grid to the array
            }
        } catch (openvdb::TypeError&) {
            std::ostringstream os;
            os << "unsupported NumPy data type ";
            os << mArrayTypeName.c_str();
            throw py::type_error(os.str());
        }
    }

protected:
    virtual void validate() const = 0;
    virtual void copyFromArray() const = 0;
    virtual void copyToArray() const = 0;

    template<typename ArrayValueType>
    void fromArray() const
    {
        validate();
        tools::Dense<ArrayValueType> valArray(mBBox, static_cast<ArrayValueType*>(mArray));
        tools::copyFromDense(valArray, *mGrid, mTolerance);
    }

    template<typename ArrayValueType>
    void toArray() const
    {
        validate();
        tools::Dense<ArrayValueType> valArray(mBBox, static_cast<ArrayValueType*>(mArray));
        tools::copyToDense(*mGrid, valArray);
    }


    bool mToGrid; // if true, copy from the array to the grid, else vice-versa
    void* mArray;
    GridType* mGrid;
    DtId mArrayTypeId;
    ArrayDimVec mArrayDims;
    std::string mArrayTypeName;
    CoordBBox mBBox;
    ValueT mTolerance;
}; // class CopyOpBase


// Helper subclass that can be specialized for various grid and NumPy array types
template<typename GridType, int VecSize> class CopyOp: public CopyOpBase<GridType> {};

// Specialization for scalar grids
template<typename GridType>
class CopyOp<GridType, /*VecSize=*/1>: public CopyOpBase<GridType>
{
public:
    CopyOp(bool toGrid, GridType& grid,  py::array array, const Coord& coord,
        const typename GridType::ValueType& tolerance = zeroVal<typename GridType::ValueType>()):
        CopyOpBase<GridType>(toGrid, grid, array, coord, tolerance)
    {
    }

protected:
    void validate() const override
    {
        if (this->mArrayDims.size() != 3) {
            std::ostringstream os;
            os << "expected 3-dimensional array, found "
                << this->mArrayDims.size() << "-dimensional array";
            throw py::value_error(os.str());
        }
    }

#ifdef __clang__
    // Suppress "enum value not explicitly handled" warnings
    PRAGMA(clang diagnostic push)
    PRAGMA(clang diagnostic ignored "-Wswitch-enum")
#endif

    void copyFromArray() const override
    {
        switch (this->mArrayTypeId) {
        case DtId::FLOAT: this->template fromArray<typename NumPyToCpp<DtId::FLOAT>::type>(); break;
        case DtId::DOUBLE:this->template fromArray<typename NumPyToCpp<DtId::DOUBLE>::type>();break;
        case DtId::BOOL:  this->template fromArray<typename NumPyToCpp<DtId::BOOL>::type>(); break;
        case DtId::INT16: this->template fromArray<typename NumPyToCpp<DtId::INT16>::type>(); break;
        case DtId::INT32: this->template fromArray<typename NumPyToCpp<DtId::INT32>::type>(); break;
        case DtId::INT64: this->template fromArray<typename NumPyToCpp<DtId::INT64>::type>(); break;
        case DtId::UINT32:this->template fromArray<typename NumPyToCpp<DtId::UINT32>::type>();break;
        case DtId::UINT64:this->template fromArray<typename NumPyToCpp<DtId::UINT64>::type>();break;
        default: throw openvdb::TypeError(); break;
        }
    }

    void copyToArray() const override
    {
        switch (this->mArrayTypeId) {
        case DtId::FLOAT:  this->template toArray<typename NumPyToCpp<DtId::FLOAT>::type>(); break;
        case DtId::DOUBLE: this->template toArray<typename NumPyToCpp<DtId::DOUBLE>::type>(); break;
        case DtId::BOOL:   this->template toArray<typename NumPyToCpp<DtId::BOOL>::type>(); break;
        case DtId::INT16:  this->template toArray<typename NumPyToCpp<DtId::INT16>::type>(); break;
        case DtId::INT32:  this->template toArray<typename NumPyToCpp<DtId::INT32>::type>(); break;
        case DtId::INT64:  this->template toArray<typename NumPyToCpp<DtId::INT64>::type>(); break;
        case DtId::UINT32: this->template toArray<typename NumPyToCpp<DtId::UINT32>::type>(); break;
        case DtId::UINT64: this->template toArray<typename NumPyToCpp<DtId::UINT64>::type>(); break;
        default: throw openvdb::TypeError(); break;
        }
    }

#ifdef __clang__
    PRAGMA(clang diagnostic pop)
#endif

}; // class CopyOp

// Specialization for Vec3 grids
template<typename GridType>
class CopyOp<GridType, /*VecSize=*/3>: public CopyOpBase<GridType>
{
public:
    CopyOp(bool toGrid, GridType& grid, py::array array, const Coord& coord,
        const typename GridType::ValueType& tolerance = zeroVal<typename GridType::ValueType>()):
        CopyOpBase<GridType>(toGrid, grid, array, coord, tolerance)
    {
    }

protected:
    void validate() const override
    {
        if (this->mArrayDims.size() != 4) {
            std::ostringstream os;
            os << "expected 4-dimensional array, found "
                << this->mArrayDims.size() << "-dimensional array";
            throw py::value_error(os.str());
        }
        if (this->mArrayDims[3] != 3) {
            std::ostringstream os;
            os << "expected " << this->mArrayDims[0] << "x" << this->mArrayDims[1]
                << "x" << this->mArrayDims[2] << "x3 array, found " << this->mArrayDims[0]
                << "x" << this->mArrayDims[1] << "x" << this->mArrayDims[2]
                << "x" << this->mArrayDims[3] << " array";
            throw py::value_error(os.str());
        }
    }

#ifdef __clang__
    // Suppress "enum value not explicitly handled" warnings
    PRAGMA(clang diagnostic push)
    PRAGMA(clang diagnostic ignored "-Wswitch-enum")
#endif

    void copyFromArray() const override
    {
        switch (this->mArrayTypeId) {
        case DtId::FLOAT:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::FLOAT>::type>>(); break;
        case DtId::DOUBLE:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::DOUBLE>::type>>(); break;
        case DtId::BOOL:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::BOOL>::type>>(); break;
        case DtId::INT16:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::INT16>::type>>(); break;
        case DtId::INT32:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::INT32>::type>>(); break;
        case DtId::INT64:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::INT64>::type>>(); break;
        case DtId::UINT32:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::UINT32>::type>>(); break;
        case DtId::UINT64:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::UINT64>::type>>(); break;
        default: throw openvdb::TypeError(); break;
        }
    }

    void copyToArray() const override
    {
        switch (this->mArrayTypeId) {
        case DtId::FLOAT:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::FLOAT>::type>>(); break;
        case DtId::DOUBLE:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::DOUBLE>::type>>(); break;
        case DtId::BOOL:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::BOOL>::type>>(); break;
        case DtId::INT16:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::INT16>::type>>(); break;
        case DtId::INT32:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::INT32>::type>>(); break;
        case DtId::INT64:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::INT64>::type>>(); break;
        case DtId::UINT32:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::UINT32>::type>>(); break;
        case DtId::UINT64:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::UINT64>::type>>(); break;
        default: throw openvdb::TypeError(); break;
        }
    }

#ifdef __clang__
    PRAGMA(clang diagnostic pop)
#endif

}; // class CopyOp


template<typename GridType>
inline void
copyFromArray(GridType& grid, py::array array, const Coord& coord, const typename GridType::ValueType& tolerance)
{
    using ValueT = typename GridType::ValueType;
    CopyOp<GridType, VecTraits<ValueT>::Size>
        op(/*toGrid=*/true, grid, array, coord, tolerance);
    op();
}


template<typename GridType>
inline void
copyToArray(GridType& grid, py::array array, const Coord& coord)
{
    using ValueT = typename GridType::ValueType;
    CopyOp<GridType, VecTraits<ValueT>::Size>
        op(/*toGrid=*/false, grid, array, coord);
    op();
}


template<>
inline void
copyFromArray(points::PointDataGrid& /*grid*/, py::array /*array*/,
    const Coord& /*coord*/, const typename points::PointDataGrid::ValueType& /*tolerance*/)
{
    PyErr_SetString(PyExc_NotImplementedError,
        "copying NumPy arrays for PointDataGrids is not supported");
    throw py::error_already_set();
}


template<>
inline void
copyToArray(points::PointDataGrid& /*grid*/, py::array /*array*/, const Coord& /*coord*/)
{
    PyErr_SetString(PyExc_NotImplementedError,
        "copying NumPy arrays for PointDataGrids is not supported");
    throw py::error_already_set();
}


#endif // defined(PY_OPENVDB_USE_NUMPY)


////////////////////////////////////////


#ifndef PY_OPENVDB_USE_NUMPY

template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(py::object, py::object, py::object, py::object, py::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw py::error_already_set();
    return typename GridType::Ptr();
}

template<typename GridType>
inline py::object
volumeToQuadMesh(const GridType&, py::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw py::error_already_set();
    return py::object();
}

template<typename GridType>
inline py::object
volumeToMesh(const GridType&, py::object, py::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    throw py::error_already_set();
    return py::object();
}

#else // if defined(PY_OPENVDB_USE_NUMPY)

// Helper class for meshToLevelSet()
template<typename SrcT, typename DstT>
struct CopyVecOp {
    void operator()(const void* srcPtr, DstT* dst, size_t count) {
        const SrcT* src = static_cast<const SrcT*>(srcPtr);
        for (size_t i = count; i > 0; --i, ++src, ++dst) {
            *dst = static_cast<DstT>(*src);
        }
    }
};
// Partial specialization for source and destination arrays of the same type
template<typename T>
struct CopyVecOp<T, T> {
    void operator()(const void* srcPtr, T* dst, size_t count) {
        const T* src = static_cast<const T*>(srcPtr);
        ::memcpy(dst, src, count * sizeof(T));
    }
};


// Helper function for use with meshToLevelSet() to copy vectors of various types
// and sizes from NumPy arrays to STL vectors
template<typename VecT>
inline void
copyVecArray(py::array& arrayObj, std::vector<VecT>& vec)
{
    using ValueT = typename VecT::ValueType;

    // Get the input array dimensions.
    const auto dims = arrayDimensions(arrayObj);
    const size_t M = dims.empty() ? 0 : dims[0];
    const size_t N = VecT().numElements();
    if (M == 0 || N == 0) return;

    // Preallocate the output vector.
    vec.resize(M);

    // Copy values from the input array to the output vector (with type conversion, if necessary).
    const void* src = arrayObj.data();
    ValueT* dst = &vec[0][0];
    switch (arrayTypeId(arrayObj)) {
    case DtId::FLOAT:  CopyVecOp<NumPyToCpp<DtId::FLOAT>::type, ValueT>()(src, dst, M*N); break;
    case DtId::DOUBLE: CopyVecOp<NumPyToCpp<DtId::DOUBLE>::type, ValueT>()(src, dst, M*N); break;
    case DtId::INT16:  CopyVecOp<NumPyToCpp<DtId::INT16>::type, ValueT>()(src, dst, M*N); break;
    case DtId::INT32:  CopyVecOp<NumPyToCpp<DtId::INT32>::type, ValueT>()(src, dst, M*N); break;
    case DtId::INT64:  CopyVecOp<NumPyToCpp<DtId::INT64>::type, ValueT>()(src, dst, M*N); break;
    case DtId::UINT32: CopyVecOp<NumPyToCpp<DtId::UINT32>::type, ValueT>()(src, dst, M*N); break;
    case DtId::UINT64: CopyVecOp<NumPyToCpp<DtId::UINT64>::type, ValueT>()(src, dst, M*N); break;
    default: break;
    }
}


/// @brief Given NumPy arrays of points, triangle indices, and quad indices,
/// call tools::meshToLevelSet() to generate a level set grid.
template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(py::array_t<float> pointsObj, py::array_t<Index32> trianglesObj, py::array_t<Index32> quadsObj, math::Transform::Ptr xform, float halfWidth)
{
    auto validate2DArray = [](py::array array, ssize_t N) {
        if (array.ndim() != 2 || array.shape(1) != N) {
            std::ostringstream os;
            os << "Expected a 2-dimensional numpy.ndarray with shape(1) = "<< N;
            os << ", found " << array.ndim() << "-dimensional array with shape = (";
            for (ssize_t i = 0; i < array.ndim(); ++i) {
                os << array.shape(i);
                if (i != array.ndim() - 1)
                    os << ", ";
            }
            os <<").";
            throw py::type_error(os.str());
        }
    };

    // Extract the list of mesh vertices from the arguments to this method.
    std::vector<Vec3s> points;
    // Throw an exception if the array has the wrong type or dimensions.
    validate2DArray(pointsObj, 3);
    // Copy values from the array to the vector.
    copyVecArray(pointsObj, points);

    // Extract the list of triangle indices from the arguments to this method.
    std::vector<Vec3I> triangles;
    validate2DArray(trianglesObj, 3);
    copyVecArray(trianglesObj, triangles);

    // Extract the list of quad indices from the arguments to this method.
    std::vector<Vec4I> quads;
    validate2DArray(quadsObj, 4);
    copyVecArray(quadsObj, quads);

    // Generate and return a level set grid.
    if (xform) {
        return tools::meshToLevelSet<GridType>(*xform, points, triangles, quads, halfWidth);
    }
    else {
        math::Transform::Ptr identity = math::Transform::createLinearTransform();
        return tools::meshToLevelSet<GridType>(*identity, points, triangles, quads, halfWidth);
    }
}

template<typename GridType, typename std::enable_if_t<!std::is_scalar<typename GridType::ValueType>::value>* = nullptr>
inline std::tuple<py::array_t<float>, py::array_t<Index32> >
volumeToQuadMesh(const GridType&, double)
{
    OPENVDB_THROW(TypeError, "volume to mesh conversion is supported only for scalar grids");
}

template<typename GridType, typename std::enable_if_t<std::is_scalar<typename GridType::ValueType>::value>* = nullptr>
inline std::tuple<py::array_t<float>, py::array_t<Index32> >
volumeToQuadMesh(const GridType& grid, double isovalue)
{
    // Mesh the input grid and populate lists of mesh vertices and face vertex indices.
    std::vector<Vec3s> points;
    std::vector<Vec4I> quads;
    tools::volumeToMesh(grid, points, quads, isovalue);

    std::vector<ssize_t> shape = { static_cast<ssize_t>(points.size()), 3 };
    std::vector<ssize_t> strides = { 3 * static_cast<ssize_t>(sizeof(float)), static_cast<ssize_t>(sizeof(float))};
    py::array_t<float> pointArrayObj(py::buffer_info(points.data(), sizeof(float), py::format_descriptor<float>::format(), 2, shape, strides));

    shape = { static_cast<ssize_t>(quads.size()), 4 };
    strides = { 4 * static_cast<ssize_t>(sizeof(Index32)), static_cast<ssize_t>(sizeof(Index32))};
    py::array_t<Index32> quadArrayObj(py::buffer_info(quads.data(), sizeof(Index32), py::format_descriptor<Index32>::format(), 2, shape, strides));

    return std::make_tuple(pointArrayObj, quadArrayObj);
}

template<typename GridType, typename std::enable_if_t<!std::is_scalar<typename GridType::ValueType>::value>* = nullptr>
inline std::tuple<py::array_t<float>, py::array_t<Index32>, py::array_t<Index32> >
volumeToMesh(const GridType&, double, double)
{
    OPENVDB_THROW(TypeError, "volume to mesh conversion is supported only for scalar grids");
}

template<typename GridType, typename std::enable_if_t<std::is_scalar<typename GridType::ValueType>::value>* = nullptr>
inline std::tuple<py::array_t<float>, py::array_t<Index32>, py::array_t<Index32> >
volumeToMesh(const GridType& grid, double isovalue, double adaptivity)
{
    // Mesh the input grid and populate lists of mesh vertices and face vertex indices.
    std::vector<Vec3s> points;
    std::vector<Vec3I> triangles;
    std::vector<Vec4I> quads;
    tools::volumeToMesh(grid, points, triangles, quads, isovalue, adaptivity);

    // Create a deep copy of the array (because the point vector will be destroyed
    // when this function returns).

    std::vector<ssize_t> shape = { static_cast<ssize_t>(points.size()), 3 };
    std::vector<ssize_t> strides = { 3 * static_cast<ssize_t>(sizeof(float)), static_cast<ssize_t>(sizeof(float))};
    py::buffer_info pointInfo(points.data(), sizeof(float), py::format_descriptor<float>::format(), 2, shape, strides);
    py::array_t<float> pointArray(pointInfo);

    shape = { static_cast<ssize_t>(triangles.size()), 3 };
    strides = { 3 * static_cast<ssize_t>(sizeof(Index32)), static_cast<ssize_t>(sizeof(Index32))};
    py::buffer_info triangleInfo(triangles.data(), sizeof(Index32), py::format_descriptor<Index32>::format(), 2, shape, strides);
    py::array_t<Index32> triangleArray(triangleInfo);

    shape = { static_cast<ssize_t>(quads.size()), 4 };
    strides = { 4 * static_cast<ssize_t>(sizeof(Index32)), static_cast<ssize_t>(sizeof(Index32))};
    py::buffer_info quadInfo(quads.data(), sizeof(Index32), py::format_descriptor<Index32>::format(), 2, shape, strides);
    py::array_t<Index32> quadArray(quadInfo);

    return std::make_tuple(pointArray, triangleArray, quadArray);
}

#endif // defined(PY_OPENVDB_USE_NUMPY)


////////////////////////////////////////


template<typename GridType, typename IterType>
inline void
applyMap(const char* methodName, GridType& grid, py::object funcObj)
{
    using ValueT = typename GridType::ValueType;

    for (IterType it = grid.tree().template begin<IterType>(); it; ++it) {
        // Evaluate the functor.
        py::object result = funcObj(*it);

        // Verify that the result is of type GridType::ValueType.
        try {
            py::cast<ValueT>(result);
        } catch (py::cast_error&) {
            std::ostringstream os;
            os << "expected callable argument to ";
            os << pyutil::GridTraits<GridType>::name();
            os << "." << methodName << "() to return ";
            os << openvdb::typeNameAsString<ValueT>();
            os << ", found " << pyutil::className(result);
            throw py::type_error(os.str());
        }

        it.setValue(py::cast<ValueT>(result));
    }
}


template<typename GridType>
inline void
mapOn(GridType& grid, py::object funcObj)
{
    applyMap<GridType, typename GridType::ValueOnIter>("mapOn", grid, funcObj);
}


template<typename GridType>
inline void
mapOff(GridType& grid, py::object funcObj)
{
    applyMap<GridType, typename GridType::ValueOffIter>("mapOff", grid, funcObj);
}


template<typename GridType>
inline void
mapAll(GridType& grid, py::object funcObj)
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
        throw py::attribute_error("can't set attribute 'value'");
    }
    static void setActive(const IterT&, bool /*on*/)
    {
        throw py::attribute_error("can't set attribute 'active'");
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
    py::object getItem(py::object keyObj) const
    {
        if (py::isinstance<std::string>(keyObj)) {
            const std::string key = py::cast<std::string>(keyObj);
            if (key == "value") return py::cast(this->getValue());
            else if (key == "active") return py::cast(this->getActive());
            else if (key == "depth") return py::cast(this->getDepth());
            else if (key == "min") return py::cast(this->getBBoxMin());
            else if (key == "max") return py::cast(this->getBBoxMax());
            else if (key == "count") return py::cast(this->getVoxelCount());
        }
        throw py::key_error(py::cast<std::string>(keyObj.attr("__repr__")()));
        return py::object();
    }

    /// @brief Set the value for the given key.
    /// @throw KeyError if the key is invalid
    /// @throw AttributeError if the key refers to a read-only item
    void setItem(py::object keyObj, py::object valObj)
    {
        if (py::isinstance<std::string>(keyObj)) {
            const std::string key = py::cast<std::string>(keyObj);
            if (key == "value") {
                this->setValue(py::cast<ValueT>(valObj)); return;
            } else if (key == "active") {
                this->setActive(py::cast<bool>(valObj)); return;
            } else if (this->hasKey(key)) {
                std::ostringstream os;
                os << "can't set attribute '";
                os << py::cast<std::string>(keyObj.attr("__repr__")());
                os << "'";
                throw py::attribute_error(os.str());
            }
        }
        throw py::key_error(py::cast<std::string>(keyObj.attr("__repr__")()));
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
        py::list valuesAsStrings;
        for (int i = 0; this->keys()[i] != nullptr; ++i) {
            py::str
                key(this->keys()[i]),
                val(this->getItem(key).attr("__repr__")());
            valuesAsStrings.append(py::str("'%s': %s").format(py::make_tuple(key, val)));
        }
        // print ", ".join(valuesAsStrings)
        py::object joined = py::str(", ").attr("join")(valuesAsStrings);
        std::string s = py::cast<std::string>(joined);
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
            throw py::stop_iteration("no more values");
        }
        IterValueProxyT result(mGrid, mIter);
        ++mIter;
        return result;
    }

    static py::object returnSelf(const py::object& obj) { return obj; }

    /// @brief Define a Python wrapper class for this C++ class and another for
    /// the IterValueProxy class returned by iterators of this type.
    static void wrap(py::module_ m)
    {
        const std::string
            gridClassName = pyutil::GridTraits<typename std::remove_const<GridT>::type>::name(),
            iterClassName = gridClassName + Traits::name(),
            valueClassName = gridClassName + Traits::name() + "Value";

        py::class_<IterWrap>(m,
            iterClassName.c_str(),
            /*docstring=*/Traits::descr().c_str())

            .def_property_readonly("parent", &IterWrap::parent,
                ("the " + gridClassName + " over which to iterate").c_str())

            .def("next", &IterWrap::next, ("next() -> " + valueClassName).c_str())
            .def("__next__", &IterWrap::next, ("__next__() -> " + valueClassName).c_str())
            .def("__iter__", &returnSelf);

        py::class_<IterValueProxyT>(m,
            valueClassName.c_str(),
            /*docstring=*/("Proxy for a tile or voxel value in a " + gridClassName).c_str())

            .def("copy", &IterValueProxyT::copy,
                ("copy() -> " + valueClassName + "\n\n"
                "Return a shallow copy of this value, i.e., one that shares\n"
                "its data with the original.").c_str())

            .def_property_readonly("parent", &IterValueProxyT::parent,
                ("the " + gridClassName + " to which this value belongs").c_str())

            .def("__str__", &IterValueProxyT::info)
            .def("__repr__", &IterValueProxyT::info)

            .def("__eq__", &IterValueProxyT::operator==)
            .def("__ne__", &IterValueProxyT::operator!=)

            .def_property("value", &IterValueProxyT::getValue, &IterValueProxyT::setValue,
                "value of this tile or voxel")
            .def_property("active", &IterValueProxyT::getActive, &IterValueProxyT::setActive,
                "active state of this tile or voxel")
            .def_property_readonly("depth", &IterValueProxyT::getDepth,
                "tree depth at which this value is stored")
            .def_property_readonly("min", &IterValueProxyT::getBBoxMin,
                "lower bound of the axis-aligned bounding box of this tile or voxel")
            .def_property_readonly("max", &IterValueProxyT::getBBoxMax,
                "upper bound of the axis-aligned bounding box of this tile or voxel")
            .def_property_readonly("count", &IterValueProxyT::getVoxelCount,
                "number of voxels spanned by this value")

            .def_static("keys", &IterValueProxyT::getKeys,
                "keys() -> list\n\n"
                "Return a list of keys for this tile or voxel.")
            .def_static("__contains__", &IterValueProxyT::hasKey,
                "__contains__(key) -> bool\n\n"
                "Return True if the given key exists.")
            .def("__getitem__", &IterValueProxyT::getItem,
                "__getitem__(key) -> value\n\n"
                "Return the value of the item with the given key.")
            .def("__setitem__", &IterValueProxyT::setItem,
                "__setitem__(key, value)\n\n"
                "Set the value of the item with the given key.");
    }

private:
    // To keep this iterator's grid from being deleted, leaving the iterator dangling,
    // store a shared pointer to the grid.
    const typename GridT::ConstPtr mGrid;
    IterT mIter;
}; // class IterWrap


////////////////////////////////////////


template<typename GridT>
struct PickleSuite
{
    using GridPtrT = typename GridT::Ptr;

    /// Return a tuple representing the state of the given Grid.
    static py::tuple getState(const GridPtrT& grid)
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
        py::bytes bytesObj(s);
        return py::make_tuple(bytesObj);
    }

    /// Restore the given Grid to a saved state.
    static GridPtrT setState(py::tuple state)
    {
        bool badState = (py::len(state) != 1);

        std::string serialized;
        if (!badState) {
            // Extract the sequence containing the serialized Grid.
            if (py::isinstance<py::bytes>(state[0]))
                serialized = py::cast<py::bytes>(state[0]);
            else
                badState = true;
        }

        if (badState) {
            std::ostringstream os;
            os << "expected (dict, bytes) tuple in call to __setstate__; found ";
            os << py::cast<std::string>(state.attr("__repr__")());
            throw py::value_error(os.str());
        }

        // Restore the internal state of the C++ object.
        GridPtrVecPtr grids;
        {
            std::istringstream istr(serialized, std::ios_base::binary);
            io::Stream strm(istr);
            grids = strm.getGrids(); // (note: file-level metadata is ignored)
        }
        if (grids && !grids->empty()) {
            if (GridPtrT savedGrid = gridPtrCast<GridT>((*grids)[0])) {
                return savedGrid;
            }
        }

        return GridPtrT();
    }
}; // struct PickleSuite


////////////////////////////////////////


/// Create a Python wrapper for a particular template instantiation of Grid.
template<typename GridType>
inline void
exportGrid(py::module_ m)
{
    using ValueT = typename GridType::ValueType;
    using GridPtr = typename GridType::Ptr;
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
    py::class_<GridType, GridPtr, GridBase>(m,
        /*classname=*/pyGridTypeName.c_str(),
        /*docstring=*/(Traits::descr()).c_str())
        .def(py::init<>(), docstring.c_str())
        .def(py::init<const ValueT&>(), py::arg("background"),
            "Initialize with the given background value.")

        .def("copy", &pyGrid::copyGrid<GridType>,
            ("copy() -> " + pyGridTypeName + "\n\n"
            "Return a shallow copy of this grid, i.e., a grid\n"
            "that shares its voxel data with this grid.").c_str())
        .def("deepCopy", &GridType::deepCopy,
            ("deepCopy() -> " + pyGridTypeName + "\n\n"
            "Return a deep copy of this grid.\n").c_str())

        .def(py::pickle(&PickleSuite<GridType>::getState, &PickleSuite<GridType>::setState))

        .def("sharesWith", &pyGrid::sharesWith<GridType>,
            ("sharesWith(" + pyGridTypeName + ") -> bool\n\n"
            "Return True if this grid shares its voxel data with the given grid.").c_str())

        /// @todo Any way to set a docstring for a class property?
        .def_property_readonly_static("valueTypeName", [](const py::object&) { return pyGrid::getValueType<GridType>(); })
            /// @todo docstring = "name of this grid's value type"
        .def_property_readonly_static("zeroValue", [](const py::object&) { return pyGrid::getZeroValue<GridType>(); })
            /// @todo docstring = "zero, as expressed in this grid's value type"
        .def_property_readonly_static("oneValue", [](const py::object&) { return pyGrid::getOneValue<GridType>(); })
            /// @todo docstring = "one, as expressed in this grid's value type"
        /// @todo Is Grid.typeName ever needed?
        //.def_property_static("typeName", &GridType::gridType)
            /// @todo docstring = to "name of this grid's type"

        .def_property("background",
            &pyGrid::getGridBackground<GridType>, &pyGrid::setGridBackground<GridType>,
            "value of this grid's background voxels")

        .def("getAccessor", &pyGrid::getAccessor<GridType>,
            ("getAccessor() -> " + pyGridTypeName + "Accessor\n\n"
            "Return an accessor that provides random read and write access\n"
            "to this grid's voxels.").c_str())
        .def("getConstAccessor", &pyGrid::getConstAccessor<GridType>,
            ("getConstAccessor() -> " + pyGridTypeName + "Accessor\n\n"
            "Return an accessor that provides random read-only access\n"
            "to this grid's voxels.").c_str())

        //
        // Statistics
        //
        .def("evalLeafBoundingBox", &pyGrid::evalLeafBoundingBox<GridType>,
            "evalLeafBoundingBox() -> xyzMin, xyzMax\n\n"
            "Return the coordinates of opposite corners of the axis-aligned\n"
            "bounding box of all leaf nodes.")
        .def("evalLeafDim", &pyGrid::evalLeafDim<GridType>,
            "evalLeafDim() -> x, y, z\n\n"
            "Return the dimensions of the axis-aligned bounding box\n"
            "of all leaf nodes.")

        .def_property_readonly("treeDepth", &pyGrid::treeDepth<GridType>,
            "depth of this grid's tree from root node to leaf node")
        .def("nodeLog2Dims", &pyGrid::getNodeLog2Dims<GridType>,
            "list of Log2Dims of the nodes of this grid's tree\n"
            "in order from root to leaf")

        .def("leafCount", &pyGrid::leafCount<GridType>,
            "leafCount() -> int\n\n"
            "Return the number of leaf nodes in this grid's tree.")
        .def("nonLeafCount", &pyGrid::nonLeafCount<GridType>,
            "nonLeafCount() -> int\n\n"
            "Return the number of non-leaf nodes in this grid's tree.")

        .def("activeLeafVoxelCount", &pyGrid::activeLeafVoxelCount<GridType>,
            "activeLeafVoxelCount() -> int\n\n"
            "Return the number of active voxels that are stored\n"
            "in the leaf nodes of this grid's tree.")

        .def("evalMinMax", &pyGrid::evalMinMax<GridType>,
            "evalMinMax() -> min, max\n\n"
            "Return the minimum and maximum active values in this grid.")

        .def("getIndexRange", &pyGrid::getIndexRange<GridType>,
            "getIndexRange() -> min, max\n\n"
            "Return the minimum and maximum coordinates that are represented\n"
            "in this grid.  These might include background voxels.")
        //.def("expand", &pyGrid::expandIndexRange<GridType>,
        //    py::arg("xyz"),
        //    "expand(xyz)\n\n"
        //    "Expand this grid's index range to include the given coordinates.")

        //
        // Tools
        //
        .def("fill", &pyGrid::fill<GridType>,
            py::arg("min"), py::arg("max"), py::arg("value"), py::arg("active")=true,
            "fill(min, max, value, active=True)\n\n"
            "Set all voxels within a given axis-aligned box to\n"
            "a constant value (either active or inactive).")
        .def("signedFloodFill", &pyGrid::signedFloodFill<GridType>,
            "signedFloodFill()\n\n"
            "Propagate the sign from a narrow-band level set into inactive\n"
            "voxels and tiles.")

        .def("copyFromArray", &pyGrid::copyFromArray<GridType>,
            py::arg("array"), py::arg("ijk")=Coord(0,0,0),
                 py::arg("tolerance")=pyGrid::getZeroValue<GridType>(),
            ("copyFromArray(array, ijk=(0, 0, 0), tolerance=0)\n\n"
            "Populate this grid, starting at voxel (i, j, k), with values\nfrom a "
            + std::string(openvdb::VecTraits<ValueT>::IsVec ? "four" : "three")
            + "-dimensional array.  Mark voxels as inactive\n"
            "if and only if their values are equal to this grid's\n"
            "background value within the given tolerance.").c_str())
        .def("copyToArray", &pyGrid::copyToArray<GridType>,
            py::arg("array"), py::arg("ijk")=Coord(0,0,0),
            ("copyToArray(array, ijk=(0, 0, 0))\n\nPopulate a "
            + std::string(openvdb::VecTraits<ValueT>::IsVec ? "four" : "three")
            + "-dimensional array with values\n"
            "from this grid, starting at voxel (i, j, k).").c_str())

        .def("convertToQuads",
            &pyGrid::volumeToQuadMesh<GridType>,
            py::arg("isovalue")=0,
            "convertToQuads(isovalue=0) -> points, quads\n\n"
            "Uniformly mesh a scalar grid that has a continuous isosurface\n"
            "at the given isovalue.  Return a NumPy array of world-space\n"
            "points and a NumPy array of 4-tuples of point indices, which\n"
            "specify the vertices of the quadrilaterals that form the mesh.")
        .def("convertToPolygons",
            &pyGrid::volumeToMesh<GridType>,
            py::arg("isovalue")=0, py::arg("adaptivity")=0,
            "convertToPolygons(isovalue=0, adaptivity=0) -> points, triangles, quads\n\n"
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
            py::arg("points"),
#ifdef PY_OPENVDB_USE_NUMPY
            py::arg("triangles")=py::array_t<Index32>({ 0, 3 }, { 3 * static_cast<ssize_t>(sizeof(Index32)), static_cast<ssize_t>(sizeof(Index32))} ),
            py::arg("quads")=py::array_t<Index32>({ 0, 4 }, { 4 * static_cast<ssize_t>(sizeof(Index32)), static_cast<ssize_t>(sizeof(Index32))} ),
#else
            py::arg("triangles")=std::vector<Index32>(),
            py::arg("quads")=std::vector<Index32>(),
#endif
            py::arg("transform")=py::none(),
            py::arg("halfWidth")=openvdb::LEVEL_SET_HALF_WIDTH,
            ("createLevelSetFromPolygons(points, triangles=None, quads=None,\n"
             "    transform=None, halfWidth="
             + std::to_string(openvdb::LEVEL_SET_HALF_WIDTH) + ") -> "
             + pyGridTypeName + "\n\n"
            "Convert a triangle and/or quad mesh to a narrow-band level set volume.\n"
            "The mesh must form a closed surface, but the surface need not be\n"
            "manifold and may have self intersections and degenerate faces.\n"
            "The mesh is described by a NumPy array of world-space points\n"
            "and NumPy arrays of 3- and 4-tuples of point indices that specify\n"
            "the vertices of the triangles and quadrilaterals that form the mesh.\n"
            "Either the triangle or the quad array may be empty or None.\n"
            "The resulting volume will have the given transform (or the identity\n"
            "transform if no transform is given) and a narrow band width of\n"
            "2 x halfWidth voxels.").c_str())

        .def("prune", &pyGrid::prune<GridType>,
            py::arg("tolerance") = 0,
            "prune(tolerance=0)\n\n"
            "Remove nodes whose values all have the same active state\n"
            "and are equal to within a given tolerance.")
        .def("pruneInactive", &pyGrid::pruneInactive<GridType>,
            py::arg("value") = py::none(),
            "pruneInactive()\n\n"
            "Remove nodes whose values are all inactive and replace them\n"
            "with background tiles.")

        .def("merge", &GridType::merge,
            ("merge(" + pyGridTypeName + ")\n\n"
            "Move child nodes from the other grid into this grid wherever\n"
            "those nodes correspond to constant-value tiles in this grid,\n"
            "and replace leaf-level inactive voxels in this grid with\n"
            "corresponding voxels in the other grid that are active.\n\n"
            "Note: this operation always empties the other grid.").c_str())

        .def("mapOn", &pyGrid::mapOn<GridType>,
            py::arg("function"),
            "mapOn(function)\n\n"
            "Iterate over all the active (\"on\") values (tile and voxel)\n"
            "of this grid and replace each value with function(value).\n\n"
            "Example: grid.mapOn(lambda x: x * 2 if x < 0.5 else x)")

        .def("mapOff", &pyGrid::mapOff<GridType>,
            py::arg("function"),
            "mapOff(function)\n\n"
            "Iterate over all the inactive (\"off\") values (tile and voxel)\n"
            "of this grid and replace each value with function(value).\n\n"
            "Example: grid.mapOff(lambda x: x * 2 if x < 0.5 else x)")

        .def("mapAll", &pyGrid::mapAll<GridType>,
            py::arg("function"),
            "mapAll(function)\n\n"
            "Iterate over all values (tile and voxel) of this grid\n"
            "and replace each value with function(value).\n\n"
            "Example: grid.mapAll(lambda x: x * 2 if x < 0.5 else x)")

        .def("combine", &pyGrid::combine<GridType>,
            py::arg("grid"), py::arg("function"),
            "combine(grid, function)\n\n"
            "Compute function(self, other) over all corresponding pairs\n"
            "of values (tile or voxel) of this grid and the other grid\n"
            "and store the result in this grid.\n\n"
            "Note: this operation always empties the other grid.\n\n"
            "Example: grid.combine(otherGrid, lambda a, b: min(a, b))")

        //
        // Iterators
        //
        .def("citerOnValues", &pyGrid::IterTraits<GridType, ValueOnCIterT>::begin,
            "citerOnValues() -> iterator\n\n"
            "Return a read-only iterator over this grid's active\ntile and voxel values.")
        .def("citerOffValues", &pyGrid::IterTraits<GridType, ValueOffCIterT>::begin,
            "iterOffValues() -> iterator\n\n"
            "Return a read-only iterator over this grid's inactive\ntile and voxel values.")
        .def("citerAllValues", &pyGrid::IterTraits<GridType, ValueAllCIterT>::begin,
            "iterAllValues() -> iterator\n\n"
            "Return a read-only iterator over all of this grid's\ntile and voxel values.")

        .def("iterOnValues", &pyGrid::IterTraits<GridType, ValueOnIterT>::begin,
            "iterOnValues() -> iterator\n\n"
            "Return a read/write iterator over this grid's active\ntile and voxel values.")
        .def("iterOffValues", &pyGrid::IterTraits<GridType, ValueOffIterT>::begin,
            "iterOffValues() -> iterator\n\n"
            "Return a read/write iterator over this grid's inactive\ntile and voxel values.")
        .def("iterAllValues", &pyGrid::IterTraits<GridType, ValueAllIterT>::begin,
            "iterAllValues() -> iterator\n\n"
            "Return a read/write iterator over all of this grid's\ntile and voxel values.");

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
    py::cast<py::list>(m.attr("GridTypes")).append(m.attr(pyGridTypeName.c_str()));
}

} // namespace pyGrid

#endif // OPENVDB_PYGRID_HAS_BEEN_INCLUDED
