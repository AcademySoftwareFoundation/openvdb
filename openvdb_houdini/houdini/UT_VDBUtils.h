///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 * 	Adrian Saldanha
 *	Side Effects Software Inc.
 *	123 Front Street West, Suite 1401
 *	Toronto, Ontario
 *	Canada   M5J 2M2
 *	416-366-4607
 */

#include <UT/UT_Version.h>
#if !defined(SESI_OPENVDB) && (UT_VERSION_INT >= 0x0c050157) // 12.5.343 or later

#include <UT/UT_VDBUtils.h>

#else // earlier than 12.5.343

#ifndef __HDK_UT_VDBUtils__
#define __HDK_UT_VDBUtils__

enum UT_VDBType
{
    UT_VDB_INVALID,
    UT_VDB_FLOAT,
    UT_VDB_DOUBLE,
    UT_VDB_INT32,
    UT_VDB_INT64,
    UT_VDB_BOOL,
    UT_VDB_VEC3F,
    UT_VDB_VEC3D,
    UT_VDB_VEC3I,
};

#include <openvdb/openvdb.h>

#include <UT/UT_Assert.h>
#include <UT/UT_BoundingBox.h>
#include <UT/UT_Matrix4.h>
#include <UT/UT_Matrix3.h>
#include <UT/UT_Matrix2.h>
#include <SYS/SYS_Math.h>

#include <boost/typeof/typeof.hpp>


#if (UT_VERSION_INT < 0x0c010072) // 12.1.114 or earlier
#define UTverify_cast static_cast
#endif


/// Calls openvdb::initialize()
inline void UTvdbInitialize() { openvdb::initialize(); }

/// Find the UT_VDBType from a grid
inline UT_VDBType
UTvdbGetGridType(const openvdb::GridBase &grid)
{
    if (grid.isType<openvdb::FloatGrid>())
	return UT_VDB_FLOAT;
    if (grid.isType<openvdb::DoubleGrid>())
	return UT_VDB_DOUBLE;
    if (grid.isType<openvdb::Int32Grid>())
	return UT_VDB_INT32;
    if (grid.isType<openvdb::Int64Grid>())
	return UT_VDB_INT64;
    if (grid.isType<openvdb::BoolGrid>())
	return UT_VDB_BOOL;
    if (grid.isType<openvdb::Vec3fGrid>())
	return UT_VDB_VEC3F;
    if (grid.isType<openvdb::Vec3dGrid>())
	return UT_VDB_VEC3D;
    if (grid.isType<openvdb::Vec3IGrid>())
	return UT_VDB_VEC3I;

    return UT_VDB_INVALID;
}

/// Return the string representation of a grid's underlying value type
inline const char *
UTvdbGetGridTypeString(const openvdb::GridBase &grid)
{
    switch(UTvdbGetGridType(grid))
    {
    case UT_VDB_FLOAT:
	return "float";
    case UT_VDB_DOUBLE:
	return "double";
    case UT_VDB_INT32:
	return "int32";
    case UT_VDB_INT64:
	return "int64";
    case UT_VDB_BOOL:
	return "bool";
    case UT_VDB_VEC3F:
	return "Vec3f";
    case UT_VDB_VEC3D:
	return "Vec3d";
    case UT_VDB_VEC3I:
	return "Vec3i";
    default:
	return "invalid type";
    }
}

/// Returns the tuple size of a grid given its value type.
inline int
UTvdbGetGridTupleSize(UT_VDBType type)
{
    switch(type)
    {
    case UT_VDB_FLOAT:
    case UT_VDB_DOUBLE:
    case UT_VDB_INT32:
    case UT_VDB_INT64:
    case UT_VDB_BOOL:
	return 1;

    case UT_VDB_VEC3F:
    case UT_VDB_VEC3D:
    case UT_VDB_VEC3I:
	return 3;

    case UT_VDB_INVALID:
    default:
	break;
    }

    return 0;
}

/// Returns the tuple size of a grid
inline int
UTvdbGetGridTupleSize(const openvdb::GridBase &grid)
{
    return UTvdbGetGridTupleSize(UTvdbGetGridType(grid));
}

/// Special plusEqual class to avoid bool warnings
/// @{
template <typename T>
struct UT_VDBMath
{
    static void plusEqual(T &lhs, const T &rhs)
	{ lhs += rhs; }
};
template <>
struct UT_VDBMath<bool>
{
    static void plusEqual(bool &lhs, const bool &rhs)
	{ lhs = lhs | rhs; }
};
/// @}

/// Helpers for downcasting to a specific grid type
/// @{
template <typename GridType>
inline const GridType *
UTvdbGridCast(const openvdb::GridBase *grid)
    { return UTverify_cast<const GridType *>(grid); }

template <typename GridType>
inline GridType *
UTvdbGridCast(openvdb::GridBase *grid)
    { return UTverify_cast<GridType *>(grid); }

template <typename GridType>
inline const GridType &
UTvdbGridCast(const openvdb::GridBase &grid)
    { return *UTverify_cast<const GridType *>(&grid); }

template <typename GridType>
inline GridType &
UTvdbGridCast(openvdb::GridBase &grid)
    { return *UTverify_cast<GridType *>(&grid); }

template <typename GridType>
inline typename GridType::ConstPtr
UTvdbGridCast(openvdb::GridBase::ConstPtr grid)
    { return openvdb::gridConstPtrCast<GridType>(grid); }

template <typename GridType>
inline typename GridType::Ptr
UTvdbGridCast(openvdb::GridBase::Ptr grid)
    { return openvdb::gridPtrCast<GridType>(grid); }
/// @}

////////////////////////////////////////

namespace UT_VDBUtils {

// Helper function used internally by UTvdbProcessTypedGrid()
// to instantiate a templated functor for a specific grid type
// and then to call the functor with a grid of that type
template<typename GridType, typename OpType, typename GridBaseType>
inline void
callTypedGrid(GridBaseType &grid, OpType& op)
{
#ifdef _MSC_VER
    op.operator()<GridType>(UTvdbGridCast<GridType>(grid));
#else
    op.template operator()<GridType>(UTvdbGridCast<GridType>(grid));
#endif
}

} // namespace UT_VDBUtils

////////////////////////////////////////


/// @brief Utility function that, given a generic grid pointer,
/// calls a functor on the fully-resolved grid
///
/// @par Example:
/// @code
/// using openvdb::Coord;
/// using openvdb::CoordBBox;
///
/// struct FillOp {
///     const CoordBBox bbox;
///
///     FillOp(const CoordBBox& b): bbox(b) {}
///
///     template<typename GridT>
///     void operator()(GridT& grid) const {
///         typedef typename GridT::ValueType ValueT;
///         grid.fill(bbox, ValueT(1));
///     }
/// };
///
/// GU_PrimVDB* vdb = ...;
/// vdb->makeGridUnique();
/// CoordBBox bbox(Coord(0,0,0), Coord(10,10,10));
/// UTvdbProcessTypedGrid(vdb->getStorageType(), vdb->getGrid(), FillOp(bbox));
/// @endcode
///
/// @return @c false if the grid type is unknown or unhandled.
/// @{
#define UT_VDB_DECL_PROCESS_TYPED_GRID(GRID_BASE_T) \
template<typename OpType> \
inline bool \
UTvdbProcessTypedGrid(UT_VDBType grid_type, GRID_BASE_T grid, OpType& op) \
{ \
    using namespace openvdb; \
    using namespace UT_VDBUtils; \
    switch (grid_type) \
    { \
	case UT_VDB_FLOAT:  callTypedGrid<FloatGrid>(grid, op); break; \
	case UT_VDB_DOUBLE: callTypedGrid<DoubleGrid>(grid, op); break; \
	case UT_VDB_INT32:  callTypedGrid<Int32Grid>(grid, op); break; \
	case UT_VDB_INT64:  callTypedGrid<Int64Grid>(grid, op); break; \
	case UT_VDB_VEC3F:  callTypedGrid<Vec3SGrid>(grid, op); break; \
	case UT_VDB_VEC3D:  callTypedGrid<Vec3DGrid>(grid, op); break; \
	case UT_VDB_VEC3I:  callTypedGrid<Vec3IGrid>(grid, op); break; \
	default:	    UT_ASSERT(!"Unknown grid type"); return false; \
    } \
    return true; \
} \
template<typename OpType> \
inline bool \
UTvdbProcessTypedGridTopology(UT_VDBType grid_type, GRID_BASE_T grid, OpType& op) \
{ \
    using namespace openvdb; \
    using namespace UT_VDBUtils; \
    switch (grid_type) \
    { \
	case UT_VDB_FLOAT:  callTypedGrid<FloatGrid>(grid, op); break; \
	case UT_VDB_DOUBLE: callTypedGrid<DoubleGrid>(grid, op); break; \
	case UT_VDB_INT32:  callTypedGrid<Int32Grid>(grid, op); break; \
	case UT_VDB_INT64:  callTypedGrid<Int64Grid>(grid, op); break; \
	case UT_VDB_VEC3F:  callTypedGrid<Vec3SGrid>(grid, op); break; \
	case UT_VDB_VEC3D:  callTypedGrid<Vec3DGrid>(grid, op); break; \
	case UT_VDB_VEC3I:  callTypedGrid<Vec3IGrid>(grid, op); break; \
	case UT_VDB_BOOL:   callTypedGrid<BoolGrid>(grid, op); break; \
	default:	    UT_ASSERT(!"Unknown grid type"); return false; \
    } \
    return true; \
} \
template<typename OpType> \
inline bool \
UTvdbProcessTypedGridVec3(UT_VDBType grid_type, GRID_BASE_T grid, OpType& op) \
{ \
    using namespace openvdb; \
    using namespace UT_VDBUtils; \
    switch (grid_type) \
    { \
	case UT_VDB_VEC3F:  callTypedGrid<Vec3SGrid>(grid, op); break; \
	case UT_VDB_VEC3D:  callTypedGrid<Vec3DGrid>(grid, op); break; \
	case UT_VDB_VEC3I:  callTypedGrid<Vec3IGrid>(grid, op); break; \
	default:	    return false; \
    } \
    return true; \
} \
template<typename OpType> \
inline bool \
UTvdbProcessTypedGridScalar(UT_VDBType grid_type, GRID_BASE_T grid, OpType& op) \
{ \
    using namespace openvdb; \
    using namespace UT_VDBUtils; \
    switch (grid_type) \
    { \
	case UT_VDB_FLOAT:  callTypedGrid<FloatGrid>(grid, op); break; \
	case UT_VDB_DOUBLE: callTypedGrid<DoubleGrid>(grid, op); break; \
	case UT_VDB_INT32:  callTypedGrid<Int32Grid>(grid, op); break; \
	case UT_VDB_INT64:  callTypedGrid<Int64Grid>(grid, op); break; \
	default:	    return false; \
    } \
    return true; \
} \
template<typename OpType> \
inline bool \
UTvdbProcessTypedGridReal(UT_VDBType grid_type, GRID_BASE_T grid, OpType& op) \
{ \
    using namespace openvdb; \
    using namespace UT_VDBUtils; \
    switch (grid_type) \
    { \
	case UT_VDB_FLOAT:  callTypedGrid<FloatGrid>(grid, op); break; \
	case UT_VDB_DOUBLE: callTypedGrid<DoubleGrid>(grid, op); break; \
	default:	    return false; \
    } \
    return true; \
} \
/**/
UT_VDB_DECL_PROCESS_TYPED_GRID(const openvdb::GridBase &)
UT_VDB_DECL_PROCESS_TYPED_GRID(const openvdb::GridBase *)
UT_VDB_DECL_PROCESS_TYPED_GRID(openvdb::GridBase::ConstPtr)
UT_VDB_DECL_PROCESS_TYPED_GRID(openvdb::GridBase &)
UT_VDB_DECL_PROCESS_TYPED_GRID(openvdb::GridBase *)
UT_VDB_DECL_PROCESS_TYPED_GRID(openvdb::GridBase::Ptr)

/// @}


// Helper macro for UTvdbCall* macros, do not outside of this file!
#define UT_VDB_CALL(GRIDT, RETURN, FNAME, GRIDBASE, ...) \
    { \
	RETURN FNAME <GRIDT> (UTvdbGridCast<GRIDT>(GRIDBASE), __VA_ARGS__ ); \
    } \
    /**/

//@{
/// Macro to invoke the correct type of grid.
/// Use like:
/// @code
/// UTvdbCallScalarType(grid_type, myfunction, grid, parms)
/// @endcode
/// to invoke
/// @code
/// template <typename GridType>
/// static void
/// myfunction(const GridType &grid, parms)
/// {  }
/// @endcode

#define UTvdbCallRealType(TYPE, FNAME, GRIDBASE, ...)	\
    if (TYPE == UT_VDB_FLOAT)	\
	UT_VDB_CALL(openvdb::FloatGrid,(void),FNAME,GRIDBASE,__VA_ARGS__) \
    else if (TYPE == UT_VDB_DOUBLE)	\
	UT_VDB_CALL(openvdb::DoubleGrid,(void),FNAME,GRIDBASE,__VA_ARGS__) \
    /**/
#define UTvdbCallScalarType(TYPE, FNAME, GRIDBASE, ...)	\
    UTvdbCallRealType(TYPE, FNAME, GRIDBASE, __VA_ARGS__)		\
    else if (TYPE == UT_VDB_INT32)	\
	UT_VDB_CALL(openvdb::Int32Grid,(void),FNAME,GRIDBASE,__VA_ARGS__) \
    else if (TYPE == UT_VDB_INT64)	\
	UT_VDB_CALL(openvdb::Int64Grid,(void),FNAME,GRIDBASE,__VA_ARGS__) \
    /**/
#define UTvdbCallVec3Type(TYPE, FNAME, GRIDBASE, ...)	\
    if (TYPE == UT_VDB_VEC3F)	\
	UT_VDB_CALL(openvdb::Vec3fGrid,(void),FNAME,GRIDBASE,__VA_ARGS__) \
    else if (TYPE == UT_VDB_VEC3D)	\
	UT_VDB_CALL(openvdb::Vec3dGrid,(void),FNAME,GRIDBASE,__VA_ARGS__) \
    else if (TYPE == UT_VDB_VEC3I)	\
	UT_VDB_CALL(openvdb::Vec3IGrid,(void),FNAME,GRIDBASE,__VA_ARGS__) \
    /**/
#define UTvdbCallAllType(TYPE, FNAME, GRIDBASE, ...)	\
    UTvdbCallScalarType(TYPE, FNAME, GRIDBASE, __VA_ARGS__)		\
    else UTvdbCallVec3Type(TYPE, FNAME, GRIDBASE, __VA_ARGS__) \
    else UT_ASSERT(!"Unknown grid type"); \
    /**/
#define UTvdbCallAllTopology(TYPE, FNAME, GRIDBASE, ...)	\
    UTvdbCallScalarType(TYPE, FNAME, GRIDBASE, __VA_ARGS__)		\
    else UTvdbCallVec3Type(TYPE, FNAME, GRIDBASE, __VA_ARGS__) \
    else if (TYPE == UT_VDB_BOOL) \
	UT_VDB_CALL(openvdb::BoolGrid,(void),FNAME,GRIDBASE,__VA_ARGS__) \
    else UT_ASSERT(!"Unknown grid type"); \
    /**/
//@}

//@{
/// Macro to invoke the correct type of grid.
/// Use like:
/// @code
/// UTvdbReturnScalarType(grid_type, myfunction, grid, parms)
/// @endcode
/// to invoke
/// @code
/// return myfunction(grid, parms);
/// @endcode
/// via:
/// @code
/// template <typename GridType>
/// static RESULT
/// myfunction(const GridType &grid, parms)
/// {  }
/// @endcode

#define UTvdbReturnRealType(TYPE, FNAME, GRIDBASE, ...)	\
    if (TYPE == UT_VDB_FLOAT)	\
	UT_VDB_CALL(openvdb::FloatGrid,return,FNAME,GRIDBASE,__VA_ARGS__) \
    else if (TYPE == UT_VDB_DOUBLE)	\
	UT_VDB_CALL(openvdb::DoubleGrid,return,FNAME,GRIDBASE,__VA_ARGS__) \
    /**/
#define UTvdbReturnScalarType(TYPE, FNAME, GRIDBASE, ...)	\
    UTvdbReturnRealType(TYPE, FNAME, GRIDBASE, __VA_ARGS__)		\
    else if (TYPE == UT_VDB_INT32)	\
	UT_VDB_CALL(openvdb::Int32Grid,return,FNAME,GRIDBASE,__VA_ARGS__) \
    else if (TYPE == UT_VDB_INT64)	\
	UT_VDB_CALL(openvdb::Int64Grid,return,FNAME,GRIDBASE,__VA_ARGS__) \
    /**/
#define UTvdbReturnVec3Type(TYPE, FNAME, GRIDBASE, ...)	\
    if (TYPE == UT_VDB_VEC3F)	\
	UT_VDB_CALL(openvdb::Vec3fGrid,return,FNAME,GRIDBASE,__VA_ARGS__) \
    else if (TYPE == UT_VDB_VEC3D)	\
	UT_VDB_CALL(openvdb::Vec3dGrid,return,FNAME,GRIDBASE,__VA_ARGS__) \
    else if (TYPE == UT_VDB_VEC3I)	\
	UT_VDB_CALL(openvdb::Vec3IGrid,return,FNAME,GRIDBASE,__VA_ARGS__) \
    /**/
#define UTvdbReturnAllType(TYPE, FNAME, GRIDBASE, ...)	\
    UTvdbReturnScalarType(TYPE, FNAME, GRIDBASE, __VA_ARGS__) \
    else UTvdbReturnVec3Type(TYPE, FNAME, GRIDBASE, __VA_ARGS__) \
    else UT_ASSERT(!"Unknown grid type"); \
    /**/
#define UTvdbReturnAllTopology(TYPE, FNAME, GRIDBASE, ...)	\
    UTvdbReturnScalarType(TYPE, FNAME, GRIDBASE, __VA_ARGS__) \
    else UTvdbReturnVec3Type(TYPE, FNAME, GRIDBASE, __VA_ARGS__) \
    else if (TYPE == UT_VDB_BOOL) \
	UT_VDB_CALL(openvdb::BoolGrid,return,FNAME,GRIDBASE,__VA_ARGS__) \
    else UT_ASSERT(!"Unknown grid type"); \
    /**/
//@}


////////////////////////////////////////


/// Matrix conversion from openvdb to UT
// @{
template <typename S>
UT_Matrix4T<S>
UTvdbConvert(const openvdb::math::Mat4<S> &src)
{
    return UT_Matrix4T<S>(src(0,0), src(0,1), src(0,2), src(0,3),
			  src(1,0), src(1,1), src(1,2), src(1,3),
			  src(2,0), src(2,1), src(2,2), src(2,3),
			  src(3,0), src(3,1), src(3,2), src(3,3));
}

template <typename S>
UT_Matrix3T<S>
UTvdbConvert(const openvdb::math::Mat3<S> &src)
{
    return UT_Matrix3T<S>(src(0,0), src(0,1), src(0,2),
			  src(1,0), src(1,1), src(1,2),
			  src(2,0), src(2,1), src(2,2));
}

template <typename S>
UT_Matrix2T<S>
UTvdbConvert(const openvdb::math::Mat2<S> &src)
{
    return UT_Matrix2T<S>(src(0,0), src(0,1),
			  src(1,0), src(1,1));
}
// @}

/// Matrix conversion from UT to openvdb
// @{
template <typename S>
openvdb::math::Mat4<S>
UTvdbConvert(const UT_Matrix4T<S> &src)
{
    return openvdb::math::Mat4<S>(src(0,0), src(0,1), src(0,2), src(0,3),
				  src(1,0), src(1,1), src(1,2), src(1,3),
				  src(2,0), src(2,1), src(2,2), src(2,3),
				  src(3,0), src(3,1), src(3,2), src(3,3));
}
template <typename S>
openvdb::math::Mat3<S>
UTvdbConvert(const UT_Matrix3T<S> &src)
{
    return openvdb::math::Mat3<S>(src(0,0), src(0,1), src(0,2),
				  src(1,0), src(1,1), src(1,2),
				  src(2,0), src(2,1), src(2,2));
}
template <typename S>
openvdb::math::Mat2<S>
UTvdbConvert(const UT_Matrix2T<S> &src)
{
    return openvdb::math::Mat2<S>(src(0,0), src(0,1),
				  src(1,0), src(1,1));
}
// @}

/// Vector conversion from openvdb to UT
// @{
template <typename S>
UT_Vector4T<S>
UTvdbConvert(const openvdb::math::Vec4<S> &src)
{
    return UT_Vector4T<S>(src.asPointer());
}
template <typename S>
UT_Vector3T<S>
UTvdbConvert(const openvdb::math::Vec3<S> &src)
{
    return UT_Vector3T<S>(src.asPointer());
}
template <typename S>
UT_Vector2T<S>
UTvdbConvert(const openvdb::math::Vec2<S> &src)
{
    return UT_Vector2T<S>(src.asPointer());
}
// @}

/// Vector conversion from UT to openvdb
// @{
template <typename S>
openvdb::math::Vec4<S>
UTvdbConvert(const UT_Vector4T<S> &src)
{
    return openvdb::math::Vec4<S>(src.data());
}
template <typename S>
openvdb::math::Vec3<S>
UTvdbConvert(const UT_Vector3T<S> &src)
{
    return openvdb::math::Vec3<S>(src.data());
}
template <typename S>
openvdb::math::Vec2<S>
UTvdbConvert(const UT_Vector2T<S> &src)
{
    return openvdb::math::Vec2<S>(src.data());
}
// @}


/// Bounding box conversion from openvdb to UT
inline UT_BoundingBoxD
UTvdbConvert(const openvdb::CoordBBox &bbox)
{
    return UT_BoundingBoxD(UTvdbConvert(bbox.getStart().asVec3d()),
	UTvdbConvert(bbox.getEnd().asVec3d()));
}


template <typename T>
inline openvdb::math::Vec4<T>	SYSabs(const openvdb::math::Vec4<T> &v1)
{  return openvdb::math::Vec4<T>( SYSabs(v1[0]),
				  SYSabs(v1[1]),
				  SYSabs(v1[2]),
				  SYSabs(v1[3])
				);
}
template <typename T>
inline openvdb::math::Vec3<T>	SYSabs(const openvdb::math::Vec3<T> &v1)
{  return openvdb::math::Vec3<T>( SYSabs(v1[0]),
				  SYSabs(v1[1]),
				  SYSabs(v1[2])
				);
}
template <typename T>
inline openvdb::math::Vec2<T>	SYSabs(const openvdb::math::Vec2<T> &v1)
{  return openvdb::math::Vec2<T>( SYSabs(v1[0]),
				  SYSabs(v1[1])
				);
}

template <typename T>
inline openvdb::math::Vec4<T>	SYSmin(const openvdb::math::Vec4<T> &v1, const openvdb::math::Vec4<T> &v2)
{  return openvdb::math::Vec4<T>( SYSmin(v1[0], v2[0]),
				  SYSmin(v1[1], v2[1]),
				  SYSmin(v1[2], v2[2]),
				  SYSmin(v1[3], v2[3])
				);
}
template <typename T>
inline openvdb::math::Vec4<T>	SYSmax(const openvdb::math::Vec4<T> &v1, const openvdb::math::Vec4<T> &v2)
{  return openvdb::math::Vec4<T>( SYSmax(v1[0], v2[0]),
				  SYSmax(v1[1], v2[1]),
				  SYSmax(v1[2], v2[2]),
				  SYSmax(v1[3], v2[3])
				);
}
template <typename T>
inline openvdb::math::Vec4<T>	SYSmin(const openvdb::math::Vec4<T> &v1, const openvdb::math::Vec4<T> &v2, const openvdb::math::Vec4<T> &v3)
{  return openvdb::math::Vec4<T>( SYSmin(v1[0], v2[0], v3[0]),
				  SYSmin(v1[1], v2[1], v3[1]),
				  SYSmin(v1[2], v2[2], v3[2]),
				  SYSmin(v1[3], v2[3], v3[3])
				);
}
template <typename T>
inline openvdb::math::Vec4<T>	SYSmax(const openvdb::math::Vec4<T> &v1, const openvdb::math::Vec4<T> &v2, const openvdb::math::Vec4<T> &v3)
{  return openvdb::math::Vec4<T>( SYSmax(v1[0], v2[0], v3[0]),
				  SYSmax(v1[1], v2[1], v3[1]),
				  SYSmax(v1[2], v2[2], v3[2]),
				  SYSmax(v1[3], v2[3], v3[3])
				);
}
template <typename T>
inline openvdb::math::Vec3<T>	SYSmin(const openvdb::math::Vec3<T> &v1, const openvdb::math::Vec3<T> &v2)
{  return openvdb::math::Vec3<T>( SYSmin(v1[0], v2[0]),
				  SYSmin(v1[1], v2[1]),
				  SYSmin(v1[2], v2[2])
				);
}
template <typename T>
inline openvdb::math::Vec3<T>	SYSmax(const openvdb::math::Vec3<T> &v1, const openvdb::math::Vec3<T> &v2)
{  return openvdb::math::Vec3<T>( SYSmax(v1[0], v2[0]),
				  SYSmax(v1[1], v2[1]),
				  SYSmax(v1[2], v2[2])
				);
}
template <typename T>
inline openvdb::math::Vec3<T>	SYSmin(const openvdb::math::Vec3<T> &v1, const openvdb::math::Vec3<T> &v2, const openvdb::math::Vec3<T> &v3)
{  return openvdb::math::Vec3<T>( SYSmin(v1[0], v2[0], v3[0]),
				  SYSmin(v1[1], v2[1], v3[1]),
				  SYSmin(v1[2], v2[2], v3[2])
				);
}
template <typename T>
inline openvdb::math::Vec3<T>	SYSmax(const openvdb::math::Vec3<T> &v1, const openvdb::math::Vec3<T> &v2, const openvdb::math::Vec3<T> &v3)
{  return openvdb::math::Vec3<T>( SYSmax(v1[0], v2[0], v3[0]),
				  SYSmax(v1[1], v2[1], v3[1]),
				  SYSmax(v1[2], v2[2], v3[2])
				);
}
template <typename T>
inline openvdb::math::Vec2<T>	SYSmin(const openvdb::math::Vec2<T> &v1, const openvdb::math::Vec2<T> &v2)
{  return openvdb::math::Vec2<T>( SYSmin(v1[0], v2[0]),
				  SYSmin(v1[1], v2[1])
				);
}
template <typename T>
inline openvdb::math::Vec2<T>	SYSmax(const openvdb::math::Vec2<T> &v1, const openvdb::math::Vec2<T> &v2)
{  return openvdb::math::Vec2<T>( SYSmax(v1[0], v2[0]),
				  SYSmax(v1[1], v2[1])
				);
}
template <typename T>
inline openvdb::math::Vec2<T>	SYSmin(const openvdb::math::Vec2<T> &v1, const openvdb::math::Vec2<T> &v2, const openvdb::math::Vec2<T> &v3)
{  return openvdb::math::Vec2<T>( SYSmin(v1[0], v2[0], v3[0]),
				  SYSmin(v1[1], v2[1], v3[1])
				);
}
template <typename T>
inline openvdb::math::Vec2<T>	SYSmax(const openvdb::math::Vec2<T> &v1, const openvdb::math::Vec2<T> &v2, const openvdb::math::Vec2<T> &v3)
{  return openvdb::math::Vec2<T>( SYSmax(v1[0], v2[0], v3[0]),
				  SYSmax(v1[1], v2[1], v3[1])
				);
}

#endif // __HDK_UT_VDBUtils__

#endif // UT_VERSION_INT < 0x0c050157 // earlier than 12.5.343

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
