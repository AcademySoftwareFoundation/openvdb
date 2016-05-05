///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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

#ifndef OPENVDB_TYPES_HAS_BEEN_INCLUDED
#define OPENVDB_TYPES_HAS_BEEN_INCLUDED

#include "version.h"
#include "Platform.h"
#include <OpenEXR/half.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/BBox.h>
#include <openvdb/math/Quat.h>
#include <openvdb/math/Vec2.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/math/Vec4.h>
#include <openvdb/math/Mat3.h>
#include <openvdb/math/Mat4.h>
#include <openvdb/math/Coord.h>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/static_assert.hpp>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

// One-dimensional scalar types
typedef uint32_t            Index32;
typedef uint64_t            Index64;
typedef Index32             Index;
typedef int16_t             Int16;
typedef int32_t             Int32;
typedef int64_t             Int64;
typedef Int32               Int;
typedef unsigned char       Byte;
typedef double              Real;

// Two-dimensional vector types
typedef math::Vec2<Real>    Vec2R;
typedef math::Vec2<Index32> Vec2I;
typedef math::Vec2<float>   Vec2f;
typedef math::Vec2<half>    Vec2H;
using math::Vec2i;
using math::Vec2s;
using math::Vec2d;

// Three-dimensional vector types
typedef math::Vec3<Real>    Vec3R;
typedef math::Vec3<Index32> Vec3I;
typedef math::Vec3<float>   Vec3f;
typedef math::Vec3<half>    Vec3H;
using math::Vec3i;
using math::Vec3s;
using math::Vec3d;

using math::Coord;
using math::CoordBBox;
typedef math::BBox<Vec3d>   BBoxd;

// Four-dimensional vector types
typedef math::Vec4<Real>    Vec4R;
typedef math::Vec4<Index32> Vec4I;
typedef math::Vec4<float>   Vec4f;
typedef math::Vec4<half>    Vec4H;
using math::Vec4i;
using math::Vec4s;
using math::Vec4d;

// Three-dimensional matrix types
typedef math::Mat3<Real>    Mat3R;

// Four-dimensional matrix types
typedef math::Mat4<Real>    Mat4R;
typedef math::Mat4<double>  Mat4d;
typedef math::Mat4<float>   Mat4s;

// Quaternions
typedef math::Quat<Real>    QuatR;

// Dummy type for a voxel with a binary mask value, e.g. the active state
class ValueMask {};


////////////////////////////////////////


/// @brief  Integer wrapper, required to distinguish PointIndexGrid and
///         PointDataGrid from Int32Grid and Int64Grid
/// @note   @c Kind is a dummy parameter used to create distinct types.
template<typename IntType_, Index Kind>
struct PointIndex
{
    BOOST_STATIC_ASSERT(boost::is_integral<IntType_>::value);

    typedef IntType_ IntType;

    PointIndex(IntType i = IntType(0)): mIndex(i) {}

    operator IntType() const { return mIndex; }

    /// Needed to support the <tt>(zeroVal<PointIndex>() + val)</tt> idiom.
    template<typename T>
    PointIndex operator+(T x) { return PointIndex(mIndex + IntType(x)); }

private:
    IntType mIndex;
};


typedef PointIndex<Index32, 0> PointIndex32;
typedef PointIndex<Index64, 0> PointIndex64;

typedef PointIndex<Index32, 1> PointDataIndex32;
typedef PointIndex<Index64, 1> PointDataIndex64;


////////////////////////////////////////


template<typename T> struct VecTraits {
    static const bool IsVec = false;
    static const int Size = 1;
    typedef T ElementType;
};
template<typename T> struct VecTraits<math::Vec2<T> > {
    static const bool IsVec = true;
    static const int Size = 2;
    typedef T ElementType;
};
template<typename T> struct VecTraits<math::Vec3<T> > {
    static const bool IsVec = true;
    static const int Size = 3;
    typedef T ElementType;
};
template<typename T> struct VecTraits<math::Vec4<T> > {
    static const bool IsVec = true;
    static const int Size = 4;
    typedef T ElementType;
};


////////////////////////////////////////


/// @brief CanConvertType<FromType, ToType>::value is @c true if a value
/// of type @a ToType can be constructed from a value of type @a FromType.
///
/// @note @c boost::is_convertible tests for implicit convertibility only.
/// What we want is the equivalent of C++11's @c std::is_constructible,
/// which allows for explicit conversions as well.  Unfortunately, not all
/// compilers support @c std::is_constructible yet, so for now, types that
/// can only be converted explicitly have to be indicated with specializations
/// of this template.
template<typename FromType, typename ToType>
struct CanConvertType { enum { value = boost::is_convertible<FromType, ToType>::value }; };

// Specializations for vector types, which can be constructed from values
// of their own ValueTypes (or values that can be converted to their ValueTypes),
// but only explicitly
template<typename T> struct CanConvertType<T, math::Vec2<T> > { enum { value = true }; };
template<typename T> struct CanConvertType<T, math::Vec3<T> > { enum { value = true }; };
template<typename T> struct CanConvertType<T, math::Vec4<T> > { enum { value = true }; };
template<typename T> struct CanConvertType<math::Vec2<T>, math::Vec2<T> > { enum {value = true}; };
template<typename T> struct CanConvertType<math::Vec3<T>, math::Vec3<T> > { enum {value = true}; };
template<typename T> struct CanConvertType<math::Vec4<T>, math::Vec4<T> > { enum {value = true}; };
template<typename T0, typename T1>
struct CanConvertType<T0, math::Vec2<T1> > { enum { value = CanConvertType<T0, T1>::value }; };
template<typename T0, typename T1>
struct CanConvertType<T0, math::Vec3<T1> > { enum { value = CanConvertType<T0, T1>::value }; };
template<typename T0, typename T1>
struct CanConvertType<T0, math::Vec4<T1> > { enum { value = CanConvertType<T0, T1>::value }; };
template<> struct CanConvertType<PointIndex32, PointDataIndex32> { enum {value = true}; };
template<> struct CanConvertType<PointDataIndex32, PointIndex32> { enum {value = true}; };    
template<typename T>
struct CanConvertType<T, ValueMask> { enum {value = CanConvertType<T, bool>::value}; };
template<typename T>
struct CanConvertType<ValueMask, T> { enum {value = CanConvertType<bool, T>::value}; };
    
////////////////////////////////////////


// Add new items to the *end* of this list, and update NUM_GRID_CLASSES.
enum GridClass {
    GRID_UNKNOWN = 0,
    GRID_LEVEL_SET,
    GRID_FOG_VOLUME,
    GRID_STAGGERED
};
enum { NUM_GRID_CLASSES = GRID_STAGGERED + 1 };

static const Real LEVEL_SET_HALF_WIDTH = 3;

/// The type of a vector determines how transforms are applied to it:
/// <dl>
/// <dt><b>Invariant</b>
/// <dd>Does not transform (e.g., tuple, uvw, color)
///
/// <dt><b>Covariant</b>
/// <dd>Apply inverse-transpose transformation: @e w = 0, ignores translation
///     (e.g., gradient/normal)
///
/// <dt><b>Covariant Normalize</b>
/// <dd>Apply inverse-transpose transformation: @e w = 0, ignores translation,
///     vectors are renormalized (e.g., unit normal)
///
/// <dt><b>Contravariant Relative</b>
/// <dd>Apply "regular" transformation: @e w = 0, ignores translation
///     (e.g., displacement, velocity, acceleration)
///
/// <dt><b>Contravariant Absolute</b>
/// <dd>Apply "regular" transformation: @e w = 1, vector translates (e.g., position)
/// </dl>
enum VecType {
    VEC_INVARIANT = 0,
    VEC_COVARIANT,
    VEC_COVARIANT_NORMALIZE,
    VEC_CONTRAVARIANT_RELATIVE,
    VEC_CONTRAVARIANT_ABSOLUTE
};
enum { NUM_VEC_TYPES = VEC_CONTRAVARIANT_ABSOLUTE + 1 };


/// Specify how grids should be merged during certain (typically multithreaded) operations.
/// <dl>
/// <dt><b>MERGE_ACTIVE_STATES</b>
/// <dd>The output grid is active wherever any of the input grids is active.
///
/// <dt><b>MERGE_NODES</b>
/// <dd>The output grid's tree has a node wherever any of the input grids' trees
///     has a node, regardless of any active states.
///
/// <dt><b>MERGE_ACTIVE_STATES_AND_NODES</b>
/// <dd>The output grid is active wherever any of the input grids is active,
///     and its tree has a node wherever any of the input grids' trees has a node.
/// </dl>
enum MergePolicy {
    MERGE_ACTIVE_STATES = 0,
    MERGE_NODES,
    MERGE_ACTIVE_STATES_AND_NODES
};


////////////////////////////////////////


template<typename T> const char* typeNameAsString()                 { return typeid(T).name(); }
template<> inline const char* typeNameAsString<bool>()              { return "bool"; }
template<> inline const char* typeNameAsString<ValueMask>()         { return "mask"; }
template<> inline const char* typeNameAsString<float>()             { return "float"; }
template<> inline const char* typeNameAsString<double>()            { return "double"; }
template<> inline const char* typeNameAsString<int32_t>()           { return "int32"; }
template<> inline const char* typeNameAsString<uint32_t>()          { return "uint32"; }
template<> inline const char* typeNameAsString<int64_t>()           { return "int64"; }
template<> inline const char* typeNameAsString<Vec2i>()             { return "vec2i"; }
template<> inline const char* typeNameAsString<Vec2s>()             { return "vec2s"; }
template<> inline const char* typeNameAsString<Vec2d>()             { return "vec2d"; }
template<> inline const char* typeNameAsString<Vec3i>()             { return "vec3i"; }
template<> inline const char* typeNameAsString<Vec3f>()             { return "vec3s"; }
template<> inline const char* typeNameAsString<Vec3d>()             { return "vec3d"; }
template<> inline const char* typeNameAsString<std::string>()       { return "string"; }
template<> inline const char* typeNameAsString<Mat4s>()             { return "mat4s"; }
template<> inline const char* typeNameAsString<Mat4d>()             { return "mat4d"; }
template<> inline const char* typeNameAsString<PointIndex32>()      { return "ptidx32"; }
template<> inline const char* typeNameAsString<PointIndex64>()      { return "ptidx64"; }
template<> inline const char* typeNameAsString<PointDataIndex32>()  { return "ptdataidx32"; }
template<> inline const char* typeNameAsString<PointDataIndex64>()  { return "ptdataidx64"; }


////////////////////////////////////////


/// @brief This struct collects both input and output arguments to "grid combiner" functors
/// used with the tree::TypedGrid::combineExtended() and combine2Extended() methods.
/// AValueType and BValueType are the value types of the two grids being combined.
///
/// @see openvdb/tree/Tree.h for usage information.
///
/// Setter methods return references to this object, to facilitate the following usage:
/// @code
///     CombineArgs<float> args;
///     myCombineOp(args.setARef(aVal).setBRef(bVal).setAIsActive(true).setBIsActive(false));
/// @endcode
template<typename AValueType, typename BValueType = AValueType>
class CombineArgs
{
public:
    typedef AValueType AValueT;
    typedef BValueType BValueT;

    CombineArgs()
        : mAValPtr(NULL)
        , mBValPtr(NULL)
        , mResultValPtr(&mResultVal)
        , mAIsActive(false)
        , mBIsActive(false)
        , mResultIsActive(false)
    {
    }

    /// Use this constructor when the result value is stored externally.
    CombineArgs(const AValueType& a, const BValueType& b, AValueType& result,
                bool aOn = false, bool bOn = false)
        : mAValPtr(&a)
        , mBValPtr(&b)
        , mResultValPtr(&result)
        , mAIsActive(aOn)
        , mBIsActive(bOn)
    {
        this->updateResultActive();
    }

    /// Use this constructor when the result value should be stored in this struct.
    CombineArgs(const AValueType& a, const BValueType& b, bool aOn = false, bool bOn = false)
        : mAValPtr(&a)
        , mBValPtr(&b)
        , mResultValPtr(&mResultVal)
        , mAIsActive(aOn)
        , mBIsActive(bOn)
    {
        this->updateResultActive();
    }

    /// Get the A input value.
    const AValueType& a() const { return *mAValPtr; }
    /// Get the B input value.
    const BValueType& b() const { return *mBValPtr; }
    //@{
    /// Get the output value.
    const AValueType& result() const { return *mResultValPtr; }
    AValueType& result() { return *mResultValPtr; }
    //@}

    /// Set the output value.
    CombineArgs& setResult(const AValueType& val) { *mResultValPtr = val; return *this; }

    /// Redirect the A value to a new external source.
    CombineArgs& setARef(const AValueType& a) { mAValPtr = &a; return *this; }
    /// Redirect the B value to a new external source.
    CombineArgs& setBRef(const BValueType& b) { mBValPtr = &b; return *this; }
    /// Redirect the result value to a new external destination.
    CombineArgs& setResultRef(AValueType& val) { mResultValPtr = &val; return *this; }

    /// @return true if the A value is active
    bool aIsActive() const { return mAIsActive; }
    /// @return true if the B value is active
    bool bIsActive() const { return mBIsActive; }
    /// @return true if the output value is active
    bool resultIsActive() const { return mResultIsActive; }

    /// Set the active state of the A value.
    CombineArgs& setAIsActive(bool b) { mAIsActive = b; updateResultActive(); return *this; }
    /// Set the active state of the B value.
    CombineArgs& setBIsActive(bool b) { mBIsActive = b; updateResultActive(); return *this; }
    /// Set the active state of the output value.
    CombineArgs& setResultIsActive(bool b) { mResultIsActive = b; return *this; }

protected:
    /// By default, the result value is active if either of the input values is active,
    /// but this behavior can be overridden by calling setResultIsActive().
    void updateResultActive() { mResultIsActive = mAIsActive || mBIsActive; }

    const AValueType* mAValPtr;   // pointer to input value from A grid
    const BValueType* mBValPtr;   // pointer to input value from B grid
    AValueType mResultVal;        // computed output value (unused if stored externally)
    AValueType* mResultValPtr;    // pointer to either mResultVal or an external value
    bool mAIsActive, mBIsActive;  // active states of A and B values
    bool mResultIsActive;         // computed active state (default: A active || B active)
};


/// This struct adapts a "grid combiner" functor to swap the A and B grid values
/// (e.g., so that if the original functor computes a + 2 * b, the adapted functor
/// will compute b + 2 * a).
template<typename ValueType, typename CombineOp>
struct SwappedCombineOp
{
    SwappedCombineOp(CombineOp& _op): op(_op) {}

    void operator()(CombineArgs<ValueType>& args)
    {
        CombineArgs<ValueType> swappedArgs(args.b(), args.a(), args.result(),
            args.bIsActive(), args.aIsActive());
        op(swappedArgs);
    }

    CombineOp& op;
};


////////////////////////////////////////


/// In copy constructors, members stored as shared pointers can be handled
/// in several ways:
/// <dl>
/// <dt><b>CP_NEW</b>
/// <dd>Don't copy the member; default construct a new member object instead.
///
/// <dt><b>CP_SHARE</b>
/// <dd>Copy the shared pointer, so that the original and new objects share
///     the same member.
///
/// <dt><b>CP_COPY</b>
/// <dd>Create a deep copy of the member.
/// </dl>
enum CopyPolicy { CP_NEW, CP_SHARE, CP_COPY };


// Dummy class that distinguishes shallow copy constructors from
// deep copy constructors
class ShallowCopy {};
// Dummy class that distinguishes topology copy constructors from
// deep copy constructors
class TopologyCopy {};
// Dummy class that distinguishes constructors during file input
class PartialCreate {};

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#if defined(__ICC)

// Use these defines to bracket a region of code that has safe static accesses.
// Keep the region as small as possible.
#define OPENVDB_START_THREADSAFE_STATIC_REFERENCE   __pragma(warning(disable:1710))
#define OPENVDB_FINISH_THREADSAFE_STATIC_REFERENCE  __pragma(warning(default:1710))
#define OPENVDB_START_THREADSAFE_STATIC_WRITE       __pragma(warning(disable:1711))
#define OPENVDB_FINISH_THREADSAFE_STATIC_WRITE      __pragma(warning(default:1711))
#define OPENVDB_START_THREADSAFE_STATIC_ADDRESS     __pragma(warning(disable:1712))
#define OPENVDB_FINISH_THREADSAFE_STATIC_ADDRESS    __pragma(warning(default:1712))

// Use these defines to bracket a region of code that has unsafe static accesses.
// Keep the region as small as possible.
#define OPENVDB_START_NON_THREADSAFE_STATIC_REFERENCE   __pragma(warning(disable:1710))
#define OPENVDB_FINISH_NON_THREADSAFE_STATIC_REFERENCE  __pragma(warning(default:1710))
#define OPENVDB_START_NON_THREADSAFE_STATIC_WRITE       __pragma(warning(disable:1711))
#define OPENVDB_FINISH_NON_THREADSAFE_STATIC_WRITE      __pragma(warning(default:1711))
#define OPENVDB_START_NON_THREADSAFE_STATIC_ADDRESS     __pragma(warning(disable:1712))
#define OPENVDB_FINISH_NON_THREADSAFE_STATIC_ADDRESS    __pragma(warning(default:1712))

// Simpler version for one-line cases
#define OPENVDB_THREADSAFE_STATIC_REFERENCE(CODE) \
    __pragma(warning(disable:1710)); CODE; __pragma(warning(default:1710))
#define OPENVDB_THREADSAFE_STATIC_WRITE(CODE) \
    __pragma(warning(disable:1711)); CODE; __pragma(warning(default:1711))
#define OPENVDB_THREADSAFE_STATIC_ADDRESS(CODE) \
    __pragma(warning(disable:1712)); CODE; __pragma(warning(default:1712))

#else // GCC does not support these compiler warnings

#define OPENVDB_START_THREADSAFE_STATIC_REFERENCE
#define OPENVDB_FINISH_THREADSAFE_STATIC_REFERENCE
#define OPENVDB_START_THREADSAFE_STATIC_WRITE
#define OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
#define OPENVDB_START_THREADSAFE_STATIC_ADDRESS
#define OPENVDB_FINISH_THREADSAFE_STATIC_ADDRESS

#define OPENVDB_START_NON_THREADSAFE_STATIC_REFERENCE
#define OPENVDB_FINISH_NON_THREADSAFE_STATIC_REFERENCE
#define OPENVDB_START_NON_THREADSAFE_STATIC_WRITE
#define OPENVDB_FINISH_NON_THREADSAFE_STATIC_WRITE
#define OPENVDB_START_NON_THREADSAFE_STATIC_ADDRESS
#define OPENVDB_FINISH_NON_THREADSAFE_STATIC_ADDRESS

#define OPENVDB_THREADSAFE_STATIC_REFERENCE(CODE) CODE
#define OPENVDB_THREADSAFE_STATIC_WRITE(CODE) CODE
#define OPENVDB_THREADSAFE_STATIC_ADDRESS(CODE) CODE

#endif // defined(__ICC)

#endif // OPENVDB_TYPES_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
