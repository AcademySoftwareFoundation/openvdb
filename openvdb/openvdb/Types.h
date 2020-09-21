// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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
#include <cstdint>
#include <memory>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

// One-dimensional scalar types
using Index32 = uint32_t;
using Index64 = uint64_t;
using Index   = Index32;
using Int16   = int16_t;
using Int32   = int32_t;
using Int64   = int64_t;
using Int     = Int32;
using Byte    = unsigned char;
using Real    = double;

// Two-dimensional vector types
using Vec2R = math::Vec2<Real>;
using Vec2I = math::Vec2<Index32>;
using Vec2f = math::Vec2<float>;
using Vec2H = math::Vec2<half>;
using math::Vec2i;
using math::Vec2s;
using math::Vec2d;

// Three-dimensional vector types
using Vec3R = math::Vec3<Real>;
using Vec3I = math::Vec3<Index32>;
using Vec3f = math::Vec3<float>;
using Vec3H = math::Vec3<half>;
using Vec3U8 = math::Vec3<uint8_t>;
using Vec3U16 = math::Vec3<uint16_t>;
using math::Vec3i;
using math::Vec3s;
using math::Vec3d;

using math::Coord;
using math::CoordBBox;
using BBoxd = math::BBox<Vec3d>;

// Four-dimensional vector types
using Vec4R = math::Vec4<Real>;
using Vec4I = math::Vec4<Index32>;
using Vec4f = math::Vec4<float>;
using Vec4H = math::Vec4<half>;
using math::Vec4i;
using math::Vec4s;
using math::Vec4d;

// Three-dimensional matrix types
using Mat3R = math::Mat3<Real>;
using math::Mat3s;
using math::Mat3d;

// Four-dimensional matrix types
using Mat4R = math::Mat4<Real>;
using math::Mat4s;
using math::Mat4d;

// Quaternions
using QuatR = math::Quat<Real>;
using math::Quats;
using math::Quatd;

// Dummy type for a voxel with a binary mask value, e.g. the active state
class ValueMask {};

// Use STL shared pointers from OpenVDB 4 on.
template<typename T> using SharedPtr = std::shared_ptr<T>;
template<typename T> using WeakPtr = std::weak_ptr<T>;

/// @brief Return a new shared pointer that points to the same object
/// as the given pointer but with possibly different <TT>const</TT>-ness.
/// @par Example:
/// @code
/// FloatGrid::ConstPtr grid = ...;
/// FloatGrid::Ptr nonConstGrid = ConstPtrCast<FloatGrid>(grid);
/// FloatGrid::ConstPtr constGrid = ConstPtrCast<const FloatGrid>(nonConstGrid);
/// @endcode
template<typename T, typename U> inline SharedPtr<T>
ConstPtrCast(const SharedPtr<U>& ptr) { return std::const_pointer_cast<T, U>(ptr); }

/// @brief Return a new shared pointer that is either null or points to
/// the same object as the given pointer after a @c dynamic_cast.
/// @par Example:
/// @code
/// GridBase::ConstPtr grid = ...;
/// FloatGrid::ConstPtr floatGrid = DynamicPtrCast<const FloatGrid>(grid);
/// @endcode
template<typename T, typename U> inline SharedPtr<T>
DynamicPtrCast(const SharedPtr<U>& ptr) { return std::dynamic_pointer_cast<T, U>(ptr); }

/// @brief Return a new shared pointer that points to the same object
/// as the given pointer after a @c static_cast.
/// @par Example:
/// @code
/// FloatGrid::Ptr floatGrid = ...;
/// GridBase::Ptr grid = StaticPtrCast<GridBase>(floatGrid);
/// @endcode
template<typename T, typename U> inline SharedPtr<T>
StaticPtrCast(const SharedPtr<U>& ptr) { return std::static_pointer_cast<T, U>(ptr); }


////////////////////////////////////////


/// @brief  Integer wrapper, required to distinguish PointIndexGrid and
///         PointDataGrid from Int32Grid and Int64Grid
/// @note   @c Kind is a dummy parameter used to create distinct types.
template<typename IntType_, Index Kind>
struct PointIndex
{
    static_assert(std::is_integral<IntType_>::value, "PointIndex requires an integer value type");

    using IntType = IntType_;

    PointIndex(IntType i = IntType(0)): mIndex(i) {}

    /// Explicit type conversion constructor
    template<typename T> explicit PointIndex(T i): mIndex(static_cast<IntType>(i)) {}

    operator IntType() const { return mIndex; }

    /// Needed to support the <tt>(zeroVal<PointIndex>() + val)</tt> idiom.
    template<typename T>
    PointIndex operator+(T x) { return PointIndex(mIndex + IntType(x)); }

private:
    IntType mIndex;
};


using PointIndex32 = PointIndex<Index32, 0>;
using PointIndex64 = PointIndex<Index64, 0>;

using PointDataIndex32 = PointIndex<Index32, 1>;
using PointDataIndex64 = PointIndex<Index64, 1>;


////////////////////////////////////////


/// @brief Helper metafunction used to determine if the first template
/// parameter is a specialization of the class template given in the second
/// template parameter
template <typename T, template <typename...> class Template>
struct IsSpecializationOf: public std::false_type {};

template <typename... Args, template <typename...> class Template>
struct IsSpecializationOf<Template<Args...>, Template>: public std::true_type {};


////////////////////////////////////////


template<typename T, bool = IsSpecializationOf<T, math::Vec2>::value ||
                            IsSpecializationOf<T, math::Vec3>::value ||
                            IsSpecializationOf<T, math::Vec4>::value>
struct VecTraits
{
    static const bool IsVec = true;
    static const int Size = T::size;
    using ElementType = typename T::ValueType;
};

template<typename T>
struct VecTraits<T, false>
{
    static const bool IsVec = false;
    static const int Size = 1;
    using ElementType = T;
};

template<typename T, bool = IsSpecializationOf<T, math::Quat>::value>
struct QuatTraits
{
    static const bool IsQuat = true;
    static const int Size = T::size;
    using ElementType = typename T::ValueType;
};

template<typename T>
struct QuatTraits<T, false>
{
    static const bool IsQuat = false;
    static const int Size = 1;
    using ElementType = T;
};

template<typename T, bool = IsSpecializationOf<T, math::Mat3>::value ||
                            IsSpecializationOf<T, math::Mat4>::value>
struct MatTraits
{
    static const bool IsMat = true;
    static const int Size = T::size;
    using ElementType = typename T::ValueType;
};

template<typename T>
struct MatTraits<T, false>
{
    static const bool IsMat = false;
    static const int Size = 1;
    using ElementType = T;
};

template<typename T, bool = VecTraits<T>::IsVec ||
                            QuatTraits<T>::IsQuat ||
                            MatTraits<T>::IsMat>
struct ValueTraits
{
    static const bool IsVec = VecTraits<T>::IsVec;
    static const bool IsQuat = QuatTraits<T>::IsQuat;
    static const bool IsMat = MatTraits<T>::IsMat;
    static const bool IsScalar = false;
    static const int Size = T::size;
    static const int Elements = IsMat ? Size*Size : Size;
    using ElementType = typename T::ValueType;
};

template<typename T>
struct ValueTraits<T, false>
{
    static const bool IsVec = false;
    static const bool IsQuat = false;
    static const bool IsMat = false;
    static const bool IsScalar = true;
    static const int Size = 1;
    static const int Elements = 1;
    using ElementType = T;
};


////////////////////////////////////////


/// @brief CanConvertType<FromType, ToType>::value is @c true if a value
/// of type @a ToType can be constructed from a value of type @a FromType.
template<typename FromType, typename ToType>
struct CanConvertType { enum { value = std::is_constructible<ToType, FromType>::value }; };

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


/// @brief CopyConstness<T1, T2>::Type is either <tt>const T2</tt>
/// or @c T2 with no @c const qualifier, depending on whether @c T1 is @c const.
/// @details For example,
/// - CopyConstness<int, int>::Type is @c int
/// - CopyConstness<int, const int>::Type is @c int
/// - CopyConstness<const int, int>::Type is <tt>const int</tt>
/// - CopyConstness<const int, const int>::Type is <tt>const int</tt>
template<typename FromType, typename ToType> struct CopyConstness {
    using Type = typename std::remove_const<ToType>::type;
};

/// @cond OPENVDB_TYPES_INTERNAL
template<typename FromType, typename ToType> struct CopyConstness<const FromType, ToType> {
    using Type = const ToType;
};
/// @endcond


////////////////////////////////////////


/// @cond OPENVDB_TYPES_INTERNAL

template<typename... Ts> struct TypeList; // forward declaration

namespace internal {

// Implementation details of @c TypeList

/// @brief Dummy struct, used as the return type from invalid or out-of-range
///        @c TypeList queries.
struct NullType {};


/// @brief   Type resolver for index queries
/// @details Defines a type at a given location within a @c TypeList or the
///          @c NullType if the index is out-of-range. The last template
///          parameter is used to determine if the index is in range.
/// @tparam ListT The @c TypeList
/// @tparam Idx The index of the type to get
template<typename ListT, size_t Idx, typename = void> struct TSGetElementImpl;

/// @brief Partial specialization for valid (in range) index queries.
/// @tparam Ts  Unpacked types from a @c TypeList
/// @tparam Idx The index of the type to get
template<typename... Ts, size_t Idx>
struct TSGetElementImpl<TypeList<Ts...>, Idx,
    typename std::enable_if<(Idx < sizeof...(Ts) && sizeof...(Ts))>::type> {
    using type = typename std::tuple_element<Idx, std::tuple<Ts...>>::type;
};

/// @brief Partial specialization for invalid index queries (i.e. out-of-range
///        indices such as @c TypeList<Int32>::Get<1>). Defines the NullType.
/// @tparam Ts  Unpacked types from a @c TypeList
/// @tparam Idx The index of the type to get
template<typename... Ts, size_t Idx>
struct TSGetElementImpl<TypeList<Ts...>, Idx,
    typename std::enable_if<!(Idx < sizeof...(Ts) && sizeof...(Ts))>::type> {
    using type = NullType;
};


/// @brief   Search for a given type within a @c TypeList.
/// @details If the type is found, a @c bool constant @c Value is set to true
///          and an @c int64_t @c Index points to the location of the type. If
///          multiple versions of the types exist, the value of @c Index is
///          always the location of the first matching type. If the type is not
///          found, @c Value is set to false and @c Index is set to -1.
/// @note    This implementation is recursively defined until the type is found
///          or until the end of the list is reached. The last template argument
///          is used as an internal counter to track the current index being
///          evaluated.
/// @tparam ListT The @c TypeList
/// @tparam T     The type to find
template <typename ListT, typename T, size_t=0>
struct TSHasTypeImpl;

/// @brief  Partial specialization on an empty @c TypeList, instantiated when
///         @c TSHasTypeImpl has been invoked with an empty @c TypeList or when
///         a recursive search reaches the end of a @c TypeList.
/// @tparam T The type to find
/// @tparam Idx Current index
template <typename T, size_t Idx>
struct TSHasTypeImpl<TypeList<>, T, Idx> {
    static constexpr bool Value = false;
    static constexpr int64_t Index = -1;
};

/// @brief Partial specialization on a @c TypeList which still contains types,
///        but the current type being evaluated @c U does not match the given
///        type @C T.
/// @tparam U The current type being evaluated within the @c TypeList
/// @tparam T The type to find
/// @tparam Ts Remaining types
/// @tparam Idx Current index
template <typename U, typename T, typename... Ts, size_t Idx>
struct TSHasTypeImpl<TypeList<U, Ts...>, T, Idx> :
    TSHasTypeImpl<TypeList<Ts...>, T, Idx+1> {};

/// @brief  Partial specialization on a @c TypeList where @c T matches the
///         current type (i.e. the type has been found).
/// @tparam T The type to find
/// @tparam Ts Remaining types
/// @tparam Idx Current index
template <typename T, typename... Ts, size_t Idx>
struct TSHasTypeImpl<TypeList<T, Ts...>, T, Idx>
{
    static constexpr bool Value = true;
    static constexpr int64_t Index = static_cast<int64_t>(Idx);
};


/// @brief    Remove any duplicate types from a @c TypeList.
/// @details  This implementation effectively rebuilds a @c TypeList by starting
///           with an empty @c TypeList and recursively defining an expanded
///           @c TypeList for every type (first to last), only if the type does
///           not already exist in the new @c TypeList. This has the effect of
///           dropping all but the first of duplicate types.
/// @note     Each type must define a new instantiation of this object.
/// @tparam ListT The starting @c TypeList, usually (but not limited to) an
///               empty @c TypeList
/// @tparam Ts    The list of types to make unique
template <typename ListT, typename... Ts>
struct TSMakeUniqueImpl {
    using type = ListT;
};

/// @brief  Partial specialization for type packs, where by the next type @c U
///         is checked in the existing type set @c Ts for duplication. If the
///         type does not exist, it is added to the new @c TypeList definition,
///         otherwise it is dropped. In either case, this class is recursively
///         defined with the remaining types @c Us.
/// @tparam Ts  Current types in the @c TypeList
/// @tparam U   Type to check for duplication in @c Ts
/// @tparam Us  Remaining types
template <typename... Ts, typename U, typename... Us>
struct TSMakeUniqueImpl<TypeList<Ts...>, U, Us...>
{
    using type = typename std::conditional<
        TSHasTypeImpl<TypeList<Ts...>, U>::Value,
        typename TSMakeUniqueImpl<TypeList<Ts...>, Us...>::type,
        typename TSMakeUniqueImpl<TypeList<Ts..., U>, Us...>::type  >::type;
};


/// @brief   Append any number of types to a @c TypeList
/// @details Defines a new @c TypeList with the provided types appended
/// @tparam ListT  The @c TypeList to append to
/// @tparam Ts     Types to append
template<typename ListT, typename... Ts> struct TSAppendImpl;

/// @brief  Partial specialization for a @c TypeList with a list of zero or more
///         types to append
/// @tparam Ts Current types within the @c TypeList
/// @tparam OtherTs Other types to append
template<typename... Ts, typename... OtherTs>
struct TSAppendImpl<TypeList<Ts...>, OtherTs...> {
    using type = TypeList<Ts..., OtherTs...>;
};

/// @brief  Partial specialization for a @c TypeList with another @c TypeList.
///         Appends the other TypeList's members.
/// @tparam Ts Types within the first @c TypeList
/// @tparam OtherTs Types within the second @c TypeList
template<typename... Ts, typename... OtherTs>
struct TSAppendImpl<TypeList<Ts...>, TypeList<OtherTs...>> {
    using type = TypeList<Ts..., OtherTs...>;
};


/// @brief   Remove all occurrences of type T from a @c TypeList
/// @details Defines a new @c TypeList with the provided types removed
/// @tparam ListT  The @c TypeList
/// @tparam T      Type to remove
template<typename ListT, typename T> struct TSEraseImpl;

/// @brief  Partial specialization for an empty @c TypeList
/// @tparam T Type to remove, has no effect
template<typename T>
struct TSEraseImpl<TypeList<>, T> { using type = TypeList<>; };

/// @brief  Partial specialization where the currently evaluating type in a
///         @c TypeList matches the type to remove. Recursively defines this
///         implementation with the remaining types.
/// @tparam Ts Unpacked types within the @c TypeList
/// @tparam T Type to remove
template<typename... Ts, typename T>
struct TSEraseImpl<TypeList<T, Ts...>, T> {
    using type = typename TSEraseImpl<TypeList<Ts...>, T>::type;
};

/// @brief  Partial specialization where the currently evaluating type @c T2 in
///         a @c TypeList does not match the type to remove @c T. Recursively
///         defines this implementation with the remaining types.
/// @tparam T2 Current type within the @c TypeList, which does not match @c T
/// @tparam Ts Other types within the @c TypeList
/// @tparam T  Type to remove
template<typename T2, typename... Ts, typename T>
struct TSEraseImpl<TypeList<T2, Ts...>, T> {
    using type = typename TSAppendImpl<TypeList<T2>,
        typename TSEraseImpl<TypeList<Ts...>, T>::type>::type;
};

/// @brief  Front end implementation to call TSEraseImpl which removes all
///         occurrences of a type from a @c TypeList. This struct handles the
///         case where the type to remove is another @c TypeList, in which case
///         all types in the second @c TypeList are removed from the first.
/// @tparam ListT  The @c TypeList
/// @tparam Ts     Types in the @c TypeList
template<typename ListT, typename... Ts> struct TSRemoveImpl;

/// @brief  Partial specialization when there are no types in the @c TypeList.
/// @tparam ListT  The @c TypeList
template<typename ListT>
struct TSRemoveImpl<ListT> { using type = ListT; };

/// @brief  Partial specialization when the type to remove @c T is not another
///         @c TypeList. @c T is removed from the @c TypeList.
/// @tparam ListT  The @c TypeList
/// @tparam T      Type to remove
/// @tparam Ts     Types in the @c TypeList
template<typename ListT, typename T, typename... Ts>
struct TSRemoveImpl<ListT, T, Ts...> {
    using type = typename TSRemoveImpl<typename TSEraseImpl<ListT, T>::type, Ts...>::type;
};

/// @brief  Partial specialization when the type to remove is another
///         @c TypeList. All types within the other type list are removed from
///         the first list.
/// @tparam ListT  The @c TypeList
/// @tparam Ts     Types from the second @c TypeList to remove from the first
template<typename ListT, typename... Ts>
struct TSRemoveImpl<ListT, TypeList<Ts...>> {
    using type = typename TSRemoveImpl<ListT, Ts...>::type;
};

/// @brief  Remove the first element of a type list. If the list is empty,
///         nothing is done. This base configuration handles the empty list.
/// @note   Much cheaper to instantiate than TSRemoveIndicesImpl
/// @tparam T  The @c TypeList
template<typename T>
struct TSRemoveFirstImpl {
    using type = TypeList<>;
};

/// @brief  Partial specialization for removing the first type of a @c TypeList
///         when the list is not empty i.e. does that actual work.
/// @tparam T   The first type in the @c TypeList.
/// @tparam Ts  Remaining types in the @c TypeList
template<typename T, typename... Ts>
struct TSRemoveFirstImpl<TypeList<T, Ts...>> {
    using type = TypeList<Ts...>;
};


/// @brief  Remove the last element of a type list. If the list is empty,
///         nothing is done. This base configuration handles the empty list.
/// @note   Cheaper to instantiate than TSRemoveIndicesImpl
/// @tparam T  The @c TypeList
template<typename T>
struct TSRemoveLastImpl { using type = TypeList<>; };

/// @brief  Partial specialization for removing the last type of a @c TypeList.
///         This instance is instantiated when the @c TypeList contains a
///         single type, or the primary struct which recursively removes types
///         (see below) hits the last type. Evaluates the last type to the empty
///         list (see above).
/// @tparam T   The last type in the @c TypeList
template<typename T>
struct TSRemoveLastImpl<TypeList<T>> : TSRemoveLastImpl<T> {};

/// @brief  Partial specialization for removing the last type of a @c TypeList
///         with a type list size of two or more. Recursively defines this
///         implementation with the remaining types, effectively rebuilding the
///         @c TypeList until the last type is hit, which is dropped.
/// @tparam T   The current type in the @c TypeList
/// @tparam Ts  Remaining types in the @c TypeList
template<typename T, typename... Ts>
struct TSRemoveLastImpl<TypeList<T, Ts...>>
{
    using type =
        typename TypeList<T>::template
            Append<typename TSRemoveLastImpl<TypeList<Ts...>>::type>;
};


/// @brief  Remove a number of types from a @c TypeList based on a @c First and
///         @c Last index.
/// @details Both indices are inclusive, such that when <tt>First == Last</tt>
///          a single type is removed (assuming the index exists). If
///          <tt>Last < First</tt>, nothing is done. Any indices which do not
///          exist are ignored. If @c Last is greater than the number of types
///          in the @c TypeList, all types from @c First to the end of the list
///          are dropped.
/// @tparam  ListT  The @c TypeList
/// @tparam  First  The first index
/// @tparam  Last   The last index
/// @tparam  Idx    Internal counter for the current index
template<typename ListT, size_t First, size_t Last, size_t Idx=0>
struct TSRemoveIndicesImpl;

/// @brief  Partial specialization for an empty @c TypeList
/// @tparam  First  The first index
/// @tparam  Last   The last index
/// @tparam  Idx    Internal counter for the current index
template<size_t First, size_t Last, size_t Idx>
struct TSRemoveIndicesImpl<TypeList<>, First, Last, Idx> {
     using type = TypeList<>;
};

/// @brief  Partial specialization for a @c TypeList containing a single element.
/// @tparam  T      The last or only type in a @c TypeList
/// @tparam  First  The first index
/// @tparam  Last   The last index
/// @tparam  Idx    Internal counter for the current index
template<typename T, size_t First, size_t Last, size_t Idx>
struct TSRemoveIndicesImpl<TypeList<T>, First, Last, Idx>
{
private:
    static constexpr bool Remove = Idx >= First && Idx <= Last;
public:
    using type = typename std::conditional<Remove, TypeList<>, TypeList<T>>::type;
};

/// @brief Partial specialization for a @c TypeList containing two or more types.
/// @details  This implementation effectively rebuilds a @c TypeList by starting
///           with an empty @c TypeList and recursively defining an expanded
///           @c TypeList for every type (first to last), only if the type's
///           index does not fall within the range of indices defines by
///           @c First and @c Last. Recursively defines this implementation with
///           all but the last type.
/// @tparam  T      The currently evaluating type within a @c TypeList
/// @tparam  Ts     Remaining types in the @c TypeList
/// @tparam  First  The first index
/// @tparam  Last   The last index
/// @tparam  Idx    Internal counter for the current index
template<typename T, typename... Ts, size_t First, size_t Last, size_t Idx>
struct TSRemoveIndicesImpl<TypeList<T, Ts...>, First, Last, Idx>
{
private:
    using ThisList = typename TSRemoveIndicesImpl<TypeList<T>, First, Last, Idx>::type;
    using NextList = typename TSRemoveIndicesImpl<TypeList<Ts...>, First, Last, Idx+1>::type;
public:
    using type = typename ThisList::template Append<NextList>;
};


template<typename OpT> inline void TSForEachImpl(OpT) {}
template<typename OpT, typename T, typename... Ts>
inline void TSForEachImpl(OpT op) { op(T()); TSForEachImpl<OpT, Ts...>(op); }

} // namespace internal

/// @endcond


/// @brief A list of types (not necessarily unique)
/// @details Example:
/// @code
/// using MyTypes = openvdb::TypeList<int, float, int, double, float>;
/// @endcode
template<typename... Ts>
struct TypeList
{
    /// The type of this list
    using Self = TypeList;

    /// @brief The number of types in the type list
    static constexpr size_t Size = sizeof...(Ts);

    /// @brief Access a particular element of this type list. If the index
    ///        is out of range, internal::NullType is returned.
    template<size_t N>
    using Get = typename internal::TSGetElementImpl<Self, N>::type;
    using Front = Get<0>;
    using Back = Get<Size-1>;

    /// @brief True if this list contains the given type, false otherwise
    /// @details Example:
    /// @code
    /// {
    ///     using IntTypes = openvdb::TypeList<Int16, Int32, Int64>;
    ///     using RealTypes = openvdb::TypeList<float, double>;
    /// }
    /// {
    ///     openvdb::TypeList<IntTypes>::Contains<Int32>; // true
    ///     openvdb::TypeList<RealTypes>::Contains<Int32>; // false
    /// }
    /// @endcode
    template<typename T>
    static constexpr bool Contains = internal::TSHasTypeImpl<Self, T>::Value;

    /// @brief Returns the index of the first found element of the given type, -1 if
    /// no matching element exists.
    /// @details Example:
    /// @code
    /// {
    ///     using IntTypes = openvdb::TypeList<Int16, Int32, Int64>;
    ///     using RealTypes = openvdb::TypeList<float, double>;
    /// }
    /// {
    ///     const int64_t L1 = openvdb::TypeList<IntTypes>::Index<Int32>;  // 1
    ///     const int64_t L2 = openvdb::TypeList<RealTypes>::Index<Int32>; // -1
    /// }
    /// @endcode
    template<typename T>
    static constexpr int64_t Index = internal::TSHasTypeImpl<Self, T>::Index;

    /// @brief Remove any duplicate types from this TypeList by rotating the
    /// next valid type left (maintains the order of other types). Optionally
    /// combine the result with another TypeList.
    /// @details Example:
    /// @code
    /// {
    ///     using Types = openvdb::TypeList<Int16, Int32, Int16, float, float, Int64>;
    /// }
    /// {
    ///     using UniqueTypes = Types::Unique<>; // <Int16, Int32, float, Int64>
    /// }
    /// @endcode
    template<typename ListT = TypeList<>>
    using Unique = typename internal::TSMakeUniqueImpl<ListT, Ts...>::type;

    /// @brief Append types, or the members of another TypeList, to this list.
    /// @details Example:
    /// @code
    /// {
    ///     using IntTypes = openvdb::TypeList<Int16, Int32, Int64>;
    ///     using RealTypes = openvdb::TypeList<float, double>;
    ///     using NumericTypes = IntTypes::Append<RealTypes>;
    /// }
    /// {
    ///     using IntTypes = openvdb::TypeList<Int16>::Append<Int32, Int64>;
    ///     using NumericTypes = IntTypes::Append<float>::Append<double>;
    /// }
    /// @endcode
    template<typename... TypesToAppend>
    using Append = typename internal::TSAppendImpl<Self, TypesToAppend...>::type;

    /// @brief Remove all occurrences of one or more types, or the members of
    /// another TypeList, from this list.
    /// @details Example:
    /// @code
    /// {
    ///     using NumericTypes = openvdb::TypeList<float, double, Int16, Int32, Int64>;
    ///     using LongTypes = openvdb::TypeList<Int64, double>;
    ///     using ShortTypes = NumericTypes::Remove<LongTypes>; // float, Int16, Int32
    /// }
    /// @endcode
    template<typename... TypesToRemove>
    using Remove = typename internal::TSRemoveImpl<Self, TypesToRemove...>::type;

    /// @brief Remove the first element of this type list. Has no effect if the
    ///        type list is already empty.
    /// @details Example:
    /// @code
    /// {
    ///     using IntTypes = openvdb::TypeList<Int16, Int32, Int64>;
    ///     using EmptyTypes = openvdb::TypeList<>;
    /// }
    /// {
    ///     IntTypes::PopFront; // openvdb::TypeList<Int32, Int64>;
    ///     EmptyTypes::PopFront; // openvdb::TypeList<>;
    /// }
    /// @endcode
    using PopFront = typename internal::TSRemoveFirstImpl<Self>::type;

    /// @brief Remove the last element of this type list. Has no effect if the
    ///        type list is already empty.
    /// @details Example:
    /// @code
    /// {
    ///     using IntTypes = openvdb::TypeList<Int16, Int32, Int64>;
    ///     using EmptyTypes = openvdb::TypeList<>;
    /// }
    /// {
    ///     IntTypes::PopBack; // openvdb::TypeList<Int16, Int32>;
    ///     EmptyTypes::PopBack; // openvdb::TypeList<>;
    /// }
    /// @endcode
    using PopBack = typename internal::TSRemoveLastImpl<Self>::type;

    /// @brief Return a new list with types removed by their location within the list.
    ///        If First is equal to Last, a single element is removed (if it exists).
    ///        If First is greater than Last, the list remains unmodified.
    /// @details Example:
    /// @code
    /// {
    ///     using NumericTypes = openvdb::TypeList<float, double, Int16, Int32, Int64>;
    /// }
    /// {
    ///     using IntTypes = NumericTypes::RemoveByIndex<0,1>; // openvdb::TypeList<Int16, Int32, Int64>;
    ///     using RealTypes = NumericTypes::RemoveByIndex<2,4>; // openvdb::TypeList<float, double>;
    ///     using RemoveFloat = NumericTypes::RemoveByIndex<0,0>; // openvdb::TypeList<double, Int16, Int32, Int64>;
    /// }
    /// @endcode
    template <size_t First, size_t Last>
    using RemoveByIndex = typename internal::TSRemoveIndicesImpl<Self, First, Last>::type;

    /// @brief Invoke a templated, unary functor on a value of each type in this list.
    /// @details Example:
    /// @code
    /// #include <typeinfo>
    ///
    /// template<typename ListT>
    /// void printTypeList()
    /// {
    ///     std::string sep;
    ///     auto op = [&](auto x) {  // C++14
    ///         std::cout << sep << typeid(decltype(x)).name(); sep = ", "; };
    ///     ListT::foreach(op);
    /// }
    ///
    /// using MyTypes = openvdb::TypeList<int, float, double>;
    /// printTypeList<MyTypes>(); // "i, f, d" (exact output is compiler-dependent)
    /// @endcode
    ///
    /// @note The functor object is passed by value.  Wrap it with @c std::ref
    /// to use the same object for each type.
    template<typename OpT>
    static void foreach(OpT op) { internal::TSForEachImpl<OpT, Ts...>(op); }
};


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
template<> inline const char* typeNameAsString<half>()              { return "half"; }
template<> inline const char* typeNameAsString<float>()             { return "float"; }
template<> inline const char* typeNameAsString<double>()            { return "double"; }
template<> inline const char* typeNameAsString<int8_t>()            { return "int8"; }
template<> inline const char* typeNameAsString<uint8_t>()           { return "uint8"; }
template<> inline const char* typeNameAsString<int16_t>()           { return "int16"; }
template<> inline const char* typeNameAsString<uint16_t>()          { return "uint16"; }
template<> inline const char* typeNameAsString<int32_t>()           { return "int32"; }
template<> inline const char* typeNameAsString<uint32_t>()          { return "uint32"; }
template<> inline const char* typeNameAsString<int64_t>()           { return "int64"; }
template<> inline const char* typeNameAsString<Vec2i>()             { return "vec2i"; }
template<> inline const char* typeNameAsString<Vec2s>()             { return "vec2s"; }
template<> inline const char* typeNameAsString<Vec2d>()             { return "vec2d"; }
template<> inline const char* typeNameAsString<Vec3U8>()            { return "vec3u8"; }
template<> inline const char* typeNameAsString<Vec3U16>()           { return "vec3u16"; }
template<> inline const char* typeNameAsString<Vec3i>()             { return "vec3i"; }
template<> inline const char* typeNameAsString<Vec3f>()             { return "vec3s"; }
template<> inline const char* typeNameAsString<Vec3d>()             { return "vec3d"; }
template<> inline const char* typeNameAsString<Vec4i>()             { return "vec4i"; }
template<> inline const char* typeNameAsString<Vec4f>()             { return "vec4s"; }
template<> inline const char* typeNameAsString<Vec4d>()             { return "vec4d"; }
template<> inline const char* typeNameAsString<std::string>()       { return "string"; }
template<> inline const char* typeNameAsString<Mat3s>()             { return "mat3s"; }
template<> inline const char* typeNameAsString<Mat3d>()             { return "mat3d"; }
template<> inline const char* typeNameAsString<Mat4s>()             { return "mat4s"; }
template<> inline const char* typeNameAsString<Mat4d>()             { return "mat4d"; }
template<> inline const char* typeNameAsString<math::Quats>()       { return "quats"; }
template<> inline const char* typeNameAsString<math::Quatd>()       { return "quatd"; }
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
    using AValueT = AValueType;
    using BValueT = BValueType;

    CombineArgs()
        : mAValPtr(nullptr)
        , mBValPtr(nullptr)
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


/// @brief Tag dispatch class that distinguishes shallow copy constructors
/// from deep copy constructors
class ShallowCopy {};
/// @brief Tag dispatch class that distinguishes topology copy constructors
/// from deep copy constructors
class TopologyCopy {};
/// @brief Tag dispatch class that distinguishes constructors during file input
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
