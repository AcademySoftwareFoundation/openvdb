// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file TypeList.h
///
/// @brief  A TypeList provides a compile time sequence of heterogeneous types
///   which can be accessed, transformed and executed over in various ways.
///   It incorporates a subset of functionality similar to boost::mpl::vector
///   however provides most of its content through using declarations rather
///   than additional typed classes.

#ifndef OPENVDB_TYPELIST_HAS_BEEN_INCLUDED
#define OPENVDB_TYPELIST_HAS_BEEN_INCLUDED

#include "version.h"

#include <tuple>
#include <type_traits>

/// We should usually not be decorating public API functions with attributes
/// such as always_inline. However many compilers are notoriously bad at
/// inlining recursive template loops with default inline settings. The
/// TypeList and TupleList metaprogram constructs heavily use this C++ feature
/// and the performance difference can be substantial, even for very small
/// lists. You can disable this behaviour by setting the define:
///    OPENVDB_TYPELIST_NO_FORCE_INLINE
/// This will disable the force inling on public API methods in this file.
#ifdef OPENVDB_TYPELIST_NO_FORCE_INLINE
#define OPENVDB_TYPELIST_FORCE_INLINE inline
#else
#define OPENVDB_TYPELIST_FORCE_INLINE OPENVDB_FORCE_INLINE
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @cond OPENVDB_DOCS_INTERNAL

// forward declarations
template<typename... Ts> struct TypeList;
template<typename... Ts> struct TupleList;

namespace typelist_internal {

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


/// @brief  Similar to TsAppendImpl but only appends types to a list if the
///   type does not alreay exist in the list.
/// @details Defines a new @c TypeList with non-unique types appended
/// @tparam U      Type to append
/// @tparam ListT  The @c TypeList to append to
template <typename U, typename ListT,
  bool ListContainsType = TSHasTypeImpl<ListT, U>::Value>
struct TSAppendUniqueImpl;

/// @brief  Partial specialization where the currently evaluating type @c U in
///   a @c TypeList already exists in the list. Returns the unmodified list.
/// @tparam U  Type to append
/// @tparam Ts Other types within the @c TypeList
template <typename U, typename... Ts>
struct TSAppendUniqueImpl<U, TypeList<Ts...>, true> {
private:
    using RemovedU = typename TypeList<Ts...>::template Remove<U>;
public:
    /// @note  It's simpler to remove the current type U and append the rest by
    ///   just having "using type = TypeList<Ts...>". However this ends up with
    ///   with keeping the last seen type rather than the first which this
    ///   method historically did. e.g:
    ///      TypeList<float, int, float>::Unique<> can become:
    ///        a)  TypeList<float, int>  currently
    ///        b)  TypeList<int, float>  if we used the afformentioned technique
    ///  Might be useful to have both? Complexity in (a) is currently linear so
    ///  this shouldn't be a problem, but be careful this doesn't change.
    //using type = TypeList<Ts...>;
    using type = typename TypeList<U>::template Append<RemovedU>;
};

/// @brief  Partial specialization where the currently evaluating type @c U in
///   a @c TypeList does not exists in the list. Returns the appended list.
/// @tparam U  Type to append
/// @tparam Ts Other types within the @c TypeList
template <typename U, typename... Ts>
struct TSAppendUniqueImpl<U, TypeList<Ts...>, false> {
    using type = TypeList<U, Ts...>;
};

/// @brief    Reconstruct a @c TypeList containing only unique types.
/// @details  This implementation effectively rebuilds a @c TypeList by
///   starting with an empty @c TypeList and recursively defining an expanded
///   @c TypeList for every type (first to last), only if the type does not
///   already exist in the new @c TypeList. This has the effect of dropping all
///   but the first of duplicate types.
/// @warning  This implementation previously used an embdedded std::conditional
///   which resulted in drastically slow compilation times. If you're changing
///   this implementation make sure to profile compile times with larger lists.
/// @tparam Ts Types within the @c TypeList
template <typename... Ts>
struct TSRecurseAppendUniqueImpl;

/// @brief  Terminate type recursion when the end of a @c TypeList is reached.
template <>
struct TSRecurseAppendUniqueImpl<> {
    using type = TypeList<>;
};

/// @brief  Merge and unpack an initial @c TypeList from the first argument if
///   such a @c TypeList has been provided.
/// @tparam Ts      Types within the first @c TypeList
/// @tparam OtherTs Other types
template <typename... Ts, typename... OtherTs>
struct TSRecurseAppendUniqueImpl<TypeList<Ts...>, OtherTs...> {
    using type = typename TSRecurseAppendUniqueImpl<OtherTs..., Ts...>::type;
};

/// @brief  Recursively call TSRecurseAppendUniqueImpl with each type in the
///   provided @c TypeLists, rebuilding a new list with only the unique set
///   of types.
/// @tparam U  Next type to check for uniqueness and append
/// @tparam Ts Remaining types within the @c TypeList
template <typename U, typename... Ts>
struct TSRecurseAppendUniqueImpl<U, Ts...>
{
    using type = typename TSAppendUniqueImpl<U,
            typename TSRecurseAppendUniqueImpl<Ts...>::type
        >::type;
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

/// @brief  Transform a @c TypeList, converting each type into a new type based
///         on a transformation struct @c OpT.
/// @details This implementation iterates through each type in a @c TypeList
///          and builds a new @c TypeList where each element is resolved through
///          a user provided converter which provides a @c Type definition.
/// @tparam  OpT  User struct to convert each type
/// @tparam Ts  Remaining types in the @c TypeList
template<template <typename> class OpT, typename... Ts> struct TSTranformImpl;

/// @brief  Partial specialization for an empty @c TypeList
/// @tparam  OpT  User struct to convert each type
template<template <typename> class OpT>
struct TSTranformImpl<OpT> {
    using type = TypeList<>;
};

/// @brief  Implementation of TSTranformImpl. See fwd declaration for details.
/// @tparam OpT  User struct to convert each type
/// @tparam Ts   Remaining types in the @c TypeList
/// @tparam T    Current type being converted
template<template <typename> class OpT, typename T, typename... Ts>
struct TSTranformImpl<OpT, T, Ts...> {
private:
    using NextList = typename TSTranformImpl<OpT, Ts...>::type;
public:
    // Invoke Append for each type to match the behaviour should OpT<T> be a
    // TypeList<>
    using type = typename TSTranformImpl<OpT>::type::template
        Append<OpT<T>>::template
        Append<NextList>;
};

/// @brief  Partial apply specialization for an empty @c TypeList
/// @tparam  OpT    User functor to apply to the first valid type
/// @tparam  BaseT  Type of the provided obj
/// @tparam  T      Current type
/// @tparam  Ts     Remaining types
template<typename OpT, typename BaseT, typename T, typename ...Ts>
struct TSApplyImpl { static bool apply(BaseT&, OpT&) { return false; } };

/// @brief  Apply a unary functor to a provided object only if the object
///   satisfies the cast requirement of isType<T> for a type in a TypeList.
/// @note  Iteration terminates immediately on the first valid type and true
///    is returned.
/// @tparam  OpT    User functor to apply to the first valid type
/// @tparam  BaseT  Type of the provided obj
/// @tparam  T      Current type
/// @tparam  Ts     Remaining types
template<typename OpT, typename BaseT, typename T, typename ...Ts>
struct TSApplyImpl<OpT, BaseT, TypeList<T, Ts...>>
{
    using CastT =
        typename std::conditional<std::is_const<BaseT>::value, const T, T>::type;

    static bool apply(BaseT& obj, OpT& op)
    {
        if (obj.template isType<T>()) {
            op(static_cast<CastT&>(obj));
            return true;
        }
        return TSApplyImpl<OpT, BaseT, TypeList<Ts...>>::apply(obj, op);
    }
};

template<template <typename> class OpT> inline void TSForEachImpl() {}
template<template <typename> class OpT, typename T, typename... Ts>
inline void TSForEachImpl() { OpT<T>()(); TSForEachImpl<OpT, Ts...>(); }

template<typename OpT> inline void TSForEachImpl(OpT) {}
template<typename OpT, typename T, typename... Ts>
constexpr OPENVDB_FORCE_INLINE void TSForEachImpl(OpT op) {
    op(T()); TSForEachImpl<OpT, Ts...>(op);
}

///////////////////////////////////////////////////////////////////////////////

// Implementation details of @c TupleList

template<size_t Iter, size_t End, typename OpT, typename TupleT>
constexpr OPENVDB_FORCE_INLINE void TSForEachImpl(
    [[maybe_unused]] OpT op,
    [[maybe_unused]] TupleT& tup)
{
    if constexpr(Iter<End) {
        op(std::get<Iter>(tup));
        TSForEachImpl<Iter+1, End, OpT, TupleT>(op, tup);
    }
}

template<typename OpT, size_t Iter, size_t End>
constexpr OPENVDB_FORCE_INLINE void TSForEachIndexImpl([[maybe_unused]] OpT op)
{
    if constexpr(Iter<End) {
        op(std::integral_constant<std::size_t, Iter>());
        TSForEachIndexImpl<OpT, Iter+1, End>(op);
    }
}

template<typename OpT, typename RetT, size_t Iter, size_t End>
constexpr OPENVDB_FORCE_INLINE RetT TSEvalFirstIndex([[maybe_unused]] OpT op, const RetT def)
{
    if constexpr(Iter<End) {
        if (auto ret = op(std::integral_constant<std::size_t, Iter>())) return ret;
        return TSEvalFirstIndex<OpT, RetT, Iter+1, End>(op, def);
    }
    else return def;
}

template<class Pred, class OpT, typename TupleT, size_t Iter, size_t End>
constexpr OPENVDB_FORCE_INLINE
void TSEvalFirstPredImpl(
    [[maybe_unused]] Pred pred,
    [[maybe_unused]] OpT op,
    [[maybe_unused]] TupleT& tup)
{
    if constexpr (Iter<End) {
        constexpr auto Idx = std::integral_constant<std::size_t, Iter>();
        if (pred(Idx)) op(std::get<Idx>(tup));
        else TSEvalFirstPredImpl<Pred, OpT, TupleT, Iter+1, End>(pred, op, tup);
    }
}

template<class Pred, class OpT, typename TupleT, typename RetT, size_t Iter, size_t End>
constexpr OPENVDB_FORCE_INLINE
RetT TSEvalFirstPredImpl(
    [[maybe_unused]] Pred pred,
    [[maybe_unused]] OpT op,
    [[maybe_unused]] TupleT& tup,
    RetT def)
{
    if constexpr (Iter<End) {
        constexpr auto Idx = std::integral_constant<std::size_t, Iter>();
        if (pred(Idx)) return op(std::get<Idx>(tup));
        else return TSEvalFirstPredImpl
            <Pred, OpT, TupleT, RetT, Iter+1, End>(pred, op, tup, def);
    }
    else return def;
}

} // namespace internal

/// @endcond


/// @brief
template<size_t Start, size_t End, typename OpT>
OPENVDB_TYPELIST_FORCE_INLINE auto foreachIndex(OpT op)
{
    typelist_internal::TSForEachIndexImpl<OpT, Start, End>(op);
}

template<size_t Start, size_t End, typename OpT, typename RetT>
OPENVDB_TYPELIST_FORCE_INLINE RetT evalFirstIndex(OpT op, const RetT def = RetT())
{
    return typelist_internal::TSEvalFirstIndex<OpT, RetT, Start, End>(op, def);
}

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

    using AsTupleList = TupleList<Ts...>;

    /// @brief The number of types in the type list
    static constexpr size_t Size = sizeof...(Ts);

    /// @brief Access a particular element of this type list. If the index
    ///        is out of range, typelist_internal::NullType is returned.
    template<size_t N>
    using Get = typename typelist_internal::TSGetElementImpl<Self, N>::type;
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
    static constexpr bool Contains = typelist_internal::TSHasTypeImpl<Self, T>::Value;

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
    static constexpr int64_t Index = typelist_internal::TSHasTypeImpl<Self, T>::Index;

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
    using Unique = typename typelist_internal::TSRecurseAppendUniqueImpl<ListT, Ts...>::type;

    /// @brief Append types, or the members of another TypeList, to this list.
    /// @warning Appending nested TypeList<> objects causes them to expand to
    ///          their contained list of types.
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
    using Append = typename typelist_internal::TSAppendImpl<Self, TypesToAppend...>::type;

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
    using Remove = typename typelist_internal::TSRemoveImpl<Self, TypesToRemove...>::type;

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
    using PopFront = typename typelist_internal::TSRemoveFirstImpl<Self>::type;

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
    using PopBack = typename typelist_internal::TSRemoveLastImpl<Self>::type;

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
    using RemoveByIndex = typename typelist_internal::TSRemoveIndicesImpl<Self, First, Last>::type;

    /// @brief Transform each type of this TypeList, rebuiling a new list of
    ///        converted types. This method instantiates a user provided Opt<T> to
    ///        replace each type in the current list.
    /// @warning Transforming types to new TypeList<> objects causes them to expand to
    ///          their contained list of types.
    /// @details Example:
    /// @code
    /// {
    ///     // Templated type decl, where the type T will be subsituted for each type
    ///     // in the TypeList being transformed.
    ///     template <typename T>
    ///     using ConvertedType = typename openvdb::PromoteType<T>::Next;
    ///
    ///     // Results in: openvdb::TypeList<Int64, double>;
    ///     using PromotedType = openvdb::TypeList<Int32, float>::Transform<ConvertedType>;
    /// }
    /// @endcode
    template<template <typename> class OpT>
    using Transform = typename typelist_internal::TSTranformImpl<OpT, Ts...>::type;

    /// @brief Invoke a templated class operator on each type in this list. Use
    ///   this method if you only need access to the type for static methods.
    /// @details Example:
    /// @code
    /// #include <typeinfo>
    ///
    /// template <typename T>
    /// struct PintTypes() {
    ///     inline void operator()() { std::cout << typeid(T).name() << std::endl; }
    /// };
    ///
    /// using MyTypes = openvdb::TypeList<int, float, double>;
    /// MyTypes::foreach<PintTypes>(); // "i, f, d" (exact output is compiler-dependent)
    /// @endcode
    ///
    /// @note OpT must be a templated class. It is created and invoked for each
    ///   type in this list.
    template<template <typename> class OpT>
    static OPENVDB_TYPELIST_FORCE_INLINE void foreach() {
        typelist_internal::TSForEachImpl<OpT, Ts...>();
    }

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
    static OPENVDB_TYPELIST_FORCE_INLINE void foreach(OpT op) {
        typelist_internal::TSForEachImpl<OpT, Ts...>(op);
    }

    template<typename OpT>
    static OPENVDB_TYPELIST_FORCE_INLINE void foreachIndex(OpT op) {
        foreachIndex<OpT, 0, Size>(op);
    }

    template<typename OpT, typename RetT>
    static OPENVDB_TYPELIST_FORCE_INLINE RetT foreachIndex(OpT op, RetT def) {
        return foreachIndex<OpT, RetT, 0, Size>(op, def);
    }

    /// @brief Invoke a templated, unary functor on a provide @c obj of type
    ///        @c BaseT only if said object is an applicable (derived) type
    ///        also contained in the current @c TypeList.
    /// @details  This method loops over every type in the type list and calls
    ///   an interface method on @c obj to check to see if the @c obj is
    ///   interpretable as the given type. If it is, the method static casts
    ///   @c obj to the type, invokes the provided functor with the casted type
    ///   and returns, stopping further list iteration. @c obj is expected to
    ///   supply an interface to validate the type which satisfies the
    ///   prototype:
    /// @code
    ///   template <typename T> bool isType()
    /// @endcode
    ///
    ///   A full example (using dynamic_cast - see Grid/Tree implementations
    ///   for string based comparisons:
    /// @code
    ///   struct Base {
    ///       virtual ~Base() = default;
    ///       template<typename T> bool isType() { return dynamic_cast<const T*>(this); }
    ///   };
    ///   struct MyType1 : public Base { void print() { std::cerr << "MyType1" << std::endl; } };
    ///   struct MyType2 : public Base { void print() { std::cerr << "MyType2" << std::endl; } };
    ///
    ///   using MyTypeList = TypeList<MyType1, MyType2>;
    ///   Base* getObj() { return new MyType2(); }
    ///
    ///   std::unique_ptr<Base> obj = getObj();
    ///   // Returns 'true', prints 'MyType2'
    ///   const bool success =
    ///       MyTypeList::apply([](const auto& type) { type.print(); }, *obj);
    /// @endcode
    ///
    /// @note The functor object is passed by value.  Wrap it with @c std::ref
    ///   pass by reference.
    template<typename OpT, typename BaseT>
    static OPENVDB_TYPELIST_FORCE_INLINE bool apply(OpT op, BaseT& obj) {
        return typelist_internal::TSApplyImpl<OpT, BaseT, Self>::apply(obj, op);
    }
};

/// @brief  A trivial wrapper around a std::tuple but with compatible TypeList
///   methods. Importantly can be instatiated from a TypeList and implements a
///   similar ::foreach interface
/// @warning  Some member methods here run on actual instances of types in the
///   list. As such, it's unlikely that they can always be resolved at compile
///   time (unlike methods in TypeList). Compilers are notriously bad at
///   automatically inlining recursive/nested template instations (without fine
///   tuning inline options to the frontend) so the public API of this class is
///   marked as force inlined. You can disable this behaviour by defining:
///      OPENVDB_TYPELIST_NO_FORCE_INLINE
///   before including this header. Note however that the ValueAccessor uses
///   this API and disabling force inlining can cause significant performance
///   degredation.
template<typename... Ts>
struct TupleList
{
    using AsTypeList = TypeList<Ts...>;
    using TupleT = std::tuple<Ts...>;

    TupleList() = default;
    TupleList(Ts&&... args) : mTuple(std::forward<Ts>(args)...) {}

    constexpr auto size() { return std::tuple_size_v<TupleT>; }
    constexpr TupleT& tuple() { return mTuple; }
    constexpr TupleT& tuple() const { return mTuple; }

    template <size_t Idx> constexpr auto& get() { return std::get<Idx>(mTuple); }
    template <size_t Idx> constexpr auto& get() const { return std::get<Idx>(mTuple); }

    /// @brief  Run a function on each type instance in the underlying
    ///   std::tuple. Effectively calls op(std::get<I>(mTuple)) where
    ///   I = [0,Size). Does not support returning a value.
    ///
    /// @param op  Function to run on each type
    /// @details Example:
    /// @code
    /// {
    ///     using Types = openvdb::TypeList<Int32, float, std::string>;
    /// }
    /// {
    ///     Types::AsTupleList tuple(Int32(1), float(3.3), std::string("foo"));
    ///     tuple.foreach([](auto value) { std::cout << value << ' '; }); // prints '1 3.3 foo'
    /// }
    /// @endcode
    template<typename OpT>
    OPENVDB_TYPELIST_FORCE_INLINE constexpr void foreach(OpT op) {
        typelist_internal::TSForEachImpl<0, AsTypeList::Size>(op, mTuple);
    }

    /// @brief  Run a function on the first element in the underlying
    ///   std::tuple that satisfies the provided predicate. Effectively
    ///   calls op(std::get<I>(mTuple)) when pred(I) returns true, then exits,
    ///   where I = [0,Size). Does not support returning a value.
    /// @note  This is mainly useful to avoid the overhead of calling std::get<I>
    ///   on every element when only a single unknown element needs processing.
    ///
    /// @param pred  Predicate to run on each index, should return true/false
    /// @param op    Function to run on the first element that satisfies pred
    /// @details Example:
    /// @code
    /// {
    ///     using Types = openvdb::TypeList<Int32, float, std::string>;
    /// }
    /// {
    ///     Types::AsTupleList tuple(Int32(1), float(3.3), std::string("foo"));
    ///     bool runtimeFlags[tuple.size()] = { .... } // some runtime flags
    ///     tuple.foreach(
    ///         [&](auto Idx)  { return runtimeFlags[Idx]; },
    ///         [](auto value) { std::cout << value << std::endl; }
    ///      );
    /// }
    /// @endcode
    template<class Pred, class OpT>
    OPENVDB_TYPELIST_FORCE_INLINE void evalFirstPred(Pred pred, OpT op)
    {
        typelist_internal::TSEvalFirstPredImpl
            <Pred, OpT, TupleT, 0, AsTypeList::Size>
                (pred, op, mTuple);
    }

    /// @brief  Run a function on the first element in the underlying
    ///   std::tuple that satisfies the provided predicate. Effectively
    ///   calls op(std::get<I>(mTuple)) when pred(I) returns true, then exits,
    ///   where I = [0,Size). Supports returning a value, but a default return
    ///   value must be provided.
    ///
    /// @param pred  Predicate to run on each index, should return true/false
    /// @param op    Function to run on the first element that satisfies pred
    /// @param def   Default return value
    /// @details Example:
    /// @code
    /// {
    ///     using Types = openvdb::TypeList<Int32, float, std::string>;
    /// }
    /// {
    ///     Types::AsTupleList tuple(Int32(1), float(3.3), std::string("foo"));
    ///     // returns 3
    ///     auto size = tuple.foreach(
    ///         [](auto Idx) { return std::is_same<std::string, Types::template Get<Idx>>::value; },
    ///         [](auto value) { return value.size(); },
    ///         -1
    ///      );
    /// }
    /// @endcode
    template<class Pred, class OpT, typename RetT>
    OPENVDB_TYPELIST_FORCE_INLINE RetT evalFirstPred(Pred pred, OpT op, RetT def)
    {
        return typelist_internal::TSEvalFirstPredImpl
            <Pred, OpT, TupleT, RetT, 0, AsTypeList::Size>
                (pred, op, mTuple, def);
    }

private:
    TupleT mTuple;
};

/// @brief  Specilization of an empty TupleList. Required due to constructor
///   selection.
template<>
struct TupleList<>
{
    using AsTypeList = TypeList<>;
    using TupleT = std::tuple<>;

    TupleList() = default;

    constexpr auto size() { return std::tuple_size_v<TupleT>; }
    inline TupleT& tuple() { return mTuple; }
    inline const TupleT& tuple() const { return mTuple; }

    template <size_t Idx> inline constexpr auto& get() { return std::get<Idx>(mTuple); }
    template <size_t Idx> inline constexpr auto& get() const { return std::get<Idx>(mTuple); }

    template<typename OpT> constexpr void foreach(OpT) {}
    template<class Pred, class OpT> constexpr void evalFirstPred(Pred, OpT) {}
    template<class Pred, class OpT, typename RetT>
    constexpr RetT evalFirstPred(Pred, OpT, RetT def) { return def; }

private:
    TupleT mTuple;
};

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TYPELIST_HAS_BEEN_INCLUDED
