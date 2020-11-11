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

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @cond OPENVDB_TYPES_INTERNAL

template<typename... Ts> struct TypeList; // forward declaration

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
    using Unique = typename typelist_internal::TSMakeUniqueImpl<ListT, Ts...>::type;

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
    static void foreach(OpT op) { typelist_internal::TSForEachImpl<OpT, Ts...>(op); }
};


} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TYPELIST_HAS_BEEN_INCLUDED
