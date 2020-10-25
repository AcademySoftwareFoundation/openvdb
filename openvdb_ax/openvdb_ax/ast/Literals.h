// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ast/Literals.h
///
/// @authors Nick Avramoussis
///
/// @brief  Basic string to numerical conversion for the parser.
///

#ifndef OPENVDB_AX_AST_LITERALS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_AST_LITERALS_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <limits>
#include <string>
#include <type_traits>
#include <memory>
#include <cassert>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

/// @brief  The backend representation of strings in AX. This is also how
///         strings are passed from the AX code generation to functions.
struct AXString
{
    // usually size_t. Used to match the implementation of std:string
    using SizeType = std::allocator<char>::size_type;
    const char* ptr = nullptr;
    SizeType size = 0;
};

/// @brief  Literal language limits and overflow behaviour

/// @brief Struct which manages numeric limits and overflow behaviour for a target numeric type T
/// @note  Used by the AST value node for handling numeric overflows
template <typename T, typename T2=void>
struct LiteralLimits
{
    using Type = T;
    using ContainerT = T;
};

template <typename T>
struct LiteralLimits<T,
    typename std::enable_if<std::is_integral<T>::value>::type>
{
    using Type = T;
    using ContainerT = uint64_t;
    /// @brief  The maximum value which can be converted from a string before
    ///         being unable to be represented as an integral
    inline static ContainerT limit() { return std::numeric_limits<ContainerT>::max(); }
    /// @brief  The value which is used if the string representation cannot be cast to
    ///         a valid integral. Note that this is NOT the signed integer wrap behaviour.
    inline static ContainerT onLimitOverflow() { return limit(); }
    /// @brief  The language conversion method for converting a string to an integer
    /// @param  number String holding a number to be converted
    /// @note   See ast::Value<Integral> for exception handling
    inline static ContainerT convert(const std::string& number) {
        assert(number.empty() || number[0] != '-');
        return std::stoull(number);
    }
};

template <typename T>
struct LiteralLimits<T,
    typename std::enable_if<std::is_floating_point<T>::value>::type>
{
    using Type = T;
    using ContainerT = double;

    /// @brief  The maximum value which can be converted from a string before
    ///         being unable to be represented as an floating point value
    inline static ContainerT limit() { return std::numeric_limits<ContainerT>::max(); }
    /// @brief  The value which is used if the string representation cannot be held in
    ///         a value defined by the above limit
    inline static ContainerT onLimitOverflow() { return std::numeric_limits<ContainerT>::infinity(); }
    /// @brief  The language conversion method for converting a string to an integer
    /// @param  number String holding a number to be converted
    /// @note   See ast::Value<Integral> for exception handling
    inline static ContainerT convert(const std::string& number) {
        assert(number.empty() || number[0] != '-');
        return std::stod(number);
    }
};

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_AST_LITERALS_HAS_BEEN_INCLUDED

