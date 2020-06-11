///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
// of its contributors may be used to endorse or promote products derived
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

/// @file ast/Literals.h
///
/// @authors Nick Avramoussis
///
/// @brief  Basic string to numerical conversion for the parser.
///

#ifndef OPENVDB_AX_AST_LITERALS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_AST_LITERALS_HAS_BEEN_INCLUDED

#include <openvdb_ax/version.h>

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

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
