// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
///
/// @author Nick Avramoussis
///
/// @file Simd.h
///
/// @brief Main header to bring in and access wrapped SIMD types. To future
///   proof the underlying SIMD wrapper in use, and to provide extended
///   functionality (such as traits and matching tuple/scalar API for writing
///   generic tools), manual SIMD usage should be restricted to the
///   openvdb::simd namespace.
///
/// @note OpenVDB currently supports SIMD usage of intel x86 intrinsics via
///   Agner Fog's VCL wrapper library. If not in use, SIMD usage relies on
///   autovectorization of the aliased Tuple containers.
///

#ifndef OPENVDB_SIMD_SIMD_HAS_BEEN_INCLUDED
#define OPENVDB_SIMD_SIMD_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Platform.h>
#include <openvdb/math/Tuple.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/HalfDecl.h>

#if defined(OPENVDB_USE_VCL)
#if defined(INSTRSET) && INSTRSET != OPENVDB_X86_INSTRSET
    // If we're here, then the value of INSTRSET does not match what OpenVDB
    // was configured with. This can only happen if VCL is included before
    // simd/Simd.h and detects different x86 compiler flags than what
    // OPENVDB_X86_INSTRSET was configured for. It's not clear what to do in
    // this situation as, technically, you can instrument openvdb libs with
    // ISA A and recompile downstream with ISA B, so long as both are valid for
    // the target architecture.
    //
    // For now we simply use the value of INSTRSET for VCL's selection of
    // intrinsics, but we continue to use the value of OPENVDB_X86_INSTRSET
    // for VDB's native vector size selection. Use a #warning directive which
    // will error with -Werror but can ultimately be suppressed (pragmas won't
    // show up with isystem etc).
    #warning "OpenVDB: Mismatching requested ISA's detected during downstream compilation."
#else
    /// Tell VCL what instruction set to use.
    #define INSTRSET OPENVDB_X86_INSTRSET
#endif
#include <openvdb/ext/vcl/vectorclass.h>
#include <openvdb/ext/vcl/vectorfp16.h>
#else
/// No VCL, just import SIMD Intrinsic Headers
#if OPENVDB_X86_INSTRSET > 0
    #if defined(_WIN32)
        #include <intrin.h>
    #elif defined(__GNUC__)
        #if defined(__x86_64__) || defined(__i386__)
            #include <x86intrin.h>
        #elif defined(__ARM_NEON__)
            #include <arm_neon.h>
        #endif
    #endif
#endif
#endif // OPENVDB_USE_VCL

#include <climits> // CHAR_BIT

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace simd {

/// Simd type selection macro, aliasing between vcl and Tuple types
#ifdef OPENVDB_USE_VCL
#define OPENVDB_SELECT_SIMD_T(A, ...) A
#else
#define OPENVDB_SELECT_SIMD_T(A, ...) __VA_ARGS__
#endif

/// Cofigure the default bit size for our simd vectors based on the configured
/// X86 instruction set. If no specific x86 set has been specified, then we
/// fallback to the assumption of 128 registers. Note that this decision only
/// impacts code using the openvdb::simd API and, if VCL is OFF, simply
/// configures the default (native) Tuple types to fit into this default size.
#if   OPENVDB_X86_INSTRSET >= 9
static inline constexpr size_t OPENVDB_DEFAULT_VECTOR_SIZE = 512; // AVX512
#elif OPENVDB_X86_INSTRSET >= 7
static inline constexpr size_t OPENVDB_DEFAULT_VECTOR_SIZE = 256; // AVX
#else
static inline constexpr size_t OPENVDB_DEFAULT_VECTOR_SIZE = 128; // fallback
#endif

static_assert(OPENVDB_DEFAULT_VECTOR_SIZE >= 128 &&
    (OPENVDB_DEFAULT_VECTOR_SIZE % 128 == 0));

/// @brief  Type selection based on if VCL is enabled. If so, all simd::Vec
///   types alias to representative VCL classes. Otherwise, types are
///   represented by contiguous tuples with elements of matching precision

// Compact boolean vectors
#if defined(OPENVDB_USE_VCL) && INSTRSET >= 10
// These types only exist with AVX512
using Vec2b   = OPENVDB_VCL_NAMESPACE::Vec2b;
using Vec4b   = OPENVDB_VCL_NAMESPACE::Vec4b;
using Vec32b  = OPENVDB_VCL_NAMESPACE::Vec32b;
using Vec64b  = OPENVDB_VCL_NAMESPACE::Vec64b;
#else
// We have to define these as these types are used as broad masks for boolean
// logic when VCL is not in use.
using Vec2b   = math::Tuple<2, bool>;
using Vec4b   = math::Tuple<4, bool>;
using Vec32b  = math::Tuple<32, bool>;
using Vec64b  = math::Tuple<64, bool>;
#endif

using Vec8b   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8b,   math::Tuple<8, bool>);
using Vec16b  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16b,  math::Tuple<16, bool>);
/*
Don't expose internal bit types for now
using Vec128b = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec128b, math::Tuple<128, bool>);
using Vec256b = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec256b, math::Tuple<256, bool>);
using Vec512b = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec512b, math::Tuple<512, bool>);
*/

//  Broad boolean vectors
using Vec16cb = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16cb, math::Tuple<16, bool>);
using Vec16fb = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16fb, math::Tuple<16, bool>);
using Vec16ib = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16ib, math::Tuple<16, bool>);
using Vec16sb = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16sb, math::Tuple<16, bool>);
using Vec2db  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec2db, math::Tuple<2, bool>);
using Vec2qb  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec2qb, math::Tuple<2, bool>);
using Vec32cb = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec32cb, math::Tuple<32, bool>);
using Vec32sb = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec32sb, math::Tuple<32, bool>);
using Vec4db  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec4db, math::Tuple<4, bool>);
using Vec4fb  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec4fb, math::Tuple<4, bool>);
using Vec4ib  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec4ib, math::Tuple<4, bool>);
using Vec4qb  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec4qb, math::Tuple<4, bool>);
using Vec64cb = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec64cb, math::Tuple<64, bool>);
using Vec8db  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8db, math::Tuple<8, bool>);
using Vec8fb  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8fb, math::Tuple<8, bool>);
using Vec8ib  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8ib, math::Tuple<8, bool>);
using Vec8qb  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8qb, math::Tuple<8, bool>);
using Vec8sb  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8sb, math::Tuple<8, bool>);

// Signed integer vectors
using Vec16c  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16c, math::Tuple<16, int8_t>);
using Vec32c  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec32c, math::Tuple<32, int8_t>);
using Vec64c  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec64c, math::Tuple<64, int8_t>);
using Vec8s   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8s,  math::Tuple<8, int16_t>);
using Vec16s  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16s, math::Tuple<16, int16_t>);
using Vec32s  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec32s, math::Tuple<32, int16_t>);
using Vec4i   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec4i,  math::Tuple<4, int32_t>);
using Vec8i   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8i,  math::Tuple<8, int32_t>);
using Vec16i  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16i, math::Tuple<16, int32_t>);
using Vec2q   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec2q,  math::Tuple<2, int64_t>);
using Vec4q   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec4q,  math::Tuple<4, int64_t>);
using Vec8q   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8q,  math::Tuple<8, int64_t>);

// Unsigned integer vectors
using Vec16uc = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16uc, math::Tuple<16, uint8_t>);
using Vec32uc = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec32uc, math::Tuple<32, uint8_t>);
using Vec64uc = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec64uc, math::Tuple<64, uint8_t>);
using Vec8us  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8us,  math::Tuple<8, uint16_t>);
using Vec16us = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16us, math::Tuple<16, uint16_t>);
using Vec32us = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec32us, math::Tuple<32, uint16_t>);
using Vec4ui  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec4ui,  math::Tuple<4, uint32_t>);
using Vec8ui  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8ui,  math::Tuple<8, uint32_t>);
using Vec16ui = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16ui, math::Tuple<16, uint32_t>);
using Vec2uq  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec2uq,  math::Tuple<2, uint64_t>);
using Vec4uq  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec4uq,  math::Tuple<4, uint64_t>);
using Vec8uq  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8uq,  math::Tuple<8, uint64_t>);

/*
// half precision vectors
// @TODO  Figure out nice type mapping with OPENVDB_VCL_NAMESPACE::Float16 and math::half
using Vec8h   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8h,  math::Tuple<8, math::half>);
using Vec16h  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16h, math::Tuple<16, math::half>);
using Vec32h  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec32h, math::Tuple<32, math::half>);
*/

// float precision vectors
using Vec4f   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec4f,  math::Tuple<4, float>);
using Vec8f   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8f,  math::Tuple<8, float>);
using Vec16f  = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec16f, math::Tuple<16, float>);

// double precision vectors
using Vec2d   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec2d,  math::Tuple<2, double>);
using Vec4d   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec4d,  math::Tuple<4, double>);
using Vec8d   = OPENVDB_SELECT_SIMD_T(OPENVDB_VCL_NAMESPACE::Vec8d,  math::Tuple<8, double>);

#undef OPENVDB_SELECT_SIMD_T

/// @brief  Various traits for SIMD types. Note that, if VCL is disabled, these
///   will simply represent canonical tuple types
template <typename T, size_t S> struct SimdT;

#if !defined(OPENVDB_USE_VCL) || INSTRSET >= 10
template <> struct SimdT<bool, 2>    { using Type = simd::Vec2b; };
template <> struct SimdT<bool, 4>    { using Type = simd::Vec4b; };
template <> struct SimdT<bool, 32>   { using Type = simd::Vec32b; };
template <> struct SimdT<bool, 64>   { using Type = simd::Vec64b; };
#endif

template <> struct SimdT<bool, 8>    { using Type = simd::Vec8b; };
template <> struct SimdT<bool, 16>   { using Type = simd::Vec16b; };
/*
template <> struct SimdT<bool, 128>  { using Type = simd::Vec128b; };
template <> struct SimdT<bool, 256>  { using Type = simd::Vec256b; };
template <> struct SimdT<bool, 512>  { using Type = simd::Vec512b; };
*/
template <> struct SimdT<int8_t, 16>  { using Type = simd::Vec16c; };
template <> struct SimdT<int8_t, 32>  { using Type = simd::Vec32c; };
template <> struct SimdT<int8_t, 64>  { using Type = simd::Vec64c; };
template <> struct SimdT<int16_t, 8>  { using Type = simd::Vec8s; };
template <> struct SimdT<int16_t, 16> { using Type = simd::Vec16s; };
template <> struct SimdT<int16_t, 32> { using Type = simd::Vec32s; };
template <> struct SimdT<int32_t, 4>  { using Type = simd::Vec4i; };
template <> struct SimdT<int32_t, 8>  { using Type = simd::Vec8i; };
template <> struct SimdT<int32_t, 16> { using Type = simd::Vec16i; };
template <> struct SimdT<int64_t, 2>  { using Type = simd::Vec2q; };
template <> struct SimdT<int64_t, 4>  { using Type = simd::Vec4q; };
template <> struct SimdT<int64_t, 8>  { using Type = simd::Vec8q; };

template <> struct SimdT<uint8_t, 16>  { using Type = simd::Vec16uc; };
template <> struct SimdT<uint8_t, 32>  { using Type = simd::Vec32uc; };
template <> struct SimdT<uint8_t, 64>  { using Type = simd::Vec64uc; };
template <> struct SimdT<uint16_t, 8>  { using Type = simd::Vec8us; };
template <> struct SimdT<uint16_t, 16> { using Type = simd::Vec16us; };
template <> struct SimdT<uint16_t, 32> { using Type = simd::Vec32us; };
template <> struct SimdT<uint32_t, 4>  { using Type = simd::Vec4ui; };
template <> struct SimdT<uint32_t, 8>  { using Type = simd::Vec8ui; };
template <> struct SimdT<uint32_t, 16> { using Type = simd::Vec16ui; };
template <> struct SimdT<uint64_t, 2>  { using Type = simd::Vec2uq; };
template <> struct SimdT<uint64_t, 4>  { using Type = simd::Vec4uq; };
template <> struct SimdT<uint64_t, 8>  { using Type = simd::Vec8uq; };

/*
template <> struct SimdT<math::half, 8>  { using Type = simd::Vec8h; };
template <> struct SimdT<math::half, 16> { using Type = simd::Vec16h; };
template <> struct SimdT<math::half, 32> { using Type = simd::Vec32h; };
*/

template <> struct SimdT<float, 4>  { using Type = simd::Vec4f; };
template <> struct SimdT<float, 8>  { using Type = simd::Vec8f; };
template <> struct SimdT<float, 16> { using Type = simd::Vec16f; };

template <> struct SimdT<double, 2> { using Type = simd::Vec2d; };
template <> struct SimdT<double, 4> { using Type = simd::Vec4d; };
template <> struct SimdT<double, 8> { using Type = simd::Vec8d; };

/// @brief  Native type selection. Given a compatible POD type for SIMD
///   intrinsics, select the best (in this case largest) SIMD container
///   type.
template <typename T>
using SimdNativeT = SimdT<T, OPENVDB_DEFAULT_VECTOR_SIZE/(sizeof(T)*CHAR_BIT)>;

/// @brief  Native type selection for compatible POD types when there
///   may not be an available Simd type. For example:
///
/// @code
///    // compiled WITHOUT -msse2 (or higher)
///    static_assert(std::is_same_v<NativeSimdOrScalar<double>::Type, double>);
///    // default behaviour with __m128
///    static_assert(std::is_same_v<NativeSimdOrScalar<double>::Type, SimdT<double, 2>>);
///    // compiled WITH -mavx
///    static_assert(std::is_same_v<NativeSimdOrScalar<double>::Type, SimdT<double, 4>>);
///@endcode
template <typename T, typename = void>
struct NativeSimdOrScalar
{
    using Type = T;
    static constexpr size_t Size = 1;
};

template <typename T>
struct NativeSimdOrScalar<T, std::void_t<typename SimdNativeT<T>::Type>>
{
    using Type = typename SimdNativeT<T>::Type;
#ifdef OPENVDB_USE_VCL
    static constexpr size_t Size = Type::size();
#else
    static constexpr size_t Size = Type::size;
#endif
};

/// @{
/// @brief  Traits for determining if a type is a SIMD type
template <typename T> struct IsSimdBroadMaskT : std::false_type {};
template <typename T> struct IsSimdCompactMaskT : std::false_type {};
template <typename T> struct IsSimdIntT : std::false_type {};
template <typename T> struct IsSimdFloatT : std::false_type {};
/// }@

// compact bool vectors
#if !defined(OPENVDB_USE_VCL) || INSTRSET >= 10
template <> struct IsSimdCompactMaskT<simd::Vec2b> : std::true_type {};
template <> struct IsSimdCompactMaskT<simd::Vec4b> : std::true_type {};
template <> struct IsSimdCompactMaskT<simd::Vec32b> : std::true_type {};
template <> struct IsSimdCompactMaskT<simd::Vec64b> : std::true_type {};
#endif
template <> struct IsSimdCompactMaskT<simd::Vec8b> : std::true_type {};
template <> struct IsSimdCompactMaskT<simd::Vec16b> : std::true_type {};
/*
template <> struct IsSimdCompactMaskT<simd::Vec128b> : std::true_type {};
template <> struct IsSimdCompactMaskT<simd::Vec256b> : std::true_type {};
template <> struct IsSimdCompactMaskT<simd::Vec512b> : std::true_type {};
*/

// broad bool masks
#ifdef OPENVDB_USE_VCL // Duplicate specializations when VCL is NOT in use
#if INSTRSET < 9 // These alias to corresponding VecNb containers with AVX512f
template <> struct IsSimdBroadMaskT<simd::Vec16fb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec8db> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec8qb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec16ib> : std::true_type {};
#endif
#if INSTRSET < 10  // These alias to corresponding VecNb containers with AVX512+
template <> struct IsSimdBroadMaskT<simd::Vec16cb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec16sb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec2db> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec2qb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec32cb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec4fb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec4db> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec4ib> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec64cb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec4qb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec8fb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec8ib> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec8sb> : std::true_type {};
template <> struct IsSimdBroadMaskT<simd::Vec32sb> : std::true_type {};
#endif
#endif

// 8-bit signed integer vectors
template <> struct IsSimdIntT<simd::Vec16c> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec32c> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec64c> : std::true_type {};
// 16-bit signed integer vectors
template <> struct IsSimdIntT<simd::Vec8s> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec16s> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec32s> : std::true_type {};
// 32-bit signed integer vectors
template <> struct IsSimdIntT<simd::Vec4i> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec8i> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec16i> : std::true_type {};
// 64-bit signed integer vectors
template <> struct IsSimdIntT<simd::Vec2q> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec4q> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec8q> : std::true_type {};

// 8-bit unsigned integer vectors
template <> struct IsSimdIntT<simd::Vec16uc> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec32uc> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec64uc> : std::true_type {};
// 16-bit unsigned integer vectors
template <> struct IsSimdIntT<simd::Vec8us> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec16us> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec32us> : std::true_type {};
// 32-bit unsigned integer vectors
template <> struct IsSimdIntT<simd::Vec4ui> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec8ui> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec16ui> : std::true_type {};
// 64-bit unsigned integer vectors
template <> struct IsSimdIntT<simd::Vec2uq> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec4uq> : std::true_type {};
template <> struct IsSimdIntT<simd::Vec8uq> : std::true_type {};

/*
// 16-bit float vectors
template <> struct IsSimdFloatT<simd::Vec8h> : std::true_type {};
template <> struct IsSimdFloatT<simd::Vec16h> : std::true_type {};
template <> struct IsSimdFloatT<simd::Vec32h> : std::true_type {};
*/
// 32-bit float vectors
template <> struct IsSimdFloatT<simd::Vec4f> : std::true_type {};
template <> struct IsSimdFloatT<simd::Vec8f> : std::true_type {};
template <> struct IsSimdFloatT<simd::Vec16f> : std::true_type {};
// 64-bit double vectors
template <> struct IsSimdFloatT<simd::Vec2d> : std::true_type {};
template <> struct IsSimdFloatT<simd::Vec4d> : std::true_type {};
template <> struct IsSimdFloatT<simd::Vec8d> : std::true_type {};

/// @brief  Alias for determining if a type is a "compatible SIMD type". This
///   definition is satisfied for VCL types, when VCL is enabled, *OR* Tuple
///   types, when VCL is DISABLED, allowing code to be written for containers
///   of multiple scalars, regardless whether VCL is enabled or not.
template <typename T> struct IsSimdT :
    std::conditional<(IsSimdFloatT<T>::value || IsSimdCompactMaskT<T>::value || IsSimdBroadMaskT<T>::value || IsSimdIntT<T>::value),
        std::true_type,
        std::false_type>::type {};

template<typename T> struct IsTupleT : std::false_type {};
template<int N, typename T> struct IsTupleT<math::Tuple<N, T>> : std::true_type {};

/// @brief SFINAE enable_if template defines for customizing the acceptable
///   arguments for an agnostic container type.
#define OPENVDB_ENABLE_IF_ARITHMETIC \
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
#define OPENVDB_ENABLE_IF_TUPLE \
    template<typename T, typename std::enable_if<IsTupleT<T>::value>::type* = nullptr>
#define OPENVDB_ENABLE_IF_TUPLE_MASK \
    template<typename T, typename std::enable_if<IsTupleT<T>::value && (IsSimdCompactMaskT<T>::value || IsSimdBroadMaskT<T>::value)>::type* = nullptr>
#define OPENVDB_ENABLE_IF_VCL \
    template<typename T, typename std::enable_if<IsSimdT<T>::value && !IsTupleT<T>::value>::type* = nullptr>
#define OPENVDB_ENABLE_IF_VCL_MASK \
    template<typename T, typename std::enable_if<(IsSimdCompactMaskT<T>::value || IsSimdBroadMaskT<T>::value) && !IsTupleT<T>::value>::type* = nullptr>

#ifdef OPENVDB_USE_VCL

/// @cond OPENVDB_DOCS_INTERNAL
namespace simd_internal
{
template <int> struct elem;
//template <> struct elem<1>  { using Type = bool; }; // bits (internal base class)
template <> struct elem<2>  { using Type = bool; }; // compact
template <> struct elem<3>  { using Type = bool; }; // broad
template <> struct elem<4>  { using Type = int8_t; };
template <> struct elem<5>  { using Type = uint8_t; };
template <> struct elem<6>  { using Type = int16_t; };
template <> struct elem<7>  { using Type = uint16_t; };
template <> struct elem<8>  { using Type = int32_t; };
template <> struct elem<9>  { using Type = uint32_t; };
template <> struct elem<10> { using Type = int64_t; };
template <> struct elem<11> { using Type = uint64_t; };
//template <> struct elem<15> { using Type = math::half; };
template <> struct elem<16> { using Type = float; };
template <> struct elem<17> { using Type = double; };
}
/// @endcond

/// @{
/// @brief Simd Traits. For a compatible simd type, expose various type
///   information, including its size, element type and equivalent mask type.
///   Also provides a converter to other simd types of the same size with a
///   different value type (although these types are not guaranteed to exist).
/// @note  These traits do not work with compact masks, they are only designed
///   to work with integer, float, or broad mask vectors.
template <typename T>
struct SimdTraits
{
    static constexpr size_t size = T::size();
    using ElementT = typename simd_internal::elem<T::elementtype()>::Type;
    using MaskT = decltype(std::declval<T>() == std::declval<T>());
    template <typename S> using ConvertT = typename SimdT<S, size>::Type;
};
#else
template <typename T>
struct SimdTraits
{
    static constexpr size_t size = T::size;
    using ElementT = typename T::ValueType;
    using MaskT = decltype(std::declval<T>() == std::declval<T>());
    template <typename S> using ConvertT = typename SimdT<S, size>::Type;
};
/// }@
#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// The follow implements generic APIs for VCL, Tuple and arithmetic POD types.
/// The idea here is that tools can be written for a templated type, T, whose
/// interface is implemented via the below methods. For example, the below
/// will work for all supported OPENVDB_VCL_NAMESPACE, openvdb::math::Tuple and
/// arithmetic POD types:
///
/// template <typename T>
/// T Lerp(const T a, const T b, const T t) {
///   using namespace openvdb::simd;
///   return ((T(1) - t) * a) + (t * b);
/// }

//// Vector API

#ifdef OPENVDB_USE_VCL

/// @{
/// @brief  Load method for VCL types
template <size_t N, typename T> inline auto load(const T* a)
{
    typename SimdT<T, N>::Type r; r.load(a); return r;
}
/// @}

/// @{
/// @brief  Various math for VCL types. These should be extended when necessary
OPENVDB_ENABLE_IF_VCL inline T min(const T& a, const T& b) { return OPENVDB_VCL_NAMESPACE::min(a, b); }
OPENVDB_ENABLE_IF_VCL inline T max(const T& a, const T& b) { return OPENVDB_VCL_NAMESPACE::max(a, b); }
OPENVDB_ENABLE_IF_VCL inline T abs(const T& a) { return OPENVDB_VCL_NAMESPACE::abs(a); }
OPENVDB_ENABLE_IF_VCL inline T sqrt(const T& a) { return OPENVDB_VCL_NAMESPACE::sqrt(a); }
OPENVDB_ENABLE_IF_VCL inline T square(const T& a) { return a * a; }
OPENVDB_ENABLE_IF_VCL inline T pow2(const T& a) { return a * a; }
OPENVDB_ENABLE_IF_VCL inline T pow3(const T& a) { return a * a * a; }
template <int N, typename T, typename std::enable_if<IsSimdT<T>::value>::type* = nullptr>
inline auto pow(const T& a) { return OPENVDB_VCL_NAMESPACE::pow_n<T, N>(a); }
/// @}

/// @brief  Select between `a` and `b`, based on the equivalently sized VCL
///   bool mask, and return the blended VCL type.
OPENVDB_ENABLE_IF_VCL inline T select(const typename SimdTraits<T>::MaskT& m, const T& a, const T& b)
{
    return OPENVDB_VCL_NAMESPACE::select(m, a, b);
}

/// @brief  Returns an equivalently sized VCL bool type, where each lane is
///   true if the corresponding lane was finite, or false otherwise
OPENVDB_ENABLE_IF_VCL inline auto is_finite(const T& a) { return OPENVDB_VCL_NAMESPACE::is_finite(a); }

/// @{
/// @brief  Horizontal reductions of VCL types
OPENVDB_ENABLE_IF_VCL inline auto horizontal_min(const T& a) { return OPENVDB_VCL_NAMESPACE::horizontal_min(a); }
OPENVDB_ENABLE_IF_VCL inline auto horizontal_max(const T& a) { return OPENVDB_VCL_NAMESPACE::horizontal_max(a); }
OPENVDB_ENABLE_IF_VCL inline auto horizontal_add(const T& a) { return OPENVDB_VCL_NAMESPACE::horizontal_add(a); }
OPENVDB_ENABLE_IF_VCL_MASK inline auto horizontal_and(const T& a) { return OPENVDB_VCL_NAMESPACE::horizontal_and(a); }
OPENVDB_ENABLE_IF_VCL_MASK inline auto horizontal_or(const T& a) { return OPENVDB_VCL_NAMESPACE::horizontal_or(a); }
OPENVDB_ENABLE_IF_VCL_MASK inline int horizontal_count(const T& a) { return OPENVDB_VCL_NAMESPACE::horizontal_count(a); }
OPENVDB_ENABLE_IF_VCL_MASK inline int horizontal_find_first(const T& a) { return OPENVDB_VCL_NAMESPACE::horizontal_find_first(a); }
/// @}

#else

/// Tuple API

/// @cond OPENVDB_DOCS_INTERNAL
namespace simd_internal
{

struct MinOp {
    template <typename T>
    constexpr T operator()(const T& a, const T& b) const { return std::min(a, b); }
};

struct MaxOp {
    template <typename T>
    constexpr T operator()(const T& a, const T& b) const { return std::max(a, b); }
};

struct AbsOp {
    template <typename T>
    constexpr T operator()(const T& a) const { return T(std::abs(a)); }
};

struct SqrtOp {
    template <typename T>
    constexpr T operator()(const T& a) const { return std::sqrt(a); }
};

struct IsFiniteOp {
    template <typename T>
    constexpr bool operator()(const T& a) const { return std::isfinite(a); }
};

template <typename UnaryOpT, typename T>
inline auto unaryop(const T& a, const UnaryOpT& op = UnaryOpT{})
{
    using ElementT = typename T::ValueType;
    using RetElementT = std::invoke_result_t<UnaryOpT, ElementT>;
    using RetTupleT = math::Tuple<T::size, RetElementT>;
    RetTupleT r; for (int i = 0; i < T::size; ++i) r[i] = op(a[i]); return r;
}

template <typename BinaryOpT, typename T>
inline auto binop(const T& a, const T& b, const BinaryOpT& op = BinaryOpT{})
{
    using ElementT = typename T::ValueType;
    using RetElementT = std::invoke_result_t<BinaryOpT, ElementT, ElementT>;
    using RetTupleT = math::Tuple<T::size, RetElementT>;
    RetTupleT r; for (int i = 0; i < T::size; ++i) r[i] = op(a[i], b[i]); return r;
}

template <typename BinaryOpT, typename T>
inline auto hbinop(const T& a, const BinaryOpT& op = BinaryOpT{})
{
    using ElementT = typename T::ValueType;
    using RetElementT = std::invoke_result_t<BinaryOpT, ElementT, ElementT>;
    RetElementT r = a[0]; for (int i = 1; i < T::size; ++i) r = op(r, a[i]); return r;
}

} // namespace simd_internal

/// @endcond

/// @{
/// @brief  Equivalent load method for Tuples
template <size_t N, typename T>
inline typename SimdT<T, N>::Type load(const T* a)
{
    typename SimdT<T, N>::Type r;
    std::memcpy(r.asV(), a, N*sizeof(T));
    return r;
}
/// @}

/// @{
/// @brief  Equivalent methods for Tuples
OPENVDB_ENABLE_IF_TUPLE inline auto min(const T& a, const T& b) { return simd_internal::binop<simd_internal::MinOp>(a, b); }
OPENVDB_ENABLE_IF_TUPLE inline auto max(const T& a, const T& b) { return simd_internal::binop<simd_internal::MaxOp>(a, b); }
OPENVDB_ENABLE_IF_TUPLE inline auto abs(const T& a) { return simd_internal::unaryop<simd_internal::AbsOp>(a); }
OPENVDB_ENABLE_IF_TUPLE inline auto sqrt(const T& a) { return simd_internal::unaryop<simd_internal::SqrtOp>(a); }
OPENVDB_ENABLE_IF_TUPLE inline auto square(const T& a) { return a * a; }
OPENVDB_ENABLE_IF_TUPLE inline auto pow2(const T& a) { return a * a; }
OPENVDB_ENABLE_IF_TUPLE inline auto pow3(const T& a) { return a * a * a; }
template <typename T, typename ExpT, typename std::enable_if<IsTupleT<T>::value>::type* = nullptr>
inline auto pow(const T& a, const ExpT n)
{
    T r; for (int i = 0; i < T::size; ++i) { r[i] = math::Pow(a[i],n); } return r;
}

OPENVDB_ENABLE_IF_TUPLE inline T select(const typename SimdTraits<T>::MaskT& m, const T& a, const T& b)
{
    T r; for (int i = 0; i < T::size; ++i) { r[i] = m[i] ? a[i] : b[i]; } return r;
}

OPENVDB_ENABLE_IF_TUPLE inline auto is_finite(const T& a) { return simd_internal::unaryop<simd_internal::IsFiniteOp>(a); }
OPENVDB_ENABLE_IF_TUPLE inline auto horizontal_min(const T& a) { return simd_internal::hbinop<simd_internal::MinOp>(a); }
OPENVDB_ENABLE_IF_TUPLE inline auto horizontal_max(const T& a) { return simd_internal::hbinop<simd_internal::MaxOp>(a); }
OPENVDB_ENABLE_IF_TUPLE_MASK inline auto horizontal_and(const T& a) { return simd_internal::hbinop<std::logical_and<>>(a); }
OPENVDB_ENABLE_IF_TUPLE_MASK inline auto horizontal_or(const T& a)  { return simd_internal::hbinop<std::logical_or<>>(a); }
OPENVDB_ENABLE_IF_TUPLE_MASK inline int horizontal_count(const T& a)
{
    int count = 0;
    for (int i = 0; i < T::size; ++i) { if (a[i]) ++count; }
    return count;
}

OPENVDB_ENABLE_IF_TUPLE_MASK inline int horizontal_find_first(const T& a)
{
    for (int i = 0; i < T::size; ++i) { if (a[i]) return i; }
    return -1;
}

OPENVDB_ENABLE_IF_TUPLE inline auto horizontal_add(const T& a)
{
    static_assert((T::size % 2) == 0);
    using ValueType = typename T::ValueType;
    ValueType r(0);
    for (int i = 0; i < T::size; i+=2) {
        r += ValueType(a[i] + a[i+1]);
    }
    return r;
}
/// @}

#endif

//// Scalar API
OPENVDB_ENABLE_IF_ARITHMETIC inline auto load(const T a) { return a; }
OPENVDB_ENABLE_IF_ARITHMETIC inline T min(const T& a, const T& b) { return std::min(a, b); }
OPENVDB_ENABLE_IF_ARITHMETIC inline T max(const T& a, const T& b) { return std::max(a, b); }
OPENVDB_ENABLE_IF_ARITHMETIC inline auto sqrt(const T& a) { return std::sqrt(a); }
OPENVDB_ENABLE_IF_ARITHMETIC inline T square(const T& a) { return math::Pow2(a); }
OPENVDB_ENABLE_IF_ARITHMETIC inline T pow2(const T& a) { return math::Pow2(a); }
OPENVDB_ENABLE_IF_ARITHMETIC inline T pow3(const T& a) { return math::Pow3(a); }

OPENVDB_ENABLE_IF_ARITHMETIC inline T select(const bool m, const T& a, const T& b) { return m ? a : b; }

OPENVDB_ENABLE_IF_ARITHMETIC inline bool is_finite(const T& a) { return std::isfinite(a); }
OPENVDB_ENABLE_IF_ARITHMETIC inline T horizontal_max(const T& a) { return a; }
OPENVDB_ENABLE_IF_ARITHMETIC inline T horizontal_min(const T& a) { return a; }
OPENVDB_ENABLE_IF_ARITHMETIC inline T horizontal_add(const T& a) { return a; }
inline bool horizontal_and(const bool a) { return a; }
inline bool horizontal_or(const bool a) { return a; }
inline int horizontal_find_first(const bool a) { return a ? 0 : -1; }
inline int horizontal_count(const bool a) { return a ? 1 : 0; }

#undef OPENVDB_ENABLE_IF_ARITHMETIC
#undef OPENVDB_ENABLE_IF_TUPLE
#undef OPENVDB_ENABLE_IF_VCL

} // namespace simd
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#if defined(OPENVDB_USE_VCL)
namespace OPENVDB_VCL_NAMESPACE {
/// @brief  ostream operator for VCL types
template<typename T, typename std::enable_if<openvdb::simd::IsSimdT<T>::value &&
    !openvdb::simd::IsTupleT<T>::value>::type* = nullptr>
inline std::ostream& operator<<(std::ostream& os, const T& v)
{
    os << "[";
    for (unsigned j(0); j < T::size(); j++) {
        if (j) os << ", ";
        os << openvdb::math::PrintCast(v[j]);
    }
    os << "]";
    return os;
}
} // OPENVDB_VCL_NAMESPACE
#endif // OPENVDB_USE_VCL

#endif // OPENVDB_SIMD_SIMD_HAS_BEEN_INCLUDED
