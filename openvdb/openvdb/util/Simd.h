// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_UTIL_SIMD_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_SIMD_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/math/Tuple.h>
#include <openvdb/math/Math.h>

#ifdef OPENVDB_USE_VCL
#include <openvdb/ext/vcl/vectorclass.h>
#endif // OPENVDB_USE_VCL

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {
namespace simd {

#ifdef OPENVDB_USE_VCL
#define OPENVDB_SELECT_SIMD_T(A, ...) A
static inline constexpr size_t MAX_REGISTER_SIZE =
    (INSTRSET < 7 ? 128 : // no AVX, SSE __m128 registers
        (INSTRSET < 9 ? 256 : // AVX  __m256 registers
            (INSTRSET > 9 ? 512 : 0))); // AVX  __m512 registers, else 128
static_assert(MAX_REGISTER_SIZE != 0 && (MAX_REGISTER_SIZE % 128 == 0));
#else
#define OPENVDB_SELECT_SIMD_T(A, ...) __VA_ARGS__
static inline constexpr size_t MAX_REGISTER_SIZE = 128;
#endif

using Vec8b   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8b,   math::Tuple<8, bool>);
using Vec16b  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16b,  math::Tuple<16, bool>);
using Vec128b = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec128b, math::Tuple<128, bool>);
using Vec256b = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec256b, math::Tuple<256, bool>);
using Vec512b = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec512b, math::Tuple<512, bool>);

using Vec16cb = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16cb, math::Tuple<16, bool>);
using Vec16fb = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16fb, math::Tuple<16, bool>);
using Vec16ib = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16ib, math::Tuple<16, bool>);
using Vec16sb = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16sb, math::Tuple<16, bool>);
using Vec2db  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec2db, math::Tuple<2, bool>);
using Vec2qb  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec2qb, math::Tuple<2, bool>);
using Vec32cb = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec32cb, math::Tuple<32, bool>);
using Vec32sb = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec32sb, math::Tuple<32, bool>);
using Vec4db  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec4db, math::Tuple<4, bool>);
using Vec4fb  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec4fb, math::Tuple<4, bool>);
using Vec4ib  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec4ib, math::Tuple<4, bool>);
using Vec4qb  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec4qb, math::Tuple<4, bool>);
using Vec64cb = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec64cb, math::Tuple<64, bool>);
using Vec8db  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8db, math::Tuple<8, bool>);
using Vec8fb  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8fb, math::Tuple<8, bool>);
using Vec8ib  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8ib, math::Tuple<8, bool>);
using Vec8qb  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8qb, math::Tuple<8, bool>);
using Vec8sb  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8sb, math::Tuple<8, bool>);

using Vec16c  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16c, math::Tuple<16, int8_t>);
using Vec32c  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec32c, math::Tuple<32, int8_t>);
using Vec64c  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec64c, math::Tuple<64, int8_t>);
using Vec8s   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8s,  math::Tuple<8, int16_t>);
using Vec16s  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16s, math::Tuple<16, int16_t>);
using Vec32s  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec32s, math::Tuple<32, int16_t>);
using Vec4i   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec4i,  math::Tuple<4, int32_t>);
using Vec8i   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8i,  math::Tuple<8, int32_t>);
using Vec16i  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16i, math::Tuple<16, int32_t>);
using Vec2q   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec2q,  math::Tuple<2, int64_t>);
using Vec4q   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec4q,  math::Tuple<4, int64_t>);
using Vec8q   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8q,  math::Tuple<8, int64_t>);

using Vec16uc = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16uc, math::Tuple<16, uint8_t>);
using Vec32uc = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec32uc, math::Tuple<32, uint8_t>);
using Vec64uc = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec64uc, math::Tuple<64, uint8_t>);
using Vec8us  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8us,  math::Tuple<8, uint16_t>);
using Vec16us = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16us, math::Tuple<16, uint16_t>);
using Vec32us = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec32us, math::Tuple<32, uint16_t>);
using Vec4ui  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec4ui,  math::Tuple<4, uint32_t>);
using Vec8ui  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8ui,  math::Tuple<8, uint32_t>);
using Vec16ui = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16ui, math::Tuple<16, uint32_t>);
using Vec2uq  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec2uq,  math::Tuple<2, uint64_t>);
using Vec4uq  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec4uq,  math::Tuple<4, uint64_t>);
using Vec8uq  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8uq,  math::Tuple<8, uint64_t>);

using Vec4f   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec4f,  math::Tuple<4, float>);
using Vec8f   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8f,  math::Tuple<8, float>);
using Vec16f  = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec16f, math::Tuple<16, float>);
using Vec2d   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec2d,  math::Tuple<2, double>);
using Vec4d   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec4d,  math::Tuple<4, double>);
using Vec8d   = OPENVDB_SELECT_SIMD_T(openvdb_vcl::Vec8d,  math::Tuple<8, double>);

#undef OPENVDB_SELECT_SIMD_T

/// Helper structs/defines

template <typename T, size_t S> struct SimdT;
template <> struct SimdT<bool, 8>    { using Type = simd::Vec8b; };
template <> struct SimdT<bool, 16>   { using Type = simd::Vec16b; };
template <> struct SimdT<bool, 128>  { using Type = simd::Vec128b; };
template <> struct SimdT<bool, 256>  { using Type = simd::Vec256b; };
template <> struct SimdT<bool, 512>  { using Type = simd::Vec512b; };

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

template <> struct SimdT<float, 4>  { using Type = simd::Vec4f; };
template <> struct SimdT<float, 8>  { using Type = simd::Vec8f; };
template <> struct SimdT<float, 16> { using Type = simd::Vec16f; };
template <> struct SimdT<double, 2> { using Type = simd::Vec2d; };
template <> struct SimdT<double, 4> { using Type = simd::Vec4d; };
template <> struct SimdT<double, 8> { using Type = simd::Vec8d; };

/// Native type selection
template <typename T> struct SimdNativeT;
template <> struct SimdNativeT<bool>     : SimdT<bool, MAX_REGISTER_SIZE> {};
template <> struct SimdNativeT<int8_t>   : SimdT<int8_t, MAX_REGISTER_SIZE/(sizeof(int8_t)*CHAR_BIT)> {};
template <> struct SimdNativeT<int16_t>  : SimdT<int16_t, MAX_REGISTER_SIZE/(sizeof(int16_t)*CHAR_BIT)> {};
template <> struct SimdNativeT<int32_t>  : SimdT<int32_t, MAX_REGISTER_SIZE/(sizeof(int32_t)*CHAR_BIT)> {};
template <> struct SimdNativeT<int64_t>  : SimdT<int64_t, MAX_REGISTER_SIZE/(sizeof(int64_t)*CHAR_BIT)> {};
template <> struct SimdNativeT<uint8_t>  : SimdT<uint8_t, MAX_REGISTER_SIZE/(sizeof(uint8_t)*CHAR_BIT)> {};
template <> struct SimdNativeT<uint16_t> : SimdT<uint16_t, MAX_REGISTER_SIZE/(sizeof(uint16_t)*CHAR_BIT)> {};
template <> struct SimdNativeT<uint32_t> : SimdT<uint32_t, MAX_REGISTER_SIZE/(sizeof(uint32_t)*CHAR_BIT)> {};
template <> struct SimdNativeT<uint64_t> : SimdT<uint64_t, MAX_REGISTER_SIZE/(sizeof(uint64_t)*CHAR_BIT)> {};
template <> struct SimdNativeT<float>    : SimdT<float, MAX_REGISTER_SIZE/(sizeof(float)*CHAR_BIT)> {};
template <> struct SimdNativeT<double>   : SimdT<double, MAX_REGISTER_SIZE/(sizeof(double)*CHAR_BIT)> {};

/// @brief  VCL traits for determining if a type is a VCL type
template <typename T> struct IsSimdMaskT : std::false_type {};
template <typename T> struct IsSimdIntT : std::false_type {};
template <typename T> struct IsSimdFloatT : std::false_type {};

#ifndef OPENVDB_USE_VCL
template <typename T> struct IsSimdT : std::false_type {};
#else
// bool vectors
template <> struct IsSimdMaskT<simd::Vec128b> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec256b> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec512b> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec16b> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec8b> : std::true_type {};

// bool masks for various types
template <> struct IsSimdMaskT<simd::Vec16cb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec16fb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec16ib> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec16sb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec2db> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec2qb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec32cb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec32sb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec4db> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec4fb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec4ib> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec4qb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec64cb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec8db> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec8fb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec8ib> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec8qb> : std::true_type {};
template <> struct IsSimdMaskT<simd::Vec8sb> : std::true_type {};

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

// 32-bit float vectors
template <> struct IsSimdFloatT<simd::Vec4f> : std::true_type {};
template <> struct IsSimdFloatT<simd::Vec8f> : std::true_type {};
template <> struct IsSimdFloatT<simd::Vec16f> : std::true_type {};
// 64-bit double vectors
template <> struct IsSimdFloatT<simd::Vec2d> : std::true_type {};
template <> struct IsSimdFloatT<simd::Vec4d> : std::true_type {};
template <> struct IsSimdFloatT<simd::Vec8d> : std::true_type {};

template <typename T> struct IsSimdT :
    std::conditional<(IsSimdFloatT<T>::value || IsSimdMaskT<T>::value || IsSimdIntT<T>::value),
        std::true_type,
        std::false_type>::type {};
#endif

template<typename T> struct IsTupleT : std::false_type {};
template<int N, typename T> struct IsTupleT<math::Tuple<N, T>> : std::true_type {};

#define OPENVDB_ENABLE_IF_ARITHMETIC \
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>

#define OPENVDB_ENABLE_IF_TUPLE \
    template <typename T, typename std::enable_if<IsTupleT<T>::value>::type* = nullptr>

#define OPENVDB_ENABLE_IF_SIMD(...) \
    template <typename T, __VA_ARGS__ typename std::enable_if<IsSimdT<T>::value>::type* = nullptr>

template <typename T> struct SimdTraits; // fwd declare

//// Vector API

#ifdef OPENVDB_USE_VCL

template <size_t N, typename T> inline auto load(const T* a) { typename SimdT<T, N>::Type r; r.load(a); return r; }
template <size_t N, typename T> inline auto load_partial(int C, const T* a)
{
    assert(C <= int(N));
    typename SimdT<T, N>::Type r;
    r.load_partial(C, a);
    return r;
}

OPENVDB_ENABLE_IF_SIMD() inline auto compress(const T& a) { return openvdb_vcl::compress(a); }

OPENVDB_ENABLE_IF_SIMD() inline T add(const T& a, const T& b) { return a + b; }
OPENVDB_ENABLE_IF_SIMD() inline T sub(const T& a, const T& b) { return a - b; }
OPENVDB_ENABLE_IF_SIMD() inline T mul(const T& a, const T& b) { return a * b; }
OPENVDB_ENABLE_IF_SIMD() inline T div(const T& a, const T& b) { return a / b; }

OPENVDB_ENABLE_IF_SIMD() inline T max(const T& a, const T& b) { return openvdb_vcl::max(a, b); }
OPENVDB_ENABLE_IF_SIMD() inline T min(const T& a, const T& b) { return openvdb_vcl::min(a, b); }

OPENVDB_ENABLE_IF_SIMD() inline T abs(const T& a) { return openvdb_vcl::abs(a); }
OPENVDB_ENABLE_IF_SIMD() inline T sqrt(const T& a) { return openvdb_vcl::sqrt(a); }
OPENVDB_ENABLE_IF_SIMD() inline T pow2(const T& a) { return openvdb_vcl::square(a); }
OPENVDB_ENABLE_IF_SIMD() inline T pow3(const T& a) { return a*openvdb_vcl::square(a); }

template <int N, typename T, typename std::enable_if<IsSimdT<T>::value>::type* = nullptr>
inline auto pow(const T& a) { return openvdb_vcl::pow_n<T, N>(a); }

OPENVDB_ENABLE_IF_SIMD() inline auto eq(const T& a, const T& b) { return a == b; }
OPENVDB_ENABLE_IF_SIMD() inline auto neq(const T& a, const T& b) { return a != b; }
OPENVDB_ENABLE_IF_SIMD() inline auto gt(const T& a, const T& b) { return a > b; }
OPENVDB_ENABLE_IF_SIMD() inline auto gte(const T& a, const T& b) { return a >= b; }
OPENVDB_ENABLE_IF_SIMD() inline auto lt(const T& a, const T& b) { return a < b; }
OPENVDB_ENABLE_IF_SIMD() inline auto lte(const T& a, const T& b) { return a <= b; }
OPENVDB_ENABLE_IF_SIMD() inline auto logical_and(const T& a, const T& b) { return a && b; }

OPENVDB_ENABLE_IF_SIMD() inline T select(const typename SimdTraits<T>::MaskT& m, const T& a, const T& b) {
    return openvdb_vcl::select(m, a, b);
}

OPENVDB_ENABLE_IF_SIMD() inline auto is_finite(const T& a) { return openvdb_vcl::is_finite(a); }

OPENVDB_ENABLE_IF_SIMD() inline auto horizontal_max(const T& a) { return openvdb_vcl::horizontal_max(a); }
OPENVDB_ENABLE_IF_SIMD() inline auto horizontal_min(const T& a) { return openvdb_vcl::horizontal_min(a); }
OPENVDB_ENABLE_IF_SIMD() inline auto horizontal_and(const T& a) { return openvdb_vcl::horizontal_and(a); }
OPENVDB_ENABLE_IF_SIMD() inline auto horizontal_or(const T& a) { return openvdb_vcl::horizontal_or(a); }
OPENVDB_ENABLE_IF_SIMD() inline auto horizontal_add(const T& a) { return openvdb_vcl::horizontal_add(a); }
OPENVDB_ENABLE_IF_SIMD() inline int horizontal_count(const T& a) { return openvdb_vcl::horizontal_count(a); }
OPENVDB_ENABLE_IF_SIMD() inline int horizontal_find_first(const T& a) { return openvdb_vcl::horizontal_find_first(a); }

#else

namespace simd_internal
{
static auto minop = [](auto x, auto y) { return std::min(x,y); };
static auto maxop = [](auto x, auto y) { return std::max(x,y); };
static auto absop = [](auto x) { return std::abs(x); };
static auto sqrtop = [](auto x) { return std::sqrt(x); };
static auto compressop = [](auto x) { return typename PromoteType<decltype(x)>::Previous(x); };

template <typename UnaryOpT, typename T> inline auto unaryop(const T& a, const UnaryOpT& op)
{
    using ElementT = typename T::ValueType;
    using RetElementT = typename std::invoke_result<UnaryOpT,ElementT>::type;
    using RetTupleT = math::Tuple<T::size, RetElementT>;
    RetTupleT r; for (int i = 0; i < T::size; ++i) r[i] = op(a[i]); return r;
}

template <typename BinaryOpT, typename T> inline auto binop(const T& a, const T& b, const BinaryOpT& op)
{
    using ElementT = typename T::ValueType;
    using RetElementT = typename std::invoke_result<BinaryOpT,ElementT,ElementT>::type;
    using RetTupleT = math::Tuple<T::size, RetElementT>;
    RetTupleT r; for (int i = 0; i < T::size; ++i) r[i] = op(a[i], b[i]); return r;
}

template <template <typename> class BinaryOpT, typename T>
inline auto binop(const T& a, const T& b) {
    return binop(a,b,BinaryOpT<typename T::ValueType>{});
}

template <typename BinaryOpT, typename T> inline auto binop(const T& a, const BinaryOpT& op)
{
    using ElementT = typename T::ValueType;
    using RetElementT = typename std::invoke_result<BinaryOpT,ElementT,ElementT>::type;
    RetElementT r = a[0]; for (int i = 1; i < T::size; ++i) r = op(r, a[i]); return r;
}

template <template <typename> class BinaryOpT, typename T>
inline auto binop(const T& a) {
    return binop(a,BinaryOpT<typename T::ValueType>{});
}
}

template <size_t N, typename T>
inline typename SimdT<T, N>::Type load(const T* a)
{
    typename SimdT<T, N>::Type r;
    std::memcpy(r.asV(), a, N*sizeof(T));
    return r;
}

OPENVDB_ENABLE_IF_TUPLE inline auto compress(const T& a) { return simd_internal::unaryop(a,simd_internal::compressop); }

OPENVDB_ENABLE_IF_TUPLE inline auto add(const T& a, const T& b) { return simd_internal::binop<std::plus>(a,b); }
OPENVDB_ENABLE_IF_TUPLE inline auto sub(const T& a, const T& b) { return simd_internal::binop<std::minus>(a,b); }
OPENVDB_ENABLE_IF_TUPLE inline auto mul(const T& a, const T& b) { return simd_internal::binop<std::multiplies>(a,b); }
OPENVDB_ENABLE_IF_TUPLE inline auto div(const T& a, const T& b) { return simd_internal::binop<std::divides>(a,b); }

OPENVDB_ENABLE_IF_TUPLE inline auto abs(const T& a) { return simd_internal::unaryop(a,simd_internal::absop); }
OPENVDB_ENABLE_IF_TUPLE inline auto sqrt(const T& a) { return simd_internal::unaryop(a,simd_internal::sqrtop); }
OPENVDB_ENABLE_IF_TUPLE inline auto pow2(const T& a) { return mul(a,a); }
OPENVDB_ENABLE_IF_TUPLE inline auto pow3(const T& a) { return mul(a, mul(a,a)); }
template <typename T, typename ExpT> inline auto pow(const T& a, const ExpT n)
{
    T r; for (int i = 0; i < T::size; ++i) { r[i] = math::Pow(a[i],n); } return r;
}

OPENVDB_ENABLE_IF_TUPLE inline auto eq(const T& a, const T& b) { return simd_internal::binop<std::equal_to>(a,b); }
OPENVDB_ENABLE_IF_TUPLE inline auto neq(const T& a, const T& b) { return simd_internal::binop<std::not_equal_to>(a,b); }
OPENVDB_ENABLE_IF_TUPLE inline auto gt(const T& a, const T& b) { return simd_internal::binop<std::greater>(a,b); }
OPENVDB_ENABLE_IF_TUPLE inline auto gte(const T& a, const T& b) { return simd_internal::binop<std::greater_equal>(a,b); }
OPENVDB_ENABLE_IF_TUPLE inline auto lt(const T& a, const T& b) { return simd_internal::binop<std::less>(a,b); }
OPENVDB_ENABLE_IF_TUPLE inline auto lte(const T& a, const T& b) { return simd_internal::binop<std::less_equal>(a,b); }
OPENVDB_ENABLE_IF_TUPLE inline auto logical_and(const T& a, const T& b) { return simd_internal::binop<std::logical_and>(a,b); }

OPENVDB_ENABLE_IF_TUPLE inline T select(const typename SimdTraits<T>::MaskT& m, const T& a, const T& b)
{
    T r; for (int i = 0; i < T::size; ++i) { r[i] = m[i] ? a[i] : b[i]; } return r;
}

OPENVDB_ENABLE_IF_TUPLE inline bool is_finite(const T& a)
{
    for (int i = 0; i < T::size; ++i) { if (!std::isfinite(a[i])) return false; } return true;
}

OPENVDB_ENABLE_IF_TUPLE inline auto min(const T& a, const T& b) { return simd_internal::binop(a,b,simd_internal::minop); }
OPENVDB_ENABLE_IF_TUPLE inline auto max(const T& a, const T& b) { return simd_internal::binop(a,b,simd_internal::maxop); }

OPENVDB_ENABLE_IF_TUPLE inline auto horizontal_min(const T& a) { return simd_internal::binop(a,simd_internal::minop); }
OPENVDB_ENABLE_IF_TUPLE inline auto horizontal_max(const T& a) { return simd_internal::binop(a,simd_internal::maxop); }
OPENVDB_ENABLE_IF_TUPLE inline auto horizontal_and(const T& a) { return simd_internal::binop<std::logical_and>(a); }
OPENVDB_ENABLE_IF_TUPLE inline auto horizontal_or(const T& a)  { return simd_internal::binop<std::logical_or>(a); }
OPENVDB_ENABLE_IF_TUPLE inline int horizontal_count(const T& a)
{
    int count = 0;
    for (int i = 0; i < T::size; ++i) { if (a[i]) ++count; }
    return count;
}

OPENVDB_ENABLE_IF_TUPLE inline int horizontal_find_first(const T& a)
{
    for (int i = 0; i < T::size; ++i) { if (a[i]) return i; }
    return -1;
}

OPENVDB_ENABLE_IF_TUPLE inline auto horizontal_add(const T& a)
{
    static_assert((T::size % 2) == 0);
    using ValueType = typename T::ValueType;
    ValueType r(0);
    for (size_t i = 0; i < T::size; i+=2) {
        r += (a[i] + a[i+1]);
    }
    return r;
}

#endif

#ifdef OPENVDB_USE_VCL

namespace simd_internal
{
template <int> struct elem;
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
template <> struct elem<15> { using Type = math::half; };
template <> struct elem<16> { using Type = float; };
template <> struct elem<17> { using Type = double; };
}

template <typename T>
struct SimdTraits
{
    static constexpr size_t size = T::size();
    using ElementT = typename simd_internal::elem<T::elementtype()>::Type;
    using MaskT = decltype(eq<T>(std::declval<T>(), std::declval<T>()));
    template <typename S> using ConvertT = SimdT<S, size>;
};

#else

template <typename T>
struct SimdTraits
{
    static constexpr size_t size = T::size;
    using ElementT = typename T::ValueType;
    using MaskT = decltype(eq<T>(std::declval<T>(), std::declval<T>()));
    template <typename S> using ConvertT = SimdT<S, size>;
};

#endif

//// Scalar API

OPENVDB_ENABLE_IF_ARITHMETIC inline T add(const T& a, const T& b) { return a + b; }
OPENVDB_ENABLE_IF_ARITHMETIC inline T sub(const T& a, const T& b) { return a - b; }
OPENVDB_ENABLE_IF_ARITHMETIC inline T mul(const T& a, const T& b) { return a * b; }
OPENVDB_ENABLE_IF_ARITHMETIC inline T div(const T& a, const T& b) { return a / b; }

OPENVDB_ENABLE_IF_ARITHMETIC inline T max(const T& a, const T& b) { return std::max(a, b); }
OPENVDB_ENABLE_IF_ARITHMETIC inline T min(const T& a, const T& b) { return std::min(a, b); }
OPENVDB_ENABLE_IF_ARITHMETIC inline auto sqrt(const T& a) { return std::sqrt(a); }
OPENVDB_ENABLE_IF_ARITHMETIC inline T pow2(const T& a) { return math::Pow2(a); }
OPENVDB_ENABLE_IF_ARITHMETIC inline T pow3(const T& a) { return math::Pow3(a); }

OPENVDB_ENABLE_IF_ARITHMETIC inline auto square(const T& a) { return mul(a, a); }
OPENVDB_ENABLE_IF_ARITHMETIC inline bool is_finite(const T& a) { return std::isfinite(a); }

OPENVDB_ENABLE_IF_ARITHMETIC inline bool eq(const T& a, const T& b) { return a == b; }
OPENVDB_ENABLE_IF_ARITHMETIC inline bool neq(const T& a, const T& b) { return a != b; }
OPENVDB_ENABLE_IF_ARITHMETIC inline bool gt(const T& a, const T& b) { return a > b; }
OPENVDB_ENABLE_IF_ARITHMETIC inline bool gte(const T& a, const T& b) { return a >= b; }
OPENVDB_ENABLE_IF_ARITHMETIC inline bool lt(const T& a, const T& b) { return a < b; }
OPENVDB_ENABLE_IF_ARITHMETIC inline bool lte(const T& a, const T& b) { return a <= b; }
OPENVDB_ENABLE_IF_ARITHMETIC inline auto logical_and(const T& a, const T& b) { return a && b; }

OPENVDB_ENABLE_IF_ARITHMETIC inline T select(const bool m, const T& a, const T& b) { return m ? a : b; }

OPENVDB_ENABLE_IF_ARITHMETIC inline T horizontal_max(const T& a) { return a; }
OPENVDB_ENABLE_IF_ARITHMETIC inline T horizontal_min(const T& a) { return a; }
OPENVDB_ENABLE_IF_ARITHMETIC inline T horizontal_add(const T& a) { return a; }
inline bool horizontal_and(const bool a) { return a; }
inline bool horizontal_or(const bool a) { return a; }
inline int horizontal_find_first(const bool a) { return a ? 0 : -1; }
inline int horizontal_count(const bool a) { return a ? 1 : 0; }

#undef OPENVDB_ENABLE_IF_ARITHMETIC
#undef OPENVDB_ENABLE_IF_TUPLE
#undef OPENVDB_ENABLE_IF_SIMD

} // namespace simd
} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_SIMD_HAS_BEEN_INCLUDED
