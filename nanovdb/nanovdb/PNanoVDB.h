
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file   PNanoVDB.h

    \author Andrew Reidmeyer

    \brief  This file is a portable (e.g. pointer-less) C99/GLSL/HLSL port 
	        of NanoVDB.h, which is compatible with most graphics APIs.
*/

#ifndef NANOVDB_PNANOVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_PNANOVDB_H_HAS_BEEN_INCLUDED

// ------------------------------------------------ Configuration -----------------------------------------------------------

// platforms
//#define PNANOVDB_C
//#define PNANOVDB_HLSL
//#define PNANOVDB_GLSL

// addressing mode
// PNANOVDB_ADDRESS_32
// PNANOVDB_ADDRESS_64
#if defined(PNANOVDB_C)
#ifndef PNANOVDB_ADDRESS_32
#define PNANOVDB_ADDRESS_64
#endif
#elif defined(PNANOVDB_HLSL)
#define PNANOVDB_ADDRESS_32
#elif defined(PNANOVDB_GLSL)
#define PNANOVDB_ADDRESS_32
#endif

// bounds checking
//#define PNANOVDB_BUF_BOUNDS_CHECK

// enable HDDA by default on HLSL/GLSL, make explicit on C
#if defined(PNANOVDB_C)
//#define PNANOVDB_HDDA
#ifdef PNANOVDB_HDDA
#ifndef PNANOVDB_CMATH
#define PNANOVDB_CMATH
#endif
#endif
#elif defined(PNANOVDB_HLSL)
#define PNANOVDB_HDDA
#elif defined(PNANOVDB_GLSL)
#define PNANOVDB_HDDA
#endif

#ifdef PNANOVDB_CMATH
#include <math.h>
#endif

// ------------------------------------------------ Buffer -----------------------------------------------------------

#if defined(PNANOVDB_BUF_CUSTOM)
// NOP
#elif defined(PNANOVDB_C)
#define PNANOVDB_BUF_C
#elif defined(PNANOVDB_HLSL)
#define PNANOVDB_BUF_HLSL
#elif defined(PNANOVDB_GLSL)
#define PNANOVDB_BUF_GLSL
#endif

#if defined(PNANOVDB_BUF_C)
#include <stdint.h>
#if defined(_WIN32)
#define PNANOVDB_BUF_FORCE_INLINE static inline __forceinline
#else
#define PNANOVDB_BUF_FORCE_INLINE static inline __attribute__((always_inline))
#endif
typedef struct pnanovdb_buf_t
{
	uint32_t* data;
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
	uint64_t size_in_words;
#endif
}pnanovdb_buf_t;
PNANOVDB_BUF_FORCE_INLINE pnanovdb_buf_t pnanovdb_make_buf(uint32_t* data, uint64_t size_in_words)
{
	pnanovdb_buf_t ret;
	ret.data = data;
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
	ret.size_in_words = size_in_words;
#endif
	return ret;
}
#if defined(PNANOVDB_ADDRESS_32)
PNANOVDB_BUF_FORCE_INLINE uint32_t pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint32_t byte_offset)
{
	uint32_t wordaddress = (byte_offset >> 2u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
	return wordaddress < buf.size_in_words ? buf.data[wordaddress] : 0u;
#else
	return buf.data[wordaddress];
#endif
}
PNANOVDB_BUF_FORCE_INLINE uint64_t pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint32_t byte_offset)
{
	uint64_t* data64 = (uint64_t*)buf.data;
	uint32_t wordaddress64 = (byte_offset >> 3u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
	uint64_t size_in_words64 = buf.size_in_words >> 1u;
	return wordaddress64 < size_in_words64 ? data64[wordaddress64] : 0llu;
#else
	return data64[wordaddress64];
#endif
}
#elif defined(PNANOVDB_ADDRESS_64)
PNANOVDB_BUF_FORCE_INLINE uint32_t pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint64_t byte_offset)
{
	uint64_t wordaddress = (byte_offset >> 2u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
	return wordaddress < buf.size_in_words ? buf.data[wordaddress] : 0u;
#else
	return buf.data[wordaddress];
#endif
}
PNANOVDB_BUF_FORCE_INLINE uint64_t pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint64_t byte_offset)
{
	uint64_t* data64 = (uint64_t*)buf.data;
	uint64_t wordaddress64 = (byte_offset >> 3u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
	uint64_t size_in_words64 = buf.size_in_words >> 1u;
	return wordaddress64 < size_in_words64 ? data64[wordaddress64] : 0llu;
#else
	return data64[wordaddress64];
#endif
}
#endif
typedef uint32_t pnanovdb_grid_type_t;
#define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) pnanovdb_grid_type_constants[grid_typeIn].nameIn
#elif defined(PNANOVDB_BUF_HLSL)
#define pnanovdb_buf_t StructuredBuffer<uint>
uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byte_offset)
{
	return buf[(byte_offset >> 2u)];
}
uint2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byte_offset)
{
	uint2 ret;
	ret.x = pnanovdb_buf_read_uint32(buf, byte_offset + 0u);
	ret.y = pnanovdb_buf_read_uint32(buf, byte_offset + 4u);
	return ret;
}
#define pnanovdb_grid_type_t uint
#define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) pnanovdb_grid_type_constants[grid_typeIn].nameIn
#elif defined(PNANOVDB_BUF_GLSL)
struct pnanovdb_buf_t
{
	uint unused;	// to satisfy min struct size?
};
uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byte_offset)
{
	return pnanovdb_buf_data[(byte_offset >> 2u)];
}
uvec2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byte_offset)
{
	uvec2 ret;
	ret.x = pnanovdb_buf_read_uint32(buf, byte_offset + 0u);
	ret.y = pnanovdb_buf_read_uint32(buf, byte_offset + 4u);
	return ret;
}
#define pnanovdb_grid_type_t uint
#define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) pnanovdb_grid_type_constants[grid_typeIn].nameIn
#endif

// ------------------------------------------------ Basic Types -----------------------------------------------------------

// force inline
#if defined(PNANOVDB_C)
#if defined(_WIN32)
#define PNANOVDB_FORCE_INLINE static inline __forceinline
#else
#define PNANOVDB_FORCE_INLINE static inline __attribute__((always_inline))
#endif
#elif defined(PNANOVDB_HLSL)
#define PNANOVDB_FORCE_INLINE
#elif defined(PNANOVDB_GLSL)
#define PNANOVDB_FORCE_INLINE
#endif

// struct typedef, static const, inout
#if defined(PNANOVDB_C)
#define PNANOVDB_STRUCT_TYPEDEF(X) typedef struct X X;
#define PNANOVDB_STATIC_CONST static const
#define PNANOVDB_INOUT(X) X*
#define PNANOVDB_IN(X) const X*
#define PNANOVDB_DEREF(X) (*X)
#define PNANOVDB_REF(X) &X
#elif defined(PNANOVDB_HLSL)
#define PNANOVDB_STRUCT_TYPEDEF(X)
#define PNANOVDB_STATIC_CONST static const
#define PNANOVDB_INOUT(X) inout X
#define PNANOVDB_IN(X) X
#define PNANOVDB_DEREF(X) X
#define PNANOVDB_REF(X) X
#elif defined(PNANOVDB_GLSL)
#define PNANOVDB_STRUCT_TYPEDEF(X)
#define PNANOVDB_STATIC_CONST const
#define PNANOVDB_INOUT(X) inout X
#define PNANOVDB_IN(X) X
#define PNANOVDB_DEREF(X) X
#define PNANOVDB_REF(X) X
#endif

// basic types, type conversion
#if defined(PNANOVDB_C)
#define PNANOVDB_NATIVE_64
#include <stdint.h>
typedef uint32_t pnanovdb_uint32_t;
typedef int32_t pnanovdb_int32_t;
typedef int32_t pnanovdb_bool_t;
#define PNANOVDB_FALSE 0
#define PNANOVDB_TRUE 1
typedef uint64_t pnanovdb_uint64_t;
typedef int64_t pnanovdb_int64_t;
typedef struct pnanovdb_coord_t
{
	pnanovdb_int32_t x, y, z;
}pnanovdb_coord_t;
typedef struct pnanovdb_vec3_t
{
	float x, y, z;
}pnanovdb_vec3_t;
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_uint32_as_int32(pnanovdb_uint32_t v) { return (pnanovdb_int32_t)v; }
PNANOVDB_FORCE_INLINE pnanovdb_int64_t pnanovdb_uint64_as_int64(pnanovdb_uint64_t v) { return (pnanovdb_int64_t)v; }
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_int64_as_uint64(pnanovdb_int64_t v) { return (pnanovdb_uint64_t)v; }
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_int32_as_uint32(pnanovdb_int32_t v) { return (pnanovdb_uint32_t)v; }
PNANOVDB_FORCE_INLINE float pnanovdb_uint32_as_float(pnanovdb_uint32_t v) { return *((float*)(&v)); }
PNANOVDB_FORCE_INLINE double pnanovdb_uint64_as_double(pnanovdb_uint64_t v) { return *((double*)(&v)); }
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_uint64_low(pnanovdb_uint64_t v) { return (pnanovdb_uint32_t)v; }
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_uint64_high(pnanovdb_uint64_t v) { return (pnanovdb_uint32_t)(v >> 32u); }
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint32_as_uint64(pnanovdb_uint32_t x, pnanovdb_uint32_t y) { return ((pnanovdb_uint64_t)x) | (((pnanovdb_uint64_t)y) << 32u); }
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint32_as_uint64_low(pnanovdb_uint32_t x) { return ((pnanovdb_uint64_t)x); }
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_uint64_is_equal(pnanovdb_uint64_t a, pnanovdb_uint64_t b) { return a == b; }
#ifdef PNANOVDB_CMATH
PNANOVDB_FORCE_INLINE float pnanovdb_floor(float v) { return floorf(v); }
#endif
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_float_to_int32(float v) { return (pnanovdb_int32_t)v; }
PNANOVDB_FORCE_INLINE float pnanovdb_int32_to_float(pnanovdb_int32_t v) { return (float)v; }
PNANOVDB_FORCE_INLINE float pnanovdb_min(float a, float b) { return a < b ? a : b; }
PNANOVDB_FORCE_INLINE float pnanovdb_max(float a, float b) { return a > b ? a : b; }
#elif defined(PNANOVDB_HLSL)
typedef uint pnanovdb_uint32_t;
typedef int pnanovdb_int32_t;
typedef bool pnanovdb_bool_t;
#define PNANOVDB_FALSE false
#define PNANOVDB_TRUE true
typedef uint2 pnanovdb_uint64_t;
typedef int2 pnanovdb_int64_t;
typedef int3 pnanovdb_coord_t;
typedef float3 pnanovdb_vec3_t;
pnanovdb_int32_t pnanovdb_uint32_as_int32(pnanovdb_uint32_t v) { return int(v); }
pnanovdb_int64_t pnanovdb_uint64_as_int64(pnanovdb_uint64_t v) { return int2(v); }
pnanovdb_uint64_t pnanovdb_int64_as_uint64(pnanovdb_int64_t v) { return uint2(v); }
pnanovdb_uint32_t pnanovdb_int32_as_uint32(pnanovdb_int32_t v) { return uint(v); }
float pnanovdb_uint32_as_float(pnanovdb_uint32_t v) { return asfloat(v); }
double pnanovdb_uint64_as_double(pnanovdb_uint64_t v) { return asdouble(v.x, v.y); }
pnanovdb_uint32_t pnanovdb_uint64_low(pnanovdb_uint64_t v) { return v.x; }
pnanovdb_uint32_t pnanovdb_uint64_high(pnanovdb_uint64_t v) { return v.y; }
pnanovdb_uint64_t pnanovdb_uint32_as_uint64(pnanovdb_uint32_t x, pnanovdb_uint32_t y) { return uint2(x, y); }
pnanovdb_uint64_t pnanovdb_uint32_as_uint64_low(pnanovdb_uint32_t x) { return uint2(x, 0); }
bool pnanovdb_uint64_is_equal(pnanovdb_uint64_t a, pnanovdb_uint64_t b) { return (a.x == b.x) && (a.y == b.y); }
float pnanovdb_floor(float v) { return floor(v); }
pnanovdb_int32_t pnanovdb_float_to_int32(float v) { return int(v); }
float pnanovdb_int32_to_float(pnanovdb_int32_t v) { return float(v); }
float pnanovdb_min(float a, float b) { return min(a, b); }
float pnanovdb_max(float a, float b) { return max(a, b); }
#elif defined(PNANOVDB_GLSL)
#define pnanovdb_uint32_t uint
#define pnanovdb_int32_t int
#define pnanovdb_bool_t bool
#define PNANOVDB_FALSE false
#define PNANOVDB_TRUE true
#define pnanovdb_uint64_t uvec2
#define pnanovdb_int64_t ivec2
#define pnanovdb_coord_t ivec3
#define pnanovdb_vec3_t vec3
pnanovdb_int32_t pnanovdb_uint32_as_int32(pnanovdb_uint32_t v) { return int(v); }
pnanovdb_int64_t pnanovdb_uint64_as_int64(pnanovdb_uint64_t v) { return ivec2(v); }
pnanovdb_uint64_t pnanovdb_int64_as_uint64(pnanovdb_int64_t v) { return uvec2(v); }
pnanovdb_uint32_t pnanovdb_int32_as_uint32(pnanovdb_int32_t v) { return uint(v); }
float pnanovdb_uint32_as_float(pnanovdb_uint32_t v) { return uintBitsToFloat(v); }
double pnanovdb_uint64_as_double(pnanovdb_uint64_t v) { return packDouble2x32(uvec2(v.x, v.y)); }
pnanovdb_uint32_t pnanovdb_uint64_low(pnanovdb_uint64_t v) { return v.x; }
pnanovdb_uint32_t pnanovdb_uint64_high(pnanovdb_uint64_t v) { return v.y; }
pnanovdb_uint64_t pnanovdb_uint32_as_uint64(pnanovdb_uint32_t x, pnanovdb_uint32_t y) { return uvec2(x, y); }
pnanovdb_uint64_t pnanovdb_uint32_as_uint64_low(pnanovdb_uint32_t x) { return uvec2(x, 0); }
bool pnanovdb_uint64_is_equal(pnanovdb_uint64_t a, pnanovdb_uint64_t b) { return (a.x == b.x) && (a.y == b.y); }
float pnanovdb_floor(float v) { return floor(v); }
pnanovdb_int32_t pnanovdb_float_to_int32(float v) { return int(v); }
float pnanovdb_int32_to_float(pnanovdb_int32_t v) { return float(v); }
float pnanovdb_min(float a, float b) { return min(a, b); }
float pnanovdb_max(float a, float b) { return max(a, b); }
#endif

// ------------------------------------------------ Coord/Vec3 Utilties -----------------------------------------------------------

#if defined(PNANOVDB_C)
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_uniform(float a)
{
	pnanovdb_vec3_t v;
	v.x = a;
	v.y = a;
	v.z = a;
	return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_add(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
	pnanovdb_vec3_t v;
	v.x = a.x + b.x;
	v.y = a.y + b.y;
	v.z = a.z + b.z;
	return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_sub(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
	pnanovdb_vec3_t v;
	v.x = a.x - b.x;
	v.y = a.y - b.y;
	v.z = a.z - b.z;
	return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_mul(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
	pnanovdb_vec3_t v;
	v.x = a.x * b.x;
	v.y = a.y * b.y;
	v.z = a.z * b.z;
	return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_div(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
	pnanovdb_vec3_t v;
	v.x = a.x / b.x;
	v.y = a.y / b.y;
	v.z = a.z / b.z;
	return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_min(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
	pnanovdb_vec3_t v;
	v.x = a.x < b.x ? a.x : b.x;
	v.y = a.y < b.y ? a.y : b.y;
	v.z = a.z < b.z ? a.z : b.z;
	return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_max(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
	pnanovdb_vec3_t v;
	v.x = a.x > b.x ? a.x : b.x;
	v.y = a.y > b.y ? a.y : b.y;
	v.z = a.z > b.z ? a.z : b.z;
	return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_coord_to_vec3(const pnanovdb_coord_t coord)
{
	pnanovdb_vec3_t v;
	v.x = pnanovdb_int32_to_float(coord.x);
	v.y = pnanovdb_int32_to_float(coord.y);
	v.z = pnanovdb_int32_to_float(coord.z);
	return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_coord_uniform(const pnanovdb_int32_t a)
{
	pnanovdb_coord_t v;
	v.x = a;
	v.y = a;
	v.z = a;
	return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_coord_add(pnanovdb_coord_t a, pnanovdb_coord_t b)
{
	pnanovdb_coord_t v;
	v.x = a.x + b.x;
	v.y = a.y + b.y;
	v.z = a.z + b.z;
	return v;
}
#elif defined(PNANOVDB_HLSL)
pnanovdb_vec3_t pnanovdb_vec3_uniform(float a) { return float3(a, a, a); }
pnanovdb_vec3_t pnanovdb_vec3_add(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a + b; }
pnanovdb_vec3_t pnanovdb_vec3_sub(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a - b; }
pnanovdb_vec3_t pnanovdb_vec3_mul(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a * b; }
pnanovdb_vec3_t pnanovdb_vec3_div(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a / b; }
pnanovdb_vec3_t pnanovdb_vec3_min(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return min(a, b); }
pnanovdb_vec3_t pnanovdb_vec3_max(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return max(a, b); }
pnanovdb_vec3_t pnanovdb_coord_to_vec3(pnanovdb_coord_t coord) { return float3(coord); }
pnanovdb_coord_t pnanovdb_coord_uniform(pnanovdb_int32_t a) { return int3(a, a, a); }
pnanovdb_coord_t pnanovdb_coord_add(pnanovdb_coord_t a, pnanovdb_coord_t b) { return a + b; }
#elif defined(PNANOVDB_GLSL)
pnanovdb_vec3_t pnanovdb_vec3_uniform(float a) { return vec3(a, a, a); }
pnanovdb_vec3_t pnanovdb_vec3_add(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a + b; }
pnanovdb_vec3_t pnanovdb_vec3_sub(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a - b; }
pnanovdb_vec3_t pnanovdb_vec3_mul(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a * b; }
pnanovdb_vec3_t pnanovdb_vec3_div(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a / b; }
pnanovdb_vec3_t pnanovdb_vec3_min(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return min(a, b); }
pnanovdb_vec3_t pnanovdb_vec3_max(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return max(a, b); }
pnanovdb_vec3_t pnanovdb_coord_to_vec3(const pnanovdb_coord_t coord) { return vec3(coord); }
pnanovdb_coord_t pnanovdb_coord_uniform(pnanovdb_int32_t a) { return ivec3(a, a, a); }
pnanovdb_coord_t pnanovdb_coord_add(pnanovdb_coord_t a, pnanovdb_coord_t b) { return a + b; }
#endif

// ------------------------------------------------ Address Type -----------------------------------------------------------

#if defined(PNANOVDB_ADDRESS_32)
struct pnanovdb_address_t
{
	pnanovdb_uint32_t byte_offset;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_address_t)

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset(pnanovdb_address_t address, pnanovdb_uint32_t byte_offset)
{
	pnanovdb_address_t ret = address;
	ret.byte_offset += byte_offset;
	return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset_product(pnanovdb_address_t address, pnanovdb_uint32_t byte_offset, pnanovdb_uint32_t multiplier)
{
	pnanovdb_address_t ret = address;
	ret.byte_offset += byte_offset * multiplier;
	return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset64(pnanovdb_address_t address, pnanovdb_uint64_t byte_offset)
{
	pnanovdb_address_t ret = address;
	// lose high bits on 32-bit
	ret.byte_offset += pnanovdb_uint64_low(byte_offset);
	return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_address_mask(pnanovdb_address_t address, pnanovdb_uint32_t mask)
{
	return address.byte_offset & mask;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_mask_inv(pnanovdb_address_t address, pnanovdb_uint32_t mask)
{
	pnanovdb_address_t ret = address;
	ret.byte_offset &= (~mask);
	return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_null()
{
	pnanovdb_address_t ret = { 0 };
	return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_address_is_null(pnanovdb_address_t address)
{
	return address.byte_offset == 0u;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_address_in_interval(pnanovdb_address_t address, pnanovdb_address_t min_address, pnanovdb_address_t max_address)
{
	return address.byte_offset >= min_address.byte_offset && address.byte_offset < max_address.byte_offset;
}
#elif defined(PNANOVDB_ADDRESS_64)
struct pnanovdb_address_t
{
	pnanovdb_uint64_t byte_offset;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_address_t)

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset(pnanovdb_address_t address, pnanovdb_uint32_t byte_offset)
{
	pnanovdb_address_t ret = address;
	ret.byte_offset += byte_offset;
	return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset_product(pnanovdb_address_t address, pnanovdb_uint32_t byte_offset, pnanovdb_uint32_t multiplier)
{
	pnanovdb_address_t ret = address;
	ret.byte_offset += pnanovdb_uint32_as_uint64_low(byte_offset) * pnanovdb_uint32_as_uint64_low(multiplier);
	return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset64(pnanovdb_address_t address, pnanovdb_uint64_t byte_offset)
{
	pnanovdb_address_t ret = address;
	ret.byte_offset += byte_offset;
	return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_address_mask(pnanovdb_address_t address, pnanovdb_uint32_t mask)
{
	return pnanovdb_uint64_low(address.byte_offset) & mask;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_mask_inv(pnanovdb_address_t address, pnanovdb_uint32_t mask)
{
	pnanovdb_address_t ret = address;
	ret.byte_offset &= (~pnanovdb_uint32_as_uint64_low(mask));
	return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_null()
{
	pnanovdb_address_t ret = { 0 };
	return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_address_is_null(pnanovdb_address_t address)
{
	return address.byte_offset == 0llu;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_address_in_interval(pnanovdb_address_t address, pnanovdb_address_t min_address, pnanovdb_address_t max_address)
{
	return address.byte_offset >= min_address.byte_offset && address.byte_offset < max_address.byte_offset;
}
#endif

// ------------------------------------------------ High Level Buffer Read -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_read_uint32(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
	return pnanovdb_buf_read_uint32(buf, address.byte_offset);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_read_uint64(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
	return pnanovdb_buf_read_uint64(buf, address.byte_offset);
}
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_read_int32(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
	return pnanovdb_uint32_as_int32(pnanovdb_read_uint32(buf, address));
}
PNANOVDB_FORCE_INLINE float pnanovdb_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
	return pnanovdb_uint32_as_float(pnanovdb_read_uint32(buf, address));
}
PNANOVDB_FORCE_INLINE pnanovdb_int64_t pnanovdb_read_int64(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
	return pnanovdb_uint64_as_int64(pnanovdb_read_uint64(buf, address));
}
PNANOVDB_FORCE_INLINE double pnanovdb_read_double(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
	return pnanovdb_uint64_as_double(pnanovdb_read_uint64(buf, address));
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_read_coord(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
	pnanovdb_coord_t ret;
	ret.x = pnanovdb_uint32_as_int32(pnanovdb_read_uint32(buf, pnanovdb_address_offset(address, 0u)));
	ret.y = pnanovdb_uint32_as_int32(pnanovdb_read_uint32(buf, pnanovdb_address_offset(address, 4u)));
	ret.z = pnanovdb_uint32_as_int32(pnanovdb_read_uint32(buf, pnanovdb_address_offset(address, 8u)));
	return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_read_bit(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_uint32_t bit_offset)
{
	pnanovdb_address_t word_address = pnanovdb_address_mask_inv(address, 3u);
	pnanovdb_uint32_t bit_index = (pnanovdb_address_mask(address, 3u) << 3u) + bit_offset;
	pnanovdb_uint32_t value_word = pnanovdb_buf_read_uint32(buf, word_address.byte_offset);
	return ((value_word >> bit_index) & 1) != 0u;
}

#if defined(PNANOVDB_C)
PNANOVDB_FORCE_INLINE short pnanovdb_read_half(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
	pnanovdb_uint32_t raw = pnanovdb_read_uint32(buf, address);
	return (short)(raw >> (pnanovdb_address_mask(address, 2) << 3));
}
#elif defined(PNANOVDB_HLSL)
PNANOVDB_FORCE_INLINE float pnanovdb_read_half(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
	pnanovdb_uint32_t raw = pnanovdb_read_uint32(buf, address);
	return f16tof32(raw >> (pnanovdb_address_mask(address, 2) << 3));
}
#elif defined(PNANOVDB_GLSL)
PNANOVDB_FORCE_INLINE float pnanovdb_read_half(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
	pnanovdb_uint32_t raw = pnanovdb_read_uint32(buf, address);
	return unpackHalf2x16(raw >> (pnanovdb_address_mask(address, 2) << 3)).x;
}
#endif

// ------------------------------------------------ Core Structures -----------------------------------------------------------

#define PNANOVDB_MAGIC_NUMBER 0x304244566f6e614eUL// "NanoVDB0" in hex - little endian (uint64_t)

#define PNANOVDB_MAJOR_VERSION_NUMBER 29// reflects changes to the ABI
#define PNANOVDB_MINOR_VERSION_NUMBER  3// reflects changes to the API but not ABI
#define PNANOVDB_PATCH_VERSION_NUMBER  0// reflects bug-fixes with no ABI or API changes

#define PNANOVDB_GRID_TYPE_UNKNOWN 0
#define PNANOVDB_GRID_TYPE_FLOAT 1
#define PNANOVDB_GRID_TYPE_DOUBLE 2
#define PNANOVDB_GRID_TYPE_INT16 3
#define PNANOVDB_GRID_TYPE_INT32 4
#define PNANOVDB_GRID_TYPE_INT64 5
#define PNANOVDB_GRID_TYPE_VEC3F 6
#define PNANOVDB_GRID_TYPE_VEC3D 7
#define PNANOVDB_GRID_TYPE_MASK 8
#define PNANOVDB_GRID_TYPE_FP16 9
#define PNANOVDB_GRID_TYPE_UINT32 10
#define PNANOVDB_GRID_TYPE_BOOLEAN 11
#define PNANOVDB_GRID_TYPE_PACKED_RGBA8 12
#define PNANOVDB_GRID_TYPE_END 13

#define PNANOVDB_GRID_CLASS_UNKNOWN 0
#define PNANOVDB_GRID_CLASS_LEVEL_SET 1
#define PNANOVDB_GRID_CLASS_FOG_VOLUME 2
#define PNANOVDB_GRID_CLASS_STAGGERED 3
#define PNANOVDB_GRID_CLASS_POINT_INDEX 4
#define PNANOVDB_GRID_CLASS_POINT_DATA 5
#define PNANOVDB_GRID_CLASS_TOPOLOGY 6
#define PNANOVDB_GRID_CLASS_VOXEL_VOLUME 7
#define PNANOVDB_GRID_CLASS_END 8

#define PNANOVDB_GRID_FLAGS_HAS_LONG_GRID_NAME (1 << 0)
#define PNANOVDB_GRID_FLAGS_HAS_BBOX (1 << 1)
#define PNANOVDB_GRID_FLAGS_HAS_MIN_MAX (1 << 2)
#define PNANOVDB_GRID_FLAGS_HAS_AVERAGE (1 << 3)
#define PNANOVDB_GRID_FLAGS_HAS_STD_DEVIATION (1 << 4)
#define PNANOVDB_GRID_FLAGS_END (1 << 5)

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_value_strides_bits[PNANOVDB_GRID_TYPE_END]  = { 0, 32, 64, 16, 32, 64, 96, 192, 0, 16, 32, 1 };
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_minmax_strides_bits[PNANOVDB_GRID_TYPE_END] = { 0, 32, 64, 16, 32, 64, 96, 192, 8, 16, 32, 8 };
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_minmax_aligns_bits[PNANOVDB_GRID_TYPE_END]  = { 0, 32, 64, 16, 32, 64, 32,  64, 8, 16, 32, 8 };
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_stat_strides_bits[PNANOVDB_GRID_TYPE_END]   = { 0, 32, 64, 32, 32, 64, 32,  64, 8, 32, 32, 8 };
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_leaf_lite[PNANOVDB_GRID_TYPE_END]           = { 0,  0,  0,  0,  0,  0,  0,   0, 1,  0,  0, 1 };

struct pnanovdb_map_t
{
	float matf[9];
	float invmatf[9];
	float vecf[3];
	float taperf;
	double matd[9];
	double invmatd[9];
	double vecd[3];
	double taperd;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_map_t)
struct pnanovdb_map_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_map_handle_t)

#define PNANOVDB_MAP_SIZE 264

#define PNANOVDB_MAP_OFF_MATF 0
#define PNANOVDB_MAP_OFF_INVMATF 36
#define PNANOVDB_MAP_OFF_VECF 72
#define PNANOVDB_MAP_OFF_TAPERF 84
#define PNANOVDB_MAP_OFF_MATD 88
#define PNANOVDB_MAP_OFF_INVMATD 160
#define PNANOVDB_MAP_OFF_VECD 232
#define PNANOVDB_MAP_OFF_TAPERD 256

PNANOVDB_FORCE_INLINE float pnanovdb_map_get_matf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_MATF + 4u * index));
}
PNANOVDB_FORCE_INLINE float pnanovdb_map_get_invmatf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_INVMATF + 4u * index));
}
PNANOVDB_FORCE_INLINE float pnanovdb_map_get_vecf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_VECF + 4u * index));
}
PNANOVDB_FORCE_INLINE float pnanovdb_map_get_taperf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_TAPERF));
}
PNANOVDB_FORCE_INLINE double pnanovdb_map_get_matd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_MATD + 8u * index));
}
PNANOVDB_FORCE_INLINE double pnanovdb_map_get_invmatd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_INVMATD + 8u * index));
}
PNANOVDB_FORCE_INLINE double pnanovdb_map_get_vecd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_VECD + 8u * index));
}
PNANOVDB_FORCE_INLINE double pnanovdb_map_get_taperd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_TAPERD));
}

struct pnanovdb_grid_t
{
	pnanovdb_uint64_t magic;					// 8 bytes, 	0
	pnanovdb_uint64_t checksum;					// 8 bytes,		8
	pnanovdb_uint32_t version;					// 4 bytes,		16
	pnanovdb_uint32_t flags;					// 4 bytes,		20
	pnanovdb_uint64_t grid_size;				// 8 bytes,		24
	pnanovdb_uint32_t grid_name[256 / 4];		// 256 bytes, 	32
	pnanovdb_map_t map;							// 264 bytes,	288
	double world_bbox[6];						// 48 bytes,	552
	double voxel_size[3];						// 24 bytes,	600
	pnanovdb_uint32_t grid_class;				// 4 bytes,		624
	pnanovdb_uint32_t grid_type;				// 4 bytes,		628
	pnanovdb_uint64_t blind_metadata_offset;	// 8 bytes,		632
	pnanovdb_uint32_t blind_metadata_count;		// 4 bytes,		640
	pnanovdb_uint32_t pad[6];					// 24 bytes,	644
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_grid_t)
struct pnanovdb_grid_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_grid_handle_t)

#define PNANOVDB_GRID_SIZE 672

#define PNANOVDB_GRID_OFF_MAGIC 0
#define PNANOVDB_GRID_OFF_CHECKSUM 8
#define PNANOVDB_GRID_OFF_VERSION 16
#define PNANOVDB_GRID_OFF_FLAGS 20
#define PNANOVDB_GRID_OFF_GRID_SIZE 24
#define PNANOVDB_GRID_OFF_GRID_NAME 32
#define PNANOVDB_GRID_OFF_MAP 288
#define PNANOVDB_GRID_OFF_WORLD_BBOX 552
#define PNANOVDB_GRID_OFF_VOXEL_SIZE 600
#define PNANOVDB_GRID_OFF_GRID_CLASS 624
#define PNANOVDB_GRID_OFF_GRID_TYPE 628
#define PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET 632
#define PNANOVDB_GRID_OFF_BLIND_METADATA_COUNT 640

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_grid_get_magic(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_MAGIC));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_grid_get_checksum(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_CHECKSUM));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_version(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_VERSION));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_flags(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_FLAGS));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_grid_get_grid_size(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_SIZE));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_grid_name(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_NAME + 4u * index));
}
PNANOVDB_FORCE_INLINE pnanovdb_map_handle_t pnanovdb_grid_get_map(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
	pnanovdb_map_handle_t ret;
	ret.address = pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_MAP);
	return ret;
}
PNANOVDB_FORCE_INLINE double pnanovdb_grid_get_world_bbox(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_WORLD_BBOX + 8u * index));
}
PNANOVDB_FORCE_INLINE double pnanovdb_grid_get_voxel_size(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_VOXEL_SIZE + 8u * index));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_grid_class(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_CLASS));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_grid_type(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_TYPE));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_grid_get_blind_metadata_offset(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_blind_metadata_count(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) { 
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_BLIND_METADATA_COUNT));
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_version_get_major(pnanovdb_uint32_t version)
{
	return (version >> 21u) & ((1u << 11u) - 1u);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_version_get_minor(pnanovdb_uint32_t version)
{
	return (version >> 10u) & ((1u << 11u) - 1u);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_version_get_patch(pnanovdb_uint32_t version)
{
	return version & ((1u << 10u) - 1u);
}

struct pnanovdb_gridblindmetadata_t
{
	pnanovdb_int64_t byte_offset;		// 8 bytes,		0
	pnanovdb_uint64_t element_count;	// 8 bytes,		8
	pnanovdb_uint32_t flags;			// 4 bytes,		16
	pnanovdb_uint32_t semantic;			// 4 bytes,		20
	pnanovdb_uint32_t data_class;		// 4 bytes,		24
	pnanovdb_uint32_t data_type;		// 4 bytes,		28
	pnanovdb_uint32_t name[256 / 4];	// 256 bytes,	32
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_gridblindmetadata_t)
struct pnanovdb_gridblindmetadata_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_gridblindmetadata_handle_t)

#define PNANOVDB_GRIDBLINDMETADATA_SIZE 288

#define PNANOVDB_GRIDBLINDMETADATA_OFF_BYTE_OFFSET 0
#define PNANOVDB_GRIDBLINDMETADATA_OFF_ELEMENT_COUNT 8
#define PNANOVDB_GRIDBLINDMETADATA_OFF_FLAGS 16
#define PNANOVDB_GRIDBLINDMETADATA_OFF_SEMANTIC 20
#define PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_CLASS 24
#define PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_TYPE 28
#define PNANOVDB_GRIDBLINDMETADATA_OFF_NAME 32

PNANOVDB_FORCE_INLINE pnanovdb_int64_t pnanovdb_gridblindmetadata_get_byte_offset(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
	return pnanovdb_read_int64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_BYTE_OFFSET));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_gridblindmetadata_get_element_count(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_ELEMENT_COUNT));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_gridblindmetadata_get_flags(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_FLAGS));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_gridblindmetadata_get_semantic(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_SEMANTIC));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_gridblindmetadata_get_data_class(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_CLASS));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_gridblindmetadata_get_data_type(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_TYPE));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_gridblindmetadata_get_name(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p, pnanovdb_uint32_t index) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_NAME + 4u * index));
}

struct pnanovdb_tree_t
{
	pnanovdb_uint64_t bytes0;
	pnanovdb_uint64_t bytes1;
	pnanovdb_uint64_t bytes2;
	pnanovdb_uint64_t bytes3;
	pnanovdb_uint32_t count0;
	pnanovdb_uint32_t count1;
	pnanovdb_uint32_t count2;
	pnanovdb_uint32_t count3;
	pnanovdb_uint32_t pad[4u];
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_tree_t)
struct pnanovdb_tree_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_tree_handle_t)

#define PNANOVDB_TREE_SIZE 64

#define PNANOVDB_TREE_OFF_BYTES0 0
#define PNANOVDB_TREE_OFF_BYTES1 8
#define PNANOVDB_TREE_OFF_BYTES2 16
#define PNANOVDB_TREE_OFF_BYTES3 24
#define PNANOVDB_TREE_OFF_COUNT0 32
#define PNANOVDB_TREE_OFF_COUNT1 36
#define PNANOVDB_TREE_OFF_COUNT2 40
#define PNANOVDB_TREE_OFF_COUNT3 44

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_tree_get_bytes0(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_BYTES0));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_tree_get_bytes1(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_BYTES1));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_tree_get_bytes2(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_BYTES2));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_tree_get_bytes3(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_BYTES3));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_tree_get_count0(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_COUNT0));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_tree_get_count1(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_COUNT1));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_tree_get_count2(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_COUNT2));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_tree_get_count3(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_COUNT3));
}

struct pnanovdb_root_t
{
	pnanovdb_coord_t bbox_min;
	pnanovdb_coord_t bbox_max;
	pnanovdb_uint64_t active_voxel_count;
	pnanovdb_uint32_t tile_count;
	pnanovdb_uint32_t pad1;						// background can start here
	// background, min, max
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_root_t)
struct pnanovdb_root_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_root_handle_t)

#define PNANOVDB_ROOT_SIZE 36

#define PNANOVDB_ROOT_OFF_BBOX_MIN 0
#define PNANOVDB_ROOT_OFF_BBOX_MAX 12
#define PNANOVDB_ROOT_OFF_ACTIVE_VOXEL_COUNT 24
#define PNANOVDB_ROOT_OFF_TILE_COUNT 32

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_root_get_bbox_min(pnanovdb_buf_t buf, pnanovdb_root_handle_t p) {
	return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_BBOX_MIN));
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_root_get_bbox_max(pnanovdb_buf_t buf, pnanovdb_root_handle_t p) {
	return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_BBOX_MAX));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_root_get_active_voxel_count(pnanovdb_buf_t buf, pnanovdb_root_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_ACTIVE_VOXEL_COUNT));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_root_get_tile_count(pnanovdb_buf_t buf, pnanovdb_root_handle_t p) { 
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_TILE_COUNT));
}

struct pnanovdb_root_tile_t
{
	pnanovdb_uint64_t key;
	pnanovdb_int32_t child_id;
	pnanovdb_uint32_t state;
	// value
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_root_tile_t)
struct pnanovdb_root_tile_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_root_tile_handle_t)

#define PNANOVDB_ROOT_TILE_SIZE 16

#define PNANOVDB_ROOT_TILE_OFF_KEY 0
#define PNANOVDB_ROOT_TILE_OFF_CHILD_ID 8
#define PNANOVDB_ROOT_TILE_OFF_STATE 12

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_root_tile_get_key(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p) {
	return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_KEY));
}
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_root_tile_get_child_id(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p) { 
	return pnanovdb_read_int32(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_CHILD_ID));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_root_tile_get_state(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_STATE));
}

struct pnanovdb_node2_t
{
	pnanovdb_coord_t bbox_min;
	pnanovdb_coord_t bbox_max;
	pnanovdb_int32_t offset;
	pnanovdb_uint32_t flags;
	pnanovdb_uint32_t value_mask[1024];
	pnanovdb_uint32_t child_mask[1024];
	// min, max
	// alignas(32) pnanovdb_uint32_t table[];
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_node2_t)
struct pnanovdb_node2_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_node2_handle_t)

#define PNANOVDB_NODE2_TABLE_COUNT 32768
#define PNANOVDB_NODE2_SIZE 8224

#define PNANOVDB_NODE2_OFF_BBOX_MIN 0
#define PNANOVDB_NODE2_OFF_BBOX_MAX 12
#define PNANOVDB_NODE2_OFF_OFFSET 24
#define PNANOVDB_NODE2_OFF_FLAGS 28
#define PNANOVDB_NODE2_OFF_VALUE_MASK 32
#define PNANOVDB_NODE2_OFF_CHILD_MASK 4128

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node2_get_bbox_min(pnanovdb_buf_t buf, pnanovdb_node2_handle_t p) {
	return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE2_OFF_BBOX_MIN));
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node2_get_bbox_max(pnanovdb_buf_t buf, pnanovdb_node2_handle_t p) {
	return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE2_OFF_BBOX_MAX));
}
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_node2_get_offset(pnanovdb_buf_t buf, pnanovdb_node2_handle_t p) {
	return pnanovdb_read_int32(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE2_OFF_OFFSET));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_get_flags(pnanovdb_buf_t buf, pnanovdb_node2_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE2_OFF_FLAGS));
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_node2_get_value_mask(pnanovdb_buf_t buf, pnanovdb_node2_handle_t p, pnanovdb_uint32_t bit_index) {
	pnanovdb_uint32_t value = pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE2_OFF_VALUE_MASK + 4u * (bit_index >> 5u)));
	return ((value >> (bit_index & 31u)) & 1) != 0u;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_node2_get_child_mask(pnanovdb_buf_t buf, pnanovdb_node2_handle_t p, pnanovdb_uint32_t bit_index) {
	pnanovdb_uint32_t value = pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE2_OFF_CHILD_MASK + 4u * (bit_index >> 5u)));
	return ((value >> (bit_index & 31u)) & 1) != 0u;
}

struct pnanovdb_node1_t
{
	pnanovdb_coord_t bbox_min;
	pnanovdb_coord_t bbox_max;
	pnanovdb_int32_t offset;
	pnanovdb_uint32_t flags;
	pnanovdb_uint32_t value_mask[128];
	pnanovdb_uint32_t child_mask[128];
	// min, max
	// alignas(32) pnanovdb_uint32_t table[];
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_node1_t)
struct pnanovdb_node1_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_node1_handle_t)

#define PNANOVDB_NODE1_TABLE_COUNT 4096
#define PNANOVDB_NODE1_SIZE 1056

#define PNANOVDB_NODE1_OFF_BBOX_MIN 0
#define PNANOVDB_NODE1_OFF_BBOX_MAX 12
#define PNANOVDB_NODE1_OFF_OFFSET 24
#define PNANOVDB_NODE1_OFF_FLAGS 28
#define PNANOVDB_NODE1_OFF_VALUE_MASK 32
#define PNANOVDB_NODE1_OFF_CHILD_MASK 544

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node1_get_bbox_min(pnanovdb_buf_t buf, pnanovdb_node1_handle_t p) {
	return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE1_OFF_BBOX_MIN));
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node1_get_bbox_max(pnanovdb_buf_t buf, pnanovdb_node1_handle_t p) {
	return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE1_OFF_BBOX_MAX));
}
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_node1_get_offset(pnanovdb_buf_t buf, pnanovdb_node1_handle_t p) {
	return pnanovdb_read_int32(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE1_OFF_OFFSET));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node1_get_flags(pnanovdb_buf_t buf, pnanovdb_node1_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE1_OFF_FLAGS));
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_node1_get_value_mask(pnanovdb_buf_t buf, pnanovdb_node1_handle_t p, pnanovdb_uint32_t bit_index) {
	pnanovdb_uint32_t value = pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE1_OFF_VALUE_MASK + 4u * (bit_index >> 5u)));
	return ((value >> (bit_index & 31u)) & 1) != 0u;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_node1_get_child_mask(pnanovdb_buf_t buf, pnanovdb_node1_handle_t p, pnanovdb_uint32_t bit_index) {
	pnanovdb_uint32_t value = pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE1_OFF_CHILD_MASK + 4u * (bit_index >> 5u)));
	return ((value >> (bit_index & 31u)) & 1) != 0u;
}

struct pnanovdb_node0_t
{
	pnanovdb_coord_t bbox_min;
	pnanovdb_uint32_t bbox_dif_and_flags;
	pnanovdb_uint32_t value_mask[16];
	// min, max
	// alignas(32) pnanovdb_uint32_t values[];
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_node0_t)
struct pnanovdb_node0_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_node0_handle_t)

#define PNANOVDB_NODE0_TABLE_COUNT 512
#define PNANOVDB_NODE0_SIZE 80

#define PNANOVDB_NODE0_OFF_BBOX_MIN 0
#define PNANOVDB_NODE0_OFF_BBOX_DIF_AND_FLAGS 12
#define PNANOVDB_NODE0_OFF_VALUE_MASK 16

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node0_get_bbox_min(pnanovdb_buf_t buf, pnanovdb_node0_handle_t p) {
	return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE0_OFF_BBOX_MIN));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node0_get_bbox_dif_and_flags(pnanovdb_buf_t buf, pnanovdb_node0_handle_t p) {
	return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE0_OFF_BBOX_DIF_AND_FLAGS));
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_node0_get_value_mask(pnanovdb_buf_t buf, pnanovdb_node0_handle_t p, pnanovdb_uint32_t bit_index) {
	pnanovdb_uint32_t value = pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_NODE0_OFF_VALUE_MASK + 4u * (bit_index >> 5u)));
	return ((value >> (bit_index & 31u)) & 1) != 0u;
}

struct pnanovdb_grid_type_constants_t
{
	pnanovdb_uint32_t root_off_background;
	pnanovdb_uint32_t root_off_min;
	pnanovdb_uint32_t root_off_max;
	pnanovdb_uint32_t root_off_ave;
	pnanovdb_uint32_t root_off_stddev;
	pnanovdb_uint32_t root_size;
	pnanovdb_uint32_t value_stride_bits;
	pnanovdb_uint32_t table_stride;
	pnanovdb_uint32_t root_tile_off_value;
	pnanovdb_uint32_t root_tile_size;
	pnanovdb_uint32_t node2_off_min;
	pnanovdb_uint32_t node2_off_max;
	pnanovdb_uint32_t node2_off_ave;
	pnanovdb_uint32_t node2_off_stddev;
	pnanovdb_uint32_t node2_off_table;
	pnanovdb_uint32_t node2_size;
	pnanovdb_uint32_t node1_off_min;
	pnanovdb_uint32_t node1_off_max;
	pnanovdb_uint32_t node1_off_ave;
	pnanovdb_uint32_t node1_off_stddev;
	pnanovdb_uint32_t node1_off_table;
	pnanovdb_uint32_t node1_size;
	pnanovdb_uint32_t node0_off_min;
	pnanovdb_uint32_t node0_off_max;
	pnanovdb_uint32_t node0_off_ave;
	pnanovdb_uint32_t node0_off_stddev;
	pnanovdb_uint32_t node0_off_table;
	pnanovdb_uint32_t node0_size;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_grid_type_constants_t)

PNANOVDB_STATIC_CONST pnanovdb_grid_type_constants_t pnanovdb_grid_type_constants[PNANOVDB_GRID_TYPE_END] =
{
	{36, 36, 36, 36, 36, 64,  0, 4, 16, 32,  8224, 8224, 8224, 8224, 8224, 139296,  1056, 1056, 1056, 1056, 1056, 17440,  80, 80, 80, 80, 96, 96},
    {36, 40, 44, 48, 52, 64,  32, 4, 16, 32,  8224, 8228, 8232, 8236, 8256, 139328,  1056, 1060, 1064, 1068, 1088, 17472,  80, 84, 88, 92, 96, 2144},
    {40, 48, 56, 64, 72, 96,  64, 8, 16, 32,  8224, 8232, 8240, 8248, 8256, 270400,  1056, 1064, 1072, 1080, 1088, 33856,  80, 88, 96, 104, 128, 4224},
    {36, 38, 40, 44, 48, 64,  16, 4, 16, 32,  8224, 8226, 8228, 8232, 8256, 139328,  1056, 1058, 1060, 1064, 1088, 17472,  80, 82, 84, 88, 96, 1120},
    {36, 40, 44, 48, 52, 64,  32, 4, 16, 32,  8224, 8228, 8232, 8236, 8256, 139328,  1056, 1060, 1064, 1068, 1088, 17472,  80, 84, 88, 92, 96, 2144},
    {40, 48, 56, 64, 72, 96,  64, 8, 16, 32,  8224, 8232, 8240, 8248, 8256, 270400,  1056, 1064, 1072, 1080, 1088, 33856,  80, 88, 96, 104, 128, 4224},
    {36, 48, 60, 72, 76, 96,  96, 12, 16, 32,  8224, 8236, 8248, 8252, 8256, 401472,  1056, 1068, 1080, 1084, 1088, 50240,  80, 92, 104, 108, 128, 6272},
    {40, 64, 88, 112, 120, 128,  192, 24, 16, 64,  8224, 8248, 8272, 8280, 8288, 794720,  1056, 1080, 1104, 1112, 1120, 99424,  80, 104, 128, 136, 160, 12448},
    {36, 37, 38, 39, 40, 64,  0, 4, 16, 32,  8224, 8225, 8226, 8227, 8256, 139328,  1056, 1057, 1058, 1059, 1088, 17472,  80, 80, 80, 80, 96, 96},
    {36, 38, 40, 44, 48, 64,  16, 4, 16, 32,  8224, 8226, 8228, 8232, 8256, 139328,  1056, 1058, 1060, 1064, 1088, 17472,  80, 82, 84, 88, 96, 1120},
    {36, 40, 44, 48, 52, 64,  32, 4, 16, 32,  8224, 8228, 8232, 8236, 8256, 139328,  1056, 1060, 1064, 1068, 1088, 17472,  80, 84, 88, 92, 96, 2144},
    {36, 37, 38, 39, 40, 64,  1, 4, 16, 32,  8224, 8225, 8226, 8227, 8256, 139328,  1056, 1057, 1058, 1059, 1088, 17472,  80, 80, 80, 80, 96, 160},
    {36, 36, 36, 36, 36, 64,  0, 4, 16, 32,  8224, 8224, 8224, 8224, 8224, 139296,  1056, 1056, 1056, 1056, 1056, 17440,  80, 80, 80, 80, 96, 96},
};

// ------------------------------------------------ Basic Lookup -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_gridblindmetadata_handle_t pnanovdb_grid_get_gridblindmetadata(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, pnanovdb_uint32_t index)
{
	pnanovdb_gridblindmetadata_handle_t meta = { grid.address };
	pnanovdb_uint64_t byte_offset = pnanovdb_grid_get_blind_metadata_offset(buf, grid);
	meta.address = pnanovdb_address_offset64(meta.address, byte_offset);
	meta.address = pnanovdb_address_offset_product(meta.address, PNANOVDB_GRIDBLINDMETADATA_SIZE, index);
	return meta;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanodvb_grid_get_gridblindmetadata_value_address(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, pnanovdb_uint32_t index)
{
	pnanovdb_gridblindmetadata_handle_t meta = pnanovdb_grid_get_gridblindmetadata(buf, grid, index);
	pnanovdb_int64_t byte_offset = pnanovdb_gridblindmetadata_get_byte_offset(buf, meta);
	pnanovdb_address_t address = grid.address;
	address = pnanovdb_address_offset64(address, pnanovdb_int64_as_uint64(byte_offset));
	return address;
}

PNANOVDB_FORCE_INLINE pnanovdb_tree_handle_t pnanovdb_grid_get_tree(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid)
{
	pnanovdb_tree_handle_t tree = { grid.address };
	tree.address = pnanovdb_address_offset(tree.address, PNANOVDB_GRID_SIZE);
	return tree;
}

PNANOVDB_FORCE_INLINE pnanovdb_root_handle_t pnanovdb_tree_get_root(pnanovdb_buf_t buf, pnanovdb_tree_handle_t tree)
{
	pnanovdb_root_handle_t root = { tree.address };
	pnanovdb_uint64_t byte_offset = pnanovdb_tree_get_bytes3(buf, tree);
	root.address = pnanovdb_address_offset64(root.address, byte_offset);
	return root;
}

PNANOVDB_FORCE_INLINE pnanovdb_root_tile_handle_t pnanovdb_root_get_tile(pnanovdb_grid_type_t grid_type, pnanovdb_root_handle_t root, pnanovdb_uint32_t n)
{
	pnanovdb_root_tile_handle_t tile = { root.address };
	tile.address = pnanovdb_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_size));
	tile.address = pnanovdb_address_offset_product(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size), n);
	return tile;
}

PNANOVDB_FORCE_INLINE pnanovdb_root_tile_handle_t pnanovdb_root_get_tile_zero(pnanovdb_grid_type_t grid_type, pnanovdb_root_handle_t root)
{
	pnanovdb_root_tile_handle_t tile = { root.address };
	tile.address = pnanovdb_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_size));
	return tile;
}

PNANOVDB_FORCE_INLINE pnanovdb_node2_handle_t pnanovdb_root_get_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, pnanovdb_root_tile_handle_t tile)
{
	pnanovdb_node2_handle_t node2 = { root.address };
	node2.address = pnanovdb_address_offset(node2.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_size));
	node2.address = pnanovdb_address_offset_product(node2.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size), pnanovdb_root_get_tile_count(buf, root));
	node2.address = pnanovdb_address_offset_product(node2.address, PNANOVDB_GRID_TYPE_GET(grid_type, node2_size), pnanovdb_root_tile_get_child_id(buf, tile));
	return node2;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_coord_to_key(PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
#if defined(PNANOVDB_NATIVE_64)
	pnanovdb_uint64_t iu = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).x) >> 12u;
	pnanovdb_uint64_t ju = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).y) >> 12u;
	pnanovdb_uint64_t ku = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).z) >> 12u;
	return (ku) | (ju << 21u) | (iu << 42u);
#else
	pnanovdb_uint32_t iu = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).x) >> 12u;
	pnanovdb_uint32_t ju = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).y) >> 12u;
	pnanovdb_uint32_t ku = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).z) >> 12u;
	pnanovdb_uint32_t key_x = ku | (ju << 21);
	pnanovdb_uint32_t key_y = (iu << 10) | (ju >> 11);
	return pnanovdb_uint32_as_uint64(key_x, key_y);
#endif
}

PNANOVDB_FORCE_INLINE pnanovdb_root_tile_handle_t pnanovdb_root_find_tile(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	pnanovdb_uint32_t tile_count = pnanovdb_uint32_as_int32(pnanovdb_root_get_tile_count(buf, root));
	pnanovdb_root_tile_handle_t tile = pnanovdb_root_get_tile_zero(grid_type, root);
	pnanovdb_uint64_t key = pnanovdb_coord_to_key(ijk);
	for (pnanovdb_uint32_t i = 0u; i < tile_count; i++)
	{
		if (pnanovdb_uint64_is_equal(key, pnanovdb_root_tile_get_key(buf, tile)))
		{
			return tile;
		}
		tile.address = pnanovdb_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size));
	}
	pnanovdb_root_tile_handle_t null_handle = { pnanovdb_address_null() };
	return null_handle;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node0_coord_to_offset(PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	return (((PNANOVDB_DEREF(ijk).x & 7) >> 0) << (2 * 3)) +
		(((PNANOVDB_DEREF(ijk).y & 7) >> 0) << (3)) +
		((PNANOVDB_DEREF(ijk).z & 7) >> 0);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node0_get_min_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node0_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node0_off_min);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node0_get_max_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node0_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node0_off_max);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node0_get_ave_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node0_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node0_off_ave);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node0_get_stdddev_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node0_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node0_off_stddev);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node0_get_table_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node0_handle_t node, pnanovdb_uint32_t n)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node0_off_table) + ((PNANOVDB_GRID_TYPE_GET(grid_type, value_stride_bits) * n) >> 3u);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node0_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node0_handle_t node0, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	pnanovdb_uint32_t n = pnanovdb_node0_coord_to_offset(ijk);
	return pnanovdb_node0_get_table_address(grid_type, buf, node0, n);
}


PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node1_coord_to_offset(PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	return (((PNANOVDB_DEREF(ijk).x & 127) >> 3) << (2 * 4)) +
		(((PNANOVDB_DEREF(ijk).y & 127) >> 3) << (4)) +
		((PNANOVDB_DEREF(ijk).z & 127) >> 3);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node1_get_min_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node1_off_min);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node1_get_max_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node1_off_max);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node1_get_ave_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node1_off_ave);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node1_get_stddev_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node1_off_stddev);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node1_get_table_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node, pnanovdb_uint32_t n)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node1_off_table) + PNANOVDB_GRID_TYPE_GET(grid_type, table_stride) * n;
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node1_get_table_child_id(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node, pnanovdb_uint32_t n)
{
	pnanovdb_address_t table_address = pnanovdb_node1_get_table_address(grid_type, buf, node, n);
	return pnanovdb_read_uint32(buf, table_address);
}

PNANOVDB_FORCE_INLINE pnanovdb_node0_handle_t pnanovdb_node1_get_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node1, pnanovdb_uint32_t n)
{
	pnanovdb_node0_handle_t node0 = { node1.address };
	node0.address = pnanovdb_address_offset_product(node0.address, PNANOVDB_GRID_TYPE_GET(grid_type, node1_size), pnanovdb_node1_get_offset(buf, node1));
	node0.address = pnanovdb_address_offset_product(node0.address, PNANOVDB_GRID_TYPE_GET(grid_type, node0_size), pnanovdb_node1_get_table_child_id(grid_type, buf, node1, n));
	return node0;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node1_get_value_address_and_level(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node1, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
	pnanovdb_uint32_t n = pnanovdb_node1_coord_to_offset(ijk);
	pnanovdb_address_t value_address;
	if (pnanovdb_node1_get_child_mask(buf, node1, n))
	{
		pnanovdb_node0_handle_t child = pnanovdb_node1_get_child(grid_type, buf, node1, n);
		value_address = pnanovdb_node0_get_value_address(grid_type, buf, child, ijk);
		PNANOVDB_DEREF(level) = 0u;
	}
	else
	{
		value_address = pnanovdb_node1_get_table_address(grid_type, buf, node1, n);
		PNANOVDB_DEREF(level) = 1u;
	}
	return value_address;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node1_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node1, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	pnanovdb_uint32_t level;
	return pnanovdb_node1_get_value_address_and_level(grid_type, buf, node1, ijk, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_coord_to_offset(PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	return (((PNANOVDB_DEREF(ijk).x & 4095) >> 7) << (2 * 5)) +
		(((PNANOVDB_DEREF(ijk).y & 4095) >> 7) << (5)) +
		((PNANOVDB_DEREF(ijk).z & 4095) >> 7);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node2_get_min_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node2_off_min);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node2_get_max_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node2_off_max);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node2_get_ave_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node2_off_ave);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node2_get_stddev_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node2_off_stddev);
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node2_get_table_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node, pnanovdb_uint32_t n)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, node2_off_table) + PNANOVDB_GRID_TYPE_GET(grid_type, table_stride) * n;
	return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_get_table_child_id(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node, pnanovdb_uint32_t n)
{
	pnanovdb_address_t bufAddress = pnanovdb_node2_get_table_address(grid_type, buf, node, n);
	return pnanovdb_read_uint32(buf, bufAddress);
}

PNANOVDB_FORCE_INLINE pnanovdb_node1_handle_t pnanovdb_node2_get_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node2, pnanovdb_uint32_t n)
{
	pnanovdb_node1_handle_t node1 = { node2.address };
	node1.address = pnanovdb_address_offset_product(node1.address, PNANOVDB_GRID_TYPE_GET(grid_type, node2_size), pnanovdb_node2_get_offset(buf, node2));
	node1.address = pnanovdb_address_offset_product(node1.address, PNANOVDB_GRID_TYPE_GET(grid_type, node1_size), pnanovdb_node2_get_table_child_id(grid_type, buf, node2, n));
	return node1;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node2_get_value_address_and_level(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node2, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
	pnanovdb_uint32_t n = pnanovdb_node2_coord_to_offset(ijk);
	pnanovdb_address_t value_address;
	if (pnanovdb_node2_get_child_mask(buf, node2, n))
	{
		pnanovdb_node1_handle_t child = pnanovdb_node2_get_child(grid_type, buf, node2, n);
		value_address = pnanovdb_node1_get_value_address_and_level(grid_type, buf, child, ijk, level);
	}
	else
	{
		value_address = pnanovdb_node2_get_table_address(grid_type, buf, node2, n);
		PNANOVDB_DEREF(level) = 2u;
	}
	return value_address;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node2_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node2, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	pnanovdb_uint32_t level;
	return pnanovdb_node2_get_value_address_and_level(grid_type, buf, node2, ijk, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_min_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, root_off_min);
	return pnanovdb_address_offset(root.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_max_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, root_off_max);
	return pnanovdb_address_offset(root.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_ave_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, root_off_ave);
	return pnanovdb_address_offset(root.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_stddev_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, root_off_stddev);
	return pnanovdb_address_offset(root.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_tile_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t root_tile)
{
	pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_off_value);
	return pnanovdb_address_offset(root_tile.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_value_address_and_level(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
	pnanovdb_root_tile_handle_t tile = pnanovdb_root_find_tile(grid_type, buf, root, ijk);
	pnanovdb_address_t ret;
	if (pnanovdb_address_is_null(tile.address))
	{
		ret = pnanovdb_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_off_background));
		PNANOVDB_DEREF(level) = 4u;
	}
	else if (pnanovdb_root_tile_get_child_id(buf, tile) < 0)
	{
		ret = pnanovdb_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_off_value));
		PNANOVDB_DEREF(level) = 3u;
	}
	else
	{
		pnanovdb_node2_handle_t child = pnanovdb_root_get_child(grid_type, buf, root, tile);
		ret = pnanovdb_node2_get_value_address_and_level(grid_type, buf, child, ijk, level);
	}
	return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	pnanovdb_uint32_t level;
	return pnanovdb_root_get_value_address_and_level(grid_type, buf, root, ijk, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_value_address_bit(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) bit_index)
{
	pnanovdb_uint32_t level;
	pnanovdb_address_t address = pnanovdb_root_get_value_address_and_level(grid_type, buf, root, ijk, PNANOVDB_REF(level));
	PNANOVDB_DEREF(bit_index) = level == 0u ? pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).x & 7) : 0u;
	return address;
}

// ------------------------------------------------ ReadAccessor -----------------------------------------------------------

struct pnanovdb_readaccessor_t
{
	pnanovdb_coord_t key;
	pnanovdb_node0_handle_t node0;
	pnanovdb_node1_handle_t node1;
	pnanovdb_node2_handle_t node2;
	pnanovdb_root_handle_t root;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_readaccessor_t)

PNANOVDB_FORCE_INLINE void pnanovdb_readaccessor_init(PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, pnanovdb_root_handle_t root)
{
	PNANOVDB_DEREF(acc).key.x = 0x7FFFFFFF;
	PNANOVDB_DEREF(acc).key.y = 0x7FFFFFFF;
	PNANOVDB_DEREF(acc).key.z = 0x7FFFFFFF;
	PNANOVDB_DEREF(acc).node0.address = pnanovdb_address_null();
	PNANOVDB_DEREF(acc).node1.address = pnanovdb_address_null();
	PNANOVDB_DEREF(acc).node2.address = pnanovdb_address_null();
	PNANOVDB_DEREF(acc).root = root;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_readaccessor_iscached0(PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, int dirty)
{
	if (pnanovdb_address_is_null(PNANOVDB_DEREF(acc).node0.address)) { return PNANOVDB_FALSE; }
	if ((dirty & ~((1u << 3) - 1u)) != 0)
	{
		PNANOVDB_DEREF(acc).node0.address = pnanovdb_address_null();
		return PNANOVDB_FALSE;
	}
	return PNANOVDB_TRUE;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_readaccessor_iscached1(PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, int dirty)
{
	if (pnanovdb_address_is_null(PNANOVDB_DEREF(acc).node1.address)) { return PNANOVDB_FALSE; }
	if ((dirty & ~((1u << 7) - 1u)) != 0)
	{
		PNANOVDB_DEREF(acc).node1.address = pnanovdb_address_null();
		return PNANOVDB_FALSE;
	}
	return PNANOVDB_TRUE;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_readaccessor_iscached2(PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, int dirty)
{
	if (pnanovdb_address_is_null(PNANOVDB_DEREF(acc).node2.address)) { return PNANOVDB_FALSE; }
	if ((dirty & ~((1u << 12) - 1u)) != 0)
	{
		PNANOVDB_DEREF(acc).node2.address = pnanovdb_address_null();
		return PNANOVDB_FALSE;
	}
	return PNANOVDB_TRUE;
}
PNANOVDB_FORCE_INLINE int pnanovdb_readaccessor_computedirty(PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	return (PNANOVDB_DEREF(ijk).x ^ PNANOVDB_DEREF(acc).key.x) | (PNANOVDB_DEREF(ijk).y ^ PNANOVDB_DEREF(acc).key.y) | (PNANOVDB_DEREF(ijk).z ^ PNANOVDB_DEREF(acc).key.z);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node0_get_value_address_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node0_handle_t node0, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_uint32_t n = pnanovdb_node0_coord_to_offset(ijk);
	return pnanovdb_node0_get_table_address(grid_type, buf, node0, n);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node1_get_value_address_and_level_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node1, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
	pnanovdb_uint32_t n = pnanovdb_node1_coord_to_offset(ijk);
	pnanovdb_address_t value_address;
	if (pnanovdb_node1_get_child_mask(buf, node1, n))
	{
		pnanovdb_node0_handle_t child = pnanovdb_node1_get_child(grid_type, buf, node1, n);
		PNANOVDB_DEREF(acc).node0 = child;
		PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
		value_address = pnanovdb_node0_get_value_address_and_cache(grid_type, buf, child, ijk, acc);
		PNANOVDB_DEREF(level) = 0u;
	}
	else
	{
		value_address = pnanovdb_node1_get_table_address(grid_type, buf, node1, n);
		PNANOVDB_DEREF(level) = 1u;
	}
	return value_address;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node1_get_value_address_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node1, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_uint32_t level;
	return pnanovdb_node1_get_value_address_and_level_and_cache(grid_type, buf, node1, ijk, acc, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node2_get_value_address_and_level_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node2, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
	pnanovdb_uint32_t n = pnanovdb_node2_coord_to_offset(ijk);
	pnanovdb_address_t value_address;
	if (pnanovdb_node2_get_child_mask(buf, node2, n))
	{
		pnanovdb_node1_handle_t child = pnanovdb_node2_get_child(grid_type, buf, node2, n);
		PNANOVDB_DEREF(acc).node1 = child;
		PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
		value_address = pnanovdb_node1_get_value_address_and_level_and_cache(grid_type, buf, child, ijk, acc, level);
	}
	else
	{
		value_address = pnanovdb_node2_get_table_address(grid_type, buf, node2, n);
		PNANOVDB_DEREF(level) = 2u;
	}
	return value_address;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_node2_get_value_address_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node2, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_uint32_t level;
	return pnanovdb_node2_get_value_address_and_level_and_cache(grid_type, buf, node2, ijk, acc, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_value_address_and_level_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
	pnanovdb_root_tile_handle_t tile = pnanovdb_root_find_tile(grid_type, buf, root, ijk);
	pnanovdb_address_t ret;
	if (pnanovdb_address_is_null(tile.address))
	{
		ret = pnanovdb_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_off_background));
		PNANOVDB_DEREF(level) = 4u;
	}
	else if (pnanovdb_root_tile_get_child_id(buf, tile) < 0)
	{
		ret = pnanovdb_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_off_value));
		PNANOVDB_DEREF(level) = 3u;
	}
	else
	{
		pnanovdb_node2_handle_t child = pnanovdb_root_get_child(grid_type, buf, root, tile);
		PNANOVDB_DEREF(acc).node2 = child;
		PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
		ret = pnanovdb_node2_get_value_address_and_level_and_cache(grid_type, buf, child, ijk, acc, level);
	}
	return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_value_address_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_uint32_t level;
	return pnanovdb_root_get_value_address_and_level_and_cache(grid_type, buf, root, ijk, acc, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_readaccessor_get_value_address_and_level(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
	int dirty = pnanovdb_readaccessor_computedirty(acc, ijk);

	pnanovdb_address_t value_address;
	if (pnanovdb_readaccessor_iscached0(acc, dirty))
	{
		value_address = pnanovdb_node0_get_value_address_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).node0, ijk, acc);
		PNANOVDB_DEREF(level) = 0u;
	}
	else if (pnanovdb_readaccessor_iscached1(acc, dirty))
	{
		value_address = pnanovdb_node1_get_value_address_and_level_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).node1, ijk, acc, level);
	}
	else if (pnanovdb_readaccessor_iscached2(acc, dirty))
	{
		value_address = pnanovdb_node2_get_value_address_and_level_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).node2, ijk, acc, level);
	}
	else
	{
		value_address = pnanovdb_root_get_value_address_and_level_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).root, ijk, acc, level);
	}
	return value_address;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_readaccessor_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	pnanovdb_uint32_t level;
	return pnanovdb_readaccessor_get_value_address_and_level(grid_type, buf, acc, ijk, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_readaccessor_get_value_address_bit(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) bit_index)
{
	pnanovdb_uint32_t level;
	pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address_and_level(grid_type, buf, acc, ijk, PNANOVDB_REF(level));
	PNANOVDB_DEREF(bit_index) = level == 0u ? pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).x & 7) : 0u;
	return address;
}

// ------------------------------------------------ ReadAccessor GetDim -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node0_get_dim_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node0_handle_t node0, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_uint32_t n = pnanovdb_node0_coord_to_offset(ijk);
	return 1u;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node1_get_dim_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node1, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_uint32_t n = pnanovdb_node1_coord_to_offset(ijk);
	pnanovdb_uint32_t ret;
	if (pnanovdb_node1_get_child_mask(buf, node1, n))
	{
		pnanovdb_node0_handle_t child = pnanovdb_node1_get_child(grid_type, buf, node1, n);
		PNANOVDB_DEREF(acc).node0 = child;
		PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
		ret = pnanovdb_node0_get_dim_and_cache(grid_type, buf, child, ijk, acc);
	}
	else
	{
		ret = (1u << (3u)); // node 0 dim
	}
	return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_get_dim_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node2, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_uint32_t n = pnanovdb_node2_coord_to_offset(ijk);
	pnanovdb_uint32_t ret;
	if (pnanovdb_node2_get_child_mask(buf, node2, n))
	{
		pnanovdb_node1_handle_t child = pnanovdb_node2_get_child(grid_type, buf, node2, n);
		PNANOVDB_DEREF(acc).node1 = child;
		PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
		ret = pnanovdb_node1_get_dim_and_cache(grid_type, buf, child, ijk, acc);
	}
	else
	{
		ret = (1u << (4u + 3u)); // node 1 dim
	}
	return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_root_get_dim_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_root_tile_handle_t tile = pnanovdb_root_find_tile(grid_type, buf, root, ijk);
	pnanovdb_uint32_t ret;
	if (pnanovdb_address_is_null(tile.address))
	{
		ret = 1u << (5u + 4u + 3u); // background, node 2 dim
	}
	else if (pnanovdb_root_tile_get_child_id(buf, tile) < 0)
	{
		ret = 1u << (5u + 4u + 3u); // tile value, node 2 dim
	}
	else
	{
		pnanovdb_node2_handle_t child = pnanovdb_root_get_child(grid_type, buf, root, tile);
		PNANOVDB_DEREF(acc).node2 = child;
		PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
		ret = pnanovdb_node2_get_dim_and_cache(grid_type, buf, child, ijk, acc);
	}
	return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_readaccessor_get_dim(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	int dirty = pnanovdb_readaccessor_computedirty(acc, ijk);

	pnanovdb_uint32_t dim;
	if (pnanovdb_readaccessor_iscached0(acc, dirty))
	{
		dim = pnanovdb_node0_get_dim_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).node0, ijk, acc);
	}
	else if (pnanovdb_readaccessor_iscached1(acc, dirty))
	{
		dim = pnanovdb_node1_get_dim_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).node1, ijk, acc);
	}
	else if (pnanovdb_readaccessor_iscached2(acc, dirty))
	{
		dim = pnanovdb_node2_get_dim_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).node2, ijk, acc);
	}
	else
	{
		dim = pnanovdb_root_get_dim_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).root, ijk, acc);
	}
	return dim;
}

// ------------------------------------------------ ReadAccessor IsActive -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_node0_is_active_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node0_handle_t node0, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_uint32_t n = pnanovdb_node0_coord_to_offset(ijk);
	return pnanovdb_node0_get_value_mask(buf, node0, n);
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_node1_is_active_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node1_handle_t node1, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_uint32_t n = pnanovdb_node1_coord_to_offset(ijk);
	pnanovdb_bool_t is_active;
	if (pnanovdb_node1_get_child_mask(buf, node1, n))
	{
		pnanovdb_node0_handle_t child = pnanovdb_node1_get_child(grid_type, buf, node1, n);
		PNANOVDB_DEREF(acc).node0 = child;
		PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
		is_active = pnanovdb_node0_is_active_and_cache(grid_type, buf, child, ijk, acc);
	}
	else
	{
		is_active = pnanovdb_node1_get_value_mask(buf, node1, n);
	}
	return is_active;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_node2_is_active_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_node2_handle_t node2, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_uint32_t n = pnanovdb_node2_coord_to_offset(ijk);
	pnanovdb_bool_t is_active;
	if (pnanovdb_node2_get_child_mask(buf, node2, n))
	{
		pnanovdb_node1_handle_t child = pnanovdb_node2_get_child(grid_type, buf, node2, n);
		PNANOVDB_DEREF(acc).node1 = child;
		PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
		is_active = pnanovdb_node1_is_active_and_cache(grid_type, buf, child, ijk, acc);
	}
	else
	{
		is_active = pnanovdb_node2_get_value_mask(buf, node2, n);
	}
	return is_active;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_root_is_active_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
	pnanovdb_root_tile_handle_t tile = pnanovdb_root_find_tile(grid_type, buf, root, ijk);
	pnanovdb_bool_t is_active;
	if (pnanovdb_address_is_null(tile.address))
	{
		is_active = PNANOVDB_FALSE; // background
	}
	else if (pnanovdb_root_tile_get_child_id(buf, tile) < 0)
	{
		pnanovdb_uint32_t state = pnanovdb_root_tile_get_state(buf, tile);
		is_active = state != 0u; // tile value
	}
	else
	{
		pnanovdb_node2_handle_t child = pnanovdb_root_get_child(grid_type, buf, root, tile);
		PNANOVDB_DEREF(acc).node2 = child;
		PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
		is_active = pnanovdb_node2_is_active_and_cache(grid_type, buf, child, ijk, acc);
	}
	return is_active;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_readaccessor_is_active(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
	int dirty = pnanovdb_readaccessor_computedirty(acc, ijk);

	pnanovdb_bool_t is_active;
	if (pnanovdb_readaccessor_iscached0(acc, dirty))
	{
		is_active = pnanovdb_node0_is_active_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).node0, ijk, acc);
	}
	else if (pnanovdb_readaccessor_iscached1(acc, dirty))
	{
		is_active = pnanovdb_node1_is_active_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).node1, ijk, acc);
	}
	else if (pnanovdb_readaccessor_iscached2(acc, dirty))
	{
		is_active = pnanovdb_node2_is_active_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).node2, ijk, acc);
	}
	else
	{
		is_active = pnanovdb_root_is_active_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).root, ijk, acc);
	}
	return is_active;
}

// ------------------------------------------------ Map Transforms -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_map_apply(pnanovdb_buf_t buf, pnanovdb_map_handle_t map, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
	pnanovdb_vec3_t  dst;
	float sx = PNANOVDB_DEREF(src).x;
	float sy = PNANOVDB_DEREF(src).y;
	float sz = PNANOVDB_DEREF(src).z;
	dst.x = sx * pnanovdb_map_get_matf(buf, map, 0) + sy * pnanovdb_map_get_matf(buf, map, 1) + sz * pnanovdb_map_get_matf(buf, map, 2) + pnanovdb_map_get_vecf(buf, map, 0);
	dst.y = sx * pnanovdb_map_get_matf(buf, map, 3) + sy * pnanovdb_map_get_matf(buf, map, 4) + sz * pnanovdb_map_get_matf(buf, map, 5) + pnanovdb_map_get_vecf(buf, map, 1);
	dst.z = sx * pnanovdb_map_get_matf(buf, map, 6) + sy * pnanovdb_map_get_matf(buf, map, 7) + sz * pnanovdb_map_get_matf(buf, map, 8) + pnanovdb_map_get_vecf(buf, map, 2);
	return dst;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_map_apply_inverse(pnanovdb_buf_t buf, pnanovdb_map_handle_t map, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
	pnanovdb_vec3_t  dst;
	float sx = PNANOVDB_DEREF(src).x - pnanovdb_map_get_vecf(buf, map, 0);
	float sy = PNANOVDB_DEREF(src).y - pnanovdb_map_get_vecf(buf, map, 1);
	float sz = PNANOVDB_DEREF(src).z - pnanovdb_map_get_vecf(buf, map, 2);
	dst.x = sx * pnanovdb_map_get_invmatf(buf, map, 0) + sy * pnanovdb_map_get_invmatf(buf, map, 1) + sz * pnanovdb_map_get_invmatf(buf, map, 2);
	dst.y = sx * pnanovdb_map_get_invmatf(buf, map, 3) + sy * pnanovdb_map_get_invmatf(buf, map, 4) + sz * pnanovdb_map_get_invmatf(buf, map, 5);
	dst.z = sx * pnanovdb_map_get_invmatf(buf, map, 6) + sy * pnanovdb_map_get_invmatf(buf, map, 7) + sz * pnanovdb_map_get_invmatf(buf, map, 8);
	return dst;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_map_apply_jacobi(pnanovdb_buf_t buf, pnanovdb_map_handle_t map, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
	pnanovdb_vec3_t  dst;
	float sx = PNANOVDB_DEREF(src).x;
	float sy = PNANOVDB_DEREF(src).y;
	float sz = PNANOVDB_DEREF(src).z;
	dst.x = sx * pnanovdb_map_get_matf(buf, map, 0) + sy * pnanovdb_map_get_matf(buf, map, 1) + sz * pnanovdb_map_get_matf(buf, map, 2);
	dst.y = sx * pnanovdb_map_get_matf(buf, map, 3) + sy * pnanovdb_map_get_matf(buf, map, 4) + sz * pnanovdb_map_get_matf(buf, map, 5);
	dst.z = sx * pnanovdb_map_get_matf(buf, map, 6) + sy * pnanovdb_map_get_matf(buf, map, 7) + sz * pnanovdb_map_get_matf(buf, map, 8);
	return dst;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_map_apply_inverse_jacobi(pnanovdb_buf_t buf, pnanovdb_map_handle_t map, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
	pnanovdb_vec3_t  dst;
	float sx = PNANOVDB_DEREF(src).x;
	float sy = PNANOVDB_DEREF(src).y;
	float sz = PNANOVDB_DEREF(src).z;
	dst.x = sx * pnanovdb_map_get_invmatf(buf, map, 0) + sy * pnanovdb_map_get_invmatf(buf, map, 1) + sz * pnanovdb_map_get_invmatf(buf, map, 2);
	dst.y = sx * pnanovdb_map_get_invmatf(buf, map, 3) + sy * pnanovdb_map_get_invmatf(buf, map, 4) + sz * pnanovdb_map_get_invmatf(buf, map, 5);
	dst.z = sx * pnanovdb_map_get_invmatf(buf, map, 6) + sy * pnanovdb_map_get_invmatf(buf, map, 7) + sz * pnanovdb_map_get_invmatf(buf, map, 8);
	return dst;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_grid_world_to_indexf(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
	pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
	return pnanovdb_map_apply_inverse(buf, map, src);
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_grid_index_to_worldf(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
	pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
	return pnanovdb_map_apply(buf, map, src);
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_grid_world_to_index_dirf(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
	pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
	return pnanovdb_map_apply_inverse_jacobi(buf, map, src);
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_grid_index_to_world_dirf(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
	pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
	return pnanovdb_map_apply_jacobi(buf, map, src);
}

// ------------------------------------------------ HDDA -----------------------------------------------------------

#ifdef PNANOVDB_HDDA

// Comment out to disable this explicit round-off check
#define PNANOVDB_ENFORCE_FORWARD_STEPPING

#define PNANOVDB_HDDA_FLOAT_MAX 1e38f

struct pnanovdb_hdda_t
{
	pnanovdb_int32_t dim;
	float tmin;
	float tmax;
	pnanovdb_coord_t voxel;
	pnanovdb_coord_t step;
	pnanovdb_vec3_t delta;
	pnanovdb_vec3_t next;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_hdda_t)

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_hdda_pos_to_ijk(PNANOVDB_IN(pnanovdb_vec3_t) pos)
{
	pnanovdb_coord_t voxel;
	voxel.x = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).x));
	voxel.y = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).y));
	voxel.z = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).z));
	return voxel;
}

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_hdda_pos_to_voxel(PNANOVDB_IN(pnanovdb_vec3_t) pos, int dim)
{
	pnanovdb_coord_t voxel;
	voxel.x = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).x)) & (~(dim - 1));
	voxel.y = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).y)) & (~(dim - 1));
	voxel.z = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).z)) & (~(dim - 1));
	return voxel;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_hdda_ray_start(PNANOVDB_IN(pnanovdb_vec3_t) origin, float tmin, PNANOVDB_IN(pnanovdb_vec3_t) direction)
{
	pnanovdb_vec3_t pos = pnanovdb_vec3_add(
		pnanovdb_vec3_mul(PNANOVDB_DEREF(direction), pnanovdb_vec3_uniform(tmin)),
		PNANOVDB_DEREF(origin)
	);
	return pos;
}

PNANOVDB_FORCE_INLINE void pnanovdb_hdda_init(PNANOVDB_INOUT(pnanovdb_hdda_t) hdda, PNANOVDB_IN(pnanovdb_vec3_t) origin, float tmin, PNANOVDB_IN(pnanovdb_vec3_t) direction, float tmax, int dim)
{
	PNANOVDB_DEREF(hdda).dim = dim;
	PNANOVDB_DEREF(hdda).tmin = tmin;
	PNANOVDB_DEREF(hdda).tmax = tmax;

	pnanovdb_vec3_t pos = pnanovdb_hdda_ray_start(origin, tmin, direction);
	pnanovdb_vec3_t dir_inv = pnanovdb_vec3_div(pnanovdb_vec3_uniform(1.f), PNANOVDB_DEREF(direction));

	PNANOVDB_DEREF(hdda).voxel = pnanovdb_hdda_pos_to_voxel(PNANOVDB_REF(pos), dim);

	// x
	if (PNANOVDB_DEREF(direction).x == 0.f)
	{
		PNANOVDB_DEREF(hdda).next.x = PNANOVDB_HDDA_FLOAT_MAX;
		PNANOVDB_DEREF(hdda).step.x = 0;
		PNANOVDB_DEREF(hdda).delta.x = 0.f;
	}
	else if (dir_inv.x > 0.f)
	{
		PNANOVDB_DEREF(hdda).step.x = 1;
		PNANOVDB_DEREF(hdda).next.x = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.x + dim - pos.x) * dir_inv.x;
		PNANOVDB_DEREF(hdda).delta.x = dir_inv.x;
	}
	else
	{
		PNANOVDB_DEREF(hdda).step.x = -1;
		PNANOVDB_DEREF(hdda).next.x = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.x - pos.x) * dir_inv.x;
		PNANOVDB_DEREF(hdda).delta.x = -dir_inv.x;
	}

	// y
	if (PNANOVDB_DEREF(direction).y == 0.f)
	{
		PNANOVDB_DEREF(hdda).next.y = PNANOVDB_HDDA_FLOAT_MAX;
		PNANOVDB_DEREF(hdda).step.y = 0;
		PNANOVDB_DEREF(hdda).delta.y = 0.f;
	}
	else if (dir_inv.y > 0.f)
	{
		PNANOVDB_DEREF(hdda).step.y = 1;
		PNANOVDB_DEREF(hdda).next.y = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.y + dim - pos.y) * dir_inv.y;
		PNANOVDB_DEREF(hdda).delta.y = dir_inv.y;
	}
	else
	{
		PNANOVDB_DEREF(hdda).step.y = -1;
		PNANOVDB_DEREF(hdda).next.y = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.y - pos.y) * dir_inv.y;
		PNANOVDB_DEREF(hdda).delta.y = -dir_inv.y;
	}

	// z
	if (PNANOVDB_DEREF(direction).z == 0.f)
	{
		PNANOVDB_DEREF(hdda).next.z = PNANOVDB_HDDA_FLOAT_MAX;
		PNANOVDB_DEREF(hdda).step.z = 0;
		PNANOVDB_DEREF(hdda).delta.z = 0.f;
	}
	else if (dir_inv.z > 0.f)
	{
		PNANOVDB_DEREF(hdda).step.z = 1;
		PNANOVDB_DEREF(hdda).next.z = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.z + dim - pos.z) * dir_inv.z;
		PNANOVDB_DEREF(hdda).delta.z = dir_inv.z;
	}
	else
	{
		PNANOVDB_DEREF(hdda).step.z = -1;
		PNANOVDB_DEREF(hdda).next.z = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.z - pos.z) * dir_inv.z;
		PNANOVDB_DEREF(hdda).delta.z = -dir_inv.z;
	}
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_hdda_update(PNANOVDB_INOUT(pnanovdb_hdda_t) hdda, PNANOVDB_IN(pnanovdb_vec3_t) origin, PNANOVDB_IN(pnanovdb_vec3_t) direction, int dim)
{
	if (PNANOVDB_DEREF(hdda).dim == dim)
	{
		return PNANOVDB_FALSE;
	}
	PNANOVDB_DEREF(hdda).dim = dim;

	pnanovdb_vec3_t pos = pnanovdb_vec3_add(
		pnanovdb_vec3_mul(PNANOVDB_DEREF(direction), pnanovdb_vec3_uniform(PNANOVDB_DEREF(hdda).tmin)),
		PNANOVDB_DEREF(origin)
	);
	pnanovdb_vec3_t dir_inv = pnanovdb_vec3_div(pnanovdb_vec3_uniform(1.f), PNANOVDB_DEREF(direction));

	PNANOVDB_DEREF(hdda).voxel = pnanovdb_hdda_pos_to_voxel(PNANOVDB_REF(pos), dim);

	if (PNANOVDB_DEREF(hdda).step.x != 0)
	{
		PNANOVDB_DEREF(hdda).next.x = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.x - pos.x) * dir_inv.x;
		if (PNANOVDB_DEREF(hdda).step.x > 0)
		{
			PNANOVDB_DEREF(hdda).next.x += dim * dir_inv.x;
		}
	}
	if (PNANOVDB_DEREF(hdda).step.y != 0)
	{
		PNANOVDB_DEREF(hdda).next.y = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.y - pos.y) * dir_inv.y;
		if (PNANOVDB_DEREF(hdda).step.y > 0)
		{
			PNANOVDB_DEREF(hdda).next.y += dim * dir_inv.y;
		}
	}
	if (PNANOVDB_DEREF(hdda).step.z != 0)
	{
		PNANOVDB_DEREF(hdda).next.z = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.z - pos.z) * dir_inv.z;
		if (PNANOVDB_DEREF(hdda).step.z > 0)
		{
			PNANOVDB_DEREF(hdda).next.z += dim * dir_inv.z;
		}
	}

	return PNANOVDB_TRUE;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_hdda_step(PNANOVDB_INOUT(pnanovdb_hdda_t) hdda)
{
	pnanovdb_bool_t ret;
	if (PNANOVDB_DEREF(hdda).next.x < PNANOVDB_DEREF(hdda).next.y && PNANOVDB_DEREF(hdda).next.x < PNANOVDB_DEREF(hdda).next.z)
	{
#ifdef PNANOVDB_ENFORCE_FORWARD_STEPPING
		if (PNANOVDB_DEREF(hdda).next.x <= PNANOVDB_DEREF(hdda).tmin)
		{
			PNANOVDB_DEREF(hdda).next.x += PNANOVDB_DEREF(hdda).tmin - 0.999999f * PNANOVDB_DEREF(hdda).next.x + 1.0e-6f;
		}
#endif
		PNANOVDB_DEREF(hdda).tmin = PNANOVDB_DEREF(hdda).next.x;
		PNANOVDB_DEREF(hdda).next.x += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).delta.x;
		PNANOVDB_DEREF(hdda).voxel.x += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).step.x;
		ret = PNANOVDB_DEREF(hdda).tmin <= PNANOVDB_DEREF(hdda).tmax;
	}
	else if (PNANOVDB_DEREF(hdda).next.y < PNANOVDB_DEREF(hdda).next.z)
	{
#ifdef PNANOVDB_ENFORCE_FORWARD_STEPPING
		if (PNANOVDB_DEREF(hdda).next.y <= PNANOVDB_DEREF(hdda).tmin)
		{
			PNANOVDB_DEREF(hdda).next.y += PNANOVDB_DEREF(hdda).tmin - 0.999999f * PNANOVDB_DEREF(hdda).next.y + 1.0e-6f;
		}
#endif
		PNANOVDB_DEREF(hdda).tmin = PNANOVDB_DEREF(hdda).next.y;
		PNANOVDB_DEREF(hdda).next.y += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).delta.y;
		PNANOVDB_DEREF(hdda).voxel.y += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).step.y;
		ret = PNANOVDB_DEREF(hdda).tmin <= PNANOVDB_DEREF(hdda).tmax;
	}
	else
	{
#ifdef PNANOVDB_ENFORCE_FORWARD_STEPPING
		if (PNANOVDB_DEREF(hdda).next.z <= PNANOVDB_DEREF(hdda).tmin)
		{
			PNANOVDB_DEREF(hdda).next.z += PNANOVDB_DEREF(hdda).tmin - 0.999999f * PNANOVDB_DEREF(hdda).next.z + 1.0e-6f;
		}
#endif
		PNANOVDB_DEREF(hdda).tmin = PNANOVDB_DEREF(hdda).next.z;
		PNANOVDB_DEREF(hdda).next.z += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).delta.z;
		PNANOVDB_DEREF(hdda).voxel.z += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).step.z;
		ret = PNANOVDB_DEREF(hdda).tmin <= PNANOVDB_DEREF(hdda).tmax;
	}
	return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_hdda_ray_clip(
	PNANOVDB_IN(pnanovdb_vec3_t) bbox_min, 
	PNANOVDB_IN(pnanovdb_vec3_t) bbox_max, 
	PNANOVDB_IN(pnanovdb_vec3_t) origin, PNANOVDB_INOUT(float) tmin, 
	PNANOVDB_IN(pnanovdb_vec3_t) direction, PNANOVDB_INOUT(float) tmax
)
{
	pnanovdb_vec3_t dir_inv = pnanovdb_vec3_div(pnanovdb_vec3_uniform(1.f), PNANOVDB_DEREF(direction));
	pnanovdb_vec3_t t0 = pnanovdb_vec3_mul(pnanovdb_vec3_sub(PNANOVDB_DEREF(bbox_min), PNANOVDB_DEREF(origin)), dir_inv);
	pnanovdb_vec3_t t1 = pnanovdb_vec3_mul(pnanovdb_vec3_sub(PNANOVDB_DEREF(bbox_max), PNANOVDB_DEREF(origin)), dir_inv);
	pnanovdb_vec3_t tmin3 = pnanovdb_vec3_min(t0, t1);
	pnanovdb_vec3_t tmax3 = pnanovdb_vec3_max(t0, t1);
	float tnear = pnanovdb_max(tmin3.x, pnanovdb_max(tmin3.y, tmin3.z));
	float tfar = pnanovdb_min(tmax3.x, pnanovdb_min(tmax3.y, tmax3.z));
	pnanovdb_bool_t hit = tnear <= tfar;
	PNANOVDB_DEREF(tmin) = pnanovdb_max(PNANOVDB_DEREF(tmin), tnear);
	PNANOVDB_DEREF(tmax) = pnanovdb_min(PNANOVDB_DEREF(tmax), tfar);
	return hit;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_hdda_zero_crossing(
	pnanovdb_grid_type_t grid_type,
	pnanovdb_buf_t buf, 
	PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, 
	PNANOVDB_IN(pnanovdb_vec3_t) origin, float tmin,
	PNANOVDB_IN(pnanovdb_vec3_t) direction, float tmax,
	PNANOVDB_INOUT(float) thit,
	PNANOVDB_INOUT(float) v
)
{
	pnanovdb_coord_t bbox_min = pnanovdb_root_get_bbox_min(buf, PNANOVDB_DEREF(acc).root);
	pnanovdb_coord_t bbox_max = pnanovdb_root_get_bbox_max(buf, PNANOVDB_DEREF(acc).root);
	pnanovdb_vec3_t bbox_minf = pnanovdb_coord_to_vec3(bbox_min);
	pnanovdb_vec3_t bbox_maxf = pnanovdb_coord_to_vec3(pnanovdb_coord_add(bbox_max, pnanovdb_coord_uniform(1)));

	pnanovdb_bool_t hit = pnanovdb_hdda_ray_clip(PNANOVDB_REF(bbox_minf), PNANOVDB_REF(bbox_maxf), origin, PNANOVDB_REF(tmin), direction, PNANOVDB_REF(tmax));
	if (!hit || tmax > 1.0e20f)
	{
		return PNANOVDB_FALSE;
	}

	pnanovdb_vec3_t pos = pnanovdb_hdda_ray_start(origin, tmin, direction);
	pnanovdb_coord_t ijk = pnanovdb_hdda_pos_to_ijk(PNANOVDB_REF(pos));

	pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, PNANOVDB_REF(ijk));
	float v0 = pnanovdb_read_float(buf, address);

	pnanovdb_int32_t dim = pnanovdb_uint32_as_int32(pnanovdb_readaccessor_get_dim(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, PNANOVDB_REF(ijk)));
	pnanovdb_hdda_t hdda;
	pnanovdb_hdda_init(PNANOVDB_REF(hdda), origin, tmin, direction, tmax, dim);
	while (pnanovdb_hdda_step(PNANOVDB_REF(hdda)))
	{
		pnanovdb_vec3_t pos_start = pnanovdb_hdda_ray_start(origin, hdda.tmin + 1.0001f, direction);
		ijk = pnanovdb_hdda_pos_to_ijk(PNANOVDB_REF(pos_start));
		dim = pnanovdb_uint32_as_int32(pnanovdb_readaccessor_get_dim(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, PNANOVDB_REF(ijk)));
		pnanovdb_hdda_update(PNANOVDB_REF(hdda), origin, direction, dim);
		if (hdda.dim > 1 || !pnanovdb_readaccessor_is_active(grid_type, buf, acc, PNANOVDB_REF(ijk)))
		{
			continue;
		}
		while (pnanovdb_hdda_step(PNANOVDB_REF(hdda)) && pnanovdb_readaccessor_is_active(grid_type, buf, acc, PNANOVDB_REF(hdda.voxel)))
		{
			ijk = hdda.voxel;
			pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, PNANOVDB_REF(ijk));
			PNANOVDB_DEREF(v) = pnanovdb_read_float(buf, address);
			if (PNANOVDB_DEREF(v) * v0 < 0.f)
			{
				PNANOVDB_DEREF(thit) = hdda.tmin;
				return PNANOVDB_TRUE;
			}
		}
	}
	return PNANOVDB_FALSE;
}

#endif

#endif // end of NANOVDB_PNANOVDB_H_HAS_BEEN_INCLUDED
