// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////

#if defined(__OPENCL_VERSION__)

#define CNANOVDB_RETURN(c) return
#ifndef CNANOVDB_COMPILER_OPENCL
#define CNANOVDB_COMPILER_OPENCL
#endif
#define CNANOVDB_INLINE inline
#define CNANOVDB_KERNEL __kernel
#define CNANOVDB_GLOBAL __global
#define CNANOVDB_RESTRICT restrict
#define CNANOVDB_CONSTANT_MEM __constant
#define CNANOVDB_REF(t) t* restrict
#define CNANOVDB_DEREF(x) (*x)
#define CNANOVDB_ADDRESS(x) (&x)
#define CNANOVDB_DECLARE_STRUCT_BEGIN(name) \
    struct _##name \
    {
#define CNANOVDB_DECLARE_STRUCT_END(name) \
    } \
    ; \
    typedef struct _##name name;
#define CNANOVDB_DECLARE_UNION_BEGIN(name) \
    union _##name \
    {
#define CNANOVDB_DECLARE_UNION_END(name) \
    } \
    ; \
    typedef union _##name name;
#define CNANOVDB_NODEDATA(cxt, level) cxt->mNodeLevel##level.nodes
#define CNANOVDB_ROOTDATA(cxt) cxt->mRootData.root
#define CNANOVDB_ROOTDATATILES(cxt) cxt->mRootData.tiles
#define CNANOVDB_GRIDDATA(cxt) cxt->mGridData.grid
#define CNANOVDB_CONTEXT const TreeContext* restrict
#define CNANOVDB_DECLARE_UNIFORMS_BEGIN \
    struct _ArgUniforms \
    {
#define CNANOVDB_DECLARE_UNIFORMS_END \
    } \
    ; \
    typedef struct _ArgUniforms ArgUniforms;
#define CNANOVDB_MAKE(t) (t)
#define CNANOVDB_MAKE_VEC3(a, b, c) (vec3)((float)a, (float)b, (float)c)
#define CNANOVDB_MAKE_IVEC3(a, b, c) (ivec3)((int)a, (int)b, (int)c)
#define CNANOVDB_MAKE_IVEC2(a, b) (ivec2)((int)a, (int)b)
#define CNANOVDB_MAKE_VEC4(a, b, c, d) (vec4)((float)a, (float)b, (float)c, (float)d)

// OpenCL doesn't define these basic types:
typedef unsigned long  uint64_t;
typedef long           int64_t;
typedef unsigned int   uint32_t;
typedef int            int32_t;
typedef unsigned short uint16_t;
typedef short          int16_t;
typedef unsigned char  uint8_t;
typedef signed char    int8_t;
typedef float3         vec3;
typedef float4         vec4;
typedef int2           ivec2;
typedef uint2          uvec2;
typedef int3           ivec3;
typedef bool           boolean;
#define CNANOVDB_TRUE true
#define CNANOVDB_FALSE false

#define CNANOVDB_WORD_TYPE uint64_t
#define CNANOVDB_WORD_LOG2SIZE 6
#define CNANOVDB_WORD_MASK 63

CNANOVDB_INLINE ivec2 getThreadId()
{
    return (ivec2)(get_global_id(0), get_global_id(1));
}

#define CNANOVDB_IMAGE_TYPE __global vec4*

CNANOVDB_INLINE vec4 imageLoadPixel(const CNANOVDB_IMAGE_TYPE image, int w, ivec2 p)
{
    return image[p.x + w * p.y];
}
CNANOVDB_INLINE void imageStorePixel(CNANOVDB_IMAGE_TYPE image, int w, ivec2 p, vec4 color)
{
    image[p.x + w * p.y] = color;
}

#define expf(x) exp(x)
#define logf(x) log(x)
#define cosf(x) cos(x)
#define sinf(x) sin(x)
#define tanf(x) tan(x)
#define sqrtf(x) sqrt(x)
#define fmin(a, b) min(a, b)
#define fmax(a, b) max(a, b)
#define vec3_length(x) length(x)
#define vec3_normalize(x) normalize(x)
#define vec3_dot(a, b) dot(a, b)
#define vec3_sub(a, b) ((a) - (b))
#define vec3_add(a, b) ((a) + (b))
#define vec3_mul(a, b) ((a) * (b))
#define vec3_div(a, b) ((a) / (b))
#define vec3_fdiv(a, b) ((a) / (b))
#define vec3_divf(a, b) ((a) / (b))
#define vec3_fmul(a, b) ((a) * (b))
#define vec3_mulf(a, b) ((a) * (b))
#define vec3_neg(a) (-(a))
#define ivec3_andi(a, b) ((a) & (b))
#define vec4_sub(a, b) ((a) - (b))
#define vec4_add(a, b) ((a) + (b))
#define vec4_fmul(a, b) ((a) * (b))

#elif defined(CNANOVDB_COMPILER_GLSL)

#extension GL_ARB_shader_image_load_store : enable
#pragma optimize(on)
#line 115

#define CNANOVDB_RETURN(c) return
#define CNANOVDB_INLINE
#define CNANOVDB_KERNEL
#define CNANOVDB_GLOBAL
#define CNANOVDB_RESTRICT
#define CNANOVDB_CONSTANT_MEM const
#define CNANOVDB_REF(t) inout t
#define CNANOVDB_DEREF(x) x
#define CNANOVDB_ADDRESS(x) x
#define CNANOVDB_DECLARE_STRUCT_BEGIN(name) \
    struct name \
    {
#define CNANOVDB_DECLARE_STRUCT_END(name) \
    } \
    ;
#define CNANOVDB_NODEDATA(cxt, level) kNodeLevel##level.nodes
#define CNANOVDB_ROOTDATA(cxt) kRootData.root
#define CNANOVDB_ROOTDATATILES(cxt) kRootData.tiles
#define CNANOVDB_GRIDDATA(cxt) kGridData.grid
#define CNANOVDB_CONTEXT int
#define CNANOVDB_DECLARE_UNIFORMS_BEGIN \
    uniform ArgUniforms \
    {
#define CNANOVDB_DECLARE_UNIFORMS_END \
    } \
    kArgs;
#define CNANOVDB_MAKE(t) t
#define CNANOVDB_MAKE_VEC3(a, b, c) vec3(float(a), float(b), float(c))
#define CNANOVDB_MAKE_IVEC3(a, b, c) ivec3(int(a), int(b), int(c))
#define CNANOVDB_MAKE_IVEC2(a, b) ivec2(int(a), int(b))
#define CNANOVDB_MAKE_VEC4(a, b, c, d) vec4(float(a), float(b), float(c), float(d))

#define uint32_t uint
#define int32_t int
#define uint64_t uvec2
#define int64_t ivec2
#define boolean bool
#define CNANOVDB_TRUE true
#define CNANOVDB_FALSE false

#define CNANOVDB_WORD_TYPE uint32_t
#define CNANOVDB_WORD_LOG2SIZE 5
#define CNANOVDB_WORD_MASK 31

#define getThreadId() ivec2(gl_GlobalInvocationID.xy)

#define expf(x) exp(x)
#define logf(x) log(x)
#define cosf(x) cos(x)
#define sinf(x) sin(x)
#define tanf(x) tan(x)
#define sqrtf(x) sqrt(x)
#define fabs(x) abs(x)
#define fmod(a, b) mod(a, b)
#define fmin(a, b) min(a, b)
#define fmax(a, b) max(a, b)
#define vec3_length(x) length(x)
#define vec3_normalize(x) normalize(x)
#define vec3_dot(a, b) dot(a, b)
#define vec3_sub(a, b) ((a) - (b))
#define vec3_add(a, b) ((a) + (b))
#define vec3_mul(a, b) ((a) * (b))
#define vec3_div(a, b) ((a) / (b))
#define vec3_fdiv(a, b) ((a) / (b))
#define vec3_divf(a, b) ((a) / (b))
#define vec3_fmul(a, b) ((a) * (b))
#define vec3_mulf(a, b) ((a) * (b))
#define vec3_neg(a) (-(a))
#define ivec3_andi(a, b) ((a) & (b))
#define vec4_sub(a, b) ((a) - (b))
#define vec4_add(a, b) ((a) + (b))
#define vec4_fmul(a, b) ((a) * (b))

#define CNANOVDB_IMAGE_TYPE int
#define imageLoadPixel(img, w, p) imageLoad(kOutImage, p);
#define imageStorePixel(img, w, p, color) imageStore(kOutImage, p, color)

#elif defined(__cplusplus)

#define CNANOVDB_RETURN(c) return
#define CNANOVDB_INLINE inline
#define CNANOVDB_KERNEL inline
#define CNANOVDB_GLOBAL
#define CNANOVDB_RESTRICT __restrict
#define CNANOVDB_CONSTANT_MEM const
#define CNANOVDB_REF(t) t&
#define CNANOVDB_DEREF(x) (x)
#define CNANOVDB_ADDRESS(x) x
#define CNANOVDB_NODEDATA(cxt, level) cxt->mNodeLevel##level.nodes
#define CNANOVDB_ROOTDATA(cxt) cxt->mRootData.root
#define CNANOVDB_ROOTDATATILES(cxt) cxt->mRootData.tiles
#define CNANOVDB_GRIDDATA(cxt) cxt->mGridData.grid
#define CNANOVDB_CONTEXT const TreeContext*
#define CNANOVDB_MAKE(t) (t)
#define CNANOVDB_MAKE_VEC3(a, b, c) \
    vec3 { (float)a, (float)b, (float)c }
#define CNANOVDB_MAKE_IVEC3(a, b, c) \
    ivec3 { (int)a, (int)b, (int)c }
#define CNANOVDB_MAKE_IVEC2(a, b) \
    ivec2 { (int)a, (int)b }
#define CNANOVDB_MAKE_VEC4(a, b, c, d) \
    vec4 { (float)a, (float)b, (float)c, (float)d }
#define CNANOVDB_DECLARE_STRUCT_BEGIN(name) \
    struct _##name \
    {
#define CNANOVDB_DECLARE_STRUCT_END(name) \
    } \
    ; \
    typedef struct _##name name;
#define CNANOVDB_DECLARE_UNION_BEGIN(name) \
    union _##name \
    {
#define CNANOVDB_DECLARE_UNION_END(name) \
    } \
    ; \
    typedef union _##name name;
#define CNANOVDB_DECLARE_UNIFORMS_BEGIN \
    struct _ArgUniforms \
    {
#define CNANOVDB_DECLARE_UNIFORMS_END \
    } \
    ; \
    typedef struct _ArgUniforms ArgUniforms;

#include <stdint.h>
#include <math.h>

typedef bool boolean;
#define CNANOVDB_TRUE true
#define CNANOVDB_FALSE false

#define CNANOVDB_WORD_TYPE uint64_t
#define CNANOVDB_WORD_LOG2SIZE 6
#define CNANOVDB_WORD_MASK 63

//float nanovdb_Clamp_float(float x, float a, float b)
struct uvec2
{
    unsigned int x;
    unsigned int y;
};
struct ivec2
{
    int x;
    int y;
};
struct vec4
{
    float x;
    float y;
    float z;
    float w;
};
struct vec3
{
    float x;
    float y;
    float z;
};
struct ivec3
{
    int x;
    int y;
    int z;
};
CNANOVDB_INLINE vec3 vec3_sub(const vec3 a, const vec3 b)
{
    return vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}
CNANOVDB_INLINE vec3 vec3_add(const vec3 a, const vec3 b)
{
    return vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
CNANOVDB_INLINE vec3 vec3_div(const vec3 a, const vec3 b)
{
    return vec3{a.x / b.x, a.y / b.y, a.z / b.z};
}
CNANOVDB_INLINE vec3 vec3_mul(const vec3 a, const vec3 b)
{
    return vec3{a.x * b.x, a.y * b.y, a.z * b.z};
}
CNANOVDB_INLINE vec3 vec3_divf(const vec3 a, const float b)
{
    return vec3{a.x / b, a.y / b, a.z / b};
}
CNANOVDB_INLINE vec3 vec3_mulf(const vec3 a, const float b)
{
    return vec3{a.x * b, a.y * b, a.z * b};
}
CNANOVDB_INLINE vec3 vec3_fdiv(const float a, const vec3 b)
{
    return vec3{a / b.x, a / b.y, a / b.z};
}
CNANOVDB_INLINE vec3 vec3_fmul(const float a, const vec3 b)
{
    return vec3{a * b.x, a * b.y, a * b.z};
}
CNANOVDB_INLINE vec3 vec3_neg(const vec3 a)
{
    return vec3{-a.x, -a.y, -a.z};
}
CNANOVDB_INLINE float vec3_dot(const vec3 a, const vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
CNANOVDB_INLINE float vec3_length(const vec3 a)
{
    return sqrtf(vec3_dot(a, a));
}
CNANOVDB_INLINE vec3 vec3_normalize(const vec3 a)
{
    return vec3_divf(a, vec3_length(a));
}
CNANOVDB_INLINE ivec3 ivec3_andi(const ivec3 a, const int b)
{
    return ivec3{a.x & b, a.y & b, a.z & b};
}
CNANOVDB_INLINE vec4 vec4_sub(const vec4 a, const vec4 b)
{
    return vec4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
CNANOVDB_INLINE vec4 vec4_fmul(const float b, const vec4 a)
{
    return vec4{a.x * b, a.y * b, a.z * b, a.w * b};
}
CNANOVDB_INLINE vec4 vec4_add(const vec4 a, const vec4 b)
{
    return vec4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
CNANOVDB_INLINE ivec2 getThreadId()
{
    return ivec2{0, 0};
}

#define CNANOVDB_IMAGE_TYPE vec4*

CNANOVDB_INLINE vec4 imageLoadPixel(const CNANOVDB_IMAGE_TYPE image, int w, ivec2 p)
{
    return image[p.x + w * p.y];
}
CNANOVDB_INLINE void imageStorePixel(CNANOVDB_IMAGE_TYPE image, int w, ivec2 p, vec4 color)
{
    image[p.x + w * p.y] = color;
}

#else

#define CNANOVDB_RETURN(c) return
#define CNANOVDB_INLINE static inline
#define CNANOVDB_KERNEL static inline
#define CNANOVDB_GLOBAL
#define CNANOVDB_RESTRICT __restrict
#define CNANOVDB_CONSTANT_MEM const
#define CNANOVDB_REF(t) t*
#define CNANOVDB_DEREF(x) (*x)
#define CNANOVDB_ADDRESS(x) &x
#define CNANOVDB_NODEDATA(cxt, level) cxt->mNodeLevel##level.nodes
#define CNANOVDB_ROOTDATA(cxt) cxt->mRootData.root
#define CNANOVDB_ROOTDATATILES(cxt) cxt->mRootData.tiles
#define CNANOVDB_GRIDDATA(cxt) cxt->mGridData.grid
#define CNANOVDB_CONTEXT const TreeContext*
#define CNANOVDB_MAKE(t) (t)
#define CNANOVDB_MAKE_VEC3(a, b, c) \
    (vec3) { (float)a, (float)b, (float)c }
#define CNANOVDB_MAKE_IVEC3(a, b, c) \
    (ivec3) { (int)a, (int)b, (int)c }
#define CNANOVDB_MAKE_IVEC2(a, b) \
    (ivec2) { (int)a, (int)b }
#define CNANOVDB_MAKE_VEC4(a, b, c, d) \
    (vec4) { (float)a, (float)b, (float)c, (float)d }
#define CNANOVDB_DECLARE_STRUCT_BEGIN(name) \
    struct _##name \
    {
#define CNANOVDB_DECLARE_STRUCT_END(name) \
    } \
    ; \
    typedef struct _##name name;
#define CNANOVDB_DECLARE_UNION_BEGIN(name) \
    union _##name \
    {
#define CNANOVDB_DECLARE_UNION_END(name) \
    } \
    ; \
    typedef union _##name name;
#define CNANOVDB_DECLARE_UNIFORMS_BEGIN \
    struct _ArgUniforms \
    {
#define CNANOVDB_DECLARE_UNIFORMS_END \
    } \
    ; \
    typedef struct _ArgUniforms ArgUniforms;

#include <stdint.h>
#include <math.h>

typedef int boolean;
#define CNANOVDB_TRUE 1
#define CNANOVDB_FALSE 0

#define CNANOVDB_WORD_TYPE uint64_t
#define CNANOVDB_WORD_LOG2SIZE 6
#define CNANOVDB_WORD_MASK 63

typedef struct
{
    unsigned int x;
    unsigned int y;
} uvec2;
typedef struct
{
    int x;
    int y;
} ivec2;
typedef struct
{
    float x;
    float y;
    float z;
    float w;
} vec4;
typedef struct
{
    float x;
    float y;
    float z;
} vec3;
typedef struct
{
    int x;
    int y;
    int z;
} ivec3;
CNANOVDB_INLINE vec3 vec3_sub(const vec3 a, const vec3 b)
{
    return (vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}
CNANOVDB_INLINE vec3 vec3_add(const vec3 a, const vec3 b)
{
    return (vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}
CNANOVDB_INLINE vec3 vec3_div(const vec3 a, const vec3 b)
{
    return (vec3){a.x / b.x, a.y / b.y, a.z / b.z};
}
CNANOVDB_INLINE vec3 vec3_mul(const vec3 a, const vec3 b)
{
    return (vec3){a.x * b.x, a.y * b.y, a.z * b.z};
}
CNANOVDB_INLINE vec3 vec3_divf(const vec3 a, const float b)
{
    return (vec3){a.x / b, a.y / b, a.z / b};
}
CNANOVDB_INLINE vec3 vec3_mulf(const vec3 a, const float b)
{
    return (vec3){a.x * b, a.y * b, a.z * b};
}
CNANOVDB_INLINE vec3 vec3_fdiv(const float a, const vec3 b)
{
    return (vec3){a / b.x, a / b.y, a / b.z};
}
CNANOVDB_INLINE vec3 vec3_fmul(const float a, const vec3 b)
{
    return (vec3){a * b.x, a * b.y, a * b.z};
}
CNANOVDB_INLINE vec3 vec3_neg(const vec3 a)
{
    return (vec3){-a.x, -a.y, -a.z};
}
CNANOVDB_INLINE float vec3_dot(const vec3 a, const vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
CNANOVDB_INLINE float vec3_length(const vec3 a)
{
    return sqrtf(vec3_dot(a, a));
}
CNANOVDB_INLINE vec3 vec3_normalize(const vec3 a)
{
    return vec3_divf(a, vec3_length(a));
}
CNANOVDB_INLINE ivec3 ivec3_andi(const ivec3 a, const int b)
{
    return (ivec3){a.x & b, a.y & b, a.z & b};
}
CNANOVDB_INLINE vec4 vec4_sub(const vec4 a, const vec4 b)
{
    return (vec4){a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
CNANOVDB_INLINE vec4 vec4_fmul(const float b, const vec4 a)
{
    return (vec4){a.x * b, a.y * b, a.z * b, a.w * b};
}
CNANOVDB_INLINE vec4 vec4_add(const vec4 a, const vec4 b)
{
    return (vec4){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
CNANOVDB_INLINE ivec2 getThreadId()
{
    return (ivec2){0, 0};
}

#define CNANOVDB_IMAGE_TYPE vec4*

CNANOVDB_INLINE vec4 imageLoadPixel(const CNANOVDB_IMAGE_TYPE image, int w, ivec2 p)
{
    return image[p.x + w * p.y];
}
CNANOVDB_INLINE void imageStorePixel(CNANOVDB_IMAGE_TYPE image, int w, ivec2 p, vec4 color)
{
    image[p.x + w * p.y] = color;
}

#endif

////////////////////////////////////////////////////////