// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_editor/putil/Reflect.h

    \author Andrew Reidmeyer

    \brief  This file provides a simple C reflection system.
*/

#ifndef NANOVDB_PUTILS_REFLECT_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_REFLECT_H_HAS_BEEN_INCLUDED

#if defined(PNANOVDB_HLSL) || defined(PNANOVDB_GLSL)
#error PNanoVDBEditor interfaces are for PNANOVDB_C only
#endif
// This header is for C target only
#if !defined(PNANOVDB_C)
#define PNANOVDB_C
#endif

#include "nanovdb/PNanoVDB.h"

// ------------------------------------------------ Basic Types -----------------------------------------------------------

#ifndef PNANOVDB_ABI
#if defined(_WIN32)
#define PNANOVDB_ABI __cdecl
#else
#define PNANOVDB_ABI
#endif
#endif

#if defined(PNANOVDB_C)
#if defined(__CUDACC__)
#define PNANOVDB_INLINE static __host__ __device__
#elif defined(_WIN32)
#define PNANOVDB_INLINE static inline
#else
#define PNANOVDB_INLINE static inline
#endif
#elif defined(PNANOVDB_HLSL)
#define PNANOVDB_INLINE
#elif defined(PNANOVDB_GLSL)
#define PNANOVDB_INLINE
#endif

#ifndef PNANOVDB_API
#if defined(_WIN32)
#if defined(__cplusplus)
#define PNANOVDB_API extern "C" __declspec(dllexport)
#else
#define PNANOVDB_API __declspec(dllexport)
#endif
#else
#if defined(__cplusplus)
#define PNANOVDB_API extern "C" __attribute__((visibility("default")))
#else
#define PNANOVDB_API __attribute__((visibility("default")))
#endif
#endif
#endif

typedef uint8_t pnanovdb_uint8_t;
typedef uint16_t pnanovdb_uint16_t;

// ------------------------------------------------ Reflection -----------------------------------------------------------

typedef pnanovdb_uint32_t pnanovdb_reflect_type_t;

#define PNANOVDB_REFLECT_TYPE_UNKNOWN 0
#define PNANOVDB_REFLECT_TYPE_VOID 1
#define PNANOVDB_REFLECT_TYPE_FUNCTION 2
#define PNANOVDB_REFLECT_TYPE_STRUCT 3
#define PNANOVDB_REFLECT_TYPE_INT32 4
#define PNANOVDB_REFLECT_TYPE_UINT32 5
#define PNANOVDB_REFLECT_TYPE_FLOAT 6
#define PNANOVDB_REFLECT_TYPE_BOOL32 7
#define PNANOVDB_REFLECT_TYPE_UINT8 8
#define PNANOVDB_REFLECT_TYPE_UINT16 9
#define PNANOVDB_REFLECT_TYPE_UINT64 10
#define PNANOVDB_REFLECT_TYPE_CHAR 11
#define PNANOVDB_REFLECT_TYPE_DOUBLE 12

PNANOVDB_INLINE int pnanovdb_reflect_string_compare(const char* a, const char* b)
{
    a = a ? a : "\0";
    b = b ? b : "\0";
    int idx = 0;
    while (a[idx] || b[idx])
    {
        if (a[idx] != b[idx])
        {
            return a[idx] < b[idx] ? -1 : +1;
        }
        idx++;
    }
    return 0;
}

PNANOVDB_INLINE const char* pnanovdb_reflect_type_to_string(pnanovdb_uint32_t type)
{
    switch (type)
    {
    case PNANOVDB_REFLECT_TYPE_UNKNOWN: return "unknown";
    case PNANOVDB_REFLECT_TYPE_VOID: return "void";
    case PNANOVDB_REFLECT_TYPE_FUNCTION: return "function";
    case PNANOVDB_REFLECT_TYPE_STRUCT: return "struct";
    case PNANOVDB_REFLECT_TYPE_INT32: return "int32";
    case PNANOVDB_REFLECT_TYPE_UINT32: return "uint32";
    case PNANOVDB_REFLECT_TYPE_FLOAT: return "float";
    case PNANOVDB_REFLECT_TYPE_BOOL32: return "bool32";
    case PNANOVDB_REFLECT_TYPE_UINT8: return "uint8";
    case PNANOVDB_REFLECT_TYPE_UINT16: return "uint16";
    case PNANOVDB_REFLECT_TYPE_UINT64: return "uint64";
    case PNANOVDB_REFLECT_TYPE_CHAR: return "char";
    case PNANOVDB_REFLECT_TYPE_DOUBLE: return "double";
    default: return "unknown";
    }
}

PNANOVDB_INLINE pnanovdb_uint32_t pnanovdb_reflect_type_from_string(const char* name)
{
    if (pnanovdb_reflect_string_compare(name, "unknown") == 0) { return PNANOVDB_REFLECT_TYPE_UNKNOWN; }
    else if (pnanovdb_reflect_string_compare(name, "void") == 0) { return PNANOVDB_REFLECT_TYPE_VOID; }
    else if (pnanovdb_reflect_string_compare(name, "function") == 0) { return PNANOVDB_REFLECT_TYPE_FUNCTION; }
    else if (pnanovdb_reflect_string_compare(name, "struct") == 0) { return PNANOVDB_REFLECT_TYPE_STRUCT; }
    else if (pnanovdb_reflect_string_compare(name, "int32") == 0) { return PNANOVDB_REFLECT_TYPE_INT32; }
    else if (pnanovdb_reflect_string_compare(name, "uint32") == 0) { return PNANOVDB_REFLECT_TYPE_UINT32; }
    else if (pnanovdb_reflect_string_compare(name, "float") == 0) { return PNANOVDB_REFLECT_TYPE_FLOAT; }
    else if (pnanovdb_reflect_string_compare(name, "bool32") == 0) { return PNANOVDB_REFLECT_TYPE_BOOL32; }
    else if (pnanovdb_reflect_string_compare(name, "uint8") == 0) { return PNANOVDB_REFLECT_TYPE_UINT8; }
    else if (pnanovdb_reflect_string_compare(name, "uint16") == 0) { return PNANOVDB_REFLECT_TYPE_UINT16; }
    else if (pnanovdb_reflect_string_compare(name, "uint64") == 0) { return PNANOVDB_REFLECT_TYPE_UINT64; }
    else if (pnanovdb_reflect_string_compare(name, "char") == 0) { return PNANOVDB_REFLECT_TYPE_CHAR; }
    else if (pnanovdb_reflect_string_compare(name, "double") == 0) { return PNANOVDB_REFLECT_TYPE_DOUBLE; }
    else return PNANOVDB_REFLECT_TYPE_UNKNOWN;
}

PNANOVDB_INLINE void pnanovdb_reflect_memcpy(void* dst, const void* src, pnanovdb_uint64_t num_bytes)
{
    for (pnanovdb_uint64_t byteIdx = 0u; byteIdx < num_bytes; byteIdx++)
    {
        ((char*)dst)[byteIdx] = ((const char*)src)[byteIdx];
    }
}

PNANOVDB_INLINE void pnanovdb_reflect_clear(void* dst, pnanovdb_uint64_t num_bytes)
{
    for (pnanovdb_uint64_t byteIdx = 0u; byteIdx < num_bytes; byteIdx++)
    {
        ((char*)dst)[byteIdx] = 0;
    }
}

typedef pnanovdb_uint32_t pnanovdb_reflect_hint_flags_t;
typedef pnanovdb_uint32_t pnanovdb_reflect_mode_flags_t;

#define PNANOVDB_REFLECT_MODE_VALUE 0
#define PNANOVDB_REFLECT_MODE_POINTER 1
#define PNANOVDB_REFLECT_MODE_ARRAY 2
#define PNANOVDB_REFLECT_MODE_POINTER_ARRAY 3
#define PNANOVDB_REFLECT_MODE_VALUE_VERSIONED 4
#define PNANOVDB_REFLECT_MODE_POINTER_VERSIONED 5
#define PNANOVDB_REFLECT_MODE_ARRAY_VERSIONED 6
#define PNANOVDB_REFLECT_MODE_POINTER_ARRAY_VERSIONED 7

struct pnanovdb_reflect_data_type_t;
typedef struct pnanovdb_reflect_data_type_t pnanovdb_reflect_data_type_t;

typedef struct pnanovdb_reflect_data_t
{
    pnanovdb_reflect_hint_flags_t reflect_hint;
    pnanovdb_reflect_mode_flags_t reflect_mode;
    const pnanovdb_reflect_data_type_t* data_type;
    const char* name;
    pnanovdb_uint64_t data_offset;
    pnanovdb_uint64_t array_size_offset;
    pnanovdb_uint64_t version_offset;
    const char* metadata;
}pnanovdb_reflect_data_t;

typedef struct pnanovdb_reflect_data_type_t
{
    pnanovdb_reflect_type_t data_type;
    pnanovdb_uint64_t element_size;
    const char* struct_typename;
    const pnanovdb_reflect_data_t* child_reflect_datas;
    pnanovdb_uint64_t child_reflect_data_count;
    const void* default_value;
}pnanovdb_reflect_data_type_t;

PNANOVDB_INLINE pnanovdb_bool_t pnanovdb_reflect_layout_compare(
    const pnanovdb_reflect_data_type_t* dst_type,
    const pnanovdb_reflect_data_type_t* src_type
)
{
    if (dst_type == src_type)
    {
        return PNANOVDB_TRUE;
    }
    if (dst_type->data_type != src_type->data_type ||
        dst_type->element_size != src_type->element_size ||
        dst_type->child_reflect_data_count != src_type->child_reflect_data_count ||
        pnanovdb_reflect_string_compare(dst_type->struct_typename, src_type->struct_typename) != 0)
    {
        return PNANOVDB_FALSE;
    }
    for (pnanovdb_uint64_t child_idx = 0u; child_idx < dst_type->child_reflect_data_count; child_idx++)
    {
        const pnanovdb_reflect_data_t* child_dst = dst_type->child_reflect_datas + child_idx;
        const pnanovdb_reflect_data_t* child_src = src_type->child_reflect_datas + child_idx;
        if (child_dst->reflect_mode != child_src->reflect_mode ||
            child_dst->data_offset != child_src->data_offset ||
            child_dst->array_size_offset != child_src->array_size_offset ||
            child_dst->version_offset != child_src->version_offset ||
            pnanovdb_reflect_string_compare(child_dst->name, child_src->name) != 0)
        {
            return PNANOVDB_FALSE;
        }
        if (!pnanovdb_reflect_layout_compare(child_dst->data_type, child_src->data_type))
        {
            return PNANOVDB_FALSE;
        }
    }
    return PNANOVDB_TRUE;
}

PNANOVDB_INLINE void pnanovdb_reflect_copy_by_name(
    void* dst_data, const pnanovdb_reflect_data_type_t* dst_type,
    const void* src_data, const pnanovdb_reflect_data_type_t* src_type
)
{
    pnanovdb_uint8_t* dst_data8 = (pnanovdb_uint8_t*)dst_data;
    const pnanovdb_uint8_t* src_data8 = (const pnanovdb_uint8_t*)src_data;

    // For safety, take min of elementSize
    pnanovdb_uint64_t safe_copy_size = src_type->element_size < dst_type->element_size ? src_type->element_size : dst_type->element_size;

    // Start with raw copy, to potential cover non-reflect data
    pnanovdb_reflect_memcpy(dst_data, src_data, safe_copy_size);

    // Copy by name
    if (dst_type != src_type)
    {
        pnanovdb_uint64_t src_idx = 0u;
        for (pnanovdb_uint64_t dst_idx = 0u; dst_idx < dst_type->child_reflect_data_count; dst_idx++)
        {
            const pnanovdb_reflect_data_t* child_dst = dst_type->child_reflect_datas + dst_idx;
            pnanovdb_uint64_t match_src_idx = ~0llu;
            for (pnanovdb_uint64_t srcCount = 0u; match_src_idx == ~0llu && srcCount < src_type->child_reflect_data_count; srcCount++)
            {
                const pnanovdb_reflect_data_t* child_src = src_type->child_reflect_datas + src_idx;
                if (child_dst->name == child_src->name ||
                    pnanovdb_reflect_string_compare(child_dst->name, child_src->name) == 0)
                {
                    match_src_idx = src_idx;
                }
                src_idx++;
                if (src_idx >= src_type->child_reflect_data_count)
                {
                    src_idx = 0u;
                }
            }
            if (match_src_idx < src_type->child_reflect_data_count)
            {
                const pnanovdb_reflect_data_t* child_src = src_type->child_reflect_datas + match_src_idx;
                if (child_src->data_type->data_type == PNANOVDB_REFLECT_TYPE_STRUCT &&
                    (child_src->reflect_mode == PNANOVDB_REFLECT_MODE_VALUE || child_src->reflect_mode == PNANOVDB_REFLECT_MODE_VALUE_VERSIONED))
                {
                    pnanovdb_reflect_copy_by_name(
                        dst_data8 + child_dst->data_offset, child_dst->data_type,
                        src_data8 + child_src->data_offset, child_src->data_type
                    );
                }
                else
                {
                    // only copy if not covered by bulk memcpy
                    if (child_dst->data_offset != child_src->data_offset)
                    {
                        pnanovdb_reflect_memcpy(
                            dst_data8 + child_dst->data_offset,
                            src_data8 + child_src->data_offset,
                            (child_dst->reflect_mode & PNANOVDB_REFLECT_MODE_POINTER_ARRAY) ? sizeof(void*) : child_dst->data_type->element_size
                        );
                    }
                    if (child_dst->reflect_mode & PNANOVDB_REFLECT_MODE_ARRAY)
                    {
                        if (child_dst->array_size_offset != child_src->array_size_offset)
                        {
                            pnanovdb_reflect_memcpy(
                                dst_data8 + child_dst->array_size_offset,
                                src_data8 + child_src->array_size_offset,
                                sizeof(pnanovdb_uint64_t)
                            );
                        }
                    }
                    if (child_dst->reflect_mode & PNANOVDB_REFLECT_MODE_VALUE_VERSIONED)
                    {
                        if (child_dst->version_offset != child_src->version_offset)
                        {
                            pnanovdb_reflect_memcpy(
                                dst_data8 + child_dst->version_offset,
                                src_data8 + child_src->version_offset,
                                sizeof(pnanovdb_uint64_t)
                            );
                        }
                    }
                }
            }
            else
            {
                pnanovdb_reflect_clear(
                    dst_data8 + child_dst->data_offset,
                    (child_dst->reflect_mode & PNANOVDB_REFLECT_MODE_POINTER_ARRAY) ? sizeof(void*) : child_dst->data_type->element_size
                );
                if (child_dst->reflect_mode & PNANOVDB_REFLECT_MODE_ARRAY)
                {
                    pnanovdb_reflect_clear(
                        dst_data8 + child_dst->array_size_offset,
                        sizeof(pnanovdb_uint64_t)
                    );
                }
                if (child_dst->reflect_mode & PNANOVDB_REFLECT_MODE_VALUE_VERSIONED)
                {
                    pnanovdb_reflect_clear(
                        dst_data8 + child_dst->version_offset,
                        sizeof(pnanovdb_uint64_t)
                    );
                }
            }
        }
    }
}

// Reflect blocks must start with #define PNANOVDB_REFLECT_TYPE typename
// And end with #undef PNANOVDB_REFLECT_TYPE

#define PNANOVDB_REFLECT_XSTR(X) PNANOVDB_REFLECT_STR(X)
#define PNANOVDB_REFLECT_STR(X) #X

#define PNANOVDB_REFLECT_XCONCAT(A, B) PNANOVDB_REFLECT_CONCAT(A, B)
#define PNANOVDB_REFLECT_CONCAT(A, B) A##B

#define PNANOVDB_REFLECT_VALIDATE(type) \
    PNANOVDB_INLINE type* PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr)(const type* v) { return (type*)v; } \
    PNANOVDB_INLINE type** PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr_ptr)(const type* const * v) { return (type**)v; } \
    PNANOVDB_INLINE type*** PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr_ptr_ptr)(const type*const *const * v) { return (type***)v; }

#if defined(__cplusplus)
#define PNANOVDB_REFLECT_VALIDATE_VALUE(type, name) PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr)(&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_POINTER(type, name) PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr_ptr)(&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_ARRAY(type, name) PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr_ptr)(&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_POINTER_ARRAY(type, name) PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr_ptr_ptr)(&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_VALUE_VERSIONED(type, name) PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr)(&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_POINTER_VERSIONED(type, name) PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr_ptr)(&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_ARRAY_VERSIONED(type, name) PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr_ptr)(&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_POINTER_ARRAY_VERSIONED(type, name) PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_validate_ptr_ptr_ptr)(&((PNANOVDB_REFLECT_TYPE*)0)->name)
#else
#define PNANOVDB_REFLECT_VALIDATE_VALUE(type, name) (&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_POINTER(type, name) (&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_ARRAY(type, name) (&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_POINTER_ARRAY(type, name) (&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_VALUE_VERSIONED(type, name) (&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_POINTER_VERSIONED(type, name) (&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_ARRAY_VERSIONED(type, name) (&((PNANOVDB_REFLECT_TYPE*)0)->name)
#define PNANOVDB_REFLECT_VALIDATE_POINTER_ARRAY_VERSIONED(type, name) (&((PNANOVDB_REFLECT_TYPE*)0)->name)
#endif

#define PNANOVDB_REFLECT_BUILTIN_IMPL(enumName, typeName) \
    static const pnanovdb_reflect_data_type_t PNANOVDB_REFLECT_XCONCAT(typeName,_pnanovdb_reflect_data_type) = { enumName, sizeof(typeName), 0, 0, 0, 0 }; \
    PNANOVDB_REFLECT_VALIDATE(typeName)

#define PNANOVDB_REFLECT_STRUCT_OPAQUE_IMPL(name) \
    static const pnanovdb_reflect_data_type_t PNANOVDB_REFLECT_XCONCAT(name,_pnanovdb_reflect_data_type) = { PNANOVDB_REFLECT_TYPE_STRUCT, 0llu, #name, 0, 0, 0 }; \
    PNANOVDB_REFLECT_VALIDATE(name)

#define PNANOVDB_REFLECT_BEGIN() \
        static const pnanovdb_reflect_data_t PNANOVDB_REFLECT_XCONCAT(PNANOVDB_REFLECT_TYPE,_reflect_datas)[] = {

#define PNANOVDB_REFLECT_END(defaultValue) \
        }; \
        static const pnanovdb_reflect_data_type_t PNANOVDB_REFLECT_XCONCAT(PNANOVDB_REFLECT_TYPE,_pnanovdb_reflect_data_type) = { \
            PNANOVDB_REFLECT_TYPE_STRUCT, \
            sizeof(PNANOVDB_REFLECT_TYPE), \
            PNANOVDB_REFLECT_XSTR(PNANOVDB_REFLECT_TYPE), \
            PNANOVDB_REFLECT_XCONCAT(PNANOVDB_REFLECT_TYPE,_reflect_datas), \
            sizeof(PNANOVDB_REFLECT_XCONCAT(PNANOVDB_REFLECT_TYPE,_reflect_datas)) / sizeof(pnanovdb_reflect_data_t), \
            defaultValue \
        }; \
        PNANOVDB_REFLECT_VALIDATE(PNANOVDB_REFLECT_TYPE)

#define PNANOVDB_REFLECT_TYPE_ALIAS(SRC, DST) \
    typedef SRC DST; \
    static const pnanovdb_reflect_data_type_t PNANOVDB_REFLECT_XCONCAT(DST,_pnanovdb_reflect_data_type) = { \
        PNANOVDB_REFLECT_TYPE_STRUCT, \
        sizeof(SRC), \
        #DST, \
        PNANOVDB_REFLECT_XCONCAT(SRC,_reflect_datas), \
        sizeof(PNANOVDB_REFLECT_XCONCAT(SRC,_reflect_datas)) / sizeof(pnanovdb_reflect_data_t), \
        &PNANOVDB_REFLECT_XCONCAT(SRC,_default) \
        }; \
    PNANOVDB_REFLECT_VALIDATE(DST)

PNANOVDB_REFLECT_BUILTIN_IMPL(PNANOVDB_REFLECT_TYPE_INT32, pnanovdb_int32_t)
PNANOVDB_REFLECT_BUILTIN_IMPL(PNANOVDB_REFLECT_TYPE_UINT32, pnanovdb_uint32_t)
PNANOVDB_REFLECT_BUILTIN_IMPL(PNANOVDB_REFLECT_TYPE_FLOAT, float)
PNANOVDB_REFLECT_BUILTIN_IMPL(PNANOVDB_REFLECT_TYPE_BOOL32, pnanovdb_bool_t)
PNANOVDB_REFLECT_BUILTIN_IMPL(PNANOVDB_REFLECT_TYPE_UINT8, pnanovdb_uint8_t)
PNANOVDB_REFLECT_BUILTIN_IMPL(PNANOVDB_REFLECT_TYPE_UINT16, pnanovdb_uint16_t)
PNANOVDB_REFLECT_BUILTIN_IMPL(PNANOVDB_REFLECT_TYPE_UINT64, pnanovdb_uint64_t)
PNANOVDB_REFLECT_BUILTIN_IMPL(PNANOVDB_REFLECT_TYPE_CHAR, char)
PNANOVDB_REFLECT_BUILTIN_IMPL(PNANOVDB_REFLECT_TYPE_DOUBLE, double)

#if defined(__cplusplus)
#define PNANOVDB_REFLECT_SIZE_OFFSET(name_size) (pnanovdb_uint64_t)PNANOVDB_REFLECT_VALIDATE_VALUE(pnanovdb_uint64_t, name_size)
#define PNANOVDB_REFLECT_VERSION_OFFSET(version) (pnanovdb_uint64_t)PNANOVDB_REFLECT_VALIDATE_VALUE(pnanovdb_uint64_t, version)
#else
#define PNANOVDB_REFLECT_SIZE_OFFSET(name_size) (pnanovdb_uint64_t)(&((PNANOVDB_REFLECT_TYPE*)0)->name_size)
#define PNANOVDB_REFLECT_VERSION_OFFSET(version) (pnanovdb_uint64_t)(&((PNANOVDB_REFLECT_TYPE*)0)->version)
#endif

/// Builtin
#define PNANOVDB_REFLECT_GENERIC(reflectMode, type, name, ARRAY, VERSION, reflectHints, metadata) { \
    reflectHints, \
    PNANOVDB_REFLECT_XCONCAT(PNANOVDB_REFLECT_MODE_,reflectMode), \
    &PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_reflect_data_type), \
    #name, \
    (pnanovdb_uint64_t)PNANOVDB_REFLECT_XCONCAT(PNANOVDB_REFLECT_VALIDATE_,reflectMode)(type, name), \
    ARRAY, \
    VERSION, \
    metadata \
    },
#define PNANOVDB_REFLECT_VALUE(type, name, reflectHints, metadata) PNANOVDB_REFLECT_GENERIC(VALUE, type, name, 0, 0, reflectHints, metadata)
#define PNANOVDB_REFLECT_POINTER(type, name, reflectHints, metadata) PNANOVDB_REFLECT_GENERIC(POINTER, type, name, 0, 0, reflectHints, metadata)
#define PNANOVDB_REFLECT_ARRAY(type, name, name_size, reflectHints, metadata) PNANOVDB_REFLECT_GENERIC(ARRAY, type, name, PNANOVDB_REFLECT_SIZE_OFFSET(name_size), 0, reflectHints, metadata)
#define PNANOVDB_REFLECT_POINTER_ARRAY(type, name, name_size, reflectHints, metadata) PNANOVDB_REFLECT_GENERIC(POINTER_ARRAY, type, name, PNANOVDB_REFLECT_SIZE_OFFSET(name_size), 0, reflectHints, metadata)
#define PNANOVDB_REFLECT_VALUE_VERSIONED(type, name, version, reflectHints, metadata) PNANOVDB_REFLECT_GENERIC(VALUE_VERSIONED, type, name, 0, PNANOVDB_REFLECT_VERSION_OFFSET(version), reflectHints, metadata)
#define PNANOVDB_REFLECT_POINTER_VERSIONED(type, name, version, reflectHints, metadata) PNANOVDB_REFLECT_GENERIC(POINTER_VERSIONED, type, name, 0, PNANOVDB_REFLECT_VERSION_OFFSET(version), reflectHints, metadata)
#define PNANOVDB_REFLECT_ARRAY_VERSIONED(type, name, name_size, version, reflectHints, metadata) PNANOVDB_REFLECT_GENERIC(ARRAY_VERSIONED, type, name, PNANOVDB_REFLECT_SIZE_OFFSET(name_size), PNANOVDB_REFLECT_VERSION_OFFSET(version), reflectHints, metadata)
#define PNANOVDB_REFLECT_POINTER_ARRAY_VERSIONED(type, name, name_size, version, reflectHints, metadata) PNANOVDB_REFLECT_GENERIC(POINTER_ARRAY_VERSIONED, type, name, PNANOVDB_REFLECT_SIZE_OFFSET(name_size), PNANOVDB_REFLECT_VERSION_OFFSET(version), reflectHints, metadata)

/// Function Pointer
static const pnanovdb_reflect_data_type_t function_pnanovdb_reflect_data_type = { PNANOVDB_REFLECT_TYPE_FUNCTION, 0llu, 0, 0, 0, 0 };
#define PNANOVDB_REFLECT_FUNCTION_POINTER(name, reflectHints, metadata) { \
    reflectHints, \
    PNANOVDB_REFLECT_MODE_POINTER, \
    &function_pnanovdb_reflect_data_type, \
    #name, \
    (pnanovdb_uint64_t)(&((PNANOVDB_REFLECT_TYPE*)0)->name), \
    0, \
    0, \
    metadata, \
    },

/// Void
static const pnanovdb_reflect_data_type_t void_pnanovdb_reflect_data_type = { PNANOVDB_REFLECT_TYPE_VOID, 0llu, 0, 0, 0, 0 };
#define PNANOVDB_REFLECT_VOID_POINTER(name, reflectHints, metadata) { \
    reflectHints, \
    PNANOVDB_REFLECT_MODE_POINTER, \
    &void_pnanovdb_reflect_data_type, \
    #name, \
    (pnanovdb_uint64_t)(&((PNANOVDB_REFLECT_TYPE*)0)->name), \
    0, \
    0, \
    metadata, \
    },

/// Enum
#define PNANOVDB_REFLECT_ENUM(name, reflectHints, metadata) { \
    reflectHints, \
    PNANOVDB_REFLECT_MODE_VALUE, \
    &pnanovdb_uint32_t_pnanovdb_reflect_data_type, \
    #name, \
    (pnanovdb_uint64_t)(&((PNANOVDB_REFLECT_TYPE*)0)->name), \
    0, \
    0, \
    metadata, \
    },

#define PNANOVDB_REFLECT_INTERFACE() const pnanovdb_reflect_data_type_t* interface_pnanovdb_reflect_data_type

#define PNANOVDB_REFLECT_INTERFACE_INIT(type) &PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_reflect_data_type)

#define PNANOVDB_REFLECT_INTERFACE_IMPL() \
    PNANOVDB_INLINE void PNANOVDB_REFLECT_XCONCAT(PNANOVDB_REFLECT_TYPE,_duplicate)(PNANOVDB_REFLECT_TYPE* dst, const PNANOVDB_REFLECT_TYPE* src) \
    { \
        const pnanovdb_reflect_data_type_t* dst_type = &PNANOVDB_REFLECT_XCONCAT(PNANOVDB_REFLECT_TYPE,_pnanovdb_reflect_data_type); \
        pnanovdb_reflect_copy_by_name( \
            dst, dst_type, \
            src, src->interface_pnanovdb_reflect_data_type  \
        ); \
        dst->interface_pnanovdb_reflect_data_type = dst_type; \
    }

#define PNANOVDB_REFLECT_DATA_TYPE(type) &PNANOVDB_REFLECT_XCONCAT(type,_pnanovdb_reflect_data_type)

#define PNANOVDB_REFLECT_TYPE pnanovdb_reflect_data_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_VALUE(pnanovdb_uint32_t, reflect_hint, 0, 0)
PNANOVDB_REFLECT_VALUE(pnanovdb_uint32_t, reflect_mode, 0, 0)
PNANOVDB_REFLECT_VOID_POINTER(/*pnanovdb_reflect_data_type_t,*/ data_type, 0, 0)    // void to break circular reference
PNANOVDB_REFLECT_POINTER(char, name, 0, 0)
PNANOVDB_REFLECT_VALUE(pnanovdb_uint64_t, data_offset, 0, 0)
PNANOVDB_REFLECT_VALUE(pnanovdb_uint64_t, array_size_offset, 0, 0)
PNANOVDB_REFLECT_END(0)
#undef PNANOVDB_REFLECT_TYPE

#define PNANOVDB_REFLECT_TYPE pnanovdb_reflect_data_type_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_ENUM(data_type, 0, 0)
PNANOVDB_REFLECT_VALUE(pnanovdb_uint64_t, element_size, 0, 0)
PNANOVDB_REFLECT_POINTER(char, struct_typename, 0, 0)
PNANOVDB_REFLECT_ARRAY(pnanovdb_reflect_data_t, child_reflect_datas, child_reflect_data_count, 0, 0)
PNANOVDB_REFLECT_VOID_POINTER(default_value, 0, 0)
PNANOVDB_REFLECT_END(0)
#undef PNANOVDB_REFLECT_TYPE

#endif // end of NANOVDB_PUTILS_REFLECT_H_HAS_BEEN_INCLUDED
