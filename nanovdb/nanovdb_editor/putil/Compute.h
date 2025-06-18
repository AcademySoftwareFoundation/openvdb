// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_editor/putil/Compute.h

    \author Andrew Reidmeyer

    \brief  This file provides a GPU compute abstraction.
*/

#ifndef NANOVDB_PUTILS_COMPUTE_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_COMPUTE_H_HAS_BEEN_INCLUDED

#include "nanovdb_editor/putil/Compiler.h"

/// ********************************* GPU Data Formats ***************************************

typedef pnanovdb_uint32_t pnanovdb_compute_format_t;
#define PNANOVDB_COMPUTE_FORMAT_UNKNOWN 0
// 128-bit
#define PNANOVDB_COMPUTE_FORMAT_R32G32B32A32_FLOAT 1
#define PNANOVDB_COMPUTE_FORMAT_R32G32B32A32_UINT 2
#define PNANOVDB_COMPUTE_FORMAT_R32G32B32A32_SINT 3
// 64-bit
#define PNANOVDB_COMPUTE_FORMAT_R16G16B16A16_FLOAT 4
#define PNANOVDB_COMPUTE_FORMAT_R16G16B16A16_UNORM 5
#define PNANOVDB_COMPUTE_FORMAT_R16G16B16A16_UINT 6
#define PNANOVDB_COMPUTE_FORMAT_R16G16B16A16_SNORM 7
#define PNANOVDB_COMPUTE_FORMAT_R16G16B16A16_SINT 8
#define PNANOVDB_COMPUTE_FORMAT_R32G32_FLOAT 9
#define PNANOVDB_COMPUTE_FORMAT_R32G32_UINT 10
#define PNANOVDB_COMPUTE_FORMAT_R32G32_SINT 11
// 32-bit
#define PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_UNORM 12
#define PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_UNORM_SRGB 13
#define PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_UINT 14
#define PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_SNORM 15
#define PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_SINT 16
#define PNANOVDB_COMPUTE_FORMAT_B8G8R8A8_UNORM 17
#define PNANOVDB_COMPUTE_FORMAT_B8G8R8A8_UNORM_SRGB 18
#define PNANOVDB_COMPUTE_FORMAT_R16G16_FLOAT 19
#define PNANOVDB_COMPUTE_FORMAT_R16G16_UNORM 20
#define PNANOVDB_COMPUTE_FORMAT_R16G16_UINT 21
#define PNANOVDB_COMPUTE_FORMAT_R16G16_SNORM 22
#define PNANOVDB_COMPUTE_FORMAT_R16G16_SINT 23
#define PNANOVDB_COMPUTE_FORMAT_R32_FLOAT 24
#define PNANOVDB_COMPUTE_FORMAT_R32_UINT 25
#define PNANOVDB_COMPUTE_FORMAT_R32_SINT 26
// 16-bit
#define PNANOVDB_COMPUTE_FORMAT_R8G8_UNORM 27
#define PNANOVDB_COMPUTE_FORMAT_R8G8_UINT 28
#define PNANOVDB_COMPUTE_FORMAT_R8G8_SNORM 29
#define PNANOVDB_COMPUTE_FORMAT_R8G8_SINT 30
#define PNANOVDB_COMPUTE_FORMAT_R16_FLOAT 31
#define PNANOVDB_COMPUTE_FORMAT_R16_UNORM 32
#define PNANOVDB_COMPUTE_FORMAT_R16_UINT 33
#define PNANOVDB_COMPUTE_FORMAT_R16_SNORM 34
#define PNANOVDB_COMPUTE_FORMAT_R16_SINT 35
// 8-bit
#define PNANOVDB_COMPUTE_FORMAT_R8_UNORM 36
#define PNANOVDB_COMPUTE_FORMAT_R8_UINT 37
#define PNANOVDB_COMPUTE_FORMAT_R8_SNORM 38
#define PNANOVDB_COMPUTE_FORMAT_R8_SINT 39
// Reserved maximum count
#define PNANOVDB_COMPUTE_FORMAT_COUNT 256

/// ********************************* Compute Context ***************************************

struct pnanovdb_compute_context_t;
typedef struct pnanovdb_compute_context_t pnanovdb_compute_context_t;

PNANOVDB_REFLECT_STRUCT_OPAQUE_IMPL(pnanovdb_compute_context_t)

typedef pnanovdb_uint32_t pnanovdb_compute_api_t;
#define PNANOVDB_COMPUTE_API_ABSTRACT 0
#define PNANOVDB_COMPUTE_API_VULKAN 1

typedef struct pnanovdb_compute_context_config_t
{
    pnanovdb_compute_api_t api;
}pnanovdb_compute_context_config_t;

typedef struct pnanovdb_compute_bytecode_t
{
    const void* data;
    pnanovdb_uint64_t size_in_bytes;
}pnanovdb_compute_bytecode_t;

struct pnanovdb_compute_buffer_t;
typedef struct pnanovdb_compute_buffer_t pnanovdb_compute_buffer_t;

struct pnanovdb_compute_buffer_transient_t;
typedef struct pnanovdb_compute_buffer_transient_t pnanovdb_compute_buffer_transient_t;

struct pnanovdb_compute_buffer_acquire_t;
typedef struct pnanovdb_compute_buffer_acquire_t pnanovdb_compute_buffer_acquire_t;

typedef pnanovdb_uint32_t pnanovdb_compute_memory_type_t;
#define PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE 0
#define PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD 1
#define PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK 2

typedef pnanovdb_uint32_t pnanovdb_compute_buffer_usage_t;
#define PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT 1
#define PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED 2
#define PNANOVDB_COMPUTE_BUFFER_USAGE_BUFFER 4
#define PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED 8
#define PNANOVDB_COMPUTE_BUFFER_USAGE_RW_BUFFER 16
#define PNANOVDB_COMPUTE_BUFFER_USAGE_INDIRECT 32
#define PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC 64
#define PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST 128

typedef struct pnanovdb_compute_buffer_desc_t
{
    pnanovdb_compute_buffer_usage_t usage;
    pnanovdb_compute_format_t format;
    pnanovdb_uint32_t structure_stride;
    pnanovdb_uint64_t size_in_bytes;
}pnanovdb_compute_buffer_desc_t;

struct pnanovdb_compute_texture_t;
typedef struct pnanovdb_compute_texture_t pnanovdb_compute_texture_t;

struct pnanovdb_compute_texture_transient_t;
typedef struct pnanovdb_compute_texture_transient_t pnanovdb_compute_texture_transient_t;

struct pnanovdb_compute_texture_acquire_t;
typedef struct pnanovdb_compute_texture_acquire_t pnanovdb_compute_texture_acquire_t;

struct pnanovdb_compute_sampler_t;
typedef struct pnanovdb_compute_sampler_t pnanovdb_compute_sampler_t;

typedef pnanovdb_uint32_t pnanovdb_compute_texture_type_t;
#define PNANOVDB_COMPUTE_TEXTURE_TYPE_1D 0
#define PNANOVDB_COMPUTE_TEXTURE_TYPE_2D 1
#define PNANOVDB_COMPUTE_TEXTURE_TYPE_3D 2

typedef pnanovdb_uint32_t pnanovdb_compute_texture_usage_t;
#define PNANOVDB_COMPUTE_TEXTURE_USAGE_TEXTURE 1
#define PNANOVDB_COMPUTE_TEXTURE_USAGE_RW_TEXTURE 2
#define PNANOVDB_COMPUTE_TEXTURE_USAGE_COPY_SRC 4
#define PNANOVDB_COMPUTE_TEXTURE_USAGE_COPY_DST 8

typedef struct pnanovdb_compute_texture_desc_t
{
    pnanovdb_compute_texture_type_t texture_type;
    pnanovdb_compute_texture_usage_t usage;
    pnanovdb_compute_format_t format;
    pnanovdb_uint32_t width;
    pnanovdb_uint32_t height;
    pnanovdb_uint32_t depth;
    pnanovdb_uint32_t mip_levels;
}pnanovdb_compute_texture_desc_t;

typedef pnanovdb_uint32_t pnanovdb_compute_sampler_address_mode_t;
#define PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_WRAP 0
#define PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_CLAMP 1
#define PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_MIRROR 2
#define PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_BORDER 3

typedef pnanovdb_uint32_t pnanovdb_compute_sampler_filter_mode_t;
#define PNANOVDB_COMPUTE_SAMPLER_FILTER_MODE_POINT 0
#define PNANOVDB_COMPUTE_SAMPLER_FILTER_MODE_LINEAR 1

typedef struct pnanovdb_compute_sampler_desc_t
{
    pnanovdb_compute_sampler_address_mode_t address_mode_u;
    pnanovdb_compute_sampler_address_mode_t address_mode_v;
    pnanovdb_compute_sampler_address_mode_t address_mode_w;
    pnanovdb_compute_sampler_filter_mode_t filter_mode;
}pnanovdb_compute_sampler_desc_t;

typedef pnanovdb_uint32_t pnanovdb_compute_descriptor_type_t;
// used in shaders and barrier
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_UNKNOWN 0
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_CONSTANT_BUFFER 1
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_STRUCTURED_BUFFER 2
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER 3
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE 4
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_SAMPLER 5
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_STRUCTURED_BUFFER 6
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_BUFFER 7
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_TEXTURE 8
// used only for barriers
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_INDIRECT_BUFFER 9
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER_COPY_SRC 10
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER_COPY_DST 11
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE_COPY_SRC 12
#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE_COPY_DST 13

#define PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_COUNT 14

typedef struct pnanovdb_compute_resource_t
{
    pnanovdb_compute_buffer_transient_t* buffer_transient;
    pnanovdb_compute_texture_transient_t* texture_transient;
    pnanovdb_compute_sampler_t* sampler;
}pnanovdb_compute_resource_t;

typedef struct pnanovdb_compute_descriptor_write_vulkan_t
{
    pnanovdb_uint32_t binding;
    pnanovdb_uint32_t array_index;
    pnanovdb_uint32_t set;
}pnanovdb_compute_descriptor_write_vulkan_t;

typedef struct pnanovdb_compute_descriptor_write_union_t
{
    pnanovdb_compute_descriptor_write_vulkan_t vulkan;
    // optionally other low level API here
}pnanovdb_compute_descriptor_write_union_t;

typedef struct pnanovdb_compute_descriptor_write_t
{
    pnanovdb_compute_descriptor_type_t type;
    pnanovdb_compute_descriptor_write_union_t write;
}pnanovdb_compute_descriptor_write_t;

typedef struct pnanovdb_compute_binding_desc_vulkan_t
{
    pnanovdb_uint32_t binding;
    pnanovdb_uint32_t descriptor_count;
    pnanovdb_uint32_t set;
}pnanovdb_compute_binding_desc_vulkan_t;

typedef struct pnanovdb_compute_binding_desc_union_t
{
    pnanovdb_compute_binding_desc_vulkan_t vulkan;
    // optionally other low level API here
}pnanovdb_compute_binding_desc_union_t;

typedef struct pnanovdb_compute_binding_desc_t
{
    pnanovdb_compute_descriptor_type_t type;
    pnanovdb_compute_binding_desc_union_t binding_desc;
}pnanovdb_compute_binding_desc_t;

struct pnanovdb_compute_pipeline_t;
typedef struct pnanovdb_compute_pipeline_t pnanovdb_compute_pipeline_t;

typedef struct pnanovdb_compute_pipeline_desc_t
{
    pnanovdb_compute_binding_desc_t* binding_descs;
    pnanovdb_uint32_t binding_desc_count;

    pnanovdb_compute_bytecode_t bytecode;
}pnanovdb_compute_pipeline_desc_t;

typedef struct pnanovdb_compute_dispatch_params_t
{
    pnanovdb_compute_pipeline_t* pipeline;
    pnanovdb_uint32_t grid_dim_x;
    pnanovdb_uint32_t grid_dim_y;
    pnanovdb_uint32_t grid_dim_z;

    const pnanovdb_compute_descriptor_write_t* descriptor_writes;
    const pnanovdb_compute_resource_t* resources;
    pnanovdb_uint32_t descriptor_write_count;

    const char* debug_label;
}pnanovdb_compute_dispatch_params_t;

typedef struct pnanovdb_compute_copy_buffer_params_t
{
    pnanovdb_uint64_t src_offset;
    pnanovdb_uint64_t dst_offset;
    pnanovdb_uint64_t num_bytes;

    pnanovdb_compute_buffer_transient_t* src;
    pnanovdb_compute_buffer_transient_t* dst;

    const char* debug_label;
}pnanovdb_compute_copy_buffer_params_t;

typedef pnanovdb_uint32_t pnanovdb_compute_interop_handle_type_t;
#define PNANOVDB_COMPUTE_INTEROP_HANDLE_TYPE_UNKNOWN 0
#define PNANOVDB_COMPUTE_INTEROP_HANDLE_TYPE_OPAQUE_FD 1
#define PNANOVDB_COMPUTE_INTEROP_HANDLE_TYPE_OPAQUE_WIN32 2

typedef struct pnanovdb_compute_interop_handle_t
{
    pnanovdb_compute_interop_handle_type_t type;
    pnanovdb_uint64_t value;
    pnanovdb_uint64_t resource_size_in_bytes;
}pnanovdb_compute_interop_handle_t;

#define pnanovdb_compute_interop_handle_default_init { \
    PNANOVDB_COMPUTE_INTEROP_HANDLE_TYPE_UNKNOWN, /*type*/ \
    0u, /*value*/  \
    0u, /*resource_size_in_bytes*/  \
}
static const pnanovdb_compute_interop_handle_t pnanovdb_compute_interop_handle_default = pnanovdb_compute_interop_handle_default_init;

PNANOVDB_REFLECT_STRUCT_OPAQUE_IMPL(pnanovdb_compute_interop_handle_t)

typedef pnanovdb_uint32_t pnanovdb_compute_feature_t;
#define PNANOVDB_COMPUTE_FEATURE_UNKNOWN 0
#define PNANOVDB_COMPUTE_FEATURE_ALIAS_RESOURCE_FORMATS 1
#define PNANOVDB_COMPUTE_FEATURE_BUFFER_EXTERNAL_HANDLE 2

typedef pnanovdb_uint32_t pnanovdb_compute_log_level_t;
#define PNANOVDB_COMPUTE_LOG_LEVEL_ERROR 0
#define PNANOVDB_COMPUTE_LOG_LEVEL_WARNING 1
#define PNANOVDB_COMPUTE_LOG_LEVEL_INFO 2

typedef void(PNANOVDB_ABI* pnanovdb_compute_log_print_t)(pnanovdb_compute_log_level_t level, const char* format, ...);

typedef void(PNANOVDB_ABI*pnanovdb_compute_thread_pool_task_t)(pnanovdb_uint32_t task_idx, pnanovdb_uint32_t thread_idx, void* shared_mem, void* userdata);

typedef struct pnanovdb_compute_frame_info_t
{
    pnanovdb_uint64_t frame_local_current;
    pnanovdb_uint64_t frame_local_completed;
    pnanovdb_uint64_t frame_global_current;
    pnanovdb_uint64_t frame_global_completed;
}pnanovdb_compute_frame_info_t;

typedef struct pnanovdb_compute_interface_t
{
    PNANOVDB_REFLECT_INTERFACE();

    pnanovdb_compute_api_t(PNANOVDB_ABI* get_api)(pnanovdb_compute_context_t* context);

    pnanovdb_bool_t(PNANOVDB_ABI* is_feature_supported)(pnanovdb_compute_context_t* context, pnanovdb_compute_feature_t feature);

    void(PNANOVDB_ABI* get_frame_info)(pnanovdb_compute_context_t* context, pnanovdb_compute_frame_info_t* frame_info);

    pnanovdb_compute_log_print_t(PNANOVDB_ABI* get_log_print)(pnanovdb_compute_context_t* context);


    void(PNANOVDB_ABI* execute_tasks)(pnanovdb_compute_context_t* context, pnanovdb_uint32_t task_count, pnanovdb_uint32_t task_granularity, pnanovdb_compute_thread_pool_task_t task, void* userdata);


    pnanovdb_compute_buffer_t*(PNANOVDB_ABI* create_buffer)(pnanovdb_compute_context_t* context, pnanovdb_compute_memory_type_t memory_type, const pnanovdb_compute_buffer_desc_t* desc);

    void(PNANOVDB_ABI* destroy_buffer)(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer);

    pnanovdb_compute_buffer_transient_t*(PNANOVDB_ABI* get_buffer_transient)(pnanovdb_compute_context_t* context, const pnanovdb_compute_buffer_desc_t* desc);

    pnanovdb_compute_buffer_transient_t*(PNANOVDB_ABI* register_buffer_as_transient)(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer);

    pnanovdb_compute_buffer_transient_t* (PNANOVDB_ABI* alias_buffer_transient)(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_transient_t* buffer, pnanovdb_compute_format_t format, pnanovdb_uint32_t structure_stride);

    pnanovdb_compute_buffer_acquire_t*(PNANOVDB_ABI* enqueue_acquire_buffer)(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_transient_t* buffer);

    pnanovdb_bool_t(PNANOVDB_ABI* get_acquired_buffer)(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_acquire_t* acquire, pnanovdb_compute_buffer_t** outBuffer);

    void*(PNANOVDB_ABI* map_buffer)(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer);

    void(PNANOVDB_ABI* unmap_buffer)(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer);

    void(PNANOVDB_ABI* get_buffer_external_handle)(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer, pnanovdb_compute_interop_handle_t* dstHandle);

    void(PNANOVDB_ABI* close_buffer_external_handle)(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer, const pnanovdb_compute_interop_handle_t* srcHandle);

    pnanovdb_compute_buffer_t*(PNANOVDB_ABI* create_buffer_from_external_handle)(pnanovdb_compute_context_t* context, const pnanovdb_compute_buffer_desc_t* desc, const pnanovdb_compute_interop_handle_t* interopHandle);


    pnanovdb_compute_texture_t*(PNANOVDB_ABI* create_texture)(pnanovdb_compute_context_t* context, const pnanovdb_compute_texture_desc_t* desc);

    void(PNANOVDB_ABI* destroy_texture)(pnanovdb_compute_context_t* context, pnanovdb_compute_texture_t* texture);

    pnanovdb_compute_texture_transient_t*(PNANOVDB_ABI* get_texture_transient)(pnanovdb_compute_context_t* context, const pnanovdb_compute_texture_desc_t* desc);

    pnanovdb_compute_texture_transient_t*(PNANOVDB_ABI* register_texture_as_transient)(pnanovdb_compute_context_t* context, pnanovdb_compute_texture_t* texture);

    pnanovdb_compute_texture_transient_t* (PNANOVDB_ABI* alias_texture_transient)(pnanovdb_compute_context_t* context, pnanovdb_compute_texture_transient_t* texture, pnanovdb_compute_format_t format);

    pnanovdb_compute_texture_acquire_t*(PNANOVDB_ABI* enqueue_acquire_texture)(pnanovdb_compute_context_t* context, pnanovdb_compute_texture_transient_t* texture);

    pnanovdb_bool_t(PNANOVDB_ABI* get_acquired_texture)(pnanovdb_compute_context_t* context, pnanovdb_compute_texture_acquire_t* acquire, pnanovdb_compute_texture_t** outTexture);


    pnanovdb_compute_sampler_t*(PNANOVDB_ABI* create_sampler)(pnanovdb_compute_context_t* context, const pnanovdb_compute_sampler_desc_t* desc);

    pnanovdb_compute_sampler_t*(PNANOVDB_ABI* get_default_sampler)(pnanovdb_compute_context_t* context);

    void(PNANOVDB_ABI* destroy_sampler)(pnanovdb_compute_context_t* context, pnanovdb_compute_sampler_t* sampler);


    pnanovdb_compute_pipeline_t*(PNANOVDB_ABI* create_compute_pipeline)(pnanovdb_compute_context_t* context, const pnanovdb_compute_pipeline_desc_t* desc);

    void(PNANOVDB_ABI* destroy_compute_pipeline)(pnanovdb_compute_context_t* context, pnanovdb_compute_pipeline_t* pipeline);


    void(PNANOVDB_ABI* dispatch)(pnanovdb_compute_context_t* context, const pnanovdb_compute_dispatch_params_t* params);

    void(PNANOVDB_ABI* copy_buffer)(pnanovdb_compute_context_t* context, const pnanovdb_compute_copy_buffer_params_t* params);

}pnanovdb_compute_interface_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_compute_interface_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_FUNCTION_POINTER(get_api, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(is_feature_supported, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_frame_info, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_log_print, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(execute_tasks, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_buffer, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_buffer, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_buffer_transient, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(register_buffer_as_transient, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(alias_buffer_transient, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(enqueue_acquire_buffer, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_acquired_buffer, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(map_buffer, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(unmap_buffer, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_buffer_external_handle, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(close_buffer_external_handle, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_buffer_from_external_handle, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_texture, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_texture, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_texture_transient, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(register_texture_as_transient, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(alias_texture_transient, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(enqueue_acquire_texture, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_acquired_texture, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_sampler, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_default_sampler, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_sampler, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_compute_pipeline, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_compute_pipeline, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(dispatch, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(copy_buffer, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

/// ********************************* Compute Device ***************************************

struct pnanovdb_compute_device_manager_t;
typedef struct pnanovdb_compute_device_manager_t pnanovdb_compute_device_manager_t;

typedef struct pnanovdb_compute_physical_device_desc_t
{
    pnanovdb_uint8_t device_uuid[16u];
    pnanovdb_uint8_t device_luid[8u];
    pnanovdb_uint32_t device_node_mask;
    pnanovdb_bool_t device_luid_valid;
}pnanovdb_compute_physical_device_desc_t;

typedef struct pnanovdb_compute_device_desc_t
{
    pnanovdb_uint32_t device_index;
    pnanovdb_bool_t enable_external_usage;
    pnanovdb_compute_log_print_t log_print;
}pnanovdb_compute_device_desc_t;

struct pnanovdb_compute_swapchain_desc_t;
typedef struct pnanovdb_compute_swapchain_desc_t pnanovdb_compute_swapchain_desc_t;

#if defined(PNANOVDB_SWAPCHAIN_DESC)
struct pnanovdb_compute_swapchain_desc_t
{
#if defined(_WIN32)
    HINSTANCE hinstance;
    HWND hwnd;
#elif defined(__APPLE__)
    void* nsview;
    void* window_userdata;
    void(*get_framebuffer_size)(void* window_userdata, int* width, int* height);
    void(*create_surface)(void* window_userdata, void* vkinstance, void** out_surface);
#else
    Display* dpy;
    Window window;
#endif
    pnanovdb_compute_format_t format;
};
#endif

struct pnanovdb_compute_device_t;
typedef struct pnanovdb_compute_device_t pnanovdb_compute_device_t;

struct pnanovdb_compute_queue_t;
typedef struct pnanovdb_compute_queue_t pnanovdb_compute_queue_t;

struct pnanovdb_compute_semaphore_t;
typedef struct pnanovdb_compute_semaphore_t pnanovdb_compute_semaphore_t;

struct pnanovdb_compute_swapchain_t;
typedef struct pnanovdb_compute_swapchain_t pnanovdb_compute_swapchain_t;

typedef struct pnanovdb_compute_profiler_entry_t
{
    const char* label;
    float cpu_delta_time;
    float gpu_delta_time;
}pnanovdb_compute_profiler_entry_t;

typedef struct pnanovdb_compute_device_memory_stats_t
{
    pnanovdb_uint64_t device_memory_bytes;
    pnanovdb_uint64_t upload_memory_bytes;
    pnanovdb_uint64_t readback_memory_bytes;
    pnanovdb_uint64_t other_memory_bytes;
}pnanovdb_compute_device_memory_stats_t;

typedef void(PNANOVDB_ABI* pnanovdb_profiler_report_t)(void* userdata, pnanovdb_uint64_t capture_id, pnanovdb_uint32_t num_entries, pnanovdb_compute_profiler_entry_t* entries);

typedef struct pnanovdb_compute_device_interface_t
{
    PNANOVDB_REFLECT_INTERFACE();

    pnanovdb_compute_device_manager_t*(PNANOVDB_ABI* create_device_manager)(pnanovdb_bool_t enable_validation_on_debug_build);

    void(PNANOVDB_ABI* destroy_device_manager)(pnanovdb_compute_device_manager_t* manager);

    pnanovdb_bool_t(PNANOVDB_ABI* enumerate_devices)(pnanovdb_compute_device_manager_t* manager, pnanovdb_uint32_t device_index, pnanovdb_compute_physical_device_desc_t* p_desc);


    pnanovdb_compute_device_t*(PNANOVDB_ABI* create_device)(pnanovdb_compute_device_manager_t* manager, const pnanovdb_compute_device_desc_t* desc);

    void(PNANOVDB_ABI* destroy_device)(pnanovdb_compute_device_manager_t* manager, pnanovdb_compute_device_t* device);

    void(PNANOVDB_ABI* get_memory_stats)(pnanovdb_compute_device_t* device, pnanovdb_compute_device_memory_stats_t* dst_stats);


    pnanovdb_compute_semaphore_t*(PNANOVDB_ABI* create_semaphore)(pnanovdb_compute_device_t* device);

    void(PNANOVDB_ABI* destroy_semaphore)(pnanovdb_compute_semaphore_t* semaphore);

    void(PNANOVDB_ABI* get_semaphore_external_handle)(pnanovdb_compute_semaphore_t* semaphore, void* dst_handle, pnanovdb_uint64_t dst_handle_size);

    void(PNANOVDB_ABI* close_semaphore_external_handle)(pnanovdb_compute_semaphore_t* semaphore, const void* src_handle, pnanovdb_uint64_t src_handle_size);


    pnanovdb_compute_queue_t*(PNANOVDB_ABI* get_device_queue)(const pnanovdb_compute_device_t* device);

    pnanovdb_compute_queue_t*(PNANOVDB_ABI* get_compute_queue)(const pnanovdb_compute_device_t* device);

    int(PNANOVDB_ABI* flush)(pnanovdb_compute_queue_t* queue, pnanovdb_uint64_t* flushed_frame, pnanovdb_compute_semaphore_t* waitSemaphore, pnanovdb_compute_semaphore_t* signalSemaphore);

    pnanovdb_uint64_t(PNANOVDB_ABI* get_frame_global_completed)(pnanovdb_compute_queue_t* queue);

    void(PNANOVDB_ABI* wait_for_frame)(pnanovdb_compute_queue_t* queue, pnanovdb_uint64_t frame);

    void(PNANOVDB_ABI* wait_idle)(pnanovdb_compute_queue_t* queue);

    pnanovdb_compute_interface_t*(PNANOVDB_ABI* get_compute_interface)(const pnanovdb_compute_queue_t* queue);

    pnanovdb_compute_context_t*(PNANOVDB_ABI* get_compute_context)(const pnanovdb_compute_queue_t* queue);


    pnanovdb_compute_swapchain_t*(PNANOVDB_ABI* create_swapchain)(pnanovdb_compute_queue_t* queue, const pnanovdb_compute_swapchain_desc_t* desc);

    void(PNANOVDB_ABI* destroy_swapchain)(pnanovdb_compute_swapchain_t* swapchain);

    void(PNANOVDB_ABI* resize_swapchain)(pnanovdb_compute_swapchain_t* swapchain, pnanovdb_uint32_t width, pnanovdb_uint32_t height);

    int(PNANOVDB_ABI* present_swapchain)(pnanovdb_compute_swapchain_t* swapchain, pnanovdb_bool_t vsync, pnanovdb_uint64_t* flushedFrameID);

    pnanovdb_compute_texture_t*(PNANOVDB_ABI* get_swapchain_front_texture)(pnanovdb_compute_swapchain_t* swapchain);


    void(PNANOVDB_ABI* enable_profiler)(pnanovdb_compute_context_t* context, void* userdata, pnanovdb_profiler_report_t report_entries);

    void(PNANOVDB_ABI* disable_profiler)(pnanovdb_compute_context_t* context);

    void(PNANOVDB_ABI* set_resource_min_lifetime)(pnanovdb_compute_context_t* context, pnanovdb_uint64_t min_lifetime);

}pnanovdb_compute_device_interface_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_compute_device_interface_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_FUNCTION_POINTER(create_device_manager, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_device_manager, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(enumerate_devices, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_device, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_device, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_memory_stats, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_semaphore, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_semaphore, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_semaphore_external_handle, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(close_semaphore_external_handle, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_device_queue, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_compute_queue, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(flush, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_frame_global_completed, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(wait_for_frame, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(wait_idle, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_compute_interface, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_compute_context, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_swapchain, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_swapchain, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(resize_swapchain, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(present_swapchain, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(get_swapchain_front_texture, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(enable_profiler, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(disable_profiler, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(set_resource_min_lifetime, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_compute_device_interface_t* (PNANOVDB_ABI* PFN_pnanovdb_get_compute_device_interface)(pnanovdb_compute_api_t api);

PNANOVDB_API pnanovdb_compute_device_interface_t* pnanovdb_get_compute_device_interface(pnanovdb_compute_api_t api);

/// ********************************* Compute Shader ***************************************

struct pnanovdb_compute_shader_t;
typedef struct pnanovdb_compute_shader_t pnanovdb_compute_shader_t;

struct pnanovdb_shader_context_t;
typedef struct pnanovdb_shader_context_t pnanovdb_shader_context_t;

typedef const char*(PNANOVDB_ABI* pnanovdb_compute_shader_get_source_t)(void* userdata, const char* path);

typedef struct pnanovdb_compute_shader_source_t
{
    const char* source;
    const char* source_filename;
    pnanovdb_compute_shader_get_source_t get_source_include;
    void* get_source_include_userdata;
}pnanovdb_compute_shader_source_t;

typedef struct pnanovdb_compute_shader_build_t
{
    pnanovdb_compute_pipeline_desc_t pipeline_desc;
    const pnanovdb_compute_descriptor_write_t* descriptor_writes;
    const char** resource_names;
    pnanovdb_uint32_t descriptor_write_count;
    const char* debug_label;
}pnanovdb_compute_shader_build_t;

typedef struct pnanovdb_compute_shader_interface_t
{
    PNANOVDB_REFLECT_INTERFACE();

    pnanovdb_compute_shader_t*(PNANOVDB_ABI* create_shader)(const pnanovdb_compute_shader_source_t* source);

    pnanovdb_bool_t(PNANOVDB_ABI* map_shader_build)(pnanovdb_compute_shader_t* shader, pnanovdb_compute_shader_build_t** out_build);

    void(PNANOVDB_ABI* destroy_shader)(pnanovdb_compute_shader_t* shader);

}pnanovdb_compute_shader_interface_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_compute_shader_interface_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_FUNCTION_POINTER(create_shader, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(map_shader_build, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_shader, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_compute_shader_interface_t* (PNANOVDB_ABI* PFN_pnanovdb_get_compute_shader_interface)();

PNANOVDB_API pnanovdb_compute_shader_interface_t* pnanovdb_get_compute_shader_interface();

/// ********************************* Compute ***************************************

struct pnanovdb_compute_t;
typedef struct pnanovdb_compute_t pnanovdb_compute_t;

typedef struct pnanovdb_compute_array_t
{
    void* data;
    pnanovdb_uint64_t element_size;
    pnanovdb_uint64_t element_count;
    const char* filepath;
}pnanovdb_compute_array_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_compute_array_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_VOID_POINTER(data, 0, 0)
PNANOVDB_REFLECT_VALUE(pnanovdb_uint64_t, element_size, 0, 0)
PNANOVDB_REFLECT_VALUE(pnanovdb_uint64_t, element_count, 0, 0)
PNANOVDB_REFLECT_POINTER(char, filepath, 0, 0)
PNANOVDB_REFLECT_END(0)
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_uint32_t pnanovdb_compiler_api_t;

typedef struct pnanovdb_compute_t
{
    PNANOVDB_REFLECT_INTERFACE();

    const pnanovdb_compiler_t* compiler;
    pnanovdb_compute_shader_interface_t shader_interface;
    pnanovdb_compute_device_interface_t device_interface;

    pnanovdb_compute_array_t* (PNANOVDB_ABI* load_nanovdb)(const char* filepath);
    pnanovdb_bool_t (PNANOVDB_ABI* save_nanovdb)(pnanovdb_compute_array_t* array, const char* filepath);
    pnanovdb_shader_context_t* (PNANOVDB_ABI* create_shader_context)(const char* filename);
    void (PNANOVDB_ABI* destroy_shader_context)(const pnanovdb_compute_t* compute,
                                                pnanovdb_compute_queue_t* queue,
                                                pnanovdb_shader_context_t* context);
    pnanovdb_bool_t (PNANOVDB_ABI* init_shader)(const pnanovdb_compute_t* compute,
                                    pnanovdb_compute_queue_t* queue,
                                    pnanovdb_shader_context_t* shaderContext,
                                    pnanovdb_compiler_settings_t* compileSettings);
    void (PNANOVDB_ABI* destroy_shader)(pnanovdb_compute_interface_t* computeInterface,
                                        const pnanovdb_compute_shader_interface_t* shaderInterface,
                                        pnanovdb_compute_context_t* computeContext,
                                        pnanovdb_shader_context_t* shaderContext);
    void (PNANOVDB_ABI* dispatch_shader)(pnanovdb_compute_interface_t* contextInterface,
                                         pnanovdb_compute_context_t* computeContext,
                                         const pnanovdb_shader_context_t* shaderContext,
                                         pnanovdb_compute_resource_t* resources,
                                         pnanovdb_uint32_t grid_dim_x,
                                         pnanovdb_uint32_t grid_dim_y,
                                         pnanovdb_uint32_t grid_dim_z,
                                         const char* debug_label);
    pnanovdb_bool_t (PNANOVDB_ABI* dispatch_shader_on_array)(const pnanovdb_compute_t* compute,
                                                 const pnanovdb_compute_device_t* device,
                                                 const char* shader_path,
                                                 pnanovdb_uint32_t grid_dim_x,
                                                 pnanovdb_uint32_t grid_dim_y,
                                                 pnanovdb_uint32_t grid_dim_z,
                                                 pnanovdb_compute_array_t* data_in,
                                                 pnanovdb_compute_array_t* constants,
                                                 pnanovdb_compute_array_t* data_out,
                                                 pnanovdb_uint32_t dispatch_count,
                                                 pnanovdb_uint64_t scratch_size,
                                                 pnanovdb_uint64_t scratch_clear_size);
    pnanovdb_bool_t (PNANOVDB_ABI* dispatch_shader_on_nanovdb_array)(const pnanovdb_compute_t* compute,
                                                         const pnanovdb_compute_device_t* device,
                                                         const pnanovdb_shader_context_t* shaderContext,
                                                         pnanovdb_compute_array_t* nanovdb_array,
                                                         pnanovdb_int32_t image_width,
                                                         pnanovdb_int32_t image_height,
                                                         pnanovdb_compute_texture_t* background_image,
                                                         pnanovdb_compute_buffer_transient_t* upload_buffer,
                                                         pnanovdb_compute_buffer_transient_t* user_upload_buffer,
                                                         pnanovdb_compute_buffer_t** nanovdb_buffer,
                                                         pnanovdb_compute_buffer_transient_t** readback_buffer);
    pnanovdb_compute_array_t*(PNANOVDB_ABI* create_array)(size_t element_size, pnanovdb_uint64_t element_count, void* data);
    void (PNANOVDB_ABI* destroy_array)(pnanovdb_compute_array_t* array);
    void*(PNANOVDB_ABI* map_array)(pnanovdb_compute_array_t* array);
    void (PNANOVDB_ABI* unmap_array)(pnanovdb_compute_array_t* array);
    void (PNANOVDB_ABI* compute_array_print_range)(const pnanovdb_compute_t* compute,
                                                   pnanovdb_compute_log_print_t log_print,
                                                   const char* name,
                                                   pnanovdb_compute_array_t* arr,
                                                   pnanovdb_uint32_t channel_count);

    void* module;
}pnanovdb_compute_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_compute_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_POINTER(pnanovdb_compiler_t, compiler, 0, 0)
PNANOVDB_REFLECT_VALUE(pnanovdb_compute_shader_interface_t, shader_interface, 0, 0)
PNANOVDB_REFLECT_VALUE(pnanovdb_compute_device_interface_t, device_interface, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(load_nanovdb, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(save_nanovdb, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_shader_context, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_shader_context, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(init_shader, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_shader, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(dispatch_shader, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(dispatch_shader_on_array, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(dispatch_shader_on_nanovdb_array, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_array, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_array, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(map_array, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(unmap_array, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(compute_array_print_range, 0, 0)
PNANOVDB_REFLECT_VOID_POINTER(module, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_compute_t* (PNANOVDB_ABI* PFN_pnanovdb_get_compute)();

PNANOVDB_API pnanovdb_compute_t* pnanovdb_get_compute();

static void pnanovdb_compute_load(pnanovdb_compute_t* compute, const pnanovdb_compiler_t* compiler)
{
    void* compute_module = pnanovdb_load_library("pnanovdbcompute.dll", "libpnanovdbcompute.so", "libpnanovdbcompute.dylib");
    if (!compute_module)
    {
#if defined(_WIN32)
        printf("Error: Compute module failed to load\n");
#else
        printf("Error: Compute module failed to load: %s\n", dlerror());
#endif
        return;
    }

    PFN_pnanovdb_get_compute get_compute = (PFN_pnanovdb_get_compute)pnanovdb_get_proc_address(compute_module, "pnanovdb_get_compute");
    if (!get_compute)
    {
        printf("Error: Failed to acquire compute getter\n");
        return;
    }
    pnanovdb_compute_t_duplicate(compute, get_compute());
    if (!compute)
    {
        printf("Error: Failed to acquire compute\n");
        return;
    }

    PFN_pnanovdb_get_compute_shader_interface get_compute_shader_interface = (PFN_pnanovdb_get_compute_shader_interface)pnanovdb_get_proc_address(compute_module, "pnanovdb_get_compute_shader_interface");
    if (!get_compute_shader_interface)
    {
        printf("Error: Failed to acquire compute shader interface getter\n");
        return;
    }
    pnanovdb_compute_shader_interface_t_duplicate(&compute->shader_interface, get_compute_shader_interface());

    PFN_pnanovdb_get_compute_device_interface get_compute_device_interface = (PFN_pnanovdb_get_compute_device_interface)pnanovdb_get_proc_address(compute_module, "pnanovdb_get_compute_device_interface");
    if (!get_compute_device_interface)
    {
        printf("Error: Failed to acquire compute device interface getter\n");
        return;
    }
    pnanovdb_compute_device_interface_t_duplicate(&compute->device_interface, get_compute_device_interface(PNANOVDB_COMPUTE_API_VULKAN));

    compute->module = compute_module;
    compute->compiler = compiler;
}

static void pnanovdb_compute_free(pnanovdb_compute_t* compute)
{
    pnanovdb_free_library(compute->module);
}

#endif
