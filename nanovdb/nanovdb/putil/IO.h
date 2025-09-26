// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/putil/IO.h

    \author Andrew Reidmeyer

    \brief  This file contains IO utilities for PNanoVDB.
*/

#ifndef NANOVDB_PUTILS_IO_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_IO_H_HAS_BEEN_INCLUDED

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__CUDACC__)
#define PNANOVDB_IO_INLINE static __host__ __device__
#elif defined(_WIN32)
#define PNANOVDB_IO_INLINE static inline
#else
#define PNANOVDB_IO_INLINE static inline
#endif

// needs to match NANOVDB_DATA_ALIGNMENT from nanovdb/NanoVDB.h
#define PNANOVDB_DATA_ALIGNMENT 32
PNANOVDB_IO_INLINE void* pnanovdb_aligned_malloc(size_t alignment, size_t size)
{
    size_t aligned_size = size + alignment + sizeof(void*);
    void* alloc_ptr = malloc(aligned_size);
    void* ptr = 0;
    if (alloc_ptr)
    {
        ptr = (void*)(alignment * (((size_t)alloc_ptr + sizeof(void*) + alignment - 1u) / alignment));
        void** header_ptr = (void**)((size_t)ptr - sizeof(void*));
        *header_ptr = alloc_ptr;
    }
    return ptr;
}
PNANOVDB_IO_INLINE void  pnanovdb_aligned_free(void* ptr)
{
    if (ptr)
    {
        void** header_ptr = (void**)((size_t)ptr - sizeof(void*));
        free(*header_ptr);
    }
}

// abstraction for file-like read/write
typedef struct pnanovdb_io_buf_t
{
    void* file;
    void(*read)(void* file, uint64_t offset, void* dst_data, uint64_t size);
    void(*write)(void* file, uint64_t offset, void* src_data, uint64_t size);
}pnanovdb_io_buf_t;

// implementation for fread/fwrite
PNANOVDB_IO_INLINE void pnanovdb_io_buf_fread(void* file, uint64_t offset, void* dst_data, uint64_t size)
{
    fseek((FILE*)file, offset, SEEK_SET);
    fread(dst_data, size, 1u, (FILE*)file);
}
PNANOVDB_IO_INLINE void pnanovdb_io_buf_fwrite(void* file, uint64_t offset, void* src_data, uint64_t size)
{
    fseek((FILE*)file, offset, SEEK_SET);
    fwrite(src_data, size, 1u, (FILE*)file);
}

// implementation for dynamic buffer
typedef struct pnanovdb_io_dynamic_buf_t
{
    void* data;
    uint64_t size;
    uint64_t capacity;
}pnanovdb_io_dynamic_buf_t;
PNANOVDB_IO_INLINE void pnanovdb_io_dynamic_buf_read(void* file, uint64_t offset, void* dst_data, uint64_t size)
{
    pnanovdb_io_dynamic_buf_t* ptr = (pnanovdb_io_dynamic_buf_t*)file;
    uint8_t* dst8 = (uint8_t*)dst_data;
    uint8_t* src8 = (uint8_t*)ptr->data;
    uint64_t size_in_bounds = size;
    if (offset >= ptr->size)
    {
        size_in_bounds = 0;
    }
    else if (offset + size > ptr->size)
    {
        size_in_bounds = ptr->size - offset;
    }
    memcpy(dst8, src8 + offset, size_in_bounds);
    memset(dst8 + size_in_bounds, 0, size - size_in_bounds);
}
PNANOVDB_IO_INLINE void pnanovdb_io_dynamic_buf_write(void* file, uint64_t offset, void* src_data, uint64_t size)
{
    pnanovdb_io_dynamic_buf_t* ptr = (pnanovdb_io_dynamic_buf_t*)file;
    // grow capacity as needed
    if (offset + size > ptr->capacity)
    {
        uint64_t new_capacity = 1024u;
        while (new_capacity < offset + size)
        {
            new_capacity = (new_capacity << 1u) - (new_capacity >> 1u);
        }
        void* new_data = pnanovdb_aligned_malloc(PNANOVDB_DATA_ALIGNMENT, new_capacity);
        memcpy(new_data, ptr->data, ptr->size);
        if (ptr->capacity != 0u)
        {
            pnanovdb_aligned_free(ptr->data);
        }
        ptr->data = new_data;
        ptr->capacity = new_capacity;
    }
    // expand size and zero initialize as needed
    if (offset + size > ptr->size)
    {
        uint8_t* data8 = (uint8_t*)ptr->data;
        memset(data8 + ptr->size, 0, offset + size - ptr->size);
        ptr->size = offset + size;
    }
    // write data
    uint8_t* src8 = (uint8_t*)src_data;
    uint8_t* dst8 = (uint8_t*)ptr->data;
    memcpy(dst8 + offset, src8, size);
}

#if defined(PNANOVDB_IO_BUF_CUSTOM)
#define PNANOVDB_BUF_CUSTOM

#define pnanovdb_buf_t pnanovdb_io_buf_t*
PNANOVDB_IO_INLINE uint32_t pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint64_t byte_offset)
{
    uint32_t ret = 0u;
    buf->read(buf->file, byte_offset, &ret, 4u);
    return ret;
}
PNANOVDB_IO_INLINE uint64_t pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint64_t byte_offset)
{
    uint64_t ret = 0u;
    buf->read(buf->file, byte_offset, &ret, 8u);
    return ret;
}
PNANOVDB_IO_INLINE void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint64_t byte_offset, uint32_t value)
{
    buf->write(buf->file, byte_offset, &value, 4u);
}
PNANOVDB_IO_INLINE void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint64_t byte_offset, uint64_t value)
{
    buf->write(buf->file, byte_offset, &value, 8u);
}
#define pnanovdb_grid_type_t uint32_t
#define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) pnanovdb_grid_type_constants[grid_typeIn].nameIn

#endif

#ifndef PNANOVDB_C
#define PNANOVDB_C
#endif
#include <nanovdb/PNanoVDB.h>

typedef struct pnanovdb_io_fileheader_t
{
    pnanovdb_uint64_t magic;                 // 8 bytes,        0
    pnanovdb_uint32_t version;                 // 4 bytes,        8
    pnanovdb_uint32_t grid_count_and_codec; // 4 bytes,        12
}pnanovdb_io_fileheader_t;

#define PNANOVDB_IO_FILEHEADER_SIZE 16

typedef struct pnanovdb_io_filemetadata_t
{
    pnanovdb_uint64_t grid_size;            // 8 bytes,     0
    pnanovdb_uint64_t file_size;            // 8 bytes,     8
    pnanovdb_uint64_t name_key;                // 8 bytes,     16
    pnanovdb_uint64_t voxel_count;            // 8 bytes,     24
    pnanovdb_uint32_t grid_type;            // 4 bytes,     32
    pnanovdb_uint32_t grid_class;            // 4 bytes,     36
    double world_bbox[6u];                    // 48 bytes,     40
    pnanovdb_coord_t index_bbox_min;        // 12 bytes,    88
    pnanovdb_coord_t index_bbox_max;        // 12 bytes,    100
    double voxel_size[3u];                    // 24 bytes,    112
    pnanovdb_uint32_t name_size;            // 4 bytes,        136
    pnanovdb_uint32_t node_count_leaf;        // 4 bytes,        140
    pnanovdb_uint32_t node_count_lower;        // 4 bytes,        144
    pnanovdb_uint32_t node_count_upper;        // 4 bytes,        148
    pnanovdb_uint32_t node_count_root;        // 4 bytes,        152
    pnanovdb_uint32_t tile_count_lower;        // 4 bytes,        156
    pnanovdb_uint32_t tile_count_upper;        // 4 bytes,        160
    pnanovdb_uint32_t tile_count_root;        // 4 bytes,        164
    pnanovdb_uint32_t codec_and_padding;    // 4 bytes,        168
    pnanovdb_uint32_t version;                // 4 bytes,        172
}pnanovdb_io_filemetadata_t;

#define PNANOVDB_IO_FILEMETADATA_SIZE 176

#if defined(PNANOVDB_IO_BUF_CUSTOM)

PNANOVDB_IO_INLINE pnanovdb_grid_handle_t pnanovdb_io_fread_fileheader(
    pnanovdb_buf_t file,
    pnanovdb_io_fileheader_t* dst_header,
    pnanovdb_io_filemetadata_t* dst_metadatas,
    pnanovdb_uint64_t dst_metadata_count)
{
    pnanovdb_grid_t local_grid = {};
    file->read(file->file, 0u, &local_grid, PNANOVDB_GRID_SIZE);
    pnanovdb_bool_t is_raw_grid = PNANOVDB_FALSE;
    if (local_grid.magic == PNANOVDB_MAGIC_GRID ||
        local_grid.data2 == PNANOVDB_MAGIC_GRID)
    {
        is_raw_grid = PNANOVDB_TRUE;
    }
    if (is_raw_grid)
    {
        pnanovdb_grid_handle_t raw_grid = {pnanovdb_address_null()};
        return raw_grid;
    }

    pnanovdb_uint64_t offset = 0u;
    pnanovdb_io_fileheader_t local_header = {};
    file->read(file->file, offset, &local_header, PNANOVDB_IO_FILEHEADER_SIZE);
    if (dst_header)
    {
        *dst_header = local_header;
    }
    offset += PNANOVDB_IO_FILEHEADER_SIZE;
    pnanovdb_uint32_t grid_count = local_header.grid_count_and_codec & 0xFFFF;
    for (uint32_t idx = 0u; idx < grid_count; idx++)
    {
        pnanovdb_io_filemetadata_t local_metadata = {};
        file->read(file->file, offset, &local_metadata, PNANOVDB_IO_FILEMETADATA_SIZE);
        if (idx < dst_metadata_count)
        {
            dst_metadatas[idx] = local_metadata;
        }
        offset += PNANOVDB_IO_FILEMETADATA_SIZE;
        // skip past name
        offset += local_metadata.name_size;
    }
    pnanovdb_grid_handle_t first_grid = {pnanovdb_address_offset64(pnanovdb_address_null(), offset)};
    return first_grid;
}

PNANOVDB_IO_INLINE void pnanovdb_io_memcpy(
    pnanovdb_buf_t dst_buf, pnanovdb_address_t dst_addr,
    pnanovdb_buf_t src_buf, pnanovdb_address_t src_addr,
    pnanovdb_uint64_t num_bytes)
{
    pnanovdb_uint64_t num_words = (num_bytes >> 2u);
    for (pnanovdb_uint64_t word_idx = 0u; word_idx < num_words; word_idx++)
    {
        pnanovdb_uint32_t val = pnanovdb_read_uint32(src_buf, pnanovdb_address_offset_product(src_addr, word_idx, 4u));
        pnanovdb_write_uint32(dst_buf, pnanovdb_address_offset_product(dst_addr, word_idx, 4u), val);
    }
}

PNANOVDB_IO_INLINE pnanovdb_uint32_t pnanovdb_io_get_leaf_size(pnanovdb_uint32_t grid_type, pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    pnanovdb_uint32_t size = PNANOVDB_GRID_TYPE_GET(grid_type, leaf_size);
    if (grid_type == PNANOVDB_GRID_TYPE_FPN)
    {
        pnanovdb_uint32_t flags = pnanovdb_leaf_get_bbox_dif_and_flags(buf, leaf) >> 24u;
        pnanovdb_uint32_t value_log_bits = flags >> 5;
        pnanovdb_uint32_t value_bits = 1u << value_log_bits;
        pnanovdb_uint32_t table_bytes = value_bits << (9u - 3u);
        size += table_bytes;
    }
    return size;
}

PNANOVDB_IO_INLINE void pnanovdb_io_copy_subregion(
    pnanovdb_buf_t src_buf,
    pnanovdb_grid_handle_t src_grid,
    pnanovdb_buf_t dst_buf,
    pnanovdb_grid_handle_t dst_grid,
    pnanovdb_coord_t ijk_min,
    pnanovdb_coord_t ijk_max)
{
    // leaf precise, not voxel precise for now
    ijk_min.x &= ~7;
    ijk_min.y &= ~7;
    ijk_min.z &= ~7;
    ijk_max.x |= 7;
    ijk_max.y |= 7;
    ijk_max.z |= 7;

    pnanovdb_buf_t buf = src_buf;
    pnanovdb_grid_handle_t grid = src_grid;

    // copy grid
    pnanovdb_io_memcpy(dst_buf, dst_grid.address, buf, grid.address, PNANOVDB_GRID_SIZE);

    pnanovdb_grid_type_t grid_type = pnanovdb_grid_get_grid_type(buf, grid);

    // copy tree
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_tree_handle_t dst_tree = pnanovdb_grid_get_tree(dst_buf, dst_grid);
    pnanovdb_io_memcpy(dst_buf, dst_tree.address, buf, tree.address, PNANOVDB_TREE_SIZE);

    // copy root
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
    pnanovdb_root_handle_t dst_root = pnanovdb_tree_get_root(dst_buf, dst_tree);
    pnanovdb_io_memcpy(dst_buf, dst_root.address, buf, root.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_size));

    // set new root to clipped bounds
    pnanovdb_root_set_bbox_min(dst_buf, dst_root, &ijk_min);
    pnanovdb_root_set_bbox_max(dst_buf, dst_root, &ijk_max);

    // copy root tiles
    pnanovdb_uint32_t tile_count = pnanovdb_root_get_tile_count(buf, root);
    for (pnanovdb_uint32_t tile_idx = 0u; tile_idx < tile_count; tile_idx++)
    {
        pnanovdb_root_tile_handle_t root_tile = pnanovdb_root_get_tile(grid_type, root, tile_idx);
        pnanovdb_root_tile_handle_t dst_root_tile = pnanovdb_root_get_tile(grid_type, dst_root, tile_idx);
        pnanovdb_io_memcpy(dst_buf, dst_root_tile.address, buf, root_tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size));
    }

    // copy upper
    for (pnanovdb_uint32_t tile_idx = 0u; tile_idx < tile_count; tile_idx++)
    {
        pnanovdb_root_tile_handle_t root_tile = pnanovdb_root_get_tile(grid_type, root, tile_idx);
        pnanovdb_root_tile_handle_t dst_root_tile = pnanovdb_root_get_tile(grid_type, dst_root, tile_idx);
        if (pnanovdb_root_tile_get_child_mask(buf, root_tile))
        {
            pnanovdb_upper_handle_t upper = pnanovdb_root_get_child(grid_type, buf, root, root_tile);
            pnanovdb_upper_handle_t dst_upper = pnanovdb_root_get_child(grid_type, dst_buf, dst_root, dst_root_tile);
            pnanovdb_io_memcpy(dst_buf, dst_upper.address, buf, upper.address, PNANOVDB_GRID_TYPE_GET(grid_type, upper_size));
        }
    }

    // copy lower
    for (pnanovdb_uint32_t tile_idx = 0u; tile_idx < tile_count; tile_idx++)
    {
        pnanovdb_root_tile_handle_t root_tile = pnanovdb_root_get_tile(grid_type, root, tile_idx);
        pnanovdb_root_tile_handle_t dst_root_tile = pnanovdb_root_get_tile(grid_type, dst_root, tile_idx);
        if (pnanovdb_root_tile_get_child_mask(buf, root_tile))
        {
            pnanovdb_upper_handle_t upper = pnanovdb_root_get_child(grid_type, buf, root, root_tile);
            pnanovdb_upper_handle_t dst_upper = pnanovdb_root_get_child(grid_type, dst_buf, dst_root, dst_root_tile);
            for (pnanovdb_uint32_t upper_n = 0u; upper_n < PNANOVDB_UPPER_TABLE_COUNT; upper_n++)
            {
                if (pnanovdb_upper_get_child_mask(buf, upper, upper_n))
                {
                    pnanovdb_lower_handle_t lower = pnanovdb_upper_get_child(grid_type, buf, upper, upper_n);
                    pnanovdb_lower_handle_t dst_lower = pnanovdb_upper_get_child(grid_type, dst_buf, dst_upper, upper_n);
                    pnanovdb_io_memcpy(dst_buf, dst_lower.address, buf, lower.address, PNANOVDB_GRID_TYPE_GET(grid_type, lower_size));
                }
            }
        }
    }

    // copy leaf
    pnanovdb_uint32_t dst_node_count_leaf = 0u;
    pnanovdb_leaf_handle_t dst_current_leaf = {
        pnanovdb_address_offset(dst_tree.address, pnanovdb_tree_get_node_offset_leaf(dst_buf, dst_tree))};
    for (pnanovdb_uint32_t tile_idx = 0u; tile_idx < tile_count; tile_idx++)
    {
        pnanovdb_root_tile_handle_t root_tile = pnanovdb_root_get_tile(grid_type, root, tile_idx);
        pnanovdb_root_tile_handle_t dst_root_tile = pnanovdb_root_get_tile(grid_type, dst_root, tile_idx);
        if (pnanovdb_root_tile_get_child_mask(buf, root_tile))
        {
            pnanovdb_upper_handle_t upper = pnanovdb_root_get_child(grid_type, buf, root, root_tile);
            pnanovdb_upper_handle_t dst_upper = pnanovdb_root_get_child(grid_type, dst_buf, dst_root, dst_root_tile);
            for (pnanovdb_uint32_t upper_n = 0u; upper_n < PNANOVDB_UPPER_TABLE_COUNT; upper_n++)
            {
                if (pnanovdb_upper_get_child_mask(buf, upper, upper_n))
                {
                    pnanovdb_lower_handle_t lower = pnanovdb_upper_get_child(grid_type, buf, upper, upper_n);
                    pnanovdb_lower_handle_t dst_lower = pnanovdb_upper_get_child(grid_type, dst_buf, dst_upper, upper_n);
                    for (pnanovdb_uint32_t lower_n = 0u; lower_n < PNANOVDB_LOWER_TABLE_COUNT; lower_n++)
                    {
                        if (pnanovdb_lower_get_child_mask(buf, lower, lower_n))
                        {
                            pnanovdb_leaf_handle_t leaf = pnanovdb_lower_get_child(grid_type, buf, lower, lower_n);

                            pnanovdb_coord_t leaf_ijk_min = pnanovdb_leaf_get_bbox_min(buf, leaf);
                            pnanovdb_uint32_t dif_and_flags = pnanovdb_leaf_get_bbox_dif_and_flags(buf, leaf);
                            pnanovdb_coord_t leaf_ijk_max = {
                                pnanovdb_uint32_as_int32(dif_and_flags & 255u) + leaf_ijk_min.x,
                                pnanovdb_uint32_as_int32((dif_and_flags >> 8u) & 255u) + leaf_ijk_min.y,
                                pnanovdb_uint32_as_int32((dif_and_flags >> 16u) & 255u) + leaf_ijk_min.z
                            };

                            // leaf precise, not voxel precise for now
                            leaf_ijk_min.x &= ~7;
                            leaf_ijk_min.y &= ~7;
                            leaf_ijk_min.z &= ~7;
                            leaf_ijk_max.x |= 7;
                            leaf_ijk_max.y |= 7;
                            leaf_ijk_max.z |= 7;

                            pnanovdb_bool_t pred = !(
                                leaf_ijk_max.x < ijk_min.x || leaf_ijk_min.x > ijk_max.x ||
                                leaf_ijk_max.y < ijk_min.y || leaf_ijk_min.y > ijk_max.y ||
                                leaf_ijk_max.z < ijk_min.z || leaf_ijk_min.z > ijk_max.z );
                            if (pred)
                            {
                                pnanovdb_uint32_t leaf_size = pnanovdb_io_get_leaf_size(grid_type, buf, leaf);
                                pnanovdb_io_memcpy(dst_buf, dst_current_leaf.address, buf, leaf.address, leaf_size);

                                pnanovdb_lower_set_child(grid_type, dst_buf, dst_lower, lower_n, dst_current_leaf);

                                // voxel precise
                                #if 0
                                for (pnanovdb_uint32_t leaf_n = 0u; leaf_n < PNANOVDB_LEAF_TABLE_COUNT; leaf_n++)
                                {
                                    pnanovdb_coord_t voxel_ijk = {
                                        (leaf_ijk_min.x & ~7) + (pnanovdb_uint32_as_int32(leaf_n >> 6u) & 7),
                                        (leaf_ijk_min.y & ~7) + (pnanovdb_uint32_as_int32(leaf_n >> 3u) & 7),
                                        (leaf_ijk_min.z & ~7) + (pnanovdb_uint32_as_int32(leaf_n >> 0u) & 7)
                                    };

                                    pnanovdb_bool_t voxel_pred = !(
                                        voxel_ijk.x < ijk_min.x || voxel_ijk.x > ijk_max.x ||
                                        voxel_ijk.y < ijk_min.y || voxel_ijk.y > ijk_max.y ||
                                        voxel_ijk.z < ijk_min.z || voxel_ijk.z > ijk_max.z );
                                    if (!voxel_pred)
                                    {
                                        pnanovdb_leaf_set_value_mask(dst_buf, dst_current_leaf, leaf_n, PNANOVDB_FALSE);
                                        pnanovdb_address_t val_addr =
                                            pnanovdb_leaf_get_table_address(grid_type, dst_buf, dst_current_leaf, leaf_n);
                                        pnanovdb_address_t background_addr = pnanovdb_root_get_background_address(grid_type, buf, root);

                                        pnanovdb_io_memcpy(dst_buf, val_addr, buf, background_addr, (PNANOVDB_GRID_TYPE_GET(grid_type, value_stride_bits) >> 3u));
                                    }
                                }
                                #endif

                                dst_current_leaf.address = pnanovdb_address_offset(dst_current_leaf.address, leaf_size);
                                dst_node_count_leaf++;
                            }
                            else
                            {
                                pnanovdb_address_t background_addr = pnanovdb_root_get_background_address(grid_type, buf, root);
                                pnanovdb_uint64_t background_index = pnanovdb_read_uint64(buf, background_addr);
                                pnanovdb_lower_set_table_child(grid_type, dst_buf, dst_lower, lower_n, pnanovdb_uint64_as_int64(background_index));
                                pnanovdb_lower_set_child_mask(dst_buf, dst_lower, lower_n, PNANOVDB_FALSE);
                                pnanovdb_lower_set_value_mask(dst_buf, dst_lower, lower_n, PNANOVDB_FALSE);
                            }
                        }
                    }
                }
            }
        }
    }

    //pnanovdb_grid_set_blind_metadata_count(dst_buf, dst_grid, 0u);

    // need to update tree for new reduced leaf range
    pnanovdb_tree_set_node_count_leaf(dst_buf, dst_tree, dst_node_count_leaf);

    pnanovdb_uint32_t blindmetadata_count = pnanovdb_grid_get_blind_metadata_count(buf, grid);

    pnanovdb_uint64_t dst_blind_metadata_offset =
        dst_current_leaf.address.byte_offset -
        dst_grid.address.byte_offset;
    pnanovdb_grid_set_blind_metadata_offset(dst_buf, dst_grid, dst_blind_metadata_offset);

    // copy blindmetadata headers
    for (pnanovdb_uint32_t metadata_idx = 0u; metadata_idx < blindmetadata_count; metadata_idx++)
    {
        pnanovdb_gridblindmetadata_handle_t metadata = pnanovdb_grid_get_gridblindmetadata(buf, grid, metadata_idx);
        pnanovdb_gridblindmetadata_handle_t dst_metadata = pnanovdb_grid_get_gridblindmetadata(dst_buf, dst_grid, metadata_idx);

        pnanovdb_io_memcpy(dst_buf, dst_metadata.address, buf, metadata.address, PNANOVDB_GRIDBLINDMETADATA_SIZE);
    }

#if 0
    // copy blindmetadata
    for (pnanovdb_uint32_t metadata_idx = 0u; metadata_idx < blindmetadata_count; metadata_idx++)
    {
        pnanovdb_gridblindmetadata_handle_t metadata = pnanovdb_grid_get_gridblindmetadata(buf, grid, metadata_idx);
        pnanovdb_gridblindmetadata_handle_t dst_metadata = pnanovdb_grid_get_gridblindmetadata(dst_buf, dst_grid, metadata_idx);

        pnanovdb_int64_t byte_offset = pnanovdb_gridblindmetadata_get_data_offset(buf, metadata);
        pnanovdb_address_t address = pnanovdb_address_offset64(metadata.address, pnanovdb_int64_as_uint64(byte_offset));

        pnanovdb_int64_t dst_byte_offset = pnanovdb_gridblindmetadata_get_data_offset(dst_buf, dst_metadata);
        pnanovdb_address_t dst_address = pnanovdb_address_offset64(dst_metadata.address, pnanovdb_int64_as_uint64(dst_byte_offset));

        pnanovdb_uint32_t value_size = pnanovdb_gridblindmetadata_get_value_size(buf, metadata);
        pnanovdb_uint64_t value_count = pnanovdb_gridblindmetadata_get_value_count(buf, metadata);

        pnanovdb_io_memcpy(dst_buf, dst_address, buf, address, value_count * value_size);
    }
#else
    for (pnanovdb_uint32_t metadata_idx = 0u; metadata_idx < blindmetadata_count; metadata_idx++)
    {
        pnanovdb_gridblindmetadata_handle_t metadata = pnanovdb_grid_get_gridblindmetadata(buf, grid, metadata_idx);
        pnanovdb_gridblindmetadata_handle_t dst_metadata = pnanovdb_grid_get_gridblindmetadata(dst_buf, dst_grid, metadata_idx);

        pnanovdb_int64_t byte_offset = pnanovdb_gridblindmetadata_get_data_offset(buf, metadata);
        pnanovdb_address_t address = pnanovdb_address_offset64(metadata.address, pnanovdb_int64_as_uint64(byte_offset));

        pnanovdb_int64_t dst_byte_offset = pnanovdb_gridblindmetadata_get_data_offset(dst_buf, dst_metadata);
        pnanovdb_address_t dst_address = pnanovdb_address_offset64(dst_metadata.address, pnanovdb_int64_as_uint64(dst_byte_offset));

        pnanovdb_uint32_t value_size = pnanovdb_gridblindmetadata_get_value_size(buf, metadata);
        pnanovdb_uint64_t value_count = pnanovdb_gridblindmetadata_get_value_count(buf, metadata);

        pnanovdb_uint64_t dst_value_index = 0llu;
        pnanovdb_uint64_t value_index = 0llu;

        pnanovdb_address_t index_addr = pnanovdb_root_get_background_address(grid_type, buf, root);
        value_index = pnanovdb_read_uint64(buf, index_addr);

        pnanovdb_address_t dst_index_addr = pnanovdb_root_get_background_address(grid_type, dst_buf, dst_root);
        pnanovdb_write_uint64(dst_buf, dst_index_addr, dst_value_index);

        pnanovdb_uint64_t last_root_tile_index = value_index;
        pnanovdb_uint64_t last_upper_tile_index = value_index;
        pnanovdb_uint64_t last_lower_tile_index = value_index;
        pnanovdb_uint64_t last_root_tile_dst_index = dst_value_index;
        pnanovdb_uint64_t last_upper_tile_dst_index = dst_value_index;
        pnanovdb_uint64_t last_lower_tile_dst_index = dst_value_index;

        pnanovdb_io_memcpy(
            dst_buf, pnanovdb_address_offset64_product(dst_address, dst_value_index, value_size),
            buf, pnanovdb_address_offset64_product(address, value_index, value_size), value_size);
        dst_value_index++;

        // walk subregion, copy referenced blindmetadata
        pnanovdb_uint32_t dst_tile_count = pnanovdb_root_get_tile_count(dst_buf, dst_root);
        for (pnanovdb_uint32_t tile_idx = 0u; tile_idx < dst_tile_count; tile_idx++)
        {
            pnanovdb_root_tile_handle_t dst_root_tile = pnanovdb_root_get_tile(grid_type, dst_root, tile_idx);
            pnanovdb_root_tile_handle_t root_tile = pnanovdb_root_get_tile(grid_type, root, tile_idx);

            if (pnanovdb_root_tile_get_child_mask(dst_buf, dst_root_tile))
            {
                pnanovdb_upper_handle_t dst_upper = pnanovdb_root_get_child(grid_type, dst_buf, dst_root, dst_root_tile);
                pnanovdb_upper_handle_t upper = pnanovdb_root_get_child(grid_type, buf, root, root_tile);
                for (pnanovdb_uint32_t upper_n = 0u; upper_n < PNANOVDB_UPPER_TABLE_COUNT; upper_n++)
                {
                    if (pnanovdb_upper_get_child_mask(dst_buf, dst_upper, upper_n))
                    {
                        pnanovdb_lower_handle_t dst_lower = pnanovdb_upper_get_child(grid_type, dst_buf, dst_upper, upper_n);
                        pnanovdb_lower_handle_t lower = pnanovdb_upper_get_child(grid_type, buf, upper, upper_n);
                        for (pnanovdb_uint32_t lower_n = 0u; lower_n < PNANOVDB_LOWER_TABLE_COUNT; lower_n++)
                        {
                            if (pnanovdb_lower_get_child_mask(dst_buf, dst_lower, lower_n))
                            {
                                pnanovdb_leaf_handle_t dst_leaf = pnanovdb_lower_get_child(grid_type, dst_buf, dst_lower, lower_n);
                                pnanovdb_leaf_handle_t leaf = pnanovdb_lower_get_child(grid_type, buf, lower, lower_n);

                                pnanovdb_uint32_t leaf_value_count = pnanovdb_leaf_onindex_get_value_count(dst_buf, dst_leaf);

                                dst_index_addr = pnanovdb_leaf_get_table_address(grid_type, dst_buf, dst_leaf, 0u);
                                pnanovdb_write_uint64(dst_buf, dst_index_addr, dst_value_index);

                                pnanovdb_address_t index_addr = pnanovdb_leaf_get_table_address(grid_type, buf, leaf, 0u);
                                value_index = pnanovdb_read_uint64(buf, index_addr);

                                pnanovdb_io_memcpy(
                                    dst_buf, pnanovdb_address_offset64_product(dst_address, dst_value_index, value_size),
                                    buf, pnanovdb_address_offset64_product(address, value_index, value_size), value_size * leaf_value_count);
                                dst_value_index+=leaf_value_count;
                            }
                            else if(pnanovdb_lower_get_child_mask(buf, lower, lower_n))
                            {
                                // this covers leaves disabled by pred above
                                pnanovdb_address_t background_addr = pnanovdb_root_get_background_address(grid_type, dst_buf, dst_root);
                                pnanovdb_uint64_t background_index = pnanovdb_read_uint64(buf, background_addr);

                                dst_index_addr = pnanovdb_lower_get_table_address(grid_type, dst_buf, dst_lower, lower_n);
                                pnanovdb_write_uint64(dst_buf, dst_index_addr, background_index);
                            }
                            else
                            {
                                pnanovdb_address_t index_addr = pnanovdb_lower_get_table_address(grid_type, buf, lower, lower_n);
                                value_index = pnanovdb_read_uint64(buf, index_addr);
                                if (last_lower_tile_index != value_index)
                                {
                                    last_lower_tile_index = value_index;

                                    dst_index_addr = pnanovdb_lower_get_table_address(grid_type, dst_buf, dst_lower, lower_n);
                                    pnanovdb_write_uint64(dst_buf, dst_index_addr, dst_value_index);
                                    last_lower_tile_dst_index = dst_value_index;

                                    pnanovdb_io_memcpy(
                                        dst_buf, pnanovdb_address_offset64_product(dst_address, dst_value_index, value_size),
                                        buf, pnanovdb_address_offset64_product(address, value_index, value_size), value_size);
                                    dst_value_index++;
                                }
                                else
                                {
                                    dst_index_addr = pnanovdb_lower_get_table_address(grid_type, dst_buf, dst_lower, lower_n);
                                    pnanovdb_write_uint64(dst_buf, dst_index_addr, last_lower_tile_dst_index);
                                }
                            }
                        }
                    }
                    else
                    {
                        pnanovdb_address_t index_addr = pnanovdb_upper_get_table_address(grid_type, buf, upper, upper_n);
                        value_index = pnanovdb_read_uint64(buf, index_addr);
                        if (last_upper_tile_index != value_index)
                        {
                            last_upper_tile_index = value_index;

                            dst_index_addr = pnanovdb_upper_get_table_address(grid_type, dst_buf, dst_upper, upper_n);
                            pnanovdb_write_uint64(dst_buf, dst_index_addr, dst_value_index);
                            last_upper_tile_dst_index = dst_value_index;

                            pnanovdb_io_memcpy(
                                dst_buf, pnanovdb_address_offset64_product(dst_address, dst_value_index, value_size),
                                buf, pnanovdb_address_offset64_product(address, value_index, value_size), value_size);
                            dst_value_index++;
                        }
                        else
                        {
                            dst_index_addr = pnanovdb_upper_get_table_address(grid_type, dst_buf, dst_upper, upper_n);
                            pnanovdb_write_uint64(dst_buf, dst_index_addr, last_upper_tile_dst_index);
                        }
                    }
                }
            }
            else
            {
                pnanovdb_address_t index_addr = pnanovdb_root_tile_get_value_address(grid_type, buf, root_tile);
                value_index = pnanovdb_read_uint64(buf, index_addr);
                if (last_root_tile_index != value_index)
                {
                    last_root_tile_index = value_index;

                    dst_index_addr = pnanovdb_root_tile_get_value_address(grid_type, dst_buf, dst_root_tile);
                    pnanovdb_write_uint64(dst_buf, dst_index_addr, dst_value_index);
                    last_root_tile_dst_index = dst_value_index;

                    pnanovdb_io_memcpy(
                        dst_buf, pnanovdb_address_offset64_product(dst_address, dst_value_index, value_size),
                        buf, pnanovdb_address_offset64_product(address, value_index, value_size), value_size);
                    dst_value_index++;
                }
                else
                {
                    dst_index_addr = pnanovdb_root_tile_get_value_address(grid_type, dst_buf, dst_root_tile);
                    pnanovdb_write_uint64(dst_buf, dst_index_addr, last_root_tile_dst_index);
                }
            }
        }

        pnanovdb_gridblindmetadata_set_value_count(dst_buf, dst_metadata, dst_value_index);

        pnanovdb_int64_t metadata_size_diff = pnanovdb_uint64_as_int64(dst_value_index - value_count) * value_size;
        // compact later metadata ranges against new count
        for (pnanovdb_uint32_t local_metadata_idx = metadata_idx + 1u; local_metadata_idx < blindmetadata_count; local_metadata_idx++)
        {
            pnanovdb_gridblindmetadata_handle_t local_dst_metadata = pnanovdb_grid_get_gridblindmetadata(dst_buf, dst_grid, metadata_idx);

            pnanovdb_int64_t local_dst_byte_offset = pnanovdb_gridblindmetadata_get_data_offset(dst_buf, local_dst_metadata);
            local_dst_byte_offset += metadata_size_diff;
            pnanovdb_gridblindmetadata_set_data_offset(dst_buf, local_dst_metadata, local_dst_byte_offset);
        }
    }

    // compute final grid size based on maximum of final leaf and final metadata
    pnanovdb_uint64_t grid_size =
        dst_current_leaf.address.byte_offset -
        dst_grid.address.byte_offset;
    for (pnanovdb_uint32_t metadata_idx = 0u; metadata_idx < blindmetadata_count; metadata_idx++)
    {
        pnanovdb_gridblindmetadata_handle_t dst_metadata = pnanovdb_grid_get_gridblindmetadata(dst_buf, dst_grid, metadata_idx);

        pnanovdb_int64_t dst_byte_offset = pnanovdb_gridblindmetadata_get_data_offset(dst_buf, dst_metadata);
        pnanovdb_address_t dst_address = pnanovdb_address_offset64(dst_metadata.address, pnanovdb_int64_as_uint64(dst_byte_offset));

        pnanovdb_uint32_t value_size = pnanovdb_gridblindmetadata_get_value_size(dst_buf, dst_metadata);
        pnanovdb_uint64_t value_count = pnanovdb_gridblindmetadata_get_value_count(dst_buf, dst_metadata);

        pnanovdb_uint64_t cmp_grid_size =
            dst_address.byte_offset + value_count * value_size -
            dst_grid.address.byte_offset;
        if (grid_size < cmp_grid_size)
        {
            grid_size = cmp_grid_size;
        }
    }
    pnanovdb_grid_set_grid_size(dst_buf, dst_grid, grid_size);
#endif
}

PNANOVDB_IO_INLINE void pnanovdb_io_print_slice(pnanovdb_io_buf_t* src_nanovdb, FILE* dst_bmp)
{
    pnanovdb_grid_handle_t grid = pnanovdb_io_fread_fileheader(src_nanovdb, 0, 0, 0u);

    pnanovdb_buf_t buf = src_nanovdb;

    // set grid at post header print offset
    printf("pnanovdb_io_print_slice grid(%lld)\n", (long long int)grid.address.byte_offset);

    printf("pnanovdb_io_print_slice magic(%llx)\n",
        (unsigned long long int)pnanovdb_grid_get_magic(buf, grid));
    printf("pnanovdb_io_print_slice version(%x)\n", pnanovdb_grid_get_version(buf, grid));

    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    printf("pnanovdb_io_print_slice tree(%lld)\n", (long long int)tree.address.byte_offset);

    printf("pnanovdb_io_print_slice node_count_leaf(%d)\n", pnanovdb_tree_get_node_count_leaf(buf, tree));
    printf("pnanovdb_io_print_slice node_count_lower(%d)\n", pnanovdb_tree_get_node_count_lower(buf, tree));
    printf("pnanovdb_io_print_slice node_count_upper(%d)\n", pnanovdb_tree_get_node_count_upper(buf, tree));

    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
    pnanovdb_grid_type_t grid_type = pnanovdb_grid_get_grid_type(buf, grid);

    pnanovdb_coord_t bbox_min = pnanovdb_root_get_bbox_min(buf, root);
    pnanovdb_coord_t bbox_max = pnanovdb_root_get_bbox_max(buf, root);

    printf("pnanovdb_io_print_slice range([(%d, %d, %d),(%d, %d, %d)])\n",
        bbox_min.x, bbox_min.y, bbox_min.z,
        bbox_max.x, bbox_max.y, bbox_max.z);

    pnanovdb_int32_t image_width = bbox_max.x - bbox_min.x;
    pnanovdb_int32_t image_height = bbox_max.y - bbox_min.y;
    pnanovdb_int32_t k = (bbox_max.z + bbox_min.z) / 2;

    char headerField0 = 'B';
    char headerField1 = 'M';
    uint32_t size = 54 + image_width * image_height * 4u;
    uint16_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t offset = 54;
    uint32_t headerSize = 40;
    uint32_t width = image_width;
    uint32_t height = image_height;
    uint16_t colorPlanes = 1;
    uint16_t bitsPerPixel = 32;
    uint32_t compressionMethod = 0;
    uint32_t imageSize = image_width * image_height * 4u;
    uint32_t hRes = 2000;
    uint32_t vRes = 2000;
    uint32_t numColors = 0;
    uint32_t numImportantColors = 0;

    fwrite(&headerField0, 1, 1, dst_bmp);
    fwrite(&headerField1, 1, 1, dst_bmp);
    fwrite(&size, 4, 1, dst_bmp);
    fwrite(&reserved1, 2, 1, dst_bmp);
    fwrite(&reserved2, 2, 1, dst_bmp);
    fwrite(&offset, 4, 1, dst_bmp);
    fwrite(&headerSize, 4, 1, dst_bmp);
    fwrite(&width, 4, 1, dst_bmp);
    fwrite(&height, 4, 1, dst_bmp);
    fwrite(&colorPlanes, 2, 1, dst_bmp);
    fwrite(&bitsPerPixel, 2, 1, dst_bmp);
    fwrite(&compressionMethod, 4, 1, dst_bmp);
    fwrite(&imageSize, 4, 1, dst_bmp);
    fwrite(&hRes, 4, 1, dst_bmp);
    fwrite(&vRes, 4, 1, dst_bmp);
    fwrite(&numColors, 4, 1, dst_bmp);
    fwrite(&numImportantColors, 4, 1, dst_bmp);

    for (pnanovdb_int32_t j = image_height - 1; j >= 0; j--)
    {
        for (pnanovdb_int32_t i = 0; i < image_width; i++)
        {
            pnanovdb_coord_t ijk = {
                i + bbox_min.x,
                j + bbox_min.y,
                k
            };

            pnanovdb_address_t addr = pnanovdb_root_get_value_address(grid_type, buf, root, &ijk);

            float density_value = 1.f;
            float color_value = pnanovdb_read_float(buf, addr);

            pnanovdb_uint32_t raw_value = pnanovdb_uint32_t(255.f * pnanovdb_max(0.f, pnanovdb_min(1.f, color_value))) |
                (pnanovdb_uint32_t(255.f * pnanovdb_max(0.f, pnanovdb_min(1.f, color_value))) << 8u) |
                (pnanovdb_uint32_t(255.f * pnanovdb_max(0.f, pnanovdb_min(1.f, color_value))) << 16u) |
                (pnanovdb_uint32_t(255.f * pnanovdb_max(0.f, pnanovdb_min(1.f, density_value))) << 24u);

            fwrite(&raw_value, 4u, 1u, dst_bmp);
        }
    }
}

#endif

#endif // end of NANOVDB_PUTILS_IO_H_HAS_BEEN_INCLUDED
