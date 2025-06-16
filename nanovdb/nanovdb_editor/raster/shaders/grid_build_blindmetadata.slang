// scatter_new_children.slang
#define PNANOVDB_HLSL
#define PNANOVDB_ADDRESS_64
#define PNANOVDB_BUF_HLSL_64
#define PNANOVDB_BUF_HLSL_RW
#include "PNanoVDB.h"
#include "PNanoVDB2.h"

struct constants_t
{
    uint workgroup_count;
    uint max_node_count;
    uint buf_word_count;
    uint point_count;
    uint empty_grid_word_count;
    uint active_node_type;
    uint child_node_type;
};

ConstantBuffer<constants_t> constants;

StructuredBuffer<uint> new_child_counts;
StructuredBuffer<uint> scan_new_child_counts;

RWStructuredBuffer<uint64_t> buf;

#include <workgroup_scan.slang>

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint node_idx = group_idx.x * 256u + thread_idx.x;

    if (node_idx == 0u)
    {
        pnanovdb_grid_handle_t grid = {};
        pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
        pnanovdb_node2_handle_t root = { pnanovdb_tree_get_root(buf, tree).address.byte_offset >> 3u };

        pnanovdb_uint32_t node_count = 1u +
                                       pnanovdb_tree_get_node_count_upper(buf, tree) +
                                       pnanovdb_tree_get_node_count_lower(buf, tree) +
                                       pnanovdb_tree_get_node_count_leaf(buf, tree);

        if (constants.active_node_type == constants.child_node_type)
        {
            uint value_count = scan_new_child_counts[node_count - 1u];

            pnanovdb_address_t end_addr = tree.address;
            end_addr = pnanovdb_address_offset64(end_addr,
                pnanovdb_tree_get_node_offset_root(buf, tree));
            end_addr = pnanovdb_address_offset64_product(end_addr,
                1u, pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_ROOT]);
            end_addr = pnanovdb_address_offset64_product(end_addr,
                pnanovdb_tree_get_node_count_upper(buf, tree), pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_UPPER]);
            end_addr = pnanovdb_address_offset64_product(end_addr,
                pnanovdb_tree_get_node_count_lower(buf, tree), pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_LOWER]);
            end_addr = pnanovdb_address_offset64_product(end_addr,
                pnanovdb_tree_get_node_count_leaf(buf, tree), pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_LEAF]);

            // place metadata after node segment
            pnanovdb_gridblindmetadata_handle_t metadata_bboxes = { end_addr };
            end_addr = pnanovdb_address_offset(end_addr, PNANOVDB_GRIDBLINDMETADATA_SIZE);
            pnanovdb_gridblindmetadata_handle_t metadata_values = { end_addr };
            end_addr = pnanovdb_address_offset(end_addr, PNANOVDB_GRIDBLINDMETADATA_SIZE);

            pnanovdb_address_t bboxes = end_addr;
            end_addr = pnanovdb_address_offset(end_addr, 32u);
            pnanovdb_address_t values = end_addr;
            end_addr = pnanovdb_address_offset_product(end_addr, 4u, value_count);

            pnanovdb_gridblindmetadata_set_data_offset(buf, metadata_bboxes, pnanovdb_address_diff(bboxes, metadata_bboxes.address));
            pnanovdb_gridblindmetadata_set_value_count(buf, metadata_bboxes, pnanovdb_uint32_as_uint64_low(2u));
            pnanovdb_gridblindmetadata_set_value_size(buf, metadata_bboxes, 12u);
            pnanovdb_gridblindmetadata_set_semantic(buf, metadata_bboxes, 0u);
            pnanovdb_gridblindmetadata_set_data_class(buf, metadata_bboxes, 0u);
            pnanovdb_gridblindmetadata_set_data_type(buf, metadata_bboxes, PNANOVDB_GRID_TYPE_VEC3F);
            pnanovdb_gridblindmetadata_set_name(buf, metadata_bboxes, 0u, 0u);

            pnanovdb_gridblindmetadata_set_data_offset(buf, metadata_values, pnanovdb_address_diff(values, metadata_values.address));
            pnanovdb_gridblindmetadata_set_value_count(buf, metadata_values, pnanovdb_uint32_as_uint64_low(value_count));
            pnanovdb_gridblindmetadata_set_value_size(buf, metadata_values, 4u);
            pnanovdb_gridblindmetadata_set_semantic(buf, metadata_values, 0u);
            pnanovdb_gridblindmetadata_set_data_class(buf, metadata_values, 0u);
            pnanovdb_gridblindmetadata_set_data_type(buf, metadata_values, PNANOVDB_GRID_TYPE_RGBA8);
            pnanovdb_gridblindmetadata_set_name(buf, metadata_values, 0u, 0u);

            // link headers to grid
            pnanovdb_grid_set_first_gridblindmetadata(buf, grid, metadata_bboxes);
            pnanovdb_grid_set_blind_metadata_count(buf, grid, 2u);

            // set grid size
            pnanovdb_grid_set_grid_size(buf, grid, pnanovdb_address_diff(end_addr, grid.address));
        }
    }

    // finalize keys
    if (group_idx.x == 0u)
    {
        const pnanovdb_uint32_t node_type = PNANOVDB_NODE2_TYPE_ROOT;

        pnanovdb_grid_handle_t grid = {};
        pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
        pnanovdb_node2_handle_t root = { pnanovdb_tree_get_root(buf, tree).address.byte_offset >> 3u };

        // set key active for active value or child
        for (uint root_n = thread_idx.x; root_n < 32768u; root_n += 256u)
        {
            bool is_child = pnanovdb_node2_get_child_mask_bit(buf, root, node_type, root_n);
            bool is_value = pnanovdb_node2_get_value_mask_bit(buf, root, node_type, root_n);
            pnanovdb_uint64_t key = pnanovdb_node2_end_key;
            if (is_child || is_value)
            {
                pnanovdb_uint64_t set_mask = 1llu << (root_n & 63u);
                InterlockedOr(buf[root.idx64 + pnanovdb_node2_off_key_mask[node_type] + (root_n >> 6u)], set_mask);
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // update key mask prefix sum
        {
            pnanovdb_node2_handle_t node = root;
            // pass to compute total values
            pnanovdb_uint32_t fanout = pnanovdb_node2_fanout_1d[node_type];
            pnanovdb_uint32_t mask_word_count = fanout >> 6u;
            uint word_idx = 4u * thread_idx.x;
            uint4 mask_count = uint4(0u, 0u, 0u, 0u);
            if (word_idx < mask_word_count)
            {
                mask_count.x = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_key_mask[node_type] + word_idx + 0u));
                mask_count.y = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_key_mask[node_type] + word_idx + 1u));
                mask_count.z = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_key_mask[node_type] + word_idx + 2u));
                mask_count.w = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_key_mask[node_type] + word_idx + 3u));
            }
            uint4 scan_mask_count = uint4(0u, 0u, 0u, 0u);
            uint scan_total = 0u;
            workgroup_scan(thread_idx.x, mask_count, scan_mask_count, scan_total);
            if (word_idx < mask_word_count)
            {
                uint word_idx_h = (word_idx + 4) & (mask_word_count - 1u);

                pnanovdb_uint64_t prefix_sum_l = pnanovdb_uint32_as_uint64(
                    (scan_mask_count.x << 16u), scan_mask_count.y | (scan_mask_count.z << 16u));
                pnanovdb_uint64_t prefix_sum_h = pnanovdb_uint32_as_uint64(scan_mask_count.w, 0u);

                InterlockedAnd(buf[node.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[node_type] + (word_idx >> 2u)], 0xFFFFllu);
                InterlockedOr(buf[node.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[node_type] + (word_idx >> 2u)], prefix_sum_l);
                InterlockedAnd(buf[node.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[node_type] + (word_idx_h >> 2u)], ~0xFFFFllu);
                InterlockedOr(buf[node.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[node_type] + (word_idx_h >> 2u)], prefix_sum_h);
            }
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint root_n = thread_idx.x; root_n < 32768u; root_n += 256u)
        {
            pnanovdb_bool_t key_active;
            pnanovdb_uint32_t key_idx = pnanovdb_node2_mask_n_to_idx(buf, root.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[node_type], node_type, root_n, PNANOVDB_REF(key_active));
            if (key_active)
            {
                pnanovdb_coord_t ijk;
                ijk.x = ((root_n >> 10u) & 0x1F) << 12u;
                ijk.y = ((root_n >> 5u) & 0x1F) << 12u;
                ijk.z = (root_n & 0x1F) << 12u;
                pnanovdb_int32_t range_half = (1u << (5u + 12u)) >> 1u;
                ijk.x = ijk.x >= range_half ? ijk.x - range_half - range_half : ijk.x;
                ijk.y = ijk.y >= range_half ? ijk.y - range_half - range_half : ijk.y;
                ijk.z = ijk.z >= range_half ? ijk.z - range_half - range_half : ijk.z;
                pnanovdb_uint64_t key =
                    pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.z) >> 12) |
                    (pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.y) >> 12) << 21) |
                    (pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.x) >> 12) << 42);

                pnanovdb_uint32_t child_count = pnanovdb_uint32_t(pnanovdb_node2_read(buf, root.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type])) & 0xFFFF;
                pnanovdb_node2_write(buf, root.idx64 + pnanovdb_node2_off_children[node_type] + child_count + key_idx, key);
            }
        }
    }
}
