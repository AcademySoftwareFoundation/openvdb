// count_new_children.slang
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

StructuredBuffer<uint> node_addresses;
StructuredBuffer<uint> node_types;

RWStructuredBuffer<uint64_t> buf;
RWStructuredBuffer<uint> new_child_counts;

#include <workgroup_scan.slang>

[require(spirv_1_3)]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint node_begin_idx = group_idx.x;

    pnanovdb_grid_handle_t grid = {};
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);

    pnanovdb_uint32_t node_count = 1u +
                                   pnanovdb_tree_get_node_count_upper(buf, tree) +
                                   pnanovdb_tree_get_node_count_lower(buf, tree) +
                                   pnanovdb_tree_get_node_count_leaf(buf, tree);
    if (node_count >= constants.max_node_count)
    {
        node_count = constants.max_node_count;
    }
    for (pnanovdb_uint32_t node_idx = node_begin_idx; node_idx < node_count; node_idx+=constants.workgroup_count)
    {
        pnanovdb_node2_handle_t node = { node_addresses[node_idx] >> 3u };
        pnanovdb_uint32_t node_type = node_types[node_idx];

        pnanovdb_uint32_t new_child_count = 0u;
        if (constants.active_node_type == constants.child_node_type)
        {
            // pass to compute total values
            pnanovdb_uint32_t fanout = pnanovdb_node2_fanout_1d[node_type];
            pnanovdb_uint32_t mask_word_count = fanout >> 6u;
            uint word_idx = 4u * thread_idx.x;
            uint4 mask_count = uint4(0u, 0u, 0u, 0u);
            if (word_idx < mask_word_count)
            {
                mask_count.x = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + word_idx + 0u));
                mask_count.y = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + word_idx + 1u));
                mask_count.z = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + word_idx + 2u));
                mask_count.w = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + word_idx + 3u));
            }
            uint4 scan_mask_count = uint4(0u, 0u, 0u, 0u);
            uint scan_total = 0u;
            workgroup_scan(thread_idx.x, mask_count, scan_mask_count, scan_total);
            if (word_idx < mask_word_count)
            {
                uint word_idx_h = (word_idx + 4) & (mask_word_count - 1u);

                pnanovdb_uint64_t prefix_sum_l = pnanovdb_uint32_as_uint64(
                    (scan_mask_count.x << 16u), scan_mask_count.y | (scan_mask_count.z << 16u) );
                pnanovdb_uint64_t prefix_sum_h = pnanovdb_uint32_as_uint64(scan_mask_count.w, 0u);

                InterlockedAnd(buf[node.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[node_type] + (word_idx >> 2u)], 0xFFFFllu);
                InterlockedOr(buf[node.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[node_type] + (word_idx >> 2u)], prefix_sum_l);
                InterlockedAnd(buf[node.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[node_type] + (word_idx_h >> 2u)], ~0xFFFFllu);
                InterlockedOr(buf[node.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[node_type] + (word_idx_h >> 2u)], prefix_sum_h);
            }
            new_child_count = scan_total;
            // root allocates background values for all
            if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
            {
                new_child_count += 2u;
            }
        }
        else if (node_type == constants.active_node_type && node_type != PNANOVDB_NODE2_TYPE_LEAF)
        {
            // pass to compute new child counts for active node type
            pnanovdb_uint32_t fanout = pnanovdb_node2_fanout_1d[node_type];
            pnanovdb_uint32_t mask_word_count = fanout >> 6u;
            uint word_idx = 4u * thread_idx.x;
            uint4 mask_count = uint4(0u, 0u, 0u, 0u);
            if (word_idx < mask_word_count)
            {
                mask_count.x = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + word_idx + 0u));
                mask_count.y = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + word_idx + 1u));
                mask_count.z = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + word_idx + 2u));
                mask_count.w = pnanovdb_uint64_countbits(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + word_idx + 3u));
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

                InterlockedAnd(buf[node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type] + (word_idx >> 2u)], 0xFFFFllu);
                InterlockedOr(buf[node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type] + (word_idx >> 2u)], prefix_sum_l);
                InterlockedAnd(buf[node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type] + (word_idx_h >> 2u)], ~0xFFFFllu);
                InterlockedOr(buf[node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type] + (word_idx_h >> 2u)], prefix_sum_h);
            }

            // compute new count
            if (word_idx < mask_word_count)
            {
                mask_count.x = pnanovdb_uint64_countbits(
                    pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + word_idx + 0u) &
                    pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + word_idx + 0u));
                mask_count.y = pnanovdb_uint64_countbits(
                    pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + word_idx + 1u) &
                    pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + word_idx + 1u));
                mask_count.z = pnanovdb_uint64_countbits(
                    pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + word_idx + 2u) &
                    pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + word_idx + 2u));
                mask_count.w = pnanovdb_uint64_countbits(
                    pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + word_idx + 3u) &
                    pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + word_idx + 3u));
            }
            workgroup_reduce(thread_idx.x, mask_count, new_child_count);
        }

        if (thread_idx.x == 0u)
        {
            new_child_counts[node_idx] = new_child_count;
        }
    }
}
