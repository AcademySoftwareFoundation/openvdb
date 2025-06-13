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
RWStructuredBuffer<uint> node_addresses;
RWStructuredBuffer<uint> node_types;

#include <workgroup_scan.slang>

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
    for (pnanovdb_uint32_t node_idx = node_begin_idx; node_idx < node_count; node_idx+=constants.workgroup_count)
    {
        pnanovdb_node2_handle_t node = {node_addresses[node_idx] >> 3u};
        pnanovdb_uint32_t node_type = node_types[node_idx];

        if (constants.active_node_type == constants.child_node_type)
        {
            if (thread_idx.x == 0u)
            {
                uint value_idx = scan_new_child_counts[node_idx] - new_child_counts[node_idx];
                // root allocated background values for all, offset past background
                if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
                {
                    value_idx += 2u;
                }
                pnanovdb_node2_write(buf, node.idx64 + pnanovdb_node2_off_value_idx, pnanovdb_uint32_as_uint64_low(value_idx));
            }
        }
        else if (node_type == constants.active_node_type && node_type != PNANOVDB_NODE2_TYPE_LEAF)
        {
            pnanovdb_uint32_t fanout = pnanovdb_node2_fanout_1d[node_type];

            // using value mask to detect new children, shift existing children to create holes
            uint total_new_child_count = new_child_counts[node_idx];
            uint inv_new_child_count = 0u;
            for (uint n_f = thread_idx.x; n_f < fanout; n_f += 256u)
            {
                uint n = fanout - 1u - n_f;

                uint4 val4 = uint4(0u, 0u, 0u, 0u);
                if (pnanovdb_node2_get_child_mask_bit(buf, node, node_type, n) &&
                    pnanovdb_node2_get_value_mask_bit(buf, node, node_type, n))
                {
                    val4.x = 1u;
                }

                uint4 scan4 = uint4(0u, 0u, 0u, 0u);
                uint scan_total = 0u;
                workgroup_scan(thread_idx.x, val4, scan4, scan_total);

                bool is_old_child = pnanovdb_node2_get_child_mask_bit(buf, node, node_type, n) &&
                                    !pnanovdb_node2_get_value_mask_bit(buf, node, node_type, n);
                uint dst_idx;
                uint src_idx;
                uint64_t child;
                if (is_old_child)
                {
                    pnanovdb_bool_t is_active;
                    dst_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type], node_type, n, PNANOVDB_REF(is_active));
                    src_idx = dst_idx - total_new_child_count + inv_new_child_count + scan4.x;
                    child = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + src_idx);
                }

                GroupMemoryBarrierWithGroupSync();

                if (is_old_child)
                {
                    pnanovdb_node2_write(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + dst_idx, child);
                }

                GroupMemoryBarrierWithGroupSync();

                inv_new_child_count += scan_total;
            }

            uint accum_offset = 0u;
            for (uint n = thread_idx.x; n < fanout; n += 256u)
            {
                uint4 val4 = uint4(0u, 0u, 0u, 0u);
                if (pnanovdb_node2_get_child_mask_bit(buf, node, node_type, n) &&
                    pnanovdb_node2_get_value_mask_bit(buf, node, node_type, n))
                {
                    val4.x = 1u;
                }

                uint4 scan4 = uint4(0u, 0u, 0u, 0u);
                uint scan_total = 0u;
                workgroup_scan(thread_idx.x, val4, scan4, scan_total);

                if (val4.x != 0u)
                {
                    // compute local child idx
                    uint new_child_idx = scan4.x - val4.x + accum_offset;
                    // add global idx
                    new_child_idx += (scan_new_child_counts[node_idx] - new_child_counts[node_idx]);

                    pnanovdb_address_t new_node_addr = tree.address;
                    new_node_addr = pnanovdb_address_offset64(new_node_addr,
                        pnanovdb_tree_get_node_offset_root(buf, tree));
                    new_node_addr = pnanovdb_address_offset64_product(new_node_addr,
                        1u, pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_ROOT]);
                    new_node_addr = pnanovdb_address_offset64_product(new_node_addr,
                        pnanovdb_tree_get_node_count_upper(buf, tree), pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_UPPER]);
                    new_node_addr = pnanovdb_address_offset64_product(new_node_addr,
                        pnanovdb_tree_get_node_count_lower(buf, tree), pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_LOWER]);
                    new_node_addr = pnanovdb_address_offset64_product(new_node_addr,
                        pnanovdb_tree_get_node_count_leaf(buf, tree), pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_LEAF]);
                    new_node_addr = pnanovdb_address_offset_product(new_node_addr,
                        new_child_idx, pnanovdb_node2_max_size[constants.child_node_type]);

                    pnanovdb_node2_handle_t new_node = { new_node_addr.byte_offset >> 3u };

                    // add prexisting node count
                    uint new_node_idx = new_child_idx + node_count;

                    if (new_node_idx < constants.max_node_count)
                    {
                        // register new node with arrays
                        node_addresses[new_node_idx] = uint(new_node_addr.byte_offset);
                        node_types[new_node_idx] = constants.child_node_type;

                        // connect to parent
                        pnanovdb_bool_t is_active;
                        pnanovdb_uint32_t child_idx = pnanovdb_node2_mask_n_to_idx(buf,
                            node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type], node_type, n, PNANOVDB_REF(is_active));

                        pnanovdb_address_t node_addr = { uint64_t(node.idx64) << 3u };
                        pnanovdb_node2_write(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + child_idx,
                            pnanovdb_address_diff(new_node_addr, node_addr));

                        // set child's node_idx
                        pnanovdb_node2_write(buf, new_node.idx64 + pnanovdb_node2_off_node_idx, new_node_idx);
                    }
                }
                accum_offset += scan_total;
            }

            GroupMemoryBarrierWithGroupSync();

            // clear any value mask bits that were temporarily set to indicate new child
            pnanovdb_uint32_t mask_word_count = fanout >> 6u;
            for (uint n = thread_idx.x; n < mask_word_count; n += 256u)
            {
                pnanovdb_uint64_t child_mask_word = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + n);
                pnanovdb_uint64_t value_mask_word = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + n);

                value_mask_word = value_mask_word & ~child_mask_word;

                pnanovdb_node2_write(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + n, value_mask_word);
            }
        }
    }
}
