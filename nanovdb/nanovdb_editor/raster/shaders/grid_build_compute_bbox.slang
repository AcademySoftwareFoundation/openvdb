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
RWStructuredBuffer<uint> bboxes;

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
    if (node_count >= constants.max_node_count)
    {
        node_count = constants.max_node_count;
    }
    for (pnanovdb_uint32_t node_idx = node_begin_idx; node_idx < node_count; node_idx+=constants.workgroup_count)
    {
        pnanovdb_node2_handle_t node = {node_addresses[node_idx] >> 3u};
        pnanovdb_uint32_t node_type = node_types[node_idx];

        if (node_type == constants.active_node_type &&
            constants.active_node_type == constants.child_node_type)
        {
            pnanovdb_uint32_t fanout = pnanovdb_node2_fanout_1d[node_type];

            pnanovdb_coord_t bbox_min = { WORKGROUP_INT_MAX, WORKGROUP_INT_MAX, WORKGROUP_INT_MAX };
            pnanovdb_coord_t bbox_max = { WORKGROUP_INT_MIN, WORKGROUP_INT_MIN, WORKGROUP_INT_MIN };
            for (uint n = thread_idx.x; n < fanout; n += 256u)
            {
                if (pnanovdb_node2_get_value_mask_bit(buf, node, node_type, n))
                {
                    pnanovdb_coord_t ijk;
                    uint fanout_bits = pnanovdb_node2_fanout_bits[node_type];
                    uint tiledim_bits = pnanovdb_node2_tiledim_bits[node_type];
                    uint ijk_mask = (1u << fanout_bits) - 1u;
                    ijk.x = int(((n >> (fanout_bits + fanout_bits)) & ijk_mask) << tiledim_bits);
                    ijk.y = int(((n >> (fanout_bits)) & ijk_mask) << tiledim_bits);
                    ijk.z = int((n & ijk_mask) << tiledim_bits);
                    bbox_min.x = workgroup_int_min(bbox_min.x, ijk.x);
                    bbox_min.y = workgroup_int_min(bbox_min.y, ijk.y);
                    bbox_min.z = workgroup_int_min(bbox_min.z, ijk.z);
                    bbox_max.x = workgroup_int_max(bbox_max.x, ijk.x);
                    bbox_max.y = workgroup_int_max(bbox_max.y, ijk.y);
                    bbox_max.z = workgroup_int_max(bbox_max.z, ijk.z);
                }
            }

            workgroup_reduce_min_int(thread_idx.x, bbox_min.x, bbox_min.x);
            workgroup_reduce_min_int(thread_idx.x, bbox_min.y, bbox_min.y);
            workgroup_reduce_min_int(thread_idx.x, bbox_min.z, bbox_min.z);
            workgroup_reduce_max_int(thread_idx.x, bbox_max.x, bbox_max.x);
            workgroup_reduce_max_int(thread_idx.x, bbox_max.y, bbox_max.y);
            workgroup_reduce_max_int(thread_idx.x, bbox_max.z, bbox_max.z);

            if (thread_idx.x == 0)
            {
                bboxes[6u * node_idx + 0u] = uint(bbox_min.x);
                bboxes[6u * node_idx + 1u] = uint(bbox_min.y);
                bboxes[6u * node_idx + 2u] = uint(bbox_min.z);
                bboxes[6u * node_idx + 3u] = uint(bbox_max.x);
                bboxes[6u * node_idx + 4u] = uint(bbox_max.y);
                bboxes[6u * node_idx + 5u] = uint(bbox_max.z);
            }
        }
        else if (node_type == constants.active_node_type && node_type != PNANOVDB_NODE2_TYPE_LEAF)
        {
            pnanovdb_uint32_t fanout = pnanovdb_node2_fanout_1d[node_type];

            pnanovdb_coord_t bbox_min = { WORKGROUP_INT_MAX, WORKGROUP_INT_MAX, WORKGROUP_INT_MAX };
            pnanovdb_coord_t bbox_max = { WORKGROUP_INT_MIN, WORKGROUP_INT_MIN, WORKGROUP_INT_MIN };
            for (uint n = thread_idx.x; n < fanout; n += 256u)
            {
                pnanovdb_node2_handle_t child_node = pnanovdb_node2_get_child(buf, node, node_type, n);
                if (child_node.idx64 != 0u)
                {
                    uint child_node_idx = uint(pnanovdb_node2_read(buf, child_node.idx64 + pnanovdb_node2_off_node_idx));
                    pnanovdb_coord_t ijk;
                    uint fanout_bits = pnanovdb_node2_fanout_bits[node_type];
                    uint tiledim_bits = pnanovdb_node2_tiledim_bits[node_type];
                    uint ijk_mask = (1u << fanout_bits) - 1u;
                    ijk.x = int(((n >> (fanout_bits + fanout_bits)) & ijk_mask) << tiledim_bits);
                    ijk.y = int(((n >> (fanout_bits)) & ijk_mask) << tiledim_bits);
                    ijk.z = int((n & ijk_mask) << tiledim_bits);

                    if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
                    {
                        const pnanovdb_uint32_t fanout_half = (1u << fanout_bits) >> 1u;
                        const pnanovdb_uint32_t ijk_offset = (fanout_half << tiledim_bits);
                        const pnanovdb_uint32_t ijk_range = (fanout_half + fanout_half) << tiledim_bits;
                        // create modulo wrap for negative space
                        if (ijk.x >= ijk_offset) { ijk.x -= int(ijk_range); }
                        if (ijk.y >= ijk_offset) { ijk.y -= int(ijk_range); }
                        if (ijk.z >= ijk_offset) { ijk.z -= int(ijk_range); }
                    }

                    bbox_min.x = workgroup_int_min(bbox_min.x, int(bboxes[6u * child_node_idx + 0u]) + ijk.x);
                    bbox_min.y = workgroup_int_min(bbox_min.y, int(bboxes[6u * child_node_idx + 1u]) + ijk.y);
                    bbox_min.z = workgroup_int_min(bbox_min.z, int(bboxes[6u * child_node_idx + 2u]) + ijk.z);
                    bbox_max.x = workgroup_int_max(bbox_max.x, int(bboxes[6u * child_node_idx + 3u]) + ijk.x);
                    bbox_max.y = workgroup_int_max(bbox_max.y, int(bboxes[6u * child_node_idx + 4u]) + ijk.y);
                    bbox_max.z = workgroup_int_max(bbox_max.z, int(bboxes[6u * child_node_idx + 5u]) + ijk.z);
                }
            }

            workgroup_reduce_min_int(thread_idx.x, bbox_min.x, bbox_min.x);
            workgroup_reduce_min_int(thread_idx.x, bbox_min.y, bbox_min.y);
            workgroup_reduce_min_int(thread_idx.x, bbox_min.z, bbox_min.z);
            workgroup_reduce_max_int(thread_idx.x, bbox_max.x, bbox_max.x);
            workgroup_reduce_max_int(thread_idx.x, bbox_max.y, bbox_max.y);
            workgroup_reduce_max_int(thread_idx.x, bbox_max.z, bbox_max.z);

            if (thread_idx.x == 0)
            {
                bboxes[6u * node_idx + 0u] = uint(bbox_min.x);
                bboxes[6u * node_idx + 1u] = uint(bbox_min.y);
                bboxes[6u * node_idx + 2u] = uint(bbox_min.z);
                bboxes[6u * node_idx + 3u] = uint(bbox_max.x);
                bboxes[6u * node_idx + 4u] = uint(bbox_max.y);
                bboxes[6u * node_idx + 5u] = uint(bbox_max.z);

                if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
                {
                    pnanovdb_address_t bboxes = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 0u);
                    pnanovdb_write_coord(buf, pnanovdb_address_offset(bboxes, 0u), bbox_min);
                    pnanovdb_write_coord(buf, pnanovdb_address_offset(bboxes, 12u), bbox_max);
                }
            }
        }
    }
}
