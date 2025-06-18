// set_masks.slang
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

StructuredBuffer<uint> point_ijk_in;

RWStructuredBuffer<uint64_t> buf;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint ijk_idx = group_idx.x * 256u + thread_idx.x;

    if (ijk_idx >= constants.point_count)
    {
        return;
    }

    pnanovdb_coord_t ijk;
    ijk.x = point_ijk_in[3u * ijk_idx + 0u];
    ijk.y = point_ijk_in[3u * ijk_idx + 1u];
    ijk.z = point_ijk_in[3u * ijk_idx + 2u];

    pnanovdb_grid_handle_t grid = { pnanovdb_address_null() };
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_node2_handle_t root = { pnanovdb_tree_get_root(buf, tree).address.byte_offset >> 3u };

    pnanovdb_uint32_t node_n = 0u;
    pnanovdb_uint32_t node_type = 0u;
    pnanovdb_uint32_t level = 0u;
    pnanovdb_node2_handle_t node;
    pnanovdb_node2_find_node(buf, root, PNANOVDB_REF(node), PNANOVDB_REF(node_type), PNANOVDB_REF(node_n), PNANOVDB_REF(level), ijk);

    if (node_type == constants.active_node_type && ijk.z != 0x7FFFFFFF)
    {
        pnanovdb_uint32_t base_offset = 0u;
        if (node_type == PNANOVDB_NODE2_TYPE_LEAF)
        {
            base_offset = pnanovdb_node2_off_value_mask[node_type];
        }
        else
        {
            base_offset = pnanovdb_node2_off_child_mask[node_type];
        }
        pnanovdb_uint32_t word_idx = node_n >> 6u; // 64-bit words
        pnanovdb_uint32_t bit_idx = node_n & 63u;
        pnanovdb_uint32_t idx64 = node.idx64 + base_offset + word_idx;

        pnanovdb_uint64_t or_val = 1llu << bit_idx;
        pnanovdb_uint64_t old_val;
        InterlockedOr(buf[idx64], or_val, old_val);
        if (node_type != PNANOVDB_NODE2_TYPE_LEAF && (or_val & old_val) == 0llu)
        {
            // child was newly set true, also set value mask to indicate child is new
            base_offset = pnanovdb_node2_off_value_mask[node_type];
            idx64 = node.idx64 + base_offset + word_idx;
            InterlockedOr(buf[idx64], or_val);
        }
    }
}
