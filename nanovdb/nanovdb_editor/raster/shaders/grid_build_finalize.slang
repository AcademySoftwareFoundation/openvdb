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

        pnanovdb_uint64_t root_offset = pnanovdb_tree_get_node_offset_root(buf, tree);

        pnanovdb_uint32_t node_count = 1u +
                                       pnanovdb_tree_get_node_count_upper(buf, tree) +
                                       pnanovdb_tree_get_node_count_lower(buf, tree) +
                                       pnanovdb_tree_get_node_count_leaf(buf, tree);

        if (constants.active_node_type == constants.child_node_type)
        {
            // NOP for voxel pass
        }
        else if (node_count > 0u)
        {
            uint total_new_count = scan_new_child_counts[node_count - 1u];

            if (constants.child_node_type == PNANOVDB_NODE2_TYPE_UPPER)
            {
                uint upper_count = pnanovdb_tree_get_node_count_upper(buf, tree);
                upper_count += total_new_count;
                pnanovdb_tree_set_node_count_upper(buf, tree, upper_count);
            }
            if (constants.child_node_type == PNANOVDB_NODE2_TYPE_LOWER)
            {
                uint lower_count = pnanovdb_tree_get_node_count_lower(buf, tree);
                lower_count += total_new_count;
                pnanovdb_tree_set_node_count_lower(buf, tree, lower_count);
            }
            if (constants.child_node_type == PNANOVDB_NODE2_TYPE_LEAF)
            {
                uint leaf_count = pnanovdb_tree_get_node_count_leaf(buf, tree);
                leaf_count += total_new_count;
                pnanovdb_tree_set_node_count_leaf(buf, tree, leaf_count);
            }
            // todo: set per node type offsets
        }
    }
}
