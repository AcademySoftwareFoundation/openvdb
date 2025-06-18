// scatter_new_children.slang
#define PNANOVDB_HLSL
#define PNANOVDB_ADDRESS_64
#define PNANOVDB_BUF_HLSL_64
//#define PNANOVDB_BUF_HLSL_RW
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

StructuredBuffer<uint64_t> buf;

RWStructuredBuffer<uint> node_addresses;
RWStructuredBuffer<uint> node_types;

#include <workgroup_scan.slang>

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint node_idx = group_idx.x * 256u + thread_idx.x;

    if (node_idx >= constants.max_node_count)
    {
        return;
    }

    uint byte_offset = 0u;
    uint node_type = 0u;
    if (node_idx == 0u)
    {
        pnanovdb_grid_handle_t grid = { pnanovdb_address_null() };
        pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
        pnanovdb_node2_handle_t root = { pnanovdb_tree_get_root(buf, tree).address.byte_offset >> 3u };

        byte_offset = root.idx64 << 3u;
        node_type = PNANOVDB_NODE2_TYPE_ROOT;
    }

    node_addresses[node_idx] = byte_offset;
    node_types[node_idx] = node_type;
}
