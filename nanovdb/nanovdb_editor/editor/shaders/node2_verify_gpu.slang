// node2_verify_gpu.slang
#define PNANOVDB_HLSL
#define PNANOVDB_ADDRESS_64
#define PNANOVDB_BUF_HLSL_64
#include "PNanoVDB.h"
#include "PNanoVDB2.h"

struct constants_t
{
    pnanovdb_coord_t bbox_min;
    pnanovdb_uint32_t grid_dim_x;
    pnanovdb_coord_t bbox_max;
    pnanovdb_uint32_t grid_dim_y;
};

StructuredBuffer<uint64_t> data_in;
ConstantBuffer<constants_t> constants;
RWStructuredBuffer<uint> data_out;
RWStructuredBuffer<uint> scratch;

// makes pnanovdb code more copy paste friendly
#define buf data_in

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 tidx = int2(dispatchThreadID.xy);

    if (tidx.x >= (constants.bbox_max.y - constants.bbox_min.y) ||
        tidx.y >= (constants.bbox_max.z - constants.bbox_min.z))
    {
        return;
    }

    pnanovdb_grid_handle_t grid = { pnanovdb_address_null() };
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_node2_handle_t root = { pnanovdb_tree_get_root(buf, tree).address.byte_offset >> 3u };
    pnanovdb_node2_levelset_values_t values = {};
    values.values = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 1u);
    values.node_inactive_idxs = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 2u);
    values.inactive_value_idxs = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 3u);

     pnanovdb_coord_t ijk = {
        (constants.bbox_max.x + constants.bbox_min.x) / 2,
        tidx.x + constants.bbox_min.y,
        tidx.y + constants.bbox_min.z
    };

    // pnanovdb_node2_handle_t node = root;
    pnanovdb_node2_accessor_t acc;
    pnanovdb_node2_accessor_init(PNANOVDB_REF(acc), root);

    float val_gpu = 0.f;
    for (int i = 0; i < PNANOVDB_NODE2_TEST_X_DEPTH; i++)
    {
        pnanovdb_node2_handle_t node = {};
        pnanovdb_uint32_t node_type = 0u;
        pnanovdb_uint32_t node_n = 0u;
        pnanovdb_uint32_t level = 0u;
#if PNANOVDB_NODE2_TEST_WITH_ACCESSOR
        pnanovdb_node2_accessor_find_node(buf, acc, node, node_type, node_n, level, ijk);
#else
        pnanovdb_node2_find_node(buf, root, node, node_type, node_n, level, ijk);
#endif
        pnanovdb_uint64_t value_idx = pnanovdb_node2_get_value_index(buf, node, node_type, node_n, PNANOVDB_TRUE, values.node_inactive_idxs, values.inactive_value_idxs);
        pnanovdb_address_t val_addr = pnanovdb_address_offset64_product(values.values, value_idx, 4u);

        val_gpu = max(val_gpu, pnanovdb_read_float(buf, val_addr));
        ijk.x++;
    }

    const pnanovdb_uint32_t element_words = 1u;
    pnanovdb_uint32_t idx =
        (ijk.y - constants.bbox_min.y) * (constants.bbox_max.z - constants.bbox_min.z) +
        (ijk.z - constants.bbox_min.z);
    data_out[element_words * idx + 0] = pnanovdb_float_as_uint32(val_gpu);
}
