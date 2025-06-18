// node2_verify_gpu_ref.slang
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

    pnanovdb_grid_handle_t grid = {};
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
    pnanovdb_grid_type_t grid_type = PNANOVDB_GRID_TYPE_FLOAT; //pnanovdb_grid_get_grid_type(buf, grid);

    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(PNANOVDB_REF(acc), root);

     pnanovdb_coord_t ijk = {
        (constants.bbox_max.x + constants.bbox_min.x) / 2,
        tidx.x + constants.bbox_min.y,
        tidx.y + constants.bbox_min.z
    };

    float val_gpu = 0.f;
    for (int i = 0; i < PNANOVDB_NODE2_TEST_X_DEPTH; i++)
    {
#if PNANOVDB_NODE2_TEST_WITH_ACCESSOR
        pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address(grid_type, buf, PNANOVDB_REF(acc), PNANOVDB_REF(ijk));
#else
        pnanovdb_address_t addr = pnanovdb_root_get_value_address(grid_type, buf, root, PNANOVDB_REF(ijk));
#endif
        val_gpu = max(val_gpu, pnanovdb_read_float(buf, addr));
        ijk.x++;
    }

    const pnanovdb_uint32_t element_words = 1u;
    pnanovdb_uint32_t idx =
        (ijk.y - constants.bbox_min.y) * (constants.bbox_max.z - constants.bbox_min.z) +
        (ijk.z - constants.bbox_min.z);
    data_out[element_words * idx + 0] = pnanovdb_float_as_uint32(val_gpu);
}
