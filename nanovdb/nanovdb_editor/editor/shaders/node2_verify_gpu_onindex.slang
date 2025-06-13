// node2_verify_gpu_onindex.slang
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
    pnanovdb_grid_type_t grid_type = PNANOVDB_GRID_TYPE_ONINDEX; // pnanovdb_grid_get_grid_type(buf, grid);

    pnanovdb_gridblindmetadata_handle_t meta = pnanovdb_grid_get_gridblindmetadata(buf, grid, 0u);
    pnanovdb_uint64_t value_count = pnanovdb_gridblindmetadata_get_value_count(buf, meta);
    pnanovdb_int64_t byte_offset = pnanovdb_gridblindmetadata_get_data_offset(buf, meta);
    pnanovdb_address_t val_addr = pnanovdb_address_offset64(meta.address, pnanovdb_int64_as_uint64(byte_offset));

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
        pnanovdb_uint32_t level = 0u;
#if PNANOVDB_NODE2_TEST_WITH_ACCESSOR
        pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address_and_level(grid_type, buf, PNANOVDB_REF(acc), PNANOVDB_REF(ijk), level);
#else
        pnanovdb_address_t addr = pnanovdb_root_get_value_address_and_level(grid_type, buf, root, PNANOVDB_REF(ijk), level);
#endif
        float val = 0.f;
        pnanovdb_uint64_t val_index64;
        if (level == 0)
        {
            val_index64 = pnanovdb_leaf_onindex_get_value_index(buf, addr, ijk);
        }
        else
        {
            val_index64 = pnanovdb_read_uint64(buf, addr);
        }
        pnanovdb_uint32_t val_index = pnanovdb_uint64_low(val_index64);
        if (val_index < pnanovdb_uint64_low(value_count))
        {
            val_addr = pnanovdb_address_offset_product(val_addr, val_index, 4u);
            val = pnanovdb_read_float(buf, val_addr);
        }
        val_gpu = max(val_gpu, val);
        ijk.x++;
    }

    const pnanovdb_uint32_t element_words = 1u;
    pnanovdb_uint32_t idx =
        (ijk.y - constants.bbox_min.y) * (constants.bbox_max.z - constants.bbox_min.z) +
        (ijk.z - constants.bbox_min.z);
    data_out[element_words * idx + 0] = pnanovdb_float_as_uint32(val_gpu);
}
