// point_frag_color.slang
#define PNANOVDB_HLSL
#define PNANOVDB_ADDRESS_64
#define PNANOVDB_BUF_HLSL_64
#define PNANOVDB_BUF_HLSL_RW
#include "PNanoVDB.h"
#include "PNanoVDB2.h"

struct constants_t
{
    uint point_count;
    float pad1;
    float pad2;
    float pad3;
};

ConstantBuffer<constants_t> constants;

StructuredBuffer<uint> colors_in;
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

    int3 ijk;
    ijk.x = point_ijk_in[3u * ijk_idx + 0u];
    ijk.y = point_ijk_in[3u * ijk_idx + 1u];
    ijk.z = point_ijk_in[3u * ijk_idx + 2u];

    float4 color;
    color.r = asfloat(colors_in[3u * ijk_idx + 0u]);
    color.g = asfloat(colors_in[3u * ijk_idx + 1u]);
    color.b = asfloat(colors_in[3u * ijk_idx + 2u]);
    color.a = 1.f;

    uint rgba_raw = uint(255.f * max(0.f, min(1.f, color.x))) |
                    (uint(255.f * max(0.f, min(1.f, color.y))) << 8u) |
                    (uint(255.f * max(0.f, min(1.f, color.z))) << 16u) |
                    (uint(255.f * max(0.f, min(1.f, color.w))) << 24u);

    pnanovdb_grid_handle_t grid = { pnanovdb_address_null() };
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_node2_handle_t root = { pnanovdb_tree_get_root(buf, tree).address.byte_offset >> 3u };
    pnanovdb_address_t values = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 1u);

    pnanovdb_uint32_t node_n = 0u;
    pnanovdb_uint32_t node_type = 0u;
    pnanovdb_uint32_t level = 0u;
    pnanovdb_node2_handle_t node;
    pnanovdb_node2_find_node(buf, root, PNANOVDB_REF(node), PNANOVDB_REF(node_type), PNANOVDB_REF(node_n), PNANOVDB_REF(level), ijk);

    pnanovdb_uint64_t value_idx = pnanovdb_node2_get_value_index(buf, node, node_type, node_n, PNANOVDB_FALSE, pnanovdb_address_null(), pnanovdb_address_null());
    pnanovdb_address_t val_addr = pnanovdb_address_offset64_product(values, value_idx, 4u);

    pnanovdb_uint32_t word_idx = uint(val_addr.byte_offset >> 3u);
    pnanovdb_uint32_t sub_idx = uint(val_addr.byte_offset & 4);

    pnanovdb_uint64_t old_value = 0llu;
    pnanovdb_uint64_t cmp_value = 1llu;
    while (cmp_value != old_value)
    {
        cmp_value = old_value;

        pnanovdb_uint32_t x = pnanovdb_uint32_t(old_value);
        pnanovdb_uint32_t y = pnanovdb_uint32_t(old_value >> 32u);
        if (sub_idx == 0u) { x = max(x, rgba_raw); }
        else { y = max(y, rgba_raw); }

        pnanovdb_uint64_t value = pnanovdb_uint64_t(x) | (pnanovdb_uint64_t(y) << 32u);
        if (value == old_value)
        {
            break;
        }
        InterlockedCompareExchange(buf[word_idx], cmp_value, value, old_value);
    }
}
