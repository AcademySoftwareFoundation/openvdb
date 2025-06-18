// gaussian_frag_color.slang
#define PNANOVDB_HLSL
#define PNANOVDB_ADDRESS_64
#define PNANOVDB_BUF_HLSL_64
#define PNANOVDB_BUF_HLSL_RW
#include "PNanoVDB.h"
#include "PNanoVDB2.h"

struct constants_t
{
    uint prim_batch_size;
    uint ijk_batch_size;
    uint64_t fragment_window_min;
    uint64_t fragment_window_max;
    uint global_prim_begin_idx;
    uint global_prim_count;
    float voxel_size;
    float voxel_size_inv;
};

ConstantBuffer<constants_t> constants;

StructuredBuffer<uint> positions_in;
StructuredBuffer<uint> quaterions_in;
StructuredBuffer<uint> scales_in;
StructuredBuffer<uint> opacities_in;
StructuredBuffer<uint> colors_in;

StructuredBuffer<uint> prim_idxs_in;
StructuredBuffer<uint> prim_raster_idxs_in;
StructuredBuffer<uint> prim_bboxs_in;

RWStructuredBuffer<uint64_t> buf;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint ijk_idx = group_idx.x * 256u + thread_idx.x;

    if (ijk_idx >= constants.ijk_batch_size)
    {
        return;
    }
    uint prim_raster_idx = prim_raster_idxs_in[ijk_idx];
    if (prim_raster_idx == ~0u)
    {
        return;
    }

    uint prim_idx = prim_idxs_in[ijk_idx];
    uint global_prim_idx = prim_idx + constants.global_prim_begin_idx;

    int3 ijk_min;
    int3 ijk_max;
    ijk_min.x = prim_bboxs_in[6u * prim_idx + 0u];
    ijk_min.y = prim_bboxs_in[6u * prim_idx + 1u];
    ijk_min.z = prim_bboxs_in[6u * prim_idx + 2u];
    ijk_max.x = prim_bboxs_in[6u * prim_idx + 3u];
    ijk_max.y = prim_bboxs_in[6u * prim_idx + 4u];
    ijk_max.z = prim_bboxs_in[6u * prim_idx + 5u];

    int3 range = (ijk_max + 1 - ijk_min);
    int3 ijk = int3(
        int(prim_raster_idx) / (range.z * range.y),
        (int(prim_raster_idx) / range.z) % range.y,
        int(prim_raster_idx) % range.z
    );
    ijk += ijk_min;

#if 1
    float3 posf;
    posf.x = asfloat(positions_in[3u * global_prim_idx + 0u]);
    posf.y = asfloat(positions_in[3u * global_prim_idx + 1u]);
    posf.z = asfloat(positions_in[3u * global_prim_idx + 2u]);

    float4 quatf;
    quatf.x = asfloat(quaterions_in[4u * global_prim_idx + 1u]);
    quatf.y = asfloat(quaterions_in[4u * global_prim_idx + 2u]);
    quatf.z = asfloat(quaterions_in[4u * global_prim_idx + 3u]);
    quatf.w = asfloat(quaterions_in[4u * global_prim_idx + 0u]);

    float3 scalef;
    scalef.x = asfloat(scales_in[3u * global_prim_idx + 0u]);
    scalef.y = asfloat(scales_in[3u * global_prim_idx + 1u]);
    scalef.z = asfloat(scales_in[3u * global_prim_idx + 2u]);

    float opacity = asfloat(opacities_in[global_prim_idx]);

    const float local_scale = sqrt(2.f * log(opacity / 0.01f));

    const float minScale = (1.f / 8.f);
    scalef = max(scalef, minScale * constants.voxel_size.xxx);

    float3 voxel_min;
    float3 voxel_max;
    for (uint vert_idx = 0u; vert_idx < 8u; vert_idx++)
    {
        float3 voxel_pos = float3(ijk);
        if ((vert_idx & 1u) != 0u) { voxel_pos.x += 1.f;}
        if ((vert_idx & 2u) != 0u) { voxel_pos.y += 1.f;}
        if ((vert_idx & 4u) != 0u) { voxel_pos.z += 1.f;}

        float3 local_pos = voxel_pos * constants.voxel_size;

        local_pos -= posf;

        float3 rot0_inv = float3(
            1.f - 2.f * (quatf.y * quatf.y + quatf.z * quatf.z),
            2.f * (quatf.x * quatf.y + quatf.z * quatf.w),
            2.f * (quatf.x * quatf.z - quatf.y * quatf.w));
        float3 rot1_inv = float3(
            2.f * (quatf.x * quatf.y - quatf.z * quatf.w),
            1.f - 2.f * (quatf.x * quatf.x + quatf.z * quatf.z),
            2.f * (quatf.y * quatf.z + quatf.x * quatf.w));
        float3 rot2_inv = float3(
            2.f * (quatf.x * quatf.z + quatf.y * quatf.w),
            2.f * (quatf.y * quatf.z - quatf.x * quatf.w),
            1.f - 2.f * (quatf.x * quatf.x + quatf.y * quatf.y));

        local_pos = float3(
            dot(local_pos, rot0_inv),
            dot(local_pos, rot1_inv),
            dot(local_pos, rot2_inv));

        local_pos *= (1.f / local_scale);
        local_pos *= (float3(1.f, 1.f, 1.f) / scalef);

        if (vert_idx == 0u)
        {
            voxel_min = local_pos;
            voxel_max = local_pos;
        }
        else
        {
            voxel_min = min(voxel_min, local_pos);
            voxel_max = max(voxel_max, local_pos);
        }
    }
    float dist2 = 1.f;
    if (0.f < voxel_min.x) { dist2 -= voxel_min.x * voxel_min.x;}
    else if (0.f > voxel_max.x) { dist2 -= voxel_max.x * voxel_max.x;}
    if (0.f < voxel_min.y) { dist2 -= voxel_min.y * voxel_min.y;}
    else if (0.f > voxel_max.y) { dist2 -= voxel_max.y * voxel_max.y;}
    if (0.f < voxel_min.z) { dist2 -= voxel_min.z * voxel_min.z;}
    else if (0.f > voxel_max.z) { dist2 -= voxel_max.z * voxel_max.z;}
    if (dist2 <= 0.f)
    {
        ijk.x = 0x7FFFFFFF;
        ijk.y = 0x7FFFFFFF;
        ijk.z = 0x7FFFFFFF;
    }
#endif

    if (ijk.z != 0x7FFFFFFF)
    {
        float4 color;
        color.r = asfloat(colors_in[3u * global_prim_idx + 0u]);
        color.g = asfloat(colors_in[3u * global_prim_idx + 1u]);
        color.b = asfloat(colors_in[3u * global_prim_idx + 2u]);
        // color.a = 1.f;
        color.a = 0.f;
        for (uint sample_idx = 0u; sample_idx < 64u; sample_idx++)
        {
            float3 w = (1.f / 3.f) * float3(
                float((sample_idx >> 4) & 3),
                float((sample_idx >> 2) & 3),
                float((sample_idx) & 3));
            float3 ow = 1.f - w;
            float3 local_pos = w * voxel_max + ow * voxel_min;
            color.a = max(color.a, opacity * exp(-(local_scale * dot(local_pos, local_pos))));
        }

        const float scale1d_k = 0.1f;
        float scale1d = length(scalef);
        float voxelRelativeScale = (scale1d_k * scale1d) / constants.voxel_size;
        color.a = 1.f - pow(1.f - color.a, 1.f / voxelRelativeScale);
        // due to screen space effects, large Gaussians never behave well, filter out
        // TODO: precompute some histogram of scales to better estimate this threshold
        const float cull_scale = 0.05f;
        if (scale1d > cull_scale)
        {
            color.a = 0.f;
        }

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

        if (node_type == PNANOVDB_NODE2_TYPE_LEAF)
        {
            if (pnanovdb_node2_get_value_mask_bit(buf, node, PNANOVDB_NODE2_TYPE_LEAF, node_n))
            {
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
        }
    }
}
