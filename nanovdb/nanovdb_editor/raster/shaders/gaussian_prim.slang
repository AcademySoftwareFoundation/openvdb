// gaussian_prim.slang

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

RWStructuredBuffer<uint64_t> fragment_counts_out;
RWStructuredBuffer<uint> prim_bbox_out;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint prim_idx = group_idx.x * 256u + thread_idx.x;

    if (prim_idx >= constants.prim_batch_size)
    {
        return;
    }
    uint global_prim_idx = prim_idx + constants.global_prim_begin_idx;
    if (global_prim_idx >= constants.global_prim_count)
    {
        fragment_counts_out[prim_idx] = 0u;
        prim_bbox_out[6u * prim_idx + 0u] = 0;
        prim_bbox_out[6u * prim_idx + 1u] = 0;
        prim_bbox_out[6u * prim_idx + 2u] = 0;
        prim_bbox_out[6u * prim_idx + 3u] = -1;
        prim_bbox_out[6u * prim_idx + 4u] = -1;
        prim_bbox_out[6u * prim_idx + 5u] = -1;
        return;
    }

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

    float local_scale = sqrt(2.f * log(opacity / 0.01f));

    const float minScale = (1.f / 8.f);
    scalef = max(scalef, minScale * constants.voxel_size.xxx);

    // transform cube and accumulate AABB
    int3 ijk_min;
    int3 ijk_max;
    for (uint vert_idx = 0u; vert_idx < 8u; vert_idx++)
    {
        float3 cube_pos;
        cube_pos.x = (vert_idx & 1u) == 0u ? -1.f : 1.f;
        cube_pos.y = (vert_idx & 2u) == 0u ? -1.f : 1.f;
        cube_pos.z = (vert_idx & 4u) == 0u ? -1.f : 1.f;

        cube_pos *= scalef;
        cube_pos *= local_scale;

        float3 rot0 = float3(
            1.f - 2.f * (quatf.y * quatf.y + quatf.z * quatf.z),
            2.f * (quatf.x * quatf.y - quatf.z * quatf.w),
            2.f * (quatf.x * quatf.z + quatf.y * quatf.w));
        float3 rot1 = float3(
            2.f * (quatf.x * quatf.y + quatf.z * quatf.w),
            1.f - 2.f * (quatf.x * quatf.x + quatf.z * quatf.z),
            2.f * (quatf.y * quatf.z - quatf.x * quatf.w));
        float3 rot2 = float3(
            2.f * (quatf.x * quatf.z - quatf.y * quatf.w),
            2.f * (quatf.y * quatf.z + quatf.x * quatf.w),
            1.f - 2.f * (quatf.x * quatf.x + quatf.y * quatf.y));

        cube_pos = float3(
            dot(cube_pos, rot0),
            dot(cube_pos, rot1),
            dot(cube_pos, rot2)
        );

        cube_pos += posf;

        int3 ijk;
        ijk.x = int(floor(cube_pos.x * constants.voxel_size_inv));
        ijk.y = int(floor(cube_pos.y * constants.voxel_size_inv));
        ijk.z = int(floor(cube_pos.z * constants.voxel_size_inv));

        if (vert_idx == 0u)
        {
            ijk_min = ijk;
            ijk_max = ijk;
        }
        else
        {
            ijk_min = min(ijk_min, ijk);
            ijk_max = max(ijk_max, ijk);
        }
    }

    // clamp above certain size for now
    #if 0
    if (ijk_max.x - ijk_min.x > 16)
    {
        int ave = (ijk_max.x + ijk_min.x) / 2;
        ijk_max.x = ave + 8;
        ijk_min.x = ave - 8;
    }
    if (ijk_max.y - ijk_min.y > 16)
    {
        int ave = (ijk_max.y + ijk_min.y) / 2;
        ijk_max.y = ave + 8;
        ijk_min.y = ave - 8;
    }
    if (ijk_max.z - ijk_min.z > 16)
    {
        int ave = (ijk_max.z + ijk_min.z) / 2;
        ijk_max.z = ave + 8;
        ijk_min.z = ave - 8;
    }
    #endif

    int3 range = (ijk_max + 1 - ijk_min);
    uint footprint = range.x * range.y * range.z;

    fragment_counts_out[prim_idx] = uint64_t(footprint);
    prim_bbox_out[6u * prim_idx + 0u] = ijk_min.x;
    prim_bbox_out[6u * prim_idx + 1u] = ijk_min.y;
    prim_bbox_out[6u * prim_idx + 2u] = ijk_min.z;
    prim_bbox_out[6u * prim_idx + 3u] = ijk_max.x;
    prim_bbox_out[6u * prim_idx + 4u] = ijk_max.y;
    prim_bbox_out[6u * prim_idx + 5u] = ijk_max.z;
}
