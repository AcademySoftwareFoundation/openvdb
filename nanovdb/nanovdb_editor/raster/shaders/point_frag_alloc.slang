// point_frag_alloc.slang

struct constants_t
{
    uint point_count;
    float voxel_size;
    float voxel_size_inv;
    float pad3;
};

ConstantBuffer<constants_t> constants;

StructuredBuffer<uint> positions_in;

RWStructuredBuffer<uint> point_ijk_out;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint ijk_idx = group_idx.x * 256u + thread_idx.x;

    if (ijk_idx >= constants.point_count)
    {
        return;
    }

    float3 posf;
    posf.x = asfloat(positions_in[3u * ijk_idx + 0u]);
    posf.y = asfloat(positions_in[3u * ijk_idx + 1u]);
    posf.z = asfloat(positions_in[3u * ijk_idx + 2u]);

    int3 ijk;
    ijk.x = int(floor(posf.x * constants.voxel_size_inv));
    ijk.y = int(floor(posf.y * constants.voxel_size_inv));
    ijk.z = int(floor(posf.z * constants.voxel_size_inv));

    point_ijk_out[3u * ijk_idx + 0u] = ijk.x;
    point_ijk_out[3u * ijk_idx + 1u] = ijk.y;
    point_ijk_out[3u * ijk_idx + 2u] = ijk.z;
}
