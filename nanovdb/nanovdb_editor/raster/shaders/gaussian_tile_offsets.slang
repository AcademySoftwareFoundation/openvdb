// gaussian_tile_offsets.slang

struct constants_t
{
    float4x4 view;
    float4 view_rot0;
    float4 view_rot1;
    float4 view_rot2;

    float near_plane;
    float far_plane;
    float fx;
    float fy;

    float cx;
    float cy;
    uint image_width;
    uint image_height;

    float eps2d;
    float radius_clip;
    uint prim_count;
    uint n_isects;

    uint image_origin_w;
    uint image_origin_h;
    uint tile_origin_w;
    uint tile_origin_h;

    uint tile_size;
    uint tile_width;
    uint tile_height;
    uint num_tiles_w;

    float3 view_dir;
    uint num_tiles_h;

    uint num_tiles;
    uint pad1;
    uint pad2;
    uint pad3;
};

ConstantBuffer<constants_t> constants;

StructuredBuffer<uint> sorted_intersection_keys_high_in;

RWStructuredBuffer<uint> offsets_out;

[shader("compute")]
[numthreads(256, 1)]
void main(uint3 group_idx: SV_GroupID, uint3 thread_idx: SV_GroupThreadID)
{
    uint idx = group_idx.x * 256u + thread_idx.x;

    if (idx >= constants.n_isects)
    {
        return;
    }

    uint tile_start_idx = sorted_intersection_keys_high_in[idx];

    if (idx == 0)
    {
        // The first tile in the first camera writes out 0 as the offset
        // until the first valid tile (inclusive). i.e. tiles before this one
        // have no intersections, so their offset range is [0, 0]
        for (uint i = 0u; i < tile_start_idx + 1u; i++)
        {
            offsets_out[i] = 0;
        }
    }
    if (idx == constants.n_isects - 1u)
    {
        for (uint i = tile_start_idx + 1; i < constants.num_tiles; i++)
        {
            offsets_out[i] = constants.n_isects;
        }
    }

    if (idx > 0)
    {
        uint prev_tile_start_idx = sorted_intersection_keys_high_in[idx - 1];

        if (prev_tile_start_idx == tile_start_idx)
        {
            return;
        }

        for (uint i = prev_tile_start_idx + 1; i < tile_start_idx + 1u; i++)
        {
            offsets_out[i] = idx;
        }
    }
}
