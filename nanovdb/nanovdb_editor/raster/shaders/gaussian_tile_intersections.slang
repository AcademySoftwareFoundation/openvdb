// gaussian_tile_intersections.slang

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

StructuredBuffer<float2> means2d_in;
StructuredBuffer<int> radii_in;
StructuredBuffer<float> depths_in;
StructuredBuffer<uint> scan_tiles_per_gaussian_in;

RWStructuredBuffer<uint> intersection_keys_low_out;
RWStructuredBuffer<uint> intersection_keys_high_out;
RWStructuredBuffer<uint> intersection_vals_out;

[shader("compute")]
[numthreads(256, 1)]
void main(uint3 group_idx: SV_GroupID, uint3 thread_idx: SV_GroupThreadID)
{
    uint idx = group_idx.x * 256u + thread_idx.x;

    if (idx >= constants.prim_count)
    {
        return;
    }

    int radius = radii_in[idx];
    if (radius <= 0)
    {
        return;
    }
    float radiusf = float(radius);

    float2 mean2d = means2d_in[idx];
    float tile_radius = radiusf / float(constants.tile_size);
    float tile_mean_u = mean2d.x / float(constants.tile_size);
    float tile_mean_v = mean2d.y / float(constants.tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_mean_u - tile_radius)), constants.num_tiles_w);
    tile_min.y = min(max(0, (uint32_t)floor(tile_mean_v - tile_radius)), constants.num_tiles_h);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_mean_u + tile_radius)), constants.num_tiles_w);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_mean_v + tile_radius)), constants.num_tiles_h);

    float depth = depths_in[idx];

    uint2 key;
    key.x = asuint(depth);
    key.y = 0u;

    uint cur_isect = (idx == 0u) ? 0u : scan_tiles_per_gaussian_in[idx - 1u];
    for (int i = tile_min.y; i < tile_max.y; i++)
    {
        for (int j = tile_min.x; j < tile_max.x; j++)
        {
            uint tile_idx = i * constants.num_tiles_w + j;
            key.y = tile_idx;
            intersection_keys_low_out[cur_isect] = key.x;
            intersection_keys_high_out[cur_isect] = key.y;
            intersection_vals_out[cur_isect] = idx;
            cur_isect++;
        }
    }
}
