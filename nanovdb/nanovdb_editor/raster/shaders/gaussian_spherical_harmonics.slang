// gaussian_rasterize_2d.slang

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

StructuredBuffer<float> spherical_harmonics_in;

RWStructuredBuffer<float> colors_out;

float3 read_sh(uint idx)
{
    return float3(
        spherical_harmonics_in[3u * idx + 0u],
        spherical_harmonics_in[3u * idx + 1u],
        spherical_harmonics_in[3u * idx + 2u]
    );
}

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation
float3 eval_sh_function(uint degree, uint sh_idx_begin, float3 viewDir)
{
    float3 cSH0 = read_sh(sh_idx_begin + 0u);

    float3 result = 0.2820947917738781f * cSH0;

    if (degree >= 1) {
        float inorm = rsqrt(viewDir.x * viewDir.x + viewDir.y * viewDir.y + viewDir.z * viewDir.z);
        float x = viewDir.x * inorm;
        float y = viewDir.y * inorm;
        float z = viewDir.z * inorm;

        float3 cSH1 = read_sh(sh_idx_begin + 1u);
        float3 cSH2 = read_sh(sh_idx_begin + 2u);
        float3 cSH3 = read_sh(sh_idx_begin + 3u);

        result += 0.48860251190292f * (-y * cSH1 + z * cSH2 - x * cSH3);

        if (degree >= 2) {
            float z2 = z * z;

            float fTmp0B = -1.092548430592079f * z;
            float fC1    = x * x - y * y;
            float fS1    = 2.f * x * y;
            float pSH6   = (0.9461746957575601f * z2 - 0.3153915652525201f);
            float pSH7   = fTmp0B * x;
            float pSH5   = fTmp0B * y;
            float pSH8   = 0.5462742152960395f * fC1;
            float pSH4   = 0.5462742152960395f * fS1;

            float3 cSH4 = read_sh(sh_idx_begin + 4u);
            float3 cSH5 = read_sh(sh_idx_begin + 5u);
            float3 cSH6 = read_sh(sh_idx_begin + 6u);
            float3 cSH7 = read_sh(sh_idx_begin + 7u);
            float3 cSH8 = read_sh(sh_idx_begin + 8u);

            result += (pSH4 * cSH4) + (pSH5 * cSH5) + (pSH6 * cSH6) + (pSH7 * cSH7) + (pSH8 * cSH8);

            if (degree >= 3) {
                float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                float fTmp1B = 1.445305721320277f * z;
                float fC2    = x * fC1 - y * fS1;
                float fS2    = x * fS1 + y * fC1;
                float pSH12  = z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH13  = fTmp0C * x;
                float pSH11  = fTmp0C * y;
                float pSH14  = fTmp1B * fC1;
                float pSH10  = fTmp1B * fS1;
                float pSH15  = -0.5900435899266435f * fC2;
                float pSH9   = -0.5900435899266435f * fS2;

                float3 cSH9  = read_sh(sh_idx_begin + 9u);
                float3 cSH10 = read_sh(sh_idx_begin + 10u);
                float3 cSH11 = read_sh(sh_idx_begin + 11u);
                float3 cSH12 = read_sh(sh_idx_begin + 12u);
                float3 cSH13 = read_sh(sh_idx_begin + 13u);
                float3 cSH14 = read_sh(sh_idx_begin + 14u);
                float3 cSH15 = read_sh(sh_idx_begin + 15u);

                result += (pSH9 * cSH9) + (pSH10 * cSH10) + (pSH11 * cSH11) + (pSH12 * cSH12) +
                          (pSH13 * cSH13) + (pSH14 * cSH14) + (pSH15 * cSH15);

                if (degree >= 4) {
                    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    float fTmp2B = -1.770130769779931f * z;
                    float fC3    = x * fC2 - y * fS2;
                    float fS3    = x * fS2 + y * fC2;
                    float pSH20  = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
                    float pSH21  = fTmp0D * x;
                    float pSH19  = fTmp0D * y;
                    float pSH22  = fTmp1C * fC1;
                    float pSH18  = fTmp1C * fS1;
                    float pSH23  = fTmp2B * fC2;
                    float pSH17  = fTmp2B * fS2;
                    float pSH24  = 0.6258357354491763f * fC3;
                    float pSH16  = 0.6258357354491763f * fS3;

                    float3 cSH16 = read_sh(sh_idx_begin + 16u);
                    float3 cSH17 = read_sh(sh_idx_begin + 17u);
                    float3 cSH18 = read_sh(sh_idx_begin + 18u);
                    float3 cSH19 = read_sh(sh_idx_begin + 19u);
                    float3 cSH20 = read_sh(sh_idx_begin + 20u);
                    float3 cSH21 = read_sh(sh_idx_begin + 21u);
                    float3 cSH22 = read_sh(sh_idx_begin + 22u);
                    float3 cSH23 = read_sh(sh_idx_begin + 23u);
                    float3 cSH24 = read_sh(sh_idx_begin + 24u);

                    result += (pSH16 * cSH16) + (pSH17 * cSH17) + (pSH18 * cSH18) +
                              (pSH19 * cSH19) + (pSH20 * cSH20) + (pSH21 * cSH21) +
                              (pSH22 * cSH22) + (pSH23 * cSH23) + (pSH24 * cSH24);
                }
            }
        }
    }

    return result + 0.5f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx: SV_GroupID, uint3 thread_idx: SV_GroupThreadID)
{
    uint idx = group_idx.x * 256u + thread_idx.x;

    float3 color = eval_sh_function(3, 16u * idx, constants.view_dir);

    colors_out[3u * idx + 0u] = color.r;
    colors_out[3u * idx + 1u] = color.g;
    colors_out[3u * idx + 2u] = color.b;
}
