// gaussian_projection.slang

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

StructuredBuffer<float> means_in;
StructuredBuffer<float> quats_in;
StructuredBuffer<float> scales_in;

RWStructuredBuffer<int> radii_out;
RWStructuredBuffer<float2> means2d_out;
RWStructuredBuffer<float> depths_out;
RWStructuredBuffer<float> conics_out;
RWStructuredBuffer<float> compensations_out;

float3x3 quat_to_mat(float4 quatf)
{
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
    return float3x3(rot0, rot1, rot2);
}

float3x3 quat_and_scale_to_mat(float4 quatf, float3 scalef)
{
    float3x3 R = quat_to_mat(quatf);
    float3x3 S = float3x3(scalef.x, 0.f, 0.f, 0.f, scalef.y, 0.f, 0.f, 0.f, scalef.z);
    float3x3 M = mul(R, S);
    return mul(M, transpose(M));
}

float3x3 covar_world_to_cam(float3x3 R, float3x3 covar)
{
    return mul(mul(R, covar), transpose(R));
}

void persp_proj(
    float3 mean3d,
    float3x3 cov3d,
    float fx, float fy,
    float cx, float cy,
    uint width, uint height,
    inout float2x2 cov2d,
    inout float2 mean2d)
{
    float x = mean3d.x;
    float y = mean3d.y;
    float z = mean3d.z;

    // for right handed projection, flip z sign
    if (constants.near_plane < 0.f)
    {
        z = -z;
    }

    float tan_fovx = 0.5f * float(width) / fx;
    float tan_fovy = 0.5f * float(height) / fy;
    float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    float2x3 J = float2x3(
        fx * rz, 0.f, -fx * tx * rz2,
        0.f, fy * rz, -fy * ty * rz2
    );

    cov2d = mul(mul(J, cov3d), transpose(J));
    mean2d = float2(fx * x * rz + cx, fy * y * rz + cy);
}

float add_blur(float eps2d, inout float2x2 covar, inout float compensation)
{
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

float inverse(float2x2 M, inout float2x2 Minv)
{
    float det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (det <= 0.f) {
        return det;
    }
    float invDet = 1.f / det;
    Minv = float2x2(
        M[1][1] * invDet, -M[0][1] * invDet,
        -M[0][1] * invDet, M[0][0] * invDet);
    return det;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx: SV_GroupID, uint3 thread_idx: SV_GroupThreadID)
{
    uint global_prim_idx = group_idx.x * 256u + thread_idx.x;

    if (global_prim_idx >= constants.prim_count)
    {
        return;
    }

    float4 mean = float4(
        means_in[3u * global_prim_idx + 0u],
        means_in[3u * global_prim_idx + 1u],
        means_in[3u * global_prim_idx + 2u],
        1.f);
    float4 mean_c = mul(mean, constants.view);
    if (constants.near_plane > 0.f)
    {
        if (mean_c.z < constants.near_plane ||
            mean_c.z > constants.far_plane)
        {
            radii_out[global_prim_idx] = 0u;
            return;
        }
    }
    else
    {
        if (-mean_c.z < -constants.near_plane ||
            -mean_c.z > -constants.far_plane)
        {
            radii_out[global_prim_idx] = 0u;
            return;
        }
    }

    float4 quat = float4(
        quats_in[4u * global_prim_idx + 1u],
        quats_in[4u * global_prim_idx + 2u],
        quats_in[4u * global_prim_idx + 3u],
        quats_in[4u * global_prim_idx + 0u]);
    float3 scale = float3(
        scales_in[3u * global_prim_idx + 0u],
        scales_in[3u * global_prim_idx + 1u],
        scales_in[3u * global_prim_idx + 2u]);

// filtering of different shape types
#if 0
    float3 scale_abs = abs(scale);
    float scale_max = max(max(scale_abs.x, scale_abs.y), scale_abs.z);
    int scale_type = 0;
    scale_type += (scale_abs.x >= 0.5f * scale_max) ? 1 : 0;
    scale_type += (scale_abs.y >= 0.5f * scale_max) ? 1 : 0;
    scale_type += (scale_abs.z >= 0.5f * scale_max) ? 1 : 0;
    if (scale_type != 1)
    {
        radii_out[global_prim_idx] = 0u;
        return;
    }
#endif

    float3x3 covar = quat_and_scale_to_mat(quat, scale);

    float3x3 view_rot = float3x3(
        constants.view_rot0.x, constants.view_rot1.x, constants.view_rot2.x,
        constants.view_rot0.y, constants.view_rot1.y, constants.view_rot2.y,
        constants.view_rot0.z, constants.view_rot1.z, constants.view_rot2.z);
    float3x3 covar_c = covar_world_to_cam(view_rot, covar);

    float2x2 covar2d;
    float2 mean2d;
    persp_proj(
        mean_c.xyz, covar_c,
        constants.fx, constants.fy,
        constants.cx, constants.cy,
        constants.image_width, constants.image_height,
        covar2d,
        mean2d
    );

    float compensation;
    float det = add_blur(constants.eps2d, covar2d, compensation);
    if (det <= 0.f)
    {
        radii_out[global_prim_idx] = 0u;
        return;
    }

    float2x2 covar2d_inv;
    inverse(covar2d, covar2d_inv);

    float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    float v1 = b + sqrt(max(0.01f, b * b - det));
    float radius = ceil(3.f * sqrt(v1));

    if (radius <= constants.radius_clip)
    {
        radii_out[global_prim_idx] = 0u;
        return;
    }

    if (mean2d.x + radius <= 0 || mean2d.x - radius >= constants.image_width ||
        mean2d.y + radius <= 0 || mean2d.y - radius >= constants.image_height)
    {
        radii_out[global_prim_idx] = 0u;
        return;
    }

    radii_out[global_prim_idx] = int(radius);
    means2d_out[global_prim_idx] = mean2d;
    depths_out[global_prim_idx] = mean_c.z;
    conics_out[3u * global_prim_idx + 0u] = covar2d_inv[0][0];
    conics_out[3u * global_prim_idx + 1u] = covar2d_inv[0][1];
    conics_out[3u * global_prim_idx + 2u] = covar2d_inv[1][1];
    compensations_out[global_prim_idx] = compensation;
}
