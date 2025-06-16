// editor.slang
#define PNANOVDB_HLSL
#define PNANOVDB_ADDRESS_64
#define PNANOVDB_BUF_HLSL_64
#include "PNanoVDB.h"
#include "PNanoVDB2.h"

struct EditorParams
{
    float4 view_inv_x;
    float4 view_inv_y;
    float4 view_inv_z;
    float4 view_inv_w;
    float4 projection_inv_x;
    float4 projection_inv_y;
    float4 projection_inv_z;
    float4 projection_inv_w;
    uint width;
    uint height;
    uint narrow_band_only;
    float slice_plane_thickness;
    float4 slice_plane;
    uint highlight_bbox;
    uint pad1;
    uint pad2;
    uint pad3;
};

StructuredBuffer<uint64_t> buf;
RWStructuredBuffer<uint> image_out;
RWTexture2D<float4> texture_out;
ConstantBuffer<EditorParams> editor_params;

float3 cross_product(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

// ray origin is implied zero
bool intersect_box(float3 rayDir, float3 rayDirInv, float3 boxMin, float3 boxMax, out float tnear, out float tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 tbot = boxMin * rayDirInv;
    float3 ttop = boxMax * rayDirInv;

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = min(ttop, tbot);
    float3 tmax = max(ttop, tbot);

    // find the largest tmin and the smallest tmax
    tnear = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    tfar = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

    return tfar > tnear;
}

int3 ray_march_compute_final_location(float3 rayDir, int4 location, int4 locationMin, int4 locationMax)
{
    return int3(
        rayDir.x > 0.f ? max(location.x, locationMax.x) : min(location.x, locationMin.x - 1),
        rayDir.y > 0.f ? max(location.y, locationMax.y) : min(location.y, locationMin.y - 1),
        rayDir.z > 0.f ? max(location.z, locationMax.z) : min(location.z, locationMin.z - 1)
        );
}

void ray_march_advance_ray(
    float3 blockSizeWorld,
    float3 rayDir,
    float3 rayDirInv,
    float3 rayOrigin,
    inout int4 location,
    inout float hitT
)
{
    float hitTx = (float(location.x + (rayDir.x > 0.f ? +1 : 0)) * blockSizeWorld.x - rayOrigin.x) * rayDirInv.x;
    float hitTy = (float(location.y + (rayDir.y > 0.f ? +1 : 0)) * blockSizeWorld.y - rayOrigin.y) * rayDirInv.y;
    float hitTz = (float(location.z + (rayDir.z > 0.f ? +1 : 0)) * blockSizeWorld.z - rayOrigin.z) * rayDirInv.z;

    if (rayDir.x != 0.f && (hitTx <= hitTy || rayDir.y == 0.f) && (hitTx <= hitTz || rayDir.z == 0.f))
    {
        hitT = hitTx;
        location.x += rayDir.x > 0.f ? +1 : -1;
    }
    else if (rayDir.y != 0.f && (hitTy <= hitTx || rayDir.x == 0.f) && (hitTy <= hitTz || rayDir.z == 0.f))
    {
        hitT = hitTy;
        location.y += rayDir.y > 0.f ? +1 : -1;
    }
    else
    {
        hitT = hitTz;
        location.z += rayDir.z > 0.f ? +1 : -1;
    }
}

// source: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
uint ray_march_hash(uint inputValue)
{
    uint state = inputValue * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float ray_march_rand_norm(uint inputValue)
{
    return float(ray_march_hash(inputValue) & 0xFFFF) * float(1.f / 65535.f);
}

float ray_march_noise_from_dir(float3 rayDir)
{
    float2 uv;
    if (abs(rayDir.x) > abs(rayDir.y) && abs(rayDir.x) > abs(rayDir.z))
    {
        uv = rayDir.yz;
    }
    else if (abs(rayDir.y) > abs(rayDir.x) && abs(rayDir.y) > abs(rayDir.z))
    {
        uv = rayDir.xz;
    }
    else //if (abs(rayDir.z) > abs(rayDir.x) && abs(rayDir.z) > abs(rayDir.y))
    {
        uv = rayDir.xy;
    }
    float maxAxis = max(abs(rayDir.x), max(abs(rayDir.y), abs(rayDir.z)));
    if (maxAxis > 0.f)
    {
        uv *= (1.f / maxAxis);
    }
    uv = 0.5f * uv + 0.5f;
    uint hashInput = uint(65535.f * uv.x) ^ (uint(65535.f * uv.y) << 16u);
    return ray_march_rand_norm(hashInput);
}

float4 levelset_to_color(float value)
{
    // narrow band highlight
    float far_band_bias = 0.f;
    if (editor_params.narrow_band_only == 0u)
    {
        far_band_bias = 0.03f;
    }
    if (value >= 0.f)
    {
        value = max(0.f, 1.f - 4.f * value) + far_band_bias;
    }
    else
    {
        value = min(0.f, -1.f - 4.f * value) - far_band_bias;
    }
    float4 color = float4(0.1f, 0.1f, 0.8f, value);
    if (color.a < 0.f)
    {
        color = float4(0.8f, 0.1f, 0.1f, -color.a);
    }
    return color;
}

float4 ray_march_nanovdb_leaf_fetch_float(
    pnanovdb_grid_type_t grid_type,
    StructuredBuffer<uint64_t> buf,
    inout pnanovdb_readaccessor_t acc,
    int3 ijk
)
{
    pnanovdb_grid_handle_t grid = { pnanovdb_address_null() };
    pnanovdb_uint32_t level = 0u;
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address_and_level(grid_type, buf, PNANOVDB_REF(acc), ijk, level);
    float4 color = float4(0.f, 0.f, 0.f, 0.f);
    if (grid_type == PNANOVDB_GRID_TYPE_ONINDEX)
    {
        pnanovdb_gridblindmetadata_handle_t metadata = pnanovdb_grid_get_gridblindmetadata(buf, grid, 0u);
        pnanovdb_grid_type_t blind_type = pnanovdb_gridblindmetadata_get_data_type(buf, metadata);
        if (blind_type == PNANOVDB_GRID_TYPE_RGBA8)
        {
            pnanovdb_uint64_t val_index64;
            if (level == 0)
            {
                val_index64 = pnanovdb_leaf_onindex_get_value_index(buf, address, ijk);
            }
            else
            {
                val_index64 = pnanovdb_read_uint64(buf, address);
            }
            pnanovdb_uint32_t val_index = pnanovdb_uint64_low(val_index64);
            pnanovdb_uint64_t value_count = pnanovdb_gridblindmetadata_get_value_count(buf, metadata);
            if (val_index < pnanovdb_uint64_low(value_count))
            {
                pnanovdb_int64_t byte_offset = pnanovdb_gridblindmetadata_get_data_offset(buf, metadata);
                pnanovdb_address_t val_addr = pnanovdb_address_offset64(metadata.address, pnanovdb_int64_as_uint64(byte_offset));
                val_addr = pnanovdb_address_offset_product(val_addr, val_index, 4u);
                uint value_raw = pnanovdb_read_uint32(buf, val_addr);
                color = float4(
                    float((value_raw >> 0) & 255) * (1.f / 255.f),
                    float((value_raw >> 8) & 255) * (1.f / 255.f),
                    float((value_raw >> 16) & 255) * (1.f / 255.f),
                    float((value_raw >> 24) & 255) * (1.f / 255.f)
                    );
            }
        }
        else
        {
            float value = 0.f;
            pnanovdb_uint64_t val_index64;
            if (level == 0)
            {
                val_index64 = pnanovdb_leaf_onindex_get_value_index(buf, address, ijk);
            }
            else
            {
                val_index64 = pnanovdb_read_uint64(buf, address);
            }
            pnanovdb_uint32_t val_index = pnanovdb_uint64_low(val_index64);
            pnanovdb_gridblindmetadata_handle_t meta = pnanovdb_grid_get_gridblindmetadata(buf, grid, 0u);
            pnanovdb_uint64_t value_count = pnanovdb_gridblindmetadata_get_value_count(buf, meta);
            if (val_index < pnanovdb_uint64_low(value_count))
            {
                pnanovdb_int64_t byte_offset = pnanovdb_gridblindmetadata_get_data_offset(buf, meta);
                pnanovdb_address_t val_addr = pnanovdb_address_offset64(meta.address, pnanovdb_int64_as_uint64(byte_offset));
                val_addr = pnanovdb_address_offset_product(val_addr, val_index, 4u);
                value = pnanovdb_read_float(buf, val_addr);
            }
            color = levelset_to_color(value);
        }
    }
    else
    {
        float value = 0.f;
        if (grid_type == PNANOVDB_GRID_TYPE_FLOAT)
        {
            value = pnanovdb_read_float(buf, address);
        }
        else if (!pnanovdb_address_is_null(PNANOVDB_DEREF(acc).leaf.address))
        {
            pnanovdb_uint32_t leaf_n = pnanovdb_leaf_coord_to_offset(PNANOVDB_REF(ijk));
            if (pnanovdb_leaf_get_value_mask(buf, PNANOVDB_DEREF(acc).leaf, leaf_n))
            {
                value = 1.f;
            }
        }
        color = levelset_to_color(value);
    }
    return color;
}

float4 ray_march_nanovdb_leaf_fetch_float_node2(
    pnanovdb_grid_type_t blind_type,
    StructuredBuffer<uint64_t> buf,
    inout pnanovdb_node2_accessor_t acc,
    pnanovdb_node2_levelset_values_t values,
    int3 ijk
)
{
    pnanovdb_node2_handle_t node = {};
    pnanovdb_uint32_t node_type = 0u;
    pnanovdb_uint32_t node_n = 0u;
    pnanovdb_uint32_t level = 0u;
    pnanovdb_node2_accessor_find_node(buf, acc, node, node_type, node_n, level, ijk);
    pnanovdb_uint64_t value_idx = pnanovdb_node2_get_value_index(buf, node, node_type, node_n, PNANOVDB_TRUE, values.node_inactive_idxs, values.inactive_value_idxs);
    pnanovdb_address_t val_addr = pnanovdb_address_offset64_product(values.values, value_idx, 4u);
    float4 color = float4(0.f, 0.f, 0.f, 0.f);
    if (blind_type == PNANOVDB_GRID_TYPE_RGBA8)
    {
        uint value_raw = pnanovdb_read_uint32(buf, val_addr);
        color = float4(
            float((value_raw >> 0) & 255) * (1.f / 255.f),
            float((value_raw >> 8) & 255) * (1.f / 255.f),
            float((value_raw >> 16) & 255) * (1.f / 255.f),
            float((value_raw >> 24) & 255) * (1.f / 255.f)
        );
    }
    else
    {
        float value = pnanovdb_read_float(buf, val_addr);
        color = levelset_to_color(value);
    }
    return color;
}

bool ray_march_nanovdb_leaf(
    pnanovdb_grid_type_t grid_type,
    StructuredBuffer<uint64_t> buf,
    float3 rayOrigin,
    float rayMinT,
    float3 rayDir,
    float rayMaxT,
    float3 rayDirInv,
    float rayNoise,
    int4 location,
    inout pnanovdb_readaccessor_t acc,
    inout float4 sum,
    inout float nominalT
)
{
    float3 boxMin = float3(location.xyz) * 8.f;
    float3 boxMax = float3(location.xyz + int3(1, 1, 1)) * 8.f;

    const float ep = 0.0001f;

    boxMin = (boxMin - rayOrigin) - ep;
    boxMax = (boxMax - rayOrigin) + ep;

    float boxMinT;
    float boxMaxT;
    bool isHit = intersect_box(rayDir, rayDirInv, boxMin, boxMax, boxMinT, boxMaxT);

    boxMinT = max(rayMinT, boxMinT);
    if (boxMinT > boxMaxT)
    {
        isHit = false;
    }

    bool hitMax = false;
    if (boxMaxT > rayMaxT)
    {
        boxMaxT = rayMaxT;
        hitMax = true;
    }

    if (isHit)
    {
        const float stepSize = 0.75f;
        const float stepSizeInv = 1.f / 0.75f;
        const float alphaScale = 0.1f;

        float cellMinT = stepSizeInv * boxMinT;
        float cellMaxT = stepSizeInv * boxMaxT;

        cellMinT = -floor(-(cellMinT + rayNoise)) - rayNoise;
        cellMaxT = -floor(-(cellMaxT + rayNoise)) - rayNoise;

        int numSteps = int(cellMaxT - cellMinT);

        float currentT = stepSize * cellMinT;

        float3 pos = rayOrigin + currentT * rayDir;
        float3 posStep = stepSize * rayDir;

        for (int stepIdx = 0; stepIdx < numSteps; stepIdx++)
        {
            // linear interpolation
#if 1
            int3 ijk000 = int3(floor(pos - 0.5f));
            float3 pw = (pos - 0.5f) - float3(ijk000);
            float3 nw = float3(1.f, 1.f, 1.f) - pw;

            float4 value;
            value =  nw.x * nw.y * nw.z * ray_march_nanovdb_leaf_fetch_float(grid_type, buf, acc, ijk000 + int3(0, 0, 0));
            value += pw.x * nw.y * nw.z * ray_march_nanovdb_leaf_fetch_float(grid_type, buf, acc, ijk000 + int3(1, 0, 0));
            value += nw.x * pw.y * nw.z * ray_march_nanovdb_leaf_fetch_float(grid_type, buf, acc, ijk000 + int3(0, 1, 0));
            value += pw.x * pw.y * nw.z * ray_march_nanovdb_leaf_fetch_float(grid_type, buf, acc, ijk000 + int3(1, 1, 0));
            value += nw.x * nw.y * pw.z * ray_march_nanovdb_leaf_fetch_float(grid_type, buf, acc, ijk000 + int3(0, 0, 1));
            value += pw.x * nw.y * pw.z * ray_march_nanovdb_leaf_fetch_float(grid_type, buf, acc, ijk000 + int3(1, 0, 1));
            value += nw.x * pw.y * pw.z * ray_march_nanovdb_leaf_fetch_float(grid_type, buf, acc, ijk000 + int3(0, 1, 1));
            value += pw.x * pw.y * pw.z * ray_march_nanovdb_leaf_fetch_float(grid_type, buf, acc, ijk000 + int3(1, 1, 1));
#else
            int3 ijk000 = int3(floor(pos));

            float4 value;
            value = ray_march_nanovdb_leaf_fetch_float(grid_type, buf, acc, ijk000);
#endif

            float4 color = value;

            if (editor_params.slice_plane_thickness != 0.f)
            {
                float plane_dist = dot(float4(pos, 1.f), editor_params.slice_plane);
                bool is_inside = abs(plane_dist) > abs(0.5f * editor_params.slice_plane_thickness);
                if (editor_params.slice_plane_thickness > 0.f && is_inside)
                {
                    color = float4(0.f, 0.f, 0.f, 0.f);
                }
                else if (editor_params.slice_plane_thickness < 0.f && !is_inside)
                {
                    color = float4(0.f, 0.f, 0.f, 0.f);
                }
            }

            color = max(float4(0.f, 0.f, 0.f, 0.f), color);
            color.a = min(1.f, color.a);

            color.a *= alphaScale;

            nominalT = sum.a * (color.a * currentT) + nominalT;

            sum.rgb = sum.a * (color.a * color.rgb) + sum.rgb;
            sum.a = (1.f - color.a) * sum.a;

            pos += posStep;
            currentT += stepSize;
        }

        if (sum.a < 0.00005f)
        {
            hitMax = true;
        }

    }
    return hitMax;
}

bool ray_march_nanovdb_leaf_node2(
    pnanovdb_grid_type_t blind_type,
    StructuredBuffer<uint64_t> buf,
    float3 rayOrigin,
    float rayMinT,
    float3 rayDir,
    float rayMaxT,
    float3 rayDirInv,
    float rayNoise,
    int4 location,
    inout pnanovdb_node2_accessor_t acc,
    pnanovdb_node2_levelset_values_t values,
    inout float4 sum,
    inout float nominalT
)
{
    float3 boxMin = float3(location.xyz) * 8.f;
    float3 boxMax = float3(location.xyz + int3(1, 1, 1)) * 8.f;

    const float ep = 0.0001f;

    boxMin = (boxMin - rayOrigin) - ep;
    boxMax = (boxMax - rayOrigin) + ep;

    float boxMinT;
    float boxMaxT;
    bool isHit = intersect_box(rayDir, rayDirInv, boxMin, boxMax, boxMinT, boxMaxT);

    boxMinT = max(rayMinT, boxMinT);
    if (boxMinT > boxMaxT)
    {
        isHit = false;
    }

    bool hitMax = false;
    if (boxMaxT > rayMaxT)
    {
        boxMaxT = rayMaxT;
        hitMax = true;
    }

    if (isHit)
    {
        const float stepSize = 0.75f;
        const float stepSizeInv = 1.f / 0.75f;
        const float alphaScale = 0.1f;

        float cellMinT = stepSizeInv * boxMinT;
        float cellMaxT = stepSizeInv * boxMaxT;

        cellMinT = -floor(-(cellMinT + rayNoise)) - rayNoise;
        cellMaxT = -floor(-(cellMaxT + rayNoise)) - rayNoise;

        int numSteps = int(cellMaxT - cellMinT);

        float currentT = stepSize * cellMinT;

        float3 pos = rayOrigin + currentT * rayDir;
        float3 posStep = stepSize * rayDir;

        for (int stepIdx = 0; stepIdx < numSteps; stepIdx++)
        {
            // linear interpolation
#if 1
            int3 ijk000 = int3(floor(pos - 0.5f));
            float3 pw = (pos - 0.5f) - float3(ijk000);
            float3 nw = float3(1.f, 1.f, 1.f) - pw;

            float4 value;
            value = nw.x * nw.y * nw.z * ray_march_nanovdb_leaf_fetch_float_node2(blind_type, buf, acc, values, ijk000 + int3(0, 0, 0));
            value += pw.x * nw.y * nw.z * ray_march_nanovdb_leaf_fetch_float_node2(blind_type, buf, acc, values, ijk000 + int3(1, 0, 0));
            value += nw.x * pw.y * nw.z * ray_march_nanovdb_leaf_fetch_float_node2(blind_type, buf, acc, values, ijk000 + int3(0, 1, 0));
            value += pw.x * pw.y * nw.z * ray_march_nanovdb_leaf_fetch_float_node2(blind_type, buf, acc, values, ijk000 + int3(1, 1, 0));
            value += nw.x * nw.y * pw.z * ray_march_nanovdb_leaf_fetch_float_node2(blind_type, buf, acc, values, ijk000 + int3(0, 0, 1));
            value += pw.x * nw.y * pw.z * ray_march_nanovdb_leaf_fetch_float_node2(blind_type, buf, acc, values, ijk000 + int3(1, 0, 1));
            value += nw.x * pw.y * pw.z * ray_march_nanovdb_leaf_fetch_float_node2(blind_type, buf, acc, values, ijk000 + int3(0, 1, 1));
            value += pw.x * pw.y * pw.z * ray_march_nanovdb_leaf_fetch_float_node2(blind_type, buf, acc, values, ijk000 + int3(1, 1, 1));
#else
            int3 ijk000 = int3(floor(pos));

            float4 value;
            value = ray_march_nanovdb_leaf_fetch_float_node2(blind_type, buf, acc, values, ijk000);
#endif

            float4 color = value;

            if (editor_params.slice_plane_thickness != 0.f)
            {
                float plane_dist = dot(float4(pos, 1.f), editor_params.slice_plane);
                bool is_inside = abs(plane_dist) > abs(0.5f * editor_params.slice_plane_thickness);
                if (editor_params.slice_plane_thickness > 0.f && is_inside)
                {
                    color = float4(0.f, 0.f, 0.f, 0.f);
                }
                else if (editor_params.slice_plane_thickness < 0.f && !is_inside)
                {
                    color = float4(0.f, 0.f, 0.f, 0.f);
                }
            }

            color = max(float4(0.f, 0.f, 0.f, 0.f), color);
            color.a = min(1.f, color.a);

            color.a *= alphaScale;

            nominalT = sum.a * (color.a * currentT) + nominalT;

            sum.rgb = sum.a * (color.a * color.rgb) + sum.rgb;
            sum.a = (1.f - color.a) * sum.a;

            pos += posStep;
            currentT += stepSize;
        }

        if (sum.a < 0.00005f)
        {
            hitMax = true;
        }
    }
    return hitMax;
}

bool ray_march_nanovdb_tile(
    pnanovdb_grid_type_t grid_type,
    StructuredBuffer<uint64_t> buf,
    float3 rayOrigin,
    float rayMinT,
    float3 rayDir,
    float rayMaxT,
    float3 rayDirInv,
    float rayNoise,
    int4 location,
    inout pnanovdb_readaccessor_t acc,
    inout float4 sum,
    inout float nominalT
)
{
    float3 boxMin = float3(location.xyz) * 8.f;
    float3 boxMax = float3(location.xyz + int3(1, 1, 1)) * 8.f;

    const float ep = 0.0001f;

    boxMin = (boxMin - rayOrigin) - ep;
    boxMax = (boxMax - rayOrigin) + ep;

    float boxMinT;
    float boxMaxT;
    bool isHit = intersect_box(rayDir, rayDirInv, boxMin, boxMax, boxMinT, boxMaxT);

    boxMinT = max(rayMinT, boxMinT);
    if (boxMinT > boxMaxT)
    {
        isHit = false;
    }

    bool hitMax = false;
    if (boxMaxT > rayMaxT)
    {
        boxMaxT = rayMaxT;
        hitMax = true;
    }

    if (isHit)
    {
        const float stepSize = 0.75f;
        const float stepSizeInv = 1.f / 0.75f;
        const float alphaScale = 0.1f;

        float cellMinT = stepSizeInv * boxMinT;
        float cellMaxT = stepSizeInv * boxMaxT;

        cellMinT = -floor(-(cellMinT + rayNoise)) - rayNoise;
        cellMaxT = -floor(-(cellMaxT + rayNoise)) - rayNoise;

        int numSteps = int(cellMaxT - cellMinT);

        float currentT = stepSize * cellMinT;

        float3 pos = rayOrigin + currentT * rayDir;
        float3 posStep = stepSize * rayDir;

        // capture value once and reuse many times
        float4 value;
        value = ray_march_nanovdb_leaf_fetch_float(grid_type, buf, acc, location.xyz * 8u);

        for (int stepIdx = 0; stepIdx < numSteps; stepIdx++)
        {
            float4 color = value;

            if (editor_params.slice_plane_thickness != 0.f)
            {
                float plane_dist = dot(float4(pos, 1.f), editor_params.slice_plane);
                bool is_inside = abs(plane_dist) > abs(0.5f * editor_params.slice_plane_thickness);
                if (editor_params.slice_plane_thickness > 0.f && is_inside)
                {
                    color = float4(0.f, 0.f, 0.f, 0.f);
                }
                else if (editor_params.slice_plane_thickness < 0.f && !is_inside)
                {
                    color = float4(0.f, 0.f, 0.f, 0.f);
                }
            }

            color = max(float4(0.f, 0.f, 0.f, 0.f), color);
            color.a = min(1.f, color.a);

            color.a *= alphaScale;

            nominalT = sum.a * (color.a * currentT) + nominalT;

            sum.rgb = sum.a * (color.a * color.rgb) + sum.rgb;
            sum.a = (1.f - color.a) * sum.a;

            pos += posStep;
            currentT += stepSize;
        }

        if (sum.a < 0.00005f)
        {
            hitMax = true;
        }

    }
    return hitMax;
}

bool ray_march_nanovdb_tile_node2(
    pnanovdb_grid_type_t blind_type,
    StructuredBuffer<uint64_t> buf,
    float3 rayOrigin,
    float rayMinT,
    float3 rayDir,
    float rayMaxT,
    float3 rayDirInv,
    float rayNoise,
    int4 location,
    inout pnanovdb_node2_accessor_t acc,
    pnanovdb_node2_levelset_values_t values,
    inout float4 sum,
    inout float nominalT
)
{
    float3 boxMin = float3(location.xyz) * 8.f;
    float3 boxMax = float3(location.xyz + int3(1, 1, 1)) * 8.f;

    const float ep = 0.0001f;

    boxMin = (boxMin - rayOrigin) - ep;
    boxMax = (boxMax - rayOrigin) + ep;

    float boxMinT;
    float boxMaxT;
    bool isHit = intersect_box(rayDir, rayDirInv, boxMin, boxMax, boxMinT, boxMaxT);

    boxMinT = max(rayMinT, boxMinT);
    if (boxMinT > boxMaxT)
    {
        isHit = false;
    }

    bool hitMax = false;
    if (boxMaxT > rayMaxT)
    {
        boxMaxT = rayMaxT;
        hitMax = true;
    }

    if (isHit)
    {
        const float stepSize = 0.75f;
        const float stepSizeInv = 1.f / 0.75f;
        const float alphaScale = 0.1f;

        float cellMinT = stepSizeInv * boxMinT;
        float cellMaxT = stepSizeInv * boxMaxT;

        cellMinT = -floor(-(cellMinT + rayNoise)) - rayNoise;
        cellMaxT = -floor(-(cellMaxT + rayNoise)) - rayNoise;

        int numSteps = int(cellMaxT - cellMinT);

        float currentT = stepSize * cellMinT;

        float3 pos = rayOrigin + currentT * rayDir;
        float3 posStep = stepSize * rayDir;

        // capture value once and reuse many times
        float4 value;
        value = ray_march_nanovdb_leaf_fetch_float_node2(blind_type, buf, acc, values, location.xyz * 8u);

        for (int stepIdx = 0; stepIdx < numSteps; stepIdx++)
        {
            float4 color = value;

            if (editor_params.slice_plane_thickness != 0.f)
            {
                float plane_dist = dot(float4(pos, 1.f), editor_params.slice_plane);
                bool is_inside = abs(plane_dist) > abs(0.5f * editor_params.slice_plane_thickness);
                if (editor_params.slice_plane_thickness > 0.f && is_inside)
                {
                    color = float4(0.f, 0.f, 0.f, 0.f);
                }
                else if (editor_params.slice_plane_thickness < 0.f && !is_inside)
                {
                    color = float4(0.f, 0.f, 0.f, 0.f);
                }
            }

            color = max(float4(0.f, 0.f, 0.f, 0.f), color);
            color.a = min(1.f, color.a);

            color.a *= alphaScale;

            nominalT = sum.a * (color.a * currentT) + nominalT;

            sum.rgb = sum.a * (color.a * color.rgb) + sum.rgb;
            sum.a = (1.f - color.a) * sum.a;

            pos += posStep;
            currentT += stepSize;
        }

        if (sum.a < 0.00005f)
        {
            hitMax = true;
        }
    }
    return hitMax;
}

void ray_march_nanovdb(
    StructuredBuffer<uint64_t> buf,
    float3 worldRayOrigin,
    float rayMinT,
    float3 worldRayDir,
    float rayMaxT,
    float3 worldRayDirInv,
    inout float4 sum,
    inout float nominalT
)
{
    pnanovdb_grid_handle_t grid = { pnanovdb_address_null() };
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
    pnanovdb_grid_type_t grid_type = pnanovdb_grid_get_grid_type(buf, grid);

    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(PNANOVDB_REF(acc), root);

    // transform ray from world to index space
    float3 rayOrigin = pnanovdb_grid_world_to_indexf(buf, grid, worldRayOrigin);
    float3 rayDir = pnanovdb_grid_world_to_index_dirf(buf, grid, worldRayDir);
    float rayDirMagn = length(rayDir);
    if (rayDirMagn > 0.f)
    {
        rayDir /= rayDirMagn;
        rayMinT *= rayDirMagn;
        rayMaxT *= rayDirMagn;
    }
    float3 rayDirInv = float3(1.f, 1.f, 1.f) / rayDir;

    // intersect local ray with local bbox
    int3 bbox_min = pnanovdb_root_get_bbox_min(buf, root);
    int3 bbox_max = pnanovdb_root_get_bbox_max(buf, root);
    float3 bbox_minf = float3(bbox_min);
    float3 bbox_maxf = float3(bbox_max + int3(1,1,1));

    float boxMinT;
    float boxMaxT;
    bool isHit = intersect_box(rayDir, rayDirInv, bbox_minf - rayOrigin, bbox_maxf - rayOrigin, boxMinT, boxMaxT);

    boxMinT = max(rayMinT, boxMinT);
    if (boxMinT > boxMaxT)
    {
        isHit = false;
    }

    if (isHit)
    {
        float3 rayLocation = rayDir * boxMinT + rayOrigin;
        int4 location = int4(int3(floor(rayLocation * (1.f / 8.f))), 0);

        int3 finalLocation = ray_march_compute_final_location(
            rayDir, location, int4(int3(bbox_min >> 3u), 0), int4(int3(bbox_max >> 3u) + int3(1,1,1), 0));

        float rayNoise = ray_march_noise_from_dir(rayDir);

        bool hitMax = false;
        float blockHitT = boxMinT;

        while (
            location.x != finalLocation.x &&
            location.y != finalLocation.y &&
            location.z != finalLocation.z &&
            !hitMax)
        {
            if (editor_params.highlight_bbox != 0u)
            {
                sum.g = max(0.1f, sum.g);
            }

            pnanovdb_readaccessor_get_value_address(grid_type, buf, PNANOVDB_REF(acc), int3(location.xyz << 3u));
            // disable check for now, until specialized tile value support is added.
            if (!pnanovdb_address_is_null(acc.leaf.address))
            {
                hitMax = ray_march_nanovdb_leaf(
                    grid_type,
                    buf,
                    rayOrigin,
                    rayMinT,
                    rayDir,
                    rayMaxT,
                    rayDirInv,
                    rayNoise,
                    location,
                    acc,
                    sum,
                    nominalT
                );
            }
            else if (editor_params.narrow_band_only == 0u)
            {
                hitMax = ray_march_nanovdb_tile(
                    grid_type,
                    buf,
                    rayOrigin,
                    rayMinT,
                    rayDir,
                    rayMaxT,
                    rayDirInv,
                    rayNoise,
                    location,
                    acc,
                    sum,
                    nominalT
                );
            }

            ray_march_advance_ray(
                float3(8.f, 8.f, 8.f),
                rayDir,
                rayDirInv,
                rayOrigin,
                location,
                blockHitT
            );
        }
    }
}

void ray_march_nanovdb_node2_intern(
    pnanovdb_grid_type_t blind_type,
    StructuredBuffer<uint64_t> buf,
    float3 worldRayOrigin,
    float rayMinT,
    float3 worldRayDir,
    float rayMaxT,
    float3 worldRayDirInv,
    pnanovdb_grid_handle_t grid,
    pnanovdb_node2_handle_t root,
    pnanovdb_node2_levelset_values_t values,
    inout float4 sum,
    inout float nominalT
)
{
    pnanovdb_node2_accessor_t acc;
    pnanovdb_node2_accessor_init(PNANOVDB_REF(acc), root);

    // transform ray from world to index space
    float3 rayOrigin = pnanovdb_grid_world_to_indexf(buf, grid, worldRayOrigin);
    float3 rayDir = pnanovdb_grid_world_to_index_dirf(buf, grid, worldRayDir);
    float rayDirMagn = length(rayDir);
    if (rayDirMagn > 0.f)
    {
        rayDir /= rayDirMagn;
        rayMinT *= rayDirMagn;
        rayMaxT *= rayDirMagn;
    }
    float3 rayDirInv = float3(1.f, 1.f, 1.f) / rayDir;

    // intersect local ray with local bbox
    pnanovdb_address_t bboxes = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 0u);
    pnanovdb_coord_t bbox_min = pnanovdb_read_coord(buf, pnanovdb_address_offset(bboxes, 0u));
    pnanovdb_coord_t bbox_max = pnanovdb_read_coord(buf, pnanovdb_address_offset(bboxes, 12u));
    float3 bbox_minf = float3(bbox_min);
    float3 bbox_maxf = float3(bbox_max + int3(1, 1, 1));

    float boxMinT;
    float boxMaxT;
    bool isHit = intersect_box(rayDir, rayDirInv, bbox_minf - rayOrigin, bbox_maxf - rayOrigin, boxMinT, boxMaxT);

    boxMinT = max(rayMinT, boxMinT);
    if (boxMinT > boxMaxT)
    {
        isHit = false;
    }

    if (isHit)
    {
        float3 rayLocation = rayDir * boxMinT + rayOrigin;
        int4 location = int4(int3(floor(rayLocation * (1.f / 8.f))), 0);

        int3 finalLocation = ray_march_compute_final_location(
            rayDir, location, int4(int3(bbox_min >> 3u), 0), int4(int3(bbox_max >> 3u) + int3(1, 1, 1), 0));

        float rayNoise = ray_march_noise_from_dir(rayDir);

        bool hitMax = false;
        float blockHitT = boxMinT;

        while (
            location.x != finalLocation.x &&
            location.y != finalLocation.y &&
            location.z != finalLocation.z &&
            !hitMax)
        {
            if (editor_params.highlight_bbox != 0u)
            {
                sum.g = max(0.1f, sum.g);
            }

            pnanovdb_coord_t leaf_ijk = int3(location.xyz << 3u);
            pnanovdb_node2_handle_t node = {};
            pnanovdb_uint32_t node_type = 0u;
            pnanovdb_uint32_t node_n = 0u;
            pnanovdb_uint32_t level = 0u;
            pnanovdb_node2_accessor_find_node(buf, acc, node, node_type, node_n, level, leaf_ijk);
            if (node_type == PNANOVDB_NODE2_TYPE_LEAF)
            {
                hitMax = ray_march_nanovdb_leaf_node2(
                    blind_type,
                    buf,
                    rayOrigin,
                    rayMinT,
                    rayDir,
                    rayMaxT,
                    rayDirInv,
                    rayNoise,
                    location,
                    acc,
                    values,
                    sum,
                    nominalT
                );
            }
            else if (editor_params.narrow_band_only == 0u)
            {
                hitMax = ray_march_nanovdb_tile_node2(
                    blind_type,
                    buf,
                    rayOrigin,
                    rayMinT,
                    rayDir,
                    rayMaxT,
                    rayDirInv,
                    rayNoise,
                    location,
                    acc,
                    values,
                    sum,
                    nominalT
                );
            }

            ray_march_advance_ray(
                float3(8.f, 8.f, 8.f),
                rayDir,
                rayDirInv,
                rayOrigin,
                location,
                blockHitT
            );
        }
    }
}

void ray_march_nanovdb_node2(
    StructuredBuffer<uint64_t> buf,
    float3 worldRayOrigin,
    float rayMinT,
    float3 worldRayDir,
    float rayMaxT,
    float3 worldRayDirInv,
    inout float4 sum,
    inout float nominalT
)
{
    pnanovdb_grid_handle_t grid = { pnanovdb_address_null() };
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_node2_handle_t root = { pnanovdb_tree_get_root(buf, tree).address.byte_offset >> 3u };
    pnanovdb_node2_levelset_values_t values = {};
    values.values = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 1u);
    values.node_inactive_idxs = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 2u);
    values.inactive_value_idxs = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 3u);

    pnanovdb_gridblindmetadata_handle_t metadata = pnanovdb_grid_get_gridblindmetadata(buf, grid, 1u);
    pnanovdb_grid_type_t blind_type = pnanovdb_gridblindmetadata_get_data_type(buf, metadata);

    ray_march_nanovdb_node2_intern(
        blind_type,
        buf,
        worldRayOrigin,
        rayMinT,
        worldRayDir,
        rayMaxT,
        worldRayDirInv,
        grid,
        root,
        values,
        sum,
        nominalT);
}

float4 mat_mul(float4 v, float4 x, float4 y, float4 z, float4 w)
{
    return float4(dot(v, x), dot(v, y), dot(v, z), dot(v, w));
}

[shader("compute")]
[numthreads(32, 4, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 tidx = int2(dispatchThreadID.xy);

    float2 ndc = float2(
        2.f * ((float(tidx.x) + 0.5f) / float(editor_params.width)) - 1.f,
        -2.f * ((float(tidx.y) + 0.5f) / float(editor_params.height)) + 1.f
    );

    float4 pos_d0 = mat_mul(float4(ndc.xy, 0.f, 1.f),
        editor_params.projection_inv_x, editor_params.projection_inv_y,
        editor_params.projection_inv_z, editor_params.projection_inv_w);
    float4 pos_d1 = mat_mul(float4(ndc.xy, 1.f, 1.f),
        editor_params.projection_inv_x, editor_params.projection_inv_y,
        editor_params.projection_inv_z, editor_params.projection_inv_w);

    float z_d0 = pos_d0.z * (1.f / pos_d0.w);
    float z_d1 = pos_d1.z * (1.f / pos_d1.w);
    bool is_reverse_z = abs(z_d0) > abs(z_d1);
    float4 ray_dir_near = is_reverse_z ? pos_d1 : pos_d0;

    float4 ray_dir_far = ray_dir_near + mat_mul(float4(0.f, 0.f, 1.f, 0.f),
        editor_params.projection_inv_x, editor_params.projection_inv_y,
        editor_params.projection_inv_z, editor_params.projection_inv_w);
    float3 rayDir = normalize((ray_dir_far.xyz / ray_dir_far.w) - (ray_dir_near.xyz / ray_dir_near.w));
    if (is_reverse_z)
    {
        rayDir = -rayDir;
    }

    rayDir = mat_mul(float4(rayDir, 0.f),
        editor_params.view_inv_x, editor_params.view_inv_y,
        editor_params.view_inv_z, editor_params.view_inv_w).xyz;
    float3 rayDirInv = float3(1.f, 1.f, 1.f) / rayDir;

    float4 rayOrigin4 = is_reverse_z ? pos_d1 : pos_d0;
    rayOrigin4 = mat_mul(rayOrigin4,
        editor_params.view_inv_x, editor_params.view_inv_y,
        editor_params.view_inv_z, editor_params.view_inv_w);
    float3 rayOrigin = rayOrigin4.xyz / rayOrigin4.w;

    pnanovdb_grid_handle_t grid = { pnanovdb_address_null() };
    pnanovdb_uint32_t grid_type = pnanovdb_grid_get_grid_type(buf, grid);

    float4 sum = float4(0.f, 0.f, 0.f, 1.f);
    float nominalT = 0.f;
    if (grid_type == PNANOVDB_GRID_TYPE_NODE2)
    {
        ray_march_nanovdb_node2(
            buf,
            rayOrigin,
            0.f,
            rayDir,
            1e9f,
            rayDirInv,
            sum,
            nominalT
        );
    }
    else
    {
        ray_march_nanovdb(
            buf,
            rayOrigin,
            0.f,
            rayDir,
            1e9f,
            rayDirInv,
            sum,
            nominalT
        );
    }

    float4 value = sum;
    // Note: BGRA target
    uint rawValue = uint(255.f * max(0.f, min(1.f, value.z))) |
        (uint(255.f * max(0.f, min(1.f, value.y))) << 8u) |
        (uint(255.f * max(0.f, min(1.f, value.x))) << 16u) |
        (uint(255.f * max(0.f, min(1.f, value.w))) << 24u);
    image_out[(editor_params.height - 1 - tidx.y) * editor_params.width + tidx.x] = rawValue;

    texture_out[tidx] = value;
}
