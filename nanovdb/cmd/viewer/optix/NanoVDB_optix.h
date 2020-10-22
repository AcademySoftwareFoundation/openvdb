
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <stdint.h>
#include <vector_types.h>
#include <optix_types.h>
#include "../RenderConstants.h"

enum RayType { RAY_TYPE_RADIANCE = 0,
               RAY_TYPE_OCCLUSION = 1,
               RAY_TYPE_COUNT };

struct Params
{
    uint32_t              numAccumulations;
    float4*               imgBuffer;
    uint32_t              width;
    uint32_t              height;

    MaterialParameters    materialConstants;
    SceneRenderParameters sceneConstants;

    int   maxDepth;
    float sceneEpsilon;

    OptixTraversableHandle handle;
};

struct MissData
{
};

struct VolumeGeometry
{
    const void* grid;
};

struct VolumeMaterial
{
    float  importance_cutoff;
    float3 Ksigma;
};

struct HitGroupData
{
    struct
    {
        VolumeGeometry volume;
    } geometry;

    struct
    {
        VolumeMaterial volume;
    } shading;
};

inline __hostdev__ float3 make_float3(const nanovdb::Vec3f& v)
{
    return make_float3(v[0], v[1], v[2]);
}

inline __hostdev__ float3 make_float3(const nanovdb::Vec3R& v)
{
    return make_float3(v[0], v[1], v[2]);
}

inline __hostdev__ float3 make_float3(const float3& v)
{
    return v;
}
