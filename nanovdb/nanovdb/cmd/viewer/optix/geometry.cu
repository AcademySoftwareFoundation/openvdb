
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <optix.h>
#include "NanoVDB_optix.h"
#include "helpers.h"
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>

//#define OPTIX_PERF_USE_LEAF_DDA

extern "C" {
__constant__ Params constantParams;
}

__device__ inline bool
rayIntersectAabb(nanovdb::Vec3f ro_, nanovdb::Vec3f rd_, float ray_tmin, float3 pos, float3 radius, float& s, float3& normal, float& winding)
{
    float3 ro = make_float3(ro_[0], ro_[1], ro_[2]);
    float3 rd = make_float3(rd_[0], rd_[1], rd_[2]);
    float3 rd_inv = 1.f / (rd);

    const bool rayCanStartInBox = false;
    ro -= pos;

    if (rayCanStartInBox) {
        winding = (fmaxf(fabsf(ro) * (1.0f / radius)) < 1.0f) ? -1.0f : 1.0f;
    } else {
        winding = 1.0f;
    }

    float3 sgn = -signf(rd);
    float3 distanceToPlane = radius * winding * sgn - ro;
    distanceToPlane *= rd_inv;
    int3 test = make_int3((distanceToPlane.x >= 0.0f) &&
                              all(lessThan(fabsf(make_float2(ro.y, ro.z) + make_float2(rd.y, rd.z) * distanceToPlane.x),
                                           make_float2(radius.y, radius.z))),
                          (distanceToPlane.y >= 0.0f) &&
                              all(lessThan(fabsf(make_float2(ro.z, ro.x) + make_float2(rd.z, rd.x) * distanceToPlane.y),
                                           make_float2(radius.z, radius.x))),
                          (distanceToPlane.z >= 0.0f) &&
                              all(lessThan(fabsf(make_float2(ro.x, ro.y) + make_float2(rd.x, rd.y) * distanceToPlane.z),
                                           make_float2(radius.x, radius.y))));
    sgn = test.x ? make_float3(sgn.x, 0.0f, 0.0f)
                 : (test.y ? make_float3(0.0, sgn.y, 0.0) : make_float3(0.0, 0.0, test.z ? sgn.z : 0.0));
    s = (sgn.x != 0.0) ? distanceToPlane.x : ((sgn.y != 0.0) ? distanceToPlane.y : distanceToPlane.z);
    normal = sgn;
    bool hit = (sgn.x != 0) || (sgn.y != 0) || (sgn.z != 0);
    return hit;
}

inline __device__ nanovdb::Vec3f rayBoxIntersect(nanovdb::Vec3f rpos, nanovdb::Vec3f rdir, nanovdb::Vec3f vmin, nanovdb::Vec3f vmax)
{
    float ht[8];
    ht[0] = (vmin[0] - rpos[0]) / rdir[0];
    ht[1] = (vmax[0] - rpos[0]) / rdir[0];
    ht[2] = (vmin[1] - rpos[1]) / rdir[1];
    ht[3] = (vmax[1] - rpos[1]) / rdir[1];
    ht[4] = (vmin[2] - rpos[2]) / rdir[2];
    ht[5] = (vmax[2] - rpos[2]) / rdir[2];
    ht[6] = fmax(fmax(fmin(ht[0], ht[1]), fmin(ht[2], ht[3])), fmin(ht[4], ht[5]));
    ht[7] = fmin(fmin(fmax(ht[0], ht[1]), fmax(ht[2], ht[3])), fmax(ht[4], ht[5]));
    ht[6] = (ht[6] < 0) ? 0.0f : ht[6];
    return nanovdb::Vec3f(ht[6], ht[7], (ht[7] < ht[6] || ht[7] <= 0 || ht[6] <= 0) ? -1.0f : 0.0f);
}

// -----------------------------------------------------------------------------
// LevelSet render method
//
extern "C" __global__ void __intersection__nanovdb_levelset()
{
    using RealT = float;
    using CoordT = nanovdb::Coord;
    using Vec3T = nanovdb::Vec3f;

    const VolumeGeometry* volume = reinterpret_cast<VolumeGeometry*>(optixGetSbtDataPointer());
    const auto*           grid = reinterpret_cast<const nanovdb::FloatGrid*>(volume->grid);

    const auto primIndex = optixGetPrimitiveIndex();
    const int leafIndex = volume->enumeration[primIndex];
    const auto leafNode = reinterpret_cast<const nanovdb::FloatTree::LeafNodeType*>(reinterpret_cast<const uint8_t*>(grid) + uintptr_t(leafIndex)*32);

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();

    const auto iEye = grid->worldToIndexF(reinterpret_cast<const Vec3T&>(ray_orig));
    const auto iDir = grid->worldToIndexDirF(reinterpret_cast<const Vec3T&>(ray_dir)).normalize();
    auto       iRay = nanovdb::Ray<float>(iEye, iDir);

    auto acc = grid->tree().getAccessor();
    float voxelUniformSize = float(grid->voxelSize()[0]);

#if (OPTIX_PERF_USE_LEAF_DDA)
    CoordT ijk;
    float  v = 0.0f;
    float  v0 = 0.0f;
    float  iT;
    if (nanovdb::ZeroCrossingNode(iRay, *leafNode, v0, ijk, v, iT)) {
#else
    if (!iRay.clip(leafNode->bbox()))
        return;
    float t0 = iRay.t0() - 2.f * voxelUniformSize;
    float t1 = iRay.t1() + 2.f * voxelUniformSize;
    iRay.setTimes(nanovdb::Max(t0, 0.000001f), nanovdb::Max(t1, 0.0001f));
    CoordT ijk;
    float  v;
    float  iT;
    if (nanovdb::ZeroCrossing(iRay, acc, ijk, v, iT)) {
#endif
        // compute the intersection interval.
        auto p = make_float3(grid->indexToWorldF(iRay(iT)));
        optixReportIntersection(iT * voxelUniformSize, 0, array3_as_args(ijk), float3_as_args(p));
    }

    return;
}

// -----------------------------------------------------------------------------
// FogVolume render method
//
extern "C" __global__ void __intersection__nanovdb_fogvolume()
{
    const auto* sbt_data = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(sbt_data->geometry.volume.grid);

    const float3              ray_orig = optixGetWorldRayOrigin();
    const float3              ray_dir = optixGetWorldRayDirection();
    const nanovdb::Ray<float> wRay(reinterpret_cast<const nanovdb::Vec3f&>(ray_orig),
                                   reinterpret_cast<const nanovdb::Vec3f&>(ray_dir));

    auto iRay = wRay.worldToIndexF(*grid);
    auto bbox = grid->tree().bbox().asReal<float>();
    auto hit = rayBoxIntersect(iRay.eye(), iRay.dir(), bbox.min(), bbox.max());
    if (hit[2] != -1) {
        float voxelUniformSize = float(grid->voxelSize()[0]);
        optixReportIntersection(hit[0] * voxelUniformSize, 0, float_as_int(hit[1] * voxelUniformSize));
    }
}

// -----------------------------------------------------------------------------
// Grid render method
//
extern "C" __global__ void __intersection__nanovdb_grid()
{
    const auto* sbt_data = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(sbt_data->geometry.volume.grid);
    
    const auto primIndex = optixGetPrimitiveIndex();
    const int leafIndex = sbt_data->geometry.volume.enumeration[primIndex];
    const auto leafNode = reinterpret_cast<const nanovdb::FloatTree::LeafNodeType*>(reinterpret_cast<const uint8_t*>(grid) + uintptr_t(leafIndex)*32);

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    const auto wEye = nanovdb::Vec3f(ray_orig.x, ray_orig.y, ray_orig.z);
    const auto wDir = nanovdb::Vec3f(ray_dir.x, ray_dir.y, ray_dir.z);
    const auto wRay = nanovdb::Ray<float>(wEye, wDir, ray_tmin, ray_tmax);
    auto       iRay = wRay.worldToIndexF(*grid);

    const float3 bmin = make_float3(nanovdb::Vec3f(leafNode->origin()));
    const float3 radius = 0.5f * make_float3(nanovdb::Vec3f(leafNode->dim()));
    const float3 pos = bmin + radius;

    float3 normal;
    float  s;
    float  winding;
    bool   hit = rayIntersectAabb(iRay.eye(), iRay.dir(), ray_tmin, pos, radius, s, normal, winding);

    if (hit) {
        optixReportIntersection(s, 0, float3_as_args(normal));
    }
}
