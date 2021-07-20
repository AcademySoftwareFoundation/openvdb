// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderVoxelUtils.h
	\brief General C++ implementation of the Voxel rendering code.
*/

#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>
#include "RenderConstants.h"
#include "RenderUtils.h"

namespace render {
namespace voxel {

namespace detail {

inline __hostdev__ bool rayIntersectSphere(const nanovdb::Ray<float>& ray, const nanovdb::Vec3f& center, const float radius, float& t0, nanovdb::Vec3f& normal)
{
    const bool use_robust_method = false;
    using namespace nanovdb;

    const Vec3f& rd = ray.dir();
    //const Vec3f& rd_inv = ray.invDir();
    const Vec3f  ro = ray.eye() - center;

    float b = ro.dot(rd);
    float c = ro.dot(ro) - radius * radius;
    float disc = b * b - c;
    if (disc > 0.0f) {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);

        bool do_refine = false;

        float root11 = 0.0f;

        if (use_robust_method && Abs(root1) > 10.f * radius) {
            do_refine = true;
        }

        if (do_refine) {
            // refine root1
            auto ro1 = ro + root1 * rd;
            b = ro1.dot(rd);
            c = ro1.dot(ro1) - radius * radius;
            disc = b * b - c;

            if (disc > 0.0f) {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        bool check_second = true;

        float t;
        t = (root1 + root11);
        if (t > t0) {
            normal = (ro + (root1 + root11) * rd) / radius;
            t0 = t;
            return true;
        }

        if (check_second) {
            float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
            t = root2;
            normal = (ro + root2 * rd) / radius;
            if (t > t0) {
                t0 = t;
                return true;
            }
        }
    }

    return false;
}

__hostdev__ inline bool
rayIntersectCube(const nanovdb::Ray<float>& ray, const nanovdb::Vec3f& center, float radius, float& s, nanovdb::Vec3f& normal)
{
    using namespace nanovdb;

    const Vec3f& rd = ray.dir();
    const Vec3f& rd_inv = ray.invDir();
    const Vec3f  ro = ray.eye() - center;

    Vec3f sgn = -Vec3f(nanovdb::Sign(rd[0]), nanovdb::Sign(rd[1]), nanovdb::Sign(rd[2]));
    Vec3f distanceToPlane = radius * sgn - ro;
    distanceToPlane = distanceToPlane * rd_inv;
    Vec3i test = Vec3i((distanceToPlane[0] >= 0.0f) &&
                           ((Abs(ro[1] + rd[1] * distanceToPlane[0]) < radius) && (Abs(ro[2] + rd[2] * distanceToPlane[0]) < radius)),
                       (distanceToPlane[1] >= 0.0f) &&
                           ((Abs(ro[2] + rd[2] * distanceToPlane[1]) < radius) && (Abs(ro[0] + rd[0] * distanceToPlane[1]) < radius)),
                       (distanceToPlane[2] >= 0.0f) &&
                           ((Abs(ro[0] + rd[0] * distanceToPlane[2]) < radius) && (Abs(ro[1] + rd[1] * distanceToPlane[2]) < radius)));

    sgn = test[0] ? Vec3f(sgn[0], 0.0f, 0.0f)
                  : (test[1] ? Vec3f(0.0f, sgn[1], 0.0f) : Vec3f(0.0f, 0.0f, test[2] ? sgn[2] : 0.0f));
    s = (sgn[0] != 0.0f) ? distanceToPlane[0] : ((sgn[1] != 0.0) ? distanceToPlane[1] : distanceToPlane[2]);
    normal = sgn;
    return (sgn[0] != 0) || (sgn[1] != 0) || (sgn[2] != 0);
}

__hostdev__ inline bool
rayIntersectSdf(const nanovdb::Ray<float>& ray, const nanovdb::Vec3f& center, float radius, float& s, nanovdb::Vec3f& normal)
{
    using namespace nanovdb;
    
#if 1
    auto sdSphere = [radius] __hostdev__(const Vec3f& p) -> float {
        return p.length() - radius;
    };
    auto sdf = [sdSphere, center] __hostdev__(const Vec3f& p) -> float {
        return sdSphere(p - center);
    };
#else
const float b = radius * 2;
    auto        sdRoundBox = [b] __hostdev__(const Vec3f& p) -> float {
        const auto q = Vec3f(Abs(p[0] - b), Abs(p[1] - b), Abs(p[2] - b));
        return Vec3f(Max(q[0], 0.f), Max(q[1], 0.f), Max(q[2], 0.f)).length() + Min(Max(q[0], Max(q[1], q[2])), 0.0f) - 0.0f;
    };
    auto sdf = [sdRoundBox, center] __hostdev__(const Vec3f& p) -> float {
        return sdRoundBox(p - center);
    };
#endif
    auto calcNormal = [sdf] __hostdev__(const Vec3f& p) -> Vec3f {
        return Vec3f(
                   sdf(Vec3f(p[0] + 0.001f, p[1], p[2])) - sdf(Vec3f(p[0] - 0.001f, p[1], p[2])),
                   sdf(Vec3f(p[0], p[1] + 0.001f, p[2])) - sdf(Vec3f(p[0], p[1] - 0.001f, p[2])),
                   sdf(Vec3f(p[0], p[1], p[2] + 0.001f)) - sdf(Vec3f(p[0], p[1], p[2] - 0.001f)))
            .normalize();
    };

    // OPT: we already have s. (we need to use that)
    s = ray.t0();
    for (int i = 0; i < 16; i++) {
        Vec3f p = ray(s);
        float d = sdf(p);
        if (d <= 0.001f) {
            normal = calcNormal(p);
            return true;
        }

        s += d;
        if (s >= ray.t1())
            break;
    }

    return false;
}

__hostdev__ inline bool
rayIntersectVoxel(int voxelGeometry, const nanovdb::Ray<float>& ray, const nanovdb::Vec3f& center, float radius, float& s, nanovdb::Vec3f& normal)
{
    if (voxelGeometry == 1)
        return rayIntersectSphere(ray, center, 0.5f, s, normal);
    return rayIntersectCube(ray, center, 0.5f, s, normal);
}

template<typename RayT, typename GridT, typename AccT, typename FnT>
inline __hostdev__ static nanovdb::Vec3f traceGeometry(const RayT& iRay, const GridT& grid, AccT& acc, uint32_t& seed, float& throughput, float& hitVoxelT, const SceneRenderParameters& sceneParams, const MaterialParameters materialParams, const FnT& hitFunc)
{
    using Vec3T = nanovdb::Vec3f;
    Vec3T radiance(0);
    throughput = 1.0f;

    nanovdb::TreeMarcher<typename GridT::TreeType::LeafNodeType, nanovdb::Ray<float>, AccT> marcher(acc);
    if (marcher.init(iRay)) {
        const typename GridT::TreeType::LeafNodeType* node = nullptr;
        float                                         t0 = 0, t1 = 0;

        while (throughput > 0.f && marcher.step(&node, t0, t1)) {
            nanovdb::DDA<RayT> dda;
            dda.init(marcher.ray(), t0, t1);
            do {
                auto ijk = dda.voxel();
                // CAVEAT:
                // This is currently necessary becuse the voxel returned might not actually be innside the node!
                // This is currently happening from time to time due to float precision issues,
                // so we skip out of bounds voxels here...
                auto localIjk = ijk - node->origin();
                if (localIjk[0] < 0 || localIjk[1] < 0 || localIjk[2] < 0 || localIjk[0] >= 8 || localIjk[1] >= 8 || localIjk[2] >= 8)
                    continue;

                const uint32_t offset = node->CoordToOffset(ijk);
                if (node->isActive(offset)) {
                    hitVoxelT = dda.time();
                    RayT ray(marcher.ray().eye(), marcher.ray().dir(), hitVoxelT, t1);
                    if (hitFunc(radiance, throughput, grid, acc, seed, node->getValue(offset), ray, hitVoxelT, nanovdb::Vec3f(ijk), sceneParams, materialParams)) {
                        throughput = 0.f;
                        break;
                    }
                }

            } while (dda.step());
        }
    }

    return radiance;
}

} // namespace detail

struct OcclusionHitFn
{
    template<typename ValueT, typename GridT, typename AccT>
    inline __hostdev__ bool operator()(nanovdb::Vec3f& inoutRadiance, float& inoutThroughput, const GridT& grid, const AccT& acc, uint32_t seed, const ValueT& v, nanovdb::Ray<float>& iRay, const float t, const nanovdb::Vec3f& pos, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams) const
    {
        nanovdb::Vec3f iNormal;
        float          s = t;
        return detail::rayIntersectVoxel(materialParams.voxelGeometry, iRay, pos + nanovdb::Vec3f(0.5f), 0.5f, s, iNormal);
    }
};

struct PrimaryHitFn
{
    using RayT  = nanovdb::Ray<float>;
    using Vec3T = nanovdb::Vec3f;

    template<typename ValueT, typename GridT, typename AccT>
    inline __hostdev__ bool operator()(nanovdb::Vec3f& inoutRadiance, float& inoutThroughput, const GridT& grid, const AccT& acc, uint32_t seed, const ValueT& v, const nanovdb::Ray<float>& iRay, const float t, const nanovdb::Vec3f& pos, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams) const
    {
        using namespace nanovdb;
        Vec3f iNormal;
        float s = t;
        bool  hit = detail::rayIntersectVoxel(materialParams.voxelGeometry, iRay, pos + Vec3f(0.5f), 0.5f, s, iNormal);

        if (hit) {
            auto diffuseColor = valueToColor(v);

            // do some cheap, hacky shading...
            if (sceneParams.useLighting) {
                const float occlusion = 0.0f;
                const float ambient = 1.0f;
                Vec3f       diffuseKey = Vec3f(1);
                Vec3f       diffuseFill = Vec3f(1);

                const float diffuseWrap = 0.25f;
                diffuseKey *= Max(0.0f, (iNormal.dot(mLightDir) + diffuseWrap) / (1.0f + diffuseWrap));
                diffuseFill *= Max(0.0f, iNormal.dot(-iRay.dir()));
                Vec3f H = (mLightDir + -iRay.dir()).normalize();
                float specularKey = powf(fmaxf(0.0f, iNormal.dot(H)), 10.f);

                float shadowFactor = 0.f;
                if (sceneParams.useShadows) {
                    auto iSurfacePos = iRay(t);
                    //auto iSurfacePos = pos + Vec3f(0.5f);
                    //iSurfacePos += iNormal * 0.5f;

                    const int numShadowSamples = 1;
                    for (int i = 0; i < numShadowSamples; ++i) {
                        // NOTE: we push the ray out a little from the surface.
                        RayT iOccRay(iSurfacePos, lambertNoTangent(iNormal, randomXorShift(seed), randomXorShift(seed)), 0.001f);
                        RayT iShadowRay(iSurfacePos, mLightDir, 0.001f);
                        if (randomXorShift(seed) > 0.9f)
                            iOccRay = iShadowRay; // add a bit of direct shadow.

                        float shadowThroughput = 1.f;
                        float shadowHitVoxelT;
                        detail::traceGeometry(iOccRay, grid, acc, seed, shadowThroughput, shadowHitVoxelT, sceneParams, materialParams, OcclusionHitFn());
                        shadowFactor += 1.0f - shadowThroughput;
                    }
                    shadowFactor *= 1.f / numShadowSamples;
                }

                auto intensity = ((1.0f - shadowFactor) * (nanovdb::Vec3f(specularKey * 0.6f) + (diffuseKey * 0.8f)) + (1.0f - occlusion) * (diffuseFill * 0.2f + nanovdb::Vec3f(ambient * 0.1f)));

                inoutRadiance[0] += diffuseColor[0] * intensity[0] * inoutThroughput;
                inoutRadiance[1] += diffuseColor[1] * intensity[1] * inoutThroughput;
                inoutRadiance[2] += diffuseColor[2] * intensity[2] * inoutThroughput;
            } else {
                inoutRadiance[0] += diffuseColor[0] * inoutThroughput;
                inoutRadiance[1] += diffuseColor[1] * inoutThroughput;
                inoutRadiance[2] += diffuseColor[2] * inoutThroughput;
            }

            // NOTE: this is approximate and should be using voxel geometry span distance.
            // But do we really care for this render mode?
            // But until we allow non binary throughput, just assume dense voxels.
            inoutThroughput *= 0.f; //expf(-materialParams.volumeDensityScale * 1.0f);

            return (inoutThroughput < 0.001f);
        }
        return false;
    }

    nanovdb::Vec3f mLightDir;
};

struct RenderVoxelsRgba32fFn
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    template<typename ValueType>
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int numAccumulations, const nanovdb::NanoGrid<ValueType>* grid, const SceneRenderParameters sceneParams, const MaterialParameters materialParams) const
    {
        auto outPixel = &imgBuffer[4 * (ix + width * iy)];

        nanovdb::Vec4f color{0};

        if (!grid) {
            RayT wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);

            auto envRadiance = traceEnvironment(wRay, sceneParams);

            color[0] = envRadiance;
            color[1] = envRadiance;
            color[2] = envRadiance;
        } else {
            const Vec3T wLightDir = sceneParams.sunDirection;
            const Vec3T iLightDir = grid->worldToIndexDirF(wLightDir).normalize();

            PrimaryHitFn primaryHitFn;
            primaryHitFn.mLightDir = iLightDir;

            for (int sampleIndex = 0; sampleIndex < sceneParams.samplesPerPixel; ++sampleIndex) {
                uint32_t pixelSeed = hash((sampleIndex + (numAccumulations + 1) * sceneParams.samplesPerPixel)) ^ hash(ix, iy);

                RayT wRay = getRayFromPixelCoord(ix, iy, width, height, numAccumulations, sceneParams.samplesPerPixel, pixelSeed, sceneParams);
                RayT iRay = wRay.worldToIndexF(*grid);

                const auto& tree = grid->tree();
                auto        acc = tree.getAccessor();

                float primaryThroughput = 1.0f;
                float hitVoxelT;
                auto  radiance = detail::traceGeometry(iRay, *grid, acc, pixelSeed, primaryThroughput, hitVoxelT, sceneParams, materialParams, primaryHitFn);

                if (primaryThroughput > 0.f) {
                    float bgIntensity = 0;

                    if (sceneParams.useBackground) {
                        float groundIntensity = 0.0f;
                        float groundMix = 0.0f;

                        if (sceneParams.useGround) {
                            // intersect with ground plane and draw checker if camera is above...
                            float wGroundT = (sceneParams.groundHeight - wRay.eye()[1]) / wRay.dir()[1];
                            if (wRay.dir()[1] != 0 && wGroundT > 0.f) {
                                Vec3T wGroundPos = wRay(wGroundT);
                                Vec3T iGroundPos = grid->worldToIndexF(wGroundPos);

                                groundIntensity = evalGroundMaterial(wGroundT, sceneParams.groundFalloff, wGroundPos, wRay.dir()[1], groundMix);

                                if (sceneParams.useLighting && sceneParams.useShadows) {
                                    RayT  iShadowRay(iGroundPos, iLightDir);
                                    float shadowVoxelThroughput = 1;
                                    float shadowHitVoxelT;
                                    detail::traceGeometry(iShadowRay, *grid, acc, pixelSeed, shadowVoxelThroughput, shadowHitVoxelT, sceneParams, materialParams, OcclusionHitFn());
                                    groundIntensity *= shadowVoxelThroughput;
                                }
                            }
                        }
                        float skyIntensity = evalSkyMaterial(wRay.dir());
                        bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
                    }
                    radiance += nanovdb::Vec3f(bgIntensity);
                }

                color[0] += radiance[0];
                color[1] += radiance[1];
                color[2] += radiance[2];
            }

            for (int k = 0; k < 3; ++k)
                color[k] = color[k] / sceneParams.samplesPerPixel;
        }

        compositeFn(outPixel, (float*)&color, numAccumulations, sceneParams);
    }
};

}
} // namespace render::voxel