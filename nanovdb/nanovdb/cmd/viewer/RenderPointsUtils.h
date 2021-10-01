// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderFunctions.h
	\brief General C++ implementation of the Grid rendering code.
*/

#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>
#include "RenderConstants.h"
#include "RenderUtils.h"

namespace render {
namespace points {

struct PointCloud
{
    float density;
};

template<typename OutT, typename InT>
struct AttributeValueConverter
{
    inline __hostdev__ OutT operator()(const InT& in)
    {
        return OutT(in);
    }
};

template<>
struct AttributeValueConverter<float, nanovdb::Vec3f>
{
    inline __hostdev__ float operator()(const nanovdb::Vec3f& in)
    {
        return in[0];
    }
};

template<>
struct AttributeValueConverter<float, nanovdb::Vec3d>
{
    inline __hostdev__ float operator()(const nanovdb::Vec3d& in)
    {
        return float(in[0]);
    }
};

template<typename T, typename GridT>
inline __hostdev__ static T sampleBlindData(const GridT& grid, int i, const RendererAttributeParams& attribute)
{
    if (attribute.attribute < 0)
        return T(attribute.offset);

    auto meta = grid.blindMetaData(attribute.attribute);
    auto data = grid.blindData(attribute.attribute);

    assert(i < meta.mElementCount);

    if (meta.mDataType == nanovdb::GridType::Vec3f)
        return AttributeValueConverter<T, nanovdb::Vec3f>()(*(reinterpret_cast<const nanovdb::Vec3f*>(data) + i)) * attribute.gain + T(attribute.offset);
    else if (meta.mDataType == nanovdb::GridType::Vec3d)
        return AttributeValueConverter<T, nanovdb::Vec3d>()(*(reinterpret_cast<const nanovdb::Vec3d*>(data) + i)) * attribute.gain + T(attribute.offset);
    else if (meta.mDataType == nanovdb::GridType::Float)
        return AttributeValueConverter<T, float>()(*(reinterpret_cast<const float*>(data) + i)) * attribute.gain + T(attribute.offset);

    return T(attribute.offset);
}

// This helper class will march through the Point Tree and then iterate over all
// the points in each node it finds.
struct PointIntegrator
{
    template<typename RayT, typename GridT, typename AccT, typename FnT>
    inline __hostdev__ void operator()(const RayT& iRay, const GridT& grid, AccT& acc, const PointCloud& geometry, const RendererAttributeParams& posAttr, const RendererAttributeParams& radiusAttr, const FnT& fn) const
    {
        using TreeT = typename GridT::TreeType;

        nanovdb::PointTreeMarcher<AccT, RayT> marcher(acc);
        if (marcher.init(iRay)) {
            const typename TreeT::LeafNodeType* node;
            float                               t0 = 0, t1 = 0;
            while (marcher.step(&node, t0, t1)) {

                if (node->maximum() == 0)
                    continue;
                const auto nodeStartIndex = node->minimum();

                // DDA through the node's values...
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
                        auto beginIndex = nodeStartIndex + (offset == 0 ? 0 : node->getValue(offset - 1));
                        auto endIndex = nodeStartIndex + node->getValue(offset);

                        int count = endIndex - beginIndex;
                        for (int j = 0; j < count; ++j) {
                            int i = beginIndex + j;

                            const auto iRadius = sampleBlindData<float>(grid, i, radiusAttr);
                            const auto iPos = sampleBlindData<nanovdb::Vec3f>(grid, i, posAttr) + ijk.asVec3s();

                            // distance to point from ray.
                            float s = iRay.dir().cross(iPos - iRay.eye()).lengthSqr();
                            float f = s / (iRadius * iRadius);
                            if (f < 1.0f) {
                                if (fn(i, f, geometry.density) == false)
                                    return;
                            }
                        }
                    }

                } while (dda.step());
            }
        }
    }
};

struct RenderPointsRgba32fFn
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    template<typename GridT, typename AccT>
    inline __hostdev__ static float getTransmittance(RayT& ray, const GridT& grid, AccT& acc, const PointCloud& geometry, uint32_t& /*seed*/, const nanovdb::Vec3f& /*lightDir*/, const MaterialParameters params)
    {
        const RendererAttributeParams radiusAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointRadius];
        const RendererAttributeParams positionAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointPosition];

        float transmittance = 1.0f;

        PointIntegrator()(ray, grid, acc, geometry, positionAttribute, radiusAttribute, [&](int /*i*/, float /*distanceToCenter*/, float density) -> bool {
            transmittance *= expf(-density);
            return (transmittance > 0.001f);
        });

        return transmittance;
    }

    template<typename GridT, typename AccT>
    inline __hostdev__ static Vec3T traceGeometry(RayT& ray, const GridT& grid, AccT& acc, const PointCloud& geometry, uint32_t& /*seed*/, const nanovdb::Vec3f& /*lightDir*/, float& transmittance, const MaterialParameters params)
    {
        const RendererAttributeParams radiusAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointRadius];
        const RendererAttributeParams positionAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointPosition];
        const RendererAttributeParams colorAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointColor];

        Vec3T radiance(0);
        transmittance = 1.0f;

        PointIntegrator()(ray, grid, acc, geometry, positionAttribute, radiusAttribute, [&](int i, float /*distanceToCenter*/, float density) -> bool {
            radiance += transmittance * density * sampleBlindData<Vec3T>(grid, i, colorAttribute);
            transmittance *= expf(-density);
            return (transmittance > 0.001f);
        });

        return radiance;
    }

    template<typename ValueType>
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int numAccumulations, const nanovdb::NanoGrid<ValueType>* grid, const SceneRenderParameters sceneParams, const MaterialParameters params) const
    {
        float* outPixel = &imgBuffer[4 * (ix + width * iy)];

        float color[4] = {0, 0, 0, 0};

        if (!grid) {
            RayT wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);

            auto envRadiance = traceEnvironment(wRay, sceneParams);

            color[0] = envRadiance;
            color[1] = envRadiance;
            color[2] = envRadiance;

        } else {
            const Vec3T wLightDir = sceneParams.sunDirection;
            const Vec3T iLightDir = grid->worldToIndexDirF(wLightDir).normalize();

            const auto& tree = grid->tree();
            const auto  acc = tree.getAccessor();

            PointCloud geometry;
            geometry.density = params.volumeDensityScale;

            for (int sampleIndex = 0; sampleIndex < sceneParams.samplesPerPixel; ++sampleIndex) {
                uint32_t pixelSeed = hash((sampleIndex + (numAccumulations + 1) * sceneParams.samplesPerPixel)) ^ hash(ix, iy);

                RayT wRay = getRayFromPixelCoord(ix, iy, width, height, numAccumulations, sceneParams.samplesPerPixel, pixelSeed, sceneParams);
                RayT iRay = wRay.worldToIndexF(*grid);

                float pathThroughput = 1.0f;
                Vec3T radiance = traceGeometry(iRay, *grid, acc, geometry, pixelSeed, iLightDir, pathThroughput, params);

                if (pathThroughput > 0.0f) {
                    float bgIntensity = 0;

                    if (sceneParams.useBackground) {
                        float groundIntensity = 0.0f;
                        float groundMix = 0.0f;

                        if (sceneParams.useLighting && sceneParams.useGround) {
                            // intersect with ground plane and draw checker if camera is above...
                            float wGroundT = (sceneParams.groundHeight - wRay.eye()[1]) / wRay.dir()[1];
                            if (wRay.dir()[1] != 0 && wGroundT > 0.f) {
                                Vec3T wGroundPos = wRay(wGroundT);
                                Vec3T iGroundPos = grid->worldToIndexF(wGroundPos);

                                groundIntensity = evalGroundMaterial(wGroundT, sceneParams.groundFalloff, wGroundPos, wRay.dir()[1], groundMix);

                                if (sceneParams.useShadows) {
                                    RayT  iShadowRay(iGroundPos, iLightDir);
                                    float shadowTransmittance = getTransmittance(iShadowRay, *grid, acc, geometry, pixelSeed, iLightDir, params);
                                    groundIntensity *= shadowTransmittance;
                                }
                            }
                        }
                        float skyIntensity = evalSkyMaterial(wRay.dir());
                        bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
                    }
                    radiance += nanovdb::Vec3f(pathThroughput * bgIntensity);
                }

                color[0] += radiance[0];
                color[1] += radiance[1];
                color[2] += radiance[2];
            }

            for (int k = 0; k < 3; ++k)
                color[k] = color[k] / sceneParams.samplesPerPixel;
        }

        compositeFn(outPixel, color, numAccumulations, sceneParams);
    }
};

}
} // namespace render::points