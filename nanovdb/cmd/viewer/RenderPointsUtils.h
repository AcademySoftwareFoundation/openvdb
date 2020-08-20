// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderFunctions.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief General C++ implementation of the Grid rendering code.
*/

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

struct RenderPointsRgba32fFn
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    template<typename T, typename GridT>
    inline __hostdev__ static T sampleAttribute(const GridT& grid, CoordT origin, int i, const RendererAttributeParams& attribute)
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

    template<typename GridT, typename AccT>
    inline __hostdev__ static float getTransmittance(RayT& ray, const GridT& grid, AccT& acc, const PointCloud& geometry, uint32_t& seed, const nanovdb::Vec3f& lightDir, const RenderConstants params)
    {
        using namespace nanovdb;

        float transmittance = 1.0f;

        const RendererAttributeParams radiusAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointRadius];
        const RendererAttributeParams positionAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointPosition];

        nanovdb::TreeMarcher<typename AccT::NodeT0, RayT, AccT> marcher(acc);
        if (marcher.init(ray)) {
            const typename AccT::NodeT0* node;
            float                        t0 = 0, t1 = 0;

            while (marcher.step(&node, t0, t1)) {
                if (t0 >= t1)
                    continue;

                if (node->valueMax() == 0)
                    continue;
                auto nodeStartIndex = node->valueMin();

                // DDA through the node's values...
                nanovdb::HDDA<RayT> hdda;
                hdda.init(ray, t0, t1, 1);
                do {
                    auto           ijk = hdda.voxel();
                    const uint32_t offset = node->CoordToOffset(ijk);
                    if (node->isActive(offset)) {
                        auto beginIndex = nodeStartIndex + (offset == 0 ? 0 : node->getValue(offset - 1));
                        auto endIndex = nodeStartIndex + node->getValue(offset);

                        int count = endIndex - beginIndex;
                        for (int j = 0; j < count; ++j) {
                            int i = beginIndex + j;

                            const float radius = sampleAttribute<float>(grid, ijk, i, radiusAttribute);
                            const Vec3T pos = sampleAttribute<Vec3T>(grid, ijk, i, positionAttribute) + Vec3T(0.5f);

                            // distance to point from ray.
                            float s = ray.dir().cross(pos - ray.eye()).lengthSqr();
                            float f = s / (radius * radius);
                            if (f < 1.0f) {
                                f = geometry.density;

                                // modify transmittance.
                                transmittance *= 1.0f - f;
                            }
                        }
                    }
                } while (hdda.step());
            }
        }

        return transmittance;
    }

    template<typename GridT, typename AccT>
    inline __hostdev__ static Vec3T traceGeometry(RayT& ray, const GridT& grid, AccT& acc, const PointCloud& geometry, uint32_t& seed, const nanovdb::Vec3f& lightDir, float& transmittance, const RenderConstants params)
    {
        using namespace nanovdb;

        const RendererAttributeParams radiusAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointRadius];
        const RendererAttributeParams positionAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointPosition];
        const RendererAttributeParams colorAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointColor];

        Vec3T radiance(0);
        transmittance = 1.0f;

        nanovdb::TreeMarcher<typename AccT::NodeT0, RayT, AccT> marcher(acc);
        if (marcher.init(ray)) {
            const typename AccT::NodeT0* node;
            float                        t0 = 0, t1 = 0;

            while (marcher.step(&node, t0, t1)) {
                if (t0 >= t1)
                    continue;

                if (node->valueMax() == 0)
                    continue;
                auto nodeStartIndex = node->valueMin();

                // DDA through the node's values...
                nanovdb::HDDA<RayT> hdda;
                hdda.init(ray, t0, t1, 1);
                do {
                    auto           ijk = hdda.voxel();
                    const uint32_t offset = node->CoordToOffset(ijk);
                    if (node->isActive(offset)) {
                        auto beginIndex = nodeStartIndex + (offset == 0 ? 0 : node->getValue(offset - 1));
                        auto endIndex = nodeStartIndex + node->getValue(offset);

                        int count = endIndex - beginIndex;
                        for (int j = 0; j < count; ++j) {
                            int i = beginIndex + j;

                            const float radius = sampleAttribute<float>(grid, ijk, i, radiusAttribute);
                            const Vec3T pos = sampleAttribute<Vec3T>(grid, ijk, i, positionAttribute) + Vec3T(0.5f);

                            // distance to point from ray.
                            float s = ray.dir().cross(pos - ray.eye()).lengthSqr();
                            float f = s / (radius * radius);
                            if (f < 1.0f) {
                                f = geometry.density;

                                float shadowTransmittance = 1.0f;
#if 0
								if (params.useShadows > 0)
								{
									RayT iShadowRay(pos + lightDir * 0.1f, lightDir);
									shadowTransmittance = getTransmittance(iShadowRay, grid, acc, geometry, seed, lightDir);
								}
#endif

                                radiance += shadowTransmittance * transmittance * f * sampleAttribute<Vec3T>(grid, ijk, i, colorAttribute);

                                // modify transmittance.
                                transmittance *= 1.0f - f;
                            }
                        }
                    }
                } while (hdda.step());
            }
        }

        return radiance;
    }

    template<typename ValueType>
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, Camera<float> camera, const nanovdb::NanoGrid<ValueType>* grid, int numAccumulations, const RenderConstants params) const
    {
        float* outPixel = &imgBuffer[4 * (ix + width * iy)];

        const auto& tree = grid->tree();

        const Vec3T wLightDir = Vec3T(0, 1, 0);
        const Vec3T iLightDir = grid->worldToIndexDirF(wLightDir).normalize();

        auto acc = tree.getAccessor();

        PointCloud geometry;
        geometry.density = params.volumeDensity;

        float color[4] = {0, 0, 0, 0};

        for (int sample = 0; sample < params.samplesPerPixel; ++sample) {
            uint32_t pixelSeed = hash(sample + numAccumulations * params.samplesPerPixel ^ hash(ix, iy));

            float u = ix + 0.5f;
            float v = iy + 0.5f;

            float randVar1 = randomf(pixelSeed + 0);
            float randVar2 = randomf(pixelSeed + 1);

            if (numAccumulations > 0) {
#if 1
                float jitterX, jitterY;
                cmj(jitterX, jitterY, numAccumulations % 64, 8, 8, pixelSeed);
                u += jitterX - 0.5f;
                v += jitterY - 0.5f;
#else
                u += randVar1 - 0.5f;
                v += randVar2 - 0.5f;
#endif
            }
            u /= width;
            v /= height;

            RayT wRay = camera.getRay(u, v);
            RayT iRay = wRay.worldToIndexF(*grid);

            Vec3T iRayDir = iRay.dir();
            Vec3T wRayDir = wRay.dir();
            Vec3T wRayEye = wRay.eye();

            float transmittance = 1.0f;
            Vec3T radiance = traceGeometry(iRay, *grid, acc, geometry, pixelSeed, iLightDir, transmittance, params);

            if (transmittance > 0.01f) {
                float groundIntensity = 0.0f;
                float groundMix = 0.0f;

                if (params.useGround > 0) {
                    // intersect with ground plane and draw checker if camera is above...

                    float wGroundT = (params.groundHeight - wRayEye[1]) / wRayDir[1];

                    if (wGroundT > 0.f) {
                        Vec3T wGroundPos = wRayEye + wGroundT * wRayDir;
                        Vec3T iGroundPos = grid->worldToIndexF(wGroundPos);

                        rayTraceGround(wGroundT, params.groundFalloff, wGroundPos, wRayDir[1], groundIntensity, groundMix);

                        if (params.useShadows > 0) {
                            RayT  iShadowRay(iGroundPos, iLightDir);
                            float shadowTransmittance = getTransmittance(iShadowRay, *grid, acc, geometry, pixelSeed, iLightDir, params);
                            groundIntensity *= shadowTransmittance;
                        }
                    }
                }

                float skyIntensity = 0.75f + 0.25f * wRayDir[1];

                radiance = radiance + nanovdb::Vec3f(transmittance * ((1.f - groundMix) * skyIntensity + groundMix * groundIntensity));
            }

            color[0] += radiance[0];
            color[1] += radiance[1];
            color[2] += radiance[2];
        }

        for (int k = 0; k < 3; ++k)
            color[k] = color[k] / params.samplesPerPixel;

        if (numAccumulations > 1) {
            for (int k = 0; k < 3; ++k)
                color[k] = outPixel[k] + (color[k] - outPixel[k]) * (1.0f / numAccumulations);
        }

        for (int k = 0; k < 3; ++k)
            outPixel[k] = color[k];
        outPixel[3] = 1.0;
    }
};

}
} // namespace render::points