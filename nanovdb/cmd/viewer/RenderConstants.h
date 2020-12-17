// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderConstants.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Declaration of RenderConstants structure used for render platforms.
*/

#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>

enum class VolumeTransmittanceMethod : int { kRiemannSum = 0,
                                             kDeltaTracking,
                                             kRatioTracking,
                                             kResidualRatioTracking };

struct RendererAttributeParams
{
    int   attribute;
    float gain;
    float offset;
};

struct MaterialParameters
{
    float                     useOcclusion;
    float                     volumeDensityScale;
    float                     volumeTemperatureScale;
    VolumeTransmittanceMethod transmittanceMethod;
    float                     transmittanceThreshold;
    float                     phase;
    int                       maxPathDepth;
    float                     volumeAlbedo;
    int                       interpolationOrder;
    int                       voxelGeometry;
    RendererAttributeParams   attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::End];
};

inline MaterialParameters makeMaterialParameters()
{
    MaterialParameters params;
    params.useOcclusion = 0;
    params.volumeDensityScale = 1;
    params.volumeTemperatureScale = 1;
    params.transmittanceMethod = VolumeTransmittanceMethod::kRatioTracking;
    params.transmittanceThreshold = 0.01f;
    params.phase = 0.f;
    params.maxPathDepth = 4;
    params.volumeAlbedo = 0.5f;
    params.interpolationOrder = 0;
    params.voxelGeometry = 0;
    return params;
}

struct RenderStatistics
{
    float mDuration;
};

/// @brief A minimal perspective camera for ray generation
class Camera
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using RayT = nanovdb::Ray<RealT>;

public:
    enum class LensType : int { kPinHole = 0,
                                kSpherical,
                                kODS,
                                kNumTypes };

    /// @brief default Ctor.
    Camera() = default;

    /// @brief Ctor. // vfov is top to bottom in degrees
    /// @note  up is assumed to be a unit-vector
    __hostdev__ Camera(LensType type, const Vec3T& eye, const Vec3T& lookat, const Vec3T& up, RealT vfov, RealT aspect)
        : mLensType(type)
        , mEye(eye)
        , mW((eye - lookat).normalize())
        , mU(up.cross(mW))
        , mV(up)
        , mAspect(aspect)
        , mIpd(1.0f)
        , mFovY(vfov)
    {
    }

    /// @brief {u,v} are are assumed to be [0,1]
    __hostdev__ RayT getRay(RealT u, RealT v) const
    {
        if (mLensType == LensType::kPinHole) {
            const RealT halfHeight = RealT(tanf(mFovY * 3.14159265358979323846f / 360.f));
            const RealT halfWidth = mAspect * halfHeight;
            auto        W = halfWidth * mU + halfHeight * mV + mW;
            auto        U = mU * 2 * halfWidth;
            auto        V = mV * 2 * halfHeight;
            auto        rd = (u * U + v * V - W).normalize();
            return RayT(mEye, rd);
        } else if (mLensType == LensType::kSpherical) {
            v -= 0.5f;
            float theta = -u * (3.14165f * 2.f);
            float phi = v * (3.14165f);
            auto  rd = Vec3T(sinf(theta) * cosf(phi),
                            sinf(phi),
                            cosf(theta) * cosf(phi));

            auto camDir = mU * rd[0] + mV * rd[1] + mW * rd[2];
            auto camEye = mEye;
            return RayT(camEye, camDir.normalize());
        } else if (mLensType == LensType::kODS) {
            float s = mIpd * 0.5f;
            if (v > 0.5)
                s = -s;
            v = fmod(v, 0.5f) * 2.0f;

            v -= 0.5f;
            float theta = -u * (3.14165f * 2.f);
            float phi = v * (3.14165f);

            auto rd = Vec3T(sinf(theta) * cosf(phi),
                            sinf(phi),
                            cosf(theta) * cosf(phi));

            auto camDir = mU * rd[0] + mV * rd[1] + mW * rd[2];
            auto camEye = mEye + Vec3T(cos(theta) * s, 0, sin(theta) * s);
            return RayT(camEye, camDir.normalize());
        } else {
            return RayT(mEye, Vec3T(0, 0, -1));
        }
    }

    __hostdev__ const Vec3T& P() const { return mEye; }
    __hostdev__ const Vec3T& U() const { return mU; }
    __hostdev__ const Vec3T& V() const { return mV; }
    __hostdev__ const Vec3T& W() const { return mW; }

    __hostdev__ const LensType& lensType() const { return mLensType; }
    __hostdev__ LensType& lensType() { return mLensType; }

    __hostdev__ const float& ipd() const { return mIpd; }
    __hostdev__ float&       ipd() { return mIpd; }

    __hostdev__ const float& fov() const { return mFovY; }
    __hostdev__ float&       fov() { return mFovY; }

    __hostdev__ const float& aspect() const { return mAspect; }
    __hostdev__ float&       aspect() { return mAspect; }

    static Camera makeDefaultCamera()
    {
        Camera camera;
        camera.mLensType = LensType::kPinHole;
        camera.mAspect = 1.0f;
        camera.mIpd = 1.0f;
        camera.mFovY = 60.f;
        return camera;
    }

private:
    LensType mLensType;
    Vec3T    mEye, mW, mU, mV;
    float    mAspect;
    float    mIpd;
    float    mFovY;
}; // Camera

struct SceneRenderParameters
{
    float          groundHeight;
    float          groundFalloff;
    int            useTonemapping;
    float          tonemapWhitePoint;
    int            useBackground;
    int            useGround;
    int            useShadows;
    int            useLighting;
    int            useGroundReflections;
    int            samplesPerPixel;
    nanovdb::Vec3f sunDirection;
    Camera         camera;
};

inline SceneRenderParameters makeSceneRenderParameters()
{
    SceneRenderParameters params;
    params.groundHeight = 0;
    params.sunDirection = nanovdb::Vec3f(0,1,0);
    params.groundFalloff = 0;
    params.useTonemapping = false;
    params.tonemapWhitePoint = 1.5f;
    params.useBackground = 1;
    params.useGround = 1;
    params.useShadows = 1;
    params.useLighting = 1;
    params.useGroundReflections = 0;
    params.samplesPerPixel = 1;
    params.camera = Camera::makeDefaultCamera();
    return params;
}

struct GridRenderParameters
{
    nanovdb::BBox<nanovdb::Vec3R> bounds;
    void*                         gridHandle;
};
