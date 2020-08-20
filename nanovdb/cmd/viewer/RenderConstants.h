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

struct RendererAttributeParams
{
    int   attribute;
    float gain;
    float offset;
};

struct RenderConstants
{
    float                   groundHeight;
    float                   groundFalloff;
    float                   useLighting;
    float                   useGround;
    float                   useOcclusion;
    float                   useShadows;
    float                   useGroundReflections;
    int                     samplesPerPixel;
    float                   volumeDensity;
    bool                    useTonemapping;
    float                   tonemapWhitePoint;
    RendererAttributeParams attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::End];
};

struct RenderStatistics
{
    float mDuration;
};

/// @brief A minimal perspective camera for ray generation
template<typename RealT = float, typename Vec3T = nanovdb::Vec3<RealT>, typename RayT = nanovdb::Ray<RealT>>
class Camera
{
    Vec3T mEye, mW, mU, mV;

    __hostdev__ void init(RealT vfov, RealT aspect)
    {
        const RealT halfHeight = RealT(tan(vfov * 3.14159265358979323846 / 360));
        const RealT halfWidth = aspect * halfHeight;
        mW = halfWidth * mU + halfHeight * mV + mW; // remove eye here and in getRay
        mU *= 2 * halfWidth;
        mV *= 2 * halfHeight;
    }

public:
    /// @brief default Ctor.
    Camera() = default;

    /// @brief Ctor. // vfov is top to bottom in degrees
    /// @note  up is assumed to be a unit-vector
    __hostdev__ Camera(const Vec3T& eye, const Vec3T& lookat, const Vec3T& up, RealT vfov, RealT aspect)
        : mEye(eye)
        , mW((eye - lookat).normalize())
        , mU(up.cross(mW))
        , mV(up)
    {
        this->init(vfov, aspect);
    }
    __hostdev__ void update(const Vec3T& eye, const Vec3T& lookat, const Vec3T& up, RealT vfov, RealT aspect)
    {
        mEye = eye;
        mV = up;
        mW = mEye - lookat;
        mW.normalize();
        mU = mV.cross(mW);
        this->init(vfov, aspect);
    }
    /// @brief {u,v} are are assumed to be [0,1]
    __hostdev__ RayT getRay(RealT u, RealT v) const { return RayT(mEye, u * mU + v * mV - mW); }

    __hostdev__ const Vec3T& P() const { return mEye; }
    __hostdev__ const Vec3T& U() const { return mU; }
    __hostdev__ const Vec3T& V() const { return mV; }
    __hostdev__ const Vec3T& W() const { return mW; }

}; // Camera
