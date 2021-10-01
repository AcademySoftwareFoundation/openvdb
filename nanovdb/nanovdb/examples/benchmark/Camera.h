// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file Camera.h
///
/// @author Ken Museth
///
/// @brief A simple camera class.

#ifndef NANOVDB_CAMERA_H_HAS_BEEN_INCLUDED
#define NANOVDB_CAMERA_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h> // for Vec3
#include <nanovdb/util/Ray.h>

namespace nanovdb {

/// @brief A minimal perspective camera for ray generation
template<typename RealT = float, typename Vec3T = Vec3<RealT>, typename RayT = Ray<RealT>>
class Camera
{
    Vec3T            mEye, mW, mU, mV;
    
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
    __hostdev__ RayT getRay(RealT u, RealT v) const {
        auto dir = u * mU + v * mV - mW;
        dir.normalize();
        return RayT(mEye, dir); 
    }

    __hostdev__ const Vec3T& P() const { return mEye; }
    __hostdev__ const Vec3T& U() const { return mU; }
    __hostdev__ const Vec3T& V() const { return mV; }
    __hostdev__ const Vec3T& W() const { return mW; }

}; // Camera

} // namespace nanovdb

#endif // NANOVDB_CAMERA_HAS_BEEN_INCLUDED
