// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_UTILITIES_RENDER_HAS_BEEN_INCLUDED
#define OPENVDBLINK_UTILITIES_RENDER_HAS_BEEN_INCLUDED

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/RayTracer.h>


/* openvdbmma::render members

 class MMADiffuseShader
 class MMAMatteShader
 class MMANormalShader
 class MMAPositionShader
 class DepthShader
 class PBRShader

 class RenderGridMma

 public members are

 renderImage
 a bunch of setters

*/


namespace openvdbmma {
namespace render {

//////////// custom shaders

template<typename GridT = Film::RGBA,
typename SamplerType = tools::PointSampler>
class DepthShader: public BaseShader
{
public:
    DepthShader(const Vec3R normal, const Vec3R point, const Vec2R dspan,
        const GridT& grid1, const GridT& grid2, const GridT& grid3, const bool is_closed = false,
        const Vec2R ispan = Vec2R(0.0, 1.0), const float gamma = 1.0f)
    : mNormal(normal.unitSafe()), mPoint(point), mDMin(dspan[0]), mDMax(dspan[1])
    , mIMin(ispan[0]), mIMax(ispan[1]), mGamma(gamma)
    , mInvDSpan(dspan[0] != dspan[1] ? 1.0f/(dspan[1]-dspan[0]) : 1.0f)
    , mInvISpan(ispan[0] != ispan[1] ? 1.0f/(ispan[1]-ispan[0]) : 1.0f)
    , is_closed(is_closed)
    , mAccFront(grid1.getAccessor()), mAccBack(grid2.getAccessor()), mAccClosed(grid3.getAccessor())
    , mXformFront(&grid1.transform()), mXformBack(&grid2.transform()), mXformClosed(&grid3.transform())
    {
    }

    DepthShader(const DepthShader&) = default;
    ~DepthShader() = default;

    Film::RGBA operator()(const Vec3R& xyz, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const float cos = normal.dot(rayDir);
        typename GridT::ValueType v = zeroVal<typename GridT::ValueType>();

        if (cos <= 0) {
            SamplerType::sample(mAccFront, mXformFront->worldToIndex(xyz), v);
        } else if (is_closed) {
            SamplerType::sample(mAccClosed, mXformClosed->worldToIndex(xyz), v);
        } else {
            SamplerType::sample(mAccBack, mXformBack->worldToIndex(xyz), v);
        }

        const float dist = cos <= 0 || !is_closed ? math::Abs(mNormal.dot(xyz - mPoint)) : mDMin;
        const float clampedDist = math::Clamp01((mDMax - dist) * mInvDSpan);
        const float rescaledDist =
            math::Clamp01((math::Pow(clampedDist, mGamma) - mIMin) * mInvISpan);
        v *= rescaledDist;

        return Film::RGBA(v[0],v[1],v[2]);
    }

    BaseShader* copy() const override { return new DepthShader<GridT, SamplerType>(*this); }

private:
    const float mDMin, mDMax, mInvDSpan, mIMin, mIMax, mInvISpan, mGamma;
    const Vec3R mNormal, mPoint;
    typename GridT::ConstAccessor mAccFront, mAccBack, mAccClosed;
    const math::Transform *mXformFront, *mXformBack, *mXformClosed;
    const bool is_closed;
};

template<typename SamplerType>
class DepthShader<Film::RGBA, SamplerType>: public BaseShader
{
public:
    DepthShader(const Vec3R normal, const Vec3R point, const Vec2R dspan,
        const Film::RGBA& c1 = Film::RGBA(1.0f), const Film::RGBA& c2 = Film::RGBA(1.0f),
        const Film::RGBA& c3 = Film::RGBA(1.0f), const bool is_closed = false,
        const Vec2R ispan = Vec2R(0.0, 1.0), const float gamma = 1.0f)
    : mNormal(normal.unitSafe()), mPoint(point), mDMin(dspan[0]), mDMax(dspan[1])
    , mRGBAFront(c1), mRGBABack(c2), mRGBAClosed(c3), is_closed(is_closed)
    , mIMin(ispan[0]), mIMax(ispan[1]), mGamma(gamma)
    , mInvDSpan(dspan[0] != dspan[1] ? 1.0f/(dspan[1]-dspan[0]) : 1.0f)
    , mInvISpan(ispan[0] != ispan[1] ? 1.0f/(ispan[1]-ispan[0]) : 1.0f)
    {
    }

    DepthShader(const DepthShader&) = default;
    ~DepthShader() = default;

    Film::RGBA operator()(const Vec3R& xyz, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const float cos = normal.dot(rayDir);

        const float dist = cos <= 0 || !is_closed ? math::Abs(mNormal.dot(xyz - mPoint)) : mDMin;
        const float clampedDist = math::Clamp01((mDMax - dist) * mInvDSpan);
        const float rescaledDist =
            math::Clamp01((math::Pow(clampedDist, mGamma) - mIMin) * mInvISpan);

        const Film::RGBA color = cos <= 0 ? mRGBAFront : (is_closed ? mRGBAClosed : mRGBABack);

        return Film::RGBA(rescaledDist) * color;
    }

    BaseShader* copy() const override { return new DepthShader<Film::RGBA, SamplerType>(*this); }

private:
    const float mDMin, mDMax, mInvDSpan, mIMin, mIMax, mInvISpan, mGamma;
    const Vec3R mNormal, mPoint;
    const Film::RGBA mRGBAFront, mRGBABack, mRGBAClosed;
    const bool is_closed;
};


template<typename GridT = Film::RGBA,
typename SamplerType = tools::PointSampler>
class MMADiffuseShader: public BaseShader
{
public:
    MMADiffuseShader(const GridT& grid1, const GridT& grid2, const GridT& grid3, const bool is_closed)
    : mAccFront(grid1.getAccessor()), mXformFront(&grid1.transform())
    , mAccBack(grid2.getAccessor()), mXformBack(&grid2.transform())
    , mAccClosed(grid3.getAccessor()), mXformClosed(&grid3.transform())
    , is_closed(is_closed)
    {
    }

    MMADiffuseShader(const MMADiffuseShader&) = default;
    ~MMADiffuseShader() = default;

    Film::RGBA operator()(const Vec3R& xyz, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const float cos = normal.dot(rayDir);
        const float cos2 = cos > 0 && is_closed ? 1.0 : cos;
        typename GridT::ValueType v = zeroVal<typename GridT::ValueType>();

        if (cos <= 0) {
            SamplerType::sample(mAccFront, mXformFront->worldToIndex(xyz), v);
        } else if (is_closed) {
            SamplerType::sample(mAccClosed, mXformClosed->worldToIndex(xyz), v);
        } else {
            SamplerType::sample(mAccBack, mXformBack->worldToIndex(xyz), v);
        }

        return Film::RGBA(v[0],v[1],v[2]) * static_cast<Film::RGBA::ValueT>(math::Abs(cos2));
    }
    BaseShader* copy() const override
    {
        return new MMADiffuseShader<GridT, SamplerType>(*this);
    }

private:
    typename GridT::ConstAccessor mAccFront, mAccBack, mAccClosed;
    const math::Transform *mXformFront, *mXformBack, *mXformClosed;
    const bool is_closed;
};

template <typename SamplerType>
class MMADiffuseShader<Film::RGBA, SamplerType>: public BaseShader
{
public:
    MMADiffuseShader(const Film::RGBA& c1 = Film::RGBA(1.0f), const Film::RGBA& c2 = Film::RGBA(1.0f),
        const Film::RGBA& c3 = Film::RGBA(1.0f), const bool is_closed = false)
    : mRGBAFront(c1), mRGBABack(c2), mRGBAClosed(c3), is_closed(is_closed)
    {
    }

    MMADiffuseShader(const MMADiffuseShader&) = default;
    ~MMADiffuseShader() = default;

    Film::RGBA operator()(const Vec3R&, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const float cos = normal.dot(rayDir);
        const float cos2 = cos > 0 && is_closed ? 1.0 : cos;

        return (cos <= 0 ? mRGBAFront : (is_closed ? mRGBAClosed : mRGBABack)) *
            static_cast<Film::RGBA::ValueT>(math::Abs(cos2));
    }

    BaseShader* copy() const override
    {
        return new MMADiffuseShader<Film::RGBA, SamplerType>(*this);
    }

private:
    const Film::RGBA mRGBAFront, mRGBABack, mRGBAClosed;
    const bool is_closed;
};


template<typename GridT = Film::RGBA,
typename SamplerType = tools::PointSampler>
class MMAMatteShader: public BaseShader
{
public:
    MMAMatteShader(const GridT& grid1, const GridT& grid2, const GridT& grid3, const bool is_closed)
    : mAccFront(grid1.getAccessor()), mXformFront(&grid1.transform())
    , mAccBack(grid2.getAccessor()), mXformBack(&grid2.transform())
    , mAccClosed(grid3.getAccessor()), mXformClosed(&grid3.transform())
    , is_closed(is_closed)
    {
    }

    MMAMatteShader(const MMAMatteShader&) = default;
    ~MMAMatteShader() = default;

    Film::RGBA operator()(const Vec3R& xyz, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const float cos = normal.dot(rayDir);
        typename GridT::ValueType v = zeroVal<typename GridT::ValueType>();

        if (cos <= 0) {
            SamplerType::sample(mAccFront, mXformFront->worldToIndex(xyz), v);
        } else if (is_closed) {
            SamplerType::sample(mAccClosed, mXformClosed->worldToIndex(xyz), v);
        } else {
            SamplerType::sample(mAccBack, mXformBack->worldToIndex(xyz), v);
        }

        return Film::RGBA(v[0],v[1],v[2]);
    }

    BaseShader* copy() const override { return new MMAMatteShader<GridT, SamplerType>(*this); }

private:
    typename GridT::ConstAccessor mAccFront, mAccBack, mAccClosed;
    const math::Transform *mXformFront, *mXformBack, *mXformClosed;
    const bool is_closed;
};

template<typename SamplerType>
class MMAMatteShader<Film::RGBA, SamplerType>: public BaseShader
{
public:
    MMAMatteShader(const Film::RGBA& c1 = Film::RGBA(1.0f),
        const Film::RGBA& c2 = Film::RGBA(1.0f), const Film::RGBA& c3 = Film::RGBA(1.0f),
        const bool is_closed = false)
    : mRGBAFront(c1), mRGBABack(c2), mRGBAClosed(c3), is_closed(is_closed)
    {
    }

    MMAMatteShader(const MMAMatteShader&) = default;
    ~MMAMatteShader() = default;

    Film::RGBA operator()(const Vec3R&, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        return normal.dot(rayDir) <= 0 ? mRGBAFront : (is_closed ? mRGBAClosed : mRGBABack);
    }

    BaseShader* copy() const override
    {
        return new MMAMatteShader<Film::RGBA, SamplerType>(*this);
    }

private:
    const Film::RGBA mRGBAFront, mRGBABack, mRGBAClosed;
    const bool is_closed;
};


template<typename GridT = Film::RGBA,
         typename SamplerType = tools::PointSampler>
class MMANormalShader: public BaseShader
{
public:
    MMANormalShader(const GridT& grid, const bool is_closed)
    : mAcc(grid.getAccessor()), mXform(&grid.transform())
    , is_open(!is_closed)
    {
    }

    MMANormalShader(const MMANormalShader&) = default;
    ~MMANormalShader() = default;

    Film::RGBA operator()(const Vec3R& xyz, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const Vec3R nrml = normal.dot(rayDir) <= 0 || is_open ? normal : -rayDir;

        typename GridT::ValueType v = zeroVal<typename GridT::ValueType>();
        SamplerType::sample(mAcc, mXform->worldToIndex(xyz), v);

        return Film::RGBA(v[0]*(nrml[0]+1.0), v[1]*(nrml[1]+1.0), v[2]*(nrml[2]+1.0));
    }

    BaseShader* copy() const override { return new MMANormalShader<GridT, SamplerType>(*this); }

private:
    typename GridT::ConstAccessor mAcc;
    const math::Transform* mXform;
    const bool is_open;
};

template<typename SamplerType>
class MMANormalShader<Film::RGBA, SamplerType>: public BaseShader
{
public:
    MMANormalShader(const Film::RGBA& c = Film::RGBA(1.0f), const bool is_closed = false)
    : mRGBA(c*0.5f), is_open(!is_closed)
    {
    }

    MMANormalShader(const MMANormalShader&) = default;
    ~MMANormalShader() = default;

    Film::RGBA operator()(const Vec3R&, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const Vec3R nrml = normal.dot(rayDir) <= 0 || is_open ? normal : -rayDir;

        return mRGBA * Film::RGBA(nrml[0] + 1.0, nrml[1] + 1.0, nrml[2] + 1.0);
    }

    BaseShader* copy() const override
    {
        return new MMANormalShader<Film::RGBA, SamplerType>(*this);
    }

private:
    const Film::RGBA mRGBA;
    const bool is_open;
};

template<typename GridT = Film::RGBA,
         typename SamplerType = tools::PointSampler>
class MMAPositionShader: public BaseShader
{
public:
    MMAPositionShader(const math::BBox<Vec3R>& bbox, const Vec3R& cpt,
        const GridT& grid, const bool is_closed)
    : mMin(bbox.min()), mClipPt(cpt)
    , mInvDim(1.0/bbox.extents())
    , mAcc(grid.getAccessor())
    , mXform(&grid.transform())
    , is_open(!is_closed)
    {
    }

    MMAPositionShader(const MMAPositionShader&) = default;
    ~MMAPositionShader() override = default;

    Film::RGBA operator()(const Vec3R& xyz, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const Vec3R pt = normal.dot(rayDir) <= 0 || is_open ? xyz : plane_nearest(rayDir, xyz);

        typename GridT::ValueType v = zeroVal<typename GridT::ValueType>();
        SamplerType::sample(mAcc, mXform->worldToIndex(pt), v);
        const Vec3R rgb = (pt - mMin) * mInvDim;

        return Film::RGBA(v[0],v[1],v[2]) * Film::RGBA(rgb[0], rgb[1], rgb[2]);
    }

    BaseShader* copy() const override
    {
        return new MMAPositionShader<GridT, SamplerType>(*this);
    }

private:
    const Vec3R mMin, mInvDim, mClipPt;
    typename GridT::ConstAccessor mAcc;
    const math::Transform* mXform;
    const bool is_open;

    inline Vec3R plane_nearest(const Vec3R& n, const Vec3R& c) const
    {
        return c + n*(n.dot(mClipPt - c));
    }
};

template<typename SamplerType>
class MMAPositionShader<Film::RGBA, SamplerType>: public BaseShader
{
public:
    MMAPositionShader(const math::BBox<Vec3R>& bbox, const Vec3R& cpt,
        const Film::RGBA& c = Film::RGBA(1.0f), const bool is_closed = false)
    : mMin(bbox.min()), mInvDim(1.0/bbox.extents()), mClipPt(cpt), mRGBA(c), is_open(!is_closed)
    {
    }

    MMAPositionShader(const MMAPositionShader&) = default;
    ~MMAPositionShader() override = default;

    Film::RGBA operator()(const Vec3R& xyz, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const Vec3R pt = normal.dot(rayDir) <= 0 || is_open ? xyz : plane_nearest(rayDir, xyz);

        const Vec3R rgb = (pt - mMin)*mInvDim;

        return mRGBA*Film::RGBA(rgb[0], rgb[1], rgb[2]);
    }

    BaseShader* copy() const override
    {
        return new MMAPositionShader<Film::RGBA, SamplerType>(*this);
    }

private:
    const Vec3R mMin, mInvDim, mClipPt;
    const Film::RGBA mRGBA;
    const bool is_open;

    inline Vec3R plane_nearest(const Vec3R& n, const Vec3R& c) const
    {
        return c + n*(n.dot(mClipPt - c));
    }
};


template<typename GridT = Film::RGBA,
typename SamplerType = tools::PointSampler>
class PBRShader: public BaseShader
{
public:
    PBRShader(const float& rough, const float& ani, const float& albedo, const float& f, const float& sdw, const float& fs, const float& fd,
              const Vec3R& vp, const Vec3R& vv, const GridT& gridfront, const GridT& gridback)
    : mAccFront(gridfront.getAccessor()), mXformFront(&gridfront.transform())
    , mAccBack(gridback.getAccessor()), mXformBack(&gridback.transform())
    {
    }

    PBRShader(const PBRShader&) = default;
    ~PBRShader() = default;

    Film::RGBA operator()(const Vec3R& xyz, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const float cos = normal.dot(rayDir);
        typename GridT::ValueType v = zeroVal<typename GridT::ValueType>();

        if (cos <= 0) {
            SamplerType::sample(mAccFront, mXformFront->worldToIndex(xyz), v);
        } else {
            SamplerType::sample(mAccBack, mXformBack->worldToIndex(xyz), v);
        }

        return Film::RGBA(v[0],v[1],v[2]) * static_cast<Film::RGBA::ValueT>(math::Abs(cos));
    }
    BaseShader* copy() const override
    {
        return new PBRShader<GridT, SamplerType>(*this);
    }

private:
    typename GridT::ConstAccessor mAccFront, mAccBack;
    const math::Transform *mXformFront, *mXformBack;
};

template <typename SamplerType>
class PBRShader<Film::RGBA, SamplerType>: public BaseShader
{
public:
    PBRShader(
        const Vec3R& baseColorFront, const Vec3R& baseColorBack, const Vec3R& baseColorClosed,
        const float& metallic, const float& rough, const float& ani, const float& ref,
        const Vec3R& coatColor, const float& coatRough, const float& coatAni, const float& coatRef,
        const float& fac_spec, const float& fac_diff, const float& fac_coat,
        const Vec3R& vp, const Vec3R& vv, const bool& is_closed
    )
    // specular
    : f0Front(fresnel_f0(baseColorFront, metallic, ref))
    , f0Back(fresnel_f0(baseColorBack, metallic, ref))
    , f0Closed(fresnel_f0(baseColorClosed, metallic, ref))
    , alpha(math::Pow2(math::Max(rough, 1e-3f)))
    , ax(alpha_tan(rough, ani))
    , ay(alpha_bitan(rough, ani))

    // diffuse
    , diffuseColorFront((1 - metallic) * baseColorFront)
    , diffuseColorBack((1 - metallic) * baseColorBack)
    , diffuseColorClosed((1 - metallic) * baseColorClosed)

    // coat
    , coatColor(coatColor)
    , cf0(fresnel_f0(coatColor, 0.0, coatRef))
    , calpha(math::Pow2(math::Max(coatRough, 1e-3f)))
    , cax(alpha_tan(coatRough, coatAni))
    , cay(alpha_bitan(coatRough, coatAni))

    // factors for each reflection type
    , fac_spec(1.2*fac_spec)
    , fac_diff(fac_diff)
    , fac_coat(fac_coat)

    // lighting + extra settings
    , U((vv - (vv.dot(vp))/(vp.dot(vp))*vp).unitSafe())
    , keyDir(makeKeyDir(vp, vv))
    , fillDir(makeFillDir(vp, vv))
    , backDir(makeBackDir(vp, vv))
    , is_closed(is_closed)
    {
    }

    PBRShader(const PBRShader&) = default;
    ~PBRShader() = default;

    Film::RGBA operator()(const Vec3R&, const Vec3R& normal, const Vec3R& rayDir) const override
    {
        const Vec3R V = -rayDir;

        const float cos = normal.dot(rayDir);

        const Vec3R rel_normal   = cos <= 0 ? normal : (is_closed ? V : -normal);
        const Vec3R f0           = cos <= 0 ? f0Front : (is_closed ? f0Closed : f0Back);
        const Vec3R diffuseColor = cos <= 0 ? diffuseColorFront : (is_closed ? diffuseColorClosed : diffuseColorBack);

        const Vec3R radiance =
              0.85 * reflected_light(V, keyDir, rel_normal, diffuseColor, f0)
            + 0.40 * reflected_light(V, fillDir, rel_normal, diffuseColor, f0)
            + 0.10 * reflected_light(V, backDir, rel_normal, diffuseColor, f0);

        return Film::RGBA(
            math::Clamp01(radiance[0]),
            math::Clamp01(radiance[1]),
            math::Clamp01(radiance[2])
        );
    }

    BaseShader* copy() const override
    {
        return new PBRShader<Film::RGBA, SamplerType>(*this);
    }

private:

    const Vec3R diffuseColorFront, diffuseColorBack, diffuseColorClosed, coatColor;
    const Vec3R f0Front, f0Back, f0Closed, cf0;

    const float alpha, ax, ay;
    const float calpha, cax, cay;
    const float fac_spec, fac_diff, fac_coat;

    const Vec3R U, keyDir, fillDir, backDir;

    const bool is_closed;

    inline Vec3R fresnel_f0(const Vec3R& baseColor, const float& metallic, const float& reflectance)
    {
        return Vec3R(0.16 * math::Pow2(reflectance) * (1 - metallic)) + baseColor * metallic;
    }

    inline float alpha_tan(const float& rough, const float& ani) const
    {
        return math::Max(math::Pow2(rough)/math::Sqrt(1.0 - 0.9*ani), 1e-5);
    }

    inline float alpha_bitan(const float& rough, const float& ani) const
    {
        return math::Max(math::Pow2(rough) * math::Sqrt(1.0 - 0.9*ani), 1e-5);
    }

#define PI 3.141592653589793

    inline Vec3R parallelRotate(const Vec3R& vp, const Vec3R& ivv, const float theta) {
        const Vec3R vv = ivv.unitSafe();
        const float cos = math::Cos(theta), sin = math::Sin(theta);

        const float vp1 = vp[0], vp2 = vp[1], vp3 = vp[2];
        const float vv1 = vv[0], vv2 = vv[1], vv3 = vv[2];

        return Vec3R(
            0.5*(2*sin*(vp3*vv2 - vp2*vv3) + 2*vv1*(vp2*vv2 + vp3*vv3) + vp1*(1 + vv1*vv1 - vv2*vv2 - vv3*vv3) + cos*(vp1*(1 - vv1*vv1 + vv2*vv2 + vv3*vv3) - 2*vv1*(vp2*vv2 + vp3*vv3))),
            sin*(vp1*vv3 - vp3*vv1) + vv2*(vp1*vv1 + vp3*vv3) - 0.5*vp2*(vv1*vv1 - vv2*vv2 + vv3*vv3 - 1) + 0.5*cos*(vp2*(1 + vv1*vv1 - vv2*vv2 + vv3*vv3) - 2*vv2*(vp1*vv1 + vp3*vv3)),
            0.5*(2*sin*(vp2*vv1 - vp1*vv2) - vp3*(vv1*vv1 + vv2*vv2 - 1) + 2*(vp1*vv1 + vp2*vv2)*vv3 + vp3*vv3*vv3 + cos*(vp3*(1 + vv1*vv1 + vv2*vv2 - vv3*vv3) - 2*(vp1*vv1 + vp2*vv2)*vv3))
        ).unitSafe();
    }

    inline Vec3R makeKeyDir(const Vec3R& vp, const Vec3R& vv) {
        return -parallelRotate(parallelRotate(vp, vv, 0.5), vp.cross(vv), -0.5);
    }

    inline Vec3R makeFillDir(const Vec3R& vp, const Vec3R& vv) {
        return -parallelRotate(parallelRotate(vp, vv, -0.4), vp.cross(vv), 0.6);
    }

    inline Vec3R makeBackDir(const Vec3R& vp, const Vec3R& vv) {
        return -parallelRotate(parallelRotate(vp, vv, 0.2), vp.cross(vv), -2.25);
    }

    inline float lerp(const float& v0, const float& v1, const float& t) const
    {
        return v0 + t*(v1 - v0);
    }

    Vec3R reflected_light(const Vec3R& V, const Vec3R& L, const Vec3R& N, const Vec3R& diffuseColor, const Vec3R& f0) const
    {
        /* ------------ common parameters between diffuse and specular ------------ */

        const Vec3R H = (V + L).unitSafe();

        const float NoV = math::Abs(N.dot(V)) + 1e-5;
        const float NoL = math::Clamp01(N.dot(L));
        const float NoH = math::Clamp01(N.dot(H));
        const float LoH = math::Clamp01(L.dot(H));

        const Vec3R Y = (U - (U.dot(N))*N).unitSafe();
        const Vec3R X = N.cross(Y);

        const float HoX = H.dot(X);
        const float HoY = H.dot(Y);

        /* ------------ diffuse reflection ------------ */

        const Vec3R brdf_diff = 1.05 * diffuseColor/PI * (1 - math::Pow(1 - LoH, 5));

        /* ------------ specular reflection ------------ */

        const float D_fac = math::Pow2(NoH) + math::Pow2(HoX/ax) + math::Pow2(HoY/ay);
        const float D = 1.0/(ax * ay * PI * math::Pow2(D_fac));

        const float G = 0.5/lerp(2.0*NoV*NoL, NoV + NoL, alpha);

        const float pow5 = math::Pow(1.0 - LoH, 5);
        const Vec3R F = Vec3R(pow5) + f0 * (1.0 - pow5);

        const Vec3R brdf_spec = (D * G) * F;

        /* ------------ clear coat ------------ */

        const float Dc_fac = math::Pow2(NoH) + math::Pow2(HoX/cax) + math::Pow2(HoY/cay);
        const float Dc = 1.0/(cax * cay * PI * math::Pow2(Dc_fac));

        const float Gc = 0.25/math::Pow2(LoH);

        const Vec3R Fc = (Vec3R(pow5) + cf0 * (1.0 - pow5)) * fac_coat;

        const Vec3R brdf_coat = (Dc * Gc) * Fc * coatColor;

        /* ------------ combination of brdfs ------------ */

        const Vec3R object_brdf = fac_diff * brdf_diff + fac_spec * brdf_spec;

        const Vec3R brdf_all = object_brdf*(Vec3R(1.0) - Fc) + brdf_coat;

        return brdf_all * PI * NoL;
    }

#undef PI

};


//////////// rendering class

// TODO figure out how to not need both ColorT and ColorPtr
template<typename GridT, typename ColorT = tools::Film::RGBA,
    typename ColorPtr = std::shared_ptr<tools::Film::RGBA>>
class RenderGridMma
{
public:

    using GridPtr = typename GridT::Ptr;

    RenderGridMma(GridPtr grid, int w, int h)
    : mGrid(grid)
    {
        setWidth(w);
        setHeight(h);
    }

    ~RenderGridMma() {}

    mma::ImageRef<mma::im_byte_t> renderImage() const
    {
        tools::Film film(mOpts.width, mOpts.height, mOpts.background);

        std::unique_ptr<tools::BaseCamera> camera;
        if (mOpts.camera == CAMERA_PERSPECTIVE) {
            camera.reset(new tools::PerspectiveCamera(film, mOpts.rotate, mOpts.translate,
                mOpts.focal, mOpts.aperture, mOpts.rng_near, mOpts.rng_far));
        } else {
            camera.reset(new tools::OrthographicCamera(film, mOpts.rotate, mOpts.translate,
                mOpts.frame, mOpts.rng_near, mOpts.rng_far));
        }
        camera->lookAt(mOpts.lookat, mOpts.up);

        if (mGrid->getGridClass() != GRID_FOG_VOLUME) {
            renderLevelSet(camera);
        } else {
            renderFogVolume(camera);
        }

        return filmImage(film);
    }

    //////////// parameter setters

    inline void setIsoValue(float isovalue) { mOpts.isovalue = isovalue; }
    inline void setFrame(float frame)
    {
        if (frame <= 0.0)
            throw mma::LibraryError(LIBRARY_NUMERICAL_ERROR);

        mOpts.frame = frame;
    }

    inline void setColor(ColorPtr color) { mOpts.color = color; }
    inline void setColor2(ColorPtr color2) { mOpts.color2 = color2; }
    inline void setColor3(ColorPtr color3) { mOpts.color3 = color3; }
    inline void setBackground(tools::Film::RGBA background) { mOpts.background = background; }

    inline void setTranslate(mma::RealVectorRef translate)
    {
        if (!validViewingVector(translate))
            throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

        mOpts.translate = Vec3d(translate[0], translate[1], translate[2]);
    }
    inline void setLookAt(mma::RealVectorRef lookat)
    {
        if (!validViewingVector(lookat))
            throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

        mOpts.lookat = Vec3d(lookat[0], lookat[1], lookat[2]);
    }
    inline void setUp(mma::RealVectorRef up)
    {
        if (!validViewingVector(up))
            throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

        mOpts.up = Vec3d(up[0], up[1], up[2]);
    }

    inline void setRange(mma::RealVectorRef range)
    {
        if (!validViewRange(range))
            throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

        setNear(range[0]);
        setFar(range[1]);
    }
    inline void setNear(float rng_near) { mOpts.rng_near = rng_near; }
    inline void setFar(float rng_far) { mOpts.rng_far = rng_far; }

    inline void setFOV(mma::RealVectorRef fov)
    {
        if (!validFOV(fov))
            throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

        setAperture(fov[0]);
        setFocal(fov[1]);
    }
    inline void setAperture(float aperture) { mOpts.aperture = aperture; }
    inline void setFocal(float focal) { mOpts.focal = focal; }

    inline void setIsClosed(bool is_closed) { mOpts.is_closed = is_closed; }

    inline void setShader(mint shader)
    {
        if (!validShader(shader))
            throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

        mOpts.shader = shader;
    }

    inline void setCamera(mint camera)
    {
        if (!validCamera(camera))
            throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

        mOpts.camera = camera;
    }

    inline void setResolution(mma::IntVectorRef resolution)
    {
        if (!validImageResolution(resolution))
            throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

        setWidth(resolution[0]);
        setHeight(resolution[1]);
    }
    inline void setWidth(int width) { mOpts.width = width; }
    inline void setHeight(int height) { mOpts.height = height; }

    inline void setSamples(mint samples)
    {
        if (samples <= 0)
            throw mma::LibraryError(LIBRARY_NUMERICAL_ERROR);

        mOpts.samples = samples;
    }
    inline void setStep(mma::RealVectorRef step)
    {
        if (!validStep(step))
            throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

        mOpts.step = Vec2d(step[0], step[1]);
    }
    inline void setLightDir(mma::RealVectorRef lightdir)
    {
        if (!validLightDirection(lightdir))
            throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

        mOpts.lightdir = Vec3d(lightdir[0], lightdir[1], lightdir[2]);
    }

    inline void setDepthParameters(mma::RealVectorRef depthParams)
    {
        if (!validDepthParameters(depthParams))
            throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

        setMinIntensity(depthParams[0]);
        setMaxIntensity(depthParams[1]);
        setGamma(depthParams[2]);
    }
    inline void setMinIntensity(float min_intensity) { mOpts.min_intensity = min_intensity; }
    inline void setMaxIntensity(float max_intensity) { mOpts.max_intensity = max_intensity; }
    inline void setGamma(float gamma) { mOpts.gamma = gamma; }

    inline void setPBRShader(
        Vec3R baseColorFront, Vec3R baseColorBack, Vec3R baseColorClosed,
        float metallic, float rough, float ani, float ref,
        Vec3R coatColor, float coatRough, float coatAni, float coatRef,
        float fac_spec, float fac_diff, float fac_coat, bool is_closed)
    {
        mOpts.baseColorFront  = baseColorFront;
        mOpts.baseColorBack   = baseColorBack;
        mOpts.baseColorClosed = baseColorClosed;

        mOpts.shader   = SHADER_PBR;
        mOpts.metallic = metallic;
        mOpts.rough    = rough;
        mOpts.ani      = ani;
        mOpts.ref      = ref;

        mOpts.coatColor = coatColor;
        mOpts.coatRough = coatRough;
        mOpts.coatAni   = coatAni;
        mOpts.coatRef   = coatRef;

        mOpts.fac_spec   = fac_spec;
        mOpts.fac_diff   = fac_diff;
        mOpts.fac_coat   = fac_coat;
        mOpts.is_closed  = is_closed;
    }

private:

    //////////// setting enumerations

    enum {
        SHADER_DIFFUSE = 0,
        SHADER_MATTE,
        SHADER_NORMAL,
        SHADER_POSITION,
        SHADER_DEPTH,
        SHADER_PBR
    };

    enum {
        CAMERA_PERSPECTIVE = 0,
        CAMERA_ORTHOGRAPHIC
    };

    ///////////// verify parameters are valid before setting them

    inline bool validShader(const mint& shader) const
    {
        return SHADER_DIFFUSE <= shader && shader <= SHADER_PBR;
    }

    inline bool validCamera(const mint& camera) const
    {
        return camera == CAMERA_PERSPECTIVE || camera == CAMERA_ORTHOGRAPHIC;
    }

    inline bool validViewingVector(const mma::RealVectorRef& vec) const
    {
        return vec.size() == 3;
    }

    inline bool validImageResolution(const mma::IntVectorRef& resolution) const
    {
        return resolution.size() == 2 && resolution[0] > 0 && resolution[1] > 0;
    }

    inline bool validViewRange(const mma::RealVectorRef& range) const
    {
        return range.size() == 2 && range[0] > 0 && range[0] <= range[1];
    }

    inline bool validFOV(const mma::RealVectorRef& fov) const
    {
        return fov.size() == 2 && fov[0] > 0 && fov[1] > 0;
    }

    inline bool validLightDirection(const mma::RealVectorRef& lightdir) const
    {
        return lightdir.size() == 3 &&
            (lightdir[0] != 0.0 || lightdir[1] != 0.0 || lightdir[2] != 0.0);
    }

    inline bool validStep(const mma::RealVectorRef& step) const
    {
        return step.size() == 2 && step[0] > 0 && step[1] > 0;
    }

    inline bool validDepthParameters(const mma::RealVectorRef& irange) const
    {
        return irange.size() == 3 && 0 <= irange[0] && irange[0] <= irange[1] &&
            irange[1] <= 1 && irange[2] > 0;
    }

    ///////////// rendering functionality

    inline unsigned char float_to_byte(const float& x) const
    {
        return static_cast<unsigned char>(255*x);
    }

    void renderLevelSet(std::unique_ptr<tools::BaseCamera>& camera) const;
    void renderFogVolume(std::unique_ptr<tools::BaseCamera>& camera) const;

    mma::ImageRef<mma::im_byte_t> filmImage(tools::Film& film) const;

    struct RenderOpts
    {
        float aperture, focal, frame, isovalue, rng_near, rng_far, gamma, min_intensity, max_intensity;
        ColorPtr color, color2, color3;
        tools::Film::RGBA background;
        Vec3d rotate, translate, lookat, up, lightdir;
        Vec2d step;
        int samples, shader, camera, width, height;

        // PBR settings
        Vec3R baseColorFront, baseColorBack, baseColorClosed, coatColor;
        float rough, metallic, ani, ref;
        float coatRough, coatAni, coatRef;
        float fac_spec, fac_diff, fac_coat;
        bool is_closed;

        RenderOpts():
        aperture(41.2136f),
        focal(50.0f),
        frame(1.0f),
        isovalue(0.0),
        rng_near(1.0e-3f),
        rng_far(std::numeric_limits<float>::max()),
        gamma(1.0),
        min_intensity(0.0),
        max_intensity(1.0),
        background(tools::Film::RGBA(1.0f, 1.0f, 1.0f, 1.0f)),
        rotate(0.0),
        translate(0.0),
        lookat(0.0),
        up(0.0, 0.0, 0.1),
        lightdir(1.0, 1.0, 1.0),
        samples(1),
        step(Vec2d(1.0, 3.0)),
        shader(SHADER_DIFFUSE),
        camera(CAMERA_PERSPECTIVE),
        width(1920),
        height(1080)
        {}
    };

    //////////// private members

    RenderOpts mOpts;

    GridPtr mGrid;

}; // end of RenderGridMma class


//////////// RenderGridMma private member function definitions

template<typename GridT, typename ColorT, typename ColorPtr>
inline void
RenderGridMma<GridT, ColorT, ColorPtr>::renderLevelSet(std::unique_ptr<tools::BaseCamera>& camera) const
{
    std::unique_ptr<tools::BaseShader> shader;
    switch (mOpts.shader) {
        case SHADER_DIFFUSE:
        {
            shader.reset(new MMADiffuseShader<ColorT>(*mOpts.color, *mOpts.color2,
                *mOpts.color3, mOpts.is_closed));
            break;
        }
        case SHADER_PBR:
        {
            shader.reset(new PBRShader<ColorT>(
                mOpts.baseColorFront, mOpts.baseColorBack, mOpts.baseColorClosed,
                mOpts.metallic, mOpts.rough, mOpts.ani, mOpts.ref,
                mOpts.coatColor, mOpts.coatRough, mOpts.coatAni, mOpts.coatRef,
                mOpts.fac_spec, mOpts.fac_diff, mOpts.fac_coat,
                mOpts.lookat - mOpts.translate, mOpts.up, mOpts.is_closed));
            break;
        }
        case SHADER_MATTE:
        {
            shader.reset(new MMAMatteShader<ColorT>(*mOpts.color, *mOpts.color2,
                *mOpts.color3, mOpts.is_closed));
            break;
        }
        case SHADER_NORMAL:
        {
            shader.reset(new MMANormalShader<ColorT>(*mOpts.color, mOpts.is_closed));
            break;
        }
        case SHADER_POSITION:
        {
            const Vec3R cpt = mOpts.translate +
                mOpts.rng_near * ((mOpts.lookat - mOpts.translate).unitSafe());
            const CoordBBox bbox = mGrid->evalActiveVoxelBoundingBox();
            const math::BBox<Vec3d> bboxIndex(bbox.min().asVec3d(), bbox.max().asVec3d());
            const math::BBox<Vec3R> bboxWorld = bboxIndex.applyMap(*(mGrid->transform().baseMap()));
            shader.reset(new MMAPositionShader<ColorT>(bboxWorld, cpt, *mOpts.color, mOpts.is_closed));
            break;
        }
        default: // case SHADER_DEPTH:
        {
            shader.reset(new DepthShader<ColorT>(
                mOpts.translate - mOpts.lookat, mOpts.translate,
                Vec2d(mOpts.rng_near, mOpts.rng_far), *mOpts.color, *mOpts.color2,
                *mOpts.color3, mOpts.is_closed,
                Vec2d(mOpts.min_intensity, mOpts.max_intensity), mOpts.gamma));
            break;
        }
    }

    tools::LevelSetRayIntersector<GridT> intersector(*mGrid, mOpts.isovalue);
    tools::rayTrace(mGrid, intersector, *shader, *camera, mOpts.samples, 0, true);
}

// todo (How much more can be added?)
template<typename GridT, typename ColorT, typename ColorPtr>
inline void
RenderGridMma<GridT, ColorT, ColorPtr>::renderFogVolume(std::unique_ptr<tools::BaseCamera>& camera) const
{
    using IntersectorType = tools::VolumeRayIntersector<GridT>;
    IntersectorType intersector(*mGrid);

    tools::VolumeRender<IntersectorType> renderer(intersector, *camera);

    renderer.setLightDir(mOpts.lightdir[0], mOpts.lightdir[1], mOpts.lightdir[2]);
    renderer.setLightColor(1.0, 1.0, 1.0);
    renderer.setPrimaryStep(mOpts.step[0]);
    renderer.setShadowStep(mOpts.step[1]);
    renderer.setScattering(1.5, 1.5, 1.5);
    renderer.setAbsorption(0.1, 0.1, 0.1);
    renderer.setLightGain(0.2);
    renderer.setCutOff(0.005);

    renderer.render(true);
}

template<typename GridT, typename ColorT, typename ColorPtr>
mma::ImageRef<mma::im_byte_t>
RenderGridMma<GridT, ColorT, ColorPtr>::filmImage(tools::Film& film) const
{
    const int w = film.width(), h = film.height();

    mma::ImageRef<mma::im_byte_t> im = mma::makeImage<mma::im_byte_t>(w, h, 3);

    tbb::parallel_for(
        tbb::blocked_range<int>(0, h),
        [&](tbb::blocked_range<int> rng)
        {
            for (mint y = rng.begin(); y < rng.end(); ++y) {
                for (int x = w-1; x >= 0; x--) {
                    const tools::Film::RGBA pixel = film.pixel(x, y);
                    im(y, x, 0) = float_to_byte(pixel.r);
                    im(y, x, 1) = float_to_byte(pixel.g);
                    im(y, x, 2) = float_to_byte(pixel.b);
                }
            }
        }
    );

    return im;
}


inline tools::Film::RGBA mmaRGBToColor(mma::RGBRef color)
{
    return tools::Film::RGBA(color.r(), color.g(), color.b(), 1.0);
}

} // namespace render
} // namespace openvdbmma

#endif // OPENVDBLINK_UTILITIES_RENDER_HAS_BEEN_INCLUDED
