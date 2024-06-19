// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

//////////////////////////////////////////////////////////////////////////
///
/// @file SampleFromVoxels.h
///
/// @brief NearestNeighborSampler, TrilinearSampler, TriquadraticSampler and TricubicSampler
///
/// @note These interpolators employ internal caching for better performance when used repeatedly
///       in the same voxel location, so try to reuse an instance of these classes more than once.
///
/// @warning While all the interpolators defined below work with both scalars and vectors
///          values (e.g. float and Vec3<float>) TrilinarSampler::zeroCrossing and
///          Trilinear::gradient will only compile with floating point value types.
///
/// @author Ken Museth
///
///////////////////////////////////////////////////////////////////////////

#ifndef NANOVDB_SAMPLE_FROM_VOXELS_H_HAS_BEEN_INCLUDED
#define NANOVDB_SAMPLE_FROM_VOXELS_H_HAS_BEEN_INCLUDED

// Only define __hostdev__ when compiling as NVIDIA CUDA
#if defined(__CUDACC__) || defined(__HIP__)
#define __hostdev__ __host__ __device__
#else
#include <cmath> // for floor
#define __hostdev__
#endif

#include <nanovdb/math/Math.h>

namespace nanovdb {

namespace math {

// Forward declaration of sampler with specific polynomial orders
template<typename TreeT, int Order, bool UseCache = true>
class SampleFromVoxels;

/// @brief Factory free-function for a sampler of specific polynomial orders
///
/// @details This allows for the compact syntax:
/// @code
///   auto acc = grid.getAccessor();
///   auto smp = nanovdb::math::createSampler<1>( acc );
/// @endcode
template<int Order, typename TreeOrAccT, bool UseCache = true>
__hostdev__ SampleFromVoxels<TreeOrAccT, Order, UseCache> createSampler(const TreeOrAccT& acc)
{
    return SampleFromVoxels<TreeOrAccT, Order, UseCache>(acc);
}

/// @brief Utility function that returns the Coord of the round-down of @a xyz
///        and redefined @xyz as the fractional part, ie xyz-in = return-value + xyz-out
template<typename CoordT, typename RealT, template<typename> class Vec3T>
__hostdev__ inline CoordT Floor(Vec3T<RealT>& xyz);

/// @brief Template specialization of Floor for Vec3<float>
template<typename CoordT, template<typename> class Vec3T>
__hostdev__ inline CoordT Floor(Vec3T<float>& xyz)
{
    const float ijk[3] = {floorf(xyz[0]), floorf(xyz[1]), floorf(xyz[2])};
    xyz[0] -= ijk[0];
    xyz[1] -= ijk[1];
    xyz[2] -= ijk[2];
    return CoordT(int32_t(ijk[0]), int32_t(ijk[1]), int32_t(ijk[2]));
}

/// @brief Template specialization of Floor for Vec3<float>
template<typename CoordT, template<typename> class Vec3T>
__hostdev__ inline CoordT Floor(Vec3T<double>& xyz)
{
    const double ijk[3] = {floor(xyz[0]), floor(xyz[1]), floor(xyz[2])};
    xyz[0] -= ijk[0];
    xyz[1] -= ijk[1];
    xyz[2] -= ijk[2];
    return CoordT(int32_t(ijk[0]), int32_t(ijk[1]), int32_t(ijk[2]));
}

// ------------------------------> NearestNeighborSampler <--------------------------------------

/// @brief Nearest neighbor, i.e. zero order, interpolator with caching
template<typename TreeOrAccT>
class SampleFromVoxels<TreeOrAccT, 0, true>
{
public:
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

    static const int ORDER = 0;
    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ SampleFromVoxels(const TreeOrAccT& acc)
        : mAcc(acc)
        , mPos(CoordT::max())
    {
    }

    __hostdev__ const TreeOrAccT& accessor() const { return mAcc; }

    /// @note xyz is in index space space
    template<typename Vec3T>
    inline __hostdev__ ValueT operator()(const Vec3T& xyz) const;

    inline __hostdev__ ValueT operator()(const CoordT& ijk) const;

private:
    const TreeOrAccT& mAcc;
    mutable CoordT    mPos;
    mutable ValueT    mVal; // private cache
}; // SampleFromVoxels<TreeOrAccT, 0, true>

/// @brief Nearest neighbor, i.e. zero order, interpolator without caching
template<typename TreeOrAccT>
class SampleFromVoxels<TreeOrAccT, 0, false>
{
public:
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;
    static const int ORDER = 0;

    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ SampleFromVoxels(const TreeOrAccT& acc)
        : mAcc(acc)
    {
    }

    __hostdev__ const TreeOrAccT& accessor() const { return mAcc; }

    /// @note xyz is in index space space
    template<typename Vec3T>
    inline __hostdev__ ValueT operator()(const Vec3T& xyz) const;

    inline __hostdev__ ValueT operator()(const CoordT& ijk) const { return mAcc.getValue(ijk);}

private:
    const TreeOrAccT& mAcc;
}; // SampleFromVoxels<TreeOrAccT, 0, false>

template<typename TreeOrAccT>
template<typename Vec3T>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 0, true>::operator()(const Vec3T& xyz) const
{
    const CoordT ijk = math::Round<CoordT>(xyz);
    if (ijk != mPos) {
        mPos = ijk;
        mVal = mAcc.getValue(mPos);
    }
    return mVal;
}

template<typename TreeOrAccT>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 0, true>::operator()(const CoordT& ijk) const
{
    if (ijk != mPos) {
        mPos = ijk;
        mVal = mAcc.getValue(mPos);
    }
    return mVal;
}

template<typename TreeOrAccT>
template<typename Vec3T>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 0, false>::operator()(const Vec3T& xyz) const
{
    return mAcc.getValue(math::Round<CoordT>(xyz));
}

// ------------------------------> TrilinearSampler <--------------------------------------

/// @brief Tri-linear sampler, i.e. first order, interpolator
template<typename TreeOrAccT>
class TrilinearSampler
{
protected:
    const TreeOrAccT& mAcc;

public:
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;
    static const int ORDER = 1;

    /// @brief Protected constructor from a Tree or ReadAccessor
    __hostdev__ TrilinearSampler(const TreeOrAccT& acc) : mAcc(acc) {}

    __hostdev__ const TreeOrAccT& accessor() const { return mAcc; }

    /// @brief Extract the stencil of 8 values
    inline __hostdev__ void stencil(CoordT& ijk, ValueT (&v)[2][2][2]) const;

    template<typename RealT, template<typename...> class Vec3T>
    static inline __hostdev__ ValueT sample(const Vec3T<RealT> &uvw, const ValueT (&v)[2][2][2]);

    template<typename RealT, template<typename...> class Vec3T>
    static inline __hostdev__ Vec3T<ValueT> gradient(const Vec3T<RealT> &uvw, const ValueT (&v)[2][2][2]);

    static inline __hostdev__ bool zeroCrossing(const ValueT (&v)[2][2][2]);
}; // TrilinearSamplerBase

template<typename TreeOrAccT>
__hostdev__ void TrilinearSampler<TreeOrAccT>::stencil(CoordT& ijk, ValueT (&v)[2][2][2]) const
{
    v[0][0][0] = mAcc.getValue(ijk); // i, j, k

    ijk[2] += 1;
    v[0][0][1] = mAcc.getValue(ijk); // i, j, k + 1

    ijk[1] += 1;
    v[0][1][1] = mAcc.getValue(ijk); // i, j+1, k + 1

    ijk[2] -= 1;
    v[0][1][0] = mAcc.getValue(ijk); // i, j+1, k

    ijk[0] += 1;
    ijk[1] -= 1;
    v[1][0][0] = mAcc.getValue(ijk); // i+1, j, k

    ijk[2] += 1;
    v[1][0][1] = mAcc.getValue(ijk); // i+1, j, k + 1

    ijk[1] += 1;
    v[1][1][1] = mAcc.getValue(ijk); // i+1, j+1, k + 1

    ijk[2] -= 1;
    v[1][1][0] = mAcc.getValue(ijk); // i+1, j+1, k
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ typename TreeOrAccT::ValueType TrilinearSampler<TreeOrAccT>::sample(const Vec3T<RealT> &uvw, const ValueT (&v)[2][2][2])
{
#if 0
  auto lerp = [](ValueT a, ValueT b, ValueT w){ return fma(w, b-a, a); };// = w*(b-a) + a
  //auto lerp = [](ValueT a, ValueT b, ValueT w){ return fma(w, b, fma(-w, a, a));};// = (1-w)*a + w*b
#else
    auto lerp = [](ValueT a, ValueT b, RealT w) { return a + ValueT(w) * (b - a); };
#endif
    return lerp(lerp(lerp(v[0][0][0], v[0][0][1], uvw[2]), lerp(v[0][1][0], v[0][1][1], uvw[2]), uvw[1]),
                lerp(lerp(v[1][0][0], v[1][0][1], uvw[2]), lerp(v[1][1][0], v[1][1][1], uvw[2]), uvw[1]),
                uvw[0]);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ Vec3T<typename TreeOrAccT::ValueType> TrilinearSampler<TreeOrAccT>::gradient(const Vec3T<RealT> &uvw, const ValueT (&v)[2][2][2])
{
    static_assert(util::is_floating_point<ValueT>::value, "TrilinearSampler::gradient requires a floating-point type");
#if 0
  auto lerp = [](ValueT a, ValueT b, ValueT w){ return fma(w, b-a, a); };// = w*(b-a) + a
  //auto lerp = [](ValueT a, ValueT b, ValueT w){ return fma(w, b, fma(-w, a, a));};// = (1-w)*a + w*b
#else
    auto lerp = [](ValueT a, ValueT b, RealT w) { return a + ValueT(w) * (b - a); };
#endif

    ValueT D[4] = {v[0][0][1] - v[0][0][0], v[0][1][1] - v[0][1][0], v[1][0][1] - v[1][0][0], v[1][1][1] - v[1][1][0]};

    // Z component
    Vec3T<ValueT> grad(0, 0, lerp(lerp(D[0], D[1], uvw[1]), lerp(D[2], D[3], uvw[1]), uvw[0]));

    const ValueT w = ValueT(uvw[2]);
    D[0] = v[0][0][0] + D[0] * w;
    D[1] = v[0][1][0] + D[1] * w;
    D[2] = v[1][0][0] + D[2] * w;
    D[3] = v[1][1][0] + D[3] * w;

    // X component
    grad[0] = lerp(D[2], D[3], uvw[1]) - lerp(D[0], D[1], uvw[1]);

    // Y component
    grad[1] = lerp(D[1] - D[0], D[3] - D[2], uvw[0]);

    return grad;
}

template<typename TreeOrAccT>
__hostdev__ bool TrilinearSampler<TreeOrAccT>::zeroCrossing(const ValueT (&v)[2][2][2])
{
    static_assert(util::is_floating_point<ValueT>::value, "TrilinearSampler::zeroCrossing requires a floating-point type");
    const bool less = v[0][0][0] < ValueT(0);
    return (less ^ (v[0][0][1] < ValueT(0))) ||
           (less ^ (v[0][1][1] < ValueT(0))) ||
           (less ^ (v[0][1][0] < ValueT(0))) ||
           (less ^ (v[1][0][0] < ValueT(0))) ||
           (less ^ (v[1][0][1] < ValueT(0))) ||
           (less ^ (v[1][1][1] < ValueT(0))) ||
           (less ^ (v[1][1][0] < ValueT(0)));
}

/// @brief Template specialization that does not use caching of stencil points
template<typename TreeOrAccT>
class SampleFromVoxels<TreeOrAccT, 1, false> : public TrilinearSampler<TreeOrAccT>
{
    using BaseT = TrilinearSampler<TreeOrAccT>;
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

public:

    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ SampleFromVoxels(const TreeOrAccT& acc) : BaseT(acc) {}

    /// @note xyz is in index space space
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ ValueT operator()(Vec3T<RealT> xyz) const;

    /// @note ijk is in index space space
    __hostdev__ ValueT operator()(const CoordT &ijk) const {return BaseT::mAcc.getValue(ijk);}

    /// @brief Return the gradient in index space.
    ///
    /// @warning Will only compile with floating point value types
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ Vec3T<ValueT> gradient(Vec3T<RealT> xyz) const;

    /// @brief Return true if the tr-linear stencil has a zero crossing at the specified index position.
    ///
    /// @warning Will only compile with floating point value types
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ bool zeroCrossing(Vec3T<RealT> xyz) const;

}; // SampleFromVoxels<TreeOrAccT, 1, false>

/// @brief Template specialization with caching of stencil values
template<typename TreeOrAccT>
class SampleFromVoxels<TreeOrAccT, 1, true> : public TrilinearSampler<TreeOrAccT>
{
    using BaseT = TrilinearSampler<TreeOrAccT>;
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

    mutable CoordT mPos;
    mutable ValueT mVal[2][2][2];

    template<typename RealT, template<typename...> class Vec3T>
    __hostdev__ void cache(Vec3T<RealT>& xyz) const;
public:

    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ SampleFromVoxels(const TreeOrAccT& acc) : BaseT(acc), mPos(CoordT::max()){}

    /// @note xyz is in index space space
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ ValueT operator()(Vec3T<RealT> xyz) const;

    // @note ijk is in index space space
    __hostdev__ ValueT operator()(const CoordT &ijk) const;

    /// @brief Return the gradient in index space.
    ///
    /// @warning Will only compile with floating point value types
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ Vec3T<ValueT> gradient(Vec3T<RealT> xyz) const;

    /// @brief Return true if the tr-linear stencil has a zero crossing at the specified index position.
    ///
    /// @warning Will only compile with floating point value types
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ bool zeroCrossing(Vec3T<RealT> xyz) const;

    /// @brief Return true if the cached tri-linear stencil has a zero crossing.
    ///
    /// @warning Will only compile with floating point value types
    __hostdev__ bool zeroCrossing() const { return BaseT::zeroCrossing(mVal); }

}; // SampleFromVoxels<TreeOrAccT, 1, true>

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 1, true>::operator()(Vec3T<RealT> xyz) const
{
    this->cache(xyz);
    return BaseT::sample(xyz, mVal);
}

template<typename TreeOrAccT>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 1, true>::operator()(const CoordT &ijk) const
{
    return  ijk == mPos ? mVal[0][0][0] : BaseT::mAcc.getValue(ijk);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ Vec3T<typename TreeOrAccT::ValueType> SampleFromVoxels<TreeOrAccT, 1, true>::gradient(Vec3T<RealT> xyz) const
{
    this->cache(xyz);
    return BaseT::gradient(xyz, mVal);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ bool SampleFromVoxels<TreeOrAccT, 1, true>::zeroCrossing(Vec3T<RealT> xyz) const
{
    this->cache(xyz);
    return BaseT::zeroCrossing(mVal);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ void SampleFromVoxels<TreeOrAccT, 1, true>::cache(Vec3T<RealT>& xyz) const
{
    CoordT ijk = Floor<CoordT>(xyz);
    if (ijk != mPos) {
        mPos = ijk;
        BaseT::stencil(ijk, mVal);
    }
}

#if 0

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 1, false>::operator()(Vec3T<RealT> xyz) const
{
    ValueT val[2][2][2];
    CoordT ijk = Floor<CoordT>(xyz);
    BaseT::stencil(ijk, val);
    return BaseT::sample(xyz, val);
}

#else

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 1, false>::operator()(Vec3T<RealT> xyz) const
{
    auto lerp = [](ValueT a, ValueT b, RealT w) { return a + ValueT(w) * (b - a); };

    CoordT coord = Floor<CoordT>(xyz);

    ValueT vx, vx1, vy, vy1, vz, vz1;

    vz = BaseT::mAcc.getValue(coord);
    coord[2] += 1;
    vz1 = BaseT::mAcc.getValue(coord);
    vy = lerp(vz, vz1, xyz[2]);

    coord[1] += 1;

    vz1 = BaseT::mAcc.getValue(coord);
    coord[2] -= 1;
    vz = BaseT::mAcc.getValue(coord);
    vy1 = lerp(vz, vz1, xyz[2]);

    vx = lerp(vy, vy1, xyz[1]);

    coord[0] += 1;

    vz = BaseT::mAcc.getValue(coord);
    coord[2] += 1;
    vz1 = BaseT::mAcc.getValue(coord);
    vy1 = lerp(vz, vz1, xyz[2]);

    coord[1] -= 1;

    vz1 = BaseT::mAcc.getValue(coord);
    coord[2] -= 1;
    vz = BaseT::mAcc.getValue(coord);
    vy = lerp(vz, vz1, xyz[2]);

    vx1 = lerp(vy, vy1, xyz[1]);

    return lerp(vx, vx1, xyz[0]);
}
#endif


template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ inline Vec3T<typename TreeOrAccT::ValueType> SampleFromVoxels<TreeOrAccT, 1, false>::gradient(Vec3T<RealT> xyz) const
{
    ValueT val[2][2][2];
    CoordT ijk = Floor<CoordT>(xyz);
    BaseT::stencil(ijk, val);
    return BaseT::gradient(xyz, val);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ bool SampleFromVoxels<TreeOrAccT, 1, false>::zeroCrossing(Vec3T<RealT> xyz) const
{
    ValueT val[2][2][2];
    CoordT ijk = Floor<CoordT>(xyz);
    BaseT::stencil(ijk, val);
    return BaseT::zeroCrossing(val);
}

// ------------------------------> TriquadraticSampler <--------------------------------------

/// @brief Tri-quadratic sampler, i.e. second order, interpolator
template<typename TreeOrAccT>
class TriquadraticSampler
{
protected:
    const TreeOrAccT& mAcc;

public:
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;
    static const int ORDER = 1;

    /// @brief Protected constructor from a Tree or ReadAccessor
    __hostdev__ TriquadraticSampler(const TreeOrAccT& acc) : mAcc(acc) {}

    __hostdev__ const TreeOrAccT& accessor() const { return mAcc; }

    /// @brief Extract the stencil of 27 values
    inline __hostdev__ void stencil(const CoordT &ijk, ValueT (&v)[3][3][3]) const;

    template<typename RealT, template<typename...> class Vec3T>
    static inline __hostdev__ ValueT sample(const Vec3T<RealT> &uvw, const ValueT (&v)[3][3][3]);

    static inline __hostdev__ bool zeroCrossing(const ValueT (&v)[3][3][3]);
}; // TriquadraticSamplerBase

template<typename TreeOrAccT>
__hostdev__ void TriquadraticSampler<TreeOrAccT>::stencil(const CoordT &ijk, ValueT (&v)[3][3][3]) const
{
    CoordT p(ijk[0] - 1, 0, 0);
    for (int dx = 0; dx < 3; ++dx, ++p[0]) {
        p[1] = ijk[1] - 1;
        for (int dy = 0; dy < 3; ++dy, ++p[1]) {
            p[2] = ijk[2] - 1;
            for (int dz = 0; dz < 3; ++dz, ++p[2]) {
                v[dx][dy][dz] = mAcc.getValue(p);// extract the stencil of 27 values
            }
        }
    }
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ typename TreeOrAccT::ValueType TriquadraticSampler<TreeOrAccT>::sample(const Vec3T<RealT> &uvw, const ValueT (&v)[3][3][3])
{
    auto kernel = [](const ValueT* value, double weight)->ValueT {
        return weight * (weight * (0.5f * (value[0] + value[2]) - value[1]) +
                        0.5f * (value[2] - value[0])) + value[1];
    };

    ValueT vx[3];
    for (int dx = 0; dx < 3; ++dx) {
        ValueT vy[3];
        for (int dy = 0; dy < 3; ++dy) {
            vy[dy] = kernel(&v[dx][dy][0], uvw[2]);
        }//loop over y
        vx[dx] = kernel(vy, uvw[1]);
    }//loop over x
    return kernel(vx, uvw[0]);
}

template<typename TreeOrAccT>
__hostdev__ bool TriquadraticSampler<TreeOrAccT>::zeroCrossing(const ValueT (&v)[3][3][3])
{
    static_assert(util::is_floating_point<ValueT>::value, "TrilinearSampler::zeroCrossing requires a floating-point type");
    const bool less = v[0][0][0] < ValueT(0);
    for (int dx = 0; dx < 3; ++dx) {
        for (int dy = 0; dy < 3; ++dy) {
            for (int dz = 0; dz < 3; ++dz) {
                if (less ^ (v[dx][dy][dz] < ValueT(0))) return true;
            }
        }
    }
    return false;
}

/// @brief Template specialization that does not use caching of stencil points
template<typename TreeOrAccT>
class SampleFromVoxels<TreeOrAccT, 2, false> : public TriquadraticSampler<TreeOrAccT>
{
    using BaseT = TriquadraticSampler<TreeOrAccT>;
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;
public:

    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ SampleFromVoxels(const TreeOrAccT& acc) : BaseT(acc) {}

    /// @note xyz is in index space space
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ ValueT operator()(Vec3T<RealT> xyz) const;

    __hostdev__ ValueT operator()(const CoordT &ijk) const {return BaseT::mAcc.getValue(ijk);}

    /// @brief Return true if the tr-linear stencil has a zero crossing at the specified index position.
    ///
    /// @warning Will only compile with floating point value types
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ bool zeroCrossing(Vec3T<RealT> xyz) const;

}; // SampleFromVoxels<TreeOrAccT, 2, false>

/// @brief Template specialization with caching of stencil values
template<typename TreeOrAccT>
class SampleFromVoxels<TreeOrAccT, 2, true> : public TriquadraticSampler<TreeOrAccT>
{
    using BaseT = TriquadraticSampler<TreeOrAccT>;
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

    mutable CoordT mPos;
    mutable ValueT mVal[3][3][3];

    template<typename RealT, template<typename...> class Vec3T>
    __hostdev__ void cache(Vec3T<RealT>& xyz) const;
public:

    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ SampleFromVoxels(const TreeOrAccT& acc) : BaseT(acc), mPos(CoordT::max()){}

    /// @note xyz is in index space space
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ ValueT operator()(Vec3T<RealT> xyz) const;

    inline __hostdev__ ValueT operator()(const CoordT &ijk) const;

    /// @brief Return true if the tr-linear stencil has a zero crossing at the specified index position.
    ///
    /// @warning Will only compile with floating point value types
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ bool zeroCrossing(Vec3T<RealT> xyz) const;

    /// @brief Return true if the cached tri-linear stencil has a zero crossing.
    ///
    /// @warning Will only compile with floating point value types
    __hostdev__ bool zeroCrossing() const { return BaseT::zeroCrossing(mVal); }

}; // SampleFromVoxels<TreeOrAccT, 2, true>

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 2, true>::operator()(Vec3T<RealT> xyz) const
{
    this->cache(xyz);
    return BaseT::sample(xyz, mVal);
}

template<typename TreeOrAccT>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 2, true>::operator()(const CoordT &ijk) const
{
    return  ijk == mPos ? mVal[1][1][1] : BaseT::mAcc.getValue(ijk);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ bool SampleFromVoxels<TreeOrAccT, 2, true>::zeroCrossing(Vec3T<RealT> xyz) const
{
    this->cache(xyz);
    return BaseT::zeroCrossing(mVal);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ void SampleFromVoxels<TreeOrAccT, 2, true>::cache(Vec3T<RealT>& xyz) const
{
    CoordT ijk = Floor<CoordT>(xyz);
    if (ijk != mPos) {
        mPos = ijk;
        BaseT::stencil(ijk, mVal);
    }
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 2, false>::operator()(Vec3T<RealT> xyz) const
{
    ValueT val[3][3][3];
    CoordT ijk = Floor<CoordT>(xyz);
    BaseT::stencil(ijk, val);
    return BaseT::sample(xyz, val);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ bool SampleFromVoxels<TreeOrAccT, 2, false>::zeroCrossing(Vec3T<RealT> xyz) const
{
    ValueT val[3][3][3];
    CoordT ijk = Floor<CoordT>(xyz);
    BaseT::stencil(ijk, val);
    return BaseT::zeroCrossing(val);
}

// ------------------------------> TricubicSampler <--------------------------------------

/// @brief Tri-cubic sampler, i.e. third order, interpolator.
///
/// @details See the following paper for implementation details:
/// Lekien, F. and Marsden, J.: Tricubic interpolation in three dimensions.
///                         In: International Journal for Numerical Methods
///                         in Engineering (2005), No. 63, p. 455-471

template<typename TreeOrAccT>
class TricubicSampler
{
protected:
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

    const TreeOrAccT& mAcc;

public:
    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ TricubicSampler(const TreeOrAccT& acc)
        : mAcc(acc)
    {
    }

    __hostdev__ const TreeOrAccT& accessor() const { return mAcc; }

     /// @brief Extract the stencil of 8 values
    inline __hostdev__ void stencil(const CoordT& ijk, ValueT (&c)[64]) const;

    template<typename RealT, template<typename...> class Vec3T>
    static inline __hostdev__ ValueT sample(const Vec3T<RealT> &uvw, const ValueT (&c)[64]);
}; // TricubicSampler

template<typename TreeOrAccT>
__hostdev__ void TricubicSampler<TreeOrAccT>::stencil(const CoordT& ijk, ValueT (&C)[64]) const
{
    auto fetch = [&](int i, int j, int k) -> ValueT& { return C[((i + 1) << 4) + ((j + 1) << 2) + k + 1]; };

    // fetch 64 point stencil values
    for (int i = -1; i < 3; ++i) {
        for (int j = -1; j < 3; ++j) {
            fetch(i, j, -1) = mAcc.getValue(ijk + CoordT(i, j, -1));
            fetch(i, j,  0) = mAcc.getValue(ijk + CoordT(i, j,  0));
            fetch(i, j,  1) = mAcc.getValue(ijk + CoordT(i, j,  1));
            fetch(i, j,  2) = mAcc.getValue(ijk + CoordT(i, j,  2));
        }
    }
    const ValueT half(0.5), quarter(0.25), eighth(0.125);
    const ValueT X[64] = {// values of f(x,y,z) at the 8 corners (each from 1 stencil value).
                          fetch(0, 0, 0),
                          fetch(1, 0, 0),
                          fetch(0, 1, 0),
                          fetch(1, 1, 0),
                          fetch(0, 0, 1),
                          fetch(1, 0, 1),
                          fetch(0, 1, 1),
                          fetch(1, 1, 1),
                          // values of df/dx at the 8 corners (each from 2 stencil values).
                          half * (fetch(1, 0, 0) - fetch(-1, 0, 0)),
                          half * (fetch(2, 0, 0) - fetch(0, 0, 0)),
                          half * (fetch(1, 1, 0) - fetch(-1, 1, 0)),
                          half * (fetch(2, 1, 0) - fetch(0, 1, 0)),
                          half * (fetch(1, 0, 1) - fetch(-1, 0, 1)),
                          half * (fetch(2, 0, 1) - fetch(0, 0, 1)),
                          half * (fetch(1, 1, 1) - fetch(-1, 1, 1)),
                          half * (fetch(2, 1, 1) - fetch(0, 1, 1)),
                          // values of df/dy at the 8 corners (each from 2 stencil values).
                          half * (fetch(0, 1, 0) - fetch(0, -1, 0)),
                          half * (fetch(1, 1, 0) - fetch(1, -1, 0)),
                          half * (fetch(0, 2, 0) - fetch(0, 0, 0)),
                          half * (fetch(1, 2, 0) - fetch(1, 0, 0)),
                          half * (fetch(0, 1, 1) - fetch(0, -1, 1)),
                          half * (fetch(1, 1, 1) - fetch(1, -1, 1)),
                          half * (fetch(0, 2, 1) - fetch(0, 0, 1)),
                          half * (fetch(1, 2, 1) - fetch(1, 0, 1)),
                          // values of df/dz at the 8 corners (each from 2 stencil values).
                          half * (fetch(0, 0, 1) - fetch(0, 0, -1)),
                          half * (fetch(1, 0, 1) - fetch(1, 0, -1)),
                          half * (fetch(0, 1, 1) - fetch(0, 1, -1)),
                          half * (fetch(1, 1, 1) - fetch(1, 1, -1)),
                          half * (fetch(0, 0, 2) - fetch(0, 0, 0)),
                          half * (fetch(1, 0, 2) - fetch(1, 0, 0)),
                          half * (fetch(0, 1, 2) - fetch(0, 1, 0)),
                          half * (fetch(1, 1, 2) - fetch(1, 1, 0)),
                          // values of d2f/dxdy at the 8 corners (each from 4 stencil values).
                          quarter * (fetch(1, 1, 0) - fetch(-1, 1, 0) - fetch(1, -1, 0) + fetch(-1, -1, 0)),
                          quarter * (fetch(2, 1, 0) - fetch(0, 1, 0) - fetch(2, -1, 0) + fetch(0, -1, 0)),
                          quarter * (fetch(1, 2, 0) - fetch(-1, 2, 0) - fetch(1, 0, 0) + fetch(-1, 0, 0)),
                          quarter * (fetch(2, 2, 0) - fetch(0, 2, 0) - fetch(2, 0, 0) + fetch(0, 0, 0)),
                          quarter * (fetch(1, 1, 1) - fetch(-1, 1, 1) - fetch(1, -1, 1) + fetch(-1, -1, 1)),
                          quarter * (fetch(2, 1, 1) - fetch(0, 1, 1) - fetch(2, -1, 1) + fetch(0, -1, 1)),
                          quarter * (fetch(1, 2, 1) - fetch(-1, 2, 1) - fetch(1, 0, 1) + fetch(-1, 0, 1)),
                          quarter * (fetch(2, 2, 1) - fetch(0, 2, 1) - fetch(2, 0, 1) + fetch(0, 0, 1)),
                          // values of d2f/dxdz at the 8 corners (each from 4 stencil values).
                          quarter * (fetch(1, 0, 1) - fetch(-1, 0, 1) - fetch(1, 0, -1) + fetch(-1, 0, -1)),
                          quarter * (fetch(2, 0, 1) - fetch(0, 0, 1) - fetch(2, 0, -1) + fetch(0, 0, -1)),
                          quarter * (fetch(1, 1, 1) - fetch(-1, 1, 1) - fetch(1, 1, -1) + fetch(-1, 1, -1)),
                          quarter * (fetch(2, 1, 1) - fetch(0, 1, 1) - fetch(2, 1, -1) + fetch(0, 1, -1)),
                          quarter * (fetch(1, 0, 2) - fetch(-1, 0, 2) - fetch(1, 0, 0) + fetch(-1, 0, 0)),
                          quarter * (fetch(2, 0, 2) - fetch(0, 0, 2) - fetch(2, 0, 0) + fetch(0, 0, 0)),
                          quarter * (fetch(1, 1, 2) - fetch(-1, 1, 2) - fetch(1, 1, 0) + fetch(-1, 1, 0)),
                          quarter * (fetch(2, 1, 2) - fetch(0, 1, 2) - fetch(2, 1, 0) + fetch(0, 1, 0)),
                          // values of d2f/dydz at the 8 corners (each from 4 stencil values).
                          quarter * (fetch(0, 1, 1) - fetch(0, -1, 1) - fetch(0, 1, -1) + fetch(0, -1, -1)),
                          quarter * (fetch(1, 1, 1) - fetch(1, -1, 1) - fetch(1, 1, -1) + fetch(1, -1, -1)),
                          quarter * (fetch(0, 2, 1) - fetch(0, 0, 1) - fetch(0, 2, -1) + fetch(0, 0, -1)),
                          quarter * (fetch(1, 2, 1) - fetch(1, 0, 1) - fetch(1, 2, -1) + fetch(1, 0, -1)),
                          quarter * (fetch(0, 1, 2) - fetch(0, -1, 2) - fetch(0, 1, 0) + fetch(0, -1, 0)),
                          quarter * (fetch(1, 1, 2) - fetch(1, -1, 2) - fetch(1, 1, 0) + fetch(1, -1, 0)),
                          quarter * (fetch(0, 2, 2) - fetch(0, 0, 2) - fetch(0, 2, 0) + fetch(0, 0, 0)),
                          quarter * (fetch(1, 2, 2) - fetch(1, 0, 2) - fetch(1, 2, 0) + fetch(1, 0, 0)),
                          // values of d3f/dxdydz at the 8 corners (each from 8 stencil values).
                          eighth * (fetch(1, 1, 1) - fetch(-1, 1, 1) - fetch(1, -1, 1) + fetch(-1, -1, 1) - fetch(1, 1, -1) + fetch(-1, 1, -1) + fetch(1, -1, -1) - fetch(-1, -1, -1)),
                          eighth * (fetch(2, 1, 1) - fetch(0, 1, 1) - fetch(2, -1, 1) + fetch(0, -1, 1) - fetch(2, 1, -1) + fetch(0, 1, -1) + fetch(2, -1, -1) - fetch(0, -1, -1)),
                          eighth * (fetch(1, 2, 1) - fetch(-1, 2, 1) - fetch(1, 0, 1) + fetch(-1, 0, 1) - fetch(1, 2, -1) + fetch(-1, 2, -1) + fetch(1, 0, -1) - fetch(-1, 0, -1)),
                          eighth * (fetch(2, 2, 1) - fetch(0, 2, 1) - fetch(2, 0, 1) + fetch(0, 0, 1) - fetch(2, 2, -1) + fetch(0, 2, -1) + fetch(2, 0, -1) - fetch(0, 0, -1)),
                          eighth * (fetch(1, 1, 2) - fetch(-1, 1, 2) - fetch(1, -1, 2) + fetch(-1, -1, 2) - fetch(1, 1, 0) + fetch(-1, 1, 0) + fetch(1, -1, 0) - fetch(-1, -1, 0)),
                          eighth * (fetch(2, 1, 2) - fetch(0, 1, 2) - fetch(2, -1, 2) + fetch(0, -1, 2) - fetch(2, 1, 0) + fetch(0, 1, 0) + fetch(2, -1, 0) - fetch(0, -1, 0)),
                          eighth * (fetch(1, 2, 2) - fetch(-1, 2, 2) - fetch(1, 0, 2) + fetch(-1, 0, 2) - fetch(1, 2, 0) + fetch(-1, 2, 0) + fetch(1, 0, 0) - fetch(-1, 0, 0)),
                          eighth * (fetch(2, 2, 2) - fetch(0, 2, 2) - fetch(2, 0, 2) + fetch(0, 0, 2) - fetch(2, 2, 0) + fetch(0, 2, 0) + fetch(2, 0, 0) - fetch(0, 0, 0))};

    // 4Kb of static table (int8_t has a range of -127 -> 127 which suffices)
    static const int8_t A[64][64] = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {9, -9, -9, 9, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-6, 6, 6, -6, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, 4, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-6, 6, 6, -6, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, 3, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {4, -4, -4, 4, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, 4, -2, 2, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, 3, -3, 3, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0},
        {-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {9, -9, 0, 0, -9, 9, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-6, 6, 0, 0, 6, -6, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, 0, 0, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0},
        {9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0},
        {-27, 27, 27, -27, 27, -27, -27, 27, -18, -9, 18, 9, 18, 9, -18, -9, -18, 18, -9, 9, 18, -18, 9, -9, -18, 18, 18, -18, -9, 9, 9, -9, -12, -6, -6, -3, 12, 6, 6, 3, -12, -6, 12, 6, -6, -3, 6, 3, -12, 12, -6, 6, -6, 6, -3, 3, -8, -4, -4, -2, -4, -2, -2, -1},
        {18, -18, -18, 18, -18, 18, 18, -18, 9, 9, -9, -9, -9, -9, 9, 9, 12, -12, 6, -6, -12, 12, -6, 6, 12, -12, -12, 12, 6, -6, -6, 6, 6, 6, 3, 3, -6, -6, -3, -3, 6, 6, -6, -6, 3, 3, -3, -3, 8, -8, 4, -4, 4, -4, 2, -2, 4, 4, 2, 2, 2, 2, 1, 1},
        {-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0},
        {18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 9, -9, 9, -9, -9, 9, -9, 9, 12, -12, -12, 12, 6, -6, -6, 6, 6, 3, 6, 3, -6, -3, -6, -3, 8, 4, -8, -4, 4, 2, -4, -2, 6, -6, 6, -6, 3, -3, 3, -3, 4, 2, 4, 2, 2, 1, 2, 1},
        {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -6, 6, -6, 6, 6, -6, 6, -6, -8, 8, 8, -8, -4, 4, 4, -4, -3, -3, -3, -3, 3, 3, 3, 3, -4, -4, 4, 4, -2, -2, 2, 2, -4, 4, -4, 4, -2, 2, -2, 2, -2, -2, -2, -2, -1, -1, -1, -1},
        {2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-6, 6, 0, 0, 6, -6, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {4, -4, 0, 0, -4, 4, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, 0, 0, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0},
        {-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0},
        {18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 12, -12, 6, -6, -12, 12, -6, 6, 9, -9, -9, 9, 9, -9, -9, 9, 8, 4, 4, 2, -8, -4, -4, -2, 6, 3, -6, -3, 6, 3, -6, -3, 6, -6, 3, -3, 6, -6, 3, -3, 4, 2, 2, 1, 4, 2, 2, 1},
        {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -8, 8, -4, 4, 8, -8, 4, -4, -6, 6, 6, -6, -6, 6, 6, -6, -4, -4, -2, -2, 4, 4, 2, 2, -3, -3, 3, 3, -3, -3, 3, 3, -4, 4, -2, 2, -4, 4, -2, 2, -2, -2, -1, -1, -2, -2, -1, -1},
        {4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
        {-12, 12, 12, -12, 12, -12, -12, 12, -8, -4, 8, 4, 8, 4, -8, -4, -6, 6, -6, 6, 6, -6, 6, -6, -6, 6, 6, -6, -6, 6, 6, -6, -4, -2, -4, -2, 4, 2, 4, 2, -4, -2, 4, 2, -4, -2, 4, 2, -3, 3, -3, 3, -3, 3, -3, 3, -2, -1, -2, -1, -2, -1, -2, -1},
        {8, -8, -8, 8, -8, 8, 8, -8, 4, 4, -4, -4, -4, -4, 4, 4, 4, -4, 4, -4, -4, 4, -4, 4, 4, -4, -4, 4, 4, -4, -4, 4, 2, 2, 2, 2, -2, -2, -2, -2, 2, 2, -2, -2, 2, 2, -2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 1, 1, 1, 1, 1, 1, 1, 1}};

    for (int i = 0; i < 64; ++i) { // C = A * X
        C[i] = ValueT(0);
#if 0
    for (int j = 0; j < 64; j += 4) {
      C[i] = fma(A[i][j], X[j], fma(A[i][j+1], X[j+1], fma(A[i][j+2], X[j+2], fma(A[i][j+3], X[j+3], C[i]))));
    }
#else
        for (int j = 0; j < 64; j += 4) {
            C[i] += A[i][j] * X[j] + A[i][j + 1] * X[j + 1] + A[i][j + 2] * X[j + 2] + A[i][j + 3] * X[j + 3];
        }
#endif
    }
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ typename TreeOrAccT::ValueType TricubicSampler<TreeOrAccT>::sample(const Vec3T<RealT> &xyz, const ValueT (&C)[64])
{
    ValueT zPow(1), sum(0);
    for (int k = 0, n = 0; k < 4; ++k) {
        ValueT yPow(1);
        for (int j = 0; j < 4; ++j, n += 4) {
#if 0
            sum = fma( yPow, zPow * fma(xyz[0], fma(xyz[0], fma(xyz[0], C[n + 3], C[n + 2]), C[n + 1]), C[n]), sum);
#else
            sum += yPow * zPow * (C[n] + xyz[0] * (C[n + 1] + xyz[0] * (C[n + 2] + xyz[0] * C[n + 3])));
#endif
            yPow *= xyz[1];
        }
        zPow *= xyz[2];
    }
    return sum;
}

template<typename TreeOrAccT>
class SampleFromVoxels<TreeOrAccT, 3, true> : public TricubicSampler<TreeOrAccT>
{
    using BaseT  = TricubicSampler<TreeOrAccT>;
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

    mutable CoordT mPos;
    mutable ValueT mC[64];

    template<typename RealT, template<typename...> class Vec3T>
    __hostdev__ void cache(Vec3T<RealT>& xyz) const;

public:
    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ SampleFromVoxels(const TreeOrAccT& acc)
        : BaseT(acc)
    {
    }

    /// @note xyz is in index space space
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ ValueT operator()(Vec3T<RealT> xyz) const;

    // @brief Return value at the coordinate @a ijk in index space space
    __hostdev__ ValueT operator()(const CoordT &ijk) const {return BaseT::mAcc.getValue(ijk);}

}; // SampleFromVoxels<TreeOrAccT, 3, true>

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 3, true>::operator()(Vec3T<RealT> xyz) const
{
    this->cache(xyz);
    return BaseT::sample(xyz, mC);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ void SampleFromVoxels<TreeOrAccT, 3, true>::cache(Vec3T<RealT>& xyz) const
{
    CoordT ijk = Floor<CoordT>(xyz);
    if (ijk != mPos) {
        mPos = ijk;
        BaseT::stencil(ijk, mC);
    }
}

template<typename TreeOrAccT>
class SampleFromVoxels<TreeOrAccT, 3, false> : public TricubicSampler<TreeOrAccT>
{
    using BaseT  = TricubicSampler<TreeOrAccT>;
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

public:
    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ SampleFromVoxels(const TreeOrAccT& acc)
        : BaseT(acc)
    {
    }

    /// @note xyz is in index space space
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ ValueT operator()(Vec3T<RealT> xyz) const;

    __hostdev__ ValueT operator()(const CoordT &ijk) const {return BaseT::mAcc.getValue(ijk);}

}; // SampleFromVoxels<TreeOrAccT, 3, true>

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
__hostdev__ typename TreeOrAccT::ValueType SampleFromVoxels<TreeOrAccT, 3, false>::operator()(Vec3T<RealT> xyz) const
{
    ValueT C[64];
    CoordT ijk = Floor<CoordT>(xyz);
    BaseT::stencil(ijk, C);
    return BaseT::sample(xyz, C);
}

}// namespace math

template<int Order, typename TreeOrAccT, bool UseCache = true>
[[deprecated("Use nanovdb::math::createSampler instead")]]
__hostdev__ math::SampleFromVoxels<TreeOrAccT, Order, UseCache> createSampler(const TreeOrAccT& acc)
{
    return math::SampleFromVoxels<TreeOrAccT, Order, UseCache>(acc);
}

} // namespace nanovdb

#endif // NANOVDB_SAMPLE_FROM_VOXELS_H_HAS_BEEN_INCLUDED
