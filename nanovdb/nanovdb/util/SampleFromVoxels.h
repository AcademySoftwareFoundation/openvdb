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
#ifdef __CUDACC__
#define __hostdev__ __host__ __device__
#else
#include <cmath> // for floor
#define __hostdev__
#endif

namespace nanovdb {

/// @brief Utility function that returns the Coord of the round-down of @a xyz
///        and redefined @xyz as the frational part, ie xyz-in = return-value + xyz-out
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

// Forward declaration of sampler with specific polynomial orders
template<typename TreeT, int Order, bool WithCache = true>
struct SampleFromVoxels;

/// @brief Factory free-function for a sampler of specific polynomial orders
///
/// @details This allows for the compact syntax:
/// @code
///   auto acc = grid.getAccessor();
///   auto smp = nanovdb::createSampler<1>( acc );
/// @endcode
template<int Order, typename TreeOrAccT>
__hostdev__ SampleFromVoxels<TreeOrAccT, Order, true> createSampler(const TreeOrAccT& acc)
{
    return SampleFromVoxels<TreeOrAccT, Order, true>(acc);
}

// ------------------------------> NearestNeighborSampler <--------------------------------------

/// @brief Neigherest neighbor, i.e. zero order, interpolator
template<typename TreeOrAccT, bool WithCache = true>
class NearestNeighborSampler;

template<typename TreeOrAccT>
class NearestNeighborSampler<TreeOrAccT, true>
{
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

public:
    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ NearestNeighborSampler(const TreeOrAccT& acc)
        : mAcc(&acc)
        , mPos(CoordT::max())
    {
    }

    /// @note xyz is in index space space
    template<typename Vec3T>
    inline __hostdev__ ValueT operator()(const Vec3T& xyz);

private:
    const TreeOrAccT* mAcc;
    CoordT            mPos;
    ValueT            mVal; // private cache
}; // NearestNeighborSampler

template<typename TreeOrAccT>
class NearestNeighborSampler<TreeOrAccT, false>
{
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

public:
    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ NearestNeighborSampler(const TreeOrAccT& acc)
        : mAcc(&acc)
    {
    }

    /// @note xyz is in index space space
    template<typename Vec3T>
    inline __hostdev__ ValueT operator()(const Vec3T& xyz);

private:
    const TreeOrAccT* mAcc;
}; // NearestNeighborSampler

template<typename TreeOrAccT>
template<typename Vec3T>
typename TreeOrAccT::ValueType NearestNeighborSampler<TreeOrAccT, true>::operator()(const Vec3T& xyz)
{
    const CoordT ijk = Round<CoordT>(xyz);
    if (ijk != mPos) {
        mPos = ijk;
        mVal = mAcc->getValue(mPos);
    }
    return mVal;
}

template<typename TreeOrAccT>
template<typename Vec3T>
typename TreeOrAccT::ValueType NearestNeighborSampler<TreeOrAccT, false>::operator()(const Vec3T& xyz)
{
    return mAcc->getValue(Round<CoordT>(xyz));
}

// ------------------------------> TrilinearSampler <--------------------------------------

/// @brief Tri-linear sampler, i.e. first order, interpolator
template<typename TreeOrAccT, bool WithCache = true>
class TrilinearSampler;

template<typename TreeOrAccT>
class TrilinearSampler<TreeOrAccT, true>
{
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

public:
    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ TrilinearSampler(const TreeOrAccT& acc)
        : mAcc(&acc)
        , mPos(CoordT::max())
    {
    }

    /// @note xyz is in index space space
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ ValueT operator()(Vec3T<RealT> xyz);

    /// @brief Return the gradient in index space.
    ///
    /// @warning Will only compile with floating point value types
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ Vec3T<ValueT> gradient(Vec3T<RealT> xyz);

    /// @brief Return true if the tr-linear stencil has a zero crossing at the specified index position.
    ///
    /// @warning Will only compile with floating point value types
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ bool zeroCrossing(Vec3T<RealT> xyz);

    /// @brief Return true if the cached tri-linear stencil has a zero crossing.
    ///
    /// @warning Will only compile with floating point value types
    inline __hostdev__ bool zeroCrossing();

private:
    const TreeOrAccT* mAcc;
    CoordT            mPos;
    ValueT            mVal[2][2][2];

    template<typename Vec3T>
    __hostdev__ bool update(Vec3T& xyz);
}; // TrilinearSampler

template<typename TreeOrAccT>
class TrilinearSampler<TreeOrAccT, false>
{
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

public:
    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ TrilinearSampler(const TreeOrAccT& acc)
        : mAcc(&acc)
    {
    }

    /// @note xyz is in index space space
    template<typename RealT, template<typename...> class Vec3T>
    inline __hostdev__ ValueT operator()(Vec3T<RealT> xyz);

private:
    const TreeOrAccT* mAcc;
}; // TrilinearSampler

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
typename TreeOrAccT::ValueType TrilinearSampler<TreeOrAccT, true>::operator()(Vec3T<RealT> xyz)
{
    this->update(xyz);
#if 0
  auto lerp = [](ValueT a, ValueT b, ValueT w){ return fma(w, b-a, a); };// = w*(b-a) + a
  //auto lerp = [](ValueT a, ValueT b, ValueT w){ return fma(w, b, fma(-w, a, a));};// = (1-w)*a + w*b
#else
    auto lerp = [](ValueT a, ValueT b, RealT w) { return a + ValueT(w) * (b - a); };
#endif
    return lerp(lerp(lerp(mVal[0][0][0], mVal[0][0][1], xyz[2]), lerp(mVal[0][1][0], mVal[0][1][1], xyz[2]), xyz[1]),
                lerp(lerp(mVal[1][0][0], mVal[1][0][1], xyz[2]), lerp(mVal[1][1][0], mVal[1][1][1], xyz[2]), xyz[1]),
                xyz[0]);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
typename TreeOrAccT::ValueType TrilinearSampler<TreeOrAccT, false>::operator()(Vec3T<RealT> xyz)
{
    auto lerp = [](ValueT a, ValueT b, RealT w) { return a + ValueT(w) * (b - a); };

    CoordT coord = Floor<CoordT>(xyz);

    ValueT vx, vx1, vy, vy1, vz, vz1;

    vz = mAcc->getValue(coord);
    coord[2] += 1;
    vz1 = mAcc->getValue(coord);
    vy = lerp(vz, vz1, xyz[2]);

    coord[1] += 1;

    vz1 = mAcc->getValue(coord);
    coord[2] -= 1;
    vz = mAcc->getValue(coord);
    vy1 = lerp(vz, vz1, xyz[2]);

    vx = lerp(vy, vy1, xyz[1]);

    coord[0] += 1;

    vz = mAcc->getValue(coord);
    coord[2] += 1;
    vz1 = mAcc->getValue(coord);
    vy1 = lerp(vz, vz1, xyz[2]);

    coord[1] -= 1;

    vz1 = mAcc->getValue(coord);
    coord[2] -= 1;
    vz = mAcc->getValue(coord);
    vy = lerp(vz, vz1, xyz[2]);

    vx1 = lerp(vy, vy1, xyz[1]);

    return lerp(vx, vx1, xyz[0]);
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
Vec3T<typename TreeOrAccT::ValueType> TrilinearSampler<TreeOrAccT>::gradient(Vec3T<RealT> xyz)
{
    static_assert(std::is_floating_point<ValueT>::value, "TrilinearSampler::gradient requires a floating-point type");
    this->update(xyz);
#if 0
  auto lerp = [](ValueT a, ValueT b, ValueT w){ return fma(w, b-a, a); };// = w*(b-a) + a
  //auto lerp = [](ValueT a, ValueT b, ValueT w){ return fma(w, b, fma(-w, a, a));};// = (1-w)*a + w*b
#else
    auto lerp = [](ValueT a, ValueT b, RealT w) { return a + ValueT(w) * (b - a); };
#endif

    ValueT D[4] = {mVal[0][0][1] - mVal[0][0][0], mVal[0][1][1] - mVal[0][1][0], mVal[1][0][1] - mVal[1][0][0], mVal[1][1][1] - mVal[1][1][0]};

    // Z component
    Vec3T<ValueT> grad(0, 0, lerp(lerp(D[0], D[1], xyz[1]), lerp(D[2], D[3], xyz[1]), xyz[0]));

    const ValueT w = ValueT(xyz[2]);
    D[0] = mVal[0][0][0] + D[0] * w;
    D[1] = mVal[0][1][0] + D[1] * w;
    D[2] = mVal[1][0][0] + D[2] * w;
    D[3] = mVal[1][1][0] + D[3] * w;

    // X component
    grad[0] = lerp(D[2], D[3], xyz[1]) - lerp(D[0], D[1], xyz[1]);

    // Y component
    grad[1] = lerp(D[1] - D[0], D[3] - D[2], xyz[0]);

    return grad;
}

template<typename TreeOrAccT>
bool TrilinearSampler<TreeOrAccT>::zeroCrossing()
{
    static_assert(std::is_floating_point<ValueT>::value, "TrilinearSampler::zeroCrossing requires a floating-point type");
    const bool less = mVal[0][0][0] < ValueT(0);
    return (less ^ (mVal[0][0][1] < ValueT(0))) ||
           (less ^ (mVal[0][1][1] < ValueT(0))) ||
           (less ^ (mVal[0][1][0] < ValueT(0))) ||
           (less ^ (mVal[1][0][0] < ValueT(0))) ||
           (less ^ (mVal[1][0][1] < ValueT(0))) ||
           (less ^ (mVal[1][1][1] < ValueT(0))) ||
           (less ^ (mVal[1][1][0] < ValueT(0)));
}

template<typename TreeOrAccT>
template<typename RealT, template<typename...> class Vec3T>
bool TrilinearSampler<TreeOrAccT>::zeroCrossing(Vec3T<RealT> xyz)
{
    this->update(xyz);
    return this->zeroCrossing();
}

template<typename TreeOrAccT>
template<typename Vec3T>
bool TrilinearSampler<TreeOrAccT>::update(Vec3T& xyz)
{
    CoordT ijk = Floor<CoordT>(xyz);

    if (ijk == mPos)
        return false; // early out - reuse cached values

    mPos = ijk;

    mVal[0][0][0] = mAcc->getValue(ijk); // i, j, k

    ijk[2] += 1;
    mVal[0][0][1] = mAcc->getValue(ijk); // i, j, k + 1

    ijk[1] += 1;
    mVal[0][1][1] = mAcc->getValue(ijk); // i, j+1, k + 1

    ijk[2] -= 1;
    mVal[0][1][0] = mAcc->getValue(ijk); // i, j+1, k

    ijk[0] += 1;
    ijk[1] -= 1;
    mVal[1][0][0] = mAcc->getValue(ijk); // i+1, j, k

    ijk[2] += 1;
    mVal[1][0][1] = mAcc->getValue(ijk); // i+1, j, k + 1

    ijk[1] += 1;
    mVal[1][1][1] = mAcc->getValue(ijk); // i+1, j+1, k + 1

    ijk[2] -= 1;
    mVal[1][1][0] = mAcc->getValue(ijk); // i+1, j+1, k

    return true;
}

// ------------------------------> TriquadraticSampler <--------------------------------------

/// @brief Tri-quadratic sampler, i.e. second order, interpolator
///
/// @warning TriquadraticSampler has not implemented yet!
template<typename TreeT, bool WithCache = true>
class TriquadraticSampler
{
    // TriquadraticSampler has not implemented yet!
};

// ------------------------------> TricubicSampler <--------------------------------------

/// @brief Tri-cubic sampler, i.e. third order, interpolator.
///
/// @details See the following paper for implementation details:
/// Lekien, F. and Marsden, J.: Tricubic interpolation in three dimensions.
///                         In: International Journal for Numerical Methods
///                         in Engineering (2005), No. 63, p. 455-471
template<typename TreeOrAccT, bool WithCache = true>
class TricubicSampler;

template<typename TreeOrAccT>
class TricubicSampler<TreeOrAccT, true>
{
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

public:
    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ TricubicSampler(const TreeOrAccT& acc)
        : mAcc(&acc)
        , mPos(CoordT::max())
    {
    }

    /// @note xyz is in index space space
    template<typename Vec3T>
    inline __hostdev__ ValueT operator()(Vec3T xyz);

private:
    const TreeOrAccT* mAcc;
    CoordT            mPos;
    ValueT            mC[64]; //private cache

    template<typename Vec3T>
    __hostdev__ bool update(Vec3T& xyz);
}; // TricubicSampler

template<typename TreeOrAccT>
class TricubicSampler<TreeOrAccT, false>
{
    using ValueT = typename TreeOrAccT::ValueType;
    using CoordT = typename TreeOrAccT::CoordType;

public:
    /// @brief Construction from a Tree or ReadAccessor
    __hostdev__ TricubicSampler(const TreeOrAccT& acc)
        : mAcc(&acc)
    {
    }

    /// @note xyz is in index space space
    template<typename Vec3T>
    inline __hostdev__ ValueT operator()(Vec3T xyz);

private:
    const TreeOrAccT* mAcc;
}; // TricubicSampler

template<typename TreeOrAccT>
template<typename Vec3T>
__hostdev__ typename TreeOrAccT::ValueType TricubicSampler<TreeOrAccT, true>::operator()(Vec3T xyz)
{
    this->update(xyz); // modifies xyz and re-computes mC and mPos if required

    ValueT zPow(1), sum(0);
    for (int k = 0, n = 0; k < 4; ++k) {
        ValueT yPow(1);
        for (int j = 0; j < 4; ++j, n += 4) {
#if 0
            sum = fma( yPow, zPow * fma(xyz[0], fma(xyz[0], fma(xyz[0], mC[n + 3], mC[n + 2]), mC[n + 1]), mC[n]), sum);
#else
            sum += yPow * zPow * (mC[n] + xyz[0] * (mC[n + 1] + xyz[0] * (mC[n + 2] + xyz[0] * mC[n + 3])));
#endif
            yPow *= xyz[1];
        }
        zPow *= xyz[2];
    }
    return sum;
}

template<typename TreeOrAccT>
template<typename Vec3T>
__hostdev__ typename TreeOrAccT::ValueType TricubicSampler<TreeOrAccT, false>::operator()(Vec3T xyz)
{
    return TricubicSampler<TreeOrAccT, true>(*mAcc)(xyz);
}

template<typename TreeOrAccT>
template<typename Vec3T>
bool TricubicSampler<TreeOrAccT>::update(Vec3T& xyz)
{
    const CoordT ijk = Floor<CoordT>(xyz);

    if (ijk == mPos)
        return false; // early out - reuse cached values
    mPos = ijk;

    auto fetch = [&](int i, int j, int k) -> ValueT& { return mC[((i + 1) << 4) + ((j + 1) << 2) + k + 1]; };

    // fetch 64 point stencil values
    for (int i = -1; i < 3; ++i) {
        for (int j = -1; j < 3; ++j) {
            fetch(i, j, -1) = mAcc->getValue(mPos + CoordT(i, j, -1));
            fetch(i, j, 0) = mAcc->getValue(mPos + CoordT(i, j, 0));
            fetch(i, j, 1) = mAcc->getValue(mPos + CoordT(i, j, 1));
            fetch(i, j, 2) = mAcc->getValue(mPos + CoordT(i, j, 2));
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
        mC[i] = ValueT(0);
#if 0
    for (int j = 0; j < 64; j += 4) {
      mC[i] = fma(A[i][j], X[j], fma(A[i][j+1], X[j+1], fma(A[i][j+2], X[j+2], fma(A[i][j+3], X[j+3], mC[i]))));
    }
#else
        for (int j = 0; j < 64; j += 4) {
            mC[i] += A[i][j] * X[j] + A[i][j + 1] * X[j + 1] + A[i][j + 2] * X[j + 2] + A[i][j + 3] * X[j + 3];
        }
#endif
    }

    return true;
}

// ------------------------------> Sampler <--------------------------------------

template<typename TreeOrAccT, bool WithCache>
struct SampleFromVoxels<TreeOrAccT, 0, WithCache> : public NearestNeighborSampler<TreeOrAccT, WithCache>
{
    __hostdev__ SampleFromVoxels(const TreeOrAccT& tree)
        : NearestNeighborSampler<TreeOrAccT, WithCache>(tree)
    {
    }
};

template<typename TreeOrAccT, bool WithCache>
struct SampleFromVoxels<TreeOrAccT, 1, WithCache> : public TrilinearSampler<TreeOrAccT, WithCache>
{
    __hostdev__ SampleFromVoxels(const TreeOrAccT& tree)
        : TrilinearSampler<TreeOrAccT, WithCache>(tree)
    {
    }
};

template<typename TreeOrAccT, bool WithCache>
struct SampleFromVoxels<TreeOrAccT, 2, WithCache> : public TriquadraticSampler<TreeOrAccT, WithCache>
{
    __hostdev__ SampleFromVoxels(const TreeOrAccT& tree)
        : TriquadraticSampler<TreeOrAccT, WithCache>(tree)
    {
    }
};

template<typename TreeOrAccT, bool WithCache>
struct SampleFromVoxels<TreeOrAccT, 3, WithCache> : public TricubicSampler<TreeOrAccT, WithCache>
{
    __hostdev__ SampleFromVoxels(const TreeOrAccT& tree)
        : TricubicSampler<TreeOrAccT, WithCache>(tree)
    {
    }
};

} // namespace nanovdb

#endif // NANOVDB_SAMPLE_FROM_VOXELS_H_HAS_BEEN_INCLUDED
