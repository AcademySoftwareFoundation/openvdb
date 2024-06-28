// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Ken Museth
///
/// @date  April 9, 2021
///
/// @file Stencils.h
///
/// @brief Defines various finite-difference stencils that allow for the
///        computation of gradients of order 1 to 5, mean curvatures,
///        gaussian curvatures, principal curvatures, tri-linear interpolation,
///        zero-crossing, laplacian, and closest point transform.

#ifndef NANOVDB_MATH_STENCILS_HAS_BEEN_INCLUDED
#define NANOVDB_MATH_STENCILS_HAS_BEEN_INCLUDED

#include <nanovdb/math/Math.h>// for __hostdev__, Vec3, Min, Max, Pow2, Pow3, Pow4

namespace nanovdb {

namespace math {

// ---------------------------- WENO5 ----------------------------

/// @brief Implementation of nominally fifth-order finite-difference WENO
/// @details This function returns the numerical flux.  See "High Order Finite Difference and
/// Finite Volume WENO Schemes and Discontinuous Galerkin Methods for CFD" - Chi-Wang Shu
/// ICASE Report No 2001-11 (page 6).  Also see ICASE No 97-65 for a more complete reference
/// (Shu, 1997).
/// Given v1 = f(x-2dx), v2 = f(x-dx), v3 = f(x), v4 = f(x+dx) and v5 = f(x+2dx),
/// return an interpolated value f(x+dx/2) with the special property that
/// ( f(x+dx/2) - f(x-dx/2) ) / dx  = df/dx (x) + error,
/// where the error is fifth-order in smooth regions: O(dx) <= error <=O(dx^5)
template<typename ValueType, typename RealT = ValueType>
__hostdev__ inline ValueType
WENO5(const ValueType& v1,
      const ValueType& v2,
      const ValueType& v3,
      const ValueType& v4,
      const ValueType& v5,
      RealT scale2 = 1.0)// openvdb uses scale2 = 0.01
{
    static const RealT C = 13.0 / 12.0;
    // WENO is formulated for non-dimensional equations, here the optional scale2
    // is a reference value (squared) for the function being interpolated.  For
    // example if 'v' is of order 1000, then scale2 = 10^6 is ok.  But in practice
    // leave scale2 = 1.
    const RealT eps = RealT(1.0e-6) * scale2;
    // {\tilde \omega_k} = \gamma_k / ( \beta_k + \epsilon)^2 in Shu's ICASE report)
    const RealT A1 = RealT(0.1)/Pow2(C*Pow2(v1-2*v2+v3)+RealT(0.25)*Pow2(v1-4*v2+3*v3)+eps),
                A2 = RealT(0.6)/Pow2(C*Pow2(v2-2*v3+v4)+RealT(0.25)*Pow2(v2-v4)+eps),
                A3 = RealT(0.3)/Pow2(C*Pow2(v3-2*v4+v5)+RealT(0.25)*Pow2(3*v3-4*v4+v5)+eps);

    return static_cast<ValueType>((A1*(2*v1 - 7*v2 + 11*v3) +
                                   A2*(5*v3 -   v2 +  2*v4) +
                                   A3*(2*v3 + 5*v4 -    v5))/(6*(A1+A2+A3)));
}

// ---------------------------- GodunovsNormSqrd ----------------------------

template <typename RealT>
__hostdev__ inline RealT
GodunovsNormSqrd(bool isOutside,
                 RealT dP_xm, RealT dP_xp,
                 RealT dP_ym, RealT dP_yp,
                 RealT dP_zm, RealT dP_zp)
{
    RealT dPLen2;
    if (isOutside) { // outside
        dPLen2  = Max(Pow2(Max(dP_xm, RealT(0))), Pow2(Min(dP_xp, RealT(0)))); // (dP/dx)2
        dPLen2 += Max(Pow2(Max(dP_ym, RealT(0))), Pow2(Min(dP_yp, RealT(0)))); // (dP/dy)2
        dPLen2 += Max(Pow2(Max(dP_zm, RealT(0))), Pow2(Min(dP_zp, RealT(0)))); // (dP/dz)2
    } else { // inside
        dPLen2  = Max(Pow2(Min(dP_xm, RealT(0))), Pow2(Max(dP_xp, RealT(0)))); // (dP/dx)2
        dPLen2 += Max(Pow2(Min(dP_ym, RealT(0))), Pow2(Max(dP_yp, RealT(0)))); // (dP/dy)2
        dPLen2 += Max(Pow2(Min(dP_zm, RealT(0))), Pow2(Max(dP_zp, RealT(0)))); // (dP/dz)2
    }
    return dPLen2; // |\nabla\phi|^2
}

template<typename RealT>
__hostdev__ inline RealT
GodunovsNormSqrd(bool isOutside,
                 const Vec3<RealT>& gradient_m,
                 const Vec3<RealT>& gradient_p)
{
    return GodunovsNormSqrd<RealT>(isOutside,
                                   gradient_m[0], gradient_p[0],
                                   gradient_m[1], gradient_p[1],
                                   gradient_m[2], gradient_p[2]);
}

// ---------------------------- BaseStencil ----------------------------

// BaseStencil uses curiously recurring template pattern (CRTP)
template<typename DerivedType, int SIZE, typename GridT>
class BaseStencil
{
public:
    using ValueType = typename GridT::ValueType;
    using GridType  = GridT;
    using TreeType  = typename GridT::TreeType;
    using AccessorType = typename GridT::AccessorType;// ReadAccessor<ValueType>;

    /// @brief Initialize the stencil buffer with the values of voxel (i, j, k)
    /// and its neighbors.
    /// @param ijk Index coordinates of stencil center
    __hostdev__ inline void moveTo(const Coord& ijk)
    {
        mCenter = ijk;
        mValues[0] = mAcc.getValue(ijk);
        static_cast<DerivedType&>(*this).init(mCenter);
    }

    /// @brief Initialize the stencil buffer with the values of voxel (i, j, k)
    /// and its neighbors. The method also takes a value of the center
    /// element of the stencil, assuming it is already known.
    /// @param ijk Index coordinates of stencil center
    /// @param centerValue Value of the center element of the stencil
    __hostdev__ inline void moveTo(const Coord& ijk, const ValueType& centerValue)
    {
        mCenter = ijk;
        mValues[0] = centerValue;
        static_cast<DerivedType&>(*this).init(mCenter);
    }

    /// @brief Initialize the stencil buffer with the values of voxel
    /// (x, y, z) and its neighbors.
    ///
    /// @note This version is slightly faster than the one above, since
    /// the center voxel's value is read directly from the iterator.
    template<typename IterType>
    __hostdev__ inline void moveTo(const IterType& iter)
    {
        mCenter = iter.getCoord();
        mValues[0] = *iter;
        static_cast<DerivedType&>(*this).init(mCenter);
    }

    /// @brief Initialize the stencil buffer with the values of voxel (x, y, z)
    /// and its neighbors.
    /// @param xyz Floating point voxel coordinates of stencil center
    /// @details This method will check to see if it is necessary to
    /// update the stencil based on the cached index coordinates of
    /// the center point.
    template<typename RealType>
    __hostdev__ inline void moveTo(const Vec3<RealType>& xyz)
    {
        Coord ijk = RoundDown(xyz);
        if (ijk != mCenter) this->moveTo(ijk);
    }

    /// @brief Return the value from the stencil buffer with linear
    /// offset pos.
    ///
    /// @note The default (@a pos = 0) corresponds to the first element
    /// which is typically the center point of the stencil.
    __hostdev__ inline const ValueType& getValue(unsigned int pos = 0) const
    {
        NANOVDB_ASSERT(pos < SIZE);
        return mValues[pos];
    }

    /// @brief Return the value at the specified location relative to the center of the stencil
    template<int i, int j, int k>
    __hostdev__ inline const ValueType& getValue() const
    {
        return mValues[static_cast<const DerivedType&>(*this).template pos<i,j,k>()];
    }

    /// @brief Set the value at the specified location relative to the center of the stencil
    template<int i, int j, int k>
    __hostdev__ inline void setValue(const ValueType& value)
    {
        mValues[static_cast<const DerivedType&>(*this).template pos<i,j,k>()] = value;
    }

    /// @brief Return the size of the stencil buffer.
    __hostdev__ static int size() { return SIZE; }

    /// @brief Return the mean value of the current stencil.
    __hostdev__ inline ValueType mean() const
    {
        ValueType sum = 0.0;
        for (int i = 0; i < SIZE; ++i) sum += mValues[i];
        return sum / ValueType(SIZE);
    }

    /// @brief Return the smallest value in the stencil buffer.
    __hostdev__ inline ValueType min() const
    {
        ValueType v = mValues[0];
        for (int i=1; i<SIZE; ++i) {
            if (mValues[i] < v) v = mValues[i];
        }
        return v;
    }

    /// @brief Return the largest value in the stencil buffer.
    __hostdev__ inline ValueType max() const
    {
        ValueType v = mValues[0];
        for (int i=1; i<SIZE; ++i) {
            if (mValues[i] > v) v = mValues[i];
        }
        return v;
    }

    /// @brief Return the coordinates of the center point of the stencil.
    __hostdev__ inline const Coord& getCenterCoord() const { return mCenter; }

    /// @brief Return the value at the center of the stencil
    __hostdev__ inline const ValueType& getCenterValue() const { return mValues[0]; }

    /// @brief Return true if the center of the stencil intersects the
    /// iso-contour specified by the isoValue
    __hostdev__ inline bool intersects(const ValueType &isoValue = ValueType(0) ) const
    {
        const bool less = this->getValue< 0, 0, 0>() < isoValue;
        return (less  ^  (this->getValue<-1, 0, 0>() < isoValue)) ||
               (less  ^  (this->getValue< 1, 0, 0>() < isoValue)) ||
               (less  ^  (this->getValue< 0,-1, 0>() < isoValue)) ||
               (less  ^  (this->getValue< 0, 1, 0>() < isoValue)) ||
               (less  ^  (this->getValue< 0, 0,-1>() < isoValue)) ||
               (less  ^  (this->getValue< 0, 0, 1>() < isoValue))  ;
    }
    struct Mask {
        uint8_t bits;
        __hostdev__ Mask() : bits(0u) {}
        __hostdev__ void set(int i) { bits |= (1 << i); }
        __hostdev__ bool test(int i) const { return bits & (1 << i); }
        __hostdev__ bool any() const  { return bits >  0u; }
        __hostdev__ bool all() const  { return bits == 255u; }
        __hostdev__ bool none() const { return bits == 0u; }
        __hostdev__ int count() const { return util::countOn(bits); }
    };// Mask

    /// @brief Return true a bit-mask where the 6 lower bits indicates if the
    /// center of the stencil intersects the iso-contour specified by the isoValue.
    ///
    /// @note There are 2^6 = 64 different possible cases, including no intersections!
    ///
    /// @details The ordering of bit mask is ( -x, +x, -y, +y, -z, +z ), so to
    /// check if there is an intersection in -y use (mask & (1u<<2)) where mask is
    /// ther return value from this function. To check if there are any
    /// intersections use mask!=0u, and for no intersections use mask==0u.
    /// To count the number of intersections use __builtin_popcount(mask).
    __hostdev__ inline Mask intersectionMask(ValueType isoValue = ValueType(0)) const
    {
        Mask mask;
        const bool less = this->getValue< 0, 0, 0>() < isoValue;
        if (less ^ (this->getValue<-1, 0, 0>() < isoValue)) mask.set(0);// |=  1u;
        if (less ^ (this->getValue< 1, 0, 0>() < isoValue)) mask.set(1);// |=  2u;
        if (less ^ (this->getValue< 0,-1, 0>() < isoValue)) mask.set(2);// |=  4u;
        if (less ^ (this->getValue< 0, 1, 0>() < isoValue)) mask.set(3);// |=  8u;
        if (less ^ (this->getValue< 0, 0,-1>() < isoValue)) mask.set(4);// |= 16u;
        if (less ^ (this->getValue< 0, 0, 1>() < isoValue)) mask.set(5);// |= 32u;
        return mask;
    }

    /// @brief Return a const reference to the grid from which this
    /// stencil was constructed.
    __hostdev__ inline const GridType& grid() const { return *mGrid; }

    /// @brief Return a const reference to the ValueAccessor
    /// associated with this Stencil.
    __hostdev__ inline const AccessorType& accessor() const { return mAcc; }

protected:
    // Constructor is protected to prevent direct instantiation.
    __hostdev__ BaseStencil(const GridType& grid)
        : mGrid(&grid)
        , mAcc(grid)
        , mCenter(Coord::max())
    {
    }

    const GridType* mGrid;
    AccessorType    mAcc;
    ValueType       mValues[SIZE];
    Coord           mCenter;

}; // BaseStencil class


// ---------------------------- BoxStencil ----------------------------


namespace { // anonymous namespace for stencil-layout map

    // the eight point box stencil
    template<int i, int j, int k> struct BoxPt {};
    template<> struct BoxPt< 0, 0, 0> { enum { idx = 0 }; };
    template<> struct BoxPt< 0, 0, 1> { enum { idx = 1 }; };
    template<> struct BoxPt< 0, 1, 1> { enum { idx = 2 }; };
    template<> struct BoxPt< 0, 1, 0> { enum { idx = 3 }; };
    template<> struct BoxPt< 1, 0, 0> { enum { idx = 4 }; };
    template<> struct BoxPt< 1, 0, 1> { enum { idx = 5 }; };
    template<> struct BoxPt< 1, 1, 1> { enum { idx = 6 }; };
    template<> struct BoxPt< 1, 1, 0> { enum { idx = 7 }; };

}

template<typename GridT>
class BoxStencil: public BaseStencil<BoxStencil<GridT>, 8, GridT>
{
    using SelfT     = BoxStencil<GridT>;
    using BaseType  = BaseStencil<SelfT, 8, GridT>;
public:
    using GridType  = GridT;
    using TreeType  = typename GridT::TreeType;
    using ValueType = typename GridT::ValueType;

    static constexpr int SIZE = 8;

    __hostdev__ BoxStencil(const GridType& grid) : BaseType(grid) {}

    /// Return linear offset for the specified stencil point relative to its center
    template<int i, int j, int k>
    __hostdev__ unsigned int pos() const { return BoxPt<i,j,k>::idx; }

     /// @brief Return true if the center of the stencil intersects the
    /// iso-contour specified by the isoValue
    __hostdev__ inline bool intersects(ValueType isoValue = ValueType(0)) const
    {
        const bool less = mValues[0] < isoValue;
        return (less  ^  (mValues[1] < isoValue)) ||
               (less  ^  (mValues[2] < isoValue)) ||
               (less  ^  (mValues[3] < isoValue)) ||
               (less  ^  (mValues[4] < isoValue)) ||
               (less  ^  (mValues[5] < isoValue)) ||
               (less  ^  (mValues[6] < isoValue)) ||
               (less  ^  (mValues[7] < isoValue))  ;
    }

    /// @brief Return the trilinear interpolation at the normalized position.
    /// @param xyz Floating point coordinate position. Index space and NOT world space.
    /// @warning It is assumed that the stencil has already been moved
    /// to the relevant voxel position, e.g. using moveTo(xyz).
    /// @note Trilinear interpolation kernal reads as:
    ///       v000 (1-u)(1-v)(1-w) + v001 (1-u)(1-v)w + v010 (1-u)v(1-w) + v011 (1-u)vw
    ///     + v100 u(1-v)(1-w)     + v101 u(1-v)w     + v110 uv(1-w)     + v111 uvw
    __hostdev__ inline ValueType interpolation(const Vec3<ValueType>& xyz) const
    {
        const ValueType u = xyz[0] - mCenter[0];
        const ValueType v = xyz[1] - mCenter[1];
        const ValueType w = xyz[2] - mCenter[2];

        NANOVDB_ASSERT(u>=0 && u<=1);
        NANOVDB_ASSERT(v>=0 && v<=1);
        NANOVDB_ASSERT(w>=0 && w<=1);

        ValueType V = BaseType::template getValue<0,0,0>();
        ValueType A = V + (BaseType::template getValue<0,0,1>() - V) * w;
        V = BaseType::template getValue< 0, 1, 0>();
        ValueType B = V + (BaseType::template getValue<0,1,1>() - V) * w;
        ValueType C = A + (B - A) * v;

        V = BaseType::template getValue<1,0,0>();
        A = V + (BaseType::template getValue<1,0,1>() - V) * w;
        V = BaseType::template getValue<1,1,0>();
        B = V + (BaseType::template getValue<1,1,1>() - V) * w;
        ValueType D = A + (B - A) * v;

        return C + (D - C) * u;
    }

    /// @brief Return the gradient in world space of the trilinear interpolation kernel.
    /// @param xyz Floating point coordinate position.
    /// @warning It is assumed that the stencil has already been moved
    /// to the relevant voxel position, e.g. using moveTo(xyz).
    /// @note Computed as partial derivatives of the trilinear interpolation kernel:
    ///       v000 (1-u)(1-v)(1-w) + v001 (1-u)(1-v)w + v010 (1-u)v(1-w) + v011 (1-u)vw
    ///     + v100 u(1-v)(1-w)     + v101 u(1-v)w     + v110 uv(1-w)     + v111 uvw
    __hostdev__ inline Vec3<ValueType> gradient(const Vec3<ValueType>& xyz) const
    {
        const ValueType u = xyz[0] - mCenter[0];
        const ValueType v = xyz[1] - mCenter[1];
        const ValueType w = xyz[2] - mCenter[2];

        NANOVDB_ASSERT(u>=0 && u<=1);
        NANOVDB_ASSERT(v>=0 && v<=1);
        NANOVDB_ASSERT(w>=0 && w<=1);

        ValueType D[4]={BaseType::template getValue<0,0,1>()-BaseType::template getValue<0,0,0>(),
                        BaseType::template getValue<0,1,1>()-BaseType::template getValue<0,1,0>(),
                        BaseType::template getValue<1,0,1>()-BaseType::template getValue<1,0,0>(),
                        BaseType::template getValue<1,1,1>()-BaseType::template getValue<1,1,0>()};

        // Z component
        ValueType A = D[0] + (D[1]- D[0]) * v;
        ValueType B = D[2] + (D[3]- D[2]) * v;
        Vec3<ValueType> grad(0, 0, A + (B - A) * u);

        D[0] = BaseType::template getValue<0,0,0>() + D[0] * w;
        D[1] = BaseType::template getValue<0,1,0>() + D[1] * w;
        D[2] = BaseType::template getValue<1,0,0>() + D[2] * w;
        D[3] = BaseType::template getValue<1,1,0>() + D[3] * w;

        // X component
        A = D[0] + (D[1] - D[0]) * v;
        B = D[2] + (D[3] - D[2]) * v;

        grad[0] = B - A;

        // Y component
        A = D[1] - D[0];
        B = D[3] - D[2];

        grad[1] = A + (B - A) * u;

        return BaseType::mGrid->map().applyIJT(grad);
    }

private:
    __hostdev__ inline void init(const Coord& ijk)
    {
        mValues[ 1] = mAcc.getValue(ijk.offsetBy( 0, 0, 1));
        mValues[ 2] = mAcc.getValue(ijk.offsetBy( 0, 1, 1));
        mValues[ 3] = mAcc.getValue(ijk.offsetBy( 0, 1, 0));
        mValues[ 4] = mAcc.getValue(ijk.offsetBy( 1, 0, 0));
        mValues[ 5] = mAcc.getValue(ijk.offsetBy( 1, 0, 1));
        mValues[ 6] = mAcc.getValue(ijk.offsetBy( 1, 1, 1));
        mValues[ 7] = mAcc.getValue(ijk.offsetBy( 1, 1, 0));
    }

    template<typename, int, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mAcc;
    using BaseType::mValues;
    using BaseType::mCenter;
};// BoxStencil class


// ---------------------------- GradStencil ----------------------------

namespace { // anonymous namespace for stencil-layout map

    template<int i, int j, int k> struct GradPt {};
    template<> struct GradPt< 0, 0, 0> { enum { idx = 0 }; };
    template<> struct GradPt< 1, 0, 0> { enum { idx = 2 }; };
    template<> struct GradPt< 0, 1, 0> { enum { idx = 4 }; };
    template<> struct GradPt< 0, 0, 1> { enum { idx = 6 }; };
    template<> struct GradPt<-1, 0, 0> { enum { idx = 1 }; };
    template<> struct GradPt< 0,-1, 0> { enum { idx = 3 }; };
    template<> struct GradPt< 0, 0,-1> { enum { idx = 5 }; };
}

/// This is a simple 7-point nearest neighbor stencil that supports
/// gradient by second-order central differencing, first-order upwinding,
/// Laplacian, closest-point transform and zero-crossing test.
///
/// @note For optimal random access performance this class
/// includes its own grid accessor.
template<typename GridT>
class GradStencil : public BaseStencil<GradStencil<GridT>, 7, GridT>
{
    using SelfT     = GradStencil<GridT>;
    using BaseType  = BaseStencil<SelfT, 7, GridT>;
public:
    using GridType  = GridT;
    using TreeType  = typename GridT::TreeType;
    using ValueType = typename GridT::ValueType;

    static constexpr int SIZE = 7;

    __hostdev__ GradStencil(const GridType& grid)
        : BaseType(grid)
        , mInv2Dx(ValueType(0.5 / grid.voxelSize()[0]))
        , mInvDx2(ValueType(4.0 * mInv2Dx * mInv2Dx))
    {
    }

    __hostdev__ GradStencil(const GridType& grid, double dx)
        : BaseType(grid)
        , mInv2Dx(ValueType(0.5 / dx))
        , mInvDx2(ValueType(4.0 * mInv2Dx * mInv2Dx))
    {
    }

    /// @brief Return the norm square of the single-sided upwind gradient
    /// (computed via Godunov's scheme) at the previously buffered location.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline ValueType normSqGrad() const
    {
        return mInvDx2 * GodunovsNormSqrd(mValues[0] > ValueType(0),
                                          mValues[0] - mValues[1],
                                          mValues[2] - mValues[0],
                                          mValues[0] - mValues[3],
                                          mValues[4] - mValues[0],
                                          mValues[0] - mValues[5],
                                          mValues[6] - mValues[0]);
    }

    /// @brief Return the gradient computed at the previously buffered
    /// location by second order central differencing.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline Vec3<ValueType> gradient() const
    {
        return Vec3<ValueType>(mValues[2] - mValues[1],
                               mValues[4] - mValues[3],
                               mValues[6] - mValues[5])*mInv2Dx;
    }
    /// @brief Return the first-order upwind gradient corresponding to the direction V.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline Vec3<ValueType> gradient(const Vec3<ValueType>& V) const
    {
        return Vec3<ValueType>(
               V[0]>0 ? mValues[0] - mValues[1] : mValues[2] - mValues[0],
               V[1]>0 ? mValues[0] - mValues[3] : mValues[4] - mValues[0],
               V[2]>0 ? mValues[0] - mValues[5] : mValues[6] - mValues[0])*2*mInv2Dx;
    }

    /// Return the Laplacian computed at the previously buffered
    /// location by second-order central differencing.
    __hostdev__ inline ValueType laplacian() const
    {
        return mInvDx2 * (mValues[1] + mValues[2] +
                          mValues[3] + mValues[4] +
                          mValues[5] + mValues[6] - 6*mValues[0]);
    }

    /// Return @c true if the sign of the value at the center point of the stencil
    /// is different from the signs of any of its six nearest neighbors.
    __hostdev__ inline bool zeroCrossing() const
    {
        return (mValues[0]>0 ? (mValues[1]<0 || mValues[2]<0 || mValues[3]<0 || mValues[4]<0 || mValues[5]<0 || mValues[6]<0)
                             : (mValues[1]>0 || mValues[2]>0 || mValues[3]>0 || mValues[4]>0 || mValues[5]>0 || mValues[6]>0));
    }

    /// @brief Compute the closest-point transform to a level set.
    /// @return the closest point in index space to the surface
    /// from which the level set was derived.
    ///
    /// @note This method assumes that the grid represents a level set
    /// with distances in world units and a simple affine transfrom
    /// with uniform scaling.
    __hostdev__ inline Vec3<ValueType> cpt()
    {
        const Coord& ijk = BaseType::getCenterCoord();
        const ValueType d = ValueType(mValues[0] * 0.5 * mInvDx2); // distance in voxels / (2dx^2)
        const auto value = Vec3<ValueType>(ijk[0] - d*(mValues[2] - mValues[1]),
                                           ijk[1] - d*(mValues[4] - mValues[3]),
                                           ijk[2] - d*(mValues[6] - mValues[5]));
        return value;
    }

    /// Return linear offset for the specified stencil point relative to its center
    template<int i, int j, int k>
    __hostdev__ unsigned int pos() const { return GradPt<i,j,k>::idx; }

private:

    __hostdev__ inline void init(const Coord& ijk)
    {
        mValues[ 1] = mAcc.getValue(ijk.offsetBy(-1, 0, 0));
        mValues[ 2] = mAcc.getValue(ijk.offsetBy( 1, 0, 0));

        mValues[ 3] = mAcc.getValue(ijk.offsetBy( 0,-1, 0));
        mValues[ 4] = mAcc.getValue(ijk.offsetBy( 0, 1, 0));

        mValues[ 5] = mAcc.getValue(ijk.offsetBy( 0, 0,-1));
        mValues[ 6] = mAcc.getValue(ijk.offsetBy( 0, 0, 1));
    }

    template<typename, int, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mAcc;
    using BaseType::mValues;
    const ValueType mInv2Dx, mInvDx2;
}; // GradStencil class


// ---------------------------- WenoStencil ----------------------------

namespace { // anonymous namespace for stencil-layout map

    template<int i, int j, int k> struct WenoPt {};
    template<> struct WenoPt< 0, 0, 0> { enum { idx = 0 }; };

    template<> struct WenoPt<-3, 0, 0> { enum { idx = 1 }; };
    template<> struct WenoPt<-2, 0, 0> { enum { idx = 2 }; };
    template<> struct WenoPt<-1, 0, 0> { enum { idx = 3 }; };
    template<> struct WenoPt< 1, 0, 0> { enum { idx = 4 }; };
    template<> struct WenoPt< 2, 0, 0> { enum { idx = 5 }; };
    template<> struct WenoPt< 3, 0, 0> { enum { idx = 6 }; };

    template<> struct WenoPt< 0,-3, 0> { enum { idx = 7 }; };
    template<> struct WenoPt< 0,-2, 0> { enum { idx = 8 }; };
    template<> struct WenoPt< 0,-1, 0> { enum { idx = 9 }; };
    template<> struct WenoPt< 0, 1, 0> { enum { idx =10 }; };
    template<> struct WenoPt< 0, 2, 0> { enum { idx =11 }; };
    template<> struct WenoPt< 0, 3, 0> { enum { idx =12 }; };

    template<> struct WenoPt< 0, 0,-3> { enum { idx =13 }; };
    template<> struct WenoPt< 0, 0,-2> { enum { idx =14 }; };
    template<> struct WenoPt< 0, 0,-1> { enum { idx =15 }; };
    template<> struct WenoPt< 0, 0, 1> { enum { idx =16 }; };
    template<> struct WenoPt< 0, 0, 2> { enum { idx =17 }; };
    template<> struct WenoPt< 0, 0, 3> { enum { idx =18 }; };

}

/// @brief This is a special 19-point stencil that supports optimal fifth-order WENO
/// upwinding, second-order central differencing, Laplacian, and zero-crossing test.
///
/// @note For optimal random access performance this class
/// includes its own grid accessor.
template<typename GridT, typename RealT = typename GridT::ValueType>
class WenoStencil: public BaseStencil<WenoStencil<GridT>, 19, GridT>
{
    using SelfT     = WenoStencil<GridT>;
    using BaseType  = BaseStencil<SelfT, 19, GridT>;
public:
    using GridType  = GridT;
    using TreeType  = typename GridT::TreeType;
    using ValueType = typename GridT::ValueType;

    static constexpr int SIZE = 19;

    __hostdev__ WenoStencil(const GridType& grid)
        : BaseType(grid)
        , mDx2(ValueType(Pow2(grid.voxelSize()[0])))
        , mInv2Dx(ValueType(0.5 / grid.voxelSize()[0]))
        , mInvDx2(ValueType(1.0 / mDx2))
    {
    }

    __hostdev__ WenoStencil(const GridType& grid, double dx)
        : BaseType(grid)
        , mDx2(ValueType(dx * dx))
        , mInv2Dx(ValueType(0.5 / dx))
        , mInvDx2(ValueType(1.0 / mDx2))
    {
    }

    /// @brief Return the norm-square of the WENO upwind gradient (computed via
    /// WENO upwinding and Godunov's scheme) at the previously buffered location.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline ValueType normSqGrad(ValueType isoValue = ValueType(0)) const
    {
        const ValueType* v = mValues;
        const RealT
            dP_xm = WENO5<RealT>(v[ 2]-v[ 1],v[ 3]-v[ 2],v[ 0]-v[ 3],v[ 4]-v[ 0],v[ 5]-v[ 4],mDx2),
            dP_xp = WENO5<RealT>(v[ 6]-v[ 5],v[ 5]-v[ 4],v[ 4]-v[ 0],v[ 0]-v[ 3],v[ 3]-v[ 2],mDx2),
            dP_ym = WENO5<RealT>(v[ 8]-v[ 7],v[ 9]-v[ 8],v[ 0]-v[ 9],v[10]-v[ 0],v[11]-v[10],mDx2),
            dP_yp = WENO5<RealT>(v[12]-v[11],v[11]-v[10],v[10]-v[ 0],v[ 0]-v[ 9],v[ 9]-v[ 8],mDx2),
            dP_zm = WENO5<RealT>(v[14]-v[13],v[15]-v[14],v[ 0]-v[15],v[16]-v[ 0],v[17]-v[16],mDx2),
            dP_zp = WENO5<RealT>(v[18]-v[17],v[17]-v[16],v[16]-v[ 0],v[ 0]-v[15],v[15]-v[14],mDx2);
        return mInvDx2*static_cast<ValueType>(
            GodunovsNormSqrd(v[0]>isoValue, dP_xm, dP_xp, dP_ym, dP_yp, dP_zm, dP_zp));
    }

    /// Return the optimal fifth-order upwind gradient corresponding to the
    /// direction V.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline Vec3<ValueType> gradient(const Vec3<ValueType>& V) const
    {
        const ValueType* v = mValues;
        return 2*mInv2Dx * Vec3<ValueType>(
            V[0]>0 ? WENO5<RealT>(v[ 2]-v[ 1],v[ 3]-v[ 2],v[ 0]-v[ 3], v[ 4]-v[ 0],v[ 5]-v[ 4],mDx2)
                   : WENO5<RealT>(v[ 6]-v[ 5],v[ 5]-v[ 4],v[ 4]-v[ 0], v[ 0]-v[ 3],v[ 3]-v[ 2],mDx2),
            V[1]>0 ? WENO5<RealT>(v[ 8]-v[ 7],v[ 9]-v[ 8],v[ 0]-v[ 9], v[10]-v[ 0],v[11]-v[10],mDx2)
                   : WENO5<RealT>(v[12]-v[11],v[11]-v[10],v[10]-v[ 0], v[ 0]-v[ 9],v[ 9]-v[ 8],mDx2),
            V[2]>0 ? WENO5<RealT>(v[14]-v[13],v[15]-v[14],v[ 0]-v[15], v[16]-v[ 0],v[17]-v[16],mDx2)
                   : WENO5<RealT>(v[18]-v[17],v[17]-v[16],v[16]-v[ 0], v[ 0]-v[15],v[15]-v[14],mDx2));
    }
    /// Return the gradient computed at the previously buffered
    /// location by second-order central differencing.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline Vec3<ValueType> gradient() const
    {
        return mInv2Dx * Vec3<ValueType>(mValues[ 4] - mValues[ 3],
                                         mValues[10] - mValues[ 9],
                                         mValues[16] - mValues[15]);
    }

    /// Return the Laplacian computed at the previously buffered
    /// location by second-order central differencing.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline ValueType laplacian() const
    {
        return mInvDx2 * (
            mValues[ 3] + mValues[ 4] +
            mValues[ 9] + mValues[10] +
            mValues[15] + mValues[16] - 6*mValues[0]);
    }

    /// Return @c true if the sign of the value at the center point of the stencil
    /// differs from the sign of any of its six nearest neighbors
    __hostdev__ inline bool zeroCrossing() const
    {
        const ValueType* v = mValues;
        return (v[ 0]>0 ? (v[ 3]<0 || v[ 4]<0 || v[ 9]<0 || v[10]<0 || v[15]<0 || v[16]<0)
                        : (v[ 3]>0 || v[ 4]>0 || v[ 9]>0 || v[10]>0 || v[15]>0 || v[16]>0));
    }

    /// Return linear offset for the specified stencil point relative to its center
    template<int i, int j, int k>
    __hostdev__ unsigned int pos() const { return WenoPt<i,j,k>::idx; }

private:
    __hostdev__ inline void init(const Coord& ijk)
    {
        mValues[ 1] = mAcc.getValue(ijk.offsetBy(-3,  0,  0));
        mValues[ 2] = mAcc.getValue(ijk.offsetBy(-2,  0,  0));
        mValues[ 3] = mAcc.getValue(ijk.offsetBy(-1,  0,  0));
        mValues[ 4] = mAcc.getValue(ijk.offsetBy( 1,  0,  0));
        mValues[ 5] = mAcc.getValue(ijk.offsetBy( 2,  0,  0));
        mValues[ 6] = mAcc.getValue(ijk.offsetBy( 3,  0,  0));

        mValues[ 7] = mAcc.getValue(ijk.offsetBy( 0, -3,  0));
        mValues[ 8] = mAcc.getValue(ijk.offsetBy( 0, -2,  0));
        mValues[ 9] = mAcc.getValue(ijk.offsetBy( 0, -1,  0));
        mValues[10] = mAcc.getValue(ijk.offsetBy( 0,  1,  0));
        mValues[11] = mAcc.getValue(ijk.offsetBy( 0,  2,  0));
        mValues[12] = mAcc.getValue(ijk.offsetBy( 0,  3,  0));

        mValues[13] = mAcc.getValue(ijk.offsetBy( 0,  0, -3));
        mValues[14] = mAcc.getValue(ijk.offsetBy( 0,  0, -2));
        mValues[15] = mAcc.getValue(ijk.offsetBy( 0,  0, -1));
        mValues[16] = mAcc.getValue(ijk.offsetBy( 0,  0,  1));
        mValues[17] = mAcc.getValue(ijk.offsetBy( 0,  0,  2));
        mValues[18] = mAcc.getValue(ijk.offsetBy( 0,  0,  3));
    }

    template<typename, int, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mAcc;
    using BaseType::mValues;
    const ValueType mDx2, mInv2Dx, mInvDx2;
}; // WenoStencil class


// ---------------------------- CurvatureStencil ----------------------------

namespace { // anonymous namespace for stencil-layout map

    template<int i, int j, int k> struct CurvPt {};
    template<> struct CurvPt< 0, 0, 0> { enum { idx = 0 }; };

    template<> struct CurvPt<-1, 0, 0> { enum { idx = 1 }; };
    template<> struct CurvPt< 1, 0, 0> { enum { idx = 2 }; };

    template<> struct CurvPt< 0,-1, 0> { enum { idx = 3 }; };
    template<> struct CurvPt< 0, 1, 0> { enum { idx = 4 }; };

    template<> struct CurvPt< 0, 0,-1> { enum { idx = 5 }; };
    template<> struct CurvPt< 0, 0, 1> { enum { idx = 6 }; };

    template<> struct CurvPt<-1,-1, 0> { enum { idx = 7 }; };
    template<> struct CurvPt< 1,-1, 0> { enum { idx = 8 }; };
    template<> struct CurvPt<-1, 1, 0> { enum { idx = 9 }; };
    template<> struct CurvPt< 1, 1, 0> { enum { idx =10 }; };

    template<> struct CurvPt<-1, 0,-1> { enum { idx =11 }; };
    template<> struct CurvPt< 1, 0,-1> { enum { idx =12 }; };
    template<> struct CurvPt<-1, 0, 1> { enum { idx =13 }; };
    template<> struct CurvPt< 1, 0, 1> { enum { idx =14 }; };

    template<> struct CurvPt< 0,-1,-1> { enum { idx =15 }; };
    template<> struct CurvPt< 0, 1,-1> { enum { idx =16 }; };
    template<> struct CurvPt< 0,-1, 1> { enum { idx =17 }; };
    template<> struct CurvPt< 0, 1, 1> { enum { idx =18 }; };

}

template<typename GridT, typename RealT = typename GridT::ValueType>
class CurvatureStencil: public BaseStencil<CurvatureStencil<GridT>, 19, GridT>
{
    using SelfT     = CurvatureStencil<GridT>;
    using BaseType  = BaseStencil<SelfT, 19, GridT>;
public:
    using GridType  = GridT;
    using TreeType  = typename GridT::TreeType;
    using ValueType = typename GridT::ValueType;

    static constexpr int SIZE = 19;

    __hostdev__ CurvatureStencil(const GridType& grid)
        : BaseType(grid)
        , mInv2Dx(ValueType(0.5 / grid.voxelSize()[0]))
        , mInvDx2(ValueType(4.0 * mInv2Dx * mInv2Dx))
    {
    }

    __hostdev__ CurvatureStencil(const GridType& grid, double dx)
        : BaseType(grid)
        , mInv2Dx(ValueType(0.5 / dx))
        , mInvDx2(ValueType(4.0 * mInv2Dx * mInv2Dx))
    {
    }

    /// @brief Return the mean curvature at the previously buffered location.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline ValueType meanCurvature() const
    {
        RealT alpha, normGrad;
        return this->meanCurvature(alpha, normGrad) ?
               ValueType(alpha*mInv2Dx/Pow3(normGrad)) : 0;
    }

    /// @brief Return the Gaussian curvature at the previously buffered location.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline ValueType gaussianCurvature() const
    {
        RealT alpha, normGrad;
        return this->gaussianCurvature(alpha, normGrad) ?
               ValueType(alpha*mInvDx2/Pow4(normGrad)) : 0;
    }

    /// @brief Return both the mean and the Gaussian curvature at the
    ///        previously buffered location.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline void curvatures(ValueType &mean, ValueType& gauss) const
    {
        RealT alphaM, alphaG, normGrad;
        if (this->curvatures(alphaM, alphaG, normGrad)) {
          mean  = ValueType(alphaM*mInv2Dx/Pow3(normGrad));
          gauss = ValueType(alphaG*mInvDx2/Pow4(normGrad));
        } else {
          mean = gauss = 0;
        }
    }

    /// Return the mean curvature multiplied by the norm of the
    /// central-difference gradient. This method is very useful for
    /// mean-curvature flow of level sets!
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline ValueType meanCurvatureNormGrad() const
    {
        RealT alpha, normGrad;
        return this->meanCurvature(alpha, normGrad) ?
               ValueType(alpha*mInvDx2/(2*Pow2(normGrad))) : 0;
    }

    /// Return the mean Gaussian multiplied by the norm of the
    /// central-difference gradient.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline ValueType gaussianCurvatureNormGrad() const
    {
        RealT alpha, normGrad;
        return this->gaussianCurvature(alpha, normGrad) ?
               ValueType(2*alpha*mInv2Dx*mInvDx2/Pow3(normGrad)) : 0;
    }

    /// @brief Return both the mean and the Gaussian curvature at the
    ///        previously buffered location.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline void curvaturesNormGrad(ValueType &mean, ValueType& gauss) const
    {
        RealT alphaM, alphaG, normGrad;
        if (this->curvatures(alphaM, alphaG, normGrad)) {
          mean  = ValueType(alphaM*mInvDx2/(2*Pow2(normGrad)));
          gauss = ValueType(2*alphaG*mInv2Dx*mInvDx2/Pow3(normGrad));
        } else {
          mean = gauss = 0;
        }
    }

    /// @brief Computes the minimum and maximum principal curvature at the
    ///        previously buffered location.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline void principalCurvatures(ValueType &min, ValueType &max) const
    {
        min = max = 0;
        RealT alphaM, alphaG, normGrad;
        if (this->curvatures(alphaM, alphaG, normGrad)) {
            const RealT mean = alphaM*mInv2Dx/Pow3(normGrad);
            const RealT tmp = Sqrt(mean*mean - alphaG*mInvDx2/Pow4(normGrad));
            min = ValueType(mean - tmp);
            max = ValueType(mean + tmp);
        }
    }

    /// Return the Laplacian computed at the previously buffered
    /// location by second-order central differencing.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline ValueType laplacian() const
    {
        return mInvDx2 * (
            mValues[1] + mValues[2] +
            mValues[3] + mValues[4] +
            mValues[5] + mValues[6] - 6*mValues[0]);
    }

    /// Return the gradient computed at the previously buffered
    /// location by second-order central differencing.
    ///
    /// @note This method should not be called until the stencil
    /// buffer has been populated via a call to moveTo(ijk).
    __hostdev__ inline Vec3<ValueType> gradient() const
    {
        return Vec3<ValueType>(
            mValues[2] - mValues[1],
            mValues[4] - mValues[3],
            mValues[6] - mValues[5])*mInv2Dx;
    }

    /// Return linear offset for the specified stencil point relative to its center
    template<int i, int j, int k>
    __hostdev__ unsigned int pos() const { return CurvPt<i,j,k>::idx; }

private:
    __hostdev__ inline void init(const Coord &ijk)
    {
        mValues[ 1] = mAcc.getValue(ijk.offsetBy(-1,  0,  0));
        mValues[ 2] = mAcc.getValue(ijk.offsetBy( 1,  0,  0));

        mValues[ 3] = mAcc.getValue(ijk.offsetBy( 0, -1,  0));
        mValues[ 4] = mAcc.getValue(ijk.offsetBy( 0,  1,  0));

        mValues[ 5] = mAcc.getValue(ijk.offsetBy( 0,  0, -1));
        mValues[ 6] = mAcc.getValue(ijk.offsetBy( 0,  0,  1));

        mValues[ 7] = mAcc.getValue(ijk.offsetBy(-1, -1,  0));
        mValues[ 8] = mAcc.getValue(ijk.offsetBy( 1, -1,  0));
        mValues[ 9] = mAcc.getValue(ijk.offsetBy(-1,  1,  0));
        mValues[10] = mAcc.getValue(ijk.offsetBy( 1,  1,  0));

        mValues[11] = mAcc.getValue(ijk.offsetBy(-1,  0, -1));
        mValues[12] = mAcc.getValue(ijk.offsetBy( 1,  0, -1));
        mValues[13] = mAcc.getValue(ijk.offsetBy(-1,  0,  1));
        mValues[14] = mAcc.getValue(ijk.offsetBy( 1,  0,  1));

        mValues[15] = mAcc.getValue(ijk.offsetBy( 0, -1, -1));
        mValues[16] = mAcc.getValue(ijk.offsetBy( 0,  1, -1));
        mValues[17] = mAcc.getValue(ijk.offsetBy( 0, -1,  1));
        mValues[18] = mAcc.getValue(ijk.offsetBy( 0,  1,  1));
    }

    __hostdev__ inline RealT Dx()  const { return 0.5*(mValues[2] - mValues[1]); }// * 1/dx
    __hostdev__ inline RealT Dy()  const { return 0.5*(mValues[4] - mValues[3]); }// * 1/dx
    __hostdev__ inline RealT Dz()  const { return 0.5*(mValues[6] - mValues[5]); }// * 1/dx
    __hostdev__ inline RealT Dxx() const { return mValues[2] - 2 * mValues[0] + mValues[1]; }// * 1/dx2
    __hostdev__ inline RealT Dyy() const { return mValues[4] - 2 * mValues[0] + mValues[3]; }// * 1/dx2}
    __hostdev__ inline RealT Dzz() const { return mValues[6] - 2 * mValues[0] + mValues[5]; }// * 1/dx2
    __hostdev__ inline RealT Dxy() const { return 0.25 * (mValues[10] - mValues[ 8] + mValues[ 7] - mValues[ 9]); }// * 1/dx2
    __hostdev__ inline RealT Dxz() const { return 0.25 * (mValues[14] - mValues[12] + mValues[11] - mValues[13]); }// * 1/dx2
    __hostdev__ inline RealT Dyz() const { return 0.25 * (mValues[18] - mValues[16] + mValues[15] - mValues[17]); }// * 1/dx2

    __hostdev__ inline bool meanCurvature(RealT& alpha, RealT& normGrad) const
    {
        // For performance all finite differences are unscaled wrt dx
        const RealT Dx  = this->Dx(), Dy = this->Dy(), Dz = this->Dz(),
                    Dx2 = Dx*Dx, Dy2 = Dy*Dy, Dz2 = Dz*Dz, normGrad2 = Dx2 + Dy2 + Dz2;
        if (normGrad2 <= Tolerance<RealT>::value()) {
             alpha = normGrad = 0;
             return false;
        }
        const RealT Dxx = this->Dxx(), Dyy = this->Dyy(), Dzz = this->Dzz();
        alpha = Dx2*(Dyy + Dzz) + Dy2*(Dxx + Dzz) + Dz2*(Dxx + Dyy) -
                2*(Dx*(Dy*this->Dxy() + Dz*this->Dxz()) + Dy*Dz*this->Dyz());// * 1/dx^4
        normGrad = Sqrt(normGrad2); // * 1/dx
        return true;
    }

    __hostdev__ inline bool gaussianCurvature(RealT& alpha, RealT& normGrad) const
    {
        // For performance all finite differences are unscaled wrt dx
        const RealT Dx  = this->Dx(), Dy = this->Dy(), Dz = this->Dz(),
                    Dx2 = Dx*Dx, Dy2 = Dy*Dy, Dz2 = Dz*Dz, normGrad2 = Dx2 + Dy2 + Dz2;
        if (normGrad2 <= Tolerance<RealT>::value()) {
             alpha = normGrad = 0;
             return false;
        }
        const RealT Dxx = this->Dxx(), Dyy = this->Dyy(), Dzz = this->Dzz(),
                   Dxy = this->Dxy(), Dxz = this->Dxz(), Dyz = this->Dyz();
        alpha = Dx2*(Dyy*Dzz - Dyz*Dyz) + Dy2*(Dxx*Dzz - Dxz*Dxz) + Dz2*(Dxx*Dyy - Dxy*Dxy) +
                2*( Dy*Dz*(Dxy*Dxz - Dyz*Dxx) + Dx*Dz*(Dxy*Dyz - Dxz*Dyy) + Dx*Dy*(Dxz*Dyz - Dxy*Dzz) );// * 1/dx^6
        normGrad  = Sqrt(normGrad2); // * 1/dx
        return true;
    }

    __hostdev__ inline bool curvatures(RealT& alphaM, RealT& alphaG, RealT& normGrad) const
    {
        // For performance all finite differences are unscaled wrt dx
        const RealT Dx  = this->Dx(), Dy = this->Dy(), Dz = this->Dz(),
                    Dx2 = Dx*Dx, Dy2 = Dy*Dy, Dz2 = Dz*Dz, normGrad2 = Dx2 + Dy2 + Dz2;
        if (normGrad2 <= Tolerance<RealT>::value()) {
             alphaM = alphaG =normGrad = 0;
             return false;
        }
        const RealT Dxx = this->Dxx(), Dyy = this->Dyy(), Dzz = this->Dzz(),
                    Dxy = this->Dxy(), Dxz = this->Dxz(), Dyz = this->Dyz();
        alphaM = Dx2*(Dyy + Dzz) + Dy2*(Dxx + Dzz) + Dz2*(Dxx + Dyy) -
                 2*(Dx*(Dy*Dxy + Dz*Dxz) + Dy*Dz*Dyz);// *1/dx^4
        alphaG = Dx2*(Dyy*Dzz - Dyz*Dyz) + Dy2*(Dxx*Dzz - Dxz*Dxz) + Dz2*(Dxx*Dyy - Dxy*Dxy) +
                 2*( Dy*Dz*(Dxy*Dxz - Dyz*Dxx) + Dx*Dz*(Dxy*Dyz - Dxz*Dyy) + Dx*Dy*(Dxz*Dyz - Dxy*Dzz) );// *1/dx^6
        normGrad  = Sqrt(normGrad2); // * 1/dx
        return true;
    }

    template<typename, int, typename> friend class BaseStencil; // allow base class to call init()
    using BaseType::mAcc;
    using BaseType::mValues;
    const ValueType mInv2Dx, mInvDx2;
}; // CurvatureStencil class

}// namespace math

} // end nanovdb namespace

#endif // NANOVDB_MATH_STENCILS_HAS_BEEN_INCLUDED
