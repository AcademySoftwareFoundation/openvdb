// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @author Richard Jones, Nick Avramoussis
///
/// @file PointRasterizeEllipsoidsSDFImpl.h
///

#ifndef OPENVDB_POINTS_RASTERIZE_ELLIPSOIDS_SDF_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_RASTERIZE_ELLIPSOIDS_SDF_IMPL_HAS_BEEN_INCLUDED

/// @brief  Dev option to experiment with the ellipsoid kernel
///     - 0 Distance to unit sphere
///     - 1 Project unit distance on elipsoid normal
///     - 2 Distance to axis-aligned ellipse
#define OPENVDB_ELLIPSOID_KERNEL_MODE 2

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @cond OPENVDB_DOCS_INTERNAL

namespace rasterize_sdf_internal
{

inline math::Vec3d
calcUnitEllipsoidBoundMaxSq(const math::Mat3s& ellipsoidTransform)
{
    math::Vec3d boundMax;
    for (int i = 0; i < 3; i++) {
        boundMax[i] =
            ellipsoidTransform(i,0) * ellipsoidTransform(i,0) +
            ellipsoidTransform(i,1) * ellipsoidTransform(i,1) +
            ellipsoidTransform(i,2) * ellipsoidTransform(i,2);
    }
    return boundMax;
}

// compute a tight axis-aligned ellipsoid bounding box
// based on https://tavianator.com/exact-bounding-boxes-for-spheres-ellipsoids/
inline math::Vec3d
calcEllipsoidBoundMax(const math::Mat3s& ellipsoidTransform)
{
    math::Vec3d boundMax = calcUnitEllipsoidBoundMaxSq(ellipsoidTransform);
    boundMax[0] = std::sqrt(boundMax[0]);
    boundMax[1] = std::sqrt(boundMax[1]);
    boundMax[2] = std::sqrt(boundMax[2]);
    return boundMax;
}

struct EllipseIndicies
{
    EllipseIndicies(const points::AttributeSet::Descriptor& desc,
            const std::string& rotation,
            const std::string& pws,
            const bool xformIsQuat)
        : xform(xformIsQuat ?
            EllipseIndicies::getAttributeIndex<Quats>(desc, rotation, false) :
            EllipseIndicies::getAttributeIndex<Mat3s>(desc, rotation, false))
        , positionws(EllipseIndicies::getAttributeIndex<Vec3d>(desc, pws, true)) {}

    bool hasWorldSpacePosition() const { return positionws != std::numeric_limits<size_t>::max(); }

    const size_t xform, positionws;

private:
    template<typename ValueT>
    static inline size_t
    getAttributeIndex(const points::AttributeSet::Descriptor& desc,
                      const std::string& name,
                      const bool optional)
    {
        const size_t idx = desc.find(name);
        if (idx == points::AttributeSet::INVALID_POS) {
            if (optional) return std::numeric_limits<size_t>::max();
            throw std::runtime_error("Missing attribute - " + name);
        }
        if (typeNameAsString<ValueT>() != desc.valueType(idx)) {
            throw std::runtime_error("Wrong attribute type for attribute " + name
                + ", expected " + typeNameAsString<ValueT>());
        }
        return idx;
    }
};

template <typename SdfT,
    typename PositionCodecT,
    typename RadiusType,
    typename FilterT,
    bool CPG>
struct EllipsoidTransfer :
    public rasterize_sdf_internal::SphericalTransfer<SdfT, PositionCodecT, RadiusType, FilterT, CPG>
{
    using BaseT = rasterize_sdf_internal::SphericalTransfer<SdfT, PositionCodecT, RadiusType, FilterT, CPG>;
    using typename BaseT::TreeT;
    using typename BaseT::ValueT;

    static const Index DIM = TreeT::LeafNodeType::DIM;
    static const Index LOG2DIM = TreeT::LeafNodeType::LOG2DIM;
    // The precision of the kernel arithmetic
    using RealT = double;

    using PHandleT   = points::AttributeHandle<Vec3f>;
    using PwsHandleT = points::AttributeHandle<Vec3d>;

    EllipsoidTransfer(const size_t pidx,
            const Vec3i width,
            const RadiusType& rt,
            const math::Transform& source,
            const FilterT& filter,
            util::NullInterrupter* interrupt,
            SdfT& surface,
            const EllipseIndicies& indices,
            Int64Tree* cpg = nullptr,
            const std::unordered_map<const PointDataTree::LeafNodeType*, Index>* ids = nullptr)
        : BaseT(pidx, width, rt, source, filter, interrupt, surface, cpg, ids)
        , mIndices(indices)
        , mPositionWSHandle() {}

    EllipsoidTransfer(const EllipsoidTransfer& other)
        : BaseT(other)
        , mIndices(other.mIndices)
        , mPositionWSHandle() {}

    inline bool startPointLeaf(const PointDataTree::LeafNodeType& leaf)
    {
        const bool ret = this->BaseT::startPointLeaf(leaf);
        if (mIndices.hasWorldSpacePosition()) {
            mPositionWSHandle = std::make_unique<PwsHandleT>(leaf.constAttributeArray(mIndices.positionws));
        }
        return ret;
    }

    inline void rasterizePoint(const Coord& ijk,
                    const Index id,
                    const CoordBBox& bounds,
                    const Mat3s& xform)
    {
        if (!BaseT::filter(id)) return;

        Vec3d P;
        if (this->mPositionWSHandle) {
            // Position may have been smoothed so we need to use the ws handle
            P = this->targetTransform().worldToIndex(this->mPositionWSHandle->get(id));
        }
        else {
            const Vec3d PWS = this->sourceTransform().indexToWorld(ijk.asVec3d() + Vec3d(this->mPosition->get(id)));
            P = this->targetTransform().worldToIndex(PWS);
        }

        // Evaluate any provided radius and apply it as a scale (in world space)
        // to the current xform (which will also be in world space). We can
        // skip this step if the xform is unitary, but we have to check every
        // xform anyway.
        const auto& r = this->mRadius.eval(id);
        math::Mat3s xformws = xform.timesDiagonal(r.get() * this->mDx);

        // Extract the final world space radius from the scale xform which
        // takes into account any scale on the xform
        Vec3f radius {
            xformws.col(0).length(),
            xformws.col(1).length(),
            xformws.col(2).length()
        };

        // Normalize out the scale from out transform (i.e. make it unitary)
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                xformws(i,j) /= radius[j];
            }
        }

        // finally put the combined radius back into index space
        radius /= float(this->mDx);
        OPENVDB_ASSERT(std::isfinite(radius.x()));
        OPENVDB_ASSERT(std::isfinite(radius.y()));
        OPENVDB_ASSERT(std::isfinite(radius.z()));

        // If we have a uniform radius, treat as a sphere
        // @todo  worth using a tolerance here in relation to the voxel size?
        if ((radius.x() == radius.y()) && (radius.x() == radius.z())) {
            const FixedBandRadius<float> fixed(radius.x(), r.halfband());
            this->BaseT::rasterizePoint(P, id, bounds, fixed);
            return;
        }

#if OPENVDB_ELLIPSOID_KERNEL_MODE == 0
        const FixedBandRadius<RealT> fbr(1.0, r.halfband());
        // If min2 == 0.0, then the index space radius is equal to or less than
        // the desired half band. In this case each sphere interior always needs
        // to be filled with distance values as we won't ever reach the negative
        // background value. If, however, a point overlaps a voxel coord exactly,
        // x2y2z2 will be 0.0. Forcing min2 to be less than zero here avoids
        // incorrectly setting these voxels to inactive -background values as
        // x2y2z2 will never be < 0.0. We still want the lteq logic in the
        // (x2y2z2 <= min2) check as this is valid when min2 > 0.0.
        const RealT min2 = fbr.minSq() == 0.0 ? -1.0 : fbr.minSq();
        const RealT max2 = fbr.maxSq();
#endif
        // @note  Extending the search by the halfband in this way will produce
        //  the desired halfband width, but will not necessarily mean that
        //  the ON values will be levelset up to positive (exterior) background
        //  value due to elliptical coordinates not being a constant distance
        //  apart
        const math::Mat3s xformis = xform.timesDiagonal(r.get());
        const Vec3d max = calcEllipsoidBoundMax(xformis) + r.halfband();
        CoordBBox intersectBox(Coord::round(P - max), Coord::round(P + max));
        intersectBox.intersect(bounds);
        if (intersectBox.empty()) return;

        auto* const data = this->template buffer<0>();
        [[maybe_unused]] auto* const cpg = CPG ? this->template buffer<CPG ? 1 : 0>() : nullptr;
        auto& mask = *(this->template mask<0>());

        // construct inverse transformation to create sphere out of an ellipsoid
        const Vec3d radInv = 1.0f / radius;

#if OPENVDB_ELLIPSOID_KERNEL_MODE != 2
        // Instead of trying to compute the distance from a point to an ellips,
        // stamp the ellipsoid by deforming the distance to the iterated voxel
        // by the inverse ellipsoid transform, then modifying it by projecting
        // it back to our normal coordinate system.
        math::Mat3s invDiag;
        invDiag.setSymmetric(radInv, Vec3f(0));
        const math::Mat3s ellipsoidInverse = invDiag * xformws.transpose();
#else
        // Instead of trying to compute the distance from a point to a rotated
        // ellipse, stamp the ellipsoid by deforming the distance to the
        // iterated voxel by the inverse xform. Then calculate the distance
        // to the axis-aligned ellipse.
        const Vec3d radInv2 = 1.0f / math::Pow2(radius);
        const math::Mat3s ellipsoidInverse = xformws.transpose();
#endif

        // We cache the multiples matrix for each axis component but essentially
        // this resolves to the invMat multiplied by the x,y,z position
        // difference (i.e. c-p):
        //   pointOnUnitSphere = ellipsoidInverse * Vec3d(x,y,z);
        const Coord& a(intersectBox.min());
        const Coord& b(intersectBox.max());
        const float* inv = ellipsoidInverse.asPointer(); // @todo do at RealT precision?
        for (Coord c = a; c.x() <= b.x(); ++c.x()) {
            const RealT x = static_cast<RealT>(c.x() - P[0]);
            const Vec3d xradial(x*inv[0], x*inv[3], x*inv[6]);
            const Index i = ((c.x() & (DIM-1u)) << 2*LOG2DIM); // unsigned bit shift mult
            for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                const RealT y = static_cast<RealT>(c.y() - P[1]);
                const Vec3d xyradial = xradial + Vec3d(y*inv[1], y*inv[4], y*inv[7]);
                const Index ij = i + ((c.y() & (DIM-1u)) << LOG2DIM);
                for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                    const Index offset = ij + /*k*/(c.z() & (DIM-1u));
                    if (!mask.isOn(offset)) continue; // inside existing level set or not in range
                    const RealT z = static_cast<RealT>(c.z() - P[2]);
                    // Transform by inverse of ellipsoidal transform to
                    // calculate distance to deformed surface
                    const Vec3d pointOnUnitSphere = (xyradial + Vec3d(z*inv[2], z*inv[5], z*inv[8]));

                    ValueT d;

#if OPENVDB_ELLIPSOID_KERNEL_MODE == 0
                    RealT len = pointOnUnitSphere.lengthSqr();
                    // Note that the transformation of the ellipse in this way
                    // also deforms the narrow band width - we may want to do
                    // something about this.
                    if (len >= max2) continue; //outside narrow band of particle in positive direction
                    if (len <= min2) { //outside narrow band of the particle in negative direction. can disable this to fill interior
                        data[offset] = -(this->mBackground);
                        mask.setOff(offset);
                        continue;
                    }
                    d = ValueT(this->mDx * (math::Sqrt(len) - 1.0)); // back to world space
#elif OPENVDB_ELLIPSOID_KERNEL_MODE == 1
                    RealT len = pointOnUnitSphere.lengthSqr();
                    // @todo  There may be a way to map the min2/max2 checks
                    //  (or at least the min2 check) onto this length to avoid
                    //  the sqrts and projections when outside the half band
                    len = math::Sqrt(len);
                    if (OPENVDB_UNLIKELY(len == 0)) {
                        // The minimum radius of this ellipse in world space. Used only to store
                        // a distance when a given voxel's ijk coordinates overlaps exactly with
                        // the center of an ellipse
                        d = -ValueT(std::min(radius.x(), std::min(radius.y(), radius.z()))) * ValueT(this->mDx);
                    }
                    else {
                        Vec3d ellipseNormal = (ellipsoidInverse.transpose() * pointOnUnitSphere);
                        ellipseNormal.normalize();
                        // Project xyz onto the ellipse normal, scale length by
                        // the offset correction based on the distance from the
                        // unit sphere surface and finally convert back to
                        // world space
                        //
                        // Invert the length to represent a proportional offset to
                        // the final distance when the above sphere point is
                        // projected back onto the ellipse. If the length iz zero,
                        // then this voxel's ijk is the center of the ellipse.
                        d = static_cast<ValueT>(
                                ((x * ellipseNormal.x()) +
                                 (y * ellipseNormal.y()) +
                                 (z * ellipseNormal.z()))       // dot product
                                    * (1.0 - (RealT(1.0)/len)) // scale
                                    * this->mDx);              // world space
                    }

                    if (d >=  this->mBackground) continue; //outside narrow band of particle in positive direction
                    if (d <= -this->mBackground) { //outside narrow band of the particle in negative direction. can disable this to fill interior
                        data[offset] = -(this->mBackground);
                        mask.setOff(offset);
                        continue;
                    }
#elif OPENVDB_ELLIPSOID_KERNEL_MODE == 2
                    const RealT k2 = (pointOnUnitSphere * radInv2).length();
                    if (OPENVDB_UNLIKELY(k2 == 0)) {
                        // The minimum radius of this ellipse in world space. Used only to store
                        // a distance when a given voxel's ijk coordinates overlaps exactly with
                        // the center of an ellipse
                        d = -ValueT(std::min(radius.x(), std::min(radius.y(), radius.z()))) * ValueT(this->mDx);
                    }
                    else {
                        const RealT k1 = (pointOnUnitSphere * radInv).length();
                        OPENVDB_ASSERT(k1 > 0);
                        // calc distance and then scale by voxelsize to convert to ws
                        d = static_cast<ValueT>((k1 * (k1 - RealT(1.0)) / k2) * this->mDx);
                    }
                    if (d >=  this->mBackground) continue; //outside narrow band of particle in positive direction
                    if (d <= -this->mBackground) { //outside narrow band of the particle in negative direction. can disable this to fill interior
                        data[offset] = -(this->mBackground);
                        mask.setOff(offset);
                        continue;
                    }
#endif
                    OPENVDB_ASSERT(std::isfinite(d));
                    ValueT& v = data[offset];
                    if (d < v) {
                        v = d;
                        if constexpr(CPG) cpg[offset] = Int64(this->mPLeafMask | Index64(id));
                    }
                }
            }
        }
    }

protected:
    const EllipseIndicies& mIndices;
private:
    std::unique_ptr<PwsHandleT> mPositionWSHandle;
};

/// @brief  Specializations of the EllipsoidTransfer for different Xform
///   attribute types.
/// @todo This is a bit silly and is only required because doRasterizeSurface
///   was designed for a specific templated transfer interface. This can easily
///   be extended; instead of taking a specific set of template parameters on
///   the TransferInterfaceT, it could append a variable <typename ...TArgsT>
///   or simply take a typename ResolverT which resolved to the final transfer
///   scheme type.
template <typename SdfT,
    typename PositionCodecT,
    typename RadiusType,
    typename FilterType,
    bool CPG>
struct EllipsoidTransferQuat final :
    public EllipsoidTransfer<SdfT, PositionCodecT, RadiusType, FilterType, CPG>
{
    using BaseT = EllipsoidTransfer<SdfT, PositionCodecT, RadiusType, FilterType, CPG>;
    using QuatHandleT = points::AttributeHandle<math::Quats>;

    EllipsoidTransferQuat(const size_t pidx,
            const Vec3i width,
            const RadiusType& rt,
            const math::Transform& source,
            const FilterType& filter,
            util::NullInterrupter* interrupt,
            SdfT& surface,
            const EllipseIndicies& indices,
            Int64Tree* cpg = nullptr,
            const std::unordered_map<const PointDataTree::LeafNodeType*, Index>* ids = nullptr)
        : BaseT(pidx, width, rt, source, filter, interrupt, surface, indices, cpg, ids)
        , mRotationHandle() {}

    EllipsoidTransferQuat(const EllipsoidTransferQuat& other)
        : BaseT(other)
        , mRotationHandle() {}

    inline bool startPointLeaf(const PointDataTree::LeafNodeType& leaf)
    {
        const bool ret = this->BaseT::startPointLeaf(leaf);
        mRotationHandle = std::make_unique<QuatHandleT>(leaf.constAttributeArray(this->BaseT::mIndices.xform));
        return ret;
    }

    inline void rasterizePoint(const Coord& ijk,
                    const Index id,
                    const CoordBBox& bounds)
    {
        Mat3s R;
        R.setToRotation(mRotationHandle->get(id));
        this->BaseT::rasterizePoint(ijk, id, bounds, R);
    }

private:
    std::unique_ptr<QuatHandleT> mRotationHandle;
};

/// @brief  Specialization for mat3 types (see above comment)
template <typename SdfT,
    typename PositionCodecT,
    typename RadiusType,
    typename FilterType,
    bool CPG>
struct EllipsoidTransferMat3 final :
    public EllipsoidTransfer<SdfT, PositionCodecT, RadiusType, FilterType, CPG>
{
    using BaseT = EllipsoidTransfer<SdfT, PositionCodecT, RadiusType, FilterType, CPG>;
    using XformHandleT = points::AttributeHandle<math::Mat3s>;

    EllipsoidTransferMat3(const size_t pidx,
            const Vec3i width,
            const RadiusType& rt,
            const math::Transform& source,
            const FilterType& filter,
            util::NullInterrupter* interrupt,
            SdfT& surface,
            const EllipseIndicies& indices,
            Int64Tree* cpg = nullptr,
            const std::unordered_map<const PointDataTree::LeafNodeType*, Index>* ids = nullptr)
        : BaseT(pidx, width, rt, source, filter, interrupt, surface, indices, cpg, ids)
        , mXformHandle() {}

    EllipsoidTransferMat3(const EllipsoidTransferMat3& other)
        : BaseT(other)
        , mXformHandle() {}

    inline bool startPointLeaf(const PointDataTree::LeafNodeType& leaf)
    {
        const bool ret = this->BaseT::startPointLeaf(leaf);
        mXformHandle.reset(new XformHandleT(leaf.constAttributeArray(this->BaseT::mIndices.xform)));
        return ret;
    }

    inline void rasterizePoint(const Coord& ijk,
                    const Index id,
                    const CoordBBox& bounds)
    {
        this->BaseT::rasterizePoint(ijk, id, bounds, mXformHandle->get(id));
    }

private:
    std::unique_ptr<XformHandleT> mXformHandle;
};

template<typename RadiusType,
     typename XformT,
     typename MaskTreeT>
struct EllipseSurfaceMaskOp final
    : public rasterize_sdf_internal::SurfaceMaskOp<MaskTreeT>
{
    using BaseT = rasterize_sdf_internal::SurfaceMaskOp<MaskTreeT>;
    using PointDataLeaf = points::PointDataTree::LeafNodeType;
    using LeafManagerT = tree::LeafManager<const points::PointDataTree>;
    using RadiusT = typename RadiusType::ValueType;
    using RotationHandleT = points::AttributeHandle<XformT>;
    static const Index DIM = points::PointDataTree::LeafNodeType::DIM;

    EllipseSurfaceMaskOp(
            const math::Transform& src,
            const math::Transform& trg,
            const RadiusType& rad,
            const Real halfband,
            const EllipseIndicies& indices)
        : BaseT(src, trg, nullptr)
        , mRadius(rad)
        , mHalfband(halfband)
        , mIndices(indices)
        , mMaxDist(0) {}

    EllipseSurfaceMaskOp(const EllipseSurfaceMaskOp& other, tbb::split)
        : BaseT(other)
        , mRadius(other.mRadius)
        , mHalfband(other.mHalfband)
        , mIndices(other.mIndices)
        , mMaxDist(0) {}

    Vec3i getMaxDist() const { return mMaxDist; }

    void join(EllipseSurfaceMaskOp& other)
    {
        mMaxDist = math::maxComponent(mMaxDist, other.mMaxDist);
        this->BaseT::join(other);
    }

    /// @brief  Fill activity by analyzing the axis aligned ellipse bounding
    ///   boxes on points in this leaf. Slightly slower than just looking at
    ///   ellipse stretches but produces a more accurate/tighter activation
    ///   result
    void fillFromStretchAndRotation(const typename LeafManagerT::LeafNodeType& leaf)
    {
        RadiusType rad(mRadius);
        rad.reset(leaf);

        Vec3d maxPos(std::numeric_limits<Real>::lowest()),
              minPos(std::numeric_limits<Real>::max());

        // Compute min/max point leaf positions
        if (mIndices.positionws == std::numeric_limits<size_t>::max())
        {
            const CoordBBox box = this->getActiveBoundingBox(leaf);
            if (box.empty()) return;
            minPos = this->mPointsTransform.indexToWorld(box.min().asVec3d() - 0.5);
            maxPos = this->mPointsTransform.indexToWorld(box.max().asVec3d() + 0.5);
        }
        else
        {
            // positions may have been smoothed, so we need to account for that too
            points::AttributeHandle<Vec3d> pwsHandle(leaf.constAttributeArray(mIndices.positionws));
            if (pwsHandle.size() == 0) return;
            for (Index i = 0; i < pwsHandle.size(); ++i)
            {
                const Vec3d Pws = pwsHandle.get(i);
                minPos = math::minComponent(minPos, Pws);
                maxPos = math::maxComponent(maxPos, Pws);
            }
        }

        // Compute max ellips bounds
        Vec3d maxBounds(0);
        RotationHandleT xformHandle(leaf.constAttributeArray(mIndices.xform));

        for (Index i = 0; i < xformHandle.size(); ++i)
        {
            math::Mat3s xform;
            if constexpr (std::is_same_v<XformT, math::Mat3s>) {
                xform = xformHandle.get(i);
            }
            else {
                // expected quaternion
                xform.setToRotation(xformHandle.get(i));
            }
            // We used to optimize for uniform radius here, but since the xform
            // is now allowed to contain scale (i.e. not necessarily unitary),
            // its simpler to just compute the AABB of the ellipse (in index
            // space) than to figure out if it contains scale and normalize it
            const math::Mat3s xformis = xform.timesDiagonal(rad.eval(i).get());
            const Vec3d bounds = calcUnitEllipsoidBoundMaxSq(xformis);
            maxBounds = math::maxComponent(maxBounds, bounds);
        }

        for (size_t i = 0; i < 3; ++i) {
            // We don't do the sqrt per point so resolve the actual maxBounds now
            maxBounds[i] = std::sqrt(maxBounds[i]);
        }

        // @note  This addition of the halfband here doesn't take into account
        //   the squash on the halfband itself. The subsequent rasterizer
        //   squashes the halfband but probably shouldn't, so although this
        //   expansion is more then necessary, I'm leaving the logic here for
        //   now. We ignore stretches as these are capped to the half band
        //   length anyway
        const Coord dist = Coord::round(maxBounds + mHalfband);
        // Convert point bounds to surface transform, expand and fill
        CoordBBox surfaceBounds(
            Coord::round(this->mSurfaceTransform.worldToIndex(minPos)),
            Coord::round(this->mSurfaceTransform.worldToIndex(maxPos)));
        surfaceBounds.min() -= dist;
        surfaceBounds.max() += dist;
        this->activate(surfaceBounds);
        /// @todo deactivate min
        this->updateMaxLookup(minPos, maxPos, dist.asVec3i(), leaf);
    }

    void operator()(const typename LeafManagerT::LeafRange& range)
    {
        for (auto leaf = range.begin(); leaf; ++leaf) {
            this->fillFromStretchAndRotation(*leaf);
        }
    }

private:
    void updateMaxLookup(const Vec3d& minWs,
                         const Vec3d& maxWs,
                         const Vec3i dist,
                         const typename LeafManagerT::LeafNodeType& leaf)
    {
        // Compute the maximum lookup required if points have moved outside of
        // this node by finding the voxel furthest away from our node and using
        // it's maximum index coordinate as the distance we need to search
        Coord minIdx = this->mPointsTransform.worldToIndexCellCentered(minWs);
        Coord maxIdx = this->mPointsTransform.worldToIndexCellCentered(maxWs);
        const auto bounds = leaf.getNodeBoundingBox();

        // If any of the ijk coords are > 0 then we need to subtract
        // the dimension of the current leaf node from the offset distance.
        // Note that min and max can both be in the negative or positive
        // direction
        if (!bounds.isInside(maxIdx)) {
            maxIdx -= leaf.origin();
            if (maxIdx.x() > 0) maxIdx.x() -= DIM;
            if (maxIdx.y() > 0) maxIdx.y() -= DIM;
            if (maxIdx.z() > 0) maxIdx.z() -= DIM;
            maxIdx = Abs(maxIdx);
        }
        else {
            maxIdx.reset(0);
        }
        if (!bounds.isInside(minIdx))
        {
            minIdx -= leaf.origin();
            if (minIdx.x() > 0) minIdx.x() -= DIM;
            if (minIdx.y() > 0) minIdx.y() -= DIM;
            if (minIdx.z() > 0) minIdx.z() -= DIM;
            minIdx = Abs(minIdx);
        }
        else {
            minIdx.reset(0);
        }

        // Now compute the max offset
        maxIdx.maxComponent(minIdx);
        mMaxDist = math::maxComponent(mMaxDist, dist + maxIdx.asVec3i());
    }

private:
    const RadiusType& mRadius;
    const Real mHalfband;
    const EllipseIndicies& mIndices;
    Vec3i mMaxDist;
};

template <typename PointDataGridT,
    typename SdfT,
    typename SettingsT,
    typename XformT>
GridPtrVec
rasterizeEllipsoids(const PointDataGridT& points,
                    const SettingsT& settings,
                    const typename SettingsT::FilterType& filter)
{
    static_assert(IsSpecializationOf<PointDataGridT, Grid>::value);
    static_assert(std::is_same_v<XformT, Quats> || std::is_same_v<XformT, Mat3s>);
    static_assert(IsSpecializationOf<SettingsT, EllipsoidSettings>::value);

    using namespace rasterize_sdf_internal;

    using PointDataTreeT = typename PointDataGridT::TreeType;
    using MaskTreeT = typename SdfT::TreeType::template ValueConverter<ValueMask>::Type;
    using AttributeTypes = typename SettingsT::AttributeTypes;
    using FilterT = typename SettingsT::FilterType;

    constexpr bool XformIsQuat = std::is_same_v<XformT, math::Quats>;

    const std::vector<std::string>& attributes = settings.attributes;
    const Real halfband = settings.halfband;
    const Vec3d radiusScale = settings.radiusScale;
    auto* interrupter = settings.interrupter;

    math::Transform::Ptr transform = settings.transform;
    if (!transform) transform = points.transform().copy();
    const Real vs = transform->voxelSize()[0];
    const typename SdfT::ValueType background =
        static_cast<typename SdfT::ValueType>(vs * halfband);

    // early exit here if no points
    const auto leaf = points.constTree().cbeginLeaf();
    if (!leaf) {
        typename SdfT::Ptr surface = SdfT::create(background);
        surface->setTransform(transform);
        surface->setGridClass(GRID_LEVEL_SET);
        return GridPtrVec(1, surface);
    }

    // deprecated, to remove, should always use xform
OPENVDB_NO_DEPRECATION_WARNING_BEGIN
    const auto& attr = settings.rotation.empty() ? settings.xform : settings.rotation;
OPENVDB_NO_DEPRECATION_WARNING_END

    // Get attributes
    const EllipseIndicies indices(leaf->attributeSet().descriptor(),
        attr,
        settings.pws, // pws is optional
        XformIsQuat);

    typename SdfT::Ptr surface;
    GridPtrVec grids;

    if (settings.radius.empty()) {
        // Initial Index Space radius
        FixedBandRadius<Vec3f> rad(Vec3f(radiusScale / vs), float(halfband));

        // pre-compute ellipsoidal transform bounds and surface mask. Points that
        // are not in the ellipses group are treated as spheres and follow
        // the same logic as that of the Fixed/VaryingSurfaceMaskOps. Ellipsoids
        // instead compute the max axis-aligned bounding boxes. The maximum extents
        // of the spheres/ellipses in a leaf is used for the maximum mask/lookup.
        // The minimum extent is always the smallest spherical radius.
        // @todo  Is the min extent correct?
        Vec3i width;
        {
            if (interrupter) interrupter->start("Generating ellipsoidal surface topology");

            tree::LeafManager<const PointDataTreeT> manager(points.tree());
            // pass radius scale as index space

            EllipseSurfaceMaskOp<FixedBandRadius<Vec3f>, XformT, MaskTreeT>
                op(points.transform(), *transform, rad, halfband, indices);
            tbb::parallel_reduce(manager.leafRange(), op);

            surface = rasterize_sdf_internal::initSdfFromMasks<SdfT, MaskTreeT>
                (transform, background, op.mask(), op.maskoff());
            // max possible index space radius
            width = op.getMaxDist();

            if (interrupter) interrupter->end();
        }

        if (interrupter) interrupter->start("Rasterizing particles to level set using ellipses and fixed spheres");

        if constexpr (std::is_same_v<XformT, math::Mat3s>) {
            grids = doRasterizeSurface<SdfT, EllipsoidTransferMat3, AttributeTypes, FilterT>
                (points, attributes, *surface,
                    width, rad, points.transform(), filter, interrupter, *surface, indices); // args
        }
        else {
            grids = doRasterizeSurface<SdfT, EllipsoidTransferQuat, AttributeTypes, FilterT>
                (points, attributes, *surface,
                    width, rad, points.transform(), filter, interrupter, *surface, indices); // args
        }
    }
    else {
        using RadiusT = typename SettingsT::RadiusAttributeType;

        const size_t ridx = leaf->attributeSet().find(settings.radius);
        if (ridx == AttributeSet::INVALID_POS) {
            OPENVDB_THROW(RuntimeError, "Failed to find radius attribute \"" + settings.radius + "\"");
        }

        // Initial varying Index Space radius
        VaryingBandRadius<RadiusT, Vec3f> rad(ridx, float(halfband), Vec3f(radiusScale / vs));

        // pre-compute ellipsoidal transform bounds and surface mask. Points that
        // are not in the ellipse group are treated as spheres and follow
        // the same logic as that of the Fixed/VaryingSurfaceMaskOps. Ellipsoids
        // instead compute the max axis-aligned bounding boxes. The maximum extents
        // of the spheres/ellipses in a leaf is used for the maximum mask/lookup.
        // The minimum extent is always the smallest spherical radius.
        // @todo  Is the min extent correct?
        Vec3i width;
        {
            if (interrupter) interrupter->start("Generating variable ellipsoidal surface topology");

            tree::LeafManager<const PointDataTreeT> manager(points.tree());

            // pass radius scale as index space
            EllipseSurfaceMaskOp<VaryingBandRadius<RadiusT, Vec3f>, XformT, MaskTreeT>
                op(points.transform(), *transform, rad, halfband, indices);
            tbb::parallel_reduce(manager.leafRange(), op);

            surface = rasterize_sdf_internal::initSdfFromMasks<SdfT, MaskTreeT>
                (transform, background, op.mask(), op.maskoff());
            // max possible index space radius
            width = op.getMaxDist();

            if (interrupter) interrupter->end();
        }

        if (interrupter) interrupter->start("Rasterizing particles to level set using variable ellipses");

        if constexpr (std::is_same_v<XformT, math::Mat3s>) {
            grids = doRasterizeSurface<SdfT, EllipsoidTransferMat3, AttributeTypes, FilterT>
                (points, attributes, *surface,
                    width, rad, points.transform(), filter, interrupter, *surface, indices); // args
        }
        else {
            grids = doRasterizeSurface<SdfT, EllipsoidTransferQuat, AttributeTypes, FilterT>
                (points, attributes, *surface,
                    width, rad, points.transform(), filter, interrupter, *surface, indices); // args
        }
    }

    if (interrupter) interrupter->end();
    tools::pruneLevelSet(surface->tree());
    grids.insert(grids.begin(), surface);
    return grids;
}

template <typename PointDataGridT,
    typename SdfT,
    typename SettingsT>
GridPtrVec
rasterizeEllipsoids(const PointDataGridT& points,
                    const SettingsT& settings,
                    const typename SettingsT::FilterType& filter)
{
    if (const auto leaf = points.constTree().cbeginLeaf()) {
        const auto& desc = leaf->attributeSet().descriptor();
        // deprecated, to remove, should always use xform
OPENVDB_NO_DEPRECATION_WARNING_BEGIN
        const auto& attr = settings.rotation.empty() ? settings.xform : settings.rotation;
OPENVDB_NO_DEPRECATION_WARNING_END
        const size_t idx = desc.find(attr);
        if (idx != points::AttributeSet::INVALID_POS) {
            if (typeNameAsString<math::Mat3s>() == desc.valueType(idx)) {
                return rasterizeEllipsoids<PointDataGridT, SdfT, SettingsT, math::Mat3s>(points, settings, filter);
            }
            if (typeNameAsString<math::Quats>() == desc.valueType(idx)) {
                return rasterizeEllipsoids<PointDataGridT, SdfT, SettingsT, math::Quats>(points, settings, filter);
            }
            throw std::runtime_error("Wrong attribute type for attribute " +
                attr + ", expected " +
                typeNameAsString<math::Mat3s>() + " or " +
                typeNameAsString<math::Quats>());
        }
    }

    // If we're here, either the point grid is empty or the xform attribute didn't exist.
    // Run a default instantiation which will elevate the correct runtime error.
    return rasterizeEllipsoids<PointDataGridT, SdfT, SettingsT, math::Mat3s>(points, settings, filter);
}

} // namespace rasterize_sdf_internal

/// @endcond

}
}
}

#endif // OPENVDB_POINTS_RASTERIZE_ELLIPSOIDS_SDF_IMPL_HAS_BEEN_INCLUDED
