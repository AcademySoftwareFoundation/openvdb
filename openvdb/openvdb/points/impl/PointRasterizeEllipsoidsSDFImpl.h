// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Richard Jones, Nick Avramoussis
///
/// @file PointRasterizeEllipsoidsSDFImpl.h
///

#ifndef OPENVDB_POINTS_RASTERIZE_ELLIPSOIDS_SDF_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_RASTERIZE_ELLIPSOIDS_SDF_IMPL_HAS_BEEN_INCLUDED

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

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
calcEllipsoidBoundMax(const math::Mat3s& ellipsoidTransform, const double radius)
{
    math::Vec3d boundMax = calcUnitEllipsoidBoundMaxSq(ellipsoidTransform);
    boundMax[0] = std::sqrt(boundMax[0]) * radius;
    boundMax[1] = std::sqrt(boundMax[1]) * radius;
    boundMax[2] = std::sqrt(boundMax[2]) * radius;
    return boundMax;
}

struct EllipsIndicies
{
    EllipsIndicies(const points::AttributeSet::Descriptor& desc,
            const PcaAttributes& attribs)
        : stretch(EllipsIndicies::getAttributeIndex<Vec3f>(desc, attribs.stretch))
        , rotation(EllipsIndicies::getAttributeIndex<Mat3s>(desc, attribs.rotation))
        , position(EllipsIndicies::getAttributeIndex<Vec3d>(desc, attribs.positionWS)) {}

    const size_t stretch, rotation, position;

private:
    template<typename ValueT>
    static inline size_t
    getAttributeIndex(const points::AttributeSet::Descriptor& desc,
                      const std::string& name)
    {
        const size_t idx = desc.find(name);
        if (idx == points::AttributeSet::INVALID_POS) {
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
    bool CPG>
struct EllipsoidTransfer :
    public rasterize_sdf_internal::SphericalTransfer<SdfT, PositionCodecT, RadiusType, CPG>
{
    using BaseT = rasterize_sdf_internal::SphericalTransfer<SdfT, PositionCodecT, RadiusType, CPG>;
    using typename BaseT::TreeT;
    using typename BaseT::ValueT;

    static const Index DIM = TreeT::LeafNodeType::DIM;
    static const Index LOG2DIM = TreeT::LeafNodeType::LOG2DIM;
    // The precision of the kernel arithmetic
    using RealT = double;

    using StretchHandleT = points::AttributeHandle<Vec3f>;
    using RotationHandleT = points::AttributeHandle<math::Mat3s>;
    using PwsHandleT = points::AttributeHandle<Vec3d>;

    EllipsoidTransfer(const size_t pidx,
            const Vec3i width,
            const RadiusType& rt,
            const math::Transform& source,
            SdfT& surface,
            const EllipsIndicies& indices,
            Int64Tree* cpg = nullptr,
            const std::unordered_map<const PointDataTree::LeafNodeType*, Index>* ids = nullptr)
        : BaseT(pidx, width, rt, source, surface, cpg, ids)
        , mIndices(indices)
        , mStretchHandle()
        , mRotationHandle()
        , mPositionWSHandle() {}

    EllipsoidTransfer(const EllipsoidTransfer& other)
        : BaseT(other)
        , mIndices(other.mIndices)
        , mStretchHandle()
        , mRotationHandle()
        , mPositionWSHandle() {}

    inline bool startPointLeaf(const PointDataTree::LeafNodeType& leaf)
    {
        const bool ret = this->BaseT::startPointLeaf(leaf);
        mStretchHandle.reset(new StretchHandleT(leaf.constAttributeArray(mIndices.stretch)));
        mRotationHandle.reset(new RotationHandleT(leaf.constAttributeArray(mIndices.rotation)));
        mPositionWSHandle.reset(new PwsHandleT(leaf.constAttributeArray(mIndices.position)));
        return ret;
    }

    inline void rasterizePoint(const Coord&,
                    const Index id,
                    const CoordBBox& bounds)
    {
        // Position may have been smoothed so we need to use the ws handle
        const Vec3d PWS = this->mPositionWSHandle->get(id);
        const Vec3d P = this->targetTransform().worldToIndex(PWS);
        const Vec3f stretch = mStretchHandle->get(id);

        // If we have a uniform stretch, treat as a sphere
        // @todo  worth using a tolerance here?
        if ((stretch.x() == stretch.y()) && (stretch.x() == stretch.z())) {
            this->BaseT::rasterizePoint(P, id, bounds,
                this->mRadius.eval(id, typename RadiusType::ValueType(stretch.x())));
            return;
        }

        const auto& r = this->mRadius.eval(id);
        const RealT radius = r.get(); // index space radius
        const math::Mat3s rotation = mRotationHandle->get(id);
        const math::Mat3s ellipsoidTransform = rotation.timesDiagonal(stretch);
        const Vec3d max = calcEllipsoidBoundMax(ellipsoidTransform, r.max());
        CoordBBox intersectBox(Coord::round(P - max), Coord::round(P + max));
        intersectBox.intersect(bounds);
        if (intersectBox.empty()) return;

        auto* const data = this->template buffer<0>();
        [[maybe_unused]] auto* const cpg = CPG ? this->template buffer<CPG ? 1 : 0>() : nullptr;
        auto& mask = *(this->template mask<0>());

        // If min2 == 0.0, then the index space radius is equal to or less than
        // the desired half band. In this case each sphere interior always needs
        // to be filled with distance values as we won't ever reach the negative
        // background value. If, however, a point overlaps a voxel coord exactly,
        // x2y2z2 will be 0.0. Forcing min2 to be less than zero here avoids
        // incorrectly setting these voxels to inactive -background values as
        // x2y2z2 will never be < 0.0. We still want the lteq logic in the
        // (x2y2z2 <= min2) check as this is valid when min2 > 0.0.
        const RealT min2 = r.minSq() == 0.0 ? -1.0 : r.minSq();
        const RealT max2 = r.maxSq();

        // construct modified covariance matrix inverse
        const Vec3f stretchInverse = (1.0f / stretch);
        // inverse transformation to create ellipsoid out of sphere
        const math::Mat3s ellipsoidInverse =
            rotation.timesDiagonal(stretchInverse) * rotation.transpose();

        // Stamp the ellipsoid by deforming the distance to the iterated voxel
        // by the inverse ellipsoid transform. We cache the multiples matrix
        // for each axis component but essentially this resolves to the invMat
        // multiplied by the x,y,z position difference (i.e. c-p):
        //   transformedRadialVector = ellipsoidInverse * Vec3d(x,y,z);
        // Note that the transformation of the ellipse also deforms the narrow
        // band width - we may want to do something about this.
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

                    // Transform by inverse of ellipsoidal transform to
                    // calculate distance to deformed surface.
                    const RealT z = static_cast<RealT>(c.z() - P[2]);
                    const Vec3d transformedRadialVector = xyradial + Vec3d(z*inv[2], z*inv[5], z*inv[8]);
                    const RealT x2y2z2 = transformedRadialVector.lengthSqr();

                    if (x2y2z2 >= max2) continue; //outside narrow band of particle in positive direction
                    if (x2y2z2 <= min2) { //outside narrow band of the particle in negative direction. can disable this to fill interior
                        data[offset] = -(this->mBackground);
                        mask.setOff(offset);
                        continue;
                    }

                    const ValueT d = ValueT(this->mDx * (math::Sqrt(x2y2z2) - radius)); // back to world space
                    ValueT& v = data[offset];
                    if (d < v) {
                        v = d;
                        if constexpr(CPG) cpg[offset] = Int64(this->mPLeafMask | Index64(id));
                    }
                }
            }
        }
    }

private:
    const EllipsIndicies& mIndices;
    std::unique_ptr<StretchHandleT> mStretchHandle;
    std::unique_ptr<RotationHandleT> mRotationHandle;
    std::unique_ptr<PwsHandleT> mPositionWSHandle;
};


template<typename RadiusType, typename MaskTreeT, typename InterrupterT>
struct EllipsSurfaceMaskOp
    : public rasterize_sdf_internal::SurfaceMaskOp<MaskTreeT, InterrupterT>
{
    using BaseT = rasterize_sdf_internal::SurfaceMaskOp<MaskTreeT, InterrupterT>;
    using PointDataLeaf = points::PointDataTree::LeafNodeType;
    using LeafManagerT = tree::LeafManager<const points::PointDataTree>;
    using RadiusT = typename RadiusType::ValueType;
    static const Index DIM = points::PointDataTree::LeafNodeType::DIM;

    EllipsSurfaceMaskOp(
            const math::Transform& src,
            const math::Transform& trg,
            const RadiusType& rad,
            const Real halfband,
            const EllipsIndicies& indices)
        : BaseT(src, trg, nullptr)
        , mRadius(rad)
        , mHalfband(halfband)
        , mIndices(indices)
        , mMaxDist(0) {}

    EllipsSurfaceMaskOp(const EllipsSurfaceMaskOp& other, tbb::split)
        : BaseT(other)
        , mRadius(other.mRadius)
        , mHalfband(other.mHalfband)
        , mIndices(other.mIndices)
        , mMaxDist(0) {}

    Vec3i getMaxDist() const { return mMaxDist; }

    void join(EllipsSurfaceMaskOp& other)
    {
        mMaxDist = math::maxComponent(mMaxDist, other.mMaxDist);
        this->BaseT::join(other);
    }

    /// @brief  Fill activity by analyzing the stretch values on points in this
    ///   leaf. Ignored ellipsoid rotations which results in faster but over
    ///   zealous activation region.
    void fillFromStretch(const typename LeafManagerT::LeafNodeType& leaf)
    {
        // Positions may have been smoothed, so we need to account for that too
        // @todo  detect this from PcaSettings and avoid checking if no smoothing
        points::AttributeHandle<Vec3d> pwsHandle(leaf.constAttributeArray(mIndices.position));
        points::AttributeHandle<Vec3f> stretchHandle(leaf.constAttributeArray(mIndices.stretch));
        if (stretchHandle.size() == 0) return;

        // The max stretch coefficient. We can't analyze each xyz component
        // individually as we don't take into account the ellips rotation, so
        // have to expand the worst case uniformly
        float maxs(0);
        Vec3d maxPos(std::numeric_limits<Real>::lowest()),
              minPos(std::numeric_limits<Real>::max());

        RadiusType rad(mRadius);
        rad.reset(leaf);

        for (Index i = 0; i < Index(stretchHandle.size()); ++i)
        {
            const Vec3d Pws = pwsHandle.get(i);
            maxPos = math::maxComponent(maxPos, Pws);
            minPos = math::minComponent(minPos, Pws);
            const auto stretch = stretchHandle.get(i);

            float maxcoef = std::max(stretch[0], std::max(stretch[1], stretch[2]));
            if constexpr(!RadiusType::Fixed) maxcoef *= rad.eval(i).get(); // index space radius
            maxs = std::max(maxs, maxcoef);
        }

        if constexpr(RadiusType::Fixed) {
            // scaling by r puts the stretch values in index space
            maxs *= float(rad.get());
        }

        // @note  This addition of the halfband here doesn't take into account
        //   the squash on the halfband itself. The subsequent rasterizer
        //   squashes the halfband but probably shouldn't, so although this
        //   expansion is more then necessary, I'm leaving the logic here for
        //   now.
        // @note Assumes max stretch coeef <= 1.0f (i.e. always squashes)
        const Coord dist(static_cast<int>(math::Round(maxs + float(mHalfband))));
        mMaxDist = math::maxComponent(mMaxDist, dist.asVec3i());

        // Convert point bounds to surface transform, expand and fill
        CoordBBox surfaceBounds(
            Coord::round(this->mSurfaceTransform.worldToIndex(minPos)),
            Coord::round(this->mSurfaceTransform.worldToIndex(maxPos)));
        surfaceBounds.min() -= dist;
        surfaceBounds.max() += dist;
        this->activate(surfaceBounds);
        /// @todo deactivate min
    }

    /// @brief  Fill activity by analyzing the axis aligned ellips bounding
    ///   boxes on points in this leaf. Slightly slower than just looking at
    ///   ellips stretches but produces a more accurate/tighter activation
    ///   result
    void fillFromStretchAndRotation(const typename LeafManagerT::LeafNodeType& leaf)
    {
        // positions may have been smoothed, so we need to account for that too :|
        points::AttributeHandle<Vec3d> pwsHandle(leaf.constAttributeArray(mIndices.position));
        points::AttributeHandle<Vec3f> stretchHandle(leaf.constAttributeArray(mIndices.stretch));
        points::AttributeHandle<math::Mat3s> rotHandle(leaf.constAttributeArray(mIndices.rotation));
        if (stretchHandle.size() == 0) return;

        RadiusT maxr = 0, maxs = 0;
        Vec3d maxBounds(0),
            maxPos(std::numeric_limits<Real>::lowest()),
            minPos(std::numeric_limits<Real>::max());

        RadiusType rad(mRadius);
        rad.reset(leaf);

        for (Index i = 0; i < stretchHandle.size(); ++i)
        {
            const Vec3d Pws = pwsHandle.get(i);
            maxPos = math::maxComponent(maxPos, Pws);
            minPos = math::minComponent(minPos, Pws);

            const Vec3f stretch = stretchHandle.get(i);
            const bool isEllips = (stretch.x() != stretch.y()) || (stretch.x() != stretch.z());

            if constexpr(RadiusType::Fixed)
            {
                if (!isEllips) {
                    maxs = std::max(maxs, RadiusT(stretch.x()));
                }
                else {
                    const math::Mat3s rotation = rotHandle.get(i);
                    const math::Mat3s ellipsoidTransform = rotation.timesDiagonal(stretch);
                    // For fixed radii, compared the squared distances - we sqrt at the end
                    const Vec3d bounds = calcUnitEllipsoidBoundMaxSq(ellipsoidTransform);
                    maxBounds = math::maxComponent(maxBounds, bounds);
                }
            }
            else
            {
                // This is doing an unnecessary multi by the scale for every
                // point as we could defer the radius scale multi till after
                // all min/max operations
                const auto r = rad.eval(i).get(); // index space radius
                maxr = std::max(maxr, r);

                if (!isEllips) {
                    // for variable radii, scale each stretch by r
                    maxs = std::max(maxs, r * RadiusT(stretch.x()));
                }
                else {
                    const math::Mat3s rotation = rotHandle.get(i);
                    const math::Mat3s ellipsoidTransform = rotation.timesDiagonal(stretch);
                    // scaling by r puts the bounds values in index space
                    const Vec3d bounds = calcEllipsoidBoundMax(ellipsoidTransform, r);
                    maxBounds = math::maxComponent(maxBounds, bounds);
                }
            }
        }

        // We don't do the sqrt per point for fixed radii - resolve the
        // actual maxBounds now
        if constexpr(RadiusType::Fixed)
        {
            maxr = rad.get(); // index space radius
            maxs *= maxr; // Also scale maxs by the radius
            // scaling by r puts the bounds values in index space
            maxBounds[0] = std::sqrt(maxBounds[0]) * double(maxr);
            maxBounds[1] = std::sqrt(maxBounds[1]) * double(maxr);
            maxBounds[2] = std::sqrt(maxBounds[2]) * double(maxr);
        }

        // Account for uniform stretch values - compare the ellips to isolated
        // points and choose the largest radius of the two
        maxBounds[0] = std::max(double(maxs), maxBounds[0]);
        maxBounds[1] = std::max(double(maxs), maxBounds[1]);
        maxBounds[2] = std::max(double(maxs), maxBounds[2]);

        // @note  This addition of the halfband here doesn't take into account
        //   the squash on the halfband itself. The subsequent rasterizer
        //   squashes the halfband but probably shouldn't, so although this
        //   expansion is more then necessary, I'm leaving the logic here for
        //   now.
        // @note Assumes max stretch coeef <= 1.0f (i.e. always squashes)
        const Coord dist = Coord::round(maxBounds + mHalfband);
        mMaxDist = math::maxComponent(mMaxDist, dist.asVec3i());

        // Convert point bounds to surface transform, expand and fill
        CoordBBox surfaceBounds(
            Coord::round(this->mSurfaceTransform.worldToIndex(minPos)),
            Coord::round(this->mSurfaceTransform.worldToIndex(maxPos)));
        surfaceBounds.min() -= dist;
        surfaceBounds.max() += dist;
        this->activate(surfaceBounds);
        /// @todo deactivate min
    }

    void operator()(const typename LeafManagerT::LeafRange& range)
    {
        for (auto leaf = range.begin(); leaf; ++leaf) {
            this->fillFromStretchAndRotation(*leaf);
        }
    }

private:
    const RadiusType& mRadius;
    const Real mHalfband;
    const EllipsIndicies& mIndices;
    Vec3i mMaxDist;
};

template <typename PointDataGridT,
    typename SdfT,
    typename SettingsT>
GridPtrVec
rasterizeEllipsoids(const PointDataGridT& points,
                    const SettingsT& settings,
                    const typename SettingsT::FilterType& filter)
{
    static_assert(IsSpecializationOf<PointDataGridT, Grid>::value);
    static_assert(IsSpecializationOf<SettingsT, EllipsoidSettings>::value);

    using namespace rasterize_sdf_internal;

    using PointDataTreeT = typename PointDataGridT::TreeType;
    using MaskTreeT = typename SdfT::TreeType::template ValueConverter<ValueMask>::Type;

    using AttributeTypes = typename SettingsT::AttributeTypes;
    using InterrupterType = typename SettingsT::InterrupterType;

    const std::vector<std::string>& attributes = settings.attributes;
    const Real halfband = settings.halfband;
    const Real radiusScale = settings.radiusScale;
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

    // Get attributes
    const EllipsIndicies indices(leaf->attributeSet().descriptor(), settings.pca);

    typename SdfT::Ptr surface;
    GridPtrVec grids;

    if (settings.radius.empty())
    {
        // Initial varying Index Space radius. Note that the scale here does NOT
        // take into account the sphere scale. This is applied per point depending
        // on the ellipses group during the masking and raster ops.
        FixedBandRadius<Real> rad(Real(radiusScale / vs), Real(halfband));

        // pre-compute ellipsoidal transform bounds and surface mask. Points that
        // are not in the ellipse ellipses group are treated as spheres and follow
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

            EllipsSurfaceMaskOp<FixedBandRadius<Real>, MaskTreeT, InterrupterType>
                op(points.transform(), *transform, rad, halfband, indices);
            tbb::parallel_reduce(manager.leafRange(), op);

            surface = rasterize_sdf_internal::initSdfFromMasks<SdfT, MaskTreeT>
                (transform, background, op.mask(), op.maskoff());
            // max possible index space radius
            width = op.getMaxDist();

            if (interrupter) interrupter->end();
        }

        if (interrupter) interrupter->start("Rasterizing particles to level set using ellipses and fixed spheres");

        grids = doRasterizeSurface<SdfT, EllipsoidTransfer, AttributeTypes, InterrupterType>
            (points, attributes, filter, *surface, interrupter,
                width, rad, points.transform(), *surface, indices); // args
    }
    else
    {
        using RadiusT = typename SettingsT::RadiusAttributeType;

        const size_t ridx = leaf->attributeSet().find(settings.radius);
        if (ridx == AttributeSet::INVALID_POS) {
            OPENVDB_THROW(RuntimeError, "Failed to find radius attribute \"" + settings.radius + "\"");
        }

        // Initial varying Index Space radius. Note that the scale here does NOT
        // take into account the sphere scale. This is applied per point depending
        // on the ellipses group during the masking and raster ops.
        VaryingBandRadius<RadiusT> rad(ridx, RadiusT(halfband), RadiusT(radiusScale / vs));

        // pre-compute ellipsoidal transform bounds and surface mask. Points that
        // are not in the ellipse ellipses group are treated as spheres and follow
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
            EllipsSurfaceMaskOp<VaryingBandRadius<RadiusT>, MaskTreeT, InterrupterType>
                op(points.transform(), *transform, rad, halfband, indices);
            tbb::parallel_reduce(manager.leafRange(), op);

            surface = rasterize_sdf_internal::initSdfFromMasks<SdfT, MaskTreeT>
                (transform, background, op.mask(), op.maskoff());
            // max possible index space radius
            width = op.getMaxDist();

            if (interrupter) interrupter->end();
        }

        if (interrupter) interrupter->start("Rasterizing particles to level set using variable ellipses");

        grids = doRasterizeSurface<SdfT, EllipsoidTransfer, AttributeTypes, InterrupterType>
            (points, attributes, filter, *surface, interrupter,
                    width, rad, points.transform(), *surface, indices); // args
    }

    if (interrupter) interrupter->end();
    tools::pruneLevelSet(surface->tree());
    grids.insert(grids.begin(), surface);
    return grids;
}

} // namespace rasterize_sdf_internal

}
}
}

#endif // OPENVDB_POINTS_RASTERIZE_ELLIPSOIDS_SDF_IMPL_HAS_BEEN_INCLUDED
