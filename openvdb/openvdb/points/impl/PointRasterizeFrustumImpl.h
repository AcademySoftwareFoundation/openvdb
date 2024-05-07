// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Dan Bailey
///
/// @file PointRasterizeFrustumImpl.h
///

#ifndef OPENVDB_POINTS_POINT_RASTERIZE_FRUSTUM_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_RASTERIZE_FRUSTUM_IMPL_HAS_BEEN_INCLUDED

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @cond OPENVDB_DOCS_INTERNAL

namespace point_rasterize_internal {


// By default this trait simply no-op copies the filter
template <typename FilterT>
struct RasterGroupTraits
{
    using NewFilterT = FilterT;
    static NewFilterT resolve(const FilterT& filter, const points::AttributeSet&)
    {
        return filter;
    }
};

// For RasterGroups objects, this returns a new MultiGroupFilter with names-to-indices resolved
template <>
struct RasterGroupTraits<RasterGroups>
{
    using NewFilterT = points::MultiGroupFilter;
    static NewFilterT resolve(const RasterGroups& groups, const points::AttributeSet& attributeSet)
    {
        return NewFilterT(groups.includeNames, groups.excludeNames, attributeSet);
    }
};


// Traits to indicate bool and ValueMask are bool types
template <typename T> struct BoolTraits { static const bool IsBool = false; };
template <> struct BoolTraits<bool> { static const bool IsBool = true; };
template <> struct BoolTraits<ValueMask> { static const bool IsBool = true; };


struct TrueOp {
    bool mOn;
    explicit TrueOp(double scale) : mOn(scale > 0.0) { }
    template<typename ValueType>
    void operator()(ValueType& v) const { v = static_cast<ValueType>(mOn); }
};


template <typename ValueT>
typename std::enable_if<std::is_integral<typename ValueTraits<ValueT>::ElementType>::value, ValueT>::type
castValue(const double value)
{
    return ValueT(math::Ceil(value));
}


template <typename ValueT>
typename std::enable_if<!std::is_integral<typename ValueTraits<ValueT>::ElementType>::value, ValueT>::type
castValue(const double value)
{
    return static_cast<ValueT>(value);
}


template <typename ValueT>
typename std::enable_if<!ValueTraits<ValueT>::IsVec, bool>::type
greaterThan(const ValueT& value, const float threshold)
{
    return value >= static_cast<ValueT>(threshold);
}


template <typename ValueT>
typename std::enable_if<ValueTraits<ValueT>::IsVec, bool>::type
greaterThan(const ValueT& value, const float threshold)
{
    return static_cast<double>(value.lengthSqr()) >= threshold*threshold;
}


template <typename AttributeT, typename HandleT, typename StridedHandleT>
typename std::enable_if<!ValueTraits<AttributeT>::IsVec, AttributeT>::type
getAttributeScale(HandleT& handlePtr, StridedHandleT&, Index index)
{
    if (handlePtr) {
        return handlePtr->get(index);
    }
    return AttributeT(1);
}


template <typename AttributeT, typename HandleT, typename StridedHandleT>
typename std::enable_if<ValueTraits<AttributeT>::IsVec, AttributeT>::type
getAttributeScale(HandleT& handlePtr, StridedHandleT& stridedHandlePtr, Index index)
{
    if (handlePtr) {
        return handlePtr->get(index);
    } else if (stridedHandlePtr) {
        return AttributeT(
            stridedHandlePtr->get(index, 0),
            stridedHandlePtr->get(index, 1),
            stridedHandlePtr->get(index, 2));
    }
    return AttributeT(1);
}


template <typename ValueT>
struct MultiplyOp
{
    template <typename AttributeT>
    static ValueT mul(const ValueT& a, const AttributeT& b)
    {
        return a * b;
    }
};

template <>
struct MultiplyOp<bool>
{
    template <typename AttributeT>
    static bool mul(const bool& a, const AttributeT& b)
    {
        return a && (b != zeroVal<AttributeT>());
    }
};

template <typename PointDataGridT, typename AttributeT, typename GridT,
    typename FilterT>
struct RasterizeOp
{
    using TreeT = typename GridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;
    using PointLeafT = typename PointDataGridT::TreeType::LeafNodeType;
    using PointIndexT = typename PointLeafT::ValueType;
    using CombinableT = tbb::combinable<GridT>;
    using SumOpT = tools::valxform::SumOp<ValueT>;
    using MaxOpT = tools::valxform::MaxOp<ValueT>;
    using PositionHandleT = openvdb::points::AttributeHandle<Vec3f>;
    using VelocityHandleT = openvdb::points::AttributeHandle<Vec3f>;
    using RadiusHandleT = openvdb::points::AttributeHandle<float>;

    // to prevent checking the interrupter too frequently, only check every 32 voxels cubed
    static const int interruptThreshold = 32*32*32;

    RasterizeOp(
                const PointDataGridT& grid,
                const std::vector<Index64>& offsets,
                const size_t attributeIndex,
                const Name& velocityAttribute,
                const Name& radiusAttribute,
                CombinableT& combinable,
                CombinableT* weightCombinable,
                const bool dropBuffers,
                const float scale,
                const FilterT& filter,
                const bool computeMax,
                const bool alignedTransform,
                const bool staticCamera,
                const FrustumRasterizerSettings& settings,
                const FrustumRasterizerMask& mask,
                util::NullInterrupter* interrupt)
        : mGrid(grid)
        , mOffsets(offsets)
        , mAttributeIndex(attributeIndex)
        , mVelocityAttribute(velocityAttribute)
        , mRadiusAttribute(radiusAttribute)
        , mCombinable(combinable)
        , mWeightCombinable(weightCombinable)
        , mDropBuffers(dropBuffers)
        , mScale(scale)
        , mFilter(filter)
        , mComputeMax(computeMax)
        , mAlignedTransform(alignedTransform)
        , mStaticCamera(staticCamera)
        , mSettings(settings)
        , mMask(mask)
        , mInterrupter(interrupt) { }

    template <typename PointOpT>
    static void rasterPoint(const Coord& ijk, const double scale,
        const AttributeT& attributeScale, PointOpT& op)
    {
        op(ijk, scale, attributeScale);
    }

    template <typename PointOpT>
    static void rasterPoint(const Vec3d& position, const double scale,
        const AttributeT& attributeScale, PointOpT& op)
    {
        Coord ijk = Coord::round(position);
        op(ijk, scale, attributeScale);
    }

    template <typename SphereOpT>
    static void rasterVoxelSphere(const Vec3d& position, const double scale,
        const AttributeT& attributeScale, const float radius, util::NullInterrupter* interrupter, SphereOpT& op)
    {
        OPENVDB_ASSERT(radius > 0.0f);
        Coord ijk = Coord::round(position);
        int &i = ijk[0], &j = ijk[1], &k = ijk[2];
        const int imin=math::Floor(position[0]-radius), imax=math::Ceil(position[0]+radius);
        const int jmin=math::Floor(position[1]-radius), jmax=math::Ceil(position[1]+radius);
        const int kmin=math::Floor(position[2]-radius), kmax=math::Ceil(position[2]+radius);

        const bool interrupt = interrupter && (imax-imin)*(jmax-jmin)*(kmax-kmin) > interruptThreshold;

        for (i = imin; i <= imax; ++i) {
            if (interrupt && interrupter->wasInterrupted()) break;
            const auto x2 = math::Pow2(i - position[0]);
            for (j = jmin; j <= jmax; ++j) {
                if (interrupt && interrupter->wasInterrupted()) break;
                const auto x2y2 = math::Pow2(j - position[1]) + x2;
                for (k = kmin; k <= kmax; ++k) {
                    const auto x2y2z2 = x2y2 + math::Pow2(k - position[2]);
                    op(ijk, scale, attributeScale, x2y2z2, radius*radius);
                }
            }
        }
    }

    template <typename SphereOpT>
    static void rasterApproximateFrustumSphere(const Vec3d& position, const double scale,
        const AttributeT& attributeScale, const float radiusWS,
        const math::Transform& frustum, const CoordBBox* clipBBox,
        util::NullInterrupter* interrupter, SphereOpT& op)
    {
        Vec3d voxelSize = frustum.voxelSize(position);
        Vec3d radius = Vec3d(radiusWS)/voxelSize;
        CoordBBox frustumBBox(Coord::floor(position-radius), Coord::ceil(position+radius));

        // if clipping to frustum is enabled, clip the index bounding box
        if (clipBBox) {
            frustumBBox.intersect(*clipBBox);
        }

        Vec3i outMin = frustumBBox.min().asVec3i();
        Vec3i outMax = frustumBBox.max().asVec3i();

        const bool interrupt = interrupter && frustumBBox.volume() > interruptThreshold;

        // back-project each output voxel into the input tree.
        Vec3R xyz;
        Coord outXYZ;
        int &x = outXYZ.x(), &y = outXYZ.y(), &z = outXYZ.z();
        for (x = outMin.x(); x <= outMax.x(); ++x) {
            if (interrupt && interrupter->wasInterrupted()) break;
            xyz.x() = x;
            for (y = outMin.y(); y <= outMax.y(); ++y) {
                if (interrupt && interrupter->wasInterrupted()) break;
                xyz.y() = y;
                for (z = outMin.z(); z <= outMax.z(); ++z) {
                    xyz.z() = z;

                    // approximate conversion to world-space

                    Vec3d offset = (xyz - position) * voxelSize;

                    double distanceSqr = offset.dot(offset);

                    op(outXYZ, scale, attributeScale, distanceSqr, radiusWS*radiusWS);
                }
            }
        }
    }

    template <typename SphereOpT>
    static void rasterFrustumSphere(const Vec3d& position, const double scale,
        const AttributeT& attributeScale, const float radiusWS,
        const math::Transform& frustum, const CoordBBox* clipBBox,
        util::NullInterrupter* interrupter, SphereOpT& op)
    {
        const Vec3d positionWS = frustum.indexToWorld(position);

        BBoxd inputBBoxWS(positionWS-radiusWS, positionWS+radiusWS);
        // Transform the corners of the input tree's bounding box
        // and compute the enclosing bounding box in the output tree.
        Vec3R frustumMin;
        Vec3R frustumMax;
        math::calculateBounds(frustum, inputBBoxWS.min(), inputBBoxWS.max(),
            frustumMin, frustumMax);
        CoordBBox frustumBBox(Coord::floor(frustumMin), Coord::ceil(frustumMax));

        // if clipping to frustum is enabled, clip the index bounding box
        if (clipBBox) {
            frustumBBox.intersect(*clipBBox);
        }

        Vec3i outMin = frustumBBox.min().asVec3i();
        Vec3i outMax = frustumBBox.max().asVec3i();

        const bool interrupt = interrupter && frustumBBox.volume() > interruptThreshold;

        // back-project each output voxel into the input tree.
        Vec3R xyz;
        Coord outXYZ;
        int &x = outXYZ.x(), &y = outXYZ.y(), &z = outXYZ.z();
        for (x = outMin.x(); x <= outMax.x(); ++x) {
            if (interrupt && interrupter->wasInterrupted()) break;
            xyz.x() = x;
            for (y = outMin.y(); y <= outMax.y(); ++y) {
                if (interrupt && interrupter->wasInterrupted()) break;
                xyz.y() = y;
                for (z = outMin.z(); z <= outMax.z(); ++z) {
                    xyz.z() = z;

                    Vec3R xyzWS = frustum.indexToWorld(xyz);

                    double xDist = xyzWS.x() - positionWS.x();
                    double yDist = xyzWS.y() - positionWS.y();
                    double zDist = xyzWS.z() - positionWS.z();

                    double distanceSqr = xDist*xDist+yDist*yDist+zDist*zDist;
                    op(outXYZ, scale, attributeScale, distanceSqr, radiusWS*radiusWS);
                }
            }
        }
    }

    void operator()(const PointLeafT& leaf, size_t) const
    {
        if (mInterrupter && mInterrupter->wasInterrupted()) {
            thread::cancelGroupExecution();
            return;
        }

        using AccessorT = tree::ValueAccessor<typename GridT::TreeType>;
        using MaskAccessorT = tree::ValueAccessor<const MaskTree>;

        constexpr bool isBool = BoolTraits<ValueT>::IsBool;
        const bool isFrustum = !mSettings.transform->isLinear();

        const bool useRaytrace = mSettings.velocityMotionBlur || !mStaticCamera;
        const bool useRadius = mSettings.useRadius;
        const float radiusScale = mSettings.radiusScale;
        const float radiusThreshold = std::sqrt(3.0f);

        const float threshold = mSettings.threshold;
        const bool scaleByVoxelVolume = !useRadius && mSettings.scaleByVoxelVolume;

        const float shutterStartDt = mSettings.camera.shutterStart()/mSettings.framesPerSecond;
        const float shutterEndDt = mSettings.camera.shutterEnd()/mSettings.framesPerSecond;
        const int motionSteps = std::max(1, mSettings.motionSamples-1);

        std::vector<Vec3d> motionPositions(motionSteps+1, Vec3d());
        std::vector<Vec2s> frustumRadiusSizes(motionSteps+1, Vec2s());

        const auto& pointsTransform = mGrid.constTransform();

        const float voxelSize = static_cast<float>(mSettings.transform->voxelSize()[0]);

        auto& grid = mCombinable.local();
        auto& tree = grid.tree();
        const auto& transform = *(mSettings.transform);

        grid.setTransform(mSettings.transform->copy());

        AccessorT valueAccessor(tree);

        std::unique_ptr<MaskAccessorT> maskAccessor;
        auto maskTree = mMask.getTreePtr();
        if (maskTree)  maskAccessor.reset(new MaskAccessorT(*maskTree));

        const CoordBBox* clipBBox = !mMask.clipBBox().empty() ? &mMask.clipBBox() : nullptr;

        std::unique_ptr<AccessorT> weightAccessor;
        if (mWeightCombinable) {
            auto& weightGrid = mWeightCombinable->local();
            auto& weightTree = weightGrid.tree();
            weightAccessor.reset(new AccessorT(weightTree));
        }

        // create a temporary tree when rasterizing with radius and accumulate
        // or weighted average methods

        std::unique_ptr<TreeT> tempTree;
        std::unique_ptr<TreeT> tempWeightTree;
        std::unique_ptr<AccessorT> tempValueAccessor;
        std::unique_ptr<AccessorT> tempWeightAccessor;
        if (useRadius && !mComputeMax) {
            tempTree.reset(new TreeT(tree.background()));
            tempValueAccessor.reset(new AccessorT(*tempTree));
            if (weightAccessor) {
                tempWeightTree.reset(new TreeT(weightAccessor->tree().background()));
                tempWeightAccessor.reset(new AccessorT(*tempWeightTree));
            }
        }

        // point rasterization

        // impl - modify a single voxel by coord, handles temporary trees and all supported modes
        auto doModifyVoxelOp = [&](const Coord& ijk, const double scale, const AttributeT& attributeScale,
            const bool isTemp, const bool forceSum)
        {
            if (mMask && !mMask.valid(ijk, maskAccessor.get()))   return;
            if (isBool) {
                // only modify the voxel if the attributeScale is positive and non-zero
                if (!math::isZero<AttributeT>(attributeScale) && !math::isNegative<AttributeT>(attributeScale)) {
                    valueAccessor.modifyValue(ijk, TrueOp(scale));
                }
            } else {
                ValueT weightValue = castValue<ValueT>(scale);
                ValueT newValue = MultiplyOp<ValueT>::mul(weightValue, attributeScale);
                if (scaleByVoxelVolume) {
                    newValue /= static_cast<ValueT>(transform.voxelSize(ijk.asVec3d()).product());
                }
                if (point_rasterize_internal::greaterThan(newValue, threshold)) {
                    if (isTemp) {
                        tempValueAccessor->modifyValue(ijk, MaxOpT(newValue));
                        if (tempWeightAccessor) {
                            tempWeightAccessor->modifyValue(ijk, MaxOpT(weightValue));
                        }
                    } else {
                        if (mComputeMax && !forceSum) {
                            valueAccessor.modifyValue(ijk, MaxOpT(newValue));
                        } else {
                            valueAccessor.modifyValue(ijk, SumOpT(newValue));
                        }
                        if (weightAccessor) {
                            weightAccessor->modifyValue(ijk, SumOpT(weightValue));
                        }
                    }
                }
            }
        };

        // modify a single voxel by coord, disable temporary trees
        auto modifyVoxelOp = [&](const Coord& ijk, const double scale, const AttributeT& attributeScale)
        {
            doModifyVoxelOp(ijk, scale, attributeScale, /*isTemp=*/false, /*forceSum=*/false);
        };

        // sum a single voxel by coord, no temporary trees or maximum mode
        auto sumVoxelOp = [&](const Coord& ijk, const double scale, const AttributeT& attributeScale)
        {
            doModifyVoxelOp(ijk, scale, attributeScale, /*isTemp=*/false, /*forceSum=*/true);
        };

        // sphere rasterization

        // impl - modify a single voxel by coord based on distance from sphere origin
        auto doModifyVoxelByDistanceOp = [&](const Coord& ijk, const double scale, const AttributeT& attributeScale,
            const double distanceSqr, const double radiusSqr, const bool isTemp)
        {
            if (distanceSqr >= radiusSqr)   return;
            if (isBool) {
                valueAccessor.modifyValue(ijk, TrueOp(scale));
            } else {
                double distance = std::sqrt(distanceSqr);
                double radius = std::sqrt(radiusSqr);
                double result = 1.0 - distance/radius;
                doModifyVoxelOp(ijk, result * scale, attributeScale, isTemp, /*forceSum=*/false);
            }
        };

        // modify a single voxel by coord based on distance from sphere origin, disable temporary trees
        auto modifyVoxelByDistanceOp = [&](const Coord& ijk, const double scale, const AttributeT& attributeScale,
            const double distanceSqr, const double radiusSqr)
        {
            doModifyVoxelByDistanceOp(ijk, scale, attributeScale, distanceSqr, radiusSqr, /*isTemp=*/false);
        };

        // modify a single voxel by coord based on distance from sphere origin, enable temporary trees
        auto modifyTempVoxelByDistanceOp = [&](const Coord& ijk, const double scale, const AttributeT& attributeScale,
            const double distanceSqr, const double radiusSqr)
        {
            doModifyVoxelByDistanceOp(ijk, scale, attributeScale, distanceSqr, radiusSqr, /*isTemp=*/true);
        };

        typename points::AttributeHandle<AttributeT>::Ptr attributeHandle;
        using ElementT = typename ValueTraits<AttributeT>::ElementType;
        typename points::AttributeHandle<ElementT>::Ptr stridedAttributeHandle;

        if (mAttributeIndex != points::AttributeSet::INVALID_POS) {
            const auto& attributeArray = leaf.constAttributeArray(mAttributeIndex);
            if (attributeArray.stride() == 3) {
                stridedAttributeHandle = points::AttributeHandle<ElementT>::create(attributeArray);
            } else {
                attributeHandle = points::AttributeHandle<AttributeT>::create(attributeArray);
            }
        }

        size_t positionIndex = leaf.attributeSet().find("P");
        size_t velocityIndex = leaf.attributeSet().find(mVelocityAttribute);
        size_t radiusIndex = leaf.attributeSet().find(mRadiusAttribute);

        auto positionHandle = PositionHandleT::create(
            leaf.constAttributeArray(positionIndex));
        auto velocityHandle = (useRaytrace && leaf.hasAttribute(velocityIndex)) ?
            VelocityHandleT::create(leaf.constAttributeArray(velocityIndex)) :
            VelocityHandleT::Ptr();
        auto radiusHandle = (useRadius && leaf.hasAttribute(radiusIndex)) ?
            RadiusHandleT::create(leaf.constAttributeArray(radiusIndex)) :
            RadiusHandleT::Ptr();

        for (auto iter = leaf.beginIndexOn(mFilter); iter; ++iter) {

            // read attribute value if it exists, attributes with stride 3 are composited
            // into a vector grid such as float[3] => vec3s

            const AttributeT attributeScale = getAttributeScale<AttributeT>(
                attributeHandle, stridedAttributeHandle, *iter);

            float radiusWS = 1.0f, radius = 1.0f;
            if (useRadius) {
                radiusWS *= radiusScale;
                if (radiusHandle)       radiusWS *= radiusHandle->get(*iter);
                if (isFrustum) {
                    radius = radiusWS;
                } else if (voxelSize > 0.0f) {
                    radius = radiusWS / voxelSize;
                }
            }

            // frustum thresholding is done later to factor in changing voxel sizes

            bool doRadius = useRadius;
            if (!isFrustum && radius < radiusThreshold) {
                doRadius = false;
            }

            float increment = shutterEndDt - shutterStartDt;

            // disable ray-tracing if velocity is very small

            openvdb::Vec3f velocity(0.0f);
            bool doRaytrace = useRaytrace;
            if (doRaytrace) {
                if (increment < openvdb::math::Delta<float>::value()) {
                    doRaytrace = false;
                } else {
                    if (velocityHandle)     velocity = velocityHandle->get(*iter);
                    if (mStaticCamera && velocity.lengthSqr() < openvdb::math::Delta<float>::value()) {
                        doRaytrace = false;
                    }
                }
            }

            if (motionSteps > 1)    increment /= float(motionSteps);

            Vec3d position = positionHandle->get(*iter) + iter.getCoord().asVec3d();

            if (doRaytrace) {
                // raytracing is done in index space
                position = pointsTransform.indexToWorld(position);

                for (int motionStep = 0; motionStep <= motionSteps; motionStep++) {

                    float offset = motionStep == motionSteps ? shutterEndDt :
                        (shutterStartDt + increment * static_cast<float>(motionStep));
                    Vec3d samplePosition = position + velocity * offset;

                    const math::Transform* sampleTransform = &transform;
                    if (!mSettings.camera.isStatic()) {
                        sampleTransform = &mSettings.camera.transform(motionStep);
                        if (mSettings.camera.hasWeight(motionStep)) {
                            const float weight = mSettings.camera.weight(motionStep);
                            const Vec3d referencePosition = transform.worldToIndex(samplePosition);
                            const Vec3d adjustedPosition = sampleTransform->worldToIndex(samplePosition);
                            motionPositions[motionStep] = referencePosition + (adjustedPosition - referencePosition) * weight;
                        } else {
                            motionPositions[motionStep] = sampleTransform->worldToIndex(samplePosition);
                        }
                    } else {
                        motionPositions[motionStep] = sampleTransform->worldToIndex(samplePosition);
                    }
                    if (doRadius && isFrustum) {
                        Vec3d left = sampleTransform->worldToIndex(samplePosition - Vec3d(radiusWS, 0, 0));
                        Vec3d right = sampleTransform->worldToIndex(samplePosition + Vec3d(radiusWS, 0, 0));
                        float width = static_cast<float>((right - left).length());
                        Vec3d top = sampleTransform->worldToIndex(samplePosition + Vec3d(0, radiusWS, 0));
                        Vec3d bottom = sampleTransform->worldToIndex(samplePosition - Vec3d(0, radiusWS, 0));
                        float height = static_cast<float>((top - bottom).length());
                        frustumRadiusSizes[motionStep].x() = width;
                        frustumRadiusSizes[motionStep].y() = height;
                    }
                }

                double totalDistance = 0.0;
                for (size_t i = 0; i < motionPositions.size()-1; i++) {
                    Vec3d direction = motionPositions[i+1] - motionPositions[i];
                    double distance = direction.length();
                    totalDistance += distance;
                }

                double distanceWeight = totalDistance > 0.0 ? 1.0 / totalDistance : 1.0;

                if (doRadius && !mComputeMax) {
                    // mark all voxels inactive
                    for (auto leaf = tempTree->beginLeaf(); leaf; ++leaf) {
                        leaf->setValuesOff();
                    }
                    if (tempWeightAccessor) {
                        for (auto leaf = tempWeightTree->beginLeaf(); leaf; ++leaf) {
                            leaf->setValuesOff();
                        }
                    }
                }

                for (int motionStep = 0; motionStep < motionSteps; motionStep++) {

                    Vec3d startPosition = motionPositions[motionStep];
                    Vec3d endPosition = motionPositions[motionStep+1];

                    Vec3d direction(endPosition - startPosition);
                    double distance = direction.length();

                    // if rasterizing into a frustum grid, compute the index-space radii for start
                    // and end positions and if below the radius threshold disable using radius

                    float maxRadius = radius;

                    if (doRadius && isFrustum) {
                        const Vec2s& startRadius = frustumRadiusSizes[motionStep];
                        const Vec2s& endRadius = frustumRadiusSizes[motionStep+1];

                        if (startRadius[0] < radiusThreshold && startRadius[1] < radiusThreshold &&
                            endRadius[0] < radiusThreshold && endRadius[1] < radiusThreshold) {
                            doRadius = false;
                        } else {
                            // max radius is the largest index-space radius factoring in
                            // that in frustum space a sphere is not rasterized as spherical
                            maxRadius = std::max(startRadius[0], startRadius[1]);
                            maxRadius = std::max(maxRadius, endRadius[0]);
                            maxRadius = std::max(maxRadius, endRadius[1]);
                        }
                    }

                    if (doRadius) {
                        distanceWeight = std::min(distanceWeight, 1.0);

                        // these arbitrary constants are how tightly spheres are packed together
                        // irrespective of how large they are in index-space - if it is too low, the shape of
                        // the individual spheres becomes visible in the rasterized path,
                        // if it is too high, rasterization becomes less efficient with
                        // diminishing returns towards the accuracy of the rasterized path
                        double spherePacking = mSettings.accurateSphereMotionBlur ? 4.0 : 2.0;

                        const int steps = std::max(2, math::Ceil(distance * spherePacking / double(maxRadius)) + 1);

                        Vec3d sample(startPosition);
                        const Vec3d offset(direction / (steps-1));

                        for (int step = 0; step < steps; step++) {
                            if (isFrustum) {
                                if (mComputeMax) {
                                    if (mSettings.accurateFrustumRadius) {
                                        this->rasterFrustumSphere(sample, mScale * distanceWeight,
                                            attributeScale, radiusWS, transform, clipBBox, mInterrupter, modifyVoxelByDistanceOp);
                                    } else {
                                        this->rasterApproximateFrustumSphere(sample, mScale * distanceWeight,
                                            attributeScale, radiusWS, transform, clipBBox, mInterrupter, modifyVoxelByDistanceOp);
                                    }
                                } else {
                                    if (mSettings.accurateFrustumRadius) {
                                        this->rasterFrustumSphere(sample, mScale * distanceWeight,
                                            attributeScale, radiusWS, transform, clipBBox, mInterrupter, modifyTempVoxelByDistanceOp);
                                    } else {
                                        this->rasterApproximateFrustumSphere(sample, mScale * distanceWeight,
                                            attributeScale, radiusWS, transform, clipBBox, mInterrupter, modifyTempVoxelByDistanceOp);
                                    }
                                }
                            } else {
                                if (mComputeMax) {
                                    this->rasterVoxelSphere(sample, mScale * distanceWeight,
                                        attributeScale, radius, mInterrupter, modifyVoxelByDistanceOp);
                                } else {
                                    this->rasterVoxelSphere(sample, mScale * distanceWeight,
                                        attributeScale, radius, mInterrupter, modifyTempVoxelByDistanceOp);
                                }
                            }
                            if (step < (steps-1))   sample += offset;
                            else                    sample = endPosition;
                        }
                    } else {
                        // compute direction and store vector length as max time
                        mDdaRay.setMinTime(math::Delta<double>::value());
                        mDdaRay.setMaxTime(std::max(distance, math::Delta<double>::value()*2.0));

                         // DDA requires normalized directions
                        direction.normalize();
                        mDdaRay.setDir(direction);

                        // dda assumes node-centered ray-tracing, so compensate by adding half a voxel first
                        mDdaRay.setEye(startPosition + Vec3d(0.5));

                        // clip the ray to the frustum bounding box
                        if (clipBBox) {
                            mDdaRay.clip(*clipBBox);
                        }

                        // first rasterization in a subsequent DDA traversal should always sum contributions
                        // in order to smoothly stitch the beginning and end of two rays together
                        bool forceSum = motionStep > 0;

                        mDda.init(mDdaRay);
                        while (true) {
                            const Coord& voxel = mDda.voxel();
                            double delta = (mDda.next() - mDda.time()) * distanceWeight;
                            if (forceSum) {
                                this->rasterPoint(voxel, mScale * delta,
                                    attributeScale, sumVoxelOp);
                                forceSum = false;
                            } else {
                                this->rasterPoint(voxel, mScale * delta,
                                    attributeScale, modifyVoxelOp);
                            }
                            if (!mDda.step())    break;
                        }
                    }
                }

                if (doRadius && !mComputeMax) {
                    // copy values into valueAccessor
                    for (auto iter = tempTree->cbeginValueOn(); iter; ++iter) {
                        valueAccessor.modifyValue(iter.getCoord(), SumOpT(*iter));
                    }

                    // copy values into weightAccessor
                    if (weightAccessor) {
                        for (auto iter = tempWeightTree->cbeginValueOn(); iter; ++iter) {
                            weightAccessor->modifyValue(iter.getCoord(), SumOpT(*iter));
                        }
                    }
                }

            } else {
                if (!mAlignedTransform) {
                    position = transform.worldToIndex(
                        pointsTransform.indexToWorld(position));
                }

                if (doRadius) {
                    if (isFrustum) {
                        if (mSettings.accurateFrustumRadius) {
                            this->rasterFrustumSphere(position, mScale, attributeScale,
                                radiusWS, transform, clipBBox, mInterrupter, modifyVoxelByDistanceOp);
                        } else {
                            this->rasterApproximateFrustumSphere(position, mScale, attributeScale,
                                radiusWS, transform, clipBBox, mInterrupter, modifyVoxelByDistanceOp);
                        }
                    } else {
                        this->rasterVoxelSphere(position, mScale, attributeScale,
                            radius, mInterrupter, modifyVoxelByDistanceOp);
                    }
                } else {
                    this->rasterPoint(position, mScale, attributeScale, modifyVoxelOp);
                }
            }
        }

        // if drop buffers is enabled, swap the leaf buffer with a partial (empty) buffer,
        // so the voxel data is de-allocated to reduce memory

        if (mDropBuffers) {
            typename PointLeafT::Buffer emptyBuffer(PartialCreate(), zeroVal<PointIndexT>());
            (const_cast<PointLeafT&>(leaf)).swap(emptyBuffer);
        }
    }

// TODO: would be better to move some of the immutable values into a shared struct

private:
    mutable math::Ray<double> mDdaRay;
    mutable math::DDA<openvdb::math::Ray<double>> mDda;
    const PointDataGridT& mGrid;
    const std::vector<Index64>& mOffsets;
    const size_t mAttributeIndex;
    const Name mVelocityAttribute;
    const Name mRadiusAttribute;
    CombinableT& mCombinable;
    CombinableT* mWeightCombinable;
    const bool mDropBuffers;
    const float mScale;
    const FilterT& mFilter;
    const bool mComputeMax;
    const bool mAlignedTransform;
    const bool mStaticCamera;
    const FrustumRasterizerSettings& mSettings;
    const FrustumRasterizerMask& mMask;
    util::NullInterrupter* mInterrupter;
}; // struct RasterizeOp


/// @brief Combines multiple grids into one by stealing leaf nodes and summing voxel values
/// This class is designed to work with thread local storage containers such as tbb::combinable
template<typename GridT>
struct GridCombinerOp
{
    using CombinableT = typename tbb::combinable<GridT>;

    using TreeT = typename GridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueType = typename TreeT::ValueType;
    using SumOpT = tools::valxform::SumOp<typename TreeT::ValueType>;
    using MaxOpT = tools::valxform::MaxOp<typename TreeT::ValueType>;

    GridCombinerOp(GridT& grid, bool sum)
        : mTree(grid.tree())
        , mSum(sum) {}

    void operator()(const GridT& grid)
    {
        for (auto leaf = grid.tree().beginLeaf(); leaf; ++leaf) {
            auto* newLeaf = mTree.probeLeaf(leaf->origin());
            if (!newLeaf) {
                // if the leaf doesn't yet exist in the new tree, steal it
                auto& tree = const_cast<GridT&>(grid).tree();
                mTree.addLeaf(tree.template stealNode<LeafT>(leaf->origin(),
                    zeroVal<ValueType>(), false));
            }
            else {
                // otherwise increment existing values
                if (mSum) {
                    for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                        newLeaf->modifyValue(iter.offset(), SumOpT(ValueType(*iter)));
                    }
                } else {
                    for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                        newLeaf->modifyValue(iter.offset(), MaxOpT(ValueType(*iter)));
                    }
                }
            }
        }
    }

private:
    TreeT& mTree;
    bool mSum;
}; // struct GridCombinerOp


template<typename GridT>
struct CombinableTraits {
    using OpT = GridCombinerOp<GridT>;
    using T = typename OpT::CombinableT;
};


template <typename PointDataGridT>
class GridToRasterize
{
public:
    using GridPtr = typename PointDataGridT::Ptr;
    using GridConstPtr = typename PointDataGridT::ConstPtr;
    using PointDataTreeT = typename PointDataGridT::TreeType;
    using PointDataLeafT = typename PointDataTreeT::LeafNodeType;

    GridToRasterize(GridPtr& grid, bool stream,
         const FrustumRasterizerSettings& settings,
         const FrustumRasterizerMask& mask)
        : mGrid(grid)
        , mStream(stream)
        , mSettings(settings)
        , mMask(mask) { }

    GridToRasterize(GridConstPtr& grid, const FrustumRasterizerSettings& settings,
         const FrustumRasterizerMask& mask)
        : mGrid(ConstPtrCast<PointDataGridT>(grid))
        , mStream(false)
        , mSettings(settings)
        , mMask(mask) { }

    const typename PointDataGridT::TreeType& tree() const
    {
        return mGrid->constTree();
    }

    void setLeafPercentage(int percentage)
    {
        mLeafPercentage = math::Clamp(percentage, 0, 100);
    }

    int leafPercentage() const
    {
        return mLeafPercentage;
    }

    size_t memUsage() const
    {
        return mGrid->memUsage() + mLeafOffsets.capacity();
    }

    template <typename AttributeT, typename GridT, typename FilterT>
    void rasterize(const Name& attribute,
        typename CombinableTraits<GridT>::T& combiner, typename CombinableTraits<GridT>::T* weightCombiner,
        const float scale, const FilterT& groupFilter, const bool computeMax, const bool reduceMemory,
        util::NullInterrupter* interrupter)
    {
        using point_rasterize_internal::RasterizeOp;

        const auto& velocityAttribute = mSettings.velocityMotionBlur ?
            mSettings.velocityAttribute : "";
        const auto& radiusAttribute = mSettings.useRadius ?
            mSettings.radiusAttribute : "";

        bool isPositionAttribute = attribute == "P";
        bool isVelocityAttribute = attribute == mSettings.velocityAttribute;
        bool isRadiusAttribute = attribute == mSettings.radiusAttribute;

        // find the attribute index
        const auto& attributeSet = mGrid->constTree().cbeginLeaf()->attributeSet();
        const size_t attributeIndex = attributeSet.find(attribute);
        const bool attributeExists = attributeIndex != points::AttributeSet::INVALID_POS;

        if (attributeExists)
        {
            // throw if attribute type doesn't match AttributeT
            const auto* attributeArray = attributeSet.getConst(attributeIndex);
            const Index stride = bool(attributeArray) ? attributeArray->stride() : Index(1);
            const auto& actualValueType = attributeSet.descriptor().valueType(attributeIndex);
            const auto requestedValueType = Name(typeNameAsString<AttributeT>());
            const bool packVector = stride == 3 &&
                ((actualValueType == typeNameAsString<float>() &&
                    requestedValueType == typeNameAsString<Vec3f>()) ||
                 (actualValueType == typeNameAsString<double>() &&
                    requestedValueType == typeNameAsString<Vec3d>()) ||
                 (actualValueType == typeNameAsString<int32_t>() &&
                    requestedValueType == typeNameAsString<Vec3I>()));
            if (!packVector && actualValueType != requestedValueType) {
                OPENVDB_THROW(TypeError,
                    "Attribute type requested for rasterization \"" << requestedValueType << "\""
                    " must match attribute value type \"" << actualValueType << "\"");
            }
        }

        tree::LeafManager<PointDataTreeT> leafManager(mGrid->tree());

        // de-allocate voxel buffers if the points are being streamed and the caches are being released
        const bool dropBuffers = mStream && reduceMemory;

        // turn RasterGroups into a MultiGroupFilter for this VDB Grid (if applicable)
        using ResolvedFilterT = typename point_rasterize_internal::RasterGroupTraits<FilterT>::NewFilterT;
        auto resolvedFilter = point_rasterize_internal::RasterGroupTraits<FilterT>::resolve(
            groupFilter, attributeSet);

        // generate leaf offsets (if necessary)
        if (mLeafOffsets.empty()) {
            openvdb::points::pointOffsets(mLeafOffsets, mGrid->constTree(), resolvedFilter,
                /*inCoreOnly=*/false, mSettings.threaded);
        }

        // set streaming arbitrary attribute array flags
        if (mStream) {
            if (attributeExists && !isPositionAttribute && !isVelocityAttribute && !isRadiusAttribute) {
                leafManager.foreach(
                    [&](PointDataLeafT& leaf, size_t /*idx*/) {
                        leaf.attributeArray(attributeIndex).setStreaming(true);
                    },
                mSettings.threaded);
            }

            if (reduceMemory) {
                size_t positionIndex = attributeSet.find("P");
                leafManager.foreach(
                    [&](PointDataLeafT& leaf, size_t /*idx*/) {
                        leaf.attributeArray(positionIndex).setStreaming(true);
                    },
                mSettings.threaded);
                if (mSettings.velocityMotionBlur || !mSettings.camera.isStatic()) {
                    size_t velocityIndex = attributeSet.find(velocityAttribute);
                    if (velocityIndex != points::AttributeSet::INVALID_POS) {
                        leafManager.foreach(
                            [&](PointDataLeafT& leaf, size_t /*idx*/) {
                                leaf.attributeArray(velocityIndex).setStreaming(true);
                            },
                        mSettings.threaded);
                    }
                }
                if (mSettings.useRadius) {
                    size_t radiusIndex = attributeSet.find(radiusAttribute);
                    if (radiusIndex != points::AttributeSet::INVALID_POS) {
                        leafManager.foreach(
                            [&](PointDataLeafT& leaf, size_t /*idx*/) {
                                leaf.attributeArray(radiusIndex).setStreaming(true);
                            },
                        mSettings.threaded);
                    }
                }
            }
        }

        const bool alignedTransform = *(mSettings.transform) == mGrid->constTransform();

        RasterizeOp<PointDataGridT, AttributeT, GridT, ResolvedFilterT> rasterizeOp(
            *mGrid, mLeafOffsets, attributeIndex, velocityAttribute, radiusAttribute, combiner, weightCombiner,
            dropBuffers, scale, resolvedFilter, computeMax, alignedTransform, mSettings.camera.isStatic(),
            mSettings, mMask, interrupter);
        leafManager.foreach(rasterizeOp, mSettings.threaded);

        // clean up leaf offsets (if reduce memory is enabled)
        if (reduceMemory && !mLeafOffsets.empty()) {
            mLeafOffsets.clear();
            // de-allocate the vector data to ensure we keep memory footprint as low as possible
            mLeafOffsets.shrink_to_fit();
        }
    }

private:
    GridPtr mGrid;
    const bool mStream;
    const FrustumRasterizerSettings& mSettings;
    const FrustumRasterizerMask& mMask;
    int mLeafPercentage = -1;
    std::vector<Index64> mLeafOffsets;
}; // class GridToRasterize


template <typename ValueT>
typename std::enable_if<!ValueTraits<ValueT>::IsVec, ValueT>::type
computeWeightedValue(const ValueT& value, const ValueT& weight)
{
    constexpr bool isSignedInt = std::is_integral<ValueT>() && std::is_signed<ValueT>();

    if (!math::isFinite(weight) || math::isApproxZero(weight) ||
        (isSignedInt && math::isNegative(weight))) {
        return zeroVal<ValueT>();
    } else {
        return value / weight;
    }
}


template <typename ValueT>
typename std::enable_if<ValueTraits<ValueT>::IsVec, ValueT>::type
computeWeightedValue(const ValueT& value, const ValueT& weight)
{
    using ElementT = typename ValueTraits<ValueT>::ElementType;

    constexpr bool isSignedInt = std::is_integral<ElementT>() && std::is_signed<ElementT>();

    ValueT result(value);
    for (int i=0; i<ValueTraits<ValueT>::Size; ++i) {
        if (!math::isFinite(weight[i]) || math::isApproxZero(weight[i]) ||
            (isSignedInt && math::isNegative(weight[i]))) {
            result[i] = zeroVal<ElementT>();
        } else {
            result[i] = value[i] / weight[i];
        }
    }
    return result;
}


} // namespace point_rasterize_internal

/// @endcond

////////////////////////////////////////////////////////////////////////////


RasterCamera::RasterCamera(const math::Transform& transform)
    : mTransforms{transform}
    , mWeights{1} { }

bool RasterCamera::isStatic() const
{
    return mTransforms.size() <= 1;
}

void RasterCamera::clear()
{
    mTransforms.clear();
    mWeights.clear();
}

void RasterCamera::appendTransform(const math::Transform& transform, float weight)
{
    mTransforms.push_back(transform);
    mWeights.push_back(weight);
}

size_t RasterCamera::size() const
{
    return mTransforms.size();
}

void RasterCamera::simplify()
{
    // if two or more identical transforms, only keep one
    if (mTransforms.size() >= 2) {
        const auto& transform = mTransforms.front();
        bool isStatic = true;
        for (const auto& testTransform : mTransforms) {
            if (transform != testTransform) {
                isStatic = false;
            }
        }
        if (isStatic) {
            while (mTransforms.size() > 1) {
                mTransforms.pop_back();
            }
        }
    }
    // if all weights are equal to one, delete the weights array
    if (!mWeights.empty()) {
        bool hasWeight = false;
        for (Index i = 0; i < mWeights.size(); i++) {
            if (this->hasWeight(i)) {
                hasWeight = true;
                break;
            }
        }
        if (!hasWeight)     mWeights.clear();
    }
}

bool RasterCamera::hasWeight(Index i) const
{
    if (mWeights.empty())  return false;
    OPENVDB_ASSERT(i < mWeights.size());
    return !openvdb::math::isApproxEqual(mWeights[i], 1.0f, 1e-3f);
}

float RasterCamera::weight(Index i) const
{
    if (mWeights.empty()) {
        return 1.0f;
    } else {
        OPENVDB_ASSERT(i < mWeights.size());
        return mWeights[i];
    }
}

const math::Transform& RasterCamera::transform(Index i) const
{
    if (mTransforms.size() == 1) {
        return mTransforms.front();
    } else {
        OPENVDB_ASSERT(i < mTransforms.size());
        return mTransforms[i];
    }
}

const math::Transform& RasterCamera::firstTransform() const
{
    OPENVDB_ASSERT(!mTransforms.empty());
    return mTransforms.front();
}

const math::Transform& RasterCamera::lastTransform() const
{
    OPENVDB_ASSERT(!mTransforms.empty());
    return mTransforms.back();
}

void RasterCamera::setShutter(float start, float end)
{
    mShutterStart = start;
    mShutterEnd = end;
}

float RasterCamera::shutterStart() const
{
    return mShutterStart;
}

float RasterCamera::shutterEnd() const
{
    return mShutterEnd;
}


////////////////////////////////////////////////////////////////////////////


FrustumRasterizerMask::FrustumRasterizerMask( const math::Transform& transform,
                                            const MaskGrid* mask,
                                            const BBoxd& bbox,
                                            const bool clipToFrustum,
                                            const bool invertMask)
    : mMask()
    , mClipBBox()
    , mInvert(invertMask)
{
    // TODO: the current OpenVDB implementation for resampling masks is particularly slow,
    // this is primarily because it uses a scatter reduction-style method that only samples
    // and generates leaf nodes and relies on pruning to generate tiles, this could be
    // significantly improved!

    // convert world-space clip mask to index-space
    if (mask) {
        // resample mask to index space
        mMask = MaskGrid::create();
        mMask->setTransform(transform.copy());
        tools::resampleToMatch<tools::PointSampler>(*mask, *mMask);

        // prune the clip mask
        tools::prune(mMask->tree());
    }

    // convert world-space clip bbox to index-space
    if (!bbox.empty()) {
        // create world-space mask (with identity linear transform)
        MaskGrid::Ptr tempMask = MaskGrid::create();
        CoordBBox coordBBox(Coord::floor(bbox.min()),
                            Coord::ceil(bbox.max()));
        tempMask->sparseFill(coordBBox, true, true);

        // resample mask to index space
        MaskGrid::Ptr bboxMask = MaskGrid::create();
        bboxMask->setTransform(mMask ? mMask->transformPtr() : transform.copy());
        tools::resampleToMatch<tools::PointSampler>(*tempMask, *bboxMask);

        // replace or union the mask
        if (mMask) {
            mMask->topologyUnion(*bboxMask);
        } else {
            mMask = bboxMask;
        }
    }

    if (clipToFrustum) {
        auto frustumMap = transform.template constMap<math::NonlinearFrustumMap>();
        if (frustumMap) {
            const auto& frustumBBox = frustumMap->getBBox();
            mClipBBox.reset(Coord::floor(frustumBBox.min()),
                Coord::ceil(frustumBBox.max()));
        }
    }
}

FrustumRasterizerMask::operator bool() const
{
    return mMask || !mClipBBox.empty();
}

MaskTree::ConstPtr
FrustumRasterizerMask::getTreePtr() const
{
    return mMask ? mMask->treePtr() : MaskTree::ConstPtr();
}

const CoordBBox&
FrustumRasterizerMask::clipBBox() const
{
    return mClipBBox;
}

bool
FrustumRasterizerMask::valid(const Coord& ijk, const tree::ValueAccessor<const MaskTree>* acc) const
{
    const bool maskValue = acc ? acc->isValueOn(ijk) : true;
    const bool insideMask = mInvert ? !maskValue : maskValue;
    const bool insideFrustum = mClipBBox.empty() ? true : mClipBBox.isInside(ijk);
    return insideMask && insideFrustum;
}


////////////////////////////////////////////////////////////////////////////


template <typename PointDataGridT>
FrustumRasterizer<PointDataGridT>::FrustumRasterizer(const FrustumRasterizerSettings& settings,
                                                     const FrustumRasterizerMask& mask,
                                                     util::NullInterrupter* interrupt)
    : mSettings(settings)
    , mMask(mask)
    , mInterrupter(interrupt)
{
    if (mSettings.velocityAttribute.empty() && mSettings.velocityMotionBlur) {
        OPENVDB_THROW(ValueError,
            "Using velocity motion blur during rasterization requires a velocity attribute.");
    }
}

template <typename PointDataGridT>
void
FrustumRasterizer<PointDataGridT>::addPoints(GridConstPtr& grid)
{
    // skip any empty grids
    if (!grid || grid->tree().empty())     return;

    // note that streaming is not possible with a const grid
    mPointGrids.emplace_back(grid, mSettings, mMask);
}

template <typename PointDataGridT>
void
FrustumRasterizer<PointDataGridT>::addPoints(GridPtr& grid, bool stream)
{
    // skip any empty grids
    if (!grid || grid->tree().empty())     return;

    mPointGrids.emplace_back(grid, stream, mSettings, mMask);
}

template <typename PointDataGridT>
void
FrustumRasterizer<PointDataGridT>::clear()
{
    mPointGrids.clear();
}

template <typename PointDataGridT>
size_t
FrustumRasterizer<PointDataGridT>::size() const
{
    return mPointGrids.size();
}

template <typename PointDataGridT>
size_t
FrustumRasterizer<PointDataGridT>::memUsage() const
{
    size_t mem = sizeof(*this) + sizeof(mPointGrids);
    for (const auto& grid : mPointGrids) {
        mem += grid.memUsage();
    }
    return mem;
}

template <typename PointDataGridT>
template <typename FilterT>
FloatGrid::Ptr
FrustumRasterizer<PointDataGridT>::rasterizeUniformDensity(
    RasterMode mode, bool reduceMemory, float scale, const FilterT& filter)
{
    // no attribute to rasterize, so just provide an empty string and default to float type
    auto density = rasterizeAttribute<FloatGrid, float>("", mode, reduceMemory, scale, filter);
    // hard-code grid name to density
    density->setName("density");
    return density;
}

template <typename PointDataGridT>
template <typename FilterT>
FloatGrid::Ptr
FrustumRasterizer<PointDataGridT>::rasterizeDensity(
    const openvdb::Name& attribute, RasterMode mode, bool reduceMemory, float scale, const FilterT& filter)
{
    auto density = rasterizeAttribute<FloatGrid, float>(attribute, mode, reduceMemory, scale, filter);
    // hard-code grid name to density
    density->setName("density");
    return density;
}

template <typename PointDataGridT>
template <typename FilterT>
GridBase::Ptr
FrustumRasterizer<PointDataGridT>::rasterizeAttribute(
    const Name& attribute, RasterMode mode, bool reduceMemory, float scale, const FilterT& filter)
{
    // retrieve the source type of the attribute

    Name sourceType;
    Index stride(0);
    for (const auto& points : mPointGrids) {
        auto leaf = points.tree().cbeginLeaf();
        const auto& descriptor = leaf->attributeSet().descriptor();
        const size_t targetIndex = descriptor.find(attribute);
        // ignore grids which don't have the attribute
        if (targetIndex == points::AttributeSet::INVALID_POS)   continue;
        const auto* attributeArray = leaf->attributeSet().getConst(attribute);
        if (!attributeArray)    continue;
        stride = attributeArray->stride();
        sourceType = descriptor.valueType(targetIndex);
        if (!sourceType.empty())    break;
    }

    // no valid point attributes for rasterization
    // TODO: add a warning / error in the case that there are non-zero grids
    if (sourceType.empty())     return {};

    if (stride == 1 && sourceType == typeNameAsString<float>()) {
        using GridT = typename PointDataGridT::template ValueConverter<float>::Type;
        return rasterizeAttribute<GridT, float>(attribute, mode, reduceMemory, scale, filter);
// this define is for lowering compilation time when debugging by instantiating only the float
// code path (default is to instantiate all code paths)
#ifndef ONLY_RASTER_FLOAT
    } else if ( sourceType == typeNameAsString<Vec3f>() ||
                (stride == 3 && sourceType == typeNameAsString<float>())) {
        using GridT = typename PointDataGridT::template ValueConverter<Vec3f>::Type;
        return rasterizeAttribute<GridT, Vec3f>(attribute, mode, reduceMemory, scale, filter);
    } else if ( sourceType == typeNameAsString<Vec3d>() ||
                (stride == 3 && sourceType == typeNameAsString<double>())) {
        using GridT = typename PointDataGridT::template ValueConverter<Vec3d>::Type;
        return rasterizeAttribute<GridT, Vec3d>(attribute, mode, reduceMemory, scale, filter);
    } else if ( sourceType == typeNameAsString<Vec3i>() ||
                (stride == 3 && sourceType == typeNameAsString<int32_t>())) {
        using GridT = typename PointDataGridT::template ValueConverter<Vec3i>::Type;
        return rasterizeAttribute<GridT, Vec3i>(attribute, mode, reduceMemory, scale, filter);
    } else if (stride == 1 && sourceType == typeNameAsString<int16_t>()) {
        using GridT = typename PointDataGridT::template ValueConverter<Int32>::Type;
        return rasterizeAttribute<GridT, int16_t>(attribute, mode, reduceMemory, scale, filter);
    } else if (stride == 1 && sourceType == typeNameAsString<int32_t>()) {
        using GridT = typename PointDataGridT::template ValueConverter<Int32>::Type;
        return rasterizeAttribute<GridT, int32_t>(attribute, mode, reduceMemory, scale, filter);
    } else if (stride == 1 && sourceType == typeNameAsString<int64_t>()) {
        using GridT = typename PointDataGridT::template ValueConverter<Int64>::Type;
        return rasterizeAttribute<GridT, int64_t>(attribute, mode, reduceMemory, scale, filter);
    } else if (stride == 1 && sourceType == typeNameAsString<double>()) {
        using GridT = typename PointDataGridT::template ValueConverter<double>::Type;
        return rasterizeAttribute<GridT, double>(attribute, mode, reduceMemory, scale, filter);
    } else if (stride == 1 && sourceType == typeNameAsString<bool>()) {
        using GridT = typename PointDataGridT::template ValueConverter<bool>::Type;
        return rasterizeAttribute<GridT, bool>(attribute, mode, reduceMemory, true, filter);
#endif
    } else {
        std::ostringstream ostr;
        ostr << "Cannot rasterize attribute of type - " << sourceType;
        if (stride > 1)    ostr << " x " << stride;
        OPENVDB_THROW(TypeError, ostr.str());
    }
}

template <typename PointDataGridT>
template <typename GridT, typename AttributeT, typename FilterT>
typename GridT::Ptr
FrustumRasterizer<PointDataGridT>::rasterizeAttribute(const Name& attribute, RasterMode mode,
    bool reduceMemory, float scale, const FilterT& filter)
{
    if (attribute == "P") {
        OPENVDB_THROW(ValueError, "Cannot rasterize position attribute.")
    }

    auto grid = GridT::create();
    grid->setName(attribute);
    grid->setTransform(mSettings.transform->copy());

    this->performRasterization<AttributeT>(*grid, mode, attribute, reduceMemory, scale, filter);

    return grid;
}

template <typename PointDataGridT>
template <typename GridT, typename FilterT>
typename GridT::Ptr
FrustumRasterizer<PointDataGridT>::rasterizeMask(bool reduceMemory, const FilterT& filter)
{
    using ValueT = typename GridT::ValueType;

    static_assert(point_rasterize_internal::BoolTraits<ValueT>::IsBool,
        "Value type of mask to be rasterized must be bool or ValueMask.");

    auto grid = rasterizeAttribute<GridT, ValueT>("", RasterMode::ACCUMULATE, reduceMemory, true, filter);
    grid->setName("mask");

    return grid;
}

template <typename PointDataGridT>
template <typename AttributeT, typename GridT, typename FilterT>
void
FrustumRasterizer<PointDataGridT>::performRasterization(
    GridT& grid, RasterMode mode, const openvdb::Name& attribute, bool reduceMemory,
    float scale, const FilterT& filter)
{
    using openvdb::points::point_mask_internal::GridCombinerOp;
    using point_rasterize_internal::computeWeightedValue;

    using TreeT = typename GridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;
    using CombinerOpT = typename point_rasterize_internal::CombinableTraits<GridT>::OpT;
    using CombinableT = typename point_rasterize_internal::CombinableTraits<GridT>::T;

    // start interrupter with a descriptive name
    if (mInterrupter) {
        std::stringstream ss;
        ss << "Rasterizing Points ";
        if (!mSettings.camera.isStatic() || mSettings.velocityMotionBlur) {
            ss << "with Motion Blur ";
        }
        if (mSettings.transform->isLinear()) {
            ss << "to Linear Volume";
        } else {
            ss << "to Frustum Volume";
        }
        mInterrupter->start(ss.str().c_str());
    }

    const bool useMaximum = mode == RasterMode::MAXIMUM;
    const bool useWeight = mode == RasterMode::AVERAGE;

    if (useMaximum && VecTraits<ValueT>::IsVec) {
        OPENVDB_THROW(ValueError,
            "Cannot use maximum mode when rasterizing vector attributes.");
    }

    if ((useMaximum || useWeight) && point_rasterize_internal::BoolTraits<ValueT>::IsBool) {
        OPENVDB_THROW(ValueError,
            "Cannot use maximum or average modes when rasterizing bool attributes.");
    }

    CombinableT combiner;
    CombinableT weightCombiner;
    CombinableT* weightCombinerPtr = useWeight ? &weightCombiner : nullptr;

    // use leaf count as an approximate indicator for progress as it
    // doesn't even require loading the topology

    if (mInterrupter) {
        if (mPointGrids.size() == 1) {
            mPointGrids[0].setLeafPercentage(100);
        }
        else {
            // compute cumulative leaf counts per grid
            Index64 leafCount(0);
            std::vector<Index64> leafCounts;
            leafCounts.reserve(mPointGrids.size());
            for (auto& points : mPointGrids) {
                leafCount += points.tree().leafCount();
                leafCounts.push_back(leafCount);
            }

            // avoid dividing by zero
            if (leafCount == Index64(0))    leafCount++;

            // compute grid percentages based on leaf count
            for (size_t i = 0; i < leafCounts.size(); i++) {
                int percentage = static_cast<int>(math::Round((static_cast<float>(leafCounts[i]))/
                    static_cast<float>(leafCount)));
                mPointGrids[i].setLeafPercentage(percentage);
            }
        }
    }

    // rasterize each point grid into each grid in turn

    for (auto& points : mPointGrids) {

        points.template rasterize<AttributeT, GridT>(
            attribute, combiner, weightCombinerPtr, scale, filter, mode == RasterMode::MAXIMUM, reduceMemory, mInterrupter);

        // interrupt if requested and update the progress percentage
        // note that even when interrupting, the operation to combine the local grids
        // is completed so that the user receives a partially rasterized volume

        if (mInterrupter &&
            mInterrupter->wasInterrupted(points.leafPercentage())) {
            break;
        }
    }

    // combine the value grids into one

    CombinerOpT combineOp(grid, mode != RasterMode::MAXIMUM);
    combiner.combine_each(combineOp);

    if (useWeight) {

        // combine the weight grids into one

        auto weightGrid = GridT::create(ValueT(1));
        CombinerOpT weightCombineOp(*weightGrid, /*sum=*/true);
        weightCombiner.combine_each(weightCombineOp);

        tree::LeafManager<TreeT> leafManager(grid.tree());
        leafManager.foreach(
            [&](LeafT& leaf, size_t /*idx*/) {
                auto weightAccessor = weightGrid->getConstAccessor();
                for (auto iter = leaf.beginValueOn(); iter; ++iter) {
                    auto weight = weightAccessor.getValue(iter.getCoord());
                    iter.setValue(computeWeightedValue(iter.getValue(), weight));
                }
            },
        mSettings.threaded);
    }

    if (mInterrupter)   mInterrupter->end();
}


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_RASTERIZE_FRUSTUM_IMPL_HAS_BEEN_INCLUDED
