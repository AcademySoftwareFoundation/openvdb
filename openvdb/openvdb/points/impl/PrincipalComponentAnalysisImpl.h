// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_POINTS_POINT_PRINCIPAL_COMPONENT_ANALYSIS_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_PRINCIPAL_COMPONENT_ANALYSIS_IMPL_HAS_BEEN_INCLUDED

#ifdef OPENVDB_PROFILE_PCA
#include <openvdb/util/CpuTimer.h>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

namespace pca_internal {

#ifdef OPENVDB_PROFILE_PCA
using PcaTimer = util::CpuTimer;
#else
struct PcaTimer {
inline void start(const char*) {}
inline void stop() {}
};
#endif

using WeightSumT = double;
using WeightedPositionSumT = Vec3d;
using GroupIndexT = points::AttributeSet::Descriptor::GroupIndex;

struct AttrIndices
{
    size_t mPosSumIndex;
    size_t mWeightSumIndex;
    size_t mCovMatrixIndex;
    size_t mPWsIndex;
    GroupIndexT mEllipsesGroupIndex;
};

#if OPENVDB_ABI_VERSION_NUMBER >= 9
template <typename T, typename LeafNodeT>
inline T* initPcaArrayAttribute(LeafNodeT& leaf, const size_t idx, const bool fill = true)
{
    auto& array = leaf.attributeArray(idx);
    assert(array.valueType() == typeNameAsString<T>());
    assert(array.codecType() == std::string(NullCodec::name()));
    // No need to call loadData on these attributes, we know we created them
    array.expand(fill);
    const char* data = array.constDataAsByteArray();
    return reinterpret_cast<T*>(const_cast<char*>(data));
}
#else
template <typename T, typename LeafNodeT>
inline typename AttributeWriteHandle<T, NullCodec>::Ptr
initPcaArrayAttribute(LeafNodeT& leaf, const size_t idx)
{
    return AttributeWriteHandle<T, NullCodec>::create(leaf.attributeArray(idx), /*expand=*/true);
}
#endif

template <typename PointDataTreeT>
struct PcaTransfer
    : public VolumeTransfer<PointDataTreeT>
{
    using BaseT = VolumeTransfer<PointDataTreeT>;
    using LeafNodeType = typename PointDataTreeT::LeafNodeType;
    using PositionHandleT = points::AttributeHandle<Vec3d, NullCodec>;

    PcaTransfer(const AttrIndices& indices,
                const float searchRadius,
                const Real vs,
                tree::LeafManager<PointDataTreeT>& manager)
        : BaseT(manager.tree())
        , mIndices(indices)
        , mSearchRadius(searchRadius)
        , mDxInv(1.0/vs)
        , mManager(manager)
        , mTargetPosition()
        , mSourcePosition() {}

    PcaTransfer(const PcaTransfer& other)
        : BaseT(other)
        , mIndices(other.mIndices)
        , mSearchRadius(other.mSearchRadius)
        , mDxInv(other.mDxInv)
        , mManager(other.mManager)
        , mTargetPosition()
        , mSourcePosition() {}

    /*
    // This is no longer used but kept for reference; each axis iterations of
    // derived PCA methods uses this technique to skip voxels entirely if a
    // point's position and search radius does not intersect it

    static bool VoxelIntersectsSphere(const Coord& ijk, const Vec3d& PosIS, const Real r2)
    {
        const Vec3d min = ijk.asVec3d() - 0.5;
        const Vec3d max = ijk.asVec3d() + 0.5;
        Real dmin = 0;
        for (int i = 0; i < 3; ++i) {
            if (PosIS[i] < min[i])      dmin += math::Pow2(PosIS[i] - min[i]);
            else if (PosIS[i] > max[i]) dmin += math::Pow2(PosIS[i] - max[i]);
        }
        return dmin <= r2;
    }
    */

    Vec3i range(const Coord&, size_t) const { return this->range(); }
    Vec3i range() const { return Vec3i(math::Round(mSearchRadius * mDxInv)); }

    inline LeafNodeType* initialize(const Coord& origin, const size_t idx, const CoordBBox& bounds)
    {
        BaseT::initialize(origin, idx, bounds);
        auto& leaf = mManager.leaf(idx);
        mTargetPosition.reset(new PositionHandleT(leaf.constAttributeArray(mIndices.mPWsIndex)));
        return &leaf;
    }

    inline bool startPointLeaf(const typename PointDataTreeT::LeafNodeType& leaf)
    {
        mSourcePosition.reset(new PositionHandleT(leaf.constAttributeArray(mIndices.mPWsIndex)));
        return true;
    }

    bool endPointLeaf(const typename PointDataTreeT::LeafNodeType&) { return true; }

protected:
    const AttrIndices& mIndices;
    const float mSearchRadius;
    const Real mDxInv;
    const tree::LeafManager<PointDataTreeT>& mManager;
    std::unique_ptr<PositionHandleT> mTargetPosition;
    std::unique_ptr<PositionHandleT> mSourcePosition;
};


template <typename PointDataTreeT>
struct WeightPosSumsTransfer
    : public PcaTransfer<PointDataTreeT>
{
    using BaseT = PcaTransfer<PointDataTreeT>;

    static const Index DIM = PointDataTreeT::LeafNodeType::DIM;
    static const Index LOG2DIM = PointDataTreeT::LeafNodeType::LOG2DIM;

    WeightPosSumsTransfer(const AttrIndices& indices,
                          const float searchRadius,
                          const int32_t neighbourThreshold,
                          const Real vs,
                          tree::LeafManager<PointDataTreeT>& manager)
        : BaseT(indices, searchRadius, vs, manager)
        , mNeighbourThreshold(neighbourThreshold)
        , mWeights()
        , mWeightedPositions()
        , mCounts() {}

    WeightPosSumsTransfer(const WeightPosSumsTransfer& other)
        : BaseT(other)
        , mNeighbourThreshold(other.mNeighbourThreshold)
        , mWeights()
        , mWeightedPositions()
        , mCounts() {}

    inline void initialize(const Coord& origin, const size_t idx, const CoordBBox& bounds)
    {
        auto& leaf = (*BaseT::initialize(origin, idx, bounds));
        mWeights = initPcaArrayAttribute<WeightSumT>(leaf, this->mIndices.mWeightSumIndex);
        mWeightedPositions = initPcaArrayAttribute<WeightedPositionSumT>(leaf, this->mIndices.mPosSumIndex);
        // track neighbours
        mCounts.assign(this->mTargetPosition->size(), 0);
    }

    inline void rasterizePoint(const Coord&,
                    const Index pid,
                    const CoordBBox& bounds)
    {
        const Vec3d Psrc(this->mSourcePosition->get(pid));
        const Vec3d PsrcIS = Psrc * this->mDxInv;

        const auto* const data = this->template buffer<0>();
        const auto& mask = *(this->template mask<0>());

        const float searchRadiusInv = 1.0f / this->mSearchRadius;
        const Real searchRadius2 = math::Pow2(this->mSearchRadius);
        const Real searchRadiusIS2 = math::Pow2(this->mSearchRadius * this->mDxInv);

        const Coord& a(bounds.min());
        const Coord& b(bounds.max());
        for (Coord c = a; c.x() <= b.x(); ++c.x()) {
            const Real minx = c.x() - 0.5;
            const Real maxx = c.x() + 0.5;
            const Real dminx =
                (PsrcIS[0] < minx ? math::Pow2(PsrcIS[0] - minx) :
                (PsrcIS[0] > maxx ? math::Pow2(PsrcIS[0] - maxx) : 0));
            if (dminx > searchRadiusIS2) continue;
            const Index i = ((c.x() & (DIM-1u)) << 2*LOG2DIM); // unsigned bit shift mult
            for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                const Real miny = c.y() - 0.5;
                const Real maxy = c.y() + 0.5;
                const Real dminxy = dminx +
                    (PsrcIS[1] < miny ? math::Pow2(PsrcIS[1] - miny) :
                    (PsrcIS[1] > maxy ? math::Pow2(PsrcIS[1] - maxy) : 0));
                if (dminxy > searchRadiusIS2) continue;
                const Index ij = i + ((c.y() & (DIM-1u)) << LOG2DIM);
                for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                    const Index offset = ij + /*k*/(c.z() & (DIM-1u));
                    if (!mask.isOn(offset)) continue;

                    const Real minz = c.z() - 0.5;
                    const Real maxz = c.z() + 0.5;
                    const Real dminxyz = dminxy +
                        (PsrcIS[2] < minz ? math::Pow2(PsrcIS[2] - minz) :
                        (PsrcIS[2] > maxz ? math::Pow2(PsrcIS[2] - maxz) : 0));
                    // Does this point's radius overlap the voxel c
                    if (dminxyz > searchRadiusIS2) continue;

                    const Index end = data[offset];
                    Index id = (offset == 0) ? 0 : Index(data[offset - 1]);
                    for (; id < end; ++id) {
                        const Vec3d Ptgt(this->mTargetPosition->get(id));
                        const Real d2 = (Psrc - Ptgt).lengthSqr();
                        if (d2 > searchRadius2) continue;

                        const float weight = 1.0f - math::Pow3(float(math::Sqrt(d2)) * searchRadiusInv);
                        assert(weight >= 0.0f && weight <= 1.0f);

#if OPENVDB_ABI_VERSION_NUMBER >= 9
                        mWeights[id] += weight;
                        mWeightedPositions[id] += Psrc * weight; // @note: world space position is weighted
#else
                        // @warning  much slower than accessing the buffers directly
                        mWeights->set(id, mWeights->get(id) + weight);
                        mWeightedPositions->set(id, mWeightedPositions->get(id) + (Psrc * weight)); // @note: world space position is weighted
#endif
                        ++mCounts[id];
                    } //point idx
                }
            }
        } // outer sdf voxel
    } // point idx

    bool finalize(const Coord&, size_t idx)
    {
        // Add points to group with counts which are more than the neighbouring threshold
        auto& leaf = this->mManager.leaf(idx);

        {
            // @todo add API to get the array from the group handle. The handle
            //   calls loadData but not expand.
            auto& array = leaf.attributeArray(this->mIndices.mEllipsesGroupIndex.first);
            array.loadData(); // so we can call setUnsafe/getUnsafe
            array.expand();
        }

        points::GroupWriteHandle group(leaf.groupWriteHandle(this->mIndices.mEllipsesGroupIndex));

        for (Index i = 0; i < this->mTargetPosition->size(); ++i)
        {
            // every point will have a self contribution (unless this operation
            // has been interrupted or the search radius was 0) so account for
            // that here
            assert(mCounts[i] >= 1);
            assert(mWeights[i] > 0.0f);
            mCounts[i] -= 1;

            // turn points OFF if they are ON and don't meet max neighbour requirements
            if ((mCounts[i] < mNeighbourThreshold) && group.getUnsafe(i)) {
                group.setUnsafe(i, false);
            }

            // only self contribution, don't bothering normalizing
            if (mCounts[i] <= 0) continue;

#if OPENVDB_ABI_VERSION_NUMBER >= 9
            // Account for self contribution
            mWeights[i] -= 1.0f;
            mWeightedPositions[i] -= this->mTargetPosition->get(i);
            assert(mWeights[i] > 0.0f);
            // Now normalize
            mWeights[i] = 1.0 / mWeights[i];
            mWeightedPositions[i] *= mWeights[i];
#else
            // Account for self contribution
            mWeights->set(i, mWeights->get(i) - 1.0f);
            mWeightedPositions->set(i, mWeightedPositions->get(i) - this->mTargetPosition->get(i));
            assert(mWeights->get(i) > 0.0f);
            // Now normalize
            mWeights->set(i, 1.0 / mWeights->get(i));
            mWeightedPositions->set(i, mWeightedPositions->get(i) * mWeights->get(i));
#endif
        }
        return true;
    }

private:
    const int32_t mNeighbourThreshold;
#if OPENVDB_ABI_VERSION_NUMBER >= 9
    WeightSumT* mWeights;
    WeightedPositionSumT* mWeightedPositions;
#else
    AttributeWriteHandle<WeightSumT, NullCodec>::Ptr mWeights;
    AttributeWriteHandle<WeightedPositionSumT, NullCodec>::Ptr mWeightedPositions;
#endif
    std::vector<int32_t> mCounts;
};

template <typename PointDataTreeT>
struct CovarianceTransfer
    : public PcaTransfer<PointDataTreeT>
{
    using BaseT = PcaTransfer<PointDataTreeT>;

    static const Index DIM = PointDataTreeT::LeafNodeType::DIM;
    static const Index LOG2DIM = PointDataTreeT::LeafNodeType::LOG2DIM;

    CovarianceTransfer(const AttrIndices& indices,
                       const float searchRadius,
                       const Real vs,
                       tree::LeafManager<PointDataTreeT>& manager)
        : BaseT(indices, searchRadius, vs, manager)
        , mIsSameLeaf()
        , mInclusionGroupHandle()
        , mWeights()
        , mWeightedPositions()
        , mCovMats() {}

    CovarianceTransfer(const CovarianceTransfer& other)
        : BaseT(other)
        , mIsSameLeaf()
        , mInclusionGroupHandle()
        , mWeights()
        , mWeightedPositions()
        , mCovMats() {}

    inline void initialize(const Coord& origin, const size_t idx, const CoordBBox& bounds)
    {
        auto& leaf = (*BaseT::initialize(origin, idx, bounds));
        mInclusionGroupHandle.reset(new points::GroupHandle(leaf.groupHandle(this->mIndices.mEllipsesGroupIndex)));
        mWeights = initPcaArrayAttribute<WeightSumT>(leaf, this->mIndices.mWeightSumIndex);
        mWeightedPositions = initPcaArrayAttribute<WeightedPositionSumT>(leaf, this->mIndices.mPosSumIndex);
        mCovMats = initPcaArrayAttribute<math::Mat3s>(leaf, this->mIndices.mCovMatrixIndex);
    }

    inline bool startPointLeaf(const typename PointDataTreeT::LeafNodeType& leaf)
    {
        BaseT::startPointLeaf(leaf);
        mIsSameLeaf = this->mTargetPosition->array() == this->mSourcePosition->array();
        return true;
    }

    inline void rasterizePoint(const Coord&,
                    const Index pid,
                    const CoordBBox& bounds)
    {
        const Vec3d Psrc(this->mSourcePosition->get(pid));
        const Vec3d PsrcIS = Psrc * this->mDxInv;

        const auto* const data = this->template buffer<0>();
        const auto& mask = *(this->template mask<0>());

        const float searchRadiusInv = 1.0f/this->mSearchRadius;
        const Real searchRadius2 = math::Pow2(this->mSearchRadius);
        const Real searchRadiusIS2 = math::Pow2(this->mSearchRadius * this->mDxInv);

        const Coord& a(bounds.min());
        const Coord& b(bounds.max());
        for (Coord c = a; c.x() <= b.x(); ++c.x()) {
            const Real minx = c.x() - 0.5;
            const Real maxx = c.x() + 0.5;
            const Real dminx =
                (PsrcIS[0] < minx ? math::Pow2(PsrcIS[0] - minx) :
                (PsrcIS[0] > maxx ? math::Pow2(PsrcIS[0] - maxx) : 0));
            if (dminx > searchRadiusIS2) continue;
            const Index i = ((c.x() & (DIM-1u)) << 2*LOG2DIM); // unsigned bit shift mult
            for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                const Real miny = c.y() - 0.5;
                const Real maxy = c.y() + 0.5;
                const Real dminxy = dminx +
                    (PsrcIS[1] < miny ? math::Pow2(PsrcIS[1] - miny) :
                    (PsrcIS[1] > maxy ? math::Pow2(PsrcIS[1] - maxy) : 0));
                if (dminxy > searchRadiusIS2) continue;
                const Index ij = i + ((c.y() & (DIM-1u)) << LOG2DIM);
                for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                    const Index offset = ij + /*k*/(c.z() & (DIM-1u));
                    if (!mask.isOn(offset)) continue;

                    const Real minz = c.z() - 0.5;
                    const Real maxz = c.z() + 0.5;
                    const Real dminxyz = dminxy +
                        (PsrcIS[2] < minz ? math::Pow2(PsrcIS[2] - minz) :
                        (PsrcIS[2] > maxz ? math::Pow2(PsrcIS[2] - maxz) : 0));
                    // Does this point's radius overlap the voxel c
                    if (dminxyz > searchRadiusIS2) continue;

                    const Index end = data[offset];
                    Index id = (offset == 0) ? 0 : Index(data[offset - 1]);
                    for (; id < end; ++id) {
                        if (!mInclusionGroupHandle->get(id)) continue;
                        // @note  Could remove self contribution as in WeightPosSumsTransfer
                        //   it adds a small % overhead to this entire operator
                        if (OPENVDB_UNLIKELY(mIsSameLeaf && id == pid)) continue;
                        const Vec3d Ptgt(this->mTargetPosition->get(id));
                        const Real d2 = (Psrc - Ptgt).lengthSqr();
                        if (d2 > searchRadius2) continue;

#if OPENVDB_ABI_VERSION_NUMBER >= 9
                        // @note  I've observed some performance degradation if
                        //   we don't take copies of the buffers here (aliasing?)
                        const WeightSumT totalWeightInv = mWeights[id];
                        const WeightedPositionSumT currWeightSum = mWeightedPositions[id];
#else
                        const WeightSumT totalWeightInv = mWeights->get(id);
                        const WeightedPositionSumT currWeightSum = mWeightedPositions->get(id);
#endif

                        const WeightSumT weight = 1.0f - math::Pow3(float(math::Sqrt(d2)) * searchRadiusInv);
                        const WeightedPositionSumT posMeanDiff = Psrc - currWeightSum;
                        const WeightedPositionSumT x = (totalWeightInv * weight) * posMeanDiff;

#if OPENVDB_ABI_VERSION_NUMBER >= 9
                        float* const m = mCovMats[id].asPointer();
#else
                        // @warning  much slower than accessing the buffers directly
                        auto cov = mCovMats->get(id);
                        float* const m = cov.asPointer();
#endif
                        /// @note: equal to:
                        // mat.setCol(0, mat.col(0) + (x * posMeanDiff[0]));
                        // mat.setCol(1, mat.col(1) + (x * posMeanDiff[1]));
                        // mat.setCol(2, mat.col(2) + (x * posMeanDiff[2]));
                        m[0] += float(x[0] * posMeanDiff[0]);
                        m[1] += float(x[0] * posMeanDiff[1]);
                        m[2] += float(x[0] * posMeanDiff[2]);
                        //
                        m[3] += float(x[1] * posMeanDiff[0]);
                        m[4] += float(x[1] * posMeanDiff[1]);
                        m[5] += float(x[1] * posMeanDiff[2]);
                        //
                        m[6] += float(x[2] * posMeanDiff[0]);
                        m[7] += float(x[2] * posMeanDiff[1]);
                        m[8] += float(x[2] * posMeanDiff[2]);

#if OPENVDB_ABI_VERSION_NUMBER < 9
                        mCovMats->set(id, cov);
#endif
                    } //point idx
                }
            }
        } // outer sdf voxel
    } // point idx

    bool finalize(const Coord&, size_t) { return true; }

private:
    bool mIsSameLeaf;
    points::GroupHandle::UniquePtr mInclusionGroupHandle;
#if OPENVDB_ABI_VERSION_NUMBER >= 9
    const WeightSumT* mWeights;
    const WeightedPositionSumT* mWeightedPositions;
    math::Mat3s* mCovMats;
#else
    AttributeHandle<WeightSumT, NullCodec>::Ptr mWeights;
    AttributeHandle<WeightedPositionSumT, NullCodec>::Ptr mWeightedPositions;
    AttributeWriteHandle<math::Mat3s, NullCodec>::Ptr mCovMats;
#endif
};

/// @brief Sort a vector into descending order and output a vector of the resulting order
/// @param vector Vector to sort
template <typename Scalar>
inline Vec3i
descendingOrder(math::Vec3<Scalar>& vector)
{
    Vec3i order(0,1,2);
    if (vector[0] < vector[1]) {
        std::swap(vector[0], vector[1]);
        std::swap(order[0], order[1]);
    }
    if (vector[1] < vector[2]) {
        std::swap(vector[1], vector[2]);
        std::swap(order[1], order[2]);
    }
    if (vector[0] < vector[1]) {
        std::swap(vector[0], vector[1]);
        std::swap(order[0], order[1]);
    }
    return order;
}

/// @brief Decomposes a symmetric matrix into its eigenvalues and a rotation matrix of eigenvectors.
///        Note that if mat is positive-definite, this will be equivalent to a singular value
///        decomposition where V = U.
/// @param mat Matrix to decompose
/// @param U rotation matrix.  The order of its columns (which will be eigenvectors) will match
///          the eigenvalues in sigma
/// @param sigma vector of eigenvalues
template <typename Scalar>
inline bool
decomposeSymmetricMatrix(const math::Mat3<Scalar>& mat,
                         math::Mat3<Scalar>& U,
                         math::Vec3<Scalar>& sigma)
{
    math::Mat3<Scalar> Q;
    const bool diagonalized = math::diagonalizeSymmetricMatrix(mat, Q, sigma);

    if (!diagonalized) return false;

    // need to sort eigenvalues and eigenvectors
    Vec3i order = descendingOrder(sigma);

    // we need to re-order the matrix ("Q") columns to match the new eigenvalue order
    // to obtain the correct "U" matrix
    U.setColumns(Q.col(order[0]), Q.col(order[1]), Q.col(order[2]));

    return true;
}

} // namespace pca_internal


template <typename PointDataGridT,
    typename FilterT,
    typename InterrupterT>
inline void
pca(PointDataGridT& points,
    const PcaSettings& settings,
    const PcaAttributes& attrs,
    InterrupterT* interrupt)
{
    static_assert(IsSpecializationOf<PointDataGridT, Grid>::value);

    using namespace pca_internal;

    using PointDataTreeT = typename PointDataGridT::TreeType;
    using LeafManagerT = tree::LeafManager<PointDataTreeT>;
    using LeafNodeT = typename PointDataTreeT::LeafNodeType;

    auto& tree = points.tree();
    const auto leaf = tree.cbeginLeaf();
    if (!leaf) return;

    // Small lambda to init any one of the necessary ellipsoid attributes
    // @note  Various algorithms here assume that we are responsible for creating
    //   these attributes and so can optimize accordingly. If this changes we'll
    //   need to also change those optimizations (e.g. NullCodec, loadData, etc)
    const auto initAttribute = [&](const std::string& name, const auto val)
    {
        using ValueT = std::decay_t<decltype(val)>;
        if (leaf->hasAttribute(name)) {
            OPENVDB_THROW(KeyError, "PCA attribute '"  << name << "' already exists!");
        }

        points::appendAttribute<ValueT>(tree, name, val);
        return leaf->attributeSet().find(name);
    };

    //

    const size_t pvsIdx = leaf->attributeSet().find("P");
    const auto& xform = points.constTransform();
    const double vs = xform.voxelSize()[0];
    LeafManagerT manager(tree);

    // 1) Create persisting attributes
    const size_t pwsIdx = initAttribute(attrs.positionWS, zeroVal<PcaAttributes::PosWsT>());
    const size_t rotIdx = initAttribute(attrs.rotation, zeroVal<PcaAttributes::RotationT>());
    const size_t strIdx = initAttribute(attrs.stretch, PcaAttributes::StretchT(1));

    // 2) Create temporary attributes
    const auto& descriptor = leaf->attributeSet().descriptor();
    const std::vector<std::string> temps {
        descriptor.uniqueName("_weightedpositionsums"),
        descriptor.uniqueName("_inv_weightssum")
    };

    const size_t posSumIndex = initAttribute(temps[0], zeroVal<WeightedPositionSumT>());
    const size_t weightSumIndex = initAttribute(temps[1], zeroVal<WeightSumT>());

    // 3) Create ellipses group
    if (!leaf->attributeSet().descriptor().hasGroup(attrs.ellipses)) {
        points::appendGroup(tree, attrs.ellipses);
        // Include everything by default to start with
        points::setGroup(tree, attrs.ellipses, true);
    }

    // Re-acquire the updated descriptor and get the group idx
    const GroupIndexT ellipsesIdx =
        leaf->attributeSet().descriptor().groupIndex(attrs.ellipses);

    PcaTimer timer;

    // 3) Store the world space position on the PDG to speed up subsequent
    //    calculations.
    timer.start("Compute position world spaces");
    manager.foreach([&](LeafNodeT& leafnode, size_t)
    {
        using PvsT = Vec3f;
        using PwsT = PcaAttributes::PosWsT;

        points::AttributeHandle<PvsT> Pvs(leafnode.constAttributeArray(pvsIdx));
#if OPENVDB_ABI_VERSION_NUMBER >= 9
        PwsT* Pws = initPcaArrayAttribute<PwsT>(leafnode, pwsIdx, /*fill=*/false);
#else
        points::AttributeWriteHandle<PwsT, NullCodec> Pws(leafnode.attributeArray(pwsIdx));
#endif

        for (auto voxel = leafnode.cbeginValueOn(); voxel; ++voxel) {
            const Coord voxelCoord = voxel.getCoord();
            const Vec3d coordVec = voxelCoord.asVec3d();
            for (auto iter = leafnode.beginIndexVoxel(voxelCoord); iter; ++iter) {
#if OPENVDB_ABI_VERSION_NUMBER >= 9
                Pws[*iter] = xform.indexToWorld(Pvs.get(*iter) + coordVec);
#else
                Pws.set(*iter, xform.indexToWorld(Pvs.get(*iter) + coordVec));
#endif
            }
        }
    });

    timer.stop();
    if (util::wasInterrupted(interrupt)) return;

    AttrIndices indices;
    indices.mPosSumIndex = posSumIndex;
    indices.mWeightSumIndex = weightSumIndex;
    indices.mCovMatrixIndex = rotIdx;
    indices.mPWsIndex = pwsIdx;
    indices.mEllipsesGroupIndex = ellipsesIdx;

    // 4) Init temporary attributes and calculate:
    //        sum_j w_{i,j} * x_j / (sum_j w_j)
    //    And neighbour counts for each point.
    // simultaneously calculates the sum of weighted vector positions (sum w_{i,j} * x_i)
    // weighted against the inverse sum of weights (1.0 / sum w_{i,j}). Also counts number
    // of neighours each point has and updates the ellipses group based on minimum
    // neighbour threshold. Those points which are "included" but which lack sufficient
    // neighbours will be marked as "not included".
    timer.start("Compute position weights");
    {
        WeightPosSumsTransfer<PointDataTreeT> transfer(indices,
            settings.searchRadius,
            int32_t(settings.neighbourThreshold),
            float(vs),
            manager);

        points::rasterize<PointDataGridT,
            decltype(transfer),
            NullFilter,
            InterrupterT>(points, transfer, NullFilter(), interrupt);
    }

    timer.stop();
    if (util::wasInterrupted(interrupt)) return;

    // 5) Principal axes define the rotation matrix of the ellipsoid.
    //    Calculates covariance matrices given weighted sums of positions and
    //    sums of weights per-particle
    timer.start("Compute covariance matrices");
    {
        CovarianceTransfer<PointDataTreeT>
            transfer(indices, settings.searchRadius, float(vs), manager);

        points::rasterize<PointDataGridT,
            decltype(transfer),
            NullFilter,
            InterrupterT>(points, transfer, NullFilter(), interrupt);
    }

    timer.stop();
    if (util::wasInterrupted(interrupt)) return;

    // 6) radii stretches are given by the scaled singular values. Decompose
    //    the covariance matrix into its principal axes and their lengths
    timer.start("Decompose covariance matrices");
    manager.foreach([&](LeafNodeT& leafnode, size_t)
    {
        AttributeWriteHandle<Vec3f, NullCodec> stretchHandle(leafnode.attributeArray(strIdx));
        AttributeWriteHandle<math::Mat3s, NullCodec> rotHandle(leafnode.attributeArray(rotIdx));
        GroupHandle ellipsesGroupHandle(leafnode.groupHandle(ellipsesIdx));

        // we don't use a group filter here since we need to set the rotation
        // matrix for excluded points

        for (Index idx = 0; idx < stretchHandle.size(); ++idx) {
            if (!ellipsesGroupHandle.get(idx)) {
                rotHandle.set(idx, math::Mat3s::identity());
                continue;
            }

            // get singular values of the covariance matrix
            math::Mat3s u;
            Vec3s sigma;
            decomposeSymmetricMatrix(rotHandle.get(idx), u, sigma);

            // fix sigma values, the principal lengths
            auto maxs = sigma[0] * settings.allowedAnisotropyRatio;
            sigma[1] = std::max(sigma[1], maxs);
            sigma[2] = std::max(sigma[2], maxs);

            // should only happen if all neighbours are coincident
            // @note  The specific tolerance here relates to the normalization
            //   of the stetch values in step (7) e.g. s*(1.0/cbrt(s.product())).
            //   math::Tolerance<float>::value is 1e-7f, but we have a bit more
            //   flexibility here, we can deal with smaller values, common for
            //   the case where a point only has one neighbour
            // @todo  have to manually construct the tolerance because
            //   math::Tolerance<Vec3f> resolves to 0.0. fix this in the math lib
            if (math::isApproxZero(sigma, Vec3f(1e-11f))) {
                sigma = Vec3f::ones();
            }

            stretchHandle.set(idx, sigma);
            rotHandle.set(idx, u);
        }
    });
    timer.stop();

    // 7) normalise the principal lengths such that the transformation they
    //    describe 'preserves volume' thus becoming the stretch of the ellipsoids

    // Calculates the average volume change that would occur if applying the
    // transformations defined by the calculate principal axes and their
    // lengths using the determinant of the stretch component of the
    // transformation (given by the sum of the diagonal values) and uses this
    // value to normalise these lengths such that they are relative to the
    // identity. This ensures that the transformation preserves volume.
    timer.start("Normalise the principal lengths");
    manager.foreach([&](LeafNodeT& leafnode, size_t)
    {
        points::GroupFilter filter(ellipsesIdx);
        filter.reset(leafnode);
        AttributeWriteHandle<Vec3f, NullCodec> stretchHandle(leafnode.attributeArray(strIdx));

        for (Index i = 0; i < stretchHandle.size(); ++i)
        {
            if (!filter.valid(&i)) continue;
            const Vec3f stretch = stretchHandle.get(i);
            assert(stretch != Vec3f::zero());
            const float stretchScale = 1.0f / std::cbrt(stretch.product());
            stretchHandle.set(i, stretchScale * stretch);
        }
    });

    timer.stop();
    if (util::wasInterrupted(interrupt)) return;

    // 8) do laplacian position smoothing here as we have the weights already
    ///   calculated (calculates the smoothed kernel origins as described in
    ///   the paper). averagePositions value biases the smoothed positions
    ///   towards the weighted mean positions. 1.0  will use the weighted means
    ///   while 0.0 will use the original world-space positions
    timer.start("Laplacian smooth positions");
    if (settings.averagePositions > 0.0f)
    {
        manager.foreach([&](LeafNodeT& leafnode, size_t) {
            AttributeWriteHandle<Vec3d, NullCodec> Pws(leafnode.attributeArray(pwsIdx));
            AttributeHandle<WeightedPositionSumT, NullCodec> weightedPosSumHandle(leafnode.constAttributeArray(posSumIndex));

            for (Index i = 0; i < Pws.size(); ++i) {
                const Vec3d smoothedPosition = (1.0f - settings.averagePositions) *
                    Pws.get(i) + settings.averagePositions * weightedPosSumHandle.get(i);
                Pws.set(i, smoothedPosition);
            }
        });
    }

    timer.stop();

    // Remove temporary attributes
    points::dropAttributes(tree, temps);
}

}
}
}

#endif // OPENVDB_POINTS_POINT_PRINCIPAL_COMPONENT_ANALYSIS_IMPL_HAS_BEEN_INCLUDED
