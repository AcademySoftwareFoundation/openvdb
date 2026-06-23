// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file    Blend.h
///
/// @author  Andre Pradhana
///
/// @brief   Define methods to blend two level-sets together. One such approach
///          is by carving an excess fillet so that the resulting blended
///          level-sets can appear to be smoother than a regular union.
///
/// @details The algorithm used in the function unionFillet is based on
///          a 2007 SIGGRAPH talk titled "Levelsets in production: Spider-man 3"
///          by Allen et al. Paper is available here:
///          https://dl.acm.org/doi/10.1145/1278780.1278815

#ifndef OPENVDB_TOOLS_BLEND_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_BLEND_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h> // for isExactlyEqual()
#include <openvdb/openvdb.h>
#include <openvdb/tools/FastSweeping.h> // for maskSdf
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/util/NullInterrupter.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief  Threaded VDB union with fillet that produces a new grid or tree from
///         immutable inputs.
///
/// @param lhs Level-set grid to be combined with a second input.
///
/// @param rhs Level-set grid to be combined with the first input.
///
/// @param mask Optional float grid that controls the strength of the blending.
///
/// @param alpha Controls the blend-radius.
///
/// @param beta Controls the exponent used to control the strength of the blend.
///
/// @param gamma Controls the strength of the blend by providing a multiplier component.
///
/// @param supportDilation Optional number of voxels by which to dilate the
///        local blend support region before extending both inputs into that
///        region. A value of zero preserves the default behavior.
///
/// @return The filleted union of the @lhs and @rhs level set inputs.
///
/// @throw  If the transforms of @lsh, @rhs, and @mask do not match.
template<typename GridT,
         typename MaskT = typename GridT::template ValueConverter<float>::Type>
typename GridT::Ptr
unionFillet(const GridT& lhs,
     const GridT& rhs,
     typename MaskT::ConstPtr mask,
     typename GridT::ValueType alpha,
     typename GridT::ValueType beta,
     typename GridT::ValueType gamma,
     int supportDilation = 0);

//
// Main class to handle UnionWithFillet
//
template<typename GridT,
         typename MaskT = typename GridT::template ValueConverter<float>::Type>
struct UnionWithFillet {
    using TreeT = typename GridT::TreeType;
    using ValueType = typename TreeT::ValueType;
    using TreePtrType = typename TreeT::Ptr;
    using LeafNodeType = typename TreeT::LeafNodeType;
    using NodeMaskType = typename LeafNodeType::NodeMaskType;
    using RootNodeType = typename TreeT::RootNodeType;
    using NodeChainType = typename RootNodeType::NodeChainType;
    using InternalNodeType = typename NodeChainType::template Get<1>;
    using MaskTreeType = typename MaskT::TreeType;
    using MaskValueType = typename MaskT::ValueType;

    UnionWithFillet(const GridT& lhsGrid,
                    const GridT& rhsGrid,
                    typename MaskT::ConstPtr mask,
                    const ValueType& bandwidth,
                    const ValueType& exponent,
                    const ValueType& multiplier)
        : mLhsGrid(&lhsGrid)
        , mRhsGrid(&rhsGrid)
        , mLhsTree(&(lhsGrid.tree()))
        , mRhsTree(&(rhsGrid.tree()))
        , mBandwidth(bandwidth)
        , mExponent(exponent)
        , mMultiplier(multiplier)
    {
        mMaskTree = mask ? mask->treePtr() : nullptr;
        static_assert(std::is_floating_point<typename TreeT::ValueType>::value,
                      "assert in UnionFillet Constructor: "
                      "level set grids must have scalar/floating-point value types.");
    }

    /// @brief Perform blending of two level-sets.
    typename GridT::Ptr blend();

    TreePtrType& segment() { return mSegment; }

private:
    struct FilletParms {
        float mAlpha; // Bandwidth Falloff
        float mBeta; // Exponent
        float mGamma; // Amplitude/Multiplier
    };

    struct BuildPrimarySegment;
    struct BuildSecondarySegment;
    struct BuildFusedSegment;

    TreePtrType         mSegment;
    GridT const * const mLhsGrid;
    GridT const * const mRhsGrid;
    TreeT const * const mLhsTree;
    TreeT const * const mRhsTree;
    typename MaskT::TreeType::ConstPtr mMaskTree = nullptr;
    ValueType mBandwidth;
    ValueType mExponent;
    ValueType mMultiplier;
};

template<typename GridT>
MaskGrid::Ptr
createBlendSupportMask(const GridT& lhs,
                       const GridT& rhs,
                       typename GridT::ValueType alpha,
                       int supportDilation)
{
    MaskGrid::Ptr supportMask = MaskGrid::create();
    supportMask->setTransform(lhs.transform().copy());

    const typename GridT::ValueType supportRadius =
        alpha + typename GridT::ValueType(supportDilation * lhs.voxelSize()[0]);

    typename GridT::ConstAccessor lhsAcc = lhs.getConstAccessor();
    typename GridT::ConstAccessor rhsAcc = rhs.getConstAccessor();
    MaskGrid::Accessor supportAcc = supportMask->getAccessor();

    for (typename GridT::ValueOnCIter iter = lhs.cbeginValueOn(); iter; ++iter) {
        const Coord& ijk = iter.getCoord();
        // only prepare extra SDF samples where both shapes could participate in the fillet
        if (iter.getValue() < supportRadius && rhsAcc.getValue(ijk) < supportRadius) {
            supportAcc.setValueOn(ijk);
        }
    }

    for (typename GridT::ValueOnCIter iter = rhs.cbeginValueOn(); iter; ++iter) {
        const Coord& ijk = iter.getCoord();
        if (lhsAcc.getValue(ijk) < supportRadius && iter.getValue() < supportRadius) {
            supportAcc.setValueOn(ijk);
        }
    }

    if (supportMask->activeVoxelCount() > 0 && supportDilation > 0) {
        tools::dilateActiveValues(
            supportMask->tree(), supportDilation, tools::NN_FACE_EDGE_VERTEX, tools::IGNORE_TILES);
    }

    return supportMask;
}

/// @cond OPENVDB_DOCS_INTERNAL
/// @brief Go through the lhs nodes (both internal and leaf nodes).
// TODO: should just be only for float
template<typename TreeT, typename MaskT>
struct UnionWithFillet<TreeT, MaskT>::BuildPrimarySegment {
    using MaskTreeType = typename MaskT::TreeType;
    using ValueType = typename TreeT::ValueType;
    using TreePtrType = typename TreeT::Ptr;
    using LeafNodeType = typename TreeT::LeafNodeType;
    using NodeMaskType = typename LeafNodeType::NodeMaskType;
    using RootNodeType = typename TreeT::RootNodeType;
    using NodeChainType = typename RootNodeType::NodeChainType;
    using InternalNodeType = typename NodeChainType::template Get<1>;

    BuildPrimarySegment(TreeT const * const lhs,
                        TreeT const * const rhs,
                        typename MaskT::TreeType::ConstPtr mask,
                        FilletParms parms)
                        : mSegment(new TreeT(lhs->background()))
                        , mLhsTree(lhs)
                        , mRhsTree(rhs)
                        , mMaskTree(mask)
                        , mParms(parms) { }

    void operator()() const
    {
        std::vector<const LeafNodeType*> leafNodes;

        {
            std::vector<const InternalNodeType*> internalNodes;
            mLhsTree->getNodes(internalNodes);

            ProcessInternalNodes op(internalNodes, *mRhsTree, *mSegment, leafNodes, mParms);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, internalNodes.size()), op);
        }
        if (mMaskTree) {
            ProcessLeafNodesMask op(leafNodes, *mRhsTree, mMaskTree, *mSegment, mParms);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leafNodes.size()), op);
        } else {
            ProcessLeafNodes op(leafNodes, *mRhsTree, *mSegment, mParms);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leafNodes.size()), op);
        }
    }

    TreePtrType& segment() { return mSegment; }

private:

    struct ProcessInternalNodes {

        ProcessInternalNodes(std::vector<const InternalNodeType*>& lhsNodes,
            const TreeT& rhsTree, TreeT& outputTree,
            std::vector<const LeafNodeType*>& outputLeafNodes, FilletParms parms)
            : mLhsNodes(lhsNodes)
            , mRhsTree(&rhsTree)
            , mLocalTree(mRhsTree->background())
            , mOutputTree(&outputTree)
            , mLocalLeafNodes()
            , mOutputLeafNodes(&outputLeafNodes)
            , mParms(parms)
        { }

        ProcessInternalNodes(ProcessInternalNodes& other, tbb::split)
            : mLhsNodes(other.mLhsNodes)
            , mRhsTree(other.mRhsTree)
            , mLocalTree(mRhsTree->background())
            , mOutputTree(&mLocalTree)
            , mLocalLeafNodes()
            , mOutputLeafNodes(&mLocalLeafNodes)
            , mParms(other.mParms)
        { }

        void join(ProcessInternalNodes& other)
        {
            mOutputTree->merge(*other.mOutputTree);
            mOutputLeafNodes->insert(mOutputLeafNodes->end(),
                other.mOutputLeafNodes->begin(), other.mOutputLeafNodes->end());
        }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            openvdb::tree::ValueAccessor<const TreeT> rhsAcc(*mRhsTree);
            openvdb::tree::ValueAccessor<TreeT>       outputAcc(*mOutputTree);

            std::vector<const LeafNodeType*> tmpLeafNodes;

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

                const InternalNodeType& lhsNode = *mLhsNodes[n];
                const openvdb::math::Coord& ijk = lhsNode.origin();
                const InternalNodeType * rhsNode =
                    rhsAcc.template probeConstNode<InternalNodeType>(ijk);

                if (rhsNode) {
                    lhsNode.getNodes(*mOutputLeafNodes);
                } else {
                    if (!(rhsAcc.getValue(ijk) < ValueType(0))) {
                        tmpLeafNodes.clear();
                        lhsNode.getNodes(tmpLeafNodes);
                        for (const LeafNodeType* leaf : tmpLeafNodes) {
                            outputAcc.addLeaf(new LeafNodeType(*leaf));
                        }
                    }
                }
            } //  end range loop
        }

        const std::vector<const InternalNodeType*>& mLhsNodes;
        TreeT                    const * const      mRhsTree;
        TreeT                                       mLocalTree;
        TreeT                          * const      mOutputTree;

        std::vector<const LeafNodeType*>         mLocalLeafNodes;
        std::vector<const LeafNodeType*> * const mOutputLeafNodes;
        FilletParms                              mParms;
    }; // struct ProcessInternalNodes

    struct ProcessLeafNodes {

        ProcessLeafNodes(std::vector<const LeafNodeType*>& lhsNodes,
            const TreeT& rhsTree, TreeT& output, FilletParms parms)
            : mLhsNodes(lhsNodes)
            , mRhsTree(&rhsTree)
            , mLocalTree(mRhsTree->background())
            , mOutputTree(&output)
            , mParms(parms)
        { }

        ProcessLeafNodes(ProcessLeafNodes& other, tbb::split)
            : mLhsNodes(other.mLhsNodes)
            , mRhsTree(other.mRhsTree)
            , mLocalTree(mRhsTree->background())
            , mOutputTree(&mLocalTree)
            , mParms(other.mParms)
        { }

        void join(ProcessLeafNodes& rhs) { mOutputTree->merge(*rhs.mOutputTree); }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            openvdb::tree::ValueAccessor<const TreeT> rhsAcc(*mRhsTree);
            openvdb::tree::ValueAccessor<TreeT>       outputAcc(*mOutputTree);

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

                const LeafNodeType& lhsNode = *mLhsNodes[n];
                const openvdb::math::Coord& ijk = lhsNode.origin();

                const LeafNodeType* rhsNodePt = rhsAcc.probeConstLeaf(ijk);

                if (rhsNodePt) { // combine overlapping nodes

                    LeafNodeType* outputNode = outputAcc.touchLeaf(ijk);
                    ValueType * outputData = outputNode->buffer().data();
                    NodeMaskType& outputMask = outputNode->getValueMask();

                    const ValueType * lhsData = lhsNode.buffer().data();
                    const NodeMaskType& lhsMask = lhsNode.getValueMask();

                    const ValueType * rhsData = rhsNodePt->buffer().data();
                    const NodeMaskType& rhsMask = rhsNodePt->getValueMask();

                    const float alpha = mParms.mAlpha;
                    const float beta = mParms.mBeta;
                    const float gamma = mParms.mGamma;

                    for (openvdb::Index pos = 0; pos < LeafNodeType::SIZE; ++pos) {
                        const float A = lhsData[pos];
                        const float B = rhsData[pos];
                        const float m = openvdb::math::Clamp((alpha - A) / alpha, 0.f, 1.f) *
                                        openvdb::math::Clamp((alpha - B) / alpha, 0.f, 1.f);
                        // Only use offset if both lhs and rhs are on, else defaults to regular union
                        const bool hasValidFilletSamples = lhsMask.isOn(pos) && rhsMask.isOn(pos);
                        const float offset = hasValidFilletSamples ?
                            openvdb::math::Pow(m, beta) * gamma : 0.0f;
                        const bool isAMin = A < B;
                        // - sign here. Otherwise, we'll get empty space between geos
                        outputData[pos] = isAMin ? A - offset : B - offset;
                        outputMask.set(pos, isAMin ? lhsMask.isOn(pos) : rhsMask.isOn(pos));
                    }
                } else {
                    if (!(rhsAcc.getValue(ijk) < ValueType(0.0))) {
                        outputAcc.addLeaf(new LeafNodeType(lhsNode));
                    }
                }
            } //  end range loop
        }

        const std::vector<const LeafNodeType*>& mLhsNodes;
        TreeT                const * const      mRhsTree;
        TreeT                                   mLocalTree;
        TreeT                      * const      mOutputTree;
        FilletParms                             mParms;
    }; // struct ProcessLeafNodes

    struct ProcessLeafNodesMask {
        using MaskTreeType = typename MaskT::TreeType;
        using MaskLeafNodeType = typename MaskT::TreeType::LeafNodeType;
        using MaskNodeMaskType = typename MaskLeafNodeType::NodeMaskType;
        using MaskValueType = typename MaskT::ValueType;

        ProcessLeafNodesMask(std::vector<const LeafNodeType*>& lhsNodes,
            const TreeT& rhsTree, typename MaskT::TreeType::ConstPtr maskTree, TreeT& output, FilletParms parms)
            : mLhsNodes(lhsNodes)
            , mRhsTree(&rhsTree)
            , mLocalTree(mRhsTree->background())
            , mMaskTree(maskTree)
            , mOutputTree(&output)
            , mParms(parms)
        { }

        ProcessLeafNodesMask(ProcessLeafNodesMask& other, tbb::split)
            : mLhsNodes(other.mLhsNodes)
            , mRhsTree(other.mRhsTree)
            , mLocalTree(mRhsTree->background())
            , mMaskTree(other.mMaskTree)
            , mOutputTree(&mLocalTree)
            , mParms(other.mParms)
        { }

        void join(ProcessLeafNodesMask& rhs) { mOutputTree->merge(*rhs.mOutputTree); }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            openvdb::tree::ValueAccessor<const TreeT> rhsAcc(*mRhsTree);
            openvdb::tree::ValueAccessor<TreeT>       outputAcc(*mOutputTree);
            openvdb::tree::ValueAccessor<const MaskTreeType> maskAcc(*mMaskTree);
            const float maskBackground = mMaskTree->background();

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

                const LeafNodeType& lhsNode = *mLhsNodes[n];
                const openvdb::math::Coord& ijk = lhsNode.origin();

                const LeafNodeType* rhsNodePt = rhsAcc.probeConstLeaf(ijk);
                const MaskLeafNodeType* maskNodePt = maskAcc.probeConstLeaf(ijk);

                if (rhsNodePt) { // combine overlapping nodes
                    LeafNodeType* outputNode = outputAcc.touchLeaf(ijk);
                    ValueType * outputData = outputNode->buffer().data();
                    NodeMaskType& outputMask = outputNode->getValueMask();

                    const ValueType* lhsData = lhsNode.buffer().data();
                    const NodeMaskType& lhsMask = lhsNode.getValueMask();

                    const ValueType* rhsData = rhsNodePt->buffer().data();
                    const NodeMaskType& rhsMask = rhsNodePt->getValueMask();

                    const float alpha = mParms.mAlpha;
                    const float beta = mParms.mBeta;
                    const float gamma = mParms.mGamma;

                    MaskValueType const * maskData = nullptr;
                    if (maskNodePt) maskData = maskNodePt->buffer().data();

                    for (openvdb::Index pos = 0; pos < LeafNodeType::SIZE; ++pos) {
                        const float A = lhsData[pos];
                        const float B = rhsData[pos];
                        const float m = openvdb::math::Clamp((alpha - A) / alpha, 0.f, 1.f) *
                                        openvdb::math::Clamp((alpha - B) / alpha, 0.f, 1.f);
                        // Only use offset if both lhs and rhs are on, else defaults to regular union
                        const bool hasValidFilletSamples = lhsMask.isOn(pos) && rhsMask.isOn(pos);
                        const float offset = hasValidFilletSamples ?
                            openvdb::math::Pow(m, beta) * gamma : 0.0f;
                        const bool isAMin = A < B;
                        // multiply by mask value if it exists
                        const float multiplier = maskData ? maskData[pos] : maskBackground;
                        // - sign here. Otherwise, we'll get empty space between geos
                        outputData[pos] = isAMin ? (A - offset * multiplier) : (B - offset * multiplier);
                        outputMask.set(pos, isAMin ? lhsMask.isOn(pos) : rhsMask.isOn(pos));
                    }
                } else {
                    if (!(rhsAcc.getValue(ijk) < ValueType(0.0))) {
                        outputAcc.addLeaf(new LeafNodeType(lhsNode));
                    }
                }
            } //  end range loop
        }

        const std::vector<const LeafNodeType*>& mLhsNodes;
        TreeT                const * const      mRhsTree;
        TreeT                                   mLocalTree;
        typename MaskT::TreeType::ConstPtr      mMaskTree;
        TreeT                      * const      mOutputTree;
        FilletParms                             mParms;
    }; // struct ProcessLeafNodesMask


private:
    TreePtrType         mSegment;
    TreeT const * const mLhsTree;
    TreeT const * const mRhsTree;
    typename MaskT::TreeType::ConstPtr mMaskTree;
    FilletParms         mParms;
};


template<typename TreeT, typename MaskT>
struct UnionWithFillet<TreeT, MaskT>::BuildSecondarySegment {
    using MaskTreeType = typename MaskT::TreeType;
    using ValueType = typename TreeT::ValueType;
    using TreePtrType = typename TreeT::Ptr;
    using LeafNodeType = typename TreeT::LeafNodeType;
    using NodeMaskType = typename LeafNodeType::NodeMaskType;
    using RootNodeType = typename TreeT::RootNodeType;
    using NodeChainType = typename RootNodeType::NodeChainType;
    using InternalNodeType = typename NodeChainType::template Get<1>;
    using MaskValueType = typename MaskTreeType::ValueType;

    BuildSecondarySegment(TreeT const * const lhs,
                          TreeT const * const rhs,
                          FilletParms parms)
                          : mSegment(new TreeT(lhs->background()))
                          , mLhsTree(lhs)
                          , mRhsTree(rhs)
                          , mParms(parms) { }


    void operator()() const
    {
        std::vector<const LeafNodeType*> leafNodes;

        {
            std::vector<const InternalNodeType*> internalNodes;
            mRhsTree->getNodes(internalNodes);

            ProcessInternalNodes op(internalNodes, *mLhsTree, *mSegment, leafNodes, mParms);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, internalNodes.size()), op);
        }

        ProcessLeafNodes op(leafNodes, *mLhsTree, *mSegment, mParms);
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leafNodes.size()), op);
    }

    TreePtrType& segment() { return mSegment; }

private:

    struct ProcessInternalNodes {

        ProcessInternalNodes(std::vector<const InternalNodeType*>& rhsNodes,
            const TreeT& lhsTree, TreeT& outputTree,
            std::vector<const LeafNodeType*>& outputLeafNodes,
            FilletParms parms)
            : mRhsNodes(rhsNodes)
            , mLhsTree(&lhsTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&outputTree)
            , mLocalLeafNodes()
            , mOutputLeafNodes(&outputLeafNodes)
            , mParms(parms)
        { }

        ProcessInternalNodes(ProcessInternalNodes& other, tbb::split)
            : mRhsNodes(other.mRhsNodes)
            , mLhsTree(other.mLhsTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&mLocalTree)
            , mLocalLeafNodes()
            , mOutputLeafNodes(&mLocalLeafNodes)
            , mParms(other.mParms)
        { }

        void join(ProcessInternalNodes& other)
        {
            mOutputTree->merge(*other.mOutputTree);
            mOutputLeafNodes->insert(mOutputLeafNodes->end(),
                other.mOutputLeafNodes->begin(), other.mOutputLeafNodes->end());
        }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            openvdb::tree::ValueAccessor<const TreeT> lhsAcc(*mLhsTree);
            openvdb::tree::ValueAccessor<TreeT>       outputAcc(*mOutputTree);

            std::vector<const LeafNodeType*> tmpLeafNodes;

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

                const InternalNodeType& rhsNode = *mRhsNodes[n];
                const openvdb::math::Coord& ijk = rhsNode.origin();
                const InternalNodeType * lhsNode =
                    lhsAcc.template probeConstNode<InternalNodeType>(ijk);

                if (lhsNode) {
                   rhsNode.getNodes(*mOutputLeafNodes);
                } else {
                    if (!(lhsAcc.getValue(ijk) < ValueType(0))) {
                        tmpLeafNodes.clear();
                        rhsNode.getNodes(tmpLeafNodes);
                        for (const LeafNodeType* leaf : tmpLeafNodes) {
                            outputAcc.addLeaf(new LeafNodeType(*leaf));
                        }
                    }
                }
            } //  end range loop
        }

        const std::vector<const InternalNodeType*>& mRhsNodes;
        TreeT                    const * const      mLhsTree;
        TreeT                                       mLocalTree;
        TreeT                          * const      mOutputTree;

        std::vector<const LeafNodeType*>         mLocalLeafNodes;
        std::vector<const LeafNodeType*> * const mOutputLeafNodes;
        FilletParms                              mParms;
    }; // struct ProcessInternalNodes

    struct ProcessLeafNodes {

        ProcessLeafNodes(std::vector<const LeafNodeType*>& rhsNodes,
            const TreeT& lhsTree, TreeT& output,
            FilletParms parms)
            : mRhsNodes(rhsNodes)
            , mLhsTree(&lhsTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&output)
            , mParms(parms)
        { }

        ProcessLeafNodes(ProcessLeafNodes& rhs, tbb::split)
            : mRhsNodes(rhs.mRhsNodes)
            , mLhsTree(rhs.mLhsTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&mLocalTree)
            , mParms(rhs.mParms)
        { }

        void join(ProcessLeafNodes& rhs) { mOutputTree->merge(*rhs.mOutputTree); }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            openvdb::tree::ValueAccessor<const TreeT> lhsAcc(*mLhsTree);
            openvdb::tree::ValueAccessor<TreeT>       outputAcc(*mOutputTree);

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

                const LeafNodeType& rhsNode = *mRhsNodes[n];
                const openvdb::math::Coord& ijk = rhsNode.origin();

                const LeafNodeType* lhsNode = lhsAcc.probeConstLeaf(ijk);

                if (!lhsNode) {
                    if (!(lhsAcc.getValue(ijk) < ValueType(0))) {
                        outputAcc.addLeaf(new LeafNodeType(rhsNode));
                    }
                }
            } //  end range loop
        }

        const std::vector<const LeafNodeType*>& mRhsNodes;
        TreeT                const * const      mLhsTree;
        TreeT                                   mLocalTree;
        TreeT                      * const      mOutputTree;
        FilletParms                             mParms;
    }; // struct ProcessLeafNodes


private:
    TreePtrType         mSegment;
    TreeT const * const mLhsTree;
    TreeT const * const mRhsTree;
    FilletParms         mParms;
};

template<typename TreeT, typename MaskT>
struct UnionWithFillet<TreeT, MaskT>::BuildFusedSegment {
    using MaskTreeType = typename MaskT::TreeType;
    using ValueType = typename TreeT::ValueType;
    using TreePtrType = typename TreeT::Ptr;
    using LeafNodeType = typename TreeT::LeafNodeType;
    using NodeMaskType = typename LeafNodeType::NodeMaskType;
    using RootNodeType = typename TreeT::RootNodeType;
    using NodeChainType = typename RootNodeType::NodeChainType;
    using InternalNodeType = typename NodeChainType::template Get<1>;
    using MaskLeafNodeType = typename MaskT::TreeType::LeafNodeType;
    using MaskValueType = typename MaskT::ValueType;

    BuildFusedSegment(TreeT const * const lhs,
                      TreeT const * const rhs,
                      typename MaskT::TreeType::ConstPtr mask,
                      FilletParms parms)
        : mSegment(new TreeT(lhs->background()))
        , mLhsTree(lhs)
        , mRhsTree(rhs)
        , mMaskTree(mask)
        , mParms(parms)
    {
    }

    void operator()() const
    {
        // Gather both lhs and rhs internal-node lists up front. One combined
        // internal-node reduce walks both sides. Internal nodes that do not
        // overlap the opposite tree and are outside the opposite SDF are copied
        // directly into the output tree. Overlapping internal nodes are expanded
        // into two leaf work queues: primaryLeafNodes contains lhs leaves that
        // need overlap/primary handling, and secondaryLeafNodes contains rhs
        // leaves that need rhs-only handling.
        //
        // One combined leaf reduce processes both queues into a single output
        // tree. There is no separate primary tree, secondary tree, or final
        // primary/secondary merge.
        std::vector<const InternalNodeType*> lhsInternalNodes;
        std::vector<const InternalNodeType*> rhsInternalNodes;
        std::vector<const LeafNodeType*> primaryLeafNodes;
        std::vector<const LeafNodeType*> secondaryLeafNodes;

        mLhsTree->getNodes(lhsInternalNodes);
        mRhsTree->getNodes(rhsInternalNodes);

        ProcessInternalNodes internalOp(lhsInternalNodes, rhsInternalNodes,
            *mLhsTree, *mRhsTree, *mSegment, primaryLeafNodes, secondaryLeafNodes, mParms);
        tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, lhsInternalNodes.size() + rhsInternalNodes.size()),
            internalOp);

        if (mMaskTree) {
            ProcessLeafNodesMask leafOp(primaryLeafNodes, secondaryLeafNodes,
                *mLhsTree, *mRhsTree, mMaskTree, *mSegment, mParms);
            tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, primaryLeafNodes.size() + secondaryLeafNodes.size()),
                leafOp);
        } else {
            ProcessLeafNodes leafOp(primaryLeafNodes, secondaryLeafNodes,
                *mLhsTree, *mRhsTree, *mSegment, mParms);
            tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, primaryLeafNodes.size() + secondaryLeafNodes.size()),
                leafOp);
        }
    }

    TreePtrType& segment() { return mSegment; }

private:
    struct ProcessInternalNodes {
        ProcessInternalNodes(
            std::vector<const InternalNodeType*>& lhsNodes,
            std::vector<const InternalNodeType*>& rhsNodes,
            const TreeT& lhsTree,
            const TreeT& rhsTree,
            TreeT& outputTree,
            std::vector<const LeafNodeType*>& primaryLeafNodes,
            std::vector<const LeafNodeType*>& secondaryLeafNodes,
            FilletParms parms)
            : mLhsNodes(lhsNodes)
            , mRhsNodes(rhsNodes)
            , mLhsTree(&lhsTree)
            , mRhsTree(&rhsTree)
            , mLocalTree(lhsTree.background())
            , mOutputTree(&outputTree)
            , mPrimaryLocalLeafNodes()
            , mSecondaryLocalLeafNodes()
            , mPrimaryLeafNodes(&primaryLeafNodes)
            , mSecondaryLeafNodes(&secondaryLeafNodes)
            , mParms(parms)
        {
        }

        ProcessInternalNodes(ProcessInternalNodes& other, tbb::split)
            : mLhsNodes(other.mLhsNodes)
            , mRhsNodes(other.mRhsNodes)
            , mLhsTree(other.mLhsTree)
            , mRhsTree(other.mRhsTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&mLocalTree)
            , mPrimaryLocalLeafNodes()
            , mSecondaryLocalLeafNodes()
            , mPrimaryLeafNodes(&mPrimaryLocalLeafNodes)
            , mSecondaryLeafNodes(&mSecondaryLocalLeafNodes)
            , mParms(other.mParms)
        {
        }

        void join(ProcessInternalNodes& other)
        {
            mOutputTree->merge(*other.mOutputTree);
            mPrimaryLeafNodes->insert(mPrimaryLeafNodes->end(),
                other.mPrimaryLeafNodes->begin(), other.mPrimaryLeafNodes->end());
            mSecondaryLeafNodes->insert(mSecondaryLeafNodes->end(),
                other.mSecondaryLeafNodes->begin(), other.mSecondaryLeafNodes->end());
        }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            openvdb::tree::ValueAccessor<const TreeT> lhsAcc(*mLhsTree);
            openvdb::tree::ValueAccessor<const TreeT> rhsAcc(*mRhsTree);
            openvdb::tree::ValueAccessor<TreeT> outputAcc(*mOutputTree);

            std::vector<const LeafNodeType*> tmpLeafNodes;
            const size_t lhsSize = mLhsNodes.size();

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
                if (n < lhsSize) {
                    const InternalNodeType& lhsNode = *mLhsNodes[n];
                    const openvdb::math::Coord& ijk = lhsNode.origin();
                    const InternalNodeType* rhsNode =
                        rhsAcc.template probeConstNode<InternalNodeType>(ijk);

                    if (rhsNode) {
                        lhsNode.getNodes(*mPrimaryLeafNodes);
                    } else if (!(rhsAcc.getValue(ijk) < ValueType(0))) {
                        tmpLeafNodes.clear();
                        lhsNode.getNodes(tmpLeafNodes);
                        for (const LeafNodeType* leaf : tmpLeafNodes) {
                            outputAcc.addLeaf(new LeafNodeType(*leaf));
                        }
                    }
                } else {
                    const InternalNodeType& rhsNode = *mRhsNodes[n - lhsSize];
                    const openvdb::math::Coord& ijk = rhsNode.origin();
                    const InternalNodeType* lhsNode =
                        lhsAcc.template probeConstNode<InternalNodeType>(ijk);

                    if (lhsNode) {
                        rhsNode.getNodes(*mSecondaryLeafNodes);
                    } else if (!(lhsAcc.getValue(ijk) < ValueType(0))) {
                        tmpLeafNodes.clear();
                        rhsNode.getNodes(tmpLeafNodes);
                        for (const LeafNodeType* leaf : tmpLeafNodes) {
                            outputAcc.addLeaf(new LeafNodeType(*leaf));
                        }
                    }
                }
            }
        }

        const std::vector<const InternalNodeType*>& mLhsNodes;
        const std::vector<const InternalNodeType*>& mRhsNodes;
        TreeT const * const mLhsTree;
        TreeT const * const mRhsTree;
        TreeT mLocalTree;
        TreeT * const mOutputTree;
        std::vector<const LeafNodeType*> mPrimaryLocalLeafNodes;
        std::vector<const LeafNodeType*> mSecondaryLocalLeafNodes;
        std::vector<const LeafNodeType*> * const mPrimaryLeafNodes;
        std::vector<const LeafNodeType*> * const mSecondaryLeafNodes;
        FilletParms mParms;
    };

    struct ProcessLeafNodes {
        ProcessLeafNodes(
            std::vector<const LeafNodeType*>& primaryLeafNodes,
            std::vector<const LeafNodeType*>& secondaryLeafNodes,
            const TreeT& lhsTree,
            const TreeT& rhsTree,
            TreeT& outputTree,
            FilletParms parms)
            : mPrimaryLeafNodes(primaryLeafNodes)
            , mSecondaryLeafNodes(secondaryLeafNodes)
            , mLhsTree(&lhsTree)
            , mRhsTree(&rhsTree)
            , mLocalTree(lhsTree.background())
            , mOutputTree(&outputTree)
            , mParms(parms)
        {
        }

        ProcessLeafNodes(ProcessLeafNodes& other, tbb::split)
            : mPrimaryLeafNodes(other.mPrimaryLeafNodes)
            , mSecondaryLeafNodes(other.mSecondaryLeafNodes)
            , mLhsTree(other.mLhsTree)
            , mRhsTree(other.mRhsTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&mLocalTree)
            , mParms(other.mParms)
        {
        }

        void join(ProcessLeafNodes& rhs) { mOutputTree->merge(*rhs.mOutputTree); }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            openvdb::tree::ValueAccessor<const TreeT> lhsAcc(*mLhsTree);
            openvdb::tree::ValueAccessor<const TreeT> rhsAcc(*mRhsTree);
            openvdb::tree::ValueAccessor<TreeT> outputAcc(*mOutputTree);

            const float alpha = mParms.mAlpha;
            const float beta = mParms.mBeta;
            const float gamma = mParms.mGamma;
            const size_t primarySize = mPrimaryLeafNodes.size();

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
                if (n < primarySize) {
                    const LeafNodeType& lhsNode = *mPrimaryLeafNodes[n];
                    const openvdb::math::Coord& ijk = lhsNode.origin();
                    const LeafNodeType* rhsNode = rhsAcc.probeConstLeaf(ijk);

                    if (rhsNode) {
                        LeafNodeType* outputNode = outputAcc.touchLeaf(ijk);
                        ValueType* outputData = outputNode->buffer().data();
                        NodeMaskType& outputMask = outputNode->getValueMask();

                        const ValueType* lhsData = lhsNode.buffer().data();
                        const NodeMaskType& lhsMask = lhsNode.getValueMask();
                        const ValueType* rhsData = rhsNode->buffer().data();
                        const NodeMaskType& rhsMask = rhsNode->getValueMask();

                        for (openvdb::Index pos = 0; pos < LeafNodeType::SIZE; ++pos) {
                            const float A = lhsData[pos];
                            const float B = rhsData[pos];
                            const float m = openvdb::math::Clamp((alpha - A) / alpha, 0.f, 1.f) *
                                            openvdb::math::Clamp((alpha - B) / alpha, 0.f, 1.f);
                            const bool hasValidFilletSamples = lhsMask.isOn(pos) && rhsMask.isOn(pos);
                            const float offset = hasValidFilletSamples ?
                                openvdb::math::Pow(m, beta) * gamma : 0.0f;
                            const bool isAMin = A < B;
                            outputData[pos] = isAMin ? A - offset : B - offset;
                            outputMask.set(pos, isAMin ? lhsMask.isOn(pos) : rhsMask.isOn(pos));
                        }
                    } else if (!(rhsAcc.getValue(ijk) < ValueType(0.0))) {
                        outputAcc.addLeaf(new LeafNodeType(lhsNode));
                    }
                } else {
                    const LeafNodeType& rhsNode = *mSecondaryLeafNodes[n - primarySize];
                    const openvdb::math::Coord& ijk = rhsNode.origin();
                    const LeafNodeType* lhsNode = lhsAcc.probeConstLeaf(ijk);

                    if (!lhsNode && !(lhsAcc.getValue(ijk) < ValueType(0))) {
                        outputAcc.addLeaf(new LeafNodeType(rhsNode));
                    }
                }
            }
        }

        const std::vector<const LeafNodeType*>& mPrimaryLeafNodes;
        const std::vector<const LeafNodeType*>& mSecondaryLeafNodes;
        TreeT const * const mLhsTree;
        TreeT const * const mRhsTree;
        TreeT mLocalTree;
        TreeT * const mOutputTree;
        FilletParms mParms;
    };

    struct ProcessLeafNodesMask {
        ProcessLeafNodesMask(
            std::vector<const LeafNodeType*>& primaryLeafNodes,
            std::vector<const LeafNodeType*>& secondaryLeafNodes,
            const TreeT& lhsTree,
            const TreeT& rhsTree,
            typename MaskT::TreeType::ConstPtr maskTree,
            TreeT& outputTree,
            FilletParms parms)
            : mPrimaryLeafNodes(primaryLeafNodes)
            , mSecondaryLeafNodes(secondaryLeafNodes)
            , mLhsTree(&lhsTree)
            , mRhsTree(&rhsTree)
            , mMaskTree(maskTree)
            , mLocalTree(lhsTree.background())
            , mOutputTree(&outputTree)
            , mParms(parms)
        {
        }

        ProcessLeafNodesMask(ProcessLeafNodesMask& other, tbb::split)
            : mPrimaryLeafNodes(other.mPrimaryLeafNodes)
            , mSecondaryLeafNodes(other.mSecondaryLeafNodes)
            , mLhsTree(other.mLhsTree)
            , mRhsTree(other.mRhsTree)
            , mMaskTree(other.mMaskTree)
            , mLocalTree(mLhsTree->background())
            , mOutputTree(&mLocalTree)
            , mParms(other.mParms)
        {
        }

        void join(ProcessLeafNodesMask& rhs) { mOutputTree->merge(*rhs.mOutputTree); }

        void operator()(const tbb::blocked_range<size_t>& range)
        {
            openvdb::tree::ValueAccessor<const TreeT> lhsAcc(*mLhsTree);
            openvdb::tree::ValueAccessor<const TreeT> rhsAcc(*mRhsTree);
            openvdb::tree::ValueAccessor<const MaskTreeType> maskAcc(*mMaskTree);
            openvdb::tree::ValueAccessor<TreeT> outputAcc(*mOutputTree);

            const float alpha = mParms.mAlpha;
            const float beta = mParms.mBeta;
            const float gamma = mParms.mGamma;
            const float maskBackground = mMaskTree->background();
            const size_t primarySize = mPrimaryLeafNodes.size();

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
                if (n < primarySize) {
                    const LeafNodeType& lhsNode = *mPrimaryLeafNodes[n];
                    const openvdb::math::Coord& ijk = lhsNode.origin();
                    const LeafNodeType* rhsNode = rhsAcc.probeConstLeaf(ijk);
                    const MaskLeafNodeType* maskNode = maskAcc.probeConstLeaf(ijk);

                    if (rhsNode) {
                        LeafNodeType* outputNode = outputAcc.touchLeaf(ijk);
                        ValueType* outputData = outputNode->buffer().data();
                        NodeMaskType& outputMask = outputNode->getValueMask();

                        const ValueType* lhsData = lhsNode.buffer().data();
                        const NodeMaskType& lhsMask = lhsNode.getValueMask();
                        const ValueType* rhsData = rhsNode->buffer().data();
                        const NodeMaskType& rhsMask = rhsNode->getValueMask();
                        const MaskValueType* maskData = maskNode ? maskNode->buffer().data() : nullptr;

                        for (openvdb::Index pos = 0; pos < LeafNodeType::SIZE; ++pos) {
                            const float A = lhsData[pos];
                            const float B = rhsData[pos];
                            const float m = openvdb::math::Clamp((alpha - A) / alpha, 0.f, 1.f) *
                                            openvdb::math::Clamp((alpha - B) / alpha, 0.f, 1.f);
                            const bool hasValidFilletSamples = lhsMask.isOn(pos) && rhsMask.isOn(pos);
                            const float offset = hasValidFilletSamples ?
                                openvdb::math::Pow(m, beta) * gamma : 0.0f;
                            const bool isAMin = A < B;
                            const float multiplier = maskData ? maskData[pos] : maskBackground;
                            outputData[pos] = isAMin ? (A - offset * multiplier) : (B - offset * multiplier);
                            outputMask.set(pos, isAMin ? lhsMask.isOn(pos) : rhsMask.isOn(pos));
                        }
                    } else if (!(rhsAcc.getValue(ijk) < ValueType(0.0))) {
                        outputAcc.addLeaf(new LeafNodeType(lhsNode));
                    }
                } else {
                    const LeafNodeType& rhsNode = *mSecondaryLeafNodes[n - primarySize];
                    const openvdb::math::Coord& ijk = rhsNode.origin();
                    const LeafNodeType* lhsNode = lhsAcc.probeConstLeaf(ijk);

                    if (!lhsNode && !(lhsAcc.getValue(ijk) < ValueType(0))) {
                        outputAcc.addLeaf(new LeafNodeType(rhsNode));
                    }
                }
            }
        }

        const std::vector<const LeafNodeType*>& mPrimaryLeafNodes;
        const std::vector<const LeafNodeType*>& mSecondaryLeafNodes;
        TreeT const * const mLhsTree;
        TreeT const * const mRhsTree;
        typename MaskT::TreeType::ConstPtr mMaskTree;
        TreeT mLocalTree;
        TreeT * const mOutputTree;
        FilletParms mParms;
    };

    TreePtrType mSegment;
    TreeT const * const mLhsTree;
    TreeT const * const mRhsTree;
    typename MaskT::TreeType::ConstPtr mMaskTree;
    FilletParms mParms;
};


template<typename GridT, typename MaskT>
typename GridT::Ptr UnionWithFillet<GridT, MaskT>::blend() {
    FilletParms parms;
    parms.mAlpha = mBandwidth;
    parms.mBeta = mExponent;
    parms.mGamma = mMultiplier;
    BuildFusedSegment fused(mLhsTree, mRhsTree, mMaskTree, parms);

    fused();

    // The leafnode (level = 0) sign is set in the segment construction.
    openvdb::tools::signedFloodFill(*fused.segment(), /*threaded=*/true, /*grainSize=*/1, /*minLevel=*/1);

    mSegment = fused.segment();

    typename GridT::Ptr ret = GridT::create(mSegment);
    ret->setTransform((mRhsGrid->transform()).copy());
    ret->setGridClass(openvdb::GRID_LEVEL_SET);
    return ret;
}


template<typename GridT,
         typename MaskT>
typename GridT::Ptr
unionFillet(const GridT& lhs,
     const GridT& rhs,
     typename MaskT::ConstPtr mask,
     typename GridT::ValueType alpha,
     typename GridT::ValueType beta,
     typename GridT::ValueType gamma,
     int supportDilation)
{
    static_assert(std::is_floating_point<typename GridT::ValueType>::value,
        "assert in unionFillet: "
        "level set grids must have scalar/floating-point value types.");

    // sanitizer
    const openvdb::math::Transform& lhsXform = lhs.constTransform();
    const openvdb::math::Transform& rhsXform = rhs.constTransform();
    if (lhsXform != rhsXform) throw std::runtime_error("The two grids need to have the same transforms.");
    if (mask) {
        const openvdb::math::Transform& maskXform = mask->constTransform();
        if (lhsXform != maskXform) throw std::runtime_error("The grids and the mask need to have the same transforms.");
    }

    if (supportDilation > 0) {
        MaskGrid::Ptr supportMask =
            createBlendSupportMask(lhs, rhs, alpha, supportDilation);
        if (supportMask->activeVoxelCount() == 0) {
            UnionWithFillet<GridT, MaskT> uf(lhs, rhs, mask, alpha, beta, gamma);
            return uf.blend();
        }

        typename GridT::Ptr lhsExtended =
            tools::maskSdf(lhs, *supportMask);
        typename GridT::Ptr rhsExtended =
            tools::maskSdf(rhs, *supportMask);

        UnionWithFillet<GridT, MaskT> uf(*lhsExtended, *rhsExtended, mask, alpha, beta, gamma);
        return uf.blend();
    }

    UnionWithFillet<GridT, MaskT> uf(lhs, rhs, mask, alpha, beta, gamma);
    return uf.blend();
}
} // tools
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_TOOLS_BLEND_HAS_BEEN_INCLUDED
