///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @file SOP_OpenVDB_Rasterize_Points.cc
///
/// @author Mihai Alden
///
/// @decription Rasterize points into density and attribute grids.
///
/// @note   This SOP has a accompanying creation script that adds a default VOP
///         subnetwork and UI parameters for cloud and velocity field modeling.
///         See creation script file header for installation details.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/GU_VDBPointTools.h>
#include <openvdb/tools/PointIndexGrid.h>


#include <UT/UT_Interrupt.h>
#include <UT/UT_WorkArgs.h>
#include <GA/GA_Handle.h>
#include <GA/GA_PageIterator.h>
#include <GA/GA_Types.h>
#include <GU/GU_Detail.h>
#include <PRM/PRM_Parm.h>
#include <CH/CH_Manager.h>

//#include <SHOP/SHOP_Node.h>

#include <VOP/VOP_CodeGenerator.h>
#include <VOP/VOP_CodeCompilerArgs.h>

#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
#include <VOP/VOP_ExportedParmsManager.h>
#endif

#include <VOP/VOP_LanguageContextTypeList.h>

#include <VEX/VEX_Error.h>
#include <CVEX/CVEX_Context.h>
#include <CVEX/CVEX_Value.h>
#include <GU/GU_SopResolver.h>

#include <OP/OP_Channels.h>
#include <OP/OP_Operator.h>
#include <OP/OP_Director.h>
#include <OP/OP_OperatorTable.h>
#include <OP/OP_Caller.h>
#include <OP/OP_NodeInfoParms.h>
#include <OP/OP_VexFunction.h>


#include <algorithm> // std::sort
#include <math.h> // trigonometric functions
#include <set>
#include <sstream>
#include <string>
#include <vector>


#include <tbb/blocked_range.h>
#include <tbb/task_group.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/atomic.h>


#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string/classification.hpp> // is_any_of
#include <boost/algorithm/string/split.hpp>


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////

// Local Utility Methods

namespace {


////////////////////////////////////////


///@brief Utility structure that caches commonly used point attributes
struct PointCache
{
    typedef boost::shared_ptr<PointCache>   Ptr;
    typedef openvdb::Vec3s                  PosType;
    typedef PosType::value_type             ScalarType;

    PointCache(const GU_Detail& detail, const float radiusScale, const GA_PointGroup* group = NULL)
        : mIndexMap(&detail.getP()->getIndexMap())
        , mSize(mIndexMap->indexSize())
        , mOffsets()
        , mRadius()
        , mPos()
    {
        if (group) {
            mSize = group->entries();
            mOffsets.reset(new GA_Offset[mSize]);

            GA_Offset start, end;
            GA_Offset* offset = mOffsets.get();
            GA_Range range(*group);
            for (GA_Iterator it = range.begin(); it.blockAdvance(start, end); ) {
                for (GA_Offset off = start; off < end; ++off, ++offset) {
                    *offset = off;
                }
            }

            mRadius.reset(new float[mSize]);
            mPos.reset(new openvdb::Vec3s[mSize]);

            tbb::parallel_for(tbb::blocked_range<size_t>(0, mSize),
                IFOCachePointGroupData(mOffsets, detail, mRadius, mPos, radiusScale));

            getOffset = &PointCache::offsetFromGroupMap;
        } else if (mIndexMap->isTrivialMap()) {
            getOffset = &PointCache::offsetFromIndexCast;
        } else {
            getOffset = &PointCache::offsetFromGeoMap;
        }

        if (!group) {
            mRadius.reset(new float[mSize]);
            mPos.reset(new openvdb::Vec3s[mSize]);

            UTparallelFor(GA_SplittableRange(detail.getPointRange(group)),
                IFOCachePointData(detail, mRadius, mPos, radiusScale));
        }
    }

    PointCache(const PointCache& rhs, const std::vector<unsigned>& indices)
        : mIndexMap(rhs.mIndexMap)
        , mSize(indices.size())
        , mOffsets()
        , mRadius()
        , mPos()
    {
        mOffsets.reset(new GA_Offset[mSize]);
        mRadius.reset(new float[mSize]);
        mPos.reset(new openvdb::Vec3s[mSize]);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, mSize),
            IFOCopyPointData(indices, mOffsets, mRadius, mPos, rhs));

        getOffset = &PointCache::offsetFromGroupMap;
    }

    size_t size() const { return mSize; }

    const float& radius(size_t n) const { return mRadius[n]; }

    const openvdb::Vec3s& pos(size_t n) const { return mPos[n]; }

    void getPos(size_t n, openvdb::Vec3s& xyz) const { xyz = mPos[n]; }

    GA_Offset offsetFromIndex(size_t n) const { return (this->*getOffset)(n); }

    const float* radiusData() const { return mRadius.get(); }
    const openvdb::Vec3s* posData() const { return mPos.get(); }

    float evalMaxRadius() const {
        IFOEvalMaxRadius op(mRadius.get());
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mSize), op);
        return op.result;
    }

    float evalMinRadius() const {
        IFOEvalMinRadius op(mRadius.get());
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mSize), op);
        return op.result;
    }

private:
    // Disallow copying
    PointCache(const PointCache&);
    PointCache& operator=(const PointCache&);

    GA_Offset (PointCache::* getOffset)(const size_t) const;

    GA_Offset offsetFromGeoMap(const size_t n) const {
        return mIndexMap->offsetFromIndex(GA_Index(n));
    }

    GA_Offset offsetFromGroupMap(const size_t n) const {
        return mOffsets[n];
    }

    GA_Offset offsetFromIndexCast(const size_t n) const {
        return GA_Offset(n);
    }

    //////////

    // Internal TBB function objects

    struct IFOCopyPointData {
        IFOCopyPointData(
            const std::vector<unsigned>& indices,
            boost::scoped_array<GA_Offset>& offsets,
            boost::scoped_array<float>& radius,
            boost::scoped_array<openvdb::Vec3s>& pos,
            const PointCache& PointCache)
            : mIndices(&indices[0])
            , mOffsets(offsets.get())
            , mRadiusData(radius.get())
            , mPosData(pos.get())
            , mPointCache(&PointCache)
        {
        }

        void operator()(const tbb::blocked_range<size_t>& range) const
        {
            const float* radiusData = mPointCache->radiusData();
            const openvdb::Vec3s* posData = mPointCache->posData();

            for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
                const size_t idx = size_t(mIndices[n]);
                mOffsets[n] = mPointCache->offsetFromIndex(idx);
                mRadiusData[n] = radiusData[idx];
                mPosData[n] = posData[idx];
            }
        }

        unsigned        const * const mIndices;
        GA_Offset             * const mOffsets;
        float                 * const mRadiusData;
        openvdb::Vec3s        * const mPosData;
        PointCache      const * const mPointCache;
    }; // struct IFOCopyPointData

    struct IFOCachePointData
    {
        IFOCachePointData(const GU_Detail& detail,
            boost::scoped_array<float>& radius,
            boost::scoped_array<openvdb::Vec3s>& pos,
            float radiusScale = 1.0)
            : mDetail(&detail)
            , mRadiusData(radius.get())
            , mPosData(pos.get())
            , mRadiusScale(radiusScale)
        {
        }

        void operator()(const GA_SplittableRange& range) const
        {
            GA_Offset start, end;
            UT_Vector3 xyz;

            GA_ROHandleV3 posHandle(mDetail->getP());
            GA_ROHandleF scaleHandle;

            GA_ROAttributeRef aRef = mDetail->findFloatTuple(GA_ATTRIB_POINT, GEO_STD_ATTRIB_PSCALE);
            bool hasScale = false;

            if (aRef.isValid()) {
                hasScale = true;
                scaleHandle.bind(aRef.getAttribute());
            }

            const float scale = mRadiusScale;


            for (GA_PageIterator pageIt = range.beginPages(); !pageIt.atEnd(); ++pageIt) {
                for (GA_Iterator blockIt(pageIt.begin()); blockIt.blockAdvance(start, end); ) {
                    for (GA_Offset i = start; i < end; ++i) {

                        const GA_Index idx = mDetail->pointIndex(i);

                        mRadiusData[idx] = hasScale ? scaleHandle.get(i) * scale : scale;

                        xyz = posHandle.get(i);
                        openvdb::Vec3s& p = mPosData[idx];
                        p[0] = xyz[0];
                        p[1] = xyz[1];
                        p[2] = xyz[2];
                    }
                }
            }
        }

        GU_Detail const * const mDetail;
        float           * const mRadiusData;
        openvdb::Vec3s  * const mPosData;
        float             const mRadiusScale;
    }; // struct IFOCachePointData

    struct IFOCachePointGroupData
    {
        IFOCachePointGroupData(const boost::scoped_array<GA_Offset>& offsets,
            const GU_Detail& detail,
            boost::scoped_array<float>& radius,
            boost::scoped_array<openvdb::Vec3s>& pos,
            float radiusScale = 1.0)
            : mOffsets(offsets.get())
            , mDetail(&detail)
            , mRadiusData(radius.get())
            , mPosData(pos.get())
            , mRadiusScale(radiusScale)
        {
        }

        void operator()(const tbb::blocked_range<size_t>& range) const
        {
            GA_ROHandleV3 posHandle(mDetail->getP());
            // bind again after construction to remove uninitialized member variable compiler warning..
            posHandle.bind(mDetail->getP());

            bool hasScale = false;
            GA_ROHandleF scaleHandle;

            GA_ROAttributeRef aRef = mDetail->findFloatTuple(GA_ATTRIB_POINT, GEO_STD_ATTRIB_PSCALE);
            if (aRef.isValid()) {
                hasScale = true;
                scaleHandle.bind(aRef.getAttribute());
            }

            const float scale = mRadiusScale;
            UT_Vector3 xyz;

            for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

                const GA_Offset offset = mOffsets[n];
                mRadiusData[n] = hasScale ? scaleHandle.get(offset) * scale : scale;

                xyz = posHandle.get(offset);
                openvdb::Vec3s& p = mPosData[n];
                p[0] = xyz[0];
                p[1] = xyz[1];
                p[2] = xyz[2];
            }
        }

        GA_Offset const * const mOffsets;
        GU_Detail const * const mDetail;
        float           * const mRadiusData;
        openvdb::Vec3s  * const mPosData;
        float             const mRadiusScale;
    }; // struct IFOCachePointGroupData

    struct IFOEvalMaxRadius {
        IFOEvalMaxRadius(float const * const radiusArray)
            : mRadiusArray(radiusArray), result(-std::numeric_limits<float>::max()) {}

        IFOEvalMaxRadius(IFOEvalMaxRadius& rhs, tbb::split) // thread safe copy constructor
            : mRadiusArray(rhs.mRadiusArray), result(-std::numeric_limits<float>::max()){}

        void operator()(const tbb::blocked_range<size_t>& range) {
            for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
                 result = std::max(mRadiusArray[n], result);
            }
        }

        void join(const IFOEvalMaxRadius& rhs) { result = std::max(rhs.result, result); }

        float const * const mRadiusArray;
        float result;
    }; // struct IFOEvalMaxRadius

    struct IFOEvalMinRadius {
        IFOEvalMinRadius(float const * const radiusArray)
            : mRadiusArray(radiusArray), result(std::numeric_limits<float>::max()) {}

        IFOEvalMinRadius(IFOEvalMinRadius& rhs, tbb::split) // thread safe copy constructor
            : mRadiusArray(rhs.mRadiusArray), result(std::numeric_limits<float>::max()){}

        void operator()(const tbb::blocked_range<size_t>& range) {
            for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
                 result = std::min(mRadiusArray[n], result);
            }
        }

        void join(const IFOEvalMinRadius& rhs) { result = std::min(rhs.result, result); }

        float const * const mRadiusArray;
        float result;
    }; // struct IFOEvalMinRadius

    //////////

    GA_IndexMap const * const           mIndexMap;
    size_t                              mSize;
    boost::scoped_array<GA_Offset>      mOffsets;
    boost::scoped_array<float>          mRadius;
    boost::scoped_array<openvdb::Vec3s> mPos;
}; // struct PointCache


///@brief   Radius based partitioning of points into multiple @c openvdb::tools::PointIndexGrid
///         acceleration structures. Improves spatial query time for points with varying radius.
struct PointIndexGridCollection
{
    typedef openvdb::tools::PointIndexGrid              PointIndexGrid;
    typedef PointIndexGrid::TreeType                    PointIndexTree;
    typedef PointIndexTree::LeafNodeType                PointIndexLeafNode;
    typedef PointIndexTree::ValueConverter<bool>::Type  BoolTreeType;

    PointIndexGridCollection(const GU_Detail& detail, const float radiusScale,
        const float minVoxelSize, const GA_PointGroup* group = NULL, hvdb::Interrupter* interrupter = NULL)
        : mPointCacheArray() , mIdxGridArray(), mMinRadiusArray(), mMaxRadiusArray()
    {
        mPointCacheArray.push_back(PointCache::Ptr(new PointCache(detail, radiusScale, group)));

        std::vector<double> voxelSizeList;
        voxelSizeList.push_back(std::max(minVoxelSize, mPointCacheArray.back()->evalMinRadius()));

        for (size_t n = 0; n < 50; ++n) {

            if (interrupter && interrupter->wasInterrupted()) break;

            PointCache& pointCache = *mPointCacheArray.back();

            const float maxRadius = pointCache.evalMaxRadius();
            const float limit = float(voxelSizeList.back() * (n < 40 ? 2.0 : 8.0));

            if (!(maxRadius > limit)) {
                break;
            }

            std::vector<unsigned> lhsIdx, rhsIdx;

            float minRadius = maxRadius;
            const float* radiusData = pointCache.radiusData();
            for (unsigned i = 0, I = unsigned(pointCache.size()); i < I; ++i) {
                if (radiusData[i] > limit) {
                    rhsIdx.push_back(i);
                    minRadius = std::min(minRadius, radiusData[i]);
                } else lhsIdx.push_back(i);
            }

            voxelSizeList.push_back(minRadius);

            PointCache::Ptr lhsPointCache(new PointCache(pointCache, lhsIdx));
            PointCache::Ptr rhsPointCache(new PointCache(pointCache, rhsIdx));

            mPointCacheArray.back() = lhsPointCache;
            mPointCacheArray.push_back(rhsPointCache);
        }

        const size_t collectionSize = mPointCacheArray.size();

        mIdxGridArray.resize(collectionSize);
        mMinRadiusArray.reset(new float[collectionSize]);
        mMaxRadiusArray.reset(new float[collectionSize]);

        tbb::task_group tasks;

        for (size_t n = 0; n < collectionSize; ++n) {

            if (interrupter && interrupter->wasInterrupted()) break;

            tasks.run(IFOCreateAuxiliaryData(mIdxGridArray[n], *mPointCacheArray[n],
                voxelSizeList[n], mMinRadiusArray[n], mMaxRadiusArray[n]));
        }

        tasks.wait();
    }

    //////////

    size_t size() const { return mPointCacheArray.size(); }

    float minRadius(size_t n) const { return mMinRadiusArray[n]; }
    float maxRadius(size_t n) const { return mMaxRadiusArray[n]; }

    const PointCache& pointCache(size_t n) const { return *mPointCacheArray[n]; }
    const PointIndexGrid& idxGrid(size_t n) const { return *mIdxGridArray[n]; }

private:
    // Disallow copying
    PointIndexGridCollection(const PointIndexGridCollection&);
    PointIndexGridCollection& operator=(const PointIndexGridCollection&);

    //////////

    // Internal TBB function objects

    struct IFOCreateAuxiliaryData  {
        IFOCreateAuxiliaryData(PointIndexGrid::Ptr& idxGridPt, PointCache& points,
            double voxelSize, float& minRadius, float& maxRadius)
            : mIdxGrid(&idxGridPt), mPointCache(&points), mVoxelSize(voxelSize)
            , mMinRadius(&minRadius), mMaxRadius(&maxRadius)
        {}

        void operator()() const {
            const openvdb::math::Transform::Ptr transform =
                openvdb::math::Transform::createLinearTransform(mVoxelSize);
            *mIdxGrid = openvdb::tools::createPointIndexGrid<PointIndexGrid>(*mPointCache, *transform);
            *mMinRadius = mPointCache->evalMinRadius();
            *mMaxRadius = mPointCache->evalMaxRadius();
        }

        PointIndexGrid::Ptr       * const mIdxGrid;
        PointCache          const * const mPointCache;
        double                      const mVoxelSize;
        float                     * const mMinRadius;
        float                     * const mMaxRadius;
    }; // struct IFOCreateAuxiliaryData

    //////////

    std::vector<PointCache::Ptr>        mPointCacheArray;
    std::vector<PointIndexGrid::Ptr>    mIdxGridArray;
    boost::scoped_array<float>          mMinRadiusArray, mMaxRadiusArray;
}; // struct PointIndexGridCollection


///@brief   TBB function object to construct a @c BoolTree region of interest
///         mask for the gather based rasterization step.
struct ConstructCandidateVoxelMask
{
    typedef PointCache::PosType                         PosType;
    typedef PosType::value_type                         ScalarType;

    typedef openvdb::tools::PointIndexGrid::TreeType    PointIndexTree;
    typedef PointIndexTree::LeafNodeType                PointIndexLeafNode;
    typedef PointIndexLeafNode::ValueType               PointIndexType;

    typedef PointIndexTree::ValueConverter<bool>::Type  BoolTreeType;
    typedef BoolTreeType::LeafNodeType                   BoolLeafNode;

    /////

    ConstructCandidateVoxelMask(BoolTreeType& maskTree, const PointCache& points,
        const std::vector<const PointIndexLeafNode*>& pointIndexLeafNodes,
        const openvdb::math::Transform& xform,
        hvdb::Interrupter* interrupter = NULL)
        : mMaskTree(false)
        , mMaskTreePt(&maskTree)
        , mMaskAccessor(*mMaskTreePt)
        , mPoints(&points)
        , mPointIndexNodes(&pointIndexLeafNodes.front())
        , mXform(xform)
        , mInterrupter(interrupter)
    {
    }

    /// Thread safe copy constructor
    ConstructCandidateVoxelMask(ConstructCandidateVoxelMask& rhs, tbb::split)
        : mMaskTree(false)
        , mMaskTreePt(&mMaskTree)
        , mMaskAccessor(*mMaskTreePt)
        , mPoints(rhs.mPoints)
        , mPointIndexNodes(rhs.mPointIndexNodes)
        , mXform(rhs.mXform)
        , mInterrupter(rhs.mInterrupter)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {

        openvdb::CoordBBox box;
        PosType pos, bboxMin, bboxMax, pMin, pMax;
        ScalarType radius(0.0);

        const PointIndexType *pointIdxPt = NULL, *endIdxPt = NULL;

        std::vector<PointIndexType> largeParticleIndices;
        double leafnodeSize = mXform.voxelSize()[0] * double(PointIndexLeafNode::DIM);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            if (this->wasInterrupted()) {
                tbb::task::self().cancel_group_execution();
                break;
            }

            const PointIndexLeafNode& node = *mPointIndexNodes[n];

            for (PointIndexLeafNode::ValueOnCIter it = node.cbeginValueOn(); it; ++it) {

                node.getIndices(it.pos(), pointIdxPt, endIdxPt);

                bboxMin[0] = std::numeric_limits<ScalarType>::max();
                bboxMin[1] = std::numeric_limits<ScalarType>::max();
                bboxMin[2] = std::numeric_limits<ScalarType>::max();

                bboxMax[0] = -bboxMin[0];
                bboxMax[1] = -bboxMin[1];
                bboxMax[2] = -bboxMin[2];

                bool regionIsValid = false;

                while (pointIdxPt < endIdxPt) {

                    radius = mPoints->radius(*pointIdxPt);

                    if (radius > leafnodeSize) {
                        largeParticleIndices.push_back(*pointIdxPt);
                    } else {
                        pos = mPoints->pos(*pointIdxPt);

                        pMin[0] = pos[0] - radius;
                        pMin[1] = pos[1] - radius;
                        pMin[2] = pos[2] - radius;

                        pMax[0] = pos[0] + radius;
                        pMax[1] = pos[1] + radius;
                        pMax[2] = pos[2] + radius;

                        bboxMin[0] = std::min(bboxMin[0], pMin[0]);
                        bboxMin[1] = std::min(bboxMin[1], pMin[1]);
                        bboxMin[2] = std::min(bboxMin[2], pMin[2]);

                        bboxMax[0] = std::max(bboxMax[0], pMax[0]);
                        bboxMax[1] = std::max(bboxMax[1], pMax[1]);
                        bboxMax[2] = std::max(bboxMax[2], pMax[2]);

                        regionIsValid = true;
                    }

                    ++pointIdxPt;
                }

                if (regionIsValid) {
                    box.min() = mXform.worldToIndexCellCentered(bboxMin);
                    box.max() = mXform.worldToIndexCellCentered(bboxMax);
                    activateRegion(box);
                }
            }
        }

        for (size_t n = 0, N = largeParticleIndices.size(); n != N; ++n) {

            radius = mPoints->radius(largeParticleIndices[n]);
            pos = mPoints->pos(largeParticleIndices[n]);

            bboxMin[0] = std::numeric_limits<ScalarType>::max();
            bboxMin[1] = std::numeric_limits<ScalarType>::max();
            bboxMin[2] = std::numeric_limits<ScalarType>::max();

            bboxMax[0] = -bboxMin[0];
            bboxMax[1] = -bboxMin[1];
            bboxMax[2] = -bboxMin[2];

            pMin[0] = pos[0] - radius;
            pMin[1] = pos[1] - radius;
            pMin[2] = pos[2] - radius;

            pMax[0] = pos[0] + radius;
            pMax[1] = pos[1] + radius;
            pMax[2] = pos[2] + radius;

            bboxMin[0] = std::min(bboxMin[0], pMin[0]);
            bboxMin[1] = std::min(bboxMin[1], pMin[1]);
            bboxMin[2] = std::min(bboxMin[2], pMin[2]);

            bboxMax[0] = std::max(bboxMax[0], pMax[0]);
            bboxMax[1] = std::max(bboxMax[1], pMax[1]);
            bboxMax[2] = std::max(bboxMax[2], pMax[2]);

            box.min() = mXform.worldToIndexCellCentered(bboxMin);
            box.max() = mXform.worldToIndexCellCentered(bboxMax);

            activateRadialRegion(box);
        }
    }

    void join(ConstructCandidateVoxelMask& rhs) {

        std::vector<BoolLeafNode*> rhsLeafNodes, overlappingLeafNodes;

        rhsLeafNodes.reserve(rhs.mMaskTreePt->leafCount());
        rhs.mMaskTreePt->getNodes(rhsLeafNodes);

        // Steal unique leafnodes

        openvdb::tree::ValueAccessor<BoolTreeType> lhsAcc(*mMaskTreePt);
        openvdb::tree::ValueAccessor<BoolTreeType> rhsAcc(*rhs.mMaskTreePt);

        typedef BoolTreeType::RootNodeType BoolRootNodeType;
        typedef BoolRootNodeType::NodeChainType BoolNodeChainType;
        typedef boost::mpl::at<BoolNodeChainType, boost::mpl::int_<1> >::type BoolInternalNodeType;

        for (size_t n = 0, N = rhsLeafNodes.size(); n < N; ++n) {
            const openvdb::Coord& ijk = rhsLeafNodes[n]->origin();
            if (!lhsAcc.probeLeaf(ijk)) {

                // add node to lhs tree
                lhsAcc.addLeaf(rhsLeafNodes[n]);

                // remove leaf node from rhs tree
                BoolInternalNodeType* internalNode = rhsAcc.probeNode<BoolInternalNodeType>(ijk);

                if (internalNode) {
                    internalNode->stealNode<BoolLeafNode>(ijk, false, false);
                } else {
                    rhs.mMaskTreePt->stealNode<BoolLeafNode>(ijk, false, false);
                }

            } else {
                overlappingLeafNodes.push_back(rhsLeafNodes[n]);
            }
        }

        // Combine overlapping leaf nodes
        tbb::parallel_for(tbb::blocked_range<size_t>(0, overlappingLeafNodes.size()),
            IFOTopologyUnion(*mMaskTreePt, &overlappingLeafNodes[0]));
    }

private:

    bool wasInterrupted() const { return mInterrupter && mInterrupter->wasInterrupted(); }

    // just a rough estimate, but more accurate than activateRegion(...) for large spheres.
    void activateRadialRegion(const openvdb::CoordBBox& bbox)
    {
        typedef BoolTreeType::LeafNodeType LeafNodeType;

        const openvdb::Vec3d center = bbox.getCenter();
        const double radius = double(bbox.dim()[0]) * 0.5;

        // inscribed box

        const double iRadius = radius * double(1.0 / std::sqrt(3.0));
        openvdb::CoordBBox ibox(
            openvdb::Coord::round(openvdb::Vec3d(center[0] - iRadius, center[1] - iRadius, center[2] - iRadius)),
            openvdb::Coord::round(openvdb::Vec3d(center[0] + iRadius, center[1] + iRadius, center[2] + iRadius)));

        ibox.min() &= ~(LeafNodeType::DIM - 1);
        ibox.max() &= ~(LeafNodeType::DIM - 1);

        openvdb::Coord ijk(0);

        for (ijk[0] = ibox.min()[0]; ijk[0] <= ibox.max()[0]; ijk[0] += LeafNodeType::DIM) {
            for (ijk[1] = ibox.min()[1]; ijk[1] <= ibox.max()[1]; ijk[1] += LeafNodeType::DIM) {
                for (ijk[2] = ibox.min()[2]; ijk[2] <= ibox.max()[2]; ijk[2] += LeafNodeType::DIM) {
                    mMaskAccessor.touchLeaf(ijk)->setValuesOn();
                }
            }
        }

        const openvdb::Coord leafMin = bbox.min() & ~(LeafNodeType::DIM - 1);
        const openvdb::Coord leafMax = bbox.max() & ~(LeafNodeType::DIM - 1);

        openvdb::Vec3d xyz;
        const double leafNodeRadius = double(LeafNodeType::DIM) * std::sqrt(3.0) * 0.5;
        double distSqr = radius + leafNodeRadius;
        distSqr *= distSqr;

        for (ijk[0] = leafMin[0]; ijk[0] <= leafMax[0]; ijk[0] += LeafNodeType::DIM) {
            for (ijk[1] = leafMin[1]; ijk[1] <= leafMax[1]; ijk[1] += LeafNodeType::DIM) {
                for (ijk[2] = leafMin[2]; ijk[2] <= leafMax[2]; ijk[2] += LeafNodeType::DIM) {

                    if (!ibox.isInside(ijk)) {

                        xyz[0] = double(ijk[0]);
                        xyz[1] = double(ijk[1]);
                        xyz[2] = double(ijk[2]);
                        xyz += double(LeafNodeType::DIM - 1) * 0.5;
                        xyz -= center;

                        if (!(xyz.lengthSqr() > distSqr)) {
                            activateLeafNodeRegion(bbox, *mMaskAccessor.touchLeaf(ijk));
                        }
                    }
                }
            }
        }
    }

    void activateRegion(const openvdb::CoordBBox& bbox)
    {
        typedef BoolTreeType::LeafNodeType LeafNodeType;
        const openvdb::Coord leafMin = bbox.min() & ~(LeafNodeType::DIM - 1);
        const openvdb::Coord leafMax = bbox.max() & ~(LeafNodeType::DIM - 1);
        openvdb::Coord ijk(0);

        for (ijk[0] = leafMin[0]; ijk[0] <= leafMax[0]; ijk[0] += LeafNodeType::DIM) {
            for (ijk[1] = leafMin[1]; ijk[1] <= leafMax[1]; ijk[1] += LeafNodeType::DIM) {
                for (ijk[2] = leafMin[2]; ijk[2] <= leafMax[2]; ijk[2] += LeafNodeType::DIM) {
                    activateLeafNodeRegion(bbox, *mMaskAccessor.touchLeaf(ijk));
                }
            }
        }
    }

    template <typename LeafNodeType>
    void activateLeafNodeRegion(const openvdb::CoordBBox& bbox, LeafNodeType& node) const
    {
        const openvdb::Coord& origin = node.origin();
        openvdb::Coord ijk = origin;
        ijk.offset(LeafNodeType::DIM - 1);

        if (bbox.isInside(origin) && bbox.isInside(ijk)) {
            node.setValuesOn();
        } else if (!node.isValueMaskOn()) {
            const openvdb::Coord ijkMin = openvdb::Coord::maxComponent(bbox.min(), origin);
            const openvdb::Coord ijkMax = openvdb::Coord::minComponent(bbox.max(), ijk);

            openvdb::Index xPos(0), yPos(0);

            for (ijk[0] = ijkMin[0]; ijk[0] <= ijkMax[0]; ++ijk[0]) {
                xPos = (ijk[0] & (LeafNodeType::DIM - 1u)) << (2 * LeafNodeType::LOG2DIM);
                for (ijk[1] = ijkMin[1]; ijk[1] <= ijkMax[1]; ++ijk[1]) {
                    yPos = xPos + ((ijk[1] & (LeafNodeType::DIM - 1u)) << LeafNodeType::LOG2DIM);
                    for (ijk[2] = ijkMin[2]; ijk[2] <= ijkMax[2]; ++ijk[2]) {
                        node.setValueOn(yPos + (ijk[2] & (LeafNodeType::DIM - 1u)));
                    }
                }
            }
        }
    }

    //////////

    // Internal TBB function objects

    struct IFOTopologyUnion {
        IFOTopologyUnion(BoolTreeType& tree, BoolLeafNode ** nodes)
            : mTree(&tree), mNodes(nodes) { }

        void operator()(const tbb::blocked_range<size_t>& range) const {
            openvdb::tree::ValueAccessor<BoolTreeType> acc(*mTree);
            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
                acc.probeLeaf(mNodes[n]->origin())->topologyUnion(*mNodes[n]);
            }
        }

        BoolTreeType           * const mTree;
        BoolLeafNode  * const * const mNodes;
    }; // struct IFOTopologyUnion

    //////////

    BoolTreeType                                       mMaskTree;
    BoolTreeType                               * const mMaskTreePt;
    openvdb::tree::ValueAccessor<BoolTreeType>         mMaskAccessor;

    PointCache                          const * const mPoints;
    PointIndexLeafNode          const * const * const mPointIndexNodes;
    openvdb::math::Transform                    const mXform;
    hvdb::Interrupter                         * const mInterrupter;
}; // struct ConstructCandidateVoxelMask


///@brief Constructs a region of interest mask for the gather based rasterization step.
PointIndexGridCollection::BoolTreeType::Ptr
constructROIMask(const PointIndexGridCollection& idxGridCollection,
    const openvdb::math::Transform& volumeTransform,
    hvdb::Interrupter* interrupter = NULL)
{
    PointIndexGridCollection::BoolTreeType::Ptr maskTree(new PointIndexGridCollection::BoolTreeType(false));

    for (size_t n = 0; n < idxGridCollection.size(); ++n) {

        if (interrupter && interrupter->wasInterrupted()) break;

        const PointCache& pointCache = idxGridCollection.pointCache(n);
        const PointIndexGridCollection::PointIndexGrid& idxGrid = idxGridCollection.idxGrid(n);

        const double voxelSize = idxGrid.transform().voxelSize()[0];

        PointIndexGridCollection::PointIndexGrid::Ptr regionPointGridPtr; // optionally used
        const PointIndexGridCollection::PointIndexTree* regionPointIndexTree = &idxGrid.tree();

        const double maxPointRadius = idxGridCollection.maxRadius(n);

        if (maxPointRadius * 1.5 > voxelSize) {
            const openvdb::math::Transform::Ptr xform =
                openvdb::math::Transform::createLinearTransform(maxPointRadius);

            regionPointGridPtr =
                openvdb::tools::createPointIndexGrid<PointIndexGridCollection::PointIndexGrid>(pointCache, *xform);

            regionPointIndexTree = &regionPointGridPtr->tree();
        }

        std::vector<const PointIndexGridCollection::PointIndexLeafNode*> pointIndexLeafNodes;

        pointIndexLeafNodes.reserve(regionPointIndexTree->leafCount());
        regionPointIndexTree->getNodes(pointIndexLeafNodes);

        ConstructCandidateVoxelMask op(*maskTree, pointCache,
            pointIndexLeafNodes, volumeTransform, interrupter);

        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, pointIndexLeafNodes.size()), op);
    }

    return maskTree;
}


////////////////////////////////////////

// Type traits

template<typename T> struct ValueTypeTraits {
    static const bool IsVec = false;
    static const int TupleSize = 1;
    typedef T ScalarType;
    typedef T HoudiniType;

    static void convert(T& lhs, const HoudiniType rhs) {
        lhs = rhs;
    }
};


template<typename T> struct ValueTypeTraits<openvdb::math::Vec3<T> > {
    static const bool IsVec = true;
    static const int TupleSize = 3;
    typedef T               ScalarType;
    typedef UT_Vector3T<T>  HoudiniType;

    static void convert(openvdb::math::Vec3<T>& lhs, const HoudiniType& rhs) {
        lhs[0] = rhs[0];
        lhs[1] = rhs[1];
        lhs[2] = rhs[2];
    }
};


////////////////////////////////////////


///@brief Composites point attributes into voxel data using a weighted average approach.
template<typename _ValueType>
struct WeightedAverageOp
{
    enum { LOG2DIM = openvdb::tools::PointIndexTree::LeafNodeType::LOG2DIM };

    typedef boost::shared_ptr<WeightedAverageOp>                Ptr;
    typedef boost::shared_ptr<const WeightedAverageOp>          ConstPtr;

    typedef _ValueType                                          ValueType;
    typedef openvdb::tree::LeafNode<ValueType, LOG2DIM>         LeafNodeType;
    typedef typename ValueTypeTraits<ValueType>::ScalarType     ScalarType;
    typedef typename ValueTypeTraits<ValueType>::HoudiniType    HoudiniType;

    /////

    WeightedAverageOp(const GA_Attribute& attrib, boost::scoped_array<LeafNodeType*>& nodes)
        : mHandle(&attrib), mNodes(nodes.get()), mNode(NULL), mNodeVoxelData(NULL)
        , mNodeOffset(0), mValue(ScalarType(0.0)), mVaryingDataBuffer(NULL), mVaryingData(false)
    {
    }

    ~WeightedAverageOp() { if (mNode) delete mNode; }

    const char* getName() const { return mHandle.getAttribute()->getName(); }

    void beginNodeProcessing(const openvdb::Coord& origin, size_t nodeOffset)
    {
        mVaryingData = false;
        mNodeOffset = nodeOffset;
        if (mNode) mNode->setOrigin(origin);
        else mNode = new LeafNodeType(origin, openvdb::zeroVal<ValueType>());
        mNodeVoxelData = const_cast<ValueType*>(&mNode->getValue(0));//mNode->buffer().data();
    }

    void updateValue(const GA_Offset pointOffset) {
        const HoudiniType val = mHandle.get(pointOffset);
        ValueTypeTraits<ValueType>::convert(mValue, val);
    }

    void updateVoxelData(const std::vector<std::pair<float, openvdb::Index> >& densitySamples) {

        typedef std::pair<float, openvdb::Index> DensitySample;

        for (size_t n = 0, N = densitySamples.size(); n < N; ++n) {

            const DensitySample& sample = densitySamples[n];

            ValueType& value = mNodeVoxelData[sample.second];

            if (mVaryingData) {
                ValueTypeTraits<ValueType>::convert(mValue, mVaryingDataBuffer[n]);
            }

            value += mValue * sample.first;
        }
    }


    template<typename LeafNodeT>
    void endNodeProcessing(const LeafNodeT& maskNode, float *voxelWeightArray)
    {
        mNode->topologyUnion(maskNode);

        ValueType* values = const_cast<ValueType*>(&mNode->getValue(0));
        for (size_t n = 0; n < LeafNodeType::SIZE; ++n) {
            values[n] *= voxelWeightArray[n];
        }

        mNodes[mNodeOffset] = mNode;
        mNode = NULL;
    }


    HoudiniType* varyingData()
    {
        mVaryingData = true;

        if (!mVaryingDataBuffer) {
            mVaryingDataBuffer.reset(new HoudiniType[LeafNodeType::SIZE]);
        }

        return mVaryingDataBuffer.get();
    }


private:
    GA_ROHandleT<HoudiniType>           mHandle;
    LeafNodeType ** const               mNodes;
    LeafNodeType *                      mNode;
    ValueType *                         mNodeVoxelData;
    size_t                              mNodeOffset;
    ValueType                           mValue;
    boost::scoped_array<HoudiniType>    mVaryingDataBuffer;
    bool                                mVaryingData;
}; // struct WeightedAverageOp


///@brief Composites point density into voxel data.
template<typename _ValueType>
struct DensityOp
{
    enum { LOG2DIM = openvdb::tools::PointIndexTree::LeafNodeType::LOG2DIM };

    typedef boost::shared_ptr<DensityOp>                Ptr;
    typedef boost::shared_ptr<const DensityOp>          ConstPtr;

    typedef _ValueType                                  ValueType;
    typedef openvdb::tree::LeafNode<ValueType, LOG2DIM> LeafNodeType;

    /////

    DensityOp(const GA_Attribute& attrib, boost::scoped_array<LeafNodeType*>& nodes)
        : mPosHandle(&attrib), mNodes(nodes.get()), mNode(NULL), mNodeOffset(0)
    {
    }

    ~DensityOp() { if (mNode) delete mNode; }

    void beginNodeProcessing(const openvdb::Coord& origin, size_t nodeOffset)
    {
        mNodeOffset = nodeOffset;
        if (mNode) mNode->setOrigin(origin);
        else mNode = new LeafNodeType(origin, openvdb::zeroVal<ValueType>());
    }

    ValueType* data() { return const_cast<ValueType*>(&mNode->getValue(0)); /*mNode->buffer().data();*/ }

    template<typename LeafNodeT>
    void endNodeProcessing(const LeafNodeT& maskNode)
    {
        mNode->topologyUnion(maskNode);
        mNodes[mNodeOffset] = mNode;
        mNode = NULL;
    }

private:
    GA_ROHandleV3           mPosHandle;
    LeafNodeType * * const  mNodes;
    LeafNodeType   *        mNode;
    size_t                  mNodeOffset;
}; // struct DensityOp


///@brief Wrapper object for Houdini point attributes.
template<typename _ValueType, typename _OperatorType = WeightedAverageOp<_ValueType> >
struct Attribute
{
    enum { LOG2DIM = openvdb::tools::PointIndexTree::LeafNodeType::LOG2DIM };

    typedef boost::shared_ptr<Attribute>                        Ptr;
    typedef boost::shared_ptr<const Attribute>                  ConstPtr;

    typedef _OperatorType                                       OperatorType;
    typedef _ValueType                                          ValueType;
    typedef openvdb::tree::LeafNode<ValueType, LOG2DIM>         LeafNodeType;

    typedef openvdb::tools::PointIndexTree                                          PointIndexTreeType;
    typedef typename PointIndexTreeType::template ValueConverter<ValueType>::Type   TreeType;
    typedef typename openvdb::Grid<TreeType>                                        GridType;

    /////

    static Ptr create(const std::string& name, const GU_Detail& detail, size_t nodeCount)
    {
        GA_ROAttributeRef ref;
        std::string gridName;

        if (name == std::string(GEO_STD_ATTRIB_POSITION)) {
            ref = detail.getP();
            gridName = "density";
        } else {
            ref = detail.findFloatTuple(GA_ATTRIB_POINT, name.c_str(), ValueTypeTraits<ValueType>::TupleSize);
            gridName = name;
        }

        if (ref.isValid()) return Ptr(new Attribute<ValueType, OperatorType>(*ref.getAttribute(), gridName, nodeCount));
        return Ptr();
    }

    typename OperatorType::Ptr getAccessor()
    {
        return typename OperatorType::Ptr(new OperatorType(*mAttrib, mNodes));
    }

    void exportVdb(GU_Detail& detail, const openvdb::math::Transform::Ptr& xform)
    {
        typename GridType::Ptr grid= GridType::create();

        IFOPopulateTree op(grid->tree(), mNodes);
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mNodeCount), op);

        if (!grid->tree().empty()) {
            grid->setTransform(xform);

            if (mName == std::string("density")) {
                grid->setGridClass(openvdb::GRID_FOG_VOLUME);
            }

            if (ValueTypeTraits<ValueType>::IsVec) {
                if (mName == std::string(GEO_STD_ATTRIB_VELOCITY)) {
                    grid->setVectorType(openvdb::VEC_CONTRAVARIANT_RELATIVE);
                } else if (mName == std::string(GEO_STD_ATTRIB_NORMAL)) {
                    grid->setVectorType(openvdb::VEC_COVARIANT_NORMALIZE);
                } else if (mName == std::string(GEO_STD_ATTRIB_POSITION)) {
                    grid->setVectorType(openvdb::VEC_CONTRAVARIANT_ABSOLUTE);
                }
            }

            hvdb::createVdbPrimitive(detail, grid, mName.c_str());
        }
    }

    ~Attribute()
    {
        for (size_t n = 0; n < mNodeCount; ++n) {
            if (mNodes[n] != NULL) delete mNodes[n];
        }
    }

private:

    Attribute(const GA_Attribute& attrib, const std::string& name, size_t nodeCount)
        : mAttrib(&attrib), mName(name), mNodeCount(nodeCount), mNodes(new LeafNodeType*[nodeCount])
    {
        for (size_t n = 0; n < mNodeCount; ++n) mNodes[n] = NULL;
    }

    //////////

    // Internal TBB function objects

    struct IFOPopulateTree {
        IFOPopulateTree(TreeType& tree, boost::scoped_array<LeafNodeType*>& nodes)
            : mTree(), mAccessor(tree) , mNodes(nodes.get()) {}

        IFOPopulateTree(IFOPopulateTree& rhs, tbb::split) // Thread safe copy constructor
            : mTree(rhs.mAccessor.tree().background()), mAccessor(mTree), mNodes(rhs.mNodes) {}

        void operator()(const tbb::blocked_range<size_t>& range) {
            for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
                if (mNodes[n]) {
                    mAccessor.addLeaf(mNodes[n]);
                    mNodes[n] = NULL;
                }
            }
        }

        void join(const IFOPopulateTree& rhs) {
            mAccessor.tree().merge(rhs.mAccessor.tree());
        }

        TreeType mTree;
        openvdb::tree::ValueAccessor<TreeType> mAccessor;
        LeafNodeType * * const mNodes;
    }; // struct IFOPopulateTree

    //////////

    GA_Attribute const * const          mAttrib;
    const std::string                   mName;
    const size_t                        mNodeCount;
    boost::scoped_array<LeafNodeType*>  mNodes;
}; // struct Attribute


////////////////////////////////////////

// VEX Utilities

struct VEXProgram {

    typedef boost::shared_ptr<VEXProgram>   Ptr;

    VEXProgram(OP_Caller& opcaller, const UT_WorkArgs& vexArgs, fpreal time, size_t maxArraySize
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
        , GU_VexGeoInputs& geoinputs
#endif
              )
        : mCVEX()
        , mRunData()
        , mMaxArraySize(maxArraySize)
        , mWorldCoordBuffer()
        , mNoiseBuffer()
        , mVEXLoaded(false)
        , mIsTimeDependant(false)
    {
        mRunData.setOpCaller(&opcaller);
        mRunData.setTime(time);
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
        mRunData.setGeoInputs(&geoinputs);
#endif

        // array attributes
        mCVEX.addInput("voxelpos", CVEX_TYPE_VECTOR3, true);
        mCVEX.addInput("Time", CVEX_TYPE_FLOAT, false);
        mCVEX.addInput("TimeInc", CVEX_TYPE_FLOAT, false);
        mCVEX.addInput("Frame", CVEX_TYPE_FLOAT, false);

        // uniform attributes
        mCVEX.addInput("voxelsize", CVEX_TYPE_FLOAT, false);
        mCVEX.addInput("pcenter", CVEX_TYPE_VECTOR3, false);
        mCVEX.addInput("pradius", CVEX_TYPE_FLOAT, false);
        mCVEX.addInput("pindex", CVEX_TYPE_INTEGER, false);

#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
        mVEXLoaded = mCVEX.load(vexArgs.getArgc(), vexArgs.getArgv());
#else
        mVEXLoaded = mCVEX.load(vexArgs.getArgc(), const_cast<UT_WorkArgs&>(vexArgs).getArgv());
#endif
    }

    void run(size_t arraySize) {
        if (mVEXLoaded) {
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
            mRunData.setTimeDependent(false);
#else
            mRunData.myTimeDependent = false;
#endif

            mCVEX.run(int(arraySize), false, &mRunData);
            mIsTimeDependant = mRunData.isTimeDependent();
        }
    }

    bool isTimeDependant() const { return mIsTimeDependant; }

    CVEX_Value* findInput(const char *name, CVEX_Type type) {
        return mVEXLoaded ? mCVEX.findInput(name, type) : NULL;
    }

    CVEX_Value* findOutput(const char *name, CVEX_Type type) {
        return mVEXLoaded ? mCVEX.findOutput(name, type) : NULL;
    }

    //////////
    // Array buffers

    UT_Vector3* getWorldCoordBuffer() {
        if (!mWorldCoordBuffer) mWorldCoordBuffer.reset(new UT_Vector3[mMaxArraySize]);
        return mWorldCoordBuffer.get();
    }

    fpreal32* getNoiseBuffer() {
        if (!mNoiseBuffer) mNoiseBuffer.reset(new fpreal32[mMaxArraySize]);
        return mNoiseBuffer.get();
    }

private:
    CVEX_Context mCVEX;
    CVEX_RunData mRunData;
    const size_t mMaxArraySize;
    boost::scoped_array<UT_Vector3> mWorldCoordBuffer;
    boost::scoped_array<fpreal32> mNoiseBuffer;
    bool mVEXLoaded, mIsTimeDependant;
}; // struct VEXProgram


struct VEXContext {

    VEXContext(OP_Caller& opcaller, const UT_String& script, size_t maxArraySize)
        : mThreadLocalTable()
        , mCaller(&opcaller)
        , mVexScript(script)
        , mVexArgs()
        , mMaxArraySize(maxArraySize)
        , mTime(0.0f)
        , mTimeInc(0.0f)
        , mFrame(0.0f)
        , mIsTimeDependant()
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
        , mVexInputs()
#endif
    {
        mIsTimeDependant = 0;
        mVexScript.parse(mVexArgs);
    }

    void setTime(fpreal time, fpreal timeinc, fpreal frame) {
        mTime = time; mTimeInc = timeinc; mFrame = frame;
    }

#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
    void setInput(int idx,  const GU_Detail *geo) { mVexInputs.setInput(idx, geo); }
#endif

    VEXProgram& getThereadLocalVEXProgram() {
        VEXProgram::Ptr& ptr = mThreadLocalTable.local();
        if (!ptr) {
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
            ptr.reset(new VEXProgram(*mCaller, mVexArgs, mTime, mMaxArraySize, mVexInputs));
#else
            ptr.reset(new VEXProgram(*mCaller, mVexArgs, mTime, mMaxArraySize));
#endif
        }

        return *ptr;
    }

    fpreal time() const { return mTime; }
    fpreal timeInc() const { return mTimeInc; }
    fpreal frame() const { return mFrame; }

    void setTimeDependantFlag() { mIsTimeDependant = 1; }
    bool isTimeDependant() const { return mIsTimeDependant == 1; }

private:
    tbb::enumerable_thread_specific<VEXProgram::Ptr> mThreadLocalTable;
    OP_Caller* mCaller;
    UT_String mVexScript;
    UT_WorkArgs mVexArgs;
    const size_t mMaxArraySize;
    fpreal mTime, mTimeInc, mFrame;
    tbb::atomic<int> mIsTimeDependant;
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
    GU_VexGeoInputs mVexInputs;
#endif
};


////////////////////////////////////////


///@brief Gather based point rasterization.
struct RasterizePoints
{
    typedef PointCache::PosType                                         PosType;
    typedef PosType::value_type                                         ScalarType;

    typedef openvdb::tools::PointIndexGrid::TreeType                    PointIndexTree;
    typedef PointIndexTree::LeafNodeType                                PointIndexLeafNode;
    typedef PointIndexLeafNode::ValueType                               PointIndexType;

    typedef openvdb::tree::LeafNode<bool, PointIndexLeafNode::LOG2DIM>  BoolLeafNodeType;

    typedef Attribute<float, DensityOp<float> >                         DensityAttribute;
    typedef Attribute<openvdb::Vec3s>                                   Vec3sAttribute;
    typedef Attribute<float>                                            FloatAttribute;

    typedef std::pair<float, openvdb::Index>                            DensitySample;

    enum DensityTreatment { ACCUMULATE = 0, MAXIMUM, MINIMUM };

    /////

    RasterizePoints(const GU_Detail& detail,
        const PointIndexGridCollection& idxGridCollection,
        std::vector<const BoolLeafNodeType*>& regionMaskLeafNodes,
        const openvdb::math::Transform& volumeXform,
        DensityTreatment treatment,
        const float densityScale = 1.0,
        const float solidRatio = 0.0,
        hvdb::Interrupter* interrupter = NULL)
        : mDetail(&detail)
        , mIdxGridCollection(&idxGridCollection)
        , mRegionMaskNodes(&regionMaskLeafNodes.front())
        , mInterrupter(interrupter)
        , mDensityAttribute(NULL)
        , mVectorAttributes(NULL)
        , mFloatAttributes(NULL)
        , mVEXContext(NULL)
        , mVolumeXform(volumeXform)
        , mDensityScale(densityScale)
        , mSolidRatio(solidRatio)
        , mDensityTreatment(treatment)
    {
    }

    void setDensityAttribute(DensityAttribute& v) { mDensityAttribute = &v; }

    void setVectorAttributes(std::vector<Vec3sAttribute::Ptr>& v) { mVectorAttributes = &v; }

    void setFloatAttributes(std::vector<FloatAttribute::Ptr>& v) { mFloatAttributes = &v; }

    void setVEXContext(VEXContext& v) { mVEXContext = &v; }

    /////

    /// Thread safe copy constructor
    RasterizePoints(const RasterizePoints& rhs)
        : mDetail(rhs.mDetail)
        , mIdxGridCollection(rhs.mIdxGridCollection)
        , mRegionMaskNodes(rhs.mRegionMaskNodes)
        , mInterrupter(rhs.mInterrupter)
        , mDensityAttribute(rhs.mDensityAttribute)
        , mVectorAttributes(rhs.mVectorAttributes)
        , mFloatAttributes(rhs.mFloatAttributes)
        , mVEXContext(rhs.mVEXContext)
        , mVolumeXform(rhs.mVolumeXform)
        , mDensityScale(rhs.mDensityScale)
        , mSolidRatio(rhs.mSolidRatio)
        , mDensityTreatment(rhs.mDensityTreatment)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        // Setup attribute operators

        DensityAttribute::OperatorType::Ptr densityAttribute;
        if (mDensityAttribute) densityAttribute = mDensityAttribute->getAccessor();

        std::vector<Vec3sAttribute::OperatorType::Ptr> vecAttributes;

        if (mVectorAttributes && !mVectorAttributes->empty()) {
            vecAttributes.reserve(mVectorAttributes->size());
            for (size_t n = 0, N = mVectorAttributes->size(); n < N; ++n) {
                vecAttributes.push_back((*mVectorAttributes)[n]->getAccessor());
            }
        }

        std::vector<FloatAttribute::OperatorType::Ptr> floatAttributes;

        if (mFloatAttributes && !mFloatAttributes->empty()) {
            floatAttributes.reserve(mFloatAttributes->size());
            for (size_t n = 0, N = mFloatAttributes->size(); n < N; ++n) {
                floatAttributes.push_back((*mFloatAttributes)[n]->getAccessor());
            }
        }

        const bool transferAttributes = !vecAttributes.empty() || !floatAttributes.empty();

        // Bind optional density attribute

        GA_ROHandleF densityHandle;
        GA_ROAttributeRef densityRef = mDetail->findFloatTuple(GA_ATTRIB_POINT, "density", 1);
        if (densityRef.isValid()) densityHandle.bind(densityRef.getAttribute());

        openvdb::tools::PointIndexIterator<PointIndexTree> pointIndexIter;

        typedef openvdb::tree::ValueAccessor<const PointIndexTree>  IndexTreeAccessor;
        typedef boost::scoped_ptr<IndexTreeAccessor>                IndexTreeAccessorPtr;

        boost::scoped_array<IndexTreeAccessorPtr> accessorList(new IndexTreeAccessorPtr[mIdxGridCollection->size()]);

        for (size_t i = 0; i < mIdxGridCollection->size(); ++i) {
            const PointIndexTree& tree = mIdxGridCollection->idxGrid(i).tree();
            accessorList[i].reset(new IndexTreeAccessor(tree));
        }

        // scratch space
        boost::scoped_array<float> voxelWeightArray(new float[BoolLeafNodeType::SIZE]);
        std::vector<DensitySample> densitySamples;
        densitySamples.reserve(BoolLeafNodeType::SIZE);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            if (this->wasInterrupted()) {
                tbb::task::self().cancel_group_execution();
                break;
            }

            const BoolLeafNodeType& maskNode = *mRegionMaskNodes[n];
            const openvdb::Coord& origin = maskNode.origin();

            if (transferAttributes) {
                memset(voxelWeightArray.get(), 0, BoolLeafNodeType::SIZE * sizeof(float));
            }

            if (densityAttribute) {
                densityAttribute->beginNodeProcessing(origin, n);
            }

            for (size_t i = 0, I = vecAttributes.size(); i < I; ++i) {
                vecAttributes[i]->beginNodeProcessing(origin, n);
            }

            for (size_t i = 0, I = floatAttributes.size(); i < I; ++i) {
                floatAttributes[i]->beginNodeProcessing(origin, n);
            }

            const openvdb::CoordBBox nodeBoundingBox(origin, origin.offsetBy(BoolLeafNodeType::DIM - 1));

            const openvdb::Vec3d bMin = mVolumeXform.indexToWorld(nodeBoundingBox.min());
            const openvdb::Vec3d bMax = mVolumeXform.indexToWorld(nodeBoundingBox.max());

            bool transferData = false;

            for (size_t i = 0; i < mIdxGridCollection->size(); ++i) {

                if (this->wasInterrupted()) break;

                const double sarchRadius = double(mIdxGridCollection->maxRadius(i));
                const PointCache& pointCache = mIdxGridCollection->pointCache(i);
                const openvdb::math::Transform& idxGridTransform = mIdxGridCollection->idxGrid(i).transform();

                const openvdb::CoordBBox searchRegion(
                    idxGridTransform.worldToIndexCellCentered(bMin - sarchRadius),
                    idxGridTransform.worldToIndexCellCentered(bMax + sarchRadius));

                pointIndexIter.searchAndUpdate(searchRegion, *accessorList[i]);

                transferData |= gatherDensityAndAttributes(densityHandle, pointIndexIter, sarchRadius,
                    pointCache, nodeBoundingBox, densityAttribute, vecAttributes, floatAttributes,
                    voxelWeightArray, densitySamples);
            }

            if (transferData && !this->wasInterrupted()) {

                if (densityAttribute) densityAttribute->endNodeProcessing(maskNode);

                if (transferAttributes) {
                    for (size_t n = 0; n < BoolLeafNodeType::SIZE; ++n) {
                        voxelWeightArray[n] = voxelWeightArray[n] > 0.0f ? 1.0f / voxelWeightArray[n] : 0.0f;
                    }

                    for (size_t i = 0, I = vecAttributes.size(); i < I; ++i) {
                        vecAttributes[i]->endNodeProcessing(maskNode, voxelWeightArray.get());
                    }

                    for (size_t i = 0, I = floatAttributes.size(); i < I; ++i) {
                        floatAttributes[i]->endNodeProcessing(maskNode, voxelWeightArray.get());
                    }
                }
            }

        } // end node loop

    } // operator::()


private:
    bool wasInterrupted() const { return mInterrupter && mInterrupter->wasInterrupted(); }

    bool gatherDensityAndAttributes(
        GA_ROHandleF& densityHandle,
        openvdb::tools::PointIndexIterator<PointIndexTree>& pointIndexIter,
        double sarchRadius,
        const PointCache& pointCache,
        const openvdb::CoordBBox& nodeBoundingBox,
        DensityAttribute::OperatorType::Ptr& densityAttribute,
        std::vector<Vec3sAttribute::OperatorType::Ptr>& vecAttributes,
        std::vector<FloatAttribute::OperatorType::Ptr>& floatAttributes,
        boost::scoped_array<float>& voxelWeightArray,
        std::vector<DensitySample>& densitySamples) const
    {
        const bool hasPointDensity = densityHandle.isValid();
        const bool transferVec3sAttributes = !vecAttributes.empty();
        const bool transferFloatAttributes = !floatAttributes.empty();
        const bool transferAttributes = transferVec3sAttributes || transferFloatAttributes;

        bool hasNonzeroDensityValues = false;

        VEXProgram * cvex = mVEXContext ? &mVEXContext->getThereadLocalVEXProgram() : NULL;
        ScalarType * const densityData = densityAttribute ? densityAttribute->data() : NULL;
        const bool exportDensity = densityData != NULL;
        const float * pointRadiusData = pointCache.radiusData();
        const openvdb::Vec3s * pointPosData = pointCache.posData();

        const double dx = mVolumeXform.voxelSize()[0];
        const double dxSqr = dx * dx;

        openvdb::Coord ijk, pMin, pMax;
        PosType center, xyz;

        for (; pointIndexIter; ++pointIndexIter) {

            if (this->wasInterrupted()) break;

            // Get attribute values for the given point offset

            const GA_Offset pointOffset = pointCache.offsetFromIndex(*pointIndexIter);

            if (transferVec3sAttributes) {
                for (size_t i = 0, I = vecAttributes.size(); i < I; ++i) {
                    vecAttributes[i]->updateValue(pointOffset);
                }
            }

            if (transferFloatAttributes) {
                for (size_t i = 0, I = floatAttributes.size(); i < I; ++i) {
                    floatAttributes[i]->updateValue(pointOffset);
                }
            }

            // Compute point properties

            xyz = pointPosData[*pointIndexIter];
            openvdb::Vec3d localPos = mVolumeXform.worldToIndex(xyz);

            ScalarType radius = pointRadiusData[*pointIndexIter];
            const float radiusSqr = radius * radius;

            const ScalarType densityScale = mDensityScale * (hasPointDensity ? densityHandle.get(pointOffset) : 1.0f);
            const ScalarType solidRadius = std::min(radius * mSolidRatio, radius);
            const ScalarType residualRadius = std::max(ScalarType(0.0), radius - solidRadius);
            const ScalarType invResidualRadius = residualRadius > 0.0f ? 1.0f / residualRadius : 0.0f;

            openvdb::Index xPos(0), yPos(0), pos(0);
            double xSqr, ySqr, zSqr;

            densitySamples.clear();

            // Intersect (point + radius) bbox with leafnode bbox to
            // define the overlapping voxel region.

            pMin = mVolumeXform.worldToIndexCellCentered(xyz - sarchRadius);
            pMax = mVolumeXform.worldToIndexCellCentered(xyz + sarchRadius);

            pMin = openvdb::Coord::maxComponent(nodeBoundingBox.min(), pMin);
            pMax = openvdb::Coord::minComponent(nodeBoundingBox.max(), pMax);

            for (ijk[0] = pMin[0]; ijk[0] <= pMax[0]; ++ijk[0]) {

                if (this->wasInterrupted()) break;

                xPos = (ijk[0] & (BoolLeafNodeType::DIM - 1u)) << (2 * BoolLeafNodeType::LOG2DIM);
                xSqr = localPos[0] - double(ijk[0]);
                xSqr *= xSqr;

                for (ijk[1] = pMin[1]; ijk[1] <= pMax[1]; ++ijk[1]) {

                    yPos = xPos + ((ijk[1] & (BoolLeafNodeType::DIM - 1u)) << BoolLeafNodeType::LOG2DIM);
                    ySqr = localPos[1] - double(ijk[1]);
                    ySqr *= ySqr;

                    for (ijk[2] = pMin[2]; ijk[2] <= pMax[2]; ++ijk[2]) {

                        pos = yPos + (ijk[2] & (BoolLeafNodeType::DIM - 1u));
                        zSqr = localPos[2] - double(ijk[2]);
                        zSqr *= zSqr;

                        const float distSqr = float( (xSqr + ySqr + zSqr) * dxSqr );

                        if (distSqr < radiusSqr) {
                            const float dist = std::sqrt(distSqr) - solidRadius;
                            const float weight = dist > 0.0f ? densityScale * (1.0f - invResidualRadius * dist) : 1.0f;
                            if (weight > 0.0f) densitySamples.push_back(DensitySample(weight, pos));
                        }
                    }
                }
            } // end overlapping voxel region loop

            hasNonzeroDensityValues |= !densitySamples.empty();

            // Apply VEX shader program to density samples
            if (cvex && !densitySamples.empty()) {
                hasNonzeroDensityValues |= executeVEXShader(*cvex, densitySamples, exportDensity,
                    vecAttributes, floatAttributes, nodeBoundingBox.min(), xyz, radius, pointOffset);
            }

            // Transfer density data to leafnode buffer
            if (densityData && mDensityTreatment == MAXIMUM) { // max
                for (size_t n = 0, N = densitySamples.size(); n < N; ++n) {
                    const DensitySample& sample = densitySamples[n];
                    ScalarType& value = densityData[sample.second];
                    value = std::max(value, sample.first);
                }
            } else if (densityData && mDensityTreatment == ACCUMULATE) { // add
                for (size_t n = 0, N = densitySamples.size(); n < N; ++n) {
                    const DensitySample& sample = densitySamples[n];
                    densityData[sample.second] += sample.first;
                }
            } else if (densityData && mDensityTreatment == MINIMUM) { // min
                for (size_t n = 0, N = densitySamples.size(); n < N; ++n) {
                    const DensitySample& sample = densitySamples[n];
                    ScalarType& value = densityData[sample.second];
                    value = std::min(value, sample.first);
                }
            }

            // Transfer attribute data to leafnode buffers
            if (transferAttributes && hasNonzeroDensityValues) {

                for (size_t n = 0, N = densitySamples.size(); n < N; ++n) {
                    const DensitySample& sample = densitySamples[n];
                    voxelWeightArray[sample.second] += sample.first;
                }

                for (size_t i = 0, I = vecAttributes.size(); i < I; ++i) {
                    vecAttributes[i]->updateVoxelData(densitySamples);
                }

                for (size_t i = 0, I = floatAttributes.size(); i < I; ++i) {
                    floatAttributes[i]->updateVoxelData(densitySamples);
                }
            }

        } // end point loop

        return hasNonzeroDensityValues;
    } // end gatherDensityAndAttributes method


    bool executeVEXShader(VEXProgram& cvex,
            std::vector<DensitySample>& densitySamples,
            bool exportDensity,
            std::vector<Vec3sAttribute::OperatorType::Ptr>& vecAttributes,
            std::vector<FloatAttribute::OperatorType::Ptr>& floatAttributes,
            const openvdb::Coord& nodeOrigin,
            const PosType& point,
            ScalarType radius,
            GA_Offset pointOffset) const
    {
        bool timeDependantVEX = false;
        const int numValues = int(densitySamples.size());

        if (CVEX_Value* val = cvex.findInput("voxelpos", CVEX_TYPE_VECTOR3)) {

            UT_Vector3* data = cvex.getWorldCoordBuffer();

            openvdb::Coord coord;
            openvdb::Vec3d ws;

            for (int n = 0; n < numValues; ++n) {
                coord = BoolLeafNodeType::offsetToLocalCoord(densitySamples[n].second);
                coord += nodeOrigin;
                ws = mVolumeXform.indexToWorld(coord);

                UT_Vector3& point = data[n];
                point[0] = float(ws[0]);
                point[1] = float(ws[1]);
                point[2] = float(ws[2]);
            }
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
            val->setTypedData(data, numValues);
#else
            val->setData(data, numValues);
#endif
        }

        UT_Vector3 particleCenter(point[0], point[1], point[2]);
        if (CVEX_Value* val = cvex.findInput("pcenter", CVEX_TYPE_VECTOR3)) {
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
            val->setTypedData(&particleCenter, 1);
#else
            val->setData(&particleCenter, numValues);
#endif
        }

        fpreal32 particleRadius(radius);
        if (CVEX_Value* val = cvex.findInput("pradius", CVEX_TYPE_FLOAT)) {
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
            val->setTypedData(&particleRadius, 1);
#else
            val->setData(&particleRadius, numValues);
#endif
        }

        int particleIndex = 0;
        if (CVEX_Value* val = cvex.findInput("pindex", CVEX_TYPE_INTEGER)) {
            particleIndex = int(mDetail->pointIndex(pointOffset));
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
            val->setTypedData(&particleIndex, 1);
#else
            val->setData(&particleIndex, numValues);
#endif
        }

        fpreal32 voxelSize(1.0f);
        if (CVEX_Value* val = cvex.findInput("voxelsize", CVEX_TYPE_FLOAT)) {
            voxelSize = fpreal32(mVolumeXform.voxelSize()[0]);
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
            val->setTypedData(&voxelSize, 1);
#else
            val->setData(&voxelSize, numValues);
#endif
        }

        fpreal32 time = fpreal32(mVEXContext->time());
        if (CVEX_Value* val = cvex.findInput("Time", CVEX_TYPE_FLOAT)) {
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
            val->setTypedData(&time, 1);
#else
            val->setData(&time, numValues);
#endif
            timeDependantVEX = true;
        }

        fpreal32 timeInc = fpreal32(mVEXContext->timeInc());
        if (CVEX_Value* val = cvex.findInput("TimeInc", CVEX_TYPE_FLOAT)) {
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
            val->setTypedData(&timeInc, 1);
#else
            val->setData(&timeInc, numValues);
#endif
            timeDependantVEX = true;
        }

        fpreal32 frame = fpreal32(mVEXContext->frame());
        if (CVEX_Value* val = cvex.findInput("Frame", CVEX_TYPE_FLOAT)) {

#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
            val->setTypedData(&frame, 1);
#else
            val->setData(&frame, numValues);
#endif
            timeDependantVEX = true;
        }

        bool hasNonzeroDensityValues = false, runProcess = false;


        fpreal32* densityScales = NULL;

        if (exportDensity) {
            if (CVEX_Value* val = cvex.findOutput("output", CVEX_TYPE_FLOAT)) {
                runProcess = true;
                densityScales = cvex.getNoiseBuffer();
                for (int n = 0; n < numValues; ++n) densityScales[n] = 1.0f;
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
                val->setTypedData(densityScales, numValues);
#else
                val->setData(densityScales, numValues);
#endif
            }
        }


        for (size_t i = 0, I = vecAttributes.size(); i < I; ++i) {
            if (CVEX_Value* val = cvex.findOutput(vecAttributes[i]->getName(), CVEX_TYPE_VECTOR3)) {
                runProcess = true;
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
                val->setTypedData(vecAttributes[i]->varyingData(), numValues);
#else
                val->setData(vecAttributes[i]->varyingData(), numValues);
#endif
            }
        }


        for (size_t i = 0, I = floatAttributes.size(); i < I; ++i) {
            if (CVEX_Value* val = cvex.findOutput(floatAttributes[i]->getName(), CVEX_TYPE_FLOAT)) {
                runProcess = true;
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
                val->setTypedData(floatAttributes[i]->varyingData(), numValues);
#else
                val->setData(floatAttributes[i]->varyingData(), numValues);
#endif
            }
        }


        if (runProcess) {

            cvex.run(numValues);

            timeDependantVEX |= cvex.isTimeDependant();

            if (densityScales) {
                for (int n = 0; n < numValues; ++n) {
                    densitySamples[n].first *= densityScales[n];
                    hasNonzeroDensityValues |= densitySamples[n].first > 0.0f;
                }
            }
        }

        if (timeDependantVEX) mVEXContext->setTimeDependantFlag();

        return hasNonzeroDensityValues;
    } // end executeVEXShader method

    //////////

    GU_Detail                   const * const mDetail;
    PointIndexGridCollection    const * const mIdxGridCollection;
    BoolLeafNodeType    const * const * const mRegionMaskNodes;
    hvdb::Interrupter                 * const mInterrupter;
    DensityAttribute                  *       mDensityAttribute;
    std::vector<Vec3sAttribute::Ptr>  *       mVectorAttributes;
    std::vector<FloatAttribute::Ptr>  *       mFloatAttributes;
    VEXContext                        *       mVEXContext;

    openvdb::math::Transform            const mVolumeXform;
    ScalarType                          const mDensityScale, mSolidRatio;
    DensityTreatment                    const mDensityTreatment;
}; // struct RasterizePoints


////////////////////////////////////////


inline int
lookupAttrInput(const PRM_SpareData* spare)
{
    const char  *istring;
    if (!spare) return 0;
    istring = spare->getValue("sop_input");
    return istring ? atoi(istring) : 0;
}


inline void
populateMeshMenu(void *data, PRM_Name *choicenames, int themenusize,
    const PRM_SpareData *spare, const PRM_Parm *parm)
{
    choicenames[0].setToken(0);
    choicenames[0].setLabel(0);

    SOP_Node* sop = CAST_SOPNODE((OP_Node *)data);
    if (sop == NULL) return;

    size_t count = 0;

    try {
        const int inputIndex = lookupAttrInput(spare);
        const GU_Detail* gdp = sop->getInputLastGeo(inputIndex, CHgetEvalTime());

        if (gdp) {
            GA_AttributeDict::iterator iter = gdp->pointAttribs().begin(GA_SCOPE_PUBLIC);
            size_t maxSize(themenusize - 1);

            std::vector<std::string> scalarNames, vectorNames;
            scalarNames.reserve(gdp->pointAttribs().entries(GA_SCOPE_PUBLIC));
            vectorNames.reserve(scalarNames.capacity());

            for (; !iter.atEnd(); ++iter) {
                GA_Attribute const * const attrib = iter.attrib();

                if (attrib->getStorageClass() == GA_STORECLASS_FLOAT) {

                    const int tupleSize = attrib->getTupleSize();

#if (UT_VERSION_INT >= 0x0f000000) // 15.0 or later
                    const UT_StringHolder& attribName = attrib->getName();
                    if (tupleSize == 1) scalarNames.push_back(attribName.buffer());
                    else if (tupleSize == 3) vectorNames.push_back(attribName.buffer());
#else
                    if (tupleSize == 1) scalarNames.push_back(attrib->getName());
                    else if (tupleSize == 3) vectorNames.push_back(attrib->getName());
#endif
                }
            }

            std::sort(scalarNames.begin(), scalarNames.end());

            for (size_t n = 0, N = scalarNames.size(); n < N && count < maxSize; ++n) {
                const char * str = scalarNames[n].c_str();
                if (std::strcmp(str, "density") != 0) {
                    choicenames[count].setToken(str);
                    choicenames[count++].setLabel(str);
                }
            }

            if (!scalarNames.empty() && !vectorNames.empty() && count < maxSize) {
                choicenames[count].setToken(PRM_Name::mySeparator);
                choicenames[count++].setLabel(PRM_Name::mySeparator);
            }

            std::sort(vectorNames.begin(), vectorNames.end());

            for (size_t n = 0, N = vectorNames.size(); n < N && count < maxSize; ++n) {
                choicenames[count].setToken(vectorNames[n].c_str());
                choicenames[count++].setLabel(vectorNames[n].c_str());
            }
        }
    } catch (...) {}

    // Terminate the list.
    choicenames[count].setToken(0);
    choicenames[count].setLabel(0);
}

inline bool
hasValidPointAttributes(const GU_Detail& detail, const std::vector<std::string>& attribNames)
{
    for (size_t n = 0, N = attribNames.size(); n < N; ++n) {
        const char* name = attribNames[n].c_str();

        GA_ROAttributeRef attrib = detail.findFloatTuple(GA_ATTRIB_POINT, name, 1);
        if (attrib.isValid()) {
            return true;
        } else {
            attrib = detail.findFloatTuple(GA_ATTRIB_POINT, name, 3);
            if (attrib.isValid()) return true;
        }
    }
    return false;
}


} // unnamed namespace



////////////////////////////////////////

// SOP Implementation

struct SOP_OpenVDB_Rasterize_Points: public hvdb::SOP_NodeVDB
{
    SOP_OpenVDB_Rasterize_Points(OP_Network*, const char* name, OP_Operator*);
    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    // Overriding these are what allow us to contain VOPs

    using hvdb::SOP_NodeVDB::evalVariableValue;

    virtual bool evalVariableValue(UT_String &value, int index, int thread);
    virtual OP_OperatorFilter* getOperatorFilter() { return mCodeGenerator.getOperatorFilter(); }
    virtual const char* getChildType() const { return VOP_OPTYPE_NAME; }
    virtual OP_OpTypeId getChildTypeID() const { return VOP_OPTYPE_ID; }
    virtual VOP_CodeGenerator* getVopCodeGenerator() { return &mCodeGenerator; }
    virtual void opChanged(OP_EventType reason, void *data = 0);

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    virtual bool hasVexShaderParameter(const char* name) { return mCodeGenerator.hasShaderParameter(name); }
#endif

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

    /// VOP and VEX functions
    virtual void finishedLoadingNetwork(bool is_child_call = false);
    virtual void addNode(OP_Node *node, int notify = 1, int explicitly = 1);

#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
    virtual void ensureSpareParmsAreUpdatedSubclass();
#endif

    VOP_CodeGenerator mCodeGenerator;
    int mInitialParmNum;
}; // struct SOP_OpenVDB_Rasterize_Points

////////////////////////////////////////

// VEX related methods

bool
SOP_OpenVDB_Rasterize_Points::evalVariableValue(UT_String &value, int index, int thread)
{
    if (mCodeGenerator.getVariableString(index, value)) return true;
    // else delegate to base class
    return SOP_Node::evalVariableValue(value, index, thread);
}

void
SOP_OpenVDB_Rasterize_Points::opChanged(OP_EventType reason, void *data)
{
    int update_id = mCodeGenerator.beginUpdate();
    SOP_Node::opChanged(reason, data);
    mCodeGenerator.ownerChanged(reason, data);
    mCodeGenerator.endUpdate(update_id);
}

void
SOP_OpenVDB_Rasterize_Points::finishedLoadingNetwork(bool is_child_call)
{
    mCodeGenerator.ownerFinishedLoadingNetwork();
    SOP_Node::finishedLoadingNetwork(is_child_call);
}

void
SOP_OpenVDB_Rasterize_Points::addNode(OP_Node *node, int notify, int explicitly)
{
    mCodeGenerator.beforeAddNode(node);
    SOP_Node::addNode(node, notify, explicitly);
    mCodeGenerator.afterAddNode(node);
}

#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
void
SOP_OpenVDB_Rasterize_Points::ensureSpareParmsAreUpdatedSubclass()
{
    // Check if the spare parameter templates are out-of-date.
    if (getVopCodeGenerator() && eventMicroNode(OP_SPAREPARM_MODIFIED).requiresUpdate(0.0)) {
        // Call into the code generator to update the spare parameter templates.
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
        getVopCodeGenerator()->exportedParmsManager()->updateOwnerSpareParmLayout();
#else
        getVopCodeGenerator()->updateExportedParameterLayout();
#endif
    }
}
#endif

////////////////////////////////////////

void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "pointgroup", "Point Group")
        .setChoiceList(&SOP_Node::pointGroupMenu)
        .setHelpText("A group of points to rasterize."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelsize", "Voxel Size")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 5)
        .setHelpText("The size (length of a side) of the cubic voxels, in world units."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "createdensity", "Create Density Volume")
        .setDefault(PRMoneDefaults)
        .setHelpText("Toggle to enable or disable the density volume generation. "
            "Attribute volumes are still constructed as usual."));

    { // density compositing
        const char* items[] = {
            "add",  "Accumulate",
            "max",  "Maximum",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "compositing", "Density Merge")
            .setDefault(PRMoneDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setHelpText("How to blend point densities in the density volume. "));
    }

    parms.add(hutil::ParmFactory(PRM_FLT_J, "densityscale", "Density Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setHelpText("The density attribute will be scaled by this. If there is "
            "no density, 1 will be used as the reference."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "particlescale", "Particle Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setHelpText("The pscale attribute will be scaled by this. If there is "
            "no pscale, 1 will be used as the reference."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "solidratio", "Solid Ratio")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_RESTRICTED, 1)
        .setHelpText("The amount of the particle that gets full density. "
            "0 means only the very center of the particle will have full density. "
            "1 means the entire particle out to the pscale will have full density."));

    parms.add(hutil::ParmFactory(PRM_STRING, "attributes", "Attributes")
        .setChoiceList(new PRM_ChoiceList(PRM_CHOICELIST_TOGGLE, populateMeshMenu))
        .setHelpText("List of (float or vector) point attributes that will be "
            "rasterized using weighted average blending."));

    /////

    parms.add(hutil::ParmFactory(PRM_HEADING,"noiseheading", ""));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "modeling", "Enable VEX Processing")
        .setDefault(PRMzeroDefaults)
        .setHelpText("Use the contained VOP network to define a VEX procedure that "
            "determines density and attribute values."));

    hvdb::OpenVDBOpFactory("OpenVDB Rasterize Points",
        SOP_OpenVDB_Rasterize_Points::factory, parms, *table)
        .setLocalVariables(VOP_CodeGenerator::theLocalVariables)
        .addInput("points");
}

bool
SOP_OpenVDB_Rasterize_Points::updateParmsFlags()
{
    bool changed = false;
    changed |= enableParm("compositing", evalInt("createdensity", 0, 0));

    const bool createDensity = evalInt("createdensity", 0, 0) != 0;

    bool transferAttributes = false;
    {
        UT_String attributeNameStr;
        evalString(attributeNameStr, "attributes", 0, 0);
        transferAttributes = attributeNameStr.length() > 0;
    }

    bool enableVEX = createDensity || transferAttributes;

    changed |= enableParm("modeling", enableVEX);

    /*const bool proceduralModeling = evalInt("modeling", 0, 0) != 0 && enableVEX;
    for (int i = mInitialParmNum; i < this->getParmList()->getEntries(); ++i) {
        changed |= enableParm(i, proceduralModeling);
    }*/

    return changed;
}

OP_Node*
SOP_OpenVDB_Rasterize_Points::factory(OP_Network* net, const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Rasterize_Points(net, name, op);
}

SOP_OpenVDB_Rasterize_Points::SOP_OpenVDB_Rasterize_Points(OP_Network* net, const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDB(net, name, op)
    , mCodeGenerator(this, new VOP_LanguageContextTypeList(VOP_LANGUAGE_VEX,
        VOPconvertToContextType(VEX_CVEX_CONTEXT)), 1, 1)
    , mInitialParmNum(this->getParmList()->getEntries())
{
    setOperatorTable(getOperatorTable(VOP_TABLE_NAME));
}


OP_ERROR
SOP_OpenVDB_Rasterize_Points::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        gdp->clearAndDestroy();

        const fpreal time = context.getTime();
        const fpreal samplesPerSec = OPgetDirector()->getChannelManager()->getSamplesPerSec();
        const fpreal timeinc = samplesPerSec > 0.0f ? 1.0f / samplesPerSec : 0.0f;
        const fpreal frame = OPgetDirector()->getChannelManager()->getSample(time);

        const GU_Detail* pointsGeo = inputGeo(0);
        const bool createDensity = 0 != evalInt("createdensity", 0, time);
        const bool applyVEX = evalInt("modeling", 0, time);


        std::vector<std::string> attributeNames;
        {
            UT_String attributeNameStr;
            evalString(attributeNameStr, "attributes", 0, time);

            std::vector<std::string> tmpAttributeNames;

            if (attributeNameStr.length() > 0) {
                std::string tmpStr = attributeNameStr.toStdString();
                boost::algorithm::split(tmpAttributeNames, tmpStr, boost::is_any_of(", "));
            }

            std::set<std::string> uniqueAttributeNames(tmpAttributeNames.begin(), tmpAttributeNames.end());

            if (applyVEX && hasParm("process_velocity") && evalInt("process_velocity", 0, time) == 1) {
                uniqueAttributeNames.insert("v");
            }

            attributeNames.insert(attributeNames.end(), uniqueAttributeNames.begin(), uniqueAttributeNames.end());
        }

        if (createDensity || hasValidPointAttributes(*pointsGeo, attributeNames)) {

            const GA_PointGroup* pointGroup = NULL;

            {
                UT_String groupStr;
                evalString(groupStr, "pointgroup", 0, time);
#if (UT_MAJOR_VERSION_INT >= 15)
                pointGroup = parsePointGroups(groupStr, GroupCreator(pointsGeo));
#else
                pointGroup = parsePointGroups(groupStr, const_cast<GU_Detail*>(pointsGeo));
#endif
            }

            const float densityScale = float(evalFloat("densityscale", 0, time));
            const float particleScale = float(evalFloat("particlescale", 0, time));
            const float solidRatio = float(evalFloat("solidratio", 0, time));

            const RasterizePoints::DensityTreatment treatment = evalInt("compositing", 0, time) == 0 ?
                RasterizePoints::ACCUMULATE : RasterizePoints::MAXIMUM;

            const float voxelSize = float(evalFloat("voxelsize", 0, time));


            const openvdb::math::Transform::Ptr volumeTransform =
                openvdb::math::Transform::createLinearTransform(voxelSize);

            /////

            hvdb::Interrupter boss("Rasterize Points");

            // partition points

            PointIndexGridCollection idxGridCollection(*pointsGeo, particleScale,
                voxelSize, pointGroup, &boss);

            // construct region of intrest mask

            PointIndexGridCollection::BoolTreeType::Ptr regionMaskTree;

            if (!boss.wasInterrupted()) {
                regionMaskTree = constructROIMask(idxGridCollection, *volumeTransform, &boss);
            }

            OP_Caller caller(this);
            boost::shared_ptr<VEXContext> vexContextPtr;

            if (applyVEX) {
                UT_String shoppath = "", script = "op:";
                getFullPath(shoppath);
                script += shoppath;
                buildVexCommand(script, getSpareParmTemplates(), time);

                vexContextPtr.reset(new VEXContext(caller, script,
                    PointIndexGridCollection::BoolTreeType::LeafNodeType::SIZE));

                vexContextPtr->setTime(time, timeinc, frame);

#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
                vexContextPtr->setInput(0, pointsGeo);
#endif
            }

            // rasterize points

            if (!boss.wasInterrupted()) {

                const size_t leafNodeCount = regionMaskTree->leafCount();

                typedef Attribute<float, DensityOp<float> > DensityAttributeType;
                typedef Attribute<openvdb::Vec3s>           Vec3sAttribute;
                typedef Attribute<float>                    FloatAttribute;

                DensityAttributeType::Ptr densityAttribute;

                if (createDensity) {
                    densityAttribute = DensityAttributeType::create(
                        GEO_STD_ATTRIB_POSITION, *pointsGeo, leafNodeCount);
                }

                std::vector<Vec3sAttribute::Ptr> vectorAttributes;
                vectorAttributes.reserve(attributeNames.size());

                std::vector<FloatAttribute::Ptr> floatAttributes;
                floatAttributes.reserve(attributeNames.size());

                bool skippedAttributes = false;

                if (!boss.wasInterrupted() && !attributeNames.empty()) {
                    Vec3sAttribute::Ptr vAttr;
                    FloatAttribute::Ptr fAttr;

                    for (size_t n = 0; n < attributeNames.size(); ++n) {
                        vAttr = Vec3sAttribute::create(attributeNames[n], *pointsGeo, leafNodeCount);
                        if (vAttr) {
                            vectorAttributes.push_back(vAttr);
                        } else {
                            fAttr = FloatAttribute::create(attributeNames[n], *pointsGeo, leafNodeCount);
                            if (fAttr) {
                                floatAttributes.push_back(fAttr);
                            } else {
                                skippedAttributes = true;
                                std::string msg = "Skipped '" + attributeNames[n] + "' attribute.";
                                addWarning(SOP_MESSAGE, msg.c_str());
                            }
                        }
                    }
                }

                if (skippedAttributes) {
                    addWarning(SOP_MESSAGE, "Attributes are skipped if the name does not match "
                        "a scalar or vector point attribute.");
                }

                if (!boss.wasInterrupted() && (densityAttribute || !vectorAttributes.empty() || !floatAttributes.empty())) {

                    std::vector<const PointIndexGridCollection::BoolTreeType::LeafNodeType*> regionMaskLeafNodes;

                    regionMaskLeafNodes.reserve(leafNodeCount);
                    regionMaskTree->getNodes(regionMaskLeafNodes);

                    RasterizePoints op(*pointsGeo, idxGridCollection, regionMaskLeafNodes,
                        *volumeTransform, treatment, densityScale, solidRatio, &boss);

                    if (densityAttribute) op.setDensityAttribute(*densityAttribute);
                    if (!vectorAttributes.empty()) op.setVectorAttributes(vectorAttributes);
                    if (!floatAttributes.empty()) op.setFloatAttributes(floatAttributes);
                    if (vexContextPtr) op.setVEXContext(*vexContextPtr);

                    tbb::parallel_for(tbb::blocked_range<size_t>(0, leafNodeCount), op);

                    if (vexContextPtr && vexContextPtr->isTimeDependant()) {
                        OP_Node::flags().timeDep = true;
                    }

                    // export volumes

                    if (!boss.wasInterrupted() && densityAttribute) {
                        densityAttribute->exportVdb(*gdp, volumeTransform);
                    }

                    for (size_t n = 0; n < vectorAttributes.size() && !boss.wasInterrupted(); ++n) {
                        vectorAttributes[n]->exportVdb(*gdp, volumeTransform);
                    }

                    for (size_t n = 0; n < floatAttributes.size() && !boss.wasInterrupted(); ++n) {
                        floatAttributes[n]->exportVdb(*gdp, volumeTransform);
                    }
                }
            }

        } else {
            addWarning(SOP_MESSAGE, "No output volume selected");
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
