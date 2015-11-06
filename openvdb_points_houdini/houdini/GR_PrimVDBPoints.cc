///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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
/// @file GR_PrimVDBPoints.cc
///
/// @author Dan Bailey
///
/// @brief GR Render Hook and Primitive for VDB PointDataGrid


#include <UT/UT_Version.h>
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later

#include <openvdb/Grid.h>
#include <openvdb/Platform.h>
#include <openvdb/Types.h>
#include <openvdb_points/tools/PointDataGrid.h>

#include <boost/algorithm/string/predicate.hpp>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glx.h>
#endif

#include <DM/DM_RenderTable.h>
#include <GEO/GEO_PrimVDB.h>
#include <GR/GR_Primitive.h>
#include <GT/GT_PrimitiveTypes.h>
#include <GT/GT_PrimVDB.h>
#include <GUI/GUI_PrimitiveHook.h>
#include <RE/RE_Geometry.h>
#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
#include <RE/RE_BufferCache.h>
#else
#include <RE/RE_GraphicsCache.h>
#endif
#include <RE/RE_Render.h>
#include <RE/RE_ShaderHandle.h>
#include <RE/RE_VertexArray.h>
#include <UT/UT_DSOVersion.h>

#include <tbb/mutex.h>


////////////////////////////////////////


#define THIS_HOOK_NAME "GUI_PrimVDBPointsHook"
#define THIS_PRIMITIVE_NAME "GR_PrimVDBPoints"

static RE_ShaderHandle thePointShader("basic/GL32/const_color.prog");

/// @note The render hook guard should not be required..

// Declare this at file scope to ensure thread-safe initialization.
tbb::mutex sRenderHookRegistryMutex;
static bool renderHookRegistered = false;


////////////////////////////////////////


/// Primitive Render Hook for VDB Points
class OPENVDB_HOUDINI_API GUI_PrimVDBPointsHook : public GUI_PrimitiveHook
{
public:
    GUI_PrimVDBPointsHook() : GUI_PrimitiveHook("VDB Points") { }
    virtual ~GUI_PrimVDBPointsHook() { }

    /// This is called when a new GR_Primitive is required for a VDB Points primitive.
    virtual GR_Primitive* createPrimitive(
        const GT_PrimitiveHandle& gt_prim,
        const GEO_Primitive* geo_prim,
        const GR_RenderInfo* info,
        const char* cache_name,
        GR_PrimAcceptResult& processed);
}; // class GUI_PrimVDBPointsHook


/// Primitive object that is created by GUI_PrimVDBPointsHook whenever a
/// VDB Points primitive is found. This object can be persistent between
/// renders, though display flag changes or navigating though SOPs can cause
/// it to be deleted and recreated later.
class OPENVDB_HOUDINI_API GR_PrimVDBPoints : public GR_Primitive
{
public:
    GR_PrimVDBPoints(const GR_RenderInfo *info,
                     const char *cache_name,
                     const GEO_Primitive* geo_prim);

    virtual ~GR_PrimVDBPoints();

    virtual const char *className() const { return "GR_PrimVDBPoints"; }

    /// See if the tetra primitive can be consumed by this primitive.
    virtual GR_PrimAcceptResult acceptPrimitive(GT_PrimitiveType t,
                                                int geo_type,
                                                const GT_PrimitiveHandle &ph,
                                                const GEO_Primitive *prim);

    /// This should reset any lists of primitives.
    virtual void resetPrimitives() { }

    /// Called whenever the parent detail is changed, draw modes are changed,
    /// selection is changed, or certain volatile display options are changed
    /// (such as level of detail).
    virtual void update(RE_Render *r,
                        const GT_PrimitiveHandle &primh,
                        const GR_UpdateParms &p);

    /// Called whenever the primitive is required to render, which may be more
    /// than one time per viewport redraw (beauty, shadow passes, wireframe-over)
    /// It also may be called outside of a viewport redraw to do picking of the
    /// geometry.
    virtual void render(RE_Render *r,
                        GR_RenderMode render_mode,
                        GR_RenderFlags flags,
                        const GR_DisplayOption *opt,
                        const RE_MaterialList  *materials);

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    virtual void renderInstances(RE_Render*, GR_RenderMode, GR_RenderFlags,
                                 const GR_DisplayOption*, const RE_MaterialList*, int) {}

    virtual int renderPick(RE_Render*, const GR_DisplayOption*, unsigned int,
                           GR_PickStyle, bool) { return 0; }
#endif

private:
    RE_Geometry *myGeo;
};


////////////////////////////////////////


void
#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
newRenderHook(DM_RenderTable* table)
#else
newRenderHook(GR_RenderTable* table)
#endif
{
    tbb::mutex::scoped_lock lock(sRenderHookRegistryMutex);

    if (!renderHookRegistered) {

        static_cast<DM_RenderTable*>(table)->registerGTHook(
            new GUI_PrimVDBPointsHook(),
            GT_PRIM_VDB_VOLUME,
            /*hook_priority=*/0,
            GUI_HOOK_FLAG_AUGMENT_PRIM);

OPENVDB_START_THREADSAFE_STATIC_WRITE
        renderHookRegistered = true; // mutex-protected
OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
    }
}


GR_Primitive*
GUI_PrimVDBPointsHook::createPrimitive(
    const GT_PrimitiveHandle& gt_prim,
    const GEO_Primitive* geo_prim,
    const GR_RenderInfo* info,
    const char* cache_name,
    GR_PrimAcceptResult&)
{
    if (gt_prim->getPrimitiveType() != GT_PRIM_VDB_VOLUME) {
        return NULL;
    }

    const GT_PrimVDB* gtPrimVDB = static_cast<const GT_PrimVDB*>(gt_prim.get());
    const GEO_PrimVDB* primVDB = gtPrimVDB->getGeoPrimitive();

    if (primVDB->getGrid().isType<openvdb::tools::PointDataGrid>()) {
        return new GR_PrimVDBPoints(info, cache_name, geo_prim);
    }

    return NULL;
}


////////////////////////////////////////


GR_PrimVDBPoints::GR_PrimVDBPoints(
    const GR_RenderInfo *info,
    const char *cache_name,
    const GEO_Primitive*)
    : GR_Primitive(info,cache_name, GA_PrimCompat::TypeMask(0))
{
    myGeo = NULL;
}


GR_PrimVDBPoints::~GR_PrimVDBPoints()
{
    delete myGeo;
}


GR_PrimAcceptResult
GR_PrimVDBPoints::acceptPrimitive(GT_PrimitiveType,
                  int geo_type,
                  const GT_PrimitiveHandle&,
                  const GEO_Primitive*)
{
    if (geo_type == GT_PRIM_VDB_VOLUME)
    {
        return GR_PROCESSED;
    }

    return GR_NOT_PROCESSED;
}

namespace gr_primitive_internal {

template<   typename PointDataTreeType,
            typename AttributeType,
            typename HoudiniBufferType>
struct FillGPUBuffersPosition {

    typedef openvdb::tools::PointDataAccessor<const PointDataTreeType> ConstAccessor;

    typedef typename PointDataTreeType::LeafNodeType LeafNode;
    typedef typename openvdb::tree::LeafManager<PointDataTreeType> LeafManagerT;
    typedef typename LeafManagerT::LeafRange LeafRangeT;

    typedef std::vector<std::pair<const LeafNode*, openvdb::Index64> > LeafOffsets;

    FillGPUBuffersPosition( HoudiniBufferType* buffer,
                            const LeafOffsets& leafOffsets,
                            const PointDataTreeType& pointDataTree,
                            const openvdb::math::Transform& transform,
                            const unsigned attributeIndex)
        : mBuffer(buffer)
        , mPointDataTree(pointDataTree)
        , mLeafOffsets(leafOffsets)
        , mTransform(transform)
        , mAcc(pointDataTree)
        , mAttributeIndex(attributeIndex) { }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const LeafNode* leaf = mLeafOffsets[n].first;
            const openvdb::Index64 leafOffset = mLeafOffsets[n].second;

            typename openvdb::tools::AttributeHandle<AttributeType>::Ptr handle =
                openvdb::tools::AttributeHandle<AttributeType>::create(
                    leaf->template attributeArray(mAttributeIndex));

            openvdb::Index64 offset = 0;

            for (typename LeafNode::ValueOnCIter value=leaf->cbeginValueOn(); value; ++value) {

                openvdb::Coord ijk = value.getCoord();

                const openvdb::Vec3f gridIndexSpace = ijk.asVec3d();

                typename ConstAccessor::PointDataIndex pointIndex = mAcc.get(ijk);

                const unsigned start = pointIndex.first;
                const unsigned end = pointIndex.second;

                for (unsigned index = start; index < end; index++)
                {
                    const openvdb::Vec3f positionVoxelSpace = handle->get(openvdb::Index64(index));
                    const openvdb::Vec3f positionIndexSpace = positionVoxelSpace + gridIndexSpace;
                    const openvdb::Vec3f positionWorldSpace = mTransform.indexToWorld(positionIndexSpace);

                    mBuffer[leafOffset + offset] = UT_Vector3F(
                        positionWorldSpace.x(), positionWorldSpace.y(), positionWorldSpace.z());

                    offset++;
                }
            }
        }
    }

    //////////

    HoudiniBufferType*                  mBuffer;
    const LeafOffsets&                   mLeafOffsets;
    const PointDataTreeType&            mPointDataTree;
    const openvdb::math::Transform&     mTransform;
    const ConstAccessor                 mAcc;
    const unsigned                      mAttributeIndex;
}; // class FillGPUBuffersPosition


template<   typename PointDataTreeType,
            typename AttributeType,
            typename HoudiniBufferType>
struct FillGPUBuffersColor {

    typedef openvdb::tools::PointDataAccessor<const PointDataTreeType> ConstAccessor;

    typedef typename PointDataTreeType::LeafNodeType LeafNode;
    typedef typename openvdb::tree::LeafManager<PointDataTreeType> LeafManagerT;
    typedef typename LeafManagerT::LeafRange LeafRangeT;

    typedef std::vector<std::pair<const LeafNode*, openvdb::Index64> > LeafOffsets;

    FillGPUBuffersColor( HoudiniBufferType* buffer,
                            const LeafOffsets& leafOffsets,
                            const PointDataTreeType& pointDataTree,
                            const unsigned attributeIndex)
        : mBuffer(buffer)
        , mPointDataTree(pointDataTree)
        , mLeafOffsets(leafOffsets)
        , mAcc(pointDataTree)
        , mAttributeIndex(attributeIndex) { }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const LeafNode* leaf = mLeafOffsets[n].first;
            const openvdb::Index64 leafOffset = mLeafOffsets[n].second;

            typename openvdb::tools::AttributeHandle<AttributeType>::Ptr handle =
                openvdb::tools::AttributeHandle<AttributeType>::create(
                    leaf->template attributeArray(mAttributeIndex));

            openvdb::Index64 offset = 0;

            for (typename LeafNode::ValueOnCIter value=leaf->cbeginValueOn(); value; ++value) {

                openvdb::Coord ijk = value.getCoord();

                typename ConstAccessor::PointDataIndex pointIndex = mAcc.get(ijk);

                const unsigned start = pointIndex.first;
                const unsigned end = pointIndex.second;

                for (unsigned index = start; index < end; index++)
                {
                    const openvdb::Vec3f color = handle->get(index);
                    mBuffer[leafOffset + offset] = UT_Vector3F(color.x(), color.y(), color.z());

                    offset++;
                }
            }
        }
    }

    //////////

    HoudiniBufferType*                  mBuffer;
    const LeafOffsets&                   mLeafOffsets;
    const PointDataTreeType&            mPointDataTree;
    const ConstAccessor                 mAcc;
    const unsigned                      mAttributeIndex;
}; // class FillGPUBuffersColor

} // namespace gr_primitive_internal

void
GR_PrimVDBPoints::update(RE_Render *r,
             const GT_PrimitiveHandle &primh,
             const GR_UpdateParms &p)
{
    bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    // Initialize the geometry with the proper name for the GL cache
    if (!myGeo)
        myGeo = new RE_Geometry;
    myGeo->cacheBuffers(getCacheName());

    const GT_PrimVDB& gt_primVDB = static_cast<const GT_PrimVDB&>(*primh);

    typedef openvdb::tools::PointDataGrid GridType;
    typedef GridType::TreeType TreeType;
    typedef TreeType::LeafNodeType LeafNode;
    typedef openvdb::tools::PointDataAccessor<const TreeType> AccessorType;
    typedef openvdb::tools::AttributeSet AttributeSet;

    const openvdb::GridBase* grid =
        const_cast<GT_PrimVDB&>((static_cast<const GT_PrimVDB&>(gt_primVDB))).getGrid();

    const GridType& pointDataGrid = static_cast<const GridType&>(*grid);

    const TreeType& tree = pointDataGrid.tree();

    const AccessorType accessor((tree));

    const size_t num_points = accessor.totalPointCount();

    if (num_points <= 0)    return;

    // Initialize the number of points in the geometry.

    myGeo->setNumPoints(int(num_points));

    TreeType::LeafCIter iter = tree.cbeginLeaf();

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    const size_t positionIndex = descriptor.find("P");
    const size_t colorIndex = descriptor.find("Cd");

    // determine whether position and color exist

    bool hasPosition = positionIndex != AttributeSet::INVALID_POS;
    bool hasColor = colorIndex != AttributeSet::INVALID_POS;

    const openvdb::Name colorType = hasColor ? descriptor.type(colorIndex).first : "vec3s";

    if (tree.leafCount() == 0)  return;

    // Fetch P (point position). If its cache version matches, no upload is required.

    RE_VertexArray *pos = NULL;
    RE_VertexArray *col = NULL;

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    pos = myGeo->findCachedAttrib(r, "P", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);
#else
    pos = myGeo->findCachedAttribOrArray(r, "P", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);
#endif

    if (hasColor)
    {
#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
        col = myGeo->findCachedAttrib(r, "Cd", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);
#else
        col = myGeo->findCachedAttribOrArray(r, "Cd", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);
#endif
    }

    if (pos->getCacheVersion() != p.geo_version ||
       (hasColor && col->getCacheVersion() != p.geo_version))
    {
#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
        RE_BufferCache* gCache = RE_BufferCache::getCache();

        size_t availableGraphicsMemory(gCache->getMaxSizeB() -  gCache->getCurSizeB());
#else
        RE_GraphicsCache* gCache = RE_GraphicsCache::getCache();

        size_t availableGraphicsMemory(gCache->getMaxSize() -  gCache->getCurrentSize());
#endif

        size_t sizeOfVector3InBytes = (REsizeOfGPUType(RE_GPU_FLOAT32) * 3) / 8;
        size_t pointAttributeBytes = hasColor ? 2 * sizeOfVector3InBytes : sizeOfVector3InBytes;

        // if the points will not fit into the remaining graphics memory then don't bother doing anything
        if (num_points * pointAttributeBytes > availableGraphicsMemory) return;

        // build cumulative leaf offset array

        typedef std::vector<std::pair<const LeafNode*, openvdb::Index64> > LeafOffsets;

        LeafOffsets offsets;

        openvdb::Index64 cumulativeOffset = 0;

        for (; iter; ++iter) {
            const LeafNode& leaf = *iter;

            const openvdb::Index64 count = leaf.pointCount(openvdb::tools::point_masks::Active);

            offsets.push_back(LeafOffsets::value_type(&leaf, cumulativeOffset));

            cumulativeOffset += count;
        }

        using gr_primitive_internal::FillGPUBuffersPosition;
        using gr_primitive_internal::FillGPUBuffersColor;

        if (hasPosition)
        {
            // map() returns a pointer to the GL buffer
            UT_Vector3F *pdata = static_cast<UT_Vector3F*>(pos->map(r));

            FillGPUBuffersPosition< TreeType,
                                    openvdb::Vec3f,
                                    UT_Vector3F> fill(pdata,
                                                      offsets,
                                                      pointDataGrid.tree(),
                                                      pointDataGrid.transform(),
                                                      positionIndex);

            const tbb::blocked_range<size_t> range(0, offsets.size());
            tbb::parallel_for(range, fill);
        }

        if (hasColor)
        {
            UT_Vector3F *cdata = static_cast<UT_Vector3F*>(col->map(r));

            if (colorType == "vec3s") {
                FillGPUBuffersColor<    TreeType,
                                        openvdb::Vec3f,
                                        UT_Vector3F> fill(cdata,
                                                          offsets,
                                                          pointDataGrid.tree(),
                                                          colorIndex);

                const tbb::blocked_range<size_t> range(0, offsets.size());
                tbb::parallel_for(range, fill);
            }
            else if (colorType == "vec3h") {
                FillGPUBuffersColor<    TreeType,
                        openvdb::math::Vec3<half>,
                        UT_Vector3F> fill(cdata,
                                          offsets,
                                          pointDataGrid.tree(),
                                          colorIndex);

                const tbb::blocked_range<size_t> range(0, offsets.size());
                tbb::parallel_for(range, fill);
            }
        }

        // unmap the buffer so it can be used by GL and set the cache version

        pos->unmap(r);
        pos->setCacheVersion(p.geo_version);

        if (hasColor)
        {
            col->unmap(r);
            col->setCacheVersion(p.geo_version);
        }
    }

    if (gl3)
    {
        // Extra constant inputs for the GL3 default shaders we are using.
        // This isn't required unless

        fpreal32 constcol[3] = { 0.0, 0.0, 0.0 };
        fpreal32 uv[2]  = { 0.0, 0.0 };
        fpreal32 alpha  = 1.0;
        fpreal32 pnt    = 0.0;
        UT_Matrix4F instance;
        instance.identity();

        if (!hasColor)  myGeo->createConstAttribute(r, "Cd",    RE_GPU_FLOAT32, 3, constcol);

        myGeo->createConstAttribute(r, "uv",    RE_GPU_FLOAT32, 2, uv);
        myGeo->createConstAttribute(r, "Alpha", RE_GPU_FLOAT32, 1, &alpha);
        myGeo->createConstAttribute(r, "pointSelection", RE_GPU_FLOAT32, 1,&pnt);
        myGeo->createConstAttribute(r, "instmat", RE_GPU_MATRIX4, 1,
                        instance.data());
    }

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    myGeo->connectAllPrims(r, RE_GEO_WIRE_IDX, RE_PRIM_POINTS, NULL, true);
#else
    myGeo->connectAllPrimsI(r, RE_GEO_WIRE_IDX, RE_PRIM_POINTS, NULL, true);
#endif
}


void
GR_PrimVDBPoints::render(RE_Render *r,
             GR_RenderMode,
             GR_RenderFlags,
             const GR_DisplayOption*,
             const RE_MaterialList*)
{
    if (!myGeo)
    return;

    bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    if (!gl3)   return;

    r->pushShader();
    r->bindShader(thePointShader);

    myGeo->draw(r, RE_GEO_WIRE_IDX);

    r->popShader();
}


////////////////////////////////////////


#endif // 13.0.0 or later

// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
