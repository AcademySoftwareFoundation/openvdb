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
#include <openvdb_points/tools/PointCount.h>
#include <boost/algorithm/string/predicate.hpp>

#if (UT_VERSION_INT < 0x0f000000) // earlier than 15.0.0
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glx.h>
#endif
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

protected:
    void updatePoints(  RE_Render* r,
                        const openvdb::tools::PointDataGrid& grid,
                        const RE_CacheVersion& version);

    void updateWire(    RE_Render* r,
                        const openvdb::tools::PointDataGrid& grid,
                        const RE_CacheVersion& version);

private:
    RE_Geometry *myGeo;
    RE_Geometry *myWire;
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
    myWire = NULL;
}


GR_PrimVDBPoints::~GR_PrimVDBPoints()
{
    delete myGeo;
    delete myWire;
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

                openvdb::tools::IndexIter iter = leaf->beginIndex(ijk);

                for (; iter; ++iter)
                {
                    const openvdb::Vec3f positionVoxelSpace = handle->get(openvdb::Index64(*iter));
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
    const LeafOffsets&                  mLeafOffsets;
    const PointDataTreeType&            mPointDataTree;
    const openvdb::math::Transform&     mTransform;
    const unsigned                      mAttributeIndex;
}; // class FillGPUBuffersPosition


template<   typename PointDataTreeType,
            typename AttributeType,
            typename HoudiniBufferType>
struct FillGPUBuffersColor {

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

                openvdb::tools::IndexIter iter = leaf->beginIndex(ijk);

                for (; iter; ++iter)
                {
                    const openvdb::Vec3f color = handle->get(*iter);
                    mBuffer[leafOffset + offset] = UT_Vector3F(color.x(), color.y(), color.z());

                    offset++;
                }
            }
        }
    }

    //////////

    HoudiniBufferType*                  mBuffer;
    const LeafOffsets&                  mLeafOffsets;
    const PointDataTreeType&            mPointDataTree;
    const unsigned                      mAttributeIndex;
}; // class FillGPUBuffersColor

struct FillGPUBuffersLeafBoxes
{
    FillGPUBuffersLeafBoxes(UT_Vector3F* buffer,
                         const std::vector<openvdb::Coord>& coords,
                         const openvdb::math::Transform& transform)
        : mBuffer(buffer)
        , mCoords(coords)
        , mTransform(transform) { }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        std::vector<UT_Vector3F> corners;
        corners.reserve(8);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            const openvdb::Coord& origin = mCoords[n];

            // define 8 corners

            const openvdb::Vec3f pos000 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 0.0, 0.0));
            corners.push_back(UT_Vector3F(pos000.x(), pos000.y(), pos000.z()));
            const openvdb::Vec3f pos001 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 0.0, 8.0));
            corners.push_back(UT_Vector3F(pos001.x(), pos001.y(), pos001.z()));
            const openvdb::Vec3f pos010 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 8.0, 0.0));
            corners.push_back(UT_Vector3F(pos010.x(), pos010.y(), pos010.z()));
            const openvdb::Vec3f pos011 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 8.0, 8.0));
            corners.push_back(UT_Vector3F(pos011.x(), pos011.y(), pos011.z()));
            const openvdb::Vec3f pos100 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 0.0, 0.0));
            corners.push_back(UT_Vector3F(pos100.x(), pos100.y(), pos100.z()));
            const openvdb::Vec3f pos101 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 0.0, 8.0));
            corners.push_back(UT_Vector3F(pos101.x(), pos101.y(), pos101.z()));
            const openvdb::Vec3f pos110 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 8.0, 0.0));
            corners.push_back(UT_Vector3F(pos110.x(), pos110.y(), pos110.z()));
            const openvdb::Vec3f pos111 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 8.0, 8.0));
            corners.push_back(UT_Vector3F(pos111.x(), pos111.y(), pos111.z()));

            openvdb::Index64 offset = n*8*3;

            // Z axis

            mBuffer[offset++] = corners[0]; mBuffer[offset++] = corners[1];
            mBuffer[offset++] = corners[2]; mBuffer[offset++] = corners[3];
            mBuffer[offset++] = corners[4]; mBuffer[offset++] = corners[5];
            mBuffer[offset++] = corners[6]; mBuffer[offset++] = corners[7];

            // Y axis

            mBuffer[offset++] = corners[0]; mBuffer[offset++] = corners[2];
            mBuffer[offset++] = corners[1]; mBuffer[offset++] = corners[3];
            mBuffer[offset++] = corners[4]; mBuffer[offset++] = corners[6];
            mBuffer[offset++] = corners[5]; mBuffer[offset++] = corners[7];

            // X axis

            mBuffer[offset++] = corners[0]; mBuffer[offset++] = corners[4];
            mBuffer[offset++] = corners[1]; mBuffer[offset++] = corners[5];
            mBuffer[offset++] = corners[2]; mBuffer[offset++] = corners[6];
            mBuffer[offset++] = corners[3]; mBuffer[offset++] = corners[7];
        }
    }

    //////////

    UT_Vector3F*                        mBuffer;
    const std::vector<openvdb::Coord>&  mCoords;
    const openvdb::math::Transform&     mTransform;
}; // class FillGPUBuffersLeafBoxes

} // namespace gr_primitive_internal

void
GR_PrimVDBPoints::updatePoints(RE_Render* r,
             const openvdb::tools::PointDataGrid& grid,
             const RE_CacheVersion& version)
{
    bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    // Initialize the geometry with the proper name for the GL cache
    if (!myGeo)
        myGeo = new RE_Geometry;
    myGeo->cacheBuffers(getCacheName());

    typedef openvdb::tools::PointDataGrid GridType;
    typedef GridType::TreeType TreeType;
    typedef TreeType::LeafNodeType LeafNode;
    typedef openvdb::tools::AttributeSet AttributeSet;

    const TreeType& tree = grid.tree();

    if (tree.leafCount() == 0)  return;

    // count up total points ignoring any leaf nodes that are out of core

    size_t numPoints = 0;
    for (typename TreeType::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter)
    {
        if (!iter->buffer().isOutOfCore()) {
            numPoints += iter->pointCount();
        }
    }

    if (numPoints == 0)    return;

    // Initialize the number of points in the geometry.

    myGeo->setNumPoints(int(numPoints));

    TreeType::LeafCIter iter = tree.cbeginLeaf();

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    const size_t positionIndex = descriptor.find("P");
    const size_t colorIndex = descriptor.find("Cd");

    // determine whether position and color exist

    bool hasPosition = positionIndex != AttributeSet::INVALID_POS;
    bool hasColor = colorIndex != AttributeSet::INVALID_POS;

    const openvdb::Name colorType = hasColor ? descriptor.type(colorIndex).first : "vec3s";

    // Fetch P (point position). If its cache version matches, no upload is required.

    RE_VertexArray *posGeo = NULL;
    RE_VertexArray *colGeo = NULL;

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    posGeo = myGeo->findCachedAttrib(r, "P", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);
#else
    posGeo = myGeo->findCachedAttribOrArray(r, "P", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);
#endif

    if (hasColor)
    {
#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
        colGeo = myGeo->findCachedAttrib(r, "Cd", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);
#else
        colGeo = myGeo->findCachedAttribOrArray(r, "Cd", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);
#endif
    }

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
    if (numPoints * pointAttributeBytes > availableGraphicsMemory) return;

    if (posGeo->getCacheVersion() != version ||
       (hasColor && colGeo->getCacheVersion() != version))
    {
        // build cumulative leaf offset array

        typedef std::vector<std::pair<const LeafNode*, openvdb::Index64> > LeafOffsets;

        LeafOffsets offsets;

        openvdb::Index64 cumulativeOffset = 0;

        for (; iter; ++iter) {
            const LeafNode& leaf = *iter;

            // skip out-of-core leaf nodes (used when delay loading VDBs)
            if (leaf.buffer().isOutOfCore())    continue;

            const openvdb::Index64 count = leaf.pointCount();

            offsets.push_back(LeafOffsets::value_type(&leaf, cumulativeOffset));

            cumulativeOffset += count;
        }

        using gr_primitive_internal::FillGPUBuffersPosition;
        using gr_primitive_internal::FillGPUBuffersColor;

        if (hasPosition)
        {
            // map() returns a pointer to the GL buffer
            UT_Vector3F *pdata = static_cast<UT_Vector3F*>(posGeo->map(r));

            FillGPUBuffersPosition< TreeType,
                                    openvdb::Vec3f,
                                    UT_Vector3F> fill(pdata,
                                                      offsets,
                                                      grid.tree(),
                                                      grid.transform(),
                                                      positionIndex);

            const tbb::blocked_range<size_t> range(0, offsets.size());
            tbb::parallel_for(range, fill);
        }

        if (hasColor)
        {
            UT_Vector3F *cdata = static_cast<UT_Vector3F*>(colGeo->map(r));

            if (colorType == "vec3s") {
                FillGPUBuffersColor<    TreeType,
                                        openvdb::Vec3f,
                                        UT_Vector3F> fill(cdata,
                                                          offsets,
                                                          grid.tree(),
                                                          colorIndex);

                const tbb::blocked_range<size_t> range(0, offsets.size());
                tbb::parallel_for(range, fill);
            }
            else if (colorType == "vec3h") {
                FillGPUBuffersColor<    TreeType,
                        openvdb::math::Vec3<half>,
                        UT_Vector3F> fill(cdata,
                                          offsets,
                                          grid.tree(),
                                          colorIndex);

                const tbb::blocked_range<size_t> range(0, offsets.size());
                tbb::parallel_for(range, fill);
            }
        }

        // unmap the buffer so it can be used by GL and set the cache version

        posGeo->unmap(r);
        posGeo->setCacheVersion(version);

        if (hasColor)
        {
            colGeo->unmap(r);
            colGeo->setCacheVersion(version);
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

        // TODO: point scale !?

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
GR_PrimVDBPoints::updateWire(RE_Render *r,
             const openvdb::tools::PointDataGrid& grid,
             const RE_CacheVersion& version)
{
    bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    // Initialize the geometry with the proper name for the GL cache
    if (!myWire)
        myWire = new RE_Geometry;
    myWire->cacheBuffers(getCacheName());

    typedef openvdb::tools::PointDataGrid GridType;
    typedef GridType::TreeType TreeType;
    typedef TreeType::LeafNodeType LeafNode;
    typedef openvdb::tools::AttributeSet AttributeSet;

    const TreeType& tree = grid.tree();

    if (tree.leafCount() == 0)  return;

    // count up total points ignoring any leaf nodes that are out of core

    size_t outOfCoreLeaves = 0;
    for (typename TreeType::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter)
    {
        if (iter->buffer().isOutOfCore())   outOfCoreLeaves++;
    }

    if (outOfCoreLeaves == 0)    return;

    // Initialize the number of points for the wireframe box per leaf.

    myWire->setNumPoints(int(outOfCoreLeaves*8*3));

    // Fetch P (point position). If its cache version matches, no upload is required.

    RE_VertexArray *posWire = NULL;

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    posWire = myWire->findCachedAttrib(r, "P", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);
#else
    posWire = myWire->findCachedAttribOrArray(r, "P", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);
#endif

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    RE_BufferCache* gCache = RE_BufferCache::getCache();

    size_t availableGraphicsMemory(gCache->getMaxSizeB() -  gCache->getCurSizeB());
#else
    RE_GraphicsCache* gCache = RE_GraphicsCache::getCache();

    size_t availableGraphicsMemory(gCache->getMaxSize() -  gCache->getCurrentSize());
#endif

    size_t sizeOfVector3InBytes = (REsizeOfGPUType(RE_GPU_FLOAT32) * 3) / 8;
    size_t pointAttributeBytes = sizeOfVector3InBytes;

    // if the points will not fit into the remaining graphics memory then don't bother doing anything
    if (outOfCoreLeaves * 8 * 3 * pointAttributeBytes > availableGraphicsMemory) return;

    if (posWire->getCacheVersion() != version)
    {
        // build cumulative leaf offset array

        typedef std::vector<std::pair<const LeafNode*, openvdb::Index64> > LeafOffsets;

        LeafOffsets offsets;

        openvdb::Index64 cumulativeOffset = 0;

        for (typename TreeType::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
            const LeafNode& leaf = *iter;

            // skip out-of-core leaf nodes (used when delay loading VDBs)
            if (leaf.buffer().isOutOfCore())    continue;

            const openvdb::Index64 count = leaf.pointCount();

            offsets.push_back(LeafOffsets::value_type(&leaf, cumulativeOffset));

            cumulativeOffset += count;
        }

        using gr_primitive_internal::FillGPUBuffersLeafBoxes;

        // fill the wire data

        UT_Vector3F *pdata = static_cast<UT_Vector3F*>(posWire->map(r));

        std::vector<openvdb::Coord> coords;
        coords.reserve(outOfCoreLeaves);

        for (typename TreeType::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
            const LeafNode& leaf = *iter;

            // skip in-core leaf nodes (for use when delay loading VDBs)
            if (!leaf.buffer().isOutOfCore())   continue;

            coords.push_back(leaf.origin());
        }

        assert(coords.size() == outOfCoreLeaves);

        FillGPUBuffersLeafBoxes fill(pdata, coords, grid.transform());
        const tbb::blocked_range<size_t> range(0, coords.size());
        tbb::parallel_for(range, fill);

        // unmap the buffer so it can be used by GL and set the cache version

        posWire->unmap(r);
        posWire->setCacheVersion(version);
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

        myWire->createConstAttribute(r, "Cd",    RE_GPU_FLOAT32, 3, constcol);
        myWire->createConstAttribute(r, "uv",    RE_GPU_FLOAT32, 2, uv);
        myWire->createConstAttribute(r, "Alpha", RE_GPU_FLOAT32, 1, &alpha);
        myWire->createConstAttribute(r, "pointSelection", RE_GPU_FLOAT32, 1,&pnt);
        myWire->createConstAttribute(r, "instmat", RE_GPU_MATRIX4, 1,
                    instance.data());
    }

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    myWire->connectAllPrims(r, RE_GEO_WIRE_IDX, RE_PRIM_LINES, NULL, true);
#else
    myWire->connectAllPrimsI(r, RE_GEO_WIRE_IDX, RE_PRIM_LINES, NULL, true);
#endif
}

void
GR_PrimVDBPoints::update(RE_Render *r,
             const GT_PrimitiveHandle &primh,
             const GR_UpdateParms &p)
{
    const GT_PrimVDB& gt_primVDB = static_cast<const GT_PrimVDB&>(*primh);

    const openvdb::GridBase* grid =
        const_cast<GT_PrimVDB&>((static_cast<const GT_PrimVDB&>(gt_primVDB))).getGrid();

    typedef openvdb::tools::PointDataGrid PointDataGrid;
    const PointDataGrid& pointDataGrid = static_cast<const PointDataGrid&>(*grid);

    updatePoints(r, pointDataGrid, p.geo_version);
    updateWire(r, pointDataGrid, p.geo_version);
}


void
GR_PrimVDBPoints::render(RE_Render *r,
             GR_RenderMode,
             GR_RenderFlags,
             const GR_DisplayOption*,
             const RE_MaterialList*)
{
    if (!myGeo && !myWire)  return;

    bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    if (!gl3)   return;

    // TODO: replace sprites with spheres and remove manual point size

    r->pushShader();
    r->pushPointSize(2.0f);
    r->bindShader(thePointShader);

    if (myGeo)  myGeo->draw(r, RE_GEO_WIRE_IDX);
    if (myWire) myWire->draw(r, RE_GEO_WIRE_IDX);

    r->popPointSize();
    r->popShader();
}


////////////////////////////////////////


#endif // 13.0.0 or later

// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
