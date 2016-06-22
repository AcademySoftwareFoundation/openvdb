///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
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
#include <openvdb_points/tools/PointConversion.h>

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
#include <RE/RE_Render.h>
#include <RE/RE_ShaderHandle.h>
#include <RE/RE_VertexArray.h>
#include <UT/UT_DSOVersion.h>

#include <tbb/mutex.h>


////////////////////////////////////////


#define THIS_HOOK_NAME "GUI_PrimVDBPointsHook"
#define THIS_PRIMITIVE_NAME "GR_PrimVDBPoints"

static RE_ShaderHandle thePixelShader("particle/GL32/pixel.prog");
#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
static RE_ShaderHandle thePointShader("particle/GL32/point.prog");
#else
static RE_ShaderHandle thePointShader("basic/GL32/const_color.prog");
#endif

static RE_ShaderHandle theLineShader("basic/GL32/wire_color.prog");

/// @note The render hook guard should not be required..

// Declare this at file scope to ensure thread-safe initialization.
tbb::mutex sRenderHookRegistryMutex;
static bool renderHookRegistered = false;


using namespace openvdb;
using namespace openvdb::tools;

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

    virtual void renderDecoration(RE_Render*, GR_Decoration, const GR_DecorationParms&);
#endif

protected:
    void computeCentroid(const openvdb::tools::PointDataGrid& grid);

    void updatePosBuffer(  RE_Render* r,
                        const openvdb::tools::PointDataGrid& grid,
                        const RE_CacheVersion& version);

    void updateWireBuffer(    RE_Render* r,
                        const openvdb::tools::PointDataGrid& grid,
                        const RE_CacheVersion& version);

    bool updateVec3Buffer(   RE_Render* r,
                            const openvdb::tools::PointDataGrid& grid,
                            const std::string& name,
                            const RE_CacheVersion& version);

    bool updateVec3Buffer(   RE_Render* r, const std::string& name, const RE_CacheVersion& version);

    bool updateIdBuffer(    RE_Render* r,
                            const openvdb::tools::PointDataGrid& grid,
                            const std::string& name,
                            const RE_CacheVersion& version);

    bool updateIdBuffer(    RE_Render* r, const std::string& name, const RE_CacheVersion& version);

    void removeBuffer(const std::string& name);

private:
    RE_Geometry *myGeo;
    RE_Geometry *myWire;
    bool mDefaultPointColor;
    openvdb::Vec3f mCentroid;
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


namespace {

void patchVertexShader(RE_Render* r, RE_ShaderHandle& shader)
{
    // check if the point shader has already been patched

    r->pushShader();
    r->bindShader(shader);

    RE_ShaderStage* patchedVertexShader = shader->getShader("pointOffset", RE_SHADER_VERTEX);

    if (patchedVertexShader) {
        r->popShader();
    }
    else {

        // retrieve the vertex shader source

        UT_String vertexSource;
        shader->getShaderSource(r, vertexSource, RE_SHADER_VERTEX);

        r->popShader();

        // patch the shader to add a uniform offset to the position

        vertexSource.substitute("void main()", "uniform vec3 offset;\n\nvoid main()", /*all=*/false);
        vertexSource.substitute("vec4(P, 1.0)", "vec4(P + offset, 1.0)", /*all=*/false);

        // move the version up to the top of the file

        vertexSource.substitute("#version ", "// #version");
        vertexSource.insert(0, "#version 150\n");

        // remove the existing shader and add the patched one

        shader->clearShaders(r, RE_SHADER_VERTEX);

        UT_String message;

        const bool success = shader->addShader(r, RE_SHADER_VERTEX, vertexSource, "pointOffset", 150, &message);

        if (!success) {
            std::cerr << message.toStdString() << std::endl;
        }

        assert(success);
    }
}

} // namespace


////////////////////////////////////////


GR_PrimVDBPoints::GR_PrimVDBPoints(
    const GR_RenderInfo *info,
    const char *cache_name,
    const GEO_Primitive*)
    : GR_Primitive(info,cache_name, GA_PrimCompat::TypeMask(0))
    , myGeo(NULL)
    , myWire(NULL)
    , mDefaultPointColor(true)
    , mCentroid(openvdb::Vec3f(0, 0, 0))
{
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
                            const openvdb::Vec3f& positionOffset,
                            const unsigned attributeIndex,
                            const std::string& groupName = "")
        : mBuffer(buffer)
        , mPointDataTree(pointDataTree)
        , mLeafOffsets(leafOffsets)
        , mTransform(transform)
        , mPositionOffset(positionOffset)
        , mAttributeIndex(attributeIndex)
        , mGroupName(groupName) { }

    inline UT_Vector3H voxelSpaceToUTVector(const openvdb::Vec3f& positionVoxelSpace,
                                            const openvdb::Vec3f& gridIndexSpace,
                                            const openvdb::math::Transform& transform) const
    {
        const openvdb::Vec3f positionWorldSpace = transform.indexToWorld(positionVoxelSpace + gridIndexSpace) - mPositionOffset;
        return UT_Vector3H(positionWorldSpace.x(), positionWorldSpace.y(), positionWorldSpace.z());
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        const bool useGroup = !mGroupName.empty();

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const LeafNode* leaf = mLeafOffsets[n].first;
            const openvdb::Index64 leafOffset = mLeafOffsets[n].second;

            typename openvdb::tools::AttributeHandle<AttributeType>::Ptr handle =
                openvdb::tools::AttributeHandle<AttributeType>::create(
                    leaf->template attributeArray(mAttributeIndex));

            openvdb::Vec3f positionVoxelSpace;

            const bool uniform = handle->isUniform();

            if (uniform)    positionVoxelSpace = handle->get(openvdb::Index64(0));

            openvdb::Index64 offset = 0;

            for (typename LeafNode::ValueOnCIter value=leaf->cbeginValueOn(); value; ++value) {

                openvdb::Coord ijk = value.getCoord();

                const openvdb::Vec3f gridIndexSpace = ijk.asVec3d();

                openvdb::tools::IndexIter iter = leaf->beginIndex(ijk);

                if (useGroup) {
                    const GroupFilter filter = GroupFilter::create(*leaf, GroupFilter::Data(mGroupName));
                    openvdb::tools::FilterIndexIter<openvdb::tools::IndexIter, GroupFilter> filterIndexIter(iter, filter);

                    for (; filterIndexIter; ++filterIndexIter)
                    {
                        if (!uniform)   positionVoxelSpace = handle->get(openvdb::Index64(*filterIndexIter));
                        mBuffer[leafOffset + offset++] = voxelSpaceToUTVector(positionVoxelSpace, gridIndexSpace, mTransform);
                    }
                }
                else {

                    for (; iter; ++iter)
                    {
                        if (!uniform)   positionVoxelSpace = handle->get(openvdb::Index64(*iter));
                        mBuffer[leafOffset + offset++] = voxelSpaceToUTVector(positionVoxelSpace, gridIndexSpace, mTransform);
                    }
                }
            }
        }
    }

    //////////

    HoudiniBufferType*                  mBuffer;
    const LeafOffsets&                  mLeafOffsets;
    const PointDataTreeType&            mPointDataTree;
    const openvdb::math::Transform&     mTransform;
    const openvdb::Vec3f                mPositionOffset;
    const unsigned                      mAttributeIndex;
    const std::string                   mGroupName;
}; // class FillGPUBuffersPosition


template<   typename PointDataTreeType,
            typename AttributeType,
            typename HoudiniBufferType>
struct FillGPUBuffersVec3 {

    typedef typename PointDataTreeType::LeafNodeType LeafNode;
    typedef typename openvdb::tree::LeafManager<PointDataTreeType> LeafManagerT;
    typedef typename LeafManagerT::LeafRange LeafRangeT;

    typedef std::vector<std::pair<const LeafNode*, openvdb::Index64> > LeafOffsets;

    FillGPUBuffersVec3( HoudiniBufferType* buffer,
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

            openvdb::Vec3f color;

            const bool uniform = handle->isUniform();

            if (uniform)    color = handle->get(openvdb::Index64(0));

            openvdb::Index64 offset = 0;

            for (typename LeafNode::ValueOnCIter value=leaf->cbeginValueOn(); value; ++value) {

                openvdb::Coord ijk = value.getCoord();

                openvdb::tools::IndexIter iter = leaf->beginIndex(ijk);

                for (; iter; ++iter)
                {
                    if (!uniform)   color = handle->get(*iter);
                    mBuffer[leafOffset + offset] = HoudiniBufferType(color.x(), color.y(), color.z());

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
}; // class FillGPUBuffersVec3


template<   typename PointDataTreeType,
            typename AttributeType,
            typename HoudiniBufferType>
struct FillGPUBuffersId {

    typedef typename PointDataTreeType::LeafNodeType LeafNode;
    typedef typename openvdb::tree::LeafManager<PointDataTreeType> LeafManagerT;
    typedef typename LeafManagerT::LeafRange LeafRangeT;

    typedef std::vector<std::pair<const LeafNode*, openvdb::Index64> > LeafOffsets;

    FillGPUBuffersId( HoudiniBufferType* buffer,
                            const LeafOffsets& leafOffsets,
                            const PointDataTreeType& pointDataTree,
                            const unsigned attributeIndex)
        : mBuffer(buffer)
        , mPointDataTree(pointDataTree)
        , mLeafOffsets(leafOffsets)
        , mAttributeIndex(attributeIndex) { }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        const long maxId = std::numeric_limits<HoudiniBufferType>::max();

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const LeafNode* leaf = mLeafOffsets[n].first;
            const openvdb::Index64 leafOffset = mLeafOffsets[n].second;

            typename openvdb::tools::AttributeHandle<AttributeType>::Ptr handle =
                openvdb::tools::AttributeHandle<AttributeType>::create(
                    leaf->template attributeArray(mAttributeIndex));

            HoudiniBufferType scalarValue;

            // note id attribute (in the GPU cache) is only 32-bit, so use zero if id overflows

            const bool uniform = handle->isUniform();

            if (uniform) {
                const long id = handle->get(openvdb::Index64(0));
                scalarValue = id <= maxId ? HoudiniBufferType(id) : HoudiniBufferType(0);
            }

            openvdb::Index64 offset = 0;

            for (typename LeafNode::ValueOnCIter value=leaf->cbeginValueOn(); value; ++value) {

                openvdb::Coord ijk = value.getCoord();

                openvdb::tools::IndexIter iter = leaf->beginIndex(ijk);

                for (; iter; ++iter)
                {
                    if (!uniform) {
                        const long id = handle->get(*iter);
                        scalarValue = id <= maxId ? HoudiniBufferType(id) : HoudiniBufferType(0);
                    }
                    mBuffer[leafOffset + offset] = scalarValue;

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
}; // class FillGPUBuffersId


struct FillGPUBuffersLeafBoxes
{
    FillGPUBuffersLeafBoxes(UT_Vector3H* buffer,
                         const std::vector<openvdb::Coord>& coords,
                         const openvdb::math::Transform& transform,
                         const openvdb::Vec3f& positionOffset)
        : mBuffer(buffer)
        , mCoords(coords)
        , mTransform(transform)
        , mPositionOffset(positionOffset) { }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        std::vector<UT_Vector3H> corners;
        corners.reserve(8);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            const openvdb::Coord& origin = mCoords[n];

            // define 8 corners

            corners.clear();

            const openvdb::Vec3f pos000 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 0.0, 0.0)) - mPositionOffset;
            corners.push_back(UT_Vector3H(pos000.x(), pos000.y(), pos000.z()));
            const openvdb::Vec3f pos001 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 0.0, 8.0)) - mPositionOffset;
            corners.push_back(UT_Vector3H(pos001.x(), pos001.y(), pos001.z()));
            const openvdb::Vec3f pos010 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 8.0, 0.0)) - mPositionOffset;
            corners.push_back(UT_Vector3H(pos010.x(), pos010.y(), pos010.z()));
            const openvdb::Vec3f pos011 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 8.0, 8.0)) - mPositionOffset;
            corners.push_back(UT_Vector3H(pos011.x(), pos011.y(), pos011.z()));
            const openvdb::Vec3f pos100 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 0.0, 0.0)) - mPositionOffset;
            corners.push_back(UT_Vector3H(pos100.x(), pos100.y(), pos100.z()));
            const openvdb::Vec3f pos101 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 0.0, 8.0)) - mPositionOffset;
            corners.push_back(UT_Vector3H(pos101.x(), pos101.y(), pos101.z()));
            const openvdb::Vec3f pos110 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 8.0, 0.0)) - mPositionOffset;
            corners.push_back(UT_Vector3H(pos110.x(), pos110.y(), pos110.z()));
            const openvdb::Vec3f pos111 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 8.0, 8.0)) - mPositionOffset;
            corners.push_back(UT_Vector3H(pos111.x(), pos111.y(), pos111.z()));

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

    UT_Vector3H*                        mBuffer;
    const std::vector<openvdb::Coord>&  mCoords;
    const openvdb::math::Transform&     mTransform;
    const openvdb::Vec3f                mPositionOffset;
}; // class FillGPUBuffersLeafBoxes

} // namespace gr_primitive_internal

void
GR_PrimVDBPoints::computeCentroid(const openvdb::tools::PointDataGrid& grid)
{
    // compute the leaf bounding box in index space

    openvdb::CoordBBox coordBBox;
    if (!grid.tree().evalLeafBoundingBox(coordBBox)) {
        mCentroid = openvdb::Vec3f(0, 0, 0);
        return;
    }

    // get the centroid and convert to world space

    mCentroid = grid.transform().indexToWorld(coordBBox.getCenter());
}

struct PositionAttribute
{
    typedef Vec3f ValueType;

    struct Handle
    {
        Handle(PositionAttribute& attribute)
            : mBuffer(attribute.mBuffer)
            , mPositionOffset(attribute.mPositionOffset) { }

        void set(openvdb::Index offset, const ValueType& value) {
            const ValueType transformedValue = value - mPositionOffset;
            mBuffer[offset] = UT_Vector3H(transformedValue.x(), transformedValue.y(), transformedValue.z());
        }

    private:
        UT_Vector3H* mBuffer;
        ValueType& mPositionOffset;
    }; // struct Handle

    PositionAttribute(UT_Vector3H* buffer, const ValueType& positionOffset)
        : mBuffer(buffer)
        , mPositionOffset(positionOffset) { }

    void expand() { }
    void compact() { }

private:
    UT_Vector3H* mBuffer;
    ValueType mPositionOffset;
}; // struct PositionAttribute

template <typename T>
struct VectorAttribute
{
    typedef T ValueType;

    struct Handle
    {
        Handle(VectorAttribute& attribute)
            : mBuffer(attribute.mBuffer) { }

        template <typename ValueType>
        void set(openvdb::Index offset, const openvdb::math::Vec3<ValueType>& value) {
            mBuffer[offset] = UT_Vector3H(float(value.x()), float(value.y()), float(value.z()));
        }

    private:
        UT_Vector3H* mBuffer;
    }; // struct Handle

    VectorAttribute(UT_Vector3H* buffer)
        : mBuffer(buffer) { }

    void expand() { }
    void compact() { }

private:
    UT_Vector3H* mBuffer;
}; // struct VectorAttribute

void
GR_PrimVDBPoints::updatePosBuffer(RE_Render* r,
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

    TreeType::LeafCIter iter = tree.cbeginLeaf();

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    // check if group viewport is in use

    std::string groupName = "";
    if (openvdb::StringMetadata::ConstPtr s = grid.getMetadata<openvdb::StringMetadata>(openvdb::META_GROUP_VIEWPORT)) {
        groupName = s->value();
    }
    const bool useGroup = !groupName.empty() && descriptor.hasGroup(groupName);

    // count up total points ignoring any leaf nodes that are out of core

    const size_t numPoints = useGroup ? groupPointCount(tree, groupName, /*inCoreOnly=*/true) : pointCount(tree, /*inCoreOnly=*/true);

    if (numPoints == 0)    return;

    // Initialize the number of points in the geometry.

    myGeo->setNumPoints(int(numPoints));

    const size_t positionIndex = descriptor.find("P");

    // determine whether position exists

    bool hasPosition = positionIndex != AttributeSet::INVALID_POS;

    if (!hasPosition)   return;

    // fetch point position attribute, if its cache version matches, no upload is required.

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    RE_VertexArray* posGeo = myGeo->findCachedAttrib(r, "P", RE_GPU_FLOAT16, 3, RE_ARRAY_POINT, true);
#else
    RE_VertexArray* posGeo = myGeo->findCachedAttribOrArray(r, "P", RE_GPU_FLOAT16, 3, RE_ARRAY_POINT, true);
#endif

    if (posGeo->getCacheVersion() != version)
    {
        using gr_primitive_internal::FillGPUBuffersPosition;

        // map() returns a pointer to the GL buffer
        UT_Vector3H *pdata = static_cast<UT_Vector3H*>(posGeo->map(r));

        std::vector<Name> includeGroups;
        if (useGroup)   includeGroups.push_back(groupName);

        std::vector<Index64> pointOffsets;
        getPointOffsets(pointOffsets, grid.tree(), includeGroups);

        PositionAttribute positionAttribute(pdata, mCentroid);
        convertPointDataGridPosition(positionAttribute, grid, pointOffsets,
                                    /*startOffset=*/ 0, includeGroups);

        // unmap the buffer so it can be used by GL and set the cache version

        posGeo->unmap(r);
        posGeo->setCacheVersion(version);
    }

    if (gl3)
    {
        // Extra constant inputs for the GL3 default shaders we are using.

        fpreal32 uv[2]    = { 0.0, 0.0 };
        fpreal32 alpha    = 1.0;
        fpreal32 pnt      = 0.0;
        UT_Matrix4F instance;
        instance.identity();

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
GR_PrimVDBPoints::updateWireBuffer(RE_Render *r,
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
    for (TreeType::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter)
    {
        if (iter->buffer().isOutOfCore())   outOfCoreLeaves++;
    }

    if (outOfCoreLeaves == 0)    return;

    // Initialize the number of points for the wireframe box per leaf.

    myWire->setNumPoints(int(outOfCoreLeaves*8*3));

    // fetch wireframe position, if its cache version matches, no upload is required.

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    RE_VertexArray* posWire = myWire->findCachedAttrib(r, "P", RE_GPU_FLOAT16, 3, RE_ARRAY_POINT, true);
#else
    RE_VertexArray* posWire = myWire->findCachedAttribOrArray(r, "P", RE_GPU_FLOAT16, 3, RE_ARRAY_POINT, true);
#endif

    if (posWire->getCacheVersion() != version)
    {
        using gr_primitive_internal::FillGPUBuffersLeafBoxes;

        // fill the wire data

        UT_Vector3H *data = static_cast<UT_Vector3H*>(posWire->map(r));

        std::vector<openvdb::Coord> coords;

        for (TreeType::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
            const LeafNode& leaf = *iter;

            // skip in-core leaf nodes (for use when delay loading VDBs)
            if (!leaf.buffer().isOutOfCore())   continue;

            coords.push_back(leaf.origin());
        }

        FillGPUBuffersLeafBoxes fill(data, coords, grid.transform(), mCentroid);
        const tbb::blocked_range<size_t> range(0, coords.size());
        tbb::parallel_for(range, fill);

        // unmap the buffer so it can be used by GL and set the cache version

        posWire->unmap(r);
        posWire->setCacheVersion(version);
    }

    if (gl3)
    {
        // Extra constant inputs for the GL3 default shaders we are using.

        fpreal32 uv[2]  = { 0.0, 0.0 };
        fpreal32 alpha  = 1.0;
        fpreal32 pnt    = 0.0;
        UT_Matrix4F instance;
        instance.identity();

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
    // patch the shaders at run-time to add an offset (does nothing if already patched)

    patchVertexShader(r, theLineShader);
    patchVertexShader(r, thePixelShader);
    patchVertexShader(r, thePointShader);

    const GT_PrimVDB& gt_primVDB = static_cast<const GT_PrimVDB&>(*primh);

    const openvdb::GridBase* grid =
        const_cast<GT_PrimVDB&>((static_cast<const GT_PrimVDB&>(gt_primVDB))).getGrid();

    typedef openvdb::tools::PointDataGrid PointDataGrid;
    const PointDataGrid& pointDataGrid = static_cast<const PointDataGrid&>(*grid);

    computeCentroid(pointDataGrid);
    updatePosBuffer(r, pointDataGrid, p.geo_version);
    updateWireBuffer(r, pointDataGrid, p.geo_version);

    mDefaultPointColor = !updateVec3Buffer(r, pointDataGrid, "Cd", p.geo_version);
}

bool
GR_PrimVDBPoints::updateVec3Buffer( RE_Render* r,
                                    const openvdb::tools::PointDataGrid& grid,
                                    const std::string& name,
                                    const RE_CacheVersion& version)
{
    // Initialize the geometry with the proper name for the GL cache
    if (!myGeo)     return false;

    typedef openvdb::tools::PointDataGrid GridType;
    typedef GridType::TreeType TreeType;
    typedef TreeType::LeafNodeType LeafNode;
    typedef openvdb::tools::AttributeSet AttributeSet;

    const TreeType& tree = grid.tree();

    if (tree.leafCount() == 0)  return false;

    const size_t numPoints = myGeo->getNumPoints();

    if (numPoints == 0)         return false;

    TreeType::LeafCIter iter = tree.cbeginLeaf();

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    const size_t index = descriptor.find(name);

    // early exit if attribute does not exist

    if (index == AttributeSet::INVALID_POS)     return false;

    const openvdb::Name type = descriptor.type(index).first;

    // fetch vector attribute, if its cache version matches, no upload is required.

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    RE_VertexArray* bufferGeo = myGeo->findCachedAttrib(r, name.c_str(), RE_GPU_FLOAT16, 3, RE_ARRAY_POINT, true);
#else
    RE_VertexArray* bufferGeo = myGeo->findCachedAttribOrArray(r, name.c_str(), RE_GPU_FLOAT16, 3, RE_ARRAY_POINT, true);
#endif

    if (bufferGeo->getCacheVersion() != version)
    {
        // check if group viewport is in use

        std::string groupName = "";
        if (openvdb::StringMetadata::ConstPtr s = grid.getMetadata<openvdb::StringMetadata>(openvdb::META_GROUP_VIEWPORT)) {
            groupName = s->value();
        }
        const bool useGroup = !groupName.empty() && descriptor.hasGroup(groupName);

        UT_Vector3H *data = static_cast<UT_Vector3H*>(bufferGeo->map(r));

        std::vector<Name> includeGroups;
        if (useGroup)   includeGroups.push_back(groupName);

        std::vector<Index64> pointOffsets;
        getPointOffsets(pointOffsets, grid.tree(), includeGroups);

        if (type == "vec3s") {
            VectorAttribute<Vec3f> typedAttribute(data);
            convertPointDataGridAttribute(typedAttribute, grid.tree(), pointOffsets,
                                         /*startOffset=*/ 0, index, includeGroups);
        }
        else if (type == "vec3h") {
            VectorAttribute<openvdb::math::Vec3<half> > typedAttribute(data);
            convertPointDataGridAttribute(typedAttribute, grid.tree(), pointOffsets,
                                         /*startOffset=*/ 0, index, includeGroups);
        }

        // unmap the buffer so it can be used by GL and set the cache version

        bufferGeo->unmap(r);
        bufferGeo->setCacheVersion(version);
    }

    return true;
}

bool
GR_PrimVDBPoints::updateVec3Buffer(RE_Render* r, const std::string& name, const RE_CacheVersion& version)
{
    const GT_PrimVDB& gt_primVDB = static_cast<const GT_PrimVDB&>(*getCachedGTPrimitive());

    const openvdb::GridBase* grid =
        const_cast<GT_PrimVDB&>((static_cast<const GT_PrimVDB&>(gt_primVDB))).getGrid();

    typedef openvdb::tools::PointDataGrid PointDataGrid;
    const PointDataGrid& pointDataGrid = static_cast<const PointDataGrid&>(*grid);

    return updateVec3Buffer(r, pointDataGrid, name, version);
}

bool
GR_PrimVDBPoints::updateIdBuffer(   RE_Render* r,
                                    const openvdb::tools::PointDataGrid& grid,
                                    const std::string& name,
                                    const RE_CacheVersion& version)
{
    // Initialize the geometry with the proper name for the GL cache
    if (!myGeo)     return false;

    typedef openvdb::tools::PointDataGrid GridType;
    typedef GridType::TreeType TreeType;
    typedef TreeType::LeafNodeType LeafNode;
    typedef openvdb::tools::AttributeSet AttributeSet;

    const TreeType& tree = grid.tree();

    if (tree.leafCount() == 0)  return false;

    const size_t numPoints = myGeo->getNumPoints();

    if (numPoints == 0)         return false;

    TreeType::LeafCIter iter = tree.cbeginLeaf();

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    const size_t index = descriptor.find("id");

    // early exit if id does not exist

    if (index == AttributeSet::INVALID_POS)     return false;

    const openvdb::Name type = descriptor.type(index).first;

    // fetch id, if its cache version matches, no upload is required.

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    RE_VertexArray* bufferGeo = myGeo->findCachedAttrib(r, name.c_str(), RE_GPU_INT32, 1, RE_ARRAY_POINT, true);
#else
    RE_VertexArray* bufferGeo = myGeo->findCachedAttribOrArray(r, name.c_str(), RE_GPU_INT32, 1, RE_ARRAY_POINT, true);
#endif

    if (bufferGeo->getCacheVersion() != version)
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

        using gr_primitive_internal::FillGPUBuffersId;

        int32_t *data = static_cast<int32_t*>(bufferGeo->map(r));

        if (type == "int16") {
            FillGPUBuffersId<   TreeType,
                                int16_t,
                                int32_t> fill(data, offsets, grid.tree(), index);

            const tbb::blocked_range<size_t> range(0, offsets.size());
            tbb::parallel_for(range, fill);
        }
        else if (type == "int32") {
            FillGPUBuffersId<   TreeType,
                                int32_t,
                                int32_t> fill(data, offsets, grid.tree(), index);

            const tbb::blocked_range<size_t> range(0, offsets.size());
            tbb::parallel_for(range, fill);
        }
        else if (type == "int64") {
            FillGPUBuffersId<   TreeType,
                                int64_t,
                                int32_t> fill(data, offsets, grid.tree(), index);

            const tbb::blocked_range<size_t> range(0, offsets.size());
            tbb::parallel_for(range, fill);
        }

        // unmap the buffer so it can be used by GL and set the cache version

        bufferGeo->unmap(r);
        bufferGeo->setCacheVersion(version);
    }

    return true;
}

bool
GR_PrimVDBPoints::updateIdBuffer(RE_Render* r, const std::string& name, const RE_CacheVersion& version)
{
    const GT_PrimVDB& gt_primVDB = static_cast<const GT_PrimVDB&>(*getCachedGTPrimitive());

    const openvdb::GridBase* grid =
        const_cast<GT_PrimVDB&>((static_cast<const GT_PrimVDB&>(gt_primVDB))).getGrid();

    typedef openvdb::tools::PointDataGrid PointDataGrid;
    const PointDataGrid& pointDataGrid = static_cast<const PointDataGrid&>(*grid);

    return updateIdBuffer(r, pointDataGrid, name, version);
}

void
GR_PrimVDBPoints::removeBuffer(const std::string& name)
{
    myGeo->clearAttribute(name.c_str());
}

void
GR_PrimVDBPoints::render(RE_Render *r,
             GR_RenderMode,
             GR_RenderFlags,
             const GR_DisplayOption* dopts,
             const RE_MaterialList*)
{
    if (!myGeo && !myWire)  return;

    bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    if (!gl3)   return;

    const GR_CommonDispOption& commonOpts = dopts->common();

    // draw points

    if (myGeo) {

        const bool pointDisplay = commonOpts.particleDisplayType() == GR_PARTICLE_POINTS;

        RE_ShaderHandle& shader = pointDisplay ? thePointShader : thePixelShader;

        // bind the shader

        r->pushShader();
        r->bindShader(shader);

        // bind the position offset

        UT_Vector3F positionOffset(mCentroid.x(), mCentroid.y(), mCentroid.z());
        shader->bindVector(r, "offset", positionOffset);

        // for default point colors, use white if dark viewport background, black otherwise

        if (mDefaultPointColor) {
            const bool darkBackground = (commonOpts.color(GR_BACKGROUND_COLOR) == UT_Color(0));
            fpreal32 white[3] = { 0.6, 0.6, 0.5 };
            fpreal32 black[3] = { 0.01, 0.01, 0.01 };
            myGeo->createConstAttribute(r, "Cd", RE_GPU_FLOAT32, 3, (darkBackground ? white : black));
        }

        if (pointDisplay)   r->pushPointSize(commonOpts.pointSize());
        myGeo->draw(r, RE_GEO_WIRE_IDX);
        if (pointDisplay)   r->popPointSize();
        r->popShader();
    }

    // draw leaf bboxes

    if (myWire) {

        // bind the shader

        r->pushShader();
        r->bindShader(theLineShader);

        // bind the position offset

        UT_Vector3F positionOffset(mCentroid.x(), mCentroid.y(), mCentroid.z());
        theLineShader->bindVector(r, "offset", positionOffset);

        fpreal32 constcol[3] = { 0.6, 0.6, 0.6 };
        myWire->createConstAttribute(r, "Cd", RE_GPU_FLOAT32, 3, constcol);

        r->pushLineWidth(commonOpts.wireWidth());
        myWire->draw(r, RE_GEO_WIRE_IDX);
        r->popLineWidth();
        r->popShader();
    }
}


#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
void
GR_PrimVDBPoints::renderDecoration(RE_Render* r, GR_Decoration decor, const GR_DecorationParms& p)
{
    // just render native GR_Primitive decorations if position not available

    const bool hasPosition = myGeo->getAttribute("P");
    if (!hasPosition) {
        GR_Primitive::renderDecoration(r, decor, p);
        return;
    }

    const RE_CacheVersion version = myGeo->getAttribute("P")->getCacheVersion();

    // update point number buffer

    GR_Decoration numberMarkers[2] = {GR_POINT_NUMBER, GR_NO_DECORATION};
    const bool numberMarkerChanged = standardMarkersChanged(*p.opts, numberMarkers, false);

    if (numberMarkerChanged)
    {
        if (p.opts->drawPointNums()) {
            updateIdBuffer(r, "pointID", version);
        }
        else {
            removeBuffer("pointID");
        }
    }

    // update normal buffer

    GR_Decoration normalMarkers[2] = {GR_POINT_NORMAL, GR_NO_DECORATION};
    const bool normalMarkerChanged = standardMarkersChanged(*p.opts, normalMarkers, false);

    if (normalMarkerChanged)
    {
        if (p.opts->drawPointNmls()) {
            updateVec3Buffer(r, "N", version);
        }
        else {
            removeBuffer("N");
        }
    }

    // update velocity buffer

    GR_Decoration velocityMarkers[2] = {GR_POINT_VELOCITY, GR_NO_DECORATION};
    const bool velocityMarkerChanged = standardMarkersChanged(*p.opts, velocityMarkers, false);

    if (velocityMarkerChanged)
    {
        if (p.opts->drawPointVelocity()) {
            updateVec3Buffer(r, "v", version);
        }
        else {
            removeBuffer("v");
        }
    }

    // render markers

    if (decor == GR_POINT_NUMBER ||
        decor == GR_POINT_MARKER ||
        decor == GR_POINT_NORMAL ||
        decor == GR_POINT_POSITION ||
        decor == GR_POINT_VELOCITY)
    {
        drawDecorationForGeo(r, myGeo, decor, p.opts, p.render_flags,
                 p.overlay, p.override_vis, p.instance_group,
                 GR_SELECT_NONE);
    }

    GR_Primitive::renderDecoration(r, decor, p);
}
#endif


////////////////////////////////////////


#endif // 13.0.0 or later

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
