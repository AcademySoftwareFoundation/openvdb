///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb_houdini/PointUtils.h>

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
#include <GR/GR_Utils.h> // inViewFrustum
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

static RE_ShaderHandle theMarkerDecorShader("decor/GL32/point_marker.prog");
static RE_ShaderHandle theNormalDecorShader("decor/GL32/point_normal.prog");
static RE_ShaderHandle theVelocityDecorShader("decor/GL32/point_normal.prog");

namespace {

RE_ShaderHandle thePixelShader("particle/GL32/pixel.prog");
#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
RE_ShaderHandle thePointShader("particle/GL32/point.prog");
#else
RE_ShaderHandle thePointShader("basic/GL32/const_color.prog");
#endif

RE_ShaderHandle theLineShader("basic/GL32/wire_color.prog");

/// @note The render hook guard should not be required..

// Declare this at file scope to ensure thread-safe initialization.
tbb::mutex sRenderHookRegistryMutex;
bool renderHookRegistered = false;

} // anonymous namespace


using namespace openvdb;
using namespace openvdb::points;


////////////////////////////////////////


/// Primitive Render Hook for VDB Points
class OPENVDB_HOUDINI_API GUI_PrimVDBPointsHook : public GUI_PrimitiveHook
{
public:
    GUI_PrimVDBPointsHook() : GUI_PrimitiveHook("DWA VDB Points") { }
    ~GUI_PrimVDBPointsHook() override = default;

    /// This is called when a new GR_Primitive is required for a VDB Points primitive.
    GR_Primitive* createPrimitive(
        const GT_PrimitiveHandle& gt_prim,
        const GEO_Primitive* geo_prim,
        const GR_RenderInfo* info,
        const char* cache_name,
        GR_PrimAcceptResult& processed) override;
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

    ~GR_PrimVDBPoints() override = default;

    const char *className() const override { return "GR_PrimVDBPoints"; }

    /// See if the tetra primitive can be consumed by this primitive.
    GR_PrimAcceptResult acceptPrimitive(GT_PrimitiveType, int geo_type,
        const GT_PrimitiveHandle&, const GEO_Primitive*) override;

    /// This should reset any lists of primitives.
    void resetPrimitives() override {}

    /// Called whenever the parent detail is changed, draw modes are changed,
    /// selection is changed, or certain volatile display options are changed
    /// (such as level of detail).
    void update(RE_Render*, const GT_PrimitiveHandle&, const GR_UpdateParms&) override;

    /// return true if the primitive is in or overlaps the view frustum.
    /// always returning true will effectively disable frustum culling.
    bool inViewFrustum(const UT_Matrix4D &objviewproj) override;

    /// Called whenever the primitive is required to render, which may be more
    /// than one time per viewport redraw (beauty, shadow passes, wireframe-over)
    /// It also may be called outside of a viewport redraw to do picking of the
    /// geometry.
#if (UT_VERSION_INT >= 0x10000000) // 16.0.0 or later
    void render(RE_Render*, GR_RenderMode, GR_RenderFlags, GR_DrawParms) override;
#else
    void render(RE_Render*, GR_RenderMode, GR_RenderFlags,
        const GR_DisplayOption*, const RE_MaterialList*) override;
#endif

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
#if (UT_VERSION_INT < 0x10000000) // earlier than 16.0.0
    void renderInstances(RE_Render*, GR_RenderMode, GR_RenderFlags,
        const GR_DisplayOption*, const RE_MaterialList*, int) override {}
#endif

    int renderPick(RE_Render*, const GR_DisplayOption*, unsigned int,
        GR_PickStyle, bool) override { return 0; }

    void renderDecoration(RE_Render*, GR_Decoration, const GR_DecorationParms&) override;
#endif

protected:
    void computeCentroid(const openvdb::points::PointDataGrid& grid);
    void computeBbox(const openvdb::points::PointDataGrid& grid);

    void updatePosBuffer(  RE_Render* r,
                        const openvdb::points::PointDataGrid& grid,
                        const RE_CacheVersion& version);

    void updateWireBuffer(    RE_Render* r,
                        const openvdb::points::PointDataGrid& grid,
                        const RE_CacheVersion& version);

    bool updateVec3Buffer(   RE_Render* r,
                            const openvdb::points::PointDataGrid& grid,
                            const std::string& name,
                            const RE_CacheVersion& version);

    bool updateVec3Buffer(   RE_Render* r, const std::string& name, const RE_CacheVersion& version);

    void removeBuffer(const std::string& name);

private:
    std::unique_ptr<RE_Geometry> myGeo;
    std::unique_ptr<RE_Geometry> myWire;
    bool mDefaultPointColor = true;
    openvdb::Vec3f mCentroid{0, 0, 0};
    openvdb::BBoxd mBbox;
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
            /*hook_priority=*/1,
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
        return nullptr;
    }

    const GT_PrimVDB* gtPrimVDB = static_cast<const GT_PrimVDB*>(gt_prim.get());
    const GEO_PrimVDB* primVDB = gtPrimVDB->getGeoPrimitive();

    if (primVDB->getGrid().isType<openvdb::points::PointDataGrid>()) {
        return new GR_PrimVDBPoints(info, cache_name, geo_prim);
    }

    return nullptr;
}


////////////////////////////////////////


namespace {

using StringPair = std::pair<std::string, std::string>;

void patchShader(RE_Render* r, RE_ShaderHandle& shader, RE_ShaderType type,
                 const std::vector<StringPair>& stringReplacements,
                 const std::vector<std::string>& stringInsertions = {})
{
    // check if the point shader has already been patched

    r->pushShader();
    r->bindShader(shader);

    RE_ShaderStage* patchedShader = shader->getShader("pointOffset", type);

    if (patchedShader) {
        r->popShader();
    }
    else {

        // retrieve the shader source

        UT_String source;
        shader->getShaderSource(r, source, type);

        const int version = shader->getCodeVersion();

        r->popShader();

        // patch the shader to replace the strings

        for (const auto& stringPair : stringReplacements) {
            source.substitute(stringPair.first.c_str(), stringPair.second.c_str(), /*all=*/true);
        }

        // patch the shader to insert the strings

        for (const auto& str: stringInsertions) {
            source.insert(0, str.c_str());
        }

        // move the version up to the top of the file

        source.substitute("#version ", "// #version");

        std::stringstream ss;
        ss << "#version " << version << "\n";
        source.insert(0, ss.str().c_str());

        // remove the existing shader and add the patched one

        shader->clearShaders(r, type);

        UT_String message;

        const bool success = shader->addShader(r, type, source, "pointOffset", version, &message);

        if (!success) {
            if (type == RE_SHADER_VERTEX)           std::cerr << "Vertex Shader (";
            else if (type == RE_SHADER_GEOMETRY)    std::cerr << "Geometry Shader (";
            else if (type == RE_SHADER_FRAGMENT)    std::cerr << "Fragment Shader (";
            std::cerr << shader->getName();
            std::cerr << ") Compile Failure: " << message.toStdString() << std::endl;
        }

        assert(success);
    }
}

void patchShaderVertexOffset(RE_Render* r, RE_ShaderHandle& shader)
{
    // patch the shader to add a uniform offset to the position

    std::vector<StringPair> stringReplacements;
    stringReplacements.push_back(StringPair("void main()", "uniform vec3 offset;\n\nvoid main()"));
    stringReplacements.push_back(StringPair("vec4(P, 1.0)", "vec4(P + offset, 1.0)"));
    stringReplacements.push_back(StringPair("vec4(P,1.0)", "vec4(P + offset, 1.0)"));

    patchShader(r, shader, RE_SHADER_VERTEX, stringReplacements);
}

void patchShaderVertexOffsetVelocity(RE_Render* r, RE_ShaderHandle& shader)
{
    // patch the shader to add a uniform offset to the position and swap "N" for "v"

    std::vector<StringPair> stringReplacements;
    stringReplacements.push_back(StringPair("void main()", "uniform vec3 offset;\n\nvoid main()"));
    stringReplacements.push_back(StringPair("vec4(P, 1.0)", "vec4(P + offset, 1.0)"));
    stringReplacements.push_back(StringPair("vec4(P,1.0)", "vec4(P + offset, 1.0)"));
    stringReplacements.push_back(StringPair("N)", "v)"));
    stringReplacements.push_back(StringPair("in vec3 N;", "in vec3 v;"));
    stringReplacements.push_back(StringPair("normalize(", "-0.04 * normalize("));

    patchShader(r, shader, RE_SHADER_VERTEX, stringReplacements);
}

void patchShaderGeomDecorationScale(RE_Render* r, RE_ShaderHandle& shader)
{
    // patch the shader to rename decoration scale

    std::vector<StringPair> stringReplacements;
    stringReplacements.push_back(StringPair("glH_DecorationScale", "decorationScale"));

    patchShader(r, shader, RE_SHADER_GEOMETRY, stringReplacements);
}

void patchShaderGeomDecorationScaleNoRedeclarations(RE_Render* r, RE_ShaderHandle& shader)
{
    // patch the shader to rename decoration scale

    std::vector<StringPair> stringReplacements;
    stringReplacements.push_back(StringPair("glH_DecorationScale", "decorationScale"));
    stringReplacements.push_back(StringPair("\t", " "));
    stringReplacements.push_back(StringPair("  ", " "));
    stringReplacements.push_back(StringPair("  ", " "));
    stringReplacements.push_back(StringPair("uniform vec2 glH_DepthProject;", "//uniform vec2 glH_DepthProject;"));
    stringReplacements.push_back(StringPair("uniform vec2 glH_ScreenSize", "//uniform vec2 glH_ScreenSize"));

    std::vector<std::string> stringInsertions;
    stringInsertions.push_back("uniform vec2 glH_DepthProject;");
    stringInsertions.push_back("uniform vec2 glH_ScreenSize;");

    patchShader(r, shader, RE_SHADER_GEOMETRY, stringReplacements, stringInsertions);
}

void patchShaderFragmentBlue(RE_Render* r, RE_ShaderHandle& shader)
{
    // patch the shader to hard-code the color to blue

    std::vector<StringPair> stringReplacements;
    stringReplacements.push_back(StringPair("col.rgb", "vec3(0,0,1)"));

    patchShader(r, shader, RE_SHADER_FRAGMENT, stringReplacements);
}

void patchShaderFragmentTurqoise(RE_Render* r, RE_ShaderHandle& shader)
{
    // patch the shader to hard-code the color to blue

    std::vector<StringPair> stringReplacements;
    stringReplacements.push_back(StringPair("col.rgb", "vec3(0.2,0.55,0.55)"));

    patchShader(r, shader, RE_SHADER_FRAGMENT, stringReplacements);
}

void patchShaderFragmentBlueDiscard(RE_Render* r, RE_ShaderHandle& shader)
{
    // patch the shader to discard pixels with alpha < 0.25 and hard-code color to blue

    std::vector<StringPair> stringReplacements;
    stringReplacements.push_back(StringPair("color = vec4(fsIn.color.rgb * d, d);",
                                            "if (d < 0.25) discard;\ncolor = vec4(vec3(0,0,1) * d, d);"));

    patchShader(r, shader, RE_SHADER_FRAGMENT, stringReplacements);
}


} // namespace


////////////////////////////////////////


GR_PrimVDBPoints::GR_PrimVDBPoints(
    const GR_RenderInfo *info,
    const char *cache_name,
    const GEO_Primitive*)
    : GR_Primitive(info, cache_name, GA_PrimCompat::TypeMask(0))
{
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

    using LeafNode = typename PointDataTreeType::LeafNodeType;
    using LeafManagerT = typename openvdb::tree::LeafManager<PointDataTreeType>;
    using LeafRangeT = typename LeafManagerT::LeafRange;

    using LeafOffsets = std::vector<std::pair<const LeafNode*, openvdb::Index64>>;

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

            auto handle = openvdb::points::AttributeHandle<AttributeType>::create(
                    leaf->template constAttributeArray(mAttributeIndex));

            openvdb::Vec3f positionVoxelSpace;

            const bool uniform = handle->isUniform();

            if (uniform)    positionVoxelSpace = handle->get(openvdb::Index64(0));

            openvdb::Index64 offset = 0;

            if (useGroup) {
                GroupFilter filter(mGroupName);

                auto iter = leaf->beginIndexOn(filter);

                for (; iter; ++iter)
                {
                    if (!uniform)   positionVoxelSpace = handle->get(openvdb::Index64(*iter));
                    mBuffer[leafOffset + offset++] = voxelSpaceToUTVector(positionVoxelSpace, iter.getCoord().asVec3d(), mTransform);
                }
            }
            else {
                auto iter = leaf->beginIndexOn();

                for (; iter; ++iter)
                {
                    if (!uniform)   positionVoxelSpace = handle->get(openvdb::Index64(*iter));
                    mBuffer[leafOffset + offset++] = voxelSpaceToUTVector(positionVoxelSpace, iter.getCoord().asVec3d(), mTransform);
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

    using LeafNode = typename PointDataTreeType::LeafNodeType;
    using LeafManagerT = typename openvdb::tree::LeafManager<PointDataTreeType>;
    using LeafRangeT = typename LeafManagerT::LeafRange;

    using LeafOffsets = std::vector<std::pair<const LeafNode*, openvdb::Index64>>;

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

            auto handle = openvdb::points::AttributeHandle<AttributeType>::create(
                leaf->constAttributeArray(mAttributeIndex));

            openvdb::Vec3f color;

            const bool uniform = handle->isUniform();

            if (uniform) color = handle->get(openvdb::Index64(0));

            openvdb::Index64 offset = 0;

            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                if (!uniform) color = handle->get(*iter);
                mBuffer[leafOffset + offset] = HoudiniBufferType(color.x(), color.y(), color.z());

                offset++;
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

    using LeafNode = typename PointDataTreeType::LeafNodeType;
    using LeafManagerT = typename openvdb::tree::LeafManager<PointDataTreeType>;
    using LeafRangeT = typename LeafManagerT::LeafRange;

    using LeafOffsets = std::vector<std::pair<const LeafNode*, openvdb::Index64>>;

    FillGPUBuffersId( HoudiniBufferType* buffer,
                            const LeafOffsets& leafOffsets,
                            const PointDataTreeType& pointDataTree,
                            const unsigned attributeIndex)
        : mBuffer(buffer)
        , mLeafOffsets(leafOffsets)
        , mPointDataTree(pointDataTree)
        , mAttributeIndex(attributeIndex) { }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        const long maxId = std::numeric_limits<HoudiniBufferType>::max();

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const LeafNode* leaf = mLeafOffsets[n].first;
            const openvdb::Index64 leafOffset = mLeafOffsets[n].second;

            auto handle = openvdb::points::AttributeHandle<AttributeType>::create(
                leaf->constAttributeArray(mAttributeIndex));

            HoudiniBufferType scalarValue{0};

            // note id attribute (in the GPU cache) is only 32-bit, so use zero if id overflows

            const bool uniform = handle->isUniform();

            if (uniform) {
                const long id = handle->get(openvdb::Index64(0));
                scalarValue = id <= maxId ? HoudiniBufferType(id) : HoudiniBufferType(0);
            }

            openvdb::Index64 offset = 0;

            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                if (!uniform) {
                    const long id = handle->get(*iter);
                    scalarValue = id <= maxId ? HoudiniBufferType(id) : HoudiniBufferType(0);
                }
                mBuffer[leafOffset + offset] = scalarValue;

                offset++;
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
            corners.emplace_back(pos000.x(), pos000.y(), pos000.z());
            const openvdb::Vec3f pos001 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 0.0, 8.0)) - mPositionOffset;
            corners.emplace_back(pos001.x(), pos001.y(), pos001.z());
            const openvdb::Vec3f pos010 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 8.0, 0.0)) - mPositionOffset;
            corners.emplace_back(pos010.x(), pos010.y(), pos010.z());
            const openvdb::Vec3f pos011 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(0.0, 8.0, 8.0)) - mPositionOffset;
            corners.emplace_back(pos011.x(), pos011.y(), pos011.z());
            const openvdb::Vec3f pos100 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 0.0, 0.0)) - mPositionOffset;
            corners.emplace_back(pos100.x(), pos100.y(), pos100.z());
            const openvdb::Vec3f pos101 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 0.0, 8.0)) - mPositionOffset;
            corners.emplace_back(pos101.x(), pos101.y(), pos101.z());
            const openvdb::Vec3f pos110 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 8.0, 0.0)) - mPositionOffset;
            corners.emplace_back(pos110.x(), pos110.y(), pos110.z());
            const openvdb::Vec3f pos111 = mTransform.indexToWorld(origin.asVec3d() + openvdb::Vec3f(8.0, 8.0, 8.0)) - mPositionOffset;
            corners.emplace_back(pos111.x(), pos111.y(), pos111.z());

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
GR_PrimVDBPoints::computeCentroid(const openvdb::points::PointDataGrid& grid)
{
    // compute the leaf bounding box in index space

    openvdb::CoordBBox coordBBox;
    if (!grid.tree().evalLeafBoundingBox(coordBBox)) {
        mCentroid = openvdb::Vec3f(0, 0, 0);
        return;
    }

    // get the centroid and convert to world space

    mCentroid = openvdb::Vec3f{grid.transform().indexToWorld(coordBBox.getCenter())};
}

void
GR_PrimVDBPoints::computeBbox(const openvdb::points::PointDataGrid& grid)
{
    // compute and store the world-space bounding box of the grid

    const CoordBBox bbox = grid.evalActiveVoxelBoundingBox();
    const BBoxd bboxIndex(bbox.min().asVec3d(), bbox.max().asVec3d());
    mBbox = bboxIndex.applyMap(*(grid.transform().baseMap()));
}

struct PositionAttribute
{
    using ValueType = Vec3f;

    struct Handle
    {
        Handle(PositionAttribute& attribute)
            : mBuffer(attribute.mBuffer)
            , mPositionOffset(attribute.mPositionOffset)
            , mStride(attribute.mStride) { }

        void set(openvdb::Index offset, openvdb::Index /*stride*/, const ValueType& value) {
            const size_t vertices = mStride;
            const ValueType transformedValue = value - mPositionOffset;
            mBuffer[offset * vertices] = UT_Vector3H(transformedValue.x(), transformedValue.y(), transformedValue.z());
        }

    private:
        UT_Vector3H* mBuffer;
        ValueType& mPositionOffset;
        Index mStride;
    }; // struct Handle

    PositionAttribute(UT_Vector3H* buffer, const ValueType& positionOffset, Index stride = 1)
        : mBuffer(buffer)
        , mPositionOffset(positionOffset)
        , mStride(stride) { }

    void expand() { }
    void compact() { }

private:
    UT_Vector3H* mBuffer;
    ValueType mPositionOffset;
    Index mStride;
}; // struct PositionAttribute

template <typename T>
struct VectorAttribute
{
    using ValueType = T;

    struct Handle
    {
        Handle(VectorAttribute& attribute)
            : mBuffer(attribute.mBuffer) { }

        template <typename ValueType>
        void set(openvdb::Index offset, openvdb::Index /*stride*/, const openvdb::math::Vec3<ValueType>& value) {
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
             const openvdb::points::PointDataGrid& grid,
             const RE_CacheVersion& version)
{
    bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    // Initialize the geometry with the proper name for the GL cache
    if (!myGeo)
        myGeo.reset(new RE_Geometry);
    myGeo->cacheBuffers(getCacheName());

    using GridType = openvdb::points::PointDataGrid;
    using TreeType = GridType::TreeType;
    using AttributeSet = openvdb::points::AttributeSet;

    const TreeType& tree = grid.tree();

    if (tree.leafCount() == 0)  return;

    auto iter = tree.cbeginLeaf();

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    // check if group viewport is in use

    std::string groupName = "";
    if (openvdb::StringMetadata::ConstPtr s = grid.getMetadata<openvdb::StringMetadata>(openvdb_houdini::META_GROUP_VIEWPORT)) {
        groupName = s->value();
    }
    const bool useGroup = !groupName.empty() && descriptor.hasGroup(groupName);

    // count up total points ignoring any leaf nodes that are out of core

    int numPoints = static_cast<int>(useGroup ?
        groupPointCount(tree, groupName, /*inCoreOnly=*/true) :
        pointCount(tree, /*inCoreOnly=*/true));

    if (numPoints == 0)    return;

    size_t stride = 1;

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

        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups;
        if (useGroup)   includeGroups.push_back(groupName);

        std::vector<Index64> pointOffsets;
        getPointOffsets(pointOffsets, grid.tree(),
                        includeGroups, excludeGroups, /*inCoreOnly=*/true);

        std::unique_ptr<UT_Vector3H[]> pdata(new UT_Vector3H[numPoints]);

        PositionAttribute positionAttribute(pdata.get(), mCentroid, static_cast<Index>(stride));
        convertPointDataGridPosition(positionAttribute, grid, pointOffsets,
                                    /*startOffset=*/ 0, includeGroups, excludeGroups,
                                    /*inCoreOnly=*/true);

        const int maxVertexSize = RE_OGLBuffer::getMaxVertexArraySize(r);

        if (numPoints < maxVertexSize) {
            posGeo->setArray(r, pdata.get(), /*offset = */ 0, /*sublen = */ int(numPoints));
        }
        else {
            for (int offset = 0; offset < numPoints; offset += maxVertexSize) {
                const int sublength = (offset + maxVertexSize) > numPoints ?
                    numPoints - offset : maxVertexSize;

                posGeo->setArray(r, pdata.get()+offset, /*offset=*/ offset, /*sublen=*/ sublength);
            }
        }

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

    RE_PrimType primType = RE_PRIM_POINTS;

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    myGeo->connectAllPrims(r, RE_GEO_WIRE_IDX, primType, nullptr, true);
#else
    myGeo->connectAllPrimsI(r, RE_GEO_WIRE_IDX, primType, nullptr, true);
#endif
}

void
GR_PrimVDBPoints::updateWireBuffer(RE_Render *r,
             const openvdb::points::PointDataGrid& grid,
             const RE_CacheVersion& version)
{
    bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    // Initialize the geometry with the proper name for the GL cache
    if (!myWire)
        myWire.reset(new RE_Geometry);
    myWire->cacheBuffers(getCacheName());

    using GridType = openvdb::points::PointDataGrid;
    using TreeType = GridType::TreeType;
    using LeafNode = TreeType::LeafNodeType;

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

    int numPoints = static_cast<int>(outOfCoreLeaves*8*3);
    myWire->setNumPoints(numPoints);

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

        std::unique_ptr<UT_Vector3H[]> data(new UT_Vector3H[numPoints]);

        std::vector<openvdb::Coord> coords;

        for (TreeType::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
            const LeafNode& leaf = *iter;

            // skip in-core leaf nodes (for use when delay loading VDBs)
            if (!leaf.buffer().isOutOfCore())   continue;

            coords.push_back(leaf.origin());
        }

        FillGPUBuffersLeafBoxes fill(data.get(), coords, grid.transform(), mCentroid);
        const tbb::blocked_range<size_t> range(0, coords.size());
        tbb::parallel_for(range, fill);

        const int maxVertexSize = RE_OGLBuffer::getMaxVertexArraySize(r);
        for (int offset = 0; offset < numPoints; offset += maxVertexSize) {
            const int sublength = (offset + maxVertexSize) > numPoints ?
                                                numPoints - offset : maxVertexSize;

            posWire->setArray(r, data.get()+offset, offset, sublength);
        }

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
    myWire->connectAllPrims(r, RE_GEO_WIRE_IDX, RE_PRIM_LINES, nullptr, true);
#else
    myWire->connectAllPrimsI(r, RE_GEO_WIRE_IDX, RE_PRIM_LINES, nullptr, true);
#endif
}

void
GR_PrimVDBPoints::update(RE_Render *r,
             const GT_PrimitiveHandle &primh,
             const GR_UpdateParms &p)
{
    // patch the point shaders at run-time to add an offset (does nothing if already patched)

    patchShaderVertexOffset(r, theLineShader);
    patchShaderVertexOffset(r, thePixelShader);
    patchShaderVertexOffset(r, thePointShader);

    // patch the decor shaders at run-time to add an offset, change color, etc (does nothing if already patched)

    patchShaderVertexOffset(r, theNormalDecorShader);
    patchShaderGeomDecorationScale(r, theNormalDecorShader);
    patchShaderFragmentBlue(r, theNormalDecorShader);

    patchShaderVertexOffset(r, theMarkerDecorShader);
    patchShaderGeomDecorationScaleNoRedeclarations(r, theMarkerDecorShader);
    patchShaderFragmentBlueDiscard(r, theMarkerDecorShader);

    patchShaderVertexOffsetVelocity(r, theVelocityDecorShader);
    patchShaderGeomDecorationScale(r, theVelocityDecorShader);
    patchShaderFragmentTurqoise(r, theVelocityDecorShader);

    // geometry itself changed. GR_GEO_TOPOLOGY changed indicates a large
    // change, such as changes in the point, primitive or vertex counts
    // GR_GEO_CHANGED indicates that some attribute data may have changed.

    if (p.reason & (GR_GEO_CHANGED | GR_GEO_TOPOLOGY_CHANGED))
    {
        const GT_PrimVDB& gt_primVDB = static_cast<const GT_PrimVDB&>(*primh);

        const openvdb::GridBase* grid =
            const_cast<GT_PrimVDB&>((static_cast<const GT_PrimVDB&>(gt_primVDB))).getGrid();

        using PointDataGrid = openvdb::points::PointDataGrid;

        const PointDataGrid& pointDataGrid = static_cast<const PointDataGrid&>(*grid);

        computeCentroid(pointDataGrid);
        computeBbox(pointDataGrid);
        updatePosBuffer(r, pointDataGrid, p.geo_version);
        updateWireBuffer(r, pointDataGrid, p.geo_version);

        mDefaultPointColor = !updateVec3Buffer(r, pointDataGrid, "Cd", p.geo_version);
    }
}

bool
GR_PrimVDBPoints::inViewFrustum(const UT_Matrix4D& objviewproj)
{
    const UT_BoundingBoxD bbox( mBbox.min().x(), mBbox.min().y(), mBbox.min().z(),
                                mBbox.max().x(), mBbox.max().y(), mBbox.max().z());
    return GR_Utils::inViewFrustum(bbox, objviewproj);
}

bool
GR_PrimVDBPoints::updateVec3Buffer( RE_Render* r,
                                    const openvdb::points::PointDataGrid& grid,
                                    const std::string& name,
                                    const RE_CacheVersion& version)
{
    // Initialize the geometry with the proper name for the GL cache
    if (!myGeo)     return false;

    using GridType = openvdb::points::PointDataGrid;
    using TreeType = GridType::TreeType;
    using AttributeSet = openvdb::points::AttributeSet;

    const TreeType& tree = grid.tree();

    if (tree.leafCount() == 0)  return false;

    const int numPoints = myGeo->getNumPoints();

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
        if (openvdb::StringMetadata::ConstPtr s = grid.getMetadata<openvdb::StringMetadata>(openvdb_houdini::META_GROUP_VIEWPORT)) {
            groupName = s->value();
        }
        const bool useGroup = !groupName.empty() && descriptor.hasGroup(groupName);

        std::unique_ptr<UT_Vector3H[]> data(new UT_Vector3H[numPoints]);

        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups;
        if (useGroup)   includeGroups.push_back(groupName);

        std::vector<Index64> pointOffsets;
        getPointOffsets(pointOffsets, grid.tree(), includeGroups, excludeGroups, /*inCoreOnly=*/true);

        if (type == "vec3s") {
            VectorAttribute<Vec3f> typedAttribute(data.get());
            convertPointDataGridAttribute(typedAttribute, grid.tree(), pointOffsets,
                /*startOffset=*/ 0, static_cast<unsigned>(index), /*stride=*/1,
                includeGroups, excludeGroups, /*inCoreOnly=*/true);
        }

        const int maxVertexSize = RE_OGLBuffer::getMaxVertexArraySize(r);
        for (int offset = 0; offset < numPoints; offset += maxVertexSize) {
            const int sublength = (offset + maxVertexSize) > numPoints ?
                                                numPoints - offset : maxVertexSize;

            bufferGeo->setArray(r, data.get()+offset, offset, sublength);
        }

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

    using PointDataGrid = openvdb::points::PointDataGrid;
    const PointDataGrid& pointDataGrid = static_cast<const PointDataGrid&>(*grid);

    return updateVec3Buffer(r, pointDataGrid, name, version);
}

void
GR_PrimVDBPoints::removeBuffer(const std::string& name)
{
    myGeo->clearAttribute(name.c_str());
}

#if (UT_VERSION_INT >= 0x10000000) // 16.0.0 or later
void
GR_PrimVDBPoints::render(RE_Render *r, GR_RenderMode, GR_RenderFlags, GR_DrawParms dp)
#else
void
GR_PrimVDBPoints::render(RE_Render *r, GR_RenderMode, GR_RenderFlags,
    const GR_DisplayOption* dopts, const RE_MaterialList*)
#endif
{
    if (!myGeo && !myWire)  return;

    bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    if (!gl3)   return;

#if (UT_VERSION_INT >= 0x10000000) // 16.0.0 or later
    const GR_CommonDispOption& commonOpts = dp.opts->common();
#else
    const GR_CommonDispOption& commonOpts = dopts->common();
#endif

    // draw points

    if (myGeo) {

        const bool pointDisplay = commonOpts.particleDisplayType() == GR_PARTICLE_POINTS;

        RE_ShaderHandle* shader;

        if (pointDisplay)       shader = &thePointShader;
        else                    shader = &thePixelShader;

        // bind the shader

        r->pushShader();
        r->bindShader(*shader);

        // bind the position offset

        UT_Vector3F positionOffset(mCentroid.x(), mCentroid.y(), mCentroid.z());
        (*shader)->bindVector(r, "offset", positionOffset);

        // for default point colors, use white if dark viewport background, black otherwise

        if (mDefaultPointColor) {
            const bool darkBackground = (commonOpts.color(GR_BACKGROUND_COLOR) == UT_Color(0));
            fpreal32 white[3] = { 0.6f, 0.6f, 0.5f };
            fpreal32 black[3] = { 0.01f, 0.01f, 0.01f };
            myGeo->createConstAttribute(r, "Cd", RE_GPU_FLOAT32, 3,
                (darkBackground ? white : black));
        }

        if (pointDisplay)      r->pushPointSize(commonOpts.pointSize());

        myGeo->draw(r, RE_GEO_WIRE_IDX);

        if (pointDisplay)      r->popPointSize();

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

        fpreal32 constcol[3] = { 0.6f, 0.6f, 0.6f };
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

    const GR_CommonDispOption& commonOpts = p.opts->common();

    const RE_CacheVersion version = myGeo->getAttribute("P")->getCacheVersion();

    // update normal buffer

    GR_Decoration normalMarkers[2] = {GR_POINT_NORMAL, GR_NO_DECORATION};
    const bool normalMarkerChanged = standardMarkersChanged(*p.opts, normalMarkers, false);

    if (normalMarkerChanged)
    {
        if (p.opts->drawPointNmls())        updateVec3Buffer(r, "N", version);
        else                                removeBuffer("N");
    }

    // update velocity buffer

    GR_Decoration velocityMarkers[2] = {GR_POINT_VELOCITY, GR_NO_DECORATION};
    const bool velocityMarkerChanged = standardMarkersChanged(*p.opts, velocityMarkers, false);

    if (velocityMarkerChanged)
    {
        if (p.opts->drawPointVelocity())    updateVec3Buffer(r, "v", version);
        else                                removeBuffer("v");
    }

    // setup shader and scale

    RE_ShaderHandle* shader = nullptr;
    float scale = 1.0f;

    if (decor == GR_POINT_MARKER) {
        shader = &theMarkerDecorShader;
        scale = static_cast<float>(commonOpts.markerSize());
    }
    else if (decor == GR_POINT_NORMAL) {
        shader = &theNormalDecorShader;
        scale = commonOpts.normalScale();
    }
    else if (decor == GR_POINT_VELOCITY) {
        shader = &theVelocityDecorShader;
        scale = static_cast<float>(commonOpts.vectorScale());

#if (UT_VERSION_INT >= 0x10000000) // 16.0.0 or later
        // use the explicit attribute mapping

        (*shader)->useDefaultAttribMap(false);
        (*shader)->useExplicitAttribMap(true);
#endif
    }
    else if (decor == GR_POINT_NUMBER ||
             decor == GR_POINT_POSITION) {
        // not currently supported
        return;
    }

    if (shader) {
        // bind the shader

        r->pushShader();
        r->bindShader(*shader);

        // bind the position offset and decoration scale

        UT_Vector3F positionOffset(mCentroid.x(), mCentroid.y(), mCentroid.z());
        (*shader)->bindVector(r, "offset", positionOffset);

        fpreal32 decorationScale(scale);
        (*shader)->bindFloat(r, "decorationScale", decorationScale);

        // render and pop the shader

        myGeo->draw(r, RE_GEO_WIRE_IDX);

        r->popShader();
    }
    else {
        // fall back on default rendering

        GR_Primitive::renderDecoration(r, decor, p);
    }
}
#endif


////////////////////////////////////////


#endif // 13.0.0 or later

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
