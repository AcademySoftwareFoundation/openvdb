// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file GR_PrimVDBPoints.cc
///
/// @author Dan Bailey, Nick Avramoussis
///
/// @brief GR Render Hook and Primitive for VDB PointDataGrid

#include <openvdb/Grid.h>
#include <openvdb/Platform.h>
#include <openvdb/Types.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb_houdini/PointUtils.h>

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
#include <UT/UT_UniquePtr.h>

#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

////////////////////////////////////////

static RE_ShaderHandle theMarkerDecorShader("decor/GL32/point_marker.prog");
static RE_ShaderHandle theNormalDecorShader("decor/GL32/point_normal.prog");
static RE_ShaderHandle theVelocityDecorShader("decor/GL32/user_point_vector3.prog");
static RE_ShaderHandle thePixelShader("particle/GL32/pixel.prog");
static RE_ShaderHandle thePointShader("particle/GL32/point.prog");

/// @note  An additional scale for velocity trails to accurately match
///        the visualization of velocity for Houdini points
#define VELOCITY_DECOR_SCALE -0.041f;

namespace {

/// @note The render hook guard should not be required..

// Declare this at file scope to ensure thread-safe initialization.
std::mutex sRenderHookRegistryMutex;
bool renderHookRegistered = false;

} // anonymous namespace


using namespace openvdb;
using namespace openvdb::points;


////////////////////////////////////////


/// Primitive Render Hook for VDB Points
class GUI_PrimVDBPointsHook : public GUI_PrimitiveHook
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
class GR_PrimVDBPoints : public GR_Primitive
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
    void update(RE_RenderContext, const GT_PrimitiveHandle&, const GR_UpdateParms&) override;

    /// return true if the primitive is in or overlaps the view frustum.
    /// always returning true will effectively disable frustum culling.
    bool inViewFrustum(const UT_Matrix4D &objviewproj,
                       const UT_BoundingBoxD *bbox) override;

    /// Called whenever the primitive is required to render, which may be more
    /// than one time per viewport redraw (beauty, shadow passes, wireframe-over)
    /// It also may be called outside of a viewport redraw to do picking of the
    /// geometry.
    void render(RE_RenderContext, GR_RenderMode, GR_RenderFlags, GR_DrawParms) override;

    int renderPick(RE_RenderContext, const GR_DisplayOption*, unsigned int,
        GR_PickStyle, bool) override { return 0; }

    void renderDecoration(RE_RenderContext, GR_Decoration, const GR_DecorationParms&) override;

protected:
    void computeCentroid(const openvdb::points::PointDataGrid& grid);
    void computeBbox(const openvdb::points::PointDataGrid& grid);

    void updatePosBuffer(RE_Render* r,
                         const openvdb::points::PointDataGrid& grid,
                         const RE_CacheVersion& version);

    bool updateVec3Buffer(RE_Render* r,
                          const openvdb::points::PointDataGrid& grid,
                          const std::string& attributeName,
                          const std::string& bufferName,
                          const RE_CacheVersion& version);

    bool updateVec3Buffer(RE_Render* r,
                          const std::string& attributeName,
                          const std::string& bufferName,
                          const RE_CacheVersion& version);

    void removeBuffer(const std::string& name);

private:
    UT_UniquePtr<RE_Geometry> myGeo;
    bool mDefaultPointColor = true;
    openvdb::Vec3f mCentroid{0, 0, 0};
    openvdb::BBoxd mBbox;
};


////////////////////////////////////////


void
newRenderHook(DM_RenderTable* table)
{
    std::lock_guard<std::mutex> lock(sRenderHookRegistryMutex);

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


static inline bool
grIsPointDataGrid(const GT_PrimitiveHandle& gt_prim)
{
    if (gt_prim->getPrimitiveType() != GT_PRIM_VDB_VOLUME)
        return false;

    const GT_PrimVDB* gt_vdb = static_cast<const GT_PrimVDB*>(gt_prim.get());
    const GEO_PrimVDB* gr_vdb = gt_vdb->getGeoPrimitive();

    return (gr_vdb->getStorageType() == UT_VDB_POINTDATA);
}


GR_Primitive*
GUI_PrimVDBPointsHook::createPrimitive(
    const GT_PrimitiveHandle& gt_prim,
    const GEO_Primitive* geo_prim,
    const GR_RenderInfo* info,
    const char* cache_name,
    GR_PrimAcceptResult& processed)
{
    if (grIsPointDataGrid(gt_prim)) {
        processed = GR_PROCESSED;
        return new GR_PrimVDBPoints(info, cache_name, geo_prim);
    }
    processed = GR_NOT_PROCESSED;
    return nullptr;
}


////////////////////////////////////////


namespace {

using StringPair = std::pair<std::string, std::string>;

bool patchShader(RE_Render* r, RE_ShaderHandle& shader, RE_ShaderType type,
                 const std::vector<StringPair>& stringReplacements,
                 const std::vector<std::string>& stringInsertions = {})
{
    // check if the point shader has already been patched

    r->pushShader();
    r->bindShader(shader);

    const RE_ShaderStage* const patchedShader = shader->getShader("pointOffset", type);
    if (patchedShader) {
        r->popShader();
        return false;
    }

    // retrieve the shader source and version

    UT_String source;
    shader->getShaderSource(r, source, type);
    const int version = shader->getCodeVersion();

    // normalize whitespace (collapse tabs and runs of spaces to single space)

    source.substitute("\t", " ");
    while (source.substitute("  ", " ")) {}

    // patch the shader to replace the strings

    for (const auto& stringPair : stringReplacements) {
        source.substitute(stringPair.first.c_str(), stringPair.second.c_str());
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

    r->popShader();

    if (!success) {
        if (type == RE_SHADER_VERTEX)           std::cerr << "Vertex Shader (";
        else if (type == RE_SHADER_GEOMETRY)    std::cerr << "Geometry Shader (";
        else if (type == RE_SHADER_FRAGMENT)    std::cerr << "Fragment Shader (";
        std::cerr << shader->getName();
        std::cerr << ") Compile Failure: " << message.toStdString() << std::endl;
    }

    assert(success);

    return true;
}

void patchShaderVertexOffset(RE_Render* r, RE_ShaderHandle& shader)
{
    // patch the shader to add a uniform offset to the position

    static const std::vector<StringPair> stringReplacements
    {
        StringPair("void main()", "uniform vec3 offset;\n\nvoid main()"),
        StringPair("vec4(P, 1.0)", "vec4(P + offset, 1.0)"),
        StringPair("vec4(P,1.0)", "vec4(P + offset, 1.0)")
    };

    patchShader(r, shader, RE_SHADER_VERTEX, stringReplacements);
}

void patchShaderNoRedeclarations(RE_Render* r, RE_ShaderHandle& shader)
{
    static const std::vector<StringPair> stringReplacements
    {
        StringPair("uniform vec2 glH_DepthProject;", "//uniform vec2 glH_DepthProject;"),
        StringPair("uniform vec2 glH_ScreenSize", "//uniform vec2 glH_ScreenSize")
    };

    static const std::vector<std::string> stringInsertions
    {
        "uniform vec2 glH_DepthProject;",
        "uniform vec2 glH_ScreenSize;"
    };

    patchShader(r, shader, RE_SHADER_GEOMETRY, stringReplacements, stringInsertions);
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
                  const GT_PrimitiveHandle& gt_prim,
                  const GEO_Primitive*)
{
    if (geo_type == GT_PRIM_VDB_VOLUME && grIsPointDataGrid(gt_prim))
        return GR_PROCESSED;

    return GR_NOT_PROCESSED;
}

void
GR_PrimVDBPoints::computeCentroid(const openvdb::points::PointDataGrid& grid)
{
    // compute the leaf bounding box in index space

    openvdb::CoordBBox coordBBox;
    if (!grid.tree().evalLeafBoundingBox(coordBBox)) {
        mCentroid.init(0.0f, 0.0f, 0.0f);
    }
    else {
        // get the centroid and convert to world space
        mCentroid = openvdb::Vec3f(grid.transform().indexToWorld(coordBBox.getCenter()));
    }
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

        void set(openvdb::Index offset,
                 openvdb::Index /*stride*/,
                 const ValueType& value)
        {
            const ValueType transformedValue = value - mPositionOffset;
            mBuffer[offset * mStride] = UT_Vector3F(transformedValue.x(), transformedValue.y(), transformedValue.z());
        }

    private:
        UT_Vector3F* mBuffer;
        const ValueType& mPositionOffset;
        const Index mStride;
    }; // struct Handle

    PositionAttribute(UT_Vector3F* buffer,
                      const ValueType& positionOffset,
                      Index stride = 1)
        : mBuffer(buffer)
        , mPositionOffset(positionOffset)
        , mStride(stride) { }

    void expand() { }
    void compact() { }

private:
    UT_Vector3F* mBuffer;
    const ValueType mPositionOffset;
    const Index mStride;
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
        void set(openvdb::Index offset,
                 openvdb::Index /*stride*/,
                 const openvdb::math::Vec3<ValueType>& value)
        {
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
    const bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    // Initialize the geometry with the proper name for the GL cache
    if (!myGeo) myGeo.reset(new RE_Geometry);
    myGeo->cacheBuffers(getCacheName());

    using GridType = openvdb::points::PointDataGrid;
    using TreeType = GridType::TreeType;
    using AttributeSet = openvdb::points::AttributeSet;

    const TreeType& tree = grid.tree();
    auto iter = tree.cbeginLeaf();

    if (!iter) return;
    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    // check if group viewport is in use
    const openvdb::StringMetadata::ConstPtr s =
        grid.getMetadata<openvdb::StringMetadata>(openvdb_houdini::META_GROUP_VIEWPORT);

    const std::string groupName = s ? s->value() : "";
    const bool useGroup = !groupName.empty() && descriptor.hasGroup(groupName);

    // count up total points ignoring any leaf nodes that are out of core

    int numPoints = 0;
    if (useGroup) {
        GroupFilter filter(groupName, iter->attributeSet());
        numPoints = static_cast<int>(pointCount(tree, filter));
    } else {
        NullFilter filter;
        numPoints = static_cast<int>(pointCount(tree, filter));
    }

    if (numPoints == 0) return;

    // Initialize the number of points in the geometry.

    myGeo->setNumPoints(numPoints);

    const size_t positionIndex = descriptor.find("P");

    // determine whether position exists

    if (positionIndex == AttributeSet::INVALID_POS) return;

    // fetch point position attribute, if its cache version matches, no upload is required.

    RE_VertexArray* posGeo = myGeo->findCachedAttrib(r, "P", RE_GPU_FLOAT32, 3, RE_ARRAY_POINT, true);

    if (posGeo->getCacheVersion() != version)
    {
        std::vector<Name> includeGroups, excludeGroups;
        if (useGroup) includeGroups.emplace_back(groupName);

        // @note  We've tried using UT_Vector3H here but we get serious aliasing in
        // leaf nodes which are a small distance away from the origin of the VDB
        // primitive (~5-6 nodes away)
        MultiGroupFilter filter(includeGroups, excludeGroups, iter->attributeSet());

        std::vector<Index64> offsets;
        pointOffsets(offsets, grid.tree(), filter);

        UT_UniquePtr<UT_Vector3F[]> pdata(new UT_Vector3F[numPoints]);

        PositionAttribute positionAttribute(pdata.get(), mCentroid);
        convertPointDataGridPosition(positionAttribute, grid, offsets,
                                    /*startOffset=*/ 0, filter);

        posGeo->setArray(r, pdata.get());
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
        myGeo->createConstAttribute(r, "instmat", RE_GPU_MATRIX4, 1, instance.data());
    }

    RE_PrimType primType = RE_PRIM_POINTS;

    myGeo->connectAllPrims(r, RE_GEO_WIRE_IDX, primType, nullptr, true);
}

void
GR_PrimVDBPoints::update(RE_RenderContext r,
             const GT_PrimitiveHandle &primh,
             const GR_UpdateParms &p)
{
    // patch the point shaders at run-time to add an offset (does nothing if already patched)

    patchShaderVertexOffset(r, thePixelShader);
    patchShaderVertexOffset(r, thePointShader);

    // patch the decor shaders at run-time to add an offset etc (does nothing if already patched)

    patchShaderVertexOffset(r, theNormalDecorShader);
    patchShaderVertexOffset(r, theVelocityDecorShader);
    patchShaderVertexOffset(r, theMarkerDecorShader);
    patchShaderNoRedeclarations(r, theMarkerDecorShader);

    // geometry itself changed. GR_GEO_TOPOLOGY changed indicates a large
    // change, such as changes in the point, primitive or vertex counts
    // GR_GEO_CHANGED indicates that some attribute data may have changed.

    if (p.reason & (GR_GEO_CHANGED | GR_GEO_TOPOLOGY_CHANGED))
    {
        const GT_PrimVDB& gt_primVDB = static_cast<const GT_PrimVDB&>(*primh);

        const openvdb::GridBase* grid =
            const_cast<GT_PrimVDB&>((static_cast<const GT_PrimVDB&>(gt_primVDB))).getGrid();

        const openvdb::points::PointDataGrid& pointDataGrid =
            static_cast<const openvdb::points::PointDataGrid&>(*grid);

        computeCentroid(pointDataGrid);
        computeBbox(pointDataGrid);
        updatePosBuffer(r, pointDataGrid, p.geo_version);

        mDefaultPointColor = !updateVec3Buffer(r, pointDataGrid, "Cd", "Cd", p.geo_version);
    }
}

bool
GR_PrimVDBPoints::inViewFrustum(const UT_Matrix4D& objviewproj,
                                const UT_BoundingBoxD *passed_bbox)
{
    const UT_BoundingBoxD bbox( mBbox.min().x(), mBbox.min().y(), mBbox.min().z(),
                                mBbox.max().x(), mBbox.max().y(), mBbox.max().z());
    return GR_Utils::inViewFrustum(passed_bbox ? *passed_bbox : bbox,
                                   objviewproj);
}

bool
GR_PrimVDBPoints::updateVec3Buffer(RE_Render* r,
                                   const openvdb::points::PointDataGrid& grid,
                                   const std::string& attributeName,
                                   const std::string& bufferName,
                                   const RE_CacheVersion& version)
{
    // Initialize the geometry with the proper name for the GL cache
    if (!myGeo) return false;

    using GridType = openvdb::points::PointDataGrid;
    using TreeType = GridType::TreeType;
    using AttributeSet = openvdb::points::AttributeSet;

    const TreeType& tree = grid.tree();
    auto iter = tree.cbeginLeaf();
    if (!iter) return false;

    const int numPoints = myGeo->getNumPoints();
    if (numPoints == 0) return false;

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();
    const size_t index = descriptor.find(attributeName);

    // early exit if attribute does not exist

    if (index == AttributeSet::INVALID_POS) return false;

    // fetch vector attribute, if its cache version matches, no upload is required.

    RE_VertexArray* bufferGeo = myGeo->findCachedAttrib(r, bufferName.c_str(), RE_GPU_FLOAT16, 3, RE_ARRAY_POINT, true);

    if (bufferGeo->getCacheVersion() != version)
    {
        UT_UniquePtr<UT_Vector3H[]> data(new UT_Vector3H[numPoints]);

        const openvdb::Name& type = descriptor.type(index).first;

        if (type == "vec3s") {

            // check if group viewport is in use

            const openvdb::StringMetadata::ConstPtr s =
                grid.getMetadata<openvdb::StringMetadata>(openvdb_houdini::META_GROUP_VIEWPORT);

            const std::string groupName = s ? s->value() : "";
            const bool useGroup = !groupName.empty() && descriptor.hasGroup(groupName);

            std::vector<Name> includeGroups, excludeGroups;
            if (useGroup) includeGroups.emplace_back(groupName);

            MultiGroupFilter filter(includeGroups, excludeGroups, iter->attributeSet());

            std::vector<Index64> offsets;
            pointOffsets(offsets, grid.tree(), filter);

            VectorAttribute<Vec3f> typedAttribute(data.get());
            convertPointDataGridAttribute(typedAttribute, grid.tree(), offsets,
                /*startOffset=*/ 0, static_cast<unsigned>(index), /*stride=*/1,
                filter);
        }

        bufferGeo->setArray(r, data.get());
        bufferGeo->setCacheVersion(version);
    }

    return true;
}

bool
GR_PrimVDBPoints::updateVec3Buffer(RE_Render* r,
                                   const std::string& attributeName,
                                   const std::string& bufferName,
                                   const RE_CacheVersion& version)
{
    const GT_PrimVDB& gt_primVDB = static_cast<const GT_PrimVDB&>(*getCachedGTPrimitive());

    const openvdb::GridBase* grid =
        const_cast<GT_PrimVDB&>((static_cast<const GT_PrimVDB&>(gt_primVDB))).getGrid();

    using PointDataGrid = openvdb::points::PointDataGrid;
    const PointDataGrid& pointDataGrid = static_cast<const PointDataGrid&>(*grid);

    return updateVec3Buffer(r, pointDataGrid, attributeName, bufferName, version);
}

void
GR_PrimVDBPoints::removeBuffer(const std::string& name)
{
    myGeo->clearAttribute(name.c_str());
}

void
GR_PrimVDBPoints::render(RE_RenderContext r, GR_RenderMode, GR_RenderFlags, GR_DrawParms dp)
{
    if (!myGeo)  return;

    const bool gl3 = (getRenderVersion() >= GR_RENDER_GL3);

    if (!gl3)   return;

    const GR_CommonDispOption& commonOpts = dp.opts->common();

    // draw points

    if (myGeo) {

        const bool pointDisplay = commonOpts.particleDisplayType() == GR_PARTICLE_POINTS;

        RE_ShaderHandle* const shader = pointDisplay ? &thePointShader : &thePixelShader;

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
}


void
GR_PrimVDBPoints::renderDecoration(RE_RenderContext r, GR_Decoration decor, const GR_DecorationParms& p)
{
    // just render native GR_Primitive decorations if position not available

    const RE_VertexArray* const position = myGeo->getAttribute("P");
    if (!position) {
        GR_Primitive::renderDecoration(r, decor, p);
        return;
    }

    const GR_CommonDispOption& commonOpts = p.opts->common();
    const RE_CacheVersion version = position->getCacheVersion();

    // update normal buffer

    GR_Decoration normalMarkers[2] = {GR_POINT_NORMAL, GR_NO_DECORATION};
    const bool normalMarkerChanged = standardMarkersChanged(*p.opts, normalMarkers, false);

    if (normalMarkerChanged) {
        const bool drawNormals = p.opts->drawPointNmls() && updateVec3Buffer(r, "N", "N", version);
        if (!drawNormals) {
            removeBuffer("N");
        }
    }

    // update velocity buffer

    GR_Decoration velocityMarkers[2] = {GR_POINT_VELOCITY, GR_NO_DECORATION};
    const bool velocityMarkerChanged = standardMarkersChanged(*p.opts, velocityMarkers, false);

    if (velocityMarkerChanged) {
        const bool drawVelocity = p.opts->drawPointVelocity() && updateVec3Buffer(r, "v", "V", version);
        if (!drawVelocity) {
            removeBuffer("V");
        }
    }

    // setup shader and scale

    RE_ShaderHandle* shader = nullptr;
    float scale = 1.0f;
    UT_Color color;

    if (decor == GR_POINT_MARKER) {
        shader = &theMarkerDecorShader;
        scale = static_cast<float>(commonOpts.markerSize());
        color = commonOpts.getColor(GR_POINT_COLOR);
    }
    else if (decor == GR_POINT_NORMAL) {
        if (static_cast<bool>(myGeo->getAttribute("N"))) {
            shader = &theNormalDecorShader;
            scale = commonOpts.normalScale();
            color = commonOpts.getColor(GR_POINT_COLOR); // No normal enum, use GR_POINT_COLOR
        }
    }
    else if (decor == GR_POINT_VELOCITY) {
        if (static_cast<bool>(myGeo->getAttribute("V"))) {
            shader = &theVelocityDecorShader;
            scale = static_cast<float>(commonOpts.vectorScale()) * VELOCITY_DECOR_SCALE;
            color = commonOpts.getColor(GR_POINT_TRAIL_COLOR);
        }
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

        // enable alpha usage in the fragment shader

        r->pushBlendState();
        r->blendAlpha(/*onoff=*/1);

        // bind the position offset

        const UT_Vector3F positionOffset(mCentroid.x(), mCentroid.y(), mCentroid.z());
        (*shader)->bindVector(r, "offset", positionOffset);

        r->pushUniformColor(RE_UNIFORM_WIRE_COLOR, color);
        r->pushUniformData(RE_UNIFORM_DECORATION_SCALE, &scale);

        // render

        myGeo->draw(r, RE_GEO_WIRE_IDX);

        // pop uniforms, blend state and the shader

        r->popUniform(RE_UNIFORM_WIRE_COLOR);
        r->popUniform(RE_UNIFORM_DECORATION_SCALE);
        r->popBlendState();
        r->popShader();
    }
    else {
        // fall back on default rendering

        GR_Primitive::renderDecoration(r, decor, p);
    }
}

