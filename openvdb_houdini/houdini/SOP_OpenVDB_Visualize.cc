///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Visualize.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Visualize VDB grids and their tree topology

#include <houdini_utils/ParmFactory.h>
#include <houdini_utils/geometry.h>
#include <openvdb_houdini/GeometryUtil.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/PointIndexGrid.h>

#ifdef DWA_OPENVDB
#include <openvdb_houdini/DW_VDBUtils.h>
#endif

#include <UT/UT_Interrupt.h>
#include <UT/UT_StopWatch.h>
#include <UT/UT_Version.h>
#include <GA/GA_Types.h>
#include <GA/GA_Handle.h>
#include <GU/GU_ConvertParms.h>
#include <GU/GU_Detail.h>
#include <GU/GU_Surfacer.h>
#include <GU/GU_PolyReduce.h>
#include <GU/GU_PrimPoly.h>
#include <PRM/PRM_Parm.h>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_arithmetic.hpp>

namespace boost {
template<> struct is_integral<openvdb::PointIndex32>: public boost::true_type {};
template<> struct is_integral<openvdb::PointIndex64>: public boost::true_type {};
}

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

// HAVE_SURFACING_PARM is disabled in H12.5
#ifdef SESI_OPENVDB
#define HAVE_SURFACING_PARM 0
#else
#define HAVE_SURFACING_PARM 1
#endif

class SOP_OpenVDB_Visualize: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Visualize(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Visualize() {};

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i == 1); }

    static UT_Vector3 colorLevel(int level) {return mColors[std::max(3-level,0)];}
    static const UT_Vector3& colorSign(bool negative) {return mColors[negative ? 5 : 4];}

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
    virtual void resolveObsoleteParms(PRM_ParmList*);
    static  UT_Vector3 mColors[];
};


// Same color scheme as the VDB TOG paper.
UT_Vector3 SOP_OpenVDB_Visualize::mColors[] = {
    UT_Vector3(0.045f, 0.045f, 0.045f),         // 0. Root
    UT_Vector3(0.0432f, 0.33f, 0.0411023f),     // 1. First internal node level
    UT_Vector3(0.871f, 0.394f, 0.01916f),       // 2. Intermediate internal node levels
    UT_Vector3(0.00608299f, 0.279541f, 0.625f), // 3. Leaf level
    UT_Vector3(0.523f, 0.0325175f, 0.0325175f), // 4. Value >= ZeroVal (for voxels or tiles)
    UT_Vector3(0.92f, 0.92f, 0.92f)             // 5. Value < ZeroVal (for voxels or tiles)
};


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Group pattern
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to be processed.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

#if HAVE_SURFACING_PARM
    // Surfacing
    parms.add(hutil::ParmFactory(PRM_HEADING,"surfacing", "Surfacing"));

    {   // Meshing scheme
        const char* items[] = {
            "none",     "Disabled",
            "opevdb",   "OpenVDB Mesher",
            "houdini",  "Houdini Surfacer",
            NULL
        };
        parms.add(hutil::ParmFactory(PRM_ORD, "meshing", "Meshing")
            .setHelpText("Select meshing scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    parms.add(hutil::ParmFactory(PRM_FLT_J, "adaptivity", "Adaptivity")
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0));

    //parms.add(hutil::ParmFactory(PRM_TOGGLE, "computeNormals", "Compute Point Normals"));
    parms.add(hutil::ParmFactory(PRM_FLT_J, "isoValue", "Iso Value")
        .setRange(PRM_RANGE_FREE, -2.0, PRM_RANGE_FREE, 2.0));
    parms.add(
        hutil::ParmFactory(PRM_RGB_J, "surfaceColor", "Surface Color")
        .setDefault(std::vector<PRM_Default>(3, PRM_Default(0.84))) // RGB = (0.84, 0.84, 0.84)
        .setVectorSize(3));

    // Tree Topology
    parms.add(hutil::ParmFactory(PRM_HEADING,"treeTopology", "Tree Topology"));
#endif // HAVE_SURFACING_PARM

    {   // Tree Nodes
        const char* items[] = {
#ifdef DWA_OPENVDB
            "none",     "Disabled",
            "leaf",     "Leaf Nodes and Active Tiles",
            "nonconst", "Leaf and Internal Nodes",
#else
            "none",     "Disabled",
            "leaf",     "Leaf Nodes", // includes constant
            "nonconst", "All Non-Constant Nodes",
#endif
            NULL
        };
        parms.add(hutil::ParmFactory(PRM_ORD, "nodes", "Tree Nodes")
            .setDefault(PRMoneDefaults)
            .setHelpText("Select render mode for tree nodes")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }
    {   // Active Tiles
        const char* items[] = {
            "none",     "Disabled",
            "points",   "Points",
            "pvalue",   "Points with Values",
            "wirebox",  "Wireframe Box",
            "box",      "Solid Box",
            NULL
        };
        parms.add(hutil::ParmFactory(PRM_ORD, "tiles", "Active Constant Tiles")
            .setHelpText("Select render mode for active tiles")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }
    {   // Active Voxels
        const char* items[] = {
            "none",     "Disabled",
            "points",   "Points",
            "pvalue",   "Points with Values",
            "wirebox",  "Wireframe Box",
            "box",      "Solid Box",
            NULL
        };
        parms.add(hutil::ParmFactory(PRM_ORD, "voxels", "Active Voxels")
            .setHelpText("Select render mode for active voxels")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
        // toggle for staggered vf
        parms.add(hutil::ParmFactory(PRM_TOGGLE, "ignorestaggered", "Ignore Staggered Vectors")
                  .setDefault(PRMzeroDefaults)
                  .setHelpText("Draws staggered vectors as if they were collocated."));
    }

    // Toggle to preview the frustum
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "previewFrustum", "Preview Frustum"));

#ifdef DWA_OPENVDB
    // Toggle to preview the ROI
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "previewroi", "Preview Region of Interest"));
#endif

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.beginSwitcher("tabMenu");
    obsoleteParms.addFolder("Tree Topology");
    obsoleteParms.endSwitcher();
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING,"renderOptions", "Render Options"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "extractMesh", "Extract Mesh"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "computeNormals", "Compute Point Normals"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "reverse", "Reverse Faces"));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "optionsHeading", "Options"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "verbose", "Verbose"));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "Other", "Other"));

#ifndef DWA_OPENVDB
    // We probably need this to share hip files.
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "previewroi", ""));
#endif

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Visualize", SOP_OpenVDB_Visualize::factory, parms, *table)
        .addAlias("OpenVDB Visualizer")
        .setObsoleteParms(obsoleteParms)
        .addInput("Input with VDBs to visualize");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Visualize::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Visualize(net, name, op);
}


SOP_OpenVDB_Visualize::SOP_OpenVDB_Visualize(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


void
SOP_OpenVDB_Visualize::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;

    // The "extractMesh" toggle was replaced by the "meshing" menu.
    // If meshing was enabled, enable Houdini surfacing.
    PRM_Parm* parm = obsoleteParms->getParmPtr("extractMesh");
    if (parm && !parm->isFactoryDefault()) {
        setInt("meshing", 0, 0.0,
            (obsoleteParms->evalInt("extractMesh", 0, /*time=*/0.0) ? 2 : 0));
    }

    // Delegate to the base class.
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}


// Update UI parm display
bool
SOP_OpenVDB_Visualize::updateParmsFlags()
{
    bool changed = false;

#if HAVE_SURFACING_PARM
    const bool extractMesh = evalInt("meshing", 0, 0) > 0;
    //changed += enableParm("computeNormals", extractMesh);
    changed |= enableParm("adaptivity", extractMesh);
    changed |= enableParm("surfaceColor", extractMesh);
    changed |= enableParm("isoValue", extractMesh);
#endif

    const bool drawVoxels = evalInt("voxels",  0, 0) > 0;
    changed |= enableParm("ignorestaggered", drawVoxels);

    return changed;
}


////////////////////////////////////////

void
createBox(GU_Detail& geo, const openvdb::math::Transform& xform,
    const openvdb::CoordBBox& bbox, const UT_Vector3& color, bool solid = false)
{
    struct Local {
        static inline UT_Vector3 Vec3dToUTV3(const openvdb::Vec3d& v) {
            return UT_Vector3(float(v.x()), float(v.y()), float(v.z()));
        }
    };

    UT_Vector3 corners[8];

#if 1
    // Nodes are rendered as cell-centered (0.5 voxel dilated) AABBox in world space
    const openvdb::Vec3d min(bbox.min().x()-0.5, bbox.min().y()-0.5, bbox.min().z()-0.5);
    const openvdb::Vec3d max(bbox.max().x()+0.5, bbox.max().y()+0.5, bbox.max().z()+0.5);
#else
    // Render as node-centered (used for debugging)
    const openvdb::Vec3d min(bbox.min().x(), bbox.min().y(), bbox.min().z());
    const openvdb::Vec3d max(bbox.max().x()+1.0, bbox.max().y()+1.0, bbox.max().z()+1.0);
#endif

    openvdb::Vec3d ptn = xform.indexToWorld(min);
    corners[0] = Local::Vec3dToUTV3(ptn);

    ptn = openvdb::Vec3d(min.x(), min.y(), max.z());
    ptn = xform.indexToWorld(ptn);
    corners[1] = Local::Vec3dToUTV3(ptn);

    ptn = openvdb::Vec3d(max.x(), min.y(), max.z());
    ptn = xform.indexToWorld(ptn);
    corners[2] = Local::Vec3dToUTV3(ptn);

    ptn = openvdb::Vec3d(max.x(), min.y(), min.z());
    ptn = xform.indexToWorld(ptn);
    corners[3] = Local::Vec3dToUTV3(ptn);

    ptn = openvdb::Vec3d(min.x(), max.y(), min.z());
    ptn = xform.indexToWorld(ptn);
    corners[4] = Local::Vec3dToUTV3(ptn);

    ptn = openvdb::Vec3d(min.x(), max.y(), max.z());
    ptn = xform.indexToWorld(ptn);
    corners[5] = Local::Vec3dToUTV3(ptn);

    ptn = xform.indexToWorld(max);
    corners[6] = Local::Vec3dToUTV3(ptn);

    ptn = openvdb::Vec3d(max.x(), max.y(), min.z());
    ptn = xform.indexToWorld(ptn);
    corners[7] = Local::Vec3dToUTV3(ptn);

    hutil::createBox(geo, corners, &color, solid);
}


////////////////////////////////////////


template <typename RenderType>
struct RenderTilesAndLeafs {
    RenderTilesAndLeafs(RenderType& r) : render(r) {}
    template<openvdb::Index LEVEL>
    inline bool descent() { return LEVEL>0; } // only descend to leaf nodes

    template<openvdb::Index LEVEL>
    inline void operator()(const openvdb::CoordBBox &bbox) {
        render.addWireBox(bbox,SOP_OpenVDB_Visualize::colorLevel(LEVEL));
    }
    RenderType& render;
};


////////////////////////////////////////


class VDBTopologyVisualizer
{
public:

    enum { NODE_WIRE_BOX = 1, NODE_SOLID_BOX = 2};
    enum { POINTS = 1, POINTS_WITH_VALUES = 2, WIRE_BOX = 3, SOLID_BOX = 4};

    VDBTopologyVisualizer(GU_Detail&,
        int nodeRenderMode, int tileRenderMode, int voxelRenderMode,
        bool ignoreStaggeredVectors = false, hvdb::Interrupter* interrupter = NULL);

    template<typename GridType>
    void operator()(const GridType&);

    void addWireBox(const openvdb::CoordBBox&, const UT_Vector3& color);

private:

    int mNodeRenderMode, mTileRenderMode, mVoxelRenderMode;
    bool mIgnoreStaggered;

    GU_Detail* mGeo;
    hvdb::Interrupter* mInterrupter;

    const openvdb::math::Transform* mXform;
    GA_RWHandleF      mFloatHandle;
    GA_RWHandleI      mInt32Handle;
    GA_RWHandleV3     mVec3fHandle;
    GA_RWHandleV3     mCdHandle;

    /// @param pos position in index coordinates
    GA_Offset createPoint(const openvdb::Vec3d& pos);

    GA_Offset createPoint(const openvdb::CoordBBox&, const UT_Vector3& color);

    template <typename ValType>
    typename boost::enable_if<boost::is_integral<ValType>, void>::type
    addPoint(const openvdb::CoordBBox&, const UT_Vector3& color, ValType s, bool);

    template <typename ValType>
    typename boost::enable_if<boost::is_floating_point<ValType>, void>::type
    addPoint(const openvdb::CoordBBox&, const UT_Vector3& color, ValType s, bool);

    template <typename ValType>
    typename boost::disable_if<boost::is_arithmetic<ValType>, void>::type
    addPoint(const openvdb::CoordBBox&, const UT_Vector3& color, ValType v, bool staggered);

    void addPoint(const openvdb::CoordBBox&, const UT_Vector3& color, bool staggered);

    void addBox(const openvdb::CoordBBox&, const UT_Vector3& color, bool solid);

    bool wasInterrupted(int percent = -1) const {
        return mInterrupter && mInterrupter->wasInterrupted(percent);
    }
};


VDBTopologyVisualizer::VDBTopologyVisualizer(GU_Detail& geo,
    int nodeRenderMode, int tileRenderMode, int voxelRenderMode,
    bool ignoreStaggeredVectors, hvdb::Interrupter* interrupter)
    : mNodeRenderMode(nodeRenderMode)
    , mTileRenderMode(tileRenderMode)
    , mVoxelRenderMode(voxelRenderMode)
    , mIgnoreStaggered(ignoreStaggeredVectors)
    , mGeo(&geo)
    , mInterrupter(interrupter)
    , mXform(NULL)
{
}


void
VDBTopologyVisualizer::addWireBox(const openvdb::CoordBBox& bbox,
    const UT_Vector3& color)
{
    addBox(bbox, color, false);
}


template<typename GridType>
void
VDBTopologyVisualizer::operator()(const GridType& grid)
{
    typedef typename GridType::TreeType TreeType;

    openvdb::CoordBBox bbox;
    mXform = &grid.transform();

    const std::string valueType = grid.valueType();

    // Node rendering
    if (mNodeRenderMode == NODE_WIRE_BOX) {

        RenderTilesAndLeafs<VDBTopologyVisualizer> op(*this);
        grid.tree().visitActiveBBox(op);

         // don't render tiles twice!
        if (mTileRenderMode == WIRE_BOX) mTileRenderMode = 0;

    } else if (mNodeRenderMode == NODE_SOLID_BOX) {

        // for each node..
        for (typename TreeType::NodeCIter iter(grid.tree()); iter; ++iter) {

            if (iter.getDepth() == 0) continue; // don't draw the root node

            iter.getBoundingBox(bbox);
            addWireBox(bbox, SOP_OpenVDB_Visualize::colorLevel(iter.getLevel()));
        }
    }

    // Tile and voxel rendering
    if (mTileRenderMode || mVoxelRenderMode) {

        // add value attributes
        if (mTileRenderMode == POINTS_WITH_VALUES || mVoxelRenderMode == POINTS_WITH_VALUES) {

            if (valueType == openvdb::typeNameAsString<float>() ||
                valueType == openvdb::typeNameAsString<double>()) {

                GA_RWAttributeRef attribHandle =
                    mGeo->findFloatTuple(GA_ATTRIB_POINT, "vdb_float", 1);

                if (!attribHandle.isValid()) {
                    attribHandle = mGeo->addFloatTuple(
                        GA_ATTRIB_POINT, "vdb_float", 1, GA_Defaults(0));
                }

                mFloatHandle = attribHandle.getAttribute();
                mGeo->addVariableName("vdb_float", "VDB_FLOAT");

            } else if (valueType == openvdb::typeNameAsString<int32_t>() ||
                valueType == openvdb::typeNameAsString<int64_t>() ||
                valueType == openvdb::typeNameAsString<bool>()) {

                GA_RWAttributeRef attribHandle =
                mGeo->findIntTuple(GA_ATTRIB_POINT, "vdb_int", 1);

                if (!attribHandle.isValid()) {
                    attribHandle = mGeo->addIntTuple(
                    GA_ATTRIB_POINT, "vdb_int", 1, GA_Defaults(0));
                }

                mInt32Handle = attribHandle.getAttribute();
                mGeo->addVariableName("vdb_int", "VDB_INT");

            } else if (valueType == openvdb::typeNameAsString<openvdb::Vec3s>() ||
                valueType == openvdb::typeNameAsString<openvdb::Vec3d>()) {

                GA_RWAttributeRef attribHandle =
                    mGeo->findFloatTuple(GA_ATTRIB_POINT, "vdb_vec3f", 3);

                if (!attribHandle.isValid()) {
                    attribHandle = mGeo->addFloatTuple(
                    GA_ATTRIB_POINT, "vdb_vec3f", 3, GA_Defaults(0));
                }

                mVec3fHandle = attribHandle.getAttribute();
                mGeo->addVariableName("vdb_vec3f", "VDB_VEC3F");

            } else {
                throw std::runtime_error(
                    "value attributes are not supported for values of type " + valueType);
            }
        }

#if (UT_VERSION_INT >= 0x0e0000b4) // 14.0.180 or later
        mCdHandle.bind(mGeo->findDiffuseAttribute(GA_ATTRIB_POINT));
        if (!mCdHandle.isValid()) {
            mCdHandle.bind(mGeo->addDiffuseAttribute(GA_ATTRIB_POINT));
        }
#else
        mCdHandle.bind(mGeo->findDiffuseAttribute(GA_ATTRIB_POINT).getAttribute());
        if (!mCdHandle.isValid()) {
            mCdHandle.bind(mGeo->addDiffuseAttribute(GA_ATTRIB_POINT).getAttribute());
        }
#endif

        const bool staggered = !mIgnoreStaggered &&
            (grid.getGridClass() == openvdb::GRID_STAGGERED);

        // for each active value..
        for (typename TreeType::ValueOnCIter iter = grid.cbeginValueOn(); iter; iter.next()) {

            if (wasInterrupted()) break;

            const int renderMode = iter.isVoxelValue() ? mVoxelRenderMode : mTileRenderMode;
            if (renderMode == 0) continue; // nothing to do!

            const bool negative =
                iter.getValue() < openvdb::zeroVal<typename TreeType::ValueType>();
            const UT_Vector3& color = SOP_OpenVDB_Visualize::colorSign(negative);
            iter.getBoundingBox(bbox);

            switch (renderMode) {
                case POINTS: addPoint(bbox, color, staggered); break;
                case POINTS_WITH_VALUES: addPoint(bbox, color, iter.getValue(), staggered); break;
                default: addBox(bbox, color, renderMode == SOLID_BOX); break;
            }
        }
    }
}


inline GA_Offset
VDBTopologyVisualizer::createPoint(const openvdb::Vec3d& pos)
{
    openvdb::Vec3d wpos = mXform->indexToWorld(pos);
    GA_Offset offset = mGeo->appendPointOffset();
    mGeo->setPos3(offset, wpos[0], wpos[1], wpos[2]);
    return offset;
}


inline GA_Offset
VDBTopologyVisualizer::createPoint(const openvdb::CoordBBox& bbox,
    const UT_Vector3& color)
{
    openvdb::Vec3d pos = openvdb::Vec3d(0.5*(bbox.min().x()+bbox.max().x()),
                                        0.5*(bbox.min().y()+bbox.max().y()),
                                        0.5*(bbox.min().z()+bbox.max().z()));
    GA_Offset offset = createPoint(pos);
    mCdHandle.set(offset, color);
    return offset;
}


template <typename ValType>
typename boost::enable_if<boost::is_integral<ValType>, void>::type
VDBTopologyVisualizer::addPoint(const openvdb::CoordBBox& bbox,
    const UT_Vector3& color, ValType s, bool)
{
    mInt32Handle.set(createPoint(bbox, color), int(s));
}


template <typename ValType>
typename boost::enable_if<boost::is_floating_point<ValType>, void>::type
VDBTopologyVisualizer::addPoint(const openvdb::CoordBBox& bbox,
    const UT_Vector3& color, ValType s, bool)
{
    mFloatHandle.set(createPoint(bbox, color), float(s));
}


template <typename ValType>
typename boost::disable_if<boost::is_arithmetic<ValType>, void>::type
VDBTopologyVisualizer::addPoint(const openvdb::CoordBBox& bbox,
    const UT_Vector3& color, ValType v, bool staggered)
{
    if (!staggered) {
        mVec3fHandle.set(createPoint(bbox, color),
            UT_Vector3(float(v[0]), float(v[1]), float(v[2])));
    } else {
        openvdb::Vec3d pos = openvdb::Vec3d(0.5*(bbox.min().x()+bbox.max().x()),
                                            0.5*(bbox.min().y()+bbox.max().y()),
                                            0.5*(bbox.min().z()+bbox.max().z()));
        pos[0] -= 0.5; // -x
        GA_Offset offset = createPoint(pos);
        mCdHandle.set(offset, UT_Vector3(1.0, 0.0, 0.0)); // r
        mVec3fHandle.set(offset, UT_Vector3(float(v[0]), 0.0, 0.0));

        pos[0] += 0.5;
        pos[1] -= 0.5; // -y
        offset = createPoint(pos);
        mCdHandle.set(offset, UT_Vector3(0.0, 1.0, 0.0)); // g
        mVec3fHandle.set(offset, UT_Vector3(0.0, float(v[1]), 0.0));

        pos[1] += 0.5;
        pos[2] -= 0.5; // -z
        offset = createPoint(pos);
        mCdHandle.set(offset, UT_Vector3(0.0, 0.0, 1.0)); // b
        mVec3fHandle.set(offset, UT_Vector3(0.0, 0.0, float(v[2])));
    }
}


void
VDBTopologyVisualizer::addPoint(const openvdb::CoordBBox& bbox,
    const UT_Vector3& color, bool staggered)
{
    if (!staggered) {
        createPoint(bbox, color);
    } else {
        openvdb::Vec3d pos = openvdb::Vec3d(0.5*(bbox.min().x()+bbox.max().x()),
                                            0.5*(bbox.min().y()+bbox.max().y()),
                                            0.5*(bbox.min().z()+bbox.max().z()));
        pos[0] -= 0.5; // -x
        GA_Offset offset = createPoint(pos);
        mCdHandle.set(offset, color);

        pos[0] += 0.5;
        pos[1] -= 0.5; // -y
        offset = createPoint(pos);
        mCdHandle.set(offset, color);

        pos[1] += 0.5;
        pos[2] -= 0.5; // -z
        offset = createPoint(pos);
        mCdHandle.set(offset, color);
    }
}

void
VDBTopologyVisualizer::addBox(const openvdb::CoordBBox& bbox,
    const UT_Vector3& color, bool solid)
{
    createBox(*mGeo, *mXform, bbox, color, solid);
}


////////////////////////////////////////


#if HAVE_SURFACING_PARM

class VDBGridSurfacer
{
public:

    VDBGridSurfacer(GU_Detail& geo, float iso = 0.0, float adaptivityThreshold = 0.0,
        bool generateNormals = false, hvdb::Interrupter* interrupter = NULL);

    template<typename GridType>
    void operator()(const GridType&);

private:

    bool wasInterrupted(int percent = -1) const {
        return mInterrupter && mInterrupter->wasInterrupted(percent);
    }

    GU_Detail* mGeo;
    const float mIso, mAdaptivityThreshold;
    const bool mGenerateNormals;
    hvdb::Interrupter* mInterrupter;
};


VDBGridSurfacer::VDBGridSurfacer(GU_Detail& geo, float iso,
    float adaptivityThreshold, bool generateNormals, hvdb::Interrupter* interrupter)
    : mGeo(&geo)
    , mIso(iso)
    , mAdaptivityThreshold(adaptivityThreshold)
    , mGenerateNormals(generateNormals)
    , mInterrupter(interrupter)
{
}


template<typename GridType>
void
VDBGridSurfacer::operator()(const GridType& grid)
{
    typedef typename GridType::TreeType TreeType;
    typedef typename TreeType::LeafNodeType LeafNodeType;
    openvdb::CoordBBox bbox;

    // Gets min & max and checks if the grid is empty
    if (grid.tree().evalLeafBoundingBox(bbox)) {

        openvdb::Coord dim(bbox.max() - bbox.min());

        GU_Detail tmpGeo;

        GU_Surfacer surfacer(tmpGeo,
            UT_Vector3(bbox.min().x(), bbox.min().y(), bbox.min().z()),
            UT_Vector3(dim[0], dim[1], dim[2]),
            dim[0], dim[1], dim[2], mGenerateNormals);

        typename GridType::ConstAccessor accessor = grid.getConstAccessor();

        openvdb::Coord xyz;
        fpreal density[8];

        // for each leaf..
        for (typename TreeType::LeafCIter iter = grid.tree().cbeginLeaf(); iter; iter.next()) {

            if (wasInterrupted()) break;

            bool isLess = false, isMore = false;

            // for each active voxel..
            typename LeafNodeType::ValueOnCIter it = iter.getLeaf()->cbeginValueOn();
            for ( ; it; ++it) {
                xyz = it.getCoord();

                // Sample values at each corner of the voxel
                for (unsigned int d = 0; d < 8; ++d) {

                    openvdb::Coord valueCoord(
                        xyz.x() +  (d & 1),
                        xyz.y() + ((d & 2) >> 1),
                        xyz.z() + ((d & 4) >> 2));

                    // Houdini uses the inverse sign convention for level sets!
                    density[d] = mIso - float(accessor.getValue(valueCoord));
                    density[d] <= 0.0f ? isLess = true : isMore = true;
                }

                // If there is a crossing, surface this voxel
                if (isLess && isMore) {
                    surfacer.addCell(
                        xyz.x() - bbox.min().x(),
                        xyz.y() - bbox.min().y(),
                        xyz.z() - bbox.min().z(),
                        density, 0);
                }
            } // end active voxel traversal
        } // end leaf traversal

        if (wasInterrupted()) return;

        if (mAdaptivityThreshold > 1e-6) {
            GU_PolyReduceParms parms;
            parms.percentage =
                static_cast<float>(100.0 * (1.0 - std::min(mAdaptivityThreshold, 0.99f)));
            parms.usepercent = 1;
            tmpGeo.polyReduce(parms);
        }

        // world space transform
        for (GA_Iterator it(tmpGeo.getPointRange()); !it.atEnd(); it.advance()) {
            GA_Offset ptOffset = it.getOffset();

            UT_Vector3 pos = tmpGeo.getPos3(ptOffset);
            openvdb::Vec3d vPos(pos.x(), pos.y(), pos.z());
            openvdb::Vec3d wPos = grid.indexToWorld(vPos);

            tmpGeo.setPos3(ptOffset, UT_Vector3(wPos.x(), wPos.y(), wPos.z()));
        }

        mGeo->merge(tmpGeo);
    }
}

#endif // HAVE_SURFACING_PARM


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Visualize::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();
        gdp->clearAndDestroy();

        hvdb::Interrupter boss("Visualizer");

        const GU_Detail* refGdp = inputGeo(0);
        if(refGdp == NULL) return error();

        // Get the group of grids to vizualize.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group =
            matchGroup(const_cast<GU_Detail&>(*refGdp), groupStr.toStdString());


        // Evaluate the UI parameters
#if HAVE_SURFACING_PARM
        const int meshing  = evalInt("meshing", 0, time);
        const double adaptivity = evalFloat("adaptivity", 0, time);
        const double iso = double(evalFloat("isoValue", 0, time));
#else
        const int meshing = 0;
#endif
        const int nodeOptions  = evalInt("nodes",  0, time);
        const int tileOptions  = evalInt("tiles",  0, time);
        const int voxelOptions = evalInt("voxels",  0, time);
        const int ignorestaggered    = evalInt("ignorestaggered", 0, time);
        const bool showFrustum = bool(evalInt("previewFrustum", 0, time));
#ifdef DWA_OPENVDB
        const bool showROI = bool(evalInt("previewroi", 0, time));
#else
        const bool showROI = false;
#endif
        const bool drawTree = (nodeOptions + tileOptions + voxelOptions) != 0;

#if HAVE_SURFACING_PARM
        if (meshing != 0) {

            fpreal values[3] = {
                evalFloat("surfaceColor", 0, time),
                evalFloat("surfaceColor", 1, time),
                evalFloat("surfaceColor", 2, time)};

            GA_Defaults color;
            color.set(values, 3);
            gdp->addFloatTuple(GA_ATTRIB_POINT, "Cd", 3, color);
        }

        // mesh using OpenVDB mesher
        if (meshing == 1) {
            GU_ConvertParms parms;
#if (UT_VERSION_INT < 0x0d0000b1) // before 13.0.177
            parms.toType = GEO_PrimTypeCompat::GEOPRIMPOLY;
#else
            parms.setToType(GEO_PrimTypeCompat::GEOPRIMPOLY);
#endif
            parms.myOffset = static_cast<float>(iso);
            parms.preserveGroups = false;
            parms.primGroup = const_cast<GA_PrimitiveGroup*>(group);
            GU_PrimVDB::convertVDBs(*gdp, *refGdp, parms,
                adaptivity, /*keep_original*/true);
        }
#endif

        if (!boss.wasInterrupted() && (meshing == 2 || drawTree || showFrustum || showROI)) {

            // for each VDB primitive...
            for (hvdb::VdbPrimCIterator it(refGdp, group); it; ++it) {

                if (boss.wasInterrupted()) break;

                const GU_PrimVDB *vdb = *it;

#if HAVE_SURFACING_PARM
                // mesh using houdini surfacer
                if (meshing == 2) {
                    VDBGridSurfacer surfacer(*gdp, static_cast<float>(iso),
                        static_cast<float>(adaptivity), false, &boss);
                    GEOvdbProcessTypedGridScalar(*vdb, surfacer);
                }
#endif

                // draw tree topology
                if (drawTree) {
                    VDBTopologyVisualizer draw(*gdp, nodeOptions, tileOptions,
                        voxelOptions, ignorestaggered, &boss);

                    if (!GEOvdbProcessTypedGridTopology(*vdb, draw)) {

                        if (vdb->getGrid().type() == openvdb::tools::PointIndexGrid::gridType()) {
                            openvdb::tools::PointIndexGrid::ConstPtr grid =
                                 openvdb::gridConstPtrCast<openvdb::tools::PointIndexGrid>(
                                     vdb->getGridPtr());
                            draw(*grid);
                        }
                    }
                }

                if (showFrustum) {
                    UT_Vector3 box_color(0.6f, 0.6f, 0.6f);
                    UT_Vector3 tick_color(0.0f, 0.0f, 0.0f);
                    hvdb::drawFrustum(*gdp, vdb->getGrid().transform(),
                        &box_color, &tick_color, /*shaded*/true);
                }

#ifdef DWA_OPENVDB
                if (showROI) {
                    const openvdb::GridBase& grid = vdb->getConstGrid();
                    openvdb::Vec3IMetadata::ConstPtr metaMin =
                        grid.getMetadata<openvdb::Vec3IMetadata>(
                            openvdb::Name(openvdb_houdini::METADATA_ROI_MIN));
                    openvdb::Vec3IMetadata::ConstPtr metaMax =
                        grid.getMetadata<openvdb::Vec3IMetadata>(
                            openvdb::Name(openvdb_houdini::METADATA_ROI_MAX));

                    if (metaMin && metaMax) {
                        openvdb::CoordBBox roi(
                            openvdb::Coord(metaMin->value()), openvdb::Coord(metaMax->value()));
                        createBox(*gdp, grid.transform(), roi, UT_Vector3(1.0, 0.0, 0.0));
                    }
                }
#endif
            }
        }

        if (boss.wasInterrupted()) {
            addWarning(SOP_MESSAGE, "Process was interrupted");
        }

        boss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
