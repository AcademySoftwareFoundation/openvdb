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
//
/// @file SOP_OpenVDB_Convert.cc
///
/// @author SESI and FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/GeometryUtil.h>
#include <openvdb_houdini/AttributeTransferUtil.h>
#include <openvdb_houdini/Utils.h>

#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tree/ValueAccessor.h>

#include <CH/CH_Manager.h>
#include <GA/GA_PageIterator.h>
#include <GU/GU_ConvertParms.h>
#include <GU/GU_PrimPoly.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_Version.h>
#include <UT/UT_VoxelArray.h>
#include <SYS/SYS_Math.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/math/special_functions/round.hpp>

#include <list>

#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
#define HAVE_POLYSOUP 1
#include <GU/GU_PrimPolySoup.h>
#else
#define HAVE_POLYSOUP 0
#endif

#if (UT_VERSION_INT >= 0x0d0000ed) // 13.0.237 or later
#define HAVE_SPLITTING 1
#else
#define HAVE_SPLITTING 0
#endif

#if (UT_VERSION_INT >= 0x0e05005b) // 14.5.91 or later
#define HAVE_ACTIVATEINSIDE 1
#else
#define HAVE_ACTIVATEINSIDE 0
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
// GA_RWHandleV3 fails to initialize its member variables in some cases.
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

namespace {
#if HAVE_POLYSOUP
enum ConvertTo { HVOLUME, OPENVDB, POLYGONS, POLYSOUP /*, SIMDATA*/ };
#else
enum ConvertTo { HVOLUME, OPENVDB, POLYGONS /*, SIMDATA*/ };
#endif
enum ConvertClass { CLASS_NO_CHANGE, CLASS_SDF, CLASS_FOG_VOLUME };
}


class SOP_OpenVDB_Convert: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Convert(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Convert() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    // Return true for a given input if the connector to the input
    // should be drawn dashed rather than solid.
    virtual int isRefInput(unsigned idx) const { return (idx == 1); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

private:
    void convertToPoly(
        fpreal time,
        GA_PrimitiveGroup *group,
        bool buildpolysoup,
        hvdb::Interrupter &boss);

    template <class GridType>
    void referenceMeshing(
        std::list<openvdb::GridBase::ConstPtr>& grids,
        std::list<const GU_PrimVDB*> vdbs,
        GA_PrimitiveGroup *group,
        openvdb::tools::VolumeToMesh& mesher,
        const GU_Detail* refGeo,
        bool computeNormals,
        hvdb::Interrupter& boss,
        const fpreal time);
};


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to surface.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));


    { // Convert To Menu
        const char* items[] = {
            "volume",   "Volume",
            "vdb",      "VDB",
            "poly",     "Polygons",
#if HAVE_POLYSOUP
            "polysoup", "Polygon Soup",
#endif
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "conversion", "Convert To")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    { // Grid Class Menu
        const char* class_items[] = {
            "none", "No Change",
            "sdf",  "Convert Fog to SDF",
            "fog",  "Convert SDF to Fog",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "vdbclass", "VDB Class")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, class_items));
    }

    //////////

    // Parms for converting to volumes

#if HAVE_SPLITTING
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "splitdisjointvolumes", "Split Disjoint Volumes")
        .setHelpText("When converting to volumes, create multiple "
            "volume primitives per VDB for unconnected regions where "
            "possible. This allows very large and sparse VDBs to be converted "
            "with less memory usage."));
#endif

    //////////

    // Parms for converting to polygons

    parms.add(hutil::ParmFactory(PRM_FLT_J, "isoValue", "Isovalue")
        .setRange(PRM_RANGE_UI, -1.0, PRM_RANGE_UI, 1.0)
        .setHelpText("The crossing point of the VDB values that is considered "
            "the surface when converting to polygons"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "fogisovalue", "Fog Isovalue")
        .setRange(PRM_RANGE_UI, 0.0, PRM_RANGE_UI, 1.0)
        .setDefault(PRMpointFiveDefaults)
        .setHelpText("The crossing point of the VDB values that is considered "
            "the surface when converting to level sets from fog volumes"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "adaptivity", "Adaptivity")
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 2.0)
        .setHelpText("When converting to polygons the adaptivity threshold determines "
            "how closely the isosurface is matched by the resulting mesh. Higher "
            "thresholds will allow more variation in polygon size, using fewer "
            "polygons to express the surface."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "computenormals", "Compute Vertex Normals")
        .setHelpText("Compute edge-preserving vertex normals"));


    //////////

    // Reference input options

    parms.add(hutil::ParmFactory(PRM_FLT_J, "internaladaptivity", "Internal Adaptivity")
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0)
        .setHelpText("When converting to polygons with a second input, this overrides "
            "the adaptivity threshold for all internal surfaces."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "transferattributes", "Transfer Surface Attributes")
        .setHelpText("When converting to polygons with a second input, this transfers "
            "all attributes (primitive, vertex and point) from the reference surface. "
            "Will override computed vertex normals for primitives in the surface group."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "sharpenfeatures", "Sharpen Features")
        .setHelpText("Sharpen edges and corners."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "edgetolerance", "Edge Tolerance")
        .setDefault(0.5)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0)
        .setHelpText("Controls the edge adaptivity mask"));

    parms.add(hutil::ParmFactory(PRM_STRING, "surfacegroup", "Surface Group")
        .setDefault("surface_polygons")
        .setHelpText("When converting to polygons with a second input, this "
            "specifies a group for all polygons that are coincident with the "
            "reference surface. This group is useful for transferring attributes such "
            "as uv coordinates, normals etc. from the reference surface."));

    parms.add(hutil::ParmFactory(PRM_STRING, "interiorgroup", "Interior Group")
        .setDefault("interior_polygons")
        .setHelpText("When converting to polygons with a second input, this "
            "specifies a group for all polygons that are interior to the "
            "reference surface. This group can be used to identify surface regions "
            "that might require projected uv coordinates or new materials."));

    parms.add(hutil::ParmFactory(PRM_STRING, "seamlinegroup", "Seam Line Group")
        .setDefault("seam_polygons")
        .setHelpText("When converting to polygons with a second input, this "
            "specifies a group for all polygons that are in proximity to "
            "the seam lines. This group can be used to drive secondary elements such "
            "as debris and dust."));

    parms.add(hutil::ParmFactory(PRM_STRING, "seampoints", "Seam Points")
        .setDefault("seam_points")
        .setHelpText("When converting to polygons with a second input, this "
            "specifies a group of the fracture seam points. This can be "
            "used to drive local pre-fracture dynamics e.g. local surface buckling."));

    //////////

    // Mask input options


   parms.add(hutil::ParmFactory(PRM_TOGGLE, "surfacemask", "")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Enable / disable the surface mask"));

    parms.add(hutil::ParmFactory(PRM_STRING, "surfacemaskname", "Surface Mask")
        .setHelpText("A single level-set or sdf grid whose interior defines the region to mesh")
        .setChoiceList(&hutil::PrimGroupMenuInput3));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "surfacemaskoffset", "Mask Offset")
        .setDefault(PRMzeroDefaults)
        .setHelpText("Isovalue used to offset the interior region of the surface mask")
        .setRange(PRM_RANGE_UI, -1.0, PRM_RANGE_UI, 1.0));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "invertmask", "Invert Surface Mask")
        .setHelpText("Used to mesh the complement of the mask"));


    parms.add(hutil::ParmFactory(PRM_TOGGLE, "adaptivityfield", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Enable / disable the the adaptivity field"));

    parms.add(hutil::ParmFactory(PRM_STRING, "adaptivityfieldname", "Adaptivity Field")
        .setHelpText(
            "A single scalar grid used as a spatial multiplier for the adaptivity threshold")
        .setChoiceList(&hutil::PrimGroupMenuInput3));



    //////////

    // Parms for converting to volumes

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "prune", "")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Collapse regions of constant value in output grids. "
            "Voxel values are considered equal if they differ "
            "by less than the specified threshold."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "tolerance", "Prune Tolerance")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "flood", "Signed-Flood Fill Output")
        .setDefault(PRMoneDefaults)
        .setHelpText("Reclassify inactive output voxels as either inside or outside."));

#if HAVE_ACTIVATEINSIDE
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "activateinsidesdf", "Activate Inside Voxels")
        .setDefault(PRMoneDefaults)
        .setHelpText("Activate all voxels inside a converted level set."));
#endif

    //////////

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR,"sep1", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "smoothseams", "Smooth Seams"));
    obsoleteParms.add(hutil::ParmFactory(PRM_INT_J, "automaticpartitions", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_INT_J, "activepart", ""));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Convert",
        SOP_OpenVDB_Convert::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("VDBs to convert")
        .addOptionalInput("Optional reference surface. Can be used "
            "to transfer attributes, sharpen features and to "
            "eliminate seams from fractured pieces.")
        .addOptionalInput("Optional VDB masks");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Convert::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Convert(net, name, op);
}


SOP_OpenVDB_Convert::SOP_OpenVDB_Convert(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


namespace {

/// @brief Convert a collection of OpenVDB grids into Houdini volumes.
/// @return @c true if all grids were successfully converted, @c false if one
/// or more grids were skipped due to unrecognized grid types.
void
convertToVolumes(
    GU_Detail& dst,
    GA_PrimitiveGroup* group,
#if HAVE_SPLITTING
    bool split_disjoint = false
#else
    bool = false
#endif
)
{
    GU_ConvertParms parms;

#if UT_VERSION_INT < 0x0d0000b1 // 13.0.177 or earlier
    parms.toType = GEO_PrimTypeCompat::GEOPRIMVOLUME;
#else
    parms.setToType(GEO_PrimTypeCompat::GEOPRIMVOLUME);
#endif
    parms.primGroup = group;
    parms.preserveGroups = true;
#if HAVE_SPLITTING
    GU_PrimVDB::convertVDBs(dst, dst, parms,
        /*adaptivity=*/0, /*keep_original*/false , split_disjoint);
#else
    GU_PrimVDB::convertVDBs(dst, dst, parms, /*adaptivity=*/0, /*keep_original*/false);
#endif
}


////////////////////////////////////////


void
convertToOpenVDB(
    GU_Detail& dst,
    GA_PrimitiveGroup* group,
    bool flood,
    bool prune,
    fpreal tolerance,
    bool activateinsidesdf)
{
    GU_ConvertParms parms;
    parms.primGroup = group;
    parms.preserveGroups = true;
    GU_PrimVDB::convertVolumesToVDBs(
        dst, dst, parms, flood, prune, tolerance, /*keep_original*/false
#if HAVE_ACTIVATEINSIDE
	, activateinsidesdf
#endif
	);
}


////////////////////////////////////////


void
convertVDBClass(
    GU_Detail& dst,
    GA_PrimitiveGroup* group,
    openvdb::GridClass new_class,
    float isovalue)
{
    using namespace openvdb;

    for (hvdb::VdbPrimIterator it(&dst, group); it; ++it) {

        if (it->getStorageType() != UT_VDB_FLOAT)
            continue;
        if (it->getGrid().getGridClass() == new_class)
            continue;

        switch (new_class) {
            case GRID_LEVEL_SET: { // from fog volume
                // *** FIXME:TODO: Hack until we have a good method ***
                // Convert to polygons
                FloatGrid &grid = UTvdbGridCast<FloatGrid>(it->getGrid());
                tools::VolumeToMesh mesher(isovalue);
                mesher(grid);
                // Convert to SDF
                math::Transform::Ptr transform = grid.transformPtr();
                std::vector<Vec3s> points;
                points.reserve(mesher.pointListSize());
                for (size_t i = 0, n = mesher.pointListSize(); i < n; i++) {
                    // The MeshToVolume conversion further down, requires the
                    // points to be in grid index space.
                    points.push_back(
                        transform->worldToIndex(mesher.pointList()[i]));
                }

                openvdb::tools::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();

                std::vector<Vec4I> primitives;
                size_t numPrimitives = 0;
                for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
                    const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
                    numPrimitives += polygons.numQuads();
                    numPrimitives += polygons.numTriangles();
                }
                primitives.reserve(numPrimitives);

                for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {

                    const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

                    // Copy quads
                    for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
                        primitives.push_back(polygons.quad(i));
                    }

                    // Copy triangles (adaptive mesh)
                    if (polygons.numTriangles() != 0) {
                        openvdb::Vec4I quad;
                        quad[3] = openvdb::util::INVALID_IDX;
                        for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {
                            const openvdb::Vec3I& triangle = polygons.triangle(i);
                            quad[0] = triangle[0];
                            quad[1] = triangle[1];
                            quad[2] = triangle[2];
                            primitives.push_back(quad);
                        }
                    }
                }

                openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec4I> mesh(points, primitives);

                openvdb::FloatGrid::Ptr sdfGrid = openvdb::tools::meshToVolume<openvdb::FloatGrid>(mesh, *transform);

                // Set grid and visualization
                it->setGrid(*sdfGrid);
                it->setVisualization(
                    GEO_VOLUMEVIS_ISO, it->getVisIso(), it->getVisDensity());
                break;
             } case GRID_FOG_VOLUME: { // from level set
                 it->makeGridUnique();
                 FloatGrid &grid = UTvdbGridCast<FloatGrid>(it->getGrid());
                 tools::sdfToFogVolume(grid, std::numeric_limits<float>::max());
                 it->setVisualization(GEO_VOLUMEVIS_SMOKE, it->getVisIso(), it->getVisDensity());
                 break;
             } default: {
                 // ignore everything else
                 break;
             }
        }
    }
}


////////////////////////////////////////


void
copyMesh(
    GU_Detail& detail,
    const GU_PrimVDB* srcvdb,
#if (UT_VERSION_INT < 0x0c0500F5) // earlier than 12.5.245
    GA_PrimitiveGroup* /*delgroup*/,
#else
    GA_PrimitiveGroup* delgroup,
#endif
    openvdb::tools::VolumeToMesh& mesher,
#if (UT_VERSION_INT < 0x0c0500F5) // earlier than 12.5.245
    bool /*toPolySoup*/,
#else
    bool toPolySoup,
#endif
    GA_PrimitiveGroup* surfaceGroup = NULL,
    GA_PrimitiveGroup* interiorGroup = NULL,
    GA_PrimitiveGroup* seamGroup = NULL,
    GA_PointGroup* seamPointGroup = NULL)
{
    const openvdb::tools::PointList& points = mesher.pointList();
    openvdb::tools::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();

    const char exteriorFlag = char(openvdb::tools::POLYFLAG_EXTERIOR);
    const char seamLineFlag = char(openvdb::tools::POLYFLAG_FRACTURE_SEAM);

    // Disable adding to seamPointGroup if we don't have pointFlags()
    if (mesher.pointFlags().size() != mesher.pointListSize()) {
        seamPointGroup = NULL;
    }

#if (UT_VERSION_INT < 0x0c0500F5) // earlier than 12.5.245

    bool groupSeamPoints = seamPointGroup && !mesher.pointFlags().empty();

    const GA_Offset lastIdx(detail.getNumPoints());
    for (size_t n = 0, N = mesher.pointListSize(); n < N; ++n) {
        GA_Offset ptoff = detail.appendPointOffset();
        detail.setPos3(ptoff, points[n].x(), points[n].y(), points[n].z());

        if (groupSeamPoints && mesher.pointFlags()[n]) {
            seamPointGroup->addOffset(ptoff);
        }
    }

    GU_ConvertMarker marker(detail);

    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {

        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

        // Copy quads
        for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {

            openvdb::Vec4I& quad = polygons.quad(i);
            GEO_PrimPoly& prim = *GU_PrimPoly::build(&detail, 4, GU_POLY_CLOSED, 0);

            for (int v = 0; v < 4; ++v) {
                prim(v).setPointOffset(lastIdx + quad[v]);
            }

            const bool surfacePrim = polygons.quadFlags(i) & exteriorFlag;
            if (surfaceGroup && surfacePrim) surfaceGroup->add(&prim);
            else if (interiorGroup && !surfacePrim) interiorGroup->add(&prim);

            if (seamGroup && (polygons.quadFlags(i) & seamLineFlag)) {
                seamGroup->add(&prim);
            }
        }


        // Copy triangles (if adaptive mesh.)
        for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {

            openvdb::Vec3I& triangle = polygons.triangle(i);
            GEO_PrimPoly& prim = *GU_PrimPoly::build(&detail, 3, GU_POLY_CLOSED, 0);

            for (int v = 0; v < 3; ++v) {
                prim(v).setPointOffset(lastIdx + triangle[v]);
            }

            const bool surfacePrim = (polygons.triangleFlags(i) & exteriorFlag);
            if (surfaceGroup && surfacePrim) surfaceGroup->add(&prim);
            else if (interiorGroup && !surfacePrim) interiorGroup->add(&prim);

            if (seamGroup && (polygons.triangleFlags(i) & seamLineFlag)) {
                seamGroup->add(&prim);
            }
        }
    }

    GA_Range primRange = marker.getPrimitives();
    GA_Range pntRange = marker.getPoints();
    GU_ConvertParms parms;
    parms.preserveGroups = true;
    GUconvertCopySingleVertexPrimAttribsAndGroups(parms,
        *srcvdb->getParent(), srcvdb->getMapOffset(), detail,
        primRange, pntRange);

#else // 12.5.245 or later
    GA_Size npoints = mesher.pointListSize();
    const GA_Offset startpt = detail.appendPointBlock(npoints);
    UT_ASSERT_COMPILETIME(sizeof(openvdb::tools::PointList::element_type) == sizeof(UT_Vector3));
    GA_RWHandleV3 pthandle(detail.getP());
    pthandle.setBlock(startpt, npoints, reinterpret_cast<UT_Vector3*>(points.get()));

    // group fracture seam points
    if (seamPointGroup && GA_Size(mesher.pointFlags().size()) == npoints) {
        GA_Offset ptoff = startpt;
        for (GA_Size i = 0; i < npoints; ++i) {

            if (mesher.pointFlags()[i]) {
                seamPointGroup->addOffset(ptoff);
            }
            ++ptoff;
        }
    }

    // index 0 --> interior, not on seam
    // index 1 --> interior, on seam
    // index 2 --> surface,  not on seam
    // index 3 --> surface,  on seam
    GA_Size nquads[4] = {0, 0, 0, 0};
    GA_Size ntris[4]  = {0, 0, 0, 0};
    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
        const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
        for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
            int flags = (((polygons.quadFlags(i) & exteriorFlag)!=0) << 1)
                       | ((polygons.quadFlags(i) & seamLineFlag)!=0);
            ++nquads[flags];
        }
        for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {
            int flags = (((polygons.triangleFlags(i) & exteriorFlag)!=0) << 1)
                       | ((polygons.triangleFlags(i) & seamLineFlag)!=0);
            ++ntris[flags];
        }
    }

    GA_Size nverts[4] = {
        nquads[0]*4 + ntris[0]*3,
        nquads[1]*4 + ntris[1]*3,
        nquads[2]*4 + ntris[2]*3,
        nquads[3]*4 + ntris[3]*3
    };
    UT_IntArray verts[4];
    for (int flags = 0; flags < 4; ++flags) {
#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later
        verts[flags].setCapacity(nverts[flags]);
#else
        verts[flags].resize(nverts[flags]);
#endif
        verts[flags].entries(nverts[flags]);
    }

    GA_Size iquad[4] = {0, 0, 0, 0};
    GA_Size itri[4]  = {nquads[0]*4, nquads[1]*4, nquads[2]*4, nquads[3]*4};

    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
        const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

        // Copy quads
        for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
            const openvdb::Vec4I& quad = polygons.quad(i);
            int flags = (((polygons.quadFlags(i) & exteriorFlag)!=0) << 1)
                       | ((polygons.quadFlags(i) & seamLineFlag)!=0);
            verts[flags](iquad[flags]++) = quad[0];
            verts[flags](iquad[flags]++) = quad[1];
            verts[flags](iquad[flags]++) = quad[2];
            verts[flags](iquad[flags]++) = quad[3];
        }

        // Copy triangles (adaptive mesh)
        for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {
            const openvdb::Vec3I& triangle = polygons.triangle(i);
            int flags = (((polygons.triangleFlags(i) & exteriorFlag)!=0) << 1)
                       | ((polygons.triangleFlags(i) & seamLineFlag)!=0);
            verts[flags](itri[flags]++) = triangle[0];
            verts[flags](itri[flags]++) = triangle[1];
            verts[flags](itri[flags]++) = triangle[2];
        }
    }

    bool shared_vertices = true;
    if (toPolySoup) {
        // NOTE: Since we could be using the same points for multiple
        //       polysoups, and the shared vertices option assumes that
        //       the points are only used by this polysoup, we have to
        //       use the unique vertices option.
        int num_prims = 0;
        for (int flags = 0; flags < 4; ++flags) {
            if (!nquads[flags] && !ntris[flags]) continue;
            num_prims++;
        }
        shared_vertices = (num_prims <= 1);
    }

    for (int flags = 0; flags < 4; ++flags) {
        if (!nquads[flags] && !ntris[flags]) continue;

        GEO_PolyCounts sizelist;
        if (nquads[flags]) sizelist.append(4, nquads[flags]);
        if (ntris[flags])  sizelist.append(3, ntris[flags]);

#if (UT_VERSION_INT >= 0x0d050013) // 13.5.19 or later
        GA_Detail::OffsetMarker marker(detail);
#else
        GU_ConvertMarker marker(detail);
#endif

        if (toPolySoup) {
            GU_PrimPolySoup::build(
                &detail, startpt, npoints, sizelist, verts[flags].array(), shared_vertices);
        } else {
            GU_PrimPoly::buildBlock(&detail, startpt, npoints, sizelist, verts[flags].array());
        }

#if (UT_VERSION_INT >= 0x0d050013) // 13.5.19 or later
        GA_Range range(marker.primitiveRange());
        GA_Range pntRange(marker.pointRange());
#else
        GA_Range range(marker.getPrimitives());
        GA_Range pntRange(marker.getPoints());
#endif
        GU_ConvertParms parms;
        parms.preserveGroups = true;
        GUconvertCopySingleVertexPrimAttribsAndGroups(parms,
            *srcvdb->getParent(), srcvdb->getMapOffset(), detail,
            range, pntRange);

        if (delgroup)                       delgroup->removeRange(range);
        if (seamGroup && (flags & 1))       seamGroup->addRange(range);
        if (surfaceGroup && (flags & 2))    surfaceGroup->addRange(range);
        if (interiorGroup && !(flags & 2))  interiorGroup->addRange(range);
    }
#endif // 12.5.245 or later
}

} // unnamed namespace


////////////////////////////////////////


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_Convert::updateParmsFlags()
{
    bool changed = false;
    const fpreal time = CHgetEvalTime();

    ConvertTo target = static_cast<ConvertTo>(evalInt("conversion", 0, time));
#if HAVE_SPLITTING
    bool toVolume = (target == HVOLUME);
#endif
    bool toOpenVDB = (target == OPENVDB);
    bool toPoly = (target == POLYGONS);
    bool toPolySoup = false;
#if HAVE_POLYSOUP
    toPolySoup = (target == POLYSOUP);
    toPoly |= toPolySoup;
#endif

    bool toSDF = (evalInt("vdbclass", 0, time) == CLASS_SDF);

    changed |= enableParm("adaptivity", toPoly);
    changed |= enableParm("isoValue", toPoly || (toOpenVDB && toSDF));
    changed |= enableParm("fogisovalue", toOpenVDB && toSDF);

    if (toOpenVDB) {
        changed |= enableParm("tolerance", evalInt("prune",  0, time));
    }

    bool refexists = (nInputs() == 2);
    changed |= enableParm("computenormals", toPoly && !toPolySoup);
    changed |= enableParm("internaladaptivity", toPoly && refexists);
    changed |= enableParm("surfacegroup", toPoly && refexists);
    changed |= enableParm("interiorgroup", toPoly && refexists);
    changed |= enableParm("seamlinegroup", toPoly && refexists);
    changed |= enableParm("seampoints", toPoly && refexists);
    changed |= enableParm("transferattributes", toPoly && refexists);
    changed |= enableParm("sharpenfeatures", toPoly && refexists);
    changed |= enableParm("edgetolerance", toPoly && refexists);


    const bool maskexists = (nInputs() == 3);


    changed |= enableParm("surfacemask", toPoly && maskexists);
    changed |= enableParm("adaptivityfield", toPoly && maskexists);

    const bool surfacemask = bool(evalInt("surfacemask", 0, 0));
    changed |= enableParm("surfacemaskname", toPoly && maskexists && surfacemask);
    changed |= enableParm("surfacemaskoffset", toPoly && maskexists && surfacemask);
    changed |= enableParm("invertmask", toPoly && maskexists && surfacemask);


    changed |= enableParm("adaptivityfield", toPoly && maskexists);

    const bool adaptivityfield = bool(evalInt("adaptivityfield", 0, 0));
    changed |= enableParm("adaptivityfieldname", toPoly && maskexists && adaptivityfield);

#if HAVE_SPLITTING
    changed |= setVisibleState("splitdisjointvolumes", toVolume);
#endif

    changed |= setVisibleState("adaptivity", toPoly);
    changed |= setVisibleState("isoValue", toPoly);
    changed |= setVisibleState("fogisovalue", toOpenVDB);
    changed |= setVisibleState("computenormals", toPoly);

    changed |= setVisibleState("internaladaptivity", toPoly);
    changed |= setVisibleState("transferattributes", toPoly);
    changed |= setVisibleState("sharpenfeatures", toPoly);
    changed |= setVisibleState("edgetolerance", toPoly);
    changed |= setVisibleState("surfacegroup", toPoly);
    changed |= setVisibleState("interiorgroup", toPoly);
    changed |= setVisibleState("seamlinegroup", toPoly);
    changed |= setVisibleState("seampoints", toPoly);

    changed |= setVisibleState("surfacemask", toPoly);
    changed |= setVisibleState("surfacemaskname", toPoly);
    changed |= setVisibleState("surfacemaskoffset", toPoly);
    changed |= setVisibleState("invertmask", toPoly);
    changed |= setVisibleState("adaptivityfield", toPoly);
    changed |= setVisibleState("adaptivityfieldname", toPoly);

    changed |= setVisibleState("flood", toOpenVDB);
    changed |= setVisibleState("prune", toOpenVDB);
    changed |= setVisibleState("tolerance", toOpenVDB);
    changed |= setVisibleState("vdbclass", toOpenVDB);

#if HAVE_ACTIVATEINSIDE
    changed |= setVisibleState("activateinsidesdf", toOpenVDB);
    if (toOpenVDB)
	changed |= enableParm("activateinsidesdf", evalInt("flood",  0, time));
#endif

    return changed;
}


////////////////////////////////////////


template <class GridType>
void
SOP_OpenVDB_Convert::referenceMeshing(
    std::list<openvdb::GridBase::ConstPtr>& grids,
    std::list<const GU_PrimVDB*> vdbs,
    GA_PrimitiveGroup* delgroup,
    openvdb::tools::VolumeToMesh& mesher,
    const GU_Detail* refGeo,
    bool computeNormals,
    hvdb::Interrupter& boss,
    const fpreal time)
{
    if (refGeo == NULL) return;

    typedef typename GridType::TreeType TreeType;
    typedef typename GridType::ValueType ValueType;

    const bool transferAttributes = evalInt("transferattributes", 0, time);
    const bool sharpenFeatures = evalInt("sharpenfeatures", 0, time);

    // Get the first grid's transform and background value.
    openvdb::math::Transform::Ptr transform = grids.front()->transform().copy();

    typename GridType::ConstPtr firstGrid = openvdb::gridConstPtrCast<GridType>(grids.front());

    if (!firstGrid) {
        addError(SOP_MESSAGE, "Unsupported grid type.");
        return;
    }

    const ValueType backgroundValue = firstGrid->background();
    const openvdb::GridClass gridClass = firstGrid->getGridClass();

    typename GridType::ConstPtr refGrid;

    typedef typename GridType::template ValueConverter<openvdb::Int32>::Type IntGridT;
    typename IntGridT::Ptr indexGrid;

    openvdb::tools::MeshToVoxelEdgeData edgeData;

    boost::shared_ptr<GU_Detail> geoPtr;
    if (!refGrid) {
        std::string warningStr;
        geoPtr = hvdb::validateGeometry(*refGeo, warningStr, &boss);

        if (geoPtr) {
            refGeo = geoPtr.get();
            if (!warningStr.empty()) addWarning(SOP_MESSAGE, warningStr.c_str());
        }

        std::vector<openvdb::Vec3s> pointList;
        std::vector<openvdb::Vec4I> primList;

        pointList.resize(refGeo->getNumPoints());
        primList.resize(refGeo->getNumPrimitives());

        UTparallelFor(GA_SplittableRange(refGeo->getPointRange()),
            hvdb::TransformOp(refGeo, *transform, pointList));

        UTparallelFor(GA_SplittableRange(refGeo->getPrimitiveRange()),
            hvdb::PrimCpyOp(refGeo, primList));

        if (boss.wasInterrupted()) return;

        openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec4I> mesh(pointList, primList);

        float bandWidth = 3.0;

        if (gridClass != openvdb::GRID_LEVEL_SET) {
            bandWidth = float(backgroundValue) / float(transform->voxelSize()[0]);
        }

        indexGrid.reset(new IntGridT(0));

        refGrid = openvdb::tools::meshToVolume<GridType>(boss,
            mesh, *transform, bandWidth, bandWidth, 0, indexGrid.get());

        if (sharpenFeatures) edgeData.convert(pointList, primList);
    }

    if (boss.wasInterrupted()) return;

    typedef typename TreeType::template ValueConverter<bool>::Type BoolTreeType;
    typename BoolTreeType::Ptr maskTree;

    if (sharpenFeatures) {

        const float edgetolerance = static_cast<float>(evalFloat("edgetolerance", 0, time));

        maskTree = typename BoolTreeType::Ptr(new BoolTreeType(false));
        maskTree->topologyUnion(indexGrid->tree());
        openvdb::tree::LeafManager<BoolTreeType> maskLeafs(*maskTree);

        hvdb::GenAdaptivityMaskOp<typename IntGridT::TreeType, BoolTreeType>
            op(*refGeo, indexGrid->tree(), maskLeafs, edgetolerance);
        op.run();

        openvdb::tools::pruneInactive(*maskTree);

        openvdb::tools::dilateVoxels(*maskTree, 2);

        mesher.setAdaptivityMask(maskTree);
    }


    if (boss.wasInterrupted()) return;

    const double iadaptivity = double(evalFloat("internaladaptivity", 0, time));
    mesher.setRefGrid(refGrid, iadaptivity);


    std::vector<std::string> badTransformList, badBackgroundList, badTypeList;

    GA_PrimitiveGroup *surfaceGroup = NULL, *interiorGroup = NULL, *seamGroup = NULL;
    GA_PointGroup* seamPointGroup = NULL;

    {
        UT_String newGropStr;
        evalString(newGropStr, "surfacegroup", 0, time);
        if(newGropStr.length() > 0) {
            surfaceGroup = gdp->findPrimitiveGroup(newGropStr);
            if (!surfaceGroup) surfaceGroup = gdp->newPrimitiveGroup(newGropStr);
        }

        evalString(newGropStr, "interiorgroup", 0, time);
        if(newGropStr.length() > 0) {
            interiorGroup = gdp->findPrimitiveGroup(newGropStr);
            if (!interiorGroup) interiorGroup = gdp->newPrimitiveGroup(newGropStr);
        }

        evalString(newGropStr, "seamlinegroup", 0, time);
        if(newGropStr.length() > 0) {
            seamGroup = gdp->findPrimitiveGroup(newGropStr);
            if (!seamGroup) seamGroup = gdp->newPrimitiveGroup(newGropStr);
        }

        evalString(newGropStr, "seampoints", 0, time);
        if(newGropStr.length() > 0) {
            seamPointGroup = gdp->findPointGroup(newGropStr);
            if (!seamPointGroup) seamPointGroup = gdp->newPointGroup(newGropStr);
        }
    }

    std::vector<typename GridType::ConstPtr> fragments;
    std::vector<const GU_PrimVDB*> fragment_vdbs;
    std::list<openvdb::GridBase::ConstPtr>::iterator it = grids.begin();
    std::list<const GU_PrimVDB*>::iterator vdbit = vdbs.begin();

    for (; it != grids.end(); ++it, ++vdbit) {

        if (boss.wasInterrupted()) break;

        typename GridType::ConstPtr grid = openvdb::gridConstPtrCast<GridType>(*it);

        if (!grid) {
            badTypeList.push_back(grid->getName());
            continue;
        }

        if (grid->transform() != *transform) {
            badTransformList.push_back(grid->getName());
            continue;
        }

        if (!openvdb::math::isApproxEqual(grid->background(), backgroundValue)) {
            badBackgroundList.push_back(grid->getName());
            continue;
        }

        fragments.push_back(grid);
        fragment_vdbs.push_back(*vdbit);
    }

    grids.clear();

    for (size_t i = 0, I = fragments.size(); i < I; ++i) {
        mesher(*fragments[i]);
#if HAVE_POLYSOUP
        ConvertTo target = static_cast<ConvertTo>(evalInt("conversion", 0, time));
        bool toPolySoup = (target == POLYSOUP);
#else
        bool toPolySoup = false;
#endif
        copyMesh(*gdp, fragment_vdbs[i], delgroup, mesher, toPolySoup,
            surfaceGroup, interiorGroup, seamGroup, seamPointGroup);
    }

    // Sharpen Features
    if (!boss.wasInterrupted() && sharpenFeatures) {
        UTparallelFor(GA_SplittableRange(gdp->getPointRange()),
            hvdb::SharpenFeaturesOp(*gdp, *refGeo, edgeData, *transform,
                surfaceGroup, maskTree.get()));
    }

    // Compute vertex normals
    if (!boss.wasInterrupted() && computeNormals) {

        UTparallelFor(GA_SplittableRange(gdp->getPrimitiveRange()),
            hvdb::VertexNormalOp(*gdp, interiorGroup, (transferAttributes ? -1.0f : 0.7f) ));

        if (!interiorGroup) {
            addWarning(SOP_MESSAGE, "More accurate vertex normals can be generated "
                "if the interior polygon group is enabled.");
        }
    }

    // Transfer Primitive Attributes
    if (!boss.wasInterrupted() && transferAttributes && refGeo && indexGrid) {
        hvdb::transferPrimitiveAttributes(*refGeo, *gdp, *indexGrid, boss, surfaceGroup);
    }


    if (!badTransformList.empty()) {
        std::string s = "The following grids were skipped: '" +
            boost::algorithm::join(badTransformList, ", ") +
            "' because they don't match the transform of the first grid.";
        addWarning(SOP_MESSAGE, s.c_str());
    }

    if (!badBackgroundList.empty()) {
        std::string s = "The following grids were skipped: '" +
            boost::algorithm::join(badBackgroundList, ", ") +
            "' because they don't match the background value of the first grid.";
        addWarning(SOP_MESSAGE, s.c_str());
    }

    if (!badTypeList.empty()) {
        std::string s = "The following grids were skipped: '" +
            boost::algorithm::join(badTypeList, ", ") +
            "' because they don't have the same data type as the first grid.";
        addWarning(SOP_MESSAGE, s.c_str());
    }
}

void
SOP_OpenVDB_Convert::convertToPoly(
    fpreal time,
    GA_PrimitiveGroup *group,
    bool buildpolysoup,
    hvdb::Interrupter &boss)
{

    hvdb::VdbPrimCIterator vdbIt(gdp, group);
    if (!vdbIt) {
        addWarning(SOP_MESSAGE, "No VDB primitives found.");
        return;
    }


    const bool      computeNormals = !buildpolysoup && (evalInt("computenormals", 0, time) != 0);
    const bool      invertmask = evalInt("invertmask", 0, time);
    const fpreal    adaptivity = evalFloat("adaptivity", 0, time);
    const fpreal    iso = evalFloat("isoValue", 0, time);
    const fpreal    maskoffset = evalFloat("surfacemaskoffset", 0, time);


    openvdb::tools::VolumeToMesh mesher(iso, adaptivity);

    // Check mask input
    const GU_Detail* maskGeo = inputGeo(2);
    if (maskGeo) {

        if (evalInt("surfacemask", 0, time)) {
            UT_String maskStr;
            evalString(maskStr, "surfacemaskname", 0, time);
#if (UT_MAJOR_VERSION_INT >= 15)
            const GA_PrimitiveGroup * maskGroup =
                parsePrimitiveGroups(maskStr.buffer(), GroupCreator(maskGeo));
#else
            const GA_PrimitiveGroup * maskGroup =
                parsePrimitiveGroups(maskStr.buffer(), const_cast<GU_Detail*>(maskGeo));
#endif

            if (!maskGroup && maskStr.length() > 0) {
                addWarning(SOP_MESSAGE, "Surface mask not found.");
            } else {
                hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);
                if (maskIt) {
                    const openvdb::GridClass gridClass = maskIt->getGrid().getGridClass();
                    if (gridClass == openvdb::GRID_LEVEL_SET) {

                        openvdb::FloatGrid::ConstPtr grid =
                            openvdb::gridConstPtrCast<openvdb::FloatGrid>(maskIt->getGridPtr());

                        mesher.setSurfaceMask(
                            openvdb::tools::sdfInteriorMask(*grid, static_cast<float>(maskoffset)),
                            invertmask);
                    } else {
                        addWarning(SOP_MESSAGE, "Currently only supporting level set masks.");
                    }
                }
            }

        }


        if (evalInt("adaptivityfield", 0, time)) {
            UT_String maskStr;
            evalString(maskStr, "adaptivityfieldname", 0, time);

            const GA_PrimitiveGroup *maskGroup =
                matchGroup(const_cast<GU_Detail&>(*maskGeo), maskStr.toStdString());

            if (!maskGroup && maskStr.length() > 0) {
                addWarning(SOP_MESSAGE, "Adaptivity field not found.");
            } else {
                hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);
                if (maskIt) {
                    openvdb::FloatGrid::ConstPtr grid =
                        openvdb::gridConstPtrCast<openvdb::FloatGrid>(maskIt->getGridPtr());

                    mesher.setSpatialAdaptivity(grid);
                }
            }
        }
    }


    const GU_Detail* refGeo = inputGeo(1);
    GU_ConvertParms parms;
    GA_PrimitiveGroup *delGroup = parms.getDeletePrimitives(gdp);

    if (refGeo) {
        // Collect all level set grids.
        std::list<openvdb::GridBase::ConstPtr> grids;
        std::list<const GU_PrimVDB*> vdbs;
        std::vector<std::string> nonLevelSetList, nonLinearList;
        for (; vdbIt; ++vdbIt) {
            if (boss.wasInterrupted()) break;

            const openvdb::GridClass gridClass = vdbIt->getGrid().getGridClass();
            if (gridClass != openvdb::GRID_LEVEL_SET) {
                nonLevelSetList.push_back(vdbIt.getPrimitiveNameOrIndex().toStdString());
                continue;
            }

            if (!vdbIt->getGrid().transform().isLinear()) {
                nonLinearList.push_back(vdbIt.getPrimitiveNameOrIndex().toStdString());
                continue;
            }

            delGroup->addOffset(vdbIt.getOffset());
            grids.push_back(vdbIt->getConstGridPtr());
            vdbs.push_back(*vdbIt);
        }

        if (!nonLevelSetList.empty()) {
            std::string s = "Reference meshing is only supported for "
                "Level Set grids, the following grids were skipped: '" +
                boost::algorithm::join(nonLevelSetList, ", ") + "'.";
            addWarning(SOP_MESSAGE, s.c_str());
        }

        if (!nonLinearList.empty()) {
            std::string s = "The following grids were skipped: '" +
                boost::algorithm::join(nonLinearList, ", ") +
                "' because they don't have a linear/affine transform.";
            addWarning(SOP_MESSAGE, s.c_str());
        }

        // Mesh using a reference surface
        if (!grids.empty() && !boss.wasInterrupted()) {

            if (grids.front()->isType<openvdb::FloatGrid>()) {
                referenceMeshing<openvdb::FloatGrid>(
                    grids, vdbs, delGroup, mesher, refGeo, computeNormals, boss, time);
            } else if (grids.front()->isType<openvdb::DoubleGrid>()) {
                referenceMeshing<openvdb::DoubleGrid>(
                    grids, vdbs, delGroup, mesher, refGeo, computeNormals, boss, time);
            } else {
                addError(SOP_MESSAGE, "Unsupported grid type.");
            }

            // Delete old VDB primitives
            if (error() < UT_ERROR_ABORT)
                gdp->destroyPrimitives(gdp->getPrimitiveRange(delGroup), /*and_points*/true);
        }

        if (delGroup) gdp->destroyGroup(delGroup);

    } else {

#if HAVE_POLYSOUP
        ConvertTo target = static_cast<ConvertTo>(evalInt("conversion", 0, time));
        bool toPolySoup = (target == POLYSOUP);
#else
        bool toPolySoup = false;
#endif

        // Mesh each VDB primitive independently
        for (; vdbIt; ++vdbIt) {

            if (boss.wasInterrupted()) break;
            GEOvdbProcessTypedGridScalar(*vdbIt.getPrimitive(), mesher);

            delGroup->addOffset(vdbIt.getOffset());

            copyMesh(*gdp, *vdbIt, delGroup, mesher, toPolySoup);

        }

        // Delete old VDB primitives
        if (error() < UT_ERROR_ABORT)
            gdp->destroyPrimitives(gdp->getPrimitiveRange(delGroup), /*and_points*/true);

        if (!boss.wasInterrupted() && computeNormals) {
            UTparallelFor(GA_SplittableRange(gdp->getPrimitiveRange()),
                hvdb::VertexNormalOp(*gdp));
        }

        if (delGroup) gdp->destroyGroup(delGroup);
    }
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Convert::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        // We are intentionally not performing a duplicateSourceStealable() here due to
        // specific implementation in this SOP which causes undesirable behavior when
        // attempting to "steal" the geometry
        duplicateSource(0, context);

        const fpreal t = context.getTime();

        UT_String group_str;
        evalString(group_str, "group", 0, t);

#if (UT_MAJOR_VERSION_INT >= 15)
        GA_PrimitiveGroup* group = parsePrimitiveGroupsCopy(group_str, GroupCreator(gdp));
#else
        GA_PrimitiveGroup* group = parsePrimitiveGroupsCopy(group_str, gdp);
#endif

        hvdb::Interrupter interrupter("Convert");

        switch (evalInt("conversion",  0, t))
        {
            case HVOLUME: {
#if HAVE_SPLITTING
                const bool splitDisjointVols = (evalInt("splitdisjointvolumes", 0, t) != 0);
#else
                const bool splitDisjointVols = false;
#endif
                convertToVolumes(*gdp, group, splitDisjointVols);
                break;
            }
            case OPENVDB: {
#if HAVE_ACTIVATEINSIDE
		const bool activateinside = (evalInt("activateinsidesdf", 0, t) != 0);
#else
		const bool activateinside = true;
#endif
                convertToOpenVDB(*gdp, group,
                    (evalInt("flood", 0, t) != 0),
                    (evalInt("prune", 0, t) != 0),
                    evalFloat("tolerance", 0, t),
		    activateinside);

                switch (evalInt("vdbclass", 0, t)) {
                    case CLASS_SDF:
                        convertVDBClass(*gdp, group, openvdb::GRID_LEVEL_SET,
                            static_cast<float>(evalFloat("fogisovalue", 0, t)));
                        break;
                    case CLASS_FOG_VOLUME:
                        convertVDBClass(*gdp, group, openvdb::GRID_FOG_VOLUME, /*unused*/0);
                        break;
                    default:
                        // ignore
                        break;
                }
                break;
            }
            case POLYGONS: {
                convertToPoly(t, group, false, interrupter);
                break;
            }

#if HAVE_POLYSOUP
            case POLYSOUP: {
                convertToPoly(t, group, true, interrupter);
                break;
            }
#endif

            default: {
                addWarning(SOP_MESSAGE, "Unrecognized conversion type");
                break;
            }
        }

        if (interrupter.wasInterrupted()) {
            addWarning(SOP_MESSAGE, "Process was interrupted");
        }

        interrupter.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
