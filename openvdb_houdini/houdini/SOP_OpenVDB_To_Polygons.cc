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
/// @file SOP_OpenVDB_To_Polygons.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief OpenVDB level set to polygon conversion

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/AttributeTransferUtil.h>
#include <openvdb_houdini/GeometryUtil.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/GeometryUtil.h>

#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/Mask.h> // for tools::interiorMask()
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/math/Operators.h>
#include <openvdb/math/Mat3.h>

#include <CH/CH_Manager.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_Version.h>
#include <GA/GA_PageIterator.h>

#if (UT_VERSION_INT >= 0x0c0500F5) // 12.5.245 or later
#include <GEO/GEO_PolyCounts.h>
#endif

#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
#define HAVE_POLYSOUP 1
#include <GU/GU_PrimPolySoup.h>
#else
#define HAVE_POLYSOUP 0
#endif


#include <GU/GU_Detail.h>
#include <GU/GU_Surfacer.h>
#include <GU/GU_PolyReduce.h>
#include <GU/GU_PrimPoly.h>
#include <GU/GU_ConvertParms.h>
#include <PRM/PRM_Parm.h>


#include <boost/algorithm/string/join.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/math/special_functions/round.hpp>
#include <string>
#include <list>
#include <vector>


#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
// GA_RWHandleV3 fails to initialize its member variables in some cases.
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


class SOP_OpenVDB_To_Polygons: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_To_Polygons(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_To_Polygons() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned i) const override { return (i > 0); }

protected:
    OP_ERROR cookMySop(OP_Context&) override;
    bool updateParmsFlags() override;
    void resolveObsoleteParms(PRM_ParmList*) override;

    template <class GridType>
    void referenceMeshing(
        std::list<openvdb::GridBase::ConstPtr>& grids,
        openvdb::tools::VolumeToMesh& mesher,
        const GU_Detail* refGeo,
        hvdb::Interrupter& boss,
        const fpreal time);
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of the input VDB grids to surface.")
        .setDocumentation(
            "A subset of the input VDB grids to be surfaced"
            " (see [specifying volumes|/model/volumes#group])"));

    { // Geometry Type
        char const * const items[] = {
            "polysoup", "Polygon Soup",
            "poly",     "Polygons",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "geometrytype", "Geometry Type")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip(
                "Specify the type of geometry to output. A polygon soup is a primitive"
                " that stores polygons using a compact memory representation."
                " Not all geometry nodes can operate directly on this primitive.")
            .setDocumentation(
                "The type of geometry to output, either polygons or a polygon soup\n\n"
                "A [polygon soup|/model/primitives#polysoup] is a primitive"
                " that stores polygons using a compact memory representation.\n\n"
                "WARNING:\n"
                "    Not all geometry nodes can operate directly on polygon soups.\n"));
    }

    parms.add(hutil::ParmFactory(PRM_FLT_J, "isovalue", "Isovalue")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_UI, -1.0, PRM_RANGE_UI, 1.0)
        .setTooltip(
            "The voxel value that determines the surface\n\n"
            "Zero works for signed distance fields, while fog volumes require"
            " a larger positive value (0.5 is a good initial guess)."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "adaptivity", "Adaptivity")
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0)
        .setTooltip(
            "The adaptivity threshold determines how closely the output mesh follows"
            " the isosurface.  A higher threshold enables more variation in polygon size,"
            " allowing the surface to be represented with fewer polygons."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "computenormals", "Compute Vertex Normals")
        .setTooltip("Compute edge-preserving vertex normals.")
        .setDocumentation(
            "Compute edge-preserving vertex normals."
            " This uses the optional second input to help eliminate seams."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "keepvdbname", "Preserve VDB Name")
        .setTooltip("Mark each primitive with the corresponding VDB name."));

    //////////

    parms.add(hutil::ParmFactory(PRM_HEADING,"sep1", "Reference Options"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "internaladaptivity", "Internal Adaptivity")
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0)
        .setTooltip("Overrides the adaptivity threshold for all internal surfaces.")
        .setDocumentation(
            "When a reference surface is provided, this is the adaptivity threshold"
            " for regions that are inside the surface."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "transferattributes", "Transfer Surface Attributes")
        .setTooltip("Transfers all attributes (primitive, vertex and point) from the "
            "reference surface. Will override computed vertex normals for primitives "
            " in the surface group.")
        .setDocumentation(
            "When a reference surface is provided, this option transfers all attributes\n"
            "(primitive, vertex and point) from the reference surface.\n\n"
            "NOTE:\n"
            "    Computed vertex normals for primitives in the surface group\n"
            "    will be overridden.\n"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "sharpenfeatures", "Sharpen Features")
        .setDefault(PRMoneDefaults)
        .setTooltip("Sharpen edges and corners."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "edgetolerance", "Edge Tolerance")
        .setDefault(0.5)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0)
        .setTooltip("Controls the edge adaptivity mask."));

    parms.add(hutil::ParmFactory(PRM_STRING, "surfacegroup", "Surface Group")
        .setDefault("surface_polygons")
        .setTooltip(
            "Specify a group for all polygons that are coincident with the reference surface.\n\n"
            "The group is useful for transferring attributes such as UV coordinates,"
            " normals, etc. from the reference surface."));

    parms.add(hutil::ParmFactory(PRM_STRING, "interiorgroup", "Interior Group")
        .setDefault("interior_polygons")
        .setTooltip(
            "Specify a group for all polygons that are interior to the reference surface.\n\n"
            "The group can be used to identify surface regions that might require"
            " projected UV coordinates or new materials."));

    parms.add(hutil::ParmFactory(PRM_STRING, "seamlinegroup", "Seam Line Group")
        .setDefault("seam_polygons")
        .setTooltip(
            "Specify a group for all polygons that are in proximity to the seam lines.\n\n"
            "This group can be used to drive secondary elements such as debris and dust."));

    parms.add(hutil::ParmFactory(PRM_STRING, "seampoints", "Seam Points")
        .setDefault("seam_points")
        .setTooltip(
            "Specify a group of the fracture seam points.\n\n"
            "This can be used to drive local pre-fracture dynamics,"
            " e.g., local surface buckling."));

    //////////

    parms.add(hutil::ParmFactory(PRM_HEADING,"sep2", "Masking Options"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "surfacemask", "")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setTooltip("Enable / disable the surface mask."));

    parms.add(hutil::ParmFactory(PRM_STRING, "surfacemaskname", "Surface Mask")
        .setChoiceList(&hutil::PrimGroupMenuInput3)
        .setTooltip(
            "A single VDB whose active voxels or (if the VDB is a level set or SDF)\n"
            "interior voxels define the region to be meshed"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "surfacemaskoffset", "Offset")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_UI, -1.0, PRM_RANGE_UI, 1.0)
        .setTooltip(
            "Isovalue that determines the interior of the surface mask\n"
            "when the mask is a level set or SDF"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "invertsurfacemask", "Invert Surface Mask")
        .setTooltip("If enabled, mesh the complement of the mask."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "adaptivityfield", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setTooltip("Enable / disable the the adaptivity field."));

    parms.add(hutil::ParmFactory(PRM_STRING, "adaptivityfieldname", "Adaptivity Field")
        .setChoiceList(&hutil::PrimGroupMenuInput3)
        .setTooltip(
            "A single scalar VDB to be used as a spatial multiplier"
            " for the adaptivity threshold"));

    //////////

    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "smoothseams", "Smooth Seams"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "invertmask", "").setDefault(PRMoneDefaults));
    obsoleteParms.add(hutil::ParmFactory(PRM_INT_J, "automaticpartitions", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_INT_J, "activepart", ""));

    hvdb::OpenVDBOpFactory("OpenVDB To Polygons", SOP_OpenVDB_To_Polygons::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("OpenVDB grids to surface")
        .addOptionalInput("Optional reference surface. Can be used "
            "to transfer attributes, sharpen features and to "
            "eliminate seams from fractured pieces.")
        .addOptionalInput("Optional VDB masks")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Convert VDB volumes into polygonal meshes.\"\"\"\n\
\n\
@overview\n\
\n\
This node converts the surfaces of VDB volumes, including level sets,\n\
into polygonal meshes.\n\
\n\
The second and third inputs are optional.\n\
The second input provides a reference polygon surface, which is useful\n\
for converting fractured VDBs back to polygons.\n\
The third input provides additional VDBs that can be used for masking\n\
(specifying which voxels to convert to polygons) and/or to specify\n\
an adaptivity multiplier.\n\
\n\
@related\n\
- [OpenVDB Convert|Node:sop/DW_OpenVDBConvert]\n\
- [OpenVDB Create|Node:sop/DW_OpenVDBCreate]\n\
- [OpenVDB From Particles|Node:sop/DW_OpenVDBFromParticles]\n\
- [Node:sop/convert]\n\
- [Node:sop/convertvolume]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


OP_Node*
SOP_OpenVDB_To_Polygons::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_To_Polygons(net, name, op);
}


SOP_OpenVDB_To_Polygons::SOP_OpenVDB_To_Polygons(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


void
SOP_OpenVDB_To_Polygons::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;

    // using the invertmask attribute to detect old houdini files that
    // had the regular polygon representation.
    PRM_Parm* parm = obsoleteParms->getParmPtr("invertmask");
    if (parm && !parm->isFactoryDefault()) {
        setInt("geometrytype", 0, 0, 1);
    }
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}


bool
SOP_OpenVDB_To_Polygons::updateParmsFlags()
{
    bool changed = false;
    const fpreal time = CHgetEvalTime();

    const bool refexists = (nInputs() == 2);
    bool usePolygonSoup = evalInt("geometrytype", 0, time) == 0;

#if !HAVE_POLYSOUP
    changed |= setVisibleState("geometrytype", false);
    usePolygonSoup = false;
#endif

    changed |= enableParm("computenormals", !usePolygonSoup);
    changed |= enableParm("internaladaptivity", refexists);
    changed |= enableParm("surfacegroup", refexists);
    changed |= enableParm("interiorgroup", refexists);
    changed |= enableParm("seamlinegroup", refexists);
    changed |= enableParm("seampoints", refexists);
    changed |= enableParm("transferattributes", refexists);
    changed |= enableParm("sharpenfeatures", refexists);
    changed |= enableParm("edgetolerance", refexists);

    const bool maskexists = (nInputs() == 3);

    changed |= enableParm("surfacemask", maskexists);
    changed |= enableParm("adaptivitymask", maskexists);

    const bool surfacemask = bool(evalInt("surfacemask", 0, 0));
    changed |= enableParm("surfacemaskname", maskexists && surfacemask);
    changed |= enableParm("surfacemaskoffset", maskexists && surfacemask);
    changed |= enableParm("invertsurfacemask", maskexists && surfacemask);

    const bool adaptivitymask = bool(evalInt("adaptivityfield", 0, 0));
    changed |= enableParm("adaptivityfieldname", maskexists && adaptivitymask);

    return changed;
}


////////////////////////////////////////


void copyMesh(GU_Detail&, openvdb::tools::VolumeToMesh&, hvdb::Interrupter&,
    const bool usePolygonSoup = true, const char* gridName = nullptr,
    GA_PrimitiveGroup* surfaceGroup = nullptr, GA_PrimitiveGroup* interiorGroup = nullptr,
    GA_PrimitiveGroup* seamGroup = nullptr, GA_PointGroup* seamPointGroup = nullptr);


void
copyMesh(
    GU_Detail& detail,
    openvdb::tools::VolumeToMesh& mesher,
#if (UT_VERSION_INT < 0x0c0500F5) // earlier than 12.5.245
    hvdb::Interrupter& boss,
#else
    hvdb::Interrupter&,
#endif
    const bool usePolygonSoup,
    const char* gridName,
    GA_PrimitiveGroup* surfaceGroup,
    GA_PrimitiveGroup* interiorGroup,
    GA_PrimitiveGroup* seamGroup,
    GA_PointGroup* seamPointGroup)
{
    const openvdb::tools::PointList& points = mesher.pointList();
    openvdb::tools::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();

    const char exteriorFlag = char(openvdb::tools::POLYFLAG_EXTERIOR);
    const char seamLineFlag = char(openvdb::tools::POLYFLAG_FRACTURE_SEAM);

    const GA_Index firstPrim = detail.getNumPrimitives();

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

    if (boss.wasInterrupted()) return;

    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {

        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

        if (boss.wasInterrupted()) break;

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

        if (boss.wasInterrupted()) break;

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
    if (usePolygonSoup) {
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

        if (usePolygonSoup) {
            GU_PrimPolySoup::build(
                &detail, startpt, npoints, sizelist, verts[flags].array(), shared_vertices);
        } else {
            GU_PrimPoly::buildBlock(&detail, startpt, npoints, sizelist, verts[flags].array());
        }

#if (UT_VERSION_INT >= 0x0d050013) // 13.5.19 or later
        GA_Range range(marker.primitiveRange());
        //GA_Range pntRange(marker.pointRange());
#else
        GA_Range range(marker.getPrimitives());
        //GA_Range pntRange(marker.getPoints());
#endif
        /*GU_ConvertParms parms;
        parms.preserveGroups = true;
        GUconvertCopySingleVertexPrimAttribsAndGroups(parms,
            *srcvdb->getParent(), srcvdb->getMapOffset(), detail,
            range, pntRange);*/

        //if (delgroup)                       delgroup->removeRange(range);
        if (seamGroup && (flags & 1))       seamGroup->addRange(range);
        if (surfaceGroup && (flags & 2))    surfaceGroup->addRange(range);
        if (interiorGroup && !(flags & 2))  interiorGroup->addRange(range);
    }

#endif // 12.5.245 or later


    // Keep VDB grid name
    const GA_Index lastPrim = detail.getNumPrimitives();
    if (gridName != nullptr && firstPrim != lastPrim) {

        GA_RWAttributeRef aRef = detail.findPrimitiveAttribute("name");
        if (!aRef.isValid()) aRef = detail.addStringTuple(GA_ATTRIB_PRIMITIVE, "name", 1);

        GA_Attribute * nameAttr = aRef.getAttribute();
        if (nameAttr) {
            const GA_AIFSharedStringTuple * stringAIF = nameAttr->getAIFSharedStringTuple();
            if (stringAIF) {
                GA_Range range(detail.getPrimitiveMap(),
                    detail.primitiveOffset(firstPrim),  detail.primitiveOffset(lastPrim));
                stringAIF->setString(nameAttr, range, gridName, 0);
            }
        }
    }
}


////////////////////////////////////////


namespace {

struct InteriorMaskOp
{
    InteriorMaskOp(double iso = 0.0): inIsovalue(iso) {}

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        outGridPtr = openvdb::tools::interiorMask(grid, inIsovalue);
    }

    const double inIsovalue;
    openvdb::BoolGrid::Ptr outGridPtr;
};


// Extract a boolean mask from a grid of any type.
inline hvdb::GridCPtr
getMaskFromGrid(const hvdb::GridCPtr& gridPtr, double isovalue = 0.0)
{
    hvdb::GridCPtr maskGridPtr;
    if (gridPtr) {
        if (gridPtr->isType<openvdb::BoolGrid>()) {
            // If the input grid is already boolean, return it.
            maskGridPtr = gridPtr;
        } else {
            InteriorMaskOp op{isovalue};
            UTvdbProcessTypedGridTopology(UTvdbGetGridType(*gridPtr), *gridPtr, op);
            maskGridPtr = op.outGridPtr;
        }
    }
    return maskGridPtr;
}

} // unnamed namespace


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_To_Polygons::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        gdp->clearAndDestroy();

        hvdb::Interrupter boss("Surfacing VDB primitives");

        const GU_Detail* vdbGeo = inputGeo(0);
        if (vdbGeo == nullptr) return error();

        // Get the group of grids to surface.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*vdbGeo, groupStr.toStdString());
        hvdb::VdbPrimCIterator vdbIt(vdbGeo, group);

        if (!vdbIt) {
            addWarning(SOP_MESSAGE, "No VDB primitives found.");
            return error();
        }

        // Eval attributes
#if HAVE_POLYSOUP
        const bool usePolygonSoup = evalInt("geometrytype", 0, time) == 0;
#else
        const bool usePolygonSoup = false;
#endif

        const double adaptivity = double(evalFloat("adaptivity", 0, time));
        const double iso = double(evalFloat("isovalue", 0, time));
        const bool computeNormals = !usePolygonSoup && evalInt("computenormals", 0, time);
        const bool keepVdbName = evalInt("keepvdbname", 0, time);
        const float maskoffset = static_cast<float>(evalFloat("surfacemaskoffset", 0, time));
        const bool invertmask = evalInt("invertsurfacemask", 0, time);


        // Setup level set mesher
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
                        if (auto maskGridPtr = getMaskFromGrid(maskIt->getGridPtr(), maskoffset)) {
                            mesher.setSurfaceMask(maskGridPtr, invertmask);
                        } else {
                            std::string mesg = "Surface mask "
                                + maskIt.getPrimitiveNameOrIndex().toStdString()
                                + " of type " + maskIt->getGrid().type() + " is not supported.";
                            addWarning(SOP_MESSAGE, mesg.c_str());
                        }
                    }
                }
            }

            if (evalInt("adaptivityfield", 0, time)) {
                UT_String maskStr;
                evalString(maskStr, "adaptivityfieldname", 0, time);

                const GA_PrimitiveGroup *maskGroup = matchGroup(*maskGeo, maskStr.toStdString());

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


        // Check reference input
        const GU_Detail* refGeo = inputGeo(1);
        if (refGeo) {

            // Collect all level set grids.
            std::list<openvdb::GridBase::ConstPtr> grids;
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

                // (We need a shallow copy to sync primitive & grid names).
                grids.push_back(vdbIt->getGrid().copyGrid());
                openvdb::ConstPtrCast<openvdb::GridBase>(grids.back())->setName(
                    vdbIt->getGridName());
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
                    referenceMeshing<openvdb::FloatGrid>(grids, mesher, refGeo, boss, time);
                } else if (grids.front()->isType<openvdb::DoubleGrid>()) {
                    referenceMeshing<openvdb::DoubleGrid>(grids, mesher, refGeo, boss, time);
                } else {
                    addError(SOP_MESSAGE, "Unsupported grid type.");
                }
            }

        } else {

            // Mesh each VDB primitive independently
            for (; vdbIt; ++vdbIt) {

                if (boss.wasInterrupted()) break;
                //GEOvdbProcessTypedGridScalar(*vdbIt.getPrimitive(), mesher);

                if (!GEOvdbProcessTypedGridScalar(*vdbIt.getPrimitive(), mesher)) {

                    if (vdbIt->getGrid().type() == openvdb::BoolGrid::gridType()) {

                        openvdb::BoolGrid::ConstPtr gridPtr =
                            openvdb::gridConstPtrCast<openvdb::BoolGrid>(vdbIt->getGridPtr());

                        mesher(*gridPtr);
                    }
                }

                copyMesh(*gdp, mesher, boss, usePolygonSoup,
                    keepVdbName ? vdbIt.getPrimitive()->getGridName() : nullptr);
            }

            if (!boss.wasInterrupted() && computeNormals) {
                UTparallelFor(GA_SplittableRange(gdp->getPrimitiveRange()),
                    hvdb::VertexNormalOp(*gdp));
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


template<class GridType>
void
SOP_OpenVDB_To_Polygons::referenceMeshing(
    std::list<openvdb::GridBase::ConstPtr>& grids,
    openvdb::tools::VolumeToMesh& mesher,
    const GU_Detail* refGeo,
    hvdb::Interrupter& boss,
    const fpreal time)
{
    if (refGeo == nullptr) return;
    const bool usePolygonSoup = evalInt("geometrytype", 0, time) == 0;
    const bool computeNormals = !usePolygonSoup && evalInt("computenormals", 0, time);
    const bool transferAttributes = evalInt("transferattributes", 0, time);
    const bool keepVdbName = evalInt("keepvdbname", 0, time);
    const bool sharpenFeatures = evalInt("sharpenfeatures", 0, time);
    const float edgetolerance = static_cast<float>(evalFloat("edgetolerance", 0, time));

    using TreeType = typename GridType::TreeType;
    using ValueType = typename GridType::ValueType;

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
    using IntGridT = typename GridType::template ValueConverter<openvdb::Int32>::Type;
    typename IntGridT::Ptr indexGrid; // replace

    openvdb::tools::MeshToVoxelEdgeData edgeData;

# if 0
    // Check for reference VDB
    {
        const GA_PrimitiveGroup *refGroup = matchGroup(*refGeo, "");
        hvdb::VdbPrimCIterator refIt(refGeo, refGroup);
        if (refIt) {
            const openvdb::GridClass refClass = refIt->getGrid().getGridClass();
            if (refIt && refClass == openvdb::GRID_LEVEL_SET) {
                refGrid = openvdb::gridConstPtrCast<GridType>(refIt->getGridPtr());
            }
        }
    }
#endif

    // Check for reference mesh
    std::unique_ptr<GU_Detail> geoPtr;
    if (!refGrid) {
        std::string warningStr;
        geoPtr = hvdb::convertGeometry(*refGeo, warningStr, &boss);

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

        openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec4I>
            mesh(pointList, primList);

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


    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    typename BoolTreeType::Ptr maskTree;

    if (sharpenFeatures) {
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

    GA_PrimitiveGroup *surfaceGroup = nullptr, *interiorGroup = nullptr, *seamGroup = nullptr;
    GA_PointGroup* seamPointGroup = nullptr;

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

    for (auto it = grids.begin(); it != grids.end(); ++it) {

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

        mesher(*grid);

        copyMesh(*gdp, mesher, boss, usePolygonSoup,
            keepVdbName ? grid->getName().c_str() : nullptr,
            surfaceGroup, interiorGroup, seamGroup, seamPointGroup);
    }

    grids.clear();

    // Sharpen Features
    if (!boss.wasInterrupted() && sharpenFeatures) {
        UTparallelFor(GA_SplittableRange(gdp->getPointRange()),
            hvdb::SharpenFeaturesOp(
                *gdp, *refGeo, edgeData, *transform, surfaceGroup, maskTree.get()));
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

    // Transfer primitive attributes
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

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
