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
/// @file SOP_OpenVDB_Metadata.cc
///
/// @author FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <UT/UT_Interrupt.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Metadata: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Metadata(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Metadata() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    bool updateParmsFlags() override;
    OP_ERROR cookMySop(OP_Context&) override;
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setTooltip("Specify a subset of the input VDBs to be modified.")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setDocumentation(
            "A subset of the input VDBs to be modified"
            " (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "setclass", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));
    {
        std::vector<std::string> items;
        for (int n = 0; n < openvdb::NUM_GRID_CLASSES; ++n) {
            openvdb::GridClass gridclass = static_cast<openvdb::GridClass>(n);
            items.push_back(openvdb::GridBase::gridClassToString(gridclass));
            items.push_back(openvdb::GridBase::gridClassToMenuName(gridclass));
        }
        parms.add(
            hutil::ParmFactory(PRM_STRING, "class", "Class")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("Specify how voxel values should be interpreted.")
            .setDocumentation("\
How voxel values should be interpreted\n\
\n\
Fog Volume:\n\
    The volume represents a density field.  Values should be positive,\n\
    with zero representing empty regions.\n\
Level Set:\n\
    The volume is treated as a narrow-band signed distance field level set.\n\
    The voxels within a certain distance&mdash;the \"narrow band width\"&mdash;of\n\
    an isosurface are expected to define positive (exterior) and negative (interior)\n\
    distances to the surface.  Outside the narrow band, the distance value\n\
    is constant and equal to the band width.\n\
Staggered Vector Field:\n\
    If the volume is vector-valued, the _x_, _y_ and _z_ vector components\n\
    are to be treated as lying on the respective faces of voxels,\n\
    not at their centers.\n\
Other:\n\
    No special meaning is assigned to the volume's data.\n"));
    }

    /// @todo Do we really need to expose this?
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "setcreator", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));
    parms.add(hutil::ParmFactory(PRM_STRING, "creator", "Creator")
        .setTooltip("Who (or what node) created the VDB"));

    /// @todo Currently, no SOP pays attention to this setting.
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "setworld", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "world", "Transform Values")
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "For vector-valued VDBs, specify whether voxel values\n"
            "are in world space and should be affected by transforms\n"
            "or in local space and should not be transformed."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "setvectype", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));
    {
        std::string help =
            "For vector-valued VDBs, specify an interpretation of the vectors"
            " that determines how they are affected by transforms.\n";
        std::vector<std::string> items;
        for (int n = 0; n < openvdb::NUM_VEC_TYPES; ++n) {
            const auto vectype = static_cast<openvdb::VecType>(n);
            items.push_back(openvdb::GridBase::vecTypeToString(vectype));
            items.push_back(openvdb::GridBase::vecTypeExamples(vectype));
            help += "\n" + openvdb::GridBase::vecTypeExamples(vectype) + "\n    "
                + openvdb::GridBase::vecTypeDescription(vectype) + ".";
        }
        parms.add(hutil::ParmFactory(PRM_STRING, "vectype", "Vector Type")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip(::strdup(help.c_str())));
    }

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "setfloat16", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "float16", "Write 16-Bit Floats")
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "When saving the VDB to a file, write floating-point\n"
            "scalar or vector voxel values as 16-bit half floats."));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Metadata", SOP_OpenVDB_Metadata::factory, parms, *table)
        .addInput("Input with VDBs")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Modify the metadata associated with a VDB volume.\"\"\"\n\
\n\
@overview\n\
\n\
This node allows one to create and edit metadata attached to a VDB volume.\n\
Some standard VDB metadata, such as the\n\
[grid class|http://www.openvdb.org/documentation/doxygen/overview.html#secGrid],\n\
is exposed via intrinsic attributes on the primitive and can be viewed\n\
and in some cases edited either from the [geometry spreadsheet|/ref/panes/geosheet]\n\
or with the [Node:sop/attribcreate] node.\n\
This node provides more direct access to the standard VDB metadata.\n\
\n\
@related\n\
- [OpenVDB Create|Node:sop/DW_OpenVDBCreate]\n\
- [Node:sop/attribcreate]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


bool
SOP_OpenVDB_Metadata::updateParmsFlags()
{
    bool changed = false;
    const fpreal time = 0; // No point using CHgetTime as that is unstable.

    changed |= enableParm("class",   bool(evalInt("setclass", 0, time)));
    changed |= enableParm("creator", bool(evalInt("setcreator", 0, time)));
    changed |= enableParm("float16", bool(evalInt("setfloat16", 0, time)));
    changed |= enableParm("world",   bool(evalInt("setworld", 0, time)));
    changed |= enableParm("vectype", bool(evalInt("setvectype", 0, time)));

    return changed;
}


OP_Node*
SOP_OpenVDB_Metadata::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Metadata(net, name, op);
}


SOP_OpenVDB_Metadata::SOP_OpenVDB_Metadata(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


OP_ERROR
SOP_OpenVDB_Metadata::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal time = context.getTime();

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSource(0, context);

        // Get UI parameter values.
        const bool
            setclass = evalInt("setclass", 0, time),
            setcreator = evalInt("setcreator", 0, time),
            setfloat16 = evalInt("setfloat16", 0, time),
            float16 = evalInt("float16", 0, time),
            setvectype = evalInt("setvectype", 0, time),
            setworld = evalInt("setworld", 0, time),
            world = evalInt("world", 0, time);

        UT_String s;
        evalString(s, "creator", 0, time);
        const std::string creator = s.toStdString();

        evalString(s, "class", 0, time);
        const openvdb::GridClass gridclass =
            openvdb::GridBase::stringToGridClass(s.toStdString());

        evalString(s, "vectype", 0, time);
        const openvdb::VecType vectype =
            openvdb::GridBase::stringToVecType(s.toStdString());

        // Get the group of grids to be modified.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        UT_AutoInterrupt progress("Set VDB grid metadata");

        // For each VDB primitive in the given group...
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (progress.wasInterrupted()) throw std::runtime_error("was interrupted");

            GU_PrimVDB* vdb = *it;

            // No need to make the grid unique, since we're not modifying its voxel data.
            hvdb::Grid& grid = vdb->getGrid();

            // Set various grid metadata items.
            if (setclass) {
                grid.setGridClass(gridclass);

                // Update view port visualization options
                if (gridclass == openvdb::GRID_LEVEL_SET) {
                    const GEO_VolumeOptions& visOps = vdb->getVisOptions();
                    vdb->setVisualization(GEO_VOLUMEVIS_ISO, visOps.myIso, visOps.myDensity);
                } else if (gridclass == openvdb::GRID_FOG_VOLUME) {
                    const GEO_VolumeOptions& visOps = vdb->getVisOptions();
                    vdb->setVisualization(GEO_VOLUMEVIS_SMOKE, visOps.myIso, visOps.myDensity);
                }
            }


            if (setcreator) grid.setCreator(creator);
            if (setfloat16) grid.setSaveFloatAsHalf(float16);
            if (setvectype) grid.setVectorType(vectype);
            if (setworld)   grid.setIsInWorldSpace(world);
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
