///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Platonic.cc
/// @author FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <UT/UT_Interrupt.h>
#include <boost/math/constants/constants.hpp>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/LevelSetUtil.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Platonic: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Platonic(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Platonic() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    { // Shapes
        const char* items[] = {
            "sphere", "Sphere",
            "tetrahedron", "Tetrahedron",
            "cube", "Cube",
            "octahedron", "Octahedron",
            "dodecahedron", "Dodecahedron",
            "icosahedron", "Icosahedron",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "solidType", "Solid Type")
                  .setHelpText("Select a sphere or one of the five platonic solids)")
                  .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    { // Grid Class
        std::vector<std::string> items;
        items.push_back(openvdb::GridBase::gridClassToString(openvdb::GRID_LEVEL_SET)); // token
        items.push_back(openvdb::GridBase::gridClassToMenuName(openvdb::GRID_LEVEL_SET)); // label
        items.push_back("sdf");
        items.push_back("Signed Distance Field");

        parms.add(hutil::ParmFactory(PRM_STRING, "gridclass", "Grid Class")
            .setDefault(openvdb::GridBase::gridClassToString(openvdb::GRID_LEVEL_SET))
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    // Radius
    parms.add(hutil::ParmFactory(PRM_FLT_J, "scalarRadius", "Radius/Size")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_FREE, 10));

    // Center
    parms.add(hutil::ParmFactory(PRM_XYZ_J, "center", "Center")
        .setVectorSize(3)
        .setDefault(PRMzeroDefaults));

    // Voxel size
    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelSize", "Voxel size")
        .setDefault(0.1f)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_FREE, 10));

    // Narrow-band half-width
    parms.add(hutil::ParmFactory(PRM_FLT_J, "halfWidth", "Half-band width")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1.0, PRM_RANGE_UI, 10)
        .setHelpText("Half the width of the narrow band in voxel units. "
            "(3 is optimal for level set operations.)"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "fogVolume", "Convert to fog volume"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Platonic",
        SOP_OpenVDB_Platonic::factory, parms, *table);
}


OP_Node*
SOP_OpenVDB_Platonic::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Platonic(net, name, op);
}


SOP_OpenVDB_Platonic::SOP_OpenVDB_Platonic(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


bool
SOP_OpenVDB_Platonic::updateParmsFlags()
{
    bool changed = false;

    bool sdfGrid = false;
    {
        UT_String gridClassStr;
        evalString(gridClassStr, "gridclass", 0, 0);
        sdfGrid = (gridClassStr.toStdString() == "sdf");
    }

    changed |= enableParm("halfWidth", sdfGrid);

    return changed;
}


OP_ERROR
SOP_OpenVDB_Platonic::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        gdp->clearAndDestroy();

        const fpreal time = context.getTime();

        hvdb::Interrupter boss("OpenVDB Platonic");

        // Read GUI parameters and generate narrow-band level set of sphere
        const float radius = static_cast<float>(evalFloat("scalarRadius", 0, time));
        const openvdb::Vec3f center = evalVec3f("center", time);
        const float voxelSize = static_cast<float>(evalFloat("voxelSize", 0, time));
        float halfWidth = static_cast<float>(evalFloat("halfWidth", 0, time));

        {
            UT_String gridClassStr;
            evalString(gridClassStr, "gridclass", 0, 0);
            if (gridClassStr.toStdString() != "sdf") {
                halfWidth = 3.0;
            }
        }

        openvdb::FloatGrid::Ptr grid;
        switch (evalInt("solidType", 0, time)) {
        case 0://Sphere
            grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid, hvdb::Interrupter>
                (radius, center, voxelSize, halfWidth, &boss);
            break;
        case 1:// Tetrahedraon
            grid = openvdb::tools::createLevelSetTetrahedron<openvdb::FloatGrid, hvdb::Interrupter>
                (radius, center, voxelSize, halfWidth, &boss);
            break;
        case 2:// Cube
            grid = openvdb::tools::createLevelSetCube<openvdb::FloatGrid, hvdb::Interrupter>
                (radius, center, voxelSize, halfWidth, &boss);
            break;
        case 3:// Octahedron
            grid = openvdb::tools::createLevelSetOctahedron<openvdb::FloatGrid, hvdb::Interrupter>
                (radius, center, voxelSize, halfWidth, &boss);
            break;
        case 4:// Dodecahedron
            grid = openvdb::tools::createLevelSetDodecahedron<openvdb::FloatGrid, hvdb::Interrupter>
                (radius, center, voxelSize, halfWidth, &boss);
            break;
        case 5:// Icosahedron
            grid = openvdb::tools::createLevelSetIcosahedron<openvdb::FloatGrid, hvdb::Interrupter>
                (radius, center, voxelSize, halfWidth, &boss);
            break;
        default:
            addError(SOP_MESSAGE, "Illegal shape.");
            return error();
        }

        // Fog volume conversion
        if (bool(evalInt("fogVolume", 0, time))) {
            openvdb::tools::sdfToFogVolume(*grid);
        }

        // Store the grid in a new VDB primitive and add the primitive to the output gdp
        hvdb::createVdbPrimitive(*gdp, grid, "PlatonicSolid");

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
