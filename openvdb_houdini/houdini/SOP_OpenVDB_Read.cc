///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Read.cc
///
/// @author FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/GEO_PrimVDB.h>
#include <openvdb_houdini/GU_PrimVDB.h>
#include <UT/UT_Interrupt.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Read: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Read(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Read() {};

    virtual void getDescriptiveParmName(UT_String& s) const { s = "file_name"; }

    static void registerSop(OP_OperatorTable*);
    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
};


////////////////////////////////////////


namespace {

// Populate a choice list with grid names read from a VDB file.
void
populateGridMenu(void* data, PRM_Name* choicenames, int listsize,
    const PRM_SpareData*, const PRM_Parm*)
{
    choicenames[0].setToken(0);
    choicenames[0].setLabel(0);

    hvdb::SOP_NodeVDB* sop = static_cast<hvdb::SOP_NodeVDB*>(data);
    if (sop == NULL) return;

    // Get the parameters from the GUI
    // The file name of the vdb we would like to load
    UT_String file_name;
    sop->evalString(file_name, "file_name", 0, 0);

     // Keep track of how many names we have entered
    int count = 0;

    // Add the star token to the menu
    choicenames[0].setToken("*");
    choicenames[0].setLabel("*");
    ++count;

    try {
        // Open the file and read the header, but don't read in any grids.
        // An exception is thrown if the file is not a valid VDB file.
        openvdb::io::File file(file_name.toStdString());
        file.open();

        // Loop over the names of all of the grids in the file.
        for (openvdb::io::File::NameIterator nameIter = file.beginName();
            nameIter != file.endName(); ++nameIter)
        {
            // Make sure we don't write more than the listsize,
            // and reserve a spot for the terminating 0.
            if (count > listsize - 2) break;

            // Add the grid's name to the list.
            std::string gridName = nameIter.gridName();
            choicenames[count].setToken(gridName.c_str());
            choicenames[count].setLabel(gridName.c_str());
            ++count;
        }

        file.close();
    } catch (...) {}

    // Terminate the list.
    choicenames[count].setToken(0);
    choicenames[count].setLabel(0);
}


// Callback to trigger a file reload
int
reloadCB(void* data, int /*idx*/, float /*time*/, const PRM_Template*)
{
    SOP_OpenVDB_Read* sop = static_cast<SOP_OpenVDB_Read*>(data);
    if (NULL != sop) {
        sop->forceRecook();
        return 1; // request a refresh of the parameter pane
    }
    return 0; // no refresh
}

} // unnamed namespace


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Metadata-only toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "metadata_only", "Read Metadata Only")
        .setDefault(PRMzeroDefaults)
        .setHelpText(
            "If enabled, output empty grids populated with\n"
            "their metadata and transforms only.\n"));

    // Filename
    parms.add(hutil::ParmFactory(PRM_FILE, "file_name", "File Name")
        .setDefault(0, "./filename.vdb")
        .setHelpText("Select a VDB file."));

    // Grid name mask
    parms.add(hutil::ParmFactory(PRM_STRING, "grids",  "Grid(s)")
        .setDefault(0, "*")
        .setChoiceList(new PRM_ChoiceList(PRM_CHOICELIST_TOGGLE, populateGridMenu))
        .setHelpText("Grid names separated by white space (wildcards allowed)"));

    // Toggle to enable/disable grouping
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "enable_grouping", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setDefault(PRMoneDefaults)
        .setHelpText(
            "If enabled, create a group with the given name\n"
            "that comprises the selected grids.\n"
            "If disabled, do not group the selected grids."));

    // Name for the output group
    parms.add(hutil::ParmFactory(PRM_STRING, "group",  "Group")
        .setDefault(0,
            "import os.path\n"
            "return os.path.splitext(os.path.basename(ch('file_name')))[0]",
            CH_PYTHON_EXPRESSION)
        .setHelpText("Specify a name for this group of grids."));

    // Reload button
    parms.add(hutil::ParmFactory(PRM_CALLBACK, "reload",  "Reload File")
        .setCallbackFunc(&reloadCB)
        .setHelpText("Reread the VDB file."));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Read", SOP_OpenVDB_Read::factory, parms, *table)
        .addAlias("OpenVDB Reader");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Read::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Read(net, name, op);
}


SOP_OpenVDB_Read::SOP_OpenVDB_Read(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


// Disable parms in the UI.
bool
SOP_OpenVDB_Read::updateParmsFlags()
{
    bool changed = false;
    float t = 0.0;

    changed |= enableParm("group", evalInt("enable_grouping", 0, t));

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Read::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        gdp->clearAndDestroy();

        const fpreal t = context.getTime();

        const bool readMetadataOnly = evalInt("metadata_only", 0, t);

        // Get the file name string from the UI.
        std::string filename;
        {
            UT_String s;
            evalString(s, "file_name", 0, t);
            filename = s.toStdString();
        }

        // Get the grid mask string.
        UT_String gridStr;
        evalString(gridStr, "grids", 0, t);

        // Get the group name string.
        UT_String groupStr;
        if (evalInt("enable_grouping", 0, t)) {
            evalString(groupStr, "group", 0, t);
            // If grouping is enabled but no group name was given, derive the group name
            // from the filename (e.g., "bar", given filename "/foo/bar.vdb").
            /// @internal Currently this is done with an expression on the parameter,
            /// but it could alternatively be done as follows:
            //if (!groupStr.isstring()) {
            //    groupStr = filename;
            //    groupStr = UT_String(groupStr.fileName());
            //    groupStr = groupStr.pathUpToExtension();
            //}
        }

        UT_AutoInterrupt progress(("Reading " + filename).c_str());

        // Create and open a VDB file, but don't read any grids yet.
        openvdb::io::File file(filename);
        file.open();

        // Read the file-level metadata.
        openvdb::MetaMap::Ptr fileMetadata = file.getMetadata();
        if (!fileMetadata) fileMetadata.reset(new openvdb::MetaMap);

        // Create a group for the grid primitives.
        GA_PrimitiveGroup* group = NULL;
        if (groupStr.isstring()) {
            group = gdp->newPrimitiveGroup(groupStr.buffer());
        }

        // Loop over all grids in the file.
        for (openvdb::io::File::NameIterator nameIter = file.beginName();
            nameIter != file.endName(); ++nameIter)
        {
            if (progress.wasInterrupted()) throw std::runtime_error("Was Interrupted");

            // Skip grids whose names don't match the user-supplied mask.
            const std::string& gridName = nameIter.gridName();
            if (!UT_String(gridName).multiMatch(gridStr.buffer(), 1, " ")) continue;

            hvdb::GridPtr grid;
            if (readMetadataOnly) {
                grid = file.readGridMetadata(gridName);
            } else {
                grid = file.readGrid(gridName);
            }
            if (grid) {
                // Copy file-level metadata into the grid, then create (if necessary)
                // and set a primitive attribute for each metadata item.
                for (openvdb::MetaMap::ConstMetaIterator fileMetaIt = fileMetadata->beginMeta(),
                    end = fileMetadata->endMeta(); fileMetaIt != end; ++fileMetaIt)
                {
                    // Resolve file- and grid-level metadata name conflicts
                    // in favor of the grid-level metadata.
                    if (openvdb::Metadata::Ptr meta = fileMetaIt->second) {
                        const std::string name = fileMetaIt->first;
                        if (!(*grid)[name]) {
                            grid->insertMeta(name, *meta);
                        }
                    }
                }
                // Transfer the grid name from the input grid map to the grid's metadata.
                grid->removeMeta("name");
                grid->insertMeta("name", openvdb::StringMetadata(gridName));

                // Add a new VDB primitive for this grid.
                // Note: this clears the grid's metadata.

                GEO_PrimVDB* vdb = hvdb::createVdbPrimitive(*gdp, grid);

                // Add the primitive to the group.
                if (group) group->add(vdb);
            }
        }
        file.close();

        // If a group was created but no grids were added to it, delete the group.
        if (group && group->isEmpty()) gdp->destroyPrimitiveGroup(group);

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
