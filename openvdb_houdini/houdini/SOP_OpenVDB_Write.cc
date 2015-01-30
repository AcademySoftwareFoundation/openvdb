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
/// @file SOP_OpenVDB_Write.cc
///
/// @author FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/GEO_PrimVDB.h>
#include <openvdb_houdini/GU_PrimVDB.h>
#include <PRM/PRM_Parm.h>
#include <UT/UT_Interrupt.h>
#include <set>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Write: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Write(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Write() {}

    virtual void getDescriptiveParmName(UT_String& s) const { s = "file_name"; }

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);
    static int writeNowCallback(void* data, int index, float now, const PRM_Template*);

protected:
    virtual void resolveObsoleteParms(PRM_ParmList*);

    virtual OP_ERROR cookMySop(OP_Context&);

    void writeOnNextCook(bool write = true) { mWriteOnNextCook = write; }

    typedef std::set<std::string> StringSet;
    void reportFloatPrecisionConflicts(const StringSet& conflicts);

private:
    void doCook(const fpreal time = 0);
    bool mWriteOnNextCook;
};


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms, obsoleteParms;

    // File name
    parms.add(hutil::ParmFactory(PRM_FILE, "file_name", "File Name")
        .setDefault(0, "./filename.vdb")
        .setHelpText("Path name for the output VDB file"));

    // Group
    parms.add(hutil::ParmFactory(PRM_STRING, "group",  "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setHelpText("Write only a subset of the input grids."));

    // Compression
    {
        const char* items[] = {
            "none",     "None",
            "zip",      "Zip",
            "blosc",    "Blosc",
            NULL
        };

#ifdef OPENVDB_USE_BLOSC
        parms.add(hutil::ParmFactory(PRM_ORD, "compression", "Compression")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault("blosc")
            .setHelpText(
                "Zip is slow but compresses very well.  Blosc is fast and compresses well,\n"
                "but files written with Blosc cannot be read by older versions of Houdini.\n"));

        obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "compress_zip", "Zip Compression"));
#else
        parms.add(hutil::ParmFactory(PRM_TOGGLE, "compress_zip", "Zip Compression")
            .setDefault(true)
            .setHelpText(
            "Apply Zip \"deflate\" compression to non-SDF and non-fog grids.\n"
            "(Zip compression can be slow for large volumes.)"));

        obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "compression", "Compression")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
#endif
    }

    {   // Write mode (manual/auto)

        const char* items[] = {
            "manual",   "Manual",
            "auto",     "Automatic",
            NULL
        };

        parms.add(
            hutil::ParmFactory(PRM_ORD, "writeMode", "Write Mode")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setHelpText(
                "In Manual mode, click the Write Now button\n"
                "to write the output file.\n"
                "In Automatic mode, the file is written\n"
                " each time this node cooks."));
    }

    // "Write Now" button
    parms.add(hutil::ParmFactory(PRM_CALLBACK, "write", "Write Now")
        .setCallbackFunc(&SOP_OpenVDB_Write::writeNowCallback)
        .setHelpText("Click to write the output file."));

    {   // Float precision
        parms.add(hutil::ParmFactory(PRM_HEADING, "float_header", "Float Precision"));

        parms.add(hutil::ParmFactory(PRM_STRING, "float_16_group", "Write 16-Bit")
            .setChoiceList(&hutil::PrimGroupMenuInput1)
            .setHelpText(
                "For grids that belong to the group(s) listed here,\n"
                "write floating-point scalar or vector voxel values\n"
                "using 16-bit half floats.\n"
                "If no groups are listed, all grids will be written\n"
                "using their existing precision settings."));

        parms.add(hutil::ParmFactory(PRM_STRING, "float_full_group", "Write Full-Precision")
            .setChoiceList(&hutil::PrimGroupMenuInput1)
            .setHelpText(
                "For grids that belong to the group(s) listed here,\n"
                "write floating-point scalar or vector voxel values\n"
                "using full-precision floats or doubles.\n"
                "If no groups are listed, all grids will be written\n"
                "using their existing precision settings."));
    }

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Write", SOP_OpenVDB_Write::factory, parms, *table)
        .addAlias("OpenVDB Writer")
        .setObsoleteParms(obsoleteParms)
        .addInput("Input with VDB grids write out");
}


void
SOP_OpenVDB_Write::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;

#ifdef OPENVDB_USE_BLOSC
    PRM_Parm* parm = obsoleteParms->getParmPtr("compress_zip");
    if (parm && !parm->isFactoryDefault()) {
        const bool zip = obsoleteParms->evalInt("compress_zip", 0, /*time=*/0.0);
        const UT_String compression(zip ? "zip" : "none");
        setString(compression, CH_STRING_LITERAL, "compression", 0, 0.0);
    }
#else
    if (NULL != obsoleteParms->getParmPtr("compression")) {
        UT_String compression;
        obsoleteParms->evalString(compression, "compression", 0, /*time=*/0.0);
        setInt("compress_zip", 0, 0.0, (compression == "zip" ? 1 : 0));
    }
#endif

    // Delegate to the base class.
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}


OP_Node*
SOP_OpenVDB_Write::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Write(net, name, op);
}


SOP_OpenVDB_Write::SOP_OpenVDB_Write(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op),
    mWriteOnNextCook(false)
{
}


////////////////////////////////////////


int
SOP_OpenVDB_Write::writeNowCallback(
    void *data, int /*index*/, float /*now*/,
    const PRM_Template*)
{
    if (SOP_OpenVDB_Write* self = static_cast<SOP_OpenVDB_Write*>(data)) {
        self->writeOnNextCook();
        self->forceRecook();
        return 1;
    }
    return 0;
}


////////////////////////////////////////


/// If any grids belong to both the "Write 16-Bit" and "Write Full-Precision"
/// groups, display a warning.
void
SOP_OpenVDB_Write::reportFloatPrecisionConflicts(const StringSet& conflicts)
{
    if (conflicts.empty()) return;

    std::ostringstream ostr;
    if (conflicts.size() == 1) {
        ostr << "For grid \"" << *conflicts.begin() << "\"";
    } else {
        StringSet::const_iterator i = conflicts.begin(), e = conflicts.end();
        ostr << "For grids \"" << *i << "\"";
        // Join grid names into a string of the form "grid1, grid2 and grid3".
        size_t count = conflicts.size(), n = 1;
        for (++i; i != e; ++i, ++n) {
            if (n + 1 < count) ostr << ", "; else ostr << " and ";
            ostr << "\"" << *i << "\"";
        }
    }
    ostr << ", specify either 16-bit output or full-precision output"
        << " or neither, but not both.";

    // Word wrap the message at 60 columns and indent the first line.
    const std::string prefix(20, '#');
    UT_String word_wrapped(prefix + ostr.str());
    word_wrapped.format(60/*cols*/);
    word_wrapped.replacePrefix(prefix.c_str(), "");

    addWarning(SOP_MESSAGE, word_wrapped);
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Write::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal t = context.getTime();

        if (mWriteOnNextCook || 1 == evalInt("writeMode", 0, t)) {
            duplicateSource(0, context);
            doCook(t);
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}


void
SOP_OpenVDB_Write::doCook(const fpreal time)
{
    // Get the filename of the output file.
    UT_String fileNameStr;
    evalString(fileNameStr, "file_name", 0, time);
    const std::string filename = fileNameStr.toStdString();
    if (filename.empty()) {
        addWarning(SOP_MESSAGE, "no name given for the output file");
        return;
    }

    // Get grid groups.
    UT_String groupStr, halfGroupStr, fullGroupStr;
    evalString(groupStr, "group", 0, time);
    evalString(halfGroupStr, "float_16_group", 0, time);
    evalString(fullGroupStr, "float_full_group", 0, time);

    const GA_PrimitiveGroup
        *group = matchGroup(*gdp, groupStr.toStdString()),
        *halfGroup = NULL,
        *fullGroup = NULL;
    if (halfGroupStr.isstring()) {
        // Normally, an empty group pattern matches all primitives, but
        // for the float precision filters, we want it to match nothing.
        halfGroup = matchGroup(*gdp, halfGroupStr.toStdString());
    }
    if (fullGroupStr.isstring()) {
        fullGroup = matchGroup(*gdp, fullGroupStr.toStdString());
    }

    // Get compression options.
#ifdef OPENVDB_USE_BLOSC
    UT_String compression;
    evalString(compression, "compression", 0, time);
#else
    const bool zip = evalInt("compress_zip", 0, time);
#endif

    UT_AutoInterrupt progress(("Writing " + filename).c_str());

    // Set of names of grids that the user selected for both 16-bit
    // and full-precision conversion
    StringSet conflicts;

    // Collect pointers to grids from VDB primitives found in the geometry.
    openvdb::GridPtrSet outGrids;
    for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
        if (progress.wasInterrupted()) {
            throw std::runtime_error("Interrupted");
        }

        const GU_PrimVDB* vdb = *it;

        // Create a new grid that shares the primitive's tree and transform
        // and then transfer primitive attributes to the new grid as metadata.
        hvdb::GridPtr grid = vdb->getGrid().copyGrid();
        GU_PrimVDB::createMetadataFromGridAttrs(*grid, *vdb, *gdp);
        grid->removeMeta("is_vdb");

        // Retrieve the grid's name from the primitive attribute.
        const std::string gridName = it.getPrimitiveName().toStdString();

        // Check if the user has overridden this grid's saveFloatAsHalf setting.
        if (halfGroup && halfGroup->contains(vdb)) {
            if (fullGroup && fullGroup->contains(vdb)) {
                // This grid belongs to both the 16-bit and full-precision groups.
                conflicts.insert(gridName);
            } else {
                grid->setSaveFloatAsHalf(true);
            }
        } else if (fullGroup && fullGroup->contains(vdb)) {
            if (halfGroup && halfGroup->contains(vdb)) {
                // This grid belongs to both the 16-bit and full-precision groups.
                conflicts.insert(gridName);
            } else {
                grid->setSaveFloatAsHalf(false);
            }
        } else {
            // Preserve this grid's existing saveFloatAsHalf setting.
        }

        outGrids.insert(grid);
    }

    if (outGrids.empty()) {
        addWarning(SOP_MESSAGE, ("No grids were written to " + filename).c_str());
    }

    reportFloatPrecisionConflicts(conflicts);

    // Add file-level metadata.
    openvdb::MetaMap outMeta;
    outMeta.insertMeta("creator",
        openvdb::StringMetadata("Houdini/SOP_OpenVDB_Write"));

    // Create a VDB file object.
    openvdb::io::File file(filename);

#ifdef OPENVDB_USE_BLOSC
    uint32_t compressionFlags = file.compression();
    if (compression == "none") {
        compressionFlags &= ~(openvdb::io::COMPRESS_ZIP | openvdb::io::COMPRESS_BLOSC);
    } else if (compression == "blosc") {
        compressionFlags &= ~openvdb::io::COMPRESS_ZIP;
        compressionFlags |= openvdb::io::COMPRESS_BLOSC;
    } else if (compression == "zip") {
        compressionFlags |= openvdb::io::COMPRESS_ZIP;
        compressionFlags &= ~openvdb::io::COMPRESS_BLOSC;
    }
#else
    uint32_t compressionFlags = openvdb::io::COMPRESS_ACTIVE_MASK;
    if (zip) compressionFlags |= openvdb::io::COMPRESS_ZIP;
#endif
    file.setCompression(compressionFlags);

    file.write(outGrids, outMeta);
    file.close();

    mWriteOnNextCook = false;
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
