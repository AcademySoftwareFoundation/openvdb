///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
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

/*
 * Copyright (c) 2012
 *	Side Effects Software Inc.  All rights reserved.
 *
 * Redistribution and use of Houdini Development Kit samples in source and
 * binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. The name of Side Effects Software may not be used to endorse or
 *    promote products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY SIDE EFFECTS SOFTWARE `AS IS' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN
 * NO EVENT SHALL SIDE EFFECTS SOFTWARE BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *----------------------------------------------------------------------------
 */

#include "GU_PrimVDB.h"
#include "Utils.h"

#ifndef SESI_OPENVDB
#include <UT/UT_DSOVersion.h>
#endif

#if (UT_VERSION_INT >= 0x0d00023d) // 13.0.573 or later
#include <UT/UT_EnvControl.h>
#endif

#include <UT/UT_IOTable.h>
#include <UT/UT_IStream.h>
#include <UT/UT_Version.h>

#include <GU/GU_Detail.h>
#include <SOP/SOP_Node.h>
#include <GEO/GEO_IOTranslator.h>

#include <openvdb/io/Stream.h>
#include <openvdb/io/File.h>

#include <stdio.h>
#include <iostream>

using namespace openvdb_houdini;
using std::cerr;

namespace {

class GEO_VDBTranslator : public GEO_IOTranslator
{
public:
	     GEO_VDBTranslator() {}
    virtual ~GEO_VDBTranslator() {}

    virtual GEO_IOTranslator *duplicate() const;

    virtual const char *formatName() const;

    virtual int		checkExtension(const char *name);
    virtual void    getFileExtensions(UT_StringArray &extensions) const;

    virtual int		checkMagicNumber(unsigned magic);

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    virtual GA_Detail::IOStatus fileLoad(GEO_Detail *gdp, UT_IStream &is, bool ate_magic);
#else
    virtual GA_Detail::IOStatus fileLoad(GEO_Detail *gdp, UT_IStream &is, int ate_magic);
#endif
    virtual GA_Detail::IOStatus fileSave(const GEO_Detail *gdp, std::ostream &os);
#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
    virtual GA_Detail::IOStatus fileSaveToFile(const GEO_Detail *gdp, const char *fname);
#else
    virtual GA_Detail::IOStatus fileSaveToFile(const GEO_Detail *gdp, std::ostream &os,
					       const char *fname);
#endif
};

GEO_IOTranslator *
GEO_VDBTranslator::duplicate() const
{
    return new GEO_VDBTranslator();
}

const char *
GEO_VDBTranslator::formatName() const
{
    return "VDB Format";
}

int
GEO_VDBTranslator::checkExtension(const char *name)
{
    return UT_String(name).matchFileExtension(".vdb");
}

void
GEO_VDBTranslator::getFileExtensions(UT_StringArray &extensions) const
{
    extensions.clear();
    extensions.append(".vdb");
}

int
GEO_VDBTranslator::checkMagicNumber(unsigned /*magic*/)
{
    return 0;
}

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
GA_Detail::IOStatus
GEO_VDBTranslator::fileLoad(GEO_Detail *geogdp, UT_IStream &is, bool /*ate_magic*/)
#else
GA_Detail::IOStatus
GEO_VDBTranslator::fileLoad(GEO_Detail *geogdp, UT_IStream &is, int /*ate_magic*/)
#endif
{
    UT_WorkBuffer   buf;
    GU_Detail       *gdp = static_cast<GU_Detail*>(geogdp);

    if (!is.isRandomAccessFile(buf)) {
        cerr << "Error: Attempt to load VDB from non-file source.\n";
        return false;
    }

    try {
        // Create and open a VDB file, but don't read any grids yet.
        openvdb::io::File file(buf.buffer());

        file.open(/*delayLoad*/false);

        // Read the file-level metadata into global attributes.
        openvdb::MetaMap::Ptr fileMetadata = file.getMetadata();
#if !defined(SESI_OPENVDB) && (UT_VERSION_INT >= 0x0c050157)
        if (!fileMetadata) fileMetadata.reset(new openvdb::MetaMap);
#else
        if (fileMetadata) {
            GU_PrimVDB::createAttrsFromMetadata(
                GA_ATTRIB_GLOBAL, GA_Offset(0), *fileMetadata, *geogdp);
        }
#endif

        // Loop over all grids in the file.
        for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
            const std::string& gridName = nameIter.gridName();

            if (GridPtr grid = file.readGrid(gridName)) {

#if !defined(SESI_OPENVDB) && (UT_VERSION_INT >= 0x0c050157)
                // Copy file-level metadata into the grid, then create (if
                // necessary)
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
#endif
                // Add a new VDB primitive for this grid.
                // Note: this clears the grid's metadata.
                createVdbPrimitive(*gdp, grid);
            }
        }
        file.close();

    } catch (std::exception &e) {
        cerr << "Load failure: " << e.what() << "\n";
        return false;
    }

    return true;
}

template <typename FileT, typename OutputT>
bool
fileSaveVDB(const GEO_Detail *geogdp, OutputT os)
{
    GU_Detail *gdp = static_cast<GU_Detail*>(const_cast<GEO_Detail*>(geogdp));
    if (!gdp) return false;

    try {
        // Populate an output GridMap with VDB grid primitives found in the
        // geometry.
        openvdb::GridPtrVec outGrids;
        for (VdbPrimIterator it(gdp); it; ++it) {
            const GU_PrimVDB* vdb = *it;

            // Create a new grid that shares the primitive's tree and transform
            // and then transfer primitive attributes to the new grid as metadata.
            GridPtr grid = vdb->getGrid().copyGrid();
            GU_PrimVDB::createMetadataFromGridAttrs(*grid, *vdb, *gdp);
            grid->removeMeta("is_vdb");

            // Retrieve the grid's name from the primitive attribute.
            grid->setName(it.getPrimitiveName().toStdString());

            outGrids.push_back(grid);
        }

        // Add file-level metadata.
        openvdb::MetaMap fileMetadata;

        std::string versionStr = "Houdini ";
        versionStr += UTgetFullVersion();
        versionStr += "/GEO_VDBTranslator";

        fileMetadata.insertMeta("creator", openvdb::StringMetadata(versionStr));

#if defined(SESI_OPENVDB) || (UT_VERSION_INT < 0x0c050157)
        GU_PrimVDB::createMetadataFromAttrs(
            fileMetadata, GA_ATTRIB_GLOBAL, GA_Offset(0), *gdp);
#endif
        // Create a VDB file object.
        FileT file(os);

        // Always enable active mask compression, since it is fast
        // and compresses level sets and fog volumes well.
        uint32_t compression = openvdb::io::COMPRESS_ACTIVE_MASK;

#if (UT_VERSION_INT >= 0x0d00023d) // 13.0.573 or later
        // Enable Blosc unless backwards compatibility is requested.
        if (openvdb::io::Archive::hasBloscCompression()
            && !UT_EnvControl::getInt(ENV_HOUDINI13_VOLUME_COMPATIBILITY)) {
            compression |= openvdb::io::COMPRESS_BLOSC;
        }
#endif
        file.setCompression(compression);

        file.write(outGrids, fileMetadata);

    } catch (std::exception &e) {
        cerr << "Save failure: " << e.what() << "\n";
        return false;
    }

    return true;
}

GA_Detail::IOStatus
GEO_VDBTranslator::fileSave(const GEO_Detail *geogdp, std::ostream &os)
{
    // Saving via io::Stream will NOT save grid offsets, disabling partial
    // reading.
    return fileSaveVDB<openvdb::io::Stream, std::ostream &>(geogdp, os);
}

#if (UT_VERSION_INT >= 0x0e000000) // 14.0.0 or later
GA_Detail::IOStatus
GEO_VDBTranslator::fileSaveToFile(const GEO_Detail *geogdp, const char *fname)
#else
GA_Detail::IOStatus
GEO_VDBTranslator::fileSaveToFile(const GEO_Detail *geogdp, std::ostream &,
    const char *fname)
#endif
{
    // Saving via io::File will save grid offsets that allow for partial
    // reading.
    return fileSaveVDB<openvdb::io::File, const char *>(geogdp, fname);
}

} // unnamed namespace

void
new_VDBGeometryIO(void *)
{
    GU_Detail::registerIOTranslator(new GEO_VDBTranslator());

    UT_ExtensionList		*geoextension;
    geoextension = UTgetGeoExtensions();
    if (!geoextension->findExtension("vdb"))
	geoextension->addExtension("vdb");
}

#ifndef SESI_OPENVDB
void
newGeometryIO(void *data)
{
    // Initialize the version of the OpenVDB library that this library is built against
    // (i.e., not the HDK native OpenVDB library).
    openvdb::initialize();
    // Register a .vdb file translator.
    new_VDBGeometryIO(data);
}
#endif

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
