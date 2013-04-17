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

#include "Stream.h"

#include <iostream>
#include <vector>
#include <boost/cstdint.hpp>
#include <openvdb/Exceptions.h>
#include "GridDescriptor.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

Stream::Stream(std::istream& is)
{
    if (!is) return;

    readHeader(is);

    // Tag the input stream with the file format and library version numbers.
    is.iword(Archive::sFormatVersionIndex) = fileVersion();
    is.iword(Archive::sLibraryMajorVersionIndex) = libraryVersion().first;
    is.iword(Archive::sLibraryMinorVersionIndex) = libraryVersion().second;
    is.iword(Archive::sDataCompressionIndex) = compressionFlags();

    // Read in the VDB metadata.
    mMeta.reset(new MetaMap);
    mMeta->readMeta(is);

    // Read in the number of grids.
    const boost::int32_t gridCount = readGridCount(is);

    // Read in all grids and insert them into mGrids.
    mGrids.reset(new GridPtrVec);
    std::vector<GridDescriptor> descriptors;
    descriptors.reserve(gridCount);
    Archive::NamedGridMap namedGrids;
    for (boost::int32_t i = 0; i < gridCount; ++i) {
        GridDescriptor gd;
        gd.read(is);
        descriptors.push_back(gd);
        GridBase::Ptr grid = readGrid(gd, is);
        mGrids->push_back(grid);
        namedGrids[gd.uniqueName()] = grid;
    }

    // Connect instances (grids that share trees with other grids).
    for (size_t i = 0, N = descriptors.size(); i < N; ++i) {
        Archive::connectInstance(descriptors[i], namedGrids);
    }
}


Stream::Stream()
{
}


Stream::~Stream()
{
}


////////////////////////////////////////


GridBase::Ptr
Stream::readGrid(const GridDescriptor& gd, std::istream& is) const
{
    GridBase::Ptr grid;

    if (!GridBase::isRegistered(gd.gridType())) {
        OPENVDB_THROW(TypeError, "can't read grid \""
            << GridDescriptor::nameAsString(gd.uniqueName()) <<
            "\" from input stream because grid type " << gd.gridType() << " is unknown");
    } else {
        grid = GridBase::createGrid(gd.gridType());
        if (grid) grid->setSaveFloatAsHalf(gd.saveFloatAsHalf());

        Archive::readGrid(grid, gd, is);
    }
    return grid;
}


void
Stream::writeGrids(std::ostream& os, const GridCPtrVec& grids, const MetaMap& metadata) const
{
    Archive::write(os, grids, /*seekable=*/false, metadata);
}


////////////////////////////////////////


MetaMap::Ptr
Stream::getMetadata() const
{
    // Return a deep copy of the file-level metadata, which was read
    // when this object was constructed.
    return MetaMap::Ptr(new MetaMap(*mMeta));
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
