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

#ifndef OPENVDB_IO_STREAM_HAS_BEEN_INCLUDED
#define OPENVDB_IO_STREAM_HAS_BEEN_INCLUDED

#include <iosfwd>
#include "Archive.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

class GridDescriptor;


/// Grid archive associated with arbitrary input and output streams (not necessarily files)
class OPENVDB_API Stream: public Archive
{
public:
    /// Read grids from an input stream.
    explicit Stream(std::istream&);
    Stream();
    ~Stream();

    /// Return the file-level metadata in a newly created MetaMap.
    MetaMap::Ptr getMetadata() const;

    /// Return pointers to the grids that were read from the input stream.
    GridPtrVecPtr getGrids() { return mGrids; }

    /// Write the grids in the given container to an output stream.
    template<typename GridPtrContainerT>
    void write(std::ostream&, const GridPtrContainerT&, const MetaMap& = MetaMap()) const;

private:
    /// Create a new grid of the type specified by the given descriptor,
    /// then populate the grid from the given input stream.
    /// @return the newly created grid.
    GridBase::Ptr readGrid(const GridDescriptor&, std::istream&) const;

    void writeGrids(std::ostream&, const GridCPtrVec&, const MetaMap&) const;

    // Disallow copying of instances of this class.
    Stream(const Stream&);
    Stream& operator=(const Stream&);


    MetaMap::Ptr mMeta;
    GridPtrVecPtr mGrids;
};


////////////////////////////////////////


template<typename GridPtrContainerT>
inline void
Stream::write(std::ostream& os, const GridPtrContainerT& container,
    const MetaMap& metadata) const
{
    GridCPtrVec grids;
    std::copy(container.begin(), container.end(), std::back_inserter(grids));
    this->writeGrids(os, grids, metadata);
}

template<>
inline void
Stream::write<GridCPtrVec>(std::ostream& os, const GridCPtrVec& grids,
    const MetaMap& metadata) const
{
    this->writeGrids(os, grids, metadata);
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_STREAM_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
