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

/// @author FX R&D OpenVDB team


#include "OpenVDBData.h"
#include <openvdb/io/Stream.h>

#include <maya/MGlobal.h>
#include <maya/MIOStream.h>
#include <maya/MPoint.h>
#include <maya/MArgList.h>

#include <limits>
#include <iostream>

////////////////////////////////////////


const MTypeId OpenVDBData::id(0x00108A50);
const MString OpenVDBData::typeName("OpenVDBData");


////////////////////////////////////////


void*
OpenVDBData::creator()
{
    return new OpenVDBData();
}


OpenVDBData::OpenVDBData()
    : MPxData()
    , mGrids(new openvdb::GridPtrVec())
{
}


OpenVDBData::~OpenVDBData()
{
}


void
OpenVDBData::copy(const MPxData& other)
{
    if(other.typeId() == OpenVDBData::id) {

        const OpenVDBData& rhs = static_cast<const OpenVDBData&>(other);

        if (mGrids.get() == rhs.mGrids.get()) return;

        // shallow-copy the grids from the rhs container.
        mGrids->clear();
        insert(*rhs.mGrids);
    }
}


MTypeId
OpenVDBData::typeId() const
{
    return OpenVDBData::id;
}


MString
OpenVDBData::name() const
{
    return OpenVDBData::typeName;
}


MStatus
OpenVDBData::readASCII(const MArgList&, unsigned&)
{
    return MS::kFailure;
}


MStatus
OpenVDBData::writeASCII(ostream&)
{
    return MS::kFailure;
}


MStatus
OpenVDBData::readBinary(istream& in, unsigned)
{
    mGrids = openvdb::io::Stream(in).getGrids();
    return in.fail() ? MS::kFailure : MS::kSuccess;
}


MStatus
OpenVDBData::writeBinary(ostream& out)
{
    openvdb::io::Stream(out).write(*mGrids);
    return out.fail() ? MS::kFailure : MS::kSuccess;
}


////////////////////////////////////////


size_t
OpenVDBData::numberOfGrids() const
{
    return mGrids->size();
}


const openvdb::GridBase&
OpenVDBData::grid(size_t index) const
{
    return *((*mGrids)[index]);
}


openvdb::GridBase::ConstPtr
OpenVDBData::gridPtr(size_t index) const
{
    return (*mGrids)[index];
}


void
OpenVDBData::duplicate(const OpenVDBData& rhs)
{
    mGrids->clear();
    openvdb::GridPtrVec::const_iterator it = rhs.mGrids->begin();
    for ( ; it != rhs.mGrids->end(); ++it) {
        mGrids->push_back((*it)->copyGrid());
    }
}

void
OpenVDBData::insert(const openvdb::GridBase::Ptr& grid)
{
    mGrids->push_back(grid);
}


void
OpenVDBData::insert(const openvdb::GridBase& grid)
{
     mGrids->push_back(grid.copyGrid());
}


void
OpenVDBData::insert(const openvdb::GridPtrVec& rhs)
{
    openvdb::GridPtrVec& lhs = *mGrids;
    lhs.reserve(lhs.size() + rhs.size());
    for (openvdb::GridPtrVec::const_iterator it = rhs.begin(); it != rhs.end(); ++it) {
        lhs.push_back((*it)->copyGrid());
    }
}


void
OpenVDBData::write(const openvdb::io::File& file,
    const openvdb::MetaMap& metadata) const
{
    file.write(*mGrids, metadata);
}


////////////////////////////////////////


// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
