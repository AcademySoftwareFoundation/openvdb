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

/// @file OpenVDBData.h
/// @author FX R&D OpenVDB team


#ifndef OPENVDB_MAYA_DATA_HAS_BEEN_INCLUDED
#define OPENVDB_MAYA_DATA_HAS_BEEN_INCLUDED


#include <openvdb/Platform.h>
#include <openvdb/openvdb.h>

#include <maya/MPxData.h>
#include <maya/MTypeId.h>
#include <maya/MString.h>
#include <maya/MObjectArray.h>

#include <iosfwd>

////////////////////////////////////////


class OpenVDBData : public MPxData
{
public:
    OpenVDBData();
    virtual ~OpenVDBData();

    size_t numberOfGrids() const;

    /// @brief  return a constant reference to the specified grid.
    const openvdb::GridBase& grid(size_t index) const;

    /// @brief  return a constant pointer to the specified grid.
    openvdb::GridBase::ConstPtr gridPtr(size_t index) const;

    /// @brief clears this container and duplicates the @c rhs grid container.
    void duplicate(const OpenVDBData& rhs);

    /// @brief Append the given grid to this container.
    void insert(const openvdb::GridBase::ConstPtr&);
    /// @brief Append a shallow copy of the given grid to this container.
    void insert(const openvdb::GridBase&);
    /// @brief Append shallow copies of the given grids to this container.
    void insert(const openvdb::GridPtrVec&);
    /// @brief Append shallow copies of the given grids to this container.
    void insert(const openvdb::GridCPtrVec&);


    void write(const openvdb::io::File& file,
        const openvdb::MetaMap& = openvdb::MetaMap()) const;


    /// @{
    // required maya interface methods
    static void* creator();

    virtual MStatus readASCII(const MArgList&, unsigned&);
    virtual MStatus writeASCII(ostream&);

    virtual MStatus readBinary(istream&, unsigned length);
    virtual MStatus writeBinary(ostream&);

    virtual void copy(const MPxData&);
    MTypeId typeId() const;
    MString name() const;

    static const MString typeName;
    static const MTypeId id;
    /// @}

private:
    openvdb::GridCPtrVec mGrids;
};


////////////////////////////////////////


#endif // OPENVDB_MAYA_DATA_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
