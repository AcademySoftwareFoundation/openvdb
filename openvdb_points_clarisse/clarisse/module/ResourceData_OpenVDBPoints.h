///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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
/// @file ResourceData_OpenVDBPoints.h
///
/// @author Dan Bailey
///
/// @brief  A Resource to store OpenVDB Points data
///

#ifndef OPENVDB_CLARISSE_RESOURCEDATA_OPENVDBPOINTS_HAS_BEEN_INCLUDED
#define OPENVDB_CLARISSE_RESOURCEDATA_OPENVDBPOINTS_HAS_BEEN_INCLUDED

#include <resource_data.h>

#include <openvdb/openvdb.h>
#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>


////////////////////////////////////////


namespace openvdb_points
{

openvdb::tools::PointDataGrid::Ptr
load(   const std::string& filename,
        const std::string& gridname,
        const bool doPrune = true);

void localise(openvdb::tools::PointDataGrid::Ptr& grid);

} // namespace openvdb_points


////////////////////////////////////////


class ResourceData_OpenVDBPoints : public ResourceData
{
public:
    typedef openvdb::tools::PointDataTree::LeafNodeType PointDataLeaf;

    static ResourceData_OpenVDBPoints* create(const openvdb::tools::PointDataGrid::Ptr& grid);

    ResourceData_OpenVDBPoints(const openvdb::tools::PointDataGrid::Ptr& grid);

    const openvdb::tools::PointDataGrid::Ptr grid() const;

    const PointDataLeaf* leaf(const unsigned int id) const;

    std::string attribute_type(const std::string& name) const;

    void *create_thread_data() const;
    void destroy_thread_data(void *data) const;

    size_t get_memory_size() const;

private:
    const openvdb::tools::PointDataGrid::Ptr m_grid;
    const openvdb::tools::AttributeSet::Descriptor::Ptr m_descriptor;

    std::vector<PointDataLeaf*> m_leaves;
}; // ResourceData_OpenVDBPoints

#endif // OPENVDB_CLARISSE_RESOURCEDATA_OPENVDBPOINTS_HAS_BEEN_INCLUDED
