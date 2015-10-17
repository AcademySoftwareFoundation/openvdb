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
/// @file GeometryProperty_OpenVDBPoints.h
///
/// @author Dan Bailey
///
/// @brief  OpenVDB Points Geometry Property for use with an ExtractProperty node
///


#ifndef OPENVDB_CLARISSE_GEOMETRY_PROPERTY_OPENVDBPOINTS_HAS_BEEN_INCLUDED
#define OPENVDB_CLARISSE_GEOMETRY_PROPERTY_OPENVDBPOINTS_HAS_BEEN_INCLUDED

#include <geometry_property.h>

#include <openvdb/openvdb.h>
#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>

class ResourceData_OpenVDBPoints;

class GeometryProperty_OpenVDBPoints : public GeometryProperty
{
public:
    typedef openvdb::tools::PointDataGrid PointDataGrid;
    typedef openvdb::tools::PointDataTree PointDataTree;
    typedef PointDataTree::LeafNodeType PointDataLeaf;

    GeometryProperty_OpenVDBPoints(const ResourceData_OpenVDBPoints* geometry, const std::string& name, const std::string& type);
    virtual ~GeometryProperty_OpenVDBPoints();

    const PointDataLeaf* leaf(const unsigned int id) const;

    const std::string& name() const { return m_name; }

private:

    static unsigned int extract_value_long( const GeometryProperty& property, const CtxEval& eval_ctx,
                                            const GeometryFragment& fragment, const unsigned int& sample_index,
                                            long long *values, long long *values_du, long long *values_dv, long long *values_dw,
                                            const unsigned int& value_count);

    template<typename ValueType>
    static unsigned int extract_value_float(const GeometryProperty& property, const CtxEval& eval_ctx,
                                            const GeometryFragment& fragment, const unsigned int& sample_index,
                                            double *values, double *values_du, double *values_dv, double *values_dw,
                                            const unsigned int& value_count);

    const ResourceData_OpenVDBPoints* m_geometry;
    const std::string m_name;
    const std::string m_type;

    DECLARE_CLASS;
};

#endif // OPENVDB_CLARISSE_GEOMETRY_OPENVDBPOINTS_HAS_BEEN_INCLUDED
