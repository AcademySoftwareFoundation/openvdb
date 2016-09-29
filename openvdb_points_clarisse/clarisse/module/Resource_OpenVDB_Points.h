///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
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
/// @file Resource_OpenVDB_Points.h
///
/// @author Dan Bailey
///
/// @brief OpenVDB Points Resource for Clarisse
///


#ifndef OPENVDB_CLARISSE_RESOURCE_OPENVDB_POINTS_HAS_BEEN_INCLUDED
#define OPENVDB_CLARISSE_RESOURCE_OPENVDB_POINTS_HAS_BEEN_INCLUDED


#include <of_object.h>
#include <resource_data.h>


#include "ResourceData_OpenVDBPoints.h"


////////////////////////////////////////


class Geometry_OpenVDBPoints;
class ParticleCloud;
class CurveMesh;
class GeometryPropertyCollection;
class GeometryPointPropertyCollection;


////////////////////////////////////////


namespace openvdb_points
{
    /// Create a ResourceData_OpenVDBPoints object from parameters on the OpenVDBPoints node
    /// @param application      used for adding progress bars
    /// @param filename         the filename of the VDB
    /// @param gridname         the name of the VDB grid to use
    /// @param localise         if true, pscale and v attributes will be transformed to index-space on resource creation
    /// @param cacheLeaves      if true, leaves are cached to a local array to reduce tree traversal time
    ResourceData_OpenVDBPoints*
    create_vdb_grid(OfApp& application, const CoreString& filename, const CoreString& gridname,
                    const bool localise = true, const bool cacheLeaves = true);

    /// Create a Clarisse ParticleCloud object from a ResourceData_OpenVDBPoints object
    /// @param application      used for adding progress bars
    /// @param data             the ResourceData_OpenVDBPoints
    /// @param loadVelocities if true and v attribute available, store velocities on point cloud
    ParticleCloud*
    create_clarisse_particle_cloud( OfApp& application, ResourceData_OpenVDBPoints& data,
                                    const bool loadVelocities = false);

    /// Create a Clarisse GeometryPointPropertyCollection from a ResourceData_OpenVDBPoints object
    /// @param application      used for adding progress bars
    /// @param data             the ResourceData_OpenVDBPoints
    GeometryPointPropertyCollection*
    create_clarisse_particle_cloud_geometry_property(OfApp& application, ResourceData_OpenVDBPoints& data);

    /// Create a Clarisse CurveMesh object from a ResourceData_OpenVDBPoints object
    /// @param application      used for adding progress bars
    /// @param data             the ResourceData_OpenVDBPoints
    CurveMesh*
    create_clarisse_curve_mesh(OfApp& application, ResourceData_OpenVDBPoints& data);

    /// Create a Geometry_OpenVDBPoints object from a ResourceData_OpenVDBPoints object
    /// @param application      used for adding progress bars
    /// @param data             the ResourceData_OpenVDBPoints
    /// @param fps              frames / second
    /// @param overrideRadius   if true, radius parameter is interpreted as an explicit value otherwise as a scale
    /// @param radius           an explicit or scale value depending on the value of overrideRadius
    Geometry_OpenVDBPoints*
    create_openvdb_points_geometry( OfApp& application, ResourceData_OpenVDBPoints& data,
                                    const double fps, const bool overrideRadius, const double radius);

    /// Create a property array of GeometryProperty_OpenVDBPoints objects from a Geometry_OpenVDBPoints node
    /// @param application      used for adding progress bars
    /// @param data             the ResourceData_OpenVDBPoints
    GeometryPropertyCollection*
    create_openvdb_points_geometry_property(OfApp& application, ResourceData_OpenVDBPoints& data);
} // namespace openvdb_points


////////////////////////////////////////


#endif // OPENVDB_CLARISSE_RESOURCE_OPENVDB_POINTS_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
