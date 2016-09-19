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
/// @file Module_OpenVDB_Points.h
///
/// @author Dan Bailey
///
/// @brief OpenVDB Points Module for Clarisse
///

#include <of_object.h>
#include <resource_data.h>

////////////////////////////////////////


namespace resource
{
    // create a ResourceData_OpenVDBPoints object from parameters on the OpenVDBPoints node
    // @param localise      if true, pscale and v attributes will be transformed to index-space on resource creation
    // @param cacheLeaves   if true, leaves are cached to a local array to reduce tree traversal time
    ResourceData* create_vdb_grid(OfObject& object, const bool localise = true, const bool cacheLeaves = true);

    // create a Geometry_OpenVDBPoints object from a ResourceData_OpenVDBPoints object
    // and parameters on the OpenVDBPoints node
    ResourceData* create_openvdb_points_geometry(OfObject& object);

    // create a property array of GeometryProperty_OpenVDBPoints objects from a Geometry_OpenVDBPoints node
    ResourceData* create_openvdb_points_geometry_property(OfObject& object);

    // create a Clarisse ParticleCloud object from a ResourceData_OpenVDBPoints object
    ResourceData* create_clarisse_particle_cloud(OfObject& object);
} // namespace resource


////////////////////////////////////////


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
