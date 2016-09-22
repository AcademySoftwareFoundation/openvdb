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
/// @file Resource_OpenVDBPoints.cc
///
/// @author Dan Bailey
///

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointCount.h>

#include <iostream>

#include <particle_cloud.h>
#include <geometry_point_cloud.h>
#include <geometry_property_collection.h>
#include <of_app.h>
#include <sys_filesystem.h>
#include <app_progress_bar.h>

#include "ResourceData_OpenVDBPoints.h"
#include "Geometry_OpenVDBPoints.h"
#include "GeometryProperty_OpenVDBPoints.h"

#include "Resource_OpenVDB_Points.h"


using namespace openvdb;
using namespace openvdb::tools;


////////////////////////////////////////


namespace openvdb_points
{
    ResourceData_OpenVDBPoints*
    create_vdb_grid(OfApp& application, const CoreString& filename, const CoreString& gridname, const bool doLocalise, const bool cacheLeaves)
    {
        // early exit if file does not exist on disk

        if (!SysFilesystem::file_exists(filename))      return 0;

        AppProgressBar* read_progress_bar = application.create_progress_bar(CoreString("Loading VDB Points Data: ") + filename + " " + gridname);

        // attempt to open and load the VDB grid from the file

        PointDataGrid::Ptr grid;

        int error = 0;

        try {
            grid = load(filename.get_data(), gridname.get_data());
        }
        catch (const openvdb::IoError& e) {
            std::cerr << "ERROR: Unable to open VDB file (" << filename.get_data() << "): " << e.what() << "\n";
            error = 1;
        }
        catch (const openvdb::KeyError& e) {
            std::cerr << "ERROR: Unable to retrieve grid (" << gridname.get_data() << ") from VDB file: " << e.what() << "\n";
            error = 1;
        }
        catch (const std::exception& e) {
            std::cerr << "ERROR: Unknown error accessing (" << gridname.get_data() << "): " << e.what() << "\n";
            error = 1;
        }

        if (!grid)  error = 1;

        read_progress_bar->destroy();

        if (error)  return 0;

        // localising velocity and radius

        if(doLocalise)
        {
            AppProgressBar* localise_progress_bar = application.create_progress_bar(CoreString("Localising Velocity and Radius for ") + gridname);

            localise(grid);

            localise_progress_bar->destroy();
        }

        // create the resource

        return ResourceData_OpenVDBPoints::create(grid, cacheLeaves);
    }

    Geometry_OpenVDBPoints*
    create_openvdb_points_geometry(OfApp& application, ResourceData_OpenVDBPoints& data, const double fps, const bool overrideRadius, const double radius)
    {
        PointDataGrid::Ptr grid = data.grid();

        if (!grid)  return 0;
        if (!grid->tree().cbeginLeaf())     return 0;

        // check descriptor for velocity and radius (pscale)

        PointDataTree::LeafCIter iter = grid->tree().cbeginLeaf();

        assert(iter);

        // retrieve motion blur length and direction

        const double motionBlurLength = application.get_motion_blur_length();
        const int motionBlurDirection = application.get_motion_blur_direction();
        const AppBase::MotionBlurDirectionMode motionBlurMode =
                            application.get_motion_blur_direction_mode_from_value(motionBlurDirection);

        Geometry_OpenVDBPoints* geometry = Geometry_OpenVDBPoints::create(  grid, overrideRadius, radius,
                                                                            fps, motionBlurLength, motionBlurMode);

        if (geometry)
        {
            AppProgressBar* acc_progress_bar = application.create_progress_bar(CoreString("Computing VDB Point Acceleration Structures"));

            geometry->computeAccelerationStructures();

            acc_progress_bar->destroy();
        }

        return geometry;
    }

    GeometryPropertyCollection*
    create_openvdb_points_geometry_property(OfApp& application, ResourceData_OpenVDBPoints& data)
    {
        GeometryPropertyCollection *property_array = new GeometryPropertyCollection;

        CoreVector<GeometryProperty*> properties;

        PointDataGrid::Ptr grid = data.grid();

        if (!grid)  return property_array;

        const PointDataTree& tree = grid->tree();

        PointDataTree::LeafCIter iter = tree.cbeginLeaf();

        if (!iter)  return property_array;

        const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

        for (AttributeSet::Descriptor::ConstIterator it = descriptor.map().begin(),
            end = descriptor.map().end(); it != end; ++it) {

            const openvdb::NamePair& type = descriptor.type(it->second);

            // only floating point property types are supported

            if (type.first != "float" && type.first != "vec3s")   continue;

            properties.add(new GeometryProperty_OpenVDBPoints(&data, it->first, type.first));
        }

        property_array->set(properties);

        return property_array;
    }

    ParticleCloud*
    create_clarisse_particle_cloud(OfApp& application, ResourceData_OpenVDBPoints& data, const bool loadVelocities)
    {
        const PointDataGrid::Ptr grid = data.grid();

        if (!grid)                  return new ParticleCloud();

        const PointDataTree& tree = grid->tree();

        if (!tree.cbeginLeaf())     return new ParticleCloud();

        const openvdb::Index64 size = pointCount(tree);

        if (size == 0)              return new ParticleCloud();

        CoreArray<GMathVec3d> array(size);

        // load Clarisse particle positions

        AppProgressBar* convert_progress_bar = application.create_progress_bar(CoreString("Converting Point Positions into Clarisse"));

        const openvdb::math::Transform& transform = grid->transform();

        unsigned arrayIndex = 0;

        for (PointDataTree::LeafCIter leaf = tree.cbeginLeaf(); leaf; ++leaf)
        {
            const AttributeHandle<openvdb::Vec3f>::Ptr positionHandle =
                AttributeHandle<openvdb::Vec3f>::create(leaf->constAttributeArray("P"));

            for (PointDataTree::LeafNodeType::IndexOnIter iter = leaf->beginIndexOn(); iter; ++iter)
            {
                const openvdb::Vec3f positionVoxelSpace = positionHandle->get(*iter);
                const openvdb::Vec3d positionIndexSpace = positionVoxelSpace + iter.getCoord().asVec3d();
                const openvdb::Vec3d positionWorldSpace = transform.indexToWorld(positionIndexSpace);

                array[arrayIndex][0] = positionWorldSpace[0];
                array[arrayIndex][1] = positionWorldSpace[1];
                array[arrayIndex][2] = positionWorldSpace[2];

                arrayIndex++;
            }
        }

        convert_progress_bar->destroy();

        // construct the point cloud

        AppProgressBar* cloud_progress_bar = application.create_progress_bar(CoreString("Creating Clarisse Point Cloud"));

        GeometryPointCloud pointCloud;

        pointCloud.init(array.get_count());
        pointCloud.init_positions(array.get_data());

        cloud_progress_bar->destroy();

        // load the velocities (only if v attribute exists)

        if (loadVelocities && tree.cbeginLeaf()->hasAttribute("v"))
        {
            AppProgressBar* velocity_progress_bar = application.create_progress_bar(CoreString("Loading Velocities into Point Cloud"));

            arrayIndex = 0;

            for (PointDataTree::LeafCIter leaf = tree.cbeginLeaf(); leaf; ++leaf)
            {
                const AttributeHandle<openvdb::Vec3f>::Ptr velocityHandle =
                    AttributeHandle<openvdb::Vec3f>::create(leaf->constAttributeArray("v"));

                for (PointDataTree::LeafNodeType::IndexOnIter iter = leaf->beginIndexOn(); iter; ++iter)
                {
                    const openvdb::Vec3f velocity = velocityHandle->get(*iter);

                    array[arrayIndex][0] = velocity.x();
                    array[arrayIndex][1] = velocity.y();
                    array[arrayIndex][2] = velocity.z();

                    arrayIndex++;
                }
            }

            pointCloud.init_velocities(array.get_data());

            velocity_progress_bar->destroy();
        }

        return new ParticleCloud(pointCloud);
    }
} // namespace openvdb_points


////////////////////////////////////////


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
