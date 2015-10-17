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
/// @file Module_OpenVDB_Points.cc
///
/// @author Dan Bailey
///
/// @brief OpenVDB Points Module for Clarisse
///

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>

#include <iostream>

#include <boost/algorithm/string/predicate.hpp>

#include <dso_export.h>
#include <module_geometry.h>
#include <module_object.h>
#include <particle_cloud.h>
#include <geometry_point_cloud.h>
#include <app_object.h>
#include <of_app.h>
#include <of_class.h>
#include <sys_filesystem.h>
#include <app_progress_bar.h>

#include "Geometry_OpenVDBPoints.cma"

#include "ResourceData_OpenVDBPoints.h"
#include "Geometry_OpenVDBPoints.h"
#include "GeometryProperty_OpenVDBPoints.h"


////////////////////////////////////////


IX_BEGIN_DECLARE_MODULE_CALLBACKS(OpenVDBPoints, ModuleGeometryCallbacks)
    static void init_class(OfClass& cls);
    static ResourceData* create_resource(OfObject&, const int&, void*);
    static void* create_thread_data(const OfObject& object, const CtxEval& eval_ctx);
    static void destroy_thread_data(const OfObject& object, const CtxEval& eval_ctx, void *thread_data);
    static void on_attribute_change(OfObject&, const OfAttr&, int&, const int&);
    static bool on_time_change(OfObject&, const double&);
    static void get_grid_tag_candidates(const OfObject&, const OfAttr&, CoreVector<CoreString>&, CoreArray<bool>&);
IX_END_DECLARE_MODULE_CALLBACKS(OpenVDBPoints)


////////////////////////////////////////


void registerVDBPoints(OfApp& app, CoreVector<OfClass *>& new_classes)
{
    // Geometry VDB Points

    OfClass* new_class = IX_DECLARE_MODULE_CLASS(OpenVDBPoints);
    new_classes.add(new_class);

    IX_MODULE_CLBK* module_callbacks;
    IX_CREATE_MODULE_CLBK(new_class, module_callbacks)
    IX_MODULE_CLBK::init_class(*new_class);

    IX_INIT_CLASS_TAG_CALLBACK(new_class, grid)

    module_callbacks->cb_on_attribute_change = IX_MODULE_CLBK::on_attribute_change;
    module_callbacks->cb_create_resource = IX_MODULE_CLBK::create_resource;
    module_callbacks->cb_create_thread_data = IX_MODULE_CLBK::create_thread_data;
    module_callbacks->cb_destroy_thread_data = IX_MODULE_CLBK::destroy_thread_data;
    module_callbacks->cb_on_new_time = IX_MODULE_CLBK::on_time_change;
}


////////////////////////////////////////


namespace resource
{
    enum {
        RESOURCE_ID_VDB_GRID = ModuleGeometry::RESOURCE_ID_COUNT,
    };

    ResourceData* create_vdb_grid(OfObject& object);

    ResourceData* create_openvdb_points_geometry(OfObject& object);

    ResourceData* create_openvdb_points_geometry_property(OfObject& object);

    ResourceData* create_clarisse_particle_cloud(OfObject& object);
} // namespace resource


////////////////////////////////////////


// Callback for initialising the resources and their dependencies

void
IX_MODULE_CLBK::init_class(OfClass& cls)
{
    openvdb::initialize();
    openvdb::points::initialize();

    CoreVector<int> deps;

    CoreVector<CoreString> data_attrs;
    data_attrs.add("filename");
    data_attrs.add("grid");
    cls.add_resource(resource::RESOURCE_ID_VDB_GRID, data_attrs, deps, "vdb_points_data");

    deps.add(resource::RESOURCE_ID_VDB_GRID);

    cls.set_resource_attrs(ModuleGeometry::RESOURCE_ID_GEOMETRY_PROPERTIES, data_attrs);
    cls.set_resource_deps(ModuleGeometry::RESOURCE_ID_GEOMETRY_PROPERTIES, deps);

    CoreVector<CoreString> geo_attrs;
    geo_attrs.add("explicit_radius");
    geo_attrs.add("override_radius");
    geo_attrs.add("radius_scale");
    geo_attrs.add("mode");

    cls.set_resource_attrs(ModuleGeometry::RESOURCE_ID_GEOMETRY, geo_attrs);
    cls.set_resource_deps(ModuleGeometry::RESOURCE_ID_GEOMETRY, deps);
}


// Callback for populating the grid names attribute

void
IX_MODULE_CLBK::get_grid_tag_candidates(const OfObject& object,
                                        const OfAttr& attr,
                                        CoreVector<CoreString>& candidates,
                                        CoreArray<bool>& preset_hints)
{
    CoreString filename = object.get_attribute("output_filename")->get_string();

    // early exit if filename isn't set or does not exist on disk

    if (filename.is_empty())                    return;
    if (!SysFilesystem::file_exists(filename))  return;

    // load the VDB metadata and update the candidate array with grid names

    openvdb::io::File file(filename.get_data());
    file.open();

    openvdb::GridPtrVecPtr grids = file.readAllGridMetadata();
    for (openvdb::GridPtrVecCIter iter = grids->begin(); iter != grids->end(); ++iter)
    {
        openvdb::tools::PointDataGrid::ConstPtr grid = openvdb::gridPtrCast<openvdb::tools::PointDataGrid>(*iter);

        if (grid)   candidates.add(grid->getName().c_str());
    }

    file.close();
}


// Callback to update filename and radius attributes

void
IX_MODULE_CLBK::on_attribute_change(OfObject& object,
                                    const OfAttr& attr,
                                    int& dirtiness,
                                    const int& dirtiness_flags)
{
    CoreString attr_name = attr.get_name();
    if (attr_name == "filename")
    {
        const std::string filename = object.get_attribute("filename")->get_string().get_data();
        object.get_attribute("output_filename")->set_string(filename.c_str());
    }
    else if (attr_name == "override_radius")
    {
        // adjust whether radius attributes are read-only based on override

        if (attr.get_bool())
        {
            object.get_attribute("explicit_radius")->set_read_only(false);
            object.get_attribute("radius_scale")->set_read_only(true);
        }
        else
        {
            object.get_attribute("explicit_radius")->set_read_only(true);
            object.get_attribute("radius_scale")->set_read_only(false);
        }
    }
}


// Callback to refresh the (hidden) filename when the frame changes

bool
IX_MODULE_CLBK::on_time_change(OfObject& object, const double& time)
{
    const std::string filename = object.get_attribute("filename")->get_string().get_data();
    object.get_attribute("output_filename")->set_string(filename.c_str());
    return true;
}


// Callback to create a PointDataAccessor per thread for efficiency

void*
IX_MODULE_CLBK::create_thread_data(const OfObject& object, const CtxEval& eval_ctx)
{
    ModuleGeometry* module = (ModuleGeometry*) object.get_module();
    ResourceData_OpenVDBPoints* data = (ResourceData_OpenVDBPoints*) module->get_resource(resource::RESOURCE_ID_VDB_GRID);
    return data ? data->create_thread_data() : 0;
}


// Callback to destroy a PointDataAccessor per thread

void
IX_MODULE_CLBK::destroy_thread_data(const OfObject& object, const CtxEval& eval_ctx, void *thread_data)
{
    ModuleGeometry *module = (ModuleGeometry *)object.get_module();
    ResourceData_OpenVDBPoints *data = (ResourceData_OpenVDBPoints *)module->get_resource(resource::RESOURCE_ID_VDB_GRID);
    if (data && thread_data) {
        data->destroy_thread_data(thread_data);
    }
}


// Create the resources

ResourceData*
IX_MODULE_CLBK::create_resource(OfObject& object,
                                const int& resource_id,
                                void* data)
{
    const long mode = object.get_attribute("mode")->get_long();

    if (resource_id == resource::RESOURCE_ID_VDB_GRID)
    {
        return resource::create_vdb_grid(object);
    }
    else if (resource_id == ModuleGeometry::RESOURCE_ID_GEOMETRY)
    {
        if (mode == 0) {
            return resource::create_openvdb_points_geometry(object);
        }
        else if (mode == 1) {
            return resource::create_clarisse_particle_cloud(object);
        }
    }
    else if (resource_id == ModuleGeometry::RESOURCE_ID_GEOMETRY_PROPERTIES)
    {
        return resource::create_openvdb_points_geometry_property(object);
    }

    return 0;
}


////////////////////////////////////////


namespace resource
{
    ResourceData*
    create_vdb_grid(OfObject& object)
    {
        const CoreString filename = object.get_attribute("output_filename")->get_string();
        const CoreString gridname = object.get_attribute("grid")->get_string();

        // early exit if file does not exist on disk

        if (!SysFilesystem::file_exists(filename))      return 0;

        AppProgressBar* read_progress_bar = object.get_application().create_progress_bar(CoreString("Loading VDB Points Data: ") + filename + " " + gridname);

        ResourceData_OpenVDBPoints* resourceData;

        // attempt to open and load the VDB grid from the file

        openvdb::tools::PointDataGrid::Ptr grid;

        int error = 0;

        try {
            grid = openvdb_points::load(filename.get_data(), gridname.get_data());
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

        AppProgressBar* localise_progress_bar = object.get_application().create_progress_bar(CoreString("Localising Velocity and Radius for ") + gridname);

        openvdb_points::localise(grid);

        localise_progress_bar->destroy();

        // create the resource

        resourceData = ResourceData_OpenVDBPoints::create(grid);

        // if radius is available on the grid, enable the option to override it

        if (resourceData->attribute_type("pscale") == "") {
            object.get_attribute("override_radius")->set_bool(true);
            object.get_attribute("override_radius")->set_read_only(true);
        }
        else {
            object.get_attribute("override_radius")->set_read_only(false);
        }

        return resourceData;
    }

    ResourceData*
    create_openvdb_points_geometry(OfObject& object)
    {
        ModuleGeometry* module = (ModuleGeometry*) object.get_module();
        ResourceData_OpenVDBPoints* data = (ResourceData_OpenVDBPoints*) module->get_resource(resource::RESOURCE_ID_VDB_GRID);

        if (!data)  return 0;

        const bool override_radius = object.get_attribute("override_radius")->get_bool();
        const double radius = object.get_attribute("explicit_radius")->get_double();
        const double radius_scale = object.get_attribute("radius_scale")->get_double();

        openvdb::tools::PointDataGrid::Ptr grid = data->grid();

        if (!grid)  return 0;
        if (!grid->tree().cbeginLeaf())     return 0;

        openvdb::tools::PointDataTree& tree = grid->tree();

        // check descriptor for velocity and radius (pscale)

        openvdb::tools::PointDataTree::LeafCIter iter = grid->tree().cbeginLeaf();

        assert(iter);

        const openvdb::tools::AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

        // retrieve fps, motion blur length and direction

        const OfApp& application = object.get_application();

        const double fps = object.get_factory().get_time().get_fps();
        const double motionBlurLength = application.get_motion_blur_length();
        const int motionBlurDirection = application.get_motion_blur_direction();
        const AppBase::MotionBlurDirectionMode motionBlurMode =
                            application.get_motion_blur_direction_mode_from_value(motionBlurDirection);

        const double radiusArg = override_radius ? radius : radius_scale;

        Geometry_OpenVDBPoints* geometry = Geometry_OpenVDBPoints::create(  grid, override_radius, radiusArg,
                                                                            fps, motionBlurLength, motionBlurMode);

        if (geometry)
        {
            AppProgressBar* acc_progress_bar = object.get_application().create_progress_bar(CoreString("Computing VDB Point Acceleration Structures"));

            geometry->computeAccelerationStructures();

            acc_progress_bar->destroy();
        }

        return geometry;
    }

    ResourceData*
    create_openvdb_points_geometry_property(OfObject& object)
    {
        GeometryPropertyArray *property_array = new GeometryPropertyArray;

        CoreVector<GeometryProperty*> properties;

        ModuleGeometry* module = (ModuleGeometry*) object.get_module();
        ResourceData_OpenVDBPoints* data = (ResourceData_OpenVDBPoints*) module->get_resource(resource::RESOURCE_ID_VDB_GRID);

        if (!data)  return property_array;

        openvdb::tools::PointDataGrid::Ptr grid = data->grid();

        if (!grid)      return property_array;

        const openvdb::tools::PointDataTree& tree = grid->tree();

        openvdb::tools::PointDataTree::LeafCIter iter = tree.cbeginLeaf();

        if (!iter)  return property_array;

        const openvdb::tools::AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

        for (openvdb::tools::AttributeSet::Descriptor::ConstIterator it = descriptor.map().begin(),
            end = descriptor.map().end(); it != end; ++it) {

            const openvdb::NamePair& type = descriptor.type(it->second);

            // only floating point property types are supported

            if (type.first != "float" &&
                type.first != "half" &&
                type.first != "vec3s" &&
                type.first != "vec3h")   continue;

            properties.add(new GeometryProperty_OpenVDBPoints(data, it->first, type.first));
        }

        property_array->set(properties);

        return property_array;
    }

    ResourceData*
    create_clarisse_particle_cloud(OfObject& object)
    {
        ModuleGeometry* module = (ModuleGeometry*) object.get_module();
        ResourceData_OpenVDBPoints* data = (ResourceData_OpenVDBPoints*) module->get_resource(resource::RESOURCE_ID_VDB_GRID);

        if (!data)                          return new ParticleCloud();

        const openvdb::tools::PointDataGrid::Ptr grid = data->grid();

        if (!grid)                          return new ParticleCloud();
        if (!grid->tree().cbeginLeaf())     return new ParticleCloud();

        typedef openvdb::tools::PointDataTree PointDataTree;
        typedef openvdb::tools::PointDataAccessor<PointDataTree> PointDataAccessor;

        const openvdb::math::Transform& transform = grid->transform();
        const PointDataTree& tree = grid->tree();
        const PointDataAccessor accessor(tree);

        const openvdb::Index64 size = accessor.totalPointCount();

        if (size == 0)                      return new ParticleCloud();

        CoreArray<GMathVec3d> array;

        array.resize(size);

        // load Clarisse particle positions

        AppProgressBar* convert_progress_bar = object.get_application().create_progress_bar(CoreString("Converting Point Positions into Clarisse"));

        unsigned arrayIndex = 0;

        for (PointDataTree::LeafCIter leaf = tree.cbeginLeaf(); leaf; ++leaf)
        {
            const openvdb::tools::AttributeHandle<openvdb::Vec3f>::Ptr positionHandle = leaf->attributeHandle<openvdb::Vec3f>("P");

            for (PointDataTree::LeafNodeType::ValueOnCIter value = leaf->cbeginValueOn(); value; ++value)
            {
                const openvdb::Coord ijk = value.getCoord();

                if (accessor.pointCount(ijk) == 0)  continue;

                const openvdb::Vec3i gridIndexSpace = ijk.asVec3i();

                PointDataAccessor::PointDataIndex pointDataIndex = accessor.get(ijk);

                const unsigned start = pointDataIndex.first;
                const unsigned end = pointDataIndex.second;

                for (unsigned index = start; index < end; index++) {

                    const openvdb::Index64 index64(index);

                    const openvdb::Vec3f positionVoxelSpace = positionHandle->get(index64);
                    const openvdb::Vec3f positionIndexSpace = positionVoxelSpace + gridIndexSpace;
                    const openvdb::Vec3f positionWorldSpace = transform.indexToWorld(positionIndexSpace);

                    array[arrayIndex][0] = positionWorldSpace[0];
                    array[arrayIndex][1] = positionWorldSpace[1];
                    array[arrayIndex][2] = positionWorldSpace[2];

                    arrayIndex++;
                }
            }
        }

        convert_progress_bar->destroy();

        // construct the point cloud

        AppProgressBar* cloud_progress_bar = object.get_application().create_progress_bar(CoreString("Creating Clarisse Point Cloud"));

        GeometryPointCloud pointCloud;

        pointCloud.init(array.get_count());
        pointCloud.init_positions(array.get_data());

        cloud_progress_bar->destroy();

        // load the velocities

        if (tree.cbeginLeaf()->hasAttribute<openvdb::tools::TypedAttributeArray<openvdb::Vec3f> >("v"))
        {
            AppProgressBar* velocity_progress_bar = object.get_application().create_progress_bar(CoreString("Loading Velocities into Point Cloud"));

            arrayIndex = 0;

            for (PointDataTree::LeafCIter leaf = tree.cbeginLeaf(); leaf; ++leaf)
            {
                const openvdb::tools::AttributeHandle<openvdb::Vec3f>::Ptr velocityHandle = leaf->attributeHandle<openvdb::Vec3f>("v");

                for (PointDataTree::LeafNodeType::ValueOnCIter value = leaf->cbeginValueOn(); value; ++value)
                {
                    const openvdb::Coord ijk = value.getCoord();

                    if (accessor.pointCount(ijk) == 0)  continue;

                    PointDataAccessor::PointDataIndex pointDataIndex = accessor.get(ijk);

                    const unsigned start = pointDataIndex.first;
                    const unsigned end = pointDataIndex.second;

                    for (unsigned index = start; index < end; index++) {

                        const openvdb::Index64 index64(index);

                        // VDB Points resource uses index-space velocity so need to revert this back to world-space

                        const openvdb::Vec3f velocity = transform.indexToWorld(velocityHandle->get(index64));

                        array[arrayIndex][0] = velocity.x();
                        array[arrayIndex][1] = velocity.y();
                        array[arrayIndex][2] = velocity.z();

                        arrayIndex++;
                    }
                }
            }

            pointCloud.init_velocities(array.get_data());

            velocity_progress_bar->destroy();
        }

        return new ParticleCloud(pointCloud);
    }
} // namespace resource


////////////////////////////////////////


// Register the plugin

IX_BEGIN_EXTERN_C
    DSO_EXPORT void
    on_register_module(OfApp& app, CoreVector<OfClass *>& new_classes)
    {
        registerVDBPoints(app, new_classes);
    }
IX_END_EXTERN_C
