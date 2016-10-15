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
/// @file Module_OpenVDB_Curves.cc
///
/// @author Dan Bailey
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
#include <curve_mesh.h>
#include <app_object.h>
#include <of_app.h>
#include <of_class.h>
#include <sys_filesystem.h>

#include "Module_OpenVDB_Curves.cma"

#include "ResourceData_OpenVDBPoints.h"
#include "Resource_OpenVDB_Points.h"


using namespace openvdb_points;


////////////////////////////////////////


IX_BEGIN_DECLARE_MODULE_CALLBACKS(OpenVDBCurves, ModuleGeometryCallbacks)
    static void init_class(OfClass& cls);
    static ResourceData* create_resource(OfObject&, const int&, void*);
    static void on_attribute_change(OfObject&, const OfAttr&, int&, const int&);
    static bool on_time_change(OfObject&, const double&);
    static void get_grid_tag_candidates(const OfObject&, const OfAttr&, CoreVector<CoreString>&, CoreArray<bool>&);
IX_END_DECLARE_MODULE_CALLBACKS(OpenVDBCurves)


////////////////////////////////////////


void registerVDBCurves(OfApp& app, CoreVector<OfClass *>& new_classes)
{
    // Geometry VDB Curves

    OfClass* new_class = IX_DECLARE_MODULE_CLASS(OpenVDBCurves);
    new_classes.add(new_class);

    IX_MODULE_CLBK* module_callbacks;
    IX_CREATE_MODULE_CLBK(new_class, module_callbacks)
    IX_MODULE_CLBK::init_class(*new_class);

    IX_INIT_CLASS_TAG_CALLBACK(new_class, grid)

    module_callbacks->cb_on_attribute_change = IX_MODULE_CLBK::on_attribute_change;
    module_callbacks->cb_create_resource = IX_MODULE_CLBK::create_resource;
    module_callbacks->cb_on_new_time = IX_MODULE_CLBK::on_time_change;
}


////////////////////////////////////////


namespace
{
    // define the resource id to use for VDB grids
    enum {
        RESOURCE_ID_VDB_GRID = ModuleGeometry::RESOURCE_ID_COUNT,
    };
} // unnamed namespace


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
    cls.add_resource(RESOURCE_ID_VDB_GRID, data_attrs, deps, "vdb_points_data");

    deps.add(RESOURCE_ID_VDB_GRID);

    CoreVector<CoreString> geo_attrs;
    geo_attrs.add("enable_motion_blur");

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
    const CoreString attr_name = attr.get_name();
    if (attr_name == "filename")
    {
        const std::string filename = object.get_attribute("filename")->get_string().get_data();
        object.get_attribute("output_filename")->set_string(filename.c_str());
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


// Create the resources

ResourceData*
IX_MODULE_CLBK::create_resource(OfObject& object,
                                const int& resource_id,
                                void* data)
{
    OfApp& application = object.get_application();

    if (resource_id == RESOURCE_ID_VDB_GRID)
    {
        const CoreString filename = object.get_attribute("output_filename")->get_string();
        const CoreString gridname = object.get_attribute("grid")->get_string();

        ResourceData_OpenVDBPoints* resourceData = create_vdb_grid(application, filename, gridname, false, false);
        if (!resourceData)  return 0;

        return resourceData;
    }
    else if (resource_id == ModuleGeometry::RESOURCE_ID_GEOMETRY)
    {
        ModuleGeometry* module = (ModuleGeometry*) object.get_module();
        ResourceData_OpenVDBPoints* data = (ResourceData_OpenVDBPoints*) module->get_resource(RESOURCE_ID_VDB_GRID);
        if (!data) return 0;

        CurveMesh* curveMesh = create_clarisse_curve_mesh(application, *data);

        return curveMesh;
    }

    return 0;
}


////////////////////////////////////////


// Register the plugin

IX_BEGIN_EXTERN_C
    DSO_EXPORT void
    on_register_module(OfApp& app, CoreVector<OfClass *>& new_classes)
    {
        registerVDBCurves(app, new_classes);
    }
IX_END_EXTERN_C

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
