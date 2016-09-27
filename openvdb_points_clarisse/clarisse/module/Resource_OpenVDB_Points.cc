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

#include <boost/algorithm/string/predicate.hpp>
#include <boost/utility/enable_if.hpp>

#include <particle_cloud.h>
#include <geometry_point_cloud.h>
#include <geometry_property.h>
#include <geometry_property_collection.h>
#include <geometry_point_property.h>
#include <geometry_point_property_collection.h>
#include <of_app.h>
#include <sys_filesystem.h>
#include <app_progress_bar.h>

#include "ResourceData_OpenVDBPoints.h"
#include "Geometry_OpenVDBPoints.h"
#include "GeometryProperty_OpenVDBPoints.h"

#include "Resource_OpenVDB_Points.h"


////////////////////////////////////////


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

// add VecTraits specialization for Quat<T> and Mat<SIZE, T> as the following
// classes only need to know about the container size and type to flatten them

template<typename T> struct VecTraits<math::Quat<T> > {
    static const bool IsVec = true;
    static const int Size = 4;
    typedef T ElementType;
};

template<unsigned SIZE, typename T> struct VecTraits<math::Mat<SIZE, T> > {
    static const bool IsVec = true;
    static const int Size = SIZE*SIZE;
    typedef T ElementType;
};

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


////////////////////////////////////////


using namespace openvdb;
using namespace openvdb::tools;


namespace openvdb_points
{
namespace resource_internal
{

class PropertyDataInfo : public GeometryProperty::LoadDataInfo
{
public:
    PropertyDataInfo(
        const ResourceData_OpenVDBPoints& resourceData,
        GeometryPointProperty* property,
        GeometryPointPropertyCollection* propertyCollection)
    : mResourceData(resourceData)
    , mProperty(property)
    , mPropertyCollection(propertyCollection) {}

    unsigned int extent() const { return mProperty->get_value_extent(); }
    size_t size() const { return mProperty->get_value_extent() * mProperty->get_value_count(); }
    PointDataGrid::ConstPtr grid() const { return mResourceData.grid(); }
    std::string name() const { return mProperty->get_name().get_data(); }
    bool isPosition() const { return name() == "P"; }

    template <typename T>
    void set_values(const unsigned int& sampleIndex, const CoreArray<T>& values)
    {
        ResourceProperty* const data = new ResourceProperty(this->name().c_str());
        data->init(mProperty->get_value_type(), this->extent(), mProperty->get_value_count());
        data->set_values(values.get_data());

        if(!mProperty->init(sampleIndex, data))     delete data;
        else                                        mPropertyCollection->advert_memory_changed();
    }

private:
    const ResourceData_OpenVDBPoints&       mResourceData;
    GeometryPointProperty* const            mProperty;
    GeometryPointPropertyCollection* const  mPropertyCollection;
}; // class PropertyDataInfo

template<typename ValueType>
static typename boost::enable_if_c<openvdb::VecTraits<ValueType>::Size==3>::type
indexToWorld(ValueType& value, const openvdb::Coord& ijk, const openvdb::math::Transform& transform) {
    value = transform.indexToWorld(value + ijk.asVec3s());
}

template<typename ValueType>
static typename boost::disable_if_c<openvdb::VecTraits<ValueType>::Size==3>::type
indexToWorld(ValueType& value, const openvdb::Coord& ijk, const openvdb::math::Transform& transform) {}

template<typename ValueType>
static typename boost::enable_if_c<openvdb::VecTraits<ValueType>::IsVec>::type
setValue(CoreArray<typename openvdb::VecTraits<ValueType>::ElementType>& values, unsigned int& index, const ValueType& value)
{
    for(size_t i = 0; i < openvdb::VecTraits<ValueType>::Size; ++i)
        values[index++] = value[i];
}

template<typename ValueType>
static typename boost::disable_if_c<openvdb::VecTraits<ValueType>::IsVec>::type
setValue(CoreArray<ValueType>& values, unsigned int& index, const ValueType& value)
{
    values[index++] = value;
}

// Helper for deferred property loading using attribute data from the VDB Grid.

template<typename ValueType>
static void buildProperty(const unsigned int& sampleIndex, GeometryProperty::LoadDataInfo* loadInfo)
{
    typedef openvdb::tools::AttributeHandle<ValueType>           HandleType;
    typedef openvdb::tools::PointDataTree::LeafNodeType          LeafNodeType;
    typedef typename openvdb::VecTraits<ValueType>::ElementType  ElementType;

    PropertyDataInfo* const info = (PropertyDataInfo*)loadInfo;

    const std::string name = info->name();
    const size_t size = info->size();

    CoreArray<ElementType> values(size);
    unsigned int index = 0;

    const openvdb::tools::PointDataGrid::ConstPtr grid = info->grid();

    for (openvdb::tools::PointDataTree::LeafCIter leafIter = grid->tree().cbeginLeaf(); leafIter; ++leafIter)
    {
        typename HandleType::Ptr handle = HandleType::create(leafIter->constAttributeArray(name));

        for (LeafNodeType::IndexOnIter indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {

            ValueType value = handle->get(*indexIter);

            // Transform position to world space
            if(info->isPosition()) indexToWorld(value, indexIter.getCoord(), grid->transform());
            setValue(values, index, value);
        }
    }

    assert(index == size);

    info->set_values(sampleIndex, values);
}

} // namespace resource_internal


////////////////////////////////////////


ResourceData_OpenVDBPoints*
create_vdb_grid(OfApp& application, const CoreString& filename, const CoreString& gridname, const bool doLocalise, const bool cacheLeaves)
{
    // early exit if file does not exist on disk

    if (!SysFilesystem::file_exists(filename))      return 0;

    AppProgressBar* read_progress_bar = application.create_progress_bar(CoreString("Loading VDB Points Data: ") + filename + " " + gridname);

    // attempt to open and load the VDB grid from the file

    PointDataGrid::Ptr grid;

    int error = 0;
    std::stringstream ostr;

    try {
        grid = load(filename.get_data(), gridname.get_data());
    }
    catch (const openvdb::IoError& e) {
        ostr << "ERROR: Unable to open VDB file (" << filename.get_data() << "): " << e.what() << std::endl;
        LOG_ERROR(ostr.str().c_str());
        error = 1;
    }
    catch (const openvdb::KeyError& e) {
        ostr << "ERROR: Unable to retrieve grid (" << gridname.get_data() << ") from VDB file: " << e.what() << std::endl;
        LOG_ERROR(ostr.str().c_str());
        error = 1;
    }
    catch (const std::exception& e) {
        ostr << "ERROR: Unknown error accessing (" << gridname.get_data() << "): " << e.what() << std::endl;
        LOG_ERROR(ostr.str().c_str());
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

    const openvdb::Index64 size = activePointCount(tree);

    if (size == 0)              return new ParticleCloud();

    CoreArray<GMathVec3f> array(size);

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


GeometryPointPropertyCollection*
create_clarisse_particle_cloud_geometry_property(OfApp& application, ResourceData_OpenVDBPoints& data)
{
    using resource_internal::PropertyDataInfo;
    using resource_internal::buildProperty;

    GeometryPointPropertyCollection* collection = new GeometryPointPropertyCollection;

    openvdb::tools::PointDataGrid::Ptr grid = data.grid();

    if(!grid)  return collection;

    const openvdb::tools::PointDataTree& tree = grid->tree();
    const openvdb::tools::PointDataTree::LeafCIter iter = tree.cbeginLeaf();

    if(!iter) return collection;

    const unsigned int numPoints = activePointCount(tree);

    if(numPoints == 0) return collection;

    CoreVector<GeometryPointProperty*> properties;

    const openvdb::tools::AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    for (openvdb::tools::AttributeSet::Descriptor::ConstIterator it = descriptor.map().begin(),
        end = descriptor.map().end(); it != end; ++it)
    {
        const openvdb::Name name = it->first;

        // don't expose group attributes as properties
        if (descriptor.hasGroup(name))  continue;

        const openvdb::Name& valueType = descriptor.valueType(it->second);
        ResourceProperty::Type resourceType(ResourceProperty::TYPE_COUNT);

        if (valueType == "bool")         resourceType = ResourceProperty::TYPE_INT_8;
        else if (valueType == "int16")   resourceType = ResourceProperty::TYPE_INT_16;
        else if (valueType == "int32")   resourceType = ResourceProperty::TYPE_INT_32;
        else if (valueType == "int64")   resourceType = ResourceProperty::TYPE_INT_64;
        else if (valueType == "float")   resourceType = ResourceProperty::TYPE_FLOAT_32;
        else if (valueType == "double")  resourceType = ResourceProperty::TYPE_FLOAT_64;
        else if (valueType == "vec3i")   resourceType = ResourceProperty::TYPE_INT_32;
        else if (valueType == "vec3s")   resourceType = ResourceProperty::TYPE_FLOAT_32;
        else if (valueType == "vec3d")   resourceType = ResourceProperty::TYPE_FLOAT_64;
        else if (valueType == "mat4s")   resourceType = ResourceProperty::TYPE_FLOAT_32;
        else if (valueType == "mat4d")   resourceType = ResourceProperty::TYPE_FLOAT_64;
        else if (valueType == "quats")   resourceType = ResourceProperty::TYPE_FLOAT_32;
        else if (valueType == "quatd")   resourceType = ResourceProperty::TYPE_FLOAT_64;
        else {
            std::ostringstream ostr;
            ostr << "Unsupported Attribute \"" << name << "\" with type \"" << valueType << "\". Skipping.";
            LOG_ERROR(ostr.str().c_str());
            continue;
        }

        size_t resourceSize(1);

        if (boost::starts_with(valueType, "vec3"))          resourceSize = 3;
        else if (boost::starts_with(valueType, "quat"))     resourceSize = 4;
        else if (boost::starts_with(valueType, "mat4"))     resourceSize = 4*4;

        GeometryPointProperty* property = new GeometryPointProperty(name.c_str(), GMathTimeSampling(0), resourceType, numPoints, resourceSize, 0);
        PropertyDataInfo* info = new PropertyDataInfo(data, property, collection);

        if (valueType == "bool")         property->set_deferred_loading(&buildProperty<bool>, info);
        else if (valueType == "int16")   property->set_deferred_loading(&buildProperty<int16_t>, info);
        else if (valueType == "int32")   property->set_deferred_loading(&buildProperty<int32_t>, info);
        else if (valueType == "int64")   property->set_deferred_loading(&buildProperty<int64_t>, info);
        else if (valueType == "float")   property->set_deferred_loading(&buildProperty<float>, info);
        else if (valueType == "double")  property->set_deferred_loading(&buildProperty<double>, info);
        else if (valueType == "vec3i")   property->set_deferred_loading(&buildProperty<openvdb::Vec3i>, info);
        else if (valueType == "vec3s")   property->set_deferred_loading(&buildProperty<openvdb::Vec3s>, info);
        else if (valueType == "vec3d")   property->set_deferred_loading(&buildProperty<openvdb::Vec3d>, info);
        else if (valueType == "mat4s")   property->set_deferred_loading(&buildProperty<openvdb::math::Mat4<float> >, info);
        else if (valueType == "mat4d")   property->set_deferred_loading(&buildProperty<openvdb::math::Mat4<double> >, info);
        else if (valueType == "quats")   property->set_deferred_loading(&buildProperty<openvdb::math::Quat<float> >, info);
        else if (valueType == "quatd")   property->set_deferred_loading(&buildProperty<openvdb::math::Quat<double> >, info);
        else {
            std::ostringstream ostr;
            ostr << "Internal Error: Unsupport function callback for attribute type \"" << valueType << "\"";
            LOG_ERROR(ostr.str().c_str());
            delete info;
            delete property;
            continue;
        }

        properties.add(property);
    }

    collection->set(properties);

    return collection;
}


} // namespace openvdb_points


////////////////////////////////////////


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
