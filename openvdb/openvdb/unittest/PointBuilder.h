// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_UNITTEST_POINT_BUILDER_HAS_BEEN_INCLUDED
#define OPENVDB_UNITTEST_POINT_BUILDER_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/PointIndexGrid.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h>

/// @brief  Get 8 corner points from a cube with a given scale, ordered such
///   that if used for conversion to OpenVDB Points, that the default
///   iteration order remains consistent
inline std::vector<openvdb::Vec3f>
getBoxPoints(const float scale = 1.0f)
{
    // This order is configured to be the same layout when
    // a vdb points grid is constructed and so matches methods
    // like setGroup or populateAttribute
    std::vector<openvdb::Vec3f> pos = {
        openvdb::Vec3f(-1.0f, -1.0f, -1.0f),
        openvdb::Vec3f(-1.0f, -1.0f, 1.0f),
        openvdb::Vec3f(-1.0f, 1.0f, -1.0f),
        openvdb::Vec3f(-1.0f, 1.0f, 1.0f),
        openvdb::Vec3f(1.0f, -1.0f, -1.0f),
        openvdb::Vec3f(1.0f, -1.0f, 1.0f),
        openvdb::Vec3f(1.0f, 1.0f, -1.0f),
        openvdb::Vec3f(1.0f, 1.0f, 1.0f)
    };

    for (auto& p : pos) p *= scale;
    return pos;
}

/// @brief  Builder pattern for creating PointDataGrids which simplifies
///   a lot of the repetitive boilerplate
struct PointBuilder
{
    using PointDataTreeT = openvdb::points::PointDataTree;
    using PointIndexTreeT = openvdb::tools::PointIndexTree;

    using CallbackT1 = std::function<void(PointDataTreeT&, const PointIndexTreeT&)>;
    using CallbackT2 = std::function<void(PointDataTreeT&)>;

    // init the builder with a set of positions
    PointBuilder(const std::vector<openvdb::Vec3f>& pos) : positions(pos) {}

    // set the desired voxel size
    PointBuilder& voxelsize(double in) { vs = in; return *this; }

    // add a group to be created with membership data
    PointBuilder& group(const std::vector<short>& in,
        const std::string& name = "group")
    {
        callbacks.emplace_back([in, name](PointDataTreeT& tree, const PointIndexTreeT& index) {
            openvdb::points::appendGroup(tree, name);
            openvdb::points::setGroup(tree, index, in, name);
        });
        return *this;
    }

    // add a uniform attribute
    template <typename ValueT>
    PointBuilder& attribute(const ValueT& in, const std::string& name)
    {
        callbacks.emplace_back([in, name](PointDataTreeT& tree, const PointIndexTreeT&) {
            openvdb::points::appendAttribute<ValueT>(tree, name, in);
        });
        return *this;
    }

    // add a varying attribute
    template <typename ValueT>
    PointBuilder& attribute(const std::vector<ValueT>& in, const std::string& name)
    {
        callbacks.emplace_back([in, name](PointDataTreeT& tree, const PointIndexTreeT& index) {
            openvdb::points::PointAttributeVector<ValueT> rwrap(in);
            openvdb::points::appendAttribute<ValueT>(tree, name);
            openvdb::points::populateAttribute(tree, index, name, rwrap);
        });
        return *this;
    }

    // add a custom callback of T1
    PointBuilder& callback(const CallbackT1& c)
    {
        callbacks.emplace_back(c); return *this;
    }

    // add a custom callback of T2
    PointBuilder& callback(const CallbackT2& c)
    {
        auto wrap = [c](PointDataTreeT& tree, const PointIndexTreeT&) { c(tree); };
        callbacks.emplace_back(wrap); return *this;
    }

    // build and return the points
    openvdb::points::PointDataGrid::Ptr get()
    {
        openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(vs);
        openvdb::points::PointAttributeVector<openvdb::Vec3f> wrap(positions);
        auto index = openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(wrap, vs);
        auto points = openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
                openvdb::points::PointDataGrid>(*index, wrap, *transform);
        for (auto c : callbacks) c(points->tree(), index->tree());
        return points;
    }

private:
    double vs = 0.1;
    std::vector<openvdb::Vec3f> positions = {};
    std::vector<CallbackT1> callbacks = {};
};

#endif // OPENVDB_UNITTEST_POINT_BUILDER_HAS_BEEN_INCLUDED
