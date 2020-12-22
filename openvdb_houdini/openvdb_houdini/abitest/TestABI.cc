// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointConversion.h>

#include <stdexcept>

#ifdef HOUDINI
namespace houdini {
#endif

////////////////////////////////////////

// Validation Methods

// throw an exception if the condition is false
inline void VDB_ASSERT(const bool condition, const std::string& file, const int line)
{
    if (!condition) {
        throw std::runtime_error("Assertion Fail in file " + file + " on line " + std::to_string(line));
    }
}

#define VDB_ASSERT(condition) VDB_ASSERT(condition, __FILE__, __LINE__)

////////////////////////////////////////

// Version methods

const char* getABI()
{
    return OPENVDB_PREPROC_STRINGIFY(OPENVDB_ABI_VERSION_NUMBER);
}

const char* getNamespace()
{
    return OPENVDB_PREPROC_STRINGIFY(OPENVDB_VERSION_NAME);
}

////////////////////////////////////////

// Grid Methods

void* createFloatGrid()
{
    openvdb::initialize();

    openvdb::FloatGrid::Ptr grid =
        openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
            /*radius=*/1.0f, /*center=*/openvdb::Vec3f(0.0f), /*voxelSize=*/0.1f);

    return new openvdb::FloatGrid(*grid);
}

void* createPointsGrid()
{
    openvdb::initialize();

    const std::vector<openvdb::Vec3R> pos {
        openvdb::Vec3R(0,0,0),
        openvdb::Vec3R(10,10,10),
        openvdb::Vec3R(10,-10,10),
        openvdb::Vec3R(10,10,-10),
        openvdb::Vec3R(10,-10,-10),
        openvdb::Vec3R(-10,10,-10),
        openvdb::Vec3R(-10,10,10),
        openvdb::Vec3R(-10,-10,10),
        openvdb::Vec3R(-10,-10,-10)
    };

    auto transform = openvdb::math::Transform::createLinearTransform(0.1);

    openvdb::points::PointDataGrid::Ptr grid =
        openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
            openvdb::points::PointDataGrid, openvdb::Vec3R>(pos, *transform);

    return new openvdb::points::PointDataGrid(*grid);
}

void cleanupFloatGrid(void* gridPtr)
{
    openvdb::uninitialize();

    openvdb::FloatGrid* grid =
        static_cast<openvdb::FloatGrid*>(gridPtr);
    delete grid;
}

void cleanupPointsGrid(void* gridPtr)
{
    openvdb::uninitialize();

    openvdb::points::PointDataGrid* grid =
        static_cast<openvdb::points::PointDataGrid*>(gridPtr);
    delete grid;
}

int validateFloatGrid(void* gridPtr)
{
    openvdb::FloatGrid* grid =
        static_cast<openvdb::FloatGrid*>(gridPtr);

    VDB_ASSERT(grid);
    VDB_ASSERT(grid->tree().activeVoxelCount() > openvdb::Index64(0));
    VDB_ASSERT(grid->tree().leafCount() > openvdb::Index64(0));

    std::stringstream ss;
    grid->tree().print(ss);
    VDB_ASSERT(ss.str().length() > size_t(0));

    auto iter = grid->tree().cbeginLeaf();
    VDB_ASSERT(iter);
    VDB_ASSERT(iter->memUsage() > openvdb::Index64(0));

    return 0;
}

int validatePointsGrid(void* gridPtr)
{
    openvdb::points::PointDataGrid* grid =
        static_cast<openvdb::points::PointDataGrid*>(gridPtr);

    VDB_ASSERT(grid);
    VDB_ASSERT(grid->tree().activeVoxelCount() > openvdb::Index64(0));
    VDB_ASSERT(grid->tree().leafCount() > openvdb::Index64(0));

    std::stringstream ss;
    grid->tree().print(ss);
    VDB_ASSERT(ss.str().length() > size_t(0));

    auto iter = grid->tree().cbeginLeaf();
    VDB_ASSERT(iter);
    VDB_ASSERT(iter->memUsage() > openvdb::Index64(0));

    auto handle = openvdb::points::AttributeHandle<openvdb::Vec3f>::create(iter->constAttributeArray("P"));
    VDB_ASSERT(handle->get(0) == openvdb::Vec3f(0));

    return 0;
}

#ifdef HOUDINI
} // namespace houdini
#endif
