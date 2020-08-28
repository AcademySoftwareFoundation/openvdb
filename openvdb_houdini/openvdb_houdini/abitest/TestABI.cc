// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>

#include <stdexcept>

#include <openvdb/tools/LevelSetSphere.h>

#ifdef HOUDINI
namespace houdini {
#endif

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

void* createGrid()
{
    openvdb::FloatGrid::Ptr grid =
        openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
            /*radius=*/1.0f, /*center=*/openvdb::Vec3f(0.0f), /*voxelSize=*/0.1f);

    return new openvdb::FloatGrid(*grid);
}

void cleanupGrid(void* gridPtr)
{
    openvdb::FloatGrid* grid =
        static_cast<openvdb::FloatGrid*>(gridPtr);
    delete grid;
}

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

int validateGrid(void* gridPtr)
{
    openvdb::FloatGrid* grid =
        static_cast<openvdb::FloatGrid*>(gridPtr);

    VDB_ASSERT(grid);
    VDB_ASSERT(grid->tree().activeVoxelCount() > openvdb::Index64(0));

    std::stringstream ss;
    grid->tree().print(ss);
    VDB_ASSERT(ss.str().length() > size_t(0));

    return 0;
}

#ifdef HOUDINI
} // namespace houdini
#endif
