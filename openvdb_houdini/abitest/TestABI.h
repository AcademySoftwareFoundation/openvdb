// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <stdexcept>

#include <openvdb/tools/LevelSetSphere.h>

// throw an exception if the condition is false
inline void VDB_ASSERT(const bool condition, const std::string& file, const int line)
{
    if (!condition) {
        throw std::runtime_error("Assertion Fail in file " + file + " on line " + std::to_string(line));
    }
}

#define VDB_ASSERT(condition) VDB_ASSERT(condition, __FILE__, __LINE__)

////////////////////////////////////////

// When including this header, the following macros need to have been #defined:
//
// #define ADD_DECLARE_SUFFIX(name) name ## TEST
// #define ADD_DEFINE_SUFFIX(name) name ## MAIN
//
// This would translate the following:
//
// ADD_DECLARE_SUFFIX(exampleMethod_) => exampleMethod_TEST
// ADD_DEFINE_SUFFIX(exampleMethod_) => exampleMethod_MAIN
//
// TestABI.cc declares methods ending TEST and defines methods ending MAIN
// main.cc declares methods ending MAIN and defines methods ending TEST

////////////////////////////////////////

// Version and Namespace Methods

extern
const char* ADD_DECLARE_SUFFIX(getABI_) (void);

extern
const char* ADD_DEFINE_SUFFIX(getABI_) (void)
{
    return OPENVDB_PREPROC_STRINGIFY(OPENVDB_ABI_VERSION_NUMBER);
}

extern
const char* ADD_DECLARE_SUFFIX(getNamespace_) (void);

extern
const char* ADD_DEFINE_SUFFIX(getNamespace_) (void)
{
    return OPENVDB_PREPROC_STRINGIFY(OPENVDB_VERSION_NAME);
}

////////////////////////////////////////

// Grid Methods

extern
void* ADD_DECLARE_SUFFIX(createGrid_) (void);

extern
void* ADD_DEFINE_SUFFIX(createGrid_) (void)
{
    openvdb::FloatGrid::Ptr grid =
        openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
            /*radius=*/1.0f, /*center=*/openvdb::Vec3f(0.0f), /*voxelSize=*/0.1f);

    return new openvdb::FloatGrid(*grid);
}

extern
void ADD_DECLARE_SUFFIX(cleanupGrid_) (void*);

extern
void ADD_DEFINE_SUFFIX(cleanupGrid_) (void* gridPtr)
{
    openvdb::FloatGrid* grid =
        static_cast<openvdb::FloatGrid*>(gridPtr);
    delete grid;
}

extern
int ADD_DECLARE_SUFFIX(validateGrid_) (void*);

extern
int ADD_DEFINE_SUFFIX(validateGrid_) (void* gridPtr)
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
