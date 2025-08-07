// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb_ax/compiler/Compiler.h>

#include <openvdb/openvdb.h>

#include <gtest/gtest.h>

/// @note  Global unit test flag enabled with -g which symbolises the integration
///        tests to auto-generate their AX tests. Any previous tests will be
///        overwritten.
int sGenerateAX = false;


namespace {

} // anonymous namespace

template <typename T>
static inline void registerType()
{
    if (!openvdb::points::TypedAttributeArray<T>::isRegistered())
        openvdb::points::TypedAttributeArray<T>::registerType();
}

int
main(int argc, char *argv[])
{
    openvdb::initialize();
    openvdb::ax::initialize();
    openvdb::logging::initialize(argc, argv);

    // Also intialize Vec2/4 point attributes

    registerType<openvdb::math::Vec2<int32_t>>();
    registerType<openvdb::math::Vec2<float>>();
    registerType<openvdb::math::Vec2<double>>();
    registerType<openvdb::math::Vec4<int32_t>>();
    registerType<openvdb::math::Vec4<float>>();
    registerType<openvdb::math::Vec4<double>>();

    ::testing::InitGoogleTest(&argc, argv);
    auto gtest_result = RUN_ALL_TESTS();

    openvdb::ax::uninitialize();
    openvdb::uninitialize();

    return gtest_result;
}

