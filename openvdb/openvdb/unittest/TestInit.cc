// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "gtest/gtest.h"
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>


class TestInit: public ::testing::Test
{
};


TEST_F(TestInit, test)
{
    using namespace openvdb;

    initialize();

    // data types
    EXPECT_TRUE(DoubleMetadata::isRegisteredType());
    EXPECT_TRUE(FloatMetadata::isRegisteredType());
    EXPECT_TRUE(Int32Metadata::isRegisteredType());
    EXPECT_TRUE(Int64Metadata::isRegisteredType());
    EXPECT_TRUE(StringMetadata::isRegisteredType());
    EXPECT_TRUE(Vec2IMetadata::isRegisteredType());
    EXPECT_TRUE(Vec2SMetadata::isRegisteredType());
    EXPECT_TRUE(Vec2DMetadata::isRegisteredType());
    EXPECT_TRUE(Vec3IMetadata::isRegisteredType());
    EXPECT_TRUE(Vec3SMetadata::isRegisteredType());
    EXPECT_TRUE(Vec3DMetadata::isRegisteredType());

    // map types
    EXPECT_TRUE(math::AffineMap::isRegistered());
    EXPECT_TRUE(math::UnitaryMap::isRegistered());
    EXPECT_TRUE(math::ScaleMap::isRegistered());
    EXPECT_TRUE(math::TranslationMap::isRegistered());
    EXPECT_TRUE(math::ScaleTranslateMap::isRegistered());
    EXPECT_TRUE(math::NonlinearFrustumMap::isRegistered());

    // grid types
    EXPECT_TRUE(BoolGrid::isRegistered());
    EXPECT_TRUE(FloatGrid::isRegistered());
    EXPECT_TRUE(DoubleGrid::isRegistered());
    EXPECT_TRUE(Int32Grid::isRegistered());
    EXPECT_TRUE(Int64Grid::isRegistered());
    EXPECT_TRUE(StringGrid::isRegistered());
    EXPECT_TRUE(Vec3IGrid::isRegistered());
    EXPECT_TRUE(Vec3SGrid::isRegistered());
    EXPECT_TRUE(Vec3DGrid::isRegistered());

    uninitialize();

    EXPECT_TRUE(!DoubleMetadata::isRegisteredType());
    EXPECT_TRUE(!FloatMetadata::isRegisteredType());
    EXPECT_TRUE(!Int32Metadata::isRegisteredType());
    EXPECT_TRUE(!Int64Metadata::isRegisteredType());
    EXPECT_TRUE(!StringMetadata::isRegisteredType());
    EXPECT_TRUE(!Vec2IMetadata::isRegisteredType());
    EXPECT_TRUE(!Vec2SMetadata::isRegisteredType());
    EXPECT_TRUE(!Vec2DMetadata::isRegisteredType());
    EXPECT_TRUE(!Vec3IMetadata::isRegisteredType());
    EXPECT_TRUE(!Vec3SMetadata::isRegisteredType());
    EXPECT_TRUE(!Vec3DMetadata::isRegisteredType());

    EXPECT_TRUE(!math::AffineMap::isRegistered());
    EXPECT_TRUE(!math::UnitaryMap::isRegistered());
    EXPECT_TRUE(!math::ScaleMap::isRegistered());
    EXPECT_TRUE(!math::TranslationMap::isRegistered());
    EXPECT_TRUE(!math::ScaleTranslateMap::isRegistered());
    EXPECT_TRUE(!math::NonlinearFrustumMap::isRegistered());

    EXPECT_TRUE(!BoolGrid::isRegistered());
    EXPECT_TRUE(!FloatGrid::isRegistered());
    EXPECT_TRUE(!DoubleGrid::isRegistered());
    EXPECT_TRUE(!Int32Grid::isRegistered());
    EXPECT_TRUE(!Int64Grid::isRegistered());
    EXPECT_TRUE(!StringGrid::isRegistered());
    EXPECT_TRUE(!Vec3IGrid::isRegistered());
    EXPECT_TRUE(!Vec3SGrid::isRegistered());
    EXPECT_TRUE(!Vec3DGrid::isRegistered());
}
