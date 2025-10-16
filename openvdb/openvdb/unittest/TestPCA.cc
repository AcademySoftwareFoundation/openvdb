// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/points/PrincipalComponentAnalysis.h>
#include <openvdb/points/PointRasterizeSDF.h>
#include "PointBuilder.h"

#include <gtest/gtest.h>

using namespace openvdb;

class TestPCA: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPCA


TEST_F(TestPCA, testPCA)
{
    const auto CheckPCAAttributes = [](
        const points::PointDataTree::LeafNodeType& leaf,
        const points::PcaAttributes& a,
        const size_t line)
    {
        const auto& desc = leaf.attributeSet().descriptor();
        ASSERT_EQ(NamePair((typeNameAsString<math::Vec3<float>>()), points::NullCodec::name()), desc.type(desc.find(a.stretch))) << "line: "<< line;
        ASSERT_EQ(NamePair((typeNameAsString<math::Mat3<float>>()), points::NullCodec::name()), desc.type(desc.find(a.xform))) << "line: "<< line;
        ASSERT_EQ(NamePair((typeNameAsString<math::Vec3<double>>()), points::NullCodec::name()), desc.type(desc.find(a.positionWS))) << "line: "<< line;
        ASSERT_TRUE(desc.hasGroup(a.ellipses)) << "line: "<< line;
    };

    const auto CheckPCAAttributeValues = [&](
        const std::vector<math::Vec3<float>>& p,
        const std::vector<math::Vec3<float>>& stretches,
        const std::vector<math::Mat3<float>>& xforms,
        const std::vector<math::Vec3<double>>& ws,
        const std::vector<bool>& memberships,
        const points::PointDataTree::LeafNodeType& leaf,
        const points::PcaAttributes& a,
        const size_t line,
        const float tolerance = math::Delta<float>::value())
    {
        CheckPCAAttributes(leaf, a, line);

        const auto& desc = leaf.attributeSet().descriptor();
        points::AttributeHandle<math::Vec3<float>> pHandle(leaf.attributeArray(desc.find("P")));
        points::AttributeHandle<math::Vec3<float>> sHandle(leaf.attributeArray(desc.find(a.stretch)));
        points::AttributeHandle<math::Mat3<float>> rHandle(leaf.attributeArray(desc.find(a.xform)));
        points::AttributeHandle<math::Vec3<double>> pwsHandle(leaf.attributeArray(desc.find(a.positionWS)));
        points::GroupHandle gHandle(leaf.groupHandle(a.ellipses));
        EXPECT_EQ(pHandle.size(), p.size());

        for (Index i = 0; i < Index(p.size()); ++i) {

            EXPECT_EQ(pHandle.get(i), p[i]) << "line: "<< line << " index: " << i;

            for (size_t j = 0; j < 3; ++j)
                EXPECT_NEAR(sHandle.get(i)[j], stretches[i][j], tolerance)
                    << "line: "<< line << " index: " << i << " component " << j;

            for (size_t j = 0; j < 9; ++j)
                EXPECT_NEAR(rHandle.get(i).asPointer()[j], xforms[i].asPointer()[j], tolerance)
                    << "line: "<< line << " index: " << i << " component " << j;

            for (size_t j = 0; j < 3; ++j)
                EXPECT_NEAR(pwsHandle.get(i)[j], ws[i][j], tolerance)
                    << "line: "<< line << " index: " << i << " component " << j;

            EXPECT_EQ(gHandle.get(i), memberships[i]) << "line: "<< line << " index: " << i;
        }
    };

    // test no points
    {
        auto points = PointBuilder({}).voxelsize(0.1).get();
        points::PcaSettings s;
        points::PcaAttributes a;
        points::pca(*points, s, a);
        EXPECT_TRUE(points->empty());
    }

    // test single point
    {
        // test offset from zero to make sure smoothing does nothing
        auto points = PointBuilder({Vec3f(1.0f,2.0f,3.0f)}).voxelsize(0.1).get();
        points::PcaSettings s;
        points::PcaAttributes a;

        points::pca(*points, s, a);
        EXPECT_EQ(Index(1), points->tree().leafCount());

        ASSERT_TRUE(points->tree().cbeginLeaf());
        const auto& leaf = *(points->tree().cbeginLeaf());
        const auto& desc = leaf.attributeSet().descriptor();
        EXPECT_EQ(leaf.pointCount(), Index64(1));
        EXPECT_TRUE(leaf.hasAttribute("P"));
        EXPECT_EQ(desc.find("P"), size_t(0));
        EXPECT_EQ(desc.size(), size_t(5));

        CheckPCAAttributeValues(
                {Vec3f(0.0f)}, // position is at the center of a voxel
                {Vec3f(1.0f)},
                {Mat3s::identity()},
                {Vec3d(1.0,2.0,3.0)},
                {false},
            leaf, a, __LINE__);
    }

    // test attribute names
    {
        const auto pos = getBoxPoints();
        auto points = PointBuilder(pos).voxelsize(0.1).get();

        points::PcaAttributes a;
        points::PcaSettings s;
        a.stretch = "test1";
        a.xform = "test2";
        a.positionWS = "test3";
        a.ellipses = "test4";

        points::pca(*points, s, a);
        EXPECT_EQ(Index(8), points->tree().leafCount());
        ASSERT_TRUE(points->tree().cbeginLeaf());
        Index count = 0 ;
        for (auto iter = points->tree().cbeginLeaf(); iter; ++iter, ++count)
        {
            const auto& leaf = *iter;
            const auto& desc = leaf.attributeSet().descriptor();
            EXPECT_EQ(desc.find("P"), size_t(0));
            EXPECT_TRUE(leaf.hasAttribute(a.stretch));
            EXPECT_TRUE(leaf.hasAttribute(a.xform));
            EXPECT_TRUE(leaf.hasAttribute(a.positionWS));
            EXPECT_TRUE(leaf.attributeSet().descriptor().hasGroup(a.ellipses));
            EXPECT_EQ(desc.size(), size_t(5));
        }

        EXPECT_EQ(Index(8), count);

        // test failure if attributes exist
        EXPECT_THROW(points::pca(*points, s, a), openvdb::KeyError);
    }

    // test two points ontop of each other
    {
        auto points = PointBuilder({
                Vec3f(0.0f, 0.0f, 0.0f),
                Vec3f(0.0f, 0.0f, 0.0f)
            }).voxelsize(0.1).get();

        ASSERT_TRUE(points->tree().cbeginLeaf());
        EXPECT_EQ(Index(1), points->tree().leafCount());

        const auto& leaf = *(points->tree().cbeginLeaf());
        const auto* desc = &(leaf.attributeSet().descriptor());
        EXPECT_EQ(desc->find("P"), size_t(0));

        points::AttributeHandle<math::Vec3<float>> pHandle(leaf.attributeArray(desc->find("P")));
        EXPECT_EQ(pHandle.size(), 2);

        std::vector<Vec3f> posistions;
        for (Index i = 0; i < Index(pHandle.size()); ++i) posistions.emplace_back(pHandle.get(i));
        const std::vector<Vec3d> posistionsWs {
            Vec3d(0.0, 0.0, 0.0), Vec3d(0.0, 0.0, 0.0)
        };

        points::PcaAttributes a;
        points::PcaSettings s;
        s.searchRadius = 0.0001f;
        s.averagePositions = 0.0f; // disable position smoothing
        s.neighbourThreshold = 1; // both elipses should be identicle

        points::pca(*points, s, a);
        ASSERT_TRUE(points->tree().cbeginLeaf());
        ASSERT_EQ(&leaf, &(*points->tree().cbeginLeaf()));
        desc = &(leaf.attributeSet().descriptor());

        EXPECT_EQ(leaf.pointCount(), Index64(2));
        EXPECT_TRUE(leaf.hasAttribute("P"));
        EXPECT_EQ(desc->find("P"), size_t(0));
        EXPECT_EQ(desc->size(), size_t(5));

        CheckPCAAttributeValues(
                posistions,
                {Vec3f(1.0f), Vec3f(1.0f)},
                {Mat3s::identity(), Mat3s::identity()},
                posistionsWs,
                {true, true},
            leaf, a, __LINE__);
    }

    // test two points
    {
        auto points = PointBuilder({
                Vec3f(0.0f, 0.01f, 0.0f),
                Vec3f(0.0f,  0.0f, 0.0f)
            }).voxelsize(0.1).get();

        ASSERT_TRUE(points->tree().cbeginLeaf());
        EXPECT_EQ(Index(1), points->tree().leafCount());

        const auto& leaf = *(points->tree().cbeginLeaf());
        const auto* desc = &(leaf.attributeSet().descriptor());
        EXPECT_EQ(desc->find("P"), size_t(0));

        points::AttributeHandle<math::Vec3<float>> pHandle(leaf.attributeArray(desc->find("P")));
        EXPECT_EQ(pHandle.size(), 2);

        std::vector<Vec3f> posistions;
        for (Index i = 0; i < Index(pHandle.size()); ++i) posistions.emplace_back(pHandle.get(i));
        const std::vector<Vec3d> posistionsWs {
            Vec3d(0.0, 0.01, 0.0), Vec3d(0.0, 0.0, 0.0)
        };

        points::PcaAttributes a;
        points::PcaSettings s;
        s.searchRadius = 1;
        s.averagePositions = 0.0f; // disable position smoothing
        s.neighbourThreshold = 1; // both elipses should be identicle

        points::pca(*points, s, a);
        ASSERT_TRUE(points->tree().cbeginLeaf());
        ASSERT_EQ(&leaf, &(*points->tree().cbeginLeaf()));
        desc = &(leaf.attributeSet().descriptor());

        EXPECT_EQ(leaf.pointCount(), Index64(2));
        EXPECT_TRUE(leaf.hasAttribute("P"));
        EXPECT_EQ(desc->find("P"), size_t(0));
        EXPECT_EQ(desc->size(), size_t(5));

        CheckPCAAttributeValues(
                posistions,
                {Vec3f(2.51984f, 0.62996f, 0.62996f), Vec3f(2.51984f, 0.62996f, 0.62996f)},
                {Mat3s(0,1,0, 1,0,0, 0,0,1), Mat3s(0,1,0, 1,0,0, 0,0,1)},
                posistionsWs,
                {true, true},
            leaf, a, __LINE__);
    }

    // test three coincident points with various settings
    {
        auto points = PointBuilder({
                Vec3f(0.0f, 0.02f, 0.0f),
                Vec3f(0.0f, 0.01f, 0.0f),
                Vec3f(0.0f,  0.0f, 0.0f)
            }).voxelsize(0.1).get();

        ASSERT_TRUE(points->tree().cbeginLeaf());
        EXPECT_EQ(Index(1), points->tree().leafCount());

        const auto& leaf = *(points->tree().cbeginLeaf());
        const auto* desc = &(leaf.attributeSet().descriptor());
        EXPECT_EQ(desc->find("P"), size_t(0));

        points::AttributeHandle<math::Vec3<float>> pHandle(leaf.attributeArray(desc->find("P")));
        EXPECT_EQ(pHandle.size(), 3);

        std::vector<Vec3f> posistions;
        for (Index i = 0; i < Index(pHandle.size()); ++i) posistions.emplace_back(pHandle.get(i));
        const std::vector<Vec3d> posistionsWs {
            Vec3d(0.0, 0.02, 0.0), Vec3d(0.0, 0.01, 0.0), Vec3d(0.0)
        };

        points::PcaAttributes a;
        points::PcaSettings s;
        s.searchRadius = 1.0f;
        s.averagePositions = 0.0f; // disable position smoothing
        s.neighbourThreshold = 4; // more than 3, points should end up as spheres
        s.nonAnisotropicStretch = 2.0f;

        points::pca(*points, s, a);
        ASSERT_TRUE(points->tree().cbeginLeaf());
        ASSERT_EQ(&leaf, &(*points->tree().cbeginLeaf()));
        desc = &(leaf.attributeSet().descriptor());

        EXPECT_EQ(leaf.pointCount(), Index64(3));
        EXPECT_TRUE(leaf.hasAttribute("P"));
        EXPECT_EQ(desc->find("P"), size_t(0));
        EXPECT_EQ(desc->size(), size_t(5));

        CheckPCAAttributeValues(
                posistions,
                {Vec3f(s.nonAnisotropicStretch), Vec3f(s.nonAnisotropicStretch), Vec3f(s.nonAnisotropicStretch)},
                {Mat3s::identity(), Mat3s::identity(), Mat3s::identity()},
                posistionsWs,
                {false, false, false},
            leaf, a, __LINE__);

        // Test points don't get classified if they are out of range

        points::dropAttributes(points->tree(), {a.stretch, a.xform, a.positionWS});
        points::dropGroup(points->tree(), a.ellipses);

        s.searchRadius = 0.001f;
        s.neighbourThreshold = 0;
        s.nonAnisotropicStretch = 1.0f;

        points::pca(*points, s, a);
        ASSERT_TRUE(points->tree().cbeginLeaf());
        ASSERT_EQ(&leaf, &(*points->tree().cbeginLeaf()));
        desc = &(leaf.attributeSet().descriptor());

        EXPECT_EQ(leaf.pointCount(), Index64(3));
        EXPECT_TRUE(leaf.hasAttribute("P"));
        EXPECT_EQ(desc->find("P"), size_t(0));
        EXPECT_EQ(desc->size(), size_t(5));

        CheckPCAAttributeValues(
                posistions,
                {Vec3f(1.0f), Vec3f(1.0f), Vec3f(1.0f)},
                {Mat3s::identity(), Mat3s::identity(), Mat3s::identity()},
                posistionsWs,
                {false, false, false},
            leaf, a, __LINE__);

        // Test only the center point is classified as an ellips

        points::dropAttributes(points->tree(), {a.stretch, a.xform, a.positionWS});
        points::dropGroup(points->tree(), a.ellipses);

        // each point is 0.01 distance away from the next, so setting to 0.01
        // should mean only the middle point gets classified
        s.searchRadius = std::nextafter(0.01f, 1.0f);
        s.neighbourThreshold = 3; // only center point
        s.averagePositions = 0.0f; // disable position smoothing
        s.nonAnisotropicStretch = 1.3f;

        points::pca(*points, s, a);
        ASSERT_TRUE(points->tree().cbeginLeaf());
        ASSERT_EQ(&leaf, &(*points->tree().cbeginLeaf()));
        desc = &(leaf.attributeSet().descriptor());

        EXPECT_EQ(leaf.pointCount(), Index64(3));
        EXPECT_TRUE(leaf.hasAttribute("P"));
        EXPECT_EQ(desc->find("P"), size_t(0));
        EXPECT_EQ(desc->size(), size_t(5));

        CheckPCAAttributeValues(
                posistions,
                {Vec3f(s.nonAnisotropicStretch), Vec3f(2.51984f, 0.62996f, 0.62996f), Vec3f(s.nonAnisotropicStretch)},
                {Mat3s::identity(), Mat3s(0,1,0, 1,0,0, 0,0,1), Mat3s::identity()},
                posistionsWs,
                {false, true, false},
            leaf, a, __LINE__);

        // Now test they get classified as ellipses and re-compute the ellips
        // transformations to check they are espected. We should end up with
        // ellips' with no rotation and simply stretched along their principal
        // axis (in this case Y) and squashed in the others

        points::dropAttributes(points->tree(), {a.stretch, a.xform, a.positionWS});
        points::dropGroup(points->tree(), a.ellipses);

        // each point is 0.01 distance away from the next, so make sure
        // they all find each other i.e. min dist as 0.02
        s.searchRadius = std::nextafter(0.02f, 1.0f);
        s.neighbourThreshold = 2; // make sure they get classified
        s.averagePositions = 0.0f; // disable position smoothing
        s.allowedAnisotropyRatio = 0.25f;
        s.nonAnisotropicStretch = 1.0f;

        points::pca(*points, s, a);
        ASSERT_TRUE(points->tree().cbeginLeaf());
        ASSERT_EQ(&leaf, &(*points->tree().cbeginLeaf()));
        desc = &(leaf.attributeSet().descriptor());

        EXPECT_EQ(leaf.pointCount(), Index64(3));
        EXPECT_TRUE(leaf.hasAttribute("P"));
        EXPECT_EQ(desc->find("P"), size_t(0));
        EXPECT_EQ(desc->size(), size_t(5));

        CheckPCAAttributes(leaf, a, __LINE__);

        pHandle = points::AttributeHandle<math::Vec3<float>>(leaf.attributeArray(desc->find("P")));
        points::AttributeHandle<math::Vec3<float>> sHandle(leaf.attributeArray(desc->find(a.stretch)));
        points::AttributeHandle<math::Mat3<float>> rHandle(leaf.attributeArray(desc->find(a.xform)));
        points::AttributeHandle<math::Vec3<double>> pwsHandle(leaf.attributeArray(desc->find(a.positionWS)));
        points::GroupHandle::UniquePtr gHandle(new points::GroupHandle(leaf.groupHandle(a.ellipses)));
        EXPECT_EQ(pHandle.size(), posistions.size());

        Index i = 0;
        for (auto iter = leaf.beginIndexOn(); iter; ++iter, ++i)
        {
            for (size_t j = 0; j < 3; ++j) EXPECT_NEAR(pHandle.get(i)[j],   posistions[i][j],   1e-6f);
            for (size_t j = 0; j < 3; ++j) EXPECT_NEAR(pwsHandle.get(i)[j], posistionsWs[i][j], 1e-6f);
            EXPECT_EQ(gHandle->get(i), true);

            // Now rebuild the expected ellipse transform - should just contain stretches,
            // predominantly along Y
            const math::Mat3<float> rot = rHandle.get(i);
            const math::Vec3<float> stretch = sHandle.get(i);
            const math::Vec3<float> ordered = stretch.sorted();
            EXPECT_NEAR(ordered[0], ordered[1], 1e-6f);
            const float min = ordered[0];
            const float max = ordered[2];

            // max should be greater than min for each point as each point
            // has an identifiable PC axis
            EXPECT_TRUE(max > min);

            // check extents related to allowedAnisotropyRatio value
            EXPECT_NEAR(min/max, s.allowedAnisotropyRatio, 1e-6f);

            const math::Mat3<float> inv = (rot.timesDiagonal(1.0/stretch) * rot.transpose()).inverse();
            EXPECT_NEAR(inv(0,0), min, 1e-6f)  << "index: " << i;
            EXPECT_NEAR(inv(0,1), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(0,2), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(1,0), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(1,1), max, 1e-6f)  << "index: " << i;
            EXPECT_NEAR(inv(1,2), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(2,0), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(2,1), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(2,2), min, 1e-6f)  << "index: " << i;
        }

        EXPECT_EQ(i, 3);

        // Test with greater allows anisotropy (i.e. lower ratio). Should get
        // more stretch and squashes. Compare these to the previous test

        points::dropAttributes(points->tree(), {a.stretch, a.xform, a.positionWS});
        points::dropGroup(points->tree(), a.ellipses);

        s.allowedAnisotropyRatio = 0.0625f; // 4x more than 0.25

        points::pca(*points, s, a);
        ASSERT_TRUE(points->tree().cbeginLeaf());
        ASSERT_EQ(&leaf, &(*points->tree().cbeginLeaf()));
        desc = &(leaf.attributeSet().descriptor());

        EXPECT_EQ(leaf.pointCount(), Index64(3));
        EXPECT_TRUE(leaf.hasAttribute("P"));
        EXPECT_EQ(desc->find("P"), size_t(0));
        EXPECT_EQ(desc->size(), size_t(5));

        CheckPCAAttributes(leaf, a, __LINE__);

        pHandle = points::AttributeHandle<math::Vec3<float>>(leaf.attributeArray(desc->find("P")));
        sHandle = points::AttributeHandle<math::Vec3<float>>(leaf.attributeArray(desc->find(a.stretch)));
        rHandle = points::AttributeHandle<math::Mat3<float>>(leaf.attributeArray(desc->find(a.xform)));
        pwsHandle = points::AttributeHandle<math::Vec3<double>>(leaf.attributeArray(desc->find(a.positionWS)));
        gHandle = points::GroupHandle::UniquePtr(new points::GroupHandle(leaf.groupHandle(a.ellipses)));
        EXPECT_EQ(pHandle.size(), posistions.size());

        i = 0;
        for (auto iter = leaf.beginIndexOn(); iter; ++iter, ++i)
        {
            for (size_t j = 0; j < 3; ++j) EXPECT_NEAR(pHandle.get(i)[j],   posistions[i][j],   1e-6f);
            for (size_t j = 0; j < 3; ++j) EXPECT_NEAR(pwsHandle.get(i)[j], posistionsWs[i][j], 1e-6f);
            EXPECT_EQ(gHandle->get(i), true);

            // Now rebuild the expected ellipse transform - should just contain stretches,
            // predominantly along Y
            const math::Mat3<float> rot = rHandle.get(i);
            const math::Vec3<float> stretch = sHandle.get(i);
            const math::Vec3<float> ordered = stretch.sorted();
            EXPECT_NEAR(ordered[0], ordered[1], 1e-6f);
            const float min = ordered[0];
            const float max = ordered[2];
            // max should be greater than min for each point as each point
            // has an identifiable PC axis
            EXPECT_TRUE(max > min);

            // check extents related to allowedAnisotropyRatio value
            EXPECT_NEAR(min/max, s.allowedAnisotropyRatio, 1e-6f);

            const math::Mat3<float> inv = (rot.timesDiagonal(1.0/stretch) * rot.transpose()).inverse();
            EXPECT_NEAR(inv(0,0), min, 1e-6f)  << "index: " << i;
            EXPECT_NEAR(inv(0,1), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(0,2), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(1,0), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(1,1), max, 1e-6f)  << "index: " << i;
            EXPECT_NEAR(inv(1,2), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(2,0), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(2,1), 0.0f, 1e-6f) << "index: " << i;
            EXPECT_NEAR(inv(2,2), min, 1e-6f)  << "index: " << i;
        }

        EXPECT_EQ(i, 3);
    }
}


TEST_F(TestPCA, testPCAxforms)
{
    // Create three grids:
    //   points1 - stretch and rot
    //   points2 - combined xform
    //   points3 - stretch and quat

    auto points1 = PointBuilder({
            Vec3f(0.0f, 0.02f, 0.0f),
            Vec3f(0.0f, 0.01f, 0.0f),
            Vec3f(0.0f,  0.0f, 0.0f)
        }).voxelsize(0.1).get();
    auto points2 = PointBuilder({
            Vec3f(0.0f, 0.02f, 0.0f),
            Vec3f(0.0f, 0.01f, 0.0f),
            Vec3f(0.0f,  0.0f, 0.0f)
        }).voxelsize(0.1).get();
    auto points3 = PointBuilder({
            Vec3f(0.0f, 0.02f, 0.0f),
            Vec3f(0.0f, 0.01f, 0.0f),
            Vec3f(0.0f,  0.0f, 0.0f)
        }).voxelsize(0.1).get();

    points::PcaSettings s;
    // each point is 0.01 distance away from the next, so make sure
    // they all find each other i.e. min dist as 0.02
    s.searchRadius = std::nextafter(0.02f, 1.0f);
    s.neighbourThreshold = 2; // make sure they get classified
    s.averagePositions = 0.0f; // disable position smoothing
    s.allowedAnisotropyRatio = 0.25f;
    s.nonAnisotropicStretch = 1.0f;

    points::PcaAttributes a;
    a.format = points::PcaAttributes::AttributeOutput::STRETCH_AND_ROTATION_MATRIX;
    points::pca(*points1, s, a);

    a.format = points::PcaAttributes::AttributeOutput::COMBINED_TRANSFORM;
    points::pca(*points2, s, a);

    a.format = points::PcaAttributes::AttributeOutput::STRETCH_AND_QUATERNION;
    points::pca(*points3, s, a);

    // Check attributes of each
    const auto& desc1 = points1->tree().cbeginLeaf()->attributeSet().descriptor();
    EXPECT_TRUE(points1->tree().cbeginLeaf()->hasAttribute("P"));
    EXPECT_TRUE(points1->tree().cbeginLeaf()->hasAttribute("pws"));
    EXPECT_TRUE(points1->tree().cbeginLeaf()->hasAttribute("stretch"));
    EXPECT_TRUE(points1->tree().cbeginLeaf()->hasAttribute("xform"));
    EXPECT_TRUE(points1->tree().cbeginLeaf()->attributeSet().descriptor().hasGroup("ellipsoids"));
    EXPECT_EQ(desc1.size(), 5);
    EXPECT_EQ(desc1.valueType(desc1.find("stretch")), std::string(typeNameAsString<math::Vec3<float>>()));
    EXPECT_EQ(desc1.valueType(desc1.find("xform")),   std::string(typeNameAsString<math::Mat3<float>>()));

    const auto& desc2 = points2->tree().cbeginLeaf()->attributeSet().descriptor();
    EXPECT_TRUE(points2->tree().cbeginLeaf()->hasAttribute("P"));
    EXPECT_TRUE(points2->tree().cbeginLeaf()->hasAttribute("pws"));
    EXPECT_TRUE(points2->tree().cbeginLeaf()->hasAttribute("xform"));
    EXPECT_TRUE(points2->tree().cbeginLeaf()->attributeSet().descriptor().hasGroup("ellipsoids"));
    EXPECT_EQ(desc2.size(), 4);
    EXPECT_EQ(desc2.valueType(desc2.find("xform")), std::string(typeNameAsString<math::Mat3<float>>()));

    const auto& desc3 = points3->tree().cbeginLeaf()->attributeSet().descriptor();
    EXPECT_TRUE(points3->tree().cbeginLeaf()->hasAttribute("P"));
    EXPECT_TRUE(points3->tree().cbeginLeaf()->hasAttribute("pws"));
    EXPECT_TRUE(points3->tree().cbeginLeaf()->hasAttribute("stretch"));
    EXPECT_TRUE(points3->tree().cbeginLeaf()->hasAttribute("xform"));
    EXPECT_TRUE(points3->tree().cbeginLeaf()->attributeSet().descriptor().hasGroup("ellipsoids"));
    EXPECT_EQ(desc3.size(), 5);
    EXPECT_EQ(desc3.valueType(desc3.find("stretch")), std::string(typeNameAsString<math::Vec3<float>>()));
    EXPECT_EQ(desc3.valueType(desc3.find("xform")),   std::string(typeNameAsString<math::Quat<float>>()));

    points::AttributeHandle<math::Vec3<float>> stretch1(points1->tree().cbeginLeaf()->attributeArray(desc1.find("stretch")));
    points::AttributeHandle<math::Vec3<float>> stretch3(points3->tree().cbeginLeaf()->attributeArray(desc3.find("stretch")));
    points::AttributeHandle<math::Mat3<float>> xform1(points1->tree().cbeginLeaf()->attributeArray(desc1.find("xform")));
    points::AttributeHandle<math::Mat3<float>> xform2(points2->tree().cbeginLeaf()->attributeArray(desc2.find("xform")));
    points::AttributeHandle<math::Quat<float>> xform3(points3->tree().cbeginLeaf()->attributeArray(desc3.find("xform")));

    for (Index i = 0; i < 3; ++i)
    {
        // Check rot matrix is represented by quat output
        math::Mat3<float> rot1 = xform1.get(i);
        math::Vec3<float> str1 = stretch1.get(i);
        math::Mat3<float> rot3(xform3.get(i)); // construct from quat

        // Combined xform - decompose and check
        math::Mat3<float> rot2, str2;
        math::polarDecomposition(xform2.get(i), rot2, str2);
        EXPECT_EQ(rot1, rot2);
        EXPECT_EQ(str1, math::Vec3<float>(str2(0,0), str2(1,1), str2(2,2)));

        // Check stretch+rot against stretch+quat
        math::Vec3<float> str3 = stretch3.get(i);
        const bool rot1IsReflection = math::isApproxEqual(rot1.det(), -1.0f);
        if (rot1IsReflection) {
            // flip last column (the same as the internal behaviour of pca)
            rot1(0, 2) *= -1.0f;
            rot1(1, 2) *= -1.0f;
            rot1(2, 2) *= -1.0f;
        }
        EXPECT_EQ(rot1, rot3);
        EXPECT_EQ(str1, str3);
    }
}
