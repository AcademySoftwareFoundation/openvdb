// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <gtest/gtest.h>
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    EXPECT_NEAR((expected), (actual), /*tolerance=*/0.0);

namespace {
const int GRID_DIM = 10;
}


class TestDivergence: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


TEST_F(TestDivergence, testDivergenceTool)
{
    using namespace openvdb;

    VectorGrid::Ptr inGrid = VectorGrid::create();
    VectorTree& inTree = inGrid->tree();
    EXPECT_TRUE(inTree.empty());

    int dim = GRID_DIM;
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                inTree.setValue(Coord(x,y,z),
                    VectorTree::ValueType(float(x), float(y), 0.f));
            }
        }
    }

    EXPECT_TRUE(!inTree.empty());
    EXPECT_EQ(math::Pow3(2*dim), int(inTree.activeVoxelCount()));

    FloatGrid::Ptr divGrid = tools::divergence(*inGrid);
    EXPECT_EQ(math::Pow3(2*dim), int(divGrid->activeVoxelCount()));

    FloatGrid::ConstAccessor accessor = divGrid->getConstAccessor();
    --dim;//ignore boundary divergence
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                Coord xyz(x,y,z);
                VectorTree::ValueType v = inTree.getValue(xyz);
                ASSERT_DOUBLES_EXACTLY_EQUAL(x, v[0]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(y, v[1]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(0, v[2]);

                const float d = accessor.getValue(xyz);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);
            }
        }
    }
}



TEST_F(TestDivergence, testDivergenceMaskedTool)
{
    using namespace openvdb;

    VectorGrid::Ptr inGrid = VectorGrid::create();
    VectorTree& inTree = inGrid->tree();
    EXPECT_TRUE(inTree.empty());

    int dim = GRID_DIM;
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                inTree.setValue(Coord(x,y,z),
                    VectorTree::ValueType(float(x), float(y), 0.f));
            }
        }
    }

    EXPECT_TRUE(!inTree.empty());
    EXPECT_EQ(math::Pow3(2*dim), int(inTree.activeVoxelCount()));

    /// maked region
    openvdb::CoordBBox maskBBox(openvdb::Coord(0), openvdb::Coord(dim));
    BoolGrid::Ptr maskGrid = BoolGrid::create(false);
    maskGrid->fill(maskBBox, true /*value*/, true /*activate*/);

    FloatGrid::Ptr divGrid = tools::divergence(*inGrid, *maskGrid);
    EXPECT_EQ(math::Pow3(dim), int(divGrid->activeVoxelCount()));

    FloatGrid::ConstAccessor accessor = divGrid->getConstAccessor();
    --dim;//ignore boundary divergence
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                Coord xyz(x,y,z);

                VectorTree::ValueType v = inTree.getValue(xyz);
                ASSERT_DOUBLES_EXACTLY_EQUAL(x, v[0]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(y, v[1]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(0, v[2]);

                const float d = accessor.getValue(xyz);
                if (maskBBox.isInside(xyz)) {
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);
                } else {
                    ASSERT_DOUBLES_EXACTLY_EQUAL(0, d);
                }
            }
        }
    }
}


TEST_F(TestDivergence, testStaggeredDivergence)
{
    // This test is slightly different than the one above for sanity
    // checking purposes.

    using namespace openvdb;

    VectorGrid::Ptr inGrid = VectorGrid::create();
    inGrid->setGridClass( GRID_STAGGERED );
    VectorTree& inTree = inGrid->tree();
    EXPECT_TRUE(inTree.empty());

    int dim = GRID_DIM;
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                inTree.setValue(Coord(x,y,z),
                    VectorTree::ValueType(float(x), float(y), float(z)));
            }
        }
    }

    EXPECT_TRUE(!inTree.empty());
    EXPECT_EQ(math::Pow3(2*dim), int(inTree.activeVoxelCount()));

    FloatGrid::Ptr divGrid = tools::divergence(*inGrid);
    EXPECT_EQ(math::Pow3(2*dim), int(divGrid->activeVoxelCount()));

    FloatGrid::ConstAccessor accessor = divGrid->getConstAccessor();
    --dim;//ignore boundary divergence
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                Coord xyz(x,y,z);
                VectorTree::ValueType v = inTree.getValue(xyz);
                ASSERT_DOUBLES_EXACTLY_EQUAL(x, v[0]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(y, v[1]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(z, v[2]);

                const float d = accessor.getValue(xyz);
                ASSERT_DOUBLES_EXACTLY_EQUAL(3, d);
            }
        }
    }
}


TEST_F(TestDivergence, testISDivergence)
{
    using namespace openvdb;

    typedef VectorGrid::ConstAccessor Accessor;

    VectorGrid::Ptr inGrid = VectorGrid::create();
    VectorTree& inTree = inGrid->tree();
    EXPECT_TRUE(inTree.empty());

    int dim = GRID_DIM;
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                inTree.setValue(Coord(x,y,z),
                    VectorTree::ValueType(float(x), float(y), 0.f));
            }
        }
    }

    Accessor inAccessor = inGrid->getConstAccessor();
    EXPECT_TRUE(!inTree.empty());
    EXPECT_EQ(math::Pow3(2*dim), int(inTree.activeVoxelCount()));

    --dim;//ignore boundary divergence
    // test index space divergence
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                Coord xyz(x,y,z);
                VectorTree::ValueType v = inTree.getValue(xyz);

                ASSERT_DOUBLES_EXACTLY_EQUAL(x, v[0]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(y, v[1]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(0, v[2]);
                float d;
                d = math::ISDivergence<math::CD_2ND>::result(inAccessor, xyz);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                d = math::ISDivergence<math::BD_1ST>::result(inAccessor, xyz);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                d = math::ISDivergence<math::FD_1ST>::result(inAccessor, xyz);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);
            }
        }
    }

    --dim;//ignore boundary divergence
    // test index space divergence
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                Coord xyz(x,y,z);
                VectorTree::ValueType v = inTree.getValue(xyz);

                ASSERT_DOUBLES_EXACTLY_EQUAL(x, v[0]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(y, v[1]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(0, v[2]);
                float d;
                d = math::ISDivergence<math::CD_4TH>::result(inAccessor, xyz);

                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);
                d = math::ISDivergence<math::FD_2ND>::result(inAccessor, xyz);

                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);
                d = math::ISDivergence<math::BD_2ND>::result(inAccessor, xyz);

                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);
            }
        }
    }

    --dim;//ignore boundary divergence
    // test index space divergence
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                Coord xyz(x,y,z);
                VectorTree::ValueType v = inTree.getValue(xyz);

                ASSERT_DOUBLES_EXACTLY_EQUAL(x, v[0]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(y, v[1]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(0, v[2]);
                float d;
                d = math::ISDivergence<math::CD_6TH>::result(inAccessor, xyz);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                d = math::ISDivergence<math::FD_3RD>::result(inAccessor, xyz);
                EXPECT_NEAR(2, d, /*tolerance=*/0.00001);

                d = math::ISDivergence<math::BD_3RD>::result(inAccessor, xyz);
                EXPECT_NEAR(2, d, /*tolerance=*/0.00001);
            }
        }
    }
}


TEST_F(TestDivergence, testISDivergenceStencil)
{
    using namespace openvdb;

    VectorGrid::Ptr inGrid = VectorGrid::create();
    VectorTree& inTree = inGrid->tree();
    EXPECT_TRUE(inTree.empty());

    int dim = GRID_DIM;
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                inTree.setValue(Coord(x,y,z),
                    VectorTree::ValueType(float(x), float(y), 0.f));
            }
        }
    }

    EXPECT_TRUE(!inTree.empty());
    EXPECT_EQ(math::Pow3(2*dim), int(inTree.activeVoxelCount()));
    math::SevenPointStencil<VectorGrid> sevenpt(*inGrid);
    math::ThirteenPointStencil<VectorGrid> thirteenpt(*inGrid);
    math::NineteenPointStencil<VectorGrid> nineteenpt(*inGrid);

    --dim;//ignore boundary divergence
    // test index space divergence
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                Coord xyz(x,y,z);
                VectorTree::ValueType v = inTree.getValue(xyz);

                ASSERT_DOUBLES_EXACTLY_EQUAL(x, v[0]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(y, v[1]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(0, v[2]);
                sevenpt.moveTo(xyz);
                float d;
                d = math::ISDivergence<math::CD_2ND>::result(sevenpt);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                d = math::ISDivergence<math::BD_1ST>::result(sevenpt);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                d = math::ISDivergence<math::FD_1ST>::result(sevenpt);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);
            }
        }
    }

    --dim;//ignore boundary divergence
    // test index space divergence
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                Coord xyz(x,y,z);
                VectorTree::ValueType v = inTree.getValue(xyz);

                ASSERT_DOUBLES_EXACTLY_EQUAL(x, v[0]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(y, v[1]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(0, v[2]);
                thirteenpt.moveTo(xyz);
                float d;
                d = math::ISDivergence<math::CD_4TH>::result(thirteenpt);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                d = math::ISDivergence<math::FD_2ND>::result(thirteenpt);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                d = math::ISDivergence<math::BD_2ND>::result(thirteenpt);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);
            }
        }
    }

    --dim;//ignore boundary divergence
    // test index space divergence
    for (int x = -dim; x<dim; ++x) {
        for (int y = -dim; y<dim; ++y) {
            for (int z = -dim; z<dim; ++z) {
                Coord xyz(x,y,z);
                VectorTree::ValueType v = inTree.getValue(xyz);

                ASSERT_DOUBLES_EXACTLY_EQUAL(x, v[0]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(y, v[1]);
                ASSERT_DOUBLES_EXACTLY_EQUAL(0, v[2]);
                nineteenpt.moveTo(xyz);
                float d;
                d = math::ISDivergence<math::CD_6TH>::result(nineteenpt);
                ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                d = math::ISDivergence<math::FD_3RD>::result(nineteenpt);
                EXPECT_NEAR(2, d, /*tolerance=*/0.00001);

                d = math::ISDivergence<math::BD_3RD>::result(nineteenpt);
                EXPECT_NEAR(2, d, /*tolerance=*/0.00001);
            }
        }
    }
}


TEST_F(TestDivergence, testWSDivergence)
{
    using namespace openvdb;

    typedef VectorGrid::ConstAccessor Accessor;

    { // non-unit voxel size
        double voxel_size = 0.5;
        VectorGrid::Ptr inGrid = VectorGrid::create();
        inGrid->setTransform(math::Transform::createLinearTransform(voxel_size));

        VectorTree& inTree = inGrid->tree();
        EXPECT_TRUE(inTree.empty());

        int dim = GRID_DIM;
        for (int x = -dim; x<dim; ++x) {
            for (int y = -dim; y<dim; ++y) {
                for (int z = -dim; z<dim; ++z) {
                    Vec3d location = inGrid->indexToWorld(Vec3d(x,y,z));
                    inTree.setValue(Coord(x,y,z),
                        VectorTree::ValueType(float(location.x()), float(location.y()), 0.f));
                }
            }
        }

        Accessor inAccessor = inGrid->getConstAccessor();
        EXPECT_TRUE(!inTree.empty());
        EXPECT_EQ(math::Pow3(2*dim), int(inTree.activeVoxelCount()));

        --dim;//ignore boundary divergence

        // test with a map
            // test with a map
        math::AffineMap map(voxel_size*math::Mat3d::identity());
        math::UniformScaleMap uniform_map(voxel_size);
        math::UniformScaleTranslateMap uniform_translate_map(voxel_size, Vec3d(0,0,0));

        for (int x = -dim; x<dim; ++x) {
            for (int y = -dim; y<dim; ++y) {
                for (int z = -dim; z<dim; ++z) {
                    openvdb::Coord xyz(x,y,z);
                    //openvdb::VectorTree::ValueType v = inTree.getValue(xyz);
                    //std::cout << "vec(" << xyz << ")=" << v << std::endl;

                    float d;
                    d = math::Divergence<math::AffineMap, math::CD_2ND>::result(
                        map, inAccessor, xyz);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::AffineMap, math::BD_1ST>::result(
                        map, inAccessor, xyz);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::AffineMap, math::FD_1ST>::result(
                        map, inAccessor, xyz);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleMap, math::CD_2ND>::result(
                        uniform_map, inAccessor, xyz);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleMap, math::BD_1ST>::result(
                        uniform_map, inAccessor, xyz);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleMap, math::FD_1ST>::result(
                        uniform_map, inAccessor, xyz);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleTranslateMap, math::CD_2ND>::result(
                        uniform_translate_map, inAccessor, xyz);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleTranslateMap, math::BD_1ST>::result(
                        uniform_translate_map, inAccessor, xyz);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleTranslateMap, math::FD_1ST>::result(
                        uniform_translate_map, inAccessor, xyz);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);
                }
            }
        }
    }

    { // non-uniform scaling and rotation
        Vec3d voxel_sizes(0.25, 0.45, 0.75);
        VectorGrid::Ptr inGrid = VectorGrid::create();
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        // apply rotation
        math::MapBase::Ptr rotated_map = base_map->preRotate(1.5, math::X_AXIS);
        inGrid->setTransform(math::Transform::Ptr(new math::Transform(rotated_map)));

        VectorTree& inTree = inGrid->tree();
        EXPECT_TRUE(inTree.empty());

        int dim = GRID_DIM;
        for (int x = -dim; x<dim; ++x) {
            for (int y = -dim; y<dim; ++y) {
                for (int z = -dim; z<dim; ++z) {
                    Vec3d location = inGrid->indexToWorld(Vec3d(x,y,z));
                    inTree.setValue(Coord(x,y,z),
                        VectorTree::ValueType(float(location.x()), float(location.y()), 0.f));
                }
            }
        }

        Accessor inAccessor = inGrid->getConstAccessor();
        EXPECT_TRUE(!inTree.empty());
        EXPECT_EQ(math::Pow3(2*dim), int(inTree.activeVoxelCount()));

        --dim;//ignore boundary divergence

        // test with a map
        math::AffineMap::ConstPtr map = inGrid->transform().map<math::AffineMap>();

        for (int x = -dim; x<dim; ++x) {
            for (int y = -dim; y<dim; ++y) {
                for (int z = -dim; z<dim; ++z) {
                    openvdb::Coord xyz(x,y,z);
                    //openvdb::VectorTree::ValueType v = inTree.getValue(xyz);
                    //std::cout << "vec(" << xyz << ")=" << v << std::endl;

                    float d;
                    d = math::Divergence<math::AffineMap, math::CD_2ND>::result(
                        *map, inAccessor, xyz);
                    EXPECT_NEAR(2.0, d, 0.01);

                    d = math::Divergence<math::AffineMap, math::BD_1ST>::result(
                        *map, inAccessor, xyz);
                    EXPECT_NEAR(2.0, d, 0.01);

                    d = math::Divergence<math::AffineMap, math::FD_1ST>::result(
                        *map, inAccessor, xyz);
                    EXPECT_NEAR(2.0, d, 0.01);
                }
            }
        }
    }
}


TEST_F(TestDivergence, testWSDivergenceStencil)
{
    using namespace openvdb;

    { // non-unit voxel size
        double voxel_size = 0.5;
        VectorGrid::Ptr inGrid = VectorGrid::create();
        inGrid->setTransform(math::Transform::createLinearTransform(voxel_size));

        VectorTree& inTree = inGrid->tree();
        EXPECT_TRUE(inTree.empty());

        int dim = GRID_DIM;
        for (int x = -dim; x<dim; ++x) {
            for (int y = -dim; y<dim; ++y) {
                for (int z = -dim; z<dim; ++z) {
                    Vec3d location = inGrid->indexToWorld(Vec3d(x,y,z));
                    inTree.setValue(Coord(x,y,z),
                        VectorTree::ValueType(float(location.x()), float(location.y()), 0.f));
                }
            }
        }

        EXPECT_TRUE(!inTree.empty());
        EXPECT_EQ(math::Pow3(2*dim), int(inTree.activeVoxelCount()));

        --dim;//ignore boundary divergence

        // test with a map
        math::AffineMap map(voxel_size*math::Mat3d::identity());
        math::UniformScaleMap uniform_map(voxel_size);
        math::UniformScaleTranslateMap uniform_translate_map(voxel_size, Vec3d(0,0,0));

        math::SevenPointStencil<VectorGrid> sevenpt(*inGrid);
        math::SecondOrderDenseStencil<VectorGrid> dense_2ndOrder(*inGrid);

        for (int x = -dim; x<dim; ++x) {
            for (int y = -dim; y<dim; ++y) {
                for (int z = -dim; z<dim; ++z) {
                    openvdb::Coord xyz(x,y,z);
                    //openvdb::VectorTree::ValueType v = inTree.getValue(xyz);
                    //std::cout << "vec(" << xyz << ")=" << v << std::endl;
                    float d;

                    sevenpt.moveTo(xyz);
                    dense_2ndOrder.moveTo(xyz);

                    d = math::Divergence<math::AffineMap, math::CD_2ND>::result(
                        map, dense_2ndOrder);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::AffineMap, math::BD_1ST>::result(
                        map, dense_2ndOrder);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::AffineMap, math::FD_1ST>::result(
                        map, dense_2ndOrder);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleMap, math::CD_2ND>::result(
                        uniform_map, sevenpt);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleMap, math::BD_1ST>::result(
                        uniform_map, sevenpt);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleMap, math::FD_1ST>::result(
                        uniform_map, sevenpt);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleTranslateMap, math::CD_2ND>::result(
                        uniform_translate_map, sevenpt);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleTranslateMap, math::BD_1ST>::result(
                        uniform_translate_map, sevenpt);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);

                    d = math::Divergence<math::UniformScaleTranslateMap, math::FD_1ST>::result(
                        uniform_translate_map, sevenpt);
                    ASSERT_DOUBLES_EXACTLY_EQUAL(2, d);
                }
            }
        }
    }

    { // non-uniform scaling and rotation
        Vec3d voxel_sizes(0.25, 0.45, 0.75);
        VectorGrid::Ptr inGrid = VectorGrid::create();
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        // apply rotation
        math::MapBase::Ptr rotated_map = base_map->preRotate(1.5, math::X_AXIS);
        inGrid->setTransform(math::Transform::Ptr(new math::Transform(rotated_map)));

        VectorTree& inTree = inGrid->tree();
        EXPECT_TRUE(inTree.empty());

        int dim = GRID_DIM;
        for (int x = -dim; x<dim; ++x) {
            for (int y = -dim; y<dim; ++y) {
                for (int z = -dim; z<dim; ++z) {
                    Vec3d location = inGrid->indexToWorld(Vec3d(x,y,z));
                    inTree.setValue(Coord(x,y,z),
                        VectorTree::ValueType(float(location.x()), float(location.y()), 0.f));
                }
            }
        }

        //Accessor inAccessor = inGrid->getConstAccessor();
        EXPECT_TRUE(!inTree.empty());
        EXPECT_EQ(math::Pow3(2*dim), int(inTree.activeVoxelCount()));

        --dim;//ignore boundary divergence

        // test with a map
        math::AffineMap::ConstPtr map = inGrid->transform().map<math::AffineMap>();
        math::SecondOrderDenseStencil<VectorGrid> dense_2ndOrder(*inGrid);

        for (int x = -dim; x<dim; ++x) {
            for (int y = -dim; y<dim; ++y) {
                for (int z = -dim; z<dim; ++z) {
                    openvdb::Coord xyz(x,y,z);
                    dense_2ndOrder.moveTo(xyz);

                    float d;
                    d = math::Divergence<math::AffineMap, math::CD_2ND>::result(
                        *map, dense_2ndOrder);
                    EXPECT_NEAR(2.0, d, 0.01);

                    d = math::Divergence<math::AffineMap, math::BD_1ST>::result(
                        *map, dense_2ndOrder);
                    EXPECT_NEAR(2.0, d, 0.01);

                    d = math::Divergence<math::AffineMap, math::FD_1ST>::result(
                        *map, dense_2ndOrder);
                    EXPECT_NEAR(2.0, d, 0.01);
                }
            }
        }
    }
}
