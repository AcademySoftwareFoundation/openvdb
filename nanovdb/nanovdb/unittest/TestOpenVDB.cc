// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <iostream>
#include <cstdlib>
#include <sstream> // for std::stringstream
#include <cstdio>// for FILE
#include <cmath>

#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/NanoToOpenVDB.h>
#include <nanovdb/tools/GridValidator.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/NodeManager.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/tools/GridStats.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/util/Timer.h>
#include <nanovdb/tools/GridBuilder.h>

#if !defined(_MSC_VER) // does not compile in msvc c++ due to zero-sized arrays.
#include <nanovdb/CNanoVDB.h>
#include <nanovdb/math/CSampleFromVoxels.h>
#endif

#include <openvdb/openvdb.h>
#include <openvdb/math/Math.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/PointIndexGrid.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/util/CpuTimer.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include <gtest/gtest.h>

// define the environment variable VDB_DATA_PATH to use models from the web
// e.g. setenv VDB_DATA_PATH /home/kmu/dev/data/vdb
// or   export VDB_DATA_PATH=/Users/ken/dev/data/vdb

// The fixture for testing class.
class TestOpenVDB : public ::testing::Test
{
protected:
    TestOpenVDB() {}

    ~TestOpenVDB() override {}

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override
    {
        openvdb::initialize();
        mStr = new char[256];
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        delete [] mStr;
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    static std::vector<std::string> availableFiles(std::vector<std::string> candidates)
    {
        const char* str = std::getenv("VDB_DATA_PATH");// get environment variable
        const std::string path = str ? str : ".";// defaults path to that of this executable
        std::vector<std::string> available;
        if (candidates.empty()) return available;
        for (auto &model : candidates) {
            const std::string fileName = path + "/" + model + ".vdb";
            if (FILE *file = fopen(fileName.c_str(), "r")) {
                fclose(file);
                available.push_back(fileName);
            }
        }
        return available;
    }

    static std::vector<std::string> availableLevelSetFiles()
    {
        return TestOpenVDB::availableFiles({
            "dragon",// prioritized list
            "armadillo",
            "buddha",
            "bunny",
            "crawler",
            "iss",
            "space",
            "torus_knot_helix",
            "utahteapot"
        });
    }

    static std::vector<std::string> availablePointFiles()
    {
        return TestOpenVDB::availableFiles({
            "boat_points",// prioritized list
            "bunny_points",
            "sphere_points",
            "waterfall_points"
        });
    }

    static std::vector<std::string> availableFogFiles()
    {
        return TestOpenVDB::availableFiles({
            "bunny_cloud",// prioritized list
            "wdas_cloud",
            "fire",
            "smoke",
            "smoke2"
        });
    }

    // ModeType: 0 is Level Set, 1 is FOG volume and 2 is points
    openvdb::FloatGrid::Ptr getSrcGrid(int verbose = 0, int modelType = 0, int modelID = 0)
    {
        openvdb::FloatGrid::Ptr grid;
        std::vector<std::string> fileNames;
        if (modelType == 0) {
            fileNames = TestOpenVDB::availableLevelSetFiles();
        } else if (modelType == 1) {
            fileNames = TestOpenVDB::availableFogFiles();
        } else if (modelType == 2) {
            fileNames = TestOpenVDB::availablePointFiles();
        }
        if (int(fileNames.size()) > modelID) {
            const auto fileName = fileNames[modelID];
            if (verbose > 0)
                mTimer.start("Reading grid from the file \"" + fileName + "\"");
            try {
                openvdb::io::File file(fileName);
                file.open(false); //disable delayed loading
                grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(file.beginName().gridName()));
            } catch(const std::exception& e) {
                std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
            }
        }

        if (!grid) { // create a narrow-band level set of a platonic shape
            const float          radius = 100.0f;
            const openvdb::Vec3f center(0.0f, 0.0f, 0.0f);
            const float          voxelSize = 1.0f, width = 3.0f;
            if (verbose > 0) {
                std::stringstream ss;
                ss << "Generating level set surface with a size of " << radius << " voxel units";
                mTimer.start(ss.str());
            }
            if (modelID >= 1 && modelID <= 5 ) {
                int numFaces[] = {0, 4, 6, 8, 12, 20};
                grid = openvdb::tools::createLevelSetPlatonic<openvdb::FloatGrid>(numFaces[modelID], radius, center, voxelSize, width);
            } else {
                grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center, voxelSize, width);
            }
        }

        EXPECT_TRUE(grid);

        if (verbose > 0)
            mTimer.stop();
        if (verbose > 1)
            grid->print(std::cout, 3);
        return grid;
    }

    nanovdb::io::Codec getCodec() const
    {
#if defined(NANOVDB_USE_BLOSC)
        return nanovdb::io::Codec::BLOSC;
#elif defined(NANOVDB_USE_ZIP)
        return nanovdb::io::Codec::ZIP;
#else
        return nanovdb::io::Codec::NONE;
#endif
    }

    openvdb::util::CpuTimer mTimer;
    char *mStr;
}; // TestOpenVDB

// make -j && ./unittest/testOpenVDB --gtest_break_on_failure --gtest_filter="*getExtrema"
TEST_F(TestOpenVDB, getExtrema)
{
    using wBBoxT = openvdb::math::BBox<openvdb::Vec3d>;
    auto srcGrid = this->getSrcGrid(false, 0, 3);// level set of a bunny if available, else an octahedron
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid, nanovdb::tools::StatsMode::All);
    EXPECT_TRUE(handle);
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
    auto dstAcc = dstGrid->getAccessor();
    auto indexToWorldMap = srcGrid->transform().baseMap();
    const auto a = srcGrid->evalActiveVoxelBoundingBox();
    const auto b = wBBoxT(a.min().asVec3d(),a.max().asVec3d()).applyMap(*indexToWorldMap);
    //std::cerr << "Grid index bbox of all active values: " << a << std::endl;
    //std::cerr << "Grid world bbox of all active values: " << b << std::endl;

    const wBBoxT wBBox(b.min(), 0.5*b.extents()[b.maxExtent()]);
    const wBBoxT iBBox = wBBox.applyInverseMap(*indexToWorldMap);
    //std::cerr << "Query bbox: iBBox = " << iBBox << ", wBBox = " << wBBox << std::endl;

    const nanovdb::CoordBBox bbox(nanovdb::math::Round<nanovdb::Coord>(iBBox.min()),
                                  nanovdb::math::Round<nanovdb::Coord>(iBBox.max()));
    //std::cerr << "Query index bbox = " << bbox << std::endl;

    //nanovdb::NodeManager<nanovdb::FloatGrid> mgr(*dstGrid);
    //std::cerr << "Root child nodes: " << mgr.nodeCount(2) << std::endl;

    //mTimer.start("getExtrema");
    nanovdb::tools::Extrema<float> ext1 = nanovdb::tools::getExtrema(*dstGrid, bbox), ext2;
    //mTimer.restart("naive approach");
    for (auto it = bbox.begin(); it; ++it) ext2.add(dstAcc.getValue(*it));
    //mTimer.stop();
    //std::cerr << "min = " << ext1.min() << ", max = " << ext1.max() << std::endl;
    //std::cerr << "min = " << ext2.min() << ", max = " << ext2.max() << std::endl;
    EXPECT_EQ(ext1.min(), ext2.min());
    EXPECT_EQ(ext1.max(), ext2.max());
}

TEST_F(TestOpenVDB, Basic)
{
    { // openvdb::Vec3::operator<
        const openvdb::Vec3f a(1.0f, 10.0f, 200.0f), b(2.0f, 0.0f, 0.0f);
        EXPECT_TRUE(a < b);// default behavior inherited from openvdb::math::Tuple
        EXPECT_TRUE(a.lengthSqr() > b.lengthSqr());// behavior used in nanovdb::GridStats
    }
}

TEST_F(TestOpenVDB, MapToNano)
{
    {// Coord
        const openvdb::Coord ijk1(1, 2, -4);
        nanovdb::Coord ijk2(-2, 7, 9);
        EXPECT_NE(ijk2, nanovdb::Coord(1, 2, -4));
        ijk2 = ijk1;
        EXPECT_EQ(ijk2, nanovdb::Coord(1, 2, -4));
    }
    {// Vec3f
        constexpr bool test1 = nanovdb::util::is_same<nanovdb::Vec3f, nanovdb::tools::MapToNano<openvdb::Vec3f>::type>::value;
        EXPECT_TRUE(test1);
        constexpr bool test2 = nanovdb::util::is_same<nanovdb::Vec3d, nanovdb::tools::MapToNano<openvdb::Vec3f>::type>::value;
        EXPECT_FALSE(test2);
        const openvdb::Vec3f xyz1(1, 2, -4);
        nanovdb::Vec3f xyz2(-2, 7, 9);
        EXPECT_NE(xyz2, nanovdb::Vec3f(1, 2, -4));
        xyz2 = xyz1;
        EXPECT_EQ(xyz2, nanovdb::Vec3f(1, 2, -4));
    }
    {// Vec4d
        constexpr bool test1 = nanovdb::util::is_same<nanovdb::Vec4d, nanovdb::tools::MapToNano<openvdb::Vec4d>::type>::value;
        EXPECT_TRUE(test1);
        constexpr bool test2 = nanovdb::util::is_same<nanovdb::Vec4f, nanovdb::tools::MapToNano<openvdb::Vec4d>::type>::value;
        EXPECT_FALSE(test2);
        const openvdb::Vec4d xyz1(1, 2, -4, 7);
        nanovdb::Vec4d xyz2(-2, 7, 9, -4);
        EXPECT_NE(xyz2, nanovdb::Vec4d(1, 2, -4, 7));
        xyz2 = xyz1;
        EXPECT_EQ(xyz2, nanovdb::Vec4d(1, 2, -4, 7));
    }
    {// MaskValue
        constexpr bool test1 = nanovdb::util::is_same<nanovdb::ValueMask, nanovdb::tools::MapToNano<openvdb::ValueMask>::type>::value;
        EXPECT_TRUE(test1);
        constexpr bool test2 = nanovdb::util::is_same<nanovdb::Vec3f, nanovdb::tools::MapToNano<openvdb::ValueMask>::type>::value;
        EXPECT_FALSE(test2);
        EXPECT_EQ(sizeof(nanovdb::ValueMask), sizeof(openvdb::ValueMask));
    }
    {// Mask
        openvdb::util::NodeMask<3> mask1;
        nanovdb::Mask<3> mask2, mask3;
        for (int i=0; i<256; ++i) {
            mask1.setOn(i);
            mask2.setOn(i);
        }
        EXPECT_NE(mask2, mask3);
        mask3 = mask2;
        EXPECT_EQ(mask2, mask3);
    }
}

TEST_F(TestOpenVDB, BasicGrid)
{
    using LeafT  = nanovdb::LeafNode<float>;
    using NodeT1 = nanovdb::InternalNode<LeafT>;
    using NodeT2 = nanovdb::InternalNode<NodeT1>;
    using RootT  = nanovdb::RootNode<NodeT2>;
    using TreeT  = nanovdb::Tree<RootT>;
    using GridT  = nanovdb::Grid<TreeT>;
    using CoordT = LeafT::CoordType;

    const std::string name("test name");

    EXPECT_EQ(nanovdb::math::AlignUp<NANOVDB_DATA_ALIGNMENT>(8 + 8 + 2 + 2 + 4 + 8 + nanovdb::GridData::MaxNameSize + 48 + sizeof(nanovdb::Map) + 24 + 4 + 4 + 8 + 4), sizeof(GridT));
    EXPECT_EQ(nanovdb::math::AlignUp<NANOVDB_DATA_ALIGNMENT>(4*8 + 2 * 4 * 3 + 8), sizeof(TreeT));
    EXPECT_EQ(size_t(4*8 + 2 * 4 * 3 + 8), sizeof(TreeT));// should already be 32 byte aligned

    size_t bytes[9];
    bytes[0] = 0;//  buffer/grid begins
    bytes[1] = GridT::memUsage(); //  grid ends
    bytes[2] = TreeT::memUsage(); //  tree ends
    bytes[3] = RootT::memUsage(1); // root node ends
    bytes[4] = NodeT2::memUsage(); // 1 upper internal node
    bytes[5] = NodeT1::memUsage(); // 1 lower internal node
    bytes[6] = LeafT::DataType::memUsage();// 1 leaf node ends
    bytes[7] = 0;// blind meta data
    bytes[8] = 0;// blind data
    for (int i = 2; i < 9; ++i)
        bytes[i] += bytes[i - 1]; // Byte offsets to: tree, root, internal nodes, leafs, total
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[bytes[8]]);

    // init leaf
    LeafT* leaf = reinterpret_cast<LeafT*>(buffer.get() + bytes[5]);
    { // set members of the leaf node
        auto *data = leaf->data();
        data->mValueMask.setOff();
        auto* voxels = data->mValues;
        for (uint32_t i = 0; i < LeafT::voxelCount() / 2; ++i)
            *voxels++ = 0.0f;
        for (uint32_t i = LeafT::voxelCount() / 2; i < LeafT::voxelCount(); ++i) {
            data->mValueMask.setOn(i);
            *voxels++ = 1.0f;
        }
        data->mMinimum = 1.0f;
        data->mMaximum = 1.0f;
    }

    // lower internal node
    NodeT1* node1 = reinterpret_cast<NodeT1*>(buffer.get() + bytes[4]);
    { // set members of the  internal node
        auto  *data = node1->data();
        data->mValueMask.setOff();
        data->mChildMask.setOff();
        data->mChildMask.setOn(0);
        data->setChild(0, leaf);
        for (uint32_t i = 1; i < NodeT1::SIZE / 2; ++i)
            data->mTable[i].value = 0.0f;
        for (uint32_t i = NodeT1::SIZE / 2; i < NodeT1::SIZE; ++i) {
            data->mValueMask.setOn(i);
            data->mTable[i].value = 2.0f;
        }
        data->mMinimum = 1.0f;
        data->mMaximum = 2.0f;
        EXPECT_EQ(leaf, data->getChild(0));
    }

    // upper internal node
    NodeT2* node2 = reinterpret_cast<NodeT2*>(buffer.get() + bytes[3]);
    { // set members of the  internal node
        auto *data = node2->data();
        data->mValueMask.setOff();
        data->mChildMask.setOff();
        data->mChildMask.setOn(0);
        data->setChild(0, node1);
        for (uint32_t i = 1; i < NodeT2::SIZE / 2; ++i)
            data->mTable[i].value = 0.0f;
        for (uint32_t i = NodeT2::SIZE / 2; i < NodeT2::SIZE; ++i) {
            data->mValueMask.setOn(i);
            data->mTable[i].value = 3.0f;
        }
        data->mMinimum = 1.0f;
        data->mMaximum = 3.0f;
        EXPECT_EQ(node1, data->getChild(0));
    }

    // init root
    RootT* root = reinterpret_cast<RootT*>(buffer.get() + bytes[2]);
    { // set members of the root node
        auto *data = root->data();
        data->mBackground = 0.0f;
        data->mMinimum = 1.0f;
        data->mMaximum = 3.0f;
        data->mTableSize = 1;
        data->tile(0)->setChild(RootT::CoordType(0), node2, data);
    }

    // init tree
    TreeT* tree = reinterpret_cast<TreeT*>(buffer.get() + bytes[1]);
    {
        auto* data = tree->data();
        data->setRoot(root);
        data->mNodeCount[0] = data->mNodeCount[1] = data->mNodeCount[2] = 1;
    }

    GridT* grid = reinterpret_cast<GridT*>(buffer.get());
    { // init Grid
        auto* data = grid->data();
        {
            openvdb::math::UniformScaleTranslateMap map(2.0, openvdb::Vec3d(0.0, 0.0, 0.0));
            auto affineMap = map.getAffineMap();
            const auto mat = affineMap->getMat4(), invMat = mat.inverse();
            //for (int i=0; i<4; ++i) std::cout << "Row("<<i<<"): ["<<mat[i][0]<<", "<<mat[i][1]<<", "<<mat[i][2]<<", "<<mat[i][3]<<"]\n";
            //for (int i=0; i<4; ++i) std::cout << "Row("<<i<<"): ["<<invMat[i][0]<<", "<<invMat[i][1]<<", "<<invMat[i][2]<<", "<<invMat[i][3]<<"]\n";
#if 1
            nanovdb::Map dstMap;
            dstMap.set(mat, invMat);
            data->init({nanovdb::GridFlags::HasMinMax}, bytes[8], dstMap, nanovdb::GridType::Float);
#else
            data->mMap.set(mat, invMat, 1.0);
            data->mVoxelSize = affineMap->voxelSize();
            data->setFlagsOff();
            data->setMinMaxOn();
            data->mGridIndex = 0;
            data->mGridCount = 1;
            data->mBlindMetadataOffset = 0;
            data->mBlindMetadataCount = 0;
            data->mGridClass = nanovdb::GridClass::Unknown;
            data->mGridType = nanovdb::GridType::Float;
            data->mMagic = NANOVDB_MAGIC_NUMBER;
            data->mVersion = nanovdb::Version();
#endif
            memcpy(data->mGridName, name.c_str(), name.size() + 1);
        }
        EXPECT_EQ(tree, &grid->tree());
        const openvdb::Vec3d p1(1.0, 2.0, 3.0);
        const auto           p2 = grid->worldToIndex(p1);
        EXPECT_EQ(openvdb::Vec3d(0.5, 1.0, 1.5), p2);
        const auto p3 = grid->indexToWorld(p2);
        EXPECT_EQ(p1, p3);
        {
            openvdb::math::UniformScaleTranslateMap map(2.0, p1);
            auto                                    affineMap = map.getAffineMap();
            data->mVoxelSize = affineMap->voxelSize();
            const auto mat = affineMap->getMat4(), invMat = mat.inverse();
            //for (int i=0; i<4; ++i) std::cout << "Row("<<i<<"): ["<<mat[i][0]<<", "<<mat[i][1]<<", "<<mat[i][2]<<", "<<mat[i][3]<<"]\n";
            //for (int i=0; i<4; ++i) std::cout << "Row("<<i<<"): ["<<invMat[i][0]<<", "<<invMat[i][1]<<", "<<invMat[i][2]<<", "<<invMat[i][3]<<"]\n";
            data->mMap.set(mat, invMat, 1.0);
        }

        auto const p4 = grid->worldToIndex(p3);
        EXPECT_EQ(openvdb::Vec3d(0.0, 0.0, 0.0), p4);
        const auto p5 = grid->indexToWorld(p4);
        EXPECT_EQ(p1, p5);
    }

    { // check leaf node
        auto* ptr = leaf->data()->mValues;
        for (uint32_t i = 0; i < LeafT::voxelCount(); ++i) {
            if (i < 256) {
                EXPECT_FALSE(leaf->valueMask().isOn(i));
                EXPECT_EQ(0.0f, *ptr++);
            } else {
                EXPECT_TRUE(leaf->valueMask().isOn(i));
                EXPECT_EQ(1.0f, *ptr++);
            }
        }
        EXPECT_EQ(1.0f, leaf->minimum());
        EXPECT_EQ(1.0f, leaf->maximum());
        EXPECT_EQ(0.0f, leaf->getValue(CoordT(0)));
        EXPECT_EQ(1.0f, leaf->getValue(CoordT(8-1)));
    }

    { // check lower internal node
        auto& data = *reinterpret_cast<NodeT1::DataType*>(buffer.get() + bytes[4]);
        EXPECT_TRUE(node1->childMask().isOn(0));
        for (uint32_t i = 1; i < NodeT1::SIZE; ++i) {
            EXPECT_FALSE(node1->childMask().isOn(i));
            if (i < NodeT1::SIZE / 2) {
                EXPECT_FALSE(node1->valueMask().isOn(i));
               EXPECT_EQ(0.0f, data.mTable[i].value);
            } else {
                EXPECT_TRUE(node1->valueMask().isOn(i));
                EXPECT_EQ(2.0f, data.mTable[i].value);
            }
        }
        EXPECT_EQ(1.0f, node1->minimum());
        EXPECT_EQ(2.0f, node1->maximum());
        EXPECT_EQ(0.0f, node1->getValue(CoordT(0)));
        EXPECT_EQ(1.0f, node1->getValue(CoordT(8-1)));
        EXPECT_EQ(2.0f, node1->getValue(CoordT(8*16-1)));
    }
    { // check upper internal node
        auto& data = *reinterpret_cast<NodeT2::DataType*>(buffer.get() + bytes[3]);
        EXPECT_TRUE(node2->childMask().isOn(0));
        for (uint32_t i = 1; i < NodeT2::SIZE; ++i) {
            EXPECT_FALSE(node2->childMask().isOn(i));
            if (i < NodeT2::SIZE / 2) {
                EXPECT_FALSE(node2->valueMask().isOn(i));
                EXPECT_EQ(0.0f, data.mTable[i].value);
            } else {
                EXPECT_TRUE(node2->valueMask().isOn(i));
                EXPECT_EQ(3.0f, data.mTable[i].value);
            }
        }
        EXPECT_EQ(1.0f, node2->minimum());
        EXPECT_EQ(3.0f, node2->maximum());
        EXPECT_EQ(0.0f, node2->getValue(CoordT(0)));
        EXPECT_EQ(1.0f, node2->getValue(CoordT(8-1)));
        EXPECT_EQ(2.0f, node2->getValue(CoordT(8*16-1)));
        EXPECT_EQ(3.0f, node2->getValue(CoordT(8*16*32-1)));
    }
    { // check root
        EXPECT_EQ(0.0f, root->background());
        EXPECT_EQ(1.0f, root->minimum());
        EXPECT_EQ(3.0f, root->maximum());
        EXPECT_EQ(1u,   root->tileCount());
        EXPECT_EQ(0.0f, root->getValue(CoordT(0)));
        EXPECT_EQ(1.0f, root->getValue(CoordT(8-1)));
        EXPECT_EQ(2.0f, root->getValue(CoordT(8*16-1)));
        EXPECT_EQ(3.0f, root->getValue(CoordT(8*16*32-1)));
    }
    { // check tree
        EXPECT_EQ(0.0f, tree->background());
        float a, b;
        tree->extrema(a, b);
        EXPECT_EQ(1.0f, a);
        EXPECT_EQ(3.0f, b);
        EXPECT_EQ(0.0f, tree->getValue(CoordT(0)));
        EXPECT_EQ(1.0f, tree->getValue(CoordT(8-1)));
        EXPECT_EQ(2.0f, tree->getValue(CoordT(8*16-1)));
        EXPECT_EQ(3.0f, tree->getValue(CoordT(8*16*32-1)));
        EXPECT_EQ(1u, tree->nodeCount<LeafT>());
        EXPECT_EQ(1u, tree->nodeCount<NodeT1>());
        EXPECT_EQ(1u, tree->nodeCount<NodeT2>());
    }
    {// check grid
        EXPECT_EQ(nanovdb::Version(), grid->version());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), grid->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), grid->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), grid->version().getPatch());
        EXPECT_TRUE(grid->isValid());
        EXPECT_EQ(grid->gridType(), nanovdb::GridType::Float);
        EXPECT_EQ(grid->gridClass(),nanovdb::GridClass::Unknown);
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_FALSE(grid->isStaggered());
        EXPECT_FALSE(grid->isPointIndex());
        EXPECT_FALSE(grid->isPointData());
        EXPECT_FALSE(grid->isMask());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_TRUE(grid->hasMinMax());
        EXPECT_FALSE(grid->hasBBox());
        EXPECT_FALSE(grid->hasLongGridName());
        EXPECT_FALSE(grid->hasAverage());
        EXPECT_FALSE(grid->hasStdDeviation());
        //std::cerr << "\nName = \"" << grid->gridName() << "\"" << std::endl;
        EXPECT_EQ(name, std::string(grid->gridName()));
    }
    {// check ReadAccessor
        auto acc = grid->getAccessor();
        EXPECT_EQ(0.0f, acc.getValue(CoordT(0)));
        EXPECT_EQ(1.0f, acc.getValue(CoordT(8-1)));
        EXPECT_EQ(2.0f, acc.getValue(CoordT(8*16-1)));
        EXPECT_EQ(3.0f, acc.getValue(CoordT(8*16*32-1)));
        EXPECT_FALSE(acc.isActive(CoordT(0)));
        EXPECT_TRUE(acc.isActive(CoordT(8-1)));
        EXPECT_TRUE(acc.isActive(CoordT(16*8-1)));
        EXPECT_TRUE(acc.isActive(CoordT(32*16*8-1)));
    }
} // BaseGrid


TEST_F(TestOpenVDB, MagicType)
{
    {// toMagic(uint64_t)
        EXPECT_EQ( nanovdb::toMagic(NANOVDB_MAGIC_NUMB), nanovdb::MagicType::NanoVDB );
        EXPECT_EQ( nanovdb::toMagic(NANOVDB_MAGIC_GRID), nanovdb::MagicType::NanoGrid );
        EXPECT_EQ( nanovdb::toMagic(NANOVDB_MAGIC_FILE), nanovdb::MagicType::NanoFile );
        EXPECT_EQ( nanovdb::toMagic(NANOVDB_MAGIC_NODE), nanovdb::MagicType::NanoNode );
        EXPECT_EQ( nanovdb::toMagic(NANOVDB_MAGIC_FRAG), nanovdb::MagicType::NanoFrag );
        EXPECT_EQ( nanovdb::toMagic(      0x56444220UL), nanovdb::MagicType::OpenVDB );
    }

    {// toStr(MagicType)
        EXPECT_EQ( strcmp(nanovdb::toStr(mStr, nanovdb::MagicType::Unknown ),  "unknown"), 0 );
        EXPECT_EQ( strcmp(nanovdb::toStr(mStr, nanovdb::MagicType::OpenVDB ),  "openvdb"), 0 );
        EXPECT_EQ( strcmp(nanovdb::toStr(mStr, nanovdb::MagicType::NanoVDB ),  "nanovdb"), 0 );
        EXPECT_EQ( strcmp(nanovdb::toStr(mStr, nanovdb::MagicType::NanoGrid ), "nanovdb::Grid"), 0 );
        EXPECT_EQ( strcmp(nanovdb::toStr(mStr, nanovdb::MagicType::NanoFile ), "nanovdb::File"), 0 );
        EXPECT_EQ( strcmp(nanovdb::toStr(mStr, nanovdb::MagicType::NanoNode ), "nanovdb::NodeManager"), 0 );
        EXPECT_EQ( strcmp(nanovdb::toStr(mStr, nanovdb::MagicType::NanoFrag ), "fragmented nanovdb::Grid"), 0 );
    }
}

TEST_F(TestOpenVDB, OpenToNanoVDB_Empty)
{
    { // empty grid
        openvdb::FloatGrid srcGrid(0.0f);
        auto srcAcc = srcGrid.getAccessor();
        auto handle = nanovdb::tools::createNanoGrid(srcGrid);
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_TRUE(meta->isEmpty());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ(0u, dstGrid->activeVoxelCount());
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_EQ(0.0f, srcAcc.getValue(openvdb::Coord(1, 2, 3)));
        EXPECT_FALSE(srcAcc.isValueOn(openvdb::Coord(1, 2, 3)));
        EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(dstGrid->tree().root().minimum(), 0.0f);
        EXPECT_EQ(dstGrid->tree().root().maximum(), 0.0f);
        EXPECT_EQ(dstGrid->tree().root().average(), 0.0f);
        EXPECT_EQ(dstGrid->tree().root().variance(), 0.0f);
        EXPECT_EQ(dstGrid->tree().root().stdDeviation(), 0.0f);
    }
} // OpenToNanoVDB_Empty

TEST_F(TestOpenVDB, OpenToNanoVDB_Basic1)
{
    { // 1 grid point
        openvdb::FloatGrid srcGrid(0.0f);
        auto srcAcc = srcGrid.getAccessor();
        srcAcc.setValue(openvdb::Coord(1, 2, 3), 1.0f);
        EXPECT_TRUE(srcAcc.isValueOn(openvdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, srcAcc.getValue(openvdb::Coord(1, 2, 3)));
        auto handle = nanovdb::tools::createNanoGrid(srcGrid, nanovdb::tools::StatsMode::All);
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(sizeof(nanovdb::NanoGrid<float>) +
                  (const char*)handle.data(), (const char*)&dstGrid->tree());
        EXPECT_EQ(sizeof(nanovdb::NanoGrid<float>) +
                  sizeof(nanovdb::NanoTree<float>) +
                  (const char*)handle.data(), (const char*)&dstGrid->tree().root());
        EXPECT_EQ(nanovdb::Vec3d(1.0), dstGrid->voxelSize());
        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(nanovdb::Coord(1, 2, 3), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord(1, 2, 3), dstGrid->indexBBox()[1]);
        EXPECT_EQ(dstGrid->tree().root().minimum(), 1.0f);
        EXPECT_EQ(dstGrid->tree().root().maximum(), 1.0f);
        EXPECT_NEAR(dstGrid->tree().root().average(),  1.0f, 1e-6);
        EXPECT_NEAR(dstGrid->tree().root().variance(), 0.0f, 1e-6);
        EXPECT_NEAR(dstGrid->tree().root().stdDeviation(), 0.0f, 1e-6);
    }
} // OpenToNanoVDB_Basic1

TEST_F(TestOpenVDB, OpenToNanoVDB_Model)
{
    auto srcGrid = this->getSrcGrid(false);
    //mTimer.start("Generating NanoVDB grid");
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid);
    //mTimer.start("Writing NanoVDB grid");
    nanovdb::io::writeGrid("data/test.nvdb", handle, this->getCodec());
    //mTimer.stop();

    auto dstGrid = handle.grid<float>();
    EXPECT_TRUE(nanovdb::isAligned(dstGrid));

    auto kernel = [&](const openvdb::CoordBBox& bbox) {
        using CoordT = const nanovdb::Coord;
        auto dstAcc = handle.grid<float>()->getAccessor();
        auto srcAcc = srcGrid->getUnsafeAccessor(); // not registered
        for (auto it = bbox.begin(); it; ++it) {
            EXPECT_EQ(dstAcc.getValue(reinterpret_cast<CoordT&>(*it)), srcAcc.getValue(*it));
        }
    };

    //mTimer.start("Parallel unit test");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    //mTimer.restart("Testing bounding box");
    const auto dstBBox = dstGrid->indexBBox();
    const auto srcBBox = srcGrid->evalActiveVoxelBoundingBox();
    EXPECT_EQ(dstBBox.min()[0], srcBBox.min()[0]);
    EXPECT_EQ(dstBBox.min()[1], srcBBox.min()[1]);
    EXPECT_EQ(dstBBox.min()[2], srcBBox.min()[2]);
    EXPECT_EQ(dstBBox.max()[0], srcBBox.max()[0]);
    EXPECT_EQ(dstBBox.max()[1], srcBBox.max()[1]);
    EXPECT_EQ(dstBBox.max()[2], srcBBox.max()[2]);
    //mTimer.stop();

    EXPECT_EQ(srcGrid->activeVoxelCount(), dstGrid->activeVoxelCount());
} // OpenToNanoVDB_Model

TEST_F(TestOpenVDB, OpenToNanoVDB_Fp4)
{
    EXPECT_EQ(96u + 512u/2, sizeof(nanovdb::NanoLeaf<nanovdb::Fp4>));
    { // 3 grid point
        openvdb::FloatGrid srcGrid(0.0f);
        auto srcAcc = srcGrid.getAccessor();
        srcAcc.setValue(openvdb::Coord(  1,  2,  3), 1.0f);
        srcAcc.setValue(openvdb::Coord(-10, 20,-50), 2.0f);
        srcAcc.setValue(openvdb::Coord( 50,-12, 30), 3.0f);
        EXPECT_TRUE(srcAcc.isValueOn(openvdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, srcAcc.getValue(openvdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, srcAcc.getValue(openvdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, srcAcc.getValue(openvdb::Coord( 50,-12, 30)));

        nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> converter(srcGrid);
        //converter.setVerbose();
        converter.setStats(nanovdb::tools::StatsMode::All);
        auto handle = converter.getHandle<nanovdb::Fp4>();// (srcGrid);

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Fp4, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<nanovdb::Fp4>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(1.0f, dstGrid->tree().root().minimum());
        EXPECT_EQ(3.0f, dstGrid->tree().root().maximum());
        EXPECT_EQ(2.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_TRUE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3d(1.0), dstGrid->voxelSize());
        auto *leaf = dstGrid->tree().root().probeLeaf(nanovdb::Coord(1, 2, 3));
        EXPECT_TRUE(leaf);
        //std::cerr << leaf->origin() << ", " << leaf->data()->mBBoxMin << std::endl;
        EXPECT_EQ(nanovdb::Coord(0,0,0), leaf->origin());
        EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->data()->mBBoxMin);
        //const auto offset = nanovdb::NanoLeaf<nanovdb::Fp4>::CoordToOffset(nanovdb::Coord(1, 2, 3));
        //std::cerr << "offset = " << offset << std::endl;
        //std::cerr << "code = " <<  int(leaf->data()->mCode[offset>>1]) << std::endl;
        //std::cerr << "code = " <<  int(leaf->data()->mCode[offset>>1] >> 4) << std::endl;

        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        EXPECT_EQ(nanovdb::Coord(-10,-12,-50), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord( 50, 20, 30), dstGrid->indexBBox()[1]);
    }
    {// Model
        auto openGrid = this->getSrcGrid(false);
        const float tolerance = 0.5f*openGrid->voxelSize()[0];
        nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> converter(*openGrid);
        converter.enableDithering();
        //converter.setVerbose(2);
        auto handle = converter.getHandle<nanovdb::Fp4>();
        auto* nanoGrid = handle.grid<nanovdb::Fp4>();
        EXPECT_TRUE(nanoGrid);

        auto kernel = [&](const openvdb::CoordBBox& bbox) {
            auto nanoAcc = nanoGrid->getAccessor();
            auto openAcc = openGrid->getAccessor();
            for (auto it = bbox.begin(); it; ++it) {
                const openvdb::Coord p = *it;
                const nanovdb::Coord q(p[0], p[1], p[2]);
                EXPECT_NEAR(nanoAcc.getValue(q), openAcc.getValue(p), tolerance);
            }
        };
        tbb::parallel_for(openGrid->evalActiveVoxelBoundingBox(), kernel);

        nanovdb::io::writeGrid("data/test_fp4.nvdb", handle, this->getCodec());
        handle = nanovdb::io::readGrid("data/test_fp4.nvdb");
        nanoGrid = handle.grid<nanovdb::Fp4>();
        EXPECT_TRUE(nanoGrid);

        tbb::parallel_for(openGrid->evalActiveVoxelBoundingBox(), kernel);
    }
} // OpenToNanoVDB_Fp4

TEST_F(TestOpenVDB, OpenToNanoVDB_Fp8)
{
    EXPECT_EQ(96u + 512u, sizeof(nanovdb::NanoLeaf<nanovdb::Fp8>));
    { // 3 grid point
        openvdb::FloatGrid srcGrid(0.0f);
        auto srcAcc = srcGrid.getAccessor();
        srcAcc.setValue(openvdb::Coord(  1,  2,  3), 1.0f);
        srcAcc.setValue(openvdb::Coord(-10, 20,-50), 2.0f);
        srcAcc.setValue(openvdb::Coord( 50,-12, 30), 3.0f);
        EXPECT_TRUE(srcAcc.isValueOn(openvdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, srcAcc.getValue(openvdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, srcAcc.getValue(openvdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, srcAcc.getValue(openvdb::Coord( 50,-12, 30)));

        nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> converter(srcGrid);
        converter.setStats(nanovdb::tools::StatsMode::All);
        auto handle = converter.getHandle<nanovdb::Fp8>();

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Fp8, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<nanovdb::Fp8>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ(1.0f, dstGrid->tree().root().minimum());
        EXPECT_EQ(3.0f, dstGrid->tree().root().maximum());
        EXPECT_EQ(2.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_TRUE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3d(1.0), dstGrid->voxelSize());
        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        EXPECT_EQ(nanovdb::Coord(-10,-12,-50), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord( 50, 20, 30), dstGrid->indexBBox()[1]);
    }
    {// Model
        auto openGrid = this->getSrcGrid(false);
        const float tolerance = 0.05f*openGrid->voxelSize()[0];
        nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> converter(*openGrid);
        auto handle = converter.getHandle<nanovdb::Fp8>();
        converter.enableDithering();
        //converter.setVerbose(2);
        auto* nanoGrid = handle.grid<nanovdb::Fp8>();
        EXPECT_TRUE(nanoGrid);

        auto kernel = [&](const openvdb::CoordBBox& bbox) {
            auto nanoAcc = nanoGrid->getAccessor();
            auto openAcc = openGrid->getAccessor();
            for (auto it = bbox.begin(); it; ++it) {
                const openvdb::Coord p = *it;
                const nanovdb::Coord q(p[0], p[1], p[2]);
                EXPECT_NEAR(nanoAcc.getValue(q), openAcc.getValue(p), tolerance);
            }
        };
        tbb::parallel_for(openGrid->evalActiveVoxelBoundingBox(), kernel);

        nanovdb::io::writeGrid("data/test_fp8.nvdb", handle, this->getCodec());

        handle = nanovdb::io::readGrid("data/test_fp8.nvdb");
        nanoGrid = handle.grid<nanovdb::Fp8>();
        EXPECT_TRUE(nanoGrid);

        tbb::parallel_for(openGrid->evalActiveVoxelBoundingBox(), kernel);
    }
} // OpenToNanoVDB_Fp8

TEST_F(TestOpenVDB, OpenToNanoVDB_Fp16)
{
    EXPECT_EQ(96u + 512u*2, sizeof(nanovdb::NanoLeaf<nanovdb::Fp16>));
    { // 3 grid point
        openvdb::FloatGrid srcGrid(0.0f);
        auto srcAcc = srcGrid.getAccessor();
        srcAcc.setValue(openvdb::Coord(  1,  2,  3), 1.0f);
        srcAcc.setValue(openvdb::Coord(-10, 20,-50), 2.0f);
        srcAcc.setValue(openvdb::Coord( 50,-12, 30), 3.0f);
        EXPECT_TRUE(srcAcc.isValueOn(openvdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, srcAcc.getValue(openvdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, srcAcc.getValue(openvdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, srcAcc.getValue(openvdb::Coord( 50,-12, 30)));

        nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> converter(srcGrid);
        //converter.setVerbose(2);
        converter.setStats(nanovdb::tools::StatsMode::All);
        auto handle = converter.getHandle<nanovdb::Fp16>();

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Fp16, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<nanovdb::Fp16>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(1.0f, dstGrid->tree().root().minimum());
        EXPECT_EQ(3.0f, dstGrid->tree().root().maximum());
        EXPECT_EQ(2.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_TRUE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3d(1.0), dstGrid->voxelSize());
        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        EXPECT_EQ(nanovdb::Coord(-10,-12,-50), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord( 50, 20, 30), dstGrid->indexBBox()[1]);
    }
    {// Model
        auto openGrid = this->getSrcGrid(false);
        const float tolerance = 0.005f*openGrid->voxelSize()[0];
        nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> converter(*openGrid);
        converter.enableDithering();
        auto handle = converter.getHandle<nanovdb::Fp16>();
        //converter.setVerbose(2);
        auto* nanoGrid = handle.grid<nanovdb::Fp16>();
        EXPECT_TRUE(nanoGrid);

        auto kernel = [&](const openvdb::CoordBBox& bbox) {
            auto nanoAcc = nanoGrid->getAccessor();
            auto openAcc = openGrid->getAccessor();
            for (auto it = bbox.begin(); it; ++it) {
                const openvdb::Coord p = *it;
                const nanovdb::Coord q(p[0], p[1], p[2]);
                EXPECT_NEAR(nanoAcc.getValue(q), openAcc.getValue(p), tolerance);
            }
        };
        tbb::parallel_for(openGrid->evalActiveVoxelBoundingBox(), kernel);

        nanovdb::io::writeGrid("data/test_fp16.nvdb", handle, this->getCodec());

        handle = nanovdb::io::readGrid("data/test_fp16.nvdb");
        nanoGrid = handle.grid<nanovdb::Fp16>();
        EXPECT_TRUE(nanoGrid);

        tbb::parallel_for(openGrid->evalActiveVoxelBoundingBox(), kernel);
    }
} // OpenToNanoVDB_Fp16

TEST_F(TestOpenVDB, OpenToNanoVDB_FpN)
{
    EXPECT_EQ(96u, sizeof(nanovdb::NanoLeaf<nanovdb::FpN>));
    { // 3 grid point
        openvdb::FloatGrid srcGrid(0.0f);
        auto srcAcc = srcGrid.getAccessor();
        srcAcc.setValue(openvdb::Coord(  1,  2,  3), 1.0f);
        srcAcc.setValue(openvdb::Coord(-10, 20,-50), 2.0f);
        srcAcc.setValue(openvdb::Coord( 50,-12, 30), 3.0f);
        EXPECT_TRUE(srcAcc.isValueOn(openvdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, srcAcc.getValue(openvdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, srcAcc.getValue(openvdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, srcAcc.getValue(openvdb::Coord( 50,-12, 30)));

        nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> converter(srcGrid);
        converter.setStats(nanovdb::tools::StatsMode::All);
        auto handle = converter.getHandle<nanovdb::FpN>();

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::FpN, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<nanovdb::FpN>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(1.0f, dstGrid->tree().root().minimum());
        EXPECT_EQ(3.0f, dstGrid->tree().root().maximum());
        EXPECT_EQ(2.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_FALSE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_FALSE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3d(1.0), dstGrid->voxelSize());
        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        EXPECT_EQ(nanovdb::Coord(-10,-12,-50), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord( 50, 20, 30), dstGrid->indexBBox()[1]);
    }
    {// Model
#if 1// switch between level set and FOG volume
        auto openGrid = this->getSrcGrid(false);// level set dragon or sphere
#else
        auto openGrid = this->getSrcGrid(true, 1, 1);// FOG volume of Disney cloud or cube
#endif
        nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> converter(*openGrid);
        //converter.setVerbose(2);

        const float tolerance = 0.05f;
        nanovdb::tools::AbsDiff oracle(tolerance);

        auto handle = converter.getHandle<nanovdb::FpN>(oracle);
        auto* nanoGrid = handle.grid<nanovdb::FpN>();
        EXPECT_TRUE(nanoGrid);

        nanovdb::io::writeGrid("data/test_fpN.nvdb", handle, this->getCodec());

        auto kernel = [&](const openvdb::CoordBBox& bbox) {
            auto nanoAcc = nanoGrid->getAccessor();
            auto openAcc = openGrid->getAccessor();
            for (auto it = bbox.begin(); it; ++it) {
                const openvdb::Coord p = *it;
                const nanovdb::Coord q(p[0], p[1], p[2]);
                const float exact  = openAcc.getValue(p);
                const float approx = nanoAcc.getValue(q);
                EXPECT_NEAR( approx, exact, tolerance );
                EXPECT_TRUE( oracle(exact, approx) );
            }
        };
        nanovdb::util::forEach(openGrid->evalActiveVoxelBoundingBox(), kernel);

        handle = nanovdb::io::readGrid("data/test_fpN.nvdb");
        nanoGrid = handle.grid<nanovdb::FpN>();
        EXPECT_TRUE(nanoGrid);

        nanovdb::util::forEach(openGrid->evalActiveVoxelBoundingBox(), kernel);
    }
} // OpenToNanoVDB_FpN

// Generate random points by uniformly distributing points
// on a unit-sphere.
inline void genPoints(const int numPoints, std::vector<openvdb::Vec3d>& points)
{
    openvdb::math::Random01 randNumber(0);
    const int               n = int(std::sqrt(double(numPoints)));
    const double            xScale = (2.0 * openvdb::math::pi<double>()) / double(n);
    const double            yScale = openvdb::math::pi<double>() / double(n);

    double         x, y, theta, phi;
    openvdb::Vec3d pos;

    points.reserve(n * n);

    // loop over a [0 to n) x [0 to n) grid.
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            // jitter, move to random pos. inside the current cell
            x = double(a) + randNumber();
            y = double(b) + randNumber();

            // remap to a lat/long map
            theta = y * yScale; // [0 to PI]
            phi = x * xScale; // [0 to 2PI]

            // convert to cartesian coordinates on a unit sphere.
            // spherical coordinate triplet (r=1, theta, phi)
            pos[0] = std::sin(theta) * std::cos(phi);
            pos[1] = std::sin(theta) * std::sin(phi);
            pos[2] = std::cos(theta);

            points.push_back(pos);
        }
    }
} // genPoints
class PointList
{
    std::vector<openvdb::Vec3d> const* const mPoints;

public:
    using PosType = openvdb::Vec3d;
    PointList(const std::vector<PosType>& points)
        : mPoints(&points)
    {
    }
    size_t size() const { return mPoints->size(); }
    void   getPos(size_t n, PosType& xyz) const { xyz = (*mPoints)[n]; }
}; // PointList

// make testOpenVDB && ./unittest/testOpenVDB --gtest_filter="*PointIndexGrid" --gtest_break_on_failure
TEST_F(TestOpenVDB, PointIndexGrid)
{
    const uint64_t pointCount = 40000;
    const float    voxelSize = 0.01f;
    const auto     transform = openvdb::math::Transform::createLinearTransform(voxelSize);

    std::vector<openvdb::Vec3d> points;
    genPoints(pointCount, points);
    PointList pointList(points);
    EXPECT_EQ(pointCount, points.size());

    using SrcGridT = openvdb::tools::PointIndexGrid;
    auto srcGrid = openvdb::tools::createPointIndexGrid<SrcGridT>(pointList, *transform);

    using MgrT = openvdb::tree::LeafManager<const SrcGridT::TreeType>;
    MgrT leafMgr(srcGrid->tree());

    size_t count = 0;
    for (size_t n = 0, N = leafMgr.leafCount(); n < N; ++n) {
        count += leafMgr.leaf(n).indices().size();
    }
    EXPECT_EQ(pointCount, count);

    //mTimer.start("Generating NanoVDB grid from PointIndexGrid");
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid, nanovdb::tools::StatsMode::All, nanovdb::CheckMode::Full);
    //mTimer.stop();
    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ(nanovdb::GridType::UInt32, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::PointIndex, meta->gridClass());
    auto dstGrid = handle.grid<uint32_t>();
    EXPECT_TRUE(dstGrid);
    EXPECT_EQ(1u, dstGrid->blindDataCount());
    const auto &metaData = dstGrid->blindMetaData(0);
    EXPECT_EQ(pointCount, metaData.mValueCount);
    EXPECT_EQ(nanovdb::GridBlindDataSemantic::PointId, metaData.mSemantic);
    EXPECT_EQ(nanovdb::GridBlindDataClass::IndexArray, metaData.mDataClass);
    EXPECT_EQ(nanovdb::GridType::UInt32, metaData.mDataType);

    // first check the voxel values
    auto kernel1 = [&](const openvdb::CoordBBox& bbox) {
        using CoordT = const nanovdb::Coord;
        auto dstAcc = dstGrid->getAccessor();
        auto srcAcc = srcGrid->getAccessor();
        for (auto it = bbox.begin(); it; ++it) {
            EXPECT_EQ(srcAcc.getValue(*it), dstAcc.getValue(reinterpret_cast<CoordT&>(*it)));
        }
    };
    //mTimer.start("Parallel unit test of voxel values");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel1);
    //mTimer.stop();

    EXPECT_EQ(pointCount, dstGrid->blindMetaData(0).mValueCount);
    //std::cerr << ""

    auto kernel = [&](const MgrT::LeafRange& r) {
        using CoordT = const nanovdb::Coord;
        auto                             dstAcc = dstGrid->getAccessor();
        nanovdb::PointAccessor<uint32_t> pointAcc(*dstGrid);
        EXPECT_TRUE(pointAcc);
        const uint32_t *                 begin2 = nullptr, *end2 = nullptr;
        EXPECT_EQ(pointCount, pointAcc.gridPoints(begin2, end2));
        for (auto leaf = r.begin(); leaf; ++leaf) {
            const auto origin1 = leaf->origin();
            const auto origin2 = reinterpret_cast<const CoordT*>(&origin1);
            EXPECT_EQ(leaf->indices().size(), pointAcc.leafPoints(*origin2, begin2, end2));
            for (auto it = leaf->cbeginValueOn(); it; ++it) {
                const auto  ijk = it.getCoord();
                const auto* abc = reinterpret_cast<const CoordT*>(&ijk);
                EXPECT_TRUE(dstAcc.isActive(*abc));
                const openvdb::PointIndex32 *begin1 = nullptr, *end1 = nullptr;
                EXPECT_TRUE(leaf->getIndices(ijk, begin1, end1));
                EXPECT_TRUE(pointAcc.voxelPoints(*abc, begin2, end2));
                EXPECT_EQ(end1 - begin1, end2 - begin2);
                for (auto* i = begin1; i != end1; ++i) EXPECT_EQ(*i, *begin2++);
            }
        }
    };

    //mTimer.start("Parallel unit test");
    tbb::parallel_for(leafMgr.leafRange(), kernel);
    //mTimer.stop();

    //mTimer.start("Testing bounding box");
    const auto dstBBox = dstGrid->indexBBox();
    //std::cerr << "\nBBox = " << dstBBox << std::endl;
    const auto srcBBox = srcGrid->evalActiveVoxelBoundingBox();
    EXPECT_EQ(dstBBox.min()[0], srcBBox.min()[0]);
    EXPECT_EQ(dstBBox.min()[1], srcBBox.min()[1]);
    EXPECT_EQ(dstBBox.min()[2], srcBBox.min()[2]);
    EXPECT_EQ(dstBBox.max()[0], srcBBox.max()[0]);
    EXPECT_EQ(dstBBox.max()[1], srcBBox.max()[1]);
    EXPECT_EQ(dstBBox.max()[2], srcBBox.max()[2]);
    //mTimer.stop();

    EXPECT_EQ(srcGrid->activeVoxelCount(), dstGrid->activeVoxelCount());
} // PointIndexGrid

TEST_F(TestOpenVDB, PointDataGridBasic)
{
    // Create a vector with three point positions.
    std::vector<openvdb::Vec3d> positions;
    positions.push_back(openvdb::Vec3d(0.0, 0.0, 0.0));
    positions.push_back(openvdb::Vec3d(0.0, 0.0, 1.0));
    positions.push_back(openvdb::Vec3d(1.34, -56.1, 5.7));
    EXPECT_EQ( 3UL, positions.size() );

    // We need to define a custom search lambda function
    // to account for floating-point roundoffs!
    auto search = [&positions](const openvdb::Vec3f &p) {
        for (auto it = positions.begin(); it != positions.end(); ++it) {
            const openvdb::Vec3d delta = *it - p;
            if ( delta.length() < 1e-5 ) return it;
        }
        return positions.end();
    };

    // The VDB Point-Partitioner is used when bucketing points and requires a
    // specific interface. For convenience, we use the PointAttributeVector
    // wrapper around an stl vector wrapper here, however it is also possible to
    // write one for a custom data structure in order to match the interface
    // required.
    openvdb::points::PointAttributeVector<openvdb::Vec3d> positionsWrapper(positions);
    // This method computes a voxel-size to match the number of
    // points / voxel requested. Although it won't be exact, it typically offers
    // a good balance of memory against performance.
    int   pointsPerVoxel = 8;
    float voxelSize = openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel);
    //std::cerr << "VoxelSize = " << voxelSize << std::endl;
    auto transform = openvdb::math::Transform::createLinearTransform(voxelSize);
    auto srcGrid = openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
                                                        openvdb::points::PointDataGrid>(positions, *transform);
    srcGrid->setName("PointDataGrid");

    //mTimer.start("Generating NanoVDB grid from PointDataGrid");
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid);
    //mTimer.stop();

    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ(nanovdb::GridType::UInt32, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::PointData, meta->gridClass());

    auto *dstGrid = handle.grid<uint32_t>();
    EXPECT_TRUE(dstGrid);
    for (int i=0; i<3; ++i) {
        EXPECT_EQ(srcGrid->voxelSize()[i], dstGrid->voxelSize()[i]);
    }
    EXPECT_EQ(1u, dstGrid->blindDataCount());// only point positions
    auto &metaData = dstGrid->blindMetaData(0u);
    EXPECT_EQ(metaData.mValueCount, positions.size());
    EXPECT_EQ(strcmp("P", metaData.mName), 0);
    EXPECT_EQ(metaData.mDataClass, nanovdb::GridBlindDataClass::AttributeArray);
    EXPECT_EQ(metaData.mSemantic, nanovdb::GridBlindDataSemantic::PointPosition);
    EXPECT_EQ(metaData.mDataType, nanovdb::GridType::Vec3f);

    nanovdb::PointAccessor<nanovdb::Vec3f> acc(*dstGrid);
    EXPECT_TRUE(acc);
    const nanovdb::Vec3f *begin = nullptr, *end = nullptr; // iterators over points in a given voxel
    EXPECT_EQ(positions.size(), openvdb::points::pointCount(srcGrid->tree()));
    EXPECT_EQ(acc.gridPoints(begin, end), positions.size());
    for (auto leafIter = srcGrid->tree().cbeginLeaf(); leafIter; ++leafIter) {
        EXPECT_TRUE(leafIter->hasAttribute("P")); // Check position attribute from the leaf by name (P is position).
        // Create a read-only AttributeHandle. Position always uses Vec3f.
        openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(leafIter->constAttributeArray("P"));
        openvdb::Coord ijkSrc(openvdb::Coord::min());
        nanovdb::Coord ijkDst(nanovdb::math::Maximum<int>::value());
        for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
            // Extract the local voxel-space position of the point relative to its occupying voxel ijk.
            const openvdb::Vec3f vxlSrc = positionHandle.get(*indexIter);
            if (ijkSrc != indexIter.getCoord()) { // new voxel
                ijkSrc = indexIter.getCoord();
                for (int i=0; i<3; ++i) ijkDst[i] = ijkSrc[i];
                EXPECT_TRUE(acc.isActive(ijkDst));
                EXPECT_TRUE(acc.voxelPoints(ijkDst, begin, end));
            }
            EXPECT_NE(nullptr, begin);
            EXPECT_NE(nullptr, end);
            EXPECT_TRUE(begin < end);
            const nanovdb::Vec3f vxlDst = *begin++;// local voxel coordinates
            for (int i=0; i<3; ++i) {
                EXPECT_EQ( ijkSrc[i], ijkDst[i] );
                //EXPECT_EQ( vxlSrc[i], vxlDst[i] );
            }
            // A PointDataGrid encodes local voxel coordinates
            // so transform those to global index coordinates!
            const openvdb::Vec3f idxSrc = ijkSrc.asVec3s()       + vxlSrc;
            const nanovdb::Vec3f idxDst = nanovdb::Vec3f(ijkDst) + vxlDst;

            // Transform global index coordinates to global world coordinates
            const openvdb::Vec3f wldSrc = srcGrid->indexToWorld(idxSrc);
            const nanovdb::Vec3f wldDst = dstGrid->indexToWorld(idxDst);

            //std::cerr << "voxel = " << vxlDst << ", index = " << idxDst << ", world = " << wldDst << std::endl;
            for (int i = 0; i < 3; ++i) {
                EXPECT_EQ( idxSrc[i], idxDst[i] );
                EXPECT_EQ( wldSrc[i], wldDst[i] );
            }

            // compare to original input points
            auto it = search( wldSrc );
            EXPECT_TRUE( it != positions.end() );
            positions.erase( it );
        }
    }
    EXPECT_EQ( 0UL, positions.size() );// verify that we found all the input points
} // PointDataGridBasic

TEST_F(TestOpenVDB, PointDataGridRandom)
{
    std::vector<openvdb::Vec3d> positions;
    const size_t pointCount = 2000;
    const openvdb::Vec3d wldMin(-234.3, -135.6, -503.7);
    const openvdb::Vec3d wldMax(  57.8,  289.1,    0.2);
    const openvdb::Vec3d wldDim = wldMax - wldMin;
    openvdb::math::Random01 randNumber(0);

    // We need to define a custom search lambda function
    // to account for floating-point roundoffs!
    auto search = [&positions](const openvdb::Vec3f &p) {
        for (auto it = positions.begin(); it != positions.end(); ++it) {
            const openvdb::Vec3d delta = *it - p;
            if ( delta.length() < 1e-3 ) return it;
        }
        return positions.end();
    };

    // Create a vector with random point positions.
    for (size_t i=0; i<pointCount; ++i) {
        const openvdb::Vec3d d(randNumber(), randNumber(), randNumber());
        const openvdb::Vec3d p = wldMin + d * wldDim;
        if (search(p) != positions.end()) continue;// avoid duplicates!
        positions.push_back(p);
    }
    EXPECT_EQ( pointCount, positions.size() );

    // The VDB Point-Partitioner is used when bucketing points and requires a
    // specific interface. For convenience, we use the PointAttributeVector
    // wrapper around an stl vector wrapper here, however it is also possible to
    // write one for a custom data structure in order to match the interface
    // required.
    openvdb::points::PointAttributeVector<openvdb::Vec3d> positionsWrapper(positions);
    // This method computes a voxel-size to match the number of
    // points / voxel requested. Although it won't be exact, it typically offers
    // a good balance of memory against performance.
    int   pointsPerVoxel = 8;
    float voxelSize = openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel);
    //std::cerr << "VoxelSize = " << voxelSize << std::endl;
    auto transform = openvdb::math::Transform::createLinearTransform(voxelSize);
    auto srcGrid = openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
                                                        openvdb::points::PointDataGrid>(positions, *transform);
    srcGrid->setName("PointDataGrid");

    //mTimer.start("Generating NanoVDB grid from PointDataGrid");
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid);
    //mTimer.stop();

    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ(nanovdb::GridType::UInt32, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::PointData, meta->gridClass());
    auto dstGrid = handle.grid<uint32_t>();
    EXPECT_TRUE(dstGrid);
    for (int i=0; i<3; ++i) {
        EXPECT_EQ(srcGrid->voxelSize()[i], dstGrid->voxelSize()[i]);
    }

    nanovdb::PointAccessor<nanovdb::Vec3f> acc(*dstGrid);
    EXPECT_TRUE(acc);
    const nanovdb::Vec3f *begin = nullptr, *end = nullptr; // iterators over points in a given voxel
    EXPECT_EQ(positions.size(), openvdb::points::pointCount(srcGrid->tree()));
    EXPECT_EQ(acc.gridPoints(begin, end), positions.size());
    for (auto leafIter = srcGrid->tree().cbeginLeaf(); leafIter; ++leafIter) {
        EXPECT_TRUE(leafIter->hasAttribute("P")); // Check position attribute from the leaf by name (P is position).
        // Create a read-only AttributeHandle. Position always uses Vec3f.
        openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(leafIter->constAttributeArray("P"));
        openvdb::Coord ijkSrc(openvdb::Coord::min());
        nanovdb::Coord ijkDst(nanovdb::math::Maximum<int>::value());
        for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
            // Extract the local voxel-space position of the point relative to its occupying voxel ijk.
            const openvdb::Vec3f vxlSrc = positionHandle.get(*indexIter);
            if (ijkSrc != indexIter.getCoord()) { // new voxel
                ijkSrc = indexIter.getCoord();
                for (int i=0; i<3; ++i) ijkDst[i] = ijkSrc[i];
                EXPECT_TRUE(acc.isActive(ijkDst));
                EXPECT_TRUE(acc.voxelPoints(ijkDst, begin, end));
            }
            EXPECT_NE(nullptr, begin);
            EXPECT_NE(nullptr, end);
            EXPECT_TRUE(begin < end);
            const nanovdb::Vec3f vxlDst = *begin++;// local voxel coordinates
            for (int i=0; i<3; ++i) {
                EXPECT_EQ( ijkSrc[i], ijkDst[i] );
                EXPECT_EQ( vxlSrc[i], vxlDst[i] );
            }
            // A PointDataGrid encodes local voxel coordinates
            // so transform those to global index coordinates!
            const openvdb::Vec3f idxSrc = ijkSrc.asVec3s() + vxlSrc;
            const nanovdb::Vec3f idxDst = ijkDst.asVec3s() + vxlDst;

            // Transform global index coordinates to global world coordinates
            const openvdb::Vec3f wldSrc = srcGrid->indexToWorld(idxSrc);
            const nanovdb::Vec3f wldDst = dstGrid->indexToWorld(idxDst);

            //std::cerr << "voxel = " << vxlDst << ", index = " << idxDst << ", world = " << wldDst << std::endl;
            for (int i = 0; i < 3; ++i) {
                EXPECT_EQ( idxSrc[i], idxDst[i] );
                EXPECT_EQ( wldSrc[i], wldDst[i] );
            }

            // compair to original input points
            auto it = search( wldSrc );
            EXPECT_TRUE( it != positions.end() );
            positions.erase( it );
        }
    }
    EXPECT_EQ( 0UL, positions.size() );// verify that we found all the input points
} // PointDataGridRandom

#if !defined(_MSC_VER)
// Disabled due to error compiling CNanoVDB on some compilers due to zero-sized arrays.
// So we should probably disable on those compilers, rather than
// on all platforms...
TEST_F(TestOpenVDB, CNanoVDBSize)
{
    // Verify the sizes of structures are what we expect.
    EXPECT_EQ(sizeof(cnanovdb_mask3), sizeof(nanovdb::Mask<3>));
    EXPECT_EQ(sizeof(cnanovdb_mask4), sizeof(nanovdb::Mask<4>));
    EXPECT_EQ(sizeof(cnanovdb_mask5), sizeof(nanovdb::Mask<5>));
    EXPECT_EQ(sizeof(cnanovdb_map), sizeof(nanovdb::Map));
    EXPECT_EQ(sizeof(cnanovdb_coord), sizeof(nanovdb::Coord));
    EXPECT_EQ(sizeof(cnanovdb_Vec3F), sizeof(nanovdb::Vec3f));

    EXPECT_EQ(sizeof(cnanovdb_node0F), sizeof(nanovdb::LeafNode<float>));
    EXPECT_EQ(sizeof(cnanovdb_node1F), sizeof(nanovdb::InternalNode<nanovdb::LeafNode<float>>));
    EXPECT_EQ(sizeof(cnanovdb_node2F), sizeof(nanovdb::InternalNode<nanovdb::InternalNode<nanovdb::LeafNode<float>>>));
    EXPECT_EQ(sizeof(cnanovdb_rootdataF), sizeof(nanovdb::NanoRoot<float>));

    EXPECT_EQ(sizeof(cnanovdb_node0F3), sizeof(nanovdb::LeafNode<nanovdb::Vec3f>));
    EXPECT_EQ(sizeof(cnanovdb_node1F3), sizeof(nanovdb::InternalNode<nanovdb::LeafNode<nanovdb::Vec3f>>));
    EXPECT_EQ(sizeof(cnanovdb_node2F3), sizeof(nanovdb::InternalNode<nanovdb::InternalNode<nanovdb::LeafNode<nanovdb::Vec3f>>>));
    EXPECT_EQ(sizeof(cnanovdb_rootdataF3), sizeof(nanovdb::NanoRoot<nanovdb::Vec3f>));

    EXPECT_EQ(sizeof(cnanovdb_treedata), sizeof(nanovdb::NanoTree<float>));
    EXPECT_EQ(sizeof(cnanovdb_gridblindmetadata), sizeof(nanovdb::GridBlindMetaData));
    EXPECT_EQ(sizeof(cnanovdb_griddata), sizeof(nanovdb::NanoGrid<float>));
}// CNanoVDBSize

TEST_F(TestOpenVDB, CNanoVDB)
{
    auto srcGrid = this->getSrcGrid();
    //mTimer.start("Generating NanoVDB grid");
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid);
    //mTimer.stop();
    EXPECT_TRUE(handle);
    EXPECT_TRUE(handle.data());

    const cnanovdb_griddata*  gridData = (const cnanovdb_griddata*)(handle.data());
    const cnanovdb_treedata*  treeData = cnanovdb_griddata_tree(gridData);
    const cnanovdb_rootdataF* rootData = cnanovdb_treedata_rootF(treeData);

    auto kernel = [&](const openvdb::CoordBBox& bbox) {
        cnanovdb_readaccessor dstAcc;
        cnanovdb_readaccessor_init(&dstAcc, rootData);
        auto srcAcc = srcGrid->getUnsafeAccessor(); // not registered
        for (auto it = bbox.begin(); it; ++it) {
            auto       ijk = *it;
            const auto v = cnanovdb_readaccessor_getValueF(&dstAcc, (cnanovdb_coord*)&ijk);
            EXPECT_EQ(srcAcc.getValue(ijk), v);
            const bool t = cnanovdb_readaccessor_isActiveF(&dstAcc, (cnanovdb_coord*)&ijk);
            EXPECT_EQ(srcAcc.isValueOn(ijk), t);
        }
    };

    //mTimer.start("Parallel unit test");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    //mTimer.stop();
}// CNanoVDB

TEST_F(TestOpenVDB, CNanoVDBTrilinear)
{
    auto srcGrid = this->getSrcGrid();
    //mTimer.start("Generating NanoVDB grid");
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid);
    //mTimer.stop();
    EXPECT_TRUE(handle);
    EXPECT_TRUE(handle.data());

    const cnanovdb_griddata* gridData = (const cnanovdb_griddata*)(handle.data());
    EXPECT_TRUE(cnanovdb_griddata_valid(gridData));
    EXPECT_TRUE(cnanovdb_griddata_validF(gridData));
    EXPECT_FALSE(cnanovdb_griddata_validF3(gridData));
    const cnanovdb_treedata*  treeData = cnanovdb_griddata_tree(gridData);
    const cnanovdb_rootdataF* rootData = cnanovdb_treedata_rootF(treeData);

    auto kernel = [&](const openvdb::CoordBBox& bbox) {
        cnanovdb_readaccessor dstAcc;
        cnanovdb_readaccessor_init(&dstAcc, rootData);
        auto srcAcc = srcGrid->getUnsafeAccessor(); // not registered
        for (auto it = bbox.begin(); it; ++it) {
            auto           ijk = *it;
            cnanovdb_Vec3F cn_xyz;
            cn_xyz.mVec[0] = ijk[0] + 0.3;
            cn_xyz.mVec[1] = ijk[1] + 0.7;
            cn_xyz.mVec[2] = ijk[2] + 0.9;
            const auto v = cnanovdb_sampleF_trilinear(&dstAcc, &cn_xyz);

            openvdb::math::Vec3d xyz(ijk[0] + 0.3,
                                     ijk[1] + 0.7,
                                     ijk[2] + 0.9);
            float                truth;
            openvdb::tools::BoxSampler::sample(srcAcc, xyz, truth);
            EXPECT_NEAR(truth, v, 1e-5);
        }
    };

    //mTimer.start("Parallel unit test");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    //mTimer.stop();
}// CNanoVDBTrilinear

TEST_F(TestOpenVDB, CNanoVDBTrilinearStencil)
{
    auto srcGrid = this->getSrcGrid();
    //mTimer.start("Generating NanoVDB grid");
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid);
    //mTimer.stop();
    EXPECT_TRUE(handle);
    EXPECT_TRUE(handle.data());

    const cnanovdb_griddata*  gridData = (const cnanovdb_griddata*)(handle.data());
    const cnanovdb_treedata*  treeData = cnanovdb_griddata_tree(gridData);
    const cnanovdb_rootdataF* rootData = cnanovdb_treedata_rootF(treeData);

    auto kernel = [&](const openvdb::CoordBBox& bbox) {
        cnanovdb_readaccessor dstAcc;
        cnanovdb_readaccessor_init(&dstAcc, rootData);
        cnanovdb_stencil1F stencil;
        cnanovdb_stencil1F_clear(&stencil);
        auto srcAcc = srcGrid->getUnsafeAccessor(); // not registered
        for (auto it = bbox.begin(); it; ++it) {
            auto           ijk = *it;
            cnanovdb_Vec3F cn_xyz;
            cn_xyz.mVec[0] = ijk[0] + 0.3;
            cn_xyz.mVec[1] = ijk[1] + 0.7;
            cn_xyz.mVec[2] = ijk[2] + 0.9;
            const auto v = cnanovdb_sampleF_trilinear_stencil(&stencil, &dstAcc, &cn_xyz);

            openvdb::math::Vec3d xyz(ijk[0] + 0.3,
                                     ijk[1] + 0.7,
                                     ijk[2] + 0.9);
            float                truth;
            openvdb::tools::BoxSampler::sample(srcAcc, xyz, truth);
            EXPECT_NEAR(truth, v, 1e-5);
        }
    };

    //mTimer.start("Parallel unit test");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    //mTimer.stop();
}// CNanoVDBTrilinearStencil

#endif

TEST_F(TestOpenVDB, NanoToOpenVDB_BuildGrid)
{// test build::Grid -> NanoVDB -> OpenVDB
    nanovdb::tools::build::Grid<float> buildGrid(0.0f, "test", nanovdb::GridClass::LevelSet);
    auto buildAcc = buildGrid.getAccessor();
    buildAcc.setValue(nanovdb::Coord(1,  2, 3), 1.0f);
    buildAcc.setValue(nanovdb::Coord(2, -2, 9), 2.0f);
    EXPECT_EQ(1.0f, buildAcc.getValue(nanovdb::Coord(1,  2, 3)));
    EXPECT_EQ(2.0f, buildAcc.getValue(nanovdb::Coord(2, -2, 9)));
    auto handle = nanovdb::tools::createNanoGrid(buildGrid);
    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_FALSE(meta->isEmpty());
    EXPECT_EQ("test", std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());

    auto* nanoGrid = handle.grid<float>();
    EXPECT_TRUE(nanoGrid);
    EXPECT_EQ("test", std::string(nanoGrid->gridName()));
    auto nanoAcc = nanoGrid->getAccessor();
    EXPECT_EQ(1.0f, nanoAcc.getValue(nanovdb::Coord(1,  2, 3)));
    EXPECT_EQ(2.0f, nanoAcc.getValue(nanovdb::Coord(2, -2, 9)));

    auto openGrid = nanovdb::tools::nanoToOpenVDB(*nanoGrid);
    EXPECT_TRUE(openGrid);
    auto openAcc = openGrid->getAccessor();
    EXPECT_EQ(1.0f, openAcc.getValue(openvdb::Coord(1,  2, 3)));
    EXPECT_EQ(2.0f, openAcc.getValue(openvdb::Coord(2, -2, 9)));

    const auto nanoBBox = nanoGrid->indexBBox();
    const auto openBBox = openGrid->evalActiveVoxelBoundingBox();
    EXPECT_EQ(nanovdb::Coord(1,-2,3), nanoBBox.min());
    EXPECT_EQ(openvdb::Coord(1,-2,3), openBBox.min());
    EXPECT_EQ(nanovdb::Coord(2, 2,9), nanoBBox.max());
    EXPECT_EQ(openvdb::Coord(2, 2,9), openBBox.max());
    EXPECT_EQ(2u, nanoGrid->activeVoxelCount());
    EXPECT_EQ(2u, openGrid->activeVoxelCount());
} // NanoToOpenVDB_Basic

TEST_F(TestOpenVDB, NanoToOpenVDB)
{
    //mTimer.start("Reading NanoVDB grids from file");
    auto handles = nanovdb::io::readGrids("data/test.nvdb");
    //mTimer.stop();

    EXPECT_EQ(1u, handles.size());
    auto* srcGrid = handles.front().grid<float>();
    EXPECT_TRUE(srcGrid);

    //std::cerr << "Grid name: " << srcGrid->gridName() << std::endl;

    //mTimer.start("Deserializing NanoVDB grid");
    auto dstGrid = nanovdb::tools::nanoToOpenVDB(*srcGrid);
    //mTimer.stop();
    EXPECT_TRUE(dstGrid);

    //dstGrid->print(std::cout, 3);

    auto kernel = [&](const openvdb::CoordBBox& bbox) {
        using CoordT = const nanovdb::Coord;
        auto dstAcc = dstGrid->getAccessor();
        auto srcAcc = srcGrid->getAccessor();
        for (auto it = bbox.begin(); it; ++it) {
            EXPECT_EQ(srcAcc.getValue(reinterpret_cast<CoordT&>(*it)), dstAcc.getValue(*it));
        }
    };

    //mTimer.start("Parallel unit test");
    tbb::parallel_for(dstGrid->evalActiveVoxelBoundingBox(), kernel);
    //mTimer.stop();

    //mTimer.start("Testing bounding box");
    const auto srcBBox = srcGrid->indexBBox();
    const auto dstBBox = dstGrid->evalActiveVoxelBoundingBox();
    EXPECT_EQ(dstBBox.min()[0], srcBBox.min()[0]);
    EXPECT_EQ(dstBBox.min()[1], srcBBox.min()[1]);
    EXPECT_EQ(dstBBox.min()[2], srcBBox.min()[2]);
    EXPECT_EQ(dstBBox.max()[0], srcBBox.max()[0]);
    EXPECT_EQ(dstBBox.max()[1], srcBBox.max()[1]);
    EXPECT_EQ(dstBBox.max()[2], srcBBox.max()[2]);
    //mTimer.stop();

    EXPECT_EQ(srcGrid->activeVoxelCount(), dstGrid->activeVoxelCount());
} // NanoToOpenVDB

TEST_F(TestOpenVDB, File)
{
    auto srcGrid = this->getSrcGrid();

    //mTimer.start("Reading NanoVDB grids from file");
    auto handles = nanovdb::io::readGrids("data/test.nvdb");
    //mTimer.stop();

    EXPECT_EQ(1u, handles.size());

    auto* dstGrid = handles[0].grid<float>();
    EXPECT_TRUE(dstGrid);

    EXPECT_TRUE(handles[0].data());
    EXPECT_TRUE(handles[0].size() > 0);

    auto kernel = [&](const openvdb::CoordBBox& bbox) {
        using CoordT = const nanovdb::Coord;
        auto dstAcc = dstGrid->getAccessor();
        auto srcAcc = srcGrid->getUnsafeAccessor(); // not registered
        for (auto it = bbox.begin(); it; ++it) {
            EXPECT_EQ(dstAcc.getValue(reinterpret_cast<CoordT&>(*it)), srcAcc.getValue(*it));
        }
    };

    //mTimer.start("Parallel unit test");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    //mTimer.stop();

    //mTimer.start("Testing bounding box");
    const auto& dstBBox = dstGrid->indexBBox();
    const auto  srcBBox = srcGrid->evalActiveVoxelBoundingBox();
    EXPECT_EQ(dstBBox.min()[0], srcBBox.min()[0]);
    EXPECT_EQ(dstBBox.min()[1], srcBBox.min()[1]);
    EXPECT_EQ(dstBBox.min()[2], srcBBox.min()[2]);
    EXPECT_EQ(dstBBox.max()[0], srcBBox.max()[0]);
    EXPECT_EQ(dstBBox.max()[1], srcBBox.max()[1]);
    EXPECT_EQ(dstBBox.max()[2], srcBBox.max()[2]);
    //mTimer.stop();

    EXPECT_EQ(srcGrid->activeVoxelCount(), dstGrid->activeVoxelCount());
} // File

TEST_F(TestOpenVDB, MultiFile)
{
    std::vector<nanovdb::GridHandle<>> handles;
    { // 1: add an int32_t grid
        openvdb::Int32Grid grid(-1);
        grid.setName("Int32 grid");
        grid.tree().setValue(openvdb::Coord(-256), 10);
        EXPECT_EQ(1u, grid.activeVoxelCount());
        handles.push_back(nanovdb::tools::createNanoGrid(grid));
    }
    { // 2: add an empty int32_t grid
        openvdb::Int32Grid grid(-4);
        grid.setName("Int32 grid, empty");
        EXPECT_EQ(0u, grid.activeVoxelCount());
        handles.push_back(nanovdb::tools::createNanoGrid(grid));
    }
    { // 3: add a ValueMask grid
        openvdb::MaskGrid grid(false);
        grid.setName("Mask grid");
        const openvdb::Coord min(-10,-450,-90), max(10, 450, 90);
        grid.tree().setValue(min, true);
        EXPECT_EQ(1u, grid.activeVoxelCount());
        grid.tree().setValue(max, true);
        EXPECT_EQ(2u, grid.activeVoxelCount());
        openvdb::CoordBBox bbox;
        grid.tree().evalActiveVoxelBoundingBox(bbox);
        //std::cerr << bbox << std::endl;
        EXPECT_EQ(openvdb::CoordBBox(min, max), bbox);
        handles.push_back(nanovdb::tools::createNanoGrid(grid));
    }
    { // 4: add a bool grid
        openvdb::BoolGrid grid(false);
        grid.setName("Bool grid");
        grid.tree().setValue(openvdb::Coord(-10,-450,-90), false);
        EXPECT_EQ(1u, grid.activeVoxelCount());
        grid.tree().setValue(openvdb::Coord( 10, 450, 90), true);
        EXPECT_EQ(2u, grid.activeVoxelCount());
        handles.push_back(nanovdb::tools::createNanoGrid(grid));
    }
    { // 5: add a Vec3f grid
        openvdb::Vec3fGrid grid(openvdb::Vec3f(0.0f, 0.0f, -1.0f));
        grid.setName("Float 3D vector grid");
        grid.setGridClass(openvdb::GRID_STAGGERED);
        EXPECT_EQ(0u, grid.activeVoxelCount());
        grid.tree().setValue(openvdb::Coord(-256), openvdb::Vec3f(1.0f, 0.0f, 0.0f));
        EXPECT_EQ(1u, grid.activeVoxelCount());
        handles.push_back(nanovdb::tools::createNanoGrid(grid));
    }
    { // 6: add a Vec4f grid
        using OpenVDBVec4fGrid = openvdb::Grid<openvdb::tree::Tree4<openvdb::Vec4f, 5, 4, 3>::Type>;
        OpenVDBVec4fGrid::registerGrid();// this gid type is not registered by default in OpenVDB
        OpenVDBVec4fGrid grid(openvdb::Vec4f(0.0f, 0.0f, 0.0f, -1.0f));
        grid.setName("Float 4D vector grid");
        grid.setGridClass(openvdb::GRID_STAGGERED);
        EXPECT_EQ(0u, grid.activeVoxelCount());
        grid.tree().setValue(openvdb::Coord(-256), openvdb::Vec4f(1.0f, 0.0f, 0.0f, 0.0f));
        EXPECT_EQ(1u, grid.activeVoxelCount());
        handles.push_back(nanovdb::tools::createNanoGrid(grid));
        OpenVDBVec4fGrid::unregisterGrid();
    }
    { // 7: add an int64_t grid
        openvdb::Int64Grid grid(0);
        grid.setName("Int64 grid");
        grid.tree().setValue(openvdb::Coord(0), 10);
        EXPECT_EQ(1u, grid.activeVoxelCount());
        handles.push_back(nanovdb::tools::createNanoGrid(grid));
    }
    for (int i = 0; i < 10; ++i) {// 8 -> 17
        const float          radius = 100.0f;
        const float          voxelSize = 1.0f, width = 3.0f;
        const openvdb::Vec3f center(i * 10.0f, 0.0f, 0.0f);
        auto                 srcGrid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center, voxelSize, width);
        srcGrid->setName("Level set sphere at (" + std::to_string(i * 10) + ",0,0)");
        handles.push_back(nanovdb::tools::createNanoGrid(*srcGrid));
    }
    { // 18: add a double grid
        openvdb::DoubleGrid grid(0.0);
        grid.setName("Double grid");
        grid.setGridClass(openvdb::GRID_FOG_VOLUME);
        grid.tree().setValue(openvdb::Coord(6000), 1.0);
        EXPECT_EQ(1u, grid.activeVoxelCount());
        handles.push_back(nanovdb::tools::createNanoGrid(grid));
    }

    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>("data/multi.nvdb", handles, this->getCodec());

    { // read grid meta data and test it
        //mTimer.start("nanovdb::io::readGridMetaData");
        auto meta = nanovdb::io::readGridMetaData("data/multi.nvdb");
        //mTimer.stop();
        EXPECT_EQ(18u, meta.size());
        EXPECT_EQ(std::string("Double grid"), meta.back().gridName);
    }
    { // read in32 grid and test values
        //mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        //mTimer.stop();
        EXPECT_EQ(18u, handles.size());
        auto& handle = handles.front();
        EXPECT_EQ(std::string("Int32 grid"), handle.gridMetaData()->shortGridName());
        EXPECT_FALSE(handle.grid<float>());
        EXPECT_FALSE(handle.grid<double>());
        EXPECT_FALSE(handle.grid<int64_t>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3f>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3d>());
        auto* grid = handle.grid<int32_t>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(std::string("Int32 grid"), grid->gridName());
        EXPECT_EQ(1u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(-256);
        const auto&          tree = grid->tree();
        EXPECT_EQ(10, tree.getValue(ijk));
        EXPECT_EQ(-1, tree.getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(10, tree.root().minimum());
        EXPECT_EQ(10, tree.root().maximum());
        EXPECT_EQ(10, tree.root().average());
        EXPECT_TRUE(grid->tree().isActive(ijk));
        EXPECT_FALSE(grid->tree().isActive(nanovdb::Coord( 10, 450, 90)));
        EXPECT_FALSE(grid->tree().isActive(nanovdb::Coord(-10,-450,-90)));
        EXPECT_FALSE(grid->tree().isActive(ijk + nanovdb::Coord(1, 0, 0)));
        const nanovdb::CoordBBox bbox(ijk, ijk);
        EXPECT_EQ(bbox, grid->indexBBox());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(1u, tree.nodeCount(0));
        EXPECT_EQ(1u, tree.nodeCount(1));
        EXPECT_EQ(1u, tree.nodeCount(2));
        auto mgrHandle = nanovdb::createNodeManager(*grid);
        auto *mgr = mgrHandle.mgr<int32_t>();
        EXPECT_TRUE(nanovdb::isAligned(mgr));
        const auto& leaf = mgr->leaf(0);
        EXPECT_TRUE(nanovdb::isAligned(&leaf));
        EXPECT_EQ(bbox, leaf.bbox());
        const auto& node1 = mgr->lower(0);
        EXPECT_TRUE(nanovdb::isAligned(&node1));
        EXPECT_EQ(bbox, node1.bbox());
        const auto& node2 = mgr->upper(0);
        EXPECT_TRUE(nanovdb::isAligned(&node2));
        EXPECT_EQ(bbox, node2.bbox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
    { // read empty in32 grid and test values
        //mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        //mTimer.stop();
        EXPECT_EQ(18u, handles.size());
        auto& handle = handles[1];
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Int32 grid, empty"), handle.gridMetaData()->shortGridName());
        EXPECT_FALSE(handle.grid<float>());
        EXPECT_FALSE(handle.grid<double>());
        EXPECT_FALSE(handle.grid<int64_t>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3f>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3d>());
        auto* grid = handle.grid<int32_t>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(std::string("Int32 grid, empty"), grid->gridName());
        EXPECT_EQ(0u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(-256);
        EXPECT_EQ(-4, grid->tree().getValue(ijk));
        EXPECT_EQ(-4, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_FALSE(grid->tree().isActive(ijk));
        EXPECT_FALSE(grid->tree().isActive(nanovdb::Coord( 10, 450, 90)));
        EXPECT_FALSE(grid->tree().isActive(nanovdb::Coord(-10,-450,-90)));
        EXPECT_FALSE(grid->tree().isActive(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(-4, grid->tree().root().minimum());// background value
        EXPECT_EQ(-4, grid->tree().root().maximum());// background value
        EXPECT_EQ(nanovdb::Coord(std::numeric_limits<int>::max()), grid->indexBBox().min());
        EXPECT_EQ(nanovdb::Coord(std::numeric_limits<int>::min()), grid->indexBBox().max());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(0u, grid->tree().nodeCount(0));
        EXPECT_EQ(0u, grid->tree().nodeCount(1));
        EXPECT_EQ(0u, grid->tree().nodeCount(2));
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_FALSE(grid->isMask());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
    { // read mask grid and test values
        //mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        //mTimer.stop();
        EXPECT_EQ(18u, handles.size());
        auto& handle = handles[2];
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Mask grid"), handle.gridMetaData()->shortGridName());
        EXPECT_FALSE(handle.grid<float>());
        EXPECT_FALSE(handle.grid<double>());
        EXPECT_FALSE(handle.grid<int64_t>());
        EXPECT_FALSE(handle.grid<int32_t>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3f>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3d>());
        auto* grid = handle.grid<nanovdb::ValueMask>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(std::string("Mask grid"), grid->gridName());
        EXPECT_EQ(2u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(-256);
        EXPECT_EQ(false, grid->tree().getValue(ijk));
        EXPECT_EQ(true, grid->tree().getValue(nanovdb::Coord( 10, 450, 90)));
        EXPECT_EQ(true, grid->tree().getValue(nanovdb::Coord(-10,-450,-90)));
        EXPECT_EQ(false, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_FALSE(grid->tree().isActive(ijk));
        EXPECT_TRUE(grid->tree().isActive(nanovdb::Coord( 10, 450, 90)));
        EXPECT_TRUE(grid->tree().isActive(nanovdb::Coord(-10,-450,-90)));
        EXPECT_FALSE(grid->tree().isActive(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(false, grid->tree().root().minimum());
        EXPECT_EQ(false, grid->tree().root().maximum());
        EXPECT_EQ(nanovdb::Coord(-10,-450,-90), grid->indexBBox().min());
        EXPECT_EQ(nanovdb::Coord( 10, 450, 90), grid->indexBBox().max());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(2u, grid->tree().nodeCount(0));
        EXPECT_EQ(2u, grid->tree().nodeCount(1));
        EXPECT_EQ(2u, grid->tree().nodeCount(2));
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_FALSE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
        EXPECT_TRUE(grid->isMask());
    }
    { // read bool grid and test values
        //mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        //mTimer.stop();
        EXPECT_EQ(18u, handles.size());
        auto& handle = handles[3];
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Bool grid"), handle.gridMetaData()->shortGridName());
        EXPECT_FALSE(handle.grid<float>());
        EXPECT_FALSE(handle.grid<double>());
        EXPECT_FALSE(handle.grid<int64_t>());
        EXPECT_FALSE(handle.grid<int32_t>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3f>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3d>());
        EXPECT_FALSE(handle.grid<nanovdb::ValueMask>());
        auto* grid = handle.grid<bool>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(std::string("Bool grid"), grid->gridName());
        EXPECT_EQ(2u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(-256);
        EXPECT_EQ(false, grid->tree().getValue(ijk));
        EXPECT_EQ(true, grid->tree().getValue(nanovdb::Coord( 10, 450, 90)));
        EXPECT_EQ(false, grid->tree().getValue(nanovdb::Coord(-10,-450,-90)));
        EXPECT_EQ(false, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_FALSE(grid->tree().isActive(ijk));
        EXPECT_TRUE(grid->tree().isActive(nanovdb::Coord( 10, 450, 90)));
        EXPECT_TRUE(grid->tree().isActive(nanovdb::Coord(-10,-450,-90)));
        EXPECT_FALSE(grid->tree().isActive(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(false, grid->tree().root().minimum());
        EXPECT_EQ(false, grid->tree().root().maximum());
        EXPECT_EQ(nanovdb::Coord(-10,-450,-90), grid->indexBBox().min());
        EXPECT_EQ(nanovdb::Coord( 10, 450, 90), grid->indexBBox().max());
        EXPECT_NE(nanovdb::Coord(std::numeric_limits<int>::max()), grid->indexBBox().min());
        EXPECT_NE(nanovdb::Coord(std::numeric_limits<int>::min()), grid->indexBBox().max());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(2u, grid->tree().nodeCount(0));
        EXPECT_EQ(2u, grid->tree().nodeCount(1));
        EXPECT_EQ(2u, grid->tree().nodeCount(2));
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
        EXPECT_FALSE(grid->isMask());
    }
    { // read vec3f grid and test values
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        EXPECT_EQ(18u, handles.size());
        auto& handle = handles[4];
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Float 3D vector grid"), handle.gridMetaData()->shortGridName());
        auto* grid = handle.grid<nanovdb::Vec3f>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(std::string("Float 3D vector grid"), grid->gridName());
        EXPECT_EQ(1u, grid->activeVoxelCount());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        const nanovdb::Coord ijk(-256);
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 0.0f, 0.0f), grid->tree().getValue(ijk));
        EXPECT_EQ(nanovdb::Vec3f(0.0f, 0.0f, -1.0f), grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 0.0f, 0.0f), grid->tree().root().minimum());
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 0.0f, 0.0f), grid->tree().root().maximum());
        EXPECT_EQ(nanovdb::CoordBBox(ijk, ijk), grid->indexBBox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_FALSE(grid->isUnknown());
        EXPECT_TRUE(grid->isStaggered());
    }
    { // read vec4f grid and test values
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        EXPECT_EQ(18u, handles.size());
        auto& handle = handles[5];
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Float 4D vector grid"), handle.gridMetaData()->shortGridName());
        auto* grid = handle.grid<nanovdb::Vec4f>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(std::string("Float 4D vector grid"), grid->gridName());
        EXPECT_EQ(1u, grid->activeVoxelCount());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        const nanovdb::Coord ijk(-256);
        EXPECT_EQ(nanovdb::Vec4f(1.0f, 0.0f, 0.0f, 0.0f), grid->tree().getValue(ijk));
        EXPECT_EQ(nanovdb::Vec4f(0.0f, 0.0f, 0.0f, -1.0f), grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(nanovdb::Vec4f(1.0f, 0.0f, 0.0f, 0.0f), grid->tree().root().minimum());
        EXPECT_EQ(nanovdb::Vec4f(1.0f, 0.0f, 0.0f, 0.0f), grid->tree().root().maximum());
        EXPECT_EQ(nanovdb::CoordBBox(ijk, ijk), grid->indexBBox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_FALSE(grid->isUnknown());
        EXPECT_TRUE(grid->isStaggered());
    }
     { // read int64 grid and test values
        //mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        //mTimer.stop();
        EXPECT_EQ(18u, handles.size());
        auto& handle = handles[6];
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Int64 grid"), handle.gridMetaData()->shortGridName());
        auto* grid = handle.grid<int64_t>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(std::string("Int64 grid"), grid->gridName());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(1u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(0);
        EXPECT_EQ(10, grid->tree().getValue(ijk));
        EXPECT_EQ(0, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(10, grid->tree().root().minimum());
        EXPECT_EQ(10, grid->tree().root().maximum());
        EXPECT_EQ(10, grid->tree().root().average());
        EXPECT_EQ(nanovdb::CoordBBox(ijk, ijk), grid->indexBBox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
    { // read double grid and test values
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        EXPECT_EQ(18u, handles.size());
        auto& handle = handles.back();
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Double grid"), handle.gridMetaData()->shortGridName());
        auto* grid = handle.grid<double>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(std::string("Double grid"), grid->gridName());
        EXPECT_EQ(1u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(6000);
        EXPECT_EQ(1.0, grid->tree().getValue(ijk));
        EXPECT_EQ(0.0, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(1.0, grid->tree().root().minimum());
        EXPECT_EQ(1.0, grid->tree().root().maximum());
        EXPECT_EQ(1.0, grid->tree().root().average());
        EXPECT_EQ(nanovdb::CoordBBox(ijk, ijk), grid->tree().bbox());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_TRUE(grid->isFogVolume());
        EXPECT_FALSE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
} // MultiFile

TEST_F(TestOpenVDB, LongGridName)
{
    for (int n = -10; n <= 10; ++n) {
        openvdb::FloatGrid srcGrid(0.0f);
        const int limit = nanovdb::GridData::MaxNameSize - 1, length = limit + n;
        char buffer[limit + 10 + 1] = {'\0'};
        srand (time(NULL));
        for (int i = 0; i < length; ++i) {
            buffer[i] = 'a' + (rand() % 26);// a-z
        }
        buffer[length] = '\0';
        const std::string gridName(buffer);
        //std::cout << "Long random grid name: " << gridName << std::endl;
        EXPECT_EQ(gridName.length(), size_t(length));
        srcGrid.setName(gridName);
        EXPECT_EQ(gridName, srcGrid.getName());
        srcGrid.tree().setValue(openvdb::Coord(-256), 10.0f);
        EXPECT_EQ(1u, srcGrid.activeVoxelCount());
        const bool isLong = length > limit;
#if 1
        auto handle = nanovdb::tools::createNanoGrid(srcGrid);
#else
        nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> converter(srcGrid);
        auto handle = converter.getHandle<float>();
#endif
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ(1u, dstGrid->activeVoxelCount());
        EXPECT_EQ(isLong ? 1u : 0u, dstGrid->blindDataCount());
        EXPECT_EQ(isLong, dstGrid->hasLongGridName());
        //std::cerr << "\nHas long grid name: " << (isLong?"yes":"no") << std::endl;
        //std::cerr << "length = " << length << ", limit = " << limit << std::endl;
        EXPECT_EQ(gridName, std::string(dstGrid->gridName()));
        EXPECT_EQ( !isLong, std::string(dstGrid->shortGridName()) == std::string(dstGrid->gridName()) );
        EXPECT_EQ( 0.0, dstGrid->tree().getValue(nanovdb::Coord(-255)));
        EXPECT_EQ(10.0, dstGrid->tree().getValue(nanovdb::Coord(-256)));
    }
}// LongGridName

TEST_F(TestOpenVDB, LevelSetFiles)
{
    const auto fileNames = this->availableLevelSetFiles();
    if (fileNames.empty()) {
        std::cout << "\tSet the environment variable \"VDB_DATA_PATH\" to a directory\n"
                  << "\tcontaining OpenVDB level set files. They can be downloaded\n"
                  << "\there: https://www.openvdb.org/download/" << std::endl;
        return;
    }

    std::vector<std::string> foundModels;
    std::ofstream            os("data/ls.nvdb", std::ios::out | std::ios::binary);
    for (const auto& fileName : fileNames) {
        //mTimer.start("\nReading grid from the file \"" + fileName + "\"");
        try {
            openvdb::io::File file(fileName);
            file.open(false); //disable delayed loading
            auto srcGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(file.beginName().gridName()));

            const size_t pos = fileName.find_last_of("/\\") + 1;
            foundModels.push_back(fileName.substr(pos, fileName.size() - pos - 4 ));

            //mTimer.restart("Generating NanoVDB grid");
            //auto handle = nanovdb::tools::createNanoGrid(*srcGrid, nanovdb::tools::StatsMode::All, nanovdb::CheckMode::Partial);
            auto handle = nanovdb::tools::createNanoGrid(*srcGrid, nanovdb::tools::StatsMode::BBox, nanovdb::CheckMode::Disable);
            //mTimer.restart("Writing NanoVDB grid");

            nanovdb::io::writeGrid(os, handle, this->getCodec());

        } catch(const std::exception& e) {
            std::cerr << "Skipping " << fileName << "\n";
        }
        //mTimer.stop();
    }
    os.close();

    if (foundModels.size() == 0) {
        return;
    }

    auto getGridName = [](const std::string& name) -> std::string {
        if (name == "torus_knot_helix") { // special case
            return "TorusKnotHelix";
        } else {
            return std::string("ls_") + name;
        }
    };

    //mTimer.start("Read GridMetaData from file");
    auto meta = nanovdb::io::readGridMetaData("data/ls.nvdb");
    //mTimer.stop();
    EXPECT_EQ(foundModels.size(), meta.size());
    for (size_t i = 0; i < foundModels.size(); ++i) {
        EXPECT_EQ(nanovdb::GridType::Float, meta[i].gridType);
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta[i].gridClass);
        EXPECT_EQ(getGridName(foundModels[i]), meta[i].gridName);
    }

    // test reading from non-existing file
    EXPECT_THROW(nanovdb::io::readGrid("data/ls.vdb", getGridName(foundModels[0])), std::runtime_error);

    // test reading of non-existing grid from an existing file
    EXPECT_THROW(nanovdb::io::readGrid("data/ls.nvdb", "bunny"), std::runtime_error);

    // test reading existing grid from an existing file
    {
        auto gridName = getGridName(foundModels[0]);
        auto handle = nanovdb::io::readGrid("data/ls.nvdb", gridName);
        EXPECT_TRUE(handle);
        EXPECT_FALSE(handle.grid<double>());
        auto grid = handle.grid<float>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(nanovdb::GridType::Float, grid->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, grid->gridClass());
        EXPECT_EQ(gridName, std::string(grid->gridName()));
    }
} // LevelSetFiles

TEST_F(TestOpenVDB, FogFiles)
{
    const auto fileNames = this->availableFogFiles();
    if (fileNames.empty()) {
        std::cout << "\tSet the environment variable \"VDB_DATA_PATH\" to a directory\n"
                  << "\tcontaining OpenVDB fog volume files. They can be downloaded\n"
                  << "\there: https://www.openvdb.org/download/" << std::endl;
        return;
    }

    std::vector<std::string> foundModels;
    std::ofstream            os("data/fog.nvdb", std::ios::out | std::ios::binary);
    for (const auto& fileName : fileNames) {
        //mTimer.start("Reading grid from the file \"" + fileName + "\"");
        try {
            openvdb::io::File file(fileName);
            file.open(false); //disable delayed loading
            auto srcGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(file.beginName().gridName()));

            const size_t pos = fileName.find_last_of("/\\") + 1;
            foundModels.push_back(fileName.substr(pos, fileName.size() - pos - 4 ));

            //mTimer.restart("Generating NanoVDB grid");
            auto handle = nanovdb::tools::createNanoGrid(*srcGrid, nanovdb::tools::StatsMode::All, nanovdb::CheckMode::Partial);
            //mTimer.restart("Writing NanoVDB grid");
            nanovdb::io::writeGrid(os, handle, this->getCodec());

        } catch(const std::exception& e) {
            std::cerr << "Skipping " << fileName << "\n";
        }
        //mTimer.stop();
    }
    os.close();

    if (foundModels.size() == 0) {
        return;
    }

    auto getGridName = [](const std::string&){return std::string("density");};

    //mTimer.start("Read GridMetaData from file");
    auto meta = nanovdb::io::readGridMetaData("data/fog.nvdb");
    //mTimer.stop();
    EXPECT_EQ(foundModels.size(), meta.size());
    for (size_t i = 0; i < foundModels.size(); ++i) {
        EXPECT_EQ(nanovdb::GridType::Float, meta[i].gridType);
        EXPECT_EQ(nanovdb::GridClass::FogVolume, meta[i].gridClass);
        EXPECT_EQ(getGridName(foundModels[i]), meta[i].gridName);
    }

    // test reading from non-existing file
    EXPECT_THROW(nanovdb::io::readGrid("data/fog.vdb", getGridName(foundModels[0])), std::runtime_error);

    // test reading of non-existing grid from an existing file
    EXPECT_THROW(nanovdb::io::readGrid("data/fog.nvdb", "bunny"), std::runtime_error);

    // test reading existing grid from an existing file
    {
        //const std::string gridName("density");
        auto gridName = getGridName(foundModels[0]);
        auto handle = nanovdb::io::readGrid("data/fog.nvdb", gridName);
        EXPECT_TRUE(handle);
        EXPECT_FALSE(handle.grid<double>());
        auto grid = handle.grid<float>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(nanovdb::GridType::Float, grid->gridType());
        EXPECT_EQ(nanovdb::GridClass::FogVolume, grid->gridClass());
        EXPECT_EQ(gridName, std::string(grid->gridName()));
    }
} // FogFiles

TEST_F(TestOpenVDB, PointFiles)
{
    const auto fileNames = this->availablePointFiles();
    if (fileNames.empty()) {
        std::cout << "\tSet the environment variable \"VDB_DATA_PATH\" to a directory\n"
                  << "\tcontaining OpenVDB files with points. They can be downloaded\n"
                  << "\there: https://www.openvdb.org/download/" << std::endl;
        return;
    }

    std::ofstream os("data/points.nvdb", std::ios::out | std::ios::binary);
    for (const auto& fileName : fileNames) {
        //mTimer.start("Reading grid from the file \"" + fileName + "\"");
        try {
            openvdb::io::File file(fileName);
            file.open(false); //disable delayed loading

            auto srcGrid = openvdb::gridPtrCast<openvdb::points::PointDataGrid>(file.readGrid(file.beginName().gridName()));
            //std::cerr << "Read PointDataGrid named \"" << srcGrid->getName() << "\"" << std::endl;
            EXPECT_TRUE(srcGrid.get());
            auto leaf = srcGrid->tree().cbeginLeaf();
            EXPECT_TRUE(leaf);
            const auto&  attributeSet = leaf->attributeSet();
            const size_t positionIndex = attributeSet.find("P");
            EXPECT_TRUE(positionIndex != openvdb::points::AttributeSet::INVALID_POS);

            //mTimer.restart("Generating NanoVDB grid from PointDataGrid");
            auto handle = nanovdb::tools::createNanoGrid(*srcGrid);
            //mTimer.restart("Writing NanoVDB grid");
            nanovdb::io::writeGrid(os, handle, this->getCodec());

            //mTimer.stop();
            EXPECT_TRUE(handle);
            auto* meta = handle.gridMetaData();
            EXPECT_TRUE(meta);
            EXPECT_EQ(nanovdb::GridType::UInt32, meta->gridType());
            EXPECT_EQ(nanovdb::GridClass::PointData, meta->gridClass());
            auto dstGrid = handle.grid<uint32_t>();
            EXPECT_TRUE(dstGrid);

            nanovdb::PointAccessor<nanovdb::Vec3f> acc(*dstGrid);
            EXPECT_TRUE(acc);
            const nanovdb::Vec3f *                 begin = nullptr, *end = nullptr; // iterators over points in a given voxel
            EXPECT_EQ(acc.gridPoints(begin, end), openvdb::points::pointCount(srcGrid->tree()));
            //std::cerr << "Point count = " << acc.gridPoints(begin, end) << ", attribute count = " << attributeSet.size() << std::endl;
            for (auto leafIter = srcGrid->tree().cbeginLeaf(); leafIter; ++leafIter) {
                EXPECT_TRUE(leafIter->hasAttribute("P")); // Check position attribute from the leaf by name (P is position).
                // Create a read-only AttributeHandle. Position always uses Vec3f.
                openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(leafIter->constAttributeArray("P"));
                openvdb::Coord ijkSrc(openvdb::Coord::min());
                nanovdb::Coord ijkDst(nanovdb::math::Maximum<int>::value());
                for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
                    // Extract the index-space position of the point relative to its occupying voxel ijk.
                    const openvdb::Vec3f vxlSrc = positionHandle.get(*indexIter);
                    if (ijkSrc != indexIter.getCoord()) { // new voxel
                        ijkSrc = indexIter.getCoord();
                        for (int i=0; i<3; ++i) ijkDst[i] = ijkSrc[i];
                        EXPECT_TRUE(acc.isActive(ijkDst));
                        EXPECT_TRUE(acc.voxelPoints(ijkDst, begin, end));
                    }
                    EXPECT_NE(nullptr, begin);
                    EXPECT_NE(nullptr, end);
                    EXPECT_TRUE(begin < end);
                    const nanovdb::Vec3f vxlDst = *begin++;// local voxel coordinates
                    for (int i=0; i<3; ++i) {
                        EXPECT_EQ( ijkSrc[i], ijkDst[i] );
                        EXPECT_EQ( vxlSrc[i], vxlDst[i] );
                    }
                    // A PointDataGrid encodes local voxel coordinates
                    // so transform those to global index coordinates!
                    const openvdb::Vec3f idxSrc = ijkSrc.asVec3s() + vxlSrc;
                    const nanovdb::Vec3f idxDst = ijkDst.asVec3s() + vxlDst;

                    // Transform global index coordinates to global world coordinates
                    const openvdb::Vec3f wldSrc = srcGrid->indexToWorld(idxSrc);
                    const nanovdb::Vec3f wldDst = dstGrid->indexToWorld(idxDst);

                    //std::cerr << "voxel = " << vxlDst << ", index = " << idxDst << ", world = " << wldDst << std::endl;
                    for (int i = 0; i < 3; ++i) {
                        EXPECT_EQ( idxSrc[i], idxDst[i] );
                        EXPECT_EQ( wldSrc[i], wldDst[i] );
                    }
                }
            }
        } catch(const std::exception& e) {
            std::cerr << "Skipping " << fileName << "\n";
        }
    }
} // PointFiles

TEST_F(TestOpenVDB, Trilinear)
{
    // create a grid so sample from
    auto trilinear = [](const openvdb::Vec3d& xyz) -> float {
        return 0.34 + 1.6 * xyz[0] + 6.7 * xyz[1] - 3.5 * xyz[2]; // world coordinates
    };

    //mTimer.start("Generating a dense tri-linear openvdb grid");
    auto        srcGrid = openvdb::createGrid<openvdb::FloatGrid>(/*background=*/1.0f);
    const float voxelSize = 0.5f;
    srcGrid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));
    const openvdb::CoordBBox bbox(openvdb::Coord(0), openvdb::Coord(128));
    auto                     acc = srcGrid->getAccessor();
    for (auto iter = bbox.begin(); iter; ++iter) {
        auto ijk = *iter;
        acc.setValue(ijk, trilinear(srcGrid->indexToWorld(ijk)));
    }
    //mTimer.restart("Generating NanoVDB grid");
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid);
    //mTimer.restart("Writing NanoVDB grid");
    nanovdb::io::writeGrid("data/tmp.nvdb", handle);
    //mTimer.stop();
    handle.reset();
    EXPECT_FALSE(handle.grid<float>());
    EXPECT_FALSE(handle.grid<double>());

    //mTimer.start("Reading NanoVDB from file");
    auto handles = nanovdb::io::readGrids("data/tmp.nvdb");
    //mTimer.stop();
    EXPECT_EQ(1u, handles.size());
    auto* dstGrid = handles[0].grid<float>();
    EXPECT_TRUE(dstGrid);
    EXPECT_FALSE(handles[0].grid<double>());
    EXPECT_EQ(voxelSize, dstGrid->voxelSize()[0]);

    const openvdb::Vec3d ijk(13.4, 24.67, 5.23); // in index space
    const float          exact = trilinear(srcGrid->indexToWorld(ijk));
    const float          approx = trilinear(srcGrid->indexToWorld(openvdb::Coord(13, 25, 5)));
    //std::cerr << "Trilinear: exact = " << exact << ", approx = " << approx << std::endl;

    auto dstAcc = dstGrid->getAccessor();
    auto sampler0 = nanovdb::math::createSampler<0>(dstAcc);
    //std::cerr << "0'th order: v = " << sampler0(ijk) << std::endl;
    EXPECT_EQ(approx, sampler0(ijk));

    auto sampler1 = nanovdb::math::createSampler<1>(dstAcc); // faster since it's using an accessor!!!
    //std::cerr << "1'th order: v = " << sampler1(ijk) << std::endl;
    EXPECT_EQ(exact, sampler1(ijk));

    EXPECT_FALSE(sampler1.zeroCrossing());
    const auto gradIndex = sampler1.gradient(ijk); //in index space
    EXPECT_NEAR(1.6f, gradIndex[0] / voxelSize, 1e-5);
    EXPECT_NEAR(6.7f, gradIndex[1] / voxelSize, 1e-5);
    EXPECT_NEAR(-3.5f, gradIndex[2] / voxelSize, 1e-5);
    const auto gradWorld = dstGrid->indexToWorldGrad(gradIndex); // in world units
    EXPECT_NEAR(1.6f, gradWorld[0], 1e-5);
    EXPECT_NEAR(6.7f, gradWorld[1], 1e-5);
    EXPECT_NEAR(-3.5f, gradWorld[2], 1e-5);

    nanovdb::math::SampleFromVoxels<nanovdb::NanoTree<float>, 3> sampler3(dstGrid->tree());
    //auto sampler3 = nanovdb::math::createSampler<3>( dstAcc );
    //std::cerr << "3'rd order: v = " << sampler3(ijk) << std::endl;
    EXPECT_EQ(exact, sampler3(ijk));
} // Trilinear

TEST_F(TestOpenVDB, Triquadratic)
{
    // create a grid so sample from
    auto triquadratic = [](const openvdb::Vec3d& xyz) -> double {
        return 0.34 + 1.6 * xyz[0] + 2.7 * xyz[1] + 1.5 * xyz[2] +
               0.025 * xyz[0] * xyz[1] * xyz[2] - 0.013 * xyz[0] * xyz[0]; // world coordinates
    };

    //mTimer.start("Generating a dense tri-cubic openvdb grid");
    auto srcGrid = openvdb::createGrid<openvdb::DoubleGrid>(/*background=*/1.0);
    srcGrid->setName("Tri-Quadratic");
    srcGrid->setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.5));
    const openvdb::CoordBBox bbox(openvdb::Coord(0), openvdb::Coord(128));
    auto                     acc = srcGrid->getAccessor();
    for (auto iter = bbox.begin(); iter; ++iter) {
        auto ijk = *iter;
        acc.setValue(ijk, triquadratic(srcGrid->indexToWorld(ijk)));
    }
    //mTimer.restart("Generating NanoVDB grid");
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid);
    //mTimer.restart("Writing NanoVDB grid");
    nanovdb::io::writeGrid("data/tmp.nvdb", handle);
    //mTimer.stop();

    { //test File::hasGrid
        EXPECT_TRUE(nanovdb::io::hasGrid("data/tmp.nvdb", "Tri-Quadratic"));
        EXPECT_FALSE(nanovdb::io::hasGrid("data/tmp.nvdb", "Tri-Linear"));
    }

    //mTimer.start("Reading NanoVDB from file");
    auto handles = nanovdb::io::readGrids("data/tmp.nvdb");
    //mTimer.stop();
    auto* dstGrid = handles[0].grid<double>();
    EXPECT_TRUE(dstGrid);

    const openvdb::Vec3d ijk(3.4, 4.67, 5.23); // in index space
    const float          exact = triquadratic(srcGrid->indexToWorld(ijk));
    const float          approx = triquadratic(srcGrid->indexToWorld(openvdb::Coord(3, 5, 5)));
    //std::cerr << "Trilinear: exact = " << exact << ", approx = " << approx << std::endl;
    auto dstAcc = dstGrid->getAccessor();

    auto sampler0 = nanovdb::math::createSampler<0>(dstAcc);
    //std::cerr << "0'th order: v = " << sampler0(ijk) << std::endl;
    EXPECT_NEAR(approx, sampler0(ijk), 1e-6);

    auto sampler1 = nanovdb::math::createSampler<1>(dstAcc);
    //std::cerr << "1'rd order: nanovdb = " << sampler1(ijk) << ", openvdb: " << openvdb::tools::Sampler<1>::sample(srcGrid->tree(), ijk) << std::endl;
    EXPECT_NE(exact, sampler1(ijk)); // it's non-linear
    EXPECT_NEAR(sampler1(ijk), openvdb::tools::Sampler<1>::sample(srcGrid->tree(), ijk), 1e-6);

    auto sampler2 = nanovdb::math::createSampler<2>(dstAcc);
    //std::cerr << "2'rd order: nanovdb = " << sampler2(ijk) << ", openvdb: " << openvdb::tools::Sampler<2>::sample(srcGrid->tree(), ijk) << std::endl;
    EXPECT_NEAR(sampler2(ijk), openvdb::tools::Sampler<2>::sample(srcGrid->tree(), ijk), 1e-6);
    EXPECT_NEAR(exact, sampler2(ijk), 1e-5); // it's a 2nd order polynomial

    auto sampler3 = nanovdb::math::createSampler<3>(dstAcc);
    //std::cerr << "3'rd order: v = " << sampler3(ijk) << std::endl;
    EXPECT_NEAR(exact, sampler3(ijk), 1e-4); // it's a 2nd order polynomial
} // Triquadratic

TEST_F(TestOpenVDB, Tricubic)
{
    // create a grid so sample from
    auto tricubic = [](const openvdb::Vec3d& xyz) -> double {
        return 0.34 + 1.6 * xyz[0] + 2.7 * xyz[1] + 1.5 * xyz[2] + 0.025 * xyz[0] * xyz[1] * xyz[2] - 0.013 * xyz[0] * xyz[0] * xyz[0]; // world coordinates
    };

    //mTimer.start("Generating a dense tri-cubic openvdb grid");
    auto srcGrid = openvdb::createGrid<openvdb::DoubleGrid>(/*background=*/1.0);
    srcGrid->setName("Tri-Cubic");
    srcGrid->setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.5));
    const openvdb::CoordBBox bbox(openvdb::Coord(0), openvdb::Coord(128));
    auto                     acc = srcGrid->getAccessor();
    for (auto iter = bbox.begin(); iter; ++iter) {
        auto ijk = *iter;
        acc.setValue(ijk, tricubic(srcGrid->indexToWorld(ijk)));
    }
    //mTimer.restart("Generating NanoVDB grid");
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid);
    //mTimer.restart("Writing NanoVDB grid");
    nanovdb::io::writeGrid("data/tmp.nvdb", handle);
    //mTimer.stop();

    { //test File::hasGrid
        EXPECT_TRUE(nanovdb::io::hasGrid("data/tmp.nvdb", "Tri-Cubic"));
        EXPECT_FALSE(nanovdb::io::hasGrid("data/tmp.nvdb", "Tri-Linear"));
    }

    //mTimer.start("Reading NanoVDB from file");
    auto handles = nanovdb::io::readGrids("data/tmp.nvdb");
    //mTimer.stop();
    auto* dstGrid = handles[0].grid<double>();
    EXPECT_TRUE(dstGrid);

    const openvdb::Vec3d ijk(3.4, 4.67, 5.23); // in index space
    const float          exact = tricubic(srcGrid->indexToWorld(ijk));
    const float          approx = tricubic(srcGrid->indexToWorld(openvdb::Coord(3, 5, 5)));
    //std::cerr << "Trilinear: exact = " << exact << ", approx = " << approx << std::endl;
    auto dstAcc = dstGrid->getAccessor();

    auto sampler0 = nanovdb::math::createSampler<0>(dstAcc);
    //std::cerr << "0'th order: v = " << sampler0(ijk) << std::endl;
    EXPECT_NEAR(approx, sampler0(ijk), 1e-6);

    auto sampler1 = nanovdb::math::createSampler<1>(dstAcc);
    //std::cerr << "1'rd order: nanovdb = " << sampler1(ijk) << ", openvdb: " << openvdb::tools::Sampler<1>::sample(srcGrid->tree(), ijk) << std::endl;
    EXPECT_NE(exact, sampler1(ijk)); // it's non-linear
    EXPECT_NEAR(sampler1(ijk), openvdb::tools::Sampler<1>::sample(srcGrid->tree(), ijk), 1e-6);

    auto sampler2 = nanovdb::math::createSampler<2>(dstAcc);
    //std::cerr << "2'rd order: nanovdb = " << sampler2(ijk) << ", openvdb: " << openvdb::tools::Sampler<2>::sample(srcGrid->tree(), ijk) << std::endl;
    EXPECT_NEAR(sampler2(ijk), openvdb::tools::Sampler<2>::sample(srcGrid->tree(), ijk), 1e-6);
    EXPECT_NE(exact, sampler2(ijk)); // it's a 3nd order polynomial

    auto sampler3 = nanovdb::math::createSampler<3>(dstAcc);
    //std::cerr << "3'rd order: v = " << sampler3(ijk) << std::endl;
    EXPECT_NEAR(exact, sampler3(ijk), 1e-4); // it's a 3nd order polynomial
} // Tricubic

TEST_F(TestOpenVDB, GridValidator)
{
    auto srcGrid = this->getSrcGrid();
    auto handle = nanovdb::tools::createNanoGrid(*srcGrid, nanovdb::tools::StatsMode::All, nanovdb::CheckMode::Full);
    //mTimer.stop();
    EXPECT_TRUE(handle);
    EXPECT_TRUE(handle.data());
    auto* grid = handle.grid<float>();
    EXPECT_TRUE(grid);

    //mTimer.start("isValid - detailed");
    EXPECT_TRUE(nanovdb::tools::isValid(grid, nanovdb::CheckMode::Full, true));
    //mTimer.stop();

    //mTimer.start("isValid - not detailed");
    EXPECT_TRUE(nanovdb::tools::isValid(grid, nanovdb::CheckMode::Partial, true));
    //mTimer.stop();

    //mTimer.start("Fast CRC");
    auto fastChecksum = nanovdb::tools::evalChecksum(grid, nanovdb::CheckMode::Full);
    //mTimer.stop();
    EXPECT_EQ(fastChecksum, nanovdb::tools::evalChecksum(grid, nanovdb::CheckMode::Full));

    auto* leaf = grid->tree().getFirstLeaf();
    EXPECT_TRUE(nanovdb::isAligned(leaf));
    leaf->data()->mValues[512 >> 1] += 0.00001f; // slightly modify a single voxel value

    EXPECT_NE(fastChecksum, nanovdb::tools::evalChecksum(grid, nanovdb::CheckMode::Full));
    EXPECT_FALSE(nanovdb::tools::isValid(grid, nanovdb::CheckMode::Full, false));

    leaf->data()->mValues[512 >> 1] -= 0.00001f; // change back the single voxel value to it's original value

    EXPECT_EQ(fastChecksum, nanovdb::tools::evalChecksum(grid, nanovdb::CheckMode::Full));
    EXPECT_TRUE(nanovdb::tools::isValid(grid, nanovdb::CheckMode::Full, true));

    leaf->data()->mValueMask.toggle(512 >> 1); // change a single bit in a value mask

    EXPECT_NE(fastChecksum, nanovdb::tools::evalChecksum(grid, nanovdb::CheckMode::Full));
    EXPECT_FALSE(nanovdb::tools::isValid(grid, nanovdb::CheckMode::Full, false));
} // GridValidator

TEST_F(TestOpenVDB, BenchmarkHostBuffer)
{
    mTimer.start();
    auto pool = nanovdb::HostBuffer::createPool( 1024 );// 1 KB
    const double ms1 = mTimer.milliseconds();
    EXPECT_LT(ms1, 1.0);// less than 1 millisecond
    //std::cout << "Construct 1 KB HostBuffer: " << ms1 << " milliseconds\n";

    EXPECT_TRUE(pool.isPool());
    EXPECT_TRUE(pool.isEmpty());
    EXPECT_EQ(1024U, pool.poolSize());

    mTimer.start();
    auto tmp = std::move( pool );
    const double ms2 = mTimer.milliseconds();
    EXPECT_LT(ms2, 1.0);// less than 1 millisecond
    //std::cout << "Moving 1 KB HostBuffer: " << ms2 << " milliseconds\n";

    EXPECT_FALSE(pool.isPool());
    EXPECT_TRUE( pool.isEmpty());
    EXPECT_EQ(0U, pool.poolSize());
    EXPECT_TRUE(tmp.isPool());
    EXPECT_TRUE(tmp.isEmpty());
    EXPECT_EQ(1024U, tmp.poolSize());
}// BenchmarkHostBuffer

TEST_F(TestOpenVDB, DenseIndexGrid)
{
    // read openvdb::FloatGrid
    auto srcGrid = this->getSrcGrid(false, 0, 0);// level set of a dragon if available, else an octahedron
    auto& srcTree = srcGrid->tree();
    nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> builder(*srcGrid);
    builder.setStats(nanovdb::tools::StatsMode::All);
    // openvdb::FloatGrid -> nanovdb::FloatGrid
    auto handle = builder.getHandle();
    EXPECT_TRUE(handle);
    auto* fltGrid = handle.grid<float>();
    builder.setStats();// reset
    //std::cerr << "FloatGrid footprint: " << (fltGrid->gridSize()>>20) << "MB" << std::endl;

    // openvdb::FloatGrid -> nanovdb::IndexGrid
    //mTimer.start("Create IndexGrid");
    auto handle2 = builder.getHandle<nanovdb::ValueIndex>(1u, true, true);
    //mTimer.stop();
    auto *idxGrid = handle2.grid<nanovdb::ValueIndex>();
    auto idxAcc = idxGrid->getAccessor();
    EXPECT_TRUE(idxGrid);
    const uint64_t vCount = idxGrid->data()->mData1;
    //std::cerr << "IndexGrid value count = " << vCount << std::endl;
    //std::cerr << "IndexGrid footprint: " << (idxGrid->gridSize()>>20) << "MB" << std::endl;

    // create external value buffer
    //mTimer.start("Create value buffer");
    const float *values = idxGrid->getBlindData<float>(0);
    EXPECT_TRUE(values);
    //mTimer.stop();
    //std::cerr << "Value buffer footprint: " << (buffer.size()>>20) << "MB" << std::endl;

    // unit-test dense value buffer
    //mTimer.start("Testing dense active values");
    for (auto it = srcTree.cbeginValueOn(); it; ++it) {
        const openvdb::Coord ijk = it.getCoord();
        const uint64_t n = idxAcc(ijk[0],ijk[1],ijk[2]);
        EXPECT_TRUE(n < vCount);
        EXPECT_EQ(it.getValue(), values[n]);
    }
    //mTimer.stop();
    auto *idxLeaf0 = idxGrid->tree().getFirstNode<0>();
    nanovdb::util::forEach(nanovdb::util::Range1D(0,idxGrid->tree().nodeCount(0)),[&](const nanovdb::util::Range1D &r){
        auto fltAcc = fltGrid->getAccessor();// NOT thread-safe!
        for (auto i=r.begin(); i!=r.end(); ++i){
            auto *idxLeaf = idxLeaf0 + i;
            auto *fltLeaf = fltAcc.probeLeaf(idxLeaf->origin());
            EXPECT_TRUE(fltLeaf);
            // since idxGrid was created from an OpenVDB Grid stats were not available
            EXPECT_EQ(values[idxLeaf->minimum()], srcGrid->tree().root().background());
            //EXPECT_EQ(values[idxLeaf->minimum()], fltLeaf->minimum());// only if idxGrid was created from fltGrid
            for (auto vox = idxLeaf->beginValueOn(); vox; ++vox) {
                EXPECT_EQ(values[*vox], fltAcc.getValue(vox.getCoord()));
            }
        }
    });
}// DenseIndexGrid

TEST_F(TestOpenVDB, SparseIndexGrid)
{
    // read openvdb::FloatGrid
    auto srcGrid = this->getSrcGrid(false, 0, 0);// level set of a dragon if available, else an octahedron

    // openvdb::FloatGrid -> nanovdb::IndexGrid
    nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> builder(*srcGrid);
    //mTimer.start("Create IndexGrid");
    auto handle2 = builder.getHandle<nanovdb::ValueIndex>(1u, false, false);
    //mTimer.stop();
    auto *idxGrid = handle2.grid<nanovdb::ValueIndex>();
    auto idxAcc = idxGrid->getAccessor();
    EXPECT_TRUE(idxGrid);
    const uint64_t vCount = idxGrid->valueCount();
    //std::cerr << "IndexGrid value count = " << vCount << std::endl;
    //std::cerr << "IndexGrid footprint: " << (idxGrid->gridSize()>>20) << "MB" << std::endl;

    // unit-test sparse value buffer
    const float *values = idxGrid->getBlindData<float>(0u);
    EXPECT_TRUE(values);
    //mTimer.start("Testing sparse active values");
    for (auto it = srcGrid->tree().cbeginValueOn(); it; ++it) {
        const openvdb::Coord ijk = it.getCoord();
        const uint64_t n = idxAcc(ijk[0], ijk[1], ijk[2]);
        EXPECT_TRUE(n < vCount);
        EXPECT_EQ(it.getValue(), values[n]);
    }
    //mTimer.stop();
}// SparseIndexGrid


TEST_F(TestOpenVDB, BuildNodeManager)
{
    {// test NodeManager with build::Grid
        using GridT = nanovdb::tools::build::Grid<float>;
        GridT grid(0.0f);
        nanovdb::tools::build::NodeManager<GridT> mgr(grid);
        using TreeT = GridT::TreeType;
        static const bool test = nanovdb::util::is_same<nanovdb::NodeTrait<TreeT,0>::type, TreeT::LeafNodeType>::value;
        EXPECT_TRUE(test);
    }
    {// test NodeManager with openvdb::Grid
        using GridT = openvdb::FloatGrid;
        GridT grid(0.0f);
        nanovdb::tools::build::NodeManager<GridT> mgr(grid);
        using TreeT = GridT::TreeType;
        static const bool test = nanovdb::util::is_same<nanovdb::NodeTrait<TreeT,0>::type, TreeT::LeafNodeType>::value;
        EXPECT_TRUE(test);
    }
    {// test NodeTrait on nanovdb::Grid
        using GridT = nanovdb::NanoGrid<float>;
        using TreeT = GridT::TreeType;
        static const bool test = nanovdb::util::is_same<nanovdb::NodeTrait<TreeT,0>::type, TreeT::LeafNodeType>::value;
        EXPECT_TRUE(test);
    }
}// BuildNodeManager

#if 0// toggle to enable benchmark tests

class NanoPointList
{
    size_t mSize;
    const openvdb::Vec3f *mPoints;
public:
    using PosType    = openvdb::Vec3f;
    using value_type = openvdb::Vec3f;
    NanoPointList(const nanovdb::Vec3f *points, size_t size) : mSize(size), mPoints(reinterpret_cast<const openvdb::Vec3f*>(points)) {}
    size_t size() const {return mSize;}
    void getPos(size_t n, PosType& xyz) const {xyz = mPoints[n];}
}; // NanoPointList

// make -j  && ./unittest/testNanoVDB --gtest_filter="*CudaPointsToGrid_PointID" --gtest_repeat=3 && ./unittest/testOpenVDB --gtest_filter="*PointIndexGrid*" --gtest_repeat=3
TEST_F(TestOpenVDB, Benchmark_OpenVDB_PointIndexGrid)
{
    const double voxelSize = 0.5;

    nanovdb::util::Timer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), voxelSize);
    timer.stop();

    auto *pointGrid = pointsHandle.grid<uint32_t>();
    EXPECT_TRUE(pointGrid);
    std::cerr << "nanovdb::bbox = " << pointGrid->indexBBox() << " voxel count = " << pointGrid->activeVoxelCount() << std::endl;

    nanovdb::PointAccessor<nanovdb::Vec3f, uint32_t> acc2(*pointGrid);
    EXPECT_TRUE(acc2);
    const nanovdb::Vec3f *begin, *end;
    const size_t pointCount = acc2.gridPoints(begin, end);
    EXPECT_TRUE(begin);
    EXPECT_TRUE(end);
    EXPECT_LT(begin, end);

    // construct data structure
    timer.start("Building openvdb::PointIndexGrid on CPU from "+std::to_string(pointCount)+" points");
    using PointIndexGrid = openvdb::tools::PointIndexGrid;
    const openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(voxelSize);
    NanoPointList pointList(begin, pointCount);
    auto pointGridPtr = openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);
    timer.stop();
    openvdb::CoordBBox bbox;
    pointGridPtr->tree().evalActiveVoxelBoundingBox(bbox);
    std::cerr << "openvdb::bbox = " << bbox << " voxel count = " << pointGridPtr->tree().activeVoxelCount() << std::endl;

}// Benchmark_OpenVDB_PointIndexGrid

TEST_F(TestOpenVDB, Benchmark_OpenVDB_PointDataGrid)
{
    const double voxelSize = 0.5;

    nanovdb::util::Timer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), voxelSize);
    timer.stop();

    auto *pointGrid = pointsHandle.grid<uint32_t>();
    EXPECT_TRUE(pointGrid);
    std::cerr << "nanovdb::bbox = " << pointGrid->indexBBox() << " voxel count = " << pointGrid->activeVoxelCount() << std::endl;

    nanovdb::PointAccessor<nanovdb::Vec3f, uint32_t> acc2(*pointGrid);
    EXPECT_TRUE(acc2);
    const nanovdb::Vec3f *begin, *end;
    const size_t pointCount = acc2.gridPoints(begin, end);
    EXPECT_TRUE(begin);
    EXPECT_TRUE(end);
    EXPECT_LT(begin, end);

    // construct data structure
    timer.start("Building openvdb::PointDataGrid on CPU from "+std::to_string(pointCount)+" points");
    using PointIndexGrid = openvdb::tools::PointIndexGrid;
    const auto transform = openvdb::math::Transform::createLinearTransform(voxelSize);
    NanoPointList pointList(begin, pointCount);
    auto pointIndexGridPtr = openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);
    auto pointDataGridPtr = openvdb::points::createPointDataGrid<openvdb::points::FixedPointCodec<true>,// corresponds to PointType::Voxel8
                            openvdb::points::PointDataGrid, NanoPointList, PointIndexGrid>(*pointIndexGridPtr, pointList, *transform);
    timer.stop();
    openvdb::CoordBBox bbox;
    pointDataGridPtr->tree().evalActiveVoxelBoundingBox(bbox);
    std::cerr << "openvdb::bbox = " << bbox << " voxel count = " << pointDataGridPtr->tree().activeVoxelCount() << std::endl;

}// Benchmark_OpenVDB_PointDataGrid
#endif

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
