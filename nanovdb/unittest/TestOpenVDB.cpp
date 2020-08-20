// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <iostream>
#include <cstdlib>
#include <sstream> // for std::stringstream
#define _USE_MATH_DEFINES
#include <cmath>

#include "gtest/gtest.h"

#include <nanovdb/util/IO.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/NanoToOpenVDB.h>
#include <nanovdb/util/SampleFromVoxels.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/CNanoVDB.h>
#include <nanovdb/util/CSampleFromVoxels.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/PointIndexGrid.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/util/CpuTimer.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

// define the enviroment variable VDB_DATA_PATH to use models from the web
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
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    std::string getEnvVar(const std::string& name) const
    {
        const char* str = std::getenv(name.c_str());
        return str == nullptr ? std::string("") : std::string(str);
    }

    openvdb::FloatGrid::Ptr getSrcGrid(int verbose = 1)
    {
        openvdb::FloatGrid::Ptr grid;
        const std::string       path = this->getEnvVar("VDB_DATA_PATH");
        if (path.empty()) { // create a narrow-band level set sphere
            const float          radius = 100.0f;
            const openvdb::Vec3f center(0.0f, 0.0f, 0.0f);
            const float          voxelSize = 1.0f, width = 3.0f;
            if (verbose > 0) {
                std::stringstream ss;
                ss << "Generating level set surface with a radius/size of " << radius << " voxels";
                mTimer.start(ss.str());
            }
#if 1 // choose between a sphere or one of five platonic solids
            grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center, voxelSize, width);
#else
            const int faces[5] = {4, 6, 8, 12, 20};
            grid = openvdb::tools::createLevelSetPlatonic<openvdb::FloatGrid>(faces[4], radius, center, voxelSize, width);
#endif
        } else {
            const std::vector<std::string> models = {"armadillo.vdb", "buddha.vdb", "bunny.vdb", "crawler.vdb", "dragon.vdb", "iss.vdb", "space.vdb", "torus_knot_helix.vdb", "utahteapot.vdb"};
            const std::string              fileName = path + "/" + models[4];
            if (verbose > 0)
                mTimer.start("Reading grid from the file \"" + fileName + "\"");
            openvdb::io::File file(fileName);
            file.open(false); //disable delayed loading
            grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(file.beginName().gridName()));
        }
        if (verbose > 0)
            mTimer.stop();
        if (verbose > 1)
            grid->print(std::cout, 3);
        return grid;
    }

    openvdb::util::CpuTimer mTimer;
}; // TestOpenVDB

TEST_F(TestOpenVDB, Grid)
{
    using LeafT = nanovdb::LeafNode<float>;
    using NodeT1 = nanovdb::InternalNode<LeafT>;
    using NodeT2 = nanovdb::InternalNode<NodeT1>;
    using RootT = nanovdb::RootNode<NodeT2>;
    using TreeT = nanovdb::Tree<RootT>;
    using GridT = nanovdb::Grid<TreeT>;
    using CoordT = LeafT::CoordType;

    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(nanovdb::GridData::MaxNameSize + 8 + 8 * 2 * 3 + sizeof(nanovdb::Map) + 8 + 2 + 2 + 4), sizeof(GridT));
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>((4 + 8) * (RootT::LEVEL + 1)), sizeof(TreeT));

    size_t bytes[6] = {GridT::memUsage(0), TreeT::memUsage(), RootT::memUsage(1), NodeT2::memUsage(), NodeT1::memUsage(), LeafT::memUsage()};
    for (int i = 1; i < 6; ++i)
        bytes[i] += bytes[i - 1]; // Byte offsets to: tree, root, internal nodes, leafs, total
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[bytes[5]]);

    // init leaf
    const LeafT* leaf = reinterpret_cast<LeafT*>(buffer.get() + bytes[4]);
    { // set members of the leaf node
        auto& data = *reinterpret_cast<LeafT::DataType*>(buffer.get() + bytes[4]);
        data.mValueMask.setOff();
        auto* voxels = data.mValues;
        for (uint32_t i = 0; i < LeafT::voxelCount() / 2; ++i)
            *voxels++ = 0.0f;
        for (uint32_t i = LeafT::voxelCount() / 2; i < LeafT::voxelCount(); ++i) {
            data.mValueMask.setOn(i);
            *voxels++ = 1.234f;
        }
        data.mValueMin = 0.0f;
        data.mValueMax = 1.234f;
    }

    // lower internal node
    const NodeT1* node1 = reinterpret_cast<NodeT1*>(buffer.get() + bytes[3]);
    { // set members of the  internal node
        auto& data = *reinterpret_cast<NodeT1::DataType*>(buffer.get() + bytes[3]);
        auto* tiles = data.mTable;
        data.mValueMask.setOff();
        data.mChildMask.setOff();
        data.mChildMask.setOn(0);
        tiles->childID = 0; // the leaf node resides right after this node
        for (uint32_t i = 1; i < NodeT1::SIZE / 2; ++i, ++tiles)
            tiles->value = 0.0f;
        for (uint32_t i = NodeT1::SIZE / 2; i < NodeT1::SIZE; ++i, ++tiles) {
            data.mValueMask.setOn(i);
            tiles->value = 1.234f;
        }
        data.mValueMin = 0.0f;
        data.mValueMax = 1.234f;
        data.mOffset = 1;
        EXPECT_EQ(leaf, data.child(0));
    }

    // upper internal node
    const NodeT2* node2 = reinterpret_cast<NodeT2*>(buffer.get() + bytes[2]);
    { // set members of the  internal node
        auto& data = *reinterpret_cast<NodeT2::DataType*>(buffer.get() + bytes[2]);
        auto* tiles = data.mTable;
        data.mValueMask.setOff();
        data.mChildMask.setOff();
        data.mChildMask.setOn(0);
        tiles->childID = 0; // the leaf node resides right after this node
        for (uint32_t i = 1; i < NodeT2::SIZE / 2; ++i, ++tiles)
            tiles->value = 0.0f;
        for (uint32_t i = NodeT2::SIZE / 2; i < NodeT2::SIZE; ++i, ++tiles) {
            data.mValueMask.setOn(i);
            tiles->value = 1.234f;
        }
        data.mValueMin = 0.0f;
        data.mValueMax = 1.234f;
        data.mOffset = 1;
        EXPECT_EQ(node1, data.child(0));
    }

    // init root
    RootT* root = reinterpret_cast<RootT*>(buffer.get() + bytes[1]);
    { // set members of the root node
        auto& data = *reinterpret_cast<RootT::DataType*>(buffer.get() + bytes[1]);
        data.mBackground = data.mValueMin = data.mValueMax = 1.234f;
        data.mTileCount = 1;
        auto& tile = data.tile(0);
        tile.setChild(RootT::CoordType(0), 0);
    }

    // init tree
    TreeT* tree = reinterpret_cast<TreeT*>(buffer.get() + bytes[0]);
    {
        auto& data = *reinterpret_cast<TreeT::DataType*>(buffer.get() + bytes[0]);
        data.mCount[0] = data.mCount[1] = data.mCount[2] = data.mCount[3] = 1;
        data.mBytes[0] = bytes[4] - bytes[0];
        data.mBytes[1] = bytes[3] - bytes[0];
        data.mBytes[2] = bytes[2] - bytes[0];
        data.mBytes[3] = bytes[1] - bytes[0];
    }

    GridT* grid = reinterpret_cast<GridT*>(buffer.get());
    { // init Grid
        auto* data = reinterpret_cast<GridT::DataType*>(buffer.get());
        {
            openvdb::math::UniformScaleTranslateMap map(2.0, openvdb::Vec3R(0.0, 0.0, 0.0));
            auto                                    affineMap = map.getAffineMap();
            data->mUniformScale = affineMap->voxelSize()[0];
            const auto mat = affineMap->getMat4(), invMat = mat.inverse();
            //for (int i=0; i<4; ++i) std::cout << "Row("<<i<<"): ["<<mat[i][0]<<", "<<mat[i][1]<<", "<<mat[i][2]<<", "<<mat[i][3]<<"]\n";
            //for (int i=0; i<4; ++i) std::cout << "Row("<<i<<"): ["<<invMat[i][0]<<", "<<invMat[i][1]<<", "<<invMat[i][2]<<", "<<invMat[i][3]<<"]\n";
            data->mMap.set(mat, invMat, 1.0);
            data->mGridClass = nanovdb::GridClass::Unknown;
            data->mGridType = nanovdb::GridType::Float;
            const std::string name("");
            memcpy(data->mGridName, name.c_str(), name.size() + 1);
        }
        EXPECT_EQ(tree, &grid->tree());
        const openvdb::Vec3R p1(1.0, 2.0, 3.0);
        const auto           p2 = grid->worldToIndex(p1);
        EXPECT_EQ(openvdb::Vec3R(0.5, 1.0, 1.5), p2);
        const auto p3 = grid->indexToWorld(p2);
        EXPECT_EQ(p1, p3);
        {
            openvdb::math::UniformScaleTranslateMap map(2.0, p1);
            auto                                    affineMap = map.getAffineMap();
            data->mUniformScale = affineMap->voxelSize()[0];
            const auto mat = affineMap->getMat4(), invMat = mat.inverse();
            //for (int i=0; i<4; ++i) std::cout << "Row("<<i<<"): ["<<mat[i][0]<<", "<<mat[i][1]<<", "<<mat[i][2]<<", "<<mat[i][3]<<"]\n";
            //for (int i=0; i<4; ++i) std::cout << "Row("<<i<<"): ["<<invMat[i][0]<<", "<<invMat[i][1]<<", "<<invMat[i][2]<<", "<<invMat[i][3]<<"]\n";
            data->mMap.set(mat, invMat, 1.0);
        }

        auto const p4 = grid->worldToIndex(p3);
        EXPECT_EQ(openvdb::Vec3R(0.0, 0.0, 0.0), p4);
        const auto p5 = grid->indexToWorld(p4);
        EXPECT_EQ(p1, p5);

        EXPECT_EQ(nanovdb::GridClass::Unknown, grid->gridClass());
        EXPECT_EQ(nanovdb::GridType::Float, grid->gridType());
        //std::cerr << "\nName = \"" << grid->getName() << "\"" << std::endl;
        EXPECT_EQ("", std::string(grid->gridName()));
    }

    { // check leaf node
        auto* ptr = reinterpret_cast<LeafT::DataType*>(buffer.get() + bytes[4])->mValues;
        for (uint32_t i = 0; i < LeafT::voxelCount(); ++i) {
            if (i < 256) {
                EXPECT_FALSE(leaf->valueMask().isOn(i));
                EXPECT_EQ(0.0f, *ptr++);
            } else {
                EXPECT_TRUE(leaf->valueMask().isOn(i));
                EXPECT_EQ(1.234f, *ptr++);
            }
        }
        EXPECT_EQ(0.0f, leaf->valueMin());
        EXPECT_EQ(1.234f, leaf->valueMax());
    }

    { // check lower internal node
        auto* ptr = reinterpret_cast<NodeT1::DataType*>(buffer.get() + bytes[3])->mTable;
        EXPECT_TRUE(node1->childMask().isOn(0));
        for (uint32_t i = 1; i < NodeT1::SIZE; ++i, ++ptr) {
            EXPECT_FALSE(node1->childMask().isOn(i));
            if (i < NodeT1::SIZE / 2) {
                EXPECT_FALSE(node1->valueMask().isOn(i));
                EXPECT_EQ(0.0f, ptr->value);
            } else {
                EXPECT_TRUE(node1->valueMask().isOn(i));
                EXPECT_EQ(1.234f, ptr->value);
            }
        }
        EXPECT_EQ(0.0f, node1->valueMin());
        EXPECT_EQ(1.234f, node1->valueMax());
    }
    { // check upper internal node
        auto* ptr = reinterpret_cast<NodeT2::DataType*>(buffer.get() + bytes[2])->mTable;
        EXPECT_TRUE(node2->childMask().isOn(0));
        for (uint32_t i = 1; i < NodeT2::SIZE; ++i, ++ptr) {
            EXPECT_FALSE(node2->childMask().isOn(i));
            if (i < NodeT2::SIZE / 2) {
                EXPECT_FALSE(node2->valueMask().isOn(i));
                EXPECT_EQ(0.0f, ptr->value);
            } else {
                EXPECT_TRUE(node2->valueMask().isOn(i));
                EXPECT_EQ(1.234f, ptr->value);
            }
        }
        EXPECT_EQ(0.0f, node2->valueMin());
        EXPECT_EQ(1.234f, node2->valueMax());
    }
    { // check root
        EXPECT_EQ(1.234f, root->background());
        EXPECT_EQ(1.234f, root->valueMin());
        EXPECT_EQ(1.234f, root->valueMax());
        EXPECT_EQ(1u, root->tileCount());
        EXPECT_EQ(0.0f, root->getValue(CoordT(0, 0, 0)));
        EXPECT_EQ(1.234f, root->getValue(CoordT(7, 7, 7)));
    }
    { // check tree
        EXPECT_EQ(1.234f, tree->background());
        float a, b;
        tree->extrema(a, b);
        EXPECT_EQ(1.234f, a);
        EXPECT_EQ(1.234f, b);
        EXPECT_EQ(0.0f, tree->getValue(CoordT(0, 0, 0)));
        EXPECT_EQ(1.234f, tree->getValue(CoordT(7, 7, 7)));
        EXPECT_EQ(1u, tree->nodeCount<LeafT>());
        EXPECT_EQ(1u, tree->nodeCount<NodeT1>());
        EXPECT_EQ(1u, tree->nodeCount<NodeT2>());
        EXPECT_EQ(1u, tree->nodeCount<RootT>());
        EXPECT_EQ(reinterpret_cast<LeafT*>(buffer.get() + bytes[4]), tree->getNode<LeafT>(0));
        EXPECT_EQ(reinterpret_cast<NodeT1*>(buffer.get() + bytes[3]), tree->getNode<NodeT1>(0));
        EXPECT_EQ(reinterpret_cast<NodeT2*>(buffer.get() + bytes[2]), tree->getNode<NodeT2>(0));
    }

} // Grid

TEST_F(TestOpenVDB, Conversion)
{
    using SrcGridT = openvdb::FloatGrid;

    SrcGridT::Ptr srcGrid = this->getSrcGrid();

    using CoordT = openvdb::Coord;
    using ValueT = SrcGridT::ValueType;
    using SrcTreeT = SrcGridT::TreeType;
    static_assert(SrcTreeT::DEPTH == 4, "Converter assumes an OpenVDB tree of depth 4 (which it the default configuration)");
    using SrcRootT = SrcTreeT::RootNodeType; // OpenVDB root node
    using SrcNode2 = SrcRootT::ChildNodeType; // upper OpenVDB internal node
    using SrcNode1 = SrcNode2::ChildNodeType; // lower OpenVDB internal node
    using SrcNode0 = SrcNode1::ChildNodeType; // OpenVDB leaf node

    using DstNode0 = nanovdb::LeafNode<ValueT, CoordT, openvdb::util::NodeMask, SrcNode0::LOG2DIM>; // leaf
    using DstNode1 = nanovdb::InternalNode<DstNode0, SrcNode1::LOG2DIM>; // lower
    using DstNode2 = nanovdb::InternalNode<DstNode1, SrcNode2::LOG2DIM>; // upper
    using DstRootT = nanovdb::RootNode<DstNode2>;
    using DstTreeT = nanovdb::Tree<DstRootT>;
    using DstGridT = nanovdb::Grid<DstTreeT>;

    EXPECT_EQ(8u, DstNode0::dim());
    EXPECT_EQ(8 * 16u, DstNode1::dim());
    EXPECT_EQ(8 * 16 * 32u, DstNode2::dim());
    EXPECT_EQ(3, int(DstRootT::LEVEL));

    mTimer.start("extracting nodes from openvdb grid");
    const SrcTreeT&              srcTree = srcGrid->tree();
    const SrcRootT&              srcRoot = srcTree.root();
    std::vector<const SrcNode2*> array2; // upper OpenVDB internal nodes
    std::vector<const SrcNode1*> array1; // lower OpenVDB internal nodes
    std::vector<const SrcNode0*> array0; // OpenVDB leaf nodes
    array0.reserve(srcTree.leafCount()); // fast pre-allocation of OpenVDB leaf nodes (of which there are many)

#if 1
    tbb::parallel_invoke([&]() { srcTree.getNodes(array0); }, // multi-threaded population of node arrays from OpenVDB tree
                         [&]() { srcTree.getNodes(array1); },
                         [&]() { srcTree.getNodes(array2); });
    mTimer.stop();

#else
    tree.getNodes(array0);
    tree.getNodes(array1);
    tree.getNodes(array2);
#endif

#ifdef USE_SINGLE_ROOT_KEY
    mTimer.start("Sorting " + std::to_string(array2.size()) + " child nodes of the root node");
    {
        //EXPECT_TRUE( std::is_sorted(array2.begin(), array2.end(), comp) );// this is a fundamental assumption at the root level!
        struct Pair
        {
            const SrcNode2*          node;
            DstRootT::DataType::KeyT code;
            bool                     operator<(const Pair& rhs) const { return code < rhs.code; }
        };
        std::unique_ptr<Pair[]> tmp(new Pair[array2.size()]);
        using RangeT = tbb::blocked_range<uint32_t>;
        auto kernel = [&](const RangeT& r) {
            for (uint32_t i = r.begin(); i != r.end(); ++i) {
                const SrcNode2* node = array2[i];
                tmp[i].node = node;
                tmp[i].code = DstRootT::DataType::CoordToKey(node->origin());
            }
        };
        tbb::parallel_for(RangeT(0, array2.size()), kernel);
        tbb::parallel_sort(&tmp[0], &tmp[array2.size()]);
        EXPECT_TRUE(std::is_sorted(&tmp[0], &tmp[array2.size()]));
        tbb::parallel_for(RangeT(0, array2.size()), [&](const RangeT& r) {
            for (uint32_t i = r.begin(); i != r.end(); ++i)
                array2[i] = tmp[i].node;
        });
    }
    mTimer.stop();
#endif
    auto key = [](const SrcNode2* node) { return DstRootT::DataType::CoordToKey(node->origin()); };
    auto comp = [&key](const SrcNode2* a, const SrcNode2* b) { return key(a) < key(b); };
    EXPECT_TRUE(std::is_sorted(array2.begin(), array2.end(), comp)); // this is a fundamental assumption at the root level!
    for (size_t i = 0; i < array2.size() - 1; ++i) {
        if (!(key(array2[i]) < key(array2[i + 1])))
            std::cerr << "key(" << array2[i]->origin() << ")=" << key(array2[i])
                      << ", key(" << array2[i + 1]->origin() << ")=" << key(array2[i + 1]) << std::endl;
        EXPECT_TRUE(key(array2[i]) < key(array2[i + 1]));
    }

    EXPECT_EQ(srcTree.leafCount(), array0.size());
    //std::cerr << "Leaf nodes = " << array0.size() << ", lower internal nodes = " << array1.size() << ", upper internal nodes = " << array2.size() << std::endl;

    mTimer.start("allocating memory for NanoGrid");
    size_t bytes[6] = {DstGridT::memUsage(), // grid + bind meta data
                       DstTreeT::memUsage(), // tree
                       DstRootT::memUsage(srcRoot.getTableSize()), // root
                       array2.size() * DstNode2::memUsage(), //  upper internal nodes
                       array1.size() * DstNode1::memUsage(), //  lower internal nodes
                       array0.size() * DstNode0::memUsage()}; // leaf nodes
    for (int i = 1; i < 6; ++i)
        bytes[i] += bytes[i - 1]; // Byte offsets to: tree, root, nodes2, node1, node0=leafs, total
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[bytes[5]]);
    std::unique_ptr<int32_t[]> cache0(new int32_t[array0.size()]);

    EXPECT_EQ(srcTree.root().getTableSize(), array2.size()); //BUG - does not allow for tiles at the root level!!!!!

    mTimer.stop();
    DstGridT* dstGrid = reinterpret_cast<DstGridT*>(buffer.get());
    uint8_t * bufferBegin = buffer.get(), *bufferEnd = bufferBegin + bytes[5];

    openvdb::util::printBytes(std::cerr, bytes[5], "allocated", " for the NanoGrid\n");

    // Lambda expression to cache the x compoment of a Coord into x and
    // encode uint32_ id into x despite it being of type const int32_t
    auto cache = [](int32_t& x, const CoordT& ijk, uint32_t id) {
        x = ijk[0];
        reinterpret_cast<uint32_t&>(const_cast<CoordT&>(ijk)[0]) = id;
    };

    mTimer.start("Processing leaf nodes");
    { // process leaf nodes
        auto* start = reinterpret_cast<DstNode0::DataType*>(buffer.get() + bytes[4]);
        auto  op = [&](const tbb::blocked_range<uint32_t>& r) {
            int32_t* x = cache0.get() + r.begin();
            auto*    data = start + r.begin();
            for (auto i = r.begin(); i != r.end(); ++i, ++data) {
                EXPECT_TRUE((uint8_t*)data > bufferBegin && (uint8_t*)data < bufferEnd);
                const SrcNode0* srcLeaf = array0[i];
                data->mValueMask = srcLeaf->valueMask();
                const ValueT* src = srcLeaf->buffer().data();
                ValueT*       dst = data->mValues;
#if 0
        data->mValueMin = data->mValueMax = *src; *dst++ = *src++;// process first element
        for (int j=1; j<SrcNode0::size(); ++j) {
          if (*src < data->mValueMin) {
            data->mValueMin = *src; 
          } else if (*src > data->mValueMax) {
            data->mValueMax = *src;
          }
          *dst++ = *src++;
        }
#else
                for (uint32_t j = 0; j < SrcNode0::size(); ++j)
                    *dst++ = *src++; //copy all voxel values
                auto iter = srcLeaf->cbeginValueOn(); // iterate over active voxels
                assert(iter); //these should be at least one active voxel
                data->mValueMin = *iter, data->mValueMax = data->mValueMin;
                openvdb::CoordBBox bbox;
                for (; iter; ++iter) {
                    bbox.expand(srcLeaf->offsetToLocalCoord(iter.pos()));
                    const ValueT& v = *iter;
                    if (v < data->mValueMin) {
                        data->mValueMin = v;
                    } else if (v > data->mValueMax) {
                        data->mValueMax = v;
                    }
                }
                bbox.translate(srcLeaf->origin());
                data->mBBoxMin = bbox.min();
                for (int j = 0; j < 3; ++j)
                    data->mBBoxDif[j] = uint8_t(bbox.max()[j] - bbox.min()[j]);
#endif
                cache(*x++, srcLeaf->origin(), i);
            }
        };
#if 1
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, array0.size(), 8), op);
#else
        op(tbb::blocked_range<uint32_t>(0, array0.size()));
#endif
    }
    mTimer.stop();

    std::unique_ptr<int32_t[]> cache1(new int32_t[array1.size()]);

    mTimer.start("Processing lower internal nodes");
    {
        const uint32_t size = array1.size();
        auto*          start = reinterpret_cast<DstNode1::DataType*>(buffer.get() + bytes[3]);
        auto           op = [&](const tbb::blocked_range<size_t>& r) {
            int32_t* x = cache1.get() + r.begin();
            auto*    data = start + r.begin();
            for (auto i = r.begin(); i != r.end(); ++i, ++data) {
                EXPECT_TRUE((uint8_t*)data > bufferBegin && (uint8_t*)data < bufferEnd);
                const SrcNode1* srcNode = array1[i];
                cache(*x++, srcNode->origin(), i);
                data->mValueMask = srcNode->getValueMask();
                data->mChildMask = srcNode->getChildMask();
                data->mOffset = size - i;
                auto iter = srcNode->cbeginChildOn();
                assert(iter); // every internal node should have at least one child node
                const auto childID = static_cast<uint32_t>(iter->origin()[0]); // extract child id
                data->mTable[iter.pos()].childID = childID; // set child id
                const_cast<openvdb::Coord&>(iter->origin())[0] = cache0[childID]; // restore coordinate
                auto* dstChild = data->child(iter.pos());
                data->mValueMin = dstChild->valueMin();
                data->mValueMax = dstChild->valueMax();
                data->mBBox = dstChild->bbox();
#if 1
                for (++iter; iter; ++iter) {
                    const auto n = iter.pos();
                    const auto childID = static_cast<uint32_t>(iter->origin()[0]); // ID of leaf node
                    data->mTable[n].childID = childID;
                    const_cast<openvdb::Coord&>(iter->origin())[0] = cache0[childID]; // restore origin[0]
                    auto* dstChild = data->child(n);
                    if (dstChild->valueMin() < data->mValueMin)
                        data->mValueMin = dstChild->valueMin();
                    if (dstChild->valueMax() > data->mValueMax)
                        data->mValueMax = dstChild->valueMax();
                    const auto& bbox = dstChild->bbox();
                    data->mBBox.min().minComponent(bbox.min());
                    data->mBBox.max().maxComponent(bbox.max());
                }
                for (auto iter = srcNode->cbeginValueAll(); iter; ++iter)
                    data->mTable[iter.pos()].value = *iter;
                for (auto iter = srcNode->cbeginValueOn(); iter; ++iter) { // typically there are few active tiles
                    const auto& value = *iter;
                    if (value < data->mValueMin) {
                        data->mValueMin = value;
                    } else if (value > data->mValueMax) {
                        data->mValueMax = value;
                    }
                    data->mBBox.min().minComponent(iter.getCoord());
                    data->mBBox.max().maxComponent(iter.getCoord().offsetBy(SrcNode0::DIM - 1));
                }
#else
                for (auto iter = srcNode->cbeginChildAll(); iter; ++iter, ++dstTile) {
                    ValueT value{};
                    if (auto* srcChild = iter.probeChild(value)) {
                        dstTile->childID = static_cast<uint32_t>(srcChild->origin()[0]);
                        auto* dstChild = dstNode->child(iter.pos());
                        if (dstChild->min < min) {
                            min = dstChild->min;
                        } else if (dstChild->max > max) {
                            max = dstChild->max;
                        }
                    } else {
                        dstTile->value = value;
                        if (value < min) {
                            min = value;
                        } else if (value > max) {
                            max = value;
                        }
                    }
                }
#endif
            }
        };
#if 1
        tbb::parallel_for(tbb::blocked_range<size_t>(0, array1.size(), 4), op);
#else
        op(tbb::blocked_range<size_t>(0, array1.size()));
#endif
    }
    mTimer.stop();

    cache0.reset();
    std::unique_ptr<int32_t[]> cache2(new int32_t[array2.size()]);

    mTimer.start("Processing upper internal nodes");
    {
        const uint32_t size = array2.size();
        auto*          start = reinterpret_cast<DstNode2::DataType*>(buffer.get() + bytes[2]);
        auto           op = [&](const tbb::blocked_range<size_t>& r) {
            int32_t* x = cache2.get() + r.begin();
            auto*    data = start + r.begin();
            for (auto i = r.begin(); i != r.end(); ++i, ++data) {
                EXPECT_TRUE((uint8_t*)data > bufferBegin && (uint8_t*)data < bufferEnd);
                const SrcNode2* srcNode = array2[i];
                cache(*x++, srcNode->origin(), i);
                data->mValueMask = srcNode->getValueMask();
                data->mChildMask = srcNode->getChildMask();
                data->mOffset = size - i;
                auto iter = srcNode->cbeginChildOn();
                assert(iter); // every internal node should have at least one child node
                const auto childID = static_cast<uint32_t>(iter->origin()[0]);
                data->mTable[iter.pos()].childID = childID;
                const_cast<openvdb::Coord&>(iter->origin())[0] = cache1[childID];
                auto* dstChild = data->child(iter.pos());
                data->mValueMin = dstChild->valueMin();
                data->mValueMax = dstChild->valueMax();
                data->mBBox = dstChild->bbox();
#if 1
                for (++iter; iter; ++iter) {
                    const auto n = iter.pos();
                    const auto childID = static_cast<uint32_t>(iter->origin()[0]);
                    data->mTable[n].childID = childID;
                    const_cast<openvdb::Coord&>(iter->origin())[0] = cache1[childID]; // restore cached coordinate
                    auto* dstChild = data->child(n);
                    if (dstChild->valueMin() < data->mValueMin)
                        data->mValueMin = dstChild->valueMin();
                    if (dstChild->valueMax() > data->mValueMax)
                        data->mValueMax = dstChild->valueMax();
                    const auto& bbox = dstChild->bbox();
                    data->mBBox.min().minComponent(bbox.min());
                    data->mBBox.max().maxComponent(bbox.max());
                }
                for (auto iter = srcNode->cbeginValueAll(); iter; ++iter)
                    data->mTable[iter.pos()].value = *iter;
                for (auto iter = srcNode->cbeginValueOn(); iter; ++iter) { // typically there are few active tiles
                    const auto& value = *iter;
                    if (value < data->mValueMin) {
                        data->mValueMin = value;
                    } else if (value > data->mValueMax) {
                        data->mValueMax = value;
                    }
                    data->mBBox.min().minComponent(iter.getCoord());
                    data->mBBox.max().maxComponent(iter.getCoord().offsetBy(SrcNode1::DIM - 1));
                }
#else
                // process tile mTable
                for (auto iter = srcNode->cbeginChildAll(); iter; ++iter, ++dstTile) {
                    ValueT value{};
                    if (auto* srcChild = iter.probeChild(value)) {
                        dstTile->childID = static_cast<uint32_t>(srcChild->origin()[0]);
                        auto* dstChild = dstNode->child(iter.pos());
                        if (dstChild->min < min) {
                            min = dstChild->min;
                        } else if (dstChild->max > max) {
                            max = dstChild->max;
                        }
                    } else {
                        if (value < min) {
                            min = value;
                        } else if (value > max) {
                            max = value;
                        }
                        dstTile->value = value;
                    }
                }
#endif
            }
        };
#if 1
        tbb::parallel_for(tbb::blocked_range<size_t>(0, array2.size(), 4), op);
#else
        op(tbb::blocked_range<size_t>(0, array2.size()));
#endif
    }
    mTimer.stop();

    cache1.reset();

    mTimer.start("Processing Root node");
    { // process root node
        DstRootT* dstRoot = reinterpret_cast<DstRootT*>(buffer.get() + bytes[1]);
        auto&     data = *reinterpret_cast<DstRootT::DataType*>(buffer.get() + bytes[1]);
        EXPECT_TRUE((uint8_t*)dstRoot > bufferBegin && (uint8_t*)dstRoot < bufferEnd);
        //EXPECT_EQ((void*)&dstGrid->tree().root(), (void*)dstRoot);
        data.mBackground = srcRoot.background();
        data.mTileCount = srcRoot.getTableSize();
        // since openvdb::RootNode internally uses a std::map for child nodes its iterator
        // visits elements in the stored order required by the nanovdb::RootNode
        if (data.mTileCount == 0) { // empty root node
            data.mValueMin = data.mValueMax = data.mBackground;
            data.mBBox.min() = openvdb::Coord::max(); // set to an empty bounding box
            data.mBBox.max() = openvdb::Coord::min();
            data.mActiveVoxelCount = 0;
        } else {
            auto* node = array2[0];
            auto& tile = data.tile(0);
            EXPECT_TRUE((uint8_t*)&tile > bufferBegin && (uint8_t*)&tile < bufferEnd);
            const auto childID = static_cast<uint32_t>(node->origin()[0]);
            const_cast<openvdb::Coord&>(node->origin())[0] = cache2[childID]; // restore cached coordinate
            tile.setChild(node->origin(), childID);
            auto& dstChild = data.child(tile);
            data.mValueMin = dstChild.valueMin();
            data.mValueMax = dstChild.valueMax();
            data.mBBox = dstChild.bbox();
            for (size_t i = 1; i < array2.size(); ++i) {
                node = array2[i];
                auto& tile = data.tile(i);
                EXPECT_TRUE((uint8_t*)&tile > bufferBegin && (uint8_t*)&tile < bufferEnd);
                const auto childID = static_cast<uint32_t>(node->origin()[0]);
                const_cast<openvdb::Coord&>(node->origin())[0] = cache2[childID]; // restore cached coordinate
                tile.setChild(node->origin(), childID);
                auto& dstChild = data.child(tile);
                if (dstChild.valueMin() < data.mValueMin)
                    data.mValueMin = dstChild.valueMin();
                if (dstChild.valueMax() > data.mValueMax)
                    data.mValueMax = dstChild.valueMax();
                data.mBBox.min().minComponent(dstChild.bbox().min());
                data.mBBox.max().maxComponent(dstChild.bbox().max());
            }
            for (auto iter = srcRoot.cbeginValueAll(); iter; ++iter) { //
                std::cerr << "this is not working anymore" << std::endl;
                auto& tile = data.tile(iter.pos());
                tile.setValue(iter.getCoord(), iter.isValueOn(), *iter);
                if (iter.isValueOn()) {
                    if (tile.value < data.mValueMin) {
                        data.mValueMin = tile.value;
                    } else if (tile.value > data.mValueMax) {
                        data.mValueMax = tile.value;
                    }
                    data.mBBox.min().minComponent(iter.getCoord());
                    data.mBBox.max().maxComponent(iter.getCoord().offsetBy(SrcNode2::DIM - 1));
                }
            }
            data.mActiveVoxelCount = srcGrid->activeVoxelCount();
        }
        //dstRoot->printTable();
    }
    mTimer.stop();

    cache2.reset();

    mTimer.start("Processing Tree");
    {
        auto&          data = *reinterpret_cast<DstTreeT::DataType*>(buffer.get() + bytes[0]); // data for the tree
        const uint64_t count[4] = {array0.size(), array1.size(), array2.size(), 1};
        for (int i = 0; i < 4; ++i) {
            data.mCount[i] = count[i];
            data.mBytes[i] = bytes[4 - i] - bytes[0];
        }
    }

    mTimer.restart("Processing Grid");
    {
        EXPECT_TRUE(srcGrid->hasUniformVoxels());
        auto& data = *reinterpret_cast<DstGridT::DataType*>(dstGrid);
        { // affine transformation
            auto affineMap = srcGrid->transform().baseMap()->getAffineMap();
            data.mUniformScale = affineMap->voxelSize()[0];
            const auto mat = affineMap->getMat4();
            data.mMap.set(mat, mat.inverse(), 1.0);
            data.mBlindDataCount = 0;
        }
        switch (srcGrid->getGridClass()) { // set grid class
        case openvdb::GRID_LEVEL_SET:
            data.mGridClass = nanovdb::GridClass::LevelSet;
            break;
        case openvdb::GRID_FOG_VOLUME:
            data.mGridClass = nanovdb::GridClass::FogVolume;
            break;
        default:
            data.mGridClass = nanovdb::GridClass::Unknown;
        }
        data.mGridType = nanovdb::GridType::Float;
        { // set grid name
            const std::string name = srcGrid->getName();
            if (name.length() + 1 > nanovdb::GridData::MaxNameSize) {
                std::stringstream ss;
                ss << "Point attribute name \"" << name << "\" is more then " << nanovdb::GridData::MaxNameSize << " characters";
                OPENVDB_THROW(openvdb::ValueError, ss.str());
            }
            memcpy(data.mGridName, name.c_str(), name.size() + 1);
        }

        // check transform
        const openvdb::Vec3R xyz0(1.0, 2.0, 3.0);
        const openvdb::Vec3R xyz1 = srcGrid->worldToIndex(xyz0);
        const auto           xyz2 = dstGrid->worldToIndex(xyz0);
        EXPECT_EQ(xyz1[0], xyz2[0]);
        EXPECT_EQ(xyz1[1], xyz2[1]);
        EXPECT_EQ(xyz1[2], xyz2[2]);
        if (!openvdb::math::isApproxEqual(xyz1[0], xyz2[0]) ||
            !openvdb::math::isApproxEqual(xyz1[1], xyz2[1]) ||
            !openvdb::math::isApproxEqual(xyz1[2], xyz2[2])) {
            OPENVDB_THROW(openvdb::ValueError, "Converter only supports grids with uniform scaling and transform");
        }
        const auto xyz3 = dstGrid->indexToWorld(xyz2);
        EXPECT_EQ(xyz0[0], xyz3[0]);
        EXPECT_EQ(xyz0[1], xyz3[1]);
        EXPECT_EQ(xyz0[2], xyz3[2]);
    }
    mTimer.stop();

    EXPECT_TRUE(dstGrid->isLevelSet());
    //std::cout << "Grid names: \"" << srcGrid->getName() << "\" and \"" << dstGrid->getName() << "\"" << std::endl;
    EXPECT_EQ(srcGrid->getName(), std::string(dstGrid->gridName()));

    SrcGridT::Ptr srcGrid2 = this->getSrcGrid();

    mTimer.start("Checking nanovdb::Leaf values, origin and mask");
    { // check leaf nodes!
        DstNode0* dstLeaf = reinterpret_cast<DstNode0*>(buffer.get() + bytes[4]);
        //std::cerr << "Address of fist leaf = " << dstLeaf << std::endl;
        for (uint32_t i = 0; i < array0.size(); ++i, ++dstLeaf) {
            auto* srcLeaf = array0[i];
            EXPECT_EQ(srcLeaf->valueMask().countOn(), dstLeaf->valueMask().countOn());
            EXPECT_EQ(srcLeaf->origin(), dstLeaf->origin());
            for (int j = 0; j < 512; ++j)
                EXPECT_EQ(srcLeaf->buffer().data()[j], dstLeaf->voxels()[j]);
        }
    }
    mTimer.stop();

    mTimer.start("Checking NanoVDB lower internal values, origin and mask");
    { // check lower internal nodes!
        DstNode1* dstNode = reinterpret_cast<DstNode1*>(buffer.get() + bytes[3]);
        for (uint32_t i = 0; i < array1.size(); ++i, ++dstNode) {
            auto* srcNode = array1[i];
            EXPECT_EQ(srcNode->getValueMask().countOn(), dstNode->valueMask().countOn());
            EXPECT_EQ(srcNode->origin(), dstNode->origin());
            for (auto iter = srcNode->cbeginValueAll(); iter; ++iter)
                EXPECT_EQ(*iter, dstNode->getValue(iter.getCoord()));
        }
    }
    mTimer.stop();

    mTimer.start("Checking NanoVDB upper internal values, origin and mask");
    //std::cerr << "\nnumber of nodes = " << array2.size() << std::endl;
    { // check upper internal nodes!
        DstNode2* dstNode = reinterpret_cast<DstNode2*>(buffer.get() + bytes[2]);
        for (uint32_t i = 0; i < array2.size(); ++i, ++dstNode) {
            auto* srcNode = array2[i];
            //std::cerr << i << ", src: " << srcNode->origin() << ", dst: " << dstNode->origin() << ", min = " << dstNode->bboxMin() << std::endl;
            EXPECT_EQ(srcNode->getValueMask().countOn(), dstNode->valueMask().countOn());
            EXPECT_EQ(srcNode->origin(), dstNode->origin());
            for (auto iter = srcNode->cbeginValueAll(); iter; ++iter)
                EXPECT_EQ(*iter, dstNode->getValue(iter.getCoord()));
        }
    }
    mTimer.restart("Checking tree");
    {
        auto& tree = dstGrid->tree();
        EXPECT_EQ(array0.size(), tree.nodeCount<DstNode0>());
        EXPECT_EQ(array1.size(), tree.nodeCount<DstNode1>());
        EXPECT_EQ(array2.size(), tree.nodeCount<DstNode2>());
        EXPECT_EQ(1U, tree.nodeCount<DstRootT>());
        for (size_t i = 0; i < tree.nodeCount<DstNode0>(); ++i) {
            auto* leaf = tree.getNode<DstNode0>(i);
            EXPECT_EQ(array0[i]->origin(), leaf->origin());
            for (int j = 0; j < 512; ++j)
                EXPECT_EQ(array0[i]->getValue(j), leaf->getValue(j));
        }
        for (size_t i = 0; i < tree.nodeCount<DstNode1>(); ++i) {
            EXPECT_EQ(array1[i]->origin(), tree.getNode<DstNode1>(i)->origin());
        }
        for (size_t i = 0; i < tree.nodeCount<DstNode2>(); ++i) {
            EXPECT_EQ(array2[i]->origin(), tree.getNode<DstNode2>(i)->origin());
        }
    }

    // Check SrcGrid2 - should not be modified
    mTimer.restart("Checking that the source grid is un-modified");
    tbb::parallel_invoke(
        [&]() {for (auto iter = srcGrid2->cbeginValueOn(); iter; ++iter) {
            const auto ijk = iter.getCoord();
            const auto value = srcGrid->tree().getValue(ijk);
            EXPECT_EQ( *iter, value );
          }; },
        [&]() {for (size_t i=0; i<array0.size(); ++i) {
            const auto ijk = array0[i]->origin();
            auto *leaf = srcGrid2->tree().probeNode<SrcNode0>(ijk);
            EXPECT_TRUE(leaf);
            EXPECT_EQ( ijk, leaf->origin());
          }; },
        [&]() {for (size_t i=0; i<array1.size(); ++i) {
            const auto ijk = array1[i]->origin();
            auto *node = srcGrid2->tree().probeNode<SrcNode1>(ijk);
            EXPECT_TRUE(node);
            EXPECT_EQ( ijk, node->origin());
          }; },
        [&]() {for (size_t i=0; i<array2.size(); ++i) {
            const auto ijk = array2[i]->origin();
            auto *node = srcGrid2->tree().probeNode<SrcNode2>(ijk);
            EXPECT_TRUE(node);
            EXPECT_EQ( ijk, node->origin());
          }; });
    mTimer.stop();

    mTimer.start("Serial test of active values using a (slow) iterator");
    // Check Grid
    for (auto iter = srcGrid->cbeginValueOn(); iter; ++iter) {
        const auto ijk = iter.getCoord();
        const auto value = dstGrid->tree().getValue(ijk);
        EXPECT_EQ(*iter, value);
        EXPECT_TRUE(dstGrid->tree().isActive(iter.getCoord()));
        DstGridT::ValueType v;
        EXPECT_TRUE(dstGrid->tree().probeValue(iter.getCoord(), v));
        EXPECT_EQ(*iter, v);
    }
    mTimer.restart("Parallele test of dense values using Tree::getValue");
    {
        auto kernel = [&](const openvdb::CoordBBox& bbox) {
            for (auto it = bbox.begin(); it; ++it) {
                DstGridT::ValueType srcV = srcTree.getValue(*it), dstV;
                EXPECT_EQ(srcV, dstGrid->tree().getValue(*it));
                EXPECT_EQ(srcTree.isValueOn(*it), dstGrid->tree().isActive(*it));
                EXPECT_EQ(srcTree.isValueOn(*it), dstGrid->tree().probeValue(*it, dstV));
                EXPECT_EQ(srcV, dstV);
            }
        };
        tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    }
    mTimer.restart("src<->src: Parallel test of all values using ReadAccessor::getValue");
    {
        auto kernel = [&](const openvdb::CoordBBox& bbox) {
            auto srcAcc1 = srcGrid->getAccessor();
            auto srcAcc2 = srcGrid2->getUnsafeAccessor(); // not registered
            for (auto it = bbox.begin(); it; ++it) {
                EXPECT_EQ(srcAcc1.getValue(*it), srcAcc2.getValue(*it));
            }
        };
        tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    }
    mTimer.restart("dst<->src: Parallel test of all values using ReadAccessor::getValue");
    {
        auto kernel = [&](const openvdb::CoordBBox& bbox) {
            auto dstAcc = dstGrid->getAccessor();
            auto srcAcc = srcGrid->getUnsafeAccessor(); // not registered
            for (auto it = bbox.begin(); it; ++it) {
                DstGridT::ValueType srcV = srcAcc.getValue(*it), dstV;
                EXPECT_EQ(srcV, dstAcc.getValue(*it));
                EXPECT_EQ(srcAcc.isValueOn(*it), dstAcc.isActive(*it));
                EXPECT_EQ(srcAcc.isValueOn(*it), dstAcc.probeValue(*it, dstV));
                EXPECT_EQ(srcV, dstV);
            }
        };
        tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    }
    mTimer.stop();
} // Conversion

TEST_F(TestOpenVDB, OpenToNanoVDB)
{
    auto srcGrid = this->getSrcGrid();
    mTimer.start("Generating NanoVDB grid");
    auto handle = nanovdb::openToNanoVDB(*srcGrid, /*mortonSort*/ false, 2);
    mTimer.restart("Writing NanoVDB grid");
#if defined(NANOVDB_USE_BLOSC)
    nanovdb::io::writeGrid("data/test.nvdb", handle, nanovdb::io::Codec::BLOSC);
#elif defined(NANOVDB_USE_ZIP)
    nanovdb::io::writeGrid("data/test.nvdb", handle, nanovdb::io::Codec::ZIP);
#else
    nanovdb::io::writeGrid("data/test.nvdb", handle, nanovdb::io::Codec::NONE);
#endif
    mTimer.stop();

    auto kernel = [&](const openvdb::CoordBBox& bbox) {
        using CoordT = const nanovdb::Coord;
        auto dstAcc = handle.grid<float>()->getAccessor();
        auto srcAcc = srcGrid->getUnsafeAccessor(); // not registered
        for (auto it = bbox.begin(); it; ++it) {
            EXPECT_EQ(dstAcc.getValue(reinterpret_cast<CoordT&>(*it)), srcAcc.getValue(*it));
        }
    };

    mTimer.start("Parallel unit test");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    mTimer.stop();

    mTimer.start("Testing bounding box");
    auto dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
    const auto dstBBox = dstGrid->indexBBox();
    const auto srcBBox = srcGrid->evalActiveVoxelBoundingBox();
    EXPECT_EQ(dstBBox.min()[0], srcBBox.min()[0]);
    EXPECT_EQ(dstBBox.min()[1], srcBBox.min()[1]);
    EXPECT_EQ(dstBBox.min()[2], srcBBox.min()[2]);
    EXPECT_EQ(dstBBox.max()[0], srcBBox.max()[0]);
    EXPECT_EQ(dstBBox.max()[1], srcBBox.max()[1]);
    EXPECT_EQ(dstBBox.max()[2], srcBBox.max()[2]);
    mTimer.stop();

    EXPECT_EQ(srcGrid->activeVoxelCount(), dstGrid->activeVoxelCount());
} // Serializer

// Generate random points by uniformly distributing points
// on a unit-sphere.
inline void genPoints(const int numPoints, std::vector<openvdb::Vec3R>& points)
{
    openvdb::math::Random01 randNumber(0);
    const int               n = int(std::sqrt(double(numPoints)));
    const double            xScale = (2.0 * M_PI) / double(n);
    const double            yScale = M_PI / double(n);

    double         x, y, theta, phi;
    openvdb::Vec3R pos;

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
    std::vector<openvdb::Vec3R> const* const mPoints;

public:
    using PosType = openvdb::Vec3R;
    PointList(const std::vector<PosType>& points)
        : mPoints(&points)
    {
    }
    size_t size() const { return mPoints->size(); }
    void   getPos(size_t n, PosType& xyz) const { xyz = (*mPoints)[n]; }
}; // PointList

TEST_F(TestOpenVDB, PointIndex)
{
    const uint64_t pointCount = 40000;
    const float    voxelSize = 0.01f;
    const auto     transform = openvdb::math::Transform::createLinearTransform(voxelSize);

    std::vector<openvdb::Vec3R> points;
    genPoints(pointCount, points);
    PointList pointList(points);
    EXPECT_EQ(pointCount, points.size());

    using SrcGridT = openvdb::tools::PointIndexGrid;
    auto srcGrid = openvdb::tools::createPointIndexGrid<SrcGridT>(pointList, *transform);

    using MgrT = openvdb::tree::LeafManager<const SrcGridT::TreeType>;
    MgrT leafs(srcGrid->tree());

    size_t count = 0;
    for (size_t n = 0, N = leafs.leafCount(); n < N; ++n) {
        count += leafs.leaf(n).indices().size();
    }
    EXPECT_EQ(pointCount, count);

    mTimer.start("Generating NanoVDB grid from PointIndexGrid");
    auto handle = nanovdb::openToNanoVDB(*srcGrid, /*mortonSort*/ false, 2);
    mTimer.stop();
    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ(nanovdb::GridType::UInt32, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::PointIndex, meta->gridClass());
    auto dstGrid = handle.grid<uint32_t>();
    EXPECT_TRUE(dstGrid);

    // first check the voxel values
    auto kernel1 = [&](const openvdb::CoordBBox& bbox) {
        using CoordT = const nanovdb::Coord;
        auto dstAcc = dstGrid->getAccessor();
        auto srcAcc = srcGrid->getAccessor();
        for (auto it = bbox.begin(); it; ++it) {
            EXPECT_EQ(srcAcc.getValue(*it), dstAcc.getValue(reinterpret_cast<CoordT&>(*it)));
        }
    };
    mTimer.start("Parallel unit test of voxel values");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel1);
    mTimer.stop();

    EXPECT_EQ(pointCount, dstGrid->blindMetaData(0).mElementCount);

    auto kernel = [&](const MgrT::LeafRange& r) {
        using CoordT = const nanovdb::Coord;
        auto                             dstAcc = dstGrid->getAccessor();
        nanovdb::PointAccessor<uint32_t> pointAcc(*dstGrid);
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
                for (auto* i = begin1; i != end1; ++i)
                    EXPECT_EQ(*i, *begin2++);
            }
        }
    };

    mTimer.start("Parallel unit test");
    tbb::parallel_for(leafs.leafRange(), kernel);
    mTimer.stop();

    mTimer.start("Testing bounding box");
    const auto dstBBox = dstGrid->indexBBox();
    const auto srcBBox = srcGrid->evalActiveVoxelBoundingBox();
    EXPECT_EQ(dstBBox.min()[0], srcBBox.min()[0]);
    EXPECT_EQ(dstBBox.min()[1], srcBBox.min()[1]);
    EXPECT_EQ(dstBBox.min()[2], srcBBox.min()[2]);
    EXPECT_EQ(dstBBox.max()[0], srcBBox.max()[0]);
    EXPECT_EQ(dstBBox.max()[1], srcBBox.max()[1]);
    EXPECT_EQ(dstBBox.max()[2], srcBBox.max()[2]);
    mTimer.stop();

    EXPECT_EQ(srcGrid->activeVoxelCount(), dstGrid->activeVoxelCount());
} // PointIndex

TEST_F(TestOpenVDB, PointData)
{
    // Create a vector with four point positions.
    std::vector<openvdb::Vec3R> positions;
    positions.push_back(openvdb::Vec3R(0, 1, 0));
    positions.push_back(openvdb::Vec3R(1.5, 3.5, 1));
    positions.push_back(openvdb::Vec3R(-1, 6, -2));
    positions.push_back(openvdb::Vec3R(1.1, 1.25, 0.06));
    // The VDB Point-Partioner is used when bucketing points and requires a
    // specific interface. For convenience, we use the PointAttributeVector
    // wrapper around an stl vector wrapper here, however it is also possible to
    // write one for a custom data structure in order to match the interface
    // required.
    openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper(positions);
    // This method computes a voxel-size to match the number of
    // points / voxel requested. Although it won't be exact, it typically offers
    // a good balance of memory against performance.
    int   pointsPerVoxel = 8;
    float voxelSize = openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel);
    std::cout << "VoxelSize=" << voxelSize << std::endl;
    auto transform = openvdb::math::Transform::createLinearTransform(voxelSize);
    auto srcGrid = openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
                                                        openvdb::points::PointDataGrid>(positions, *transform);
    srcGrid->setName("PointDataGrid");

    mTimer.start("Generating NanoVDB grid from PointDataGrid");
    auto handle = nanovdb::openToNanoVDB(*srcGrid);
    mTimer.stop();

    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ(nanovdb::GridType::UInt32, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::PointData, meta->gridClass());
    auto dstGrid = handle.grid<uint32_t>();
    EXPECT_TRUE(dstGrid);

    nanovdb::PointAccessor<nanovdb::Vec3f> acc(*dstGrid);
    const nanovdb::Vec3f *                 begin = nullptr, *end = nullptr; // iterators over points in a given voxel
    EXPECT_EQ(positions.size(), openvdb::points::pointCount(srcGrid->tree()));
    EXPECT_EQ(acc.gridPoints(begin, end), positions.size());
    for (auto leafIter = srcGrid->tree().cbeginLeaf(); leafIter; ++leafIter) {
        EXPECT_TRUE(leafIter->hasAttribute("P")); // Check position attribute from the leaf by name (P is position).
        // Create a read-only AttributeHandle. Position always uses Vec3f.
        openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(leafIter->constAttributeArray("P"));
        openvdb::Coord                                   ijk(openvdb::Coord::min());
        for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
            // Extract the index-space position of the point relative to its occupying voxel ijk.
            const openvdb::Vec3f dijk = positionHandle.get(*indexIter);
            if (ijk != indexIter.getCoord()) { // new voxel
                ijk = indexIter.getCoord();
                const nanovdb::Coord* abc = reinterpret_cast<const nanovdb::Coord*>(&ijk);
                EXPECT_TRUE(acc.isActive(*abc));
                EXPECT_TRUE(acc.voxelPoints(*abc, begin, end));
            }
            const openvdb::Vec3f xyz1 = ijk.asVec3s() + dijk;
            const nanovdb::Vec3f xyz2 = *begin++;
            for (int i = 0; i < 3; ++i)
                EXPECT_EQ(xyz1[i], xyz2[i]);
        }
    }
} // PointData

TEST_F(TestOpenVDB, PointData2)
{
    const std::string path = this->getEnvVar("VDB_DATA_PATH");
    if (path.empty())
        return;
    const std::vector<std::string> models = {"boat_points", "bunny_points", "sphere_points", "waterfall_points"};
    std::ofstream                  os("data/all_points.nvdb", std::ios::out | std::ios::binary);
    for (const auto& model : models) {
        const std::string fileName = path + "/" + model + ".vdb";
        mTimer.start("Reading grid from the file \"" + fileName + "\"");
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

        mTimer.restart("Generating NanoVDB grid from PointDataGrid");
        auto handle = nanovdb::openToNanoVDB(*srcGrid);
        mTimer.restart("Writing NanoVDB grid");
#if defined(NANOVDB_USE_BLOSC)
        nanovdb::io::writeGrid(os, handle, nanovdb::io::Codec::BLOSC);
#elif defined(NANOVDB_USE_ZIP)
        nanovdb::io::writeGrid(os, handle, nanovdb::io::Codec::ZIP);
#else
        nanovdb::io::writeGrid(os, handle, nanovdb::io::Codec::NONE);
#endif
        mTimer.stop();
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_EQ(nanovdb::GridType::UInt32, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::PointData, meta->gridClass());
        auto dstGrid = handle.grid<uint32_t>();
        EXPECT_TRUE(dstGrid);

        nanovdb::PointAccessor<nanovdb::Vec3f> acc(*dstGrid);
        const nanovdb::Vec3f *                 begin = nullptr, *end = nullptr; // iterators over points in a given voxel
        EXPECT_EQ(acc.gridPoints(begin, end), openvdb::points::pointCount(srcGrid->tree()));
        //std::cerr << "Point count = " << acc.gridPoints(begin, end) << ", attribute count = " << attributeSet.size() << std::endl;
        for (auto leafIter = srcGrid->tree().cbeginLeaf(); leafIter; ++leafIter) {
            EXPECT_TRUE(leafIter->hasAttribute("P")); // Check position attribute from the leaf by name (P is position).
            // Create a read-only AttributeHandle. Position always uses Vec3f.
            openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(leafIter->constAttributeArray("P"));
            openvdb::Coord                                   ijk(openvdb::Coord::min());
            for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
                // Extract the index-space position of the point relative to its occupying voxel ijk.
                const openvdb::Vec3f dijk = positionHandle.get(*indexIter);
                if (ijk != indexIter.getCoord()) { // new voxel
                    ijk = indexIter.getCoord();
                    const nanovdb::Coord* abc = reinterpret_cast<const nanovdb::Coord*>(&ijk);
                    EXPECT_TRUE(acc.isActive(*abc));
                    EXPECT_TRUE(acc.voxelPoints(*abc, begin, end));
                }
                const openvdb::Vec3f xyz1 = ijk.asVec3s() + dijk;
                const nanovdb::Vec3f xyz2 = *begin++;
                for (int i = 0; i < 3; ++i)
                    EXPECT_EQ(xyz1[i], xyz2[i]);
            }
        }
    }
} // PointData2

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
}

TEST_F(TestOpenVDB, CNanoVDB)
{
    auto srcGrid = this->getSrcGrid();
    mTimer.start("Generating NanoVDB grid");
    auto handle = nanovdb::openToNanoVDB(*srcGrid);
    mTimer.stop();
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
        }
    };

    mTimer.start("Parallel unit test");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    mTimer.stop();
}

TEST_F(TestOpenVDB, CNanoVDBTrilinear)
{
    auto srcGrid = this->getSrcGrid();
    mTimer.start("Generating NanoVDB grid");
    auto handle = nanovdb::openToNanoVDB(*srcGrid);
    mTimer.stop();
    EXPECT_TRUE(handle);
    EXPECT_TRUE(handle.data());

    const cnanovdb_griddata*  gridData = (const cnanovdb_griddata*)(handle.data());
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
            auto          ijk = *it;
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

    mTimer.start("Parallel unit test");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    mTimer.stop();
}

TEST_F(TestOpenVDB, CNanoVDBTrilinearStencil)
{
    auto srcGrid = this->getSrcGrid();
    mTimer.start("Generating NanoVDB grid");
    auto handle = nanovdb::openToNanoVDB(*srcGrid);
    mTimer.stop();
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
            auto          ijk = *it;
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

    mTimer.start("Parallel unit test");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    mTimer.stop();
}

TEST_F(TestOpenVDB, NanoToOpenVDB)
{
    mTimer.start("Reading NanoVDB grids from file");
    auto handles = nanovdb::io::readGrids("data/test.nvdb");
    mTimer.stop();

    EXPECT_EQ(1u, handles.size());
    auto* srcGrid = handles.front().grid<float>();
    EXPECT_TRUE(srcGrid);

    mTimer.start("Deserializing NanoVDB grid");
    auto dstGrid = nanovdb::nanoToOpenVDB(*srcGrid);
    mTimer.stop();

    //dstGrid->print(std::cout, 3);

    auto kernel = [&](const openvdb::CoordBBox& bbox) {
        using CoordT = const nanovdb::Coord;
        auto dstAcc = dstGrid->getUnsafeAccessor();
        //auto dstAcc = dstGrid->getAccessor();
        auto srcAcc = srcGrid->getAccessor();
        for (auto it = bbox.begin(); it; ++it) {
            EXPECT_EQ(srcAcc.getValue(reinterpret_cast<CoordT&>(*it)), dstAcc.getValue(*it));
        }
    };

    mTimer.start("Parallel unit test");
    tbb::parallel_for(dstGrid->evalActiveVoxelBoundingBox(), kernel);
    mTimer.stop();

    mTimer.start("Testing bounding box");
    const auto srcBBox = srcGrid->indexBBox();
    const auto dstBBox = dstGrid->evalActiveVoxelBoundingBox();
    EXPECT_EQ(dstBBox.min()[0], srcBBox.min()[0]);
    EXPECT_EQ(dstBBox.min()[1], srcBBox.min()[1]);
    EXPECT_EQ(dstBBox.min()[2], srcBBox.min()[2]);
    EXPECT_EQ(dstBBox.max()[0], srcBBox.max()[0]);
    EXPECT_EQ(dstBBox.max()[1], srcBBox.max()[1]);
    EXPECT_EQ(dstBBox.max()[2], srcBBox.max()[2]);
    mTimer.stop();

    EXPECT_EQ(srcGrid->activeVoxelCount(), dstGrid->activeVoxelCount());
} // NanoToOpenVDB

TEST_F(TestOpenVDB, File)
{
    { // check nanovdb::io::stringHash
        EXPECT_EQ(nanovdb::io::stringHash("generated_id_0"), nanovdb::io::stringHash("generated_id_0"));
        EXPECT_NE(nanovdb::io::stringHash("generated_id_0"), nanovdb::io::stringHash("generated_id_1"));
    }
    auto srcGrid = this->getSrcGrid();

    mTimer.start("Reading NanoVDB grids from file");
    auto handles = nanovdb::io::readGrids("data/test.nvdb");
    mTimer.stop();

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

    mTimer.start("Parallel unit test");
    tbb::parallel_for(srcGrid->evalActiveVoxelBoundingBox(), kernel);
    mTimer.stop();

    mTimer.start("Testing bounding box");
    const auto& dstBBox = dstGrid->indexBBox();
    const auto  srcBBox = srcGrid->evalActiveVoxelBoundingBox();
    EXPECT_EQ(dstBBox.min()[0], srcBBox.min()[0]);
    EXPECT_EQ(dstBBox.min()[1], srcBBox.min()[1]);
    EXPECT_EQ(dstBBox.min()[2], srcBBox.min()[2]);
    EXPECT_EQ(dstBBox.max()[0], srcBBox.max()[0]);
    EXPECT_EQ(dstBBox.max()[1], srcBBox.max()[1]);
    EXPECT_EQ(dstBBox.max()[2], srcBBox.max()[2]);
    mTimer.stop();

    EXPECT_EQ(srcGrid->activeVoxelCount(), dstGrid->activeVoxelCount());
} // File

TEST_F(TestOpenVDB, MultiFile)
{
    std::vector<nanovdb::GridHandle<>> handles;
    { // add an int32_t grid
        openvdb::Int32Grid grid(-1);
        grid.setName("Int32 grid");
        grid.tree().setValue(openvdb::Coord(-256), 10);
        EXPECT_EQ(1u, grid.activeVoxelCount());
        handles.push_back(nanovdb::openToNanoVDB(grid));
    }
    { // add an empty int32_t grid
        openvdb::Int32Grid grid(-4);
        grid.setName("Int32 grid, empty");
        EXPECT_EQ(0u, grid.activeVoxelCount());
        handles.push_back(nanovdb::openToNanoVDB(grid));
    }
    { // add a Vec3f grid
        openvdb::Vec3fGrid grid(openvdb::Vec3f(0.0f, 0.0f, -1.0f));
        grid.setName("Float vector grid");
        grid.setGridClass(openvdb::GRID_STAGGERED);
        grid.tree().setValue(openvdb::Coord(-256), openvdb::Vec3f(1.0f, 0.0f, 0.0f));
        EXPECT_EQ(1u, grid.activeVoxelCount());
        handles.push_back(nanovdb::openToNanoVDB(grid));
    }
    { // add an int64_t grid
        openvdb::Int64Grid grid(0);
        grid.setName("Int64 grid");
        grid.tree().setValue(openvdb::Coord(0), 10);
        EXPECT_EQ(1u, grid.activeVoxelCount());
        handles.push_back(nanovdb::openToNanoVDB(grid));
    }
    for (int i = 0; i < 10; ++i) {
        const float          radius = 100.0f;
        const float          voxelSize = 1.0f, width = 3.0f;
        const openvdb::Vec3f center(i * 10.0f, 0.0f, 0.0f);
        auto                 srcGrid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center, voxelSize, width);
        srcGrid->setName("Level set sphere at (" + std::to_string(i * 10) + ",0,0)");
        handles.push_back(nanovdb::openToNanoVDB(*srcGrid));
    }
    { // add a double grid
        openvdb::DoubleGrid grid(0.0);
        grid.setName("Double grid");
        grid.setGridClass(openvdb::GRID_FOG_VOLUME);
        grid.tree().setValue(openvdb::Coord(6000), 1.0);
        EXPECT_EQ(1u, grid.activeVoxelCount());
        handles.push_back(nanovdb::openToNanoVDB(grid));
    }
#if defined(NANOVDB_USE_BLOSC)
    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>("data/multi.nvdb", handles, nanovdb::io::Codec::BLOSC);
#elif defined(NANOVDB_USE_ZIP)
    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>("data/multi.nvdb", handles, nanovdb::io::Codec::ZIP);
#else
    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>("data/multi.nvdb", handles, nanovdb::io::Codec::NONE);
#endif
    { // read grid meta data and test it
        mTimer.start("nanovdb::io::readGridMetaData");
        auto meta = nanovdb::io::readGridMetaData("data/multi.nvdb");
        mTimer.stop();
        EXPECT_EQ(15u, meta.size());
        EXPECT_EQ(std::string("Double grid"), meta.back().gridName);
    }
    { // read in32 grid and test values
        mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        mTimer.stop();
        EXPECT_EQ(15u, handles.size());
        auto& handle = handles.front();
        EXPECT_EQ(std::string("Int32 grid"), handle.gridMetaData()->gridName());
        EXPECT_FALSE(handle.grid<float>());
        EXPECT_FALSE(handle.grid<double>());
        EXPECT_FALSE(handle.grid<int64_t>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3f>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3d>());
        auto* grid = handle.grid<int32_t>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(1u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(-256);
        const auto&          tree = grid->tree();
        EXPECT_EQ(10, tree.getValue(ijk));
        EXPECT_EQ(-1, tree.getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(10, tree.root().valueMin());
        EXPECT_EQ(10, tree.root().valueMax());
        const nanovdb::CoordBBox bbox(ijk, ijk);
        EXPECT_EQ(bbox, grid->indexBBox());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(1u, tree.nodeCount(0));
        EXPECT_EQ(1u, tree.nodeCount(1));
        EXPECT_EQ(1u, tree.nodeCount(2));
        EXPECT_EQ(1u, tree.nodeCount(3));
        const auto* leaf = tree.getNode<0>(0);
        EXPECT_TRUE(leaf);
        EXPECT_EQ(bbox, leaf->bbox());
        const auto* node1 = tree.getNode<1>(0);
        EXPECT_TRUE(node1);
        EXPECT_EQ(bbox, node1->bbox());
        const auto* node2 = tree.getNode<2>(0);
        EXPECT_TRUE(node2);
        EXPECT_EQ(bbox, node2->bbox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
    { // read empty in32 grid and test values
        mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        mTimer.stop();
        EXPECT_EQ(15u, handles.size());
        auto& handle = handles[1];
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Int32 grid, empty"), handle.gridMetaData()->gridName());
        EXPECT_FALSE(handle.grid<float>());
        EXPECT_FALSE(handle.grid<double>());
        EXPECT_FALSE(handle.grid<int64_t>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3f>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3d>());
        auto* grid = handle.grid<int32_t>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(0u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(-256);
        EXPECT_EQ(-4, grid->tree().getValue(ijk));
        EXPECT_EQ(-4, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(-4, grid->tree().root().valueMin());
        EXPECT_EQ(-4, grid->tree().root().valueMax());
        EXPECT_EQ(nanovdb::Coord(std::numeric_limits<int>::max()), grid->indexBBox().min());
        EXPECT_EQ(nanovdb::Coord(std::numeric_limits<int>::min()), grid->indexBBox().max());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(0u, grid->tree().nodeCount(0));
        EXPECT_EQ(0u, grid->tree().nodeCount(1));
        EXPECT_EQ(0u, grid->tree().nodeCount(2));
        EXPECT_EQ(1u, grid->tree().nodeCount(3)); // always a root node
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
    { // read int64 grid and test values
        mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        mTimer.stop();
        EXPECT_EQ(15u, handles.size());
        auto& handle = handles[3];
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Int64 grid"), handle.gridMetaData()->gridName());
        auto* grid = handle.grid<int64_t>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(1u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(0);
        EXPECT_EQ(10, grid->tree().getValue(ijk));
        EXPECT_EQ(0, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(10, grid->tree().root().valueMin());
        EXPECT_EQ(10, grid->tree().root().valueMax());
        EXPECT_EQ(nanovdb::CoordBBox(ijk, ijk), grid->indexBBox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
    { // read vec3f grid and test values
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        EXPECT_EQ(15u, handles.size());
        auto& handle = handles[2];
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Float vector grid"), handle.gridMetaData()->gridName());
        auto* grid = handle.grid<nanovdb::Vec3f>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(1u, grid->activeVoxelCount());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        const nanovdb::Coord ijk(-256);
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 0.0f, 0.0f), grid->tree().getValue(ijk));
        EXPECT_EQ(nanovdb::Vec3f(0.0f, 0.0f, -1.0f), grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 0.0f, 0.0f), grid->tree().root().valueMin());
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 0.0f, 0.0f), grid->tree().root().valueMax());
        EXPECT_EQ(nanovdb::CoordBBox(ijk, ijk), grid->tree().bbox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_FALSE(grid->isUnknown());
        EXPECT_TRUE(grid->isStaggered());
    }
    { // read double grid and test values
        auto handles = nanovdb::io::readGrids("data/multi.nvdb");
        EXPECT_EQ(15u, handles.size());
        auto& handle = handles.back();
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Double grid"), handle.gridMetaData()->gridName());
        auto* grid = handle.grid<double>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(1u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(6000);
        EXPECT_EQ(1.0, grid->tree().getValue(ijk));
        EXPECT_EQ(0.0, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(1.0, grid->tree().root().valueMin());
        EXPECT_EQ(1.0, grid->tree().root().valueMax());
        EXPECT_EQ(nanovdb::CoordBBox(ijk, ijk), grid->tree().bbox());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_TRUE(grid->isFogVolume());
        EXPECT_FALSE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
} // MultiFile

TEST_F(TestOpenVDB, MultiFile2)
{
    const std::string path = this->getEnvVar("VDB_DATA_PATH");
    if (path.empty())
        return;
    const std::vector<std::string> models = {"armadillo", "buddha", "bunny", "crawler", "dragon", "iss", "space", "torus_knot_helix", "utahteapot"};
    std::ofstream                  os("data/all.nvdb", std::ios::out | std::ios::binary);
    for (const auto& model : models) {
        const std::string fileName = path + "/" + model + ".vdb";
        mTimer.start("Reading grid from the file \"" + fileName + "\"");
        openvdb::io::File file(fileName);
        file.open(false); //disable delayed loading
        auto srcGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(file.beginName().gridName()));
        mTimer.restart("Generating NanoVDB grid");
        auto handle = nanovdb::openToNanoVDB(*srcGrid, /*mortonSort*/ false, 1);
        mTimer.restart("Writing NanoVDB grid");
#if defined(NANOVDB_USE_BLOSC)
        nanovdb::io::writeGrid(os, handle, nanovdb::io::Codec::BLOSC);
#elif defined(NANOVDB_USE_ZIP)
        nanovdb::io::writeGrid(os, handle, nanovdb::io::Codec::ZIP);
#else
        nanovdb::io::writeGrid(os, handle, nanovdb::io::Codec::NONE);
#endif
        mTimer.stop();
    }
    mTimer.start("Read GridMetaData from file");
    auto meta = nanovdb::io::readGridMetaData("data/all.nvdb");
    mTimer.stop();
    EXPECT_EQ(models.size(), meta.size());
    for (size_t i = 0; i < models.size(); ++i) {
        EXPECT_EQ(nanovdb::GridType::Float, meta[i].gridType);
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta[i].gridClass);
        if (i == 7) { // special case
            EXPECT_EQ("TorusKnotHelix", meta[i].gridName);
        } else {
            EXPECT_EQ("ls_" + models[i], meta[i].gridName);
        }
    }
    // test reading from non-existing file
    EXPECT_THROW(nanovdb::io::readGrid("data/all.vdb", "ls_bunny"), std::runtime_error);

    // test reading non-existing grid from an existing file
    EXPECT_FALSE(nanovdb::io::readGrid("data/all.nvdb", "bunny"));

    { // test reading existing grid from an existing file
        auto handle = nanovdb::io::readGrid("data/all.nvdb", "ls_bunny");
        EXPECT_TRUE(handle);
        EXPECT_FALSE(handle.grid<double>());
        auto grid = handle.grid<float>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(nanovdb::GridType::Float, grid->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, grid->gridClass());
        EXPECT_EQ("ls_bunny", std::string(grid->gridName()));
    }
} // MultiFile2

TEST_F(TestOpenVDB, Trilinear)
{
    // create a grid so sample from
    auto trilinear = [](const openvdb::Vec3R& xyz) -> float {
        return 0.34 + 1.6 * xyz[0] + 6.7 * xyz[1] - 3.5 * xyz[2]; // world coordinates
    };

    mTimer.start("Generating a dense tri-linear openvdb grid");
    auto        srcGrid = openvdb::createLevelSet<openvdb::FloatGrid>(/*background=*/1.0f);
    const float voxelSize = 0.5f;
    srcGrid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));
    const openvdb::CoordBBox bbox(openvdb::Coord(0), openvdb::Coord(128));
    auto                     acc = srcGrid->getAccessor();
    for (auto iter = bbox.begin(); iter; ++iter) {
        auto ijk = *iter;
        acc.setValue(ijk, trilinear(srcGrid->indexToWorld(ijk)));
    }
    mTimer.restart("Generating NanoVDB grid");
    auto handle = nanovdb::openToNanoVDB(*srcGrid);
    mTimer.restart("Writing NanoVDB grid");
    nanovdb::io::writeGrid("data/tmp.nvdb", handle);
    mTimer.stop();
    handle.reset();
    EXPECT_FALSE(handle.grid<float>());
    EXPECT_FALSE(handle.grid<double>());

    mTimer.start("Reading NanoVDB from file");
    auto handles = nanovdb::io::readGrids("data/tmp.nvdb");
    mTimer.stop();
    EXPECT_EQ(1u, handles.size());
    auto* dstGrid = handles[0].grid<float>();
    EXPECT_TRUE(dstGrid);
    EXPECT_FALSE(handles[0].grid<double>());
    EXPECT_EQ(voxelSize, dstGrid->voxelSize());

    const openvdb::Vec3R ijk(13.4, 24.67, 5.23); // in index space
    const float          exact = trilinear(srcGrid->indexToWorld(ijk));
    const float          approx = trilinear(srcGrid->indexToWorld(openvdb::Coord(13, 25, 5)));
    //std::cerr << "Trilinear: exact = " << exact << ", approx = " << approx << std::endl;

    auto dstAcc = dstGrid->getAccessor();
    auto sampler0 = nanovdb::createSampler<0>(dstAcc);
    //std::cerr << "0'th order: v = " << sampler0(ijk) << std::endl;
    EXPECT_EQ(approx, sampler0(ijk));

    auto sampler1 = nanovdb::createSampler<1>(dstAcc); // faster since it's using an accessor!!!
    //std::cerr << "1'th order: v = " << sampler1(ijk) << std::endl;
    EXPECT_EQ(exact, sampler1(ijk));

    EXPECT_FALSE(sampler1.zeroCrossing());
    const auto gradIndex = sampler1.gradient(ijk); //in index space
    EXPECT_NEAR(1.6f, gradIndex[0] / voxelSize, 1e-5);
    EXPECT_NEAR(6.7f, gradIndex[1] / voxelSize, 1e-5);
    EXPECT_NEAR(-3.5f, gradIndex[2] / voxelSize, 1e-5);
    const auto gradWorld = dstGrid->indexToWorldDir(gradIndex); // in world units
    EXPECT_NEAR(1.6f, gradWorld[0], 1e-5);
    EXPECT_NEAR(6.7f, gradWorld[1], 1e-5);
    EXPECT_NEAR(-3.5f, gradWorld[2], 1e-5);

    nanovdb::SampleFromVoxels<nanovdb::NanoTree<float>, 3> sampler3(dstGrid->tree());
    //auto sampler3 = nanovdb::createSampler<3>( dstAcc );
    //std::cerr << "3'rd order: v = " << sampler3(ijk) << std::endl;
    EXPECT_EQ(exact, sampler3(ijk));
} // Trilinear

TEST_F(TestOpenVDB, Tricubic)
{
    // create a grid so sample from
    auto trilinear = [](const openvdb::Vec3R& xyz) -> double {
        return 0.34 + 1.6 * xyz[0] + 2.7 * xyz[1] + 1.5 * xyz[2] + 0.025 * xyz[0] * xyz[1] * xyz[2] - 0.013 * xyz[0] * xyz[0] * xyz[0]; // world coordinates
    };

    mTimer.start("Generating a dense tri-cubic openvdb grid");
    auto srcGrid = openvdb::createLevelSet<openvdb::DoubleGrid>(/*background=*/1.0);
    srcGrid->setName("Tri-Cubic");
    srcGrid->setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.5));
    const openvdb::CoordBBox bbox(openvdb::Coord(0), openvdb::Coord(128));
    auto                     acc = srcGrid->getAccessor();
    for (auto iter = bbox.begin(); iter; ++iter) {
        auto ijk = *iter;
        acc.setValue(ijk, trilinear(srcGrid->indexToWorld(ijk)));
    }
    mTimer.restart("Generating NanoVDB grid");
    auto handle = nanovdb::openToNanoVDB(*srcGrid);
    mTimer.restart("Writing NanoVDB grid");
    nanovdb::io::writeGrid("data/tmp.nvdb", handle);
    mTimer.stop();

    { //test File::hasGrid
        EXPECT_TRUE(nanovdb::io::hasGrid("data/tmp.nvdb", "Tri-Cubic"));
        EXPECT_FALSE(nanovdb::io::hasGrid("data/tmp.nvdb", "Tri-Linear"));
    }

    mTimer.start("Reading NanoVDB from file");
    auto handles = nanovdb::io::readGrids("data/tmp.nvdb", 1);
    mTimer.stop();
    auto* dstGrid = handles[0].grid<double>();
    EXPECT_TRUE(dstGrid);

    const openvdb::Vec3R ijk(3.4, 4.67, 5.23); // in index space
    const float          exact = trilinear(srcGrid->indexToWorld(ijk));
    const float          approx = trilinear(srcGrid->indexToWorld(openvdb::Coord(3, 5, 5)));
    //std::cerr << "Trilinear: exact = " << exact << ", approx = " << approx << std::endl;
    auto dstAcc = dstGrid->getAccessor();

    auto sampler0 = nanovdb::createSampler<0>(dstAcc);
    //std::cerr << "0'th order: v = " << sampler0(ijk) << std::endl;
    EXPECT_NEAR(approx, sampler0(ijk), 1e-6);

    //nanovdb::SampleFromVoxels<TreeT, 1> sampler1( dstGrid->tree());
    //std::cerr << "1'th order: v = " << sampler1(ijk) << std::endl;
    //EXPECT_EQ( exact, sampler1(ijk) );

    auto sampler3 = nanovdb::createSampler<3>(dstAcc);
    //std::cerr << "3'rd order: v = " << sampler3(ijk) << std::endl;
    EXPECT_NEAR(exact, sampler3(ijk), 1e-4);
} // Tricubic

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
