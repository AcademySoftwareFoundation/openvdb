// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/util/Util.h>
#include <cppunit/extensions/HelperMacros.h>

// See TestMetadataIO for an explanation with this logic
#undef CPPUNIT_TESTNAMER_DECL
#define CPPUNIT_TESTNAMER_DECL( variableName, FixtureType ) \
    CPPUNIT_NS::TestNamer variableName( FixtureType::testSuiteName() )

template<typename TreeT, openvdb::tools::NearestNeighbors NN>
class TestMorphology : public CppUnit::TestCase
{
public:
    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    static std::string testSuiteName() {
        using T = typename TreeT::ValueType;
        std::string name = openvdb::typeNameAsString<T>();
        if (!name.empty()) name[0] = static_cast<char>(::toupper(name[0]));
        if (NN == openvdb::tools::NN_FACE)             return "TestMorphologyFace" + name;
        if (NN == openvdb::tools::NN_FACE_EDGE)        return "TestMorphologyEdge" + name;
        if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) return "TestMorphologyVertex" + name;
        return "";
    }

    CPPUNIT_TEST_SUITE(TestMorphology);
    CPPUNIT_TEST(testMorphActiveLeafValues);
    CPPUNIT_TEST(testMorphActiveValues);
    CPPUNIT_TEST_SUITE_END();

    void testMorphActiveLeafValues();
    void testMorphActiveValues();
};


// required due to multi template arguments
using TDFF = TestMorphology<openvdb::FloatTree, openvdb::tools::NN_FACE>;
using TDFE = TestMorphology<openvdb::FloatTree, openvdb::tools::NN_FACE_EDGE>;
using TDFV = TestMorphology<openvdb::FloatTree, openvdb::tools::NN_FACE_EDGE_VERTEX>;
using TDMF = TestMorphology<openvdb::MaskTree, openvdb::tools::NN_FACE>;
using TDME = TestMorphology<openvdb::MaskTree, openvdb::tools::NN_FACE_EDGE>;
using TDMV = TestMorphology<openvdb::MaskTree, openvdb::tools::NN_FACE_EDGE_VERTEX>;
CPPUNIT_TEST_SUITE_REGISTRATION(TDFF);
CPPUNIT_TEST_SUITE_REGISTRATION(TDFE);
CPPUNIT_TEST_SUITE_REGISTRATION(TDFV);
CPPUNIT_TEST_SUITE_REGISTRATION(TDMF);
CPPUNIT_TEST_SUITE_REGISTRATION(TDME);
CPPUNIT_TEST_SUITE_REGISTRATION(TDMV);

template<typename TreeT, openvdb::tools::NearestNeighbors NN>
void
TestMorphology<TreeT, NN>::testMorphActiveLeafValues()
{
    using openvdb::Coord;
    using openvdb::Index32;
    using openvdb::Index64;
    using ValueType = typename TreeT::ValueType;

    size_t offsets = 0;
    if (NN == openvdb::tools::NN_FACE)             offsets = 6;
    if (NN == openvdb::tools::NN_FACE_EDGE)        offsets = 18;
    if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) offsets = 26;

    const Coord* const start = openvdb::util::COORD_OFFSETS;
    const Coord* const end = start + offsets;

    // Small methods to check neighbour activity from an xyz coordinate. Recurse
    // parameter allows for recursively checking the acitvity of the xyz
    // neighbours, with recurse=0 only checking the immediate neighbours.
    std::function<void(const TreeT&, const Coord&, const size_t)> CheckActiveNeighbours;
    CheckActiveNeighbours = [start, end, &CheckActiveNeighbours]
        (const TreeT& acc, const Coord& xyz, const size_t recurse)
    {
        CPPUNIT_ASSERT(acc.isValueOn(xyz));
        const Coord* offset(start);
        while (offset != end) {
            // optionally recurse into neighbour voxels
            const Coord ijk = xyz + *offset;
            if (recurse > 0) CheckActiveNeighbours(acc, ijk, recurse-1);
            CPPUNIT_ASSERT(acc.isValueOn(ijk));
            ++offset;
        }
    };

    auto CheckInactiveNeighbours = [start, end]
        (const TreeT& acc, const Coord& xyz) {
        const Coord* offset(start);
        while (offset != end) {
            CPPUNIT_ASSERT(acc.isValueOff(xyz + *offset));
            ++offset;
        }
    };

    constexpr bool IsMask = std::is_same<ValueType, bool>::value;
    TreeT tree(IsMask ? 0.0 : -1.0);
    CPPUNIT_ASSERT(tree.empty());

    const openvdb::Index leafDim = TreeT::LeafNodeType::DIM;
    CPPUNIT_ASSERT_EQUAL(1 << 3, int(leafDim));

    { // Set and dilate a single voxel at the center of a leaf node.
        tree.clear();
        const Coord xyz(leafDim >> 1);
        tree.setValue(xyz, ValueType(1.0));
        CPPUNIT_ASSERT(tree.isValueOn(xyz));
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeVoxelCount());
        // dilate
        openvdb::tools::dilateActiveLeafValues(tree, 1, NN);
        CheckActiveNeighbours(tree, xyz, 0);
        CPPUNIT_ASSERT_EQUAL(Index64(1 + offsets), tree.activeVoxelCount());
        // erode
        openvdb::tools::erodeActiveLeafValues(tree, 1, NN);
        CheckInactiveNeighbours(tree, xyz);
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeVoxelCount());
        openvdb::tools::erodeActiveLeafValues(tree, 1, NN);
        CPPUNIT_ASSERT_EQUAL(Index64(0), tree.activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index32(1), tree.leafCount());
        // check values
        if (!IsMask) {
            CPPUNIT_ASSERT_EQUAL(tree.getValue(xyz), ValueType(1.0));
            const Coord* offset(start);
            while (offset != end) CPPUNIT_ASSERT_EQUAL(tree.getValue(xyz + *offset++), ValueType(-1.0));
        }
    }
    { // Create an active, leaf node-sized tile and a single edge/corner voxel
        tree.clear();
        tree.addTile(/*level*/1, Coord(0), ValueType(1.0), true);
        CPPUNIT_ASSERT_EQUAL(Index32(0), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(leafDim * leafDim * leafDim), tree.activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());

        const Coord xyz(leafDim, leafDim - 1, leafDim - 1);
        tree.setValue(xyz, ValueType(1.0));

        Index64 expected = leafDim * leafDim * leafDim + 1;
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());

        // dilate
        openvdb::tools::dilateActiveLeafValues(tree, 1, NN);
        CheckActiveNeighbours(tree, xyz, 0);
        if (NN == openvdb::tools::NN_FACE)             expected += 5;  // 1 overlapping with tile
        if (NN == openvdb::tools::NN_FACE_EDGE)        expected += 15; // 3 overlapping
        if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) expected += 22; // 4 overlapping
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());
        Index32 leafs;
        if (NN == openvdb::tools::NN_FACE)             leafs = 3;
        if (NN == openvdb::tools::NN_FACE_EDGE)        leafs = 6;
        if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) leafs = 7;
        CPPUNIT_ASSERT_EQUAL(leafs, tree.leafCount());

        // erode
        openvdb::tools::erodeActiveLeafValues(tree, 1, NN);
        // tile should be umodified
        expected = leafDim * leafDim * leafDim + 1;
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());
        CPPUNIT_ASSERT_EQUAL(leafs, tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
        //
        openvdb::tools::erodeActiveLeafValues(tree, 1, NN);
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());
        CPPUNIT_ASSERT_EQUAL(leafs, tree.leafCount());
        expected = leafDim * leafDim * leafDim;
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
        // erode again, only 1 active tile, should be no change
        TreeT copy(tree);
        openvdb::tools::erodeActiveLeafValues(tree, 1, NN);
        CPPUNIT_ASSERT(copy.hasSameTopology(tree));
        // check values
        if (!IsMask) {
            CPPUNIT_ASSERT_EQUAL(tree.getValue(xyz), ValueType(1.0));
            CPPUNIT_ASSERT_EQUAL(tree.getValue(Coord(0)), ValueType(1.0));
        }
    }
    { // Set and dilate a single voxel at each of the eight corners of a leaf node.
        for (int i = 0; i < 8; ++i) {
            tree.clear();
            const Coord xyz(
                i & 1 ? leafDim - 1 : 0,
                i & 2 ? leafDim - 1 : 0,
                i & 4 ? leafDim - 1 : 0);
            tree.setValue(xyz, ValueType(1.0));
            CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeVoxelCount());
            // dilate
            openvdb::tools::dilateActiveLeafValues(tree, 1, NN);
            CheckActiveNeighbours(tree, xyz, 0);
            CPPUNIT_ASSERT_EQUAL(Index64(1 + offsets), tree.activeVoxelCount());
            // erode
            openvdb::tools::erodeActiveLeafValues(tree, 1, NN);
            CheckInactiveNeighbours(tree, xyz);
            CPPUNIT_ASSERT(tree.isValueOn(xyz));
            CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeVoxelCount());
            // check values
            if (!IsMask) {
                CPPUNIT_ASSERT_EQUAL(tree.getValue(xyz), ValueType(1.0));
                const Coord* offset(start);
                while (offset != end) CPPUNIT_ASSERT_EQUAL(tree.getValue(xyz + *offset++), ValueType(-1.0));
            }
        }
    }
    { // 3 neighbouring voxels
        tree.clear();
        const Coord xyz1(0), xyz2(1,0,0), xyz3(-1,0,0);
        tree.setValue(xyz1, ValueType(1.0));
        tree.setValue(xyz2, ValueType(1.0));
        tree.setValue(xyz3, ValueType(1.0));

        Index64 expected = 3;
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
        openvdb::tools::dilateActiveLeafValues(tree, 1, NN);
        CheckActiveNeighbours(tree, xyz1, 0);
        CheckActiveNeighbours(tree, xyz2, 0);
        CheckActiveNeighbours(tree, xyz3, 0);

        if (NN == openvdb::tools::NN_FACE)             expected += (6* 3)-4;  // dilation - overlapping
        if (NN == openvdb::tools::NN_FACE_EDGE)        expected += (18*3)-20; // dilation - overlapping
        if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) expected += (26*3)-36; // dilation - overlapping
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());

        openvdb::tools::erodeActiveLeafValues(tree, 1, NN);
        expected = 3;
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
        // check values
        if (!IsMask) {
            CPPUNIT_ASSERT_EQUAL(tree.getValue(xyz1), ValueType(1.0));
            CPPUNIT_ASSERT_EQUAL(tree.getValue(xyz2), ValueType(1.0));
            CPPUNIT_ASSERT_EQUAL(tree.getValue(xyz3), ValueType(1.0));
        }
    }
    { // Perform repeated dilations, starting with a single voxel.
        struct Info { int activeVoxelCount, leafCount, nonLeafCount; };
        Info iterInfo[33] = {
            /*    FACE                EDGE               VERTEX   */
            { 1,     1,  3 },   { 1,     1,  3 },   { 1,     1,  3 },
            { 7,     1,  3 },   { 19,    1,  3 },   { 27,    1,  3 },
            { 25,    1,  3 },   { 93,    1,  3 },   { 125,   1,  3 },
            { 63,    1,  3 },   { 263,   1,  3 },   { 343,   1,  3 },
            { 129,   4,  3 },   { 569,   7,  3 },   { 729,   8,  3 },
            { 231,   7,  9 },   { 1051, 19, 15 },   { 1331, 27, 17 },
            { 377,   7,  9 },   { 1749, 20, 15 },   { 2197, 27, 17 },
            { 575,   7,  9 },   { 2703, 26, 15 },   { 3375, 27, 17 },
            { 833,  10,  9 },   { 3953, 27, 17 },   { 4913, 27, 17 },
            { 1159, 16,  9 },   { 5539, 27, 17 },   { 6859, 27, 17 },
            { 1561, 19, 15 },   { 7501, 27, 17 },   { 9261, 27, 17 },
        };

        tree.clear();
        tree.setValue(Coord(leafDim >> 1), ValueType(1.0));

        int offset = 0;
        if (NN == openvdb::tools::NN_FACE_EDGE)        offset = 1;
        if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) offset = 2;
        int i = offset;
        CPPUNIT_ASSERT_EQUAL(iterInfo[i].activeVoxelCount, int(tree.activeVoxelCount()));
        CPPUNIT_ASSERT_EQUAL(iterInfo[i].leafCount,        int(tree.leafCount()));
        CPPUNIT_ASSERT_EQUAL(iterInfo[i].nonLeafCount,     int(tree.nonLeafCount()));

        // dilate
        i+= 3;
        for (; i < 33; i+=3) {
            openvdb::tools::dilateActiveLeafValues(tree, 1, NN);
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].activeVoxelCount, int(tree.activeVoxelCount()));
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].leafCount,        int(tree.leafCount()));
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].nonLeafCount,     int(tree.nonLeafCount()));
        }
        // erode
        i-= 6;
        for (; i >= 0; i-=3) {
            openvdb::tools::erodeActiveLeafValues(tree, 1, NN);
            // also prune inactive to clear up empty nodes
            openvdb::tools::pruneInactive(tree);
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].activeVoxelCount, int(tree.activeVoxelCount()));
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].leafCount,        int(tree.leafCount()));
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].nonLeafCount,     int(tree.nonLeafCount()));
        }
        // try with values as iterations
        int j = 0;
        i = offset;
        for (; i < 33; i+=3, ++j) {
            tree.clear();
            tree.setValue(Coord(leafDim >> 1), ValueType(1.0));
            // dilate
            openvdb::tools::dilateActiveLeafValues(tree, j, NN);
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].activeVoxelCount, int(tree.activeVoxelCount()));
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].leafCount,        int(tree.leafCount()));
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].nonLeafCount,     int(tree.nonLeafCount()));
        }
        // erode
        i-= 3;
        j = 0;
        for (; i >= 0; i-=3, ++j) {
            tree.clear();
            tree.setValue(Coord(leafDim >> 1), ValueType(1.0));
            openvdb::tools::dilateActiveLeafValues(tree, 10, NN);
            openvdb::tools::erodeActiveLeafValues(tree, j, NN);
            // also prune inactive to clear up empty nodes
            openvdb::tools::pruneInactive(tree);
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].activeVoxelCount, int(tree.activeVoxelCount()));
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].leafCount,        int(tree.leafCount()));
            CPPUNIT_ASSERT_EQUAL(iterInfo[i].nonLeafCount,     int(tree.nonLeafCount()));
        }
    }
    { // Test multiple iterations
        tree.clear();
        const Coord xyz(leafDim >> 1);
        tree.setValue(xyz, ValueType(1.0));
        CPPUNIT_ASSERT(tree.isValueOn(xyz));
        Index64 expected = 1;
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());

        if (NN == openvdb::tools::NN_FACE)             expected = 25;
        if (NN == openvdb::tools::NN_FACE_EDGE)        expected = 93;
        if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) expected = 125;
        openvdb::tools::dilateActiveLeafValues(tree, 2, NN);
        CheckActiveNeighbours(tree, xyz, /*recurse-once*/1);
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());

        if (NN == openvdb::tools::NN_FACE)             expected = 231;
        if (NN == openvdb::tools::NN_FACE_EDGE)        expected = 1051;
        if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) expected = 1331;
        openvdb::tools::dilateActiveLeafValues(tree, 3, NN);
        CheckActiveNeighbours(tree, xyz, /*recurse-four-times*/4);
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
        openvdb::tools::erodeActiveLeafValues(tree, 5, NN);
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeVoxelCount());
        CheckInactiveNeighbours(tree, xyz);
    }

    {// dilate a narrow band of a sphere
        const openvdb::FloatGrid::ConstPtr grid =
            openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(/*radius=*/20,
                /*center=*/openvdb::Vec3f(0, 0, 0),
                /*dx=*/1.0f, /*halfWidth*/ 3.0f);
        const Index64 count = grid->tree().activeVoxelCount();
        {
            TreeT copy(grid->tree());
            openvdb::tools::dilateActiveLeafValues(copy, 1, NN);
            CPPUNIT_ASSERT(copy.activeVoxelCount() > count);
        }
        {
            TreeT copy(grid->tree());
            openvdb::tools::erodeActiveLeafValues(copy, 1, NN);
            CPPUNIT_ASSERT(copy.activeVoxelCount() < count);
        }
    }

    {// dilate a fog volume of a sphere
        openvdb::FloatGrid::Ptr grid =
            openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(/*radius=*/20,
                /*center=*/openvdb::Vec3f(0, 0, 0),
                /*dx=*/1.0f, /*halfWidth*/ 3.0f);
        openvdb::tools::sdfToFogVolume(*grid);
        const Index64 count = grid->tree().activeVoxelCount();
        {
            TreeT copy(grid->tree());
            openvdb::tools::dilateActiveLeafValues(copy, 1, NN);
            CPPUNIT_ASSERT(copy.activeVoxelCount() > count);
        }
        {
            TreeT copy(grid->tree());
            openvdb::tools::erodeActiveLeafValues(copy, 1, NN);
            CPPUNIT_ASSERT(copy.activeVoxelCount() < count);
        }
    }

    { // test dilation/erosion at every position inside a 8x8x8 leaf
        for (int x=0; x<8; ++x) {
            for (int y=0; y<8; ++y) {
                for (int z=0; z<8; ++z) {
                    tree.clear();
                    const openvdb::Coord xyz(x,y,z);
                    tree.setValue(xyz, ValueType(1.0));
                    CPPUNIT_ASSERT(tree.isValueOn(xyz));
                    CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeVoxelCount());
                    // dilate
                    openvdb::tools::dilateActiveLeafValues(tree, 1, NN);
                    CheckActiveNeighbours(tree, xyz, 0);
                    CPPUNIT_ASSERT_EQUAL(Index64(1 + offsets), tree.activeVoxelCount());
                    //erode
                    openvdb::tools::erodeActiveLeafValues(tree, 1, NN);
                    CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeVoxelCount());
                    CheckInactiveNeighbours(tree, xyz);
                    CPPUNIT_ASSERT(tree.isValueOn(xyz));
                    if (!IsMask) CPPUNIT_ASSERT_EQUAL(ValueType(1.0), tree.getValue(xyz));
                }
            }
        }
    }
}

template<typename TreeT, openvdb::tools::NearestNeighbors NN>
void
TestMorphology<TreeT, NN>::testMorphActiveValues()
{
    using openvdb::Coord;
    using openvdb::CoordBBox;
    using openvdb::Index32;
    using openvdb::Index64;
    using ValueType = typename TreeT::ValueType;

    size_t offsets = 0;
    if (NN == openvdb::tools::NN_FACE)             offsets = 6;
    if (NN == openvdb::tools::NN_FACE_EDGE)        offsets = 18;
    if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) offsets = 26;

    const Coord* const start = openvdb::util::COORD_OFFSETS;
    const Coord* const end = start + offsets;

    // Small method to check neighbour activity from an xyz coordinate. Recurse
    // parameter allows for recursively checking the acitvity of the xyz
    // neighbours, with recurse=0 only checking the immediate neighbours.
    std::function<void(const TreeT&, const Coord&, const size_t)> CheckActiveNeighbours;
    CheckActiveNeighbours = [start, end, &CheckActiveNeighbours]
        (const TreeT& acc, const Coord& xyz, const size_t recurse)
    {
        CPPUNIT_ASSERT(acc.isValueOn(xyz));
        const Coord* offset(start);
        while (offset != end) {
            // optionally recurse into neighbour voxels
            if (recurse > 0) CheckActiveNeighbours(acc, xyz + *offset, recurse-1);
            CPPUNIT_ASSERT(acc.isValueOn(xyz + *offset));
            ++offset;
        }
    };

    // This test specifically tests the tile policy with various inputs

    TreeT tree;
    CPPUNIT_ASSERT(tree.empty());

    const openvdb::Index leafDim = TreeT::LeafNodeType::DIM;
    CPPUNIT_ASSERT_EQUAL(1 << 3, int(leafDim));

    { // Test behaviour with an existing active tile at (0,0,0)
        tree.clear();
        tree.addTile(/*level*/1, Coord(0), ValueType(1.0), true);
        CPPUNIT_ASSERT_EQUAL(Index32(0), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(leafDim * leafDim * leafDim), tree.activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());

        TreeT copy(tree);
        { // A single active tile exists so this has no effect
            openvdb::tools::dilateActiveValues(tree, 1, NN, openvdb::tools::IGNORE_TILES);
            CPPUNIT_ASSERT(copy.hasSameTopology(tree));
            openvdb::tools::erodeActiveValues(tree, 1, NN, openvdb::tools::IGNORE_TILES);
            CPPUNIT_ASSERT(copy.hasSameTopology(tree));
        }

        { // erode with EXPAND_TILES/PRESERVE_TILES - center tile should be expanded and eroded
            TreeT erodeexp(tree), erodepres(tree);
            openvdb::tools::erodeActiveValues(erodeexp, 1, NN, openvdb::tools::EXPAND_TILES);
            Index64 expected = (leafDim-2) * (leafDim-2) * (leafDim-2);
            CPPUNIT_ASSERT_EQUAL(Index32(1), erodeexp.leafCount());
            CPPUNIT_ASSERT_EQUAL(expected, erodeexp.activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(Index64(0), erodeexp.activeTileCount());
            CPPUNIT_ASSERT(erodeexp.probeConstLeaf(Coord(0)));

            openvdb::tools::erodeActiveValues(erodepres, 1, NN, openvdb::tools::PRESERVE_TILES);
            CPPUNIT_ASSERT(erodeexp.hasSameTopology(erodepres));
        }

        { // dilate
            openvdb::tools::dilateActiveValues(tree, 1, NN, openvdb::tools::EXPAND_TILES);
            Index64 expected = leafDim * leafDim * leafDim;
            if (NN == openvdb::tools::NN_FACE)             expected +=  (leafDim * leafDim) * 6; // faces
            if (NN == openvdb::tools::NN_FACE_EDGE)        expected += ((leafDim * leafDim) * 6) +  (leafDim) * 12; // edges
            if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) expected += ((leafDim * leafDim) * 6) + ((leafDim) * 12) + 8; // edges
            CPPUNIT_ASSERT_EQUAL(Index32(1+offsets), tree.leafCount());
            CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(Index64(0), tree.activeTileCount());
            // Check actual values around center node faces
            CPPUNIT_ASSERT(tree.probeConstLeaf(Coord(0)));
            CPPUNIT_ASSERT(tree.probeConstLeaf(Coord(0))->isDense());
            for (int i = 0; i < int(leafDim); ++i) {
                for (int j = 0; j < int(leafDim); ++j) {
                    CheckActiveNeighbours(tree, {i,j,0}, 0);
                    CheckActiveNeighbours(tree, {i,0,j}, 0);
                    CheckActiveNeighbours(tree, {0,i,j}, 0);
                    CheckActiveNeighbours(tree, {i,j,leafDim-1}, 0);
                    CheckActiveNeighbours(tree, {i,leafDim-1,j}, 0);
                    CheckActiveNeighbours(tree, {leafDim-1,i,j}, 0);
                }
            }

            // Voxelize the original copy and run with IGNORE_TILES - should produce the same result
            copy.voxelizeActiveTiles();
            openvdb::tools::dilateActiveValues(copy, 1, NN, openvdb::tools::IGNORE_TILES);
            CPPUNIT_ASSERT(copy.hasSameTopology(tree));
        }

        { // erode the dilated result
            TreeT erode(tree);
            openvdb::tools::erodeActiveValues(erode, 1, NN, openvdb::tools::IGNORE_TILES);
            Index64 expected = leafDim * leafDim * leafDim;
            CPPUNIT_ASSERT_EQUAL(Index32(1+offsets), erode.leafCount());
            CPPUNIT_ASSERT_EQUAL(expected, erode.activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(Index64(0), erode.activeTileCount());
            CPPUNIT_ASSERT(erode.probeConstLeaf(Coord(0)));
            CPPUNIT_ASSERT(erode.probeConstLeaf(Coord(0))->isDense());
        }

        // clear
        tree.clear();
        copy.clear();
        tree.addTile(/*level*/1, Coord(0), ValueType(1.0), true);
        copy.addTile(/*level*/1, Coord(0), ValueType(1.0), true);
        copy.voxelizeActiveTiles();

        { // dilate both with PRESERVE_TILES
            openvdb::tools::dilateActiveValues(tree, 1, NN, openvdb::tools::PRESERVE_TILES);
            openvdb::tools::dilateActiveValues(copy, 1, NN, openvdb::tools::PRESERVE_TILES);
            Index64 expected = leafDim * leafDim * leafDim;
            if (NN == openvdb::tools::NN_FACE)             expected +=  (leafDim * leafDim) * 6; // faces
            if (NN == openvdb::tools::NN_FACE_EDGE)        expected += ((leafDim * leafDim) * 6) +  (leafDim) * 12; // edges
            if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) expected += ((leafDim * leafDim) * 6) + ((leafDim) * 12) + 8; // edges

            CPPUNIT_ASSERT_EQUAL(Index32(offsets), tree.leafCount());
            CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());
            CPPUNIT_ASSERT(copy.hasSameTopology(tree));

            // Check actual values around center node faces
            CPPUNIT_ASSERT(!tree.probeConstLeaf(Coord(0)));
            CPPUNIT_ASSERT(tree.isValueOn(Coord(0)));
            for (int i = 0; i < int(leafDim); ++i) {
                for (int j = 0; j < int(leafDim); ++j) {
                    CheckActiveNeighbours(tree, {i,j,0}, 0);
                    CheckActiveNeighbours(tree, {i,0,j}, 0);
                    CheckActiveNeighbours(tree, {0,i,j}, 0);
                    CheckActiveNeighbours(tree, {i,j,leafDim-1}, 0);
                    CheckActiveNeighbours(tree, {i,leafDim-1,j}, 0);
                    CheckActiveNeighbours(tree, {leafDim-1,i,j}, 0);
                }
            }
        }

        { // final erode with PRESERVE_TILES
            TreeT erode(tree); // 10x10x10 filled tree, erode back down to a tile
            openvdb::tools::erodeActiveValues(erode, 1, NN, openvdb::tools::PRESERVE_TILES);
            // PRESERVE_TILES will prune the result
            Index64 expected = leafDim * leafDim * leafDim;
            CPPUNIT_ASSERT_EQUAL(Index32(0), erode.leafCount());
            CPPUNIT_ASSERT_EQUAL(expected, erode.activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(Index64(1), erode.activeTileCount());
            CPPUNIT_ASSERT(!erode.probeConstLeaf(Coord(0)));
            CPPUNIT_ASSERT(erode.isValueOn(Coord(0)));
        }
    }
    { // Test tile preservation with voxel topology - create an active, leaf node-sized tile and a single edge voxel
        tree.clear();
        tree.addTile(/*level*/1, Coord(0), ValueType(1.0), true);
        CPPUNIT_ASSERT_EQUAL(Index32(0), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(leafDim * leafDim * leafDim), tree.activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());

        const Coord xyz(leafDim, leafDim >> 1, leafDim >> 1);
        tree.setValue(xyz, 1.0);
        Index64 expected = leafDim * leafDim * leafDim + 1;
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());

        { // Test tile is preserve with IGNORE_TILES but only the corner gets dilated
            openvdb::tools::dilateActiveValues(tree, 1, NN, openvdb::tools::IGNORE_TILES);
            CheckActiveNeighbours(tree, xyz, 0);

            if (NN == openvdb::tools::NN_FACE)             expected += offsets - 1; // 1 overlapping with tile
            if (NN == openvdb::tools::NN_FACE_EDGE)        expected += offsets - 5; // 5 overlapping
            if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) expected += offsets - 9; // 9 overlapping
            CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());

            // Test all topology is dilated but tile is preserved
            openvdb::tools::dilateActiveValues(tree, 1, NN, openvdb::tools::PRESERVE_TILES);
            CheckActiveNeighbours(tree, xyz, /*recurse*/1);

            // Check actual values around center node faces
            CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());
            CPPUNIT_ASSERT_EQUAL(Index32(offsets), tree.leafCount());
            CPPUNIT_ASSERT(!tree.probeConstLeaf(Coord(0)));
            CPPUNIT_ASSERT(tree.isValueOn(Coord(0)));
            for (int i = 0; i < int(leafDim); ++i) {
                for (int j = 0; j < int(leafDim); ++j) {
                    CheckActiveNeighbours(tree, {i,j,0}, 0);
                    CheckActiveNeighbours(tree, {i,0,j}, 0);
                    CheckActiveNeighbours(tree, {0,i,j}, 0);
                    CheckActiveNeighbours(tree, {i,j,leafDim-1}, 0);
                    CheckActiveNeighbours(tree, {i,leafDim-1,j}, 0);
                    CheckActiveNeighbours(tree, {leafDim-1,i,j}, 0);
                }
            }
        }
        { // Test tile is preserved with erosions IGNORE_TILES, irrespective of iterations
            openvdb::tools::erodeActiveValues(tree, 10, NN, openvdb::tools::IGNORE_TILES);
            CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());
            CPPUNIT_ASSERT_EQUAL(Index32(offsets), tree.leafCount());
            CPPUNIT_ASSERT_EQUAL(Index64(leafDim * leafDim * leafDim), tree.activeVoxelCount());
            CPPUNIT_ASSERT(!tree.probeConstLeaf(Coord(0)));
            CPPUNIT_ASSERT(tree.isValueOn(Coord(0)));
        }
    }
    { // Test constant leaf nodes are pruned with PRESERVE_TILES
        constexpr bool IsMask = std::is_same<ValueType, bool>::value;
        tree.clear();
        // For mask trees, the bg value is the active state, so make sure its 0.
        // the second partial leaf should still become dense
        const ValueType bg = IsMask ? 0.0 : 1.0;
        tree.root().setBackground(bg, /*update-child*/false);
        // partial leaf node which will become dense, but not constant
        tree.fill(CoordBBox(Coord(0,0,1), Coord(leafDim-1)), ValueType(2.0));
        // partial leaf node which will become dense and constant
        tree.fill(CoordBBox(Coord(leafDim*3,0,1), Coord(leafDim*3 + leafDim - 1, leafDim-1, leafDim-1)), ValueType(1.0));
        // dense leaf node
        tree.touchLeaf(Coord(leafDim*6, 0, 0))->setValuesOn();
        Index64 expected = (leafDim * leafDim * leafDim) +
            ((leafDim * leafDim * leafDim) - (leafDim * leafDim)) * 2;
        CPPUNIT_ASSERT_EQUAL(Index32(3), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(0), tree.activeTileCount());

        // Dilate and preserve - first leaf node becomes dense
        // (regardless of NN) but not pruned, second dense and pruned,
        // third, already being constant, is also pruned
        openvdb::tools::dilateActiveValues(tree, 1,
            NN, openvdb::tools::PRESERVE_TILES);

        // For mask grids, both partial leaf nodes that become dense should be pruned
        if (IsMask)  CPPUNIT_ASSERT_EQUAL(Index64(3), tree.activeTileCount());
        else         CPPUNIT_ASSERT_EQUAL(Index64(2), tree.activeTileCount());

        if (NN == openvdb::tools::NN_FACE)             expected = offsets*3 -2;
        if (NN == openvdb::tools::NN_FACE_EDGE)        expected = offsets*3 -10;
        if (NN == openvdb::tools::NN_FACE_EDGE_VERTEX) expected = offsets*3 -18;
        if (!IsMask) expected += 1;
        CPPUNIT_ASSERT_EQUAL(Index32(expected), tree.leafCount());
        // first
        if (IsMask) {
            // should have been pruned
            CPPUNIT_ASSERT(!tree.probeConstLeaf(Coord(0)));
            CPPUNIT_ASSERT(tree.isValueOn(Coord(0)));
        }
        else {
            CPPUNIT_ASSERT(tree.probeConstLeaf(Coord(0)));
            CPPUNIT_ASSERT(tree.probeConstLeaf(Coord(0))->isDense());
        }
        //second
        CPPUNIT_ASSERT(!tree.probeConstLeaf(Coord(leafDim*3, 0, 0)));
        CPPUNIT_ASSERT(tree.isValueOn(Coord(leafDim*3, 0, 0)));
        // third
        CPPUNIT_ASSERT(!tree.probeConstLeaf(Coord(leafDim*6, 0, 0)));
        CPPUNIT_ASSERT(tree.isValueOn(Coord(leafDim*6, 0, 0)));

        // test erosion PRESERVE_TILES correctly erodes and prunes the result
        openvdb::tools::erodeActiveValues(tree, 1, NN,
            openvdb::tools::PRESERVE_TILES);
        expected = (leafDim * leafDim * leafDim) +
            ((leafDim * leafDim * leafDim) - (leafDim * leafDim)) * 2;
        CPPUNIT_ASSERT_EQUAL(Index32(2), tree.leafCount());
        CPPUNIT_ASSERT_EQUAL(expected, tree.activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(1), tree.activeTileCount());
    }
}
