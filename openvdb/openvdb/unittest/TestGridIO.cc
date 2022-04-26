// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <gtest/gtest.h>

#include <cstdio> // for remove()


class TestGridIO: public ::testing::Test
{
public:
    typedef openvdb::tree::Tree<
        openvdb::tree::RootNode<
        openvdb::tree::InternalNode<
        openvdb::tree::InternalNode<
        openvdb::tree::InternalNode<
        openvdb::tree::LeafNode<float, 2>, 3>, 4>, 5> > >
        Float5432Tree;
    typedef openvdb::Grid<Float5432Tree> Float5432Grid;

    void SetUp() override    { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }

protected:
    template<typename GridType> void readAllTest();
};


////////////////////////////////////////


template<typename GridType>
void
TestGridIO::readAllTest()
{
    using namespace openvdb;

    typedef typename GridType::TreeType TreeType;
    typedef typename TreeType::Ptr TreePtr;
    typedef typename TreeType::ValueType ValueT;
    typedef typename TreeType::NodeCIter NodeCIter;
    const ValueT zero = zeroVal<ValueT>();

    // For each level of the tree, compute a bit mask for use in converting
    // global coordinates to node origins for nodes at that level.
    // That is, node_origin = global_coordinates & mask[node_level].
    std::vector<Index> mask;
    TreeType::getNodeLog2Dims(mask);
    const size_t height = mask.size();
    for (size_t i = 0; i < height; ++i) {
        Index dim = 0;
        for (size_t j = i; j < height; ++j) dim += mask[j];
        mask[i] = ~((1 << dim) - 1);
    }
    const Index childDim = 1 + ~(mask[0]);

    // Choose sample coordinate pairs (coord0, coord1) and (coord0, coord2)
    // that are guaranteed to lie in different children of the root node
    // (because they are separated by more than the child node dimension).
    const Coord
        coord0(0, 0, 0),
        coord1(int(1.1 * childDim), 0, 0),
        coord2(0, int(1.1 * childDim), 0);

    // Create trees.
    TreePtr
        tree1(new TreeType(zero + 1)),
        tree2(new TreeType(zero + 2));

    // Set some values.
    tree1->setValue(coord0, zero + 5);
    tree1->setValue(coord1, zero + 6);
    tree2->setValue(coord0, zero + 10);
    tree2->setValue(coord2, zero + 11);

    // Create grids with trees and assign transforms.
    math::Transform::Ptr trans1(math::Transform::createLinearTransform(0.1)),
        trans2(math::Transform::createLinearTransform(0.1));
    GridBase::Ptr grid1 = createGrid(tree1), grid2 = createGrid(tree2);
    grid1->setTransform(trans1);
    grid1->setName("density");
    grid2->setTransform(trans2);
    grid2->setName("temperature");

    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    EXPECT_EQ(ValueT(zero + 5), tree1->getValue(coord0));
    EXPECT_EQ(ValueT(zero + 6), tree1->getValue(coord1));
    EXPECT_EQ(ValueT(zero + 10), tree2->getValue(coord0));
    EXPECT_EQ(ValueT(zero + 11), tree2->getValue(coord2));
    OPENVDB_NO_FP_EQUALITY_WARNING_END

    // count[d] is the number of nodes already visited at depth d.
    // There should be exactly two nodes at each depth (apart from the root).
    std::vector<int> count(height, 0);

    // Verify that tree1 has correct node origins.
    for (NodeCIter iter = tree1->cbeginNode(); iter; ++iter) {
        const Index depth = iter.getDepth();
        const Coord expected[2] = {
            coord0 & mask[depth], // origin of the first node at this depth
            coord1 & mask[depth]  // origin of the second node at this depth
        };
        EXPECT_EQ(expected[count[depth]], iter.getCoord());
        ++count[depth];
    }
    // Verify that tree2 has correct node origins.
    count.assign(height, 0); // reset node counts
    for (NodeCIter iter = tree2->cbeginNode(); iter; ++iter) {
        const Index depth = iter.getDepth();
        const Coord expected[2] = { coord0 & mask[depth], coord2 & mask[depth] };
        EXPECT_EQ(expected[count[depth]], iter.getCoord());
        ++count[depth];
    }

    MetaMap::Ptr meta(new MetaMap);
    meta->insertMeta("author", StringMetadata("Einstein"));
    meta->insertMeta("year", Int32Metadata(2009));

    GridPtrVecPtr grids(new GridPtrVec);
    grids->push_back(grid1);
    grids->push_back(grid2);

    // Write grids and metadata out to a file.
    {
        io::File vdbfile("something.vdb2");
        vdbfile.write(*grids, *meta);
    }
    meta.reset();
    grids.reset();

    io::File vdbfile("something.vdb2");
    EXPECT_THROW(vdbfile.getGrids(), openvdb::IoError); // file has not been opened

    // Read the grids back in.
    vdbfile.open();
    EXPECT_TRUE(vdbfile.isOpen());

    grids = vdbfile.getGrids();
    meta = vdbfile.getMetadata();

    // Ensure we have the metadata.
    EXPECT_TRUE(meta.get() != NULL);
    EXPECT_EQ(2, int(meta->metaCount()));
    EXPECT_EQ(std::string("Einstein"), meta->metaValue<std::string>("author"));
    EXPECT_EQ(2009, meta->metaValue<int32_t>("year"));

    // Ensure we got both grids.
    EXPECT_TRUE(grids.get() != NULL);
    EXPECT_EQ(2, int(grids->size()));

    grid1.reset();
    grid1 = findGridByName(*grids, "density");
    EXPECT_TRUE(grid1.get() != NULL);
    TreePtr density = gridPtrCast<GridType>(grid1)->treePtr();
    EXPECT_TRUE(density.get() != NULL);

    grid2.reset();
    grid2 = findGridByName(*grids, "temperature");
    EXPECT_TRUE(grid2.get() != NULL);
    TreePtr temperature = gridPtrCast<GridType>(grid2)->treePtr();
    EXPECT_TRUE(temperature.get() != NULL);

    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    EXPECT_EQ(ValueT(zero + 5), density->getValue(coord0));
    EXPECT_EQ(ValueT(zero + 6), density->getValue(coord1));
    EXPECT_EQ(ValueT(zero + 10), temperature->getValue(coord0));
    EXPECT_EQ(ValueT(zero + 11), temperature->getValue(coord2));
    OPENVDB_NO_FP_EQUALITY_WARNING_END

    // Check if we got the correct node origins.
    count.assign(height, 0);
    for (NodeCIter iter = density->cbeginNode(); iter; ++iter) {
        const Index depth = iter.getDepth();
        const Coord expected[2] = { coord0 & mask[depth], coord1 & mask[depth] };
        EXPECT_EQ(expected[count[depth]], iter.getCoord());
        ++count[depth];
    }
    count.assign(height, 0);
    for (NodeCIter iter = temperature->cbeginNode(); iter; ++iter) {
        const Index depth = iter.getDepth();
        const Coord expected[2] = { coord0 & mask[depth], coord2 & mask[depth] };
        EXPECT_EQ(expected[count[depth]], iter.getCoord());
        ++count[depth];
    }

    vdbfile.close();

    ::remove("something.vdb2");
}

TEST_F(TestGridIO, testReadAllBool) { readAllTest<openvdb::BoolGrid>(); }
TEST_F(TestGridIO, testReadAllFloat) { readAllTest<openvdb::FloatGrid>(); }
TEST_F(TestGridIO, testReadAllVec3S) { readAllTest<openvdb::Vec3SGrid>(); }
TEST_F(TestGridIO, testReadAllFloat5432) { Float5432Grid::registerGrid(); readAllTest<Float5432Grid>(); }
