// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/io/GridDescriptor.h>
#include <openvdb/openvdb.h>

#include <gtest/gtest.h>


class TestGridDescriptor: public ::testing::Test
{
};


TEST_F(TestGridDescriptor, testIO)
{
    using namespace openvdb::io;
    using namespace openvdb;

    typedef FloatGrid GridType;

    GridDescriptor gd(GridDescriptor::addSuffix("temperature", 2), GridType::gridType());
    gd.setInstanceParentName("temperature_32bit");

    gd.setGridPos(123);
    gd.setBlockPos(234);
    gd.setEndPos(567);

    // write out the gd.
    std::ostringstream ostr(std::ios_base::binary);

    gd.writeHeader(ostr);
    gd.writeStreamPos(ostr);

    // Read in the gd.
    std::istringstream istr(ostr.str(), std::ios_base::binary);

    // Since the input is only a fragment of a VDB file (in particular,
    // it doesn't have a header), set the file format version number explicitly.
    io::setCurrentVersion(istr);

    GridDescriptor gd2;

    EXPECT_THROW(gd2.read(istr), openvdb::LookupError);

    // Register the grid.
    GridBase::clearRegistry();
    GridType::registerGrid();

    // seek back and read again.
    istr.seekg(0, std::ios_base::beg);
    GridBase::Ptr grid;
    EXPECT_NO_THROW(grid = gd2.read(istr));

    EXPECT_EQ(gd.gridName(), gd2.gridName());
    EXPECT_EQ(gd.uniqueName(), gd2.uniqueName());
    EXPECT_EQ(gd.gridType(), gd2.gridType());
    EXPECT_EQ(gd.instanceParentName(), gd2.instanceParentName());
    EXPECT_TRUE(grid.get() != NULL);
    EXPECT_EQ(GridType::gridType(), grid->type());
    EXPECT_EQ(gd.getGridPos(), gd2.getGridPos());
    EXPECT_EQ(gd.getBlockPos(), gd2.getBlockPos());
    EXPECT_EQ(gd.getEndPos(), gd2.getEndPos());

    // Clear the registry when we are done.
    GridBase::clearRegistry();
}


TEST_F(TestGridDescriptor, testCopy)
{
    using namespace openvdb::io;
    using namespace openvdb;

    typedef FloatGrid GridType;

    GridDescriptor gd("temperature", GridType::gridType());
    gd.setInstanceParentName("temperature_32bit");

    gd.setGridPos(123);
    gd.setBlockPos(234);
    gd.setEndPos(567);

    GridDescriptor gd2;

    // do the copy
    gd2 = gd;

    EXPECT_EQ(gd.gridName(), gd2.gridName());
    EXPECT_EQ(gd.uniqueName(), gd2.uniqueName());
    EXPECT_EQ(gd.gridType(), gd2.gridType());
    EXPECT_EQ(gd.instanceParentName(), gd2.instanceParentName());
    EXPECT_EQ(gd.getGridPos(), gd2.getGridPos());
    EXPECT_EQ(gd.getBlockPos(), gd2.getBlockPos());
    EXPECT_EQ(gd.getEndPos(), gd2.getEndPos());
}


TEST_F(TestGridDescriptor, testName)
{
    using openvdb::Name;
    using openvdb::io::GridDescriptor;

    const std::string typ = openvdb::FloatGrid::gridType();

    Name name("test");
    GridDescriptor gd(name, typ);

    // Verify that the grid name and the unique name are equivalent
    // when the unique name has no suffix.
    EXPECT_EQ(name, gd.gridName());
    EXPECT_EQ(name, gd.uniqueName());
    EXPECT_EQ(name, GridDescriptor::nameAsString(name));
    EXPECT_EQ(name, GridDescriptor::stripSuffix(name));

    // Add a suffix.
    name = GridDescriptor::addSuffix("test", 2);
    gd = GridDescriptor(name, typ);

    // Verify that the grid name and the unique name differ
    // when the unique name has a suffix.
    EXPECT_EQ(name, gd.uniqueName());
    EXPECT_TRUE(gd.gridName() != gd.uniqueName());
    EXPECT_EQ(GridDescriptor::stripSuffix(name), gd.gridName());
    EXPECT_EQ(Name("test[2]"), GridDescriptor::nameAsString(name));

    // As above, but with a longer suffix
    name = GridDescriptor::addSuffix("test", 13);
    gd = GridDescriptor(name, typ);

    EXPECT_EQ(name, gd.uniqueName());
    EXPECT_TRUE(gd.gridName() != gd.uniqueName());
    EXPECT_EQ(GridDescriptor::stripSuffix(name), gd.gridName());
    EXPECT_EQ(Name("test[13]"), GridDescriptor::nameAsString(name));

    // Multiple suffixes aren't supported, but verify that
    // they behave reasonably, at least.
    name = GridDescriptor::addSuffix(name, 4);
    gd = GridDescriptor(name, typ);

    EXPECT_EQ(name, gd.uniqueName());
    EXPECT_TRUE(gd.gridName() != gd.uniqueName());
    EXPECT_EQ(GridDescriptor::stripSuffix(name), gd.gridName());
}
