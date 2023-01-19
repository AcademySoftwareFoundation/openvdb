// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/io/TempFile.h>
#include <openvdb/math/Math.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointGroup.h>

#include <gtest/gtest.h>


using namespace openvdb;
using namespace openvdb::points;

class TestPointConversion: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointConversion


// Simple Attribute Wrapper
template <typename T>
struct AttributeWrapper
{
    using ValueType     = T;
    using PosType       = T;
    using value_type    = T;

    struct Handle
    {
        Handle(AttributeWrapper<T>& attribute)
            : mBuffer(attribute.mAttribute)
            , mStride(attribute.mStride) { }

        template <typename ValueType>
        void set(size_t n, openvdb::Index m, const ValueType& value) {
            mBuffer[n * mStride + m] = static_cast<T>(value);
        }

        template <typename ValueType>
        void set(size_t n, openvdb::Index m, const openvdb::math::Vec3<ValueType>& value) {
            mBuffer[n * mStride + m] = static_cast<T>(value);
        }

    private:
        std::vector<T>& mBuffer;
        Index mStride;
    }; // struct Handle

    explicit AttributeWrapper(const Index stride) : mStride(stride) { }

    void expand() { }
    void compact() { }

    void resize(const size_t n) { mAttribute.resize(n); }
    size_t size() const { return mAttribute.size(); }

    std::vector<T>& buffer() { return mAttribute; }

    template <typename ValueT>
    void get(ValueT& value, size_t n, openvdb::Index m = 0) const { value = mAttribute[n * mStride + m]; }
    template <typename ValueT>
    void getPos(size_t n, ValueT& value) const { this->get<ValueT>(value, n); }

private:
    std::vector<T> mAttribute;
    Index mStride;
}; // struct AttributeWrapper


struct GroupWrapper
{
    GroupWrapper() = default;

    void setOffsetOn(openvdb::Index index) {
        mGroup[index] = short(1);
    }

    void finalize() { }

    void resize(const size_t n) { mGroup.resize(n, short(0)); }
    size_t size() const { return mGroup.size(); }

    std::vector<short>& buffer() { return mGroup; }

private:
    std::vector<short> mGroup;
}; // struct GroupWrapper


struct PointData
{
    int id;
    Vec3f position;
    Vec3i xyz;
    float uniform;
    openvdb::Name string;
    short group;

    bool operator<(const PointData& other) const { return id < other.id; }
}; // PointData


// Generate random points by uniformly distributing points
// on a unit-sphere.
inline void
genPoints(const int numPoints, const double scale, const bool stride,
    AttributeWrapper<Vec3f>& position,
    AttributeWrapper<int>& xyz,
    AttributeWrapper<int>& id,
    AttributeWrapper<float>& uniform,
    AttributeWrapper<openvdb::Name>& string,
    GroupWrapper& group)
{
    // init
    openvdb::math::Random01 randNumber(0);
    const int n = int(std::sqrt(double(numPoints)));
    const double xScale = (2.0 * openvdb::math::pi<double>()) / double(n);
    const double yScale = openvdb::math::pi<double>() / double(n);

    double x, y, theta, phi;
    openvdb::Vec3f pos;

    position.resize(n*n);
    xyz.resize(stride ? n*n*3 : 1);
    id.resize(n*n);
    uniform.resize(n*n);
    string.resize(n*n);
    group.resize(n*n);

    AttributeWrapper<Vec3f>::Handle positionHandle(position);
    AttributeWrapper<int>::Handle xyzHandle(xyz);
    AttributeWrapper<int>::Handle idHandle(id);
    AttributeWrapper<float>::Handle uniformHandle(uniform);
    AttributeWrapper<openvdb::Name>::Handle stringHandle(string);

    size_t i = 0;

    // loop over a [0 to n) x [0 to n) grid.
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {

            // jitter, move to random pos. inside the current cell
            x = double(a) + randNumber();
            y = double(b) + randNumber();

            // remap to a lat/long map
            theta = y * yScale; // [0 to PI]
            phi   = x * xScale; // [0 to 2PI]

            // convert to cartesian coordinates on a unit sphere.
            // spherical coordinate triplet (r=1, theta, phi)
            pos[0] = static_cast<float>(std::sin(theta)*std::cos(phi)*scale);
            pos[1] = static_cast<float>(std::sin(theta)*std::sin(phi)*scale);
            pos[2] = static_cast<float>(std::cos(theta)*scale);

            positionHandle.set(i, /*stride*/0, pos);
            idHandle.set(i, /*stride*/0, i);
            uniformHandle.set(i, /*stride*/0, 100.0f);

            if (stride)
            {
                xyzHandle.set(i, 0, static_cast<int>(i));
                xyzHandle.set(i, 1, static_cast<int>(i*i));
                xyzHandle.set(i, 2, static_cast<int>(i*i*i));
            }

            // add points with even id to the group
            if ((i % 2) == 0) {
                group.setOffsetOn(static_cast<int>(i));
                stringHandle.set(i, /*stride*/0, "testA");
            }
            else {
                stringHandle.set(i, /*stride*/0, "testB");
            }

            i++;
        }
    }
}


////////////////////////////////////////


TEST_F(TestPointConversion, testPointConversion)
{
    // generate points

    const size_t count(1000000);

    AttributeWrapper<Vec3f> position(1);
    AttributeWrapper<int> xyz(1);
    AttributeWrapper<int> id(1);
    AttributeWrapper<float> uniform(1);
    AttributeWrapper<openvdb::Name> string(1);
    GroupWrapper group;

    genPoints(count, /*scale=*/ 100.0, /*stride=*/false,
                position, xyz, id, uniform, string, group);

    EXPECT_EQ(position.size(), count);
    EXPECT_EQ(id.size(), count);
    EXPECT_EQ(uniform.size(), count);
    EXPECT_EQ(string.size(), count);
    EXPECT_EQ(group.size(), count);

    // convert point positions into a Point Data Grid

    const float voxelSize = 1.0f;
    openvdb::math::Transform::Ptr transform(openvdb::math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(position, *transform);
    PointDataGrid::Ptr pointDataGrid = createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, position, *transform);

    tools::PointIndexTree& indexTree = pointIndexGrid->tree();
    PointDataTree& tree = pointDataGrid->tree();

    // add id and populate

    appendAttribute<int>(tree, "id");
    populateAttribute<PointDataTree, tools::PointIndexTree, AttributeWrapper<int>>(tree, indexTree, "id", id);

    // add uniform and populate

    appendAttribute<float>(tree, "uniform");
    populateAttribute<PointDataTree, tools::PointIndexTree, AttributeWrapper<float>>(tree, indexTree, "uniform", uniform);

    // add string and populate

    appendAttribute<Name>(tree, "string");

    // reset the descriptors
    PointDataTree::LeafIter leafIter = tree.beginLeaf();
    const AttributeSet::Descriptor& descriptor = leafIter->attributeSet().descriptor();
    auto newDescriptor = std::make_shared<AttributeSet::Descriptor>(descriptor);
    for (; leafIter; ++leafIter) {
        leafIter->resetDescriptor(newDescriptor);
    }

    populateAttribute<PointDataTree, tools::PointIndexTree, AttributeWrapper<openvdb::Name>>(
        tree, indexTree, "string", string);

    // add group and set membership

    appendGroup(tree, "test");
    setGroup(tree, indexTree, group.buffer(), "test");

    EXPECT_EQ(indexTree.leafCount(), tree.leafCount());

    // read/write grid to a temp file

    const std::string filename = "testPointConversion.vdb";
    io::File fileOut(filename);

    GridCPtrVec grids;
    grids.push_back(pointDataGrid);

    fileOut.write(grids);

    fileOut.close();

    io::File fileIn(filename);
    fileIn.open();

    GridPtrVecPtr readGrids = fileIn.getGrids();

    fileIn.close();

    EXPECT_EQ(readGrids->size(), size_t(1));

    pointDataGrid = GridBase::grid<PointDataGrid>((*readGrids)[0]);
    PointDataTree& inputTree = pointDataGrid->tree();

    // create accessor and iterator for Point Data Tree

    PointDataTree::LeafCIter leafCIter = inputTree.cbeginLeaf();

    EXPECT_EQ(5, int(leafCIter->attributeSet().size()));

    EXPECT_TRUE(leafCIter->attributeSet().find("id") != AttributeSet::INVALID_POS);
    EXPECT_TRUE(leafCIter->attributeSet().find("uniform") != AttributeSet::INVALID_POS);
    EXPECT_TRUE(leafCIter->attributeSet().find("P") != AttributeSet::INVALID_POS);
    EXPECT_TRUE(leafCIter->attributeSet().find("string") != AttributeSet::INVALID_POS);

    const auto idIndex = static_cast<Index>(leafCIter->attributeSet().find("id"));
    const auto uniformIndex = static_cast<Index>(leafCIter->attributeSet().find("uniform"));
    const auto stringIndex = static_cast<Index>(leafCIter->attributeSet().find("string"));
    const AttributeSet::Descriptor::GroupIndex groupIndex =
        leafCIter->attributeSet().groupIndex("test");

    // convert back into linear point attribute data

    AttributeWrapper<Vec3f> outputPosition(1);
    AttributeWrapper<int> outputId(1);
    AttributeWrapper<float> outputUniform(1);
    AttributeWrapper<openvdb::Name> outputString(1);
    GroupWrapper outputGroup;

    // test offset the whole point block by an arbitrary amount

    Index64 startOffset = 10;

    outputPosition.resize(startOffset + position.size());
    outputId.resize(startOffset + id.size());
    outputUniform.resize(startOffset + uniform.size());
    outputString.resize(startOffset + string.size());
    outputGroup.resize(startOffset + group.size());

    std::vector<Name> includeGroups;
    std::vector<Name> excludeGroups;

    std::vector<Index64> offsets;
    MultiGroupFilter filter(includeGroups, excludeGroups, inputTree.cbeginLeaf()->attributeSet());
    pointOffsets(offsets, inputTree, filter);

    convertPointDataGridPosition(outputPosition, *pointDataGrid, offsets, startOffset, filter);
    convertPointDataGridAttribute(outputId, inputTree, offsets, startOffset, idIndex, 1, filter);
    convertPointDataGridAttribute(outputUniform, inputTree, offsets, startOffset, uniformIndex, 1, filter);
    convertPointDataGridAttribute(outputString, inputTree, offsets, startOffset, stringIndex, 1, filter);
    convertPointDataGridGroup(outputGroup, inputTree, offsets, startOffset, groupIndex, filter);

    // pack and sort the new buffers based on id

    std::vector<PointData> pointData(count);

    for (unsigned int i = 0; i < count; i++) {
        pointData[i].id = outputId.buffer()[startOffset + i];
        pointData[i].position = outputPosition.buffer()[startOffset + i];
        pointData[i].uniform = outputUniform.buffer()[startOffset + i];
        pointData[i].string = outputString.buffer()[startOffset + i];
        pointData[i].group = outputGroup.buffer()[startOffset + i];
    }

    std::sort(pointData.begin(), pointData.end());

    // compare old and new buffers

    for (unsigned int i = 0; i < count; i++)
    {
        EXPECT_EQ(id.buffer()[i], pointData[i].id);
        EXPECT_EQ(group.buffer()[i], pointData[i].group);
        EXPECT_EQ(uniform.buffer()[i], pointData[i].uniform);
        EXPECT_EQ(string.buffer()[i], pointData[i].string);
        EXPECT_NEAR(position.buffer()[i].x(), pointData[i].position.x(), /*tolerance=*/1e-6);
        EXPECT_NEAR(position.buffer()[i].y(), pointData[i].position.y(), /*tolerance=*/1e-6);
        EXPECT_NEAR(position.buffer()[i].z(), pointData[i].position.z(), /*tolerance=*/1e-6);
    }

    // convert based on even group

    const size_t halfCount = count / 2;

    outputPosition.resize(startOffset + halfCount);
    outputId.resize(startOffset + halfCount);
    outputUniform.resize(startOffset + halfCount);
    outputString.resize(startOffset + halfCount);
    outputGroup.resize(startOffset + halfCount);

    includeGroups.push_back("test");

    offsets.clear();
    MultiGroupFilter filter2(includeGroups, excludeGroups, inputTree.cbeginLeaf()->attributeSet());
    pointOffsets(offsets, inputTree, filter2);

    convertPointDataGridPosition(outputPosition, *pointDataGrid, offsets, startOffset, filter2);
    convertPointDataGridAttribute(outputId, inputTree, offsets, startOffset, idIndex, /*stride*/1, filter2);
    convertPointDataGridAttribute(outputUniform, inputTree, offsets, startOffset, uniformIndex, /*stride*/1, filter2);
    convertPointDataGridAttribute(outputString, inputTree, offsets, startOffset, stringIndex, /*stride*/1, filter2);
    convertPointDataGridGroup(outputGroup, inputTree, offsets, startOffset, groupIndex, filter2);

    EXPECT_EQ(size_t(outputPosition.size() - startOffset), size_t(halfCount));
    EXPECT_EQ(size_t(outputId.size() - startOffset), size_t(halfCount));
    EXPECT_EQ(size_t(outputUniform.size() - startOffset), size_t(halfCount));
    EXPECT_EQ(size_t(outputString.size() - startOffset), size_t(halfCount));
    EXPECT_EQ(size_t(outputGroup.size() - startOffset), size_t(halfCount));

    pointData.clear();

    for (unsigned int i = 0; i < halfCount; i++) {
        PointData data;
        data.id = outputId.buffer()[startOffset + i];
        data.position = outputPosition.buffer()[startOffset + i];
        data.uniform = outputUniform.buffer()[startOffset + i];
        data.string = outputString.buffer()[startOffset + i];
        data.group = outputGroup.buffer()[startOffset + i];
        pointData.push_back(data);
    }

    std::sort(pointData.begin(), pointData.end());

    // compare old and new buffers

    for (unsigned int i = 0; i < halfCount; i++)
    {
        EXPECT_EQ(id.buffer()[i*2], pointData[i].id);
        EXPECT_EQ(group.buffer()[i*2], pointData[i].group);
        EXPECT_EQ(uniform.buffer()[i*2], pointData[i].uniform);
        EXPECT_EQ(string.buffer()[i*2], pointData[i].string);
        EXPECT_NEAR(position.buffer()[i*2].x(), pointData[i].position.x(), /*tolerance=*/1e-6);
        EXPECT_NEAR(position.buffer()[i*2].y(), pointData[i].position.y(), /*tolerance=*/1e-6);
        EXPECT_NEAR(position.buffer()[i*2].z(), pointData[i].position.z(), /*tolerance=*/1e-6);
    }

    std::remove(filename.c_str());
}


////////////////////////////////////////


TEST_F(TestPointConversion, testPointConversionNans)
{
    // generate points

    const size_t count(25);

    AttributeWrapper<Vec3f> position(1);
    AttributeWrapper<int> xyz(1);
    AttributeWrapper<int> id(1);
    AttributeWrapper<float> uniform(1);
    AttributeWrapper<openvdb::Name> string(1);
    GroupWrapper group;

    genPoints(count, /*scale=*/ 1.0, /*stride=*/false,
        position, xyz, id, uniform, string, group);

    // set point numbers 0, 10, 20 and 24 to a nan position

    const std::vector<int> nanIndices = { 0, 10, 20, 24 };

    AttributeWrapper<Vec3f>::Handle positionHandle(position);
    const Vec3f nanPos(std::nan("0"));
    EXPECT_TRUE(nanPos.isNan());
    for (const int& idx : nanIndices) {
        positionHandle.set(idx, /*stride*/0, nanPos);
    }

    EXPECT_EQ(count, position.size());
    EXPECT_EQ(count, id.size());
    EXPECT_EQ(count, uniform.size());
    EXPECT_EQ(count, string.size());
    EXPECT_EQ(count, group.size());

    // convert point positions into a Point Data Grid

    openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform(/*voxelsize*/1.0f);

    tools::PointIndexGrid::Ptr pointIndexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(position, *transform);
    PointDataGrid::Ptr pointDataGrid = createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, position, *transform);

    tools::PointIndexTree& indexTree = pointIndexGrid->tree();
    PointDataTree& tree = pointDataGrid->tree();

    // set expected point count to the total minus the number of nan positions
    const size_t expected = count - nanIndices.size();
    EXPECT_EQ(expected, static_cast<size_t>(pointCount(tree)));

    // add id and populate

    appendAttribute<int>(tree, "id");
    populateAttribute<PointDataTree, tools::PointIndexTree, AttributeWrapper<int>>(tree, indexTree, "id", id);

    // add uniform and populate

    appendAttribute<float>(tree, "uniform");
    populateAttribute<PointDataTree, tools::PointIndexTree, AttributeWrapper<float>>(tree, indexTree, "uniform", uniform);

    // add string and populate

    appendAttribute<Name>(tree, "string");
    populateAttribute<PointDataTree, tools::PointIndexTree, AttributeWrapper<openvdb::Name>>(
        tree, indexTree, "string", string);

    // add group and set membership

    appendGroup(tree, "test");
    setGroup(tree, indexTree, group.buffer(), "test");

    // create accessor and iterator for Point Data Tree

    const auto leafCIter = tree.cbeginLeaf();

    EXPECT_EQ(5, int(leafCIter->attributeSet().size()));

    EXPECT_TRUE(leafCIter->attributeSet().find("id") != AttributeSet::INVALID_POS);
    EXPECT_TRUE(leafCIter->attributeSet().find("uniform") != AttributeSet::INVALID_POS);
    EXPECT_TRUE(leafCIter->attributeSet().find("P") != AttributeSet::INVALID_POS);
    EXPECT_TRUE(leafCIter->attributeSet().find("string") != AttributeSet::INVALID_POS);

    const auto idIndex = static_cast<Index>(leafCIter->attributeSet().find("id"));
    const auto uniformIndex = static_cast<Index>(leafCIter->attributeSet().find("uniform"));
    const auto stringIndex = static_cast<Index>(leafCIter->attributeSet().find("string"));
    const AttributeSet::Descriptor::GroupIndex groupIndex =
        leafCIter->attributeSet().groupIndex("test");

    // convert back into linear point attribute data

    AttributeWrapper<Vec3f> outputPosition(1);
    AttributeWrapper<int> outputId(1);
    AttributeWrapper<float> outputUniform(1);
    AttributeWrapper<openvdb::Name> outputString(1);
    GroupWrapper outputGroup;

    outputPosition.resize(position.size());
    outputId.resize(id.size());
    outputUniform.resize(uniform.size());
    outputString.resize(string.size());
    outputGroup.resize(group.size());

    std::vector<Index64> offsets;
    pointOffsets(offsets, tree);

    convertPointDataGridPosition(outputPosition, *pointDataGrid, offsets, 0);
    convertPointDataGridAttribute(outputId, tree, offsets, 0, idIndex, 1);
    convertPointDataGridAttribute(outputUniform, tree, offsets, 0, uniformIndex, 1);
    convertPointDataGridAttribute(outputString, tree, offsets, 0, stringIndex, 1);
    convertPointDataGridGroup(outputGroup, tree, offsets, 0, groupIndex);

    // pack and sort the new buffers based on id

    std::vector<PointData> pointData(expected);

    for (unsigned int i = 0; i < expected; i++) {
        pointData[i].id = outputId.buffer()[i];
        pointData[i].position = outputPosition.buffer()[i];
        pointData[i].uniform = outputUniform.buffer()[i];
        pointData[i].string = outputString.buffer()[i];
        pointData[i].group = outputGroup.buffer()[i];
    }

    std::sort(pointData.begin(), pointData.end());

    // compare old and new buffers, taking into account the nan position
    // which should not have been converted

    for (unsigned int i = 0; i < expected; ++i)
    {
        size_t iOffset = i;
        for (const int& idx : nanIndices) {
            if (int(iOffset) >= idx) iOffset += 1;
        }

        EXPECT_EQ(id.buffer()[iOffset], pointData[i].id);
        EXPECT_EQ(group.buffer()[iOffset], pointData[i].group);
        EXPECT_EQ(uniform.buffer()[iOffset], pointData[i].uniform);
        EXPECT_EQ(string.buffer()[iOffset], pointData[i].string);
        EXPECT_NEAR(position.buffer()[iOffset].x(), pointData[i].position.x(), /*tolerance=*/1e-6);
        EXPECT_NEAR(position.buffer()[iOffset].y(), pointData[i].position.y(), /*tolerance=*/1e-6);
        EXPECT_NEAR(position.buffer()[iOffset].z(), pointData[i].position.z(), /*tolerance=*/1e-6);
    }
}


////////////////////////////////////////


TEST_F(TestPointConversion, testStride)
{
    // generate points

    const size_t count(40000);

    AttributeWrapper<Vec3f> position(1);
    AttributeWrapper<int> xyz(3);
    AttributeWrapper<int> id(1);
    AttributeWrapper<float> uniform(1);
    AttributeWrapper<openvdb::Name> string(1);
    GroupWrapper group;

    genPoints(count, /*scale=*/ 100.0, /*stride=*/true,
                position, xyz, id, uniform, string, group);

    EXPECT_EQ(position.size(), count);
    EXPECT_EQ(xyz.size(), count*3);
    EXPECT_EQ(id.size(), count);

    // convert point positions into a Point Data Grid

    const float voxelSize = 1.0f;
    openvdb::math::Transform::Ptr transform(openvdb::math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(position, *transform);
    PointDataGrid::Ptr pointDataGrid = createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, position, *transform);

    tools::PointIndexTree& indexTree = pointIndexGrid->tree();
    PointDataTree& tree = pointDataGrid->tree();

    // add id and populate

    appendAttribute<int>(tree, "id");
    populateAttribute<PointDataTree, tools::PointIndexTree, AttributeWrapper<int>>(tree, indexTree, "id", id);

    // add xyz and populate

    appendAttribute<int>(tree, "xyz", 0, /*stride=*/3);
    populateAttribute<PointDataTree, tools::PointIndexTree, AttributeWrapper<int>>(tree, indexTree, "xyz", xyz, /*stride=*/3);

    // create accessor and iterator for Point Data Tree

    PointDataTree::LeafCIter leafCIter = tree.cbeginLeaf();

    EXPECT_EQ(3, int(leafCIter->attributeSet().size()));

    EXPECT_TRUE(leafCIter->attributeSet().find("id") != AttributeSet::INVALID_POS);
    EXPECT_TRUE(leafCIter->attributeSet().find("P") != AttributeSet::INVALID_POS);
    EXPECT_TRUE(leafCIter->attributeSet().find("xyz") != AttributeSet::INVALID_POS);

    const auto idIndex = static_cast<Index>(leafCIter->attributeSet().find("id"));
    const auto xyzIndex = static_cast<Index>(leafCIter->attributeSet().find("xyz"));

    // convert back into linear point attribute data

    AttributeWrapper<Vec3f> outputPosition(1);
    AttributeWrapper<int> outputXyz(3);
    AttributeWrapper<int> outputId(1);

    // test offset the whole point block by an arbitrary amount

    Index64 startOffset = 10;

    outputPosition.resize(startOffset + position.size());
    outputXyz.resize((startOffset + id.size())*3);
    outputId.resize(startOffset + id.size());

    std::vector<Index64> offsets;
    pointOffsets(offsets, tree);

    convertPointDataGridPosition(outputPosition, *pointDataGrid, offsets, startOffset);
    convertPointDataGridAttribute(outputId, tree, offsets, startOffset, idIndex);
    convertPointDataGridAttribute(outputXyz, tree, offsets, startOffset, xyzIndex, /*stride=*/3);

    // pack and sort the new buffers based on id

    std::vector<PointData> pointData;

    pointData.resize(count);

    for (unsigned int i = 0; i < count; i++) {
        pointData[i].id = outputId.buffer()[startOffset + i];
        pointData[i].position = outputPosition.buffer()[startOffset + i];
        for (unsigned int j = 0; j < 3; j++) {
            pointData[i].xyz[j] = outputXyz.buffer()[startOffset * 3 + i * 3 + j];
        }
    }

    std::sort(pointData.begin(), pointData.end());

    // compare old and new buffers

    for (unsigned int i = 0; i < count; i++)
    {
        EXPECT_EQ(id.buffer()[i], pointData[i].id);
        EXPECT_NEAR(position.buffer()[i].x(), pointData[i].position.x(), /*tolerance=*/1e-6);
        EXPECT_NEAR(position.buffer()[i].y(), pointData[i].position.y(), /*tolerance=*/1e-6);
        EXPECT_NEAR(position.buffer()[i].z(), pointData[i].position.z(), /*tolerance=*/1e-6);
        EXPECT_EQ(Vec3i(xyz.buffer()[i*3], xyz.buffer()[i*3+1], xyz.buffer()[i*3+2]), pointData[i].xyz);
    }
}


////////////////////////////////////////


TEST_F(TestPointConversion, testComputeVoxelSize)
{
    struct Local {

        static PointDataGrid::Ptr genPointsGrid(const float voxelSize, const AttributeWrapper<Vec3f>& positions)
        {
            math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));
            tools::PointIndexGrid::Ptr pointIndexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(positions, *transform);
            return createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, positions, *transform);
        }
    };

    // minimum and maximum voxel sizes

    const auto minimumVoxelSize = static_cast<float>(math::Pow(double(3e-15), 1.0/3.0));
    const auto maximumVoxelSize =
        static_cast<float>(math::Pow(double(std::numeric_limits<float>::max()), 1.0/3.0));

    AttributeWrapper<Vec3f> position(/*stride*/1);
    AttributeWrapper<Vec3d> positionD(/*stride*/1);

    // test with no positions

    {
        const float voxelSize = computeVoxelSize(position, /*points per voxel*/8);
        EXPECT_EQ(voxelSize, 0.1f);
    }

    // test with one point

    {
        position.resize(1);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        positionHandle.set(0, 0, Vec3f(0.0f));

        const float voxelSize = computeVoxelSize(position, /*points per voxel*/8);
        EXPECT_EQ(voxelSize, 0.1f);
    }

    // test with n points, where n > 1 && n <= num points per voxel

    {
        position.resize(7);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        positionHandle.set(0, 0, Vec3f(-8.6f, 0.0f,-23.8f));
        positionHandle.set(1, 0, Vec3f( 8.6f, 7.8f, 23.8f));

        for (size_t i = 2; i < 7; ++ i)
            positionHandle.set(i, 0, Vec3f(0.0f));

        float voxelSize = computeVoxelSize(position, /*points per voxel*/8);
        EXPECT_NEAR(18.5528f, voxelSize, /*tolerance=*/1e-4);

        voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_NEAR(5.51306f, voxelSize, /*tolerance=*/1e-4);

        // test decimal place accuracy

        voxelSize = computeVoxelSize(position, /*points per voxel*/1, math::Mat4d::identity(), 10);
        EXPECT_NEAR(5.5130610466f, voxelSize, /*tolerance=*/1e-9);

        voxelSize = computeVoxelSize(position, /*points per voxel*/1, math::Mat4d::identity(), 1);
        EXPECT_EQ(5.5f, voxelSize);

        voxelSize = computeVoxelSize(position, /*points per voxel*/1, math::Mat4d::identity(), 0);
        EXPECT_EQ(6.0f, voxelSize);
    }

    // test coplanar points (Y=0)

    {
        position.resize(5);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        positionHandle.set(0, 0, Vec3f(0.0f, 0.0f, 10.0f));
        positionHandle.set(1, 0, Vec3f(0.0f, 0.0f, -10.0f));
        positionHandle.set(2, 0, Vec3f(20.0f, 0.0f, -10.0f));
        positionHandle.set(3, 0, Vec3f(20.0f, 0.0f, 10.0f));
        positionHandle.set(4, 0, Vec3f(10.0f, 0.0f, 0.0f));

        float voxelSize = computeVoxelSize(position, /*points per voxel*/5);
        EXPECT_NEAR(20.0f, voxelSize, /*tolerance=*/1e-4);

        voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_NEAR(11.696f, voxelSize, /*tolerance=*/1e-4);
    }

    // test collinear points (X=0, Y=0)

    {
        position.resize(5);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        positionHandle.set(0, 0, Vec3f(0.0f, 0.0f, 10.0f));
        positionHandle.set(1, 0, Vec3f(0.0f, 0.0f, -10.0f));
        positionHandle.set(2, 0, Vec3f(0.0f, 0.0f, -10.0f));
        positionHandle.set(3, 0, Vec3f(0.0f, 0.0f, 10.0f));
        positionHandle.set(4, 0, Vec3f(0.0f, 0.0f, 0.0f));

        float voxelSize = computeVoxelSize(position, /*points per voxel*/5);
        EXPECT_NEAR(20.0f, voxelSize, /*tolerance=*/1e-4);

        voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_NEAR(8.32034f, voxelSize, /*tolerance=*/1e-4);
    }

    // test min limit collinear points (X=0, Y=0, Z=+/-float min)

    {
        position.resize(2);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        positionHandle.set(0, 0, Vec3f(0.0f, 0.0f, -std::numeric_limits<float>::min()));
        positionHandle.set(1, 0, Vec3f(0.0f, 0.0f, std::numeric_limits<float>::min()));

        float voxelSize = computeVoxelSize(position, /*points per voxel*/2);
        EXPECT_NEAR(minimumVoxelSize, voxelSize, /*tolerance=*/1e-4);

        voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_NEAR(minimumVoxelSize, voxelSize, /*tolerance=*/1e-4);
    }

    // test max limit collinear points (X=+/-float max, Y=0, Z=0)

    {
        position.resize(2);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        positionHandle.set(0, 0, Vec3f(-std::numeric_limits<float>::max(), 0.0f, 0.0f));
        positionHandle.set(1, 0, Vec3f(std::numeric_limits<float>::max(), 0.0f, 0.0f));

        float voxelSize = computeVoxelSize(position, /*points per voxel*/2);
        EXPECT_NEAR(maximumVoxelSize, voxelSize, /*tolerance=*/1e-4);

        voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_NEAR(maximumVoxelSize, voxelSize, /*tolerance=*/1e-4);
    }

    // max pointsPerVoxel

    {
        position.resize(2);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        positionHandle.set(0, 0, Vec3f(0));
        positionHandle.set(1, 0, Vec3f(1));

        float voxelSize = computeVoxelSize(position, /*points per voxel*/std::numeric_limits<uint32_t>::max());
        EXPECT_EQ(voxelSize, 1.0f);
    }

    // limits test

    {
        positionD.resize(2);
        AttributeWrapper<Vec3d>::Handle positionHandleD(positionD);
        positionHandleD.set(0, 0, Vec3d(0));
        positionHandleD.set(1, 0, Vec3d(std::numeric_limits<double>::max()));

        float voxelSize = computeVoxelSize(positionD, /*points per voxel*/2);
        EXPECT_EQ(voxelSize, maximumVoxelSize);
    }

    {
        const float smallest(std::numeric_limits<float>::min());

        position.resize(4);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        positionHandle.set(0, 0, Vec3f(0.0f));
        positionHandle.set(1, 0, Vec3f(smallest));
        positionHandle.set(2, 0, Vec3f(smallest, 0.0f, 0.0f));
        positionHandle.set(3, 0, Vec3f(smallest, 0.0f, smallest));

        float voxelSize = computeVoxelSize(position, /*points per voxel*/4);
        EXPECT_EQ(voxelSize, minimumVoxelSize);

        voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_NEAR(minimumVoxelSize, voxelSize, /*tolerance=*/1e-4);

        PointDataGrid::Ptr grid = Local::genPointsGrid(voxelSize, position);
        EXPECT_EQ(grid->activeVoxelCount(), Index64(1));
    }

    // the smallest possible vector extent that can exist from an input set
    // without being clamped to the minimum voxel size
    // is Tolerance<Real>::value() + std::numeric_limits<Real>::min()

    {
        position.resize(2);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        positionHandle.set(0, 0, Vec3f(0.0f));
        positionHandle.set(1, 0, Vec3f(math::Tolerance<Real>::value() + std::numeric_limits<Real>::min()));

        float voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_EQ(voxelSize, minimumVoxelSize);
    }

    // in-between smallest extent and ScaleMap determinant test

    {
        position.resize(2);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        positionHandle.set(0, 0, Vec3f(0.0f));
        positionHandle.set(1, 0, Vec3f(math::Tolerance<Real>::value()*1e8 + std::numeric_limits<Real>::min()));

        float voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_EQ(voxelSize, float(math::Pow(double(3e-15), 1.0/3.0)));
    }

    {
        const float smallValue(1e-5f);

        position.resize(300000);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);

        for (size_t i = 0; i < 100000; ++ i) {
            positionHandle.set(i, 0, Vec3f(smallValue*float(i), 0, 0));
            positionHandle.set(i+100000, 0, Vec3f(0, smallValue*float(i), 0));
            positionHandle.set(i+200000, 0, Vec3f(0, 0, smallValue*float(i)));
        }

        float voxelSize = computeVoxelSize(position, /*points per voxel*/10);
        EXPECT_NEAR(0.00012f, voxelSize, /*tolerance=*/1e-4);

        voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_NEAR(2e-5, voxelSize, /*tolerance=*/1e-6);

        PointDataGrid::Ptr grid = Local::genPointsGrid(voxelSize, position);
        EXPECT_EQ(grid->activeVoxelCount(), Index64(150001));

        // check zero decimal place still returns valid result

        voxelSize = computeVoxelSize(position, /*points per voxel*/1, math::Mat4d::identity(), 0);
        EXPECT_NEAR(2e-5, voxelSize, /*tolerance=*/1e-6);
    }

    // random position generation within two bounds of equal size.
    // This test distributes 1000 points within a 1x1x1 box centered at (0,0,0)
    // and another 1000 points within a separate 1x1x1 box centered at (20,20,20).
    // Points are randomly positioned however can be defined as having a stochastic
    // distribution. Tests that sparsity between these data sets causes no issues
    // and that computeVoxelSize produces accurate results

    {
        position.resize(2000);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        openvdb::math::Random01 randNumber(0);

        // positions between -0.5 and 0.5

        for (size_t i = 0; i < 1000; ++ i) {
            const Vec3f pos(randNumber() - 0.5f);
            positionHandle.set(i, 0, pos);
        }

        // positions between 19.5 and 20.5

        for (size_t i = 1000; i < 2000; ++ i) {
            const Vec3f pos(randNumber() - 0.5f + 20.0f);
            positionHandle.set(i, 0, pos);
        }

        float voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_NEAR(0.00052f, voxelSize, /*tolerance=*/1e-4);

        PointDataGrid::Ptr grid = Local::genPointsGrid(voxelSize, position);
        const auto pointsPerVoxel = static_cast<Index64>(
            math::Round(2000.0f / static_cast<float>(grid->activeVoxelCount())));
        EXPECT_EQ(pointsPerVoxel, Index64(1));
    }

    // random position generation within three bounds of varying size.
    // This test distributes 1000 points within a 1x1x1 box centered at (0.5,0.5,0,5)
    // another 1000 points within a separate 10x10x10 box centered at (15,15,15) and
    // a final 1000 points within a separate 50x50x50 box centered at (75,75,75)
    // Points are randomly positioned however can be defined as having a stochastic
    // distribution. Tests that sparsity between these data sets causes no issues as
    // well as computeVoxelSize producing a good average result

    {
        position.resize(3000);
        AttributeWrapper<Vec3f>::Handle positionHandle(position);
        openvdb::math::Random01 randNumber(0);

        // positions between 0 and 1

        for (size_t i = 0; i < 1000; ++ i) {
            const Vec3f pos(randNumber());
            positionHandle.set(i, 0, pos);
        }

        // positions between 10 and 20

        for (size_t i = 1000; i < 2000; ++ i) {
            const Vec3f pos((randNumber() * 10.0f) + 10.0f);
            positionHandle.set(i, 0, pos);
        }

        // positions between 50 and 100

        for (size_t i = 2000; i < 3000; ++ i) {
            const Vec3f pos((randNumber() * 50.0f) + 50.0f);
            positionHandle.set(i, 0, pos);
        }

        float voxelSize = computeVoxelSize(position, /*points per voxel*/10);
        EXPECT_NEAR(0.24758f, voxelSize, /*tolerance=*/1e-3);

        PointDataGrid::Ptr grid = Local::genPointsGrid(voxelSize, position);
        auto pointsPerVoxel = static_cast<Index64>(
            math::Round(3000.0f/ static_cast<float>(grid->activeVoxelCount())));
        EXPECT_TRUE(math::isApproxEqual(pointsPerVoxel, Index64(10), Index64(2)));

        voxelSize = computeVoxelSize(position, /*points per voxel*/1);
        EXPECT_NEAR(0.00231f, voxelSize, /*tolerance=*/1e-4);

        grid = Local::genPointsGrid(voxelSize, position);
        pointsPerVoxel = static_cast<Index64>(
            math::Round(3000.0f/ static_cast<float>(grid->activeVoxelCount())));
        EXPECT_EQ(pointsPerVoxel, Index64(1));
    }

    // Generate a sphere
    // NOTE: The sphere does NOT provide uniform distribution

    const size_t count(40000);

    position.resize(0);

    AttributeWrapper<int> xyz(1);
    AttributeWrapper<int> id(1);
    AttributeWrapper<float> uniform(1);
    AttributeWrapper<openvdb::Name> string(1);
    GroupWrapper group;

    genPoints(count, /*scale=*/ 100.0, /*stride=*/false, position, xyz, id, uniform, string, group);

    EXPECT_EQ(position.size(), count);
    EXPECT_EQ(id.size(), count);
    EXPECT_EQ(uniform.size(), count);
    EXPECT_EQ(string.size(), count);
    EXPECT_EQ(group.size(), count);

    // test a distributed point set around a sphere

    {
        const float voxelSize = computeVoxelSize(position, /*points per voxel*/2);

        EXPECT_NEAR(2.6275f, voxelSize, /*tolerance=*/0.01);

        PointDataGrid::Ptr grid = Local::genPointsGrid(voxelSize, position);
        const Index64 pointsPerVoxel = count / grid->activeVoxelCount();
        EXPECT_EQ(pointsPerVoxel, Index64(2));
    }

    // test with given target transforms

    {
        // test that a different scale doesn't change the result

        openvdb::math::Transform::Ptr transform1(openvdb::math::Transform::createLinearTransform(0.33));
        openvdb::math::Transform::Ptr transform2(openvdb::math::Transform::createLinearTransform(0.87));

        math::UniformScaleMap::ConstPtr scaleMap1 = transform1->constMap<math::UniformScaleMap>();
        math::UniformScaleMap::ConstPtr scaleMap2 = transform2->constMap<math::UniformScaleMap>();
        EXPECT_TRUE(scaleMap1.get());
        EXPECT_TRUE(scaleMap2.get());

        math::AffineMap::ConstPtr affineMap1 = scaleMap1->getAffineMap();
        math::AffineMap::ConstPtr affineMap2 = scaleMap2->getAffineMap();

        float voxelSize1 = computeVoxelSize(position, /*points per voxel*/2, affineMap1->getMat4());
        float voxelSize2 = computeVoxelSize(position, /*points per voxel*/2, affineMap2->getMat4());
        EXPECT_EQ(voxelSize1, voxelSize2);

        // test that applying a rotation roughly calculates to the same result for this example
        // NOTE: distribution is not uniform

        // Rotate by 45 degrees in X, Y, Z

        transform1->postRotate(openvdb::math::pi<double>() / 4.0, math::X_AXIS);
        transform1->postRotate(openvdb::math::pi<double>() / 4.0, math::Y_AXIS);
        transform1->postRotate(openvdb::math::pi<double>() / 4.0, math::Z_AXIS);

        affineMap1 = transform1->constMap<math::AffineMap>();
        EXPECT_TRUE(affineMap1.get());

        float voxelSize3 = computeVoxelSize(position, /*points per voxel*/2, affineMap1->getMat4());
        EXPECT_NEAR(voxelSize1, voxelSize3, 0.1);

        // test that applying a translation roughly calculates to the same result for this example

        transform1->postTranslate(Vec3d(-5.0f, 3.3f, 20.1f));
        affineMap1 = transform1->constMap<math::AffineMap>();
        EXPECT_TRUE(affineMap1.get());

        float voxelSize4 = computeVoxelSize(position, /*points per voxel*/2, affineMap1->getMat4());
        EXPECT_NEAR(voxelSize1, voxelSize4, 0.1);
    }
}

TEST_F(TestPointConversion, testPrecision)
{
    const double tolerance = math::Tolerance<float>::value();

    { // test values far from origin
        const double voxelSize = 0.5;
        const float halfVoxelSize = 0.25f;
        auto transform = math::Transform::createLinearTransform(voxelSize);

        float onBorder = 1000.0f + halfVoxelSize; // can be represented exactly in floating-point
        float beforeBorder = std::nextafterf(onBorder, /*to=*/0.0f);
        float afterBorder = std::nextafterf(onBorder, /*to=*/2000.0f);

        const Vec3f positionBefore(beforeBorder, afterBorder, onBorder);

        std::vector<Vec3f> points{positionBefore};
        PointAttributeVector<Vec3f> wrapper(points);
        auto pointIndexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(
            wrapper, *transform);

        Vec3f positionAfterNull;
        Vec3f positionAfterFixed16;

        { // null codec
            auto points = createPointDataGrid<NullCodec, PointDataGrid>(
                *pointIndexGrid, wrapper, *transform);

            auto leafIter = points->tree().cbeginLeaf();
            auto indexIter = leafIter->beginIndexOn();
            auto handle = AttributeHandle<Vec3f>(leafIter->constAttributeArray("P"));

            const auto& ijk = indexIter.getCoord();

            EXPECT_EQ(ijk.x(), 2000);
            EXPECT_EQ(ijk.y(), 2001);
            EXPECT_EQ(ijk.z(), 2001); // on border value is stored in the higher voxel

            const Vec3f positionVoxelSpace = handle.get(*indexIter);

            // voxel-space range: -0.5f >= value > 0.5f

            EXPECT_TRUE(positionVoxelSpace.x() > 0.49f && positionVoxelSpace.x() < 0.5f);
            EXPECT_TRUE(positionVoxelSpace.y() > -0.5f && positionVoxelSpace.y() < -0.49f);
            EXPECT_TRUE(positionVoxelSpace.z() == -0.5f); // on border value is stored at -0.5f

            positionAfterNull = Vec3f(transform->indexToWorld(positionVoxelSpace + ijk.asVec3d()));

            EXPECT_NEAR(positionAfterNull.x(), positionBefore.x(), tolerance);
            EXPECT_NEAR(positionAfterNull.y(), positionBefore.y(), tolerance);
            EXPECT_NEAR(positionAfterNull.z(), positionBefore.z(), tolerance);
        }

        { // fixed 16-bit codec
            auto points = createPointDataGrid<FixedPointCodec<false>, PointDataGrid>(
                *pointIndexGrid, wrapper, *transform);

            auto leafIter = points->tree().cbeginLeaf();
            auto indexIter = leafIter->beginIndexOn();
            auto handle = AttributeHandle<Vec3f>(leafIter->constAttributeArray("P"));

            const auto& ijk = indexIter.getCoord();

            EXPECT_EQ(ijk.x(), 2000);
            EXPECT_EQ(ijk.y(), 2001);
            EXPECT_EQ(ijk.z(), 2001); // on border value is stored in the higher voxel

            const Vec3f positionVoxelSpace = handle.get(*indexIter);

            // voxel-space range: -0.5f >= value > 0.5f

            EXPECT_TRUE(positionVoxelSpace.x() > 0.49f && positionVoxelSpace.x() < 0.5f);
            EXPECT_TRUE(positionVoxelSpace.y() > -0.5f && positionVoxelSpace.y() < -0.49f);
            EXPECT_TRUE(positionVoxelSpace.z() == -0.5f); // on border value is stored at -0.5f

            positionAfterFixed16 = Vec3f(transform->indexToWorld(
                positionVoxelSpace + ijk.asVec3d()));

            EXPECT_NEAR(positionAfterFixed16.x(), positionBefore.x(), tolerance);
            EXPECT_NEAR(positionAfterFixed16.y(), positionBefore.y(), tolerance);
            EXPECT_NEAR(positionAfterFixed16.z(), positionBefore.z(), tolerance);
        }

        // at this precision null codec == fixed-point 16-bit codec

        EXPECT_EQ(positionAfterNull.x(), positionAfterFixed16.x());
        EXPECT_EQ(positionAfterNull.y(), positionAfterFixed16.y());
        EXPECT_EQ(positionAfterNull.z(), positionAfterFixed16.z());
    }

    { // test values near to origin
        const double voxelSize = 0.5;
        const float halfVoxelSize = 0.25f;
        auto transform = math::Transform::createLinearTransform(voxelSize);

        float onBorder = 0.0f+halfVoxelSize;
        float beforeBorder = std::nextafterf(onBorder, /*to=*/0.0f);
        float afterBorder = std::nextafterf(onBorder, /*to=*/2000.0f);

        const Vec3f positionBefore(beforeBorder, afterBorder, onBorder);

        std::vector<Vec3f> points{positionBefore};
        PointAttributeVector<Vec3f> wrapper(points);
        auto pointIndexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(
            wrapper, *transform);

        Vec3f positionAfterNull;
        Vec3f positionAfterFixed16;

        { // null codec
            auto points = createPointDataGrid<NullCodec, PointDataGrid>(
                *pointIndexGrid, wrapper, *transform);

            auto leafIter = points->tree().cbeginLeaf();
            auto indexIter = leafIter->beginIndexOn();
            auto handle = AttributeHandle<Vec3f>(leafIter->constAttributeArray("P"));

            const auto& ijk = indexIter.getCoord();

            EXPECT_EQ(ijk.x(), 0);
            EXPECT_EQ(ijk.y(), 1);
            EXPECT_EQ(ijk.z(), 1); // on border value is stored in the higher voxel

            const Vec3f positionVoxelSpace = handle.get(*indexIter);

            // voxel-space range: -0.5f >= value > 0.5f

            EXPECT_TRUE(positionVoxelSpace.x() > 0.49f && positionVoxelSpace.x() < 0.5f);
            EXPECT_TRUE(positionVoxelSpace.y() > -0.5f && positionVoxelSpace.y() < -0.49f);
            EXPECT_TRUE(positionVoxelSpace.z() == -0.5f); // on border value is stored at -0.5f

            positionAfterNull = Vec3f(transform->indexToWorld(positionVoxelSpace + ijk.asVec3d()));

            EXPECT_NEAR(positionAfterNull.x(), positionBefore.x(), tolerance);
            EXPECT_NEAR(positionAfterNull.y(), positionBefore.y(), tolerance);
            EXPECT_NEAR(positionAfterNull.z(), positionBefore.z(), tolerance);
        }

        { // fixed 16-bit codec - at this precision, this codec results in lossy compression
            auto points = createPointDataGrid<FixedPointCodec<false>, PointDataGrid>(
                *pointIndexGrid, wrapper, *transform);

            auto leafIter = points->tree().cbeginLeaf();
            auto indexIter = leafIter->beginIndexOn();
            auto handle = AttributeHandle<Vec3f>(leafIter->constAttributeArray("P"));

            const auto& ijk = indexIter.getCoord();

            EXPECT_EQ(ijk.x(), 0);
            EXPECT_EQ(ijk.y(), 1);
            EXPECT_EQ(ijk.z(), 1); // on border value is stored in the higher voxel

            const Vec3f positionVoxelSpace = handle.get(*indexIter);

            // voxel-space range: -0.5f >= value > 0.5f

            EXPECT_TRUE(positionVoxelSpace.x() == 0.5f); // before border is clamped to 0.5f
            EXPECT_TRUE(positionVoxelSpace.y() == -0.5f); // after border is clamped to -0.5f
            EXPECT_TRUE(positionVoxelSpace.z() == -0.5f); // on border is stored at -0.5f

            positionAfterFixed16 = Vec3f(transform->indexToWorld(
                positionVoxelSpace + ijk.asVec3d()));

            // reduce tolerance to handle lack of precision

            EXPECT_NEAR(positionAfterFixed16.x(), positionBefore.x(), 1e-6);
            EXPECT_NEAR(positionAfterFixed16.y(), positionBefore.y(), 1e-6);
            EXPECT_NEAR(positionAfterFixed16.z(), positionBefore.z(), tolerance);
        }

        // only z matches precisely due to lossy compression

        EXPECT_TRUE(positionAfterNull.x() != positionAfterFixed16.x());
        EXPECT_TRUE(positionAfterNull.y() != positionAfterFixed16.y());
        EXPECT_EQ(positionAfterNull.z(), positionAfterFixed16.z());
    }
}

TEST_F(TestPointConversion, testExample)
{
    // this is the example from the documentation using both Vec3R and Vec3f

    { // Vec3R
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
        int pointsPerVoxel = 8;
        float voxelSize =
            openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel);

        // Create a transform using this voxel-size.
        openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

        // Create a PointDataGrid containing these four points and using the
        // transform given. This function has two template parameters, (1) the codec
        // to use for storing the position, (2) the grid we want to create
        // (ie a PointDataGrid).
        // We use no compression here for the positions.
        openvdb::points::PointDataGrid::Ptr grid =
            openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
                            openvdb::points::PointDataGrid>(positions, *transform);

        // Set the name of the grid
        grid->setName("Points");

        // Create a VDB file object and write out the grid.
        openvdb::io::File("mypoints.vdb").write({grid});

        // Create a new VDB file object for reading.
        openvdb::io::File newFile("mypoints.vdb");

        // Open the file. This reads the file header, but not any grids.
        newFile.open();

        // Read the grid by name.
        openvdb::GridBase::Ptr baseGrid = newFile.readGrid("Points");
        newFile.close();

        // From the example above, "Points" is known to be a PointDataGrid,
        // so cast the generic grid pointer to a PointDataGrid pointer.
        grid = openvdb::gridPtrCast<openvdb::points::PointDataGrid>(baseGrid);

        std::vector<Vec3R> resultingPositions;

        // Iterate over all the leaf nodes in the grid.
        for (auto leafIter = grid->tree().cbeginLeaf(); leafIter; ++leafIter) {

            // Extract the position attribute from the leaf by name (P is position).
            const openvdb::points::AttributeArray& array =
                leafIter->constAttributeArray("P");

            // Create a read-only AttributeHandle. Position always uses Vec3f.
            openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(array);

            // Iterate over the point indices in the leaf.
            for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {

                // Extract the voxel-space position of the point.
                openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);

                // Extract the index-space position of the voxel.
                const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();

                // Compute the world-space position of the point.
                openvdb::Vec3f worldPosition =
                    grid->transform().indexToWorld(voxelPosition + xyz);

                resultingPositions.push_back(worldPosition);
            }
        }

        EXPECT_EQ(size_t(4), resultingPositions.size());

        // remap the position order

        std::vector<size_t> remap;
        remap.push_back(1);
        remap.push_back(3);
        remap.push_back(0);
        remap.push_back(2);

        for (int i = 0; i < 4; i++) {
            EXPECT_NEAR(positions[i].x(), resultingPositions[remap[i]].x(), /*tolerance=*/1e-6);
            EXPECT_NEAR(positions[i].y(), resultingPositions[remap[i]].y(), /*tolerance=*/1e-6);
            EXPECT_NEAR(positions[i].z(), resultingPositions[remap[i]].z(), /*tolerance=*/1e-6);
        }

        remove("mypoints.vdb");
    }

    { // Vec3f
        // Create a vector with four point positions.
        std::vector<openvdb::Vec3f> positions;
        positions.push_back(openvdb::Vec3f(0.0f, 1.0f, 0.0f));
        positions.push_back(openvdb::Vec3f(1.5f, 3.5f, 1.0f));
        positions.push_back(openvdb::Vec3f(-1.0f, 6.0f, -2.0f));
        positions.push_back(openvdb::Vec3f(1.1f, 1.25f, 0.06f));

        // The VDB Point-Partioner is used when bucketing points and requires a
        // specific interface. For convenience, we use the PointAttributeVector
        // wrapper around an stl vector wrapper here, however it is also possible to
        // write one for a custom data structure in order to match the interface
        // required.
        openvdb::points::PointAttributeVector<openvdb::Vec3f> positionsWrapper(positions);

        // This method computes a voxel-size to match the number of
        // points / voxel requested. Although it won't be exact, it typically offers
        // a good balance of memory against performance.
        int pointsPerVoxel = 8;
        float voxelSize =
            openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel);

        // Create a transform using this voxel-size.
        openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

        // Create a PointDataGrid containing these four points and using the
        // transform given. This function has two template parameters, (1) the codec
        // to use for storing the position, (2) the grid we want to create
        // (ie a PointDataGrid).
        // We use no compression here for the positions.
        openvdb::points::PointDataGrid::Ptr grid =
            openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
                            openvdb::points::PointDataGrid>(positions, *transform);

        // Set the name of the grid
        grid->setName("Points");

        // Create a VDB file object and write out the grid.
        openvdb::io::File("mypoints.vdb").write({grid});

        // Create a new VDB file object for reading.
        openvdb::io::File newFile("mypoints.vdb");

        // Open the file. This reads the file header, but not any grids.
        newFile.open();

        // Read the grid by name.
        openvdb::GridBase::Ptr baseGrid = newFile.readGrid("Points");
        newFile.close();

        // From the example above, "Points" is known to be a PointDataGrid,
        // so cast the generic grid pointer to a PointDataGrid pointer.
        grid = openvdb::gridPtrCast<openvdb::points::PointDataGrid>(baseGrid);

        std::vector<Vec3f> resultingPositions;

        // Iterate over all the leaf nodes in the grid.
        for (auto leafIter = grid->tree().cbeginLeaf(); leafIter; ++leafIter) {

            // Extract the position attribute from the leaf by name (P is position).
            const openvdb::points::AttributeArray& array =
                leafIter->constAttributeArray("P");

            // Create a read-only AttributeHandle. Position always uses Vec3f.
            openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(array);

            // Iterate over the point indices in the leaf.
            for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {

                // Extract the voxel-space position of the point.
                openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);

                // Extract the index-space position of the voxel.
                const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();

                // Compute the world-space position of the point.
                openvdb::Vec3f worldPosition =
                    grid->transform().indexToWorld(voxelPosition + xyz);

                resultingPositions.push_back(worldPosition);
            }
        }

        EXPECT_EQ(size_t(4), resultingPositions.size());

        // remap the position order

        std::vector<size_t> remap;
        remap.push_back(1);
        remap.push_back(3);
        remap.push_back(0);
        remap.push_back(2);

        for (int i = 0; i < 4; i++) {
            EXPECT_NEAR(positions[i].x(), resultingPositions[remap[i]].x(), /*tolerance=*/1e-6f);
            EXPECT_NEAR(positions[i].y(), resultingPositions[remap[i]].y(), /*tolerance=*/1e-6f);
            EXPECT_NEAR(positions[i].z(), resultingPositions[remap[i]].z(), /*tolerance=*/1e-6f);
        }

        remove("mypoints.vdb");
    }
}
