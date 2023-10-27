// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/math/Math.h>// for math::Random01(), math::Pow3()
#include <gtest/gtest.h>

class TestLeaf: public ::testing::Test
{
public:
    void testBuffer();
    void testGetValue();
};

typedef openvdb::tree::LeafNode<int, 3> LeafType;
typedef LeafType::Buffer                BufferType;
using openvdb::Index;

void
TestLeaf::testBuffer()
{
    {// access
        BufferType buf;

        for (Index i = 0; i < BufferType::size(); ++i) {
            buf.mData[i] = i;
            EXPECT_TRUE(buf[i] == buf.mData[i]);
        }
        for (Index i = 0; i < BufferType::size(); ++i) {
            buf[i] = i;
            EXPECT_EQ(int(i), buf[i]);
        }
    }

    {// swap
        BufferType buf0, buf1, buf2;

        int *buf0Data = buf0.mData;
        int *buf1Data = buf1.mData;

        for (Index i = 0; i < BufferType::size(); ++i) {
            buf0[i] = i;
            buf1[i] = i * 2;
        }

        buf0.swap(buf1);

        EXPECT_TRUE(buf0.mData == buf1Data);
        EXPECT_TRUE(buf1.mData == buf0Data);

        buf1.swap(buf0);

        EXPECT_TRUE(buf0.mData == buf0Data);
        EXPECT_TRUE(buf1.mData == buf1Data);

        buf0.swap(buf2);

        EXPECT_TRUE(buf2.mData == buf0Data);

        buf2.swap(buf0);

        EXPECT_TRUE(buf0.mData == buf0Data);
    }

}
TEST_F(TestLeaf, testBuffer) { testBuffer(); }

void
TestLeaf::testGetValue()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));

    leaf.mBuffer[0] = 2;
    leaf.mBuffer[1] = 3;
    leaf.mBuffer[2] = 4;
    leaf.mBuffer[65] = 10;

    EXPECT_EQ(2, leaf.getValue(openvdb::Coord(0, 0, 0)));
    EXPECT_EQ(3, leaf.getValue(openvdb::Coord(0, 0, 1)));
    EXPECT_EQ(4, leaf.getValue(openvdb::Coord(0, 0, 2)));

    EXPECT_EQ(10, leaf.getValue(openvdb::Coord(1, 0, 1)));
}
TEST_F(TestLeaf, testGetValue) { testGetValue(); }

TEST_F(TestLeaf, testSetValue)
{
    LeafType leaf(openvdb::Coord(0, 0, 0), 3);

    openvdb::Coord xyz(0, 0, 0);
    leaf.setValueOn(xyz, 10);
    EXPECT_EQ(10, leaf.getValue(xyz));

    xyz.reset(7, 7, 7);
    leaf.setValueOn(xyz, 7);
    EXPECT_EQ(7, leaf.getValue(xyz));
    leaf.setValueOnly(xyz, 10);
    EXPECT_EQ(10, leaf.getValue(xyz));

    xyz.reset(2, 3, 6);
    leaf.setValueOn(xyz, 236);
    EXPECT_EQ(236, leaf.getValue(xyz));

    leaf.setValueOff(xyz, 1);
    EXPECT_EQ(1, leaf.getValue(xyz));
    EXPECT_TRUE(!leaf.isValueOn(xyz));
}

TEST_F(TestLeaf, testIsValueSet)
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(1, 5, 7), 10);

    EXPECT_TRUE(leaf.isValueOn(openvdb::Coord(1, 5, 7)));

    EXPECT_TRUE(!leaf.isValueOn(openvdb::Coord(0, 5, 7)));
    EXPECT_TRUE(!leaf.isValueOn(openvdb::Coord(1, 6, 7)));
    EXPECT_TRUE(!leaf.isValueOn(openvdb::Coord(0, 5, 6)));
}

TEST_F(TestLeaf, testProbeValue)
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(1, 6, 5), 10);

    LeafType::ValueType val;
    EXPECT_TRUE(leaf.probeValue(openvdb::Coord(1, 6, 5), val));
    EXPECT_TRUE(!leaf.probeValue(openvdb::Coord(1, 6, 4), val));
}

TEST_F(TestLeaf, testIterators)
{
    LeafType leaf(openvdb::Coord(0, 0, 0), 2);
    leaf.setValueOn(openvdb::Coord(1, 2, 3), -3);
    leaf.setValueOn(openvdb::Coord(5, 2, 3),  4);
    LeafType::ValueType sum = 0;
    for (LeafType::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) sum += *iter;
    EXPECT_EQ((-3 + 4), sum);
}

TEST_F(TestLeaf, testEquivalence)
{
    LeafType leaf( openvdb::Coord(0, 0, 0), 2);
    LeafType leaf2(openvdb::Coord(0, 0, 0), 3);

    EXPECT_TRUE(leaf != leaf2);

    for(openvdb::Index32 i = 0; i < LeafType::size(); ++i) {
        leaf.setValueOnly(i, i);
        leaf2.setValueOnly(i, i);
    }
    EXPECT_TRUE(leaf == leaf2);

    // set some values.
    leaf.setValueOn(openvdb::Coord(0, 0, 0), 1);
    leaf.setValueOn(openvdb::Coord(0, 1, 0), 1);
    leaf.setValueOn(openvdb::Coord(1, 1, 0), 1);
    leaf.setValueOn(openvdb::Coord(1, 1, 2), 1);

    leaf2.setValueOn(openvdb::Coord(0, 0, 0), 1);
    leaf2.setValueOn(openvdb::Coord(0, 1, 0), 1);
    leaf2.setValueOn(openvdb::Coord(1, 1, 0), 1);
    leaf2.setValueOn(openvdb::Coord(1, 1, 2), 1);

    EXPECT_TRUE(leaf == leaf2);

    leaf2.setValueOn(openvdb::Coord(0, 0, 1), 1);

    EXPECT_TRUE(leaf != leaf2);

    leaf2.setValueOff(openvdb::Coord(0, 0, 1), 1);

    EXPECT_TRUE(leaf == leaf2);
}

TEST_F(TestLeaf, testGetOrigin)
{
    {
        LeafType leaf(openvdb::Coord(1, 0, 0), 1);
        EXPECT_EQ(openvdb::Coord(0, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(0, 0, 0), 1);
        EXPECT_EQ(openvdb::Coord(0, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(8, 0, 0), 1);
        EXPECT_EQ(openvdb::Coord(8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(8, 1, 0), 1);
        EXPECT_EQ(openvdb::Coord(8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(1024, 1, 3), 1);
        EXPECT_EQ(openvdb::Coord(128*8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(1023, 1, 3), 1);
        EXPECT_EQ(openvdb::Coord(127*8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(512, 512, 512), 1);
        EXPECT_EQ(openvdb::Coord(512, 512, 512), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(2, 52, 515), 1);
        EXPECT_EQ(openvdb::Coord(0, 48, 512), leaf.origin());
    }
}

TEST_F(TestLeaf, testIteratorGetCoord)
{
    using namespace openvdb;

    LeafType leaf(openvdb::Coord(8, 8, 0), 2);

    EXPECT_EQ(Coord(8, 8, 0), leaf.origin());

    leaf.setValueOn(Coord(1, 2, 3), -3);
    leaf.setValueOn(Coord(5, 2, 3),  4);

    LeafType::ValueOnIter iter = leaf.beginValueOn();
    Coord xyz = iter.getCoord();
    EXPECT_EQ(Coord(9, 10, 3), xyz);

    ++iter;
    xyz = iter.getCoord();
    EXPECT_EQ(Coord(13, 10, 3), xyz);
}

TEST_F(TestLeaf, testNegativeIndexing)
{
    using namespace openvdb;

    LeafType leaf(openvdb::Coord(-9, -2, -8), 1);

    EXPECT_EQ(Coord(-16, -8, -8), leaf.origin());

    leaf.setValueOn(Coord(1, 2, 3), -3);
    leaf.setValueOn(Coord(5, 2, 3),  4);

    EXPECT_EQ(-3, leaf.getValue(Coord(1, 2, 3)));
    EXPECT_EQ(4, leaf.getValue(Coord(5, 2, 3)));

    LeafType::ValueOnIter iter = leaf.beginValueOn();
    Coord xyz = iter.getCoord();
    EXPECT_EQ(Coord(-15, -6, -5), xyz);

    ++iter;
    xyz = iter.getCoord();
    EXPECT_EQ(Coord(-11, -6, -5), xyz);
}

TEST_F(TestLeaf, testIsConstant)
{
    using namespace openvdb;
    const Coord origin(-9, -2, -8);

    {// check old version (v3.0 and older) with float
        // Acceptable range: first-value +/- tolerance
        const float val = 1.0f, tol = 0.01f;
        tree::LeafNode<float, 3> leaf(origin, val, true);
        float v = 0.0f;
        bool stat = false;
        EXPECT_TRUE(leaf.isConstant(v, stat, tol));
        EXPECT_TRUE(stat);
        EXPECT_EQ(val, v);

        leaf.setValueOff(0);
        EXPECT_TRUE(!leaf.isConstant(v, stat, tol));

        leaf.setValueOn(0);
        EXPECT_TRUE(leaf.isConstant(v, stat, tol));

        leaf.setValueOn(0, val + 0.99f*tol);
        EXPECT_TRUE(leaf.isConstant(v, stat, tol));
        EXPECT_TRUE(stat);
        EXPECT_EQ(val + 0.99f*tol, v);

        leaf.setValueOn(0, val + 1.01f*tol);
        EXPECT_TRUE(!leaf.isConstant(v, stat, tol));
    }
    {// check old version (v3.0 and older) with double
        // Acceptable range: first-value +/- tolerance
        const double val = 1.0, tol = 0.00001;
        tree::LeafNode<double, 3> leaf(origin, val, true);
        double v = 0.0;
        bool stat = false;
        EXPECT_TRUE(leaf.isConstant(v, stat, tol));
        EXPECT_TRUE(stat);
        EXPECT_EQ(val, v);

        leaf.setValueOff(0);
        EXPECT_TRUE(!leaf.isConstant(v, stat, tol));

        leaf.setValueOn(0);
        EXPECT_TRUE(leaf.isConstant(v, stat, tol));

        leaf.setValueOn(0, val + 0.99*tol);
        EXPECT_TRUE(leaf.isConstant(v, stat, tol));
        EXPECT_TRUE(stat);
        EXPECT_EQ(val + 0.99*tol, v);

        leaf.setValueOn(0, val + 1.01*tol);
        EXPECT_TRUE(!leaf.isConstant(v, stat, tol));
    }
    {// check newer version (v3.2 and newer) with float
        // Acceptable range: max - min <= tolerance
        const float val = 1.0, tol = 0.01f;
        tree::LeafNode<float, 3> leaf(origin, val, true);
        float vmin = 0.0f, vmax = 0.0f;
        bool stat = false;

        EXPECT_TRUE(leaf.isConstant(vmin, vmax, stat, tol));
        EXPECT_TRUE(stat);
        EXPECT_EQ(val, vmin);
        EXPECT_EQ(val, vmax);

        leaf.setValueOff(0);
        EXPECT_TRUE(!leaf.isConstant(vmin, vmax, stat, tol));

        leaf.setValueOn(0);
        EXPECT_TRUE(leaf.isConstant(vmin, vmax, stat, tol));

        leaf.setValueOn(0, val + tol);
        EXPECT_TRUE(leaf.isConstant(vmin, vmax, stat, tol));
        EXPECT_EQ(val, vmin);
        EXPECT_EQ(val + tol, vmax);

        leaf.setValueOn(0, val + 1.01f*tol);
        EXPECT_TRUE(!leaf.isConstant(vmin, vmax, stat, tol));
    }
    {// check newer version (v3.2 and newer) with double
        // Acceptable range: (max- min) <= tolerance
        const double val = 1.0, tol = 0.000001;
        tree::LeafNode<double, 3> leaf(origin, val, true);
        double vmin = 0.0, vmax = 0.0;
        bool stat = false;
        EXPECT_TRUE(leaf.isConstant(vmin, vmax, stat, tol));
        EXPECT_TRUE(stat);
        EXPECT_EQ(val, vmin);
        EXPECT_EQ(val, vmax);

        leaf.setValueOff(0);
        EXPECT_TRUE(!leaf.isConstant(vmin, vmax, stat, tol));

        leaf.setValueOn(0);
        EXPECT_TRUE(leaf.isConstant(vmin, vmax, stat, tol));

        leaf.setValueOn(0, val + tol);
        EXPECT_TRUE(leaf.isConstant(vmin, vmax, stat, tol));
        EXPECT_EQ(val, vmin);
        EXPECT_EQ(val + tol, vmax);

        leaf.setValueOn(0, val + 1.01*tol);
        EXPECT_TRUE(!leaf.isConstant(vmin, vmax, stat, tol));
    }
    {// check newer version (v3.2 and newer) with float and random values
        typedef tree::LeafNode<float,3> LeafNodeT;
        const float val = 1.0, tol = 1.0f;
        LeafNodeT leaf(origin, val, true);
        float min = 2.0f, max = -min;
        math::Random01 r(145);// random values in the range [0,1]
        for (Index i=0; i<LeafNodeT::NUM_VALUES; ++i) {
            const float v = float(r());
            if (v < min) min = v;
            if (v > max) max = v;
            leaf.setValueOnly(i, v);
        }
        float vmin = 0.0f, vmax = 0.0f;
        bool stat = false;
        EXPECT_TRUE(leaf.isConstant(vmin, vmax, stat, tol));
        EXPECT_TRUE(stat);
        EXPECT_TRUE(math::isApproxEqual(min, vmin));
        EXPECT_TRUE(math::isApproxEqual(max, vmax));
    }
}

TEST_F(TestLeaf, testMedian)
{
    using namespace openvdb;
    const Coord origin(-9, -2, -8);
    std::vector<float> v{5, 6, 4, 3, 2, 6, 7, 9, 3};
    tree::LeafNode<float, 3> leaf(origin, 1.0f, false);

    float val = 0.0f;
    EXPECT_EQ(Index(0), leaf.medianOn(val));
    EXPECT_EQ(0.0f, val);
    EXPECT_EQ(leaf.numValues(), leaf.medianOff(val));
    EXPECT_EQ(1.0f, val);
    EXPECT_EQ(1.0f, leaf.medianAll());

    leaf.setValue(Coord(0,0,0), v[0]);
    EXPECT_EQ(Index(1), leaf.medianOn(val));
    EXPECT_EQ(v[0], val);
    EXPECT_EQ(leaf.numValues()-1, leaf.medianOff(val));
    EXPECT_EQ(1.0f, val);
    EXPECT_EQ(1.0f, leaf.medianAll());

    leaf.setValue(Coord(0,0,1), v[1]);
    EXPECT_EQ(Index(2), leaf.medianOn(val));
    EXPECT_EQ(v[0], val);
    EXPECT_EQ(leaf.numValues()-2, leaf.medianOff(val));
    EXPECT_EQ(1.0f, val);
    EXPECT_EQ(1.0f, leaf.medianAll());

    leaf.setValue(Coord(0,2,1), v[2]);
    EXPECT_EQ(Index(3), leaf.medianOn(val));
    EXPECT_EQ(v[0], val);
    EXPECT_EQ(leaf.numValues()-3, leaf.medianOff(val));
    EXPECT_EQ(1.0f, val);
    EXPECT_EQ(1.0f, leaf.medianAll());

    leaf.setValue(Coord(1,2,1), v[3]);
    EXPECT_EQ(Index(4), leaf.medianOn(val));
    EXPECT_EQ(v[2], val);
    EXPECT_EQ(leaf.numValues()-4, leaf.medianOff(val));
    EXPECT_EQ(1.0f, val);
    EXPECT_EQ(1.0f, leaf.medianAll());

    leaf.setValue(Coord(1,2,3), v[4]);
    EXPECT_EQ(Index(5), leaf.medianOn(val));
    EXPECT_EQ(v[2], val);
    EXPECT_EQ(leaf.numValues()-5, leaf.medianOff(val));
    EXPECT_EQ(1.0f, val);
    EXPECT_EQ(1.0f, leaf.medianAll());

    leaf.setValue(Coord(2,2,1), v[5]);
    EXPECT_EQ(Index(6), leaf.medianOn(val));
    EXPECT_EQ(v[2], val);
    EXPECT_EQ(leaf.numValues()-6, leaf.medianOff(val));
    EXPECT_EQ(1.0f, val);
    EXPECT_EQ(1.0f, leaf.medianAll());

    leaf.setValue(Coord(2,4,1), v[6]);
    EXPECT_EQ(Index(7), leaf.medianOn(val));
    EXPECT_EQ(v[0], val);
    EXPECT_EQ(leaf.numValues()-7, leaf.medianOff(val));
    EXPECT_EQ(1.0f, val);
    EXPECT_EQ(1.0f, leaf.medianAll());

    leaf.setValue(Coord(2,6,1), v[7]);
    EXPECT_EQ(Index(8), leaf.medianOn(val));
    EXPECT_EQ(v[0], val);
    EXPECT_EQ(leaf.numValues()-8, leaf.medianOff(val));
    EXPECT_EQ(1.0f, val);
    EXPECT_EQ(1.0f, leaf.medianAll());

    leaf.setValue(Coord(7,2,1), v[8]);
    EXPECT_EQ(Index(9), leaf.medianOn(val));
    EXPECT_EQ(v[0], val);
    EXPECT_EQ(leaf.numValues()-9, leaf.medianOff(val));
    EXPECT_EQ(1.0f, val);
    EXPECT_EQ(1.0f, leaf.medianAll());

    leaf.fill(2.0f, true);

    EXPECT_EQ(leaf.numValues(), leaf.medianOn(val));
    EXPECT_EQ(2.0f, val);
    EXPECT_EQ(Index(0), leaf.medianOff(val));
    EXPECT_EQ(2.0f, val);
    EXPECT_EQ(2.0f, leaf.medianAll());
}

TEST_F(TestLeaf, testFill)
{
    using namespace openvdb;
    const Coord origin(-9, -2, -8);

    const float bg = 0.0f, fg = 1.0f;
    tree::LeafNode<float, 3> leaf(origin, bg, false);

    const int bboxDim = 1 + int(leaf.dim() >> 1);
    auto bbox = CoordBBox::createCube(leaf.origin(), bboxDim);
    EXPECT_EQ(math::Pow3(bboxDim), int(bbox.volume()));

    bbox = leaf.getNodeBoundingBox();
    leaf.fill(bbox, bg, false);
    EXPECT_TRUE(leaf.isEmpty());
    leaf.fill(bbox, fg, true);
    EXPECT_TRUE(leaf.isDense());

    leaf.fill(bbox, bg, false);
    EXPECT_TRUE(leaf.isEmpty());

    // Fill a region that is larger than the node but that doesn't completely enclose it.
    bbox.max() = bbox.min() + (bbox.dim() >> 1);
    bbox.expand(bbox.min() - Coord{10});
    leaf.fill(bbox, fg, true);

    // Verify that fill() correctly clips the fill region to the node.
    auto clippedBBox = leaf.getNodeBoundingBox();
    clippedBBox.intersect(bbox);
    EXPECT_EQ(int(clippedBBox.volume()), int(leaf.onVoxelCount()));
}

TEST_F(TestLeaf, testCount)
{
    using namespace openvdb;
    const Coord origin(-9, -2, -8);
    tree::LeafNode<float, 3> leaf(origin, 1.0f, false);

    EXPECT_EQ(Index(3), leaf.log2dim());
    EXPECT_EQ(Index(8), leaf.dim());
    EXPECT_EQ(Index(512), leaf.size());
    EXPECT_EQ(Index(512), leaf.numValues());
    EXPECT_EQ(Index(0), leaf.getLevel());
    EXPECT_EQ(Index(1), leaf.getChildDim());
    EXPECT_EQ(Index(1), leaf.leafCount());
    EXPECT_EQ(Index(0), leaf.nonLeafCount());
    EXPECT_EQ(Index(0), leaf.childCount());

    std::vector<Index> dims;
    leaf.getNodeLog2Dims(dims);
    EXPECT_EQ(size_t(1), dims.size());
    EXPECT_EQ(Index(3), dims[0]);
}

TEST_F(TestLeaf, testTransientData)
{
    using namespace openvdb;
    using LeafT = tree::LeafNode<float, 3>;
    const Coord origin(-9, -2, -8);
    LeafT leaf(origin, 1.0f, false);

    EXPECT_EQ(Index32(0), leaf.transientData());
    leaf.setTransientData(Index32(5));
    EXPECT_EQ(Index32(5), leaf.transientData());
    LeafT leaf2(leaf);
    EXPECT_EQ(Index32(5), leaf2.transientData());
    LeafT leaf3 = leaf;
    EXPECT_EQ(Index32(5), leaf3.transientData());
}
