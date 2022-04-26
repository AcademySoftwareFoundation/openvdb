// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/util/NodeMasks.h>
#include <openvdb/io/Compression.h>

#include <gtest/gtest.h>

using openvdb::Index;

template<typename MaskType> void TestAll();

class TestNodeMask: public ::testing::Test
{
};


template<typename MaskType>
void TestAll()
{
    EXPECT_TRUE(MaskType::memUsage() == MaskType::SIZE/8);
    const Index SIZE = MaskType::SIZE > 512 ? 512 : MaskType::SIZE;

    {// default constructor
        MaskType m;//all bits are off
        for (Index i=0; i<SIZE; ++i) EXPECT_TRUE(m.isOff(i));
        for (Index i=0; i<SIZE; ++i) EXPECT_TRUE(!m.isOn(i));
        EXPECT_TRUE(m.isOff());
        EXPECT_TRUE(!m.isOn());
        EXPECT_TRUE(m.countOn() == 0);
        EXPECT_TRUE(m.countOff()== MaskType::SIZE);
        m.toggle();//all bits are on
        EXPECT_TRUE(m.isOn());
        EXPECT_TRUE(!m.isOff());
        EXPECT_TRUE(m.countOn() == MaskType::SIZE);
        EXPECT_TRUE(m.countOff()== 0);
        for (Index i=0; i<SIZE; ++i) EXPECT_TRUE(!m.isOff(i));
        for (Index i=0; i<SIZE; ++i) EXPECT_TRUE(m.isOn(i));
    }
    {// On constructor
        MaskType m(true);//all bits are on
        EXPECT_TRUE(m.isOn());
        EXPECT_TRUE(!m.isOff());
        EXPECT_TRUE(m.countOn() == MaskType::SIZE);
        EXPECT_TRUE(m.countOff()== 0);
        for (Index i=0; i<SIZE; ++i) EXPECT_TRUE(!m.isOff(i));
        for (Index i=0; i<SIZE; ++i) EXPECT_TRUE(m.isOn(i));
        m.toggle();
        for (Index i=0; i<SIZE; ++i) EXPECT_TRUE(m.isOff(i));
        for (Index i=0; i<SIZE; ++i) EXPECT_TRUE(!m.isOn(i));
        EXPECT_TRUE(m.isOff());
        EXPECT_TRUE(!m.isOn());
        EXPECT_TRUE(m.countOn() == 0);
        EXPECT_TRUE(m.countOff()== MaskType::SIZE);
    }
    {// Off constructor
        MaskType m(false);
        EXPECT_TRUE(m.isOff());
        EXPECT_TRUE(!m.isOn());
        EXPECT_TRUE(m.countOn() == 0);
        EXPECT_TRUE(m.countOff()== MaskType::SIZE);
        m.setOn();
        EXPECT_TRUE(m.isOn());
        EXPECT_TRUE(!m.isOff());
        EXPECT_TRUE(m.countOn() == MaskType::SIZE);
        EXPECT_TRUE(m.countOff()== 0);
        m = MaskType();//copy asignment
        EXPECT_TRUE(m.isOff());
        EXPECT_TRUE(!m.isOn());
        EXPECT_TRUE(m.countOn() == 0);
        EXPECT_TRUE(m.countOff()== MaskType::SIZE);
    }
    {// test setOn, setOff, findFirstOn and findFiratOff
        MaskType m;
        for (Index i=0; i<SIZE; ++i) {
            m.setOn(i);
            EXPECT_TRUE(m.countOn() == 1);
            EXPECT_TRUE(m.findFirstOn() == i);
            EXPECT_TRUE(m.findFirstOff() == (i==0 ? 1 : 0));
            for (Index j=0; j<SIZE; ++j) {
                EXPECT_TRUE( i==j ? m.isOn(j) : m.isOff(j) );
            }
            m.setOff(i);
            EXPECT_TRUE(m.countOn() == 0);
            EXPECT_TRUE(m.findFirstOn() == MaskType::SIZE);
        }
    }
    {// OnIterator
        MaskType m;
        for (Index i=0; i<SIZE; ++i) {
            m.setOn(i);
            for (typename MaskType::OnIterator iter=m.beginOn(); iter; ++iter) {
                EXPECT_TRUE( iter.pos() == i );
            }
            EXPECT_TRUE(m.countOn() == 1);
            m.setOff(i);
            EXPECT_TRUE(m.countOn() == 0);
        }
    }
    {// OffIterator
        MaskType m(true);
        for (Index i=0; i<SIZE; ++i) {
            m.setOff(i);
            EXPECT_TRUE(m.countOff() == 1);
            for (typename MaskType::OffIterator iter=m.beginOff(); iter; ++iter) {
                EXPECT_TRUE( iter.pos() == i );
            }
            EXPECT_TRUE(m.countOn() == MaskType::SIZE-1);
            m.setOn(i);
            EXPECT_TRUE(m.countOff() == 0);
            EXPECT_TRUE(m.countOn() == MaskType::SIZE);
        }
    }
    {// isConstant
        MaskType m(true);//all bits are on
        bool isOn = false;
        EXPECT_TRUE(!m.isOff());
        EXPECT_TRUE(m.isOn());
        EXPECT_TRUE(m.isConstant(isOn));
        EXPECT_TRUE(isOn);
        m.setOff(MaskType::SIZE-1);//sets last bit off
        EXPECT_TRUE(!m.isOff());
        EXPECT_TRUE(!m.isOn());
        EXPECT_TRUE(!m.isConstant(isOn));
        m.setOff();//sets all bits off
        EXPECT_TRUE(m.isOff());
        EXPECT_TRUE(!m.isOn());
        EXPECT_TRUE(m.isConstant(isOn));
        EXPECT_TRUE(!isOn);
    }
    {// DenseIterator
        MaskType m(false);
        for (Index i=0; i<SIZE; ++i) {
            m.setOn(i);
            EXPECT_TRUE(m.countOn() == 1);
            for (typename MaskType::DenseIterator iter=m.beginDense(); iter; ++iter) {
                EXPECT_TRUE( iter.pos()==i ? *iter : !*iter );
            }
            m.setOff(i);
            EXPECT_TRUE(m.countOn() == 0);
        }
    }
}

TEST_F(TestNodeMask, testCompress)
{
    using namespace openvdb;

    using ValueT = int;
    using MaskT = openvdb::util::NodeMask<1>;

    { // no inactive values
        MaskT valueMask(true);
        MaskT childMask;
        std::vector<int> values = {0,1,2,3,4,5,6,7};
        int background = 0;

        EXPECT_EQ(valueMask.countOn(), Index32(8));
        EXPECT_EQ(childMask.countOn(), Index32(0));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(maskCompress.metadata, int8_t(openvdb::io::NO_MASK_OR_INACTIVE_VALS));
        EXPECT_EQ(maskCompress.inactiveVal[0], background);
        EXPECT_EQ(maskCompress.inactiveVal[1], background);
    }

    { // all inactive values are +background
        MaskT valueMask;
        MaskT childMask;
        std::vector<int> values = {10,10,10,10,10,10,10,10};
        int background = 10;

        EXPECT_EQ(valueMask.countOn(), Index32(0));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(maskCompress.metadata, int8_t(openvdb::io::NO_MASK_OR_INACTIVE_VALS));
        EXPECT_EQ(maskCompress.inactiveVal[0], background);
        EXPECT_EQ(maskCompress.inactiveVal[1], background);
    }

    { // all inactive values are -background
        MaskT valueMask;
        MaskT childMask;
        std::vector<int> values = {-10,-10,-10,-10,-10,-10,-10,-10};
        int background = 10;

        EXPECT_EQ(valueMask.countOn(), Index32(0));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(maskCompress.metadata, int8_t(openvdb::io::NO_MASK_AND_MINUS_BG));
        EXPECT_EQ(maskCompress.inactiveVal[0], -background);
        EXPECT_EQ(maskCompress.inactiveVal[1], background);
    }

    { // all inactive vals have the same non-background val
        MaskT valueMask(true);
        MaskT childMask;
        std::vector<int> values = {0,1,500,500,4,500,500,7};
        int background = 10;

        valueMask.setOff(2);
        valueMask.setOff(3);
        valueMask.setOff(5);
        valueMask.setOff(6);
        EXPECT_EQ(valueMask.countOn(), Index32(4));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(int(maskCompress.metadata), int(openvdb::io::NO_MASK_AND_ONE_INACTIVE_VAL));
        EXPECT_EQ(maskCompress.inactiveVal[0], 500);
        EXPECT_EQ(maskCompress.inactiveVal[1], background);
    }

    { // mask selects between -background and +background
        MaskT valueMask;
        MaskT childMask;
        std::vector<int> values = {0,10,10,-10,4,10,-10,10};
        int background = 10;

        valueMask.setOn(0);
        valueMask.setOn(4);
        EXPECT_EQ(valueMask.countOn(), Index32(2));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(int(maskCompress.metadata), int(openvdb::io::MASK_AND_NO_INACTIVE_VALS));
        EXPECT_EQ(maskCompress.inactiveVal[0], -background);
        EXPECT_EQ(maskCompress.inactiveVal[1], background);
    }

    { // mask selects between -background and +background
        MaskT valueMask;
        MaskT childMask;
        std::vector<int> values = {0,-10,-10,10,4,-10,10,-10};
        int background = 10;

        valueMask.setOn(0);
        valueMask.setOn(4);
        EXPECT_EQ(valueMask.countOn(), Index32(2));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(int(maskCompress.metadata), int(openvdb::io::MASK_AND_NO_INACTIVE_VALS));
        EXPECT_EQ(maskCompress.inactiveVal[0], -background);
        EXPECT_EQ(maskCompress.inactiveVal[1], background);
    }

    { // mask selects between backgd and one other inactive val
        MaskT valueMask;
        MaskT childMask;
        std::vector<int> values = {0,500,500,10,4,500,10,500};
        int background = 10;

        valueMask.setOn(0);
        valueMask.setOn(4);
        EXPECT_EQ(valueMask.countOn(), Index32(2));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(int(maskCompress.metadata), int(openvdb::io::MASK_AND_ONE_INACTIVE_VAL));
        EXPECT_EQ(maskCompress.inactiveVal[0], 500);
        EXPECT_EQ(maskCompress.inactiveVal[1], background);
    }

    { // mask selects between two non-background inactive vals
        MaskT valueMask;
        MaskT childMask;
        std::vector<int> values = {0,500,500,2000,4,500,2000,500};
        int background = 10;

        valueMask.setOn(0);
        valueMask.setOn(4);
        EXPECT_EQ(valueMask.countOn(), Index32(2));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(int(maskCompress.metadata), int(openvdb::io::MASK_AND_TWO_INACTIVE_VALS));
        EXPECT_EQ(maskCompress.inactiveVal[0], 500); // first unique value
        EXPECT_EQ(maskCompress.inactiveVal[1], 2000); // second unique value
    }

    { // mask selects between two non-background inactive vals
        MaskT valueMask;
        MaskT childMask;
        std::vector<int> values = {0,2000,2000,500,4,2000,500,2000};
        int background = 10;

        valueMask.setOn(0);
        valueMask.setOn(4);
        EXPECT_EQ(valueMask.countOn(), Index32(2));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(int(maskCompress.metadata), int(openvdb::io::MASK_AND_TWO_INACTIVE_VALS));
        EXPECT_EQ(maskCompress.inactiveVal[0], 2000); // first unique value
        EXPECT_EQ(maskCompress.inactiveVal[1], 500); // second unique value
    }

    { // > 2 inactive vals, so no mask compression at all
        MaskT valueMask;
        MaskT childMask;
        std::vector<int> values = {0,1000,2000,3000,4,2000,500,2000};
        int background = 10;

        valueMask.setOn(0);
        valueMask.setOn(4);
        EXPECT_EQ(valueMask.countOn(), Index32(2));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(int(maskCompress.metadata), int(openvdb::io::NO_MASK_AND_ALL_VALS));
        EXPECT_EQ(maskCompress.inactiveVal[0], 1000); // first unique value
        EXPECT_EQ(maskCompress.inactiveVal[1], 2000); // second unique value
    }

    { // mask selects between two non-background inactive vals (selective child mask)
        MaskT valueMask;
        MaskT childMask;
        std::vector<int> values = {0,1000,2000,3000,4,2000,500,2000};
        int background = 0;

        valueMask.setOn(0);
        valueMask.setOn(4);
        EXPECT_EQ(valueMask.countOn(), Index32(2));

        childMask.setOn(3);
        childMask.setOn(6);
        EXPECT_EQ(childMask.countOn(), Index32(2));

        openvdb::io::MaskCompress<ValueT, MaskT> maskCompress(
            valueMask, childMask, values.data(), background);

        EXPECT_EQ(int(maskCompress.metadata), int(openvdb::io::MASK_AND_TWO_INACTIVE_VALS));
        EXPECT_EQ(maskCompress.inactiveVal[0], 1000); // first unique value
        EXPECT_EQ(maskCompress.inactiveVal[1], 2000); // secone unique value
    }
}

TEST_F(TestNodeMask, testAll4) { TestAll<openvdb::util::NodeMask<4> >(); }
TEST_F(TestNodeMask, testAll3) { TestAll<openvdb::util::NodeMask<3> >(); }
TEST_F(TestNodeMask, testAll2) { TestAll<openvdb::util::NodeMask<2> >(); }
TEST_F(TestNodeMask, testAll1) { TestAll<openvdb::util::NodeMask<1> >(); }
