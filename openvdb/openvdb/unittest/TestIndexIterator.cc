// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/IndexIterator.h>
#include <openvdb/Types.h>
#include <openvdb/tree/LeafNode.h>

#include <gtest/gtest.h>
#include <tbb/tick_count.h>

#include <sstream>
#include <iostream>
#include <iomanip>//for setprecision

using namespace openvdb;
using namespace openvdb::points;

class TestIndexIterator: public ::testing::Test
{
}; // class TestIndexIterator


////////////////////////////////////////


/// @brief Functionality similar to openvdb::util::CpuTimer except with prefix padding and no decimals.
///
/// @code
///    ProfileTimer timer("algorithm 1");
///    // code to be timed goes here
///    timer.stop();
/// @endcode
class ProfileTimer
{
public:
    /// @brief Prints message and starts timer.
    ///
    /// @note Should normally be followed by a call to stop()
    ProfileTimer(const std::string& msg)
    {
        (void)msg;
#ifdef PROFILE
        // padd string to 50 characters
        std::string newMsg(msg);
        if (newMsg.size() < 50)     newMsg.insert(newMsg.end(), 50 - newMsg.size(), ' ');
        std::cerr << newMsg << " ... ";
#endif
        mT0 = tbb::tick_count::now();
    }

    ~ProfileTimer() { this->stop(); }

    /// Return Time diference in milliseconds since construction or start was called.
    inline double delta() const
    {
        tbb::tick_count::interval_t dt = tbb::tick_count::now() - mT0;
        return 1000.0*dt.seconds();
    }

    /// @brief Print time in milliseconds since construction or start was called.
    inline void stop() const
    {
#ifdef PROFILE
        std::stringstream ss;
        ss << std::setw(6) << ::round(this->delta());
        std::cerr << "completed in " << ss.str() << " ms\n";
#endif
    }

private:
    tbb::tick_count mT0;
};// ProfileTimer


////////////////////////////////////////


TEST_F(TestIndexIterator, testNullFilter)
{
    NullFilter filter;
    EXPECT_TRUE(filter.initialized());
    EXPECT_TRUE(filter.state() == index::ALL);
    int a = 0;
    EXPECT_TRUE(filter.valid(a));
}


TEST_F(TestIndexIterator, testValueIndexIterator)
{
    using namespace openvdb::tree;

    using LeafNode      = LeafNode<unsigned, 1>;
    using ValueOnIter   = LeafNode::ValueOnIter;

    const int size = LeafNode::SIZE;

    { // one per voxel offset, all active
        LeafNode leafNode;

        for (int i = 0; i < size; i++) {
            leafNode.setValueOn(i, i+1);
        }

        ValueOnIter valueIter = leafNode.beginValueOn();

        IndexIter<ValueOnIter, NullFilter>::ValueIndexIter iter(valueIter);

        EXPECT_TRUE(iter);

        EXPECT_EQ(iterCount(iter), Index64(size));

        // check assignment operator
        auto iter2 = iter;
        EXPECT_EQ(iterCount(iter2), Index64(size));

        ++iter;

        // check coord value
        Coord xyz;
        iter.getCoord(xyz);
        EXPECT_EQ(xyz, openvdb::Coord(0, 0, 1));
        EXPECT_EQ(iter.getCoord(), openvdb::Coord(0, 0, 1));

        // check iterators retrieval
        EXPECT_EQ(iter.valueIter().getCoord(), openvdb::Coord(0, 0, 1));
        EXPECT_EQ(iter.end(), Index32(2));

        ++iter;

        // check coord value
        iter.getCoord(xyz);
        EXPECT_EQ(xyz, openvdb::Coord(0, 1, 0));
        EXPECT_EQ(iter.getCoord(), openvdb::Coord(0, 1, 0));

        // check iterators retrieval
        EXPECT_EQ(iter.valueIter().getCoord(), openvdb::Coord(0, 1, 0));
        EXPECT_EQ(iter.end(), Index32(3));
    }

    { // one per even voxel offsets, only these active
        LeafNode leafNode;

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            if ((i % 2) == 0) {
                leafNode.setValueOn(i, ++offset);
            }
            else {
                leafNode.setValueOff(i, offset);
            }
        }

        {
            ValueOnIter valueIter = leafNode.beginValueOn();

            IndexIter<ValueOnIter, NullFilter>::ValueIndexIter iter(valueIter);

            EXPECT_TRUE(iter);

            EXPECT_EQ(iterCount(iter), Index64(size/2));
        }
    }

    { // one per odd voxel offsets, all active
        LeafNode leafNode;

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            if ((i % 2) == 1) {
                leafNode.setValueOn(i, offset++);
            }
            else {
                leafNode.setValueOn(i, offset);
            }
        }

        {
            ValueOnIter valueIter = leafNode.beginValueOn();

            IndexIter<ValueOnIter, NullFilter>::ValueIndexIter iter(valueIter);

            EXPECT_TRUE(iter);

            EXPECT_EQ(iterCount(iter), Index64(3));
        }
    }

    { // one per even voxel offsets, all active
        LeafNode leafNode;

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            if ((i % 2) == 0) {
                leafNode.setValueOn(i, offset++);
            }
            else {
                leafNode.setValueOn(i, offset);
            }
        }

        {
            ValueOnIter valueIter = leafNode.beginValueOn();

            IndexIter<ValueOnIter, NullFilter>::ValueIndexIter iter(valueIter);

            EXPECT_TRUE(iter);

            EXPECT_EQ(iterCount(iter), Index64(size/2));
        }
    }

    { // one per voxel offset, none active
        LeafNode leafNode;

        for (int i = 0; i < size; i++) {
            leafNode.setValueOff(i, i);
        }

        ValueOnIter valueIter = leafNode.beginValueOn();

        IndexIter<ValueOnIter, NullFilter>::ValueIndexIter iter(valueIter);

        EXPECT_TRUE(!iter);

        EXPECT_EQ(iterCount(iter), Index64(0));
    }
}


struct EvenIndexFilter
{
    static bool initialized() { return true; }
    static bool all() { return false; }
    static bool none() { return false; }
    template <typename IterT>
    bool valid(const IterT& iter) const {
        return ((*iter) % 2) == 0;
    }
};


struct OddIndexFilter
{
    static bool initialized() { return true; }
    static bool all() { return false; }
    static bool none() { return false; }
    OddIndexFilter() : mFilter() { }
    template <typename IterT>
    bool valid(const IterT& iter) const {
        return !mFilter.valid(iter);
    }
private:
    EvenIndexFilter mFilter;
};


struct ConstantIter
{
    ConstantIter(const int _value) : value(_value) { }
    int operator*() const { return value; }
    const int value;
};


TEST_F(TestIndexIterator, testFilterIndexIterator)
{
    { // index iterator with even filter
        EvenIndexFilter filter;
        ValueVoxelCIter indexIter(0, 5);
        IndexIter<ValueVoxelCIter, EvenIndexFilter> iter(indexIter, filter);

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, Index32(0));

        EXPECT_TRUE(iter.next());
        EXPECT_EQ(*iter, Index32(2));

        EXPECT_TRUE(iter.next());
        EXPECT_EQ(*iter, Index32(4));

        EXPECT_TRUE(!iter.next());

        EXPECT_EQ(iter.end(), Index32(5));
        EXPECT_EQ(filter.valid(ConstantIter(1)), iter.filter().valid(ConstantIter(1)));
        EXPECT_EQ(filter.valid(ConstantIter(2)), iter.filter().valid(ConstantIter(2)));
    }

    { // index iterator with odd filter
        OddIndexFilter filter;
        ValueVoxelCIter indexIter(0, 5);
        IndexIter<ValueVoxelCIter, OddIndexFilter> iter(indexIter, filter);

        EXPECT_EQ(*iter, Index32(1));

        EXPECT_TRUE(iter.next());
        EXPECT_EQ(*iter, Index32(3));

        EXPECT_TRUE(!iter.next());
    }
}

TEST_F(TestIndexIterator, testProfile)
{
    using namespace openvdb::util;
    using namespace openvdb::math;
    using namespace openvdb::tree;

#ifdef PROFILE
    const int elements(1000 * 1000 * 1000);

    std::cerr << std::endl;
#else
    const int elements(10 * 1000 * 1000);
#endif

    { // for loop
        ProfileTimer timer("ForLoop: sum");
        volatile uint64_t sum = 0;
        for (int i = 0; i < elements; i++) {
            sum = sum + i;
        }
        EXPECT_TRUE(sum);
    }

    { // index iterator
        ProfileTimer timer("IndexIter: sum");
        volatile uint64_t sum = 0;
        ValueVoxelCIter iter(0, elements);
        for (; iter; ++iter) {
            sum = sum + *iter;
        }
        EXPECT_TRUE(sum);
    }

    using LeafNode = LeafNode<unsigned, 3>;
    LeafNode leafNode;

    const int size = LeafNode::SIZE;

    for (int i = 0; i < size - 1; i++) {
        leafNode.setValueOn(i, (elements / size) * i);
    }
    leafNode.setValueOn(size - 1, elements);

    { // manual value iteration
        ProfileTimer timer("ValueIteratorManual: sum");
        volatile uint64_t sum = 0;
        auto indexIter(leafNode.cbeginValueOn());
        int offset = 0;
        for (; indexIter; ++indexIter) {
            int start = offset > 0 ? leafNode.getValue(offset - 1) : 0;
            int end = leafNode.getValue(offset);
            for (int i = start; i < end; i++) {
                sum = sum + i;
            }
            offset++;
        }
        EXPECT_TRUE(sum);
    }

    { // value on iterator (all on)
        ProfileTimer timer("ValueIndexIter: sum");
        volatile uint64_t sum = 0;
        auto indexIter(leafNode.cbeginValueAll());
        IndexIter<LeafNode::ValueAllCIter, NullFilter>::ValueIndexIter iter(indexIter);
        for (; iter; ++iter) {
            sum = sum + *iter;
        }
        EXPECT_TRUE(sum);
    }
}
