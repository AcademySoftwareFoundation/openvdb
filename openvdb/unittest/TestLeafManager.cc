///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include <cstdio> // for remove()
#include <fstream>
#include <sstream>
#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/tree/LeafManager.h>
#include "util.h" // for unittest_util::makeSphere()

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);


//typedef float                                            ValueType;
//typedef openvdb::tree::LeafNode<ValueType,3>             LeafNodeType;
//typedef openvdb::tree::InternalNode<LeafNodeType,4>      InternalNodeType1;
//typedef openvdb::tree::InternalNode<InternalNodeType1,5> InternalNodeType2;
//typedef openvdb::tree::RootNode<InternalNodeType2>       RootNodeType;


class TestLeafManager: public CppUnit::TestFixture
{
public:
    virtual void setUp() { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestLeafManager);
    CPPUNIT_TEST(testAll);
    CPPUNIT_TEST_SUITE_END();
   
    void testAll();
};


CPPUNIT_TEST_SUITE_REGISTRATION(TestLeafManager);
  
void
TestLeafManager::testAll()
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    using openvdb::Vec3f;
    using openvdb::FloatGrid;
    using openvdb::FloatTree;

    const Vec3f center(0.35f, 0.35f, 0.35f);
    const float radius = 0.15f;
    const int dim = 128, half_width = 5;
    const float voxel_size = 1.0f/dim;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/half_width*voxel_size);
    FloatTree& tree = grid->tree();
    grid->setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/voxel_size));

    unittest_util::makeSphere<FloatGrid>(
        Coord(dim), center, radius, *grid, unittest_util::SPHERE_SPARSE_NARROW_BAND);
    const size_t leafCount = tree.leafCount();

    //grid->print(std::cout, 3);
    {// test with no aux buffers
        openvdb::tree::LeafManager<FloatTree> r(tree);
        CPPUNIT_ASSERT_EQUAL(leafCount, r.leafCount());
        CPPUNIT_ASSERT_EQUAL(size_t(0), r.auxBufferCount());
        CPPUNIT_ASSERT_EQUAL(size_t(0), r.auxBuffersPerLeaf());
        size_t n = 0;
        for (FloatTree::LeafCIter iter=tree.cbeginLeaf(); iter; ++iter, ++n) {
            CPPUNIT_ASSERT(r.leaf(n) == *iter);
            CPPUNIT_ASSERT(r.getBuffer(n,0) == iter->buffer());
        }
        CPPUNIT_ASSERT_EQUAL(r.leafCount(), n);
        CPPUNIT_ASSERT(!r.swapBuffer(0,0));

        r.rebuildAuxBuffers(2);

        CPPUNIT_ASSERT_EQUAL(leafCount, r.leafCount());
        CPPUNIT_ASSERT_EQUAL(size_t(2), r.auxBuffersPerLeaf());
        CPPUNIT_ASSERT_EQUAL(size_t(2*leafCount),r.auxBufferCount());

         for (n=0; n<leafCount; ++n) {
            CPPUNIT_ASSERT(r.getBuffer(n,0) == r.getBuffer(n,1));
            CPPUNIT_ASSERT(r.getBuffer(n,1) == r.getBuffer(n,2));
            CPPUNIT_ASSERT(r.getBuffer(n,0) == r.getBuffer(n,2));
        }
    }
    {// test with 2 aux buffers
        openvdb::tree::LeafManager<FloatTree> r(tree, 2);
        CPPUNIT_ASSERT_EQUAL(leafCount, r.leafCount());
        CPPUNIT_ASSERT_EQUAL(size_t(2), r.auxBuffersPerLeaf());
        CPPUNIT_ASSERT_EQUAL(size_t(2*leafCount),r.auxBufferCount());
        size_t n = 0;
        for (FloatTree::LeafCIter iter=tree.cbeginLeaf(); iter; ++iter, ++n) {
            CPPUNIT_ASSERT(r.leaf(n) == *iter);
            CPPUNIT_ASSERT(r.getBuffer(n,0) == iter->buffer());

            CPPUNIT_ASSERT(r.getBuffer(n,0) == r.getBuffer(n,1));
            CPPUNIT_ASSERT(r.getBuffer(n,1) == r.getBuffer(n,2));
            CPPUNIT_ASSERT(r.getBuffer(n,0) == r.getBuffer(n,2));
        }
        CPPUNIT_ASSERT_EQUAL(r.leafCount(), n);
        for (n=0; n<leafCount; ++n) r.leaf(n).buffer().setValue(4,2.4f);
        for (n=0; n<leafCount; ++n) {
            CPPUNIT_ASSERT(r.getBuffer(n,0) != r.getBuffer(n,1));
            CPPUNIT_ASSERT(r.getBuffer(n,1) == r.getBuffer(n,2));
            CPPUNIT_ASSERT(r.getBuffer(n,0) != r.getBuffer(n,2));
        }
        r.syncAllBuffers();
        for (n=0; n<leafCount; ++n) {
            CPPUNIT_ASSERT(r.getBuffer(n,0) == r.getBuffer(n,1));
            CPPUNIT_ASSERT(r.getBuffer(n,1) == r.getBuffer(n,2));
            CPPUNIT_ASSERT(r.getBuffer(n,0) == r.getBuffer(n,2));
        }
        for (n=0; n<leafCount; ++n) r.getBuffer(n,1).setValue(4,5.4f);
        for (n=0; n<leafCount; ++n) {
            CPPUNIT_ASSERT(r.getBuffer(n,0) != r.getBuffer(n,1));
            CPPUNIT_ASSERT(r.getBuffer(n,1) != r.getBuffer(n,2));
            CPPUNIT_ASSERT(r.getBuffer(n,0) == r.getBuffer(n,2));
        }
        CPPUNIT_ASSERT(r.swapLeafBuffer(1));
        for (n=0; n<leafCount; ++n) {
            CPPUNIT_ASSERT(r.getBuffer(n,0) != r.getBuffer(n,1));
            CPPUNIT_ASSERT(r.getBuffer(n,1) == r.getBuffer(n,2));
            CPPUNIT_ASSERT(r.getBuffer(n,0) != r.getBuffer(n,2));
        }
        r.syncAuxBuffer(1);
        for (n=0; n<leafCount; ++n) {
            CPPUNIT_ASSERT(r.getBuffer(n,0) == r.getBuffer(n,1));
            CPPUNIT_ASSERT(r.getBuffer(n,1) != r.getBuffer(n,2));
            CPPUNIT_ASSERT(r.getBuffer(n,0) != r.getBuffer(n,2));
        }
        r.syncAuxBuffer(2);
        for (n=0; n<leafCount; ++n) {
            CPPUNIT_ASSERT(r.getBuffer(n,0) == r.getBuffer(n,1));
            CPPUNIT_ASSERT(r.getBuffer(n,1) == r.getBuffer(n,2));
        }
    }
    {// test with const tree (buffers are not swappable)
        openvdb::tree::LeafManager<const FloatTree> r(tree);

        for (size_t numAuxBuffers = 0; numAuxBuffers <= 2; ++numAuxBuffers += 2) {
            r.rebuildAuxBuffers(numAuxBuffers);

            CPPUNIT_ASSERT_EQUAL(leafCount, r.leafCount());
            CPPUNIT_ASSERT_EQUAL(int(numAuxBuffers * leafCount), int(r.auxBufferCount()));
            CPPUNIT_ASSERT_EQUAL(numAuxBuffers, r.auxBuffersPerLeaf());

            size_t n = 0;
            for (FloatTree::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter, ++n) {
                CPPUNIT_ASSERT(r.leaf(n) == *iter);
                // Verify that each aux buffer was initialized with a copy of the leaf buffer.
                for (size_t bufIdx = 0; bufIdx < numAuxBuffers; ++bufIdx) {
                    CPPUNIT_ASSERT(r.getBuffer(n, bufIdx) == iter->buffer());
                }
            }
            CPPUNIT_ASSERT_EQUAL(r.leafCount(), n);

            for (size_t i = 0; i < numAuxBuffers; ++i) {
                for (size_t j = 0; j < numAuxBuffers; ++j) {
                    // Verify that swapping buffers with themselves and swapping
                    // leaf buffers with aux buffers have no effect.
                    const bool canSwap = (i != j && i != 0 && j != 0);
                    CPPUNIT_ASSERT_EQUAL(canSwap, r.swapBuffer(i, j));
                }
            }
        }
    }
}

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
