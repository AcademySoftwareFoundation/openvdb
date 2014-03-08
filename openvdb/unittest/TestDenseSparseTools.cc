///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
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

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/DenseSparseTools.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include "util.h"

class TestDenseSparseTools: public CppUnit::TestCase
{
public:
    virtual void setUp();
    virtual void tearDown() { if (mDense) delete mDense;}

    CPPUNIT_TEST_SUITE(TestDenseSparseTools);
    CPPUNIT_TEST(testExtractSparseFloatTree);
    CPPUNIT_TEST(testExtractSparseBoolTree);
    CPPUNIT_TEST(testExtractSparseAltDenseLayout);
    CPPUNIT_TEST(testExtractSparseMaskedTree);
    CPPUNIT_TEST_SUITE_END();
    
    void testExtractSparseFloatTree();
    void testExtractSparseBoolTree();
    void testExtractSparseAltDenseLayout();
    void testExtractSparseMaskedTree();


private:
    openvdb::tools::Dense<float>* mDense;
    openvdb::math::Coord          mijk;
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDenseSparseTools);


void 
TestDenseSparseTools::setUp() 
{
    namespace vdbmath = openvdb::math;   
    
    // Domain for the dense grid

    vdbmath::CoordBBox domain(vdbmath::Coord(-100, -16, 12),
                              vdbmath::Coord( 90, 103, 100));

    // Create dense grid, filled with 0.f
    
    mDense = new openvdb::tools::Dense<float>(domain, 0.f);

    // Insert non-zero values
 
    mijk[0] = 1; mijk[1] = -2; mijk[2] = 14;
 
}

namespace {

    // Simple Rule for extracting data greater than a determined mMaskValue
    // and producing a tree that holds type ValueType
    namespace vdbmath = openvdb::math;
    
    class FloatRule
    {
    public:
        // Standard tree type (e.g. BoolTree or FloatTree in openvdb.h)
        typedef openvdb::FloatTree            ResultTreeType;
        typedef ResultTreeType::LeafNodeType  ResultLeafNodeType;

        typedef float                                  ResultValueType;
        typedef float                                  DenseValueType;
                         
        FloatRule(const DenseValueType& value): mMaskValue(value){}
    
        template <typename IndexOrCoord>
        void operator()(const DenseValueType& a, const IndexOrCoord& offset,
                    ResultLeafNodeType* leaf) const 
        {
            if (a > mMaskValue) {
                leaf->setValueOn(offset, a);
            }
        }
        
    private:
        const DenseValueType mMaskValue;
    };

    class BoolRule
    {
    public:
        // Standard tree type (e.g. BoolTree or FloatTree in openvdb.h)
        typedef openvdb::BoolTree             ResultTreeType;
        typedef ResultTreeType::LeafNodeType  ResultLeafNodeType;
      
        typedef bool                                   ResultValueType;
        typedef float                                  DenseValueType;
                         
        BoolRule(const DenseValueType& value): mMaskValue(value){}
    
        template <typename IndexOrCoord>
        void operator()(const DenseValueType& a, const IndexOrCoord& offset,
                    ResultLeafNodeType* leaf) const 
        {
            if (a > mMaskValue) {
                leaf->setValueOn(offset, true);
            }
        }
        
    private:
        const DenseValueType mMaskValue;
    };
    

}

void
TestDenseSparseTools::testExtractSparseFloatTree()
{
    namespace vdbmath = openvdb::math;
    
    
    FloatRule rule(0.5f);
    
    const float testvalue = 1.f;
    mDense->setValue(mijk, testvalue);
    const float background(0.f);
    openvdb::FloatTree::Ptr result 
        = openvdb::tools::extractSparseTree(*mDense, rule, background);

    // The result should have only one active value.

    CPPUNIT_ASSERT(result->activeVoxelCount() == 1);

    // The result should have only one leaf 
    
    CPPUNIT_ASSERT(result->leafCount() == 1);

    // The background 

    CPPUNIT_ASSERT_DOUBLES_EQUAL(background, result->background(), 1.e-6);

    // The stored value

    CPPUNIT_ASSERT_DOUBLES_EQUAL(testvalue, result->getValue(mijk), 1.e-6);
}


void
TestDenseSparseTools::testExtractSparseBoolTree()
{
   
    const float testvalue = 1.f;
    mDense->setValue(mijk, testvalue);

    const float cutoff(0.5);
    
    openvdb::BoolTree::Ptr result 
        = openvdb::tools::extractSparseTree(*mDense, BoolRule(cutoff), false);

    // The result should have only one active value.
    
    CPPUNIT_ASSERT(result->activeVoxelCount() == 1);

    // The result should have only one leaf 
    
    CPPUNIT_ASSERT(result->leafCount() == 1);

    // The background 

    CPPUNIT_ASSERT(result->background() == false);

    // The stored value

    CPPUNIT_ASSERT(result->getValue(mijk) == true);
}


void
TestDenseSparseTools::testExtractSparseAltDenseLayout()
{
    namespace vdbmath = openvdb::math;
    
    FloatRule rule(0.5f);
    // Create a dense grid with the alternate data layout
    // but the same domain as mDense
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense(mDense->bbox(), 0.f);
    
    const float testvalue = 1.f;
    dense.setValue(mijk, testvalue);

    const float background(0.f);
    openvdb::FloatTree::Ptr result 
        = openvdb::tools::extractSparseTree(dense, rule, background);


    // The result should have only one active value.
    
    CPPUNIT_ASSERT(result->activeVoxelCount() == 1);

    // The result should have only one leaf 
    
    CPPUNIT_ASSERT(result->leafCount() == 1);

    // The background 

    CPPUNIT_ASSERT_DOUBLES_EQUAL(background, result->background(), 1.e-6);

    // The stored value

    CPPUNIT_ASSERT_DOUBLES_EQUAL(testvalue, result->getValue(mijk), 1.e-6);
}

void
TestDenseSparseTools::testExtractSparseMaskedTree()
{
    namespace vdbmath = openvdb::math;

    const float testvalue = 1.f;
    mDense->setValue(mijk, testvalue);

    // Create a mask with two values.  One in the domain of 
    // interest and one outside.  The intersection of the active
    // state topology of the mask and the domain of interest will define 
    // the topology of the extracted result.

    openvdb::FloatTree mask(0.f);

    // turn on a point inside the bouding domain of the dense grid
    mask.setValue(mijk, 5.f); 

    // turn on a point outside the bounding domain of the dense grid
    vdbmath::Coord outsidePoint = mDense->bbox().min() - vdbmath::Coord(3, 3, 3);
    mask.setValue(outsidePoint, 1.f);

    float background = 10.f;
         
    openvdb::FloatTree::Ptr result 
        = openvdb::tools::extractSparseTreeWithMask(*mDense, mask, background);
 
    // The result should have only one active value.
    
    CPPUNIT_ASSERT(result->activeVoxelCount() == 1);

    // The result should have only one leaf 
    
    CPPUNIT_ASSERT(result->leafCount() == 1);

    // The background 

    CPPUNIT_ASSERT_DOUBLES_EQUAL(background, result->background(), 1.e-6);

    // The stored value

    CPPUNIT_ASSERT_DOUBLES_EQUAL(testvalue, result->getValue(mijk), 1.e-6);

}



// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
