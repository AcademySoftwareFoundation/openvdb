// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/DenseSparseTools.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>

#include <gtest/gtest.h>

#include "util.h"

class TestDenseSparseTools: public ::testing::Test
{
public:
    void SetUp() override;
    void TearDown() override { delete mDense; }

protected:
    openvdb::tools::Dense<float>* mDense;
    openvdb::math::Coord          mijk;
};


void TestDenseSparseTools::SetUp()
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


    // Square each value
    struct SqrOp
    {
        float operator()(const float& in) const
        { return in * in; }
    };
}


TEST_F(TestDenseSparseTools, testExtractSparseFloatTree)
{
    namespace vdbmath = openvdb::math;


    FloatRule rule(0.5f);

    const float testvalue = 1.f;
    mDense->setValue(mijk, testvalue);
    const float background(0.f);
    openvdb::FloatTree::Ptr result
        = openvdb::tools::extractSparseTree(*mDense, rule, background);

    // The result should have only one active value.

    EXPECT_TRUE(result->activeVoxelCount() == 1);

    // The result should have only one leaf

    EXPECT_TRUE(result->leafCount() == 1);

    // The background

    EXPECT_NEAR(background, result->background(), 1.e-6);

    // The stored value

    EXPECT_NEAR(testvalue, result->getValue(mijk), 1.e-6);
}


TEST_F(TestDenseSparseTools, testExtractSparseBoolTree)
{

    const float testvalue = 1.f;
    mDense->setValue(mijk, testvalue);

    const float cutoff(0.5);

    openvdb::BoolTree::Ptr result
        = openvdb::tools::extractSparseTree(*mDense, BoolRule(cutoff), false);

    // The result should have only one active value.

    EXPECT_TRUE(result->activeVoxelCount() == 1);

    // The result should have only one leaf

    EXPECT_TRUE(result->leafCount() == 1);

    // The background

    EXPECT_TRUE(result->background() == false);

    // The stored value

    EXPECT_TRUE(result->getValue(mijk) == true);
}


TEST_F(TestDenseSparseTools, testExtractSparseAltDenseLayout)
{
    namespace vdbmath = openvdb::math;

    FloatRule rule(0.5f);
    // Create a dense grid with the alternate data layout
    // but the same domain as mDense
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense(mDense->bbox(), 0.f);

    const float testvalue = 1.f;
    dense.setValue(mijk, testvalue);

    const float background(0.f);
    openvdb::FloatTree::Ptr result = openvdb::tools::extractSparseTree(dense, rule, background);

    // The result should have only one active value.

    EXPECT_TRUE(result->activeVoxelCount() == 1);

    // The result should have only one leaf

    EXPECT_TRUE(result->leafCount() == 1);

    // The background

    EXPECT_NEAR(background, result->background(), 1.e-6);

    // The stored value

    EXPECT_NEAR(testvalue, result->getValue(mijk), 1.e-6);
}


TEST_F(TestDenseSparseTools, testExtractSparseMaskedTree)
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

    EXPECT_TRUE(result->activeVoxelCount() == 1);

    // The result should have only one leaf

    EXPECT_TRUE(result->leafCount() == 1);

    // The background

    EXPECT_NEAR(background, result->background(), 1.e-6);

    // The stored value

    EXPECT_NEAR(testvalue, result->getValue(mijk), 1.e-6);
}


TEST_F(TestDenseSparseTools, testDenseTransform)
{

    namespace vdbmath = openvdb::math;

    vdbmath::CoordBBox domain(vdbmath::Coord(-4, -6, 10),
                              vdbmath::Coord( 1, 2, 15));

    // Create dense grid, filled with value
    const float value(2.f); const float valueSqr(value*value);

    openvdb::tools::Dense<float> dense(domain, 0.f);
    dense.fill(value);

    SqrOp op;

    vdbmath::CoordBBox smallBBox(vdbmath::Coord(-5, -5, 11),
                                 vdbmath::Coord( 0,  1, 13) );

    // Apply the transformation
    openvdb::tools::transformDense<float, SqrOp>(dense, smallBBox, op, true);

    vdbmath::Coord ijk;
    // Test results.
    for (ijk[0] = domain.min().x(); ijk[0] < domain.max().x() + 1; ++ijk[0]) {
        for (ijk[1] = domain.min().y(); ijk[1] < domain.max().y() + 1; ++ijk[1]) {
            for (ijk[2] = domain.min().z(); ijk[2] < domain.max().z() + 1; ++ijk[2]) {

                if (smallBBox.isInside(ijk)) {
                    // the functor was applied here
                    // the value should be base * base
                    EXPECT_NEAR(dense.getValue(ijk), valueSqr, 1.e-6);
                } else {
                    // the original value
                    EXPECT_NEAR(dense.getValue(ijk), value, 1.e-6);
                }
            }
        }
    }
}


TEST_F(TestDenseSparseTools, testOver)
{
    namespace vdbmath = openvdb::math;

    const vdbmath::CoordBBox domain(vdbmath::Coord(-10, 0, 5), vdbmath::Coord( 10, 5, 10));
    const openvdb::Coord ijk = domain.min() + openvdb::Coord(1, 1, 1);
    // Create dense grid, filled with value
    const float value(2.f);
    const float strength(1.f);
    const float beta(1.f);

    openvdb::FloatTree src(0.f);
    src.setValue(ijk, 1.f);
    openvdb::FloatTree alpha(0.f);
    alpha.setValue(ijk, 1.f);


    const float expected = openvdb::tools::ds::OpOver<float>::apply(
        value, alpha.getValue(ijk), src.getValue(ijk), strength, beta, 1.f);

    { // testing composite function
        openvdb::tools::Dense<float> dense(domain, 0.f);
        dense.fill(value);

        openvdb::tools::compositeToDense<openvdb::tools::DS_OVER>(
            dense, src, alpha, beta,  strength, true /*threaded*/);

        // Check for over value
        EXPECT_NEAR(dense.getValue(ijk), expected, 1.e-6);
        // Check for original value
        EXPECT_NEAR(dense.getValue(openvdb::Coord(1,1,1) + ijk), value, 1.e-6);
    }

    { // testing sparse explict sparse composite
        openvdb::tools::Dense<float> dense(domain, 0.f);
        dense.fill(value);

        typedef openvdb::tools::ds::CompositeFunctorTranslator<openvdb::tools::DS_OVER, float>
            CompositeTool;
        typedef CompositeTool::OpT Method;
        openvdb::tools::SparseToDenseCompositor<Method, openvdb::FloatTree>
            sparseToDense(dense, src, alpha, beta, strength);

        sparseToDense.sparseComposite(true);
        // Check for over value
        EXPECT_NEAR(dense.getValue(ijk), expected, 1.e-6);
        // Check for original value
        EXPECT_NEAR(dense.getValue(openvdb::Coord(1,1,1) + ijk), value, 1.e-6);
    }

    { // testing sparse explict dense composite
        openvdb::tools::Dense<float> dense(domain, 0.f);
        dense.fill(value);

        typedef openvdb::tools::ds::CompositeFunctorTranslator<openvdb::tools::DS_OVER, float>
            CompositeTool;
        typedef CompositeTool::OpT Method;
        openvdb::tools::SparseToDenseCompositor<Method, openvdb::FloatTree>
            sparseToDense(dense, src, alpha, beta, strength);

        sparseToDense.denseComposite(true);
        // Check for over value
        EXPECT_NEAR(dense.getValue(ijk), expected, 1.e-6);
        // Check for original value
        EXPECT_NEAR(dense.getValue(openvdb::Coord(1,1,1) + ijk), value, 1.e-6);
    }
}
