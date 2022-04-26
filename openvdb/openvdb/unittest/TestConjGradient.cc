// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/version.h>
#include <openvdb/math/ConjGradient.h>

#include <gtest/gtest.h>


class TestConjGradient: public ::testing::Test
{
};


////////////////////////////////////////


TEST_F(TestConjGradient, testJacobi)
{
    using namespace openvdb;

    typedef math::pcg::SparseStencilMatrix<double, 7> MatrixType;

    const math::pcg::SizeType rows = 5;

    MatrixType A(rows);
    A.setValue(0, 0, 24.0);
    A.setValue(0, 2,  6.0);
    A.setValue(1, 1,  8.0);
    A.setValue(1, 2,  2.0);
    A.setValue(2, 0,  6.0);
    A.setValue(2, 1,  2.0);
    A.setValue(2, 2,  8.0);
    A.setValue(2, 3, -6.0);
    A.setValue(2, 4,  2.0);
    A.setValue(3, 2, -6.0);
    A.setValue(3, 3, 24.0);
    A.setValue(4, 2,  2.0);
    A.setValue(4, 4,  8.0);

    EXPECT_TRUE(A.isFinite());

    MatrixType::VectorType
        x(rows, 0.0),
        b(rows, 1.0),
        expected(rows);

    expected[0] = 0.0104167;
    expected[1] = 0.09375;
    expected[2] = 0.125;
    expected[3] = 0.0729167;
    expected[4] = 0.09375;

    math::pcg::JacobiPreconditioner<MatrixType> precond(A);

    // Solve A * x = b for x.
    math::pcg::State result = math::pcg::solve(
        A, b, x, precond, math::pcg::terminationDefaults<double>());

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.iterations <= 20);
    EXPECT_TRUE(x.eq(expected, 1.0e-5));
}


TEST_F(TestConjGradient, testIncompleteCholesky)
{
    using namespace openvdb;

    typedef math::pcg::SparseStencilMatrix<double, 7> MatrixType;
    typedef math::pcg::IncompleteCholeskyPreconditioner<MatrixType> CholeskyPrecond;

    const math::pcg::SizeType rows = 5;

    MatrixType A(5);
    A.setValue(0, 0, 24.0);
    A.setValue(0, 2,  6.0);
    A.setValue(1, 1,  8.0);
    A.setValue(1, 2,  2.0);
    A.setValue(2, 0,  6.0);
    A.setValue(2, 1,  2.0);
    A.setValue(2, 2,  8.0);
    A.setValue(2, 3, -6.0);
    A.setValue(2, 4,  2.0);
    A.setValue(3, 2, -6.0);
    A.setValue(3, 3, 24.0);
    A.setValue(4, 2,  2.0);
    A.setValue(4, 4,  8.0);

    EXPECT_TRUE(A.isFinite());

    CholeskyPrecond precond(A);
    {
        const CholeskyPrecond::TriangularMatrix lower = precond.lowerMatrix();

        CholeskyPrecond::TriangularMatrix expected(5);
        expected.setValue(0, 0,  4.89898);
        expected.setValue(1, 1,  2.82843);
        expected.setValue(2, 0,  1.22474);
        expected.setValue(2, 1,  0.707107);
        expected.setValue(2, 2,  2.44949);
        expected.setValue(3, 2, -2.44949);
        expected.setValue(3, 3,  4.24264);
        expected.setValue(4, 2,  0.816497);
        expected.setValue(4, 4,  2.70801);

#if 0
        std::cout << "Expected:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "    " << expected.getConstRow(i).str() << std::endl;
        }
        std::cout << "Actual:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "    " << lower.getConstRow(i).str() << std::endl;
        }
#endif

        EXPECT_TRUE(lower.eq(expected, 1.0e-5));
    }
    {
        const CholeskyPrecond::TriangularMatrix upper = precond.upperMatrix();

        CholeskyPrecond::TriangularMatrix expected(5);
        {
            expected.setValue(0, 0,  4.89898);
            expected.setValue(0, 2,  1.22474);
            expected.setValue(1, 1,  2.82843);
            expected.setValue(1, 2,  0.707107);
            expected.setValue(2, 2,  2.44949);
            expected.setValue(2, 3, -2.44949);
            expected.setValue(2, 4,  0.816497);
            expected.setValue(3, 3,  4.24264);
            expected.setValue(4, 4,  2.70801);
        }

#if 0
        std::cout << "Expected:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "    " << expected.getConstRow(i).str() << std::endl;
        }
        std::cout << "Actual:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "    " << upper.getConstRow(i).str() << std::endl;
        }
#endif

        EXPECT_TRUE(upper.eq(expected, 1.0e-5));
    }

    MatrixType::VectorType
        x(rows, 0.0),
        b(rows, 1.0),
        expected(rows);

    expected[0] = 0.0104167;
    expected[1] = 0.09375;
    expected[2] = 0.125;
    expected[3] = 0.0729167;
    expected[4] = 0.09375;

    // Solve A * x = b for x.
    math::pcg::State result = math::pcg::solve(
        A, b, x, precond, math::pcg::terminationDefaults<double>());

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.iterations <= 20);
    EXPECT_TRUE(x.eq(expected, 1.0e-5));
}

TEST_F(TestConjGradient, testVectorDotProduct)
{
    using namespace openvdb;

    typedef math::pcg::Vector<double>  VectorType;

    // Test small vector - runs in series
    {
        const size_t length = 1000;
        VectorType aVec(length, 2.0);
        VectorType bVec(length, 3.0);

        VectorType::ValueType result = aVec.dot(bVec);

        EXPECT_NEAR(result, 6.0 * length, 1.0e-7);
    }
    // Test long vector  - runs in parallel
    {
        const size_t length = 10034502;
        VectorType aVec(length, 2.0);
        VectorType bVec(length, 3.0);

        VectorType::ValueType result = aVec.dot(bVec);

        EXPECT_NEAR(result, 6.0 * length, 1.0e-7);
    }
}
