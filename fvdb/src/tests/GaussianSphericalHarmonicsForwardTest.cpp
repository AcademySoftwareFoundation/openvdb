// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <detail/ops/Ops.h>

#include <torch/torch.h>

#include <gtest/gtest.h>
#include <tests/utils/Tensor.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>

struct TestParams {
    float   azimuth;
    float   elevation;
    int64_t shDegreeToUse;
    int64_t numGaussians;
    int64_t numChannels;
    int64_t numCameras;
    bool    setZeroRadii;
    bool    noRadii;
};

struct SphericalHarmonicsForwardTestFixture : public ::testing::TestWithParam<TestParams> {
    void
    SetUp() override {
        TestParams testParams    = GetParam();
        float      azimuth       = testParams.azimuth;
        float      elevation     = testParams.elevation;
        shDegreeToUse            = testParams.shDegreeToUse;
        numGaussians             = testParams.numGaussians;
        numChannels              = testParams.numChannels;
        numCameras               = testParams.numCameras;
        setZeroRadii             = testParams.setZeroRadii;
        noRadii                  = testParams.noRadii;
        const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);
        const auto intOptsCUDA   = fvdb::test::tensorOpts<int>(torch::kCUDA);

        const auto cosAzimuth =
            torch::cos(torch::full({ numCameras, numGaussians, 1 }, azimuth, floatOptsCUDA));
        const auto sinAzimuth =
            torch::sin(torch::full({ numCameras, numGaussians, 1 }, azimuth, floatOptsCUDA));
        const auto cosElevation =
            torch::cos(torch::full({ numCameras, numGaussians, 1 }, elevation, floatOptsCUDA));
        const auto sinElevation =
            torch::sin(torch::full({ numCameras, numGaussians, 1 }, elevation, floatOptsCUDA));

        viewDirs = torch::cat(
            { cosAzimuth * cosElevation, sinAzimuth * cosElevation, sinElevation }, 2); // [C, N, 3]

        sh0Coeffs = torch::full({ numGaussians, 1, numChannels }, 1.0f, floatOptsCUDA);

        const int64_t K = (shDegreeToUse + 1) * (shDegreeToUse + 1);
        shNCoeffs       = torch::full({ numGaussians, K - 1, numChannels }, 1.0f, floatOptsCUDA);

        radii = torch::full({ numCameras, numGaussians }, 1, intOptsCUDA);

        const float expectedShValue = shDegreeToUse == 0 ? (0.2820947917738781 + 0.5) : 0.861482;
        expectedResult =
            torch::full({ numCameras, numGaussians, numChannels }, expectedShValue, floatOptsCUDA);

        if (noRadii) {
            radii = torch::Tensor();
        }

        if (setZeroRadii) {
            setHalfOfRadiiToZero();
            expectedResult.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, -1, 2),
                                        torch::indexing::Slice() },
                                      0.0);
        }
    }

    void
    setHalfOfRadiiToZero() {
        radii         = radii.cpu();
        auto radiiAcc = radii.accessor<int, 2>();
        for (int64_t i = 0; i < radii.size(0); ++i) {
            for (int64_t j = 0; j < radii.size(1); ++j) {
                radiiAcc[i][j] = j % 2;
            }
        }
        const auto intOptsCUDA = fvdb::test::tensorOpts<int>(torch::kCUDA);
        radii                  = radii.to(intOptsCUDA);
    }

    torch::Tensor expectedResult;
    torch::Tensor sh0Coeffs;
    torch::Tensor shNCoeffs;
    torch::Tensor viewDirs;
    torch::Tensor radii;

    int64_t numCameras;
    int64_t numChannels;
    int64_t numGaussians;
    int64_t shDegreeToUse;
    bool    setZeroRadii;
    bool    noRadii;
};

TEST_P(SphericalHarmonicsForwardTestFixture, TestShForward) {
    if (shDegreeToUse == 0) {
        {
            auto result = fvdb::detail::ops::dispatchSphericalHarmonicsForward<torch::kCUDA>(
                shDegreeToUse, numCameras, viewDirs, sh0Coeffs, shNCoeffs, radii);
            EXPECT_TRUE(result.sizes() ==
                        torch::IntArrayRef({ numCameras, numGaussians, numChannels }));
            EXPECT_TRUE(torch::allclose(result, expectedResult));
        }

        {
            shNCoeffs   = torch::Tensor();
            auto result = fvdb::detail::ops::dispatchSphericalHarmonicsForward<torch::kCUDA>(
                shDegreeToUse, numCameras, viewDirs, sh0Coeffs, shNCoeffs, radii);
            EXPECT_TRUE(result.sizes() ==
                        torch::IntArrayRef({ numCameras, numGaussians, numChannels }));
            EXPECT_TRUE(torch::allclose(result, expectedResult));
        }

        {
            viewDirs    = torch::Tensor();
            auto result = fvdb::detail::ops::dispatchSphericalHarmonicsForward<torch::kCUDA>(
                shDegreeToUse, numCameras, viewDirs, sh0Coeffs, shNCoeffs, radii);
            EXPECT_TRUE(result.sizes() ==
                        torch::IntArrayRef({ numCameras, numGaussians, numChannels }));
            EXPECT_TRUE(torch::allclose(result, expectedResult));
        }
    } else {
        auto result = fvdb::detail::ops::dispatchSphericalHarmonicsForward<torch::kCUDA>(
            shDegreeToUse, numCameras, viewDirs, sh0Coeffs, shNCoeffs, radii);
        EXPECT_TRUE(result.sizes() ==
                    torch::IntArrayRef({ numCameras, numGaussians, numChannels }));
        EXPECT_TRUE(torch::allclose(result, expectedResult));
    }
}

#undef DEBUG_BENCHMARK
#ifdef DEBUG_BENCHMARK
TEST_F(SphericalHarmonicsTestFixture, TestSh0Benchmark) {
    const int64_t shDegreeToUse = 0;
    const int64_t numGaussians  = 6128356;
    const int64_t numChannels   = 3;
    const int64_t numCameras    = 4;

    initInputs(0.0f, 0.0f, shDegreeToUse, numCameras, numGaussians, numChannels);

    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    torch::Tensor expectedResult =
        torch::full({ numCameras, numGaussians, numChannels }, 0.861482, floatOptsCUDA);

    // Warm up
    for (int i = 0; i < 10; i += 1) {
        torch::cuda::synchronize();
        auto result = fvdb::detail::ops::dispatchSphericalHarmonicsForward<torch::kCUDA>(
            shDegreeToUse, numCameras, viewDirs, sh0Coeffs, shNCoeffs, radii);
        torch::cuda::synchronize();
    }

    const int totalIters = 1000;
    int64_t   totalTime  = 0;
    for (int i = 0; i < totalIters; i += 1) {
        torch::cuda::synchronize();
        auto start  = std::chrono::high_resolution_clock::now();
        auto result = fvdb::detail::ops::dispatchSphericalHarmonicsForward<torch::kCUDA>(
            shDegreeToUse, numCameras, viewDirs, sh0Coeffs, shNCoeffs, radii);
        torch::cuda::synchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        totalTime += duration.count();
    }
    std::cout << "Avg for 0-deg Spherical Harmonics Forward (over " << totalIters
              << " iters): " << (double(totalTime) / double(totalIters)) << " ms" << std::endl;
}

TEST_F(SphericalHarmonicsTestFixture, TestShNNBenchmark) {
    const int64_t shDegreeToUse = 4;
    const int64_t numGaussians  = 6128356;
    const int64_t numChannels   = 3;
    const int64_t numCameras    = 4;

    initInputs(0.0f, 0.0f, shDegreeToUse, numCameras, numGaussians, numChannels);

    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    torch::Tensor expectedResult =
        torch::full({ numCameras, numGaussians, numChannels }, 0.861482, floatOptsCUDA);

    // Warm up
    for (int i = 0; i < 10; i += 1) {
        torch::cuda::synchronize();
        auto result = fvdb::detail::ops::dispatchSphericalHarmonicsForward<torch::kCUDA>(
            shDegreeToUse, numCameras, viewDirs, sh0Coeffs, shNCoeffs, radii);
        torch::cuda::synchronize();
    }

    const int totalIters = 1000;
    int64_t   totalTime  = 0;
    for (int i = 0; i < totalIters; i += 1) {
        torch::cuda::synchronize();
        auto start  = std::chrono::high_resolution_clock::now();
        auto result = fvdb::detail::ops::dispatchSphericalHarmonicsForward<torch::kCUDA>(
            shDegreeToUse, numCameras, viewDirs, sh0Coeffs, shNCoeffs, radii);
        torch::cuda::synchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        totalTime += duration.count();
    }
    std::cout << "Avg for N-deg Spherical Harmonics Forward (over " << totalIters
              << " iters): " << (double(totalTime) / double(totalIters)) << " ms" << std::endl;
}
#endif

INSTANTIATE_TEST_SUITE_P(ShForwardTests, SphericalHarmonicsForwardTestFixture,
                         ::testing::Values(TestParams{ 0.0f, 0.0f, 0, 0, 3, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 1, 3, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 3, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 3, 1, false, true },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 3, 1, true, false },

                                           TestParams{ 0.0f, 0.0f, 0, 0, 8, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 1, 8, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 8, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 8, 1, false, true },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 8, 1, true, false },

                                           TestParams{ 0.0f, 0.0f, 0, 0, 3, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 1, 3, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 3, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 3, 2, false, true },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 3, 2, true, false },

                                           TestParams{ 0.0f, 0.0f, 0, 0, 7, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 1, 7, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 7, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 7, 2, false, true },
                                           TestParams{ 0.0f, 0.0f, 0, 10, 7, 2, true, false },

                                           TestParams{ 0.0f, 0.0f, 4, 0, 3, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 1, 3, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 3, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 3, 1, false, true },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 3, 1, true, false },

                                           TestParams{ 0.0f, 0.0f, 4, 0, 8, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 1, 8, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 8, 1, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 8, 1, false, true },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 8, 1, true, false },

                                           TestParams{ 0.0f, 0.0f, 4, 0, 3, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 1, 3, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 3, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 3, 2, false, true },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 3, 2, true, false },

                                           TestParams{ 0.0f, 0.0f, 4, 0, 7, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 1, 7, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 7, 2, false, false },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 7, 2, false, true },
                                           TestParams{ 0.0f, 0.0f, 4, 10, 7, 2, true, false }));
