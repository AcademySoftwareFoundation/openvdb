// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <detail/ops/Ops.h>
#include <tests/utils/Tensor.h>

#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>

struct TestParams {
    float azimuth;
    float elevation;
    int64_t shDegreeToUse;
    int64_t numGaussians;
    int64_t numChannels;
    int64_t numCameras;
    bool setZeroRadii;
    bool noRadii;
};

struct SphericalHarmonincsBackwardTestFixture : public ::testing::TestWithParam<TestParams> {
    void
    SetUp() override {
        TestParams testParams = GetParam();
        float azimuth         = testParams.azimuth;
        float elevation       = testParams.elevation;
        shDegreeToUse         = testParams.shDegreeToUse;
        numGaussians          = testParams.numGaussians;
        numChannels           = testParams.numChannels;
        numCameras            = testParams.numCameras;
        setZeroRadii          = testParams.setZeroRadii;
        noRadii               = testParams.noRadii;

        const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);
        const auto intOptsCUDA   = fvdb::test::tensorOpts<int>(torch::kCUDA);

        const auto cosAzimuth =
            torch::cos(torch::full({numCameras, numGaussians, 1}, azimuth, floatOptsCUDA));
        const auto sinAzimuth =
            torch::sin(torch::full({numCameras, numGaussians, 1}, azimuth, floatOptsCUDA));
        const auto cosElevation =
            torch::cos(torch::full({numCameras, numGaussians, 1}, elevation, floatOptsCUDA));
        const auto sinElevation =
            torch::sin(torch::full({numCameras, numGaussians, 1}, elevation, floatOptsCUDA));

        viewDirs = torch::cat({cosAzimuth * cosElevation, sinAzimuth * cosElevation, sinElevation},
                              2); // [C, N, 3]

        sh0Coeffs = torch::full({numGaussians, 1, numChannels}, 1.0f, floatOptsCUDA);

        K         = (shDegreeToUse + 1) * (shDegreeToUse + 1);
        shNCoeffs = torch::full({numGaussians, K - 1, numChannels}, 1.0f, floatOptsCUDA);

        radii = torch::full({numCameras, numGaussians}, 1, intOptsCUDA);

        dLossDRenderQuantities =
            torch::full({numCameras, numGaussians, numChannels}, 1.0f, floatOptsCUDA);

        if (noRadii) {
            radii = torch::Tensor();
        }
        if (setZeroRadii) {
            setHalfOfRadiiToZero();
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

    void
    checkSh(const int64_t numCameras,
            const int64_t numGaussians,
            const int64_t numChannels,
            const int64_t shDegreeToUse,
            const bool setZeroRadii = false) {
        {
            auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
                fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                    shDegreeToUse,
                    numCameras,
                    numGaussians,
                    viewDirs,
                    shNCoeffs,
                    dLossDRenderQuantities,
                    radii,
                    true);

            if (setZeroRadii) {
                const auto dLdSh0Slice = dLossDSh0Coeffs.index({torch::indexing::Slice(0, -1, 2),
                                                                torch::indexing::Slice(),
                                                                torch::indexing::Slice()});
                const auto dLdShNSlice = dLossDShNCoeffs.index({torch::indexing::Slice(0, -1, 2),
                                                                torch::indexing::Slice(),
                                                                torch::indexing::Slice()});
                const auto dLDViewDirsSlice =
                    dLossDViewDirs.index({torch::indexing::Slice(),
                                          torch::indexing::Slice(0, -1, 2),
                                          torch::indexing::Slice()});

                EXPECT_TRUE(torch::allclose(dLdSh0Slice, torch::zeros_like(dLdSh0Slice)));
                EXPECT_TRUE(torch::allclose(dLdShNSlice, torch::zeros_like(dLdShNSlice)));
                EXPECT_TRUE(torch::allclose(dLDViewDirsSlice, torch::zeros_like(dLDViewDirsSlice)));
            }
            EXPECT_TRUE(dLossDSh0Coeffs.sizes() == sh0Coeffs.sizes());
            EXPECT_TRUE(dLossDShNCoeffs.sizes() == shNCoeffs.sizes());
            EXPECT_TRUE(dLossDViewDirs.sizes() == viewDirs.sizes());
        }

        // We don't return view direction gradients if you don't ask for them
        {
            auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
                fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                    shDegreeToUse,
                    numCameras,
                    numGaussians,
                    viewDirs,
                    shNCoeffs,
                    dLossDRenderQuantities,
                    radii,
                    false);
            if (setZeroRadii) {
                const auto dLdSh0Slice = dLossDSh0Coeffs.index({torch::indexing::Slice(0, -1, 2),
                                                                torch::indexing::Slice(),
                                                                torch::indexing::Slice()});
                const auto dLdShNSlice = dLossDShNCoeffs.index({torch::indexing::Slice(0, -1, 2),
                                                                torch::indexing::Slice(),
                                                                torch::indexing::Slice()});

                EXPECT_TRUE(torch::allclose(dLdSh0Slice, torch::zeros_like(dLdSh0Slice)));
                EXPECT_TRUE(torch::allclose(dLdShNSlice, torch::zeros_like(dLdShNSlice)));
            }
            EXPECT_TRUE(dLossDSh0Coeffs.sizes() == sh0Coeffs.sizes());
            EXPECT_TRUE(dLossDShNCoeffs.sizes() == shNCoeffs.sizes());
            EXPECT_FALSE(dLossDViewDirs.defined());
        }
    }

    void
    checkOnlySh0(const int64_t numCameras,
                 const int64_t numGaussians,
                 const int64_t numChannels,
                 const int64_t shDegreeToUse,
                 bool setZeroRadii = false) {
        const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

        if (setZeroRadii) {
            setHalfOfRadiiToZero();
        }
        torch::Tensor expectedDLossDSh0Coeffs =
            torch::full({numGaussians, numCameras, numChannels}, 0.282095, floatOptsCUDA);
        if (setZeroRadii) {
            expectedDLossDSh0Coeffs.index_put_({torch::indexing::Slice(0, -1, 2),
                                                torch::indexing::Slice(),
                                                torch::indexing::Slice()},
                                               0.0f);
        }

        const auto expectedSh0Sizes = std::vector({numGaussians, int64_t(1), numChannels});

        {
            auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
                fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                    shDegreeToUse,
                    numCameras,
                    numGaussians,
                    viewDirs,
                    shNCoeffs,
                    dLossDRenderQuantities,
                    radii,
                    true);
            EXPECT_TRUE(dLossDSh0Coeffs.sizes() == expectedSh0Sizes);
            EXPECT_FALSE(dLossDShNCoeffs.defined());
            EXPECT_FALSE(dLossDViewDirs.defined());
            EXPECT_TRUE(torch::allclose(dLossDSh0Coeffs, expectedDLossDSh0Coeffs));
        }

        // You can pass in an empty tensor for shNCoeffs and we return an empty tensor for the
        // gradient of shN and viewDirs
        {
            shNCoeffs = torch::Tensor();
            auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
                fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                    shDegreeToUse,
                    numCameras,
                    numGaussians,
                    viewDirs,
                    shNCoeffs,
                    dLossDRenderQuantities,
                    radii,
                    true);
            EXPECT_TRUE(dLossDSh0Coeffs.sizes() == expectedSh0Sizes);
            EXPECT_FALSE(dLossDShNCoeffs.defined());
            EXPECT_FALSE(dLossDViewDirs.defined());
            EXPECT_TRUE(torch::allclose(dLossDSh0Coeffs, expectedDLossDSh0Coeffs));
        }

        // You can pass in an empty tensor for shNCoeffs and viewDirs and we return empty tensors
        // for their gradients
        {
            shNCoeffs = torch::Tensor();
            viewDirs  = torch::Tensor();
            auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
                fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                    shDegreeToUse,
                    numCameras,
                    numGaussians,
                    viewDirs,
                    shNCoeffs,
                    dLossDRenderQuantities,
                    radii,
                    true);
            EXPECT_TRUE(dLossDSh0Coeffs.sizes() == expectedSh0Sizes);
            EXPECT_FALSE(dLossDShNCoeffs.defined());
            EXPECT_FALSE(dLossDViewDirs.defined());
            EXPECT_TRUE(torch::allclose(dLossDSh0Coeffs, expectedDLossDSh0Coeffs));
        }
    }

    torch::Tensor sh0Coeffs;
    torch::Tensor shNCoeffs;
    torch::Tensor viewDirs;
    torch::Tensor radii;
    torch::Tensor dLossDRenderQuantities;
    int64_t K;

    int64_t numCameras;
    int64_t numChannels;
    int64_t numGaussians;
    int64_t shDegreeToUse;
    bool setZeroRadii;
    bool noRadii;
};

TEST_P(SphericalHarmonincsBackwardTestFixture, TestShBackward) {
    if (shDegreeToUse == 0) {
        checkOnlySh0(numCameras, numGaussians, numChannels, shDegreeToUse, setZeroRadii);
    } else {
        checkSh(numCameras, numGaussians, numChannels, shDegreeToUse, setZeroRadii);
    }
}

#undef DEBUG_BENCHMARK
#ifdef DEBUG_BENCHMARK
TEST_F(SphericalHarmonincsBackwardTestFixture, BenchmarkSh0) {
    const float azimuth         = 0.0f;
    const float elevation       = 0.0f;
    const int64_t shDegreeToUse = 0;
    const int64_t numGaussians  = 6128356;
    const int64_t numChannels   = 3;
    const int64_t numCameras    = 4;

    initInputs(azimuth, elevation, shDegreeToUse, numCameras, numGaussians, numChannels);

    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    for (int i = 0; i < 10; i += 1) {
        torch::cuda::synchronize();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
            fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                shDegreeToUse,
                numCameras,
                numGaussians,
                viewDirs,
                shNCoeffs,
                dLossDRenderQuantities,
                radii,
                false);
        torch::cuda::synchronize();
    }

    const int totalIters = 1000;
    int64_t totalTime    = 0;
    for (int i = 0; i < totalIters; i += 1) {
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
            fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                shDegreeToUse,
                numCameras,
                numGaussians,
                viewDirs,
                shNCoeffs,
                dLossDRenderQuantities,
                radii,
                false);
        torch::cuda::synchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        totalTime += duration.count();
    }

    std::cerr << "Avg for deg-0 Spherical Harmonics Backward with no viewDir grad (over "
              << totalIters << " iters): " << (double(totalTime) / double(totalIters)) << " ms"
              << std::endl;
}

TEST_F(SphericalHarmonincsBackwardTestFixture, BenchmarkSh0WithViewDirGrad) {
    const float azimuth         = 0.0f;
    const float elevation       = 0.0f;
    const int64_t shDegreeToUse = 0;
    const int64_t numGaussians  = 6128356;
    const int64_t numChannels   = 3;
    const int64_t numCameras    = 4;

    initInputs(azimuth, elevation, shDegreeToUse, numCameras, numGaussians, numChannels);

    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    for (int i = 0; i < 10; i += 1) {
        torch::cuda::synchronize();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
            fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                shDegreeToUse,
                numCameras,
                numGaussians,
                viewDirs,
                shNCoeffs,
                dLossDRenderQuantities,
                radii,
                false);
        torch::cuda::synchronize();
    }

    const int totalIters = 1000;
    int64_t totalTime    = 0;
    for (int i = 0; i < totalIters; i += 1) {
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
            fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                shDegreeToUse,
                numCameras,
                numGaussians,
                viewDirs,
                shNCoeffs,
                dLossDRenderQuantities,
                radii,
                true);
        torch::cuda::synchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        totalTime += duration.count();
    }

    std::cerr << "Avg for deg-0 Spherical Harmonics Backward with viewDir grad (over " << totalIters
              << " iters): " << (double(totalTime) / double(totalIters)) << " ms" << std::endl;
}

TEST_F(SphericalHarmonincsBackwardTestFixture, BenchmarkShNWithViewDirGrad) {
    const float azimuth         = 0.0f;
    const float elevation       = 0.0f;
    const int64_t shDegreeToUse = 4;
    const int64_t numGaussians  = 6128356;
    const int64_t numChannels   = 3;
    const int64_t numCameras    = 4;

    initInputs(azimuth, elevation, shDegreeToUse, numCameras, numGaussians, numChannels);

    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    for (int i = 0; i < 10; i += 1) {
        torch::cuda::synchronize();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
            fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                shDegreeToUse,
                numCameras,
                numGaussians,
                viewDirs,
                shNCoeffs,
                dLossDRenderQuantities,
                radii,
                false);
        torch::cuda::synchronize();
    }

    const int totalIters = 1000;
    int64_t totalTime    = 0;
    for (int i = 0; i < totalIters; i += 1) {
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
            fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                shDegreeToUse,
                numCameras,
                numGaussians,
                viewDirs,
                shNCoeffs,
                dLossDRenderQuantities,
                radii,
                true);
        torch::cuda::synchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        totalTime += duration.count();
    }

    std::cerr << "Avg for deg-N Spherical Harmonics Backward with viewDir grad (over " << totalIters
              << " iters): " << (double(totalTime) / double(totalIters)) << " ms" << std::endl;
}

TEST_F(SphericalHarmonincsBackwardTestFixture, BenchmarkShNWithoutViewDirGrad) {
    const float azimuth         = 0.0f;
    const float elevation       = 0.0f;
    const int64_t shDegreeToUse = 4;
    const int64_t numGaussians  = 6128356;
    const int64_t numChannels   = 3;
    const int64_t numCameras    = 4;

    initInputs(azimuth, elevation, shDegreeToUse, numCameras, numGaussians, numChannels);

    const auto floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    for (int i = 0; i < 10; i += 1) {
        torch::cuda::synchronize();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
            fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                shDegreeToUse,
                numCameras,
                numGaussians,
                viewDirs,
                shNCoeffs,
                dLossDRenderQuantities,
                radii,
                false);
        torch::cuda::synchronize();
    }

    const int totalIters = 1000;
    int64_t totalTime    = 0;
    for (int i = 0; i < totalIters; i += 1) {
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        auto [dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs] =
            fvdb::detail::ops::dispatchSphericalHarmonicsBackward<torch::kCUDA>(
                shDegreeToUse,
                numCameras,
                numGaussians,
                viewDirs,
                shNCoeffs,
                dLossDRenderQuantities,
                radii,
                false);
        torch::cuda::synchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        totalTime += duration.count();
    }

    std::cerr << "Avg for deg-N Spherical Harmonics Backward with no viewDir grad (over "
              << totalIters << " iters): " << (double(totalTime) / double(totalIters)) << " ms"
              << std::endl;
}
#endif

INSTANTIATE_TEST_SUITE_P(ShBackwardTests,
                         SphericalHarmonincsBackwardTestFixture,
                         ::testing::Values(TestParams{0.0f, 0.0f, 0, 0, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 1, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 1, false, true},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 1, true, false},

                                           TestParams{0.0f, 0.0f, 0, 0, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 1, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 8, 1, false, true},
                                           TestParams{0.0f, 0.0f, 0, 10, 8, 1, true, false},

                                           TestParams{0.0f, 0.0f, 0, 0, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 1, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 2, false, true},
                                           TestParams{0.0f, 0.0f, 0, 10, 3, 2, true, false},

                                           TestParams{0.0f, 0.0f, 0, 0, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 1, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 0, 10, 7, 2, false, true},
                                           TestParams{0.0f, 0.0f, 0, 10, 7, 2, true, false},

                                           TestParams{0.0f, 0.0f, 4, 0, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 1, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 1, false, true},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 1, true, false},

                                           TestParams{0.0f, 0.0f, 4, 0, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 1, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 8, 1, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 8, 1, false, true},
                                           TestParams{0.0f, 0.0f, 4, 10, 8, 1, true, false},

                                           TestParams{0.0f, 0.0f, 4, 0, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 1, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 2, false, true},
                                           TestParams{0.0f, 0.0f, 4, 10, 3, 2, true, false},

                                           TestParams{0.0f, 0.0f, 4, 0, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 1, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 7, 2, false, false},
                                           TestParams{0.0f, 0.0f, 4, 10, 7, 2, false, true},
                                           TestParams{0.0f, 0.0f, 4, 10, 7, 2, true, false}));
