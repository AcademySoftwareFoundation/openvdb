// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"

#include <fvdb/detail/ops/Ops.h>

#include <torch/script.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>

#ifndef FVDB_EXTERNAL_TEST_DATA_PATH
#error "FVDB_EXTERNAL_TEST_DATA_PATH must be defined"
#endif

struct GaussianProjectionForwardTestFixture : public ::testing::Test {
    void
    loadInputData(const std::string insPath) {
        const auto dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const auto inputsPath = dataPath + std::string("/") + insPath;

        auto inputs = fvdb::test::loadTensors(inputsPath, inputNames);
        means       = inputs[0].cuda();
        quats       = inputs[1].cuda();
        scales      = inputs[2].cuda();
        viewmats    = inputs[3].cuda();
        Ks          = inputs[4].cuda();

        imageWidth   = 647;
        imageHeight  = 420;
        imageOriginW = 0;
        imageOriginH = 0;
    }

    void
    storeData(const std::string outsPath, const std::vector<torch::Tensor> &outputData) {
        const auto dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const auto outputPath = dataPath + std::string("/") + outsPath;

        fvdb::test::storeTensors(outputPath, outputData, outputNames);
    }

    void
    loadTestData(const std::string insPath, const std::string outsPath) {
        // Set the random seed for reproducibility.
        torch::manual_seed(0);

        loadInputData(insPath);

        const auto dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const auto expectedOutputsPath = dataPath + std::string("/") + outsPath;

        auto expectedOutputs = fvdb::test::loadTensors(expectedOutputsPath, outputNames);
        expectedRadii        = expectedOutputs[0].cuda();
        expectedMeans2d      = expectedOutputs[1].cuda();
        expectedDepths       = expectedOutputs[2].cuda();
        expectedConics       = expectedOutputs[3].cuda();
    }

    void
    moveToDevice(const torch::Device &device) {
        means           = means.to(device);
        quats           = quats.to(device);
        scales          = scales.to(device);
        viewmats        = viewmats.to(device);
        Ks              = Ks.to(device);
        expectedMeans2d = expectedMeans2d.to(device);
        expectedRadii   = expectedRadii.to(device);
        expectedDepths  = expectedDepths.to(device);
        expectedConics  = expectedConics.to(device);
    }

    const std::vector<std::string> inputNames = {
        "means",
        "quats",
        "scales",
        "viewmats",
        "Ks",
    };

    const std::vector<std::string> outputNames = {"radii", "means2d", "depths", "conics"};

    // Input tensors
    torch::Tensor means;    // [C, N, 3] or [nnz, 3]
    torch::Tensor quats;    // [C, N, 4] or [nnz, 4]
    torch::Tensor scales;   // [C, N, 3] or [nnz, 3]
    torch::Tensor viewmats; // [C, 16] or [nnz, 16]
    torch::Tensor Ks;       // [C, 9] or [nnz, 9]

    // Expected output tensors
    torch::Tensor expectedMeans2d; // [C, 2]
    torch::Tensor expectedRadii;   // [C, N] or [nnz]
    torch::Tensor expectedDepths;  // [C, N] or [nnz]
    torch::Tensor expectedConics;  // [C, N, 3] or [nnz, 3]

    // Parameters
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t imageOriginW;
    uint32_t imageOriginH;
};

// This is a helper function to generate the output data for the test cases.
// Only enable this test when you want to update the output data.
TEST_F(GaussianProjectionForwardTestFixture, DISABLED_GenerateOutputData) {
    // Load test data using our helper method
    loadInputData("projection_forward_inputs.pt");

    {
        // Perspective projection
        const auto [radii, means2d, depths, conics, compensations] =
            fvdb::detail::ops::dispatchGaussianProjectionForward<torch::kCUDA>(means,
                                                                               quats,
                                                                               scales,
                                                                               viewmats,
                                                                               Ks,
                                                                               imageWidth,
                                                                               imageHeight,
                                                                               0.3,
                                                                               1e-2,
                                                                               1e10,
                                                                               0,
                                                                               false,
                                                                               false);

        std::vector<torch::Tensor> outputData = {radii, means2d, depths, conics};

        auto outputFilename = std::string("projection_persp_forward_outputs.pt");

        storeData(outputFilename, outputData);
    }

    {
        // Orthographic projection
        const auto [radii, means2d, depths, conics, compensations] =
            fvdb::detail::ops::dispatchGaussianProjectionForward<torch::kCUDA>(means,
                                                                               quats,
                                                                               scales,
                                                                               viewmats,
                                                                               Ks,
                                                                               imageWidth,
                                                                               imageHeight,
                                                                               0.3,
                                                                               1e-2,
                                                                               1e10,
                                                                               0,
                                                                               false,
                                                                               true);

        std::vector<torch::Tensor> outputData = {radii, means2d, depths, conics};

        auto outputFilename = std::string("projection_ortho_forward_outputs.pt");

        storeData(outputFilename, outputData);
    }
}

TEST_F(GaussianProjectionForwardTestFixture, TestPerspectiveProjection) {
    loadTestData("projection_forward_inputs.pt", "projection_persp_forward_outputs.pt");

    const auto [radii, means2d, depths, conics, compensations] =
        fvdb::detail::ops::dispatchGaussianProjectionForward<torch::kCUDA>(means,
                                                                           quats,
                                                                           scales,
                                                                           viewmats,
                                                                           Ks,
                                                                           imageWidth,
                                                                           imageHeight,
                                                                           0.3,
                                                                           1e-2,
                                                                           1e10,
                                                                           0,
                                                                           false,
                                                                           false);

    EXPECT_TRUE(torch::allclose(means2d, expectedMeans2d));
    EXPECT_TRUE(torch::allclose(radii, expectedRadii));
    EXPECT_TRUE(torch::allclose(depths, expectedDepths));
    EXPECT_TRUE(torch::allclose(conics, expectedConics));
}

TEST_F(GaussianProjectionForwardTestFixture, TestOrthographicProjection) {
    loadTestData("projection_forward_inputs.pt", "projection_ortho_forward_outputs.pt");

    const auto [radii, means2d, depths, conics, compensations] =
        fvdb::detail::ops::dispatchGaussianProjectionForward<torch::kCUDA>(means,
                                                                           quats,
                                                                           scales,
                                                                           viewmats,
                                                                           Ks,
                                                                           imageWidth,
                                                                           imageHeight,
                                                                           0.3,
                                                                           1e-2,
                                                                           1e10,
                                                                           0,
                                                                           false,
                                                                           true);

    // other outputs are undefined where radii is zero
    auto radiiNonZeroMask = radii > 0; // [C, N]

    EXPECT_TRUE(torch::allclose(means2d.index({radiiNonZeroMask}),
                                expectedMeans2d.index({radiiNonZeroMask})));
    EXPECT_TRUE(torch::allclose(radii, expectedRadii));
    EXPECT_TRUE(torch::allclose(depths.index({radiiNonZeroMask}),
                                expectedDepths.index({radiiNonZeroMask})));
    EXPECT_TRUE(torch::allclose(conics.index({radiiNonZeroMask}),
                                expectedConics.index({radiiNonZeroMask})));
}
