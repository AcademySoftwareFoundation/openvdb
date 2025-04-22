// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <detail/ops/Ops.h>

#include <torch/script.h>
#include <torch/torch.h>

#include <gtest/gtest.h>
#include <tests/utils/Tensor.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>

#ifndef FVDB_EXTERNAL_TEST_DATA_PATH
#error "FVDB_EXTERNAL_TEST_DATA_PATH must be defined"
#endif

struct GaussianProjectionBackwardTestFixture : public ::testing::Test {
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
        if (inputNames.size() > 5) {
            compensations = inputs[5].cuda();
            radii         = inputs[6].cuda();
            conics        = inputs[7].cuda();
        }

        imageWidth   = 647;
        imageHeight  = 420;
        imageOriginW = 0;
        imageOriginH = 0;
        eps2d        = 0.3;
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
        expectedDLossDMeans  = expectedOutputs[0].cuda();
        // expectedDLossDCovars      = expectedOutputs[1].cuda();
        expectedDLossDQuats                      = expectedOutputs[1].cuda();
        expectedDLossDScales                     = expectedOutputs[2].cuda();
        expectedDLossDCamToWorlds                = expectedOutputs[3].cuda();
        expectedNormalizeddLossdMeans2dNormAccum = expectedOutputs[4].cuda();
        expectedNormalizedMaxRadiiAccum          = expectedOutputs[5].cuda();
        expectedGradientStepCounts               = expectedOutputs[6].cuda();
    }

    void
    moveToDevice(const torch::Device &device) {
        means               = means.to(device);
        quats               = quats.to(device);
        scales              = scales.to(device);
        viewmats            = viewmats.to(device);
        Ks                  = Ks.to(device);
        compensations       = compensations.to(device);
        radii               = radii.to(device);
        conics              = conics.to(device);
        expectedDLossDMeans = expectedDLossDMeans.to(device);
        // expectedDLossDCovars      = expectedDLossDCovars.to(device);
        expectedDLossDQuats       = expectedDLossDQuats.to(device);
        expectedDLossDScales      = expectedDLossDScales.to(device);
        expectedDLossDCamToWorlds = expectedDLossDCamToWorlds.to(device);
        expectedNormalizeddLossdMeans2dNormAccum =
            expectedNormalizeddLossdMeans2dNormAccum.to(device);
        expectedNormalizedMaxRadiiAccum = expectedNormalizedMaxRadiiAccum.to(device);
        expectedGradientStepCounts      = expectedGradientStepCounts.to(device);
    }

    std::vector<std::string> inputNames = {
        "means",
        "quats",
        "scales",
        "viewmats",
        "Ks",    // projection inputs
        "compensations",
        "radii",
        "conics" // disable if reading projection inputs and writing backwards outputs
    };

    const std::vector<std::string> outputNames = { "dLossDMeans",
                                                   "dLossDQuats",
                                                   "dLossDScales",
                                                   "dLossDCamToWorlds",
                                                   "normalizeddLossdMeans2dNormAccum",
                                                   "normalizedMaxRadiiAccum",
                                                   "gradientStepCounts" };

    // Input tensors
    torch::Tensor means;         // [C, N, 3] or [nnz, 3]
    torch::Tensor quats;         // [C, N, 4] or [nnz, 4]
    torch::Tensor scales;        // [C, N, 3] or [nnz, 3]
    torch::Tensor viewmats;      // [C, 16] or [nnz, 16]
    torch::Tensor Ks;            // [C, 9] or [nnz, 9]
    torch::Tensor compensations; // [C, N] or [nnz]
    torch::Tensor radii;         // [C, N] or [nnz]
    torch::Tensor conics;        // [C, N, 3] or [nnz, 3]

    // Expected output tensors
    torch::Tensor expectedDLossDMeans; // [C, N, 3] or [nnz, 3]
    // torch::Tensor expectedDLossDCovars;      // [C, N, 9] or [nnz, 9]
    torch::Tensor expectedDLossDQuats;                      // [C, N, 4] or [nnz, 4]
    torch::Tensor expectedDLossDScales;                     // [C, N, 3] or [nnz, 3]
    torch::Tensor expectedDLossDCamToWorlds;                // [C, 16] or [nnz, 16]
    torch::Tensor expectedNormalizeddLossdMeans2dNormAccum; // [C]
    torch::Tensor expectedNormalizedMaxRadiiAccum;          // [C]
    torch::Tensor expectedGradientStepCounts;               // [C]

    // Parameters
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t imageOriginW;
    uint32_t imageOriginH;
    float    eps2d;
};

// This is a helper function to generate the output data for the test cases.
// Only enable this test when you want to update the output data.
TEST_F(GaussianProjectionBackwardTestFixture, DISABLED_GenerateOutputData) {
    // Load test data using our helper method
    auto inputFilename      = std::string("projection_forward_inputs.pt");
    auto originalInputNames = inputNames;
    // remove last 3 input names
    inputNames.pop_back();
    inputNames.pop_back();
    inputNames.pop_back();
    loadInputData(inputFilename);

    {
        // Perspective projection
        const auto [radii_proj, means2d_proj, depths_proj, conics_proj, compensations_proj] =
            fvdb::detail::ops::dispatchGaussianProjectionForward<torch::kCUDA>(
                means, quats, scales, viewmats, Ks, imageWidth, imageHeight, 0.3, 1e-2, 1e10, 0,
                true, false);

        // store the input data for the backward pass
        auto backwardInputFilename = std::string("projection_persp_backward_inputs.pt");
        storeData(backwardInputFilename, { means, quats, scales, viewmats, Ks, compensations_proj,
                                           radii_proj, conics_proj });

        // run backwards pass and store outputs
        torch::Tensor dLossDMeans2d =
            torch::full({ means.size(0), means.size(1), 2 }, 0.1, torch::kCUDA);
        torch::Tensor dLossDDepths =
            torch::full({ means.size(0), means.size(1) }, 0.2, torch::kCUDA);
        torch::Tensor dLossDConics =
            torch::full({ means.size(0), means.size(1), 3 }, 0.3, torch::kCUDA);
        torch::Tensor dLossDCompensations =
            torch::full({ means.size(0), means.size(1) }, 0.4, torch::kCUDA);

        torch::TensorOptions options = torch::kCUDA;
        torch::Tensor        outNormalizeddLossdMeans2dNormAccum =
            torch::zeros({ means.size(0) }, options.dtype(torch::kFloat32));
        torch::Tensor outNormalizedMaxRadiiAccum =
            torch::zeros({ means.size(0) }, options.dtype(torch::kInt32));
        torch::Tensor outGradientStepCounts =
            torch::zeros({ means.size(0) }, options.dtype(torch::kInt32));

        auto [dLossDMeans, dLossDCovars, dLossDQuats, dLossDScales, dLossDCamToWorlds] =
            fvdb::detail::ops::dispatchGaussianProjectionBackward<torch::kCUDA>(
                means, quats, scales, viewmats, Ks, compensations_proj, imageWidth, imageHeight,
                eps2d, radii_proj, conics_proj, dLossDMeans2d, dLossDDepths, dLossDConics,
                dLossDCompensations, true, false, outNormalizeddLossdMeans2dNormAccum,
                outNormalizedMaxRadiiAccum, outGradientStepCounts);

        std::vector<torch::Tensor> outputData = {
            dLossDMeans,
            // dLossDCovars, Currently dLossDCovars is not output, not exposed, see
            // dispatchGaussianProjectionBackward
            dLossDQuats, dLossDScales, dLossDCamToWorlds, outNormalizeddLossdMeans2dNormAccum,
            outNormalizedMaxRadiiAccum, outGradientStepCounts
        };
        auto outputFilename = std::string("projection_persp_backward_outputs.pt");

        storeData(outputFilename, outputData);
    }

    {
        // Orthographic projection
        const auto [radii_proj, means2d_proj, depths_proj, conics_proj, compensations_proj] =
            fvdb::detail::ops::dispatchGaussianProjectionForward<torch::kCUDA>(
                means, quats, scales, viewmats, Ks, imageWidth, imageHeight, 0.3, 1e-2, 1e10, 0,
                true, true);

        // store the input data for the backward pass
        auto backwardInputFilename = std::string("projection_ortho_backward_inputs.pt");
        storeData(backwardInputFilename, { means, quats, scales, viewmats, Ks, compensations_proj,
                                           radii_proj, conics_proj });

        // run backwards pass and store outputs
        torch::Tensor dLossDMeans2d =
            torch::full({ means.size(0), means.size(1), 2 }, 0.1, torch::kCUDA);
        torch::Tensor dLossDDepths =
            torch::full({ means.size(0), means.size(1) }, 0.2, torch::kCUDA);
        torch::Tensor dLossDConics =
            torch::full({ means.size(0), means.size(1), 3 }, 0.3, torch::kCUDA);
        torch::Tensor dLossDCompensations =
            torch::full({ means.size(0), means.size(1) }, 0.4, torch::kCUDA);

        torch::TensorOptions options = torch::kCUDA;
        torch::Tensor        outNormalizeddLossdMeans2dNormAccum =
            torch::zeros({ means.size(0) }, options.dtype(torch::kFloat32));
        torch::Tensor outNormalizedMaxRadiiAccum =
            torch::zeros({ means.size(0) }, options.dtype(torch::kInt32));
        torch::Tensor outGradientStepCounts =
            torch::zeros({ means.size(0) }, options.dtype(torch::kInt32));

        auto [dLossDMeans, dLossDCovars, dLossDQuats, dLossDScales, dLossDCamToWorlds] =
            fvdb::detail::ops::dispatchGaussianProjectionBackward<torch::kCUDA>(
                means, quats, scales, viewmats, Ks, compensations_proj, imageWidth, imageHeight,
                eps2d, radii_proj, conics_proj, dLossDMeans2d, dLossDDepths, dLossDConics,
                dLossDCompensations, true, true, outNormalizeddLossdMeans2dNormAccum,
                outNormalizedMaxRadiiAccum, outGradientStepCounts);

        std::vector<torch::Tensor> outputData = {
            dLossDMeans,
            // dLossDCovars, Currently dLossDCovars is not output, not exposed, see
            // dispatchGaussianProjectionBackward
            dLossDQuats, dLossDScales, dLossDCamToWorlds, outNormalizeddLossdMeans2dNormAccum,
            outNormalizedMaxRadiiAccum, outGradientStepCounts
        };
        auto outputFilename = std::string("projection_ortho_backward_outputs.pt");

        storeData(outputFilename, outputData);
    }

    // restore input names
    inputNames = originalInputNames;
}

TEST_F(GaussianProjectionBackwardTestFixture, TestPerspectiveProjection) {
    loadTestData("projection_persp_backward_inputs.pt", "projection_persp_backward_outputs.pt");

    torch::Tensor dLossDMeans2d =
        torch::full({ means.size(0), means.size(1), 2 }, 0.1, torch::kCUDA);
    torch::Tensor dLossDDepths = torch::full({ means.size(0), means.size(1) }, 0.2, torch::kCUDA);
    torch::Tensor dLossDConics =
        torch::full({ means.size(0), means.size(1), 3 }, 0.3, torch::kCUDA);
    torch::Tensor dLossDCompensations =
        torch::full({ means.size(0), means.size(1) }, 0.4, torch::kCUDA);

    torch::TensorOptions options = torch::kCUDA;
    torch::Tensor        outNormalizeddLossdMeans2dNormAccum =
        torch::zeros({ means.size(0) }, options.dtype(torch::kFloat32));
    torch::Tensor outNormalizedMaxRadiiAccum =
        torch::zeros({ means.size(0) }, options.dtype(torch::kInt32));
    torch::Tensor outGradientStepCounts =
        torch::zeros({ means.size(0) }, options.dtype(torch::kInt32));

    const auto [dLossDMeans, dLossDCovars, dLossDQuats, dLossDScales, dLossDCamToWorlds] =
        fvdb::detail::ops::dispatchGaussianProjectionBackward<torch::kCUDA>(
            means, quats, scales, viewmats, Ks, compensations, imageWidth, imageHeight, eps2d,
            radii, conics, dLossDMeans2d, dLossDDepths, dLossDConics, dLossDCompensations, true,
            false, outNormalizeddLossdMeans2dNormAccum, outNormalizedMaxRadiiAccum,
            outGradientStepCounts);

    EXPECT_TRUE(torch::allclose(dLossDMeans, expectedDLossDMeans));
    EXPECT_TRUE(torch::allclose(dLossDQuats, expectedDLossDQuats));
    EXPECT_TRUE(torch::allclose(dLossDScales, expectedDLossDScales));
    EXPECT_TRUE(torch::allclose(dLossDCamToWorlds, expectedDLossDCamToWorlds));
    EXPECT_TRUE(torch::allclose(outNormalizeddLossdMeans2dNormAccum,
                                expectedNormalizeddLossdMeans2dNormAccum));
    EXPECT_TRUE(torch::allclose(outNormalizedMaxRadiiAccum, expectedNormalizedMaxRadiiAccum));
    EXPECT_TRUE(torch::allclose(outGradientStepCounts, expectedGradientStepCounts));
}

TEST_F(GaussianProjectionBackwardTestFixture, TestOrthographicProjection) {
    loadTestData("projection_ortho_backward_inputs.pt", "projection_ortho_backward_outputs.pt");

    torch::Tensor dLossDMeans2d =
        torch::full({ means.size(0), means.size(1), 2 }, 0.1, torch::kCUDA);
    torch::Tensor dLossDDepths = torch::full({ means.size(0), means.size(1) }, 0.2, torch::kCUDA);
    torch::Tensor dLossDConics =
        torch::full({ means.size(0), means.size(1), 3 }, 0.3, torch::kCUDA);
    torch::Tensor dLossDCompensations =
        torch::full({ means.size(0), means.size(1) }, 0.4, torch::kCUDA);

    torch::TensorOptions options = torch::kCUDA;
    torch::Tensor        outNormalizeddLossdMeans2dNormAccum =
        torch::zeros({ means.size(0) }, options.dtype(torch::kFloat32));
    torch::Tensor outNormalizedMaxRadiiAccum =
        torch::zeros({ means.size(0) }, options.dtype(torch::kInt32));
    torch::Tensor outGradientStepCounts =
        torch::zeros({ means.size(0) }, options.dtype(torch::kInt32));

    const auto [dLossDMeans, dLossDCovars, dLossDQuats, dLossDScales, dLossDCamToWorlds] =
        fvdb::detail::ops::dispatchGaussianProjectionBackward<torch::kCUDA>(
            means, quats, scales, viewmats, Ks, compensations, imageWidth, imageHeight, eps2d,
            radii, conics, dLossDMeans2d, dLossDDepths, dLossDConics, dLossDCompensations, true,
            true, outNormalizeddLossdMeans2dNormAccum, outNormalizedMaxRadiiAccum,
            outGradientStepCounts);

    EXPECT_TRUE(torch::allclose(dLossDMeans, expectedDLossDMeans));
    // EXPECT_TRUE(torch::allclose(dLossDCovars, expectedDLossDCovars));
    EXPECT_TRUE(torch::allclose(dLossDQuats, expectedDLossDQuats));
    EXPECT_TRUE(torch::allclose(dLossDScales, expectedDLossDScales));
    EXPECT_TRUE(torch::allclose(dLossDCamToWorlds, expectedDLossDCamToWorlds));
    EXPECT_TRUE(torch::allclose(outNormalizeddLossdMeans2dNormAccum,
                                expectedNormalizeddLossdMeans2dNormAccum));
    EXPECT_TRUE(torch::allclose(outNormalizedMaxRadiiAccum, expectedNormalizedMaxRadiiAccum));
    EXPECT_TRUE(torch::allclose(outGradientStepCounts, expectedGradientStepCounts));
}
