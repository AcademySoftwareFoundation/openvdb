// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <detail/ops/Ops.h>

#include <torch/torch.h>

#include <gtest/gtest.h>
#include <tests/utils/Tensor.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>

#ifndef FVDB_EXTERNAL_TEST_DATA_PATH
#error "FVDB_EXTERNAL_TEST_DATA_PATH must be defined"
#endif

struct GaussianRasterizeTestFixture : public ::testing::Test {
    void
    loadTestData(const std::string insPath, const std::string outsPath) {
        // Set the random seed for reproducibility.
        torch::manual_seed(0);

        const std::string dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const std::string inputsPath          = dataPath + std::string("/") + insPath;
        const std::string expectedOutputsPath = dataPath + std::string("/") + outsPath;

        std::vector<torch::Tensor> inputs;
        torch::load(inputs, inputsPath);
        means2d                 = inputs[0].cuda();
        conics                  = inputs[1].cuda();
        colors                  = inputs[2].cuda();
        opacities               = inputs[3].cuda();
        tileOffsets             = inputs[4].cuda();
        tileGaussianIds         = inputs[5].cuda();
        renderedAlphas          = inputs[6].cuda();
        lastGaussianIdsPerPixel = inputs[7].cuda();
        dLossDRenderedColors    = inputs[8].cuda();
        dLossDRenderedAlphas    = inputs[9].cuda();

        std::vector<torch::Tensor> expectedOutputs;
        torch::load(expectedOutputs, expectedOutputsPath);
        expectedDLossDMeans2d   = expectedOutputs[0].cuda();
        expectedDLossDConics    = expectedOutputs[1].cuda();
        expectedDLossDColors    = expectedOutputs[2].cuda();
        expectedDLossDOpacities = expectedOutputs[3].cuda();

        imageWidth   = 1297;
        imageHeight  = 840;
        imageOriginW = 0;
        imageOriginH = 0;
        tileSize     = 16;
    }

    void
    SetUp() override {}

    torch::Tensor
    catChannelsToDim(const torch::Tensor &input, int64_t numOutChannels) {
        using namespace at::indexing;

        const int64_t numInChannels = input.size(-1);

        EXPECT_TRUE(numInChannels <= numOutChannels);

        if (numInChannels == numOutChannels) {
            return input;
        }

        std::vector<torch::Tensor> tensorsToCat;
        for (int i = 0; i < numOutChannels / numInChannels; i += 1) {
            tensorsToCat.push_back(input);
        }
        tensorsToCat.push_back(input.index({ Ellipsis, Slice(0, numOutChannels % numInChannels) }));
        torch::Tensor ret = torch::cat(tensorsToCat, -1);
        return ret;
    }

    void
    moveToDevice(torch::Device device) {
        means2d                 = means2d.to(device);
        conics                  = conics.to(device);
        colors                  = colors.to(device);
        opacities               = opacities.to(device);
        tileOffsets             = tileOffsets.to(device);
        tileGaussianIds         = tileGaussianIds.to(device);
        renderedAlphas          = renderedAlphas.to(device);
        lastGaussianIdsPerPixel = lastGaussianIdsPerPixel.to(device);
        dLossDRenderedColors    = dLossDRenderedColors.to(device);
        dLossDRenderedAlphas    = dLossDRenderedAlphas.to(device);
    }

    uint32_t imageWidth   = 1297;
    uint32_t imageHeight  = 840;
    uint32_t imageOriginW = 0;
    uint32_t imageOriginH = 0;
    uint32_t tileSize     = 16;

    torch::Tensor means2d;
    torch::Tensor conics;
    torch::Tensor colors;
    torch::Tensor opacities;
    torch::Tensor tileOffsets;
    torch::Tensor tileGaussianIds;
    torch::Tensor renderedAlphas;
    torch::Tensor lastGaussianIdsPerPixel;
    torch::Tensor dLossDRenderedColors;
    torch::Tensor dLossDRenderedAlphas;

    torch::Tensor expectedDLossDMeans2d;
    torch::Tensor expectedDLossDConics;
    torch::Tensor expectedDLossDColors;
    torch::Tensor expectedDLossDOpacities;
};

TEST_F(GaussianRasterizeTestFixture, TestBasicInputsAndOutputs) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs.pt");

    const auto rasterizeBackwardOutputs =
        fvdb::detail::ops::dispatchGaussianRasterizeBackward<torch::kCUDA>(
            means2d, conics, colors, opacities, imageWidth, imageHeight, imageOriginW, imageOriginH,
            tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel,
            dLossDRenderedColors, dLossDRenderedAlphas, false);

    EXPECT_TRUE(torch::allclose(std::get<1>(rasterizeBackwardOutputs), expectedDLossDMeans2d));

    // This is a big sum of products in parallel that is pretty ill conditioned and so we
    // only expect about 1 digit of accuracy.
    EXPECT_TRUE(torch::allclose(std::get<2>(rasterizeBackwardOutputs), expectedDLossDConics,
                                1e-1 /*rtol*/));

    EXPECT_TRUE(torch::allclose(std::get<3>(rasterizeBackwardOutputs), expectedDLossDColors));
    EXPECT_TRUE(
        torch::allclose(std::get<4>(rasterizeBackwardOutputs), expectedDLossDOpacities, 1e-4));
}

TEST_F(GaussianRasterizeTestFixture, TestConcatenatedChannels) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs_64.pt");

    colors               = catChannelsToDim(colors, 64);
    dLossDRenderedColors = catChannelsToDim(dLossDRenderedColors, 64);
    expectedDLossDColors = catChannelsToDim(expectedDLossDColors, 64);

    const auto rasterizeBackwardOutputs =
        fvdb::detail::ops::dispatchGaussianRasterizeBackward<torch::kCUDA>(
            means2d, conics, colors, opacities, imageWidth, imageHeight, imageOriginW, imageOriginH,
            tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel,
            dLossDRenderedColors, dLossDRenderedAlphas, false);

    const torch::Tensor dLossDMeans2d   = std::get<1>(rasterizeBackwardOutputs);
    const torch::Tensor dLossDConics    = std::get<2>(rasterizeBackwardOutputs);
    const torch::Tensor dLossDColors    = std::get<3>(rasterizeBackwardOutputs);
    const torch::Tensor dLossDOpacities = std::get<4>(rasterizeBackwardOutputs);

    EXPECT_TRUE(torch::allclose(std::get<1>(rasterizeBackwardOutputs), expectedDLossDMeans2d));
    EXPECT_TRUE(torch::allclose(std::get<3>(rasterizeBackwardOutputs), expectedDLossDColors));
    EXPECT_TRUE(
        torch::allclose(std::get<4>(rasterizeBackwardOutputs), expectedDLossDOpacities, 1e-4));
}

TEST_F(GaussianRasterizeTestFixture, TestConcatenatedChunkedChannelsWithUnusedChannels) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs_47.pt");

    colors               = catChannelsToDim(colors, 47);
    dLossDRenderedColors = catChannelsToDim(dLossDRenderedColors, 47);

    const auto rasterizeBackwardOutputs =
        fvdb::detail::ops::dispatchGaussianRasterizeBackward<torch::kCUDA>(
            means2d, conics, colors, opacities, imageWidth, imageHeight, imageOriginW, imageOriginH,
            tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel,
            dLossDRenderedColors, dLossDRenderedAlphas, false, 32);

    const torch::Tensor dLossDMeans2d   = std::get<1>(rasterizeBackwardOutputs);
    const torch::Tensor dLossDConics    = std::get<2>(rasterizeBackwardOutputs);
    const torch::Tensor dLossDColors    = std::get<3>(rasterizeBackwardOutputs);
    const torch::Tensor dLossDOpacities = std::get<4>(rasterizeBackwardOutputs);

    EXPECT_TRUE(torch::allclose(std::get<1>(rasterizeBackwardOutputs), expectedDLossDMeans2d));
    EXPECT_TRUE(torch::allclose(std::get<3>(rasterizeBackwardOutputs), expectedDLossDColors));
    EXPECT_TRUE(
        torch::allclose(std::get<4>(rasterizeBackwardOutputs), expectedDLossDOpacities, 1e-4));
}

TEST_F(GaussianRasterizeTestFixture, TestChunkedChannels) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs_64.pt");

    colors               = catChannelsToDim(colors, 64);
    dLossDRenderedColors = catChannelsToDim(dLossDRenderedColors, 64);
    expectedDLossDColors = catChannelsToDim(expectedDLossDColors, 64);

    const auto rasterizeBackwardOutputs =
        fvdb::detail::ops::dispatchGaussianRasterizeBackward<torch::kCUDA>(
            means2d, conics, colors, opacities, imageWidth, imageHeight, imageOriginW, imageOriginH,
            tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel,
            dLossDRenderedColors, dLossDRenderedAlphas, false, 32);

    const torch::Tensor dLossDMeans2d   = std::get<1>(rasterizeBackwardOutputs);
    const torch::Tensor dLossDConics    = std::get<2>(rasterizeBackwardOutputs);
    const torch::Tensor dLossDColors    = std::get<3>(rasterizeBackwardOutputs);
    const torch::Tensor dLossDOpacities = std::get<4>(rasterizeBackwardOutputs);

    EXPECT_TRUE(torch::allclose(std::get<1>(rasterizeBackwardOutputs), expectedDLossDMeans2d));
    EXPECT_TRUE(torch::allclose(std::get<3>(rasterizeBackwardOutputs), expectedDLossDColors));
    EXPECT_TRUE(
        torch::allclose(std::get<4>(rasterizeBackwardOutputs), expectedDLossDOpacities, 1e-4));
}

TEST_F(GaussianRasterizeTestFixture, CPUThrows) {
    loadTestData("rasterize_backward_inputs.pt", "rasterize_backward_outputs.pt");
    moveToDevice(torch::kCPU);
    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRasterizeBackward<torch::kCPU>(
                     means2d, conics, colors, opacities, imageWidth, imageHeight, imageOriginW,
                     imageOriginH, tileSize, tileOffsets, tileGaussianIds, renderedAlphas,
                     lastGaussianIdsPerPixel, dLossDRenderedColors, dLossDRenderedAlphas, false),
                 c10::NotImplementedError);
}

// NOTE: This is called as a backward pass so the forward pass will handle most of the error
//       checking. We just need to test that the backward pass doesn't throw.
// TODO: Test empty inputs
// TODO: Test error inputs
// TODO: Test with backgrounds
