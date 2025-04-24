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

struct GaussianRasterizeForwardTestFixture : public ::testing::Test {
    void
    loadInputData(const std::string insPath) {
        const std::string dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const std::string inputsPath = dataPath + std::string("/") + insPath;

        std::vector<torch::Tensor> inputs = fvdb::test::loadTensors(inputsPath, inputNames);
        means2d                           = inputs[0].cuda();
        conics                            = inputs[1].cuda();
        colors                            = inputs[2].cuda();
        opacities                         = inputs[3].cuda();
        tileOffsets                       = inputs[4].cuda();
        tileGaussianIds                   = inputs[5].cuda();

        imageWidth   = 647;
        imageHeight  = 420;
        imageOriginW = 0;
        imageOriginH = 0;
        tileSize     = 16;
    }

    void
    storeData(const std::string outsPath, const std::vector<torch::Tensor> &outputData) {
        const std::string dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const std::string outputPath = dataPath + std::string("/") + outsPath;

        fvdb::test::storeTensors(outputPath, outputData, outputNames);
    }

    void
    loadTestData(const std::string insPath, const std::string outsPath) {
        // Set the random seed for reproducibility.
        torch::manual_seed(0);

        loadInputData(insPath);

        const std::string dataPath =
            std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
        const std::string expectedOutputsPath = dataPath + std::string("/") + outsPath;

        std::vector<torch::Tensor> expectedOutputs =
            fvdb::test::loadTensors(expectedOutputsPath, outputNames);
        expectedRenderedColors = expectedOutputs[0].cuda();
        expectedRenderedAlphas = expectedOutputs[1].cuda();
        expectedLastIds        = expectedOutputs[2].cuda();
    }

    /// @brief Concatenate channels in a color tensor
    /// @param tensor The tensor to concatenate channels in
    /// @param numChannels The number of channels to concatenate
    /// @return The concatenated tensor
    torch::Tensor
    catChannelsToDim(const torch::Tensor &tensor, size_t numChannels) {
        const int64_t lastDim = tensor.dim() - 1;
        TORCH_CHECK(lastDim >= 0, "tensor must have at least one dimension");
        TORCH_CHECK(numChannels >= tensor.size(lastDim),
                    "numChannels must be at least as large as the last dimension of tensor");

        if (numChannels == tensor.size(lastDim)) {
            return tensor;
        }

        std::vector<torch::Tensor> toConcat;
        toConcat.push_back(tensor);

        const auto extraChannels = numChannels - tensor.size(lastDim);
        if (extraChannels > 0) {
            std::vector<int64_t> extraShape = tensor.sizes().vec();
            extraShape[lastDim]             = extraChannels;
            torch::Tensor extraTensor       = torch::zeros(extraShape, tensor.options());
            toConcat.push_back(extraTensor);
        }

        return torch::cat(toConcat, lastDim);
    }

    void
    moveToDevice(const torch::Device &device) {
        means2d                = means2d.to(device);
        conics                 = conics.to(device);
        colors                 = colors.to(device);
        opacities              = opacities.to(device);
        tileOffsets            = tileOffsets.to(device);
        tileGaussianIds        = tileGaussianIds.to(device);
        expectedRenderedColors = expectedRenderedColors.to(device);
        expectedRenderedAlphas = expectedRenderedAlphas.to(device);
        expectedLastIds        = expectedLastIds.to(device);
    }

    const std::vector<std::string> inputNames = {
        "means2d", "conics", "colors", "opacities", "tile_offsets", "tile_gaussian_ids"
    };
    const std::vector<std::string> outputNames = { "rendered_colors", "rendered_alphas",
                                                   "last_ids" };

    // Input tensors
    torch::Tensor means2d;         // [C, N, 2] or [nnz, 2]
    torch::Tensor conics;          // [C, N, 3] or [nnz, 3]
    torch::Tensor colors;          // [C, N, D] or [nnz, D]
    torch::Tensor opacities;       // [C, N] or [nnz]
    torch::Tensor tileOffsets;     // [C, tileHeight, tileWidth]
    torch::Tensor tileGaussianIds; // [nIsects]

    // Expected output tensors
    torch::Tensor expectedRenderedColors; // [C, imageHeight, imageWidth, D]
    torch::Tensor expectedRenderedAlphas; // [C, imageHeight, imageWidth, 1]
    torch::Tensor expectedLastIds;        // [C, imageHeight, imageWidth]

    // Parameters
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t imageOriginW;
    uint32_t imageOriginH;
    uint32_t tileSize;
};

// This is a helper function to generate the output data for the test cases.
// Only enable this test when you want to update the output data.
TEST_F(GaussianRasterizeForwardTestFixture, DISABLED_GenerateOutputData) {
    // Load test data using our helper method
    loadInputData("rasterize_forward_inputs.pt");

    // Test with 3 channels
    {
        const auto [renderedColors, renderedAlphas, lastIds] =
            fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(
                means2d, conics, colors, opacities, imageWidth, imageHeight, imageOriginW,
                imageOriginH, tileSize, tileOffsets, tileGaussianIds);

        std::vector<torch::Tensor> outputData = { renderedColors, renderedAlphas, lastIds };

        auto outputFilename = std::string("rasterize_forward_outputs.pt");

        storeData(outputFilename, outputData);
    }

    // Test with 64 channels
    {
        auto colors_64 = catChannelsToDim(colors, 64);

        const auto [renderedColors, renderedAlphas, lastIds] =
            fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(
                means2d, conics, colors_64, opacities, imageWidth, imageHeight, imageOriginW,
                imageOriginH, tileSize, tileOffsets, tileGaussianIds);

        std::vector<torch::Tensor> outputData = { renderedColors, renderedAlphas, lastIds };

        auto outputFilename = std::string("rasterize_forward_outputs_64.pt");
        storeData(outputFilename, outputData);
    }
}

TEST_F(GaussianRasterizeForwardTestFixture, TestBasicInputsAndOutputs) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs.pt");

    const auto [outColors, outAlphas, outLastIds] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(
            means2d, conics, colors, opacities, imageWidth, imageHeight, imageOriginW, imageOriginH,
            tileSize, tileOffsets, tileGaussianIds);

    EXPECT_TRUE(torch::allclose(outColors, expectedRenderedColors));
    EXPECT_TRUE(torch::allclose(outAlphas, expectedRenderedAlphas));
    EXPECT_TRUE(torch::equal(outLastIds, expectedLastIds));
}

TEST_F(GaussianRasterizeForwardTestFixture, TestConcatenatedChannels) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs_64.pt");

    colors                 = catChannelsToDim(colors, 64);
    expectedRenderedColors = catChannelsToDim(expectedRenderedColors, 64);

    const auto [outColors, outAlphas, outLastIds] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(
            means2d, conics, colors, opacities, imageWidth, imageHeight, imageOriginW, imageOriginH,
            tileSize, tileOffsets, tileGaussianIds);

    EXPECT_TRUE(torch::allclose(outColors, expectedRenderedColors));
    EXPECT_TRUE(torch::allclose(outAlphas, expectedRenderedAlphas));
    EXPECT_TRUE(torch::equal(outLastIds, expectedLastIds));
}

TEST_F(GaussianRasterizeForwardTestFixture, CPUThrows) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs.pt");
    moveToDevice(torch::kCPU);
    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCPU>(
                     means2d, conics, colors, opacities, imageWidth, imageHeight, imageOriginW,
                     imageOriginH, tileSize, tileOffsets, tileGaussianIds),
                 c10::NotImplementedError);
}
