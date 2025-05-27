// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <detail/ops/Ops.h>
#include <tests/utils/ImageUtils.h>
#include <tests/utils/Tensor.h>

#include <torch/script.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

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
    catChannelsToDim(const torch::Tensor &tensor, int numChannels) {
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
        "means2d", "conics", "colors", "opacities", "tile_offsets", "tile_gaussian_ids"};
    const std::vector<std::string> outputNames = {"rendered_colors", "rendered_alphas", "last_ids"};

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
            fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                              conics,
                                                                              colors,
                                                                              opacities,
                                                                              imageWidth,
                                                                              imageHeight,
                                                                              imageOriginW,
                                                                              imageOriginH,
                                                                              tileSize,
                                                                              tileOffsets,
                                                                              tileGaussianIds);

        std::vector<torch::Tensor> outputData = {renderedColors, renderedAlphas, lastIds};

        auto outputFilename = std::string("rasterize_forward_outputs.pt");

        storeData(outputFilename, outputData);
    }

    // Test with 64 channels
    {
        auto colors_64 = catChannelsToDim(colors, 64);

        const auto [renderedColors, renderedAlphas, lastIds] =
            fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                              conics,
                                                                              colors_64,
                                                                              opacities,
                                                                              imageWidth,
                                                                              imageHeight,
                                                                              imageOriginW,
                                                                              imageOriginH,
                                                                              tileSize,
                                                                              tileOffsets,
                                                                              tileGaussianIds);

        std::vector<torch::Tensor> outputData = {renderedColors, renderedAlphas, lastIds};

        auto outputFilename = std::string("rasterize_forward_outputs_64.pt");
        storeData(outputFilename, outputData);
    }
}

TEST_F(GaussianRasterizeForwardTestFixture, TestBasicInputsAndOutputs) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs.pt");

    const auto [outColors, outAlphas, outLastIds] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                          conics,
                                                                          colors,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds);

    EXPECT_TRUE(torch::allclose(outColors, expectedRenderedColors));
    EXPECT_TRUE(torch::allclose(outAlphas, expectedRenderedAlphas));
    EXPECT_TRUE(torch::equal(outLastIds, expectedLastIds));
}

TEST_F(GaussianRasterizeForwardTestFixture, TestConcatenatedChannels) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs_64.pt");

    colors                 = catChannelsToDim(colors, 64);
    expectedRenderedColors = catChannelsToDim(expectedRenderedColors, 64);

    const auto [outColors, outAlphas, outLastIds] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                          conics,
                                                                          colors,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds);

    EXPECT_TRUE(torch::allclose(outColors, expectedRenderedColors));
    EXPECT_TRUE(torch::allclose(outAlphas, expectedRenderedAlphas));
    EXPECT_TRUE(torch::equal(outLastIds, expectedLastIds));
}

// Compares the output of multi-camera rasterization with the output of sequentialsingle-camera
// rasterization.
TEST_F(GaussianRasterizeForwardTestFixture, TestMultipleCameras) {
    // the output here is not used in this test.
    loadTestData("rasterize_forward_inputs_3cams.pt", "rasterize_forward_outputs.pt");

    // run all 3 cameras at once
    const auto [outColorsAll, outAlphasAll, outLastIdsAll] =
        fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d,
                                                                          conics,
                                                                          colors,
                                                                          opacities,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          tileOffsets,
                                                                          tileGaussianIds);

    // rasterize each camera individually
    std::vector<torch::Tensor> outColorsList;
    std::vector<torch::Tensor> outAlphasList;
    std::vector<torch::Tensor> outLastIdsList;
    for (int i = 0; i < means2d.size(0); i++) {
        // extract the ith camera data from each tensor and add a leading dim of 1
        auto means2d_1cam     = means2d[i].unsqueeze(0);
        auto conics_1cam      = conics[i].unsqueeze(0);
        auto colors_1cam      = colors[i].unsqueeze(0);
        auto opacities_1cam   = opacities[i].unsqueeze(0);
        auto tileOffsets_1cam = tileOffsets[i].unsqueeze(0);

        auto numCameras = means2d.size(0);

        // find the start and end of the ith camera in tileGaussianIds
        auto start = tileOffsets[i][0][0].item<int64_t>();
        auto end   = i == numCameras - 1 ? tileGaussianIds.numel()
                                         : tileOffsets[i + 1][0][0].item<int64_t>();

        // slice out this camera's tileGaussianIds and adjust to 0-based
        auto tileGaussianIds_1cam =
            tileGaussianIds.index({torch::indexing::Slice(start, end)}) - i * means2d.size(1);

        // Adjust the tileOffsets to be 0-based
        tileOffsets_1cam = tileOffsets_1cam - start;

        // Kernel receives adjusted offsets and 0-based IDs for this camera
        auto [outColors, outAlphas, outLastIds] =
            fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCUDA>(means2d_1cam,
                                                                              conics_1cam,
                                                                              colors_1cam,
                                                                              opacities_1cam,
                                                                              imageWidth,
                                                                              imageHeight,
                                                                              imageOriginW,
                                                                              imageOriginH,
                                                                              tileSize,
                                                                              tileOffsets_1cam,
                                                                              tileGaussianIds_1cam);

        // add start offset back to non-background pixels
        outLastIds = outLastIds + start;
        // mask out the background pixels
        auto background_mask = (outLastIdsAll[i].unsqueeze(0) == -1);
        outLastIds.masked_fill_(background_mask, -1);

        outColorsList.push_back(outColors);
        outAlphasList.push_back(outAlphas);
        outLastIdsList.push_back(outLastIds); // Store the raw output index

// Uncomment to dump binarized lastIds images for comparison
#if 0
        // binarize the outLastIds tensor, stack it to a 3-channel image and write a PNG
        auto binarizedLastIds = (outLastIds[0] == -1);

        auto binarizedLastIds_3ch =
            torch::stack({ binarizedLastIds, binarizedLastIds, binarizedLastIds }, -1)
                .to(torch::kFloat32);
        auto alpha = torch::ones_like(outAlphas[0]);
        // print shape of binarizedLastIds_3ch
        std::cout << "binarizedLastIds_3ch shape: " << binarizedLastIds_3ch.sizes() << std::endl;
        fvdb::test::writePNG(binarizedLastIds_3ch.cpu(), alpha.cpu(),
                             "test_output_camera_" + std::to_string(i) + "_binarized.png");

        // also write the i-th 3-camera image binarized
        auto binarizedLastIdsAll = (outLastIdsAll[i] == -1);
        auto binarizedLastIdsAll_3ch =
            torch::stack({ binarizedLastIdsAll, binarizedLastIdsAll, binarizedLastIdsAll }, -1)
                .to(torch::kFloat32);
        std::cout << "binarizedLastIdsAll_3ch shape: " << binarizedLastIdsAll_3ch.sizes()
                  << std::endl;
        fvdb::test::writePNG(binarizedLastIdsAll_3ch.cpu(), alpha.cpu(),
                             "test_output_camera_" + std::to_string(i) + "_binarized_all.png");

        // compute a difference image between the two
        auto diff = binarizedLastIds_3ch - binarizedLastIdsAll_3ch;
        std::cout << "diff shape: " << diff.sizes() << std::endl;
        fvdb::test::writePNG(diff.cpu(), alpha.cpu(),
                             "test_output_camera_" + std::to_string(i) + "_diff.png");
#endif

// Uncomment to dump color images for comparison
#if 0
        // write out the ith camera's image
        fvdb::test::writePNG(outColors[0].cpu(), alpha.cpu(), // outAlphas[0].cpu(),
                             "test_output_camera_" + std::to_string(i) + ".png");
        // write out the three images from the single-pass rasterization
        fvdb::test::writePNG(outColorsAll[i].cpu(), alpha.cpu(), // outAlphasAll[i].cpu(),
                             "test_output_all_cameras_" + std::to_string(i) + ".png");
#endif
    }

    auto combinedColors  = torch::cat(outColorsList, 0);
    auto combinedAlphas  = torch::cat(outAlphasList, 0);
    auto combinedLastIds = torch::cat(outLastIdsList, 0);

    EXPECT_TRUE(torch::allclose(combinedColors, outColorsAll));
    EXPECT_TRUE(torch::allclose(combinedAlphas, outAlphasAll));
    EXPECT_TRUE(torch::equal(combinedLastIds, outLastIdsAll));
}

TEST_F(GaussianRasterizeForwardTestFixture, CPUThrows) {
    loadTestData("rasterize_forward_inputs.pt", "rasterize_forward_outputs.pt");
    moveToDevice(torch::kCPU);
    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianRasterizeForward<torch::kCPU>(means2d,
                                                                                  conics,
                                                                                  colors,
                                                                                  opacities,
                                                                                  imageWidth,
                                                                                  imageHeight,
                                                                                  imageOriginW,
                                                                                  imageOriginH,
                                                                                  tileSize,
                                                                                  tileOffsets,
                                                                                  tileGaussianIds),
                 c10::NotImplementedError);
}
