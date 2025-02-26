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

TEST(GaussianRasterizeBackwardTests, TestBasicInputsAndOutputs) {
    const std::string dataPath =
        std::string(FVDB_EXTERNAL_TEST_DATA_PATH) + std::string("/unit_tests/gsplat");
    const std::string inputsPath = dataPath + std::string("/rasterize_backward_inputs.pt");
    const std::string expectedOutputsPath =
        dataPath + std::string("/rasterize_backward_outputs.pt");

    std::vector<torch::Tensor> inputs;
    std::vector<torch::Tensor> expectedOutputs;
    torch::load(inputs, inputsPath);
    torch::load(expectedOutputs, expectedOutputsPath);

    const torch::Tensor means2d                 = inputs[0].cuda();
    const torch::Tensor conics                  = inputs[1].cuda();
    const torch::Tensor colors                  = inputs[2].cuda();
    const torch::Tensor opacities               = inputs[3].cuda();
    const torch::Tensor tileOffsets             = inputs[4].cuda();
    const torch::Tensor tileGaussianIds         = inputs[5].cuda();
    const torch::Tensor renderedAlphas          = inputs[6].cuda();
    const torch::Tensor lastGaussianIdsPerPixel = inputs[7].cuda();
    const torch::Tensor dLossDRenderedColors    = inputs[8].cuda();
    const torch::Tensor dLossDRenderedAlphas    = inputs[9].cuda();

    const torch::Tensor expectedDLossDMeans2d   = expectedOutputs[0].cuda();
    const torch::Tensor expectedDLossDConics    = expectedOutputs[1].cuda();
    const torch::Tensor expectedDLossDColors    = expectedOutputs[2].cuda();
    const torch::Tensor expectedDLossDOpacities = expectedOutputs[3].cuda();

    const uint32_t imageWidth   = 1297;
    const uint32_t imageHeight  = 840;
    const uint32_t imageOriginW = 0;
    const uint32_t imageOriginH = 0;
    const uint32_t tileSize     = 16;

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
    EXPECT_TRUE(torch::allclose(std::get<4>(rasterizeBackwardOutputs), expectedDLossDOpacities));
}
