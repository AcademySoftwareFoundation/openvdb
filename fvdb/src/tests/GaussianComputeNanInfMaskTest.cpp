// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <detail/ops/Ops.h>

#include <torch/torch.h>

#include <gtest/gtest.h>
#include <tests/utils/Tensor.h>

#include <cstddef>
#include <cstdlib>

class NanInfMaskTestFixture : public ::testing::TestWithParam<int> {};

TEST(NanInfMaskTests, TestEmptyGaussians) {
    int64_t const numGaussians = 0;

    auto const floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    // means, quats, scales, opacities, sh
    auto const means     = torch::rand({ numGaussians, 3 }, floatOptsCUDA);
    auto const quats     = torch::rand({ numGaussians, 4 }, floatOptsCUDA);
    auto const scales    = torch::rand({ numGaussians, 3 }, floatOptsCUDA);
    auto const opacities = torch::rand({ numGaussians }, floatOptsCUDA);
    auto const sh        = torch::rand({ numGaussians, 27, 3 }, floatOptsCUDA);

    auto mask = fvdb::detail::ops::dispatchGaussianNanInfMask<torch::kCUDA>(means, quats, scales,
                                                                            opacities, sh);

    EXPECT_TRUE(mask.jdata().numel() == 0);
    EXPECT_TRUE(mask.jdata().is_cuda());
}

TEST(NanInfMaskTests, TestExceptionForInconsistentGaussians) {
    auto const floatOptsCUDA = fvdb::test::tensorOpts<float>(torch::kCUDA);

    // number of gaussians for means, quats, scales, opacities, sh
    int64_t const                       ng = 10; // base number of gaussians
    std::vector<std::vector<int>> const configs{ { ng - 2, ng, ng, ng, ng },
                                                 { ng, ng + 1, ng, ng, ng },
                                                 { ng, ng, ng + 5, ng, ng },
                                                 { ng, ng, ng, ng - 5, ng },
                                                 { ng, ng, ng, ng, ng + 1 } };

    for (auto const &config: configs) {
        auto const means     = torch::rand({ config[0], 3 }, floatOptsCUDA);
        auto const quats     = torch::rand({ config[1], 4 }, floatOptsCUDA);
        auto const scales    = torch::rand({ config[2], 3 }, floatOptsCUDA);
        auto const opacities = torch::rand({ config[3] }, floatOptsCUDA);
        auto const sh        = torch::rand({ config[4], 27, 3 }, floatOptsCUDA);

        EXPECT_THROW(fvdb::detail::ops::dispatchGaussianNanInfMask<torch::kCUDA>(
                         means, quats, scales, opacities, sh),
                     c10::ValueError);
    }
}

TEST_P(NanInfMaskTestFixture, TestNanInfMaskMeansNan) {
    std::size_t batchSize = 10;

    auto const floatOptsCPU = fvdb::test::tensorOpts<float>(torch::kCPU);
    auto const boolOptsCPU  = fvdb::test::tensorOpts<bool>(torch::kCPU);

    auto whichTensorHasNans = GetParam();

    std::vector<int64_t> badIndices;

    std::vector<torch::Tensor> meansVec;
    std::vector<torch::Tensor> quatsVec;
    std::vector<torch::Tensor> scalesVec;
    std::vector<torch::Tensor> opacitiesVec;
    std::vector<torch::Tensor> shVec;

    std::int64_t totalElements = 0;
    for (auto i = 0; i < batchSize; i += 1) {
        auto const numGaussians = 10 + (std::rand() % 100);

        // means, quats, scales, opacities, sh
        std::vector<torch::Tensor> parameters{ torch::rand({ numGaussians, 3 }, floatOptsCPU),
                                               torch::rand({ numGaussians, 4 }, floatOptsCPU),
                                               torch::rand({ numGaussians, 3 }, floatOptsCPU),
                                               torch::rand({ numGaussians, 1 }, floatOptsCPU),
                                               torch::rand({ numGaussians, 27, 3 }, floatOptsCPU) };

        // -1 means no tensor has nans or infs
        if (whichTensorHasNans > 0) {
            torch::Tensor tensorToCorrupt = parameters[whichTensorHasNans];
            for (auto j = 0; j < tensorToCorrupt.size(1); j += 1) {
                auto const randomGaussianIndex = std::rand() % tensorToCorrupt.size(0);
                auto const numNans             = std::rand() % tensorToCorrupt.size(1);
                for (auto k = 0; k < numNans; k += 1) {
                    auto const randomIndex = std::rand() % tensorToCorrupt.size(1);

                    if (std::rand() % 2 == 0) {
                        tensorToCorrupt[randomGaussianIndex][randomIndex] =
                            std::numeric_limits<float>::quiet_NaN();
                    } else {
                        tensorToCorrupt[randomGaussianIndex][randomIndex] =
                            std::numeric_limits<float>::infinity();
                    }
                }
                if (numNans > 0) {
                    badIndices.push_back(totalElements + randomGaussianIndex);
                }
            }
        }

        meansVec.push_back(parameters[0]);
        quatsVec.push_back(parameters[1]);
        scalesVec.push_back(parameters[2]);
        opacitiesVec.push_back(parameters[3].squeeze());
        shVec.push_back(parameters[4]);

        totalElements += numGaussians;
    }

    auto const expectedMask = torch::ones({ totalElements }, boolOptsCPU);
    for (auto i = 0; i < badIndices.size(); i += 1) {
        expectedMask[badIndices[i]] = false;
    }

    auto const meansJT     = fvdb::JaggedTensor(meansVec).to(torch::kCUDA);
    auto const quatsJT     = fvdb::JaggedTensor(quatsVec).to(torch::kCUDA);
    auto const scalesJT    = fvdb::JaggedTensor(scalesVec).to(torch::kCUDA);
    auto const opacitiesJT = fvdb::JaggedTensor(opacitiesVec).to(torch::kCUDA);
    auto const shJT        = fvdb::JaggedTensor(shVec).to(torch::kCUDA);

    auto const mask = fvdb::detail::ops::dispatchGaussianNanInfMask<torch::kCUDA>(
        meansJT, quatsJT, scalesJT, opacitiesJT, shJT);

    EXPECT_TRUE(torch::equal(expectedMask.to(torch::kCUDA), mask.jdata()));
}

INSTANTIATE_TEST_CASE_P(NanInfMaskTests, NanInfMaskTestFixture,
                        ::testing::Values(-1, 0, 1, 2, 3, 4));
