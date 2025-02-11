// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <detail/ops/Ops.h>

#include <torch/torch.h>

#include <gtest/gtest.h>

class GaussianTileIntersectionTest : public ::testing::Test {
  protected:
    void
    SetUp() override {
        // Set default test parameters
        num_cameras   = 2;
        num_gaussians = 3;
        tile_size     = 16;
        num_tiles_h   = 4;
        num_tiles_w   = 4;
    }

    // Helper function to create a simple test case
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    createTestData() {
        // Create means2d tensor [C, N, 2]
        auto const means2d = torch::tensor(
            {
                // Camera 0
                { { 16.0, 16.0 },  // At corner between tiles (0,0), (0,1), (1,0), (1,1)
                  { 48.0, 48.0 },  // At corner between tiles (2,2), (2,3), (3,2), (3,3)
                  { 8.0, 8.0 } },  // Center of tile (0,0)
                                   // Camera 1
                { { 24.0, 24.0 },  // Center of tile (1,1)
                  { 32.0, 32.0 },  // At corner between tiles (1,1), (1,2), (2,1), (2,2)
                  { 40.0, 40.0 } } // Center of tile (2,2)
            },
            torch::kFloat32);

        // Create radii tensor [C, N]
        auto const radii = torch::tensor(
            {
                { 8, 8, 8 }, // Camera 0
                { 8, 8, 8 }  // Camera 1
            },
            torch::kInt32);

        // Create depths tensor [C, N]
        auto const depths = torch::tensor(
            {
                { 1.0, 2.0, 0.5 }, // Camera 0
                { 1.5, 2.5, 3.0 }  // Camera 1
            },
            torch::kFloat32);

        // Verify shapes match expected dimensions
        EXPECT_EQ(means2d.sizes(), std::vector<int64_t>({ num_cameras, num_gaussians, 2 }));
        EXPECT_EQ(radii.sizes(), std::vector<int64_t>({ num_cameras, num_gaussians }));
        EXPECT_EQ(depths.sizes(), std::vector<int64_t>({ num_cameras, num_gaussians }));

        return { means2d, radii, depths };
    }

    // Helper to verify depth sorting within tiles
    void
    verifyDepthSorting(const torch::Tensor &tile_offsets, const torch::Tensor &intersection_values,
                       const torch::Tensor &depths) {
        for (int c = 0; c < num_cameras; c++) {
            for (int h = 0; h < num_tiles_h; h++) {
                for (int w = 0; w < num_tiles_w; w++) {
                    int start = tile_offsets[c][h][w].item<int32_t>();
                    int end;
                    if (w < num_tiles_w - 1) {
                        end = tile_offsets[c][h][w + 1].item<int32_t>();
                    } else if (h < num_tiles_h - 1) {
                        end = tile_offsets[c][h + 1][0].item<int32_t>();
                    } else {
                        end = (c < num_cameras - 1) ? tile_offsets[c + 1][0][0].item<int32_t>()
                                                    : intersection_values.size(0);
                    }

                    // Check that depths are monotonically increasing within same camera
                    for (int i = start + 1; i < end; i++) {
                        int global_idx1 = intersection_values[i - 1].item<int32_t>();
                        int global_idx2 = intersection_values[i].item<int32_t>();

                        // Convert global indices to camera and local indices
                        int camera_idx1 = global_idx1 / num_gaussians;
                        int camera_idx2 = global_idx2 / num_gaussians;
                        int local_idx1  = global_idx1 % num_gaussians;
                        int local_idx2  = global_idx2 % num_gaussians;

                        // Only compare depths if both points are from the same camera
                        if (camera_idx1 == camera_idx2) {
                            EXPECT_LE(depths[camera_idx1][local_idx1].item<float>(),
                                      depths[camera_idx2][local_idx2].item<float>())
                                << "Depths not sorted within camera " << camera_idx1
                                << " at local indices " << local_idx1 << " and " << local_idx2;
                        }
                    }
                }
            }
        }
    }

    uint32_t num_cameras;
    uint32_t num_gaussians;
    uint32_t tile_size;
    uint32_t num_tiles_h;
    uint32_t num_tiles_w;
};

TEST_F(GaussianTileIntersectionTest, CPUNotImplementedTest) {
    auto const [means2d, radii, depths] = createTestData();

    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCPU>(
                     means2d, radii, depths,
                     /*camera_jidx=*/at::nullopt, num_cameras, tile_size, num_tiles_h, num_tiles_w),
                 c10::Error);
}

TEST_F(GaussianTileIntersectionTest, ZeroLengthGaussianTest) {
    const auto numCameras = 5;

    // Create empty tensors with correct shapes
    auto means2d = torch::empty({ numCameras, 0, 2 }, torch::kFloat32); // [C, N=0, 2]
    auto radii   = torch::empty({ numCameras, 0 }, torch::kInt32);      // [C. N=0]
    auto depths  = torch::empty({ numCameras, 0 }, torch::kFloat32);    // [C, N=0]

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(
            means2d.cuda(), radii.cuda(), depths.cuda(),
            /*camera_jidx=*/at::nullopt, numCameras, tile_size, num_tiles_h, num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();

    // For zero Gaussians, we should still get a valid tile_offsets tensor filled with zeros
    EXPECT_EQ(tile_offsets.sizes(), std::vector<int64_t>({ numCameras, num_tiles_h, num_tiles_w }));
    EXPECT_TRUE(tile_offsets.equal(torch::zeros_like(tile_offsets)));
    EXPECT_EQ(intersection_values.numel(), 0);
}

TEST_F(GaussianTileIntersectionTest, BadInputsFailTest) {
    const auto nc = 5;
    const auto ng = 3;

    const std::vector<std::vector<std::vector<int64_t>>> configs = {
        // Unpacked:

        // Wrong number of cameras
        { { nc + 1, ng, 2 }, { nc, ng }, { nc, ng } },
        { { nc, ng, 2 }, { nc - 1, ng }, { nc, ng } },
        { { nc, ng, 2 }, { nc, ng }, { nc + 8, ng } },
        // Wrong number of gaussians
        { { nc, ng + 1, 2 }, { nc, ng }, { nc, ng } },
        { { nc, ng, 2 }, { nc, ng - 1 }, { nc, ng } },
        { { nc, ng, 2 }, { nc, ng }, { nc, ng + 8 } },
        // Wrong mean dim
        { { nc, ng, 4 }, { nc, ng }, { nc, ng } },
        // Wrong number of dimensions
        { { nc, 2 }, { nc, 3 }, { nc, 3 } },
        { { nc, 3, 2 }, { nc }, { nc, 3 } },
        { { nc, 3, 2 }, { nc, 3 }, { nc } },

        // Packed:

        // Wrong number of gaussians
        { { ng + 1, 2 }, { ng }, { ng }, { ng } },
        { { ng, 2 }, { ng - 1 }, { ng }, { ng } },
        { { ng, 2 }, { ng }, { ng + 8 }, { ng } },
        { { ng, 2 }, { ng }, { ng }, { ng + 5 } },
        // Wrong mean dim
        { { ng, 4 }, { nc, ng }, { nc, ng }, { ng } },
        // Wrong number of dimensions
        { { ng, 4, 2 }, { ng }, { ng }, { ng } },
        { { ng, 2 }, { ng, 7 }, { ng }, { ng } },
        { { ng, 2 }, { ng }, { ng, 2, 1 }, { ng } },
        { { ng, 2 }, { ng }, { ng, 2, 1 }, { ng, 1 } },
    };

    for (auto config: configs) {
        // Create empty tensors with correct shapes
        auto       means2d  = torch::empty(config[0], torch::kFloat32);
        auto       radii    = torch::empty(config[1], torch::kInt32);
        auto       depths   = torch::empty(config[2], torch::kFloat32);
        const bool isPacked = config.size() == 4;

        if (isPacked) {
            const auto cameraJidx = torch::zeros(config[3], torch::kInt32);
            EXPECT_THROW(fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(
                             means2d.cuda(), radii.cuda(), depths.cuda(), cameraJidx.cuda(), nc,
                             tile_size, num_tiles_h, num_tiles_w),
                         c10::ValueError);

        } else {
            EXPECT_THROW(fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(
                             means2d.cuda(), radii.cuda(), depths.cuda(),
                             /*camera_jidx=*/at::nullopt, nc, tile_size, num_tiles_h, num_tiles_w),
                         c10::ValueError);
        }
    }
}

TEST_F(GaussianTileIntersectionTest, ZeroRadiusGaussianTest) {
    // Create test case with zero-radius Gaussians
    auto [means2d, radii, depths] = createTestData();
    radii.zero_(); // Set all radii to 0

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(
            means2d.cuda(), radii.cuda(), depths.cuda(),
            /*camera_jidx=*/at::nullopt, num_cameras, tile_size, num_tiles_h, num_tiles_w);

    // Verify that there are no intersections
    EXPECT_EQ(tile_offsets.sum().item<int32_t>(), 0);
}

TEST_F(GaussianTileIntersectionTest, BasicIntersectionTest) {
    auto [means2d, radii, depths] = createTestData();

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(
            means2d.cuda(), radii.cuda(), depths.cuda(),
            /*camera_jidx=*/at::nullopt, num_cameras, tile_size, num_tiles_h, num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();

    // Verify dimensions
    EXPECT_EQ(tile_offsets.sizes(),
              std::vector<int64_t>({ num_cameras, num_tiles_h, num_tiles_w }));

    // Verify depth sorting
    verifyDepthSorting(tile_offsets, intersection_values, depths);
}

TEST_F(GaussianTileIntersectionTest, PackedFormatTest) {
    auto [means2d, radii, depths] = createTestData();

    // Reshape tensors to packed format [M, 2] and [M]
    auto means2d_packed = means2d.reshape({ -1, 2 });
    auto radii_packed   = radii.reshape({ -1 });
    auto depths_packed  = depths.reshape({ -1 });

    // Create camera indices tensor
    auto camera_indices =
        torch::arange(num_cameras, torch::kInt32).repeat_interleave(num_gaussians);

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(
            means2d_packed.cuda(), radii_packed.cuda(), depths_packed.cuda(), camera_indices.cuda(),
            num_cameras, tile_size, num_tiles_h, num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();

    // Verify dimensions and basic properties
    EXPECT_EQ(tile_offsets.sizes(),
              std::vector<int64_t>({ num_cameras, num_tiles_h, num_tiles_w }));
    EXPECT_GT(intersection_values.numel(), 0);

    // Verify depth sorting
    verifyDepthSorting(tile_offsets, intersection_values, depths);
}
