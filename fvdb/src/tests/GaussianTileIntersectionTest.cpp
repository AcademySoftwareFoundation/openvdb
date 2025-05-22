// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <detail/ops/Ops.h>
#include <detail/ops/gsplat/GaussianSplatSparse.h>

#include <torch/torch.h>

#include <gtest/gtest.h>

class GaussianTileIntersectionTest : public ::testing::Test {
  protected:
    void
    SetUp() override {
        torch::manual_seed(0);

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
                {{16.0, 16.0}, // At corner between tiles (0,0), (0,1), (1,0), (1,1)
                 {48.0, 48.0}, // At corner between tiles (2,2), (2,3), (3,2), (3,3)
                 {8.0, 8.0}},  // Center of tile (0,0)
                               // Camera 1
                {{24.0, 24.0}, // Center of tile (1,1)
                 {32.0, 32.0}, // At corner between tiles (1,1), (1,2), (2,1), (2,2)
                 {40.0, 40.0}} // Center of tile (2,2)
            },
            torch::kFloat32);

        // Create radii tensor [C, N]
        auto const radii = torch::tensor(
            {
                {8, 8, 8}, // Camera 0
                {8, 8, 8}  // Camera 1
            },
            torch::kInt32);

        // Create depths tensor [C, N]
        auto const depths = torch::tensor(
            {
                {1.0, 2.0, 0.5}, // Camera 0
                {1.5, 2.5, 3.0}  // Camera 1
            },
            torch::kFloat32);

        // Verify shapes match expected dimensions
        EXPECT_EQ(means2d.sizes(), std::vector<int64_t>({num_cameras, num_gaussians, 2}));
        EXPECT_EQ(radii.sizes(), std::vector<int64_t>({num_cameras, num_gaussians}));
        EXPECT_EQ(depths.sizes(), std::vector<int64_t>({num_cameras, num_gaussians}));

        return {means2d, radii, depths};
    }

    // Helper to verify depth sorting within tiles for both dense and sparse formats
    void
    verifyDepthSorting(const torch::Tensor &tile_offsets,
                       const torch::Tensor &intersection_values,
                       const torch::Tensor &depths,
                       const torch::Tensor &tile_mask = torch::Tensor()) {
        // Handle dense case - iterate through 3D tensor
        if (tile_offsets.dim() == 3) {
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
                        verifyTileDepthOrder(start, end, intersection_values, depths);
                    }
                }
            }
        }
        // Handle sparse case - iterate through 1D tensor
        else {
            for (int i = 0; i < tile_offsets.size(0); i++) {
                int start = tile_offsets[i].item<int32_t>();
                int end   = (i < tile_offsets.size(0) - 1) ? tile_offsets[i + 1].item<int32_t>()
                                                           : intersection_values.size(0);
                verifyTileDepthOrder(start, end, intersection_values, depths);
            }
        }
    }

    // Helper to verify depth order within a single tile's range
    void
    verifyTileDepthOrder(int start,
                         int end,
                         const torch::Tensor &intersection_values,
                         const torch::Tensor &depths) {
        for (int i = start + 1; i < end; i++) {
            int global_idx1 = intersection_values[i - 1].item<int32_t>();
            int global_idx2 = intersection_values[i].item<int32_t>();

            int camera_idx1 = global_idx1 / num_gaussians;
            int camera_idx2 = global_idx2 / num_gaussians;
            int local_idx1  = global_idx1 % num_gaussians;
            int local_idx2  = global_idx2 % num_gaussians;

            // Only compare depths if both points are from the same camera
            if (camera_idx1 == camera_idx2) {
                EXPECT_LE(depths[camera_idx1][local_idx1].item<float>(),
                          depths[camera_idx2][local_idx2].item<float>())
                    << "Depths not sorted within camera " << camera_idx1 << " at local indices "
                    << local_idx1 << " and " << local_idx2;
            }
        }
    }

    // Helper function to verify tile intersections and their ordering
    void
    verifyTileIntersections(const torch::Tensor &tile_offsets,
                            const torch::Tensor &intersection_values,
                            const torch::Tensor &means2d,
                            const torch::Tensor &radii,
                            const torch::Tensor &depths,
                            const torch::Tensor &active_tiles) {
        // Generate expected output by computing intersections for each active tile
        std::vector<std::vector<int32_t>> expected_intersections;
        int32_t total_expected = 0;

        auto num_active_tiles = active_tiles.size(0);

        // For each active tile, find all gaussians that intersect it
        for (int64_t i = 0; i < num_active_tiles; i++) {
            auto active_idx = active_tiles[i].item<int32_t>();
            int32_t cam_idx = active_idx / (num_tiles_h * num_tiles_w);
            int32_t tile_h  = (active_idx % (num_tiles_h * num_tiles_w)) / num_tiles_w;
            int32_t tile_w  = active_idx % num_tiles_w;

            std::vector<int32_t> tile_intersections;

            // Check each gaussian in this camera for intersection
            for (int64_t g = 0; g < num_gaussians; g++) {
                float x   = means2d[cam_idx][g][0].item<float>();
                float y   = means2d[cam_idx][g][1].item<float>();
                int32_t r = radii[cam_idx][g].item<int32_t>();

                // Compute tile bounds
                float tile_left   = tile_w * tile_size;
                float tile_right  = (tile_w + 1) * tile_size;
                float tile_bottom = tile_h * tile_size;
                float tile_top    = (tile_h + 1) * tile_size;

                // Check if gaussian intersects tile bounds
                bool intersects = x + r > tile_left && x - r < tile_right && y + r > tile_bottom &&
                                  y - r < tile_top;

                if (intersects) {
                    tile_intersections.push_back(cam_idx * num_gaussians + g);
                }
            }

            // Sort intersections by depth
            std::sort(
                tile_intersections.begin(), tile_intersections.end(), [&](int32_t a, int32_t b) {
                    int32_t cam_a = a / num_gaussians;
                    int32_t cam_b = b / num_gaussians;
                    int32_t idx_a = a % num_gaussians;
                    int32_t idx_b = b % num_gaussians;
                    if (cam_a != cam_b)
                        return cam_a < cam_b;
                    return depths[cam_a][idx_a].item<float>() < depths[cam_b][idx_b].item<float>();
                });

            expected_intersections.push_back(tile_intersections);
            total_expected += tile_intersections.size();
        }

        EXPECT_EQ(intersection_values.size(0), total_expected);

        int32_t curr_offset = 0;
        for (int64_t i = 0; i < num_active_tiles; i++) {
            int32_t next_offset       = tile_offsets[i + 1].item<int32_t>();
            int32_t num_intersections = next_offset - curr_offset;

            EXPECT_EQ(num_intersections, expected_intersections[i].size())
                << "Number of intersections mismatch for tile " << i;

            for (int32_t j = 0; j < num_intersections; j++) {
                EXPECT_EQ(intersection_values[curr_offset + j].item<int32_t>(),
                          expected_intersections[i][j])
                    << "Intersection mismatch at tile " << i << " index " << j;
            }

            curr_offset = next_offset;
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
                     means2d,
                     radii,
                     depths,
                     /*camera_jidx=*/at::nullopt,
                     num_cameras,
                     tile_size,
                     num_tiles_h,
                     num_tiles_w),
                 c10::Error);
}

TEST_F(GaussianTileIntersectionTest, ZeroLengthGaussianTest) {
    const auto numCameras = 5;

    // Create empty tensors with correct shapes
    auto means2d = torch::empty({numCameras, 0, 2}, torch::kFloat32); // [C, N=0, 2]
    auto radii   = torch::empty({numCameras, 0}, torch::kInt32);      // [C. N=0]
    auto depths  = torch::empty({numCameras, 0}, torch::kFloat32);    // [C, N=0]

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(
            means2d.cuda(),
            radii.cuda(),
            depths.cuda(),
            /*camera_jidx=*/at::nullopt,
            numCameras,
            tile_size,
            num_tiles_h,
            num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();

    // For zero Gaussians, we should still get a valid tile_offsets tensor filled with zeros
    EXPECT_EQ(tile_offsets.sizes(), std::vector<int64_t>({numCameras, num_tiles_h, num_tiles_w}));
    EXPECT_TRUE(tile_offsets.equal(torch::zeros_like(tile_offsets)));
    EXPECT_EQ(intersection_values.numel(), 0);
}

TEST_F(GaussianTileIntersectionTest, BadInputsFailTest) {
    const auto nc = 5;
    const auto ng = 3;

    const std::vector<std::vector<std::vector<int64_t>>> configs = {
        // Unpacked:

        // Wrong number of cameras
        {{nc + 1, ng, 2}, {nc, ng}, {nc, ng}},
        {{nc, ng, 2}, {nc - 1, ng}, {nc, ng}},
        {{nc, ng, 2}, {nc, ng}, {nc + 8, ng}},
        // Wrong number of gaussians
        {{nc, ng + 1, 2}, {nc, ng}, {nc, ng}},
        {{nc, ng, 2}, {nc, ng - 1}, {nc, ng}},
        {{nc, ng, 2}, {nc, ng}, {nc, ng + 8}},
        // Wrong mean dim
        {{nc, ng, 4}, {nc, ng}, {nc, ng}},
        // Wrong number of dimensions
        {{nc, 2}, {nc, 3}, {nc, 3}},
        {{nc, 3, 2}, {nc}, {nc, 3}},
        {{nc, 3, 2}, {nc, 3}, {nc}},

        // Packed:

        // Wrong number of gaussians
        {{ng + 1, 2}, {ng}, {ng}, {ng}},
        {{ng, 2}, {ng - 1}, {ng}, {ng}},
        {{ng, 2}, {ng}, {ng + 8}, {ng}},
        {{ng, 2}, {ng}, {ng}, {ng + 5}},
        // Wrong mean dim
        {{ng, 4}, {nc, ng}, {nc, ng}, {ng}},
        // Wrong number of dimensions
        {{ng, 4, 2}, {ng}, {ng}, {ng}},
        {{ng, 2}, {ng, 7}, {ng}, {ng}},
        {{ng, 2}, {ng}, {ng, 2, 1}, {ng}},
        {{ng, 2}, {ng}, {ng, 2, 1}, {ng, 1}},
    };

    for (auto config: configs) {
        // Create empty tensors with correct shapes
        auto means2d        = torch::empty(config[0], torch::kFloat32);
        auto radii          = torch::empty(config[1], torch::kInt32);
        auto depths         = torch::empty(config[2], torch::kFloat32);
        const bool isPacked = config.size() == 4;

        if (isPacked) {
            const auto cameraJidx = torch::zeros(config[3], torch::kInt32);
            EXPECT_THROW(
                fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(means2d.cuda(),
                                                                                  radii.cuda(),
                                                                                  depths.cuda(),
                                                                                  cameraJidx.cuda(),
                                                                                  nc,
                                                                                  tile_size,
                                                                                  num_tiles_h,
                                                                                  num_tiles_w),
                c10::ValueError);

        } else {
            EXPECT_THROW(fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(
                             means2d.cuda(),
                             radii.cuda(),
                             depths.cuda(),
                             /*camera_jidx=*/at::nullopt,
                             nc,
                             tile_size,
                             num_tiles_h,
                             num_tiles_w),
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
            means2d.cuda(),
            radii.cuda(),
            depths.cuda(),
            /*camera_jidx=*/at::nullopt,
            num_cameras,
            tile_size,
            num_tiles_h,
            num_tiles_w);

    // Verify that there are no intersections
    EXPECT_EQ(tile_offsets.sum().item<int32_t>(), 0);
}

TEST_F(GaussianTileIntersectionTest, BasicIntersectionTest) {
    auto [means2d, radii, depths] = createTestData();

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(
            means2d.cuda(),
            radii.cuda(),
            depths.cuda(),
            /*camera_jidx=*/at::nullopt,
            num_cameras,
            tile_size,
            num_tiles_h,
            num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();

    // Verify dimensions
    EXPECT_EQ(tile_offsets.sizes(), std::vector<int64_t>({num_cameras, num_tiles_h, num_tiles_w}));

    // Verify depth sorting
    verifyDepthSorting(tile_offsets, intersection_values, depths);
}

TEST_F(GaussianTileIntersectionTest, PackedFormatTest) {
    auto [means2d, radii, depths] = createTestData();

    // Reshape tensors to packed format [M, 2] and [M]
    auto means2d_packed = means2d.reshape({-1, 2});
    auto radii_packed   = radii.reshape({-1});
    auto depths_packed  = depths.reshape({-1});

    // Create camera indices tensor
    auto camera_indices =
        torch::arange(num_cameras, torch::kInt32).repeat_interleave(num_gaussians);

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersection<torch::kCUDA>(means2d_packed.cuda(),
                                                                          radii_packed.cuda(),
                                                                          depths_packed.cuda(),
                                                                          camera_indices.cuda(),
                                                                          num_cameras,
                                                                          tile_size,
                                                                          num_tiles_h,
                                                                          num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();

    // Verify dimensions and basic properties
    EXPECT_EQ(tile_offsets.sizes(), std::vector<int64_t>({num_cameras, num_tiles_h, num_tiles_w}));
    EXPECT_GT(intersection_values.numel(), 0);

    // Verify depth sorting
    verifyDepthSorting(tile_offsets, intersection_values, depths);
}

////// Sparse tests //////

torch::Tensor
tile_mask_to_active_tiles(const torch::Tensor &tile_mask) {
    return torch::nonzero(tile_mask.flatten()).flatten().to(torch::kUInt32);
}

TEST_F(GaussianTileIntersectionTest, DenseViaSparseTest) {
    auto [means2d, radii, depths] = createTestData();

    // Create a mask with all tiles active
    auto tile_mask           = torch::ones({num_cameras, num_tiles_h, num_tiles_w}, torch::kBool);
    int32_t num_active_tiles = tile_mask.sum().item<int32_t>();
    auto active_tiles        = tile_mask_to_active_tiles(tile_mask);

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersectionSparse<torch::kCUDA>(
            means2d.cuda(),
            radii.cuda(),
            depths.cuda(),
            tile_mask.cuda(),
            active_tiles.cuda(),
            /*camera_jidx=*/at::nullopt,
            num_cameras,
            tile_size,
            num_tiles_h,
            num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();

    // Verify dimensions
    EXPECT_EQ(tile_offsets.sizes(), std::vector<int64_t>({num_active_tiles + 1}));

    // Verify that offsets are monotonically increasing
    for (int64_t i = 1; i < tile_offsets.size(0); i++) {
        EXPECT_GE(tile_offsets[i].item<int32_t>(), tile_offsets[i - 1].item<int32_t>())
            << "Tile offsets must be monotonically increasing";
    }

    // Verify that the last offset equals the size of intersection_values
    EXPECT_EQ(tile_offsets[tile_offsets.size(0) - 1].item<int32_t>(), intersection_values.size(0))
        << "Last offset should equal number of intersections";

    // Verify depth sorting
    verifyDepthSorting(tile_offsets, intersection_values, depths, tile_mask);
}

TEST_F(GaussianTileIntersectionTest, SparseIntersectionTest) {
    auto [means2d, radii, depths] = createTestData();

    // Create a mask where only the top-left quadrant of tiles is active
    auto tile_mask = torch::zeros({num_cameras, num_tiles_h, num_tiles_w}, torch::kBool);
    tile_mask.slice(1, 0, num_tiles_h / 2).slice(2, 0, num_tiles_w / 2).fill_(true);
    auto active_tiles = tile_mask_to_active_tiles(tile_mask);

    // Calculate number of active tiles
    int32_t num_active_tiles = active_tiles.size(0);

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersectionSparse<torch::kCUDA>(
            means2d.cuda(),
            radii.cuda(),
            depths.cuda(),
            tile_mask.cuda(),
            active_tiles.cuda(),
            /*camera_jidx=*/at::nullopt,
            num_cameras,
            tile_size,
            num_tiles_h,
            num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();
    tile_mask           = tile_mask.cpu();

    EXPECT_EQ(tile_offsets.sizes(), std::vector<int64_t>({num_active_tiles + 1}));

    // tile_offsets should be: [0 2 3 4 5 5 5 5 7] (note 9 elements)

    // Verify that offsets are monotonically increasing
    for (int64_t i = 1; i < tile_offsets.size(0); i++) {
        EXPECT_GE(tile_offsets[i].item<int32_t>(), tile_offsets[i - 1].item<int32_t>())
            << "Tile offsets must be monotonically increasing";
    }

    // Verify that the last offset equals the size of intersection_values minus 1
    EXPECT_EQ(tile_offsets[-1].item<int32_t>(), intersection_values.size(0));

    verifyDepthSorting(tile_offsets, intersection_values, depths, tile_mask);
}

TEST_F(GaussianTileIntersectionTest, SparseCPUNotImplementedTest) {
    auto const [means2d, radii, depths] = createTestData();
    auto tile_mask                      = torch::ones({1, num_tiles_h, num_tiles_w}, torch::kBool);
    auto num_active_tiles               = num_tiles_h * num_tiles_w;
    auto active_tiles                   = tile_mask_to_active_tiles(tile_mask);

    EXPECT_THROW(fvdb::detail::ops::dispatchGaussianTileIntersectionSparse<torch::kCPU>(
                     means2d,
                     radii,
                     depths,
                     tile_mask,
                     active_tiles,
                     /*camera_jidx=*/at::nullopt,
                     num_cameras,
                     tile_size,
                     num_tiles_h,
                     num_tiles_w),
                 c10::Error);
}

TEST_F(GaussianTileIntersectionTest, SparseZeroLengthGaussianTest) {
    const auto numCameras = 5;

    // Create empty tensors with correct shapes
    auto means2d = torch::empty({numCameras, 0, 2}, torch::kFloat32); // [C, N=0, 2]
    auto radii   = torch::empty({numCameras, 0}, torch::kInt32);      // [C. N=0]
    auto depths  = torch::empty({numCameras, 0}, torch::kFloat32);    // [C, N=0]

    auto tile_mask           = torch::ones({numCameras, num_tiles_h, num_tiles_w}, torch::kBool);
    int32_t num_active_tiles = tile_mask.sum().item<int32_t>();
    auto active_tiles        = tile_mask_to_active_tiles(tile_mask);

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersectionSparse<torch::kCUDA>(
            means2d.cuda(),
            radii.cuda(),
            depths.cuda(),
            tile_mask.cuda(),
            active_tiles.cuda(),
            /*camera_jidx=*/at::nullopt,
            numCameras,
            tile_size,
            num_tiles_h,
            num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();

    // For zero Gaussians, we should still get a valid tile_offsets tensor
    EXPECT_EQ(tile_offsets.sizes(), std::vector<int64_t>({num_active_tiles + 1}));
    EXPECT_TRUE(tile_offsets.equal(torch::zeros_like(tile_offsets)));
    EXPECT_EQ(intersection_values.numel(), 0);
}

TEST_F(GaussianTileIntersectionTest, SparsePackedFormatTest) {
    auto [means2d, radii, depths] = createTestData();

    // Reshape tensors to packed format [M, 2] and [M]
    auto means2d_packed = means2d.reshape({-1, 2});
    auto radii_packed   = radii.reshape({-1});
    auto depths_packed  = depths.reshape({-1});

    // Create camera indices tensor
    auto camera_indices =
        torch::arange(num_cameras, torch::kInt32).repeat_interleave(num_gaussians);

    auto tile_mask           = torch::ones({num_cameras, num_tiles_h, num_tiles_w}, torch::kBool);
    int32_t num_active_tiles = tile_mask.sum().item<int32_t>();
    auto active_tiles        = tile_mask_to_active_tiles(tile_mask);

    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersectionSparse<torch::kCUDA>(
            means2d_packed.cuda(),
            radii_packed.cuda(),
            depths_packed.cuda(),
            tile_mask.cuda(),
            active_tiles.cuda(),
            camera_indices.cuda(),
            num_cameras,
            tile_size,
            num_tiles_h,
            num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();

    // Verify dimensions and basic properties
    EXPECT_EQ(tile_offsets.sizes(), std::vector<int64_t>({num_active_tiles + 1}));
    EXPECT_GT(intersection_values.numel(), 0);

    verifyDepthSorting(tile_offsets, intersection_values, depths, tile_mask);
}

TEST_F(GaussianTileIntersectionTest, RandomSparsePatternTest) {
    auto [means2d, radii, depths] = createTestData();

    // Create a random sparsity pattern with ~30% active tiles
    auto tile_mask = torch::rand({num_cameras, num_tiles_h, num_tiles_w}) < 0.3;
    tile_mask      = tile_mask.to(torch::kBool);

    // Ensure at least one tile is active to avoid empty test case
    if (tile_mask.sum().item<int32_t>() == 0) {
        tile_mask[0][0][0] = true;
    }

    auto active_tiles        = tile_mask_to_active_tiles(tile_mask);
    int32_t num_active_tiles = active_tiles.size(0);

#if 0
    // Print sparsity pattern and active tile indices
    std::cout << "Active tiles pattern:\n";
    for (int c = 0; c < num_cameras; c++) {
        std::cout << "Camera " << c << ":\n";
        for (int h = 0; h < num_tiles_h; h++) {
            for (int w = 0; w < num_tiles_w; w++) {
                std::cout << (tile_mask[c][h][w].item<bool>() ? "1 " : "0 ");
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
#endif
    auto [tile_offsets, intersection_values] =
        fvdb::detail::ops::dispatchGaussianTileIntersectionSparse<torch::kCUDA>(
            means2d.cuda(),
            radii.cuda(),
            depths.cuda(),
            tile_mask.cuda(),
            active_tiles.cuda(),
            /*camera_jidx=*/at::nullopt,
            num_cameras,
            tile_size,
            num_tiles_h,
            num_tiles_w);

    // Move results back to CPU for verification
    tile_offsets        = tile_offsets.cpu();
    intersection_values = intersection_values.cpu();
    tile_mask           = tile_mask.cpu();

    verifyTileIntersections(
        tile_offsets, intersection_values, means2d, radii, depths, active_tiles);

#if 0
    // Print statistics
    float sparsity =
        static_cast<float>(num_active_tiles) / (num_cameras * num_tiles_h * num_tiles_w);
    std::cout << "Sparsity statistics:\n"
              << "Total tiles: " << (num_cameras * num_tiles_h * num_tiles_w) << "\n"
              << "Active tiles: " << num_active_tiles << "\n"
              << "Sparsity ratio: " << sparsity << "\n"
              << "Total intersections: " << intersection_values.size(0) << "\n";
#endif
}
