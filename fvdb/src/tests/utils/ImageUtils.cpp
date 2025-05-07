// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <torch/torch.h>

#include <png.h>
#include <tests/utils/ImageUtils.h>

#include <algorithm>
#include <csetjmp>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace fvdb::test {

void
writePNG(const torch::Tensor &colors, const torch::Tensor &alphas,
         const std::string &outputFilename) {
    // 1. Validate inputs
    TORCH_CHECK(colors.dim() == 3, "colors must have shape [H, W, 3]");
    TORCH_CHECK(alphas.dim() == 3 && alphas.size(2) == 1, "alphas must have shape [H, W, 1]");
    TORCH_CHECK(colors.device() == torch::kCPU, "colors must be on CPU");
    TORCH_CHECK(alphas.device() == torch::kCPU, "alphas must be on CPU");
    // Allow float or double, convert to float32 internally
    TORCH_CHECK(colors.scalar_type() == torch::kFloat32 || colors.scalar_type() == torch::kFloat64,
                "colors must be float or double type");
    TORCH_CHECK(alphas.scalar_type() == torch::kFloat32 || alphas.scalar_type() == torch::kFloat64,
                "alphas must be float or double type");

    TORCH_CHECK(colors.size(0) == alphas.size(0) && colors.size(1) == alphas.size(1),
                "Color and alpha dimensions must match");

    int height         = colors.size(0);
    int width          = colors.size(1);
    int color_channels = colors.size(2);

    // Assuming input colors are RGB for now. Modify if RGBA is possible.
    TORCH_CHECK(color_channels == 3, "This function currently expects 3 color channels (RGB)");

    // 2. Combine color and alpha, convert to uint8 RGBA
    auto rgba_tensor = torch::empty({ height, width, 4 }, torch::kUInt8);
    // Convert inputs to float32 for processing
    auto colors_float = colors.to(torch::kFloat32);
    auto alphas_float = alphas.to(torch::kFloat32);

    // Accessors for efficient element access
    auto colors_acc = colors_float.accessor<float, 3>();
    auto alphas_acc = alphas_float.accessor<float, 3>();
    auto rgba_acc   = rgba_tensor.accessor<uint8_t, 3>();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Clamp and scale [0, 1] float to [0, 255] uint8
            rgba_acc[y][x][0] = static_cast<uint8_t>(
                std::max(0.0f, std::min(255.0f, colors_acc[y][x][0] * 255.0f))); // R
            rgba_acc[y][x][1] = static_cast<uint8_t>(
                std::max(0.0f, std::min(255.0f, colors_acc[y][x][1] * 255.0f))); // G
            rgba_acc[y][x][2] = static_cast<uint8_t>(
                std::max(0.0f, std::min(255.0f, colors_acc[y][x][2] * 255.0f))); // B
            rgba_acc[y][x][3] = static_cast<uint8_t>(
                std::max(0.0f, std::min(255.0f, alphas_acc[y][x][0] * 255.0f))); // A
        }
    }

    // Ensure data is contiguous in memory for libpng
    rgba_tensor         = rgba_tensor.contiguous();
    uint8_t *image_data = rgba_tensor.data_ptr<uint8_t>();

    // 3. libpng setup
    FILE *fp = fopen(outputFilename.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("Could not open file for writing: " + outputFilename);
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        fclose(fp);
        throw std::runtime_error("Could not allocate png write struct");
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        fclose(fp);
        throw std::runtime_error("Could not allocate png info struct");
    }

    // Setup error handling using setjmp
    // Needs C linkage for the error function if defined separately
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        throw std::runtime_error("Error during png creation via libpng");
    }

    png_init_io(png_ptr, fp);

    // 4. Write header (IHDR)
    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8,                   // Bit depth per channel
                 PNG_COLOR_TYPE_RGBA, // We are writing RGBA
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    // 5. Create row pointers
    // libpng expects an array of pointers, each pointing to the start of a row
    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; ++y) {
        row_pointers[y] = image_data + y * width * 4; // 4 bytes per pixel (RGBA)
    }

    // 6. Write image data
    png_write_image(png_ptr, row_pointers.data());

    // 7. End write
    png_write_end(png_ptr, nullptr);

    // 8. Cleanup
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

} // namespace fvdb::test
