// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef FVDB_TESTS_UTILS_IMAGEUTILS_H
#define FVDB_TESTS_UTILS_IMAGEUTILS_H

#include <torch/torch.h>

#include <string>

namespace fvdb::test {

/// @brief Write out tensors representing RGB image data and alpha values as a PNG image
/// @param colors Tensor of shape [H, W, 3] representing RGB image data
/// @param alphas Tensor of shape [H, W, 1] representing alpha values
/// @param filename The filename to write the image to
void
writePNG(const torch::Tensor &colors, const torch::Tensor &alphas, const std::string &filename);

} // namespace fvdb::test

#endif // FVDB_TESTS_UTILS_IMAGEUTILS_H
