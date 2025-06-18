// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_IO_IO_H
#define FVDB_DETAIL_IO_IO_H

#include <fvdb/GridBatch.h>
#include <fvdb/Types.h>

#include <torch/torch.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace io {

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
fromNVDB(nanovdb::GridHandle<nanovdb::HostBuffer> &handle,
         const std::optional<torch::Device> maybeDevice = std::optional<torch::Device>());

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
fromNVDB(const std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> &handles,
         const std::optional<torch::Device> maybeDevice = std::optional<torch::Device>());

nanovdb::GridHandle<nanovdb::HostBuffer>
toNVDB(const GridBatch &gridBatch,
       const std::optional<JaggedTensor> maybeData = std::nullopt,
       const std::vector<std::string> &names       = {});

/// @brief Load a vector of grid handles matching the user-specified indices
std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
loadNVDB(const std::string &path,
         const std::vector<uint64_t> &indices,
         const torch::Device &device,
         bool verbose);

/// @brief Load a vector of grid handles matching the user-specified names
std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
loadNVDB(const std::string &path,
         const std::vector<std::string> &names,
         const torch::Device &device,
         bool verbose);

/// @brief Load a vector of grid handles
std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
loadNVDB(const std::string &path, const torch::Device &device, bool verbose);

void saveNVDB(const std::string &path,
              const GridBatch &gridBatch,
              const std::optional<JaggedTensor> maybeData,
              const std::vector<std::string> &names = {},
              bool compressed                       = false,
              bool verbose                          = false);

} // namespace io
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_IO_IO_H
