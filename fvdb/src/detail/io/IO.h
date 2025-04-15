// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_IO_IO_H
#define FVDB_DETAIL_IO_IO_H

#include <GridBatch.h>
#include <Types.h>

#include <torch/torch.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace io {

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
fromNVDB(nanovdb::GridHandle<nanovdb::HostBuffer> &handle,
         const std::optional<torch::Device>        maybeDevice = std::optional<torch::Device>());

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
fromNVDB(const std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> &handles,
         const std::optional<torch::Device> maybeDevice = std::optional<torch::Device>());

nanovdb::GridHandle<nanovdb::HostBuffer>
toNVDB(const GridBatch &gridBatch, const std::optional<JaggedTensor> maybeData = std::nullopt,
       const std::vector<std::string> &names = {});

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
loadNVDB(const std::string &path, const fvdb::NanoVDBFileGridIdentifier &gridIdentifier,
         const torch::Device &device, bool verbose);

void saveNVDB(const std::string &path, const GridBatch &gridBatch,
              const std::optional<JaggedTensor> maybeData,
              const std::vector<std::string> &names = {}, bool compressed = false,
              bool verbose = false);

} // namespace io
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_IO_IO_H
