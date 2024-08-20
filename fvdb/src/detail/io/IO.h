// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
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
fromNVDB(nanovdb::GridHandle<nanovdb::HostBuffer>        &handle,
         const torch::optional<fvdb::TorchDeviceOrString> maybeDevice =
             torch::optional<fvdb::TorchDeviceOrString>());

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
fromNVDB(const std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> &handles,
         const torch::optional<fvdb::TorchDeviceOrString>             maybeDevice =
             torch::optional<fvdb::TorchDeviceOrString>());

nanovdb::GridHandle<nanovdb::HostBuffer>
toNVDB(const GridBatch                             &gridBatch,
       const torch::optional<JaggedTensor>          maybeData = torch::optional<JaggedTensor>(),
       const torch::optional<StringOrListOfStrings> maybeNames =
           torch::optional<StringOrListOfStrings>());

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
loadNVDB(const std::string &path, const fvdb::NanoVDBFileGridIdentifier &gridIdentifier,
         fvdb::TorchDeviceOrString device, bool verbose);

void saveNVDB(const std::string &path, const GridBatch &gridBatch,
              const torch::optional<fvdb::JaggedTensor>          maybeData,
              const torch::optional<fvdb::StringOrListOfStrings> maybeNames,
              bool compressed = false, bool verbose = false);

} // namespace io
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_IO_IO_H