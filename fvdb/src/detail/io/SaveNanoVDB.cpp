#include "detail/io/IO.h"

#include "detail/utils/Utils.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <torch/all.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <c10/cuda/CUDACachingAllocator.h>


namespace fvdb {
namespace detail {
namespace io {

/// @brief Copy a std::string to a char buffer with a fixed size and throw an exception if the string is too long
/// @param targetBuf A pointer to the buffer to write the string to
/// @param maxSize The maximum size of the target buffer
/// @param sourceSting The source string to copy
/// @param bufName A name for this string to use when throwing an exception (default is "String")
void setFixedSizeStringBuf(char* targetBuf, size_t maxSize, std::string sourceSting, std::string bufName = "String") {
    memset(targetBuf, 0, maxSize);
    TORCH_CHECK_VALUE(sourceSting.size() < maxSize, bufName + " exceeds maximum character length of " + std::to_string(maxSize) + ".");
    strncpy(targetBuf, sourceSting.c_str(), maxSize);
}


/// @brief Get the (row) value at index rowIdx from a tensor with 2 dimensions.
///        Specialized to return useful nanovdb types (e.g. Vec3f, Vec4f, etc...)
/// @tparam ScalarT The scalar type of values
/// @tparam ValueT The return type of the row value (e.g. nanovdb::Vec3f)
/// @param acc The accessor to the tensor
/// @param rowIdx The row to read from
/// @return The rowIdx^th row of the tensor casted to ValueT
template <class ScalarT, class ValueT>
inline ValueT valueGetter(torch::TensorAccessor<ScalarT, 2>& acc, int rowIdx) {
    return acc[rowIdx][0];
}
template <>
inline nanovdb::Vec3f valueGetter(torch::TensorAccessor<float, 2>& acc, int rowIdx) {
    return {acc[rowIdx][0], acc[rowIdx][1], acc[rowIdx][2]};
}
template <>
inline nanovdb::Vec4f valueGetter(torch::TensorAccessor<float, 2>& acc, int rowIdx) {
    return {acc[rowIdx][0], acc[rowIdx][1], acc[rowIdx][2], acc[rowIdx][3]};
}
template <>
inline nanovdb::Vec3d valueGetter(torch::TensorAccessor<double, 2>& acc, int rowIdx) {
    return {acc[rowIdx][0], acc[rowIdx][1], acc[rowIdx][2]};
}
template <>
inline nanovdb::Vec4d valueGetter(torch::TensorAccessor<double, 2>& acc, int rowIdx) {
    return {acc[rowIdx][0], acc[rowIdx][1], acc[rowIdx][2], acc[rowIdx][3]};
}
template <>
inline nanovdb::Vec3i valueGetter(torch::TensorAccessor<int32_t, 2>& acc, int rowIdx) {
    return {acc[rowIdx][0], acc[rowIdx][1], acc[rowIdx][2]};
}
template <>
inline nanovdb::math::Rgba8 valueGetter(torch::TensorAccessor<uint8_t, 2>& acc, int rowIdx) {
    return {acc[rowIdx][0], acc[rowIdx][1], acc[rowIdx][2], acc[rowIdx][3]};
}


/// @brief Helper function to copy an index grid with a corresponding JaggedTensor of values to a nanovdb grid
///        with values stored directly in the leaves. This will only work for values which correspond to valid nanovdb
///        grid types (e.g. Vec3f, Vec4f, Vec3d, Vec4d, etc...)
/// @tparam OutGridType The type of data to store in the returned grid (e.g. float, nanovdb::Vec3f, etc...)
/// @tparam InScalarType The scalar type of the input jagged tensor (e.g float, double, int32_t, etc...)
/// @param gridBatch The batch of index grids to copy
/// @param data The JaggedTensor of data to copy
/// @param names The names of the grids in the batch to write to the copied output (optional)
/// @return A nanovdb grid handle with the copied data stored in the leaves
template <typename OutGridType, typename InScalarType>
nanovdb::GridHandle<nanovdb::HostBuffer> fvdbToNanovdbGridWithValues(const GridBatch& gridBatch,
                                                                     const JaggedTensor& data,
                                                                     const std::vector<std::string>& names) {

    TORCH_CHECK(names.size() == 0 || names.size() == (size_t) gridBatch.grid_count(),
                "Invalid parameter for names, must be empty or a list of the same length as the batch size. Got "
                + std::to_string(names.size()) + " names for batch size " + std::to_string(gridBatch.grid_count()));
    TORCH_CHECK(!gridBatch.is_mutable(), "Need to use indexing with mutable grids!");

    using ProxyGridT = nanovdb::tools::build::Grid<OutGridType>;
    using GridValueT = typename ProxyGridT::ValueType;
    using HostGridHandle = nanovdb::GridHandle<nanovdb::HostBuffer>;

    // We'll build each grid from the ijk values and data, so get accessors for these
    JaggedTensor ijkValues = gridBatch.ijk().cpu();

    // Get a contiguous CPU copy of the data tensor and make sure it has 2 dimensions
    torch::Tensor jdataCpu = data.cpu().jdata().squeeze().contiguous();
    if (jdataCpu.ndimension() == 0) {
        jdataCpu = jdataCpu.unsqueeze(0);
    }
    if (jdataCpu.ndimension() == 1) {
        jdataCpu = jdataCpu.unsqueeze(1);
    }
    TORCH_CHECK(jdataCpu.size(0) == gridBatch.total_voxels(), "Invalid data tensor size. Must match number of voxels in grid batch.");

    auto ijkAccessor = ijkValues.jdata().accessor<int, 2>();
    auto jdataAccessor = jdataCpu.accessor<InScalarType, 2>();

    // Populate a vector of host buffers for each grid in the batch
    std::vector<HostGridHandle> buffers(gridBatch.grid_count());
    for (int64_t bid = 0; bid < gridBatch.grid_count(); bid += 1) {
        const std::string name = names.size() > 0 ? names[bid] : "";
        TORCH_CHECK_VALUE(name.size() < nanovdb::GridData::MaxNameSize, "Grid name " + name + " exceeds maximum character length of " + std::to_string(nanovdb::GridData::MaxNameSize) + ".");

        auto proxyGrid = std::make_shared<ProxyGridT>(GridValueT(0), name);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        const int start = ijkValues.joffsets()[bid][0].item<int>();
        const int end = ijkValues.joffsets()[bid][1].item<int>();
        const int64_t numVoxels = end - start;
        const int64_t numData = data.joffsets()[bid][1].item<int>() - data.joffsets()[bid][0].item<int>();
        TORCH_CHECK_VALUE(numData == gridBatch.num_voxels_at(bid),
                          "Invalid number of voxels in jagged tensor at index " + std::to_string(bid) +
                          ". Expected it to match the number of voxels at grid index " + std::to_string(bid) + ". " +
                          "Got " + std::to_string(numVoxels) + " but expected " +
                          std::to_string(gridBatch.num_voxels_at(bid)) + ".");
        for (int i = 0; i < numVoxels; i += 1) {
            const GridValueT& value = valueGetter<InScalarType, GridValueT>(jdataAccessor, start + i);
            const nanovdb::Coord ijk(ijkAccessor[start + i][0], ijkAccessor[start + i][1], ijkAccessor[start + i][2]);
            proxyGridAccessor.setValue(ijk, value);
        }
        proxyGridAccessor.merge();

        // Write shape of tensor to blind data so we can load it back with the same shape
        // This lets us handle things like (N, 1, 3) tensors which we can save as Vec3f grids
        nanovdb::tools::CreateNanoGrid<ProxyGridT> converter(*proxyGrid);
        converter.addBlindData("fvdb_jdata",
                               nanovdb::GridBlindDataSemantic::Unknown,
                               nanovdb::GridBlindDataClass::Unknown,
                               nanovdb::GridType::Unknown,
                               data.jdata().dim() + 1,
                               sizeof(int64_t));
        buffers.push_back(converter.template getHandle<OutGridType, nanovdb::HostBuffer>(nanovdb::HostBuffer()));
        TORCH_CHECK(buffers.back().gridCount() == 1, "Internal error. Invalid grid count.");
        nanovdb::NanoGrid<OutGridType>* nanoGrid = buffers.back().grid<OutGridType>();
        TORCH_CHECK(nanoGrid->blindDataCount() == 1, "Internal error. Invalid blind metadata count.");
        int64_t* writeHead = (int64_t*) nanoGrid->blindMetaData(0).blindData();
        JaggedTensor dataBi = data.index({bid});
        *writeHead = (int64_t) dataBi.jdata().dim();
        writeHead += 1;
        for (int di = 0; di < dataBi.jdata().dim(); di += 1) {
            *writeHead = (int64_t) dataBi.jdata().size(di);
            writeHead += 1;
        }
    }

    // Merge all the buffers into a single one if we have more than one grid
    if (buffers.size() == 1) {
        return std::move(buffers[0]);
    } else {
        return nanovdb::mergeGrids(buffers);
    }
}

nanovdb::GridHandle<nanovdb::HostBuffer> maybeConvertToStandardNanovdbGrid(const fvdb::GridBatch& gridBatch,
                                                                           const fvdb::JaggedTensor data,
                                                                           const std::vector<std::string> names)
{
    // We can't convert mutable grids to a standard format because we don't know what do with disabled voxels
    if (gridBatch.is_mutable()) {
        return nanovdb::GridHandle<nanovdb::HostBuffer>();
    }

    // Get a squeezed view of the tensor so we can save data with redundant dimensions
    // (e.g. shape (N, 1, 3) can get saved as a Vec3f grid)
    torch::Tensor jdataSqueezed = data.jdata().squeeze();
    if (jdataSqueezed.numel() == 1 && jdataSqueezed.dim() == 0) {  // Make sure we have at least 1 dimension
        jdataSqueezed = jdataSqueezed.unsqueeze(0);
        TORCH_CHECK(jdataSqueezed.ndimension() == 1, "Internal error: Invalid jdata shape when saving grid.");
    }

    if (data.dtype() == torch::kHalf) {
        if (jdataSqueezed.dim() == 1 || (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 1)) {
            using GridType = nanovdb::Fp16;
            return fvdbToNanovdbGridWithValues<GridType, c10::Half>(gridBatch, data, names);
        }
    } else if (data.dtype() == torch::kFloat32) {
        if (jdataSqueezed.dim() == 1 || (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 1)) {
            using GridType = float;
            return fvdbToNanovdbGridWithValues<GridType, float>(gridBatch, data, names);
        } else if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 3) {
            using GridType = nanovdb::Vec3f;
            return fvdbToNanovdbGridWithValues<GridType, float>(gridBatch, data, names);
        } else if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 4) {
            using GridType = nanovdb::Vec4f;
            return fvdbToNanovdbGridWithValues<GridType, float>(gridBatch, data, names);
        }
    } else if (data.dtype() == torch::kFloat64) {
        if (jdataSqueezed.dim() == 1 || (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 1)) {
            using GridType = double;
            return fvdbToNanovdbGridWithValues<GridType, double>(gridBatch, data, names);
        } else if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 3) {
            using GridType = nanovdb::Vec3d;
            return fvdbToNanovdbGridWithValues<GridType, double>(gridBatch, data, names);
        } else if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 4) {
            using GridType = nanovdb::Vec4d;
            return fvdbToNanovdbGridWithValues<GridType, double>(gridBatch, data, names);
        }
    } else if (data.dtype() == torch::kInt32) {
        if (jdataSqueezed.dim() == 1 || (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 1)) {
            using GridType = int32_t;
            return fvdbToNanovdbGridWithValues<GridType, int32_t>(gridBatch, data, names);
        } else if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 3) {
            using GridType = nanovdb::Vec3i;
            return fvdbToNanovdbGridWithValues<GridType, int32_t>(gridBatch, data, names);
        }
    } else if (data.dtype() == torch::kInt64) {
        if (jdataSqueezed.dim() == 1 || (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 1)) {
            using GridType = int64_t;
            return fvdbToNanovdbGridWithValues<GridType, int64_t>(gridBatch, data, names);
        }
    } else if (data.dtype() == torch::kUInt8) {
        if (jdataSqueezed.dim() == 2 && jdataSqueezed.size(1) == 4) {
            using GridType = nanovdb::math::Rgba8;
            return fvdbToNanovdbGridWithValues<GridType, uint8_t>(gridBatch, data, names);
        }
    }

    return nanovdb::GridHandle<nanovdb::HostBuffer>();
}

bool maybeSaveStandardNanovdbGrid(const std::string& path,
                                  const GridBatch& gridBatch,
                                  const JaggedTensor data,
                                  const std::vector<std::string> names,
                                  nanovdb::io::Codec codec,
                                  bool verbose) {

    nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle = maybeConvertToStandardNanovdbGrid(gridBatch, data, names);
    if (gridHandle.isEmpty())
    {
        return false;
    }

    nanovdb::io::writeGrid(path, gridHandle, codec, verbose);
    return true;
}

nanovdb::GridHandle<nanovdb::HostBuffer> getIndexGrid(const GridBatch& gridBatch,
                                                      const std::vector<std::string> names = {}) {

    const nanovdb::GridHandle<PytorchDeviceBuffer>& nanoGridHdl = gridBatch.nanovdb_grid_handle();

    // Allocate memory and get pointer to host grid buffer
    nanovdb::HostBuffer writeBuf(nanoGridHdl.buffer().size());
    void* writeHead = writeBuf.data();

    // Get pointer to grid read from (possibly on the device)
    const bool isCuda = nanoGridHdl.buffer().device().is_cuda();
    void* readHead = isCuda ? nanoGridHdl.buffer().deviceData() : nanoGridHdl.buffer().data();
    const size_t sourceGridByteSize = nanoGridHdl.buffer().size();

    // Write out the full grid to the buffer
    if (isCuda) {
        at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(gridBatch.device().index());
        cudaMemcpyAsync(writeHead, readHead, sourceGridByteSize, cudaMemcpyDeviceToHost, defaultStream.stream());
        cudaStreamSynchronize(defaultStream.stream());
    } else {
        memcpy(writeHead, readHead, sourceGridByteSize);
    }

    nanovdb::GridHandle<nanovdb::HostBuffer> retHandle = nanovdb::GridHandle<nanovdb::HostBuffer> (std::move(writeBuf));

    // Write voxelSize and origin information to the output buffer
    for (int64_t bi = 0; bi < gridBatch.grid_count(); bi += 1)
    {
        nanovdb::GridData *retGridData = (nanovdb::GridData *)(retHandle.gridData(bi));
        torch::Tensor voxelSize = gridBatch.voxel_size_at(bi, torch::kFloat64);
        torch::Tensor origin = gridBatch.origin_at(bi, torch::kFloat64);
        retGridData->mVoxelSize = {voxelSize[0].item<double>(), voxelSize[1].item<double>(), voxelSize[2].item<double>()};
        retGridData->mMap = nanovdb::Map(voxelSize[0].item<double>(), {origin[0].item<double>(), origin[1].item<double>(), origin[2].item<double>()});
    }

    // If you passed in grid names, write them to the output buffer
    if (names.size() > 0) {
        for (int64_t bi = 0; bi < gridBatch.grid_count(); bi += 1) {
            const std::string name = names.size() > 0 ? names[bi] : "";
            TORCH_CHECK_VALUE(name.size() < nanovdb::GridData::MaxNameSize, "Grid name " + name + " exceeds maximum character length of " + std::to_string(nanovdb::GridData::MaxNameSize) + ".");
            nanovdb::GridData* retGridData = (nanovdb::GridData*) (retHandle.gridData(bi));
            #pragma GCC diagnostic ignored "-Wstringop-truncation"
            strncpy(retGridData->mGridName, names[bi].c_str(), nanovdb::GridData::MaxNameSize);
        }
    }

    // Build a grid handle
    return retHandle;
}

void saveIndexGrid(const std::string& path,
                   const GridBatch& gridBatch,
                   const std::vector<std::string> names,
                   nanovdb::io::Codec codec,
                   bool verbose) {

    // If you don't pass in data, then we just write the grid
    nanovdb::GridHandle<nanovdb::HostBuffer> writeHandle = getIndexGrid(gridBatch, names);

    // Save the grid to disk
    nanovdb::io::writeGrid(path, writeHandle, codec, verbose);
}

void saveIndexGridWithBlindData(const std::string& path,
                                const GridBatch& gridBatch,
                                const JaggedTensor data,
                                const std::vector<std::string> names,
                                nanovdb::io::Codec codec,
                                bool verbose) {

    const nanovdb::GridHandle<PytorchDeviceBuffer>& nanoGridHdl = gridBatch.nanovdb_grid_handle();

    // Make a (possible) cpu copy of the data jagged tensor
    JaggedTensor cpuData = data.cpu().contiguous();

    // Compute blind data sizes padded to be 32 byte aligned
    std::vector<uint64_t> blindDataPadding;  // Size of each blind data padded to 32 bytes
    std::vector<uint64_t> paddedBlindDataSizes;  // The amount of padding added to each blind data to achieve 32 byte alignment
    uint64_t totalBlindDataSize = 0;
    for (int bi = 0; bi < gridBatch.grid_count(); bi += 1) {
        JaggedTensor dataBi = cpuData.index({bi});
        const int64_t numVoxelsBi = gridBatch.num_voxels_at(bi);
        const int64_t jdataBytesBi = dataBi.jdata().numel() * dataBi.jdata().element_size();
        TORCH_CHECK_VALUE(numVoxelsBi == dataBi.size(0),
                          "Invalid number of voxels in jagged tensor at index " + std::to_string(bi) +
                          ". Expected it to match the number of voxels at grid index " + std::to_string(bi) + ". " +
                          "Got " + std::to_string(dataBi.jdata().size(0)) + " but expected " +
                          std::to_string(gridBatch.num_voxels_at(bi)) + ".");
        const uint64_t blindDataSizeBi = jdataBytesBi + sizeof(int64_t) * (dataBi.dim() + 1);
        const uint64_t paddedBlindDataSizeBi = nanovdb::math::AlignUp<32UL>(blindDataSizeBi);
        blindDataPadding.push_back(paddedBlindDataSizeBi - blindDataSizeBi);
        paddedBlindDataSizes.push_back(paddedBlindDataSizeBi);
        totalBlindDataSize += paddedBlindDataSizeBi;
    }

    // Allocate a big enough buffer to allocate the index grid and blind data
    const size_t allocSize = nanoGridHdl.buffer().size() +                                 // Grids (32B aligned)
                             sizeof(nanovdb::GridBlindMetaData) * gridBatch.grid_count() + // Blind metadata (32B aligned)
                             totalBlindDataSize;                                           // Blind data (32B aligned)
    nanovdb::HostBuffer writeBuf(allocSize);

    // Get pointer to read (possibly on the device) and write pointers
    const bool isCuda = nanoGridHdl.buffer().device().is_cuda();
    uint8_t* writeHead = static_cast<uint8_t*>(writeBuf.data());
    uint8_t* readHead = static_cast<uint8_t*>(isCuda ? nanoGridHdl.buffer().deviceData() : nanoGridHdl.buffer().data());

    // Copy each grid and each entry in the jagged tensor
    for (int bi = 0; bi < gridBatch.grid_count(); bi += 1) {

        // Copy the full bi^th index grid to the buffer
        const size_t sourceGridByteSize = nanoGridHdl.gridSize(bi);
        if (isCuda) {
            at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(gridBatch.device().index());
            cudaMemcpyAsync((void*) writeHead, (void*) readHead, sourceGridByteSize, cudaMemcpyDeviceToHost, defaultStream.stream());
        } else {
            memcpy((void*) writeHead, (void*) readHead, sourceGridByteSize);
        }
        // Update the metadata for the copied grid in the buffer to be a tensor grid with blind data
        nanovdb::GridData* writeGridData = reinterpret_cast<nanovdb::GridData*>(writeHead);
        writeGridData->mGridClass = nanovdb::GridClass::TensorGrid;
        writeGridData->mGridType = gridBatch.is_mutable() ? nanovdb::GridType::OnIndexMask : nanovdb::GridType::OnIndex;
        writeGridData->mBlindMetadataCount = 1;
        writeGridData->mBlindMetadataOffset = sourceGridByteSize;
        const std::string name = names.size() > 0 ? names[bi] : "";
        setFixedSizeStringBuf(writeGridData->mGridName, nanovdb::GridData::MaxNameSize, name, "Grid name " + name);
        writeGridData->mGridSize = sourceGridByteSize + sizeof(nanovdb::GridBlindMetaData) + paddedBlindDataSizes[bi];
        readHead += sourceGridByteSize;
        writeHead += sourceGridByteSize;

        // Write out blind metadata to the end of the grid
        nanovdb::GridBlindMetaData* blindMetadata = reinterpret_cast<nanovdb::GridBlindMetaData*>(writeHead);
        blindMetadata->mDataOffset = int64_t(sizeof(nanovdb::GridBlindMetaData));
        blindMetadata->mValueCount = paddedBlindDataSizes[bi]; // Number of bytes
        blindMetadata->mValueSize = 1;                         // 1 byte per value
        blindMetadata->mSemantic = nanovdb::GridBlindDataSemantic::Unknown;
        blindMetadata->mDataClass = nanovdb::GridBlindDataClass::Unknown;
        blindMetadata->mDataType = nanovdb::GridType::Unknown;
        const std::string fvdbBlindName = "fvdb_jdata" + TorchScalarTypeToStr(cpuData.scalar_type());
        setFixedSizeStringBuf(blindMetadata->mName, nanovdb::GridBlindMetaData::MaxNameSize, fvdbBlindName, "blind metadata name");
        TORCH_CHECK(blindMetadata->isValid(), "Invalid blind metadata");
        writeHead += sizeof(nanovdb::GridBlindMetaData);

        // i^th jdata entry in the jagged tensor
        JaggedTensor dataBi = cpuData.index({bi});
        TORCH_CHECK(dataBi.is_contiguous(), "Jagged tensor must be contiguous");

        // Write the shape of bi^th jdata tensor so we can load it with the same shape it was saved with
        *reinterpret_cast<int64_t*>(writeHead) = (int64_t) dataBi.dim();
        writeHead += sizeof(int64_t);
        for (int di = 0; di < dataBi.dim(); di += 1) {
            *reinterpret_cast<int64_t*>(writeHead) = (int64_t) dataBi.size(di);
            writeHead += sizeof(int64_t);
        }

        // Copy the bi^th jdata tensor as blind data to the buffer
        const int64_t jdataSize = dataBi.jdata().numel() * dataBi.jdata().element_size();
        TORCH_CHECK(dataBi.jdata().is_contiguous(), "Jagged tensor must be contiguous");
        TORCH_CHECK(dataBi.device().is_cpu(), "Jagged tensor must be on CPU");
        memcpy((void*) writeHead, (void*) dataBi.jdata().data_ptr(), jdataSize);
        writeHead += jdataSize;
        writeHead += blindDataPadding[bi];  // Add padding to make sure we're 32 byte aligned
    }

    // Synchronize cuda stream if we just did a bunch of GPU -> CPU transfers
    if (isCuda) {
        at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(gridBatch.device().index());
        cudaStreamSynchronize(defaultStream.stream());
    }

    // Write the grid to disk
    nanovdb::GridHandle<nanovdb::HostBuffer> writeHandle(std::move(writeBuf));
    nanovdb::io::writeGrid(path, writeHandle, codec, verbose);
}

nanovdb::GridHandle<nanovdb::HostBuffer>
toNVDB(const GridBatch& gridBatch,
       const torch::optional<JaggedTensor> maybeData,
       const torch::optional<StringOrListOfStrings> maybeNames) {

    // Get optional names
    std::vector<std::string> names;
    if (maybeNames.has_value())
    {
        names = maybeNames.value().value();
        TORCH_CHECK_VALUE(names.size() == 0 || names.size() == (size_t)gridBatch.grid_count(),
                          "Invalid parameter for names, must be empty or a list of the same length as the batch size. Got " + std::to_string(names.size()) + " names for batch size " + std::to_string(gridBatch.grid_count()));
    }

    if (maybeData.has_value())
    {
        return maybeConvertToStandardNanovdbGrid(gridBatch, maybeData.value(), names);
    }
    else
    {
        return getIndexGrid(gridBatch, names);
    }
}

void saveNVDB(const std::string& path,
              const GridBatch& gridBatch,
              const torch::optional<JaggedTensor> maybeData,
              const torch::optional<StringOrListOfStrings> maybeNames,
              bool compressed,
              bool verbose) {

    // Which Codec to use for saving
    nanovdb::io::Codec codec = compressed ? nanovdb::io::Codec::BLOSC : nanovdb::io::Codec::NONE;

    // Get optional names
    std::vector<std::string> names;
    if (maybeNames.has_value()) {
        names = maybeNames.value().value();
        TORCH_CHECK_VALUE(names.size() == 0 || names.size() == (size_t) gridBatch.grid_count(),
                          "Invalid parameter for names, must be empty or a list of the same length as the batch size. Got "
                          + std::to_string(names.size()) + " names for batch size " + std::to_string(gridBatch.grid_count()));
    }

    JaggedTensor data;
    if (maybeData.has_value()) {
        data = maybeData.value();
    } else {
        saveIndexGrid(path, gridBatch, names, codec, verbose);
        return;
    }

    TORCH_CHECK_VALUE(data.jdata().ndimension() >= 1, "Invalid jagged data shape in save_nvdb");
    TORCH_CHECK_VALUE(gridBatch.total_voxels() == data.jdata().size(0), "Invalid jagged data shape in save_nvdb. Must match number of voxels");
    TORCH_CHECK_VALUE(gridBatch.device() == data.device(), "Device should match between grid batch and data");

    // Heuristically determine if we can use a standard nanovdb grid (e.g. vec3f, float, vec3i, etc...) to store the data
    // If so, we save such a grid -- otherwise we save an index grid with custom blind data
    if (maybeSaveStandardNanovdbGrid(path, gridBatch, data, names, codec, verbose)) {
        return;
    } else {
        // If we didn't manage to save a standard nanovdb grid, just save a tensor grid with blind data
        saveIndexGridWithBlindData(path, gridBatch, data, names, codec, verbose);
    }
}


} // namespace io
} // namespace detail
} // namespace fvdb
