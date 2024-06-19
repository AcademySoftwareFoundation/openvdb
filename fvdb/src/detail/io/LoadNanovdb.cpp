#include "detail/io/IO.h"

#include <torch/all.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include "Types.h"
#include "detail/utils/Utils.h"
#include "detail/GridBatchImpl.h"


namespace fvdb {
namespace detail {
namespace io {

/// @brief Get the gridId^th grid with build type SourceGrid in a grid handle and throw an exception if the grid is none
/// @tparam GridType The build type of the grid to read
/// @param handle The grid handle to read from
/// @param gridId The index of the grid in the handle to read
/// @param bid The batch index of the grid in the handle to read (this is only used for logging)
/// @return A host pointer to the extracted grid
template <typename GridType>
const nanovdb::NanoGrid<GridType>* getGrid(const nanovdb::GridHandle<nanovdb::HostBuffer>& handle, uint32_t gridId, uint32_t bid) {
    const nanovdb::NanoGrid<GridType>* grid = handle.grid<GridType>(gridId);
    char gridTypeStr[nanovdb::strlen<nanovdb::GridType>()];
    nanovdb::toStr(gridTypeStr, handle.gridType(gridId));
    char expectedGridTypeStr[nanovdb::strlen<nanovdb::GridType>()];
    nanovdb::toStr(expectedGridTypeStr, nanovdb::toGridType<GridType>());
    TORCH_CHECK(gridId < handle.gridCount(),
                "Failed to load grid " + std::to_string(gridId) + " from handle at batch index " + std::to_string(bid) +
                std::string(". Grid index out of bounds."));
    TORCH_CHECK(grid != nullptr,
                "Failed to load grid " + std::to_string(gridId) + " from handle at batch index " + std::to_string(bid) +
                std::string(". Grid has type ") + gridTypeStr +
                std::string(", but expected ") + expectedGridTypeStr + ".");
    return grid;
}


/// @brief Set the (row) value at index rowIdx of a tensor with 2 dimensions.
///        Specialized to accept useful nanovdb types (e.g. Vec3f, Vec4f, etc...)
/// @tparam TensorAccessorT The type of tensor accessor to use (e.g. torch::TensorAccessor, torch::PackedTensorAccessor)
/// @tparam ValueT The input type of the row to write to the tensor (e.g. float, nanovdb::Vec3f, nanovdb::Vec4f)
/// @param acc The accessor to the tensor (must refer to a 2D tensor)
/// @param rowIdx The row to read from
/// @return The rowIdx^th row of the tensor casted to ValueT
template <typename TensorAccessorT, class ValueT>
inline void valueSetter(TensorAccessorT& acc, int idx, const ValueT& value) {
    acc[idx][0] = value;
}
template <class TensorAccessorT>
inline void valueSetter(TensorAccessorT& acc, int idx, const nanovdb::Vec3f& value) {
    acc[idx][0] = value[0]; acc[idx][1] = value[1]; acc[idx][2] = value[2];
}
template <class TensorAccessorT>
inline void valueSetter(TensorAccessorT& acc, int idx, const nanovdb::Vec3d& value) {
    acc[idx][0] = value[0]; acc[idx][1] = value[1]; acc[idx][2] = value[2];
}
template <class TensorAccessorT>
inline void valueSetter(TensorAccessorT& acc, int idx, const nanovdb::Vec4f& value) {
    acc[idx][0] = value[0]; acc[idx][1] = value[1]; acc[idx][2] = value[2]; acc[idx][3] = value[3];
}
template <class TensorAccessorT>
inline void valueSetter(TensorAccessorT& acc, int idx, const nanovdb::Vec4d& value) {
    acc[idx][0] = value[0]; acc[idx][1] = value[1]; acc[idx][2] = value[2]; acc[idx][3] = value[3];
}
template <class TensorAccessorT>
inline void valueSetter(TensorAccessorT& acc, int idx, const nanovdb::math::Rgba8& value) {
    acc[idx][0] = value.r(); acc[idx][1] = value.g(); acc[idx][2] = value.b(); acc[idx][3] = value.a();
}

/// @brief Return whether a nanovdb blind metadata is a valid FVDB tensor grid blind metadata,
///        and if so, what the dtype is (if any).
///        FVDB Blind data is named "fvdb_jdata<dtype>" where dtype is an optional dtype name. If no dtype is specified,
///        then the blind data just records the  size of the tensor, and the scalar type should be determinied from the
///        grid type itself (e.g. Vec3f grids will have a float32 scalar type).
/// @param blindMetadata The blind metadata to check
/// @return A tuple containing whether the blind metadata is valid, and the dtype of the tensor (or None if no dtype is specified)
std::tuple<bool, torch::optional<torch::Dtype>> isFvdbBlindData(const nanovdb::GridBlindMetaData& blindMetadata) {
    if(strncmp(blindMetadata.mName, "fvdb_jdata", 10) != 0) {
        return std::make_tuple(false, torch::nullopt);
    }

    // Check if we load the dtype name, we won't overrun the buffer
    const int64_t blindDataNameLen = strnlen(blindMetadata.mName, nanovdb::GridBlindMetaData::MaxNameSize);
    TORCH_CHECK(blindDataNameLen < nanovdb::GridBlindMetaData::MaxNameSize, "Invalid blind metadata for nanovdb grid.");

    // There's no scalar type specified -- we're just storing a size of the tensor
    if (blindDataNameLen == 10) {
        return std::make_tuple(true, torch::nullopt);
    }

    // Get the dtype of the blind data tensor
    const std::string blindDtypeName = std::string(blindMetadata.mName + 10);
    const torch::Dtype blindDtype = StringToTorchScalarType(blindDtypeName);
    return std::make_tuple(true, torch::optional<torch::Dtype>(blindDtype));
}


/// @brief Copy a source index grid (ValueIndex(Mask) or ValueOnIndex(Mask)) to a nanovdb::GridHandle<PytorchDeviceBuffer>.
///        If the source type is ValueIndex or ValueIndex mask it will be set to ValueOnIndex or ValueOnIndexMask respectively.
/// @tparam SourceGridType The type of the source grid (must be a nanovdb::ValueIndex or nanovdb::ValueIndexMask)
/// @tparam TargetGridType The type of the target grid (must be a form of index grid)
/// @param sourceGrid A host pointer to the source grid to copy
/// @return A handle to the copied grid
template <typename SourceGridType, typename TargetGridType>
nanovdb::GridHandle<PytorchDeviceBuffer> copyIndexGridToHandle(const nanovdb::NanoGrid<SourceGridType>* sourceGrid) {
    constexpr bool isSrcValueOnIndex = nanovdb::util::is_same<SourceGridType, nanovdb::ValueOnIndex>::value;
    constexpr bool isSrcValueOnIndexMask = nanovdb::util::is_same<SourceGridType, nanovdb::ValueOnIndexMask>::value;
    constexpr bool isSrcValueIndex = nanovdb::util::is_same<SourceGridType, nanovdb::ValueIndex>::value;
    constexpr bool isSrcValueIndexMask = nanovdb::util::is_same<SourceGridType, nanovdb::ValueIndexMask>::value;
    constexpr bool isTgtValueOnIndex = nanovdb::util::is_same<TargetGridType, nanovdb::ValueOnIndex>::value;
    constexpr bool isTgtValueOnIndexMask = nanovdb::util::is_same<TargetGridType, nanovdb::ValueOnIndexMask>::value;

    static_assert(isSrcValueOnIndex || isSrcValueOnIndexMask || isSrcValueIndex || isSrcValueIndexMask,
                  "Bad source type in copyIndexGridToHandle must be an Index grid type.");
    static_assert(isTgtValueOnIndex || isTgtValueOnIndexMask,
                  "Bad target type in copyIndexGridToHandle must be ValueOnIndex or ValueOnIndexMask.");
    static_assert((isTgtValueOnIndex && (isSrcValueIndex || isSrcValueOnIndex)) ||
                  (isTgtValueOnIndexMask && (isSrcValueIndexMask || isSrcValueOnIndexMask)),
                  "Bad target grid type for given source grid type in copyIndexGridToHandle. If source is a masked grid, then target must also be a masked grid.");

    const ptrdiff_t gridSize = sourceGrid->blindDataCount() > 0 ? nanovdb::util::PtrDiff(&sourceGrid->blindMetaData(0), sourceGrid) : sourceGrid->gridSize();
    PytorchDeviceBuffer buf(gridSize);
    memcpy(buf.data(), sourceGrid, gridSize);
    nanovdb::GridData* data = reinterpret_cast<nanovdb::GridData*>(buf.data());
    data->mGridCount = 1;
    data->mGridSize = gridSize;
    data->mGridClass = nanovdb::GridClass::IndexGrid;
    data->mGridType = nanovdb::toGridType<TargetGridType>();
    return nanovdb::GridHandle<PytorchDeviceBuffer>(std::move(buf));
}


/// @brief Load a nanovdb ValueOnIndex or ValueOnIndexMask grid with tensor blind metatada (GridClass = TensorGrid) into
///        an index grid of the same type stored in a PytorchDeviceBuffer) and a torch tensor of data
///        (i.e. the standard grid format for FVDB).
/// @tparam SourceGridType The type of the source grid (must be a nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @tparam TargetGridType The type of the target grid (must be a form of index grid)
/// @param sourceGrid A host pointer to the source grid to load
/// @return A tuple containing the index grid, the name of the grid, the tensor of data, the voxel size, and the voxel origin
template <class SourceGridType, class TargetGridType>
std::tuple<nanovdb::GridHandle<PytorchDeviceBuffer>, std::string, torch::Tensor, nanovdb::Vec3d, nanovdb::Vec3d>
nanovdbTensorGridToFVDBGrid(const nanovdb::NanoGrid<SourceGridType>* sourceGrid) {
    static_assert(nanovdb::util::is_same<SourceGridType, nanovdb::ValueOnIndex>::value ||
                  nanovdb::util::is_same<SourceGridType, nanovdb::ValueOnIndexMask>::value,
                  "Bad source grid type in nanovdbTensorGridToFVDBGrid. Must be ValueOnIndex or ValueOnIndexMask.");
    static_assert(nanovdb::util::is_same<TargetGridType, nanovdb::ValueOnIndex>::value ||
                  nanovdb::util::is_same<TargetGridType, nanovdb::ValueOnIndexMask>::value,
                  "Bad target grid type in nanovdbTensorGridToFVDBGrid. Must be ValueOnIndex or ValueOnIndexMask.");
    static_assert(nanovdb::util::is_same<SourceGridType, TargetGridType>::value,
                  "Mismatched source and target grid types in nanovdbTensorGridToFVDBGrid. They must be identical.");

    TORCH_CHECK(sourceGrid->gridClass() == nanovdb::GridClass::TensorGrid, "Invalid grid class: Index grids which are not saved with fVDB are not yet supported.");

    // Copy the index grid from the loaded buffer and update metadata to be consisten with FVDB
    nanovdb::GridHandle<PytorchDeviceBuffer> retHandle = copyIndexGridToHandle<SourceGridType, TargetGridType>(sourceGrid);

    // Check if this grid has FVDB blind data attached to it
    bool foundFVDB = false;
    torch::Dtype blindDtype;
    for (unsigned i = 0; i < sourceGrid->blindDataCount(); i += 1) {
        const nanovdb::GridBlindMetaData& blindMetadata = sourceGrid->blindMetaData(i);
        // Don't need to warn for grid name
        if (blindMetadata.mDataClass == nanovdb::GridBlindDataClass::GridName) {
            continue;
        }
        std::tuple<bool, torch::optional<torch::Dtype>> isFvdb = isFvdbBlindData(sourceGrid->blindMetaData(0));
        if (std::get<0>(isFvdb)) {
            TORCH_CHECK(!foundFVDB, "Internal Error: Grid has multiple FVDB blind data tensors. Only one is supported.");
            TORCH_CHECK(std::get<1>(isFvdb).has_value(), "Invalid blind metadata for nanovdb Tensor grid.");
            foundFVDB = true;
            blindDtype = std::get<1>(isFvdb).value();
        } else {
            TORCH_WARN("Grid has blind data, but it is not valid FVDB blind data. Blind data will be ignored.");
        }
    }

    // If there is no FVDB blind data, this is just an index grid, so just return an empty data tensor
    if (!foundFVDB) {
        return std::make_tuple(std::move(retHandle),
                               sourceGrid->gridName(),
                               torch::empty({0}),
                               sourceGrid->data()->mVoxelSize,
                               sourceGrid->data()->mMap.applyMap(nanovdb::Vec3d(0.0)));
    }

    // Pointer to actual blind data
    uint8_t* readHead = (uint8_t*)(sourceGrid->blindMetaData(0).blindData());

    // Read the shape of the tensor
    const int64_t ndim = *reinterpret_cast<int64_t*>(readHead);
    readHead += sizeof(int64_t);
    std::vector<int64_t> blindDataShape;
    blindDataShape.reserve(ndim);
    for (int i = 0; i < ndim; i++) {
        blindDataShape.push_back(*reinterpret_cast<int64_t*>(readHead));
        readHead += sizeof(int64_t);
    }

    // Copy the blind data tensor
    torch::Tensor retData = torch::from_blob(const_cast<uint8_t*>(readHead), blindDataShape, blindDtype).clone();

    // Load the name and the transform
    const std::string name = sourceGrid->gridName();
    const nanovdb::Vec3d voxSize = sourceGrid->mVoxelSize;
    const nanovdb::Vec3d voxOrigin = sourceGrid->mMap.applyMap(nanovdb::Vec3d(0.0));
    return std::make_tuple(std::move(retHandle), name, retData, voxSize, voxOrigin);
}

/// @brief Load a nanovdb index grid (ValueOnIndex(Mask) or ValueIndex(Mask)) into an ValueOnIndex or ValueIndex grid
///        (stored in a PytorchDeviceBuffer) and an empty tensor of data (i.e. the standard grid format for FVDB).
/// @tparam SourceGridType The type of the source grid (must not be an index grid)
/// @tparam TargetGridType The type of the target grid (must be a nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @param sourceGrid A host pointer to the source grid to load
/// @return A tuple containing the index grid, the name of the grid, the empty tensor of data, the voxel size, and the voxel origin
template <class SourceGridType, class TargetGridType>
std::tuple<nanovdb::GridHandle<PytorchDeviceBuffer>, std::string, torch::Tensor, nanovdb::Vec3d, nanovdb::Vec3d>
nanovdbIndexGridToFVDBGrid(const nanovdb::NanoGrid<SourceGridType>* sourceGrid) {
    nanovdb::GridHandle<PytorchDeviceBuffer> retHandle = copyIndexGridToHandle<SourceGridType, TargetGridType>(sourceGrid);
    const std::string name = sourceGrid->gridName();
    const nanovdb::Vec3d voxSize = sourceGrid->data()->mVoxelSize;
    const nanovdb::Vec3d voxOrigin = sourceGrid->data()->mMap.applyMap(nanovdb::Vec3d(0.0));
    return std::make_tuple(std::move(retHandle), name, torch::empty({0}), voxSize, voxOrigin);
}


/// @brief Load a nanovdb grid with scalar or vector data stored in the leaves into a ValueOnIndex grid
///        (stored in a PytorchDeviceBuffer) and a tensor of data (i.e. the standard grid format for FVDB).
/// @tparam SourceGridType The type of the source grid (must not be an index grid)
/// @tparam TargetGridType The type of the target grid (must be a nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @tparam ScalarType The scalar type of data stored in the source grid
/// @tparam DataDim The dimension of the data stored in the source grid
/// @param sourceGrid A host pointer to the source grid to load
/// @return A tuple containing the index grid, the name of the grid, the tensor of data, the voxel size, and the voxel origin
template <class SourceGridType, class ScalarType, class TargetGridType, int DataDim>
std::tuple<nanovdb::GridHandle<PytorchDeviceBuffer>, std::string, torch::Tensor, nanovdb::Vec3d, nanovdb::Vec3d>
nanovdbGridToFvdbGrid(const nanovdb::NanoGrid<SourceGridType>* sourceGrid) {
    static_assert(nanovdb::util::is_same<TargetGridType, nanovdb::ValueOnIndex>::value,
                  "Bad target type in copyIndexGridToHandle must be ValueOnIndex.");

    static_assert(!nanovdb::util::is_same<SourceGridType, nanovdb::ValueOnIndex>::value &&
                  !nanovdb::util::is_same<SourceGridType, nanovdb::ValueOnIndexMask>::value &&
                  !nanovdb::util::is_same<SourceGridType, nanovdb::ValueIndex>::value &&
                  !nanovdb::util::is_same<SourceGridType, nanovdb::ValueIndexMask>::value,
                  "Bad source type in nanovdbGridToIndexGridAndData must NOT be an Index grid type.");

    // Create the index grid for the loaded grid
    using ProxyGridT = nanovdb::tools::build::Grid<float>;
    auto proxyGrid = std::make_shared<ProxyGridT>(0.0f);
    auto proxyGridAccessor = proxyGrid->getWriteAccessor();
    for (auto it = ActiveVoxelIteratorIJKOnly<SourceGridType>(sourceGrid->tree()); it.isValid(); it++) {
        proxyGridAccessor.setValue(*it, 1.0f);
    }
    proxyGridAccessor.merge();
    nanovdb::GridHandle<PytorchDeviceBuffer> retHandle = nanovdb::tools::createNanoGrid<ProxyGridT, TargetGridType, PytorchDeviceBuffer>(*proxyGrid, 0u, false, false);
    nanovdb::NanoGrid<TargetGridType>* outGrid = retHandle.template grid<TargetGridType>();
    TORCH_CHECK(outGrid != nullptr, "Internal error: failed to get outGrid.");
    TORCH_CHECK(outGrid->gridClass() == nanovdb::GridClass::IndexGrid, "Internal error: outGrid is not an index grid.");
    TORCH_CHECK(outGrid->gridType() == nanovdb::GridType::OnIndex || outGrid->gridType() == nanovdb::GridType::OnIndexMask,
                "Internal error: outGrid is not an index grid.");

    // Load data at the voxels into a tensor
    int64_t numVox = outGrid->activeVoxelCount();
    int64_t dim = DataDim;
    torch::TensorOptions opts = torch::TensorOptions().device(torch::kCPU).dtype<ScalarType>();
    torch::Tensor outData = torch::empty({numVox, dim}, opts);
    auto outDataAcc = outData.accessor<ScalarType, 2>();
    auto sourceGridAccessor = sourceGrid->getAccessor();
    for (auto it = ActiveVoxelIterator<TargetGridType, -1>(outGrid->tree()); it.isValid(); it++) {
        valueSetter(outDataAcc, it->second, sourceGridAccessor.getValue(it->first));
    }

    // If there's extra blind data we need to load, check if any of it is FVDB blind data.
    // We use FVDB blind data in save to store the shape of the tensor so we can load it back in the same shape
    // the user saved it in. This lets us handle saving (N, 1), (1, N, 1), (N, )... shaped tensors properly.
    bool foundFVDB = false;
    for (unsigned i = 0; i < sourceGrid->blindDataCount(); i += 1) {
        const nanovdb::GridBlindMetaData& blindMetadata = sourceGrid->blindMetaData(i);

        // Don't need to warn for grid name
        if (blindMetadata.mDataClass == nanovdb::GridBlindDataClass::GridName) {
            continue;
        }

        // Otherwise, check if this is an FVDB blind data tensor
        std::tuple<bool, torch::optional<torch::Dtype>> isFvdb = isFvdbBlindData(blindMetadata);
        if (!std::get<0>(isFvdb)) {
            TORCH_WARN("Grid has blind data, but it is not valid FVDB blind data. Blind data will be ignored.");
        } else {
            TORCH_CHECK(!foundFVDB, "Internal Error: Grid has multiple FVDB blind data tensors. Only one is supported.");
            foundFVDB = true;
            TORCH_CHECK(!std::get<1>(isFvdb).has_value(),
                        "Invalid FVDB blind metadata for nanovdb grid. Should not have extra type.");

            // Pointer to actual blind data
            uint8_t* readHead = (uint8_t*)(sourceGrid->blindMetaData(0).blindData());

            // Read the shape of the tensor
            const int64_t ndim = *reinterpret_cast<int64_t*>(readHead);
            TORCH_CHECK(sourceGrid->blindMetaData(0).blindDataSize() == nanovdb::math::AlignUp<32U>(sizeof(int64_t) * (ndim + 1)),
                        "Invalid FVDB blind data for nanovdb grid. Unexpected size.");
            readHead += sizeof(int64_t);
            std::vector<int64_t> blindDataShape;
            blindDataShape.reserve(ndim);
            for (int i = 0; i < ndim; i++) {
                blindDataShape.push_back(*reinterpret_cast<int64_t*>(readHead));
                readHead += sizeof(int64_t);
            }

            outData = outData.reshape(blindDataShape);
        }
    }

    // Load the name and the transform
    const std::string name = sourceGrid->gridName();
    const nanovdb::Vec3d voxSize = sourceGrid->data()->mVoxelSize;
    const nanovdb::Vec3d voxOrigin = sourceGrid->data()->mMap.applyMap(nanovdb::Vec3d(0.0));

    return std::make_tuple(std::move(retHandle), name, outData, voxSize, voxOrigin);
}


/// @brief Load a single nanovdb grid in a nanovdb::GridHandle<nanovdb::HostBuffer> into an ValueOnIndex or ValueOnIndexMask grid
///        stored in a nanovdb::GridHandle<PytorchDeviceBuffer> as well as torch::Tensor encoding the data at the voxels
///        (i.e. the standard format for FVDB).
///        There are 3 cases:
///          1. The input grid has scalar or vector values at the leaves:
///            - Load a ValueOnIndex grid and torch::Tensor of values
///          2. The input grid is a ValueOnIndex or ValueOnIndexMask and has its grid class set to TensorGrid:
///            - Load a matching ValueOnIndex or ValueOnIndexMask grid and torch::Tensor of values corresponding to
///              the blind data (if it is present)
///          3. The input grid is an index grid (ValueIndex(Mask) or ValueOnIndex(Mask)) but doesn't have a TensorGrid class set:
///            - Load a ValueOnIndex or ValueOnIndexMask grid (depending if the input type has a mask or not) and an empty torch::Tensor of values
///
/// @param handle The grid handle to read from
/// @param gridId The index of the grid in the handle to read
/// @param bid The batch index of the grid in the handle to read (this is only used for logging)
/// @return A tuple containing the loaded index grid, the name of the grid, the tensor of data, the voxel size, and the voxel origin
std::tuple<nanovdb::GridHandle<PytorchDeviceBuffer>, std::string, torch::Tensor, nanovdb::Vec3d, nanovdb::Vec3d>
loadOneGrid(const nanovdb::GridHandle<nanovdb::HostBuffer>& handle, uint32_t gridId, uint32_t bid) {

    if (handle.gridMetaData()->gridClass() == nanovdb::GridClass::TensorGrid) {
        TORCH_CHECK(handle.gridType() == nanovdb::GridType::OnIndex || handle.gridType() == nanovdb::GridType::OnIndexMask,
                    "Invalid grid type: Tensor grids which are not saved with fVDB are not yet supported.");
        if (handle.gridType() == nanovdb::GridType::OnIndex) {
            const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* sourceGrid = getGrid<nanovdb::ValueOnIndex>(handle, gridId, bid);
            return nanovdbTensorGridToFVDBGrid<nanovdb::ValueOnIndex, nanovdb::ValueOnIndex>(sourceGrid);
        } else if (handle.gridType() == nanovdb::GridType::OnIndexMask) {
            const nanovdb::NanoGrid<nanovdb::ValueOnIndexMask>* sourceGrid = getGrid<nanovdb::ValueOnIndexMask>(handle, gridId, bid);
            return nanovdbTensorGridToFVDBGrid<nanovdb::ValueOnIndexMask, nanovdb::ValueOnIndexMask>(sourceGrid);
        }
    }

    switch (handle.gridType()) {
        case nanovdb::GridType::Float:
        {
            const nanovdb::NanoGrid<float>* sourceGrid = getGrid<float>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<float, float, nanovdb::ValueOnIndex, 1>(sourceGrid);
        }
        case nanovdb::GridType::Double:
        {
            const nanovdb::NanoGrid<double>* sourceGrid = getGrid<double>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<double, double, nanovdb::ValueOnIndex, 1>(sourceGrid);
        }
        case nanovdb::GridType::Int32:
        {
            const nanovdb::NanoGrid<int32_t>* sourceGrid = getGrid<int32_t>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<int32_t, int32_t, nanovdb::ValueOnIndex, 1>(sourceGrid);
        }
        case nanovdb::GridType::Int64:
        {
            const nanovdb::NanoGrid<int64_t>* sourceGrid = getGrid<int64_t>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<int64_t, int64_t, nanovdb::ValueOnIndex, 1>(sourceGrid);
        }
        case nanovdb::GridType::Mask:
        case nanovdb::GridType::Boolean:
        {
            const nanovdb::NanoGrid<nanovdb::ValueMask>* sourceGrid = getGrid<nanovdb::ValueMask>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<nanovdb::ValueMask, bool, nanovdb::ValueOnIndex, 1>(sourceGrid);
        }
        case nanovdb::GridType::Vec3f:
        {
            const nanovdb::NanoGrid<nanovdb::Vec3f>* sourceGrid = getGrid<nanovdb::Vec3f>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<nanovdb::Vec3f, float, nanovdb::ValueOnIndex, 3>(sourceGrid);
        }
        case nanovdb::GridType::Vec3d:
        {
            const nanovdb::NanoGrid<nanovdb::Vec3d>* sourceGrid = getGrid<nanovdb::Vec3d>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<nanovdb::Vec3d, double, nanovdb::ValueOnIndex, 3>(sourceGrid);
        }
        case nanovdb::GridType::RGBA8:
        {
            const nanovdb::NanoGrid<nanovdb::math::Rgba8>* sourceGrid = getGrid<nanovdb::math::Rgba8>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<nanovdb::math::Rgba8, uint8_t, nanovdb::ValueOnIndex, 4>(sourceGrid);
        }
        case nanovdb::GridType::Vec4f:
        {
            const nanovdb::NanoGrid<nanovdb::Vec4f>* sourceGrid = getGrid<nanovdb::Vec4f>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<nanovdb::Vec4f, float, nanovdb::ValueOnIndex, 4>(sourceGrid);
        }
        case nanovdb::GridType::Vec4d:
        {
            const nanovdb::NanoGrid<nanovdb::Vec4d>* sourceGrid = getGrid<nanovdb::Vec4d>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<nanovdb::Vec4d, double, nanovdb::ValueOnIndex, 4>(sourceGrid);
        }
        case nanovdb::GridType::Fp16:
        {
            const nanovdb::NanoGrid<nanovdb::Fp16>* sourceGrid = getGrid<nanovdb::Fp16>(handle, gridId, bid);
            return nanovdbGridToFvdbGrid<nanovdb::Fp16, torch::Half, nanovdb::ValueOnIndex, 1>(sourceGrid);
        }
        case nanovdb::GridType::Index:
        {
            const nanovdb::NanoGrid<nanovdb::ValueIndex>* sourceGrid = getGrid<nanovdb::ValueIndex>(handle, gridId, bid);
            return nanovdbIndexGridToFVDBGrid<nanovdb::ValueIndex, nanovdb::ValueOnIndex>(sourceGrid);
        }
        case nanovdb::GridType::IndexMask:
        {
            const nanovdb::NanoGrid<nanovdb::ValueIndexMask>* sourceGrid = getGrid<nanovdb::ValueIndexMask>(handle, gridId, bid);
            return nanovdbIndexGridToFVDBGrid<nanovdb::ValueIndexMask, nanovdb::ValueOnIndexMask>(sourceGrid);
        }
        case nanovdb::GridType::OnIndex:
        {
            const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* sourceGrid = getGrid<nanovdb::ValueOnIndex>(handle, gridId, bid);
            return nanovdbIndexGridToFVDBGrid<nanovdb::ValueOnIndex, nanovdb::ValueOnIndex>(sourceGrid);
        }
        case nanovdb::GridType::OnIndexMask:
        {
            const nanovdb::NanoGrid<nanovdb::ValueOnIndexMask>* sourceGrid = getGrid<nanovdb::ValueOnIndexMask>(handle, gridId, bid);
            return nanovdbIndexGridToFVDBGrid<nanovdb::ValueOnIndexMask, nanovdb::ValueOnIndexMask>(sourceGrid);
        }
        default:
            // Unhandled cases include: Int16, UInt32, Fp4, Fp8, FpN
            char gridTypeStr[nanovdb::strlen<nanovdb::GridType>()];
            nanovdb::toStr(gridTypeStr, handle.gridType());
            throw std::runtime_error(
                    std::string("Grid type not supported: ") + gridTypeStr);
    }
}

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
fromNVDB(nanovdb::GridHandle<nanovdb::HostBuffer>& handle,
         const torch::optional<fvdb::TorchDeviceOrString> maybeDevice) {
    return fromNVDB({handle}, maybeDevice);
}

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
fromNVDB(const std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>>& handles,
         const torch::optional<fvdb::TorchDeviceOrString> maybeDevice) {
    // Load the grids, data, names, voxel origins, and sizes
    std::vector<torch::Tensor> data;
    std::vector<nanovdb::GridHandle<PytorchDeviceBuffer>> grids;
    std::vector<nanovdb::Vec3d> voxSizes, voxOrigins;
    std::vector<std::string> names;
    uint32_t bid = 0;
    nanovdb::GridType lastGridType = nanovdb::GridType::Unknown;
    for (size_t handleId = 0; handleId < handles.size(); handleId += 1) {
        for (size_t gridId = 0; gridId < handles[handleId].gridCount(); gridId += 1) {
            auto gridData = loadOneGrid(handles[handleId], gridId, bid);
            grids.push_back(std::move(std::get<0>(gridData)));
            names.push_back(std::move(std::get<1>(gridData)));
            data.push_back(std::move(std::get<2>(gridData)));
            voxSizes.push_back(std::move(std::get<3>(gridData)));
            voxOrigins.push_back(std::move(std::get<4>(gridData)));

            // FVDB grid batches all share the same mutability. i.e. a grid batch consists of all
            // ValueOnIndex (immutable) grids or all ValueOnIndexMask (mutable) grids.
            // In all but two cases, we load a ValueOnIndex grid and a tensor of data:
            //   1. When the user saved a mutable Tensor grid with save
            //   2. When the user loaded a batch with a ValueOnIndexMask grid
            // If the file the list of grids the user loaded contains a mix of ValueOnIndex and ValueOnIndexMask grids,
            // then it's unclear what to do, so throw an exception.
            if (bid > 0) {
                TORCH_CHECK(lastGridType == grids.back().gridData()->mGridType,
                            "All grids in a batch must have the same mutability (i.e. all ValueOnIndex or all ValueOnIndexMask).");
            }
            lastGridType = grids.back().gridData()->mGridType;

            bid += 1;
        }
    }

    // Merge all the loaded grids into a single handle
    nanovdb::GridHandle<PytorchDeviceBuffer> resCpu = nanovdb::mergeGrids(grids);
    c10::intrusive_ptr<GridBatchImpl> ret = c10::make_intrusive<GridBatchImpl>(std::move(resCpu), voxSizes, voxOrigins);

    // Merge loaded data Tensors into a JaggedTensor
    JaggedTensor dataJagged(data);

    // Transfer the grid handle to the device the user requested
    if (maybeDevice.has_value()) {
        torch::Device toDevice = maybeDevice.value().value();
        if (toDevice != ret->device()) {
            ret = ret->clone(toDevice);
            dataJagged = dataJagged.to(toDevice);
        }
    }

    return std::make_tuple(GridBatch(ret), dataJagged, names);
}

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
loadNVDB(const std::string& path,
         const NanoVDBFileGridIdentifier& gridIdentifier,
         TorchDeviceOrString device,
         bool verbose) {

    // Load a std::vector of grid handles each containing a one grid to load
    // If the user specified specific indices or names of grid to load, use that as a filter
    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> sourceHandles;
    if (gridIdentifier.specifiesIndices()) {
        for (uint64_t index : gridIdentifier.indicesValue()) {
            try {
                sourceHandles.emplace_back(nanovdb::io::readGrid<nanovdb::HostBuffer>(path, index, verbose));
            } catch(std::runtime_error& e) {
                TORCH_CHECK_INDEX(false, "Grid id ", index, " is out of range.");
            }
        }
    } else if (gridIdentifier.specifiesNames()) {
        for (const std::string& name : gridIdentifier.namesValue()) {
            try {
                sourceHandles.emplace_back(nanovdb::io::readGrid<nanovdb::HostBuffer>(path, name, verbose));
            } catch(std::runtime_error& e) {
                TORCH_CHECK_INDEX(false, "Grid with name '", name, "' not found in file '", path, "'.");
            }
        }
    } else {
        sourceHandles = nanovdb::io::readGrids<nanovdb::HostBuffer, std::vector>(path, verbose);
    }

    return fromNVDB(sourceHandles, device);
}



} // namespace io
} // namespace detail
} // namespace fvdb
