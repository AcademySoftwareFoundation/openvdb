// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_GRIDBATCHIMPL_H
#define FVDB_DETAIL_GRIDBATCHIMPL_H

#include "TorchDeviceBuffer.h"
#include "VoxelCoordTransform.h"
#include "utils/Utils.h"

#include <JaggedTensor.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>

#include <torch/all.h>

#include <vector>

#if !defined(__CUDACC__) && !defined(__restrict__)
#define __restrict__
#endif

namespace fvdb {
namespace detail {

class GridBatchImpl : public torch::CustomClassHolder {
  public:
    // Metadata about a single grid in the batch
    struct GridMetadata {
        uint32_t version = 1;   // Version of this struct

        int64_t mCumLeaves = 0; // Cumulative number of leaf nodes in the batch up to this grid
        int64_t mCumVoxels = 0; // Cumulative number of voxels in the batch up to this grid
        uint64_t mCumBytes = 0; // Cumulative number of bytes in the buffer of grids up to this grid
        VoxelCoordTransform mPrimalTransform; // Primal Transform of this grid (i.e. transform which
                                              // aligns origin with voxel center)
        VoxelCoordTransform mDualTransform;   // Dual Transform of this grid (i.e. transform which
                                              // aligns origin with voxel corner)
        nanovdb::Vec3d mVoxelSize;            // Size of a single voxel in world space
        uint32_t mNumLeaves;                  // Number of leaf nodes in this grid
        int64_t mNumVoxels;                   // Number of voxels in this grid
        uint64_t mNumBytes;                   // Number of bytes in the buffer of this grid
        nanovdb::CoordBBox mBBox;             // Bounding box of this grid

        __hostdev__ nanovdb::Vec3d
        voxelOrigin() const {
            return mPrimalTransform.applyInv<double>(0., 0., 0.);
        }

        __hostdev__ void
        setTransform(const nanovdb::Vec3d &voxSize, const nanovdb::Vec3d &voxOrigin) {
            mVoxelSize = voxSize;
            voxelTransformForSizeAndOrigin(voxSize, voxOrigin, mPrimalTransform, mDualTransform);
        }
    };

    // Metadata about the whole batch
    struct GridBatchMetadata {
        uint32_t version = 1; // Version of this struct

        // Total number of leaf nodes accross all grids
        int64_t mTotalLeaves = 0;

        // Total number of voxels accross all grids
        int64_t mTotalVoxels = 0;

        // Maximum number of voxels in any grid. Used to set thread count
        int64_t mMaxVoxels = 0;

        // Maximum number of leaf nodes in any grid. Used to set thread count
        uint32_t mMaxLeafCount = 0;

        // Bounding box enclosing all the grids in the batch
        nanovdb::CoordBBox mTotalBBox;

        // Is this a mutable grid?
        bool mIsMutable = false;

        // Is this grid contiguous
        bool mIsContiguous = true;
    };

  private:
    // Metadata for each grid in the batch. There is a seperate host and device version of these.
    // The caller of this class sets the host version and is responsible for syncing the device
    // version with the host version by calling syncMetadataToDeviceIfCUDA
    GridMetadata *mHostGridMetadata{nullptr};   // CPU only
    GridMetadata *mDeviceGridMetadata{nullptr}; // CUDA only
    int64_t mBatchSize{0};

    GridBatchMetadata mBatchMetadata;           // Metadata about the whole batch

    std::shared_ptr<nanovdb::GridHandle<TorchDeviceBuffer>> mGridHdl; // NanoVDB grid handle
    torch::Tensor mLeafBatchIndices; // Indices of leaf nodes in the batch shape = [total_leafs]
    torch::Tensor mBatchOffsets;     // Batch indices for grid (ignores disabled)
    torch::Tensor mListIndices; // List indices for grid (same as JaggedTensor, ignores disabled)

    // Write back changes to host metadata to the device if we're a cuda handle
    void syncMetadataToDeviceIfCUDA(bool blocking);

    inline int64_t
    negativeToPositiveIndexWithRangecheck(int64_t bi) const {
        if (bi < 0) {
            bi += batchSize();
        }
        TORCH_CHECK_INDEX(bi >= 0 && bi < batchSize(),
                          "Batch index ",
                          bi,
                          " is out of range for grid batch of size " + std::to_string(batchSize()));
        return static_cast<int64_t>(bi);
    }

    void recomputeBatchOffsets();

    template <typename Indexable>
    c10::intrusive_ptr<GridBatchImpl> indexInternal(const Indexable &idx, int64_t size) const;

  public:
    template <typename GridType> class Accessor {
        const GridBatchImpl::GridMetadata *__restrict__ mMetadata = nullptr; // 8 bytes
        const nanovdb::NanoGrid<GridType> *__restrict__ mGridPtr  = nullptr; // 8 bytes
        fvdb::JIdxType *__restrict__ mLeafBatchIndices            = nullptr; // 8 bytes
        int64_t mTotalVoxels                                      = 0;       // 8 bytes
        int64_t mTotalLeaves                                      = 0;       // 8 bytes
        int64_t mMaxVoxels                                        = 0;       // 8 bytes
        uint32_t mMaxLeafCount                                    = 0;       // 4 bytes
        int64_t mGridCount                                        = 0;       // 8 bytes

      private:
        __hostdev__ inline int64_t
        negativeToPositiveIndexWithRangecheck(int64_t bi) const {
            if (bi < 0) {
                bi += batchSize();
            }
            assert(bi >= 0 && bi < batchSize());
            return static_cast<int64_t>(bi);
        }

      public:
        Accessor(const GridBatchImpl::GridMetadata *metadata,
                 const nanovdb::NanoGrid<GridType> *gridPtr,
                 fvdb::JIdxType *leafBatchIndices,
                 int64_t totalVoxels,
                 int64_t totalLeaves,
                 int64_t maxVoxels,
                 uint32_t maxLeafCount,
                 int64_t gridCount)
            : mMetadata(metadata), mGridPtr(gridPtr), mLeafBatchIndices(leafBatchIndices),
              mTotalVoxels(totalVoxels), mTotalLeaves(totalLeaves), mMaxVoxels(maxVoxels),
              mMaxLeafCount(maxLeafCount), mGridCount(gridCount) {}

        __hostdev__ const nanovdb::NanoGrid<GridType> *
        grid(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return reinterpret_cast<const nanovdb::NanoGrid<GridType> *>(
                reinterpret_cast<const char *>(mGridPtr) + mMetadata[bi].mCumBytes);
        }

        __hostdev__ nanovdb::CoordBBox
        bbox(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return grid(bi)->tree().bbox();
        }

        __hostdev__ nanovdb::CoordBBox
        dualBbox(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            nanovdb::CoordBBox dualBbox(bbox(bi));
            dualBbox.mCoord[1] += nanovdb::Coord(1, 1, 1);
            return dualBbox;
        }

        __hostdev__ int64_t
        batchSize() const {
            return mGridCount;
        }

        __hostdev__ int64_t
        voxelOffset(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return mMetadata[bi].mCumVoxels;
        }

        __hostdev__ int64_t
        leafOffset(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return mMetadata[bi].mCumLeaves;
        }

        __hostdev__ int64_t
        maxVoxels() const {
            return mMaxVoxels;
        }

        __hostdev__ uint32_t
        maxLeafCount() const {
            return mMaxLeafCount;
        }

        __hostdev__ int64_t
        totalVoxels() const {
            return mTotalVoxels;
        }

        __hostdev__ int64_t
        totalLeaves() const {
            return mTotalLeaves;
        }

        __hostdev__ const VoxelCoordTransform &
        primalTransform(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return mMetadata[bi].mPrimalTransform;
        }

        __hostdev__ const VoxelCoordTransform &
        dualTransform(int64_t bi) const {
            bi = negativeToPositiveIndexWithRangecheck(bi);
            return mMetadata[bi].mDualTransform;
        }

        __hostdev__ fvdb::JIdxType
        leafBatchIndex(int64_t leaf_idx) const {
            return mLeafBatchIndices[leaf_idx];
        }
    };

    template <typename GridType>
    Accessor<GridType>
    hostAccessor() const {
        TORCH_CHECK(!isEmpty(), "Cannot access empty grid");
        TORCH_CHECK(mGridHdl->template grid<GridType>(), "Failed to get host grid pointer");
        Accessor<GridType> ret(mHostGridMetadata,
                               mGridHdl->template grid<GridType>(),
                               mLeafBatchIndices.data_ptr<fvdb::JIdxType>(),
                               mBatchMetadata.mTotalVoxels,
                               mBatchMetadata.mTotalLeaves,
                               mBatchMetadata.mMaxVoxels,
                               mBatchMetadata.mMaxLeafCount,
                               mBatchSize);
        return ret;
    }

    template <typename GridType>
    Accessor<GridType>
    deviceAccessor() const {
        TORCH_CHECK(!isEmpty(), "Cannot access empty grid");
        TORCH_CHECK(device().is_cuda(), "Cannot access device accessor on non-CUDA device");
        TORCH_CHECK(mGridHdl->template deviceGrid<GridType>(), "Failed to get device grid pointer");
        Accessor<GridType> ret(mDeviceGridMetadata,
                               mGridHdl->template deviceGrid<GridType>(),
                               mLeafBatchIndices.data_ptr<fvdb::JIdxType>(),
                               mBatchMetadata.mTotalVoxels,
                               mBatchMetadata.mTotalLeaves,
                               mBatchMetadata.mMaxVoxels,
                               mBatchMetadata.mMaxLeafCount,
                               mBatchSize);
        return ret;
    }

    GridBatchImpl() = default;

    GridBatchImpl(const torch::Device &device, bool isMutable);

    GridBatchImpl(nanovdb::GridHandle<TorchDeviceBuffer> &&gridHdl,
                  const std::vector<nanovdb::Vec3d> &voxelSizes,
                  const std::vector<nanovdb::Vec3d> &voxelOrigins);

    GridBatchImpl(nanovdb::GridHandle<TorchDeviceBuffer> &&gridHdl,
                  const nanovdb::Vec3d &globalVoxelSize,
                  const nanovdb::Vec3d &globalVoxelOrigin);

    ~GridBatchImpl();

    // Cannot move make copies of this handle. There is only one owner of the underlying buffer.
    // This class should only be created and copied through c10::intrusive_ptr
    GridBatchImpl &operator=(GridBatchImpl &&other) = delete;
    GridBatchImpl(GridBatchImpl &&other)            = delete;
    GridBatchImpl(GridBatchImpl &other)             = delete;
    GridBatchImpl &operator=(GridBatchImpl &other)  = delete;

    torch::Tensor voxelOffsets(bool ignoreDisabledVoxels) const;

    torch::Tensor jlidx(bool ignoreDisabledVoxels = true) const;

    torch::Tensor jidx(bool ignoreDisabledVoxels) const;

    int64_t
    totalLeaves() const {
        return mBatchMetadata.mTotalLeaves;
    }

    int64_t
    totalVoxels() const {
        return mBatchMetadata.mTotalVoxels;
    }

    int64_t totalEnabledVoxels(bool ignoreDisabledVoxels) const;

    int64_t
    maxVoxelsPerGrid() const {
        return mBatchMetadata.mMaxVoxels;
    }

    int64_t
    maxLeavesPerGrid() const {
        return static_cast<int64_t>(mBatchMetadata.mMaxLeafCount);
    }

    int64_t
    batchSize() const {
        return mBatchSize;
    }

    uint64_t
    totalBytes() const {
        uint64_t sum = 0;
        for (int64_t i = 0; i < mBatchSize; ++i) {
            sum += mHostGridMetadata[i].mNumBytes;
        }
        return sum;
    }

    const nanovdb::GridHandle<TorchDeviceBuffer> &
    nanoGridHandle() const {
        return *mGridHdl;
    }

    nanovdb::GridHandle<TorchDeviceBuffer> &
    nanoGridHandleMut() const {
        return *mGridHdl;
    }

    bool
    isMutable() const {
        return mBatchMetadata.mIsMutable;
    }

    const c10::Device
    device() const {
        return mGridHdl->buffer().device();
    }

    bool
    isEmpty() const {
        return mGridHdl->buffer().isEmpty();
    }

    uint32_t
    numLeaves(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mNumLeaves;
    }

    int64_t
    numVoxels(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mNumVoxels;
    }

    int64_t
    cumVoxels(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mCumVoxels;
    }

    uint64_t
    numBytes(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mNumBytes;
    }

    uint64_t
    cumBytes(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mCumBytes;
    }

    const VoxelCoordTransform &
    primalTransform(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mPrimalTransform;
    }

    const VoxelCoordTransform &
    dualTransform(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mDualTransform;
    }

    void
    gridVoxelSizesAndOrigins(std::vector<nanovdb::Vec3d> &outVoxelSizes,
                             std::vector<nanovdb::Vec3d> &outVoxelOrigins) const {
        outVoxelSizes.clear();
        outVoxelOrigins.clear();
        for (int64_t i = 0; i < batchSize(); ++i) {
            outVoxelSizes.emplace_back(mHostGridMetadata[i].mVoxelSize);
            outVoxelOrigins.emplace_back(mHostGridMetadata[i].voxelOrigin());
        }
    }

    const nanovdb::CoordBBox &
    totalBBox() const {
        return mBatchMetadata.mTotalBBox;
    }

    const nanovdb::CoordBBox &
    bbox(int64_t bi) const {
        checkNonEmptyGrid();
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mBBox;
    }

    const nanovdb::CoordBBox
    dualBbox(int64_t bi) const {
        bi                          = negativeToPositiveIndexWithRangecheck(bi);
        nanovdb::CoordBBox dualBbox = bbox(bi);
        dualBbox.mCoord[1] += nanovdb::Coord(1, 1, 1);
        return dualBbox;
    }

    const nanovdb::Vec3d &
    voxelSize(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].mVoxelSize;
    }

    const nanovdb::Vec3d
    voxelOrigin(int64_t bi) const {
        bi = negativeToPositiveIndexWithRangecheck(bi);
        return mHostGridMetadata[bi].voxelOrigin();
    }

    torch::Tensor worldToGridMatrix(int64_t bi) const;

    torch::Tensor gridToWorldMatrix(int64_t bi) const;

    c10::intrusive_ptr<GridBatchImpl> clone(const torch::Device &device,
                                            bool blocking = false) const;

    void
    checkNonEmptyGrid() const {
        TORCH_CHECK(!isEmpty(), "Empty grid");
    }

    void
    checkDevice(const torch::Tensor &t) const {
        torch::Device hdlDevice = mGridHdl->buffer().device();
        TORCH_CHECK(hdlDevice == t.device(),
                    "All tensors must be on the same device (" + hdlDevice.str() +
                        ") as index grid but got " + t.device().str());
    }

    void
    checkDevice(const JaggedTensor &t) const {
        torch::Device hdlDevice = mGridHdl->buffer().device();
        TORCH_CHECK(hdlDevice == t.device(),
                    "All tensors must be on the same device (" + hdlDevice.str() +
                        ") as index grid but got " + t.device().str());
    }

    JaggedTensor jaggedTensor(const torch::Tensor &data, bool ignoreDisabledVoxels) const;

    void setGlobalPrimalTransform(const VoxelCoordTransform &transform);
    void setGlobalDualTransform(const VoxelCoordTransform &transform);
    void setGlobalVoxelSize(const nanovdb::Vec3d &voxelSize);
    void setGlobalVoxelOrigin(const nanovdb::Vec3d &voxelOrigin);
    void setGlobalVoxelSizeAndOrigin(const nanovdb::Vec3d &voxelSize,
                                     const nanovdb::Vec3d &voxelOrigin);

    void setFineTransformFromCoarseGrid(const GridBatchImpl &coarseBatch,
                                        nanovdb::Coord subdivisionFactor);
    void setCoarseTransformFromFineGrid(const GridBatchImpl &fineBatch,
                                        nanovdb::Coord coarseningFactor);
    void setPrimalTransformFromDualGrid(const GridBatchImpl &dualBatch);

    void setGrid(nanovdb::GridHandle<TorchDeviceBuffer> &&gridHdl,
                 const torch::Tensor listIndices,
                 const std::vector<nanovdb::Vec3d> &voxelSizes,
                 const std::vector<nanovdb::Vec3d> &voxelOrigins,
                 bool blocking = false);

    c10::intrusive_ptr<GridBatchImpl> index(int64_t bi) const;
    c10::intrusive_ptr<GridBatchImpl> index(ssize_t start, ssize_t stop, ssize_t step) const;
    c10::intrusive_ptr<GridBatchImpl> index(const torch::Tensor &indices) const;
    c10::intrusive_ptr<GridBatchImpl> index(const std::vector<int64_t> &indices) const;
    c10::intrusive_ptr<GridBatchImpl> index(const std::vector<bool> &indices) const;

    static c10::intrusive_ptr<GridBatchImpl>
    concatenate(const std::vector<c10::intrusive_ptr<GridBatchImpl>> &elements);

    static c10::intrusive_ptr<GridBatchImpl> contiguous(c10::intrusive_ptr<GridBatchImpl> input);

    bool
    isContiguous() const {
        return mBatchMetadata.mIsContiguous;
    }

    // Return a CPU int8 tensor with this grid packed inside it
    torch::Tensor serialize() const;

    // Load a CPU int8 tensor into a grid batch handle
    static c10::intrusive_ptr<GridBatchImpl> deserialize(const torch::Tensor &serialized);

  private:
    // We're going to version serialization. These are v0
    torch::Tensor serializeV0() const;
    static c10::intrusive_ptr<GridBatchImpl> deserializeV0(const torch::Tensor &serialized);
};

template <typename GridType> using BatchGridAccessor = typename GridBatchImpl::Accessor<GridType>;

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_GRIDBATCHIMPL_H
