// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_TORCHDEVICEBUFFER_H
#define FVDB_DETAIL_TORCHDEVICEBUFFER_H

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h> // for BufferTraits

#include <torch/types.h>

#include <cuda_runtime_api.h>

namespace fvdb {
namespace detail {

// ----------------------------> TorchDeviceBuffer <--------------------------------------

/// @brief Simple memory buffer using un-managed pinned host memory when compiled with NVCC.
///        Obviously this class is making explicit used of CUDA so replace it with your own memory
///        allocator if you are not using CUDA.
/// @note  While CUDA's pinned host memory allows for asynchronous memory copy between host and
/// device
///        it is significantly slower then cached (un-pinned) memory on the host.
class TorchDeviceBuffer {
    uint64_t      mSize; // total number of bytes for the NanoVDB grid.
    uint8_t      *mData; // raw buffer for the NanoVDB grid.
    torch::Device mDevice{ torch::kCPU };

  public:
    /// @brief Default constructor initializes a buffer with the given size and device specified by
    /// host and deviceIndex.
    /// @note This has a weird API because it has to match other buffer classes in nanovdb like
    /// nanovdb::HostBuffer
    /// @param size The size (in bytes to allocate for this buffer)
    /// @param device Specifies the device to use for the buffer
    TorchDeviceBuffer(uint64_t size = 0, const torch::Device &device = torch::kCPU);

    /// @brief Disallow copy-construction
    TorchDeviceBuffer(const TorchDeviceBuffer &) = delete;

    /// @brief Move copy-constructor
    TorchDeviceBuffer(TorchDeviceBuffer &&other) noexcept;

    /// @brief Disallow copy assignment operation
    TorchDeviceBuffer &operator=(const TorchDeviceBuffer &) = delete;

    /// @brief Move copy assignment operation
    TorchDeviceBuffer &operator=(TorchDeviceBuffer &&other) noexcept;

    /// @brief Destructor frees memory on specified device
    ~TorchDeviceBuffer();

    /// @brief Returns the device used by this buffer
    /// @return The device used by this buffer
    const torch::Device &device() const;

    /// @brief Moves the buffer to the specified device
    void to(const torch::Device &device);

    /// @brief Returns a pointer to the CPU memory buffer managed by this allocator if the device is
    /// torch::kCPU, nullptr otherwise.
    uint8_t *data() const;

    /// @brief Returns a pointer to the GPU memory buffer managed by this allocator if the device is
    /// torch::kCUDA, nullptr otherwise.
    uint8_t *deviceData() const;

    /// @brief Returns the size in bytes of the raw memory buffer managed by this allocator.
    uint64_t size() const;

    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    bool empty() const;

    /// @copydoc empty()
    bool isEmpty() const;

    /// @brief De-allocate all memory managed by this allocator and set all pointer to NULL
    void clear();

    /// @brief Static factory method that return an instance of this buffer
    /// @param size byte size of buffer to be initialized
    /// @param guide this argument is there to match the signature of the other create() methods
    /// (e.g. nanovdb::HostBuffer) and to provide a way to specify the device to be used for the
    /// buffer. i.e. if guide is non-null, the created buffer will be on the same device as guide!
    /// note you must also set the device argument to match the guide buffer device
    /// @param device Device index for the buffer. If you passed in a guide buffer, then this must
    /// match the device of the guide buffer!
    /// @return An instance of this class using move semantics
    static TorchDeviceBuffer create(uint64_t size, const TorchDeviceBuffer *guide = nullptr,
                                    int device = cudaCpuDeviceId, void *stream = nullptr);

}; // TorchDeviceBuffer class

} // namespace detail
} // namespace fvdb

namespace nanovdb {
template <> struct BufferTraits<fvdb::detail::TorchDeviceBuffer> {
    static const bool hasDeviceDual = true;
};

template <>
template <>
GridHandle<fvdb::detail::TorchDeviceBuffer>
GridHandle<fvdb::detail::TorchDeviceBuffer>::copy<fvdb::detail::TorchDeviceBuffer>(
    const fvdb::detail::TorchDeviceBuffer &guide) const;

} // namespace nanovdb

#endif // FVDB_DETAIL_TORCHDEVICEBUFFER_H
