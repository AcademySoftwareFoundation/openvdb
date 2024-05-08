#pragma once

#include <torch/types.h>

#include <nanovdb/HostBuffer.h> // for BufferTraits


namespace fvdb {
namespace detail {

// ----------------------------> PytorchDeviceBuffer <--------------------------------------

/// @brief Simple memory buffer using un-managed pinned host memory when compiled with NVCC.
///        Obviously this class is making explicit used of CUDA so replace it with your own memory
///        allocator if you are not using CUDA.
/// @note  While CUDA's pinned host memory allows for asynchronous memory copy between host and device
///        it is significantly slower then cached (un-pinned) memory on the host.
class PytorchDeviceBuffer
{
    uint64_t mSize; // total number of bytes for the NanoVDB grid.
    uint8_t *mCpuData, *mGpuData; // raw buffer for the NanoVDB grid.
    torch::Device mDevice = torch::Device(torch::kCPU);

    /// @brief Helper function to move this buffer to the CPU. If the buffer is on the GPU, the GPU memory will be freed.
    /// @param blocking If set to false, then memory allocations and copies are performed asynchronously
    void toCpu(bool blocking);

    /// @brief Helper function to move this buffer to the specified CUDA device.
    /// @param device The device on which the buffer should be moved to
    /// @param blocking If set to false, then memory allocations and copies are performed asynchronously
    void toCuda(torch::Device device, bool blocking);

    /// @brief Helper function to copy from the host to the device and then free the host buffer. If @c blocking is false the memory copy is asynchronous!
    ///
    /// @note This will allocate memory on the GPU/device if it is not already allocated.
    /// @note The device of this buffer must be CPU
    void copyHostToDeviceAndFreeHost(void* stream = 0, bool blocking = true);  // Delete

    /// @brief Helper function to copy from a device to the host and then free the device buffer. If @c blocking is false the memory copy is asynchronous!
    ///
    /// @note This will allocate memory on the host if it is not already allocated.
    /// @note The device of this buffer must be CPU
    void copyDeviceToHostAndFreeDevice(void* stream = 0, bool blocking = true);  // Delete

public:
    /// @brief Default constructor initializes a buffer with the given size and device specified by host and deviceIndex.
    /// @note This has a weird API because it has to match other buffer classes in nanovdb like nanovdb::HostBuffer
    /// @param size The size (in bytes to allocate for this buffer)
    /// @param data If non-null, the data pointer to use for this buffer
    /// @param host If true buffer is initialized only on the host/CPU, else on the device/GPU
    /// @param deviceIndex If host is false, then this specifies the device index to use for the buffer
    ///                    (must be set to a nonzero value when host is false)
    PytorchDeviceBuffer(uint64_t size = 0, void* data = nullptr, bool host = true, int deviceIndex = -1);

    /// @brief Disallow copy-construction
    PytorchDeviceBuffer(const PytorchDeviceBuffer&) = delete;

    /// @brief Move copy-constructor
    PytorchDeviceBuffer(PytorchDeviceBuffer&& other) noexcept;

    /// @brief Disallow copy assignment operation
    PytorchDeviceBuffer& operator=(const PytorchDeviceBuffer&) = delete;

    /// @brief Move copy assignment operation
    PytorchDeviceBuffer& operator=(PytorchDeviceBuffer&& other) noexcept;

    /// @brief Destructor frees memory on both the host and device
    ~PytorchDeviceBuffer() { this->clear(); };

    /// @brief Initialize buffer
    /// @param size byte size of buffer to be initialized
    /// @param host If true buffer is iniaialized only on the host/CPU, else on the device/GPU
    ///             The selected device will be this->device which must be a cuda device
    /// @note All existing buffers are first cleared
    /// @warning size is expected to be non-zero. Use clear() clear buffer!
    void init(uint64_t size, void* data = nullptr, bool host = true);

    /// @brief Set the device of this buffer and shuffle data around accordingly
    /// @param device The device to be used by this buffer (if CUDA, must specify a device index)
    /// @param blocking If true the memory copy is synchronous, else asynchronous
    void setDevice(const torch::Device& device, bool blocking);

    /// @brief Returns the device used by this buffer
    /// @return The device used by this buffer
    const torch::Device& device() const {
        return mDevice;
    }

    /// @brief Retuns a pointer to the raw memory buffer managed by this allocator.
    /// @warning Note that the pointer can be NULL is the allocator was not initialized!
    uint8_t* data() const { return mCpuData; }
    uint8_t* deviceData() const { return mGpuData; }

    /// @brief Returns the size in bytes of the raw memory buffer managed by this allocator.
    uint64_t size() const { return mSize; }

    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    bool empty() const { return mSize == 0 && mCpuData == nullptr && mGpuData == nullptr; }
    bool isEmpty() const { return empty(); }

    /// @brief De-allocate all memory managed by this allocator and set all pointer to NULL
    void clear();

    /// @brief Static factory method that return an instance of this buffer
    /// @param size byte size of buffer to be initialized
    /// @param guide this argument is there to match the signature of the other create() methods (e.g. nanovdb::HostBuffer)
    ///              and to provide a way to specify the device to be used for the buffer.
    ///              i.e. if guide is non-null, the created buffer will be on the same device as guide!
    ///              note you must also set the host argument to match the guide buffer device
    /// @param host If true buffer is initialized only on the host/CPU, else on the device/GPU. If you passed in a guide
    ///             buffer, then this must match the device of the guide buffer!
    /// @return An instance of this class using move semantics
    static PytorchDeviceBuffer create(uint64_t size, const PytorchDeviceBuffer* guide = nullptr, bool host = true, void* stream = nullptr);

}; // PytorchDeviceBuffer class


} // namespace detail
} // namespace fvdb


namespace nanovdb {
    template<>
    struct BufferTraits<fvdb::detail::PytorchDeviceBuffer>
    {
        static const bool hasDeviceDual = true;
    };
} // namespace nanovdb
