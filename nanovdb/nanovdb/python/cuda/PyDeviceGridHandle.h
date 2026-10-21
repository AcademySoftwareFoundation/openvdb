// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYDEVICEGRIDHANDLE_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYDEVICEGRIDHANDLE_HAS_BEEN_INCLUDED

#ifdef NANOVDB_USE_CUDA

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/cuda/HandleStorage.h> // for cuda::copyTo

#include <cuda_runtime_api.h>

#include <cstddef> // for std::byte, size_t
#include <cstdint> // for uint32_t, uint64_t
#include <cstdio> // for std::fprintf
#include <exception> // for std::exception
#include <stdexcept> // for std::invalid_argument, std::out_of_range, std::runtime_error
#include <string>
#include <utility> // for std::move, std::swap

namespace pynanovdb {

/// @brief The C++ object behind the Python DeviceGridHandle. It holds the same
///        grids on two sides, each possibly empty: a host side in ordinary
///        host memory, and a device side on one device. deviceUpload and
///        deviceDownload copy between them. A handle constructed from a
///        caller's pair of arrays views them instead and owns neither.
/// @details
///   - The host side is pinned in place (cudaHostRegister) at its first
///     transfer, so transfers with sync=False are asynchronous, and a handle
///     that never transfers never calls CUDA.
///   - The device side lives on the device where the grids were built or
///     first uploaded (for viewed arrays, the device of the device array).
///     Every copy and synchronization runs on that device's default stream,
///     whatever device is current.
///   - Once a side exists, later transfers copy into its storage rather than
///     reallocating it, so grids already handed to Python stay valid.
///   - A transfer left running by sync=False finishes before its storage is
///     released or replaced.
///   - CUDA errors raise std::runtime_error, except while releasing storage,
///     where they are reported on stderr.
///   - Translation units compiled by the host compiler include this header,
///     so it uses only host-callable NanoVDB API.
class DeviceGridHandle
{
public:
    using DeviceBufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    using ViewT         = nanovdb::cuda::BufferView<std::byte>;

    DeviceGridHandle() = default;

    /// @brief Takes grids built or read on the host.
    explicit DeviceGridHandle(nanovdb::GridHandle<nanovdb::HostBuffer>&& host)
        : mHost(std::move(host))
    {
    }

    /// @brief Takes grids built on the device; the host side stays empty until deviceDownload.
    explicit DeviceGridHandle(nanovdb::GridHandle<DeviceBufferT>&& device)
        : mDevice(std::move(device))
        , mDeviceId(mDevice.empty() ? -1 : deviceOf(mDevice.deviceData()))
    {
    }

    /// @brief Views a caller's host and device arrays of @a bytes each; the host
    ///        array must hold valid grids, and both must outlive the handle.
    /// @throw std::invalid_argument if @a bytes is zero, either array is not
    ///        aligned to NANOVDB_DATA_ALIGNMENT, the host array does not hold
    ///        valid grids, or the device array is not device memory
    DeviceGridHandle(void* hostData, void* deviceData, size_t bytes)
        : mExternalHost(externalHost(hostData, bytes))
        , mExternalDevice(externalView(deviceData, bytes, "cuda_t"))
        , mDeviceId(externalDevice(deviceData))
        , mExternal(true)
    {
    }

    DeviceGridHandle(const DeviceGridHandle&) = delete;
    DeviceGridHandle& operator=(const DeviceGridHandle&) = delete;

    DeviceGridHandle(DeviceGridHandle&& other) noexcept { this->swap(other); }

    DeviceGridHandle& operator=(DeviceGridHandle&& other) noexcept
    {
        DeviceGridHandle released(std::move(other));
        this->swap(released);
        return *this;
    }

    ~DeviceGridHandle() { this->release(); }

    /// @throw std::runtime_error if a pending sync=False transfer failed
    void reset()
    {
        this->synchronize("reset");
        this->release();
    }

private:
    // Metadata is the same on both sides; read it from whichever side exists.
    template<typename F>
    auto meta(F&& f) const
    {
        if (mExternal) return f(mExternalHost);
        return mHost.empty() ? f(mDevice) : f(mHost);
    }

public:
    uint64_t bufferSize() const { return this->meta([](const auto& h) { return h.bufferSize(); }); }
    bool     empty() const { return this->meta([](const auto& h) { return h.empty(); }); }
    bool     isPadded() const { return this->meta([](const auto& h) { return h.isPadded(); }); }
    uint32_t gridCount() const { return this->meta([](const auto& h) { return h.gridCount(); }); }
    uint64_t gridSize(uint32_t n) const
    {
        this->checkIndex(n);
        return this->meta([n](const auto& h) { return h.gridSize(n); });
    }
    nanovdb::GridType gridType(uint32_t n) const
    {
        this->checkIndex(n);
        return this->meta([n](const auto& h) { return h.gridType(n); });
    }

    /// @throw std::out_of_range if @a n is not the index of a grid in this handle
    void checkIndex(uint32_t n) const
    {
        if (n >= this->gridCount())
            throw std::out_of_range("DeviceGridHandle: grid index " + std::to_string(n) + " is out of range for " +
                                    std::to_string(this->gridCount()) + " grid(s)");
    }

    /// @brief The n'th grid on the host side, or nullptr if it has no host copy
    ///        or the grid is not of type ValueT.
    template<typename ValueT>
    nanovdb::NanoGrid<ValueT>* grid(uint32_t n)
    {
        return mExternal ? mExternalHost.template grid<ValueT>(n) : mHost.template grid<ValueT>(n);
    }

    /// @brief The n'th grid on the device side, or nullptr if it has no device
    ///        copy or the grid is not of type ValueT.
    template<typename ValueT>
    nanovdb::NanoGrid<ValueT>* deviceGrid(uint32_t n)
    {
        if (!mExternal) return mDevice.template deviceGrid<ValueT>(n);
        const auto* hostGrid = mExternalHost.template grid<ValueT>(n);
        if (hostGrid == nullptr) return nullptr;
        const auto offset = reinterpret_cast<const std::byte*>(hostGrid) - static_cast<const std::byte*>(mExternalHost.data());
        return reinterpret_cast<nanovdb::NanoGrid<ValueT>*>(mExternalDevice.data() + offset);
    }

    /// @brief Calls @a f with the host-side handle, for the readers and writers of host grids.
    /// @throw std::runtime_error if there is no host copy
    template<typename F>
    auto withHost(const char* caller, F&& f) const
    {
        if (mExternal) return f(mExternalHost);
        if (mHost.empty()) {
            throw std::runtime_error(std::string("DeviceGridHandle.") + caller +
                                     ": the handle has no host copy of its grids; call deviceDownload() first");
        }
        return f(mHost);
    }

    /// @brief Copies the host side to the device side, allocating the device
    ///        side on the current device the first time.
    /// @throw std::runtime_error if there is no host copy to upload, or on a CUDA error
    void upload(bool sync)
    {
        if (!mExternal && mHost.empty()) throw std::runtime_error("DeviceGridHandle.deviceUpload: the handle has no host grids to upload");
        if (mDeviceId < 0) check(cudaGetDevice(&mDeviceId), "deviceUpload");
        DeviceGuard guard(mDeviceId, "deviceUpload");
        if (mExternal) {
            check(cudaMemcpyAsync(mExternalDevice.data(), mExternalHost.data(), mExternalHost.bufferSize(), cudaMemcpyDefault, 0), "deviceUpload");
        } else {
            this->pin("deviceUpload");
            if (mDevice.empty()) {
                mDevice = nanovdb::cuda::copyTo<DeviceBufferT>(mHost, cudaStream_t(0));
            } else {
                check(cudaMemcpyAsync(mDevice.deviceData(), mHost.data(), mHost.bufferSize(), cudaMemcpyDefault, 0), "deviceUpload");
            }
        }
        this->finish(sync, "deviceUpload");
    }

    /// @brief Copies the device side to the host side, allocating the host
    ///        side the first time, which synchronizes.
    /// @throw std::runtime_error if there is no device copy to download, or on a CUDA error
    void download(bool sync)
    {
        if (!mExternal && mDevice.empty()) {
            throw std::runtime_error("DeviceGridHandle.deviceDownload: the handle has no device grids to download; call deviceUpload() first");
        }
        DeviceGuard guard(mDeviceId, "deviceDownload");
        if (mExternal) {
            check(cudaMemcpyAsync(mExternalHost.data(), mExternalDevice.data(), mExternalHost.bufferSize(), cudaMemcpyDefault, 0), "deviceDownload");
        } else if (mHost.empty()) {
            mHost = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(mDevice, cudaStream_t(0));
        } else {
            this->pin("deviceDownload");
            check(cudaMemcpyAsync(mHost.data(), mDevice.deviceData(), mDevice.bufferSize(), cudaMemcpyDefault, 0), "deviceDownload");
        }
        this->finish(sync, "deviceDownload");
    }

private:
    /// @brief Makes @a device current for its lifetime, restoring the previous one.
    class DeviceGuard
    {
        int mPrevious = -1;

    public:
        DeviceGuard(int device, const char* caller)
        {
            if (device < 0) return;
            check(cudaGetDevice(&mPrevious), caller);
            if (mPrevious != device) check(cudaSetDevice(device), caller);
        }
        ~DeviceGuard()
        {
            if (mPrevious >= 0) cudaSetDevice(mPrevious);
        }
        DeviceGuard(const DeviceGuard&) = delete;
        DeviceGuard& operator=(const DeviceGuard&) = delete;
    };

    static void check(cudaError_t err, const char* caller)
    {
        if (err == cudaSuccess) return;
        cudaGetLastError(); // clear a non-sticky error so later calls are not blamed for it
        throw std::runtime_error(std::string("DeviceGridHandle.") + caller + ": " + cudaGetErrorString(err));
    }

    static cudaPointerAttributes attributesOf(const void* ptr)
    {
        cudaPointerAttributes attributes{};
        check(cudaPointerGetAttributes(&attributes, ptr), "DeviceGridHandle");
        return attributes;
    }

    static int deviceOf(const void* ptr) { return attributesOf(ptr).device; }

    static ViewT externalView(void* data, size_t bytes, const char* name)
    {
        if (bytes == 0) throw std::invalid_argument(std::string("DeviceGridHandle: ") + name + " is empty");
        if (!nanovdb::isAligned(data)) {
            throw std::invalid_argument(std::string("DeviceGridHandle: ") + name + " must be aligned to " +
                                        std::to_string(NANOVDB_DATA_ALIGNMENT) + " bytes");
        }
        return ViewT(static_cast<std::byte*>(data), bytes);
    }

    static nanovdb::GridHandle<ViewT> externalHost(void* data, size_t bytes)
    {
        auto view = externalView(data, bytes, "cpu_t");
        try {
            return nanovdb::GridHandle<ViewT>(std::move(view));
        } catch (const std::runtime_error& e) {
            throw std::invalid_argument(std::string("DeviceGridHandle: cpu_t does not hold valid grids: ") + e.what());
        }
    }

    static int externalDevice(const void* data)
    {
        const auto attributes = attributesOf(data);
        if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged)
            throw std::invalid_argument("DeviceGridHandle: cuda_t is not device memory");
        return attributes.device;
    }

    // Pins the host side in place, once, so that copies from and to it are asynchronous.
    void pin(const char* caller)
    {
        if (mPinned || mHost.empty()) return;
        check(cudaHostRegister(mHost.data(), mHost.bufferSize(), cudaHostRegisterDefault), caller);
        mPinned = true;
    }

    // Called on the transfer's device, after the copy is issued on its default stream.
    void finish(bool sync, const char* caller)
    {
        mInFlight = !sync;
        if (sync) check(cudaStreamSynchronize(0), caller);
    }

    // Waits for a transfer left running by sync=False.
    void synchronize(const char* caller)
    {
        if (!mInFlight) return;
        mInFlight = false;
        DeviceGuard guard(mDeviceId, caller);
        check(cudaStreamSynchronize(0), caller);
    }

    void swap(DeviceGridHandle& other) noexcept
    {
        std::swap(mHost, other.mHost);
        std::swap(mDevice, other.mDevice);
        std::swap(mExternalHost, other.mExternalHost);
        std::swap(mExternalDevice, other.mExternalDevice);
        std::swap(mDeviceId, other.mDeviceId);
        std::swap(mExternal, other.mExternal);
        std::swap(mPinned, other.mPinned);
        std::swap(mInFlight, other.mInFlight);
    }

    // Waits for any pending transfer, then frees both sides on the device that owns them.
    void release() noexcept
    {
        try {
            DeviceGuard guard(mDeviceId, "release");
            if (mInFlight) check(cudaStreamSynchronize(0), "release");
            if (mPinned) check(cudaHostUnregister(mHost.data()), "release");
            mHost.reset();
            mDevice.reset();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
        }
        mInFlight = mPinned = mExternal = false;
        mDeviceId = -1;
        mExternalHost.reset();
        mExternalDevice = ViewT();
    }

    nanovdb::GridHandle<nanovdb::HostBuffer> mHost;
    nanovdb::GridHandle<DeviceBufferT>       mDevice;
    nanovdb::GridHandle<ViewT>               mExternalHost;
    ViewT                                    mExternalDevice;
    int                                      mDeviceId = -1; // device of the device side, -1 before it exists
    bool                                     mExternal = false;
    bool                                     mPinned = false; // mHost is registered with cudaHostRegister
    bool                                     mInFlight = false; // a sync=False transfer was issued on mDeviceId's default stream
};

} // namespace pynanovdb

#endif // NANOVDB_USE_CUDA

#endif // NANOVDB_CUDA_PYDEVICEGRIDHANDLE_HAS_BEEN_INCLUDED
