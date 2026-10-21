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
#include <nanovdb/cuda/PinnedResource.h>
#include <nanovdb/util/cuda/Util.h> // for cudaCheck

#include <cuda_runtime_api.h>

#include <cstddef> // for std::byte, size_t
#include <cstdint> // for uint32_t, uint64_t
#include <stdexcept> // for std::invalid_argument, std::out_of_range, std::runtime_error
#include <string>
#include <utility> // for std::exchange, std::move

namespace pynanovdb {

/// @brief The C++ object behind the Python DeviceGridHandle. It holds the same
///        grids on two sides, each possibly empty: a host side in pinned
///        memory, so transfers with sync=False stay asynchronous, and a device
///        side on the device that was current when it was first filled.
///        deviceUpload and deviceDownload copy between them. A handle
///        constructed from a caller's pair of arrays views them instead and
///        owns neither.
/// @details Once a side exists, later transfers copy into its storage rather
///          than reallocating it, so grids already handed to Python stay valid.
///          Every call that takes a host path is compiled by the host compiler,
///          so this header uses only host-callable NanoVDB API.
class DeviceGridHandle
{
public:
    using HostBufferT   = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::PinnedResource>;
    using DeviceBufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    using ViewT         = nanovdb::cuda::BufferView<std::byte>;

    DeviceGridHandle() = default;

    /// @brief Takes grids built or read on the host; they are copied into pinned memory.
    explicit DeviceGridHandle(const nanovdb::GridHandle<nanovdb::HostBuffer>& host)
        : mHost(host.template copy<HostBufferT>(HostBufferT()))
    {
    }

    /// @brief Takes grids built on the device; the host side stays empty until deviceDownload.
    explicit DeviceGridHandle(nanovdb::GridHandle<DeviceBufferT>&& device)
        : mDevice(std::move(device))
    {
    }

    /// @brief Views a caller's host and device arrays of @a bytes each; the host
    ///        array must hold valid grids, and both must outlive the handle.
    /// @throw std::invalid_argument if @a bytes is zero or either array is not
    ///        aligned to NANOVDB_DATA_ALIGNMENT
    DeviceGridHandle(void* hostData, void* deviceData, size_t bytes)
        : mExternalHost(externalView(hostData, bytes, "cpu_t"))
        , mExternalDevice(externalView(deviceData, bytes, "cuda_t"))
        , mExternal(true)
    {
    }

    DeviceGridHandle(const DeviceGridHandle&) = delete;
    DeviceGridHandle& operator=(const DeviceGridHandle&) = delete;

    DeviceGridHandle(DeviceGridHandle&& other) noexcept
        : mHost(std::move(other.mHost))
        , mDevice(std::move(other.mDevice))
        , mExternalHost(std::move(other.mExternalHost))
        , mExternalDevice(std::exchange(other.mExternalDevice, ViewT()))
        , mExternal(std::exchange(other.mExternal, false))
        , mInFlight(std::exchange(other.mInFlight, false))
    {
    }

    DeviceGridHandle& operator=(DeviceGridHandle&& other)
    {
        if (this != &other) {
            this->settle();
            mHost = std::move(other.mHost);
            mDevice = std::move(other.mDevice);
            mExternalHost = std::move(other.mExternalHost);
            mExternalDevice = std::exchange(other.mExternalDevice, ViewT());
            mExternal = std::exchange(other.mExternal, false);
            mInFlight = std::exchange(other.mInFlight, false);
        }
        return *this;
    }

    ~DeviceGridHandle() { this->settle(); }

    void reset()
    {
        this->settle();
        mHost.reset();
        mDevice.reset();
        mExternalHost.reset();
        mExternalDevice = ViewT();
        mExternal = false;
    }

private:
    static ViewT externalView(void* data, size_t bytes, const char* name)
    {
        if (bytes == 0) throw std::invalid_argument(std::string("DeviceGridHandle: ") + name + " is empty");
        if (!nanovdb::isAligned(data)) {
            throw std::invalid_argument(std::string("DeviceGridHandle: ") + name + " must be aligned to " +
                                        std::to_string(NANOVDB_DATA_ALIGNMENT) + " bytes");
        }
        return ViewT(static_cast<std::byte*>(data), bytes);
    }

    // A transfer issued with sync=False may still be reading or writing the
    // storage, so wait for it before that storage is released or replaced.
    void finish(cudaStream_t stream, bool sync)
    {
        if (sync) cudaCheck(cudaStreamSynchronize(stream));
        mInFlight = !sync;
    }
    void settle()
    {
        if (mInFlight) cudaCheck(cudaStreamSynchronize(0));
        mInFlight = false;
    }

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

    /// @brief The host-side handle, for the readers and writers of host grids.
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

    /// @brief Copies the host side to the device side on the default stream,
    ///        allocating the device side on the current device the first time.
    /// @throw std::runtime_error if there is no host copy to upload
    void upload(bool sync)
    {
        const cudaStream_t stream = 0;
        if (mExternal) {
            cudaCheck(cudaMemcpyAsync(mExternalDevice.data(), mExternalHost.data(), mExternalHost.bufferSize(), cudaMemcpyDefault, stream));
        } else if (mHost.empty()) {
            throw std::runtime_error("DeviceGridHandle.deviceUpload: the handle has no host grids to upload");
        } else if (mDevice.bufferSize() == mHost.bufferSize()) {
            cudaCheck(cudaMemcpyAsync(mDevice.deviceData(), mHost.data(), mHost.bufferSize(), cudaMemcpyDefault, stream));
        } else {
            mDevice = nanovdb::cuda::copyTo<DeviceBufferT>(mHost, stream);
        }
        this->finish(stream, sync);
    }

    /// @brief Copies the device side to the host side on the default stream,
    ///        allocating the host side the first time, which synchronizes.
    /// @throw std::runtime_error if there is no device copy to download
    void download(bool sync)
    {
        const cudaStream_t stream = 0;
        if (mExternal) {
            cudaCheck(cudaMemcpyAsync(mExternalHost.data(), mExternalDevice.data(), mExternalHost.bufferSize(), cudaMemcpyDefault, stream));
        } else if (mDevice.empty()) {
            throw std::runtime_error("DeviceGridHandle.deviceDownload: the handle has no device grids to download; call deviceUpload() first");
        } else if (mHost.bufferSize() == mDevice.bufferSize()) {
            cudaCheck(cudaMemcpyAsync(mHost.data(), mDevice.deviceData(), mDevice.bufferSize(), cudaMemcpyDefault, stream));
        } else {
            mHost = nanovdb::cuda::copyTo<HostBufferT>(mDevice, stream);
        }
        this->finish(stream, sync);
    }

private:
    nanovdb::GridHandle<HostBufferT>   mHost;
    nanovdb::GridHandle<DeviceBufferT> mDevice;
    nanovdb::GridHandle<ViewT>         mExternalHost;
    ViewT                              mExternalDevice;
    bool                               mExternal = false;
    bool                               mInFlight = false;// a sync=False transfer was issued on the default stream
};

} // namespace pynanovdb

#endif // NANOVDB_USE_CUDA

#endif // NANOVDB_CUDA_PYDEVICEGRIDHANDLE_HAS_BEEN_INCLUDED
