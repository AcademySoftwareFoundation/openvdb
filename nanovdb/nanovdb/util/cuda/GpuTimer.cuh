// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file GpuTimer.cuh
///
/// @author Ken Museth
///
/// @brief A simple GPU timing class

#ifndef NANOVDB_GPU_TIMER_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_GPU_TIMER_CUH_HAS_BEEN_INCLUDED

#include <iostream>// for std::cerr
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace nanovdb {

class GpuTimer
{
    cudaEvent_t mStart, mStop;

public:
    /// @brief Default constructor
    /// @note Starts the timer
    GpuTimer(void* stream = nullptr)
    {
        cudaEventCreate(&mStart);
        cudaEventCreate(&mStop);
        cudaEventRecord(mStart, reinterpret_cast<cudaStream_t>(stream));
    }

    /// @brief Construct and start the timer
    /// @param msg string message to be printed when timer is started
    /// @param stream CUDA stream to be timed (defaults to stream 0)
    /// @param os output stream for the message above
    GpuTimer(const std::string &msg, void* stream = nullptr, std::ostream& os = std::cerr)
    {
        os << msg << " ... " << std::flush;
        cudaEventCreate(&mStart);
        cudaEventCreate(&mStop);
        cudaEventRecord(mStart, reinterpret_cast<cudaStream_t>(stream));
    }

    /// @brief Destructor
    ~GpuTimer()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mStop);
    }

    /// @brief Start the timer
    /// @param stream CUDA stream to be timed (defaults to stream 0)
    /// @param os output stream for the message above
    void start(void* stream = nullptr)
    {
        cudaEventRecord(mStart, reinterpret_cast<cudaStream_t>(stream));
    }

    /// @brief Start the timer
    /// @param msg string message to be printed when timer is started
    /// @param stream CUDA stream to be timed (defaults to stream 0)
    /// @param os output stream for the message above
    void start(const std::string &msg, void* stream = nullptr, std::ostream& os = std::cerr)
    {
        os << msg << " ... " << std::flush;
        this->start(stream);
    }

    /// @brief Start the timer
    /// @param msg string message to be printed when timer is started
    /// @param stream CUDA stream to be timed (defaults to stream 0)
    /// @param os output stream for the message above
    void start(const char* msg, void* stream = nullptr, std::ostream& os = std::cerr)
    {
        os << msg << " ... " << std::flush;
        this->start(stream);
    }

    /// @brief elapsed time (since start) in miliseconds
    /// @param stream CUDA stream to be timed (defaults to stream 0)
    /// @return elapsed time (since start) in miliseconds
    float elapsed(void* stream = nullptr)
    {
        cudaEventRecord(mStop, reinterpret_cast<cudaStream_t>(stream));
        cudaEventSynchronize(mStop);
        float diff = 0.0f;
        cudaEventElapsedTime(&diff, mStart, mStop);
        return diff;
    }

    /// @brief stop the timer
    /// @param stream CUDA stream to be timed (defaults to stream 0)
    /// @param os output stream for the message above
    void stop(void* stream = nullptr, std::ostream& os = std::cerr)
    {
        float diff = this->elapsed(stream);
        os << "completed in " << diff << " milliseconds" << std::endl;
    }

    /// @brief stop and start the timer
    /// @param msg string message to be printed when timer is started
    /// @param os output stream for the message above
    /// @warning Remember to call start before restart
    void restart(const std::string &msg, void* stream = nullptr, std::ostream& os = std::cerr)
    {
        this->stop();
        this->start(msg, stream, os);
    }
};// GpuTimer

} // namespace nanovdb

#endif // NANOVDB_GPU_TIMER_CUH_HAS_BEEN_INCLUDED
