// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file CpuTimer.h
///
/// @author Ken Museth
///
/// @brief A simple timing class

#ifndef NANOVDB_CPU_TIMER_H_HAS_BEEN_INCLUDED
#define NANOVDB_CPU_TIMER_H_HAS_BEEN_INCLUDED

#include <iostream>
#include <chrono>

namespace nanovdb {

template <typename Accuracy = std::chrono::milliseconds>
class CpuTimer
{
    std::chrono::high_resolution_clock::time_point mStart;
public:
    CpuTimer() {}
    void start(const std::string &msg, std::ostream& os = std::cerr) { 
        os << msg << " ... " << std::flush;
        mStart = std::chrono::high_resolution_clock::now();
    }
    void restart(const std::string &msg, std::ostream& os = std::cerr) {
        this->stop(); 
        os << msg << " ... " << std::flush;
        mStart = std::chrono::high_resolution_clock::now();
    }
    void stop(std::ostream& os = std::cerr)
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<Accuracy>(end - mStart).count();
        os << "completed in " << diff;
        if (std::is_same<Accuracy, std::chrono::microseconds>::value) {// resolved at compile-time
            os << " microseconds" << std::endl;
        } else if (std::is_same<Accuracy, std::chrono::milliseconds>::value) {
            os << " milliseconds" << std::endl;
        } else if (std::is_same<Accuracy, std::chrono::seconds>::value) {
            os << " seconds" << std::endl;
        } else {
            os << " unknown time unit" << std::endl;
        }
    }
};// CpuTimer

} // namespace nanovdb

#endif // NANOVDB_CPU_TIMER_HAS_BEEN_INCLUDED
