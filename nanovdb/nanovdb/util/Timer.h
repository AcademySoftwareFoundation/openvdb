// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file nanovdb/util/Timer.h
///
/// @author Ken Museth
///
/// @brief A simple timing class (in case openvdb::util::CpuTimer is unavailable)

#ifndef NANOVDB_UTIL_TIMER_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_TIMER_H_HAS_BEEN_INCLUDED

#include <iostream>
#include <chrono>

namespace nanovdb {

namespace util {

class Timer
{
    std::chrono::high_resolution_clock::time_point mStart;
public:
    /// @brief Default constructor
    Timer() {}

    /// @brief Constructor that starts the timer
    /// @param msg string message to be printed when timer is started
    /// @param os output stream for the message above
    Timer(const std::string &msg, std::ostream& os = std::cerr) {this->start(msg, os);}

    /// @brief Start the timer
    /// @param msg string message to be printed when timer is started
    /// @param os output stream for the message above
    void start(const std::string &msg, std::ostream& os = std::cerr)
    {
        os << msg << " ... " << std::flush;
        mStart = std::chrono::high_resolution_clock::now();
    }

    /// @brief elapsed time (since start) in miliseconds
    template <typename AccuracyT = std::chrono::milliseconds>
    auto elapsed()
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<AccuracyT>(end - mStart).count();
    }

    /// @brief stop the timer
    /// @tparam AccuracyT Template parameter defining the accuracy of the reported times
    /// @param os output stream for the message above
    template <typename AccuracyT = std::chrono::milliseconds>
    void stop(std::ostream& os = std::cerr)
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<AccuracyT>(end - mStart).count();
        os << "completed in " << diff;
        if (std::is_same<AccuracyT, std::chrono::microseconds>::value) {// resolved at compile-time
            os << " microseconds" << std::endl;
        } else if (std::is_same<AccuracyT, std::chrono::milliseconds>::value) {
            os << " milliseconds" << std::endl;
        } else if (std::is_same<AccuracyT, std::chrono::seconds>::value) {
            os << " seconds" << std::endl;
        } else {
            os << " unknown time unit" << std::endl;
        }
    }

    /// @brief stop and start the timer
    /// @tparam AccuracyT Template parameter defining the accuracy of the reported times
    /// @param msg string message to be printed when timer is started
    /// @param os output stream for the message above
    template <typename AccuracyT = std::chrono::milliseconds>
    void restart(const std::string &msg, std::ostream& os = std::cerr)
    {
        this->stop<AccuracyT>();
        this->start(msg, os);
    }
};// Timer

}// namespace util

using CpuTimer [[deprecated("Use nanovdb::util::Timer instead")]] = util::Timer;

} // namespace nanovdb

#endif // NANOVDB_UTIL_TIMER_HAS_BEEN_INCLUDED
