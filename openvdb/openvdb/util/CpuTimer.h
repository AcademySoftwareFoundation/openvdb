// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_UTIL_CPUTIMER_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_CPUTIMER_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <string>
#include <chrono>
#include <iostream>// for std::cerr
#include <sstream>// for ostringstream
#include <iomanip>// for setprecision
#include "Formats.h"// for printTime

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

/// @brief Simple timer for basic profiling.
///
/// @code
///    util::CpuTimer timer;
///    // code here will not be timed!
///    timer.start("algorithm");
///    // code to be timed goes here
///    timer.stop();
/// @endcode
///
/// or to time multiple blocks of code
///
/// @code
///    util::CpuTimer timer("algorithm 1");
///    // code to be timed goes here
///    timer.restart("algorithm 2");
///    // code to be timed goes here
///    timer.stop();
/// @endcode
///
/// or to measure speedup between multiple runs
///
/// @code
///    util::CpuTimer timer("algorithm 1");
///    // code for the first run goes here
///    const double t1 = timer.restart("algorithm 2");
///    // code for the second run goes here
///    const double t2 = timer.stop();
///    std::cerr << "Algorithm 1 is " << (t2/t1)
///              << " timers faster than algorithm 2\n";
/// @endcode
///
/// or to measure multiple blocks of code with deferred output
///
/// @code
///    util::CpuTimer timer();
///    // code here will not be timed!
///    timer.start();
///    // code for the first run goes here
///    const double t1 = timer.restart();//time in milliseconds
///    // code for the second run goes here
///    const double t2 = timer.restart();//time in milliseconds
///    // code here will not be timed!
///    util::printTime(std::cout, t1, "Algorithm 1 completed in ");
///    util::printTime(std::cout, t2, "Algorithm 2 completed in ");
/// @endcode
class CpuTimer
{
public:
    /// @brief Initiate timer
    CpuTimer(std::ostream& os = std::cerr) : mOutStream(os), mT0(this->now()) {}

    /// @brief Prints message and start timer.
    ///
    /// @note Should normally be followed by a call to stop()
    CpuTimer(const std::string& msg, std::ostream& os = std::cerr) : mOutStream(os) { this->start(msg); }

    /// @brief Start timer.
    ///
    /// @note Should normally be followed by a call to milliseconds() or stop(std::string)
    inline void start() { mT0 = this->now(); }

    /// @brief Print message and start timer.
    ///
    /// @note Should normally be followed by a call to stop()
    inline void start(const std::string& msg)
    {
        mOutStream << msg << " ...";
        this->start();
    }

    /// @brief Return Time difference in microseconds since construction or start was called.
    ///
    /// @note Combine this method with start() to get timing without any outputs.
    inline int64_t microseconds() const
    {
        return (this->now() - mT0);
    }

    /// @brief Return Time difference in milliseconds since construction or start was called.
    ///
    /// @note Combine this method with start() to get timing without any outputs.
    inline double milliseconds() const
    {
        static constexpr double resolution = 1.0 / 1E3;
        return static_cast<double>(this->microseconds()) * resolution;
    }

    /// @brief Return Time difference in seconds since construction or start was called.
    ///
    /// @note Combine this method with start() to get timing without any outputs.
    inline double seconds() const
    {
        static constexpr double resolution = 1.0 / 1E6;
        return static_cast<double>(this->microseconds()) * resolution;
    }

    inline std::string time() const
    {
        const double msec = this->milliseconds();
        std::ostringstream os;
        printTime(os, msec, "", "", 4, 1, 1);
        return os.str();
    }

    /// @brief Returns and prints time in milliseconds since construction or start was called.
    ///
    /// @note Combine this method with start(std::string) to print at start and stop of task being timed.
    inline double stop() const
    {
        const double msec = this->milliseconds();
        printTime(mOutStream, msec, " completed in ", "\n", 4, 3, 1);
        return msec;
    }

    /// @brief Returns and prints time in milliseconds since construction or start was called.
    ///
    /// @note Combine this method with start() to delay output of task being timed.
    inline double stop(const std::string& msg) const
    {
        const double msec = this->milliseconds();
        mOutStream << msg << " ...";
        printTime(mOutStream, msec, " completed in ", "\n", 4, 3, 1);
        return msec;
    }

    /// @brief Re-start timer.
    /// @return time in milliseconds since previous start or restart.
    ///
    /// @note Should normally be followed by a call to stop() or restart()
    inline double restart()
    {
        const double msec = this->milliseconds();
        this->start();
        return msec;
    }

    /// @brief Stop previous timer, print message and re-start timer.
    /// @return time in milliseconds since previous start or restart.
    ///
    /// @note Should normally be followed by a call to stop() or restart()
    inline double restart(const std::string& msg)
    {
        const double delta = this->stop();
        this->start(msg);
        return delta;
    }

private:
    static int64_t now()
    {
        // steady_clock is a monotonically increasing clock designed for timing duration
        // note that high_resolution_clock is aliased to either steady_clock or system_clock
        // depending on the platform, so it is preferrable to use steady_clock
        const auto time_since_epoch =
            std::chrono::steady_clock::now().time_since_epoch();
        // cast time since epoch into microseconds (1 / 1000000 seconds)
        const auto microseconds =
            std::chrono::duration_cast<std::chrono::microseconds>(time_since_epoch).count();
        // cast to a a 64-bit signed integer as this will overflow in 2262!
        return static_cast<int64_t>(microseconds);
    }

    std::ostream&       mOutStream;
    int64_t             mT0{0};
};// CpuTimer

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_UTIL_CPUTIMER_HAS_BEEN_INCLUDED
