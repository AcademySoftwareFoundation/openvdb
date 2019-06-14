///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#ifndef OPENVDB_UTIL_CPUTIMER_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_CPUTIMER_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <string>
#include <tbb/tick_count.h>
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
    CpuTimer(std::ostream& os = std::cerr) : mOutStream(os), mT0(tbb::tick_count::now()) {}

    /// @brief Prints message and start timer.
    ///
    /// @note Should normally be followed by a call to stop()
    CpuTimer(const std::string& msg, std::ostream& os = std::cerr) : mOutStream(os), mT0() { this->start(msg); }

    /// @brief Start timer.
    ///
    /// @note Should normally be followed by a call to milliseconds() or stop(std::string)
    inline void start() { mT0 = tbb::tick_count::now(); }

    /// @brief Print message and start timer.
    ///
    /// @note Should normally be followed by a call to stop()
    inline void start(const std::string& msg)
    {
        mOutStream << msg << " ...";
        this->start();
    }

    /// @brief Return Time difference in milliseconds since construction or start was called.
    ///
    /// @note Combine this method with start() to get timing without any outputs.
    inline double milliseconds() const
    {
        tbb::tick_count::interval_t dt = tbb::tick_count::now() - mT0;
        return 1000.0*dt.seconds();
    }

    /// @brief Return Time difference in seconds since construction or start was called.
    ///
    /// @note Combine this method with start() to get timing without any outputs.
    inline double seconds() const
    {
        tbb::tick_count::interval_t dt = tbb::tick_count::now() - mT0;
        return dt.seconds();
    }

    /// @brief This method is identical to milliseconds() - deprecated
    OPENVDB_DEPRECATED inline double delta() const { return this->milliseconds(); }

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
    std::ostream&   mOutStream;
    tbb::tick_count mT0;
};// CpuTimer

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_UTIL_CPUTIMER_HAS_BEEN_INCLUDED

// Copyright (c) DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
