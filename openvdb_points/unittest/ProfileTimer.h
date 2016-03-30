///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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

#ifndef OPENVDB_UNITTEST_PROFILE_TIMER_HAS_BEEN_INCLUDED
#define OPENVDB_UNITTEST_PROFILE_TIMER_HAS_BEEN_INCLUDED

#include <string>
#include <tbb/tick_count.h>
#include <sstream>// for ostringstream
#include <iomanip>//for setprecision

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

// TODO: expose this as an option to the vdb_test application
//#define PROFILE

/// @brief Functionality similar to openvdb::util::CpuTimer except with prefix padding and no decimals.
/// @note Profile timing should switch to use CpuTimer if these features were introduced.
///
/// @code
///    ProfileTimer timer("algorithm 1");
///    // code to be timed goes here
///    timer.stop();
/// @endcode
class ProfileTimer
{
public:
    /// @brief Prints message and starts timer.
    ///
    /// @note Should normally be followed by a call to stop()
    ProfileTimer(const std::string& msg)
    {
        (void)msg;
#ifdef PROFILE
        // padd string to 50 characters
        std::string newMsg(msg);
        newMsg.insert(newMsg.end(), 50 - newMsg.size(), ' ');
        std::cerr << newMsg << " ... ";
#endif
        mT0 = tbb::tick_count::now();
    }

    ~ProfileTimer() { this->stop(); }

    /// Return Time diference in milliseconds since construction or start was called.
    inline double delta() const
    {
        tbb::tick_count::interval_t dt = tbb::tick_count::now() - mT0;
        return 1000.0*dt.seconds();
    }

    /// @brief Print time in milliseconds since construction or start was called.
    inline void stop() const
    {
#ifdef PROFILE
        std::stringstream ss;
        ss << std::setw(6) << ::round(this->delta());
        std::cerr << "completed in " << ss.str() << " ms\n";
#endif
    }

private:
    tbb::tick_count mT0;
};// ProfileTimer

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_UNITTEST_PROFILE_TIMER_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
