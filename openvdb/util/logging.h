///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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

#ifndef OPENVDB_UTIL_LOGGING_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_LOGGING_HAS_BEEN_INCLUDED

#ifndef OPENVDB_USE_LOG4CPLUS

/// Log an info message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_INFO(message)
/// Log a warning message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_WARN(message)           do { std::cerr << message << std::endl; } while (0);
/// Log an error message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_ERROR(message)          do { std::cerr << message << std::endl; } while (0);
/// Log a fatal error message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_FATAL(message)          do { std::cerr << message << std::endl; } while (0);
/// In debug builds only, log a debugging message of the form '<TT>someVar << "text" << ...</TT>'.
#define OPENVDB_LOG_DEBUG(message)
/// @brief Log a debugging message in both debug and optimized builds.
/// @warning Don't use this in performance-critical code.
#define OPENVDB_LOG_DEBUG_RUNTIME(message)

#else // ifdef OPENVDB_USE_LOG4CPLUS

#include <log4cplus/logger.h>
#include <log4cplus/loglevel.h>
#include <sstream>

#define OPENVDB_LOG(level, message) \
    do { \
        log4cplus::Logger _log = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("main")); \
        if (_log.isEnabledFor(log4cplus::level##_LOG_LEVEL)) { \
            std::ostringstream _buf; \
            _buf << message; \
            _log.forcedLog(log4cplus::level##_LOG_LEVEL, _buf.str(), __FILE__, __LINE__); \
        } \
    } while (0);

#define OPENVDB_LOG_INFO(message)           OPENVDB_LOG(INFO, message)
#define OPENVDB_LOG_WARN(message)           OPENVDB_LOG(WARN, message)
#define OPENVDB_LOG_ERROR(message)          OPENVDB_LOG(ERROR, message)
#define OPENVDB_LOG_FATAL(message)          OPENVDB_LOG(FATAL, message)
#ifdef DEBUG
#define OPENVDB_LOG_DEBUG(message)          OPENVDB_LOG(DEBUG, message)
#else
#define OPENVDB_LOG_DEBUG(message)
#endif
#define OPENVDB_LOG_DEBUG_RUNTIME(message)  OPENVDB_LOG(DEBUG, message)

#endif // OPENVDB_USE_LOG4CPLUS

#endif // OPENVDB_UTIL_LOGGING_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
