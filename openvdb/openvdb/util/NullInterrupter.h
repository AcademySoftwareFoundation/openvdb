// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file NullInterrupter.h

#ifndef OPENVDB_UTIL_NULL_INTERRUPTER_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_NULL_INTERRUPTER_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

/// @brief Base class for interrupters
///
/// The host application calls start() at the beginning of an interruptible operation,
/// end() at the end of the operation, and wasInterrupted() periodically during the
/// operation.
/// If any call to wasInterrupted() returns @c true, the operation will be aborted.
/// @note This interrupter was not virtual in a previous implementation, so it could
/// be compiled out, however it remains important to not call wasInterrupter() too
/// frequently so as to balance performance and the ability to interrupt an operation.
struct NullInterrupter
{
    /// Default constructor
    NullInterrupter() = default;
    virtual ~NullInterrupter() = default;
    /// Signal the start of an interruptible operation.
    /// @param name  an optional descriptive name for the operation
    virtual void start(const char* name = nullptr) { (void)name; }
    /// Signal the end of an interruptible operation.
    virtual void end() { }
    /// Check if an interruptible operation should be aborted.
    /// @param percent  an optional (when >= 0) percentage indicating
    ///     the fraction of the operation that has been completed
    /// @note this method is assumed to be thread-safe.
    virtual bool wasInterrupted(int percent = -1) { (void)percent; return false; }
    /// Convenience method to return a reference to the base class from a derived class.
    virtual NullInterrupter& interrupter() final {
        return static_cast<NullInterrupter&>(*this);
    }
}; // struct NullInterrupter

/// This method is primarily for backwards-compatibility as the ability to compile out
/// the call to wasInterrupted() is no longer supported.
template <typename T>
inline bool wasInterrupted(T* i, int percent = -1) { return i && i->wasInterrupted(percent); }


} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_NULL_INTERRUPTER_HAS_BEEN_INCLUDED
