// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "NullInterrupter.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

// default implementation of custom interrupter is a noop

CustomInterrupter::CustomInterrupter(const char*) { }
void CustomInterrupter::start(const char*) { }
void CustomInterrupter::end() { }
bool CustomInterrupter::wasInterrupted(int) { return false; }

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
