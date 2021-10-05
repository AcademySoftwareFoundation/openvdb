// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file Interrupter.cc
/// @brief Houdini Interrupter

#include "Interrupter.h"
#include <UT/UT_Interrupt.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

/// Override the CustomInterrupter definitions from the OpenVDB library
/// with methods that use Houdini @c UT_Interrupt
/// @sa openvdb/util/NullInterrupter.h

CustomInterrupter::CustomInterrupter(const char* title /*=nullptr*/):
    mInterrupter(reinterpret_cast<char* const>(UTgetInterrupt())),
    mRunning(false),
    mTitle(title ? title : "")
{
}

void CustomInterrupter::start(const char* name /*=nullptr*/)
{
    if (!mRunning) {
        mRunning = true;
        reinterpret_cast<UT_Interrupt* const>(mInterrupter)->opStart(name ? name : mTitle.c_str());
    }
}

void CustomInterrupter::end()
{
    if (mRunning) {
        reinterpret_cast<UT_Interrupt* const>(mInterrupter)->opEnd();
        mRunning = false;
    }
}

bool CustomInterrupter::wasInterrupted(int percent /*=-1*/)
{
    return reinterpret_cast<UT_Interrupt* const>(mInterrupter)->opInterrupt(percent);
}

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
