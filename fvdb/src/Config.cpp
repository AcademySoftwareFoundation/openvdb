// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "Config.h"

namespace fvdb {

Config::Config() = default;

Config &
Config::global() {
    static Config _config;
    return _config;
}

void
Config::setUltraSparseAcceleration(bool enabled) {
    mUltraSparseAcceleration = enabled;
}

bool
Config::ultraSparseAccelerationEnabled() const {
    return mUltraSparseAcceleration;
}

void
Config::setPendanticErrorChecking(bool enabled) {
    mPendanticErrorChecking = enabled;
}
bool
Config::pendanticErrorCheckingEnabled() const {
    return mPendanticErrorChecking;
}

} // namespace fvdb