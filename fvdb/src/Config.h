// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
#pragma once


namespace fvdb {

class Config {

public:
    Config();

    void setUltraSparseAcceleration(bool enabled);
    bool ultraSparseAccelerationEnabled() const;

    void setPendanticErrorChecking(bool enabled);
    bool pendanticErrorCheckingEnabled() const;

    static Config& global();

private:
    bool mUltraSparseAcceleration = false;
    bool mPendanticErrorChecking = false;
};

} // namespace fvdb