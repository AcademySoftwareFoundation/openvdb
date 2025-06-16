// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/ShaderMonitor.h

    \author Petra Hapalova

    \brief
*/

#pragma once

#include <nanovdb_editor/putil/Reflect.h>
#include <FileWatch.hpp>

namespace pnanovdb_editor
{
    typedef std::function<void(const std::string&)> ShaderCallback;

    class ShaderMonitor
    {
    public:
        static ShaderMonitor& getInstance()
        {
            static ShaderMonitor instance;
            return instance;
        }

        void addPath(const std::string& path, ShaderCallback callback);
        void removePath(const std::string& path);

    private:
        ShaderMonitor() = default;
        ~ShaderMonitor() = default;

        ShaderMonitor(const ShaderMonitor&) = delete;
        ShaderMonitor& operator=(const ShaderMonitor&) = delete;

        std::unordered_map<std::string, std::unique_ptr<filewatch::FileWatch<std::string>>> watchers;
        std::unordered_map<std::string, std::chrono::steady_clock::time_point> lastEventTime;
    };

    void monitor_shader_dir(const char* path, ShaderCallback callback);
}
