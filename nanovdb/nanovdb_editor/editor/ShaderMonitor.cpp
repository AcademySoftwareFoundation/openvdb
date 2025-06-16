// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/ShaderMonitor.cpp

    \author Petra Hapalova

    \brief
*/

#include "ShaderMonitor.h"
#include "nanovdb_editor/putil/Shader.hpp"

#include <filesystem>
#include <regex>
#include <chrono>
#include <thread>
#include <iostream>

namespace pnanovdb_editor
{
    static const std::string shaderExtensions = ".*\\.(slang|hlsl)$";

    void ShaderMonitor::addPath(const std::string& path, ShaderCallback callback)
    {
        std::string resolvedPath = pnanovdb_shader::resolveSymlink(path).string();
        if (watchers.find(resolvedPath) == watchers.end())
        {
            std::basic_regex<char> regexPattern(shaderExtensions);
            watchers[resolvedPath] = std::make_unique<filewatch::FileWatch<std::string>>(path, regexPattern,
                [this, path, callback](const std::string& filename, const filewatch::Event changeType)
                {
                    // runs on a worker thread
                    std::filesystem::path filePath = std::filesystem::path(path) / std::filesystem::path(filename);
                    std::string filePathStr = filePath.string();
                    if (changeType == filewatch::Event::modified)
                    {
                        auto now = std::chrono::steady_clock::now();
                        auto it = lastEventTime.find(filePathStr);
                        if (it != lastEventTime.end())
                        {
                            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second).count();
                            if (duration < 500)
                            {
                                // ignore events within 500 milliseconds
                                return;
                            }
                        }
                        lastEventTime[filePathStr] = now;

                        std::filesystem::file_time_type lastWriteTime = std::filesystem::last_write_time(filePathStr);
                        auto systemNow = std::chrono::system_clock::now();

                        // Convert file time to system time
#ifdef _WIN32
                        // Windows file time epoch is Jan 1, 1601, system clock is Jan 1, 1970
                        // Difference is 11644473600 seconds
                        constexpr auto epochDiff = std::chrono::seconds(11644473600);
                        auto fileTimeAdjusted = lastWriteTime.time_since_epoch() - epochDiff;
                        auto fileSystemTime = std::chrono::system_clock::time_point(
                            std::chrono::duration_cast<std::chrono::system_clock::duration>(fileTimeAdjusted)
                        );
#else
                        // For non-Windows systems, assume file_time_clock is compatible with system_clock
                        auto fileSystemTime = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                            std::chrono::system_clock::time_point(lastWriteTime.time_since_epoch())
                        );
#endif
                        auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(systemNow - fileSystemTime).count();
                        if (timeDiff > 500)
                        {
                            // ignore events which hasn't modified the file in the last 500 ms
                            return;
                        }

                        std::cout << "Shader changed: " << filePathStr << std::endl;
                        if (callback)
                        {
                            std::thread workerThread(callback, filePathStr);
                            workerThread.detach();
                        }
                    }
                });
            std::cout << "Started monitoring: " << path << std::endl;
        }

        // recursively check for symlinks in the path
        for (const auto& entry : std::filesystem::recursive_directory_iterator(resolvedPath))
        {
            if (std::filesystem::is_directory(entry.path()) && pnanovdb_shader::isSymlink(entry.path()))
            {
                std::filesystem::path linkedPath = pnanovdb_shader::resolveSymlink(entry.path().string());
                if (std::filesystem::is_directory(linkedPath))
                {
                    addPath(linkedPath.string(), callback);
                }
            }
        }
    }

    void ShaderMonitor::removePath(const std::string& path)
    {
        auto it = watchers.find(path);
        if (it != watchers.end())
        {
            watchers.erase(it);
            std::cout << "Stopped monitoring: " << path << std::endl;
        }
    }

    void monitor_shader_dir(const char* path, ShaderCallback callback)
    {
        ShaderMonitor::getInstance().addPath(path, callback);
    }
}
