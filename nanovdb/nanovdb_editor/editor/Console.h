// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/Console.h

    \author Petra Hapalova

    \brief
*/

#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif  // IMGUI_DEFINE_MATH_OPERATORS

#include <imgui.h>

#include <TextEditor.h>

#include <string>
#include <mutex>

#pragma once

namespace pnanovdb_editor
{
class Console
{
public:
    // singleton
    static Console& getInstance()
    {
        static Console instance;
        return instance;
    }

    Console();
    bool render();
    void addLog(const char* fmt, ...);

private:
    ImGuiTextBuffer buffer_;
    bool scrollToBottom_ = false;
    std::mutex logMutex_;
};
}
