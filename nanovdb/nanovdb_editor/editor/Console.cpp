// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/Console.cpp

    \author Petra Hapalova

    \brief
*/

#include "Console.h"

namespace pnanovdb_editor
{
Console::Console()
{
}

bool Console::render()
{
    if (ImGui::BeginChild("ScrollingRegion", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()), false, ImGuiWindowFlags_HorizontalScrollbar))
    {
        ImGui::TextUnformatted(buffer_.begin());
        if (scrollToBottom_)
        {
            ImGui::SetScrollHereY(1.0f);
        }
    }
    ImGui::EndChild();

    return true;
}

void Console::addLog(const char* fmt, ...)
{
    std::lock_guard<std::mutex> lock(logMutex_);

    // TODO: add timestamp?
    va_list args;
    va_start(args, fmt);
    buffer_.appendfv(fmt, args);
    va_end(args);
    buffer_.append("\n");
    scrollToBottom_ = true;
}

}
