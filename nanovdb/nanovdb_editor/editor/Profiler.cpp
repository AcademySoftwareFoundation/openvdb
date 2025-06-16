// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/Profiler.cpp

    \author Petra Hapalova

    \brief  Profiler window for the NanoVDB editor
*/

#include "Profiler.h"

#include <imgui.h>

namespace pnanovdb_editor
{
int Profiler::s_id = 0;

Profiler& Profiler::getInstance()
{
    static Profiler instance;
    return instance;
}

bool Profiler::render(bool* update_memory_stats, float delta_time)
{
    ImGuiIO& io = ImGui::GetIO();
    //ImGui::Text("%.1f FPS", io.Framerate);
    ImGui::Text("%.3f CPU ms/frame", delta_time * 1000.0f);

    // update memory stats timer once per second
    {
        std::lock_guard<std::mutex> lock(mutex_);
        memory_stats_timer_ += delta_time;
        if (memory_stats_timer_ > 1.0f)
        {
            *update_memory_stats = true;
            memory_stats_timer_ = 0.0f;
        }
    }

    pnanovdb_compute_device_memory_stats_t stats;
    bool show_avg = false;
    std::vector<std::string> profiler_names;
    bool has_any_data = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats = memory_stats_;
        show_avg = show_averages_;

        for (const auto& device_entry : profiler_entries_)
        {
            profiler_names.push_back(device_entry.first);
            has_any_data = has_any_data || !device_entry.second.empty();
        }
    }

    // TODO: can have a table per device
    //if (ImGui::CollapsingHeader("Device", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::CollapsingHeader("Memory Usage", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::BeginTable("MemoryStatsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
            {
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Size (MB)", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Device");
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", stats.device_memory_bytes / (1024.0f * 1024.0f));

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Upload");
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", stats.upload_memory_bytes / (1024.0f * 1024.0f));

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Readback");
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", stats.readback_memory_bytes / (1024.0f * 1024.0f));

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Other");
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", stats.other_memory_bytes / (1024.0f * 1024.0f));

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Total");
                ImGui::TableNextColumn();
                ImGui::Text("%.2f",
                    (stats.device_memory_bytes +
                        stats.upload_memory_bytes +
                        stats.readback_memory_bytes +
                        stats.other_memory_bytes) / (1024.0f * 1024.0f));

                ImGui::EndTable();
            }
        }

        ImGui::Separator();

        if (has_any_data)
        {
            if (ImGui::Button(profiler_paused_ ? "Resume" : "Pause"))
            {
                std::lock_guard<std::mutex> lock(mutex_);
                profiler_paused_ = !profiler_paused_;
            }

            ImGui::SameLine();
            if (ImGui::Button("Clear"))
            {
                std::lock_guard<std::mutex> lock(mutex_);
                profiler_capture_ids_.clear();
                profiler_entries_.clear();
                label_history_.clear();
            }

            const char* label_averages = "Show Averages";

            float window_width = ImGui::GetWindowWidth();
            float checkbox_width = ImGui::CalcTextSize(label_averages).x + ImGui::GetStyle().FramePadding.x * 2 + 20.0f;
            ImGui::SameLine(window_width - checkbox_width);

            bool temp_show_averages = show_avg;
            if (ImGui::Checkbox(label_averages, &temp_show_averages))
            {
                std::lock_guard<std::mutex> lock(mutex_);
                show_averages_ = temp_show_averages;
            }

            for (std::string profile_name : profiler_names)
            {
                if (ImGui::CollapsingHeader(profile_name.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
                {
                    std::map<std::string, ProfilerEntry> entries_copy;
                    std::unordered_map<std::string, std::vector<pnanovdb_compute_profiler_entry_t>> history_copy;

                    pnanovdb_uint64_t capture_id = 0u;
                    {
                        std::lock_guard<std::mutex> lock(mutex_);

                        if (profiler_capture_ids_.find(profile_name) != profiler_capture_ids_.end()) {
                            capture_id = profiler_capture_ids_[profile_name];
                        }

                        if (profiler_entries_.find(profile_name) != profiler_entries_.end()) {
                            entries_copy = profiler_entries_[profile_name];
                        }

                        if (show_averages_ && label_history_.find(profile_name) != label_history_.end()) {
                            history_copy = label_history_[profile_name];
                        }
                    }

                    render_profiler_table(capture_id, entries_copy, history_copy, show_avg);
                }
            }
        }
        else
        {
            ImGui::Text("No profiler data available");
            if (ImGui::Button("Start Profiling"))
            {
                profiler_paused_ = false;
            }
        }
    }
    return true;
}

void Profiler::render_profiler_table(
    pnanovdb_uint64_t capture_id,
    const std::map<std::string, ProfilerEntry>& entries,
    const std::unordered_map<std::string, std::vector<pnanovdb_compute_profiler_entry_t>>& history,
    bool show_avg)
{
    if (ImGui::BeginTable("ProfilerTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("CPU (ms)", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("GPU (ms)", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();

        float total_cpu_time = 0.0f;
        float total_gpu_time = 0.0f;

        for (const auto& pair : entries)
        {
            const auto& label = pair.first;
            const auto& entry = pair.second;

            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImGui::Text("%s", label.c_str());

            ImGui::TableNextColumn();
            float cpu_ms;
            if (show_avg && history.find(label) != history.end() && !history.at(label).empty())
            {
                float sum = 0.0f;
                for (const auto& hist_entry : history.at(label))
                {
                    sum += hist_entry.cpu_delta_time * 1000.0f;
                }
                cpu_ms = sum / history.at(label).size();
            }
            else
            {
                cpu_ms = entry.entry.cpu_delta_time * 1000.0f;
                if (entry.capture_id != capture_id)
                {
                    cpu_ms = 0.f;
                }
            }
            ImGui::Text("%.3f", cpu_ms);
            total_cpu_time += cpu_ms;

            ImGui::TableNextColumn();
            float gpu_ms;
            if (show_avg && history.find(label) != history.end() && !history.at(label).empty())
            {
                float sum = 0.0f;
                for (const auto& hist_entry : history.at(label))
                {
                    sum += hist_entry.gpu_delta_time * 1000.0f;
                }
                gpu_ms = sum / history.at(label).size();
            }
            else
            {
                gpu_ms = entry.entry.gpu_delta_time * 1000.0f;
                if (entry.capture_id != capture_id)
                {
                    gpu_ms = 0.f;
                }
            }
            ImGui::Text("%.3f", gpu_ms);
            total_gpu_time += gpu_ms;
        }

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("Total");
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", total_cpu_time);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", total_gpu_time);

        ImGui::EndTable();
    }
}

void Profiler::report_callback(void* userdata, pnanovdb_uint64_t captureID, pnanovdb_uint32_t numEntries, pnanovdb_compute_profiler_entry_t* entries)
{
    auto profiler = &Profiler::getInstance();
    if (profiler->profiler_paused_)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(profiler->mutex_);

    // Extract profiler label from userdata
    std::string name = reinterpret_cast<const char*>(userdata);
    if (name.empty())
    {
        name = "Profiler " + std::to_string(Profiler::s_id++);
    }

    profiler->profiler_capture_ids_[name] = captureID;

    for (pnanovdb_uint32_t i = 0; i < numEntries; ++i)
    {
        if (entries[i].label && entries[i].label[0] != '\0')
        {
            std::string label = entries[i].label;
            // merge entries with a common captureID
            auto& profiler_entry = profiler->profiler_entries_[name][label];
            if (profiler_entry.capture_id == captureID)
            {
                profiler_entry.entry.cpu_delta_time += entries[i].cpu_delta_time;
                profiler_entry.entry.gpu_delta_time += entries[i].gpu_delta_time;
            }
            else
            {
                profiler_entry = ProfilerEntry{entries[i], captureID};
            }

            const size_t MAX_HISTORY_PER_LABEL = 100;
            auto& history = profiler->label_history_[name][label];
            history.push_back(entries[i]);
            if (history.size() > MAX_HISTORY_PER_LABEL)
            {
                history.erase(history.begin());
            }
        }
    }
}

} // namespace pnanovdb_editor
