// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/CodeEditor.h

    \author Petra Hapalova

    \brief
*/

#include "putil/Compute.h"
#include "putil/Shader.hpp"

#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif  // IMGUI_DEFINE_MATH_OPERATORS

#include <imgui.h>
#include <TextEditor.h>

#include <string>
#include <filesystem>
#include <atomic>

struct ImGuiSettingsHandler;
struct ImGuiTextBuffer;

namespace pnanovdb_editor
{
class CodeEditor
{
    static const char* defaultName;

    enum class ShowOption
    {
        ShaderOnly,
        Generated,
        UserParams,
        Last
    };

    struct EditorTab
    {
        TextEditor editor;
        TextEditor viewer;      // viewer for generated HLSL code or user params
        std::string title;
        std::string filepath;
        std::string userParamsFilepath;
        bool opened = false;
        int editorUndoIndex = -1;

        // serialized in the settings
        std::string shaderName;
        int firstVisibleLine = -1;
        int viewerFirstVisibleLine = -1;

        void rename(const char* name)
        {
            shaderName = name;
            std::filesystem::path fsPath(shaderName);
            title = fsPath.filename().string();
            filepath = pnanovdb_shader::getShaderFilePath(shaderName.c_str());
            userParamsFilepath = pnanovdb_shader::getUserParamsFilePath(shaderName.c_str());
        }

        EditorTab()
            : EditorTab("")
        {
        }

        EditorTab(const char* shaderName)
        {
            rename(shaderName);

            opened = true;
            editorUndoIndex = 0;

            // TODO Create PR for the Slang language definition, check https://github.com/santaclose/ImGuiColorTextEdit/pull/26/
            editor.SetLanguageDefinition(TextEditor::LanguageDefinitionId::Hlsl);
        }
    };

public:
    // singleton
    static CodeEditor& getInstance()
    {
        static CodeEditor instance;
        return instance;
    }

    CodeEditor();
    bool render();
    void setup(std::string* shaderNamePtr, std::atomic<bool>* updateShaderPtr, ImVec2& dialog_size, pnanovdb_shader::run_shader_func_t run_shader);
    void setSelectedShader(const std::string& shaderName);
    void updateViewer();
    void saveUserParams();
    void registerSettingsHandler(ImGuiContext* context);
    void saveTabsState();

private:
    static void ClearAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler);
    static void* ReadOpen(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name);
    static void ReadLine(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line);
    static void WriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf);
    static void ApplyAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler);

    void setSelectedFile(const std::string& filepath);
    void addNewFile();
    void saveSelectedTabText();
    void openFileDialog(const char* dialogKey, const char* title, const char* filters);

    std::map<std::string, EditorTab> tabs_;     // key is shader name
    std::string selectedTab_;
    ShowOption showOption_;
    std::string* editorShaderPtr_;
    std::atomic<bool>* updateShaderPtr_;
    pnanovdb_shader::run_shader_func_t run_shader_;
    ImVec2 editorSize_;
    ImVec2 dialog_size_;
    bool isEditorLastClicked_ = false;
    int gridDims_[3] = { 0, 0, 0 };
};
}
