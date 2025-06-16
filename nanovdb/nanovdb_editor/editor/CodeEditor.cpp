// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/CodeEditor.cpp

    \author Petra Hapalova

    \brief
*/

#include "CodeEditor.h"

#include <imgui_internal.h>
#include <ImGuiFileDialog.h>

#include <sstream>
#include <atomic>

namespace pnanovdb_editor
{
const char* CodeEditor::defaultName = "new";

static TextEditor::LanguageDefinitionId getLanguageDefinition(const char* extension)
{
    if (strcmp(extension, pnanovdb_shader::SHADER_HLSL_EXT) == 0)
    {
        return TextEditor::LanguageDefinitionId::Hlsl;
    }
    else if (strcmp(extension, pnanovdb_shader::SHADER_CPP_EXT) == 0)
    {
        return TextEditor::LanguageDefinitionId::Cpp;
    }
    return TextEditor::LanguageDefinitionId::None;
}

CodeEditor::CodeEditor()
{
    selectedTab_.clear();
    showOption_ = ShowOption::ShaderOnly;
    editorShaderPtr_ = nullptr;
    updateShaderPtr_ = nullptr;
}

bool CodeEditor::render()
{
    bool isFocused = ImGui::IsWindowFocused();

    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("New", "Ctrl-N"))
            {
                addNewFile();
            }
            if (ImGui::MenuItem("Open...", "Ctrl-O"))
            {
                openFileDialog("OpenCodeDlgKey", "Open Shader File", ".slang");
            }
            if (ImGui::MenuItem("Save", "Ctrl-S"))
            {
                saveSelectedTabText();
            }
            if (ImGui::MenuItem("Save As...", "Ctrl-Alt-S"))
            {
                openFileDialog("SaveCodeDlgKey", "Save Shader File", ".slang,.hlsl");
            }
            // if (ImGui::MenuItem("Save User Params", ""))
            // {
            //     saveUserParams();
            // }
            if (ImGui::MenuItem("Quit", "Alt-F4"))
            {
                ImGui::EndMenu();
                ImGui::EndMenuBar();
                ImGui::End();
                return false;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
            assert(tabs_.find(selectedTab_) != tabs_.end());

            if (ImGui::MenuItem("Undo", "Ctrl-Z", nullptr, tabs_[selectedTab_].editor.CanUndo()))
            {
                tabs_[selectedTab_].editor.Undo();
            }
            if (ImGui::MenuItem("Redo", "Ctrl-Y", nullptr, tabs_[selectedTab_].editor.CanRedo()))
            {
                tabs_[selectedTab_].editor.Redo();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Copy", "Ctrl+C", nullptr, tabs_[selectedTab_].editor.AnyCursorHasSelection()))
            {
                tabs_[selectedTab_].editor.Copy();
            }
            if (ImGui::MenuItem("Cut", "Ctrl+X", nullptr, tabs_[selectedTab_].editor.AnyCursorHasSelection()))
            {
                tabs_[selectedTab_].editor.Cut();
            }
            if (ImGui::MenuItem("Paste", "Ctrl+V", nullptr, ImGui::GetClipboardText() != nullptr))
            {
                tabs_[selectedTab_].editor.Paste();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Select all", "Ctrl+A", nullptr))
            {
                tabs_[selectedTab_].editor.SelectAll();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    if (ImGuiFileDialog::Instance()->IsOpened())
    {
        ImGui::SetNextWindowSize(dialog_size_, ImGuiCond_Appearing);
        if (ImGuiFileDialog::Instance()->Display("OpenCodeDlgKey"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                auto filepath = ImGuiFileDialog::Instance()->GetFilePathName();
                setSelectedFile(filepath);
            }
            ImGuiFileDialog::Instance()->Close();
        }
        if (ImGuiFileDialog::Instance()->Display("SaveCodeDlgKey"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                assert(tabs_.find(selectedTab_) != tabs_.end());

                std::string filePath = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string newName = pnanovdb_shader::getShaderName(filePath.c_str());
                tabs_[selectedTab_].rename(newName.c_str());

                saveSelectedTabText();
            }
            ImGuiFileDialog::Instance()->Close();
        }
    }

    if (ImGui::Button("Compile"))
    {
        saveSelectedTabText();
    }
    ImGui::SameLine();
    if (ImGui::Button("Show"))
    {
        assert(editorShaderPtr_ && updateShaderPtr_);

        *editorShaderPtr_ = selectedTab_;
        *updateShaderPtr_ = true;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Show shader output in a viewport");
    }
    /*
    if (ImGui::Button("Run"))
    {
        const std::string shaderName = tabs_[selectedTab_].shaderName;
        if (run_shader_)
        {
            run_shader_(shaderName.c_str(), gridDims_[0], gridDims_[1], gridDims_[2]);
        }
        else
        {
            printf("Error: Run shader function is not set\n");
        }
    }
    ImGui::SameLine();
    ImGui::DragInt3("Grid Dim", gridDims_, 0.1f);
    */
    ShowOption selectedOption = showOption_;
    if (ImGui::RadioButton("Shader Only", selectedOption == ShowOption::ShaderOnly))
    {
        selectedOption = ShowOption::ShaderOnly;
        saveTabsState();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Generated", selectedOption == ShowOption::Generated))
    {
        selectedOption = ShowOption::Generated;
        saveTabsState();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("User Params", selectedOption == ShowOption::UserParams))
    {
        selectedOption = ShowOption::UserParams;
        saveTabsState();
    }

    if (selectedOption != showOption_)
    {
        showOption_ = selectedOption;
        if (showOption_ != ShowOption::ShaderOnly)
        {
            updateViewer();

            // set split in half by default
            ImVec2 size = ImGui::GetContentRegionAvail();
            editorSize_.y = 0.5f * size.y;
        }
    }

    ImGui::Spacing();
    if (ImGui::BeginTabBar("Tabs"))
    {
        // get absolute size for the tab
        ImVec2 size = ImGui::GetContentRegionAvail();
        if (showOption_ == ShowOption::ShaderOnly)
        {
            editorSize_ = size;
        }
        else
        {
            editorSize_.x = size.x;
            size.y -= editorSize_.y;
        }

        for (auto it = tabs_.begin(); it != tabs_.end(); )
        {
            ImGuiTabItemFlags flags = ImGuiTabItemFlags_None;
            if (it->first == selectedTab_)
            {
                flags |= ImGuiTabItemFlags_SetSelected;
            }
            if (it->second.editor.GetUndoIndex() != it->second.editorUndoIndex)
            {
                flags |= ImGuiTabItemFlags_UnsavedDocument;
            }
            bool tabOpen = ImGui::BeginTabItem(it->second.title.c_str(), &it->second.opened, flags);
            if (ImGui::IsItemClicked())
            {
                setSelectedShader(it->first);
            }
            if (tabOpen)
            {
                bool editorCursor = ImGui::GetIO().MouseDrawCursor;
                bool editorClicked = it->second.editor.Render(it->first.c_str(), isFocused && isEditorLastClicked_, editorSize_);
                if (editorClicked)
                {
                    isEditorLastClicked_ = true;
                    ImGui::GetIO().MouseDrawCursor = editorCursor;
                }

                if (showOption_ != ShowOption::ShaderOnly)
                {
                    ImRect bb;
                    bb.Min = ImGui::GetWindowPos() + ImGui::GetCursorPos();
                    bb.Min.y += 2.f;
                    bb.Max = bb.Min + ImVec2(editorSize_.x, 2.f);
                    ImGui::SplitterBehavior(bb, ImGui::GetID("##splitter"), ImGuiAxis_Y, &editorSize_.y, &size.y, 10.f, 10.f);
                    ImGui::Spacing();

                    // Store cursor visibility state before rendering viewer
                    bool viewerCursor = ImGui::GetIO().MouseDrawCursor;
                    bool viewerClicked = it->second.viewer.Render("##viewer", isFocused && !isEditorLastClicked_, size);
                    if (viewerClicked)
                    {
                        isEditorLastClicked_ = false;
                        ImGui::GetIO().MouseDrawCursor = viewerCursor;
                    }
                }
                it->second.firstVisibleLine = it->second.editor.GetFirstVisibleLine();
                it->second.viewerFirstVisibleLine = it->second.viewer.GetFirstVisibleLine();
                ImGui::EndTabItem();
            }
            if (it->second.opened)
            {
                ++it;
            }
            else
            {
                it = tabs_.erase(it);
            }
        }
        ImGui::EndTabBar();
    }

    if (isEditorLastClicked_ && ImGui::GetIO().KeyCtrl)
    {
        if (ImGui::IsKeyPressed(ImGuiKey_N, false))
        {
            addNewFile();
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_S, false))
        {
            saveSelectedTabText();
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_S, false))
        {
            if (ImGui::GetIO().KeyAlt)
            {
                // Ctrl+Alt+S for Save As
                openFileDialog("SaveCodeDlgKey", "Save Shader File", ".slang,.hlsl");
            }
            else
            {
                // Ctrl+S for Save
                saveSelectedTabText();
            }
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_O, false))
        {
            // Ctrl+O for Open
            openFileDialog("OpenCodeDlgKey", "Open Shader File", ".slang");
        }
    }

    // Save tabs state when tabs are added or removed
    static size_t lastTabCount = 0;
    if (tabs_.size() != lastTabCount)
    {
        saveTabsState();
        lastTabCount = tabs_.size();
    }

    return true;
}

void CodeEditor::setup(std::string* shaderName, std::atomic<bool>* updateShaderPtr, ImVec2& dialog_size, pnanovdb_shader::run_shader_func_t run_shader)
{
    editorShaderPtr_ = shaderName;
    updateShaderPtr_ = updateShaderPtr;
    dialog_size_ = dialog_size;
    run_shader_ = run_shader;
    if (selectedTab_.empty())
    {
        setSelectedShader(*shaderName);
    }
}

void CodeEditor::setSelectedShader(const std::string& shaderName)
{
    selectedTab_ = shaderName;

    auto it = tabs_.find(shaderName);
    if (it == tabs_.end())
    {
        tabs_[selectedTab_] = EditorTab(selectedTab_.c_str());
    }
    std::ifstream inFile(tabs_[selectedTab_].filepath);
    if (inFile)
    {
        std::stringstream buffer;
        buffer << inFile.rdbuf();
        tabs_[selectedTab_].editor.SetText(buffer.str());
    }

    updateViewer();
    saveTabsState();

    tabs_[selectedTab_].editor.SetViewAtLine(tabs_[selectedTab_].firstVisibleLine, TextEditor::SetViewAtLineMode::FirstVisibleLine);
    tabs_[selectedTab_].viewer.SetViewAtLine(tabs_[selectedTab_].viewerFirstVisibleLine, TextEditor::SetViewAtLineMode::FirstVisibleLine);
}

void CodeEditor::updateViewer()
{
    tabs_[selectedTab_].viewer.SetText("");
    tabs_[selectedTab_].viewer.SetReadOnlyEnabled(true);

    if (showOption_ == ShowOption::Generated)
    {
        const std::string shaderName = tabs_[selectedTab_].shaderName;
        const uint32_t compileTarget = pnanovdb_shader::getCompileTarget(shaderName.c_str());
        const std::string extension = pnanovdb_shader::getGeneratedExtension(compileTarget);
        const std::string generatedFilepath = pnanovdb_shader::getShaderCacheFilePath(shaderName.c_str()) + extension;
        const TextEditor::LanguageDefinitionId generatedLanguage = getLanguageDefinition(extension.c_str());

        tabs_[selectedTab_].viewer.SetLanguageDefinition(generatedLanguage);

        std::ifstream generatedFile(generatedFilepath.c_str());
        if (generatedFile)
        {
            std::stringstream buffer;
            buffer << generatedFile.rdbuf();
            tabs_[selectedTab_].viewer.SetText(buffer.str());
        }
    }
    else if (showOption_ == ShowOption::UserParams)
    {
        tabs_[selectedTab_].viewer.SetReadOnlyEnabled(false);
        tabs_[selectedTab_].viewer.SetLanguageDefinition(TextEditor::LanguageDefinitionId::Json);

        std::ifstream userParamsFile(tabs_[selectedTab_].userParamsFilepath);
        if (userParamsFile)
        {
            std::stringstream buffer;
            buffer << userParamsFile.rdbuf();
            tabs_[selectedTab_].viewer.SetText(buffer.str());
        }
    }
}

void CodeEditor::setSelectedFile(const std::string& filepath)
{
    std::string shaderName = pnanovdb_shader::getShaderName(filepath.c_str());
    setSelectedShader(shaderName);
}

void CodeEditor::addNewFile()
{
    std::string defaultNewShaderName = defaultName + std::string(" ");
    std::string newShaderName;
    int counter = 1;
    do
    {
        newShaderName = defaultNewShaderName + std::to_string(counter++);
    } while (tabs_.find(newShaderName) != tabs_.end());

    tabs_[newShaderName] = EditorTab(newShaderName.c_str());
    tabs_[newShaderName].editorUndoIndex = -1;
    tabs_[newShaderName].filepath = "";
    setSelectedShader(newShaderName);
}

static void replaceTabs(std::string& text)
{
    // Replace tabs with 4 spaces
    size_t pos = 0;
    while ((pos = text.find('\t', pos)) != std::string::npos)
    {
        text.replace(pos, 1, "    ");
        pos += 4;
    }
}

void CodeEditor::saveSelectedTabText()
{
    assert(tabs_.find(selectedTab_) != tabs_.end());

    if (tabs_[selectedTab_].filepath.empty())
    {
        openFileDialog("SaveCodeDlgKey", "Save File", ".slang,.hlsl");
        return;
    }

    std::ofstream outFile(tabs_[selectedTab_].filepath);
    if (outFile)
    {
        std::string text = tabs_[selectedTab_].editor.GetText();
        outFile << text;
        outFile.close();

        replaceTabs(text);

        int visibleLine = tabs_[selectedTab_].editor.GetFirstVisibleLine();
        tabs_[selectedTab_].editor.SetViewAtLine(visibleLine, TextEditor::SetViewAtLineMode::FirstVisibleLine);
        tabs_[selectedTab_].editorUndoIndex = tabs_[selectedTab_].editor.GetUndoIndex();
    }
}

void CodeEditor::saveUserParams()
{
    assert(tabs_.find(selectedTab_) != tabs_.end());

    std::ofstream outFile(tabs_[selectedTab_].userParamsFilepath);
    if (outFile)
    {
        std::string text = tabs_[selectedTab_].viewer.GetText();
        outFile << text;
        outFile.close();

        replaceTabs(text);

        int visibleLine = tabs_[selectedTab_].viewer.GetFirstVisibleLine();
        tabs_[selectedTab_].viewer.SetViewAtLine(visibleLine, TextEditor::SetViewAtLineMode::FirstVisibleLine);
    }
}

void CodeEditor::openFileDialog(const char* dialogKey, const char* title, const char* filters)
{
    IGFD::FileDialogConfig config;
    config.path = pnanovdb_shader::getShaderDir();

    ImGuiFileDialog::Instance()->OpenDialog(dialogKey, title, filters, config);
}

void CodeEditor::registerSettingsHandler(ImGuiContext* context)
{
    ImGuiSettingsHandler tabs_handler;
    tabs_handler.TypeName = "CodeEditorTabs";
    tabs_handler.TypeHash = ImHashStr("CodeEditorTabs");
    tabs_handler.ClearAllFn = ClearAll;
    tabs_handler.ReadOpenFn = ReadOpen;
    tabs_handler.ReadLineFn = ReadLine;
    tabs_handler.WriteAllFn = WriteAll;
    tabs_handler.ApplyAllFn = ApplyAll;
    tabs_handler.UserData = this;

    context->SettingsHandlers.push_back(tabs_handler);
}

void CodeEditor::saveTabsState()
{
    ImGui::MarkIniSettingsDirty();
}

void CodeEditor::ClearAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler)
{
    CodeEditor* editor = (CodeEditor*)handler->UserData;
    editor->tabs_.clear();
    editor->selectedTab_.clear();
}

void* CodeEditor::ReadOpen(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name)
{
    // name is either "Settings" or tab index
    if (strcmp(name, "Settings") == 0)
    {
        // General settings section
        return (void*)name;
    }
    else
    {
        // This is a tab section, format is "Tab_N" where N is an index
        // Return a temporary empty string as placeholder
        static std::string emptyStr;
        emptyStr = "";
        return (void*)&emptyStr;
    }
}

void CodeEditor::ReadLine(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line)
{
    const char* name = (const char*)entry;
    CodeEditor* editor = (CodeEditor*)handler->UserData;

    if (name && strcmp(name, "Settings") == 0)
    {
        char buffer[1024] = {};
        if (sscanf(line, "SelectedTab=%[^\n]", buffer) == 1)
        {
            editor->selectedTab_ = buffer;
        }
        else if (int option; sscanf(line, "ShowOption=%d", &option) == 1)
        {
            if (option >= 0 && option < int(ShowOption::Last))
            {
                editor->showOption_ = static_cast<ShowOption>(option);
            }
        }
    }
    else if (entry)
    {
        std::string* tabName = (std::string*)entry;
        char buffer[1024] = {};

        if (sscanf(line, "ShaderName=%[^\n]", buffer) == 1)
        {
            *tabName = buffer;
            if (editor->tabs_.find(*tabName) == editor->tabs_.end())
            {
                editor->tabs_[*tabName] = EditorTab(buffer);
            }
        }
        else if (!tabName->empty())
        {
            int value;
            if (sscanf(line, "FirstVisibleLine=%d", &value) == 1)
            {
                editor->tabs_[*tabName].firstVisibleLine = value;
            }
            else if (sscanf(line, "ViewerFirstVisibleLine=%d", &value) == 1)
            {
                editor->tabs_[*tabName].viewerFirstVisibleLine = value;
            }
        }
    }
}

void CodeEditor::WriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
{
    CodeEditor* editor = (CodeEditor*)handler->UserData;

    buf->appendf("[%s][Settings]\n", handler->TypeName);
    buf->appendf("SelectedTab=%s\n", editor->selectedTab_.c_str());
    buf->appendf("ShowOption=%d\n", static_cast<int>(editor->showOption_));
    buf->append("\n");

    int tabIndex = 0;
    for (const auto& [name, tab] : editor->tabs_)
    {
        buf->appendf("[%s][Tab_%d]\n", handler->TypeName, tabIndex++);
        buf->appendf("ShaderName=%s\n", name.c_str());
        buf->appendf("FirstVisibleLine=%d\n", tab.firstVisibleLine);
        buf->appendf("ViewerFirstVisibleLine=%d\n", tab.viewerFirstVisibleLine);
        buf->append("\n");
    }
}

void CodeEditor::ApplyAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler)
{
    CodeEditor* editor = (CodeEditor*)handler->UserData;

    if (editor->tabs_.empty())
    {
        // If no tabs were loaded, add a default tab
        editor->addNewFile();
    }
    else if (!editor->selectedTab_.empty() && editor->tabs_.find(editor->selectedTab_) != editor->tabs_.end())
    {
        editor->setSelectedShader(editor->selectedTab_);
    }
    else if (!editor->tabs_.empty())
    {
        // If the selected tab doesn't exist, select the first tab
        editor->setSelectedShader(editor->tabs_.begin()->first);
    }
}
}
