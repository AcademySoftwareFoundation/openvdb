// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/UserParams.cpp

    \author Petra Hapalova

    \brief
*/

#include "UserParams.h"

#include "Console.h"

#include "nanovdb_editor/putil/Shader.hpp"

#include <fstream>

namespace imgui_instance_user
{

void UserParams::create(const std::string& shader_name)
{
    std::string json_filePath = pnanovdb_shader::getUserParamsFilePath(shader_name.c_str());
    std::filesystem::path fsPath(json_filePath);
    if (std::filesystem::exists(fsPath))
    {
        pnanovdb_editor::Console::getInstance().addLog("User params file '%s' already exists", json_filePath.c_str());
        return;
    }

    auto json_user_params = nlohmann::json::object();

    std::string shader_json_path = pnanovdb_shader::getShaderParamsFilePath(shader_name.c_str());
    std::ifstream shader_json_file(shader_json_path);
    if (shader_json_file)
    {
        nlohmann::json shader_json;
        shader_json_file >> shader_json;
        if (shader_json.contains("userParams"))
        {
            auto& shader_user_params = shader_json["userParams"];
            for (auto& [key, value] : shader_user_params.items())
            {
                if (key.find("_pad") != std::string::npos)
                {
                    continue;
                }
                assert(value.contains("type"));
                if (value["type"] == "bool")
                {
                    createDefaultBoolParam(key, json_user_params);
                }
                else
                {
                    createDefaultScalarNParam(key, value, json_user_params);
                }
            }
        }
        shader_json_file.close();
    }

    nlohmann::json json;
    json["UserParams"] = json_user_params;

    std::ofstream json_file(json_filePath);
    json_file << json.dump(4);
    json_file.close();
}

bool UserParams::load(const std::string& shader_name)
{
    clear();

    std::string shader_json_path = pnanovdb_shader::getShaderParamsFilePath(shader_name.c_str());
    std::ifstream shader_json_file(shader_json_path);
    if (!shader_json_file)
    {
        pnanovdb_editor::Console::getInstance().addLog("User params for '%s' not found, please compile shader first", shader_name.c_str());
        return false;
    }

    nlohmann::ordered_json shader_json;
    shader_json_file >> shader_json;
    if (!shader_json.contains("userParams"))
    {
        // Shader has no user parameters defined
        return false;
    }

    auto& shader_user_params = shader_json["userParams"];
    for (auto& [key, value] : shader_user_params.items())
    {
        assert(value.contains("type"));
        if (value["type"] == "bool")
        {
            createBoolParam(key, value);
        }
        else
        {
            createScalarNParam(key, value);
        }
    }

    shader_json_file.close();

    std::string json_filePath = pnanovdb_shader::getUserParamsFilePath(shader_name.c_str());
    std::ifstream json_file(json_filePath);
    if (!json_file)
    {
        return false;
    }

    nlohmann::json json;
    try
    {
        json_file >> json;
    }
    catch (const nlohmann::json::parse_error&)
    {
        pnanovdb_editor::Console::getInstance().addLog("Error parsing user params file '%s'", json_filePath.c_str());
        json_file.close();
        return false;
    }

    if (!json.contains("UserParams"))
    {
        json_file.close();
        return false;
    }

    auto& json_user_params = json["UserParams"];
    for (auto& user_param : params_)
    {
        if (!json_user_params.contains(user_param.name))
        {
            continue;
        }

        auto& value = json_user_params[user_param.name];
        if (user_param.type == ImGuiDataType_Bool)
        {
            addToBoolParam(user_param.name, value);
        }
        else
        {
            addToScalarNParam(user_param.name, value);
        }
    }
    json_file.close();
    return true;
}

static std::pair<ImGuiDataType, size_t> getScalarTypeAndSize(const std::string& type)
{
    static const std::unordered_map<std::string, std::pair<ImGuiDataType, size_t>> typeMap = {
        {"", {ImGuiDataType_Float, sizeof(float)}},
        {"void", {ImGuiDataType_Float, sizeof(float)}},
        {"int", {ImGuiDataType_S32, sizeof(int32_t)}},
        {"uint", {ImGuiDataType_U32, sizeof(uint32_t)}},
        {"int64", {ImGuiDataType_S64, sizeof(int64_t)}},
        {"uint64", {ImGuiDataType_U64, sizeof(uint64_t)}},
        {"float16", {ImGuiDataType_Float, sizeof(float) / 2u}},
        {"float", {ImGuiDataType_Float, sizeof(float)}},
        {"double", {ImGuiDataType_Float, sizeof(double)}}
    };

    auto it = typeMap.find(type);
    if (it != typeMap.end())
    {
        return it->second;
    }
    return {ImGuiDataType_Float, sizeof(float)};
}

template<typename T>
void assignValue(void* target, const nlohmann::json& source, T defaultValue = T(0))
{
    T val = source.is_number() ? source.get<T>() : defaultValue;
    memcpy(target, &val, sizeof(T));
}

void assignTypedValue(ImGuiDataType type, void* target, const nlohmann::json& source, float defaultValue = 0.f)
{
    switch (type)
    {
        case ImGuiDataType_S32:
            assignValue<int>(target, source, static_cast<int>(defaultValue));
            break;
        case ImGuiDataType_U32:
            assignValue<unsigned int>(target, source, static_cast<unsigned int>(defaultValue));
            break;
        case ImGuiDataType_S64:
            assignValue<long long>(target, source, static_cast<long long>(defaultValue));
            break;
        case ImGuiDataType_U64:
            assignValue<unsigned long long>(target, source, static_cast<unsigned long long>(defaultValue));
            break;
        case ImGuiDataType_Float:
        default:
            assignValue<float>(target, source, static_cast<float>(defaultValue));
            break;
    }
}

template<typename T>
void assignValueOnIndex(void* target, const nlohmann::json& source, int index)
{
    nlohmann::basic_json json_val;
    try
    {
        json_val = source.at(index);
    }
    catch (const nlohmann::json::out_of_range&)
    {
        json_val = nlohmann::json(T(0));
    }

    T val = json_val.get<T>();
    memcpy(static_cast<char*>(target) + index * sizeof(T), &val, sizeof(T));
}

void assignTypedValueOnIndex(ImGuiDataType type, void* target, const nlohmann::json& source, int index)
{
    switch (type)
    {
        case ImGuiDataType_S32:
            assignValueOnIndex<int>(target, source, index);
            break;
        case ImGuiDataType_U32:
            assignValueOnIndex<unsigned int>(target, source, index);
            break;
        case ImGuiDataType_S64:
            assignValueOnIndex<long long>(target, source, index);
            break;
        case ImGuiDataType_U64:
            assignValueOnIndex<unsigned long long>(target, source, index);
            break;
        case ImGuiDataType_Float:
        default:
            assignValueOnIndex<float>(target, source, index);
            break;
    }
}
void UserParams::createDefaultScalarNParam(const std::string& name, nlohmann::json& value, nlohmann::json& json_user_params)
{
    nlohmann::json param;
    param["min"] = 0;
    param["max"] = 1;
    param["step"] = 0.01f;
    assert(value.contains("elementCount"));
    size_t num_elements = value["elementCount"];
    if (num_elements > 1)
    {
        nlohmann::json array = nlohmann::json::array();
        for (size_t i = 0; i < num_elements; i++)
        {
            array.push_back(0);
        }
        param["value"] = array;
    }
    else
    {
        param["value"] = 0;
    }
    json_user_params[name] = param;
}

void UserParams::createScalarNParam(const std::string& name, const nlohmann::json& value)
{
    UserParam user_param;
    user_param.name = name;
    auto [type, size] = getScalarTypeAndSize(value["type"]);
    user_param.type = type;
    user_param.size = size;
    assert(value.contains("elementCount"));
    user_param.num_elements = value["elementCount"];
    user_param.value = new char[user_param.num_elements * size];
    if (user_param.num_elements > 1)
    {
        for (int i = 0; i < user_param.num_elements; i++)
        {
            assignTypedValue(user_param.type, (char*)user_param.value + i * user_param.size, nlohmann::json(0));
        }
    }
    else
    {
        assignTypedValue(user_param.type, user_param.value, nlohmann::json(0));
    }

    user_param.min = new char[user_param.num_elements * size];
    assignTypedValue(user_param.type, user_param.min, nlohmann::json(-FLT_MAX));

    user_param.max = new char[user_param.num_elements * size];
    assignTypedValue(user_param.type, user_param.max, nlohmann::json(FLT_MAX));

    params_.emplace_back(std::move(user_param));
}

void UserParams::addToScalarNParam(const std::string& name, const nlohmann::json& value)
{
    auto it = std::find_if(params_.begin(), params_.end(), [&name](const UserParam& param) {
        return param.name == name;
    });

    if (it == params_.end())
    {
        return;
    }

    UserParam& user_param = *it;
    if (value.contains("value"))
    {
        if (value["value"].is_array())
        {
            for (int i = 0; i < user_param.num_elements; i++)
            {
                assignTypedValueOnIndex(user_param.type, user_param.value, value["value"], i);
            }
        }
        else
        {
            assignTypedValue(user_param.type, user_param.value, value["value"]);
        }
    }
    if (value.contains("min"))
    {
        assignTypedValue(user_param.type, user_param.min, value["min"]);
    }
    if (value.contains("max"))
    {
        assignTypedValue(user_param.type, user_param.max, value["max"]);
    }
    if (value.contains("step"))
    {
        user_param.step = value["step"];
    }
}

void UserParams::createDefaultBoolParam(const std::string& name, nlohmann::json& json_user_params)
{
    nlohmann::json param;
    param["value"] = false;
    json_user_params[name] = param;
}

void UserParams::createBoolParam(const std::string& name, const nlohmann::json& value)
{
    UserParam user_param;
    user_param.name = name;
    user_param.num_elements = 1;
    user_param.type = ImGuiDataType_Bool;
    user_param.size = sizeof(bool);
    user_param.value = new bool[user_param.num_elements];
    ((bool*)user_param.value)[0] = false;

    params_.emplace_back(std::move(user_param));
}

void UserParams::addToBoolParam(const std::string& name, const nlohmann::json& value)
{
    auto it = std::find_if(params_.begin(), params_.end(), [&name](const UserParam& param) {
        return param.name == name;
    });

    if (it == params_.end())
    {
        return;
    }

    UserParam& user_param = *it;
    if (value.contains("value") && value["value"].is_boolean())
    {
        ((bool*)user_param.value)[0] = value["value"];
    }
}

void UserParams::render()
{
    for (auto& user_param : params_)
    {
        if (user_param.name.find("_pad") != std::string::npos)
        {
            continue;
        }
        if (user_param.type == ImGuiDataType_Bool)
        {
            ImGui::Checkbox(user_param.name.c_str(), (bool*)user_param.value);
        }
        else
        {
            if (user_param.num_elements == 16)
            {
                ImGui::Text("%s", user_param.name.c_str());
                size_t num_rows = 4;
                size_t num_cols = 4;
                const char* names[] = {"x", "y", "z", "w"};
                for (int i = 0; i < num_rows; i++)
                {
                    void* row_ptr = (char*)user_param.value + i * num_cols * user_param.size;
                    ImGui::DragScalarN(names[i], user_param.type, row_ptr, num_cols, user_param.step,
                        user_param.min, user_param.max);
                }
            }
            else
            {
                ImGui::DragScalarN(user_param.name.c_str(), user_param.type, user_param.value, user_param.num_elements,
                    user_param.step, user_param.min, user_param.max);
            }
        }
    }
}
}
