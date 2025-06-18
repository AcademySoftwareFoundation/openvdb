// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/UserParams.h

    \author Petra Hapalova

    \brief
*/

#include <nlohmann/json.hpp>

#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif  // IMGUI_DEFINE_MATH_OPERATORS

#include <imgui.h>

#include <string>
#include <vector>

namespace imgui_instance_user
{
struct UserParam
{
    std::string name;
    ImGuiDataType type;
    void* value;
    size_t size;
    size_t num_elements;
    void* min;
    void* max;
    float step;

    UserParam()
    : value(nullptr), size(0), num_elements(0), min(nullptr), max(nullptr), step(0.0f)
    {
    }

    ~UserParam()
    {
        if (value)
        {
            delete[] static_cast<char*>(value);
            value = nullptr;
        }
        if (min)
        {
            delete[] static_cast<char*>(min);
            min = nullptr;
        }
        if (max)
        {
            delete[] static_cast<char*>(max);
            max = nullptr;
        }
    }

    UserParam(UserParam&& other) noexcept
    {
        name = std::move(other.name);
        type = other.type;
        value = other.value;
        size = other.size;
        num_elements = other.num_elements;
        min = other.min;
        max = other.max;
        step = other.step;
        other.value = nullptr;
        other.size = 0u;
        other.num_elements = 0u;
        other.min = nullptr;
        other.max = nullptr;
        other.step = 0.f;
    }

    UserParam& operator=(UserParam&& other) noexcept
    {
        if (this != &other)
        {
            if (value)
            {
                delete[](char*)value;
            }
            name = std::move(other.name);
            type = other.type;
            value = other.value;
            size = other.size;
            num_elements = other.num_elements;
            min = other.min;
            max = other.max;
            other.value = nullptr;
            other.size = 0u;
            other.num_elements = 0u;
            other.min = nullptr;
            other.max = nullptr;
            other.step = 0.f;
        }
        return *this;
    }
};

class UserParams
{
public:
    void create(const std::string& shader_name);
    bool load(const std::string& shader_name);
    void render();

    const std::vector<UserParam>& getParams() const
    {
        return params_;
    }

    void clear()
    {
        params_.clear();
    }

private:
    std::vector<UserParam> params_;
    void createDefaultScalarNParam(const std::string& name, nlohmann::json& value, nlohmann::json& json_user_params);
    void createScalarNParam(const std::string& name, const nlohmann::json& value);
    void addToScalarNParam(const std::string& name, const nlohmann::json& value);
    void createDefaultBoolParam(const std::string& name, nlohmann::json& json_user_params);
    void createBoolParam(const std::string& name, const nlohmann::json& value);
    void addToBoolParam(const std::string& name, const nlohmann::json& value);
};
}
