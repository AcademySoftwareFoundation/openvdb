// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file StringUtils.h
	\brief Helpful string utility methods.
*/

#pragma once

#include <string>
#include <sstream>
#include <map>
#include <vector>

//---------------------------------------------------------------------------------------------------
//! Return extracted url path without scheme or parameters.
std::string urlGetFile(const std::string& url);
//! Return extracted url path without scheme or parameters.
std::string urlGetPath(const std::string& url);
//! Return extracted url scheme. Empty string if none found.
std::string urlGetScheme(const std::string& url);
//! Return extracted url path extension. Empty string if none found.
std::string urlGetPathExtension(const std::string& url);
//! Return extracted url query. Empty string if not found.
std::string urlGetQuery(const std::string& url);
//! Return url fragment. Empty string if not found.
std::string urlGetFragment(const std::string& url);

//---------------------------------------------------------------------------------------------------
template<typename T>
struct ToString
{
    template<typename U = T, typename std::enable_if<(std::is_integral<U>::value || std::is_floating_point<U>::value), void>::type* = nullptr>
    inline std::string operator()(const T v) const
    {
        return std::to_string(v);
    }

    template<typename U = T, typename std::enable_if<!(std::is_integral<U>::value || std::is_floating_point<U>::value), void>::type* = nullptr>
    inline std::string operator()(const T& v) const
    {
        std::ostringstream ss;
        ss << v;
        return ss.str();
    }
};

template<typename T>
struct FromString
{
    inline T operator()(const std::string& s) const
    {
        std::istringstream ss(s);
        T                  v;
        ss >> v;
        return v;
    }
};

template<>
struct ToString<std::string>
{
    inline std::string operator()(const std::string& s) const
    {
        return s;
    }
};

template<>
struct ToString<bool>
{
    inline std::string operator()(const bool v) const
    {
        return (v) ? "true" : "false";
    }
};

template<>
struct FromString<std::string>
{
    inline std::string operator()(const std::string& s) const
    {
        return s;
    }
};

template<>
struct FromString<bool>
{
    inline bool operator()(const std::string& s) const
    {
        std::istringstream ss(s);
        std::string        v;
        ss >> v;
        if (v == "off" || v == "no" || v == "false" || v == "0")
            return false;
        return true;
    }
};

//---------------------------------------------------------------------------------------------------
// Very simple map of strings for property lists.
class StringMap : public std::map<std::string, std::string>
{
public:
    inline bool contains(const std::string& k) const
    {
        return (this->find(k) != this->end());
    }

    template<typename T>
    inline void set(const std::string& k, const T& v)
    {
        (*this)[k] = ToString<T>()(v);
    }

    template<typename T>
    inline T get(const std::string& k, const T& def = T()) const
    {
        auto it = this->find(k);
        if (it != this->end()) {
            return FromString<T>()(it->second);
        }
        return def;
    }

    template<typename T>
    inline void setEnum(const std::string& k, const std::vector<std::string>& enums, const T v) const
    {
        assert(v >= 0 && v < enums.size());
        this->insert(std::make_pair(k, enums[v]));
    }

    template<typename T>
    inline T getEnum(const std::string& k, const std::vector<std::string>& enums, const T& def = T(0)) const
    {
        auto it = this->find(k);
        if (it != this->end()) {
            for (int i = 0; i < enums.size(); ++i) {
                if (enums[i] == it->second)
                    return T(i);
            }
        }
        return def;
    }

    template<typename T>
    inline void setEnum(const std::string& k, const char** enums, int enumCount, const T v) const
    {
        (void)enumCount;
        assert(v >= 0 && v < enumCount);
        this->insert(std::make_pair(k, enums[v]));
    }

    template<typename T>
    inline T getEnum(const std::string& k, const char** enums, int enumCount, const T& def = T(0)) const
    {
        auto it = this->find(k);
        if (it != this->end()) {
            for (int i = 0; i < enumCount; ++i) {
                if (std::string(enums[i]) == it->second)
                    return T(i);
            }
        }
        return def;
    }
};
