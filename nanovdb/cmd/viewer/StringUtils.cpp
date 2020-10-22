// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file StringUtils.cpp

	\author Wil Braithwaite

	\date October 09, 2020

	\brief Helpful string utility methods.
*/

#include "StringUtils.h"
#include <string>
#include <map>

//---------------------------------------------------------------------------------------------------
std::string urlGetFile(const std::string& url)
{
    auto path = url;
    auto i = url.find("://", 0);
    if (i != std::string::npos) {
        path = url.substr(i + 3);
    }
    i = path.find_last_of(':');
    if (i != std::string::npos && i != 1) {
        // if not windows drive letter.
        path = path.substr(0, i);
    }
    i = path.find_last_of('/');
    if (i != std::string::npos)
        path = path.substr(i + 1);
    i = path.find_last_of('\\');
    if (i != std::string::npos)
        path = path.substr(i + 1);
    return path;
}

std::string urlGetPath(const std::string& url)
{
    auto path = url;
    auto i = url.find("://", 0);
    if (i != std::string::npos) {
        path = url.substr(i + 3);
    }
    i = path.find("?");
    if (i == std::string::npos) {
        return path;
    }
    return path.substr(0, i);
}

std::string urlGetPathExtension(const std::string& url)
{
    auto path = urlGetPath(url);
    auto i = path.find_last_of(".");
    if (i != std::string::npos) {
        return path.substr(i + 1);
    }
    return "";
}

std::string urlGetScheme(const std::string& url)
{
    std::string scheme = "";
    auto        i = url.find("://", 0);
    if (i != std::string::npos) {
        scheme = url.substr(0, i);
    }
    return scheme;
}

std::string urlGetQuery(const std::string& url)
{
    auto i = url.find("?", 0);
    if (i != std::string::npos) {
        return url.substr(i + 1);
    }
    return "";
}

std::string urlGetFragment(const std::string& url)
{
    auto i = url.find("#", 0);
    if (i != std::string::npos) {
        return url.substr(i + 1);
    }
    return "";
}
