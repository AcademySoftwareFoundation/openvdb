// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_UTIL_NAME_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_NAME_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/version.h>
#include <string>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

typedef std::string Name;

inline Name
readString(std::istream& is)
{
    uint32_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    std::string buffer(size, ' ');
    if (size>0) is.read(&buffer[0], size);
    return buffer;
}


inline void
writeString(std::ostream& os, const Name& name)
{
    uint32_t size = uint32_t(name.size());
    os.write(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    os.write(&name[0], size);
}

namespace string {

template <typename ContainerT>
inline void split(ContainerT& out, const std::string& in, const char delim)
{
    out.clear();
    if (in.empty()) return;
    std::string::size_type pos = 0u, old = 0u;
    while ((pos = in.find(delim, pos)) != std::string::npos) {
        out.insert(out.end(), in.substr(old, (pos-old)));
        old = ++pos;
    }
    // last element
    out.insert(out.end(), in.substr(old, in.length()-old));
}

template <typename ContainerT>
inline void split(ContainerT& out, const std::string& in, const std::set<char>& delims)
{
    out.clear();
    if (in.empty()) return;
    std::string::size_type pos = 0u, old = 0u;
    const std::string seq(delims.begin(), delims.end());
    while ((pos = in.find_first_of(seq, pos)) != std::string::npos) {
        out.insert(out.end(), in.substr(old, (pos-old)));
        old = ++pos;
    }
    // last element
    out.insert(out.end(), in.substr(old, in.length()-old));
}

inline bool starts_with(const std::string& in, const std::string& start)
{
    if (start.length() > in.length()) return false;
    return std::equal(start.begin(), start.end(), in.begin());
}

inline bool ends_with(const std::string& in, const std::string& end)
{
    if (end.length() > in.length()) return false;
    return std::equal(end.rbegin(), end.rend(), in.rbegin());
}

inline void trim(std::string& s)
{
    // ltrim
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));

    // rtrim
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

inline void to_lower(std::string& s)
{
    std::transform(s.begin(), s.end(), s.begin(),
        [](auto c){ return std::tolower(c); });
}

} // util

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_NAME_HAS_BEEN_INCLUDED
