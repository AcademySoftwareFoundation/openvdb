// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/String.h
///
/// @authors Nick Avramoussis
///
/// @brief Provides the class definition for the equivalent IR representation
///   and logic for strings in AX.
///

#ifndef OPENVDB_AX_CODEGEN_STRING_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_STRING_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>
#include <openvdb/util/Assert.h>

#include <cstring>
#include <cstdlib>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  An extremely basic but native representation of a string class with
///   SSO support. This exists to provide an interface between the AX C++ API
///   and backend IR string logic. It is not designed to fulfill any other use
///   and should very rarely be used directly.
struct String
{
    static constexpr int64_t SSO_LENGTH = 16; // Should never be less than 2
    static_assert(SSO_LENGTH >= 2, "SSO should be greater than or equal to 2");

    String() : String("", 0) {}
    /// Construct from null terminated character string
    String(const char* str) : String(str, std::strlen(str)) {}
    String(const std::string& str) : String(str.c_str(), str.size()) {}
    String(const char* str, const int64_t size)
    {
        OPENVDB_ASSERT(str != nullptr);
        this->ptr = this->SSO; // for the isLocal check in alloc
        this->reset(str, size);
    }
    ~String() { if (!this->isLocal()) std::free(ptr); }

    String(const String& other) : String(other.ptr, other.len) {}

    ///////////////////////////////////////////////////////////////////////////

    const std::string str() const { return std::string(this->ptr, this->len); }
    const char* c_str() const { return this->ptr; }
    int64_t size() const { return this->len; }
    bool isLocal() const { return this->ptr == this->SSO; }
    void clear() { this->reset("", 0); }

    // operators

    inline operator const char*() const { return this->ptr; }

    const String& operator=(const std::string& str)
    {
        this->reset(str.c_str(), str.size());
        return *this;
    }

    const String& operator=(const String& other)
    {
        this->reset(other.ptr, other.len);
        return *this;
    }

    bool operator==(const String& other) const
    {
        return std::strcmp(this->ptr, other.ptr) == 0;
    }

    bool operator!=(const String& other) const
    {
        return !this->operator==(other);
    }

    String operator+(const String& other) const
    {
        String s;
        s.alloc(this->size() + other.size());
        std::memcpy(s.ptr, this->c_str(), this->size());
        std::memcpy(s.ptr + this->size(), other.c_str(), other.size());
        return s;
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    // Internal methods and members. These should never be used directly, but
    // remain public for C bindings in StringFunction (allowing for behaviour
    // coupling with the IR implementations).

    inline void reset(const char* str, const int64_t size)
    {
        this->alloc(size);
        std::memcpy(this->ptr, str, size);
    }

    inline void alloc(const size_t size)
    {
        if (!this->isLocal()) std::free(this->ptr);
        if (size > SSO_LENGTH-1) this->ptr = static_cast<char*>(std::malloc(size + 1));
        else this->ptr = this->SSO;
        this->ptr[size] = '\0';
        this->len = size;
    }

    char* ptr = nullptr;
    char SSO[SSO_LENGTH];
    int64_t len = 0;
};

} // namespace codegen
} // namespace ax

template<> inline ax::codegen::String zeroVal<ax::codegen::String>() { return ""; }

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_CODEGEN_STRING_HAS_BEEN_INCLUDED

