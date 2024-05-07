// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file cmd/cli.h
///
/// @brief  This file is not intended to be part of the public API but is used
///   internally by the vdb_ax command line too _and_ volume/point executables.
/// @todo   Stabilize this API, consider making the parameter types public here
///   and on the executables so that we can use it in the Houdini SOP/other
///   components in the future.

#ifndef OPENVDB_AX_VDB_AX_CLI_HAS_BEEN_INCLUDED
#define OPENVDB_AX_VDB_AX_CLI_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb_ax/Exceptions.h>
#include <openvdb/util/Assert.h>

#include <cstring>
#include <functional>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {
namespace cli {

template <typename T> struct ParamBuilder;
template <typename T> struct BasicParamBuilder;

/// @brief Output nchar characters from str to a stream with custom indentation
/// @param os        output stream
/// @param str       null terminated char array to output
/// @param nchars    num chars of str to output
/// @param maxWidth  column number of max output
/// @param indent    function that returns an indent for each line
/// @param ignoreNewLines  whether to ignore new lines in str
/// @param trimWhitespace  whether to ignore whitespce at the start of lines
inline void oswrap(std::ostream& os,
    const char* str,
    const size_t nchars,
    const size_t maxWidth,
    const std::function<std::string(size_t)> indent = nullptr,
    const bool ignoreNewLines = false,
    const bool trimWhitespace = false)
{
    size_t pos = 0;
    size_t line = 0;
    while (pos < nchars)
    {
        // usually the same indent per line
        size_t lineMaxWidth = maxWidth;

        if (indent) { // handle custom indent
            std::string i = indent(line++);
            if (i.size() > maxWidth) i.resize(maxWidth); // clamp to maxWidth
            lineMaxWidth -= i.size(); // new limit on this line
            os << i;
        }

        const char* start = str + pos; // move to new start
        if (trimWhitespace) {
            while (pos < nchars && *start == ' ') { ++start; ++pos; } // skip spaces
        }
        if (pos >= nchars) break;
        const size_t endpos = pos + lineMaxWidth; // set new end

        if (!ignoreNewLines) {
            // scane for new line in this range (memchr doesn't stop on \0)
            // hence the std::min on the max width
            const char* nl = (const char*)std::memchr(start, '\n',
                std::min(lineMaxWidth, nchars-pos));
            if (nl) {
                while (start <= nl) { os << *start; ++start; ++pos; }
                continue;
            }
        }

        // if endpos greater than max, output rest and exit
        if (endpos >= nchars) {
            const char* end = str + nchars;
            while (start < end) { os << *start; ++start; ++pos; }
            break;
        }

        const char* end = str + endpos;
        while (end > start && *end != ' ') --end; // back scan to next space
        if (end == start) end = str + endpos; // no spaces found, reset end
        while (start != end) { os << *start; ++start; ++pos; } // output range
        if (*start == ' ') { ++start; ++pos; } // always skip the space break (replaced with \n)
        os << '\n';
    }
}

/// @brief Output a CLI parameter with its documentation with formatting.
/// @param os        output stream
/// @param opts      parameter options
/// @param doc       parameter docs
/// @param verbose   when false, only outputs the docs first sentance
/// @param argSpace  minimum allowed whitespace between opts end and doc start
/// @param docBegin  column number where doc printing should start
/// @param maxWidth  column number of max output
inline void usage(std::ostream& os,
        const std::vector<const char*>& opts,
        const char* doc,
        const bool verbose,
        const int32_t argSpace = 4,
        const int32_t docBegin = 30,
        const int32_t maxWidth = 100)
{
    const int32_t argGap = 2; // gap from arg to start of docs

    for (int32_t i = 0; i < argSpace; ++i) os << ' ';
    auto iter = opts.cbegin();
    for (; iter != opts.cend()-1; ++iter) os << *iter << ", ";
    os << *iter;

    if (!doc) {
        os << '\n';
        return;
    }

    size_t len = (opts.size() - 1) * 2; // include ", " sep
    for (const auto& name : opts) {
        len += std::strlen(name);
    }

    // compute whitespace required till doc start
    int32_t current = argSpace + static_cast<int32_t>(len);
    const int32_t whitespace = docBegin - current;
    if (whitespace < argGap) {
        os << '\n';
        current = 0;
    }

    // doc chars to output. If not verbose, only first sentance
    size_t doclen = std::strlen(doc);
    if (!verbose) {
        // memchr doesn't stop on '\0' so needs max size
        const char* stop = (const char*)std::memchr(doc, '.', doclen);
        if (stop) doclen = stop - doc;
    }

    // output docs. If docs start on a new line (current == 0) then just forward
    // to oswrap - otherwise, if the start on the same line then add whitespace,
    // output the first line, wrap the rest
    const std::string indent(docBegin, ' ');
    if (current == 0) {
        oswrap(os, doc, doclen, maxWidth, [&](size_t) { return indent; });
    }
    else {
        OPENVDB_ASSERT(whitespace >= argGap);
        // space between name and docs
        for (int32_t i = 0; i < whitespace; ++i) os << ' ';

        // if the docs don't it on one line, output to the remaining max width
        // and forward the rest to oswrap
        size_t remain = static_cast<size_t>(std::max(0, maxWidth-(current+whitespace)));
        if (doclen > remain) {
            // calculate how much goes onto the rest of this line
            while (remain > 0 && doc[remain] != ' ') --remain;
            for (size_t i = 0; i < remain; ++i, ++doc) os << *doc;
            // skip space break (if found)
            if (*doc == ' ') { ++doc; --doclen; }
            os << '\n';
            OPENVDB_ASSERT(doclen >= remain);
            doclen -= remain;
            oswrap(os, doc, doclen, maxWidth, [&](size_t) { return indent; });
        }
        else {
            for (size_t i = 0; i < doclen; ++i, ++doc) os << *doc;
        }
    }
    os << '\n';
}

struct ParamBase
{
    ParamBase() = default;
    virtual ~ParamBase() = default;
    inline const std::vector<const char*>& opts() const { return mOpts; }
    inline const char* doc() const { return mDoc; }
    virtual void init(const char* arg, const uint32_t idx = 0) = 0;
    virtual bool acceptsArg() const = 0;
    virtual bool requiresArg() const = 0;
    virtual bool isInit() const = 0;
protected:
    std::vector<const char*> mOpts;
    const char* mDoc {nullptr};
};

template <typename T>
struct BasicParam
{
    using CB = std::function<void(T&, const char*)>;
    BasicParam(T&& v) : mValue(std::move(v)) {}
    BasicParam(const T& v) : mValue(v) {}

    BasicParam(const BasicParam&) = default;
    BasicParam(BasicParam&&) = default;
    BasicParam& operator=(const BasicParam&) = default;
    BasicParam& operator=(BasicParam&&) = default;

    inline void set(const T& v) { mValue = v; }
    inline T& get() { return mValue; }
    inline const T& get() const { return mValue; }
    inline operator const T&() const { return mValue; }
private:
    friend BasicParamBuilder<T>;
protected:
    BasicParam() = default;
    T mValue;
};

template <typename T>
struct Param : public BasicParam<T>, ParamBase
{
    ~Param() override = default;
    Param(const Param&) = default;
    Param(Param&&) = default;
    Param& operator=(const Param&) = default;
    Param& operator=(Param&&) = default;

    // CB1 callback passes the value to store
    // CB2 callback passed the value and the argument provided
    // CB3 callback passed the value, the argument provided and the index to
    //   the opt that was used.
    using CB1 = std::function<void(T&)>;
    using CB2 = std::function<void(T&, const char*)>;
    using CB3 = std::function<void(T&, const char*, const uint32_t)>;

    inline bool acceptsIndex() const { return this->opts().size() > 1; }
    inline bool acceptsArg() const override { return static_cast<bool>(mCb2 || mCb3); }
    inline bool requiresArg() const override { return static_cast<bool>(!mCb1); }
    inline bool isInit() const override { return mInit; }
    inline void init(const char* arg, const uint32_t idx = 0) override
    {
        OPENVDB_ASSERT((!arg && mCb1) || (arg && mCb2) || (arg && mCb3));
        if (!arg) mCb1(BasicParam<T>::mValue);
        else if (mCb3 && this->acceptsIndex()) {
            mCb3(BasicParam<T>::mValue, arg, idx);
        }
        else {
            OPENVDB_ASSERT(mCb2);
            mCb2(BasicParam<T>::mValue, arg);
        }
        mInit = true;
    }

private:
    friend ParamBuilder<T>;
    Param() = default;
    // @todo could be a union if we need to add more
    CB1 mCb1 {nullptr};
    CB2 mCb2 {nullptr};
    CB3 mCb3 {nullptr};
    bool mInit {false};
};


template <typename T, typename E = void> struct DefaultCallback {
    static typename Param<T>::CB1 get() { return nullptr; }
};
template <> struct DefaultCallback<bool, void> {
    static typename Param<bool>::CB1 get() { return [](bool& v) { v = true; }; }
};
template <> struct DefaultCallback<std::string, void> {
    static typename Param<std::string>::CB2 get() {
        return [](std::string& v, const char* arg) { v = std::string(arg); };
    }
};
template <typename T>
struct DefaultCallback<T,
    typename std::enable_if<
        std::is_integral<T>::value &&
        !std::is_same<T, bool>::value>::type>
{
    static typename Param<T>::CB2 get() {
        return [](T& v, const char* arg) {
            try { v = T(std::stol(arg)); }
            catch(...) {
                OPENVDB_THROW(CLIError, "Unable to convert argument: '"
                    << arg << "' to a valid interger");
            }
        };
    }
};
template <typename T>
struct DefaultCallback<T,
    typename std::enable_if<
        std::is_floating_point<T>::value>::type>
{
    static typename Param<T>::CB2 get() {
        return [](T& v, const char* arg) {
            try { v = T(std::stod(arg)); }
            catch(...) {
                OPENVDB_THROW(CLIError, "Unable to convert argument '"
                    << arg << "' to a valid floating point number");
            }
        };
    }
};

template <typename T>
struct BasicParamBuilder
{
    using ParamT = BasicParam<T>;
    BasicParamBuilder() = default;
    BasicParamBuilder& addOpt(const char*) { return *this; }
    BasicParamBuilder& setDoc(const char*) { return *this; }
    BasicParamBuilder& setDefault(const T& v) { mParam.set(v); return *this; }
    BasicParamBuilder& setCB(const typename Param<T>::CB1) { return *this; }
    BasicParamBuilder& setCB(const typename Param<T>::CB2) { return *this; }
    BasicParamBuilder& setCB(const typename Param<T>::CB3) { return *this; }
    ParamT&& get() { return std::move(mParam); }
private:
    ParamT mParam;
};

template <typename T>
struct ParamBuilder
{
    using ParamT = Param<T>;
    ParamBuilder() : mParam() {
        mParam.ParamBase::mOpts.clear();
    }
    ParamBuilder& addOpt(const char* opt) {
        OPENVDB_ASSERT(opt);
        OPENVDB_ASSERT(opt[0] == '-' || std::strchr(opt, ' ') == nullptr);
        mParam.ParamBase::mOpts.emplace_back(opt);
        return *this;
    }
    ParamBuilder& setDoc(const char* doc) { mParam.ParamBase::mDoc = doc; return *this; }
    ParamBuilder& setDefault(const T& def) { mParam.mValue = def; return *this; }
    ParamBuilder& setCB(const typename ParamT::CB1 cb) { mParam.mCb1 = cb; return *this; }
    ParamBuilder& setCB(const typename ParamT::CB2 cb) { mParam.mCb2 = cb; return *this; }
    ParamBuilder& setCB(const typename ParamT::CB3 cb) { mParam.mCb3 = cb; return *this; }
    ParamT&& get() {
        OPENVDB_ASSERT(!mParam.ParamBase::mOpts.empty());
        if (!(mParam.mCb1 || mParam.mCb2 || mParam.mCb3)) {
            this->setCB(DefaultCallback<T>::get());
        }
        return std::move(mParam);
    }
private:
    ParamT mParam;
};

inline void init(const size_t argc, const char* argv[],
    const std::vector<ParamBase*>& positional,
    const std::vector<ParamBase*>& optional,
    bool* used = nullptr)
{
    size_t i = 0;
    for (auto& P : positional) {
        if (i >= argc) return;
        const char* arg = argv[i];
        if (arg[0] == '-') break;
        P->init(arg);
        if (used) used[i] = true;
        ++i;
    }

    for (; i < argc; ++i)
    {
        const char* current = argv[i];
        const char* next = (((i + 1 < argc) && argv[i+1][0] != '-') ?
            argv[++i] : nullptr);

        if (current[0] != '-') {
            OPENVDB_THROW(CLIError,
                "positional argument '" <<
                    current << "' used after optional argument");
        }

        for (auto& P : optional)
        {
            int32_t optIndex = -1;
            for (const auto& opt : P->opts())
            {
                ++optIndex;
                size_t count = std::numeric_limits<size_t>::max();
                if (auto space = std::strchr(opt, ' ')) {
                    count = space - opt;
                }

                // assumes strings are null terminated
                if (std::strncmp(current, opt, count) != 0) continue;

                if (next) {
                    if (P->requiresArg() || P->acceptsArg()) {
                        P->init(next, uint32_t(optIndex));
                        if (used) used[i-1] = used[i] = true;
                    }
                    else {
                        OPENVDB_THROW(CLIError, "argument was provided to '" << current <<
                            "' which does not accept one");
                    }
                }
                else {
                    if (P->requiresArg()) {
                        OPENVDB_THROW(CLIError, "option '" << current << "' requires an argument");
                    }
                    P->init(nullptr);
                    if (used) used[i] = true;
                }
            }
        }
    }
}

} // namespace cli
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_VDB_AX_CLI_HAS_BEEN_INCLUDED
