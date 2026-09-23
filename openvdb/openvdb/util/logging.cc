// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "logging.h"

#include <algorithm> // for std::remove()
#include <cstdio> // for fileno, stderr
#include <iostream>
#include <mutex>
#include <vector>
#if defined(_WIN32)
#include <io.h> // for _isatty
#else
#include <unistd.h> // for isatty
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace logging {

namespace {

struct Registry
{
    std::mutex mMutex;
    std::vector<Sink::Ptr> mSinks;
    bool mInitialized = false;
};

Registry&
getRegistry()
{
    static Registry sRegistry;
    return sRegistry;
}

std::atomic<Level> sLevel{Level::Warn};

// Must be called with the registry mutex held.
void
installDefaultSink(Registry& registry)
{
    registry.mInitialized = true;
    registry.mSinks.push_back(std::make_shared<ConsoleSink>());
}

} // anonymous namespace

Level
getLevel()
{
    return sLevel.load(std::memory_order_relaxed);
}

void
setLevel(Level level)
{
    sLevel.store(level, std::memory_order_relaxed);
}

bool
isEnabledFor(Level level)
{
    return level >= sLevel.load(std::memory_order_relaxed);
}

void
setLevel(int& argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) { // note: skip argv[0]
        const std::string arg{argv[i]};
        bool remove = true;
        if (arg == "-debug")      { setLevel(Level::Debug); }
        else if (arg == "-error") { setLevel(Level::Error); }
        else if (arg == "-fatal") { setLevel(Level::Fatal); }
        else if (arg == "-info")  { setLevel(Level::Info); }
        else if (arg == "-warn")  { setLevel(Level::Warn); }
        else { remove = false; }
        if (remove) argv[i] = nullptr;
    }
    auto end = std::remove(argv + 1, argv + argc, nullptr);
    argc = static_cast<int>(end - argv);
}


Sink::Sink(const std::string& name, Level threshold)
    : mName(name)
    , mThreshold(threshold)
{
}

Sink::~Sink() {}

const std::string&
Sink::name() const
{
    return mName;
}

Level
Sink::threshold() const
{
    return mThreshold.load(std::memory_order_relaxed);
}

void
Sink::setThreshold(Level threshold)
{
    mThreshold.store(threshold, std::memory_order_relaxed);
}


void
addSink(const Sink::Ptr& sink)
{
    auto& registry = getRegistry();
    std::lock_guard<std::mutex> lock(registry.mMutex);
    registry.mInitialized = true;
    auto& sinks = registry.mSinks;
    auto it = std::find_if(sinks.begin(), sinks.end(),
        [&](const Sink::Ptr& existing) { return existing->name() == sink->name(); });
    if (it != sinks.end()) *it = sink;
    else sinks.push_back(sink);
}

bool
removeSink(const std::string& name)
{
    auto& registry = getRegistry();
    std::lock_guard<std::mutex> lock(registry.mMutex);
    auto& sinks = registry.mSinks;
    auto it = std::find_if(sinks.begin(), sinks.end(),
        [&](const Sink::Ptr& sink) { return sink->name() == name; });
    if (it == sinks.end()) return false;
    sinks.erase(it);
    return true;
}

Sink::Ptr
findSink(const std::string& name)
{
    auto& registry = getRegistry();
    std::lock_guard<std::mutex> lock(registry.mMutex);
    for (const auto& sink : registry.mSinks) {
        if (sink->name() == name) return sink;
    }
    return nullptr;
}


namespace {

const char* levelPrefix(Level level, bool useColor)
{
    if (!useColor) {
        switch (level) {
            case Level::Debug: return "DEBUG";
            case Level::Info:  return "INFO";
            case Level::Warn:  return "WARNING";
            case Level::Error: return "ERROR";
            case Level::Fatal: return "FATAL";
        }
        return "";
    }
    switch (level) {
        case Level::Debug: return "\033[32mDEBUG\033[0m"; // green
        case Level::Info:  return "\033[36mINFO\033[0m"; // cyan
        case Level::Warn:  return "\033[35mWARNING\033[0m"; // magenta
        case Level::Error: return "\033[31mERROR\033[0m"; // red
        case Level::Fatal: return "\033[31mFATAL\033[0m"; // red
    }
    return "";
}

} // anonymous namespace

const char*
ConsoleSink::defaultName()
{
    return "console";
}

bool
ConsoleSink::stderrIsTerminal()
{
#if defined(_WIN32)
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(fileno(stderr)) != 0;
#endif
}

ConsoleSink::ConsoleSink(bool useColor)
    : Sink(defaultName())
    , mUseColor(useColor)
{
}

bool
ConsoleSink::useColor() const
{
    return mUseColor.load(std::memory_order_relaxed);
}

void
ConsoleSink::setUseColor(bool useColor)
{
    mUseColor.store(useColor, std::memory_order_relaxed);
}

void
ConsoleSink::append(Level level, const std::string& message, const char*, int)
{
    std::ostringstream buffer;
    buffer << levelPrefix(level, useColor()) << ": " << message << "\n";
    std::cerr << buffer.str();
}

void
initialize(bool useColor)
{
    auto& registry = getRegistry();
    std::lock_guard<std::mutex> lock(registry.mMutex);
    if (registry.mInitialized) return;
    installDefaultSink(registry);
    static_cast<ConsoleSink&>(*registry.mSinks.back()).setUseColor(useColor);
}

void
initialize(int& argc, char* argv[], bool useColor)
{
    initialize(useColor);
    setLevel(argc, argv);
}


namespace internal {

void
dispatch(Level level, const std::string& message, const char* file, int line)
{
    std::vector<Sink::Ptr> sinks;
    {
        auto& registry = getRegistry();
        std::lock_guard<std::mutex> lock(registry.mMutex);
        if (!registry.mInitialized) installDefaultSink(registry);
        sinks = registry.mSinks;
    }
    try {
        for (const auto& sink : sinks) {
            if (level < sink->threshold()) continue;
            sink->append(level, message, file, line);
        }
    } catch (...) {
        // A sink must not throw. If one does anyway, drop the message rather
        // than propagate an exception out of a logging call.
    }
}

} // namespace internal

} // namespace logging
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
