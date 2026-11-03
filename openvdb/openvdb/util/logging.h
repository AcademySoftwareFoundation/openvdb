// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_UTIL_LOGGING_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_LOGGING_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Platform.h>

#include <atomic>
#include <memory>
#include <sstream>
#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace logging {

/// @brief Message severity level, in increasing order of severity.
enum class Level { Debug = 0, Info, Warn, Error, Fatal };

/// @brief Return the current severity threshold. Messages below this are dropped.
OPENVDB_API Level getLevel();

/// @brief Set the severity threshold. Messages below this are dropped.
OPENVDB_API void setLevel(Level);

/// @brief Return true if a message of the given level would be logged.
/// @details Used by the logging macros to skip formatting a message that would
///   be discarded. This is a single relaxed atomic load.
OPENVDB_API bool isEnabledFor(Level);

/// @brief If "-debug", "-info", "-warn", "-error" or "-fatal" is found
/// in the given array of command-line arguments, set the logging level
/// appropriately and remove the relevant argument(s) from the array.
OPENVDB_API void setLevel(int& argc, char* argv[]);

/// @brief Receives formatted log messages. Subclass this to route OpenVDB
///   diagnostics into a host application.
/// @details Sinks are called on the thread that logged the message, so an
///   implementation must be thread safe. A sink must not throw.
///
/// @par Example
/// @code
/// class MySink: public openvdb::logging::Sink
/// {
/// public:
///     MySink(): Sink("mysink") {}
///     void append(openvdb::logging::Level level, const std::string& message,
///         const char* file, int line) override
///     {
///         // forward to the host application's own logging system
///     }
/// };
///
/// openvdb::logging::addSink(std::make_shared<MySink>());
/// @endcode
class OPENVDB_API Sink
{
public:
    using Ptr = std::shared_ptr<Sink>;

    /// @param name       Unique identifier, used by removeSink() and findSink().
    /// @param threshold  Messages below this level are not delivered to this sink,
    ///   in addition to the global threshold from setLevel().
    explicit Sink(const std::string& name, Level threshold = Level::Debug);
    virtual ~Sink();

    const std::string& name() const;

    Level threshold() const;
    void setThreshold(Level);

    /// @param level    Severity of the message.
    /// @param message  The formatted message text, without a trailing newline.
    /// @param file     Source file the message was logged from, may be null.
    /// @param line     Line in @a file, or 0 if unknown.
    virtual void append(Level level, const std::string& message,
        const char* file, int line) = 0;

private:
    const std::string mName;
    std::atomic<Level> mThreshold;
};

/// @brief Register a sink. Replaces any existing sink with the same name.
OPENVDB_API void addSink(const Sink::Ptr&);

/// @brief Remove the sink with the given name. Returns true if one was removed.
OPENVDB_API bool removeSink(const std::string& name);

/// @brief Return the sink with the given name, or nullptr.
OPENVDB_API Sink::Ptr findSink(const std::string& name);

/// @brief The sink installed by default, writing to std::cerr.
class OPENVDB_API ConsoleSink: public Sink
{
public:
    /// The name under which this sink is registered.
    static const char* defaultName();

    /// @brief Return true if stderr is connected to a terminal.
    static bool stderrIsTerminal();

    explicit ConsoleSink(bool useColor = stderrIsTerminal());

    bool useColor() const;
    void setUseColor(bool);

    void append(Level, const std::string& message, const char* file, int line) override;

private:
    std::atomic<bool> mUseColor;
};

/// @brief Install the default console sink if no sink has been installed yet.
/// @details By default, color is enabled only if stderr is connected to a terminal.
OPENVDB_API void initialize(bool useColor = ConsoleSink::stderrIsTerminal());

/// @brief Initialize and then apply any level flags found in the arguments.
/// @details By default, color is enabled only if stderr is connected to a terminal.
OPENVDB_API void initialize(int& argc, char* argv[], bool useColor = ConsoleSink::stderrIsTerminal());

/// @brief Sets the level on construction and restores the previous level on
/// destruction.
struct LevelScope
{
    Level level;
    explicit LevelScope(Level newLevel): level(getLevel()) { setLevel(newLevel); }
    ~LevelScope() { setLevel(level); }
};

/// @cond OPENVDB_DOCS_INTERNAL

namespace internal {

OPENVDB_API void dispatch(Level level, const std::string& message,
    const char* file, int line);

} // namespace internal

/// @endcond

} // namespace logging
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#define OPENVDB_LOG(level, message) \
    do { \
        if (openvdb::logging::isEnabledFor(openvdb::logging::Level::level)) { \
            std::ostringstream _buf; \
            _buf << message; \
            openvdb::logging::internal::dispatch( \
                openvdb::logging::Level::level, _buf.str(), __FILE__, __LINE__); \
        } \
    } while (0)

/// Log an info message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_INFO(message)   OPENVDB_LOG(Info, message)
/// Log a warning message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_WARN(message)   OPENVDB_LOG(Warn, message)
/// Log an error message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_ERROR(message)  OPENVDB_LOG(Error, message)
/// Log a fatal error message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_FATAL(message)  OPENVDB_LOG(Fatal, message)
#ifdef NDEBUG
/// In debug builds only, log a debugging message of the form '<TT>someVar << "text" << ...</TT>'.
#define OPENVDB_LOG_DEBUG(message)
#else
/// In debug builds only, log a debugging message of the form '<TT>someVar << "text" << ...</TT>'.
#define OPENVDB_LOG_DEBUG(message)  OPENVDB_LOG(Debug, message)
#endif
/// @brief Log a debugging message in both debug and optimized builds.
/// @warning Don't use this in performance-critical code.
#define OPENVDB_LOG_DEBUG_RUNTIME(message)  OPENVDB_LOG(Debug, message)

#endif // OPENVDB_UTIL_LOGGING_HAS_BEEN_INCLUDED
