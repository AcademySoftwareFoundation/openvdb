// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/util/logging.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

class TestLogging: public ::testing::Test
{
};

namespace {

class RecordingSink: public openvdb::logging::Sink
{
public:
    struct Record {
        openvdb::logging::Level level;
        std::string message;
        std::string file;
        int line;
    };

    explicit RecordingSink(const std::string& name,
        openvdb::logging::Level threshold = openvdb::logging::Level::Debug)
        : Sink(name, threshold)
    {
    }

    void append(openvdb::logging::Level level, const std::string& message,
        const char* file, int line) override
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mRecords.push_back(Record{level, message, file ? file : "", line});
    }

    std::vector<Record> mRecords;

private:
    std::mutex mMutex;
};

} // anonymous namespace


TEST_F(TestLogging, testLevelDefaultAndRoundTrip)
{
    using namespace openvdb::logging;

    LevelScope scope(getLevel());

    EXPECT_EQ(Level::Warn, scope.level);

    for (Level level : {Level::Debug, Level::Info, Level::Warn, Level::Error, Level::Fatal}) {
        setLevel(level);
        EXPECT_EQ(level, getLevel());
    }
}


TEST_F(TestLogging, testLevelScopeRestoresOnThrow)
{
    using namespace openvdb::logging;

    setLevel(Level::Warn);

    EXPECT_THROW({
        LevelScope scope(Level::Fatal);
        EXPECT_EQ(Level::Fatal, getLevel());
        throw std::runtime_error("boom");
    }, std::runtime_error);

    EXPECT_EQ(Level::Warn, getLevel());
}


TEST_F(TestLogging, testSinkThresholdFiltering)
{
    using namespace openvdb::logging;

    LevelScope scope(Level::Debug);

    auto sink = std::make_shared<RecordingSink>("test.threshold", Level::Warn);
    addSink(sink);

    OPENVDB_LOG_DEBUG("debug message");
    OPENVDB_LOG_INFO("info message");
    OPENVDB_LOG_WARN("warn message");
    OPENVDB_LOG_ERROR("error message");

    removeSink(sink->name());

    ASSERT_EQ(size_t(2), sink->mRecords.size());
    EXPECT_EQ(Level::Warn, sink->mRecords[0].level);
    EXPECT_EQ("warn message", sink->mRecords[0].message);
    EXPECT_EQ(Level::Error, sink->mRecords[1].level);
    EXPECT_EQ("error message", sink->mRecords[1].message);
}


TEST_F(TestLogging, testFileAndLineRecorded)
{
    using namespace openvdb::logging;

    LevelScope scope(Level::Debug);

    auto sink = std::make_shared<RecordingSink>("test.fileline");
    addSink(sink);

    const int line = __LINE__ + 1;
    OPENVDB_LOG_WARN("checking file and line");

    removeSink(sink->name());

    ASSERT_EQ(size_t(1), sink->mRecords.size());
    EXPECT_EQ(__FILE__, sink->mRecords[0].file);
    EXPECT_EQ(line, sink->mRecords[0].line);
}


TEST_F(TestLogging, testGlobalLevelSuppressesAllSinks)
{
    using namespace openvdb::logging;

    LevelScope scope(Level::Fatal);

    auto sink = std::make_shared<RecordingSink>("test.global");
    addSink(sink);

    OPENVDB_LOG_WARN("should not be seen");
    OPENVDB_LOG_ERROR("should not be seen either");

    removeSink(sink->name());

    EXPECT_TRUE(sink->mRecords.empty());
}


TEST_F(TestLogging, testPerSinkThresholdIndependentOfGlobalLevel)
{
    using namespace openvdb::logging;

    LevelScope scope(Level::Debug);

    auto sink = std::make_shared<RecordingSink>("test.perSink");
    addSink(sink);

    sink->setThreshold(Level::Error);
    OPENVDB_LOG_WARN("filtered by sink threshold");
    OPENVDB_LOG_ERROR("passes sink threshold");

    removeSink(sink->name());

    ASSERT_EQ(size_t(1), sink->mRecords.size());
    EXPECT_EQ("passes sink threshold", sink->mRecords[0].message);
}


TEST_F(TestLogging, testAddRemoveFindSink)
{
    using namespace openvdb::logging;

    auto sink = std::make_shared<RecordingSink>("test.addRemoveFind");
    addSink(sink);
    EXPECT_EQ(sink, findSink("test.addRemoveFind"));

    auto replacement = std::make_shared<RecordingSink>("test.addRemoveFind");
    addSink(replacement);
    EXPECT_EQ(replacement, findSink("test.addRemoveFind"));

    EXPECT_TRUE(removeSink("test.addRemoveFind"));
    EXPECT_FALSE(removeSink("test.addRemoveFind"));
    EXPECT_EQ(nullptr, findSink("test.addRemoveFind"));
    EXPECT_EQ(nullptr, findSink("test.doesNotExist"));
}


TEST_F(TestLogging, testSetLevelFromArgs)
{
    using namespace openvdb::logging;

    LevelScope scope(getLevel());

    char arg0[] = "progname";
    char arg1[] = "-info";
    char arg2[] = "otherArg";
    char* argv[] = {arg0, arg1, arg2};
    int argc = 3;

    setLevel(argc, argv);

    EXPECT_EQ(Level::Info, getLevel());
    ASSERT_EQ(2, argc);
    EXPECT_STREQ("progname", argv[0]);
    EXPECT_STREQ("otherArg", argv[1]);
}


TEST_F(TestLogging, testFilteredMessagesDoNotEvaluateArguments)
{
    using namespace openvdb::logging;

    LevelScope scope(Level::Fatal);

    int evaluationCount = 0;
    auto sideEffect = [&]() -> int { ++evaluationCount; return 0; };

    OPENVDB_LOG_WARN("value is " << sideEffect());

    EXPECT_EQ(0, evaluationCount);
}


TEST_F(TestLogging, testConcurrentLoggingAndSinkChanges)
{
    using namespace openvdb::logging;

    LevelScope scope(Level::Debug);

    std::atomic<bool> stop{false};
    std::vector<std::thread> loggers;
    for (int i = 0; i < 4; ++i) {
        loggers.emplace_back([&stop]() {
            while (!stop.load(std::memory_order_relaxed)) {
                OPENVDB_LOG_WARN("concurrent message");
            }
        });
    }

    std::thread sinkChurner([&stop]() {
        int i = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            auto sink = std::make_shared<RecordingSink>("test.churn");
            addSink(sink);
            removeSink(sink->name());
            if (++i > 1000) break;
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop = true;

    for (auto& t : loggers) t.join();
    sinkChurner.join();
}
