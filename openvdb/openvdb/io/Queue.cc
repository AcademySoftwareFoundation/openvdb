// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file Queue.cc
/// @author Peter Cucka

#include "Queue.h"
#include "File.h"
#include "Stream.h"
#include <openvdb/Exceptions.h>
#include <openvdb/util/logging.h>

#include <tbb/concurrent_hash_map.h>
#include <tbb/task_arena.h>

#include <thread>
#include <algorithm> // for std::max()
#include <atomic>
#include <iostream>
#include <map>
#include <mutex>
#include <chrono>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

namespace {

// Abstract base class for queuable TBB tasks that adds a task completion callback
class Task
{
public:
    Task(Queue::Id id): mId(id) {}
    virtual ~Task() {}

    Queue::Id id() const { return mId; }

    void setNotifier(Queue::Notifier& notifier) { mNotify = notifier; }
    virtual void execute() const = 0;

protected:
    void notify(Queue::Status status) const { if (mNotify) mNotify(this->id(), status); }

private:
    Queue::Id mId;
    Queue::Notifier mNotify;
};


// Queuable TBB task that writes one or more grids to a .vdb file or an output stream
class OutputTask : public Task
{
public:
    OutputTask(Queue::Id id, const GridCPtrVec& grids, const Archive& archive,
        const MetaMap& metadata)
        : Task(id)
        , mGrids(grids)
        , mArchive(archive.copy())
        , mMetadata(metadata) {}
    ~OutputTask() override {}

    void execute() const override
    {
        Queue::Status status = Queue::FAILED;
        try {
            mArchive->write(mGrids, mMetadata);
            status = Queue::SUCCEEDED;
        } catch (std::exception& e) {
            if (const char* msg = e.what()) {
                OPENVDB_LOG_ERROR(msg);
            }
        } catch (...) {}
        this->notify(status);
    }

private:
    GridCPtrVec mGrids;
    SharedPtr<Archive> mArchive;
    MetaMap mMetadata;
};

} // unnamed namespace


////////////////////////////////////////


// Private implementation details of a Queue
struct Queue::Impl
{
    using NotifierMap = std::map<Queue::Id, Queue::Notifier>;
    /// @todo Provide more information than just "succeeded" or "failed"?
    using StatusMap = tbb::concurrent_hash_map<Queue::Id, Queue::Status>;

    Impl()
        : mTimeout(Queue::DEFAULT_TIMEOUT)
        , mCapacity(Queue::DEFAULT_CAPACITY)
        , mNextId(1)
        , mNextNotifierId(1)
    {
        mNumTasks = 0; // note: must explicitly zero-initialize atomics
    }
    ~Impl() {}

    // Disallow copying of instances of this class.
    Impl(const Impl&);
    Impl& operator=(const Impl&);

    // This method might be called from multiple threads.
    void setStatus(Queue::Id id, Queue::Status status)
    {
        StatusMap::accessor acc;
        mStatus.insert(acc, id);
        acc->second = status;
    }

    // This method might be called from multiple threads.
    void setStatusWithNotification(Queue::Id id, Queue::Status status)
    {
        const bool completed = (status == SUCCEEDED || status == FAILED);

        // Update the task's entry in the status map with the new status.
        this->setStatus(id, status);

        // If the client registered any callbacks, call them now.
        bool didNotify = false;
        {
            // tbb::concurrent_hash_map does not support concurrent iteration
            // (i.e., iteration concurrent with insertion or deletion),
            // so we use a mutex-protected STL map instead.  But if a callback
            // invokes a notifier method such as removeNotifier() on this queue,
            // the result will be a deadlock.
            /// @todo Is it worth trying to avoid such deadlocks?
            std::lock_guard<std::mutex> lock(mNotifierMutex);
            if (!mNotifiers.empty()) {
                didNotify = true;
                for (NotifierMap::const_iterator it = mNotifiers.begin();
                    it != mNotifiers.end(); ++it)
                {
                    it->second(id, status);
                }
            }
        }
        // If the task completed and callbacks were called, remove
        // the task's entry from the status map.
        if (completed) {
            if (didNotify) {
                StatusMap::accessor acc;
                if (mStatus.find(acc, id)) {
                    mStatus.erase(acc);
                }
            }
            --mNumTasks;
        }
    }

    bool canEnqueue() const { return mNumTasks < Int64(mCapacity); }

    void enqueue(OutputTask& task)
    {
        auto start = std::chrono::steady_clock::now();
        while (!canEnqueue()) {
            std::this_thread::sleep_for(/*0.5s*/std::chrono::milliseconds(500));
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            const double seconds = double(duration.count()) / 1000.0;
            if (seconds > double(mTimeout)) {
                OPENVDB_THROW(RuntimeError,
                    "unable to queue I/O task; " << mTimeout << "-second time limit expired");
            }
        }
        Queue::Notifier notify = std::bind(&Impl::setStatusWithNotification, this,
            std::placeholders::_1, std::placeholders::_2);
        task.setNotifier(notify);
        this->setStatus(task.id(), Queue::PENDING);

        // get the global task arena
        tbb::task_arena arena(tbb::task_arena::attach{});
        arena.enqueue([task = std::move(task)] { task.execute(); });
        ++mNumTasks;
    }

    Index32 mTimeout;
    Index32 mCapacity;
    std::atomic<Int32> mNumTasks;
    Index32 mNextId;
    StatusMap mStatus;
    NotifierMap mNotifiers;
    Index32 mNextNotifierId;
    std::mutex mNotifierMutex;
};


////////////////////////////////////////


Queue::Queue(Index32 capacity): mImpl(new Impl)
{
    mImpl->mCapacity = capacity;
}


Queue::~Queue()
{
    // Wait for all queued tasks to complete (successfully or unsuccessfully).
    /// @todo Allow the queue to be destroyed while there are uncompleted tasks
    /// (e.g., by keeping a static registry of queues that also dispatches
    /// or blocks notifications)?
    while (mImpl->mNumTasks > 0) {
        std::this_thread::sleep_for(/*0.5s*/std::chrono::milliseconds(500));
    }
}


////////////////////////////////////////


bool Queue::empty() const { return (mImpl->mNumTasks == 0); }
Index32 Queue::size() const { return Index32(std::max<Int32>(0, mImpl->mNumTasks)); }
Index32 Queue::capacity() const { return mImpl->mCapacity; }
void Queue::setCapacity(Index32 n) { mImpl->mCapacity = std::max<Index32>(1, n); }

/// @todo void Queue::setCapacity(Index64 bytes);

/// @todo Provide a way to limit the number of tasks in flight
/// (e.g., by enqueueing tbb::tasks that pop Tasks off a concurrent_queue)?

/// @todo Remove any tasks from the queue that are not currently executing.
//void clear() const;

Index32 Queue::timeout() const { return mImpl->mTimeout; }
void Queue::setTimeout(Index32 sec) { mImpl->mTimeout = sec; }


////////////////////////////////////////


Queue::Status
Queue::status(Id id) const
{
    Impl::StatusMap::const_accessor acc;
    if (mImpl->mStatus.find(acc, id)) {
        const Status status = acc->second;
        if (status == SUCCEEDED || status == FAILED) {
            mImpl->mStatus.erase(acc);
        }
        return status;
    }
    return UNKNOWN;
}


Queue::Id
Queue::addNotifier(Notifier notify)
{
    std::lock_guard<std::mutex> lock(mImpl->mNotifierMutex);
    Queue::Id id = mImpl->mNextNotifierId++;
    mImpl->mNotifiers[id] = notify;
    return id;
}


void
Queue::removeNotifier(Id id)
{
    std::lock_guard<std::mutex> lock(mImpl->mNotifierMutex);
    Impl::NotifierMap::iterator it = mImpl->mNotifiers.find(id);
    if (it != mImpl->mNotifiers.end()) {
        mImpl->mNotifiers.erase(it);
    }
}


void
Queue::clearNotifiers()
{
    std::lock_guard<std::mutex> lock(mImpl->mNotifierMutex);
    mImpl->mNotifiers.clear();
}


////////////////////////////////////////


Queue::Id
Queue::writeGrid(GridBase::ConstPtr grid, const Archive& archive, const MetaMap& metadata)
{
    return writeGridVec(GridCPtrVec(1, grid), archive, metadata);
}


Queue::Id
Queue::writeGridVec(const GridCPtrVec& grids, const Archive& archive, const MetaMap& metadata)
{
    const Queue::Id taskId = mImpl->mNextId++;
    OutputTask task(taskId, grids, archive, metadata);
    mImpl->enqueue(task);
    return taskId;
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
