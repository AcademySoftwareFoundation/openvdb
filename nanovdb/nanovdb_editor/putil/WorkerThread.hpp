// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_editor/putil/WorkerThread.hpp

    \author Petra Hapalova

    \brief
*/

#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <atomic>
#include <unordered_set>
#include <unordered_map>
#include <string>

namespace pnanovdb_util
{
// Execute tasks queue synchronously on a single thread
class WorkerThread
{
public:
    using TaskId = size_t;

    static TaskId invalidTaskId()
    {
        return std::numeric_limits<TaskId>::max();
    }

    WorkerThread() : m_running(true), m_nextTaskId(0), m_currentTaskId(invalidTaskId())
    {
        m_thread = std::thread(&WorkerThread::threadFunc, this);
    }

    ~WorkerThread()
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_running = false;
            m_condition.notify_one();
        }
        if (m_thread.joinable())
        {
            m_thread.join();
        }
    }

    template<typename F, typename... Args>
    TaskId enqueue(F&& f, Args&&... args)
    {
        TaskId taskId = generateTaskId();

        // Create a wrapper that updates the task status when done
        auto taskWrapper = [this, taskId, func = std::bind(std::forward<F>(f), std::forward<Args>(args)...)]() -> bool
            {
                try
                {
                    // Mark this task as the currently running task
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_currentTaskId = taskId;
                        m_taskProgres = 0.0f;
                        m_taskProgresText = "";
                    }

                    // Execute the actual task
                    bool result = func();
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_taskResults[m_currentTaskId] = result;
                    }
                    clearTaskId();
                    return result;
                }
                catch (...)
                {
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_taskResults[m_currentTaskId] = false;
                    }
                    clearTaskId();
                    return false;
                }
            };

        // Enqueue the wrapped task
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_tasks.push(std::move(taskWrapper));
        }
        m_condition.notify_one();

        return taskId;
    }

    bool isTaskRunning(const TaskId& taskId)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_currentTaskId == taskId && m_currentTaskId != invalidTaskId();
    }

    bool hasRunningTask()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_currentTaskId != invalidTaskId();
    }

    bool isTaskCompleted(const TaskId& taskId)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_completedTasks.find(taskId) != m_completedTasks.end();
    }

    void removeCompletedTask(const TaskId& taskId)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_completedTasks.erase(taskId);
    }

    float getTaskProgress(const TaskId& taskId)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_taskProgres;
    }

    std::string getTaskProgressText(const TaskId& taskId)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_taskProgresText;
    }

    void updateTaskProgress(float progress = 0.f, std::string text = "")
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_taskProgres = std::max(0.f, std::min(1.0f, progress));
        if (!text.empty())
        {
            m_taskProgresText = text;
        }
    }

    bool isTaskSuccessful(const TaskId& taskId)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        auto it = m_taskResults.find(taskId);
        return it != m_taskResults.end() ? it->second : false;
    }

private:
    TaskId generateTaskId()
    {
        return m_nextTaskId++;
    }

    void clearTaskId()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_currentTaskId != invalidTaskId())
        {
            m_completedTasks.insert(m_currentTaskId);
            m_taskProgres = 1.0f;
            m_taskProgresText = "";
        }
        m_currentTaskId = invalidTaskId();
    }

    void threadFunc()
    {
        while (true)
        {
            std::function<bool()> task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_condition.wait(lock, [this] { return !m_running || !m_tasks.empty(); });

                if (!m_running && m_tasks.empty())
                {
                    return;
                }

                task = std::move(m_tasks.front());
                m_tasks.pop();
            }

            // Execute the task - result and exceptions are already handled in the wrapper
            task();
        }
    }

    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::queue<std::function<bool()>> m_tasks;
    bool m_running;
    std::atomic<TaskId> m_nextTaskId;
    TaskId m_currentTaskId;
    std::unordered_set<TaskId> m_completedTasks; // Stores IDs of completed tasks
    std::unordered_map<TaskId, bool> m_taskResults; // Stores results of completed tasks
    float m_taskProgres;
    std::string m_taskProgresText;
};

} // namespace pnanovdb_editor
