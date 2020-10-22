// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file CallbackPool.h

	\author Wil Braithwaite

	\date October 10, 2020

	\brief Class to handle a pool of threads.
*/

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <vector>
#include <iostream>
#include <exception>

#include "CallbackPool.h"

class CallbackPoolImpl
{
    using CallbackType = CallbackPool::CallbackType;

public:
    void process(int threadNumber);

public:
    CallbackPool* handler;

    std::vector<std::thread*> mThreads;
    std::mutex mLock;

    int mNumThreads = 0;
    int mActiveThreads = 0;

    std::condition_variable mCallbackCV;
    std::condition_variable mStartCV;
    std::condition_variable mQueueIsEmptyCV;

    std::queue<CallbackType> mCallbackQueue;

    bool mDone = false;
};

void CallbackPoolImpl::process(int threadNumber)
{
    std::unique_lock<std::mutex> lock(this->mLock);

    this->handler->onSetup(threadNumber);
    this->mActiveThreads++;

    this->mStartCV.notify_one();

    CallbackType callback;

    while (!this->mDone) {
        while (!this->mCallbackQueue.empty()) {
            callback = this->mCallbackQueue.front();
            this->mCallbackQueue.pop();
            {
                lock.unlock();

                try {
                    callback();
                }
                catch (const std::exception& exp) {
                    std::cerr << "Callback failed: " << exp.what() << std::endl;
                }

                lock.lock();
            }
        }

        mQueueIsEmptyCV.notify_all();

        this->mCallbackCV.wait(lock);
    }

    this->handler->onCleanup(threadNumber);
    this->mActiveThreads--;
}

CallbackPool::CallbackPool(int numberOfThreads)
{
    // Create behind the scenes implementation
    this->mImpl = new CallbackPoolImpl;

    this->mImpl->handler = this;
    if (numberOfThreads < 1) {
        numberOfThreads = 1;
    }

    this->mImpl->mNumThreads = numberOfThreads;
}

CallbackPool::~CallbackPool()
{
    shutdown();
    delete this->mImpl;
}

void CallbackPool::initialize()
{
    std::unique_lock<std::mutex> lock(this->mImpl->mLock);
    this->mImpl->mThreads.resize(this->mImpl->mNumThreads, nullptr);

    for (int j = 0; j < this->mImpl->mNumThreads; j++) {
        this->mImpl->mThreads[j] = new std::thread(std::bind(&CallbackPoolImpl::process, this->mImpl, j));

        if (this->mImpl->mThreads[j] == nullptr) {
            throw std::runtime_error("Could not start callback handler pool thread");
        }
    }

    // Wait for all threads to be ready...
    while (true) {
        this->mImpl->mStartCV.wait(lock);
        if (this->mImpl->mActiveThreads == this->mImpl->mNumThreads) {
            break;
        }
    }
}

void CallbackPool::shutdown()
{
    {
        std::unique_lock<std::mutex> lock(this->mImpl->mLock);
        this->mImpl->mDone = true;
        this->mImpl->mCallbackCV.notify_all();
    }

    // wait for and cleanup threads
    for (size_t j = 0; j < this->mImpl->mThreads.size(); j++) {
        if (this->mImpl->mThreads[j] != nullptr) {
            this->mImpl->mThreads[j]->join();
            delete this->mImpl->mThreads[j];
            this->mImpl->mThreads[j] = nullptr;
        }
    }
}

void CallbackPool::sync()
{
    std::unique_lock<std::mutex> lock(this->mImpl->mLock);
    while (!this->mImpl->mCallbackQueue.empty()) {
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
    }
}

void CallbackPool::sendCallback(CallbackType callback)
{
    std::unique_lock<std::mutex> lock(this->mImpl->mLock);
    this->mImpl->mCallbackQueue.push(callback);
    this->mImpl->mCallbackCV.notify_one();
}

void CallbackPool::onSetup(int threadNum)
{
}

void CallbackPool::onCleanup(int threadNum)
{
}

//---------------------------------------------------------------------------------------------------
