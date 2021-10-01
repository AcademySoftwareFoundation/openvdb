// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file CallbackPool.h
	\brief Class to handle a pool of threads.
*/

#pragma once

#include <memory>
#include <functional>

// This is a multi threaded version of the callback handler
// It spreads work over multiple threads. There is a common
// event stack, with function objects that need to called
// This is used for the loader, so we can load in parallel

class CallbackPool
{
public:
    using Ptr = std::shared_ptr<CallbackPool>;
    using CallbackType = std::function<void()>;

    // Default constructor
    CallbackPool(int numberOfThreads = 5);

    // Overloadable desctuctor
    virtual ~CallbackPool();

    // Set the number of threads to use
    // TODO: Currently implementation is not yet dynamic so call
    // it before start. Plan is to make it dynamic and scale based
    // on needs and rendering speeds
    void setNumberOfThreads(int numberOfThreads);

    // Start the callback handling thread
    // All threads are initialized after this call
    void initialize();

    // Stop the callback handling thread
    // All threads are deallocated after this call
    void shutdown();

    // Send function object for handling callback
    void sendCallback(CallbackType callback);

    void sync();

protected:
    // Called when the callback handler starts
    // (called on a every internal thread)
    // threadNum specifies which thread one is
    virtual void onSetup(int threadNum);

    // Called when the callback handler stops
    // (called on every internal thread)
    virtual void onCleanup(int threadNum);

private:
    // Internals of the event handler
    class CallbackPoolImpl* mImpl = nullptr;
    friend class CallbackPoolImpl;
};

//---------------------------------------------------------------------------------------------------
