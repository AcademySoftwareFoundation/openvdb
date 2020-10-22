// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file AssetLoader.h

	\author Wil Braithwaite

	\date October 9, 2020

	\brief Classes for asynchronous asset loading.
*/

#include <string>
#include <cstring>
#include <vector>
#include <map>
#include <iostream>
#include <mutex>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <fstream>

#include "StringUtils.h"
#include "CallbackPool.h"
#include "AssetLoader.h"

//---------------------------------------------------------------------------------------------------
class AssetResponseImpl
{
public:
    StringMap            mHeader;
    std::string          mUrl;
    AssetRequest::Ptr    mRequest; // original request
    AssetLoadStatus::Ptr mStatus;
};

AssetResponse::AssetResponse()
{
    mImpl = new AssetResponseImpl;
}

AssetResponse::AssetResponse(AssetRequest::Ptr request)
{
    mImpl = new AssetResponseImpl;
    mImpl->mRequest = request;
    mImpl->mUrl = request->getUrl();
}

AssetResponse::~AssetResponse()
{
    delete mImpl;
}

AssetLoadStatus::Ptr AssetLoadStatus::create(AssetLoadStatus::Type code, std::string message)
{
    return AssetLoadStatus::Ptr(new AssetLoadStatus(code, message));
}

AssetRequest::Ptr AssetResponse::getRequest() const
{
    return mImpl->mRequest;
}

void AssetResponse::setRequest(AssetRequest::Ptr request)
{
    mImpl->mUrl = request->getUrl();
    mImpl->mRequest = request;
}

std::string AssetResponse::getUrl() const
{
    return mImpl->mUrl;
}

void AssetResponse::setUrl(std::string url)
{
    mImpl->mUrl = url;
}

AssetLoadStatus::Ptr AssetResponse::getStatus() const
{
    return mImpl->mStatus;
}

void AssetResponse::setStatus(AssetLoadStatus::Ptr status)
{
    mImpl->mStatus = status;
}

AssetResponse::Ptr AssetResponse::create(AssetRequest::Ptr request, AssetLoadStatus::Ptr status)
{
    AssetResponse* response = new AssetResponse(request);
    response->setStatus(status);
    return AssetResponse::Ptr(response);
}

AssetResponse::Ptr AssetResponse::create(AssetRequest::Ptr request)
{
    return AssetResponse::Ptr(new AssetResponse(request));
}

//---------------------------------------------------------------------------------------------------
class AssetRequestImpl
{
public:
    std::string mUrl; // Url of item that needs to be requested
    int         mTimeOut = 30000; // Number of milliseconds to wait until giving up
};

AssetRequest::AssetRequest(std::string url)
{
    mImpl = new AssetRequestImpl;
    mImpl->mUrl = url;
}

AssetRequest::~AssetRequest()
{
    delete mImpl;
}

std::string AssetRequest::getUrl() const
{
    return mImpl->mUrl;
}

int AssetRequest::getTimeOut() const
{
    return mImpl->mTimeOut;
}

void AssetRequest::setTimeOut(int timeout_ms)
{
    mImpl->mTimeOut = timeout_ms;
}

std::string AssetRequest::getRequestKey() const
{
    return getUrl();
}

bool AssetRequest::onResponse(AssetResponse::Ptr response)
{
    if (response->getStatus()->getType() == AssetLoadStatus::Type::kSuccess) {
        std::cout << "Successfully loaded url(" << response->getRequest()->getUrl() << ")" << std::endl;
        return true;
    } else {
        std::cerr << "Failed to load url(" << response->getRequest()->getUrl() << ") error(" << response->getStatus()->getMessage() << ")" << std::endl;
        return false;
    }
}

//---------------------------------------------------------------------------------------------------
class AssetLoaderImpl : public CallbackPool
{
public:
    void processRequest(AssetRequest::Ptr request);

public:
    std::mutex                               mRequestLock;
    std::map<std::string, AssetRequest::Ptr> mPendingRequests;

private:
    bool handleLoadContentFile(AssetRequest::Ptr request);
};

//---------------------------------------------------------------------------------------------------
bool AssetLoaderImpl::handleLoadContentFile(AssetRequest::Ptr request)
{
    // Load file from disk directly

    std::string fileName = urlGetPath(request->getUrl());

    std::ifstream ifile(fileName.c_str(), std::ios::in | std::ios::binary);
    if (ifile.fail()) {
        std::string errorMsg = "File " + fileName + " cannot be opened.";
        return request->onResponse(AssetResponse::create(request, AssetLoadStatus::create(AssetLoadStatus::Type::kNotFoundError, errorMsg)));
    }

    ifile.ignore(std::numeric_limits<std::streamsize>::max());
    std::streamsize length = ifile.gcount();
    ifile.clear();

    if (length == 0) {
        std::string errorMsg = "File " + fileName + " does not contain any data.";
        return request->onResponse(AssetResponse::create(request, AssetLoadStatus::create(AssetLoadStatus::Type::kInvalidFileError, errorMsg)));
    }

    // for now we pass this work to the response callback.
    // we respond with kSuccess, but with no buffer.
    // This tells the response callback that it is its responsibility to load the url.

    AssetResponse::Ptr response(AssetResponse::create(request));
    response->setUrl("file://" + fileName);
    response->setStatus(AssetLoadStatus::create(AssetLoadStatus::Type::kSuccess));
    return request->onResponse(response);
}

void AssetLoaderImpl::processRequest(AssetRequest::Ptr request)
{
    std::string url = request->getUrl();

    bool isSuccessful = false;

    try {
        AssetResponse::Ptr response;
        
        std::string scheme = urlGetScheme(url);
        if (scheme == "file" || scheme.empty()) {
            isSuccessful = handleLoadContentFile(request);
        } else {
            // We pass through any unknown protocols.
            // This gives the requestint object a chance to support it in its callback.
            response = AssetResponse::create(request);
            response->setStatus(AssetLoadStatus::create(AssetLoadStatus::Type::kUnknownProtocol));
            isSuccessful = request->onResponse(response);
        }
    }
    catch (...) {
    }

    // now close the request...
    {
        std::lock_guard<std::mutex> scopedLock(mRequestLock);
        if (isSuccessful) {
            // keep the key in the pending requests, but release the request.
            mPendingRequests[request->getRequestKey()].reset();
        } else {
            // if it was not successful, then we remove the key AND release request.
            // This will allow the same request to occur again.
            mPendingRequests.erase(request->getRequestKey());
        }
    }
}

static const char* kAssetStatusErrorTypeStr[size_t(AssetLoadStatus::Type::kNumTypes)] = {"Success", "Unknown scheme", "Unknown error", "Not found", "Invalid file"};

AssetLoadStatus::AssetLoadStatus(Type code, const std::string& msg)
    : mCode(code)
    , mMessage((msg.length() > 0) ? msg : kAssetStatusErrorTypeStr[int(code)])
{
}

AssetLoader::AssetLoader()
{
    mImpl = new AssetLoaderImpl;
}

AssetLoader::~AssetLoader()
{
    sync();
    delete mImpl;
}

void AssetLoader::initialize(int /*numberOfThreads*/)
{
    mImpl->initialize();
}

void AssetLoader::shutdown()
{
    mImpl->shutdown();
}

void AssetLoader::sync()
{
    while (poll()) {
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
    }
}

bool AssetLoader::poll()
{
    bool                        foundPending = false;
    std::lock_guard<std::mutex> scopedLock(mImpl->mRequestLock);
    for (auto& it : this->mImpl->mPendingRequests) {
        if (it.second) {
            foundPending = true;
            break;
        }
    }
    return foundPending;
}

bool AssetLoader::load(AssetRequest::Ptr request)
{
    // check the request was not already submitted...
    {
        std::lock_guard<std::mutex> scopedLock(mImpl->mRequestLock);
        if (mImpl->mPendingRequests.find(request->getRequestKey()) == mImpl->mPendingRequests.end()) {
            mImpl->mPendingRequests.insert(std::make_pair(request->getRequestKey(), request));
        } else {
            // Ignoring duplicate request
            return false;
        }
    }

    // Post request to the threads handling loading and processing
    // After this call, execution continues on callback pool threads
    mImpl->sendCallback(std::bind(&AssetLoaderImpl::processRequest, mImpl, request));
    return true;
}

// Create content loader
AssetLoader::Ptr AssetLoader::create(int numberOfThreads)
{
    AssetLoader::Ptr loader(new AssetLoader);
    loader->initialize(numberOfThreads);
    return loader;
}
