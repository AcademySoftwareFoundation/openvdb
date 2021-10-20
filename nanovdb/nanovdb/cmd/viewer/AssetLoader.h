// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file AssetLoader.h
	\brief Classes for asynchronous asset loading.
*/

#pragma once

#include <memory>
#include <string>

//! The status of an asset load.
class AssetLoadStatus
{
public:
    using Ptr = std::shared_ptr<AssetLoadStatus>;

    enum class Type : int { kSuccess = 0,
                            kUnknownProtocol,
                            kUnknownError,
                            kNotFoundError,
                            kInvalidFileError,
                            kNumTypes };

    static AssetLoadStatus::Ptr create(Type code, std::string message = "");

    Type               getType() const { return mCode; }
    const std::string& getMessage() const { return mMessage; }

private:
    AssetLoadStatus(Type code, const std::string& msg);
    std::string mMessage;
    Type        mCode;
};

class AssetRequest
{
public:
    using Ptr = std::shared_ptr<AssetRequest>;

    AssetRequest(std::string url);
    virtual ~AssetRequest();

    // Get the url of the request
    std::string getUrl() const;

    // Set timeout in ms
    void setTimeOut(int timeout_ms);

    // Get timeout in ms
    int getTimeOut() const;

protected:
    friend class AssetLoader;
    friend class AssetLoaderImpl;

    virtual bool        onResponse(std::shared_ptr<class AssetResponse> response);
    virtual std::string getRequestKey() const;

private:
    class AssetRequestImpl* mImpl;
};

class AssetResponse
{
    AssetResponse();
    AssetResponse(AssetRequest::Ptr request);

public:
    using Ptr = std::shared_ptr<AssetResponse>;

    ~AssetResponse();

public:
    static Ptr create(AssetRequest::Ptr request);
    static Ptr create(AssetRequest::Ptr request, AssetLoadStatus::Ptr status);

    std::string getUrl() const;

    void setUrl(std::string url);

    AssetRequest::Ptr getRequest() const;

    void setRequest(AssetRequest::Ptr request);

    AssetLoadStatus::Ptr getStatus() const;

    void setStatus(AssetLoadStatus::Ptr status);

private:
    class AssetResponseImpl* mImpl;
};

class AssetLoader
{
private:
    AssetLoader();

public:
    using Ptr = std::shared_ptr<AssetLoader>;

    static Ptr create(int numberOfThreads = 5);
    virtual ~AssetLoader();

    void initialize(int numberOfThreads);

    void shutdown();

    void sync();

    bool poll();

    bool load(AssetRequest::Ptr request);

private:
    class AssetLoaderImpl* mImpl;
};