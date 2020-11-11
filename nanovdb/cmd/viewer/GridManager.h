// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file GridManager.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Class definition for a manager and loader of grid assets.
*/

#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <sstream>
#include "AssetLoader.h"
#include "GridAssetUrl.h"
#include <nanovdb/util/GridHandle.h>

class GridManager
{
public:
    enum class AssetGridStatus : int { kError = -1,
                                       kUnknown = 0,
                                       kPending = 1,
                                       kLoaded = 2 };

    struct EventMessage
    {
        enum class Type { kError,
                          kWarning,
                          kInfo,
                          kDebug
        };
        Type        mType;
        std::string mMessage;
    };

private:
    struct GridAsset
    {
        using Ptr = std::shared_ptr<GridAsset>;
        std::shared_ptr<nanovdb::GridHandle<>> mGridHandle;
        nanovdb::BBoxR                         mGridBounds;
        AssetGridStatus                        mStatus;
        std::string                            mStatusMessage;
    };

    struct Asset
    {
        using Ptr = std::shared_ptr<Asset>;

        std::vector<int>                      mEvents;
        bool                                  mHasError;
        std::map<std::string, GridAsset::Ptr> mGridAssets;
        std::recursive_mutex                  mGridAssetsMutex;
        std::recursive_mutex                  mEventMutex;
    };

public:
    using AssetStatusInfoType = std::map<std::string, std::pair<bool, std::map<std::string, GridManager::AssetGridStatus>>>;

    void                                                               sync();
    bool                                                               poll();
    void                                                               initialize();
    void                                                               shutdown();
    void                                                               addGrid(const std::string& url, const std::string& gridName);
    std::tuple<nanovdb::BBoxR, std::shared_ptr<nanovdb::GridHandle<>>> getGrid(const std::string& url, const std::string& gridName) const;
    AssetStatusInfoType                                                getGridNameStatusInfo() const;
    size_t                                                             getAssetStatusEvents(const std::string& url, std::vector<int>& eventIndices) const;
    EventMessage                                                       getEventMessage(int eventIndex) const;
    int                                                                getEventMessages(std::vector<EventMessage>& messages, int startIndex) const;

    bool                     addGridsFromInternal(const std::string& url, const std::string& params);
    bool                     addGridsFromLocalFile(const std::string& url, const std::string& gridName, const std::string& localFilename);
    bool                     addGridsMetaFromLocalFile(const std::string& url, const std::string& gridName, const std::string& localFilename);
    size_t                   addEventMessage(const EventMessage& s);
    std::vector<std::string> getGridsNamesFromLocalFile(const std::string& url, const std::string& localFilename);

private:
    Asset::Ptr     ensureAsset(const std::string& url);
    GridAsset::Ptr ensureGridAsset(Asset::Ptr asset, const std::string& gridName, AssetGridStatus status);
    GridAsset::Ptr ensureGridAsset(Asset::Ptr asset, const std::string& gridName, AssetGridStatus status, nanovdb::GridHandle<>&& gridHdl);

    AssetLoader::Ptr                  mLoader;
    std::map<std::string, Asset::Ptr> mAssets;
    mutable std::recursive_mutex      mAssetsMutex;
    std::vector<EventMessage>         mEventMessages;
    mutable std::recursive_mutex      mEventListMutex;

    friend class GridAssetRequest;
};