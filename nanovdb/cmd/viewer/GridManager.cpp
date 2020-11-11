// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file GridManager.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of GridManager.
*/

#define _USE_MATH_DEFINES
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>

#include "GridManager.h"
#include "StringUtils.h"

#include <nanovdb/util/IO.h> // for NanoVDB file import
#include <nanovdb/util/GridBuilder.h>
#if defined(NANOVDB_USE_OPENVDB)
#include <openvdb/io/Stream.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#endif

// create a nanovdb grid from a typeName.
// Returns a valid GridHandle on success.
static nanovdb::GridHandle<> createInternalGrid(std::string typeName, std::string /*url*/)
{
    if (typeName == "empty") {
        nanovdb::GridBuilder<float> builder(0);
        return builder.getHandle<>();
    } else if (typeName == "ls_sphere_100") {
        return nanovdb::createLevelSetSphere(100.0f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), typeName);
    } else if (typeName == "ls_torus_100") {
        return nanovdb::createLevelSetTorus(100.0f, 50.f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), typeName);
    } else if (typeName == "ls_box_100") {
        return nanovdb::createLevelSetBox(100.0f, 100.0f, 100.0f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), typeName);
    } else if (typeName == "fog_sphere_100") {
        return nanovdb::createFogVolumeSphere(100.0f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), typeName);
    } else if (typeName == "fog_torus_100") {
        return nanovdb::createFogVolumeTorus(100.0f, 50.f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), typeName);
    } else if (typeName == "fog_box_100") {
        return nanovdb::createFogVolumeBox(100.0f, 100.0f, 100.0f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), typeName);
    } else if (typeName == "points_sphere_100") {
        return nanovdb::createPointSphere(1, 100.0f, nanovdb::Vec3d(0), 1.0f, nanovdb::Vec3R(0), typeName);
    } else if (typeName == "points_torus_100") {
        return nanovdb::createPointTorus(1, 100.0f, 50.f, nanovdb::Vec3d(0), 1.0f, nanovdb::Vec3R(0), typeName);
    } else if (typeName == "points_box_100") {
        return nanovdb::createPointBox(1, 100.0f, 100.0f, 100.0f, nanovdb::Vec3d(0), 1.0f, nanovdb::Vec3R(0), typeName);
    } else if (typeName == "ls_bbox_100") {
        return nanovdb::createLevelSetBBox(100.0f, 100.0f, 100.0f, 10.f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), typeName);
    } else {
        return nanovdb::GridHandle<>();
    }
}

class GridAssetRequest : public AssetRequest
{
public:
    GridAssetRequest(GridManager* mgr, GridManager::Asset::Ptr asset, std::string url, std::string gridName)
        : AssetRequest(url)
        , mManager(mgr)
        , mAsset(asset)
        , mGridName(gridName)
    {
    }

private:
    GridManager*            mManager;
    GridManager::Asset::Ptr mAsset;
    std::string             mGridName;

    std::string getRequestKey() const override
    {
        return getUrl() + "|" + mGridName;
    }

    bool onResponse(AssetResponse::Ptr response) override
    {
        std::string url = getUrl();
        auto        scheme = urlGetScheme(url);
        bool        hasError = false;
        try {
            if (response->getStatus()->getType() == AssetLoadStatus::Type::kUnknownProtocol) {
                if (scheme == "internal") {
                    // If it is an internal scheme, we handle it here.
                    hasError = !mManager->addGridsFromInternal(url, mGridName);
                } else {
                    // we failed to load this asset.
                    hasError = true;
                    throw std::runtime_error("Unsupported scheme: " + scheme);
                }
            } else if (response->getStatus()->getType() == AssetLoadStatus::Type::kSuccess) {
                if (scheme == "file") {
                    // If it is a file scheme, then we expect to just be passed the local filename in the URL.
                    hasError = !mManager->addGridsFromLocalFile(url, mGridName, urlGetPath(response->getUrl()));
                }
            } else {
                hasError = true;
                throw std::runtime_error(response->getStatus()->getMessage());
            }
        }
        catch (const std::exception& e) {
            // something went wrong on import.
            hasError = true;
            std::lock_guard<std::recursive_mutex> scopedLock(mAsset->mEventMutex);
            mAsset->mEvents.push_back(mManager->addEventMessage({GridManager::EventMessage::Type::kError, "Can't load grid \"" + mGridName + "\" from \"" + url + "\"; " + e.what()}));
        }

        if (hasError) {
            mAsset->mHasError = true;
        }

        return !hasError;
    }
};

void GridManager::initialize()
{
#if defined(NANOVDB_USE_OPENVDB)
    openvdb::initialize();
#endif

    mLoader = AssetLoader::create();
}

void GridManager::shutdown()
{
}

void GridManager::sync()
{
    mLoader->sync();
}

bool GridManager::poll()
{
    return mLoader->poll();
}

GridManager::AssetStatusInfoType GridManager::getGridNameStatusInfo() const
{
    GridManager::AssetStatusInfoType      m;
    std::lock_guard<std::recursive_mutex> scopedLock(mAssetsMutex);
    for (auto& it : mAssets) {
        auto asset = it.second;
        m[it.first].first = asset->mHasError;
        for (auto& git : asset->mGridAssets) {
            m[it.first].second.insert(std::make_pair(git.first, git.second->mStatus));
        }
    }

    return m;
}

size_t GridManager::addEventMessage(const EventMessage& s)
{
    std::lock_guard<std::recursive_mutex> scopedLock(mEventListMutex);
    mEventMessages.push_back(s);
    return mEventMessages.size() - 1;
}

GridManager::EventMessage GridManager::getEventMessage(int i) const
{
    std::lock_guard<std::recursive_mutex> scopedLock(mEventListMutex);
    assert(i < mEventMessages.size());
    return mEventMessages[i];
}

int GridManager::getEventMessages(std::vector<EventMessage>& messages, int startIndex) const
{
    messages.clear();
    std::lock_guard<std::recursive_mutex> scopedLock(mEventListMutex);
    for (size_t i = startIndex; i < mEventMessages.size(); ++i)
        messages.push_back(mEventMessages[i]);
    return messages.size();
}

size_t GridManager::getAssetStatusEvents(const std::string& url, std::vector<int>& eventIndices) const
{
    std::lock_guard<std::recursive_mutex> scopedLock(mAssetsMutex);
    auto                                  it = mAssets.find(url);
    if (it != mAssets.end()) {
        auto                                  asset = it->second;
        std::lock_guard<std::recursive_mutex> scopedLock(asset->mEventMutex);
        eventIndices = asset->mEvents;
        return asset->mEvents.size();
    }
    return 0;
}

void GridManager::addGrid(const std::string& url, const std::string& gridName)
{
    // check to see if this "url & grid" asset is resident.
    bool isResident = false;

    auto asset = ensureAsset(url);
    {
        // if we specified a grid then check if it is already resident...
        if (gridName.length() > 0) {
            std::lock_guard<std::recursive_mutex> scopedLock(asset->mGridAssetsMutex);
            auto                                  git = asset->mGridAssets.find(gridName);
            if (git != asset->mGridAssets.end() && git->second->mStatus != AssetGridStatus::kError) {
                isResident = true;
            }
        }
    }

    // the grid is not resident or errored, so make a request.
    if (!isResident) {
#if 0
        // this REALLY slows down the main thread. So probably not worth it.
        if (urlGetScheme(url) == "file") {
            addGridsMetaFromLocalFile(url, gridName, urlGetPath(url));
        }
#endif
        AssetRequest::Ptr request(new GridAssetRequest(this, asset, url, gridName));
        if (mLoader->load(request)) {
            addEventMessage(EventMessage{EventMessage::Type::kInfo, "Requesting asset(" + url + ") grid(" + gridName + ")"});
        }
    }
}

std::tuple<nanovdb::BBoxR, std::shared_ptr<nanovdb::GridHandle<>>> GridManager::getGrid(const std::string& url, const std::string& gridName) const
{
    std::lock_guard<std::recursive_mutex> scopedLock(mAssetsMutex);
    auto                                  it = mAssets.find(url);
    if (it != mAssets.end()) {
        auto asset = it->second;
        if (gridName.length()) {
            std::lock_guard<std::recursive_mutex> scopedLock(asset->mGridAssetsMutex);
            auto                                  git = asset->mGridAssets.find(gridName);
            if (git != asset->mGridAssets.end()) {
                if (git->second->mStatus == AssetGridStatus::kLoaded) {
                    return std::make_tuple(git->second->mGridBounds, git->second->mGridHandle);
                } else {
                    return std::make_tuple(git->second->mGridBounds, nullptr);
                }
            }
        } else {
            // just take first grid...
            auto git = asset->mGridAssets.begin();
            if (git != asset->mGridAssets.end()) {
                if (git->second->mStatus == AssetGridStatus::kLoaded) {
                    return std::make_tuple(git->second->mGridBounds, git->second->mGridHandle);
                } else {
                    return std::make_tuple(git->second->mGridBounds, nullptr);
                }
            }
        }
    }
    return {nanovdb::BBoxR(), nullptr};
}

GridManager::Asset::Ptr GridManager::ensureAsset(const std::string& url)
{
    Asset::Ptr asset;
    mAssetsMutex.lock();
    auto it = mAssets.find(url);
    if (it != mAssets.end()) {
        asset = it->second;
    } else {
        asset = std::make_shared<Asset>();
        asset->mHasError = false;
        mAssets.insert(std::make_pair(url, asset));
    }
    mAssetsMutex.unlock();
    return asset;
}

bool GridManager::addGridsFromInternal(const std::string& url, const std::string& fragment)
{
    auto           asset = ensureAsset(url);
    GridAsset::Ptr gridAsset;

    auto internalGridType = fragment;

    auto gridHdl = createInternalGrid(internalGridType, url);
    if (!gridHdl) {
        std::lock_guard<std::recursive_mutex> scopedLock(asset->mEventMutex);
        asset->mHasError = true;
        asset->mEvents.push_back(addEventMessage({EventMessage::Type::kError, "Can't generate \"" + url + "#" + internalGridType + "\""}));
        return false;
    } else {
        gridAsset = ensureGridAsset(asset, internalGridType, AssetGridStatus::kLoaded, std::move(gridHdl));
        addEventMessage({GridManager::EventMessage::Type::kInfo, "Successfully generated \"" + url + "#" + internalGridType + "\""});
        return true;
    }
}

std::vector<std::string> GridManager::getGridsNamesFromLocalFile(const std::string& url, const std::string& localFilename)
{
    std::vector<std::string> result;
    std::string              extension = urlGetPathExtension(localFilename);
    try {
        if (extension == "vdb") {
#if defined(NANOVDB_USE_OPENVDB)
            openvdb::io::File file(localFilename);
            file.open(true);
            auto gridMetas = file.readAllGridMetadata();
            for (auto& gridMeta : *gridMetas) {
                result.push_back(gridMeta->getName());
                addEventMessage({GridManager::EventMessage::Type::kInfo, "Found grid \"" + gridMeta->getName() + "\" from \"" + url + "\""});
            }
#else
            throw std::runtime_error("OpenVDB is not supported in this build. Please recompile with OpenVDB support.");
#endif
        } else {
            // load all the grids in the file...
            auto gridMetas = nanovdb::io::readGridMetaData(localFilename);
            for (auto& gridMeta : gridMetas) {
                result.push_back(gridMeta.gridName);
                addEventMessage({GridManager::EventMessage::Type::kInfo, "Found grid \"" + gridMeta.gridName + "\" from \"" + url + "\""});
            }
        }
    }
    catch (const std::exception& e) {
        addEventMessage({GridManager::EventMessage::Type::kError, "Can't load grid meta from \"" + url + "\" - " + e.what()});
    }
    return result;
}

bool GridManager::addGridsMetaFromLocalFile(const std::string& url, const std::string& gridName, const std::string& localFilename)
{
    std::string extension = urlGetPathExtension(localFilename);

    Asset::Ptr     asset = ensureAsset(url);
    GridAsset::Ptr gridAsset;
    if (!gridName.empty()) {
        gridAsset = ensureGridAsset(asset, gridName, AssetGridStatus::kPending);
    }

    try {
        if (extension == "vdb") {
#if defined(NANOVDB_USE_OPENVDB)
            openvdb::io::File file(localFilename);
            file.open(true);
            if (gridName.empty()) {
                auto gridMetas = file.readAllGridMetadata();
                for (auto& gridMeta : *gridMetas) {
                    gridAsset = ensureGridAsset(asset, gridMeta->getName(), AssetGridStatus::kPending);
                    auto* fileBboxMin = static_cast<openvdb::TypedMetadata<openvdb::Vec3i>*>((*gridMeta)[openvdb::GridBase::META_FILE_BBOX_MIN].get());
                    auto* fileBboxMax = static_cast<openvdb::TypedMetadata<openvdb::Vec3i>*>((*gridMeta)[openvdb::GridBase::META_FILE_BBOX_MAX].get());
                    if (fileBboxMin && fileBboxMax) {
                        auto bboxMin = fileBboxMin->value();
                        auto bboxMind = gridMeta->indexToWorld(bboxMin);
                        auto bboxMax = fileBboxMax->value();
                        auto bboxMaxd = gridMeta->indexToWorld(bboxMax);
                        gridAsset->mGridBounds = nanovdb::BBoxR(nanovdb::Vec3R(bboxMind[0], bboxMind[1], bboxMind[2]), nanovdb::Vec3R(bboxMaxd[0], bboxMaxd[1], bboxMaxd[2]));
                    }
                    addEventMessage({GridManager::EventMessage::Type::kInfo, "Successfully loaded meta \"" + gridMeta->getName() + "\" from \"" + url + "\""});
                }

            } else {
                auto gridMeta = file.readGridMetadata(gridName);
                if (gridMeta) {
                    gridAsset = ensureGridAsset(asset, gridName, AssetGridStatus::kPending);
                    auto* fileBboxMin = static_cast<openvdb::TypedMetadata<openvdb::Vec3i>*>((*gridMeta)[openvdb::GridBase::META_FILE_BBOX_MIN].get());
                    auto* fileBboxMax = static_cast<openvdb::TypedMetadata<openvdb::Vec3i>*>((*gridMeta)[openvdb::GridBase::META_FILE_BBOX_MAX].get());
                    if (fileBboxMin && fileBboxMax) {
                        auto bboxMin = fileBboxMin->value();
                        auto bboxMind = gridMeta->indexToWorld(bboxMin);
                        auto bboxMax = fileBboxMax->value();
                        auto bboxMaxd = gridMeta->indexToWorld(bboxMax);
                        gridAsset->mGridBounds = nanovdb::BBoxR(nanovdb::Vec3R(bboxMind[0], bboxMind[1], bboxMind[2]), nanovdb::Vec3R(bboxMaxd[0], bboxMaxd[1], bboxMaxd[2]));
                    }
                    addEventMessage({GridManager::EventMessage::Type::kInfo, "Successfully loaded meta \"" + gridName + "\" from \"" + url + "\""});
                }
            }
#else
            throw std::runtime_error("OpenVDB is not supported in this build. Please recompile with OpenVDB support.");
#endif
        } else {
            // load all the grids in the file...
            if (gridName.length() > 0) {
                auto                             gridMetas = nanovdb::io::readGridMetaData(localFilename);
                const nanovdb::io::GridMetaData* gridMeta = nullptr;
                for (size_t i = 0; i < gridMetas.size(); ++i) {
                    if (gridMetas[i].gridName == gridName) {
                        gridMeta = &gridMetas[i];
                        break;
                    }
                }
                if (!gridMeta)
                    throw std::exception();

                gridAsset = ensureGridAsset(asset, gridName, AssetGridStatus::kPending);
                gridAsset->mGridBounds = gridMeta->worldBBox;
                addEventMessage({GridManager::EventMessage::Type::kInfo, "Successfully loaded meta \"" + gridName + "\" from \"" + url + "\""});
            } else {
                auto gridMetas = nanovdb::io::readGridMetaData(localFilename);
                for (auto& gridMeta : gridMetas) {
                    gridAsset = ensureGridAsset(asset, gridMeta.gridName, AssetGridStatus::kPending);
                    gridAsset->mGridBounds = gridMeta.worldBBox;
                    addEventMessage({GridManager::EventMessage::Type::kInfo, "Successfully loaded meta \"" + gridMeta.gridName + "\" from \"" + url + "\""});
                }
            }
        }
    }
    catch (const std::exception& e) {
        if (!gridName.empty()) {
            if (gridAsset) {
                gridAsset->mStatus = AssetGridStatus::kError;
                gridAsset->mStatusMessage = e.what();
            }
            std::lock_guard<std::recursive_mutex> scopedLock(asset->mEventMutex);
            asset->mEvents.push_back(addEventMessage({EventMessage::Type::kError, "Can't load grid \"" + gridName + "\" from \"" + url + "\"; " + e.what()}));
        } else {
            std::lock_guard<std::recursive_mutex> scopedLock(asset->mEventMutex);
            asset->mEvents.push_back(addEventMessage({EventMessage::Type::kError, "Can't load from \"" + url + "\"; " + e.what()}));
        }
        asset->mHasError = true;
        return false;
    }
    return true;
}

bool GridManager::addGridsFromLocalFile(const std::string& url, const std::string& gridName, const std::string& localFilename)
{
    const int verbosity = 0;

    Asset::Ptr     asset = ensureAsset(url);
    GridAsset::Ptr gridAsset;
#if 0
    if (!gridName.empty()) {
        // if the gridName is specified, then we can insert the grid-asset into the manager now as pending.
        // this is useful for grids which take a long time to load.
        // However it does mean that grid-assets which have errors will be in the asset list.
        gridAsset = ensureGridAsset(asset, gridName, AssetGridStatus::kPending);
    }
#endif

    try {
        std::string extension = urlGetPathExtension(localFilename);

        if (extension == "vdb") {
#if defined(NANOVDB_USE_OPENVDB)
            openvdb::io::File file(localFilename);
            file.open(true);
            if (gridName.empty()) {
                auto grids = file.getGrids();
                for (auto& grid : *grids) {
                    auto gridHdl = nanovdb::openToNanoVDB(grid, nanovdb::StatsMode::Default, nanovdb::ChecksumMode::Default, false, verbosity);
                    gridAsset = ensureGridAsset(asset, grid->getName(), AssetGridStatus::kLoaded, std::move(gridHdl));
                    addEventMessage({GridManager::EventMessage::Type::kInfo, "Successfully loaded grid \"" + grid->getName() + "\" from \"" + url + "\""});
                }

            } else {
                auto grid = file.readGrid(gridName);
                if (grid) {
                    auto gridHdl = nanovdb::openToNanoVDB(grid, nanovdb::StatsMode::Default, nanovdb::ChecksumMode::Default, false, verbosity);
                    gridAsset = ensureGridAsset(asset, gridName, AssetGridStatus::kLoaded, std::move(gridHdl));
                    addEventMessage({GridManager::EventMessage::Type::kInfo, "Successfully loaded grid \"" + gridName + "\" from \"" + url + "\""});
                }
            }
#else
            throw std::runtime_error("OpenVDB is not supported in this build. Please recompile with OpenVDB support.");
#endif
        } else {
            // load all the grids in the file...
            if (gridName.length() > 0) {
                auto gridHdl = nanovdb::io::readGrid<>(localFilename, gridName);
                gridAsset = ensureGridAsset(asset, gridName, AssetGridStatus::kLoaded, std::move(gridHdl));
                addEventMessage({GridManager::EventMessage::Type::kInfo, "Successfully loaded grid \"" + gridName + "\" from \"" + url + "\""});
            } else {
                auto list = nanovdb::io::readGridMetaData(localFilename);
                for (auto& m : list) {
                    auto gridHdl = nanovdb::io::readGrid<>(localFilename, m.gridName);
                    gridAsset = ensureGridAsset(asset, m.gridName, AssetGridStatus::kLoaded, std::move(gridHdl));
                    addEventMessage({GridManager::EventMessage::Type::kInfo, "Successfully loaded grid \"" + m.gridName + "\" from \"" + url + "\""});
                }
            }
        }
    }
    catch (const std::exception& e) {
        if (!gridName.empty()) {
            if (gridAsset) {
                gridAsset->mStatus = AssetGridStatus::kError;
                gridAsset->mStatusMessage = e.what();
            }
            std::lock_guard<std::recursive_mutex> scopedLock(asset->mEventMutex);
            asset->mEvents.push_back(addEventMessage({EventMessage::Type::kError, "Can't load grid \"" + gridName + "\" from \"" + url + "\"; " + e.what()}));
        } else {
            std::lock_guard<std::recursive_mutex> scopedLock(asset->mEventMutex);
            asset->mEvents.push_back(addEventMessage({EventMessage::Type::kError, "Can't load from \"" + url + "\"; " + e.what()}));
        }
        asset->mHasError = true;
        return false;
    }

    return true;
}

GridManager::GridAsset::Ptr GridManager::ensureGridAsset(Asset::Ptr asset, const std::string& gridName, AssetGridStatus status)
{
    std::lock_guard<std::recursive_mutex> scopedLock(asset->mGridAssetsMutex);

    GridAsset::Ptr gridAsset;
    auto           it = asset->mGridAssets.find(gridName);
    if (it != asset->mGridAssets.end()) {
        gridAsset = it->second;
    } else {
        gridAsset = std::make_shared<GridAsset>();
        asset->mGridAssets.insert(std::make_pair(gridName, gridAsset));
        gridAsset->mStatus = status;
        gridAsset->mGridBounds = nanovdb::BBoxR(nanovdb::Vec3R(-1), nanovdb::Vec3R(-1));
    }
    return gridAsset;
}

GridManager::GridAsset::Ptr GridManager::ensureGridAsset(Asset::Ptr asset, const std::string& gridName, AssetGridStatus status, nanovdb::GridHandle<>&& gridHdl)
{
    if (!gridHdl) {
        throw std::runtime_error("Invalid grid, " + gridName);
    }

    std::lock_guard<std::recursive_mutex> scopedLock(asset->mGridAssetsMutex);

    GridAsset::Ptr gridAsset;
    auto           it = asset->mGridAssets.find(gridName);
    if (it != asset->mGridAssets.end()) {
        gridAsset = it->second;
    } else {
        gridAsset = std::make_shared<GridAsset>();
        gridAsset->mGridBounds = (gridHdl.gridMetaData()->indexBBox().empty()) ? nanovdb::BBoxR() : gridHdl.gridMetaData()->worldBBox();
        gridAsset->mGridHandle = std::make_shared<nanovdb::GridHandle<>>(std::move(gridHdl));
        asset->mGridAssets.insert(std::make_pair(gridName, gridAsset));
    }

    gridAsset->mStatus = status;
    return gridAsset;
}
