// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/HostBuffer.h>
#include <algorithm>

//////////////////////////////////////////////
class MapPoolAllocatorContext
{
public:
    uint64_t                                                 mNextUniqueId;
    std::map<uint64_t, std::shared_ptr<nanovdb::HostBuffer>> mGridBufferMap;

    MapPoolAllocatorContext()
        : mNextUniqueId(0)
    {
        std::cout << "Creating MapPoolAllocatorContext" << std::endl;
    }

    ~MapPoolAllocatorContext()
    {
        mGridBufferMap.clear();
        std::cout << "Destroyed MapPoolAllocatorContext" << std::endl;
    }
};

// Note: we use std::enable_shared_from_this so that we can construct a GridHandle
// in the createGridHandle method.
class MapPoolBuffer
{
    MapPoolBuffer(const MapPoolBuffer&) = delete;
    MapPoolBuffer& operator=(const MapPoolBuffer&) = delete;

public:
    uint64_t                                 mId = -1;
    std::shared_ptr<nanovdb::HostBuffer>     mBuffer;
    std::shared_ptr<MapPoolAllocatorContext> mState;

public:
    void initialize()
    {
        if (!mState) {
            mState = std::make_shared<MapPoolAllocatorContext>();
        }
    }

    MapPoolBuffer() = default;

    MapPoolBuffer(MapPoolBuffer&& other)
    {
        mId = other.mId;
        other.mId = -1;
        mBuffer = std::move(other.mBuffer);
        mState = std::move(other.mState);
    }

    MapPoolBuffer& operator=(MapPoolBuffer&& other)
    {
        mId = other.mId;
        other.mId = -1;
        mBuffer = std::move(other.mBuffer);
        mState = std::move(other.mState);
        return *this;
    }

    static MapPoolBuffer create(uint64_t size, const MapPoolBuffer* context)
    {
        MapPoolBuffer buffer;

        if (!context || !context->mState) {
            throw std::runtime_error("context has not been created.");
        }

        buffer.mState = context->mState;
        if (!buffer.mState)
            buffer.mState = std::make_shared<MapPoolAllocatorContext>();
        buffer.mId = buffer.mState->mNextUniqueId++;
        buffer.mBuffer = std::make_shared<nanovdb::HostBuffer>(size);
        buffer.mState->mGridBufferMap.emplace(buffer.mId, buffer.mBuffer);

        printf("Creating resources id(%d) buffer(%p)\n", (int)buffer.mId, buffer.mBuffer.get());
        return buffer;
    }

    void clear()
    {
        if (mBuffer.use_count() == 2) {
            mState->mGridBufferMap.erase(mId);
            printf("Destroying resources id(%d) buffer(%p)\n", (int)mId, mBuffer.get());
        }
        mId = -1;
        mBuffer = nullptr;
    }

    uint64_t       size() const { return getBuffer(mId).size(); }
    void*          data() { return getBuffer(mId).data(); }
    const void*    data() const { return getBuffer(mId).data(); }

    std::vector<uint64_t> getGridKeys() const
    {
        std::vector<uint64_t> keys;
        std::transform(
            mState->mGridBufferMap.begin(),
            mState->mGridBufferMap.end(),
            std::back_inserter(keys),
            [](const std::map<uint64_t, std::shared_ptr<nanovdb::HostBuffer>>::value_type& pair) { return pair.first; });
        return keys;
    }

    MapPoolBuffer copy(uint64_t id) const
    {
        MapPoolBuffer buffer;
        buffer.mState = mState;
        buffer.mId = id;
        buffer.mBuffer = mState->mGridBufferMap[id];
        return buffer;
    }

private:
    const nanovdb::HostBuffer& getBuffer(uint64_t id) const
    {
        auto it = mState->mGridBufferMap.find(id);
        assert(it != mState->mGridBufferMap.end());
        return *it->second.get();
    }

    nanovdb::HostBuffer& getBuffer(uint64_t id)
    {
        auto it = mState->mGridBufferMap.find(id);
        assert(it != mState->mGridBufferMap.end());
        return *it->second.get();
    }
};

int main()
{
    // this parent scope allocator contains a std::map of buffers.
    // this allocator also acts as a manager for the grid resources,
    // and allows multiple GridHandles to point to the same data!
    MapPoolBuffer bufferContext;
    bufferContext.initialize();

    nanovdb::GridHandle<MapPoolBuffer> anotherHdl;

    try {
        std::vector<nanovdb::GridHandle<MapPoolBuffer>> gridHdls;

        // create two grids...
        gridHdls.push_back(nanovdb::tools::createLevelSetSphere<float >(100.0, nanovdb::Vec3d(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "spheref", nanovdb::tools::StatsMode::BBox, nanovdb::CheckMode::Partial, bufferContext));
        gridHdls.push_back(nanovdb::tools::createLevelSetSphere<double>(100.0, nanovdb::Vec3d( 20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "sphered", nanovdb::tools::StatsMode::BBox, nanovdb::CheckMode::Partial, bufferContext));

        // share grid[0]'s buffer into a parent-scope handle to prevent deletion.
        anotherHdl = nanovdb::GridHandle<MapPoolBuffer>(bufferContext.copy(gridHdls[0].buffer().mId));

        // use the gridAllocator to dump the buffers.
        for (auto& it : bufferContext.getGridKeys()) {
            // create a temporary GridHandle so we can parse the GridMetaData.
            auto hdl = nanovdb::GridHandle<MapPoolBuffer>(bufferContext.copy(it));
            printf("key = %d, buffer = %p, gridName = %s\n", (int)hdl.buffer().mId, hdl.data(), hdl.gridMetaData()->shortGridName());
        }

        // Get a (raw) pointer to the NanoVDB grid form the GridManager.
        auto* dstGrid = gridHdls[0].grid<float>();
        if (!dstGrid)
            throw std::runtime_error("GridHandle does not contain a grid with value type float");

        // Access and print out a single value (inside the level set) from both grids
        printf("NanoVDB cpu: %4.2f\n", dstGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));

        // Write the NanoVDB grids to file and throw if writing fails
        nanovdb::io::writeGrids<MapPoolBuffer, std::vector>("data/spheres.nvdb", gridHdls, nanovdb::io::Codec::NONE, 2);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

    // use the gridAllocator to dump the buffers.
    for (auto& it : bufferContext.getGridKeys()) {
        // create a temporary GridHandle so we can parse the GridMetaData.
        auto hdl = nanovdb::GridHandle<MapPoolBuffer>(bufferContext.copy(it));
        printf("key = %d, buffer = %p, gridName = %s\n", (int)hdl.buffer().mId, hdl.data(), hdl.gridMetaData()->shortGridName());
    }
    return 0;
}
