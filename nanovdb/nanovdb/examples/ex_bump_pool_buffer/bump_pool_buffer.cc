// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/HostBuffer.h>
#include <algorithm>

//////////////////////////////////////////////
class BumpPoolAllocatorContext
{
public:
    uint64_t            mNextFreeOffset;
    nanovdb::HostBuffer mPoolSlab;

    BumpPoolAllocatorContext(uint64_t size)
        : mNextFreeOffset(0)
    {
        mPoolSlab.init(size);
        std::cout << "Created BumpPoolAllocatorContext" << std::endl;
    }

    ~BumpPoolAllocatorContext()
    {
        mPoolSlab.clear();
        std::cout << "Destroyed BumpPoolAllocatorContext" << std::endl;
    }
};

// This is an example of a custom Buffer with has a persistent state allocator ("mState") within it.
// It must have methods for "clear", "data" (both const and non-const variants), and "size".
// It must also include a static "create" method.
class PoolBuffer
{
    uint64_t                              mOffset = -1;
    uint64_t                              mSize = 0;
    std::shared_ptr<BumpPoolAllocatorContext> mState;

public:
    PoolBuffer() = default;
    PoolBuffer(const PoolBuffer&) = delete;
    PoolBuffer& operator=(const PoolBuffer&) = delete;
    PoolBuffer(PoolBuffer&&) = default;
    PoolBuffer& operator=(PoolBuffer&&) = default;

    // Initialize the stateful allocator.
    void initialize(uint64_t size)
    {
        if (!mState) {
            mState = std::make_shared<BumpPoolAllocatorContext>(size);
        }
    }

    // Mandatory.
    // Create a buffer instance (optionally using the context as a hint)
    static PoolBuffer create(uint64_t size, const PoolBuffer* context)
    {
        if (!context || !context->mState) {
            throw std::runtime_error("context has not been created.");
        }

        PoolBuffer buffer;
        buffer.mState = context->mState;
        buffer.mOffset = context->mState->mNextFreeOffset;
        buffer.mSize = size;
        context->mState->mNextFreeOffset += size;

        printf("Creating resources offset(%d) size(%d)\n", (int)buffer.mOffset, (int)buffer.mSize);

        return buffer;
    }

    // Mandatory.
    // Clear the buffer data.
    void clear()
    {
        // nothing to do, as this is a simple "bump" allocator.
    }

    // Mandatory.
    // Return the size of the buffer data.
    uint64_t       size() const { return mSize; }

    // Mandatory.
    // Return non-const pointer to the buffer data.
    void*       data() { return nanovdb::util::PtrAdd(mState->mPoolSlab.data(), mOffset); }

    // Mandatory.
    // Return const pointer to the buffer data.
    const void* data() const { return nanovdb::util::PtrAdd(mState->mPoolSlab.data(), mOffset); }
};

// we specify this trait to avoid declaring the "device...(...)" convenience methods.
// By default, "hasDeviceDual" is false for a Buffer, but we are just being overly explicit here.
namespace nanovdb {
template<>
struct BufferTraits<PoolBuffer>
{
    static const bool hasDeviceDual = false;
};
} // namespace nanovdb

int main()
{
    PoolBuffer bufferContext;
    // We must initialize the stateful allocator.
    bufferContext.initialize(1 << 26);

    try {
        std::vector<nanovdb::GridHandle<PoolBuffer>> gridHdls;

        // create two grids...
        gridHdls.push_back(nanovdb::tools::createLevelSetSphere<float >(100.0, nanovdb::Vec3d(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "spheref", nanovdb::tools::StatsMode::BBox, nanovdb::CheckMode::Partial, bufferContext));
        gridHdls.push_back(nanovdb::tools::createLevelSetSphere<double>(100.0, nanovdb::Vec3d( 20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "sphered", nanovdb::tools::StatsMode::BBox, nanovdb::CheckMode::Partial, bufferContext));

        // Get a (raw) pointer to the NanoVDB grid form the GridManager.
        auto* dstGrid = gridHdls[0].grid<float>();
        if (!dstGrid)
            throw std::runtime_error("GridHandle does not contain a grid with value type float");

        // Access and print out a single value (inside the level set) from both grids
        printf("NanoVDB cpu: %4.2f\n", dstGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));

        // Write the NanoVDB grids to file and throw if writing fails
        nanovdb::io::writeGrids<PoolBuffer, std::vector>("data/spheres.nvdb", gridHdls, nanovdb::io::Codec::NONE, 2);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
