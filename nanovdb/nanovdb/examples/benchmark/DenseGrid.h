// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file DenseGrid.h
///
/// @author Ken Museth
///
/// @brief Simple dense grid class.

#ifndef NANOVDB_DENSEGRID_H_HAS_BEEN_INCLUDED
#define NANOVDB_DENSEGRID_H_HAS_BEEN_INCLUDED

#include <stdint.h>// for uint64_t
#include <fstream> // for std::ifstream
#include <nanovdb/util/HostBuffer.h>// for default Buffer
#include <nanovdb/util/ForEach.h>
#include <nanovdb/NanoVDB.h>// for Map, GridClass, GridType and and Coord


// use 4x4x4 tiles for better cache coherence
// else it uses dense indexing which is slow!
// 0 means disable, 1 is 2x2x2, 2 is 4x4x4 and 3 is 8x8x8
#define LOG2_TILE_SIZE 2

namespace nanovdb {

// forward decleration
template<typename BufferT = HostBuffer>
class DenseGridHandle;

#define DENSE_MAGIC_NUMBER 0x42445665736e6544UL // "DenseVDB" in hex - little endian (uint64_t)


struct DenseData
{
    Map         mMap;// defined in NanoVDB.h
    CoordBBox   mIndexBBox;// min/max of bbox
    BBox<Vec3R> mWorldBBox;// 48B. floating-point AABB of active values in WORLD SPACE (2 x 3 doubles)
    Vec3R       mVoxelSize;
    GridClass   mGridClass;// defined in NanoVDB.h
    GridType    mGridType; //  defined in NanoVDB.h
    uint64_t    mY, mX;//strides in the y and x direction
    uint64_t    mSize;

    __hostdev__ Coord dim() const { return mIndexBBox.dim(); }

    // Affine transformations based on double precision
    template<typename Vec3T>
    __hostdev__ Vec3T applyMap(const Vec3T& xyz) const { return mMap.applyMap(xyz); } // Pos: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMap(const Vec3T& xyz) const { return mMap.applyInverseMap(xyz); } // Pos: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobian(const Vec3T& xyz) const { return mMap.applyJacobian(xyz); } // Dir: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobian(const Vec3T& xyz) const { return mMap.applyInverseJacobian(xyz); } // Dir: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJT(const Vec3T& xyz) const { return mMap.applyIJT(xyz); }
    // Affine transformations based on single precision
    template<typename Vec3T>
    __hostdev__ Vec3T applyMapF(const Vec3T& xyz) const { return mMap.applyMapF(xyz); } // Pos: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMapF(const Vec3T& xyz) const { return mMap.applyInverseMapF(xyz); } // Pos: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobianF(const Vec3T& xyz) const { return mMap.applyJacobianF(xyz); } // Dir: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobianF(const Vec3T& xyz) const { return mMap.applyInverseJacobianF(xyz); } // Dir: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJTF(const Vec3T& xyz) const { return mMap.applyIJTF(xyz); }
};
/// @brief Simple dense grid class
/// @note ZYX is the memory-layout in VDB. It leads to nested
/// for-loops of the order x, y, z.
template<typename ValueT>
class DenseGrid : private DenseData
{
#if LOG2_TILE_SIZE > 0
    static constexpr uint32_t TileLog2 = LOG2_TILE_SIZE, TileMask = (1 << TileLog2) - 1, TileDim = 1 << (3*TileLog2);
#endif
    using DenseData = DenseData;

public:
    using ValueType = ValueT;

    template<typename BufferT = HostBuffer>
    inline static DenseGridHandle<BufferT> create(Coord min, // min inclusive index coordinate
                                                  Coord max, // max inclusive index coordinate
                                                  double dx = 1.0, //voxel size
                                                  const Vec3d& p0 = Vec3d(0.0), // origin
                                                  GridClass gridClass = GridClass::Unknown,
                                                  const BufferT& allocator = BufferT());
    
    __hostdev__ DenseGrid(const DenseGrid&) = delete;
    __hostdev__ ~DenseGrid() = delete;
    __hostdev__ DenseGrid& operator=(const DenseGrid&) = delete;

    __hostdev__ uint64_t size() const { return mIndexBBox.volume(); }
    __hostdev__ inline uint64_t coordToOffset(const Coord &ijk) const;
    __hostdev__ inline bool test(const Coord &ijk) const;
    __hostdev__ uint64_t memUsage() const {return mSize;}
    __hostdev__ uint64_t gridSize() const {return this->memUsage();}
    __hostdev__ const Coord& min() const { return mIndexBBox[0]; }
    __hostdev__ const Coord& max() const { return mIndexBBox[1]; }
    __hostdev__ inline bool isValidType() const;

    /// @brief Return a const reference to the Map for this grid
    __hostdev__ const Map& map() const { return DenseData::mMap; }

    // @brief Return a const reference to the size of a voxel in world units
    __hostdev__ const Vec3R& voxelSize() const { return DenseData::mVoxelSize; }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndex(const Vec3T& xyz) const { return this->applyInverseMap(xyz); }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorld(const Vec3T& xyz) const { return this->applyMap(xyz); }

    /// @brief transformation from index space direction to world space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldDir(const Vec3T& dir) const { return this->applyJacobian(dir); }

    /// @brief transformation from world space direction to index space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexDir(const Vec3T& dir) const { return this->applyInverseJacobian(dir); }

    /// @brief transform the gradient from index space to world space.
    /// @details Applies the inverse jacobian transform map.
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldGrad(const Vec3T& grad) const { return this->applyIJT(grad); }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexF(const Vec3T& xyz) const { return this->applyInverseMapF(xyz); }

    /// @brief index to world space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldF(const Vec3T& xyz) const { return this->applyMapF(xyz); }

    /// @brief transformation from index space direction to world space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldDirF(const Vec3T& dir) const { return this->applyJacobianF(dir); }

    /// @brief transformation from world space direction to index space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexDirF(const Vec3T& dir) const { return this->applyInverseJacobianF(dir); }

    /// @brief Transforms the gradient from index space to world space.
    /// @details Applies the inverse jacobian transform map.
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldGradF(const Vec3T& grad) const { return DenseData::applyIJTF(grad); }

    /// @brief Computes a AABB of active values in world space
    __hostdev__ const BBox<Vec3R>& worldBBox() const { return DenseData::mWorldBBox; }

    __hostdev__ bool  isLevelSet() const { return DenseData::mGridClass == GridClass::LevelSet; }
    __hostdev__ bool  isFogVolume() const { return DenseData::mGridClass == GridClass::FogVolume; }

    /// @brief Computes a AABB of active values in index space
    ///
    /// @note This method is returning a floating point bounding box and not a CoordBBox. This makes
    ///       it more useful for clipping rays.
    __hostdev__ const CoordBBox& indexBBox() const { return mIndexBBox; }

    __hostdev__ const GridType& gridType() const { return DenseData::mGridType; }
    __hostdev__ const GridClass& gridClass() const { return DenseData::mGridClass; }

    __hostdev__ DenseData* data() { return reinterpret_cast<DenseData*>(this); }
    __hostdev__ const DenseData* data() const { return reinterpret_cast<const DenseData*>(this); }

    __hostdev__ ValueT* values() { return reinterpret_cast<ValueT*>(this+1);}
    __hostdev__ const ValueT* values() const { return reinterpret_cast<const ValueT*>(this+1); }

    __hostdev__ inline const ValueT& getValue(const Coord &ijk) const;
    __hostdev__ inline void setValue(const Coord &ijk, const ValueT &v);
}; // Grid

template<typename ValueT>
template<typename BufferT>
DenseGridHandle<BufferT> 
DenseGrid<ValueT>::create(Coord min, 
                          Coord max,
                          double dx, //voxel size
                          const Vec3d& p0, // origin
                          GridClass gridClass, 
                          const BufferT& allocator)
{
    if (dx <= 0) {
        throw std::runtime_error("GridBuilder: voxel size is zero or negative");
    }
    max += Coord(1,1,1);// now max is exclusive

#if LOG2_TILE_SIZE > 0
    const uint64_t dim[3] = {(uint64_t(max[0] - min[0]) + TileMask) >> TileLog2,
                             (uint64_t(max[1] - min[1]) + TileMask) >> TileLog2,
                             (uint64_t(max[2] - min[2]) + TileMask) >> TileLog2};
    const uint64_t size = sizeof(DenseGrid) + sizeof(ValueT)*TileDim*dim[0]*dim[1]*dim[2];
#else
    const uint64_t dim[3] = {uint64_t(max[0] - min[0]), 
                             uint64_t(max[1] - min[1]), 
                             uint64_t(max[2] - min[2])};
    const uint64_t size = sizeof(DenseGrid) + sizeof(ValueT)*dim[0]*dim[1]*dim[2];                         
#endif
    
    DenseGridHandle<BufferT> handle(allocator.create(size));
    DenseGrid* grid = reinterpret_cast<DenseGrid*>(handle.data());
    grid->mSize = size;
    const double Tx = p0[0], Ty = p0[1], Tz = p0[2];
    const double mat[4][4] = {
        {dx, 0.0, 0.0, 0.0}, // row 0
        {0.0, dx, 0.0, 0.0}, // row 1
        {0.0, 0.0, dx, 0.0}, // row 2
        {Tx, Ty, Tz, 1.0}, // row 3
    };
    const double invMat[4][4] = {
        {1 / dx, 0.0, 0.0, 0.0}, // row 0
        {0.0, 1 / dx, 0.0, 0.0}, // row 1
        {0.0, 0.0, 1 / dx, 0.0}, // row 2
        {-Tx, -Ty, -Tz, 1.0}, // row 3
    };

    grid->mMap.set(mat, invMat, 1.0);
    for (int i=0; i<3; ++i) {
        grid->mIndexBBox[0][i] = min[i];
        grid->mIndexBBox[1][i] = max[i] - 1;
    }
    grid->mWorldBBox[0] = grid->mWorldBBox[1] = grid->mMap.applyMap(Vec3d(min[0], min[1], min[2]));
    grid->mWorldBBox.expand(grid->mMap.applyMap(Vec3d(min[0], min[1], max[2])));
    grid->mWorldBBox.expand(grid->mMap.applyMap(Vec3d(min[0], max[1], min[2])));
    grid->mWorldBBox.expand(grid->mMap.applyMap(Vec3d(max[0], min[1], min[2])));
    grid->mWorldBBox.expand(grid->mMap.applyMap(Vec3d(max[0], max[1], min[2])));
    grid->mWorldBBox.expand(grid->mMap.applyMap(Vec3d(max[0], min[1], max[2])));
    grid->mWorldBBox.expand(grid->mMap.applyMap(Vec3d(min[0], max[1], max[2])));
    grid->mWorldBBox.expand(grid->mMap.applyMap(Vec3d(max[0], max[1], max[2])));
    grid->mVoxelSize = grid->mMap.applyMap(Vec3d(1)) - grid->mMap.applyMap(Vec3d(0));
    if (gridClass == GridClass::LevelSet && !is_floating_point<ValueT>::value)
        throw std::runtime_error("Level sets are expected to be floating point types");
    if (gridClass == GridClass::FogVolume && !is_floating_point<ValueT>::value)
        throw std::runtime_error("Fog volumes are expected to be floating point types");
    grid->mGridClass = gridClass;
    grid->mGridType = mapToGridType<ValueT>();
    grid->mY = dim[2];
    grid->mX = dim[2] * dim[1];
    return handle;
}

template<typename ValueT>
bool DenseGrid<ValueT>::test(const Coord &ijk) const 
{ 
    return (ijk[0]>=mIndexBBox[0][0]) && (ijk[0]<=mIndexBBox[1][0]) && 
           (ijk[1]>=mIndexBBox[0][1]) && (ijk[1]<=mIndexBBox[1][1]) && 
           (ijk[2]>=mIndexBBox[0][2]) && (ijk[2]<=mIndexBBox[1][2]);
}

template<typename ValueT>
uint64_t DenseGrid<ValueT>::coordToOffset(const Coord &ijk) const 
{ 
    assert(this->test(ijk));
#if LOG2_TILE_SIZE > 0
    const uint32_t x = ijk[0] - mIndexBBox[0][0];
    const uint32_t y = ijk[1] - mIndexBBox[0][1];
    const uint32_t z = ijk[2] - mIndexBBox[0][2];
    return ((mX*(x>>TileLog2) + mY*(y>>TileLog2) + (z>>TileLog2))<<(3*TileLog2)) + 
           ((x&TileMask)<<(2*TileLog2)) + ((y&TileMask)<<TileLog2) + (z&TileMask);
#else
    return uint64_t(ijk[0]-mIndexBBox[0][0])*mX + 
           uint64_t(ijk[1]-mIndexBBox[0][1])*mY + 
           uint64_t(ijk[2]-mIndexBBox[0][2]);
#endif
}

template<typename ValueT>
const ValueT& DenseGrid<ValueT>::getValue(const Coord &ijk) const 
{ 
    return this->values()[this->coordToOffset(ijk)]; 
}

template<typename ValueT>
void DenseGrid<ValueT>::setValue(const Coord &ijk, const ValueT &value)
{ 
    this->values()[this->coordToOffset(ijk)] = value; 
}

template<typename ValueT>
bool DenseGrid<ValueT>::isValidType() const
{ 
    return std::is_same<float, ValueT>::value ? mGridType == GridType::Float : false;
}

/////////////////////////////////////////////

namespace io{

template<typename ValueT>
void writeDense(const DenseGrid<ValueT> &grid, const char* fileName)
{
    std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::runtime_error("Unable to open file for output");
    }
    const uint64_t tmp[2] = {DENSE_MAGIC_NUMBER, grid.memUsage()};
    os.write(reinterpret_cast<const char*>(tmp), 2*sizeof(uint64_t));
    os.write(reinterpret_cast<const char*>(&grid), tmp[1]);
}

template<typename BufferT>
void writeDense(const DenseGridHandle<BufferT> &handle, const char* fileName)
{
    std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::runtime_error("Unable to open file for output");
    }
    const uint64_t tmp[2] = {DENSE_MAGIC_NUMBER, handle.size()};
    os.write(reinterpret_cast<const char*>(tmp), 2*sizeof(uint64_t));
    os.write(reinterpret_cast<const char*>(handle.data()), tmp[1]);
}

template<typename BufferT = HostBuffer>
DenseGridHandle<BufferT> 
readDense(const char* fileName, const BufferT& allocator = BufferT())
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Unable to open file for input");
    }
    uint64_t tmp[2];
    is.read(reinterpret_cast<char*>(tmp), 2*sizeof(uint64_t));
    if (tmp[0] != DENSE_MAGIC_NUMBER) {
        throw std::runtime_error("This is not a dense NanoVDB file!");
    }
    DenseGridHandle<BufferT> handle(allocator.create(tmp[1]));
    is.read(reinterpret_cast<char*>(handle.data()), tmp[1]);
    return handle;
}
}// namespace io
/////////////////////////////////////////////

/// @brief Converts a NanoVDB grid to a DenseGrid
template<typename GridT, typename BufferT = HostBuffer>
DenseGridHandle<BufferT> convertToDense(const GridT &grid, const BufferT& allocator = BufferT())
{
    using ValueT = typename GridT::ValueType;
    using DenseT = DenseGrid<ValueT>;
    const Coord min = grid.indexBBox().min(), max = grid.indexBBox().max() + Coord(1,1,1);// max is exclusive!
#if LOG2_TILE_SIZE > 0
    static constexpr uint32_t TileLog2 = LOG2_TILE_SIZE, TileMask = (1 << TileLog2) - 1, TileDim = 1 << (3*TileLog2);
    const uint64_t dim[3] = {(uint64_t(max[0] - min[0]) + TileMask) >> TileLog2,
                             (uint64_t(max[1] - min[1]) + TileMask) >> TileLog2,
                             (uint64_t(max[2] - min[2]) + TileMask) >> TileLog2};
    const uint64_t size = sizeof(DenseT) + sizeof(ValueT)*TileDim*dim[0]*dim[1]*dim[2];
#else
    const uint64_t dim[3] = {uint64_t(max[0] - min[0]), 
                             uint64_t(max[1] - min[1]), 
                             uint64_t(max[2] - min[2])};
    const uint64_t size = sizeof(DenseT) + sizeof(ValueT)*dim[0]*dim[1]*dim[2];
#endif
    
    DenseGridHandle<BufferT> handle( allocator.create(size) );
    auto *dense = reinterpret_cast<DenseT*>(handle.data());
    auto *data = dense->data();
    
    // copy DenseData
    data->mMap = grid.map();
    data->mIndexBBox = grid.indexBBox();
    data->mWorldBBox = grid.worldBBox();
    data->mVoxelSize = grid.voxelSize();
    data->mGridClass = grid.gridClass();
    data->mGridType  = grid.gridType();
    data->mY = dim[2];
    data->mX = dim[2] * dim[1];
    data->mSize = size;

    // copy values
    auto kernel = [&](const Range<1,int> &r) {
        auto acc = grid.getAccessor();
        Coord ijk;
        for (ijk[0] = r.begin(); ijk[0] < r.end(); ++ijk[0]) {
            for (ijk[1] = min[1]; ijk[1] < max[1]; ++ijk[1]) {
                for (ijk[2] = min[2]; ijk[2] < max[2]; ++ijk[2]) {
                    dense->setValue(ijk, acc.getValue(ijk));
                }
            }
        }
    };
    Range<1,int> range(min[0], max[0]);
#if 1
    forEach(range, kernel); 
#else
    kernel(range);
#endif

    return handle;
}
/////////////////////////////////////////////

template<typename BufferT>
class DenseGridHandle
{
    BufferT mBuffer;

public:
    DenseGridHandle(BufferT&& resources) { mBuffer = std::move(resources); }

    DenseGridHandle() = default;
    /// @brief Disallow copy-construction
    DenseGridHandle(const DenseGridHandle&) = delete;
    /// @brief Disallow copy assignment operation
    DenseGridHandle& operator=(const DenseGridHandle&) = delete;
    /// @brief Move copy assignment operation
    DenseGridHandle& operator=(DenseGridHandle&& other) noexcept
    {
        mBuffer = std::move(other.mBuffer);
        return *this;
    }
    /// @brief Move copy-constructor
    DenseGridHandle(DenseGridHandle&& other) noexcept { mBuffer = std::move(other.mBuffer); }
    /// @brief Default destructor
    ~DenseGridHandle() { this->reset(); }

    void reset() { mBuffer.clear(); }

    BufferT&       buffer() { return mBuffer; }
    const BufferT& buffer() const { return mBuffer; }

    /// @brief Returns a non-const pointer to the data.
    ///
    /// @warning Note that the return pointer can be NULL if the DenseGridHandle was not initialized
    uint8_t* data() {return mBuffer.data();}

    /// @brief Returns a const pointer to the data.
    ///
    /// @warning Note that the return pointer can be NULL if the DenseGridHandle was not initialized
    const uint8_t* data() const {return mBuffer.data();}

    /// @brief Returns the size in bytes of the raw memory buffer managed by this DenseGridHandle's allocator.
    uint64_t size() const { return mBuffer.size();}

    /// @brief Returns a const pointer to the NanoVDB grid encoded in the DenseGridHandle.
    ///
    /// @warning Note that the return pointer can be NULL if the DenseGridHandle was not initialized or the template
    ///          parameter does not match!
    template<typename ValueT>
    const DenseGrid<ValueT>* grid() const
    {
        using GridT = const DenseGrid<ValueT>;
        GridT* grid = reinterpret_cast<GridT*>(mBuffer.data());
        return (grid && grid->isValidType()) ? grid : nullptr;
    }

    template<typename ValueT>
    DenseGrid<ValueT>* grid()
    {
        using GridT = DenseGrid<ValueT>;
        GridT* grid = reinterpret_cast<GridT*>(mBuffer.data());
        return (grid && grid->isValidType()) ? grid : nullptr;
    }

    template<typename ValueT, typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, const DenseGrid<ValueT>*>::type
    deviceGrid() const
    {
        using GridT = const DenseGrid<ValueT>;
        bool isValidType = reinterpret_cast<GridT*>(mBuffer.data())->isValidType();
        GridT* grid = reinterpret_cast<GridT*>(mBuffer.deviceData());
        return (grid && isValidType) ? grid : nullptr;
    }

    template<typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceUpload(void* stream = nullptr, bool sync = true) {
        mBuffer.deviceUpload(stream, sync);
    }

    template<typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceDownload(void* stream = nullptr, bool sync = true) {
        mBuffer.deviceDownload(stream, sync);
    }
}; // DenseGridHandle

} // namespace nanovdb

#endif // NANOVDB_DENSEGRID_HAS_BEEN_INCLUDED
