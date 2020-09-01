
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////

#if defined(CNANOVDB_COMPILER_GLSL)
#line 8
#endif

#define CNANOVDB_USE_READACCESSOR_CACHE
#define AVOID_64BIT_SHIFT
#define USE_SINGLE_ROOT_KEY
#define CNANOVDB_DATA_ALIGNMENT 32
#define CNANOVDB_ALIGNMENT_PADDING(x, n) (-(x) & ((n)-1))

#define Vec3T vec3

#define nanovdb_Vec3f vec3

////////////////////////////////////////////////////////

//// struct:

CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_Coord)
    int32_t mVec[3];
CNANOVDB_DECLARE_STRUCT_END(nanovdb_Coord)
#define CNANOVDB_SIZEOF_nanovdb_Coord (4 * 3)

////////////////////////////////////////////////////////

//// struct:

CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_Map)
    float  mMatF[9]; // r,c = 3*r + c
    float  mInvMatF[9]; // r,c = 3*r + c
    float  mVecF[3];
    float  mTaperF;
    double mMatD[9]; // r,c = 3*r + c
    double mInvMatD[9]; // r,c = 3*r + c
    double mVecD[3];
    double mTaperD;
CNANOVDB_DECLARE_STRUCT_END(nanovdb_Map)
#define CNANOVDB_SIZEOF_nanovdb_Map (4 * 9 + 4 * 9 + 4 * 3 + 4 + 8 * 9 + 8 * 9 + 8 * 3 + 8)

////////////////////////////////////////////////////////

//// struct:

CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_Ray)
    vec3  mEye;
    vec3  mDir;
    float mT0;
    float mT1;
CNANOVDB_DECLARE_STRUCT_END(nanovdb_Ray)
#define CNANOVDB_SIZEOF_nanovdb_Ray (4 * 3 + 4 * 3 + 4 + 4)

    ////////////////////////////////////////////////////////

#define DEFINE_MASK_(LOG2DIM, SIZE) \
    CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_BitMask##LOG2DIM) \
        CNANOVDB_WORD_TYPE mWords[SIZE >> CNANOVDB_WORD_LOG2SIZE]; \
    CNANOVDB_DECLARE_STRUCT_END(nanovdb_BitMask##LOG2DIM)

#define DEFINE_MASK(LOG2DIM) DEFINE_MASK_(LOG2DIM, (1 << (3 * LOG2DIM)))

DEFINE_MASK(3);
DEFINE_MASK(4);
DEFINE_MASK(5);

////////////////////////////////////////////////////////

CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_ReadAccessor)
    nanovdb_Coord mKey;
    int32_t       mNodeIndex[3];
CNANOVDB_DECLARE_STRUCT_END(nanovdb_ReadAccessor)

CNANOVDB_INLINE void nanovdb_ReadAccessor_insert(CNANOVDB_REF(nanovdb_ReadAccessor) acc, int childlevel, int32_t nodeIndex, nanovdb_Coord ijk)
{
    CNANOVDB_DEREF(acc).mNodeIndex[childlevel] = nodeIndex;
    CNANOVDB_DEREF(acc).mKey.mVec[0] = ijk.mVec[0];
    CNANOVDB_DEREF(acc).mKey.mVec[1] = ijk.mVec[1];
    CNANOVDB_DEREF(acc).mKey.mVec[2] = ijk.mVec[2];
}

////////////////////////////////////////////////////////

//// struct:

#define DEFINE_LEAF_NODE_struct_(LEVEL, LOG2DIM, CHILDTOTAL, TOTAL, MASK, VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_Node##LEVEL##_##SUFFIX) \
        nanovdb_BitMask##LOG2DIM mValueMask; \
        VALUET                   mVoxels[1 << (3 * LOG2DIM)]; \
        VALUET                   mValueMin; \
        VALUET                   mValueMax; \
        nanovdb_Coord            mBBox_min; \
        uint32_t                 mBBoxDifAndFlags; \
        uint32_t                 _reserved[CNANOVDB_ALIGNMENT_PADDING((8 << (3 * LOG2DIM - 6)) + (SIZEOF_VALUET << (3 * LOG2DIM)) + (SIZEOF_VALUET * 2) + (12) + (4), CNANOVDB_DATA_ALIGNMENT) / 4]; \
    CNANOVDB_DECLARE_STRUCT_END(nanovdb_Node##LEVEL##_##SUFFIX) \
\
    CNANOVDB_CONSTANT_MEM int32_t nanovdb_Node##LEVEL##_TOTAL = TOTAL; \
    CNANOVDB_CONSTANT_MEM int32_t nanovdb_Node##LEVEL##_MASK = MASK;

#define DEFINE_LEAF_NODE_struct(LEVEL, LOG2DIM, TOTAL, VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_LEAF_NODE_struct_(LEVEL, LOG2DIM, (TOTAL - LOG2DIM), TOTAL, ((1 << TOTAL) - 1), VALUET, SIZEOF_VALUET, SUFFIX)

//// functions:

#define DEFINE_LEAF_NODE_functions_(LEVEL, LOG2DIM, CHILDTOTAL, TOTAL, MASK, VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_INLINE int32_t \
        nanovdb_Node##LEVEL##_##SUFFIX##_CoordToOffset(nanovdb_Coord ijk) \
    { \
        return (((ijk.mVec[0] & MASK) >> CHILDTOTAL) << (2 * LOG2DIM)) + \
               (((ijk.mVec[1] & MASK) >> CHILDTOTAL) << (LOG2DIM)) + \
               ((ijk.mVec[2] & MASK) >> CHILDTOTAL); \
    } \
\
    CNANOVDB_INLINE VALUET \
        nanovdb_Node##LEVEL##_##SUFFIX##_getValue(CNANOVDB_CONTEXT cxt, int32_t nodeIndex, nanovdb_Coord ijk) \
    { \
        int32_t n = nanovdb_Node##LEVEL##_##SUFFIX##_CoordToOffset(ijk); \
        return CNANOVDB_NODEDATA(cxt, LEVEL)[nodeIndex].mVoxels[n]; \
    } \
    CNANOVDB_INLINE VALUET \
        nanovdb_Node##LEVEL##_##SUFFIX##_getValueAndCache(CNANOVDB_CONTEXT cxt, int32_t nodeIndex, nanovdb_Coord ijk, CNANOVDB_REF(nanovdb_ReadAccessor) acc) \
    { \
        int32_t n = nanovdb_Node##LEVEL##_##SUFFIX##_CoordToOffset(ijk); \
        return CNANOVDB_NODEDATA(cxt, LEVEL)[nodeIndex].mVoxels[n]; \
    } \
\
    CNANOVDB_INLINE int32_t \
        nanovdb_Node##LEVEL##_##SUFFIX##_getDimAndCache(CNANOVDB_CONTEXT cxt, int32_t nodeIndex, nanovdb_Coord ijk, nanovdb_Ray ray, CNANOVDB_REF(nanovdb_ReadAccessor) acc) \
    { \
        return ((CNANOVDB_NODEDATA(cxt, LEVEL)[nodeIndex].mBBoxDifAndFlags & 7) != 0u) ? 1 << TOTAL : 1; \
    }

#define DEFINE_LEAF_NODE_functions(LEVEL, LOG2DIM, TOTAL, VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_LEAF_NODE_functions_(LEVEL, LOG2DIM, (TOTAL - LOG2DIM), TOTAL, ((1 << TOTAL) - 1), VALUET, SIZEOF_VALUET, SUFFIX)

////////////////////////////////////////////////////////

#if defined(CNANOVDB_COMPILER_GLSL)

//// struct:

#define DEFINE_NODE_TILE_ENTRY_struct_(VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_TileEntry_##SUFFIX) \
        int32_t valueOrChildID; \
    CNANOVDB_DECLARE_STRUCT_END(nanovdb_TileEntry_##SUFFIX)
#define SIZEOF_NODE_TILE_ENTRY(SUFFIX) 4

//// functions:

#define DEFINE_NODE_TILE_ENTRY_functions_(VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_INLINE VALUET \
        nanovdb_TileEntry_##SUFFIX##_getValue(nanovdb_TileEntry_##SUFFIX entry) \
    { \
        return intBitsToFloat(entry.valueOrChildID); \
    } \
    CNANOVDB_INLINE int32_t \
        nanovdb_TileEntry_##SUFFIX##_getChild(nanovdb_TileEntry_##SUFFIX entry) \
    { \
        return entry.valueOrChildID; \
    }

#else

//// struct:

#define DEFINE_NODE_TILE_ENTRY_struct_(VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_DECLARE_UNION_BEGIN(nanovdb_TileEntry_##SUFFIX) \
    VALUET  value; \
    int32_t childID; \
    CNANOVDB_DECLARE_UNION_END(nanovdb_TileEntry_##SUFFIX)
#define SIZEOF_NODE_TILE_ENTRY(SUFFIX) sizeof(nanovdb_TileEntry_##SUFFIX)

//// functions:

#define DEFINE_NODE_TILE_ENTRY_functions_(VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_INLINE VALUET \
        nanovdb_TileEntry_##SUFFIX##_getValue(nanovdb_TileEntry_##SUFFIX entry) \
    { \
        return entry.value; \
    } \
    CNANOVDB_INLINE int32_t \
        nanovdb_TileEntry_##SUFFIX##_getChild(nanovdb_TileEntry_##SUFFIX entry) \
    { \
        return entry.childID; \
    }

#endif

#define DEFINE_NODE_TILE_ENTRY_struct(VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_NODE_TILE_ENTRY_struct_(VALUET, SIZEOF_VALUET, SUFFIX)

#define DEFINE_NODE_TILE_ENTRY_functions(VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_NODE_TILE_ENTRY_functions_(VALUET, SIZEOF_VALUET, SUFFIX)

////////////////////////////////////////////////////////

//// struct:

#define DEFINE_INTERNAL_NODE_struct_(CHILDLEVEL, LEVEL, LOG2DIM, CHILDTOTAL, TOTAL, MASK, VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_Node##LEVEL##_##SUFFIX) \
        nanovdb_BitMask##LOG2DIM   mValueMask, mChildMask; \
        nanovdb_TileEntry_##SUFFIX mTable[1 << (3 * LOG2DIM)]; \
        VALUET                     mValueMin, mValueMax; \
        nanovdb_Coord              mBBox_min, mBBox_max; \
        int32_t                    mOffset; \
        uint32_t                   mFlags; \
        uint32_t                   _reserved[CNANOVDB_ALIGNMENT_PADDING((8 << (3 * LOG2DIM - 6)) * 2 + (SIZEOF_NODE_TILE_ENTRY(SUFFIX) << (3 * LOG2DIM)) + (SIZEOF_VALUET * 2) + (12 * 2) + (4) + (4), CNANOVDB_DATA_ALIGNMENT) / 4]; \
    CNANOVDB_DECLARE_STRUCT_END(nanovdb_Node##LEVEL##_##SUFFIX) \
\
    CNANOVDB_CONSTANT_MEM int32_t nanovdb_Node##LEVEL##_TOTAL = TOTAL; \
    CNANOVDB_CONSTANT_MEM int32_t nanovdb_Node##LEVEL##_MASK = MASK;

#define DEFINE_INTERNAL_NODE_struct(CHILDLEVEL, LEVEL, LOG2DIM, TOTAL, VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_INTERNAL_NODE_struct_(CHILDLEVEL, LEVEL, LOG2DIM, (TOTAL - LOG2DIM), TOTAL, ((1 << TOTAL) - 1), VALUET, SIZEOF_VALUET, SUFFIX)

//// functions:

#define DEFINE_INTERNAL_NODE_functions_(CHILDLEVEL, LEVEL, LOG2DIM, CHILDTOTAL, TOTAL, MASK, VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_INLINE int32_t \
        nanovdb_Node##LEVEL##_##SUFFIX##_CoordToOffset(nanovdb_Coord ijk) \
    { \
        return (((ijk.mVec[0] & MASK) >> nanovdb_Node##CHILDLEVEL##_TOTAL) << (2 * LOG2DIM)) + \
               (((ijk.mVec[1] & MASK) >> nanovdb_Node##CHILDLEVEL##_TOTAL) << (LOG2DIM)) + \
               ((ijk.mVec[2] & MASK) >> nanovdb_Node##CHILDLEVEL##_TOTAL); \
    } \
\
    CNANOVDB_INLINE int32_t \
        nanovdb_Node##LEVEL##_##SUFFIX##_getChild(CNANOVDB_CONTEXT cxt, int32_t nodeIndex, int32_t n) \
    { \
        return nanovdb_TileEntry_##SUFFIX##_getChild(CNANOVDB_NODEDATA(cxt, LEVEL)[nodeIndex].mTable[n]); \
    } \
\
    CNANOVDB_INLINE boolean \
        nanovdb_Node##LEVEL##_##SUFFIX##_hasChild(CNANOVDB_CONTEXT cxt, int32_t nodeIndex, int32_t n) \
    { \
        return 0 != (CNANOVDB_NODEDATA(cxt, LEVEL)[nodeIndex].mChildMask.mWords[n >> CNANOVDB_WORD_LOG2SIZE] & (CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(1) << (n & CNANOVDB_WORD_MASK))); \
    } \
\
    CNANOVDB_INLINE VALUET \
        nanovdb_Node##LEVEL##_##SUFFIX##_getValue(CNANOVDB_CONTEXT cxt, int32_t nodeIndex, nanovdb_Coord ijk) \
    { \
        int32_t n = nanovdb_Node##LEVEL##_##SUFFIX##_CoordToOffset(ijk); \
        if (nanovdb_Node##LEVEL##_##SUFFIX##_hasChild(cxt, nodeIndex, n)) { \
            int32_t childIndex = nanovdb_Node##LEVEL##_##SUFFIX##_getChild(cxt, nodeIndex, n); \
            return nanovdb_Node##CHILDLEVEL##_##SUFFIX##_getValue(cxt, childIndex, ijk); \
        } \
        return nanovdb_TileEntry_##SUFFIX##_getValue(CNANOVDB_NODEDATA(cxt, LEVEL)[nodeIndex].mTable[n]); \
    } \
\
    CNANOVDB_INLINE VALUET \
        nanovdb_Node##LEVEL##_##SUFFIX##_getValueAndCache(CNANOVDB_CONTEXT cxt, int32_t nodeIndex, nanovdb_Coord ijk, CNANOVDB_REF(nanovdb_ReadAccessor) acc) \
    { \
        int32_t n = nanovdb_Node##LEVEL##_##SUFFIX##_CoordToOffset(ijk); \
        if (nanovdb_Node##LEVEL##_##SUFFIX##_hasChild(cxt, nodeIndex, n)) { \
            int32_t childIndex = nanovdb_Node##LEVEL##_##SUFFIX##_getChild(cxt, nodeIndex, n); \
            nanovdb_ReadAccessor_insert(acc, CHILDLEVEL, childIndex, ijk); \
            return nanovdb_Node##CHILDLEVEL##_##SUFFIX##_getValueAndCache(cxt, childIndex, ijk, acc); \
        } \
        return nanovdb_TileEntry_##SUFFIX##_getValue(CNANOVDB_NODEDATA(cxt, LEVEL)[nodeIndex].mTable[n]); \
    } \
\
    CNANOVDB_INLINE int32_t \
        nanovdb_Node##LEVEL##_##SUFFIX##_getDimAndCache(CNANOVDB_CONTEXT cxt, int32_t nodeIndex, nanovdb_Coord ijk, nanovdb_Ray ray, CNANOVDB_REF(nanovdb_ReadAccessor) acc) \
    { \
        int32_t n = nanovdb_Node##LEVEL##_##SUFFIX##_CoordToOffset(ijk); \
        if (nanovdb_Node##LEVEL##_##SUFFIX##_hasChild(cxt, nodeIndex, n)) { \
            int32_t childIndex = nanovdb_Node##LEVEL##_##SUFFIX##_getChild(cxt, nodeIndex, n); \
            nanovdb_ReadAccessor_insert(acc, CHILDLEVEL, childIndex, ijk); \
            return nanovdb_Node##CHILDLEVEL##_##SUFFIX##_getDimAndCache(cxt, childIndex, ijk, ray, acc); \
        } \
        return 1 << nanovdb_Node##CHILDLEVEL##_TOTAL; \
    }

#define DEFINE_INTERNAL_NODE_functions(CHILDLEVEL, LEVEL, LOG2DIM, TOTAL, VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_INTERNAL_NODE_functions_(CHILDLEVEL, LEVEL, LOG2DIM, (TOTAL - LOG2DIM), TOTAL, ((1 << TOTAL) - 1), VALUET, SIZEOF_VALUET, SUFFIX)

////////////////////////////////////////////////////////

#ifdef USE_SINGLE_ROOT_KEY
#define SIZEOF_ROOT_KEY 8
#define DEFINE_ROOT_KEY(k) uint64_t k
#define ROOT_COORD_TO_KEY(ijk) nanovdb_coord_to_key(ijk)
#if defined(CNANOVDB_COMPILER_GLSL)
#define ROOT_FIND_TILE(k) \
    for (int i = low; i < high; i++) { \
        if (CNANOVDB_ROOTDATATILES(cxt)[i].key.x == k.x && CNANOVDB_ROOTDATATILES(cxt)[i].key.y == k.y) \
            return CNANOVDB_MAKE(int32_t)(i); \
    }
#else
#define ROOT_FIND_TILE(k) \
    for (int i = low; i < high; i++) { \
        if (CNANOVDB_ROOTDATATILES(cxt)[i].key == k) \
            return CNANOVDB_MAKE(int32_t)(i); \
    }
#endif
#else
#define SIZEOF_ROOT_KEY 12
#define DEFINE_ROOT_KEY(k) nanovdb_Coord k
#define ROOT_COORD_TO_KEY(ijk) ijk
#define ROOT_FIND_TILE(k) // todo
#endif

//// struct:

#define DEFINE_ROOT_TILE_struct_(VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_RootData_Tile_##SUFFIX) \
        DEFINE_ROOT_KEY(key); \
        VALUET   value; \
        int32_t  childID; \
        uint32_t state; \
        uint32_t _reserved[CNANOVDB_ALIGNMENT_PADDING(SIZEOF_ROOT_KEY + SIZEOF_VALUET + 4 + 4, CNANOVDB_DATA_ALIGNMENT) / 4]; \
    CNANOVDB_DECLARE_STRUCT_END(nanovdb_RootData_Tile_##SUFFIX)

#define DEFINE_ROOT_TILE_struct(VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_ROOT_TILE_struct_(VALUET, SIZEOF_VALUET, SUFFIX)

#define DEFINE_ROOT_NODE_struct_(VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_RootData_##SUFFIX) \
        nanovdb_Coord mBBox_min, mBBox_max; \
        uint64_t      mActiveVoxelCount; \
        int32_t       mTileCount, padding[3]; \
        VALUET        mBackground, mValueMin, mValueMax; \
        uint32_t      _reserved[CNANOVDB_ALIGNMENT_PADDING((12 * 2) + (8) + (4 * 4) + (SIZEOF_VALUET * 3), CNANOVDB_DATA_ALIGNMENT) / 4]; \
    CNANOVDB_DECLARE_STRUCT_END(nanovdb_RootData_##SUFFIX)

//// functions:

#define DEFINE_ROOT_NODE_functions_(VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_INLINE int32_t \
        nanovdb_RootData_##SUFFIX##_findTile(CNANOVDB_CONTEXT cxt, nanovdb_Coord ijk) \
    { \
        int low = 0, high = CNANOVDB_ROOTDATA(cxt).mTileCount; \
        DEFINE_ROOT_KEY(key) = ROOT_COORD_TO_KEY(ijk); \
        ROOT_FIND_TILE(key); \
        return -1; \
    } \
\
    CNANOVDB_INLINE VALUET \
        nanovdb_RootData_##SUFFIX##_getValue(CNANOVDB_CONTEXT cxt, nanovdb_Coord ijk) \
    { \
        int32_t rootTileIndex = nanovdb_RootData_##SUFFIX##_findTile(cxt, ijk); \
        if (rootTileIndex < 0) \
            return CNANOVDB_ROOTDATA(cxt).mBackground; \
        int32_t childIndex = CNANOVDB_ROOTDATATILES(cxt)[rootTileIndex].childID; \
        if (childIndex < 0) \
            return CNANOVDB_ROOTDATATILES(cxt)[rootTileIndex].value; \
        return nanovdb_Node2_##SUFFIX##_getValue(cxt, childIndex, ijk); \
    } \
\
    CNANOVDB_INLINE VALUET \
        nanovdb_RootData_##SUFFIX##_getValueAndCache(CNANOVDB_CONTEXT cxt, nanovdb_Coord ijk, CNANOVDB_REF(nanovdb_ReadAccessor) acc) \
    { \
        int32_t rootTileIndex = nanovdb_RootData_##SUFFIX##_findTile(cxt, ijk); \
        if (rootTileIndex < 0) \
            return CNANOVDB_ROOTDATA(cxt).mBackground; \
        int32_t childIndex = CNANOVDB_ROOTDATATILES(cxt)[rootTileIndex].childID; \
        if (childIndex < 0) \
            return CNANOVDB_ROOTDATATILES(cxt)[rootTileIndex].value; \
        nanovdb_ReadAccessor_insert(acc, 2, childIndex, ijk); \
        return nanovdb_Node2_##SUFFIX##_getValueAndCache(cxt, childIndex, ijk, acc); \
    } \
\
    CNANOVDB_INLINE int32_t \
        nanovdb_RootData_##SUFFIX##_getDimAndCache(CNANOVDB_CONTEXT cxt, nanovdb_Coord ijk, nanovdb_Ray ray, CNANOVDB_REF(nanovdb_ReadAccessor) acc) \
    { \
        int32_t rootTileIndex = nanovdb_RootData_##SUFFIX##_findTile(cxt, ijk); \
        if (rootTileIndex < 0) \
            return 1 << nanovdb_Node2_TOTAL; \
        int32_t childIndex = CNANOVDB_ROOTDATATILES(cxt)[rootTileIndex].childID; \
        if (childIndex < 0) \
            return 1 << nanovdb_Node2_TOTAL; \
        nanovdb_ReadAccessor_insert(acc, 2, childIndex, ijk); \
        return nanovdb_Node2_##SUFFIX##_getDimAndCache(cxt, childIndex, ijk, ray, acc); \
    }

#define DEFINE_ROOT_NODE_struct(VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_ROOT_NODE_struct_(VALUET, SIZEOF_VALUET, SUFFIX)

#define DEFINE_ROOT_NODE_functions(VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_ROOT_NODE_functions_(VALUET, SIZEOF_VALUET, SUFFIX)

////////////////////////////////////////////////////////

#ifdef USE_SINGLE_ROOT_KEY
CNANOVDB_INLINE uint64_t
nanovdb_coord_to_key(nanovdb_Coord ijk)
{
#if defined(AVOID_64BIT_SHIFT) && defined(CNANOVDB_COMPILER_OPENCL)
    // Define to workaround a bug with 64-bit shifts in the AMD OpenCL compiler.
    uvec2 key = CNANOVDB_MAKE(uvec2)((CNANOVDB_MAKE(uint32_t)(ijk.mVec[2])) >> 12, 0) |
                CNANOVDB_MAKE(uvec2)(((CNANOVDB_MAKE(uint32_t)(ijk.mVec[1])) >> 12) << 21,
                                     (CNANOVDB_MAKE(uint32_t)(ijk.mVec[1])) >> 23) |
                CNANOVDB_MAKE(uvec2)(0, ((CNANOVDB_MAKE(uint32_t)(ijk.mVec[0])) >> 12) << 10);
    return *(uint64_t*)&key;
#elif defined(CNANOVDB_COMPILER_GLSL)
    uvec2 key = CNANOVDB_MAKE(uvec2)((CNANOVDB_MAKE(uint32_t)(ijk.mVec[2])) >> 12, 0) |
                CNANOVDB_MAKE(uvec2)(((CNANOVDB_MAKE(uint32_t)(ijk.mVec[1])) >> 12) << 21,
                                     (CNANOVDB_MAKE(uint32_t)(ijk.mVec[1])) >> 23) |
                CNANOVDB_MAKE(uvec2)(0, ((CNANOVDB_MAKE(uint32_t)(ijk.mVec[0])) >> 12) << 10);
    return key;
#else
    return (CNANOVDB_MAKE(uint64_t)((CNANOVDB_MAKE(uint32_t)(ijk.mVec[2])) >> 12)) |
           ((CNANOVDB_MAKE(uint64_t)((CNANOVDB_MAKE(uint32_t)(ijk.mVec[1])) >> 12)) << 21) |
           ((CNANOVDB_MAKE(uint64_t)((CNANOVDB_MAKE(uint32_t)(ijk.mVec[0])) >> 12)) << 42);
#endif
}
#else
CNANOVDB_INLINE void nanovdb_coord_to_key(nanovdb_Coord key, nanovdb_Coord ijk)
{
    key.mVec[0] = ijk.mVec[0] & ~((1u << 12) - 1u);
    key.mVec[1] = ijk.mVec[1] & ~((1u << 12) - 1u);
    key.mVec[2] = ijk.mVec[2] & ~((1u << 12) - 1u);
}
#endif

/*
enum nanovdb_GridType {
    nanovdb_GridType_Unknown = 0,
    nanovdb_GridType_Float = 1,
    nanovdb_GridType_Double = 2,
    nanovdb_GridType_Int16 = 3,
    nanovdb_GridType_Int32 = 4,
    nanovdb_GridType_Int64 = 5,
    nanovdb_GridType_Vec3f = 6,
    nanovdb_GridType_Vec3d = 7,
    nanovdb_GridType_Mask = 8,
    nanovdb_GridType_FP16 = 9,
    nanovdb_GridType_End = 10
};
*/

CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_GridData)
    uint64_t    mMagic; // 8 byte magic to validate it is valid grid data.
    uint32_t    mGridName[256 / 4];
    double      mBBoxMin[3]; // floating-point min-corner of active values in WORLD SPACE
    double      mBBoxMax[3]; // floating-point max-corner of active values in WORLD SPACE
    nanovdb_Map mMap; // affine transformation between index and world space in both single and double precision
    double      mUniformScale; // size of a voxel in world units
    uint32_t    mGridClass;
    uint32_t    mGridType; // packed class and type (16bits each)
    uint32_t    mBlindDataCount;
    uint32_t    _reserved[CNANOVDB_ALIGNMENT_PADDING(8 + 256 + 24 + 24 + CNANOVDB_SIZEOF_nanovdb_Map + 8 + 4 + 4 + 4, CNANOVDB_DATA_ALIGNMENT) / 4];
CNANOVDB_DECLARE_STRUCT_END(nanovdb_GridData)

CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_GridBlindMetaData)
    int64_t  mByteOffset; // byte offset to the blind data, relative to the GridData.
    int64_t  mByteSize; // byte size of blind data.
    uint32_t mFlags; // flags
    uint32_t mSemantic; // semantic meaning of the data.
    uint32_t mDataClass;
    uint32_t mDataType;
    uint32_t mName[256 / 4];
    //uint32_t  _reserved[CNANOVDB_ALIGNMENT_PADDING(8 + 8 + 4 + 4 + 4 + 4 + 256, CNANOVDB_DATA_ALIGNMENT)/4];
CNANOVDB_DECLARE_STRUCT_END(nanovdb_GridBlindMetaData)

#define DEFINE_GRID(VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_LEAF_NODE_struct(0, 3, 3, VALUET, SIZEOF_VALUET, SUFFIX); \
    DEFINE_NODE_TILE_ENTRY_struct(VALUET, SIZEOF_VALUET, SUFFIX); \
    DEFINE_INTERNAL_NODE_struct(0, 1, 4, 7, VALUET, SIZEOF_VALUET, SUFFIX); \
    DEFINE_INTERNAL_NODE_struct(1, 2, 5, 12, VALUET, SIZEOF_VALUET, SUFFIX); \
    DEFINE_ROOT_TILE_struct(VALUET, SIZEOF_VALUET, SUFFIX); \
    DEFINE_ROOT_NODE_struct(VALUET, SIZEOF_VALUET, SUFFIX);

////////////////////////////////////////////////////////

// instantiate the grid structures.
DEFINE_GRID(VALUETYPE, SIZEOF_VALUETYPE, VALUETYPE);

////////////////////////////////////////////////////////
