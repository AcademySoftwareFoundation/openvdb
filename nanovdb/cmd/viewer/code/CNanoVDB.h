
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
        nanovdb_Coord            mBBox_min; \
        uint8_t                  mBBoxDif[3]; \
        uint8_t                  mFlags; \
        nanovdb_BitMask##LOG2DIM mValueMask; \
        VALUET                   mValueMin; \
        VALUET                   mValueMax; \
        VALUET                   mAverage; \
        VALUET                   mStdDevi; \
        VALUET                   mVoxels[1 << (3 * LOG2DIM)]; \
    CNANOVDB_DECLARE_STRUCT_END(nanovdb_Node##LEVEL##_##SUFFIX) \
\
    CNANOVDB_CONSTANT_MEM int32_t nanovdb_Node##LEVEL##_TOTAL = TOTAL; \
    CNANOVDB_CONSTANT_MEM int32_t nanovdb_Node##LEVEL##_MASK = MASK;

#define DEFINE_LEAF_NODE_struct(LEVEL, LOG2DIM, TOTAL, VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_LEAF_NODE_struct_(LEVEL, LOG2DIM, (TOTAL - LOG2DIM), TOTAL, ((1 << TOTAL) - 1), VALUET, SIZEOF_VALUET, SUFFIX)

#define SIZEOF_LEAF_NODE(SIZEOF_VALUET, LOG2DIM) ((12) + (3) + (1) + (8 << (3 * LOG2DIM - 6)) + (SIZEOF_VALUET * 4) + (SIZEOF_VALUET * (1 << (3 * LOG2DIM))))
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
        return ((CNANOVDB_NODEDATA(cxt, LEVEL)[nodeIndex].mBBoxDif[2] & 7) != 0u) ? 1 << TOTAL : 1; \
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
#define SIZEOF_TILE_ENTRY(SIZEOF_VALUET) 4

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
    int64_t child; \
    CNANOVDB_DECLARE_UNION_END(nanovdb_TileEntry_##SUFFIX)
#define SIZEOF_TILE_ENTRY(SIZEOF_VALUET) (SIZEOF_VALUET <= 8 ? 8 : SIZEOF_VALUET)

//// functions:

#define DEFINE_NODE_TILE_ENTRY_functions_(VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_INLINE VALUET \
        nanovdb_TileEntry_##SUFFIX##_getValue(nanovdb_TileEntry_##SUFFIX entry) \
    { \
        return entry.value; \
    } \
    CNANOVDB_INLINE int64_t \
        nanovdb_TileEntry_##SUFFIX##_getChild(nanovdb_TileEntry_##SUFFIX entry) \
    { \
        return entry.child; \
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
        nanovdb_Coord              mBBox_min, mBBox_max; \
        uint64_t                   mFlags; \
        nanovdb_BitMask##LOG2DIM   mValueMask; \
        nanovdb_BitMask##LOG2DIM   mChildMask; \
        VALUET                     mValueMin; \
        VALUET                     mValueMax; \
        VALUET                     mAverage; \
        VALUET                     mStdDevi; \
        uint32_t                   _reserved[CNANOVDB_ALIGNMENT_PADDING((24) + (8) + (8 << (3 * LOG2DIM - 6)) * 2 + (SIZEOF_VALUET * 4), CNANOVDB_DATA_ALIGNMENT) / 4]; \
        nanovdb_TileEntry_##SUFFIX mTable[1 << (3 * LOG2DIM)]; \
    CNANOVDB_DECLARE_STRUCT_END(nanovdb_Node##LEVEL##_##SUFFIX) \
\
    CNANOVDB_CONSTANT_MEM int32_t nanovdb_Node##LEVEL##_TOTAL = TOTAL; \
    CNANOVDB_CONSTANT_MEM int32_t nanovdb_Node##LEVEL##_MASK = MASK;

#define SIZEOF_INTERNAL_NODE_(SIZEOF_VALUET, LOG2DIM) ((12 * 2) + (8) + ((8 << (3 * LOG2DIM - 6)) * 2) + (SIZEOF_VALUETYPE * 4) + (CNANOVDB_ALIGNMENT_PADDING((24) + (8) + (8 << (3 * LOG2DIM - 6)) * 2 + (SIZEOF_VALUETYPE * 4), CNANOVDB_DATA_ALIGNMENT)) + ((1 << (3 * LOG2DIM)) * SIZEOF_TILE_ENTRY(SIZEOF_VALUETYPE)))
#define SIZEOF_INTERNAL_NODE(LEVEL, SIZEOF_VALUET, LOG2DIM) (LEVEL == 0 ? SIZEOF_LEAF_NODE(SIZEOF_VALUET, LOG2DIM) : SIZEOF_INTERNAL_NODE_(SIZEOF_VALUET, LOG2DIM))

#define DEFINE_INTERNAL_NODE_struct(CHILDLEVEL, LEVEL, LOG2DIM, TOTAL, VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_INTERNAL_NODE_struct_(CHILDLEVEL, LEVEL, LOG2DIM, (TOTAL - LOG2DIM), TOTAL, ((1 << TOTAL) - 1), VALUET, SIZEOF_VALUET, SUFFIX)

//// functions:

#define DEFINE_INTERNAL_NODE_functions_(CHILDLEVEL, CHILDLOG2DIM, LEVEL, LOG2DIM, CHILDTOTAL, TOTAL, MASK, VALUET, SIZEOF_VALUET, SUFFIX) \
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
        int64_t            offset = nanovdb_TileEntry_##SUFFIX##_getChild(CNANOVDB_NODEDATA(cxt, LEVEL)[nodeIndex].mTable[n]); \
        CNANOVDB_WORD_TYPE root = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(CNANOVDB_NODEDATA(cxt, CHILDLEVEL)); \
        CNANOVDB_WORD_TYPE array_base = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(CNANOVDB_NODEDATA(cxt, LEVEL)); \
        CNANOVDB_WORD_TYPE array_item_offset = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(nodeIndex * SIZEOF_INTERNAL_NODE(LEVEL, SIZEOF_VALUETYPE, LOG2DIM)); \
        CNANOVDB_WORD_TYPE child_addr = array_base + array_item_offset + offset; \
        int32_t            diff = CNANOVDB_MAKE(int32_t)(child_addr - root); \
        int32_t            childIndex = diff / SIZEOF_INTERNAL_NODE(CHILDLEVEL, SIZEOF_VALUETYPE, CHILDLOG2DIM); \
        return childIndex; \
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

#define DEFINE_INTERNAL_NODE_functions(CHILDLEVEL, CHILDLOG2DIM, LEVEL, LOG2DIM, TOTAL, VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_INTERNAL_NODE_functions_(CHILDLEVEL, CHILDLOG2DIM, LEVEL, LOG2DIM, (TOTAL - LOG2DIM), TOTAL, ((1 << TOTAL) - 1), VALUET, SIZEOF_VALUET, SUFFIX)

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
        int64_t  child; \
        uint32_t state; \
        VALUET   value; \
        uint32_t _reserved[CNANOVDB_ALIGNMENT_PADDING((SIZEOF_ROOT_KEY) + (8) + (4) + (SIZEOF_VALUET), CNANOVDB_DATA_ALIGNMENT) / 4]; \
    CNANOVDB_DECLARE_STRUCT_END(nanovdb_RootData_Tile_##SUFFIX)

#define SIZEOF_ROOT_TILE(SIZEOF_VALUET) ((SIZEOF_ROOT_KEY) + (8) + (4) + (SIZEOF_VALUET) + (CNANOVDB_ALIGNMENT_PADDING((SIZEOF_ROOT_KEY) + (8) + (4) + (SIZEOF_VALUET), CNANOVDB_DATA_ALIGNMENT)))

#define DEFINE_ROOT_TILE_struct(VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_ROOT_TILE_struct_(VALUET, SIZEOF_VALUET, SUFFIX)

#define DEFINE_ROOT_NODE_struct_(VALUET, SIZEOF_VALUET, SUFFIX) \
    CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_RootData_##SUFFIX) \
        nanovdb_Coord mBBox_min, mBBox_max; \
        int32_t       mTileCount; \
        VALUET        mBackground, mValueMin, mValueMax, mAverage, mStdDevi; \
        uint32_t      _reserved[CNANOVDB_ALIGNMENT_PADDING((24) + (4) + (SIZEOF_VALUET * 5), CNANOVDB_DATA_ALIGNMENT) / 4]; \
    CNANOVDB_DECLARE_STRUCT_END(nanovdb_RootData_##SUFFIX)

#define SIZEOF_ROOT_NODE(SIZEOF_VALUET) ((12 * 2) + (4) + (SIZEOF_VALUET) + (CNANOVDB_ALIGNMENT_PADDING((24) + (4) + (SIZEOF_VALUET * 5), CNANOVDB_DATA_ALIGNMENT)))

//// functions:

#define DEFINE_ROOT_NODE_functions_(VALUET, SIZEOF_VALUET, SUFFIX, LOG2DIM) \
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
        int64_t offset = CNANOVDB_ROOTDATATILES(cxt)[rootTileIndex].child; \
        if (offset == 0) \
            return CNANOVDB_ROOTDATATILES(cxt)[rootTileIndex].value; \
        CNANOVDB_WORD_TYPE root = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(CNANOVDB_NODEDATA(cxt, 2)); \
        CNANOVDB_WORD_TYPE array_base = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(CNANOVDB_ROOTDATATILES(cxt)); \
        CNANOVDB_WORD_TYPE array_item_offset = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(rootTileIndex * SIZEOF_ROOT_TILE(SIZEOF_VALUETYPE)); \
        CNANOVDB_WORD_TYPE child_addr = array_base + array_item_offset + offset; \
        int32_t            diff = CNANOVDB_MAKE(int32_t)(child_addr - root); \
        int32_t            childIndex = diff / SIZEOF_INTERNAL_NODE(2, SIZEOF_VALUET, LOG2DIM); \
        return nanovdb_Node2_##SUFFIX##_getValue(cxt, childIndex, ijk); \
    } \
\
    CNANOVDB_INLINE VALUET \
        nanovdb_RootData_##SUFFIX##_getValueAndCache(CNANOVDB_CONTEXT cxt, nanovdb_Coord ijk, CNANOVDB_REF(nanovdb_ReadAccessor) acc) \
    { \
        int32_t rootTileIndex = nanovdb_RootData_##SUFFIX##_findTile(cxt, ijk); \
        if (rootTileIndex < 0) \
            return CNANOVDB_ROOTDATA(cxt).mBackground; \
        int64_t offset = CNANOVDB_ROOTDATATILES(cxt)[rootTileIndex].child; \
        if (offset == 0) \
            return CNANOVDB_ROOTDATATILES(cxt)[rootTileIndex].value; \
        CNANOVDB_WORD_TYPE root = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(CNANOVDB_NODEDATA(cxt, 2)); \
        CNANOVDB_WORD_TYPE array_base = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(CNANOVDB_ROOTDATATILES(cxt)); \
        CNANOVDB_WORD_TYPE array_item_offset = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(rootTileIndex * SIZEOF_ROOT_TILE(SIZEOF_VALUETYPE)); \
        CNANOVDB_WORD_TYPE child_addr = array_base + array_item_offset + offset; \
        int32_t            diff = CNANOVDB_MAKE(int32_t)(child_addr - root); \
        int32_t            childIndex = diff / SIZEOF_INTERNAL_NODE(2, SIZEOF_VALUET, LOG2DIM); \
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
        int64_t offset = CNANOVDB_ROOTDATATILES(cxt)[rootTileIndex].child; \
        if (offset == 0) \
            return 1 << nanovdb_Node2_TOTAL; \
        CNANOVDB_WORD_TYPE root = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(CNANOVDB_NODEDATA(cxt, 2)); \
        CNANOVDB_WORD_TYPE array_base = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(CNANOVDB_ROOTDATATILES(cxt)); \
        CNANOVDB_WORD_TYPE array_item_offset = CNANOVDB_MAKE(CNANOVDB_WORD_TYPE)(rootTileIndex * SIZEOF_ROOT_TILE(SIZEOF_VALUETYPE)); \
        CNANOVDB_WORD_TYPE child_addr = array_base + array_item_offset + offset; \
        int32_t            diff = CNANOVDB_MAKE(int32_t)(child_addr - root); \
        int32_t            childIndex = diff / SIZEOF_INTERNAL_NODE(2, SIZEOF_VALUET, LOG2DIM); \
        nanovdb_ReadAccessor_insert(acc, 2, childIndex, ijk); \
        return nanovdb_Node2_##SUFFIX##_getDimAndCache(cxt, childIndex, ijk, ray, acc); \
    }

#define DEFINE_ROOT_NODE_struct(VALUET, SIZEOF_VALUET, SUFFIX) \
    DEFINE_ROOT_NODE_struct_(VALUET, SIZEOF_VALUET, SUFFIX)

#define DEFINE_ROOT_NODE_functions(VALUET, SIZEOF_VALUET, SUFFIX, LOG2DIM) \
    DEFINE_ROOT_NODE_functions_(VALUET, SIZEOF_VALUET, SUFFIX, LOG2DIM)

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

CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_GridData)
    uint64_t    mMagic; // 8B magic to validate it is valid grid data.
    uint64_t    mChecksum; // 8B. Checksum of grid buffer.
    uint32_t    mMajor; // 4B. major version number.
    uint32_t    mFlags; // 4B. flags for grid.
    uint32_t    mGridIndex; // 4B. Index of this grid in the buffer
    uint32_t    mGridCount; // 4B. Total number of grids in the buffer
    uint64_t    mGridSize; // 8B. byte count of entire grid buffer.
    uint32_t    mGridName[256/4]; // 256B
    nanovdb_Map mMap; // 264B. affine transformation between index and world space in both single and double precision
    double      mBBoxMin[3]; // 24B. floating-point min-corner of active values in WORLD SPACE
    double      mBBoxMax[3]; // 24B. floating-point max-corner of active values in WORLD SPACE
    double      mVoxelSize[3]; // 24B. size of a voxel in world units
    uint32_t    mGridClass; // 4B.
    uint32_t    mGridType; // 4B.
    uint64_t    mBlindMetadataOffset; // 8B. offset of GridBlindMetaData structures.
    uint32_t    mBlindMetadataCount; // 4B. count of GridBlindMetaData structures.
    uint32_t    _reserved[CNANOVDB_ALIGNMENT_PADDING(8 + 8 + 4 + 4 + 8 + 256 + 24 + 24 + CNANOVDB_SIZEOF_nanovdb_Map + 24 + 4 + 4 + 8 + 4, CNANOVDB_DATA_ALIGNMENT) / 4];
CNANOVDB_DECLARE_STRUCT_END(nanovdb_GridData)

CNANOVDB_DECLARE_STRUCT_BEGIN(nanovdb_GridBlindMetaData)
    int64_t  mByteOffset; // byte offset to the blind data, relative to the GridData.
    int64_t  mByteSize; // byte size of blind data.
    uint32_t mFlags; // flags
    uint32_t mSemantic; // semantic meaning of the data.
    uint32_t mDataClass;
    uint32_t mDataType;
    uint32_t mName[256 / 4];
//uint32_t _reserved[CNANOVDB_ALIGNMENT_PADDING(8 + 8 + 4 + 4 + 4 + 4 + 256, CNANOVDB_DATA_ALIGNMENT)/4];
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
