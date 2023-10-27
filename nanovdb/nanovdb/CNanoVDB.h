// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

//
// Simple C-wrapper for the nanovdb structure
// Meant for systems where you lack a C++ compiler.
//
#ifndef __CNANOVDB__
#define __CNANOVDB__

#define CNANOVDB_DATA_ALIGNMENT 32
#define CNANOVDB_ALIGNMENT_PADDING(x, n) (-(x) & ((n)-1))

#define USE_SINGLE_ROOT_KEY

#ifdef __OPENCL_VERSION__

#define CNANOVDB_GLOBAL __global
#define RESTRICT restrict

// OpenCL doesn't define these basic types:
typedef unsigned long uint64_t;
typedef long int64_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

#else

#define CNANOVDB_GLOBAL
#define RESTRICT __restrict

#endif


enum cnanovdb_GridType
{
    cnanovdb_GridType_Unknown = 0,
    cnanovdb_GridType_Float   = 1,
    cnanovdb_GridType_Double  = 2,
    cnanovdb_GridType_Int16   = 3,
    cnanovdb_GridType_Int32   = 4,
    cnanovdb_GridType_Int64   = 5,
    cnanovdb_GridType_Vec3f   = 6,
    cnanovdb_GridType_Vec3d   = 7,
    cnanovdb_GridType_Mask    = 8,
    cnanovdb_GridType_FP16    = 9,
    cnanovdb_GridType_End     = 10
};

#define ROOT_LEVEL 3

#define DEFINEMASK_int(LOG2DIM, SIZE) \
typedef struct \
{ \
    uint64_t    mWords[SIZE >> 6]; \
} cnanovdb_mask##LOG2DIM; \
\
static void cnanovdb_mask##LOG2DIM##_clear(CNANOVDB_GLOBAL cnanovdb_mask##LOG2DIM *RESTRICT mask) \
{ for (uint32_t i = 0; i < (SIZE >> 6); i++) mask->mWords[i] = 0; } \
\
static bool cnanovdb_mask##LOG2DIM##_isOn(const CNANOVDB_GLOBAL cnanovdb_mask##LOG2DIM *RESTRICT mask, uint32_t n) \
{ return 0 != (mask->mWords[n >> 6] & (((uint64_t)(1)) << (n & 63))); } \
/**/

#define DEFINEMASK(LOG2DIM) \
    DEFINEMASK_int(LOG2DIM, (1U << (3*LOG2DIM)))

#define INSTANTIATE(LOG2DIM) \
    DEFINEMASK(LOG2DIM)

INSTANTIATE(3)
INSTANTIATE(4)
INSTANTIATE(5)

typedef struct
{
    float  mMatF[9];    // r,c = 3*r + c
    float  mInvMatF[9]; // r,c = 3*r + c
    float  mVecF[3];
    float  mTaperF;
    double mMatD[9];    // r,c = 3*r + c
    double mInvMatD[9]; // r,c = 3*r + c
    double mVecD[3];
    double mTaperD;
} cnanovdb_map;

typedef struct
{
    float       mVec[3];
} cnanovdb_Vec3F;

typedef struct
{
    int32_t     mVec[3];
} cnanovdb_coord;

static int
cnanovdb_coord_compare(const CNANOVDB_GLOBAL cnanovdb_coord *a, const cnanovdb_coord *b)
{
    if (a->mVec[0] < b->mVec[0])
        return -1;
    if (a->mVec[0] > b->mVec[0])
        return 1;
    if (a->mVec[1] < b->mVec[1])
        return -1;
    if (a->mVec[1] > b->mVec[1])
        return 1;
    if (a->mVec[2] < b->mVec[2])
        return -1;
    if (a->mVec[2] > b->mVec[2])
        return 1;
    return 0;
}

#ifdef USE_SINGLE_ROOT_KEY
static uint64_t
cnanovdb_coord_to_key(const cnanovdb_coord *RESTRICT ijk)
{
    // Define to workaround a bug with 64-bit shifts in the AMD OpenCL compiler.
#if defined(AVOID_64BIT_SHIFT)
    uint2 key = (uint2)( ((uint32_t)ijk->mVec[2]) >> 12, 0) |
                (uint2)((((uint32_t)ijk->mVec[1]) >> 12) << 21,
                         ((uint32_t)ijk->mVec[1]) >> 23) |
             (uint2)(0, (((uint32_t)ijk->mVec[0]) >> 12) << 10);
    return *(uint64_t *)&key;
#else
    return  ((uint64_t) (((uint32_t)ijk->mVec[2]) >> 12)) |
           (((uint64_t) (((uint32_t)ijk->mVec[1]) >> 12)) << 21) |
           (((uint64_t) (((uint32_t)ijk->mVec[0]) >> 12)) << 42);
#endif
}
#else
static void
cnanovdb_coord_to_key(cnanovdb_coord *RESTRICT key, const cnanovdb_coord *RESTRICT ijk)
{
    key->mVec[0] = ijk->mVec[0] & ~((1u << 12) - 1u);
    key->mVec[1] = ijk->mVec[1] & ~((1u << 12) - 1u);
    key->mVec[2] = ijk->mVec[2] & ~((1u << 12) - 1u);
}
#endif

static void
cnanovdb_map_apply(cnanovdb_Vec3F *dst, const CNANOVDB_GLOBAL cnanovdb_map *RESTRICT map, const cnanovdb_Vec3F *src)
{
    float sx = src->mVec[0];
    float sy = src->mVec[1];
    float sz = src->mVec[2];
    dst->mVec[0] = sx * map->mMatF[0] + sy * map->mMatF[1] + sz * map->mMatF[2] + map->mVecF[0];
    dst->mVec[1] = sx * map->mMatF[3] + sy * map->mMatF[4] + sz * map->mMatF[5] + map->mVecF[1];
    dst->mVec[2] = sx * map->mMatF[6] + sy * map->mMatF[7] + sz * map->mMatF[8] + map->mVecF[2];
}

static void
cnanovdb_map_applyInverse(cnanovdb_Vec3F *dst, const CNANOVDB_GLOBAL cnanovdb_map *RESTRICT map, const cnanovdb_Vec3F *src)
{
    float sx = src->mVec[0] - map->mVecF[0];
    float sy = src->mVec[1] - map->mVecF[1];
    float sz = src->mVec[2] - map->mVecF[2];
    dst->mVec[0] = sx * map->mInvMatF[0] + sy * map->mInvMatF[1] + sz * map->mInvMatF[2];
    dst->mVec[1] = sx * map->mInvMatF[3] + sy * map->mInvMatF[4] + sz * map->mInvMatF[5];
    dst->mVec[2] = sx * map->mInvMatF[6] + sy * map->mInvMatF[7] + sz * map->mInvMatF[8];
}

static void
cnanovdb_map_applyJacobi(cnanovdb_Vec3F *dst, const CNANOVDB_GLOBAL cnanovdb_map *RESTRICT map, const cnanovdb_Vec3F *src)
{
    float sx = src->mVec[0];
    float sy = src->mVec[1];
    float sz = src->mVec[2];
    dst->mVec[0] = sx * map->mMatF[0] + sy * map->mMatF[1] + sz * map->mMatF[2];
    dst->mVec[1] = sx * map->mMatF[3] + sy * map->mMatF[4] + sz * map->mMatF[5];
    dst->mVec[2] = sx * map->mMatF[6] + sy * map->mMatF[7] + sz * map->mMatF[8];
}

static void
cnanovdb_map_applyInverseJacobi(cnanovdb_Vec3F *dst, const CNANOVDB_GLOBAL cnanovdb_map *RESTRICT map, const cnanovdb_Vec3F *src)
{
    float sx = src->mVec[0];
    float sy = src->mVec[1];
    float sz = src->mVec[2];
    dst->mVec[0] = sx * map->mInvMatF[0] + sy * map->mInvMatF[1] + sz * map->mInvMatF[2];
    dst->mVec[1] = sx * map->mInvMatF[3] + sy * map->mInvMatF[4] + sz * map->mInvMatF[5];
    dst->mVec[2] = sx * map->mInvMatF[6] + sy * map->mInvMatF[7] + sz * map->mInvMatF[8];
}

static void
cnanovdb_map_applyIJT(cnanovdb_Vec3F *dst, const CNANOVDB_GLOBAL cnanovdb_map *RESTRICT map, const cnanovdb_Vec3F *src)
{
    float sx = src->mVec[0];
    float sy = src->mVec[1];
    float sz = src->mVec[2];
    dst->mVec[0] = sx * map->mInvMatF[0] + sy * map->mInvMatF[3] + sz * map->mInvMatF[6];
    dst->mVec[1] = sx * map->mInvMatF[1] + sy * map->mInvMatF[4] + sz * map->mInvMatF[7];
    dst->mVec[2] = sx * map->mInvMatF[2] + sy * map->mInvMatF[5] + sz * map->mInvMatF[8];
}

typedef struct
{
    int64_t     mByteOffset;   // byte offset to the blind data, relative to the GridData.
    uint64_t    mElementCount; // number of elements, e.g. point count
    uint32_t    mFlags;        // flags
    uint32_t    mSemantic;     // semantic meaning of the data.
    uint32_t    mDataClass;    // 4 bytes
    uint32_t    mDataType;     // 4 bytes
    char        mName[256];
    uint8_t     _reserved[CNANOVDB_ALIGNMENT_PADDING(sizeof(int64_t)+sizeof(uint64_t)+2*sizeof(uint32_t)+2*sizeof(uint32_t)+256*sizeof(char), CNANOVDB_DATA_ALIGNMENT)];
} cnanovdb_gridblindmetadata;

typedef struct
{
    uint64_t         mMagic; // 8B magic to validate it is valid grid data.
    uint64_t         mChecksum; // 8B. Checksum of grid buffer.
    uint32_t         mVersion;// 4B. compacted major.minor.path version number.
    uint32_t         mFlags; // 4B. flags for grid.
    uint32_t         mGridIndex;// 4B. Index of this grid in the buffer
    uint32_t         mGridCount; // 4B. Total number of grids in the buffer
    uint64_t         mGridSize; // 8B. byte count of this entire grid occupied in the buffer.
    char             mGridName[256]; // 256B
    cnanovdb_map     mMap; // 264B. affine transformation between index and world space in both single and double precision
    double           mBBox[6]; // 48B. floating-point bounds of active values in WORLD SPACE
    double           mVoxelSize[3]; // 24B. size of a voxel in world units
    uint32_t         mGridClass; // 4B.
    uint32_t         mGridType; // 4B.
    uint64_t         mBlindMetadataOffset; // 8B. offset of GridBlindMetaData structures.
    int32_t          mBlindMetadataCount; // 4B. count of GridBlindMetaData structures.
    uint32_t         _reserved[CNANOVDB_ALIGNMENT_PADDING(8 + 8 + 4 + 4 + 4 + 4 + 8 + 256 + 24 + 24 + sizeof(cnanovdb_map) + 24 + 4 + 4 + 8 + 4, CNANOVDB_DATA_ALIGNMENT) / 4];
} cnanovdb_griddata;

static void
cnanovdb_griddata_worldToIndex(cnanovdb_Vec3F *dst, const CNANOVDB_GLOBAL cnanovdb_griddata *RESTRICT grid, const cnanovdb_Vec3F *src)
{
    cnanovdb_map_applyInverse(dst, &grid->mMap, src);
}

static void
cnanovdb_griddata_indexToWorld(cnanovdb_Vec3F *dst, const CNANOVDB_GLOBAL cnanovdb_griddata *RESTRICT grid, const cnanovdb_Vec3F *src)
{
    cnanovdb_map_apply(dst, &grid->mMap, src);
}

static void
cnanovdb_griddata_worldToIndexDir(cnanovdb_Vec3F *dst, const CNANOVDB_GLOBAL cnanovdb_griddata *RESTRICT grid, const cnanovdb_Vec3F *src)
{
    cnanovdb_map_applyInverseJacobi(dst, &grid->mMap, src);
}

static void
cnanovdb_griddata_indexToWorldDir(cnanovdb_Vec3F *dst, const CNANOVDB_GLOBAL cnanovdb_griddata *RESTRICT grid, const cnanovdb_Vec3F *src)
{
    cnanovdb_map_applyJacobi(dst, &grid->mMap, src);
}

static void
cnanovdb_griddata_applyIJT(cnanovdb_Vec3F *dst, const CNANOVDB_GLOBAL cnanovdb_griddata *RESTRICT grid, const cnanovdb_Vec3F *src)
{
    cnanovdb_map_applyIJT(dst, &grid->mMap, src);
}

typedef struct
{
    uint64_t mNodeOffset[ROOT_LEVEL + 1];
    uint32_t mNodeCount[ROOT_LEVEL];
    uint32_t mTileCount[ROOT_LEVEL];
    uint64_t mVoxelCount;
    uint8_t  _reserved[CNANOVDB_ALIGNMENT_PADDING(4*sizeof(uint64_t)+(3+3)*sizeof(uint32_t)+sizeof(uint64_t), CNANOVDB_DATA_ALIGNMENT)];
} cnanovdb_treedata;

static const CNANOVDB_GLOBAL cnanovdb_treedata *
cnanovdb_griddata_tree(const CNANOVDB_GLOBAL cnanovdb_griddata *RESTRICT griddata)
{
    return (const CNANOVDB_GLOBAL cnanovdb_treedata *)(griddata + 1);
}

#define CREATE_TILEENTRY(VALUETYPE, SUFFIX) \
typedef union \
{ \
    VALUETYPE   value; \
    uint64_t    child; \
} cnanovdb_tileentry##SUFFIX; \
/**/

typedef struct
{
    cnanovdb_coord              mKey;
    const CNANOVDB_GLOBAL void *mNode[4];
} cnanovdb_readaccessor;


static void
cnanovdb_readaccessor_insert(cnanovdb_readaccessor *RESTRICT acc, int childlevel, const CNANOVDB_GLOBAL void *RESTRICT node, const cnanovdb_coord *RESTRICT ijk)
{
    acc->mNode[childlevel] = node;
    acc->mKey.mVec[0] = ijk->mVec[0];
    acc->mKey.mVec[1] = ijk->mVec[1];
    acc->mKey.mVec[2] = ijk->mVec[2];
}

#define CREATE_LEAF_NODE_int(LEVEL, LOG2DIM, CHILDTOTAL, TOTAL, MASK, VALUETYPE, STATSTYPE, SUFFIX) \
typedef struct \
{ \
    cnanovdb_coord              mBBox_min; \
    uint8_t                     mBBoxDif[3]; \
    uint8_t                     mFlags; \
    cnanovdb_mask##LOG2DIM      mValueMask; \
    VALUETYPE                   mMinimum; \
    VALUETYPE                   mMaximum; \
    STATSTYPE                   mAverage; \
    STATSTYPE                   mStdDevi; \
    uint32_t                    _reserved[ CNANOVDB_ALIGNMENT_PADDING(sizeof(cnanovdb_mask##LOG2DIM)+2*sizeof(VALUETYPE)+2*sizeof(STATSTYPE)+sizeof(cnanovdb_coord)+sizeof(uint8_t[3])+sizeof(uint8_t), CNANOVDB_DATA_ALIGNMENT)/4]; \
    VALUETYPE                   mVoxels[1u << (3*LOG2DIM)]; \
} cnanovdb_node##LEVEL##SUFFIX; \
\
static uint32_t \
cnanovdb_node##LEVEL##SUFFIX##_CoordToOffset(const cnanovdb_coord *RESTRICT ijk) \
{ \
    return ( ( ( ijk->mVec[0] & MASK ) >> CHILDTOTAL ) << ( 2 * LOG2DIM ) ) + \
           ( ( ( ijk->mVec[1] & MASK ) >> CHILDTOTAL ) << ( LOG2DIM ) ) + \
             ( ( ijk->mVec[2] & MASK ) >> CHILDTOTAL ); \
} \
\
static VALUETYPE \
cnanovdb_node##LEVEL##SUFFIX##_getValue(const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *RESTRICT node, const cnanovdb_coord *RESTRICT ijk) \
{ \
    uint32_t n = cnanovdb_node##LEVEL##SUFFIX##_CoordToOffset(ijk); \
    return node->mVoxels[n]; \
} \
\
static VALUETYPE \
cnanovdb_node##LEVEL##SUFFIX##_getValueAndCache(const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *RESTRICT node, const cnanovdb_coord *RESTRICT ijk, cnanovdb_readaccessor *RESTRICT /* DO NOT REMOVE: Required for C99 compliance */ acc) \
{ \
    (void)(acc); \
    uint32_t n = cnanovdb_node##LEVEL##SUFFIX##_CoordToOffset(ijk); \
    return node->mVoxels[n]; \
} \
\
static bool \
cnanovdb_node##LEVEL##SUFFIX##_isActive(const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *RESTRICT node, const cnanovdb_coord *RESTRICT ijk) \
{ \
    uint32_t n = cnanovdb_node##LEVEL##SUFFIX##_CoordToOffset(ijk); \
    if (cnanovdb_mask##LOG2DIM##_isOn(&node->mValueMask, n)) \
        return true; \
    return false; \
} \
\
static bool \
cnanovdb_node##LEVEL##SUFFIX##_isActiveAndCache(const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *RESTRICT node, const cnanovdb_coord *RESTRICT ijk, cnanovdb_readaccessor *RESTRICT /* DO NOT REMOVE: Required for C99 compliance */ acc) \
{ \
    (void)(acc); \
    uint32_t n = cnanovdb_node##LEVEL##SUFFIX##_CoordToOffset(ijk); \
    if (cnanovdb_mask##LOG2DIM##_isOn(&node->mValueMask, n)) \
        return true; \
    return false; \
} \
\
static const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX * \
cnanovdb_tree_getNode##LEVEL##SUFFIX(const CNANOVDB_GLOBAL cnanovdb_treedata *RESTRICT tree, uint64_t i) \
{ \
    const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *basenode = (const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *)((CNANOVDB_GLOBAL uint8_t *)(tree) + tree->mNodeOffset[LEVEL]); \
    return basenode + i; \
} \
\
/**/

#define CREATE_LEAF_NODE(LEVEL, LOG2DIM, TOTAL, VALUETYPE, STATSTYPE, SUFFIX) \
CREATE_LEAF_NODE_int(LEVEL, LOG2DIM, (TOTAL-LOG2DIM), TOTAL, ((1u << TOTAL) - 1u), VALUETYPE, STATSTYPE, SUFFIX)

#define CREATE_INTERNAL_NODE_int(CHILDLEVEL, LEVEL, LOG2DIM, CHILDTOTAL, TOTAL, MASK, VALUETYPE, STATSTYPE, SUFFIX) \
typedef struct \
{ \
    cnanovdb_coord               mBBox_min, mBBox_max; \
    int32_t                      mOffset; \
    uint32_t                     mFlags; \
    cnanovdb_mask##LOG2DIM       mValueMask, mChildMask; \
    VALUETYPE                    mMinimum, mMaximum; \
    STATSTYPE                    mAverage, mStdDevi; \
    uint8_t                      _reserved[CNANOVDB_ALIGNMENT_PADDING(sizeof(cnanovdb_mask##LOG2DIM)+sizeof(VALUETYPE)*2+sizeof(STATSTYPE)*2+sizeof(cnanovdb_coord)*2+sizeof(int32_t)+sizeof(uint32_t), CNANOVDB_DATA_ALIGNMENT)]; \
    cnanovdb_tileentry##SUFFIX   mTable[1u << (3*LOG2DIM)]; \
} cnanovdb_node##LEVEL##SUFFIX; \
\
static uint32_t \
cnanovdb_node##LEVEL##SUFFIX##_CoordToOffset(const cnanovdb_coord *RESTRICT ijk) \
{ \
    return ( ( ( ijk->mVec[0] & MASK ) >> CHILDTOTAL ) << ( 2 * LOG2DIM ) ) + \
           ( ( ( ijk->mVec[1] & MASK ) >> CHILDTOTAL ) << ( LOG2DIM ) ) + \
             ( ( ijk->mVec[2] & MASK ) >> CHILDTOTAL ); \
} \
\
static const CNANOVDB_GLOBAL cnanovdb_node##CHILDLEVEL##SUFFIX * \
cnanovdb_node##LEVEL##SUFFIX##_getChild(const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *RESTRICT node, uint32_t n) \
{ \
    const CNANOVDB_GLOBAL cnanovdb_node##CHILDLEVEL##SUFFIX *childnode = (const CNANOVDB_GLOBAL cnanovdb_node##CHILDLEVEL##SUFFIX *)( ((CNANOVDB_GLOBAL uint8_t *)node) + node->mTable[n].child); \
    return childnode; \
} \
\
static VALUETYPE \
cnanovdb_node##LEVEL##SUFFIX##_getValue(const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *RESTRICT node, const cnanovdb_coord *RESTRICT ijk) \
{ \
    uint32_t n = cnanovdb_node##LEVEL##SUFFIX##_CoordToOffset(ijk); \
    if (cnanovdb_mask##LOG2DIM##_isOn(&node->mChildMask, n)) \
    { \
        const CNANOVDB_GLOBAL cnanovdb_node##CHILDLEVEL##SUFFIX *child = cnanovdb_node##LEVEL##SUFFIX##_getChild(node, n); \
        return cnanovdb_node##CHILDLEVEL##SUFFIX##_getValue(child, ijk); \
    } \
    return node->mTable[n].value; \
} \
\
static VALUETYPE \
cnanovdb_node##LEVEL##SUFFIX##_getValueAndCache(const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *RESTRICT node, const cnanovdb_coord *RESTRICT ijk, cnanovdb_readaccessor *RESTRICT acc) \
{ \
    uint32_t n = cnanovdb_node##LEVEL##SUFFIX##_CoordToOffset(ijk); \
    if (cnanovdb_mask##LOG2DIM##_isOn(&node->mChildMask, n)) \
    { \
        const CNANOVDB_GLOBAL cnanovdb_node##CHILDLEVEL##SUFFIX *child = cnanovdb_node##LEVEL##SUFFIX##_getChild(node, n); \
        cnanovdb_readaccessor_insert(acc, CHILDLEVEL, child, ijk); \
        return cnanovdb_node##CHILDLEVEL##SUFFIX##_getValueAndCache(child, ijk, acc); \
    } \
    return node->mTable[n].value; \
} \
\
static bool \
cnanovdb_node##LEVEL##SUFFIX##_isActive(const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *RESTRICT node, const cnanovdb_coord *RESTRICT ijk) \
{ \
    uint32_t n = cnanovdb_node##LEVEL##SUFFIX##_CoordToOffset(ijk); \
    if (cnanovdb_mask##LOG2DIM##_isOn(&node->mChildMask, n)) \
    { \
        const CNANOVDB_GLOBAL cnanovdb_node##CHILDLEVEL##SUFFIX *child = cnanovdb_node##LEVEL##SUFFIX##_getChild(node, n); \
        return cnanovdb_node##CHILDLEVEL##SUFFIX##_isActive(child, ijk); \
    } \
    return cnanovdb_mask##LOG2DIM##_isOn(&node->mValueMask, n) ? true : false; \
} \
\
static bool \
cnanovdb_node##LEVEL##SUFFIX##_isActiveAndCache(const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *RESTRICT node, const cnanovdb_coord *RESTRICT ijk, cnanovdb_readaccessor *RESTRICT acc) \
{ \
    uint32_t n = cnanovdb_node##LEVEL##SUFFIX##_CoordToOffset(ijk); \
    if (cnanovdb_mask##LOG2DIM##_isOn(&node->mChildMask, n)) \
    { \
        const CNANOVDB_GLOBAL cnanovdb_node##CHILDLEVEL##SUFFIX *child = cnanovdb_node##LEVEL##SUFFIX##_getChild(node, n); \
        cnanovdb_readaccessor_insert(acc, CHILDLEVEL, child, ijk); \
        return cnanovdb_node##CHILDLEVEL##SUFFIX##_isActiveAndCache(child, ijk, acc); \
    } \
    return cnanovdb_mask##LOG2DIM##_isOn(&node->mValueMask, n) ? true : false; \
} \
\
static const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX * \
cnanovdb_tree_getNode##LEVEL##SUFFIX(const CNANOVDB_GLOBAL cnanovdb_treedata *RESTRICT tree, uint64_t i) \
{ \
    const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *basenode = (const CNANOVDB_GLOBAL cnanovdb_node##LEVEL##SUFFIX *)((CNANOVDB_GLOBAL uint8_t *)(tree) + tree->mNodeOffset[LEVEL]); \
    return basenode + i; \
} \
\
/**/

#define CREATE_INTERNAL_NODE(CHILDLEVEL, LEVEL, LOG2DIM, TOTAL, VALUETYPE, STATSTYPE, SUFFIX) \
CREATE_INTERNAL_NODE_int(CHILDLEVEL, LEVEL, LOG2DIM, (TOTAL-LOG2DIM), TOTAL, ((1u << TOTAL) - 1u), VALUETYPE, STATSTYPE, SUFFIX)


#ifdef USE_SINGLE_ROOT_KEY
#define DEFINE_KEY(KEY) \
        uint64_t        KEY;
#define KEYSIZE sizeof(uint64_t)

#define KEYSEARCH(SUFFIX) \
    uint64_t                key; \
    key = cnanovdb_coord_to_key(ijk); \
\
    for (int i = low; i < high; i++) \
    { \
        const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX   *tile = tiles + i; \
        if (tile->key == key) \
            return tile; \
    } \
/**/
#else
#define DEFINE_KEY(KEY) \
        cnanovdb_coord   KEY;
#define KEYSIZE sizeof(cnanovdb_coord)
#define KEYSEARCH(SUFFIX) \
    cnanovdb_coord key; \
    cnanovdb_coord_to_key(&key, ijk); \
 \
    while (low != high) \
    { \
        int32_t mid = low + (( high - low ) >> 1 ); \
        const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX   *tile = tiles + mid; \
 \
        int             keycmp = cnanovdb_coord_compare(&tile->key, &key); \
        if (keycmp == 0) \
        { \
            return tile; \
        } \
 \
        if (keycmp < 0) \
            low = mid + 1; \
        else \
            high = mid; \
    } \
/**/
#endif


#define CREATE_ROOTDATA(VALUETYPE, STATSTYPE, SUFFIX) \
typedef struct \
{ \
    DEFINE_KEY(key); \
    int64_t             child; \
    uint32_t            state; \
    VALUETYPE           value; \
    uint8_t             _reserved[CNANOVDB_ALIGNMENT_PADDING(sizeof(KEYSIZE)+sizeof(VALUETYPE)+sizeof(int64_t)+sizeof(uint32_t), CNANOVDB_DATA_ALIGNMENT)]; \
} cnanovdb_rootdata_tile##SUFFIX; \
 \
typedef struct \
{ \
    cnanovdb_coord mBBox_min, mBBox_max; \
    uint32_t       mTableSize; \
    VALUETYPE      mBackground; \
    VALUETYPE      mMinimum, mMaximum; \
    STATSTYPE      mAverage, mStdDevi; \
    uint32_t       _reserved[CNANOVDB_ALIGNMENT_PADDING(sizeof(cnanovdb_coord)*2+sizeof(uint32_t)+sizeof(VALUETYPE)*3+sizeof(STATSTYPE)*2, CNANOVDB_DATA_ALIGNMENT)/4]; \
} cnanovdb_rootdata##SUFFIX; \
 \
static const CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX * \
cnanovdb_treedata_root##SUFFIX(const CNANOVDB_GLOBAL cnanovdb_treedata *RESTRICT treedata) \
{ \
    return (const CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX *) ((const CNANOVDB_GLOBAL uint8_t *)(treedata) + treedata->mNodeOffset[ROOT_LEVEL]); \
} \
 \
static const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX * \
cnanovdb_rootdata##SUFFIX##_getTile(const CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX *RESTRICT rootdata, uint32_t n) \
{ \
    const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX *basetile = (const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX *) (rootdata + 1); \
    return basetile + n; \
} \
 \
static const CNANOVDB_GLOBAL cnanovdb_node2##SUFFIX * \
cnanovdb_rootdata##SUFFIX##_getChild(const CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX *RESTRICT rootdata, const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX *RESTRICT tile) \
{ \
    CNANOVDB_GLOBAL cnanovdb_node2##SUFFIX *basenode = (CNANOVDB_GLOBAL cnanovdb_node2##SUFFIX *) (((CNANOVDB_GLOBAL uint8_t *) rootdata) + tile->child); \
    return basenode; \
} \
 \
static const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX * \
cnanovdb_rootdata##SUFFIX##_findTile(const CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX *RESTRICT rootdata, const cnanovdb_coord *RESTRICT ijk) \
{ \
    int32_t                      low = 0, high = rootdata->mTableSize; \
    const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX *tiles = cnanovdb_rootdata##SUFFIX##_getTile(rootdata, 0); \
 \
    KEYSEARCH(SUFFIX) \
    return 0; \
} \
 \
static VALUETYPE \
cnanovdb_rootdata##SUFFIX##_getValue(const CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX *RESTRICT rootdata, const cnanovdb_coord *RESTRICT ijk) \
{ \
    const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX *tile = cnanovdb_rootdata##SUFFIX##_findTile(rootdata, ijk); \
    if (!tile) \
        return rootdata->mBackground; \
    if (tile->child == 0) \
        return tile->value; \
    return cnanovdb_node2##SUFFIX##_getValue( cnanovdb_rootdata##SUFFIX##_getChild(rootdata, tile), ijk ); \
} \
 \
static VALUETYPE \
cnanovdb_rootdata##SUFFIX##_getValueAndCache(const CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX *RESTRICT rootdata, const cnanovdb_coord *RESTRICT ijk, cnanovdb_readaccessor *RESTRICT acc) \
{ \
    const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX *tile = cnanovdb_rootdata##SUFFIX##_findTile(rootdata, ijk); \
    if (!tile) \
        return rootdata->mBackground; \
    if (tile->child == 0) \
        return tile->value; \
    const CNANOVDB_GLOBAL cnanovdb_node2##SUFFIX *child = cnanovdb_rootdata##SUFFIX##_getChild(rootdata, tile); \
    cnanovdb_readaccessor_insert(acc, 2, child, ijk); \
    return cnanovdb_node2##SUFFIX##_getValueAndCache( child, ijk, acc ); \
} \
\
static bool \
cnanovdb_rootdata##SUFFIX##_isActive(const CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX *RESTRICT rootdata, const cnanovdb_coord *RESTRICT ijk) \
{ \
    const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX *tile = cnanovdb_rootdata##SUFFIX##_findTile(rootdata, ijk); \
    if (!tile) \
        return false; \
    if (tile->child == 0) \
        return tile->state; \
    return cnanovdb_node2##SUFFIX##_isActive( cnanovdb_rootdata##SUFFIX##_getChild(rootdata, tile), ijk ); \
} \
 \
static bool \
cnanovdb_rootdata##SUFFIX##_isActiveAndCache(const CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX *RESTRICT rootdata, const cnanovdb_coord *RESTRICT ijk, cnanovdb_readaccessor *RESTRICT acc) \
{ \
    const CNANOVDB_GLOBAL cnanovdb_rootdata_tile##SUFFIX *tile = cnanovdb_rootdata##SUFFIX##_findTile(rootdata, ijk); \
    if (!tile) \
        return false; \
    if (tile->child == 0) \
        return tile->state; \
    const CNANOVDB_GLOBAL cnanovdb_node2##SUFFIX *child = cnanovdb_rootdata##SUFFIX##_getChild(rootdata, tile); \
    cnanovdb_readaccessor_insert(acc, 2, child, ijk); \
    return cnanovdb_node2##SUFFIX##_isActiveAndCache( child, ijk, acc ); \
} \
/**/


inline void
cnanovdb_readaccessor_init(cnanovdb_readaccessor *RESTRICT acc,
                    const CNANOVDB_GLOBAL void /*cnanovdb_rootdata* */ *RESTRICT rootdata)
{
    acc->mNode[0] = acc->mNode[1] = acc->mNode[2] = 0;
    acc->mNode[3] = rootdata;
}

#define DEFINE_ISCACHED(LEVEL, MASK) \
inline bool \
cnanovdb_readaccessor_isCached##LEVEL(cnanovdb_readaccessor *RESTRICT acc, int32_t dirty) \
{ \
    if (!acc->mNode[LEVEL]) \
        return false; \
    if (dirty & ~MASK) \
    { \
        acc->mNode[LEVEL] = 0; \
        return false; \
    } \
    return true; \
} \
/**/

DEFINE_ISCACHED(0, ((1u <<  3) - 1u) )
DEFINE_ISCACHED(1, ((1u <<  7) - 1u) )
DEFINE_ISCACHED(2, ((1u << 12) - 1u) )

inline int32_t
cnanovdb_readaccessor_computeDirty(const cnanovdb_readaccessor *RESTRICT acc, const cnanovdb_coord *RESTRICT ijk)
{
    return (ijk->mVec[0] ^ acc->mKey.mVec[0]) |
           (ijk->mVec[1] ^ acc->mKey.mVec[1]) |
           (ijk->mVec[2] ^ acc->mKey.mVec[2]);
}

#define CREATE_ACCESSOR(VALUETYPE, SUFFIX) \
inline VALUETYPE \
cnanovdb_readaccessor_getValue##SUFFIX(cnanovdb_readaccessor *RESTRICT acc, const cnanovdb_coord *RESTRICT ijk) \
{ \
    int32_t dirty = cnanovdb_readaccessor_computeDirty(acc, ijk); \
 \
    if (cnanovdb_readaccessor_isCached0(acc, dirty)) \
        return cnanovdb_node0##SUFFIX##_getValue( ((CNANOVDB_GLOBAL cnanovdb_node0##SUFFIX *) acc->mNode[0]), ijk); \
    if (cnanovdb_readaccessor_isCached1(acc, dirty)) \
        return cnanovdb_node1##SUFFIX##_getValueAndCache( ((CNANOVDB_GLOBAL cnanovdb_node1##SUFFIX *) acc->mNode[1]), ijk, acc); \
    if (cnanovdb_readaccessor_isCached2(acc, dirty)) \
        return cnanovdb_node2##SUFFIX##_getValueAndCache( ((CNANOVDB_GLOBAL cnanovdb_node2##SUFFIX *) acc->mNode[2]), ijk, acc); \
 \
    return cnanovdb_rootdata##SUFFIX##_getValueAndCache( ((CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX *)acc->mNode[3]), ijk, acc); \
} \
\
inline bool \
cnanovdb_readaccessor_isActive##SUFFIX(cnanovdb_readaccessor *RESTRICT acc, const cnanovdb_coord *RESTRICT ijk) \
{ \
    int32_t dirty = cnanovdb_readaccessor_computeDirty(acc, ijk); \
 \
    if (cnanovdb_readaccessor_isCached0(acc, dirty)) \
        return cnanovdb_node0##SUFFIX##_isActive( ((CNANOVDB_GLOBAL cnanovdb_node0##SUFFIX *) acc->mNode[0]), ijk); \
    if (cnanovdb_readaccessor_isCached1(acc, dirty)) \
        return cnanovdb_node1##SUFFIX##_isActiveAndCache( ((CNANOVDB_GLOBAL cnanovdb_node1##SUFFIX *) acc->mNode[1]), ijk, acc); \
    if (cnanovdb_readaccessor_isCached2(acc, dirty)) \
        return cnanovdb_node2##SUFFIX##_isActiveAndCache( ((CNANOVDB_GLOBAL cnanovdb_node2##SUFFIX *) acc->mNode[2]), ijk, acc); \
 \
    return cnanovdb_rootdata##SUFFIX##_isActiveAndCache( ((CNANOVDB_GLOBAL cnanovdb_rootdata##SUFFIX *)acc->mNode[3]), ijk, acc); \
} \
/**/


#define CREATE_GRIDTYPE(VALUETYPE, STATSTYPE, SUFFIX) \
CREATE_TILEENTRY(VALUETYPE, SUFFIX) \
CREATE_LEAF_NODE(0, 3, 3, VALUETYPE, STATSTYPE, SUFFIX) \
CREATE_INTERNAL_NODE(0, 1, 4, 7, VALUETYPE, STATSTYPE, SUFFIX) \
CREATE_INTERNAL_NODE(1, 2, 5, 12, VALUETYPE, STATSTYPE, SUFFIX) \
CREATE_ROOTDATA(VALUETYPE, STATSTYPE, SUFFIX) \
CREATE_ACCESSOR(VALUETYPE, SUFFIX) \
/**/

CREATE_GRIDTYPE(float, float, F)
CREATE_GRIDTYPE(cnanovdb_Vec3F, float, F3)

static int
cnanovdb_griddata_valid(const CNANOVDB_GLOBAL cnanovdb_griddata *RESTRICT grid)
{
    if (!grid)
        return 0;
    if (grid->mMagic != 0x304244566f6e614eUL && grid->mMagic != 0x314244566f6e614eUL)
        return 0;
    return 1;
}

static int
cnanovdb_griddata_validF(const CNANOVDB_GLOBAL cnanovdb_griddata *RESTRICT grid)
{
    if (!cnanovdb_griddata_valid(grid))
        return 0;
    if (grid->mGridType != cnanovdb_GridType_Float)
        return 0;
    return 1;
}

static int
cnanovdb_griddata_validF3(const CNANOVDB_GLOBAL cnanovdb_griddata *RESTRICT grid)
{
    if (!cnanovdb_griddata_valid(grid))
        return 0;
    if (grid->mGridType != cnanovdb_GridType_Vec3f)
        return 0;
    return 1;
}

#endif
