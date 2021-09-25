// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////

#if defined(CNANOVDB_COMPILER_GLSL)
#line 7
#endif

CNANOVDB_INLINE vec3 nanovdb_CoordToVec3f(const nanovdb_Coord x)
{
    return CNANOVDB_MAKE_VEC3(CNANOVDB_MAKE(float)(x.mVec[0]), CNANOVDB_MAKE(float)(x.mVec[1]), CNANOVDB_MAKE(float)(x.mVec[2]));
}

CNANOVDB_INLINE nanovdb_Coord nanovdb_Vec3fToCoord(const vec3 p)
{
    nanovdb_Coord x;
    x.mVec[0] = CNANOVDB_MAKE(int32_t)(floor(p.x));
    x.mVec[1] = CNANOVDB_MAKE(int32_t)(floor(p.y));
    x.mVec[2] = CNANOVDB_MAKE(int32_t)(floor(p.z));
    return x;
}

CNANOVDB_INLINE nanovdb_Coord nanovdb_Vec3iToCoord(const ivec3 p)
{
    nanovdb_Coord x;
    x.mVec[0] = p.x;
    x.mVec[1] = p.y;
    x.mVec[2] = p.z;
    return x;
}

////////////////////////////////////////////////////////

CNANOVDB_INLINE boolean nanovdb_Ray_clip(CNANOVDB_REF(nanovdb_Ray) ray, vec3 p0, vec3 p1)
{
    vec3  t0 = vec3_div(vec3_sub(p0, CNANOVDB_DEREF(ray).mEye), CNANOVDB_DEREF(ray).mDir);
    vec3  t1 = vec3_div(vec3_sub(p1, CNANOVDB_DEREF(ray).mEye), CNANOVDB_DEREF(ray).mDir);
    vec3  tmin = CNANOVDB_MAKE_VEC3(fmin(t0.x, t1.x), fmin(t0.y, t1.y), fmin(t0.z, t1.z));
    vec3  tmax = CNANOVDB_MAKE_VEC3(fmax(t0.x, t1.x), fmax(t0.y, t1.y), fmax(t0.z, t1.z));
    float mint = fmax(tmin.x, fmax(tmin.y, tmin.z));
    float maxt = fmin(tmax.x, fmin(tmax.y, tmax.z));
    boolean  hit = (mint <= maxt);
    CNANOVDB_DEREF(ray).mT0 = fmax(CNANOVDB_DEREF(ray).mT0, mint);
    CNANOVDB_DEREF(ray).mT1 = fmin(CNANOVDB_DEREF(ray).mT1, maxt);
    return hit;
}

CNANOVDB_INLINE vec3 nanovdb_Ray_start(CNANOVDB_REF(nanovdb_Ray) ray)
{
    return vec3_add(CNANOVDB_DEREF(ray).mEye, vec3_fmul(CNANOVDB_DEREF(ray).mT0, CNANOVDB_DEREF(ray).mDir));
}

CNANOVDB_INLINE vec3 nanovdb_Ray_eval(CNANOVDB_REF(nanovdb_Ray) ray, float t)
{
    return vec3_add(CNANOVDB_DEREF(ray).mEye, vec3_fmul(t, CNANOVDB_DEREF(ray).mDir));
}

////////////////////////////////////////////////////////

CNANOVDB_INLINE vec3 nanovdb_Map_apply(nanovdb_Map map, const vec3 src)
{
    vec3  dst;
    float sx = src.x;
    float sy = src.y;
    float sz = src.z;
    dst.x = sx * map.mMatF[0] + sy * map.mMatF[1] + sz * map.mMatF[2] + map.mVecF[0];
    dst.y = sx * map.mMatF[3] + sy * map.mMatF[4] + sz * map.mMatF[5] + map.mVecF[1];
    dst.z = sx * map.mMatF[6] + sy * map.mMatF[7] + sz * map.mMatF[8] + map.mVecF[2];
    return dst;
}

CNANOVDB_INLINE vec3 nanovdb_Map_applyInverse(nanovdb_Map map, const vec3 src)
{
    vec3  dst;
    float sx = src.x - map.mVecF[0];
    float sy = src.y - map.mVecF[1];
    float sz = src.z - map.mVecF[2];
    dst.x = sx * map.mInvMatF[0] + sy * map.mInvMatF[1] + sz * map.mInvMatF[2];
    dst.y = sx * map.mInvMatF[3] + sy * map.mInvMatF[4] + sz * map.mInvMatF[5];
    dst.z = sx * map.mInvMatF[6] + sy * map.mInvMatF[7] + sz * map.mInvMatF[8];
    return dst;
}

CNANOVDB_INLINE vec3 nanovdb_Map_applyJacobi(nanovdb_Map map, const vec3 src)
{
    vec3  dst;
    float sx = src.x;
    float sy = src.y;
    float sz = src.z;
    dst.x = sx * map.mMatF[0] + sy * map.mMatF[1] + sz * map.mMatF[2];
    dst.y = sx * map.mMatF[3] + sy * map.mMatF[4] + sz * map.mMatF[5];
    dst.z = sx * map.mMatF[6] + sy * map.mMatF[7] + sz * map.mMatF[8];
    return dst;
}

CNANOVDB_INLINE vec3 nanovdb_Map_applyInverseJacobi(nanovdb_Map map, const vec3 src)
{
    vec3  dst;
    float sx = src.x;
    float sy = src.y;
    float sz = src.z;
    dst.x = sx * map.mInvMatF[0] + sy * map.mInvMatF[1] + sz * map.mInvMatF[2];
    dst.y = sx * map.mInvMatF[3] + sy * map.mInvMatF[4] + sz * map.mInvMatF[5];
    dst.z = sx * map.mInvMatF[6] + sy * map.mInvMatF[7] + sz * map.mInvMatF[8];
    return dst;
}

////////////////////////////////////////////////////////

DEFINE_LEAF_NODE_functions(0, 3, 3, VALUETYPE, SIZEOF_VALUETYPE, VALUETYPE);
DEFINE_NODE_TILE_ENTRY_functions(VALUETYPE, SIZEOF_VALUETYPE, VALUETYPE);
DEFINE_INTERNAL_NODE_functions(0, 3, 1, 4, 7, VALUETYPE, SIZEOF_VALUETYPE, VALUETYPE);
DEFINE_INTERNAL_NODE_functions(1, 4, 2, 5, 12, VALUETYPE, SIZEOF_VALUETYPE, VALUETYPE);
DEFINE_ROOT_NODE_functions(VALUETYPE, SIZEOF_VALUETYPE, VALUETYPE, 5);


////////////////////////////////////////////////////////

CNANOVDB_INLINE nanovdb_ReadAccessor
nanovdb_ReadAccessor_create()
{
    nanovdb_ReadAccessor acc;
    acc.mKey.mVec[0] = acc.mKey.mVec[1] = acc.mKey.mVec[2] = 0;
    acc.mNodeIndex[0] = acc.mNodeIndex[1] = acc.mNodeIndex[2] = -1;
    return acc;
}

#define DEFINE_READACCESSOR_ISCACHED(LEVEL, MASK) \
    CNANOVDB_INLINE boolean \
        nanovdb_ReadAccessor_isCached##LEVEL(CNANOVDB_REF(nanovdb_ReadAccessor) acc, int32_t dirty) \
    { \
        if (CNANOVDB_DEREF(acc).mNodeIndex[LEVEL] < 0) \
            return CNANOVDB_FALSE; \
        if ((dirty & ~MASK) != 0) { \
            CNANOVDB_DEREF(acc).mNodeIndex[LEVEL] = -1; \
            return CNANOVDB_FALSE; \
        } \
        return CNANOVDB_TRUE; \
    }

DEFINE_READACCESSOR_ISCACHED(0, ((1u << 3) - 1u))
DEFINE_READACCESSOR_ISCACHED(1, ((1u << 7) - 1u))
DEFINE_READACCESSOR_ISCACHED(2, ((1u << 12) - 1u))

CNANOVDB_INLINE int32_t
nanovdb_ReadAccessor_computeDirty(nanovdb_ReadAccessor acc, nanovdb_Coord ijk)
{
    return (ijk.mVec[0] ^ acc.mKey.mVec[0]) |
           (ijk.mVec[1] ^ acc.mKey.mVec[1]) |
           (ijk.mVec[2] ^ acc.mKey.mVec[2]);
}

CNANOVDB_INLINE VALUETYPE
nanovdb_ReadAccessor_getValue(CNANOVDB_CONTEXT cxt, CNANOVDB_REF(nanovdb_ReadAccessor) acc, nanovdb_Coord ijk)
{
#ifdef CNANOVDB_USE_READACCESSOR_CACHE
    int32_t dirty = nanovdb_ReadAccessor_computeDirty(CNANOVDB_DEREF(acc), ijk);

    if (nanovdb_ReadAccessor_isCached0(acc, dirty))
        return nanovdb_Node0_float_getValueAndCache(cxt, CNANOVDB_DEREF(acc).mNodeIndex[0], ijk, acc);
    else if (nanovdb_ReadAccessor_isCached1(acc, dirty))
        return nanovdb_Node1_float_getValueAndCache(cxt, CNANOVDB_DEREF(acc).mNodeIndex[1], ijk, acc);
    else if (nanovdb_ReadAccessor_isCached2(acc, dirty))
        return nanovdb_Node2_float_getValueAndCache(cxt, CNANOVDB_DEREF(acc).mNodeIndex[2], ijk, acc);
    else
        return nanovdb_RootData_float_getValueAndCache(cxt, ijk, acc);
#else
    return nanovdb_RootData_float_getValue(cxt, ijk);
#endif
}

CNANOVDB_INLINE int32_t
nanovdb_ReadAccessor_getDim(CNANOVDB_CONTEXT cxt, CNANOVDB_REF(nanovdb_ReadAccessor) acc, nanovdb_Coord ijk, nanovdb_Ray ray)
{
#ifdef CNANOVDB_USE_READACCESSOR_CACHE
    int32_t dirty = nanovdb_ReadAccessor_computeDirty(CNANOVDB_DEREF(acc), ijk);

    if (nanovdb_ReadAccessor_isCached0(acc, dirty))
        return nanovdb_Node0_float_getDimAndCache(cxt, CNANOVDB_DEREF(acc).mNodeIndex[0], ijk, ray, acc);
    else if (nanovdb_ReadAccessor_isCached1(acc, dirty))
        return nanovdb_Node1_float_getDimAndCache(cxt, CNANOVDB_DEREF(acc).mNodeIndex[1], ijk, ray, acc);
    else if (nanovdb_ReadAccessor_isCached2(acc, dirty))
        return nanovdb_Node2_float_getDimAndCache(cxt, CNANOVDB_DEREF(acc).mNodeIndex[2], ijk, ray, acc);
    else
        return nanovdb_RootData_float_getDimAndCache(cxt, ijk, ray, acc);
#else
    return nanovdb_RootData_float_getDimAndCache(cxt, ijk, ray, acc);
#endif
}

CNANOVDB_INLINE boolean nanovdb_ReadAccessor_isActive(nanovdb_ReadAccessor acc, nanovdb_Coord ijk)
{
    return CNANOVDB_TRUE;
}

////////////////////////////////////////////////////////

CNANOVDB_INLINE vec3 nanovdb_Grid_worldToIndexF(nanovdb_GridData grid, const vec3 src)
{
    return nanovdb_Map_applyInverse(grid.mMap, src);
}

CNANOVDB_INLINE vec3 nanovdb_Grid_indexToWorldF(nanovdb_GridData grid, const vec3 src)
{
    return nanovdb_Map_apply(grid.mMap, src);
}

CNANOVDB_INLINE vec3 nanovdb_Grid_worldToIndexDirF(nanovdb_GridData grid, const vec3 src)
{
    return nanovdb_Map_applyInverseJacobi(grid.mMap, src);
}

CNANOVDB_INLINE vec3 nanovdb_Grid_indexToWorldDirF(nanovdb_GridData grid, const vec3 src)
{
    return nanovdb_Map_applyJacobi(grid.mMap, src);
}

CNANOVDB_INLINE nanovdb_Ray nanovdb_Ray_worldToIndexF(nanovdb_Ray ray, const nanovdb_GridData grid)
{
    nanovdb_Ray result;
    const vec3  eye = nanovdb_Grid_worldToIndexF(grid, ray.mEye);
    const vec3  dir = nanovdb_Grid_worldToIndexDirF(grid, ray.mDir);
    const float len = vec3_length(dir), invLength = CNANOVDB_MAKE(float)(1) / len;
    float       t1 = ray.mT1;
    if (t1 < MaxFloat)
        t1 *= len;
    result.mEye = eye;
    result.mDir = vec3_mulf(dir, invLength);
    result.mT0 = len * ray.mT0;
    result.mT1 = t1;
    return result;
}

////////////////////////////////////////////////////////