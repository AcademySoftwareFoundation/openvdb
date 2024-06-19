// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

//
// Simple C-wrapper for voxel interpolation functions
//

#ifndef __CSAMPLEFROMVOXELS__
#define __CSAMPLEFROMVOXELS__

#include "../CNanoVDB.h"

#ifdef __OPENCL_VERSION__
#else
#include <math.h>
#endif

void
cnanovdb_coord_round(cnanovdb_coord *RESTRICT coord, const cnanovdb_Vec3F *RESTRICT xyz)
{
#ifdef __OPENCL_VERSION__
    coord->mVec[0] = floor(xyz->mVec[0]+0.5);
    coord->mVec[1] = floor(xyz->mVec[1]+0.5);
    coord->mVec[2] = floor(xyz->mVec[2]+0.5);
#else
    coord->mVec[0] = floorf(xyz->mVec[0]+0.5);
    coord->mVec[1] = floorf(xyz->mVec[1]+0.5);
    coord->mVec[2] = floorf(xyz->mVec[2]+0.5);
#endif
}

void
cnanovdb_coord_fract(cnanovdb_coord *RESTRICT coord, cnanovdb_Vec3F *RESTRICT fraction, const cnanovdb_Vec3F *RESTRICT xyz)
{
#ifdef __OPENCL_VERSION__
    float               i0, i1, i2;
    fraction->mVec[0] = fract(xyz->mVec[0], &i0);
    coord->mVec[0] = i0;
    fraction->mVec[1] = fract(xyz->mVec[1], &i1);
    coord->mVec[1] = i1;
    fraction->mVec[2] = fract(xyz->mVec[2], &i2);
    coord->mVec[2] = i2;
#else
    float               i0, i1, i2;
    i0 = floorf(xyz->mVec[0]);
    fraction->mVec[0] = xyz->mVec[0] - i0;
    coord->mVec[0] = i0;
    i1 = floorf(xyz->mVec[1]);
    fraction->mVec[1] = xyz->mVec[1] - i1;
    coord->mVec[1] = i1;
    i2 = floorf(xyz->mVec[2]);
    fraction->mVec[2] = xyz->mVec[2] - i2;
    coord->mVec[2] = i2;
#endif
}

#define CREATE_STENCIL(VALUETYPE, SUFFIX) \
typedef struct \
{ \
    VALUETYPE mStencil[2][2][2]; \
    cnanovdb_coord mCoord; \
} cnanovdb_stencil1##SUFFIX; \
 \
void \
cnanovdb_stencil1##SUFFIX##_clear(cnanovdb_stencil1##SUFFIX *RESTRICT stencil) \
{ \
    /* Invalid coords. */ \
    stencil->mCoord.mVec[0] = 0x80000000; \
    stencil->mCoord.mVec[1] = 0x80000000; \
    stencil->mCoord.mVec[2] = 0x80000000; \
} \
 \
void \
cnanovdb_stencil1##SUFFIX##_fill(cnanovdb_stencil1##SUFFIX *RESTRICT stencil, cnanovdb_readaccessor *RESTRICT acc, cnanovdb_coord *RESTRICT coord) \
{ \
    stencil->mStencil[0][0][0] = cnanovdb_readaccessor_getValue##SUFFIX(acc, coord); \
    coord->mVec[2] += 1; \
    stencil->mStencil[0][0][1] = cnanovdb_readaccessor_getValue##SUFFIX(acc, coord); \
    coord->mVec[1] += 1; \
    stencil->mStencil[0][1][1] = cnanovdb_readaccessor_getValue##SUFFIX(acc, coord); \
    coord->mVec[2] -= 1; \
    stencil->mStencil[0][1][0] = cnanovdb_readaccessor_getValue##SUFFIX(acc, coord); \
 \
    coord->mVec[0] += 1; \
    stencil->mStencil[1][1][0] = cnanovdb_readaccessor_getValue##SUFFIX(acc, coord); \
    coord->mVec[2] += 1; \
    stencil->mStencil[1][1][1] = cnanovdb_readaccessor_getValue##SUFFIX(acc, coord); \
    coord->mVec[1] -= 1; \
    stencil->mStencil[1][0][1] = cnanovdb_readaccessor_getValue##SUFFIX(acc, coord); \
    coord->mVec[2] -= 1; \
    stencil->mStencil[1][0][0] = cnanovdb_readaccessor_getValue##SUFFIX(acc, coord); \
    coord->mVec[0] -= 1; \
     \
    stencil->mCoord.mVec[0] = coord->mVec[0]; \
    stencil->mCoord.mVec[1] = coord->mVec[1]; \
    stencil->mCoord.mVec[2] = coord->mVec[2]; \
} \
 \
void \
cnanovdb_stencil1##SUFFIX##_update(cnanovdb_stencil1##SUFFIX *RESTRICT stencil, cnanovdb_readaccessor *RESTRICT acc, cnanovdb_coord *RESTRICT coord) \
{ \
    uint32_t change = (coord->mVec[0] ^ stencil->mCoord.mVec[0]) | \
                      (coord->mVec[1] ^ stencil->mCoord.mVec[1]) | \
                      (coord->mVec[2] ^ stencil->mCoord.mVec[2]); \
    if (!change) \
        return; \
 \
    cnanovdb_stencil1##SUFFIX##_fill(stencil, acc, coord); \
} \
/**/
CREATE_STENCIL(float, F)
CREATE_STENCIL(cnanovdb_Vec3F, F3)


#define CREATE_LERPSIMPLE(VALUETYPE, SUFFIX) \
VALUETYPE \
cnanovdb_lerp##SUFFIX(VALUETYPE a, VALUETYPE b, float w) \
{ \
    return a + w * (b - a); \
} \
/**/

CREATE_LERPSIMPLE(float, F)
CREATE_LERPSIMPLE(double, D)

cnanovdb_Vec3F
cnanovdb_lerpF3(cnanovdb_Vec3F a, cnanovdb_Vec3F b, float w)
{
    a.mVec[0] = cnanovdb_lerpF(a.mVec[0], b.mVec[0], w);
    a.mVec[1] = cnanovdb_lerpF(a.mVec[1], b.mVec[1], w);
    a.mVec[2] = cnanovdb_lerpF(a.mVec[2], b.mVec[2], w);
    return a;
}

#define CREATE_SAMPLE(VALUETYPE, SUFFIX) \
VALUETYPE \
cnanovdb_sample##SUFFIX##_nearest(cnanovdb_readaccessor *RESTRICT acc, const cnanovdb_Vec3F *RESTRICT xyz) \
{ \
    cnanovdb_coord coord; \
    cnanovdb_coord_round(&coord, xyz); \
    return cnanovdb_readaccessor_getValue##SUFFIX(acc, &coord); \
} \
 \
VALUETYPE \
cnanovdb_sample##SUFFIX##_trilinear(cnanovdb_readaccessor *RESTRICT acc, const cnanovdb_Vec3F *RESTRICT xyz) \
{ \
    cnanovdb_coord coord; \
    cnanovdb_Vec3F fraction; \
    cnanovdb_coord_fract(&coord, &fraction, xyz); \
 \
    VALUETYPE               vx, vx1, vy, vy1, vz, vz1; \
 \
    vz = cnanovdb_readaccessor_getValue##SUFFIX(acc, &coord); \
    coord.mVec[2] += 1; \
    vz1 = cnanovdb_readaccessor_getValue##SUFFIX(acc, &coord); \
    vy = cnanovdb_lerp##SUFFIX(vz, vz1, fraction.mVec[2]); \
 \
    coord.mVec[1] += 1; \
 \
    vz1 = cnanovdb_readaccessor_getValue##SUFFIX(acc, &coord); \
    coord.mVec[2] -= 1; \
    vz = cnanovdb_readaccessor_getValue##SUFFIX(acc, &coord); \
    vy1 = cnanovdb_lerp##SUFFIX(vz, vz1, fraction.mVec[2]); \
 \
    vx = cnanovdb_lerp##SUFFIX(vy, vy1, fraction.mVec[1]); \
 \
    coord.mVec[0] += 1; \
 \
    vz = cnanovdb_readaccessor_getValue##SUFFIX(acc, &coord); \
    coord.mVec[2] += 1; \
    vz1 = cnanovdb_readaccessor_getValue##SUFFIX(acc, &coord); \
    vy1 = cnanovdb_lerp##SUFFIX(vz, vz1, fraction.mVec[2]); \
 \
    coord.mVec[1] -= 1; \
 \
    vz1 = cnanovdb_readaccessor_getValue##SUFFIX(acc, &coord); \
    coord.mVec[2] -= 1; \
    vz = cnanovdb_readaccessor_getValue##SUFFIX(acc, &coord); \
    vy = cnanovdb_lerp##SUFFIX(vz, vz1, fraction.mVec[2]); \
 \
    vx1 = cnanovdb_lerp##SUFFIX(vy, vy1, fraction.mVec[1]); \
 \
    return cnanovdb_lerp##SUFFIX(vx, vx1, fraction.mVec[0]); \
} \
 \
VALUETYPE \
cnanovdb_sample##SUFFIX##_trilinear_stencil(cnanovdb_stencil1##SUFFIX *RESTRICT stencil, cnanovdb_readaccessor *RESTRICT acc, const cnanovdb_Vec3F *RESTRICT xyz) \
{ \
    cnanovdb_coord coord; \
    cnanovdb_Vec3F fraction; \
    cnanovdb_coord_fract(&coord, &fraction, xyz); \
 \
    cnanovdb_stencil1##SUFFIX##_update(stencil, acc, &coord); \
 \
    VALUETYPE               vx, vx1, vy, vy1, vz, vz1; \
 \
    vz = stencil->mStencil[0][0][0]; \
    vz1 = stencil->mStencil[0][0][1]; \
    vy = cnanovdb_lerp##SUFFIX(vz, vz1, fraction.mVec[2]); \
 \
    vz = stencil->mStencil[0][1][0]; \
    vz1 = stencil->mStencil[0][1][1]; \
    vy1 = cnanovdb_lerp##SUFFIX(vz, vz1, fraction.mVec[2]); \
 \
    vx = cnanovdb_lerp##SUFFIX(vy, vy1, fraction.mVec[1]); \
 \
    vz = stencil->mStencil[1][1][0]; \
    vz1 = stencil->mStencil[1][1][1]; \
    vy1 = cnanovdb_lerp##SUFFIX(vz, vz1, fraction.mVec[2]); \
 \
    vz = stencil->mStencil[1][0][0]; \
    vz1 = stencil->mStencil[1][0][1]; \
    vy = cnanovdb_lerp##SUFFIX(vz, vz1, fraction.mVec[2]); \
 \
    vx1 = cnanovdb_lerp##SUFFIX(vy, vy1, fraction.mVec[1]); \
 \
    return cnanovdb_lerp##SUFFIX(vx, vx1, fraction.mVec[0]); \
} \
/**/
CREATE_SAMPLE(float, F)
CREATE_SAMPLE(cnanovdb_Vec3F, F3)

void
cnanovdb_sampleF_gradient(cnanovdb_Vec3F *RESTRICT ret, cnanovdb_readaccessor *RESTRICT acc, const cnanovdb_Vec3F *RESTRICT xyz)
{
    cnanovdb_Vec3F qxyz;
    qxyz.mVec[0] = xyz->mVec[0];
    qxyz.mVec[1] = xyz->mVec[1];
    qxyz.mVec[2] = xyz->mVec[2];
    for (int i = 0; i < 3; i++)
    {
        float       sp, sm;

        qxyz.mVec[i] -= 0.5;
        sm = cnanovdb_sampleF_trilinear(acc, &qxyz);
        qxyz.mVec[i] += 1.0;
        sp = cnanovdb_sampleF_trilinear(acc, &qxyz);
        qxyz.mVec[i] -= 0.5;
        ret->mVec[i] = sp - sm;
    }
}

void
cnanovdb_sampleF_gradient0(cnanovdb_Vec3F *RESTRICT ret, cnanovdb_readaccessor *RESTRICT acc, const cnanovdb_Vec3F *RESTRICT xyz)
{
    cnanovdb_coord coord;
    cnanovdb_Vec3F fraction;
    cnanovdb_coord_fract(&coord, &fraction, xyz);

    float stencil[2][2][2];

    stencil[0][0][0] = cnanovdb_readaccessor_getValueF(acc, &coord);
    coord.mVec[2] += 1;
    stencil[0][0][1] = cnanovdb_readaccessor_getValueF(acc, &coord);
    coord.mVec[1] += 1;
    stencil[0][1][1] = cnanovdb_readaccessor_getValueF(acc, &coord);
    coord.mVec[2] -= 1;
    stencil[0][1][0] = cnanovdb_readaccessor_getValueF(acc, &coord);

    coord.mVec[0] += 1;
    stencil[1][1][0] = cnanovdb_readaccessor_getValueF(acc, &coord);
    coord.mVec[2] += 1;
    stencil[1][1][1] = cnanovdb_readaccessor_getValueF(acc, &coord);
    coord.mVec[1] -= 1;
    stencil[1][0][1] = cnanovdb_readaccessor_getValueF(acc, &coord);
    coord.mVec[2] -= 1;
    stencil[1][0][0] = cnanovdb_readaccessor_getValueF(acc, &coord);

    float D[4];

    D[0] = stencil[0][0][1] - stencil[0][0][0];
    D[1] = stencil[0][1][1] - stencil[0][1][0];
    D[2] = stencil[1][0][1] - stencil[1][0][0];
    D[3] = stencil[1][1][1] - stencil[1][1][0];

    ret->mVec[2] = cnanovdb_lerpF(
                        cnanovdb_lerpF(D[0], D[1], fraction.mVec[1]),
                        cnanovdb_lerpF(D[2], D[3], fraction.mVec[1]),
                        fraction.mVec[0] );

    float w = fraction.mVec[2];
    D[0] = stencil[0][0][0] + D[0] * w;
    D[1] = stencil[0][1][0] + D[1] * w;
    D[2] = stencil[1][0][0] + D[2] * w;
    D[3] = stencil[1][1][0] + D[3] * w;

    ret->mVec[0] =   cnanovdb_lerpF(D[2], D[3], fraction.mVec[1])
                   - cnanovdb_lerpF(D[0], D[1], fraction.mVec[1]);

    ret->mVec[1] = cnanovdb_lerpF(D[1] - D[0], D[3] - D[2], fraction.mVec[0]);
}

void
cnanovdb_sampleF_gradient0_stencil(cnanovdb_Vec3F *RESTRICT ret, cnanovdb_stencil1F *RESTRICT stencil, cnanovdb_readaccessor *RESTRICT acc, const cnanovdb_Vec3F *RESTRICT xyz)
{
    cnanovdb_coord coord;
    cnanovdb_Vec3F fraction;
    cnanovdb_coord_fract(&coord, &fraction, xyz);

    cnanovdb_stencil1F_update(stencil, acc, &coord);

    float D[4];

    D[0] = stencil->mStencil[0][0][1] - stencil->mStencil[0][0][0];
    D[1] = stencil->mStencil[0][1][1] - stencil->mStencil[0][1][0];
    D[2] = stencil->mStencil[1][0][1] - stencil->mStencil[1][0][0];
    D[3] = stencil->mStencil[1][1][1] - stencil->mStencil[1][1][0];

    ret->mVec[2] = cnanovdb_lerpF(
                        cnanovdb_lerpF(D[0], D[1], fraction.mVec[1]),
                        cnanovdb_lerpF(D[2], D[3], fraction.mVec[1]),
                        fraction.mVec[0] );

    float w = fraction.mVec[2];
    D[0] = stencil->mStencil[0][0][0] + D[0] * w;
    D[1] = stencil->mStencil[0][1][0] + D[1] * w;
    D[2] = stencil->mStencil[1][0][0] + D[2] * w;
    D[3] = stencil->mStencil[1][1][0] + D[3] * w;

    ret->mVec[0] =   cnanovdb_lerpF(D[2], D[3], fraction.mVec[1])
                   - cnanovdb_lerpF(D[0], D[1], fraction.mVec[1]);

    ret->mVec[1] = cnanovdb_lerpF(D[1] - D[0], D[3] - D[2], fraction.mVec[0]);
}


#endif
