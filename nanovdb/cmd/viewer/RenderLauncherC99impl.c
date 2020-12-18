// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLauncherC99impl.c

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implmentation of C99 Render launcher.
*/

#ifdef __cplusplus
extern "C" {
#endif

#define VALUETYPE float
#define SIZEOF_VALUETYPE 4
#include "AgnosticNanoVDB.h"
#include "code/renderCommon.h"
#include "code/renderLevelSet.c"
#include "code/renderFogVolume.c"

void launchRender(int method, int width, int height, vec4* imgPtr, const nanovdb_Node0_float* node0Level, const nanovdb_Node1_float* node1Level, const nanovdb_Node2_float* node2Level, const nanovdb_RootData_float* rootData, const nanovdb_RootData_Tile_float* rootDataTiles, const nanovdb_GridData* gridData, const ArgUniforms* uniforms)
{
    const int blockSize = 16;
    const int nBlocksX = (width + (blockSize - 1)) / blockSize;
    const int nBlocksY = (height + (blockSize - 1)) / blockSize;
    const int nBlocks = nBlocksX * nBlocksY;

    for (int blockIndex = 0; blockIndex < nBlocks; ++blockIndex) {
        const int blockOffsetX = blockSize * (blockIndex % nBlocksX);
        const int blockOffsetY = blockSize * (blockIndex / nBlocksY);

        if (method == CNANOVDB_RENDERMETHOD_LEVELSET) {
            for (int iy = 0; iy < blockSize; ++iy) {
                for (int ix = 0; ix < blockSize; ++ix) {
                    renderLevelSet(
                        CNANOVDB_MAKE_IVEC2(ix + blockOffsetX, iy + blockOffsetY),
                        imgPtr,
                        node0Level,
                        node1Level,
                        node2Level,
                        rootData,
                        rootDataTiles,
                        gridData,
                        *uniforms);
                }
            }
        } else if (method == CNANOVDB_RENDERMETHOD_FOG_VOLUME) {
            for (int iy = 0; iy < blockSize; ++iy) {
                for (int ix = 0; ix < blockSize; ++ix) {
                    renderFogVolume(
                        CNANOVDB_MAKE_IVEC2(ix + blockOffsetX, iy + blockOffsetY),
                        imgPtr,
                        node0Level,
                        node1Level,
                        node2Level,
                        rootData,
                        rootDataTiles,
                        gridData,
                        *uniforms);
                }
            }
        } else {
            for (int iy = 0; iy < blockSize; ++iy) {
                for (int ix = 0; ix < blockSize; ++ix) {
                    int gx = (blockOffsetX + ix);
                    int gy = (blockOffsetY + iy);
                    if (gx >= width || gy >= height)
                        continue;
                    *((imgPtr) + gx + gy * width) = CNANOVDB_MAKE_VEC4((float)gx / width, (float)gy / height, 1, 1);
                }
            }
        }
    }
}


#ifdef __cplusplus
}
#endif
