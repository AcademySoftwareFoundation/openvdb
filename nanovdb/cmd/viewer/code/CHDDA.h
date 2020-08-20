// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////

CNANOVDB_DECLARE_STRUCT_BEGIN(HDDA)
    int32_t mDim;
    float   mT0, mT1; // min and max allowed times
    ivec3   mVoxel, mStep; // current voxel location and step to next voxel location
    vec3    mDelta, mNext; // delta time and next time
CNANOVDB_DECLARE_STRUCT_END(HDDA)

CNANOVDB_CONSTANT_MEM float DeltaFloat = 0.0001f;
CNANOVDB_CONSTANT_MEM float MaxFloat = 9999999.f;
CNANOVDB_CONSTANT_MEM int MinIndex_hashTable[8] = {2, 1, 9, 1, 2, 9, 0, 0}; //9 are dummy values

CNANOVDB_INLINE int MinIndex(vec3 v)
{
    // first expression is faster in OpenCL, but second is faster in GLSL. 
    // TODO: investigate.
#if 1
    return (v.y < v.x) ? (v.z < v.y ? 2 : 1) : (v.z < v.x ? 2 : 0);
#else
    const int        hashKey = ((int)(v.x < v.y) << 2) + ((int)(v.x < v.z) << 1) + (int)(v.y < v.z); // ?*4+?*2+?*1
    return MinIndex_hashTable[hashKey];
#endif
}
////////////////////////////////////////////////////////