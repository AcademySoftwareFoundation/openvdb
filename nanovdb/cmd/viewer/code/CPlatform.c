// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////

#if defined(CNANOVDB_COMPILER_GLSL)
#line 7

layout(std430, binding = 0) restrict readonly buffer nanovdb_Node0_Block
{
    nanovdb_Node0_float nodes[];
}
kNodeLevel0;

layout(std430, binding = 1) restrict readonly buffer nanovdb_Node1_Block
{
    nanovdb_Node1_float nodes[];
}
kNodeLevel1;

layout(std430, binding = 2) restrict readonly buffer nanovdb_Node2_Block
{
    nanovdb_Node2_float nodes[];
}
kNodeLevel2;

layout(std430, binding = 3) readonly buffer nanovdb_Root_Block
{
    nanovdb_RootData_float      root;
    nanovdb_RootData_Tile_float tiles[];
}
kRootData;

layout(std430, binding = 4) readonly buffer nanovdb_Grid_Block
{
    nanovdb_GridData grid;
}
kGridData;

#else
struct nanovdb_Node0_Block
{
    CNANOVDB_GLOBAL const nanovdb_Node0_float* nodes;
};

struct nanovdb_Node1_Block
{
    CNANOVDB_GLOBAL const nanovdb_Node1_float* nodes;
};

struct nanovdb_Node2_Block
{
    CNANOVDB_GLOBAL const nanovdb_Node2_float* nodes;
};

struct nanovdb_Root_Block
{
    nanovdb_RootData_float root;
    CNANOVDB_GLOBAL const nanovdb_RootData_Tile_float* tiles;
};

struct nanovdb_Grid_Block
{
    nanovdb_GridData grid;
};

CNANOVDB_DECLARE_STRUCT_BEGIN(TreeContext)
    struct nanovdb_Grid_Block  mGridData;
    struct nanovdb_Root_Block  mRootData;
    struct nanovdb_Node2_Block mNodeLevel2;
    struct nanovdb_Node1_Block mNodeLevel1;
    struct nanovdb_Node0_Block mNodeLevel0;
CNANOVDB_DECLARE_STRUCT_END(TreeContext)

#endif

////////////////////////////////////////////////////////