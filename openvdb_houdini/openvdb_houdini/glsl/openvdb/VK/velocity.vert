// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// VDB velocity decoration vertex shader for Vulkan.
// Reads P (position) and V (velocity) as vertex inputs.
// Transforms position to view space. Velocity is passed through
// in view space for the tessellation stage to generate isolines.

layout(location = 0) in vec3 P;
layout(location = 1) in vec3 V;

layout(location = 0) out parms
{
    vec3 velocity;
} vsOut;

layout(set=0, binding=0)
#using glH_PassInfo

layout(set=1, binding=0)
#using glH_Object

void main()
{
    // position in view space (projection applied in TCS)
    gl_Position = glH_Object.ObjView * vec4(P, 1.0);

    // velocity direction in view space
    vsOut.velocity = mat3(glH_Object.ObjView) * V;
}
