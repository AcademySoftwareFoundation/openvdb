// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Minimal VDB Points vertex shader for Vulkan.
// Transforms position P by the standard Houdini view/projection matrices,
// passes Cd (color) to the fragment shader.
//
// Uses direct vertex inputs (not ATTRIB macros) because the geometry is
// populated with createAttribute() which creates VBOs, not SSBOs.

layout(location = 0) in vec3 P;
layout(location = 1) in vec3 Cd;

layout(location = 0) out vec4 pnt_color;

layout(set=0, binding=0)
#using glH_PassInfo

layout(set=1, binding=0)
#using glH_Object

void main()
{
    vec4 pos = glH_Object.ObjView * vec4(P, 1.0);
    gl_Position = glH_PassInfo.Projection * pos;
    gl_PointSize = glH_Object.DecorationScale;
    pnt_color = vec4(Cd, 1.0);
}
