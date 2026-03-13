// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// VDB velocity decoration fragment shader.
// Applies a fade effect along the velocity trail using the u parameter
// from the tessellation evaluation shader. Trail color comes from
// glH_Object.WireColor which is set by the C++ code to the trail color.

layout(location = 0) in float fsU;

layout(location = 0) out vec4 color_out;

layout(set=1, binding=0)
#using glH_Object

void main()
{
    vec4 col = glH_Object.WireColor;

    // cubic fade from opaque (tip) to transparent (tail)
    float a = 1.0 - fsU;
    a = 1.0 - a * a * a;
    a *= col.a;

    color_out = vec4(col.rgb, a);
}
