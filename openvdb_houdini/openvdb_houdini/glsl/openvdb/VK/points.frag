// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// VDB Points fragment shader for Vulkan.
// Outputs the interpolated color from the vertex shader.
// Uses gl_PointCoord to discard fragments outside a circular disc
// with smooth anti-aliased edges.

layout(location = 0) in vec4 pnt_color;

layout(location = 0) out vec4 color_out;

void main()
{
    if (pnt_color.a == 0.0)
        discard;

    // circular point sprite with anti-aliased edge
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0)
        discard;

    // smooth edge over the outermost ~1 pixel
    float alpha = 1.0 - smoothstep(0.8, 1.0, r2);

    color_out = vec4(pnt_color.rgb, pnt_color.a * alpha);
}
