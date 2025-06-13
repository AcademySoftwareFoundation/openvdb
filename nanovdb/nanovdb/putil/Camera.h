// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/putil/Camera.h

    \author Andrew Reidmeyer

    \brief  This file contains a simple camera implementation.
*/

#ifndef NANOVDB_PUTILS_CAMERA_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_CAMERA_H_HAS_BEEN_INCLUDED

#include "nanovdb/PNanoVDB.h"

#if defined(PNANOVDB_C)
#include <math.h>
#define pnanovdb_camera_sqrt sqrtf
#else
#define pnanovdb_camera_sqrt sqrt
#endif

#define PNANOVDB_CAMERA_INFINITY ((float)(1e+300 * 1e+300))

struct pnanovdb_camera_mat_t
{
    pnanovdb_vec4_t x;
    pnanovdb_vec4_t y;
    pnanovdb_vec4_t z;
    pnanovdb_vec4_t w;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_camera_mat_t)

#define pnanovdb_camera_action_t pnanovdb_uint32_t

#define PNANOVDB_CAMERA_ACTION_UNKNOWN 0
#define PNANOVDB_CAMERA_ACTION_DOWN 1
#define PNANOVDB_CAMERA_ACTION_UP 2

#define pnanovdb_camera_mouse_button_t pnanovdb_uint32_t

#define PNANOVDB_CAMERA_MOUSE_BUTTON_UNKNOWN 0
#define PNANOVDB_CAMERA_MOUSE_BUTTON_LEFT 1
#define PNANOVDB_CAMERA_MOUSE_BUTTON_MIDDLE 2
#define PNANOVDB_CAMERA_MOUSE_BUTTON_RIGHT 3

#define pnanovdb_camera_key_t pnanovdb_uint32_t

#define PNANOVDB_CAMERA_KEY_UNKNOWN 0
#define PNANOVDB_CAMERA_KEY_UP 1
#define PNANOVDB_CAMERA_KEY_DOWN 2
#define PNANOVDB_CAMERA_KEY_LEFT 3
#define PNANOVDB_CAMERA_KEY_RIGHT 4

struct pnanovdb_camera_state_t
{
    pnanovdb_vec3_t position;
    pnanovdb_vec3_t eye_direction;
    pnanovdb_vec3_t eye_up;
    float eye_distance_from_position;
    float orthographic_scale;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_camera_state_t)

struct pnanovdb_camera_config_t
{
    pnanovdb_bool_t is_projection_rh;
    pnanovdb_bool_t is_orthographic;
    pnanovdb_bool_t is_reverse_z;
    float near_plane;
    float far_plane;
    float fov_angle_y;
    float orthographic_y;
    float pan_rate;
    float tilt_rate;
    float zoom_rate;
    float key_translation_rate;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_camera_config_t)

struct pnanovdb_camera_t
{
    pnanovdb_camera_config_t config;
    pnanovdb_camera_state_t state;

    int mouse_x_prev;
    int mouse_y_prev;
    pnanovdb_bool_t rotation_active;
    pnanovdb_bool_t zoom_active;
    pnanovdb_bool_t translate_active;
    pnanovdb_uint32_t key_translate_active_mask;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_camera_t)

PNANOVDB_FORCE_INLINE void pnanovdb_camera_config_default(PNANOVDB_INOUT(pnanovdb_camera_config_t) ptr)
{
    PNANOVDB_DEREF(ptr).is_projection_rh = PNANOVDB_TRUE;
    PNANOVDB_DEREF(ptr).is_orthographic = PNANOVDB_FALSE;
    PNANOVDB_DEREF(ptr).is_reverse_z = PNANOVDB_TRUE;
    PNANOVDB_DEREF(ptr).near_plane = 0.1f;
    PNANOVDB_DEREF(ptr).far_plane = PNANOVDB_CAMERA_INFINITY;
    PNANOVDB_DEREF(ptr).fov_angle_y = 3.14159f / 4.f;
    PNANOVDB_DEREF(ptr).orthographic_y = 500.f;
    PNANOVDB_DEREF(ptr).pan_rate = 1.f;
    PNANOVDB_DEREF(ptr).tilt_rate = 1.f;
    PNANOVDB_DEREF(ptr).zoom_rate = 1.f;
    PNANOVDB_DEREF(ptr).key_translation_rate = 800.f;
}

PNANOVDB_FORCE_INLINE void pnanovdb_camera_state_default(PNANOVDB_INOUT(pnanovdb_camera_state_t) ptr, pnanovdb_bool_t y_up)
{
    PNANOVDB_DEREF(ptr).position.x = 0.f;
    PNANOVDB_DEREF(ptr).position.y = 0.f;
    PNANOVDB_DEREF(ptr).position.z = 0.f;
    if (y_up)
    {
        PNANOVDB_DEREF(ptr).eye_direction.x = 0.f;
        PNANOVDB_DEREF(ptr).eye_direction.y = 0.f;
        PNANOVDB_DEREF(ptr).eye_direction.z = 1.f;
        PNANOVDB_DEREF(ptr).eye_up.x = 0.f;
        PNANOVDB_DEREF(ptr).eye_up.y = 1.f;
        PNANOVDB_DEREF(ptr).eye_up.z = 0.f;
    }
    else
    {
        PNANOVDB_DEREF(ptr).eye_direction.x = 0.f;
        PNANOVDB_DEREF(ptr).eye_direction.y = 1.f;
        PNANOVDB_DEREF(ptr).eye_direction.z = 0.f;
        PNANOVDB_DEREF(ptr).eye_up.x = 0.f;
        PNANOVDB_DEREF(ptr).eye_up.y = 0.f;
        PNANOVDB_DEREF(ptr).eye_up.z = 1.f;
    }
    PNANOVDB_DEREF(ptr).eye_distance_from_position = -700.f;
    PNANOVDB_DEREF(ptr).orthographic_scale = 1.f;
}

PNANOVDB_FORCE_INLINE void pnanovdb_camera_init(PNANOVDB_INOUT(pnanovdb_camera_t) ptr)
{
    pnanovdb_camera_config_default(PNANOVDB_REF(PNANOVDB_DEREF(ptr).config));
    pnanovdb_camera_state_default(PNANOVDB_REF(PNANOVDB_DEREF(ptr).state), PNANOVDB_FALSE);
    PNANOVDB_DEREF(ptr).mouse_x_prev = 0;
    PNANOVDB_DEREF(ptr).mouse_y_prev = 0;
    PNANOVDB_DEREF(ptr).rotation_active = PNANOVDB_FALSE;
    PNANOVDB_DEREF(ptr).zoom_active = PNANOVDB_FALSE;
    PNANOVDB_DEREF(ptr).translate_active = PNANOVDB_FALSE;
    PNANOVDB_DEREF(ptr).key_translate_active_mask = 0u;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_camera_vec3_normalize(const pnanovdb_vec3_t v)
{
    pnanovdb_vec3_t ret = v;
    float v_magn2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (v_magn2 != 0.f)
    {
        float scale = 1.f / pnanovdb_camera_sqrt(v_magn2);
        ret.x *= scale;
        ret.y *= scale;
        ret.z *= scale;
    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_camera_vec3_dot(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
    float v_dot = a.x * b.x + a.y * b.y + a.z * b.z;
    pnanovdb_vec3_t ret = { v_dot, v_dot, v_dot };
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_camera_vec3_cross(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
    pnanovdb_vec3_t ret = {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x };
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec4_t pnanovdb_camera_mat_mul_row(const pnanovdb_camera_mat_t b, const pnanovdb_vec4_t r)
{
    pnanovdb_vec4_t res;
    res.x = b.x.x * r.x + b.y.x * r.y + b.z.x * r.z + b.w.x * r.w;
    res.y = b.x.y * r.x + b.y.y * r.y + b.z.y * r.z + b.w.y * r.w;
    res.z = b.x.z * r.x + b.y.z * r.y + b.z.z * r.z + b.w.z * r.w;
    res.w = b.x.w * r.x + b.y.w * r.y + b.z.w * r.z + b.w.w * r.w;
    return res;
}

PNANOVDB_FORCE_INLINE pnanovdb_camera_mat_t pnanovdb_camera_mat_mul(const pnanovdb_camera_mat_t a, const pnanovdb_camera_mat_t b)
{
    pnanovdb_camera_mat_t res;
    res.x = pnanovdb_camera_mat_mul_row(b, a.x);
    res.y = pnanovdb_camera_mat_mul_row(b, a.y);
    res.z = pnanovdb_camera_mat_mul_row(b, a.z);
    res.w = pnanovdb_camera_mat_mul_row(b, a.w);
    return res;
}

PNANOVDB_FORCE_INLINE pnanovdb_camera_mat_t pnanovdb_camera_mat_transpose(const pnanovdb_camera_mat_t a)
{
    pnanovdb_camera_mat_t ret = {
        {a.x.x, a.y.x, a.z.x, a.w.x},
        {a.x.y, a.y.y, a.z.y, a.w.y},
        {a.x.z, a.y.z, a.z.z, a.w.z},
        {a.x.w, a.y.w, a.z.w, a.w.w}
    };
    return ret;
}

PNANOVDB_FORCE_INLINE void pnanovdb_camera_compute_rotation_basis(
    PNANOVDB_INOUT(pnanovdb_camera_t) ptr,
    PNANOVDB_INOUT(pnanovdb_vec3_t) out_x_axis,
    PNANOVDB_INOUT(pnanovdb_vec3_t) out_y_axis,
    PNANOVDB_INOUT(pnanovdb_vec3_t) out_z_axis)
{
    pnanovdb_vec3_t z_axis = pnanovdb_camera_vec3_normalize(PNANOVDB_DEREF(ptr).state.eye_direction);
    // RH Z is negative going into screen, so reverse eye_direction_vector, after building basis
    if (PNANOVDB_DEREF(ptr).config.is_projection_rh)
    {
        z_axis.x = -z_axis.x;
        z_axis.y = -z_axis.y;
        z_axis.z = -z_axis.z;
    }
    pnanovdb_vec3_t y_axis = PNANOVDB_DEREF(ptr).state.eye_up;

    // force yAxis to orthogonal
    pnanovdb_vec3_t vec_proj = pnanovdb_vec3_sub(
        y_axis,
        pnanovdb_vec3_mul(pnanovdb_camera_vec3_dot(z_axis, y_axis), z_axis)
    );
    y_axis = pnanovdb_camera_vec3_normalize(vec_proj);

    // generate third basis vector
    pnanovdb_vec3_t x_axis = pnanovdb_camera_vec3_cross(y_axis, z_axis);

    PNANOVDB_DEREF(out_x_axis) = x_axis;
    PNANOVDB_DEREF(out_y_axis) = y_axis;
    PNANOVDB_DEREF(out_z_axis) = z_axis;
}

PNANOVDB_FORCE_INLINE pnanovdb_camera_mat_t pnanovdb_camera_matrix_rotation_normal(
    pnanovdb_vec3_t normal, float angle)
{
    float sin_angle = sinf(angle);
    float cos_angle = cosf(angle);

    pnanovdb_vec3_t a = { sin_angle, cos_angle, 1.f - cos_angle };

    pnanovdb_vec3_t c2 = pnanovdb_vec3_uniform(a.z);
    pnanovdb_vec3_t c1 = pnanovdb_vec3_uniform(a.y);
    pnanovdb_vec3_t c0 = pnanovdb_vec3_uniform(a.x);

    pnanovdb_vec3_t n0 = { normal.y, normal.z, normal.x };
    pnanovdb_vec3_t n1 = { normal.z, normal.x, normal.y };

    pnanovdb_vec3_t v0 = pnanovdb_vec3_mul(c2, n0);
    v0 = pnanovdb_vec3_mul(v0, n1);

    pnanovdb_vec3_t r0 = pnanovdb_vec3_mul(c2, normal);
    r0 = pnanovdb_vec3_add(pnanovdb_vec3_mul(r0, normal), c1);

    pnanovdb_vec3_t r1 = pnanovdb_vec3_add(pnanovdb_vec3_mul(c0, normal), v0);
    pnanovdb_vec3_t r2 = pnanovdb_vec3_sub(v0, pnanovdb_vec3_mul(c0, normal));

    pnanovdb_camera_mat_t ret = {
        { r0.x, r1.z, r2.y, 0.f },
        { r2.z, r0.y, r1.x, 0.f },
        { r1.y, r2.x, r0.z, 0.f },
        { 0.f, 0.f, 0.f, 1.f }
    };
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_camera_mat_t pnanovdb_camera_mat_rotation_axis(pnanovdb_vec3_t axis, float angle)
{
    pnanovdb_vec3_t normal = pnanovdb_camera_vec3_normalize(axis);
    return pnanovdb_camera_matrix_rotation_normal(normal, angle);
}

PNANOVDB_FORCE_INLINE pnanovdb_vec4_t pnanovdb_camera_vec4_transform(const pnanovdb_vec4_t x, const pnanovdb_camera_mat_t A)
{
    pnanovdb_vec4_t ret = {
        A.x.x * x.x + A.y.x * x.y + A.z.x * x.z + A.w.x * x.w,
        A.x.y * x.x + A.y.y * x.y + A.z.y * x.z + A.w.y * x.w,
        A.x.z * x.x + A.y.z * x.y + A.z.z * x.z + A.w.z * x.w,
        A.x.w * x.x + A.y.w * x.y + A.z.w * x.z + A.w.w * x.w
    };
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_camera_mat_t pnanovdb_camera_mat_inverse(const pnanovdb_camera_mat_t m)
{
    float f = (float(1.0) /
        (m.x.x * m.y.y * m.z.z * m.w.w +
            m.x.x * m.y.z * m.z.w * m.w.y +
            m.x.x * m.y.w * m.z.y * m.w.z +
            m.x.y * m.y.x * m.z.w * m.w.z +
            m.x.y * m.y.z * m.z.x * m.w.w +
            m.x.y * m.y.w * m.z.z * m.w.x +
            m.x.z * m.y.x * m.z.y * m.w.w +
            m.x.z * m.y.y * m.z.w * m.w.x +
            m.x.z * m.y.w * m.z.x * m.w.y +
            m.x.w * m.y.x * m.z.z * m.w.y +
            m.x.w * m.y.y * m.z.x * m.w.z +
            m.x.w * m.y.z * m.z.y * m.w.x +
            -m.x.x * m.y.y * m.z.w * m.w.z +
            -m.x.x * m.y.z * m.z.y * m.w.w +
            -m.x.x * m.y.w * m.z.z * m.w.y +
            -m.x.y * m.y.x * m.z.z * m.w.w +
            -m.x.y * m.y.z * m.z.w * m.w.x +
            -m.x.y * m.y.w * m.z.x * m.w.z +
            -m.x.z * m.y.x * m.z.w * m.w.y +
            -m.x.z * m.y.y * m.z.x * m.w.w +
            -m.x.z * m.y.w * m.z.y * m.w.x +
            -m.x.w * m.y.x * m.z.y * m.w.z +
            -m.x.w * m.y.y * m.z.z * m.w.x +
            -m.x.w * m.y.z * m.z.x * m.w.y));

    float a00 = (m.y.y * m.z.z * m.w.w +
        m.y.z * m.z.w * m.w.y +
        m.y.w * m.z.y * m.w.z +
        -m.y.y * m.z.w * m.w.z +
        -m.y.z * m.z.y * m.w.w +
        -m.y.w * m.z.z * m.w.y);

    float a10 = (m.x.y * m.z.w * m.w.z +
        m.x.z * m.z.y * m.w.w +
        m.x.w * m.z.z * m.w.y +
        -m.x.y * m.z.z * m.w.w +
        -m.x.z * m.z.w * m.w.y +
        -m.x.w * m.z.y * m.w.z);

    float a20 = (m.x.y * m.y.z * m.w.w +
        m.x.z * m.y.w * m.w.y +
        m.x.w * m.y.y * m.w.z +
        -m.x.y * m.y.w * m.w.z +
        -m.x.z * m.y.y * m.w.w +
        -m.x.w * m.y.z * m.w.y);

    float a30 = (m.x.y * m.y.w * m.z.z +
        m.x.z * m.y.y * m.z.w +
        m.x.w * m.y.z * m.z.y +
        -m.x.y * m.y.z * m.z.w +
        -m.x.z * m.y.w * m.z.y +
        -m.x.w * m.y.y * m.z.z);

    float a01 = (m.y.x * m.z.w * m.w.z +
        m.y.z * m.z.x * m.w.w +
        m.y.w * m.z.z * m.w.x +
        -m.y.x * m.z.z * m.w.w +
        -m.y.z * m.z.w * m.w.x +
        -m.y.w * m.z.x * m.w.z);

    float a11 = (m.x.x * m.z.z * m.w.w +
        m.x.z * m.z.w * m.w.x +
        m.x.w * m.z.x * m.w.z +
        -m.x.x * m.z.w * m.w.z +
        -m.x.z * m.z.x * m.w.w +
        -m.x.w * m.z.z * m.w.x);

    float a21 = (m.x.x * m.y.w * m.w.z +
        m.x.z * m.y.x * m.w.w +
        m.x.w * m.y.z * m.w.x +
        -m.x.x * m.y.z * m.w.w +
        -m.x.z * m.y.w * m.w.x +
        -m.x.w * m.y.x * m.w.z);

    float a31 = (m.x.x * m.y.z * m.z.w +
        m.x.z * m.y.w * m.z.x +
        m.x.w * m.y.x * m.z.z +
        -m.x.x * m.y.w * m.z.z +
        -m.x.z * m.y.x * m.z.w +
        -m.x.w * m.y.z * m.z.x);

    float a02 = (m.y.x * m.z.y * m.w.w +
        m.y.y * m.z.w * m.w.x +
        m.y.w * m.z.x * m.w.y +
        -m.y.x * m.z.w * m.w.y +
        -m.y.y * m.z.x * m.w.w +
        -m.y.w * m.z.y * m.w.x);

    float a12 = (-m.x.x * m.z.y * m.w.w +
        -m.x.y * m.z.w * m.w.x +
        -m.x.w * m.z.x * m.w.y +
        m.x.x * m.z.w * m.w.y +
        m.x.y * m.z.x * m.w.w +
        m.x.w * m.z.y * m.w.x);

    float a22 = (m.x.x * m.y.y * m.w.w +
        m.x.y * m.y.w * m.w.x +
        m.x.w * m.y.x * m.w.y +
        -m.x.x * m.y.w * m.w.y +
        -m.x.y * m.y.x * m.w.w +
        -m.x.w * m.y.y * m.w.x);

    float a32 = (m.x.x * m.y.w * m.z.y +
        m.x.y * m.y.x * m.z.w +
        m.x.w * m.y.y * m.z.x +
        -m.x.y * m.y.w * m.z.x +
        -m.x.w * m.y.x * m.z.y +
        -m.x.x * m.y.y * m.z.w);

    float a03 = (m.y.x * m.z.z * m.w.y +
        m.y.y * m.z.x * m.w.z +
        m.y.z * m.z.y * m.w.x +
        -m.y.x * m.z.y * m.w.z +
        -m.y.y * m.z.z * m.w.x +
        -m.y.z * m.z.x * m.w.y);

    float a13 = (m.x.x * m.z.y * m.w.z +
        m.x.y * m.z.z * m.w.x +
        m.x.z * m.z.x * m.w.y +
        -m.x.x * m.z.z * m.w.y +
        -m.x.y * m.z.x * m.w.z +
        -m.x.z * m.z.y * m.w.x);

    float a23 = (m.x.x * m.y.z * m.w.y +
        m.x.y * m.y.x * m.w.z +
        m.x.z * m.y.y * m.w.x +
        -m.x.x * m.y.y * m.w.z +
        -m.x.y * m.y.z * m.w.x +
        -m.x.z * m.y.x * m.w.y);

    float a33 = (m.x.x * m.y.y * m.z.z +
        m.x.y * m.y.z * m.z.x +
        m.x.z * m.y.x * m.z.y +
        -m.x.x * m.y.z * m.z.y +
        -m.x.y * m.y.x * m.z.z +
        -m.x.z * m.y.y * m.z.x);

    pnanovdb_camera_mat_t ret = {
        a00*f, a10*f, a20*f, a30*f,
        a01*f, a11*f, a21*f, a31*f,
        a02*f, a12*f, a22*f, a32*f,
        a03*f, a13*f, a23*f, a33*f };
    return ret;
}

PNANOVDB_FORCE_INLINE void pnanovdb_camera_get_view(PNANOVDB_INOUT(pnanovdb_camera_t) ptr, PNANOVDB_INOUT(pnanovdb_camera_mat_t) view)
{
    pnanovdb_vec3_t eye_position = PNANOVDB_DEREF(ptr).state.position;
    eye_position.x -= PNANOVDB_DEREF(ptr).state.eye_direction.x * PNANOVDB_DEREF(ptr).state.eye_distance_from_position;
    eye_position.y -= PNANOVDB_DEREF(ptr).state.eye_direction.y * PNANOVDB_DEREF(ptr).state.eye_distance_from_position;
    eye_position.z -= PNANOVDB_DEREF(ptr).state.eye_direction.z * PNANOVDB_DEREF(ptr).state.eye_distance_from_position;

    pnanovdb_camera_mat_t translate = {
        {1.f, 0.f, 0.f, 0.f},
        {0.f, 1.f, 0.f, 0.f},
        {0.f, 0.f, 1.f, 0.f},
        {eye_position.x, eye_position.y, eye_position.z, 1.f}
    };

    // derive rotation from eye_direction, eye_up vectors
    pnanovdb_vec3_t x_axis = {};
    pnanovdb_vec3_t y_axis = {};
    pnanovdb_vec3_t z_axis = {};
    pnanovdb_camera_compute_rotation_basis(ptr, PNANOVDB_REF(x_axis), PNANOVDB_REF(y_axis), PNANOVDB_REF(z_axis));

    pnanovdb_camera_mat_t rotation = pnanovdb_camera_mat_t{
        {x_axis.x, y_axis.x, z_axis.x, 0.f},
        {x_axis.y, y_axis.y, z_axis.y, 0.f},
        {x_axis.z, y_axis.z, z_axis.z, 0.f},
        {0.f, 0.f, 0.f, 1.f}
    };

    float scale_k = 1.f;
    if (PNANOVDB_DEREF(ptr).config.is_orthographic)
    {
        scale_k = PNANOVDB_DEREF(ptr).state.orthographic_scale;
    }
    pnanovdb_camera_mat_t scale = {
        scale_k, 0.f, 0.f, 0.f,
        0.f, scale_k, 0.f, 0.f,
        0.f, 0.f, scale_k, 0.f,
        0.f, 0.f, 0.f, 1.f
    };

    PNANOVDB_DEREF(view) = pnanovdb_camera_mat_mul(pnanovdb_camera_mat_mul(translate, rotation), scale);
}

PNANOVDB_FORCE_INLINE void pnanovdb_camera_get_projection(
    PNANOVDB_INOUT(pnanovdb_camera_t) ptr,
    PNANOVDB_INOUT(pnanovdb_camera_mat_t) out_projection,
    float aspect_width,
    float aspect_height)
{
    float aspect_ratio = aspect_width / aspect_height;
    float d0_z = PNANOVDB_DEREF(ptr).config.is_reverse_z ? PNANOVDB_DEREF(ptr).config.far_plane : PNANOVDB_DEREF(ptr).config.near_plane;
    float d1_z = PNANOVDB_DEREF(ptr).config.is_reverse_z ? PNANOVDB_DEREF(ptr).config.near_plane : PNANOVDB_DEREF(ptr).config.far_plane;

    pnanovdb_camera_mat_t projection = {};
    if (PNANOVDB_DEREF(ptr).config.is_orthographic)
    {
        float width = PNANOVDB_DEREF(ptr).config.orthographic_y * aspect_ratio;
        float height = PNANOVDB_DEREF(ptr).config.orthographic_y;
        float z_sign = PNANOVDB_DEREF(ptr).config.is_projection_rh ? -1.f : 1.f;
        float frange = z_sign * 1.f / (d1_z - d0_z);

        pnanovdb_camera_mat_t projection = {
            { 2.f / width, 0.f, 0.f, 0.f },
            { 0.f, 2.f / height, 0.f, 0.f },
            { 0.f, 0.f, frange, 0.f },
            { 0.f, 0.f, -z_sign * frange * d0_z, 1.f }
        };
        PNANOVDB_DEREF(out_projection) = projection;
    }
    else
    {
        float sinfov = sinf(0.5f * PNANOVDB_DEREF(ptr).config.fov_angle_y);
        float cosfov = cosf(0.5f * PNANOVDB_DEREF(ptr).config.fov_angle_y);

        float height = cosfov / sinfov;
        float width = height / aspect_ratio;
        float z_sign = PNANOVDB_DEREF(ptr).config.is_projection_rh ? -1.f : 1.f;
        float frange = z_sign * d1_z / (d1_z - d0_z);

        if (d0_z == PNANOVDB_CAMERA_INFINITY)
        {
            pnanovdb_camera_mat_t projection = {
                { width, 0.f, 0.f, 0.f },
                { 0.f, height, 0.f, 0.f },
                { 0.f, 0.f, frange, z_sign },
                { 0.f, 0.f, d1_z, 0.f }
            };
            PNANOVDB_DEREF(out_projection) = projection;
        }
        else
        {
            pnanovdb_camera_mat_t projection = {
                { width, 0.f, 0.f, 0.f },
                { 0.f, height, 0.f, 0.f },
                { 0.f, 0.f, frange, z_sign },
                { 0.f, 0.f, -z_sign * frange * d0_z, 0.f }
            };
            PNANOVDB_DEREF(out_projection) = projection;
        }
    }
}

PNANOVDB_FORCE_INLINE void pnanovdb_camera_mouse_update(
    PNANOVDB_INOUT(pnanovdb_camera_t) ptr,
    pnanovdb_camera_mouse_button_t button,
    pnanovdb_camera_action_t action,
    int mouse_x,
    int mouse_y,
    int winw,
    int    winh)
{
    // transient mouse state
    float rotation_dx = 0.f;
    float rotation_dy = 0.f;
    float translate_dx = 0.f;
    float translate_dy = 0.f;
    int translate_win_w = 1024;
    int translate_win_h = 1024;
    float zoom_dy = 0.f;

    // process event
    if (action == PNANOVDB_CAMERA_ACTION_DOWN)
    {
        if (button == PNANOVDB_CAMERA_MOUSE_BUTTON_LEFT)
        {
            PNANOVDB_DEREF(ptr).rotation_active = PNANOVDB_TRUE;
            rotation_dx = 0.f;
            rotation_dy = 0.f;
        }
        else if (button == PNANOVDB_CAMERA_MOUSE_BUTTON_MIDDLE)
        {
            PNANOVDB_DEREF(ptr).translate_active = PNANOVDB_TRUE;
            translate_dx = 0.f;
            translate_dy = 0.f;
        }
        else if (button == PNANOVDB_CAMERA_MOUSE_BUTTON_RIGHT)
        {
            PNANOVDB_DEREF(ptr).zoom_active = PNANOVDB_TRUE;
            zoom_dy = 0.f;
        }
    }
    else if (action == PNANOVDB_CAMERA_ACTION_UP)
    {
        if (button == PNANOVDB_CAMERA_MOUSE_BUTTON_LEFT)
        {
            PNANOVDB_DEREF(ptr).rotation_active = PNANOVDB_FALSE;
            rotation_dx = 0.f;
            rotation_dy = 0.f;
        }
        else if (button == PNANOVDB_CAMERA_MOUSE_BUTTON_MIDDLE)
        {
            PNANOVDB_DEREF(ptr).translate_active = PNANOVDB_FALSE;
            translate_dx = 0.f;
            translate_dy = 0.f;
        }
        else if (button == PNANOVDB_CAMERA_MOUSE_BUTTON_RIGHT)
        {
            PNANOVDB_DEREF(ptr).zoom_active = PNANOVDB_FALSE;
            zoom_dy = 0.f;
        }
    }
    else if (action == PNANOVDB_CAMERA_ACTION_UNKNOWN)
    {
        if (PNANOVDB_DEREF(ptr).rotation_active)
        {
            int dx = +(mouse_x - PNANOVDB_DEREF(ptr).mouse_x_prev);
            int dy = +(mouse_y - PNANOVDB_DEREF(ptr).mouse_y_prev);

            rotation_dx = float(dx) * 2.f * 3.14f / (winw);
            rotation_dy = float(dy) * 2.f * 3.14f / (winh);
        }
        if (PNANOVDB_DEREF(ptr).translate_active)
        {
            float dx = float(mouse_x - PNANOVDB_DEREF(ptr).mouse_x_prev);
            float dy = -float(mouse_y - PNANOVDB_DEREF(ptr).mouse_y_prev);

            translate_dx = dx * 2.f / (winw);
            translate_dy = dy * 2.f / (winh);

            translate_win_w = winw;
            translate_win_h = winh;
        }
        if (PNANOVDB_DEREF(ptr).zoom_active)
        {
            float dy = -float(mouse_y - PNANOVDB_DEREF(ptr).mouse_y_prev);

            zoom_dy = dy * 3.14f / float(winh);
        }
    }

    // keep current mouse position for next previous
    PNANOVDB_DEREF(ptr).mouse_x_prev = mouse_x;
    PNANOVDB_DEREF(ptr).mouse_y_prev = mouse_y;

    // apply rotation
    if (rotation_dx != 0.f || rotation_dy != 0.f)
    {
        float dx = rotation_dx;
        float dy = rotation_dy;

        if (PNANOVDB_DEREF(ptr).config.is_projection_rh)
        {
            dx = -dx;
            dy = -dy;
        }

        float rot_tilt = PNANOVDB_DEREF(ptr).config.tilt_rate * float(dy);
        float rot_pan = PNANOVDB_DEREF(ptr).config.pan_rate * float(dx);

        const float eye_dot_limit = 0.99f;

        // tilt
        {
            pnanovdb_vec3_t rot_vec = {};
            pnanovdb_vec3_t b = {};
            pnanovdb_vec3_t c = {};
            pnanovdb_camera_compute_rotation_basis(ptr, PNANOVDB_REF(rot_vec), PNANOVDB_REF(b), PNANOVDB_REF(c));

            const float angle = rot_tilt;
            pnanovdb_camera_mat_t dtilt = pnanovdb_camera_mat_rotation_axis(rot_vec, angle);

            pnanovdb_vec4_t eye_direction4 = {
                PNANOVDB_DEREF(ptr).state.eye_direction.x,
                PNANOVDB_DEREF(ptr).state.eye_direction.y,
                PNANOVDB_DEREF(ptr).state.eye_direction.z,
                0.f
            };
            eye_direction4 = pnanovdb_camera_vec4_transform(eye_direction4, dtilt);
            pnanovdb_vec3_t eye_direction3 = {
                eye_direction4.x,
                eye_direction4.y,
                eye_direction4.z
            };
            // make sure eye direction stays normalized
            eye_direction3 = pnanovdb_camera_vec3_normalize(eye_direction3);

            // check dot of eye_direction and eye_up, and avoid commit if value is very low
            float eye_dot = fabsf(
                eye_direction3.x * PNANOVDB_DEREF(ptr).state.eye_up.x +
                eye_direction3.y * PNANOVDB_DEREF(ptr).state.eye_up.y +
                eye_direction3.z * PNANOVDB_DEREF(ptr).state.eye_up.z
            );

            if (eye_dot < eye_dot_limit)
            {
                PNANOVDB_DEREF(ptr).state.eye_direction = eye_direction3;
            }
        }
        // pan
        {
            const float angle = rot_pan;
            pnanovdb_camera_mat_t dpan = pnanovdb_camera_mat_rotation_axis(PNANOVDB_DEREF(ptr).state.eye_up, angle);

            pnanovdb_vec4_t eye_direction4 = {
                PNANOVDB_DEREF(ptr).state.eye_direction.x,
                PNANOVDB_DEREF(ptr).state.eye_direction.y,
                PNANOVDB_DEREF(ptr).state.eye_direction.z,
                0.f
            };
            eye_direction4 = pnanovdb_camera_vec4_transform(eye_direction4, dpan);
            pnanovdb_vec3_t eye_direction3 = {
                eye_direction4.x,
                eye_direction4.y,
                eye_direction4.z
            };
            // make sure eye direction stays normalized
            eye_direction3 = pnanovdb_camera_vec3_normalize(eye_direction3);

            PNANOVDB_DEREF(ptr).state.eye_direction = eye_direction3;
        }
    }

    // apply translation
    if (translate_dx != 0.f || translate_dy != 0.f)
    {
        // goal here is to apply an NDC offset, to the position value in world space
        pnanovdb_camera_mat_t projection = {};
        pnanovdb_camera_get_projection(ptr, &projection, float(translate_win_w), float(translate_win_h));

        pnanovdb_camera_mat_t view = {};
        pnanovdb_camera_get_view(ptr, &view);

        // project position to NDC
        pnanovdb_vec4_t position_ndc = {
            PNANOVDB_DEREF(ptr).state.position.x,
            PNANOVDB_DEREF(ptr).state.position.y,
            PNANOVDB_DEREF(ptr).state.position.z,
            1.f
        };
        position_ndc = pnanovdb_camera_vec4_transform(position_ndc, view);
        position_ndc = pnanovdb_camera_vec4_transform(position_ndc, projection);

        // normalize
        if (position_ndc.w > 0.f)
        {
            position_ndc.x = position_ndc.x / position_ndc.w;
            position_ndc.y = position_ndc.y / position_ndc.w;
            position_ndc.z = position_ndc.z / position_ndc.w;
            position_ndc.w = 1.f;
        }

        // offset using mouse data
        position_ndc.x += translate_dx;
        position_ndc.y += translate_dy;

        // move back to world space
        pnanovdb_camera_mat_t proj_view_inverse = pnanovdb_camera_mat_inverse(
            pnanovdb_camera_mat_mul(view, projection)
        );

        pnanovdb_vec4_t position_world = pnanovdb_camera_vec4_transform(position_ndc, proj_view_inverse);

        // normalize
        if (position_world.w > 0.f)
        {
            position_world.x = position_world.x / position_world.w;
            position_world.y = position_world.y / position_world.w;
            position_world.z = position_world.z / position_world.w;
            position_world.w = 1.f;
        }

        // commit update
        PNANOVDB_DEREF(ptr).state.position.x = position_world.x;
        PNANOVDB_DEREF(ptr).state.position.y = position_world.y;
        PNANOVDB_DEREF(ptr).state.position.z = position_world.z;
    }

    // apply zoom
    if (zoom_dy != 0.f)
    {
        if (PNANOVDB_DEREF(ptr).config.is_orthographic)
        {
            PNANOVDB_DEREF(ptr).state.orthographic_scale *=
                (1.f - PNANOVDB_DEREF(ptr).config.zoom_rate * zoom_dy);
        }
        else
        {
            PNANOVDB_DEREF(ptr).state.eye_distance_from_position *=
                (1.f + PNANOVDB_DEREF(ptr).config.zoom_rate * zoom_dy);
        }
    }
}

PNANOVDB_FORCE_INLINE void pnanovdb_camera_key_update(PNANOVDB_INOUT(pnanovdb_camera_t) ptr, pnanovdb_camera_key_t key, pnanovdb_camera_action_t action)
{
    if (action == PNANOVDB_CAMERA_ACTION_DOWN)
    {
        if (key == PNANOVDB_CAMERA_KEY_UP)
        {
            PNANOVDB_DEREF(ptr).key_translate_active_mask |= 2u;
        }
        if (key == PNANOVDB_CAMERA_KEY_DOWN)
        {
            PNANOVDB_DEREF(ptr).key_translate_active_mask |= 4u;
        }
        if (key == PNANOVDB_CAMERA_KEY_LEFT)
        {
            PNANOVDB_DEREF(ptr).key_translate_active_mask |= 8u;
        }
        if (key == PNANOVDB_CAMERA_KEY_RIGHT)
        {
            PNANOVDB_DEREF(ptr).key_translate_active_mask |= 16u;
        }
    }
    else if (action == PNANOVDB_CAMERA_ACTION_UP)
    {
        if (key == PNANOVDB_CAMERA_KEY_UP)
        {
            PNANOVDB_DEREF(ptr).key_translate_active_mask &= ~2u;
        }
        if (key == PNANOVDB_CAMERA_KEY_DOWN)
        {
            PNANOVDB_DEREF(ptr).key_translate_active_mask &= ~4u;
        }
        if (key == PNANOVDB_CAMERA_KEY_LEFT)
        {
            PNANOVDB_DEREF(ptr).key_translate_active_mask &= ~8u;
        }
        if (key == PNANOVDB_CAMERA_KEY_RIGHT)
        {
            PNANOVDB_DEREF(ptr).key_translate_active_mask &= ~16u;
        }
    }
}

PNANOVDB_FORCE_INLINE void pnanovdb_camera_animation_tick(PNANOVDB_INOUT(pnanovdb_camera_t) ptr, float delta_time)
{
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;

    float rate = PNANOVDB_DEREF(ptr).config.key_translation_rate * delta_time;

    if (PNANOVDB_DEREF(ptr).key_translate_active_mask & 2u)
    {
        z += -rate;
    }
    if (PNANOVDB_DEREF(ptr).key_translate_active_mask & 4u)
    {
        z += +rate;
    }
    if (PNANOVDB_DEREF(ptr).key_translate_active_mask & 8)
    {
        x += +rate;
    }
    if (PNANOVDB_DEREF(ptr).key_translate_active_mask & 16)
    {
        x += -rate;
    }

    if (PNANOVDB_DEREF(ptr).key_translate_active_mask)
    {
        PNANOVDB_DEREF(ptr).state.position.x += PNANOVDB_DEREF(ptr).state.eye_direction.x * z;
        PNANOVDB_DEREF(ptr).state.position.y += PNANOVDB_DEREF(ptr).state.eye_direction.y * z;
        PNANOVDB_DEREF(ptr).state.position.z += PNANOVDB_DEREF(ptr).state.eye_direction.z * z;

        // compute xaxis
        pnanovdb_vec3_t x_axis{};
        pnanovdb_vec3_t y_axis{};
        pnanovdb_vec3_t z_axis{};
        pnanovdb_camera_compute_rotation_basis(ptr, PNANOVDB_REF(x_axis), PNANOVDB_REF(y_axis), PNANOVDB_REF(z_axis));

        PNANOVDB_DEREF(ptr).state.position.x += x_axis.x * x;
        PNANOVDB_DEREF(ptr).state.position.y += x_axis.y * x;
        PNANOVDB_DEREF(ptr).state.position.z += x_axis.z * x;
    }
}

#endif // end of NANOVDB_PUTILS_CAMERA_H_HAS_BEEN_INCLUDED
