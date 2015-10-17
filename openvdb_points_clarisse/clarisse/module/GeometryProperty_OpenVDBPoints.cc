///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @file GeometryProperty_OpenVDBPoints.cc
///
/// @author Dan Bailey

#include <iostream>

#include <ctx_eval.h>
#include <geometry_intersection.h>

#include <core_platform.h>
#include <core_vector.h>
#include <gmath_bbox3.h>
#include <geometry_raytrace_ctx.h>
#include <ctx_eval.h>
#include <ctx_shader.h>

#include "GeometryProperty_OpenVDBPoints.h"
#include "ResourceData_OpenVDBPoints.h"

using namespace openvdb;
using namespace openvdb::tools;

IMPLEMENT_CLASS(GeometryProperty_OpenVDBPoints, GeometryProperty);

GeometryProperty_OpenVDBPoints::GeometryProperty_OpenVDBPoints( const ResourceData_OpenVDBPoints* geometry,
                                                                const std::string& name,
                                                                const std::string& type)
    : GeometryProperty(name.c_str())
    , m_geometry(geometry)
    , m_name(name)
    , m_type(type)
{
    if (m_type == "float")           set_evaluate_callbacks(extract_value_long, extract_value_float<float>);
    else if (m_type == "half")       set_evaluate_callbacks(extract_value_long, extract_value_float<half>);
    else if (m_type == "vec3s")      set_evaluate_callbacks(extract_value_long, extract_value_float<Vec3f>);
    else if (m_type == "vec3h")      set_evaluate_callbacks(extract_value_long, extract_value_float<math::Vec3<half> >);
}

GeometryProperty_OpenVDBPoints::~GeometryProperty_OpenVDBPoints()
{
}

const GeometryProperty_OpenVDBPoints::PointDataLeaf* GeometryProperty_OpenVDBPoints::leaf(const unsigned int id) const
{
    return m_geometry->leaf(id);
}

unsigned int
GeometryProperty_OpenVDBPoints::extract_value_long( const GeometryProperty& property, const CtxEval& eval_ctx,
                                                    const GeometryFragment& fragment, const unsigned int& sample_index,
                                                    long long *values, long long *values_du, long long *values_dv, long long *values_dw,
                                                    const unsigned int& value_count)
{
    if (values) {
        for (unsigned i = 0; i < value_count; i++)   values[i] = 1;
    }

    return value_count;
}

namespace {

template <typename ValueType>
void setValues(double* values, const ValueType& value, const unsigned int& value_count)
{
    for (unsigned i = 0; i < value_count; i++)   values[i] = (double) value;
}

template <>
void setValues(double* values, const math::Vec3<half>& value, const unsigned int& value_count)
{
    for (unsigned i = 0; i < value_count; i++)   values[i] = (double) value[i];
}

template <>
void setValues(double* values, const Vec3f& value, const unsigned int& value_count)
{
    for (unsigned i = 0; i < value_count; i++)   values[i] = (double) value[i];
}

} // namespace

template<typename ValueType>
unsigned int
GeometryProperty_OpenVDBPoints::extract_value_float(const GeometryProperty& property, const CtxEval& eval_ctx,
                                                    const GeometryFragment& fragment, const unsigned int& sample_index,
                                                    double *values, double *values_du, double *values_dv, double *values_dw,
                                                    const unsigned int& value_count)
{
    const GeometryProperty_OpenVDBPoints& prop = (const GeometryProperty_OpenVDBPoints&)property;

    const unsigned int primitive_id = fragment.get_primitive_id();
    const unsigned int sub_primitive_id = fragment.get_sub_primitive_id();

    const PointDataLeaf* leaf = prop.leaf(primitive_id);

    if (!leaf)  return 0;

    const typename AttributeHandle<ValueType>::Ptr attributeHandle = leaf->attributeHandle<ValueType>(prop.name());

    const ValueType& value = attributeHandle->get(Index64(sub_primitive_id));

    setValues<ValueType>(values, value, value_count);

    return value_count;
}
