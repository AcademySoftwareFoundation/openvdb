// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file primitives.h
	\brief primitives math functions.
*/

#include <nanovdb/NanoVDB.h>

namespace viewer {
namespace primitives {

nanovdb::Vec4f mandelbulb(const nanovdb::Vec3f& p);

}
} // namespace viewer::primitives