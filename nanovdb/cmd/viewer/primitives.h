// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file primitives.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief primitives math functions.
*/

#include <nanovdb/NanoVDB.h>

namespace viewer {
namespace primitives {

nanovdb::Vec4f mandelbulb(const nanovdb::Vec3f& p);

}
} // namespace viewer::primitives