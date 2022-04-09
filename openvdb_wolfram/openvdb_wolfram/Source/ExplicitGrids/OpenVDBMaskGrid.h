#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>

#include "../OpenVDBGrid.h"

using OpenVDBMaskGrid = OpenVDBGrid<openvdb::ValueMask>;
