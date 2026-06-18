// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*
 * Copyright (c) Side Effects Software Inc.
 *
 * Produced by:
 *      Side Effects Software Inc
 *      477 Richmond Street West
 *      Toronto, Ontario
 *      Canada   M5V 3E7
 *      416-504-9876
 *
 * NAME:        UT_VDBUtils.h (UT Library, C++)
 *
 * COMMENTS:
 */

#include "UT_VDBUtils.h"

#include <UT/UT_StringHolder.h>
#include <UT/UT_FSATable.h>

namespace openvdb_houdini {
// empty
}

static const UT_FSATable *
getVDBTypeTable()
{
    static const UT_FSATable theOwnerTable(
        UT_VDB_INVALID,         "unknown",
        UT_VDB_FLOAT,           "float",
        UT_VDB_DOUBLE,          "double",
        UT_VDB_INT32,           "int32",
        UT_VDB_INT64,           "int64",
        UT_VDB_BOOL,            "bool",
        UT_VDB_VEC3F,           "Vec3f",
        UT_VDB_VEC3D,           "Vec3d",
        UT_VDB_VEC3I,           "Vec3i",
        UT_VDB_POINTINDEX,      "PointIndex",
        UT_VDB_POINTDATA,       "PointData"
        );

    return &theOwnerTable;
};

#define GET_NAMETOKEN(FNAME, TYPE, TABLE, FAIL_ENUM) \
    const char *FNAME(TYPE type) { return TABLE()->getToken((int)type); } \
    TYPE FNAME(const char *name) { \
        int id = TABLE()->findSymbol(name); \
        return id >= 0 ? static_cast<TYPE>(id) : FAIL_ENUM; \
    } \
    /**/

GET_NAMETOKEN( UTvdbType, UT_VDBType, getVDBTypeTable, UT_VDB_INVALID )

size_t
UTformatBuffer(char *buffer, size_t buffer_size, const UT_VDBType &v)
{
    UT::Format::Writer writer(buffer, buffer_size);
    UT::Format::Formatter f;
    return f.format(writer, "{0}", {UTvdbType(v)});
}

