// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "ax.h"
#include "ast/AST.h"
#include "compiler/Compiler.h"
#include "compiler/PointExecutable.h"
#include "compiler/VolumeExecutable.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {

/// @note Implementation for initialize, isInitialized and unitialized
///       reamins in compiler/Compiler.cc

void run(const char* ax, openvdb::GridBase& grid)
{
    // Construct a generic compiler
    openvdb::ax::Compiler compiler;
    // Parse the provided code and produce an abstract syntax tree
    // @note  Throws with parser errors if invalid. Parsable code does not
    //        necessarily equate to compilable code
    const openvdb::ax::ast::Tree::ConstPtr ast = openvdb::ax::ast::parse(ax);

    if (grid.isType<points::PointDataGrid>()) {
        // Compile for Point support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::PointExecutable::Ptr exe =
            compiler.compile<openvdb::ax::PointExecutable>(*ast);
        // Execute on the provided points
        // @note  Throws on invalid point inputs such as mismatching types
        exe->execute(static_cast<points::PointDataGrid&>(grid));
    }
    else {
        // Compile for numerical grid support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::VolumeExecutable::Ptr exe =
            compiler.compile<openvdb::ax::VolumeExecutable>(*ast);
        // Execute on the provided numerical grid
        // @note  Throws on invalid grid inputs such as mismatching types
        exe->execute(grid);
    }
}

void run(const char* ax, openvdb::GridPtrVec& grids)
{
    if (grids.empty()) return;
    // Check the type of all grids. If they are all points, run for point data.
    // Otherwise, run for numerical volumes.
    bool points = true;
    for (auto& grid : grids) {
        if (!grid->isType<points::PointDataGrid>()) {
            points = false;
            break;
        }
    }

    // Construct a generic compiler
    openvdb::ax::Compiler compiler;
    // Parse the provided code and produce an abstract syntax tree
    // @note  Throws with parser errors if invalid. Parsable code does not
    //        necessarily equate to compilable code
    const openvdb::ax::ast::Tree::ConstPtr ast = openvdb::ax::ast::parse(ax);
    if (points) {
        // Compile for Point support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::PointExecutable::Ptr exe =
            compiler.compile<openvdb::ax::PointExecutable>(*ast);
        // Execute on the provided points individually
        // @note  Throws on invalid point inputs such as mismatching types
        for (auto& grid : grids) {
            exe->execute(static_cast<points::PointDataGrid&>(*grid));
        }
    }
    else {
        // Compile for Volume support and produce an executable
        // @note  Throws compiler errors on invalid code. On success, returns
        //        the executable which can be used multiple times on any inputs
        const openvdb::ax::VolumeExecutable::Ptr exe =
            compiler.compile<openvdb::ax::VolumeExecutable>(*ast);
        // Execute on the provided volumes
        // @note  Throws on invalid grid inputs such as mismatching types
        exe->execute(grids);
    }
}

}
}
}
