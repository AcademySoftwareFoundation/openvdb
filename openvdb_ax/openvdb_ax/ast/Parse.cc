// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "Parse.h"
#include "../Exceptions.h"

// if OPENVDB_AX_REGENERATE_GRAMMAR is defined, we've re-generated the
// grammar - include path should be set up to pull in from the temp dir
// @note We need to include this to get access to axlloc. Should look to
//   re-work this so we don't have to (would require a reentrant parser)
#ifdef OPENVDB_AX_REGENERATE_GRAMMAR
#include "axparser.h"
#else
#include "../grammar/generated/axparser.h"
#endif

#include <openvdb/util/Assert.h>

#include <mutex>
#include <string>
#include <memory>

namespace {
inline std::mutex& GetInitMutex()
{
    static std::mutex sInitMutex;
    return sInitMutex;
}
}

openvdb::ax::Logger* axlog = nullptr;
using YY_BUFFER_STATE = struct yy_buffer_state*;
extern int axparse(openvdb::ax::ast::Tree**);
extern YY_BUFFER_STATE ax_scan_string(const char * str);
extern void ax_delete_buffer(YY_BUFFER_STATE buffer);
extern void axerror (openvdb::ax::ast::Tree**, char const *s) {
    //@todo: add check for memory exhaustion
    OPENVDB_ASSERT(axlog);
    axlog->error(/*starts with 'syntax error, '*/s + 14,
        {axlloc.first_line, axlloc.first_column});
}

openvdb::ax::ast::Tree::ConstPtr
openvdb::ax::ast::parse(const char* code,
    openvdb::ax::Logger& logger)
{
    std::lock_guard<std::mutex> lock(GetInitMutex());
    axlog = &logger; // for lexer errs
    logger.setSourceCode(code);

    const size_t err = logger.errors();

    // reset all locations
    axlloc.first_line = axlloc.last_line = 1;
    axlloc.first_column = axlloc.last_column = 1;

    YY_BUFFER_STATE buffer = ax_scan_string(code);

    openvdb::ax::ast::Tree* tree(nullptr);
    axparse(&tree);
    axlog = nullptr;

    openvdb::ax::ast::Tree::ConstPtr ptr(const_cast<const openvdb::ax::ast::Tree*>(tree));

    ax_delete_buffer(buffer);

    if (logger.errors() > err) ptr.reset();

    logger.setSourceTree(ptr);
    return ptr;
}


openvdb::ax::ast::Tree::Ptr
openvdb::ax::ast::parse(const char* code)
{
    openvdb::ax::Logger logger(
        [](const std::string& error) {
            OPENVDB_THROW(openvdb::AXSyntaxError, error);
        });

    openvdb::ax::ast::Tree::ConstPtr constTree = openvdb::ax::ast::parse(code, logger);

    return std::const_pointer_cast<openvdb::ax::ast::Tree>(constTree);
}

