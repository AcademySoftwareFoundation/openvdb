
# These are the types being used in openvdb
'''
mt::blocked_range
mt::blocked_range2d
mt::blocked_range3d
mt::combinable
mt::concurrent_hash_map
mt::concurrent_vector
mt::concurrent_vector::push_back
mt::enumerable_thread_specific
mt::parallel_for
mt::parallel_invoke
mt::parallel_reduce
mt::parallel_sort
mt::simple_partitioner
mt::spin_mutex
mt::spin_mutex::scoped_lock
mt::split
mt::task_arena
mt::task_arena::attach
mt::task::current_context
mt::task_group
mt::tasks
mt::task_scheduler_init
mt::task::self
mt::this_task_arena::max_concurrency
mt::tick_count
mt::tick_count::interval_t
mt::tick_count::now
'''

headers=\
"""
<tbb/blocked_range2d.h>
<tbb/blocked_range3d.h>
<tbb/blocked_range.h>
<tbb/combinable.h>
<tbb/concurrent_hash_map.h>
<tbb/concurrent_vector.h>
<tbb/enumerable_thread_specific.h>
<tbb/parallel_for.h>
<tbb/parallel_invoke.h>
<tbb/parallel_reduce.h>
<tbb/parallel_sort.h>
<tbb/partitioner.h>
<tbb/spin_mutex.h>
<tbb/task_arena.h>
<tbb/task_group.h>
<tbb/task.h>
<tbb/tbb.h>
<tbb/tick_count.h>
"""

template=\
'''
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file {filename}

#ifndef OPENVDB_{upper}_HAS_BEEN_INCLUDED
#define OPENVDB_{upper}_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <{include}>

namespace openvdb {{
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {{

namespace mt = ::tbb;

}} // OPENVDB_VERSION_NAME
}} // openvdb

#endif // OPENVDB_{upper}_HAS_BEEN_INCLUDED
'''

import os
import re

def make_files():
    info=re.compile("<(.*/(([a-zA-Z_0-9]+)\.h))>")

    for i in headers.split():
        match = info.match(i)
        if match:
            (include, filename, name) = match.groups()
            upper = name.upper()

            with open(filename, "w") as header:
                header.write(template.format(**locals()))
make_files()
