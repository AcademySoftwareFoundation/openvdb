///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
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

/// @file compiler/PointExecutable.h
///
/// @authors Nick Avramoussis, Francisco Gochez, Richard Jones
///
/// @brief The PointExecutable, produced by the OpenVDB AX Compiler for
///   execution over OpenVDB Points Grids.
///

#ifndef OPENVDB_AX_COMPILER_POINT_EXECUTABLE_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPILER_POINT_EXECUTABLE_HAS_BEEN_INCLUDED

#include <openvdb_ax/compiler/CustomData.h>
#include <openvdb_ax/compiler/AttributeRegistry.h>

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>

#include <unordered_map>

//forward
namespace llvm {

class ExecutionEngine;
class LLVMContext;

}

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {


/// @brief Object that encapsulates compiled AX code which can be executed on a target point grid
class PointExecutable
{
public:
    using Ptr = std::shared_ptr<PointExecutable>;

    /// @brief Constructor
    /// @param exeEngine Shared pointer to an llvm::ExecutionEngine object used to build functions.
    ///        context should be the associated llvm context
    /// @param context Shared pointer to an llvm:context object associated with exeEngine
    /// @param attributeRegistry Registry of point attributes accessed by AX code
    /// @param customData Custom data object which will be shared by this executable.  It can be
    ///        used to retrieve external data from within the AX code
    /// @param functions A map of function names to physical memory addresses which were built
    ///        by llvm using exeEngine
    /// @note  This object is normally be constructed by the Compiler::compile method, rather
    ///        than directly
    PointExecutable(const std::shared_ptr<const llvm::ExecutionEngine>& exeEngine,
                    const std::shared_ptr<const llvm::LLVMContext>& context,
                    const AttributeRegistry::ConstPtr& attributeRegistry,
                    const CustomData::ConstPtr& customData,
                    const std::unordered_map<std::string, uint64_t>& functions)
        : mContext(context)
        , mExecutionEngine(exeEngine)
        , mAttributeRegistry(attributeRegistry)
        , mCustomData(customData)
        , mFunctionAddresses(functions) {}

    ~PointExecutable() = default;

    /// @brief executes compiled AX code on target grid
    /// @param grid Grid to apply code to
    /// @param group Optional name of a group for filtering.  If this is not NULL,
    ///        the code will only be applied to points in this group
    /// @param createMissing If true, any attribute which has been used but does
    ///        not exist will be created. Otherwise, a missing attribute runtime
    ///        error will be thrown.
    void execute(points::PointDataGrid& grid,
                 const std::string* const group = nullptr,
                 const bool createMissing = true) const;

private:

    // The Context and ExecutionEngine must exist _only_ for object lifetime
    // management. The ExecutionEngine must be destroyed before the Context
    const std::shared_ptr<const llvm::LLVMContext> mContext;
    const std::shared_ptr<const llvm::ExecutionEngine> mExecutionEngine;
    const AttributeRegistry::ConstPtr mAttributeRegistry;
    const CustomData::ConstPtr mCustomData;
    // addresses of actual compiled code
    const std::unordered_map<std::string, uint64_t> mFunctionAddresses;
};

}
}
}

#endif // OPENVDB_AX_COMPILER_POINT_EXECUTABLE_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
