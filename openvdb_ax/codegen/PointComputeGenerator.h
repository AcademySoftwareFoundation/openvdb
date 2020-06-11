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

/// @file codegen/PointComputeGenerator.h
///
/// @authors Nick Avramoussis, Matt Warner, Francisco Gochez, Richard Jones
///
/// @brief  The visitor framework and function definition for point data
///   grid code generation
///

#ifndef OPENVDB_AX_POINT_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED
#define OPENVDB_AX_POINT_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

#include "ComputeGenerator.h"
#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"

#include <openvdb_ax/compiler/AttributeRegistry.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  The function definition and signature which is built by the
///         PointComputeGenerator.
///
///         The argument structure is as follows:
///
///           1) - A void pointer to the CustomData
///           2) - A void pointer to the leaf AttributeSet
///           3) - An unsigned integer, representing the leaf relative point
///                id being executed
///           4) - A void pointer to a vector of void pointers, representing an
///                array of attribute handles
///           5) - A void pointer to a vector of void pointers, representing an
///                array of group handles
///           6) - A void pointer to a LeafLocalData object, used to track newly
///                initialized attributes and arrays
///
struct PointKernel
{
    /// The signature of the generated function
    using Signature =
        void(const void* const,
             const void* const,
             uint64_t,
             void**,
             void**,
             void*);

    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    /// The argument key names available during code generation
    static const std::array<std::string, N_ARGS>& argumentKeys();
    static std::string getDefaultName();
};

/// @brief  An additonal function built by the PointComputeGenerator.
///         Currently both compute and compute range functions have the same
///         signature
struct PointRangeKernel : public PointKernel
{
    static std::string getDefaultName();
};


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


/// @brief Visitor object which will generate llvm IR for a syntax tree which has been generated from
///        AX that targets point grids.  The IR will represent  2 functions : one that executes over
///        single points and one that executes over a collection of points.  This is primarily used by the
///        Compiler class.
struct PointComputeGenerator : public ComputeGenerator
{
    /// @brief Constructor
    /// @param module           llvm Module for generating IR
    /// @param options          Options for the function registry behaviour
    /// @param functionRegistry Function registry object which will be used when generating IR
    ///                         for function calls
    /// @param warnings         Vector which will hold compiler warnings.  If null, no warnings will
    ///                         be stored.
    PointComputeGenerator(llvm::Module& module,
       const FunctionOptions& options,
       FunctionRegistry& functionRegistry,
       std::vector<std::string>* const warnings = nullptr);

    ~PointComputeGenerator() override = default;

    using ComputeGenerator::traverse;
    using ComputeGenerator::visit;

    AttributeRegistry::Ptr generate(const ast::Tree& node);
    bool visit(const ast::Attribute*) override;

private:
    llvm::Value* attributeHandleFromToken(const std::string&);
    void getAttributeValue(const std::string& globalName, llvm::Value* location);
};

}
}
}
}

#endif // OPENVDB_AX_POINT_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
