///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
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
/// @file SOP_NodeVDB.h
/// @author FX R&D OpenVDB team
/// @brief Base class for OpenVDB plugins

#ifndef OPENVDB_HOUDINI_SOP_NODEVDB_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_SOP_NODEVDB_HAS_BEEN_INCLUDED

#include <houdini_utils/ParmFactory.h>
#include <openvdb/openvdb.h>
#include <openvdb/Platform.h>
#include <SOP/SOP_Node.h>
#ifndef SESI_OPENVDB
#include <UT/UT_DSOVersion.h>
#endif
#include <UT/UT_Version.h>


class GU_Detail;

namespace openvdb_houdini {

/// @brief Use this class to register a new OpenVDB operator (SOP, POP, etc.)
/// @details This class ensures that the operator uses the appropriate OpPolicy.
/// @sa houdini_utils::OpFactory, houdini_utils::OpPolicy
class OPENVDB_HOUDINI_API OpenVDBOpFactory: public houdini_utils::OpFactory
{
public:
    /// Construct an OpFactory that on destruction registers a new OpenVDB operator type.
    OpenVDBOpFactory(const std::string& english, OP_Constructor, houdini_utils::ParmList&,
        OP_OperatorTable&, houdini_utils::OpFactory::OpFlavor = SOP);
};


class OPENVDB_HOUDINI_API DWAOpenVDBOpFactory: public OpenVDBOpFactory
{
public:
    /// @brief Construct an OpFactory that on destruction registers a new
    /// OpenVDB operator type using the DreamWorks naming convention.
    /// @deprecated Use OpenVDBOpFactory instead.
    OPENVDB_DEPRECATED
    DWAOpenVDBOpFactory(const std::string& english, OP_Constructor ctor,
        houdini_utils::ParmList& plist, OP_OperatorTable& optable,
        houdini_utils::OpFactory::OpFlavor flavor = SOP)
        : OpenVDBOpFactory(english, ctor, plist, optable, flavor)
    {
    }
};


////////////////////////////////////////


/// @brief Base class from which to derive OpenVDB-related Houdini SOPs
class OPENVDB_HOUDINI_API SOP_NodeVDB: public SOP_Node
{
public:
    SOP_NodeVDB(OP_Network*, const char*, OP_Operator*);
    virtual ~SOP_NodeVDB() {}

    virtual void fillInfoTreeNodeSpecific(UT_InfoTree&, fpreal time);
    virtual void getNodeSpecificInfoText(OP_Context&, OP_NodeInfoParms&);

protected:
    virtual OP_ERROR cookMyGuide1(OP_Context&);
    //virtual OP_ERROR cookMyGuide2(OP_Context&);

    /// @brief Retrieve a group from a geometry detail by parsing a pattern
    /// (typically, the value of a Group parameter belonging to this node).
    /// @throw std::runtime_error if the pattern is nonempty but doesn't match any group.
    /// @todo This is a wrapper for SOP_Node::parsePrimitiveGroups(), so it needs access
    /// to a SOP_Node instance.  But it probably doesn't need to be a SOP_NodeVDB method.
    const GA_PrimitiveGroup* matchGroup(GU_Detail&, const std::string& pattern);

    //@{
    /// @brief Evaluate a vector-valued parameter.
    openvdb::Vec3f evalVec3f(const char* name, fpreal time) const;
    openvdb::Vec3R evalVec3R(const char* name, fpreal time) const;
    openvdb::Vec3i evalVec3i(const char* name, fpreal time) const;
    openvdb::Vec2R evalVec2R(const char* name, fpreal time) const;
    openvdb::Vec2i evalVec2i(const char* name, fpreal time) const;
    //@}

    /// @brief Transfer the value of an obsolete parameter that was renamed
    /// to the parameter with the new name.
    /// @details This convenience method is intended to be called from
    /// @c resolveObsoleteParms(), when that function is implemented.
    void resolveRenamedParm(PRM_ParmList& obsoleteParms,
        const char* oldName, const char* newName);


    /// @brief Steal the geometry on the specified input if possible, instead of copying the data.
    ///
    /// @details In certain cases where a node's input geometry isn't being shared with
    /// other nodes, it is safe for the node to directly modify the geometry.
    /// Normally, input geometry is shared with the upstream node's output cache,
    /// so for stealing to be possible, the "unload" flag must be set on the upstream node
    /// to inhibit caching.  In addition, reference counting of GEO_PrimVDB shared pointers
    /// ensures we cannot steal data that is in use elsewhere.  When stealing is not possible,
    /// this method falls back to copying the shared pointer, effectively performing
    /// a duplicateSource().
    ///
    /// @param index    the index of the input from which to perform this operation
    /// @param context  the current SOP context is used for cook time for network traversal
    /// @param pgdp     pointer to the SOP's gdp
    /// @param gdh      handle to manage input locking
    /// @param clean    (forwarded to duplicateSource())
    ///
    /// @note Prior to Houdini 13.0, this method peforms a duplicateSource() and unlocks the
    /// inputs to the SOP. From Houdini 13.0 on, this method will insert the existing data
    /// into the detail and update the detail handle in the SOP.
    ///
    /// @warning No attempt to call duplicateSource() or inputGeo() should be made after
    /// calling this method, as there will be no data on the input stream if isSourceStealable()
    /// returns @c true.
    OP_ERROR duplicateSourceStealable(const unsigned index,
        OP_Context& context, GU_Detail **pgdp, GU_DetailHandle& gdh, bool clean = true);


    /// @brief Steal the geometry on the specified input if possible, instead of copying the data.
    ///
    /// @details In certain cases where a node's input geometry isn't being shared with
    /// other nodes, it is safe for the node to directly modify the geometry.
    /// Normally, input geometry is shared with the upstream node's output cache,
    /// so for stealing to be possible, the "unload" flag must be set on the upstream node
    /// to inhibit caching.  In addition, reference counting of GEO_PrimVDB shared pointers
    /// ensures we cannot steal data that is in use elsewhere.  When stealing is not possible,
    /// this method falls back to copying the shared pointer, effectively performing
    /// a duplicateSource().
    ///
    /// @note Prior to Houdini 13.0, this method peforms a duplicateSource() and unlocks the
    /// inputs to the SOP. From Houdini 13.0 on, this method will insert the existing data
    /// into the detail and update the detail handle in the SOP.
    ///
    /// @param index    the index of the input from which to perform this operation
    /// @param context  the current SOP context is used for cook time for network traversal
    OP_ERROR duplicateSourceStealable(const unsigned index, OP_Context& context);

private:

    /// @brief Traverse the upstream network to determine if the source input can be stolen.
    ///
    /// An upstream SOP cannot be stolen if it is implicitly caching the data (no "unload" flag)
    /// or explictly caching the data (using a Cache SOP)
    ///
    /// The traversal ignores pass through nodes such as null SOPs and bypassing.
    ///
    /// @param index    the index of the input from which to perform this operation
    /// @param context  the current SOP context is used for cook time for network traversal
    bool isSourceStealable(const unsigned index, OP_Context& context) const;
};


/// @brief Namespace to hold functionality for registering info text callbacks. Whenever
/// getNodeSpecificInfoText() is called, the default info text is added to MMB output unless
/// a valid callback has been registered for the grid type.
///
/// @details Use node_info_text::registerGridSpecificInfoText<> to register a grid type to
/// a function pointer which matches the ApplyGridSpecificInfoText signature.
///
///    void floatGridText(std::ostream&, const openvdb::GridBase&);
///
///    node_info_text::registerGridSpecificInfoText<openvdb::FloatGrid>(&floatGridText);
///
namespace node_info_text
{
    // The function pointer signature expected when registering an grid type text
    // callback. The grid is passed untyped but is guaranteed to match the registered
    // type.
    typedef void (*ApplyGridSpecificInfoText)(std::ostream&, const openvdb::GridBase&);

    /// @brief Register an info text callback to a specific grid type.
    /// @note Does not add the callback if the grid type already has a registered callback.
    /// @param gridType   the grid type as a unique string (see templated
    ///                   registerGridSpecificInfoText<>)
    /// @param callback   a pointer to the callback function to execute
    void registerGridSpecificInfoText(const std::string& gridType,
        ApplyGridSpecificInfoText callback);

    /// @brief Register an info text callback to a templated grid type.
    /// @note Does not add the callback if the grid type already has a registered callback.
    /// @param callback   a pointer to the callback function to execute
    template<typename GridType>
    inline void registerGridSpecificInfoText(ApplyGridSpecificInfoText callback)
    {
        registerGridSpecificInfoText(GridType::gridType(), callback);
    }

} // namespace node_info_text

} // namespace openvdb_houdini

#endif // OPENVDB_HOUDINI_SOP_NODEVDB_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
