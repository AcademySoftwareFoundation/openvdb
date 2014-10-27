///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
/// @file ParmFactory.h
/// @author FX R&D OpenVDB team
///
/// @brief A collection of factory methods and helper functions
/// to simplify Houdini plugin development and maintenance.

#ifndef HOUDINI_UTILS_PARM_FACTORY_HAS_BEEN_INCLUDED
#define HOUDINI_UTILS_PARM_FACTORY_HAS_BEEN_INCLUDED

#include <OP/OP_Operator.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_SpareData.h>
#include <SOP/SOP_Node.h>
#if defined(PRODDEV_BUILD) || defined(DWREAL_IS_DOUBLE)
  // OPENVDB_HOUDINI_API, which has no meaning in a DWA build environment but
  // must at least exist, is normally defined by including openvdb/Platform.h.
  // For DWA builds (i.e., if either PRODDEV_BUILD or DWREAL_IS_DOUBLE exists),
  // that introduces an unwanted and unnecessary library dependency.
  #ifndef OPENVDB_HOUDINI_API
    #define OPENVDB_HOUDINI_API
  #endif
#else
  #include <openvdb/Platform.h>
#endif
#include <boost/shared_ptr.hpp>
#include <map>
#include <string>
#include <vector>


#ifdef SESI_OPENVDB
    #ifdef OPENVDB_HOUDINI_API
	#undef OPENVDB_HOUDINI_API
	#define OPENVDB_HOUDINI_API
    #endif
#endif


class GU_Detail;
class OP_OperatorTable;

namespace houdini_utils {

class ParmFactory;


/// @brief Parameter template list that is always terminated.
class OPENVDB_HOUDINI_API ParmList
{
public:
    typedef std::vector<PRM_Template> PrmTemplateVec;

    ParmList() {}

    bool empty() const { return mParmVec.empty(); }
    size_t size() const { return mParmVec.size(); }

    void clear() { mParmVec.clear(); mSwitchers.clear(); }

    ParmList& add(const PRM_Template&);
    ParmList& add(const ParmFactory&);

    ParmList& beginSwitcher(const std::string& token, const std::string& label = "");
    ParmList& endSwitcher();

    ParmList& addFolder(const std::string& label);

    /// Return a heap-allocated copy of this list's array of parameters.
    PRM_Template* get() const;

private:
    struct SwitcherInfo { size_t parmIdx; std::vector<PRM_Default> folders; };
    typedef std::vector<SwitcherInfo> SwitcherStack;

    void incFolderParmCount();
    SwitcherInfo* getCurrentSwitcher();

    PrmTemplateVec mParmVec;
    SwitcherStack mSwitchers;
};


////////////////////////////////////////


/// @class ParmFactory
/// @brief Helper class to simplify construction of PRM_Templates and
/// dynamic user interfaces.
///
/// Usage example:
/// @code
/// houdini_utils::ParmList parms;
///
/// parms.add(houdini_utils::ParmFactory(PRM_STRING, "group", "Group")
///     .setHelpText("Specify a subset of the input VDB grids to be processed.")
///     .setChoiceList(&houdini_utils::PrimGroupMenu));
///
/// parms.add(houdini_utils::ParmFactory(PRM_FLT_J, "tolerance", "Pruning Tolerance")
///     .setDefault(PRMzeroDefaults)
///     .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1));
/// @endcode
class OPENVDB_HOUDINI_API ParmFactory
{
public:
    ParmFactory(PRM_Type, const std::string& token, const std::string& label);
    ParmFactory(PRM_MultiType, const std::string& token, const std::string& label);

    // Settings
    ParmFactory& setCallbackFunc(const PRM_Callback&);

    /// Specify a menu of values for this parameter.
    ParmFactory& setChoiceList(const PRM_ChoiceList*);
    /// @brief Specify a menu type and a list of token, label, token, label,... pairs
    /// for this parameter.
    /// @param typ     specifies the menu behavior (toggle, replace, etc.)
    /// @param items   a list of token, label, token, label,... string pairs
    ParmFactory& setChoiceListItems(PRM_ChoiceListType typ, const std::vector<std::string>& items);
    /// @brief Specify a menu type and a list of token, label, token, label,... pairs
    /// for this parameter.
    /// @param typ     specifies the menu behavior (toggle, replace, etc.)
    /// @param items   a list of token, label, token, label,... string pairs
    /// @note The @a items array must be null-terminated.
    ParmFactory& setChoiceListItems(PRM_ChoiceListType typ, const char* const* items);


#if defined(GCC3)
    #define IS_DEPRECATED __attribute__ ((deprecated))
#elif defined(_MSC_VER)
    #define IS_DEPRECATED __declspec(deprecated)
#else
    #define IS_DEPRECATED
#endif

    /// @brief Specify a menu type and either a list of menu item labels or a list of
    /// token, label, token, label,... pairs for this parameter.
    /// @param typ     specifies the menu behavior (toggle, replace, etc.)
    /// @param items   a list of menu item labels or token, label, token, label,... pairs
    /// @param paired  if @c false, treat all the elements of @a items as labels and assign
    ///     them numeric tokens starting from zero; otherwise, treat the elements of @a items
    ///     as token, label, token, label,... pairs
    /// @deprecated Use setChoiceListItems() instead.  Using unpaired items may mean
    /// less typing now, but it prevents you from reordering or deleting entries later.
    IS_DEPRECATED ParmFactory& setChoiceList(PRM_ChoiceListType typ,
        const std::vector<std::string>& items, bool paired = false);
    /// @brief Specify a menu type and either a list of menu item labels or a list of
    /// token, label, token, label,... pairs for this parameter.
    /// @param typ     specifies the menu behavior (toggle, replace, etc.)
    /// @param items   a list of menu item labels or token, label, token, label,... pairs
    /// @param paired  if @c false, treat all the elements of @a items as labels and assign
    ///     them numeric tokens starting from zero; otherwise, treat the elements of @a items
    ///     as token, label, token, label,... pairs
    /// @note The @a items array must be null-terminated.
    /// @deprecated Use setChoiceListItems() instead.  Using unpaired items may mean
    /// less typing now, but it prevents you from reordering or deleting entries later.
    IS_DEPRECATED ParmFactory& setChoiceList(PRM_ChoiceListType typ,
        const char* const* items, bool paired = false);

#undef IS_DEPRECATED

    ParmFactory& setConditional(const PRM_ConditionalBase*);

    /// @brief Specify a default value for this parameter.
    /// @details If the string is null, the floating-point value will be used
    /// (but rounded if this parameter is integer-valued).
    /// @note The string pointer must not point to a temporary.
    ParmFactory& setDefault(fpreal, const char* = NULL, CH_StringMeaning = CH_STRING_LITERAL);
    /// @brief Specify a default string value for this parameter.
    ParmFactory& setDefault(const std::string&, CH_StringMeaning = CH_STRING_LITERAL);
    /// @brief Specify default numeric values for the vector elements of this parameter
    /// (assuming its vector size is > 1).
    /// @details Floating-point values will be rounded if this parameter is integer-valued.
    ParmFactory& setDefault(const std::vector<fpreal>&);
    /// @brief Specify default values for the vector elements of this parameter
    /// (assuming its vector size is > 1).
    ParmFactory& setDefault(const std::vector<PRM_Default>&);
    /// Specify a default value or values for this parameter.
    ParmFactory& setDefault(const PRM_Default*);

    ParmFactory& setHelpText(const char*);

    ParmFactory& setParmGroup(int);

    /// Specify a range for this parameter's values.
    ParmFactory& setRange(
        PRM_RangeFlag minFlag, fpreal minVal,
        PRM_RangeFlag maxFlag, fpreal maxVal);
    /// @brief Specify range for the values of this parameter's vector elements
    /// (assuming its vector size is > 1).
    ParmFactory& setRange(const std::vector<PRM_Range>&);
    /// Specify a range or ranges for this parameter's values.
    ParmFactory& setRange(const PRM_Range*);

    /// Specify (@e key, @e value) pairs of spare data for this parameter.
    ParmFactory& setSpareData(const std::map<std::string, std::string>&);
    /// Specify spare data for this parameter.
    ParmFactory& setSpareData(const PRM_SpareData*);

    /// @brief Specify the list of parameters for each instance of a multiparm.
    /// @note This setting is ignored for non-multiparm parameters.
    /// @note Parameter name tokens should include a '#' character.
    ParmFactory& setMultiparms(const ParmList&);

    /// Specify an extended type for this parameter.
    ParmFactory& setTypeExtended(PRM_TypeExtended);

    /// @brief Specify the number of vector elements for this parameter.
    /// @details (The default vector size is one element.)
    ParmFactory& setVectorSize(int);

    /// Construct and return the parameter template.
    PRM_Template get() const;

private:
    struct Impl;
    boost::shared_ptr<Impl> mImpl;

    // For internal use only, and soon to be removed:
    ParmFactory& doSetChoiceList(PRM_ChoiceListType, const std::vector<std::string>&, bool);
    ParmFactory& doSetChoiceList(PRM_ChoiceListType, const char* const* items, bool);
};


////////////////////////////////////////


class OpPolicy;
typedef boost::shared_ptr<OpPolicy> OpPolicyPtr;


/// @brief Helper class to simplify operator registration
///
/// Usage example:
/// @code
/// void
/// newPopOperator(OP_OperatorTable* table)
/// {
///     houdini_utils::ParmList parms;
///
///     parms.add(houdini_utils::ParmFactory(PRM_STRING, "group", "Group")
///         .setHelpText("Specify a subset of the input VDB grids to be processed.")
///         .setChoiceList(&houdini_utils::PrimGroupMenu));
///
///     parms.add(...);
///
///     ...
///
///     houdini_utils::OpFactory(MyOpPolicy(), My Node",
///         POP_DW_MyNode::factory, parms, *table, houdini_utils::OpFactory::POP)
///         .addInput("Input geometry")              // input 0 (required)
///         .addOptionalInput("Reference geometry"); // input 1 (optional)
/// }
/// @endcode
class OPENVDB_HOUDINI_API OpFactory
{
public:
    enum OpFlavor { SOP, POP, ROP, VOP, HDA };

    /// @brief Return "SOP" for the SOP flavor, "POP" for the POP flavor, etc.
    /// @details Useful in OpPolicy classes for constructing type and icon names.
    static std::string flavorToString(OpFlavor);

    /// @brief Construct a factory that on destruction registers a new operator type.
    /// @param english  the operator's UI name, as it should appear in menus
    /// @param ctor     a factory function that creates operators of this type
    /// @param parms    the parameter template list for operators of this type
    /// @param table    the registry to which to add this operator type
    /// @param flavor   the operator's class (SOP, POP, etc.)
    /// @details @c OpPolicyType specifies the type of OpPolicy to be used to control
    /// the factory's behavior.  The (unused) @c OpPolicyType argument is required
    /// to enable the compiler to infer the type of the template argument
    /// (there is no other way to invoke a templated constructor).
    template<typename OpPolicyType>
    OpFactory(const OpPolicyType& /*unused*/, const std::string& english,
        OP_Constructor ctor, ParmList& parms, OP_OperatorTable& table, OpFlavor flavor = SOP)
    {
        this->init(OpPolicyPtr(new OpPolicyType), english, ctor, parms, table, flavor);
    }

    /// @note Factories initialized with this constructor use the DWAOpPolicy.
    OpFactory(const std::string& english, OP_Constructor ctor,
        ParmList& parms, OP_OperatorTable& table, OpFlavor flavor = SOP);

    /// Register the operator.
    ~OpFactory();

    /// @brief Return the new operator's flavor (SOP, POP, etc.).
    /// @details This accessor is mainly for use by OpPolicy objects.
    OpFlavor flavor() const;
    /// @brief Return the new operator's flavor as a string ("SOP", "POP", etc.).
    /// @details This accessor is mainly for use by OpPolicy objects.
    std::string flavorString() const;
    /// @brief Return the new operator's type name.
    /// @details This accessor is mainly for use by OpPolicy objects.
    const std::string& name() const;
    /// @brief Return the new operator's UI name.
    /// @details This accessor is mainly for use by OpPolicy objects.
    const std::string& english() const;
    /// @brief Return the new operator's icon name.
    /// @details This accessor is mainly for use by OpPolicy objects.
    const std::string& iconName() const;
    /// @brief Return the new operator's help URL.
    /// @details This accessor is mainly for use by OpPolicy objects.
    const std::string& helpURL() const;
    /// @brief Return the operator table with which this factory is associated.
    /// @details This accessor is mainly for use by OpPolicy objects.
    const OP_OperatorTable& table() const;

    /// @brief Construct a type name for this operator from the given English name
    /// and add it as an alias.
    /// @details For backward compatibility when an operator needs to be renamed,
    /// add the old name as an alias.
    OpFactory& addAlias(const std::string& english);
    /// @brief Add an alias for this operator.
    /// @details For backward compatibility when an operator needs to be renamed,
    /// add the old name as an alias.
    /// @note This variant takes an operator type name rather than an English name.
    OpFactory& addAliasVerbatim(const std::string& name);
    /// Add a required input with the given name.
    OpFactory& addInput(const std::string& name);
    /// Add an optional input with the given name.
    OpFactory& addOptionalInput(const std::string& name);
    /// @brief Set the maximum number of inputs allowed by this operator.
    /// @note It is only necessary to set this limit if there are inputs
    /// that have not been named with addInput() or addOptionalInput().
    OpFactory& setMaxInputs(unsigned = 9999);
    /// Specify obsolete parameters to this operator.
    OpFactory& setObsoleteParms(const ParmList&);
    /// Add one or more local variables to this operator.
    OpFactory& setLocalVariables(CH_LocalVariable*);
    OpFactory& setFlags(unsigned);

private:
    OpFactory(const OpFactory&);
    OpFactory& operator=(const OpFactory&);

    void init(OpPolicyPtr, const std::string& english, OP_Constructor,
        ParmList&, OP_OperatorTable&, OpFlavor);

    struct Impl;
    boost::shared_ptr<Impl> mImpl;
};


////////////////////////////////////////


/// @brief An OpPolicy customizes the behavior of an OpFactory.
/// This base class specifies the required interface.
class OPENVDB_HOUDINI_API OpPolicy
{
public:
    OpPolicy() {}
    virtual ~OpPolicy() {}

    /// @brief Return a type name for the operator defined by the given factory.
    std::string getName(const OpFactory& factory) { return getName(factory, factory.english()); }

    /// @brief Convert an English name into a type name for the operator defined by
    /// the given factory, and return the result.
    /// @details In this base class implementation, the operator's type name is generated
    /// by calling @c UT_String::forceValidVariableName() on the English name.
    /// @note This function might be called (from OpFactory::addAlias(), for example)
    /// with an English name other than the one returned by
    /// factory.@link OpFactory::english() english()@endlink.
    virtual std::string getName(const OpFactory& factory, const std::string& english);

    /// @brief Return an icon name for the operator defined by the given factory.
    /// @details Return an empty string to use Houdini's default icon naming scheme.
    virtual std::string getIconName(const OpFactory&) { return ""; }

    /// @brief Return a help URL for the operator defined by the given factory.
    virtual std::string getHelpURL(const OpFactory&) { return ""; }
};

/// @brief Default policy for DWA operator types
class OPENVDB_HOUDINI_API DWAOpPolicy: public OpPolicy
{
public:
    /// @brief Return a type name for the operator defined by the given factory.
    /// @details The operator's type name is generated from its English name
    /// by prepending "DW_" and removing non-alphanumeric characters.
    /// For example, "My Node" becomes "DW_MyNode".
    virtual std::string getName(const OpFactory&, const std::string& english);

    /// @brief Return a help URL for the operator defined by the given factory.
    virtual std::string getHelpURL(const OpFactory&);
};

/// @brief Default policies for DWA R&D operator types
///
/// See http://mydw.anim.dreamworks.com/display/FX/Houdini+Plugin+and+HDA+Naming+Rules

class DWALevel1RnDOpPolicy : public DWAOpPolicy
{
public:
    /// @brief Level 1: show-wide
    virtual std::string getIconName(const OpFactory&) { return "DreamWorks_L1_RnD"; }
};

class DWALevel2RnDOpPolicy : public DWAOpPolicy
{
public:
    /// @brief Level 2: global
    virtual std::string getIconName(const OpFactory&) { return "DreamWorks_L2_RnD"; }
};

class DWALevel3RnDOpPolicy : public DWAOpPolicy
{
public:
    /// @brief Level 3: depot, map, most stable
    virtual std::string getIconName(const OpFactory&) { return "DreamWorks_L3_RnD"; }
};

////////////////////////////////////////


/// @brief Helper class to manage input locking.
class OPENVDB_HOUDINI_API ScopedInputLock
{
public:
    ScopedInputLock(SOP_Node& node, OP_Context& context): mNode(&node)
    {
        if (mNode->lockInputs(context) >= UT_ERROR_ABORT) {
            throw std::runtime_error("failed to lock inputs");
        }
    }

    ~ScopedInputLock() { mNode->unlockInputs(); }

private:
    SOP_Node* mNode;
};


////////////////////////////////////////


// Extended group name drop-down menu incorporating "@<attr>=<value" syntax

OPENVDB_HOUDINI_API extern const PRM_ChoiceList PrimGroupMenuInput1;
OPENVDB_HOUDINI_API extern const PRM_ChoiceList PrimGroupMenuInput2;
OPENVDB_HOUDINI_API extern const PRM_ChoiceList PrimGroupMenuInput3;

/// @note   Use this if you have more than 3 inputs, otherwise use
///         the input specific menus instead which automatically
///         handle the appropriate spare data settings.
OPENVDB_HOUDINI_API extern const PRM_ChoiceList PrimGroupMenu;


} // namespace houdini_utils

#endif // HOUDINI_UTILS_PARM_FACTORY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
