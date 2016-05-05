///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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
/// @file ParmFactory.cc
/// @author FX R&D OpenVDB team

#include "ParmFactory.h"

#include <CH/CH_Manager.h>
#include <GOP/GOP_GroupParse.h>
#include <GU/GU_Detail.h>
#include <GU/GU_PrimPoly.h>
#include <GU/GU_Selection.h>
#include <GA/GA_AIFSharedStringTuple.h>
#include <GA/GA_Attribute.h>
#include <GA/GA_AttributeRef.h>
#include <OP/OP_OperatorTable.h>
#include <PRM/PRM_Parm.h>
#include <PRM/PRM_SharedFunc.h>
#include <UT/UT_IntArray.h>
#include <UT/UT_Version.h>
#include <UT/UT_WorkArgs.h>
#include <cstring> // for ::strdup()

namespace houdini_utils {

ParmList&
ParmList::add(const PRM_Template& p)
{
    mParmVec.push_back(p);
    incFolderParmCount();
    return *this;
}


ParmList&
ParmList::add(const ParmFactory& f)
{
    add(f.get());
    return *this;
}


ParmList::SwitcherInfo*
ParmList::getCurrentSwitcher()
{
    SwitcherInfo* info = NULL;
    if (!mSwitchers.empty()) {
        info = &mSwitchers.back();
    }
    return info;
}


ParmList&
ParmList::beginSwitcher(const std::string& token, const std::string& label)
{
    if (NULL != getCurrentSwitcher()) {
        incFolderParmCount();
    }
    SwitcherInfo info;
    info.parmIdx = mParmVec.size();
    info.exclusive = false;
    mSwitchers.push_back(info);
    // Add a switcher parameter with the given token and name, but no folders.
    mParmVec.push_back(ParmFactory(PRM_SWITCHER, token, label).get());
    return *this;
}


ParmList&
ParmList::beginExclusiveSwitcher(const std::string& token, const std::string& label)
{
    if (NULL != getCurrentSwitcher()) {
        incFolderParmCount();
    }
    SwitcherInfo info;
    info.parmIdx = mParmVec.size();
    info.exclusive = true;
    mSwitchers.push_back(info);
    // Add a switcher parameter with the given token and name, but no folders.
    mParmVec.push_back(ParmFactory(PRM_SWITCHER, token, label).get());
    return *this;
}


ParmList&
ParmList::endSwitcher()
{
    if (SwitcherInfo* info = getCurrentSwitcher()) {
        if (info->folders.empty()) {
            throw std::runtime_error("added switcher that has no folders");
        } else {
            // Replace the placeholder switcher parameter that was added to
            // mParmVec in beginSwitcher() with a new parameter that has
            // the correct folder count and folder info.
            PRM_Template& switcherParm = mParmVec[info->parmIdx];
            std::string token, label;
            if (const char* s = switcherParm.getToken()) token = s;
            if (const char* s = switcherParm.getLabel()) label = s;
            mParmVec[info->parmIdx] =
                ParmFactory(info->exclusive ? PRM_SWITCHER_EXCLUSIVE : PRM_SWITCHER,
                    token.c_str(), label.c_str())
                .setVectorSize(int(info->folders.size()))
                .setDefault(info->folders)
                .get();
        }
        mSwitchers.pop_back();
    } else {
        throw std::runtime_error("endSwitcher() called with no corresponding beginSwitcher()");
    }
    return *this;
}


ParmList&
ParmList::addFolder(const std::string& label)
{
    if (SwitcherInfo* info = getCurrentSwitcher()) {
        info->folders.push_back(PRM_Default(/*numParms=*/0, ::strdup(label.c_str())));
    } else {
        throw std::runtime_error("added folder to nonexistent switcher");
    }
    return *this;
}


void
ParmList::incFolderParmCount()
{
    if (SwitcherInfo* info = getCurrentSwitcher()) {
        if (info->folders.empty()) {
            throw std::runtime_error("added parameter to switcher that has no folders");
        } else {
            // If a parameter is added to this ParmList while a switcher with at least
            // one folder is active, increment the folder's parameter count.
            PRM_Default& def = *(info->folders.rbegin());
            def.setOrdinal(def.getOrdinal() + 1);
        }
    }
}


PRM_Template*
ParmList::get() const
{
    const size_t numParms = mParmVec.size();
    PRM_Template* ret = new PRM_Template[numParms + 1];
    for (size_t n = 0; n < numParms; ++n) {
        ret[n] = mParmVec[n];
    }
    return ret;
}


////////////////////////////////////////


struct ParmFactory::Impl
{
    Impl(const std::string& token, const std::string& label):
        callbackFunc(0),
        choicelist(NULL),
        conditional(NULL),
        defaults(PRMzeroDefaults),
        helpText(NULL),
        multiType(PRM_MULTITYPE_NONE),
        name(new PRM_Name(token.c_str(), label.c_str())),
        parmGroup(0),
        range(NULL),
        spareData(NULL),
        multiparms(NULL),
        typeExtended(PRM_TYPE_NONE),
        vectorSize(1)
    {
        const_cast<PRM_Name*>(name)->harden();
    }

    PRM_Callback               callbackFunc;
    const PRM_ChoiceList*      choicelist;
    const PRM_ConditionalBase* conditional;
    const PRM_Default*         defaults;
    const char*                helpText;
    PRM_MultiType              multiType;
    const PRM_Name*            name;
    int                        parmGroup;
    const PRM_Range*           range;
    const PRM_SpareData*       spareData;
    const PRM_Template*        multiparms;
    PRM_Type                   type;
    PRM_TypeExtended           typeExtended;
    int                        vectorSize;
};


ParmFactory::ParmFactory(PRM_Type type, const std::string& token, const std::string& label):
    mImpl(new Impl(token, label))
{
    mImpl->type = type;
}


ParmFactory::ParmFactory(PRM_MultiType multiType, const std::string& token,
    const std::string& label): mImpl(new Impl(token, label))
{
    mImpl->multiType = multiType;
}

ParmFactory&
ParmFactory::setCallbackFunc(const PRM_Callback& f) { mImpl->callbackFunc = f; return *this; }

ParmFactory&
ParmFactory::setChoiceList(const PRM_ChoiceList* c)
{
    mImpl->choicelist = c;

#if (UT_VERSION_INT >= 0x0e000075) // 14.0.117 or later
    if (c == &PrimGroupMenuInput1) {
        setSpareData(SOP_Node::getGroupSelectButton(GA_GROUP_PRIMITIVE,
            NULL, 0, &SOP_Node::theFirstInput));
    } else if (c == &PrimGroupMenuInput2) {
        setSpareData(SOP_Node::getGroupSelectButton(GA_GROUP_PRIMITIVE,
            NULL, 1, &SOP_Node::theSecondInput));
    } else if (c == &PrimGroupMenuInput3) {
        setSpareData(SOP_Node::getGroupSelectButton(GA_GROUP_PRIMITIVE,
            NULL, 2, &SOP_Node::theThirdInput));
    }
#else
    if (c == &PrimGroupMenuInput1) {
        setSpareData(&SOP_Node::theFirstInput);
    } else if (c == &PrimGroupMenuInput2) {
        setSpareData(&SOP_Node::theSecondInput);
    } else if (c == &PrimGroupMenuInput3) {
        setSpareData(&SOP_Node::theThirdInput);
    }
#endif

    return *this;
}


/// @todo Merge this into setChoiceListItems() once the deprecated
/// setChoiceList() overloads have been removed.
ParmFactory&
ParmFactory::doSetChoiceList(PRM_ChoiceListType typ, const char* const* items, bool paired)
{
    size_t numItems = 0;
    for ( ; items[numItems] != NULL; ++numItems) {}
    if (paired) numItems >>= 1;
    PRM_Name* copyOfItems = new PRM_Name[numItems + 1]; // extra item is list terminator
    if (paired) {
        for (size_t i = 0, n = 0; n < numItems; ++n, i += 2) {
            copyOfItems[n].setToken(items[i]);
            copyOfItems[n].setLabel(items[i+1]);
            copyOfItems[n].harden();
        }
    } else {
        for (size_t n = 0; n < numItems; ++n) {
            UT_String idx;
            idx.itoa(n);
            copyOfItems[n].setToken(idx.buffer());
            copyOfItems[n].setLabel(items[n]);
            copyOfItems[n].harden();
        }
    }
    mImpl->choicelist = new PRM_ChoiceList(typ, copyOfItems);
    return *this;
}

/// @todo Merge this into setChoiceListItems() once the deprecated
/// setChoiceList() overloads have been removed.
ParmFactory&
ParmFactory::doSetChoiceList(PRM_ChoiceListType typ,
    const std::vector<std::string>& items, bool paired)
{
    const size_t numItems = items.size() >> (paired ? 1 : 0);
    PRM_Name* copyOfItems = new PRM_Name[numItems + 1]; // extra item is list terminator
    if (paired) {
        for (size_t i = 0, n = 0; n < numItems; ++n, i += 2) {
            copyOfItems[n].setToken(items[i].c_str());
            copyOfItems[n].setLabel(items[i+1].c_str());
            copyOfItems[n].harden();
        }
    } else {
        for (size_t n = 0; n < numItems; ++n) {
            UT_String idx;
            idx.itoa(n);
            copyOfItems[n].setToken(idx.buffer());
            copyOfItems[n].setLabel(items[n].c_str());
            copyOfItems[n].harden();
        }
    }
    mImpl->choicelist = new PRM_ChoiceList(typ, copyOfItems);
    return *this;
}

ParmFactory&
ParmFactory::setChoiceList(PRM_ChoiceListType typ, const char* const* items, bool paired)
{
    return doSetChoiceList(typ, items, paired);
}

ParmFactory&
ParmFactory::setChoiceList(PRM_ChoiceListType typ,
    const std::vector<std::string>& items, bool paired)
{
    return doSetChoiceList(typ, items, paired);
}

ParmFactory&
ParmFactory::setChoiceListItems(PRM_ChoiceListType typ, const char* const* items)
{
    return doSetChoiceList(typ, items, /*paired=*/true);
}

ParmFactory&
ParmFactory::setChoiceListItems(PRM_ChoiceListType typ, const std::vector<std::string>& items)
{
    return doSetChoiceList(typ, items, /*paired=*/true);
}

ParmFactory&
ParmFactory::setConditional(const PRM_ConditionalBase* c) { mImpl->conditional = c; return *this; }

ParmFactory&
ParmFactory::setDefault(fpreal f, const char* s, CH_StringMeaning meaning)
{
    mImpl->defaults = new PRM_Default(f, s, meaning);
    return *this;
}

ParmFactory&
ParmFactory::setDefault(const std::string& s, CH_StringMeaning meaning)
{
    mImpl->defaults = new PRM_Default(0.0, ::strdup(s.c_str()), meaning);
    return *this;
}

ParmFactory&
ParmFactory::setDefault(const std::vector<fpreal>& v)
{
    const size_t numDefaults = v.size();
    PRM_Default* defaults = new PRM_Default[numDefaults + 1];
    for (size_t n = 0; n < numDefaults; ++n) {
        defaults[n] = PRM_Default(v[n]);
    }
    mImpl->defaults = defaults;
    return *this;
}

ParmFactory&
ParmFactory::setDefault(const std::vector<PRM_Default>& defaults)
{
    const size_t numDefaults = defaults.size();
    PRM_Default* copyOfDefaults = new PRM_Default[numDefaults + 1];
    for (size_t n = 0; n < numDefaults; ++n) {
        copyOfDefaults[n] = defaults[n];
    }
    mImpl->defaults = copyOfDefaults;
    return *this;
}

ParmFactory&
ParmFactory::setDefault(const PRM_Default* d)       { mImpl->defaults = d; return *this; }

ParmFactory&
ParmFactory::setHelpText(const char* t)             { mImpl->helpText = t; return *this; }

ParmFactory&
ParmFactory::setParmGroup(int n)                    { mImpl->parmGroup = n; return *this; }

ParmFactory&
ParmFactory::setRange(PRM_RangeFlag minFlag, fpreal minVal, PRM_RangeFlag maxFlag, fpreal maxVal)
{
    mImpl->range = new PRM_Range(minFlag, minVal, maxFlag, maxVal);
    return *this;
}

ParmFactory&
ParmFactory::setRange(const std::vector<PRM_Range>& ranges)
{
    const size_t numRanges = ranges.size();
    PRM_Range* copyOfRanges = new PRM_Range[numRanges + 1];
    for (size_t n = 0; n < numRanges; ++n) {
        copyOfRanges[n] = ranges[n];
    }
    mImpl->range = copyOfRanges;
    return *this;
}

ParmFactory&
ParmFactory::setRange(const PRM_Range* r)            { mImpl->range = r; return *this; }

ParmFactory&
ParmFactory::setSpareData(const std::map<std::string, std::string>& items)
{
    typedef std::map<std::string, std::string> StringMap;
    if (!items.empty()) {
        PRM_SpareData* data = new PRM_SpareData();
        for (StringMap::const_iterator i = items.begin(), e = items.end(); i != e; ++i) {
            data->addTokenValue(i->first.c_str(), i->second.c_str());
        }
        mImpl->spareData = data;
    }
    return *this;
}

ParmFactory&
ParmFactory::setSpareData(const PRM_SpareData* d)   { mImpl->spareData = d; return *this; }

ParmFactory&
ParmFactory::setMultiparms(const ParmList& p)       { mImpl->multiparms = p.get(); return *this; }

ParmFactory&
ParmFactory::setTypeExtended(PRM_TypeExtended t)    { mImpl->typeExtended = t; return *this; }

ParmFactory&
ParmFactory::setVectorSize(int n)                   { mImpl->vectorSize = n; return *this; }

PRM_Template
ParmFactory::get() const
{
#ifdef SESI_OPENVDB
    // Help is maintained separately within Houdini
    const char *helpText = NULL;
#else
    const char *helpText = mImpl->helpText;
#endif
    if (mImpl->multiType != PRM_MULTITYPE_NONE) {
        return PRM_Template(
            mImpl->multiType,
            const_cast<PRM_Template*>(mImpl->multiparms),
            fpreal(mImpl->vectorSize),
            const_cast<PRM_Name*>(mImpl->name),
            const_cast<PRM_Default*>(mImpl->defaults),
            const_cast<PRM_Range*>(mImpl->range),
            const_cast<PRM_SpareData*>(mImpl->spareData),
            helpText,
            const_cast<PRM_ConditionalBase*>(mImpl->conditional));
    } else {
        return PRM_Template(
            mImpl->type,
            mImpl->typeExtended,
            mImpl->vectorSize,
            const_cast<PRM_Name*>(mImpl->name),
            const_cast<PRM_Default*>(mImpl->defaults),
            const_cast<PRM_ChoiceList*>(mImpl->choicelist),
            const_cast<PRM_Range*>(mImpl->range),
            mImpl->callbackFunc,
            const_cast<PRM_SpareData*>(mImpl->spareData),
            mImpl->parmGroup,
            helpText,
            const_cast<PRM_ConditionalBase*>(mImpl->conditional));
    }
}


////////////////////////////////////////


namespace {

/// @brief Operator class that adds the help link. Used by the OpFactory.
class OP_OperatorDW: public OP_Operator
{
public:
    OP_OperatorDW(
        OpFactory::OpFlavor,
        const char* name,
        const char* english,
        OP_Constructor construct,
        PRM_Template* multiparms,
        unsigned minSources,
        unsigned maxSources,
        CH_LocalVariable* variables,
        unsigned flags,
        const char** inputlabels,
        const std::string& helpUrl):
        OP_Operator(name, english, construct, multiparms, minSources,
            maxSources, variables, flags, inputlabels),
        mHelpUrl(helpUrl)
    {
    }

    virtual ~OP_OperatorDW() {}
    virtual bool getOpHelpURL(UT_String& url) { url = mHelpUrl; return !mHelpUrl.empty(); }

private:
    std::string mHelpUrl;
};

} // unnamed namespace


////////////////////////////////////////


struct OpFactory::Impl
{
    Impl(const std::string& english, OP_Constructor& constructor, PRM_Template* parms,
        OP_OperatorTable& table, OpFactory::OpFlavor flavor):
        mFlavor(flavor),
        mEnglish(english),
        mConstruct(constructor),
        mTable(&table),
        mParms(parms),
        mObsoleteParms(NULL),
        mMaxSources(0),
        mVariables(NULL),
        mFlags(0)
    {
    }

    ~Impl()
    {
        std::for_each(mInputLabels.begin(), mInputLabels.end(), ::free);
        // Note: In get(), mOptInputLabels are appended to mInputLabels.
    }

    void init(const OpFactory& factory, OpPolicyPtr policy)
    {
        // Because mPolicy is supplied by this Impl's parent OpFactory
        // (which knows which OpPolicy subclass to use), initialization
        // of the following members must be postponed until both
        // the OpFactory and this Impl have been fully constructed.
        mPolicy = policy;
        mName = mPolicy->getName(factory);
        mIconName = mPolicy->getIconName(factory);
        mHelpUrl = mPolicy->getHelpURL(factory);
    }

    OP_OperatorDW* get()
    {
        // Get the number of required inputs.
        const unsigned minSources = unsigned(mInputLabels.size());

        // Append optional input labels to required input labels.
        mInputLabels.insert(mInputLabels.end(),
            mOptInputLabels.begin(), mOptInputLabels.end());

        // Ensure that the maximum number of inputs is at least as large
        // as the number of labeled inputs.
        mMaxSources = std::max<unsigned>(unsigned(mInputLabels.size()), mMaxSources);

        mInputLabels.push_back(NULL);

        OP_OperatorDW* op = new OP_OperatorDW(mFlavor, mName.c_str(), mEnglish.c_str(),
            mConstruct, mParms, minSources, mMaxSources, mVariables, mFlags,
            const_cast<const char**>(&mInputLabels[0]), mHelpUrl);

        if (!mIconName.empty()) op->setIconName(mIconName.c_str());

        if (mObsoleteParms != NULL) op->setObsoleteTemplates(mObsoleteParms);

        return op;
    }

    OpPolicyPtr mPolicy; // polymorphic, so stored by pointer
    OpFactory::OpFlavor mFlavor;
    std::string mEnglish, mName, mIconName, mHelpUrl;
    OP_Constructor mConstruct;
    OP_OperatorTable* mTable;
    PRM_Template *mParms, *mObsoleteParms;
    unsigned mMinSources;
    unsigned mMaxSources;
    CH_LocalVariable* mVariables;
    unsigned mFlags;
    std::vector<std::string> mAliases;
    std::vector<char*> mInputLabels, mOptInputLabels;
};


OpFactory::OpFactory(const std::string& english, OP_Constructor ctor,
    ParmList& parms, OP_OperatorTable& table, OpFlavor flavor)
{
    this->init(OpPolicyPtr(new DWAOpPolicy), english, ctor, parms, table, flavor);
}


OpFactory::~OpFactory()
{
    mImpl->mTable->addOperator(mImpl->get());

    for (size_t n = 0, N = mImpl->mAliases.size(); n < N; ++n) {
        const std::string& alias = mImpl->mAliases[n];
        if (!alias.empty()) {
            mImpl->mTable->setOpAlias(/*original=*/mImpl->mName.c_str(), alias.c_str());
        }
    }
}


void
OpFactory::init(OpPolicyPtr policy, const std::string& english, OP_Constructor ctor,
    ParmList& parms, OP_OperatorTable& table, OpFlavor flavor)
{
    mImpl.reset(new Impl(english, ctor, parms.get(), table, flavor));
    mImpl->init(*this, policy);
}


//static
std::string
OpFactory::flavorToString(OpFlavor flavor)
{
    switch (flavor) {
        case SOP: return "SOP";
        case POP: return "POP";
        case ROP: return "ROP";
        case VOP: return "VOP";
        case HDA: return "HDA";
    }
    return "";
}


OpFactory::OpFlavor
OpFactory::flavor() const
{
    return mImpl->mFlavor;
}


std::string
OpFactory::flavorString() const
{
    return flavorToString(mImpl->mFlavor);
}


const std::string&
OpFactory::name() const
{
    return mImpl->mName;
}


const std::string&
OpFactory::english() const
{
    return mImpl->mEnglish;
}


const std::string&
OpFactory::iconName() const
{
    return mImpl->mIconName;
}


const std::string&
OpFactory::helpURL() const
{
    return mImpl->mHelpUrl;
}


const OP_OperatorTable&
OpFactory::table() const
{
    return *mImpl->mTable;
}


OpFactory&
OpFactory::addAlias(const std::string& english)
{
    if (!english.empty()) {
        this->addAliasVerbatim(mImpl->mPolicy->getName(*this, english));
    }
    return *this;
}


OpFactory&
OpFactory::addAliasVerbatim(const std::string& name)
{
    if (!name.empty()) {
        mImpl->mAliases.push_back(name);
    }
    return *this;
}


OpFactory&
OpFactory::addInput(const std::string& name)
{
    mImpl->mInputLabels.push_back(::strdup(name.c_str()));
    return *this;
}


OpFactory&
OpFactory::addOptionalInput(const std::string& name)
{
    mImpl->mOptInputLabels.push_back(::strdup(name.c_str()));
    return *this;
}


OpFactory&
OpFactory::setMaxInputs(unsigned n) { mImpl->mMaxSources = n; return *this; }


OpFactory&
OpFactory::setObsoleteParms(const ParmList& parms)
{
    if (mImpl->mObsoleteParms != NULL) delete mImpl->mObsoleteParms;
    mImpl->mObsoleteParms = parms.get();
    return *this;
}


OpFactory&
OpFactory::setLocalVariables(CH_LocalVariable* v) { mImpl->mVariables = v; return *this; }


OpFactory&
OpFactory::setFlags(unsigned f) { mImpl->mFlags = f; return *this; }


////////////////////////////////////////


//virtual
std::string
OpPolicy::getName(const OpFactory&, const std::string& english)
{
    UT_String s(english);
    s.forceValidVariableName();
    return s.toStdString();
}


//virtual
std::string
DWAOpPolicy::getName(const OpFactory&, const std::string& english)
{
    UT_String s(english);
    // Remove non-alphanumeric characters from the name.
    s.forceValidVariableName();
    std::string name = s.toStdString();
    // Remove spaces and underscores.
    name.erase(std::remove(name.begin(), name.end(), ' '), name.end());
    name.erase(std::remove(name.begin(), name.end(), '_'), name.end());
    name = "DW_" + name;
    return name;
}


//virtual
std::string
DWAOpPolicy::getHelpURL(const OpFactory& factory)
{
#if defined(PRODDEV_BUILD) || defined(DWREAL_IS_DOUBLE)
    std::string url("http://mydw.anim.dreamworks.com/display/FX/Houdini+");
    url += factory.flavorString(); // append "SOP", "POP", etc.
    url += "_" + factory.name();
    return url;
#else
    return "";
#endif
}


////////////////////////////////////////


#if (UT_VERSION_INT >= 0x0d000000) // 13.0.0 or later

const PRM_ChoiceList PrimGroupMenuInput1 = SOP_Node::primGroupMenu;
const PRM_ChoiceList PrimGroupMenuInput2 = SOP_Node::primGroupMenu;
const PRM_ChoiceList PrimGroupMenuInput3 = SOP_Node::primGroupMenu;

const PRM_ChoiceList PrimGroupMenu = SOP_Node::primGroupMenu;

#else // earlier than 13.0.0

namespace {

// Extended group name drop-down menu incorporating @c "@<attr>=<value>" syntax
// (this functionality was added to SOP_Node::primGroupMenu some time ago,
// possibly as early as Houdini 12.5)

inline int
lookupGroupInput(const PRM_SpareData *spare)
{
    const char  *istring;
    if (!spare) return 0;
    istring = spare->getValue("sop_input");
    return istring ? atoi(istring) : 0;
}


void
sopBuildGridMenu(void *data, PRM_Name *menuEntries, int themenusize,
    const PRM_SpareData *spare, const PRM_Parm *parm)
{
    SOP_Node* sop = CAST_SOPNODE((OP_Node *)data);
    int inputIndex = lookupGroupInput(spare);

    const GU_Detail* gdp = sop->getInputLastGeo(inputIndex, CHgetEvalTime());

    GA_GroupTable::iterator<GA_ElementGroup> ithead;

    GA_GroupTable::iterator<GA_ElementGroup> itpthead;
    UT_WorkBuffer                buf;
    UT_String                    primtoken, pttoken;
    int                          i, n_entries;
    UT_String                    origgroup;
    UT_WorkArgs                  wargs;
    UT_StringArray               allnames;
    bool                         needseparator = false;

    // Find our original group list so we can flag groups we
    // as toggleable.
    origgroup = "";
    // We don't want to evaluate the expression as the toggle
    // works on the raw code.
#if (UT_VERSION_INT >= 0x0c0100B6) // 12.1.182 or later
    if (parm) parm->getValue(0.0f, origgroup, 0, 0, SYSgetSTID());
#else
    if (parm) parm->getValue(0.0f, origgroup, 0, 0, UTgetSTID());
#endif

    origgroup.tokenize(wargs, ' ');

    UT_SymbolTable table;
    UT_Thing thing((void *) 0);

    for (i = 0; i < wargs.getArgc(); i++) {
        table.addSymbol(wargs(i), thing);
    }

    n_entries = 0;

    if (gdp) {
        ithead = gdp->primitiveGroups().beginTraverse();

        GA_ROAttributeRef aRef = gdp->findPrimitiveAttribute("name");
        if (aRef.isValid()) {
            const GA_AIFSharedStringTuple *stuple = aRef.getAttribute()->getAIFSharedStringTuple();
            if (stuple) {
                UT_IntArray handles;
                stuple->extractStrings(aRef.getAttribute(), allnames, handles);
            }
        }
    }

    if (!ithead.atEnd()) {
        GA_PrimitiveGroup *primsel;
        bool includeselection = true;

        if (gdp->selection() && (primsel = gdp->selection()->primitives()) &&
            includeselection &&
            primsel->getInternal())     // If not internal we'll catch it below
        {
            GOP_GroupParse::buildPrimGroupToken(gdp, primsel, primtoken);

            if (primtoken.length()) {
                menuEntries[n_entries].setToken(primtoken);
                menuEntries[n_entries].setLabel("Primitive Selection");
                n_entries++;
                needseparator = true;
            }
        }

        int                      numprim = 0;
        int                      startprim = n_entries;

        for (i = n_entries; (i + 1) < themenusize && !ithead.atEnd(); ++ithead, i++) {
            const GA_PrimitiveGroup* primgrp =
                static_cast<const GA_PrimitiveGroup *>(ithead.group());

            if (primgrp->getInternal()) continue;

            menuEntries[n_entries].setToken(primgrp->getName());

            // Determine if this group is already in the list.  If so,
            // we want to put a + in front of it.
            if (table.findSymbol(primgrp->getName(), &thing)) {
                buf.sprintf("%s\t*", (const char *) primgrp->getName());
                menuEntries[n_entries].setLabel(buf.buffer());
            } else {
                menuEntries[n_entries].setLabel(primgrp->getName());
            }
            numprim++;
            n_entries++;
        }

        if (numprim) {
            PRMsortMenu(&menuEntries[startprim], numprim);
            needseparator = true;
        }
    }

    if (allnames.entries()) {
        if (needseparator && (n_entries+1 < themenusize)) {
            needseparator = false;
            menuEntries[n_entries].setToken(PRM_Name::mySeparator);
            menuEntries[n_entries].setLabel(PRM_Name::mySeparator);
            n_entries++;
        }

        int             numnames = 0;
        int             startname = n_entries;

        for (int j = 0; j < allnames.entries(); j++) {
            if (n_entries+1 >= themenusize) break;

            if(!allnames(j).isstring()) continue;

            buf.sprintf("@name=%s", (const char *) allnames(j));
            menuEntries[n_entries].setToken(buf.buffer());

            // Determine if this group is already in the list.  If so,
            // we want to put a + in front of it.
            if (table.findSymbol(buf.buffer(), &thing)) {
                buf.sprintf("@%s\t*", (const char *) allnames(j));
                menuEntries[n_entries].setLabel(buf.buffer());
            } else {
                buf.sprintf("@%s", (const char *) allnames(j));
                menuEntries[n_entries].setLabel(buf.buffer());
            }
            numnames++;
            n_entries++;
        }

        if (numnames) {
            PRMsortMenu(&menuEntries[startname], numnames);
            needseparator = true;
        }
    }

    menuEntries[n_entries].setToken(0);
    menuEntries[n_entries].setLabel(0);
}

} // unnamed namespace


#ifdef _MSC_VER

OPENVDB_HOUDINI_API const PRM_ChoiceList
PrimGroupMenuInput1(PRM_CHOICELIST_TOGGLE, sopBuildGridMenu);
OPENVDB_HOUDINI_API const PRM_ChoiceList
PrimGroupMenuInput2(PRM_CHOICELIST_TOGGLE, sopBuildGridMenu);
OPENVDB_HOUDINI_API const PRM_ChoiceList
PrimGroupMenuInput3(PRM_CHOICELIST_TOGGLE, sopBuildGridMenu);

OPENVDB_HOUDINI_API const PRM_ChoiceList PrimGroupMenu(PRM_CHOICELIST_TOGGLE, sopBuildGridMenu);

#else

const PRM_ChoiceList
PrimGroupMenuInput1(PRM_CHOICELIST_TOGGLE, sopBuildGridMenu);
const PRM_ChoiceList
PrimGroupMenuInput2(PRM_CHOICELIST_TOGGLE, sopBuildGridMenu);
const PRM_ChoiceList
PrimGroupMenuInput3(PRM_CHOICELIST_TOGGLE, sopBuildGridMenu);

const PRM_ChoiceList PrimGroupMenu(PRM_CHOICELIST_TOGGLE, sopBuildGridMenu);

#endif

#endif // earlier than 13.0.0


} // namespace houdini_utils

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
