/*
 * Copyright (c) 2016
 *	Side Effects Software Inc.  All rights reserved.
 *
 * Redistribution and use of Houdini Development Kit samples in source and
 * binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. The name of Side Effects Software may not be used to endorse or
 *    promote products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY SIDE EFFECTS SOFTWARE `AS IS' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN
 * NO EVENT SHALL SIDE EFFECTS SOFTWARE BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *----------------------------------------------------------------------------
 */

#ifndef OPENVDB_HOUDINI_SOP_VDBVERBUTILS_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_SOP_VDBVERBUTILS_HAS_BEEN_INCLUDED

#include <GOP/GOP_Manager.h>

class SOP_VDBCacheOptions : public SOP_NodeCacheOptions
{
public:
	    SOP_VDBCacheOptions() 
		{}
    virtual ~SOP_VDBCacheOptions() {}

protected:
    virtual OP_ERROR	cook(OP_Context &context) override final
    {
	auto result = cookMySop(context);
	gop.destroyAdhocGroups();
	return result;
    }

    openvdb::Vec3f evalVec3f(const char* name, fpreal time) const
    {
	return openvdb::Vec3f(float(evalFloat(name, 0, time)),
			      float(evalFloat(name, 1, time)),
			      float(evalFloat(name, 2, time)));
    }
    openvdb::Vec3R evalVec3R(const char* name, fpreal time) const
    {
	return openvdb::Vec3R(evalFloat(name, 0, time),
			      evalFloat(name, 1, time),
			      evalFloat(name, 2, time));
    }
    openvdb::Vec3i evalVec3i(const char* name, fpreal time) const
    {
	return openvdb::Vec3i(evalInt(name, 0, time),
			      evalInt(name, 1, time),
			      evalInt(name, 2, time));
    }
    openvdb::Vec2R evalVec2R(const char* name, fpreal time) const
    {
	return openvdb::Vec2R(evalFloat(name, 0, time),
			      evalFloat(name, 1, time));
    }
    openvdb::Vec2i evalVec2i(const char* name, fpreal time) const
    {
	return openvdb::Vec2i(evalInt(name, 0, time),
			      evalInt(name, 1, time));
    }

    const GA_PrimitiveGroup *matchGroup(const GU_Detail &gdp, const UT_StringRef &groupname)
    {
	const GA_PrimitiveGroup *group = 0;
	if (groupname.isstring())
	{
	    bool		success;
	    group = gop.parseOrderedPrimitiveDetached(groupname, &gdp, false, success);
	    if (!success)
	    {
		UT_StringHolder	error;
		error = "Invalid group (";
		error += groupname;
		error += ")";
		throw std::runtime_error(error.c_str());
	    }
	}
	return group;
    }

    const GA_PrimitiveGroup *
    parsePrimitiveGroups(const UT_StringRef &maskStr, const GroupCreator &maskGeo)
    {
	return gop.parsePrimitiveGroups(maskStr, maskGeo);
    }

    GA_PrimitiveGroup *
    parsePrimitiveGroupsCopy(const UT_StringRef &maskStr, const GroupCreator &maskGeo)
    {
	return gop.parsePrimitiveGroupsCopy(maskStr, maskGeo);
    }

    const GA_PointGroup *
    parsePointGroups(const UT_StringRef &maskStr, const GroupCreator &maskGeo)
    {
	return gop.parsePointGroups(maskStr, maskGeo);
    }

    const GA_PointGroup *
    parsePointGroups(const UT_StringRef &maskStr, const GU_Detail *gdp)
    {
	return parsePointGroups(maskStr, GroupCreator(gdp));
    }

    virtual OP_ERROR	cookMySop(OP_Context &context) = 0;

    // Handles ad-hoc group creation.
    GOP_Manager		gop;
};

#define DEFINE_VERB_IMPLEMENTATION(CLASSNAME, SOPNAME, COOKMODE) \
\
OP_ERROR \
CLASSNAME::cookMySop(OP_Context& context) \
{ \
    return cookMyselfAsVerb(context); \
} \
 \
 \
class CLASSNAME##Verb : public SOP_NodeVerb \
{ \
public: \
    virtual SOP_NodeParms	*allocParms() const  \
    {  \
	UT_ASSERT(theParmTemplates); \
	return new SOP_NodeParmsOptions(theParmTemplates);  \
    } \
    virtual SOP_NodeCache	*allocCache() const { return new CLASSNAME##Cache(); } \
    virtual UT_StringHolder	 name() const { return #SOPNAME##_sh; } \
 \
    virtual CookMode		 cookMode(const SOP_NodeParms *)  const { return COOKMODE; } \
 \
    virtual void	cook(const CookParms &cookparms) const \
    { \
	auto cache = (SOP_NodeCacheOptions *)cookparms.cache(); \
 \
	cache->doCook(this, cookparms); \
    } \
 \
protected: \
}; \
 \
 \
static SOP_NodeVerb::Register<CLASSNAME##Verb>	the##CLASSNAME##Verb; \
 \
const SOP_NodeVerb * \
CLASSNAME::cookVerb() const  \
{  \
    return the##CLASSNAME##Verb.get(); \
} \

#endif
