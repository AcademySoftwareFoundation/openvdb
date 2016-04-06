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

/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *	Side Effects Software Inc
 *	123 Front Street West, Suite 1401
 *	Toronto, Ontario
 *	Canada   M5J 2M2
 *	416-504-9876
 *
 * NAME:	GT_GEOPrimCollectVDB.h ( GT Library, C++)
 *
 * COMMENTS:
 */

#ifndef __GT_GEOPrimCollectVDB__
#define __GT_GEOPrimCollectVDB__

#include <GT/GT_GEOPrimCollect.h>
#include <openvdb/Platform.h>

namespace openvdb_houdini {

class OPENVDB_HOUDINI_API GT_GEOPrimCollectVDB : public GT_GEOPrimCollect
{
public:
		GT_GEOPrimCollectVDB(const GA_PrimitiveTypeId &id);
    virtual	~GT_GEOPrimCollectVDB();

    static void registerPrimitive(const GA_PrimitiveTypeId &id);

    virtual GT_GEOPrimCollectData *
		beginCollecting(
			const GT_GEODetailListHandle &,
			const GT_RefineParms *) const;

    virtual GT_PrimitiveHandle
		collect(
			const GT_GEODetailListHandle &geometry,
			const GEO_Primitive *const* prim_list,
			int nsegments,
			GT_GEOPrimCollectData *data) const;

    virtual GT_PrimitiveHandle
		endCollecting(
			const GT_GEODetailListHandle &geometry,
			GT_GEOPrimCollectData *data) const;
private:

    GA_PrimitiveTypeId		myId;

};

} // namespace openvdb_houdini

#endif // __GT_GEOPrimCollectVDB__

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
