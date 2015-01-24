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

#include "QuantizedUnitVec.h"
#include <openvdb/Types.h>
#include <tbb/atomic.h>
#include <tbb/mutex.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


////////////////////////////////////////


bool QuantizedUnitVec::sInitialized = false;
float QuantizedUnitVec::sNormalizationWeights[MASK_SLOTS + 1];

namespace {
// Declare this at file scope to ensure thread-safe initialization.
tbb::mutex sInitMutex;
}


////////////////////////////////////////


void
QuantizedUnitVec::init()
{
    tbb::mutex::scoped_lock lock(sInitMutex);

    if (!sInitialized) {

        OPENVDB_START_THREADSAFE_STATIC_WRITE

        sInitialized = true;

        uint16_t xbits, ybits;
        double x, y, z, w;

        for (uint16_t b = 0; b < 8192; ++b) {

            xbits = uint16_t((b & MASK_XSLOT) >> 7);
            ybits = b & MASK_YSLOT;

            if ((xbits + ybits) > 126) {
                xbits = uint16_t(127 - xbits);
                ybits = uint16_t(127 - ybits);
            }

            x = double(xbits);
            y = double(ybits);
            z = double(126 - xbits - ybits);
            w = 1.0 / std::sqrt(x*x + y*y + z*z);

            sNormalizationWeights[b] = float(w);
        }

        OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
    }
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
