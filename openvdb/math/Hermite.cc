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

#include "Hermite.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


////////////////////////////////////////

// min and max on compressd data

Hermite
min(const Hermite& lhs, const Hermite& rhs)
{
    Hermite ret;

    if(!lhs && !rhs) {

        if(lhs.isInside()) ret = lhs;
        else ret = rhs;

        return ret;
    }

    ret.setIsInside(lhs.isInside() || rhs.isInside());

    if(lhs.isGreaterX(rhs)) ret.setX(rhs);
    else ret.setX(lhs);

    if(lhs.isGreaterY(rhs)) ret.setY(rhs);
    else ret.setY(lhs);

    if(lhs.isGreaterZ(rhs)) ret.setZ(rhs);
    else ret.setZ(lhs);

    return ret;
}

Hermite
max(const Hermite& lhs, const Hermite& rhs)
{
    Hermite ret;

    if(!lhs && !rhs) {

        if(!lhs.isInside()) ret = lhs;
        else ret = rhs;

        return ret;
    }

    ret.setIsInside(lhs.isInside() && rhs.isInside());

    if(rhs.isGreaterX(lhs)) ret.setX(rhs);
    else ret.setX(lhs);

    if(rhs.isGreaterY(lhs)) ret.setY(rhs);
    else ret.setY(lhs);

    if(rhs.isGreaterZ(lhs)) ret.setZ(rhs);
    else ret.setZ(lhs);

    return ret;
}


////////////////////////////////////////

// constructors

Hermite::Hermite():
    mXNormal(0),
    mYNormal(0),
    mZNormal(0),
    mData(0)
{
}

Hermite::Hermite(const Hermite& rhs):
    mXNormal(rhs.mXNormal),
    mYNormal(rhs.mYNormal),
    mZNormal(rhs.mZNormal),
    mData(rhs.mData)
{
}


////////////////////////////////////////

// string representation

std::string
Hermite::str() const
{
    std::ostringstream ss;

    ss << "{ " << (isInside() ? "inside" : "outside");
    if(hasOffsetX()) ss << " |x " << getOffsetX() << " " << getNormalX();
    if(hasOffsetY()) ss << " |y " << getOffsetY() << " " << getNormalY();
    if(hasOffsetZ()) ss << " |z " << getOffsetZ() << " " << getNormalZ();
    ss << " }";

    return ss.str();
}


////////////////////////////////////////


void
Hermite::read(std::istream& is)
{
    is.read(reinterpret_cast<char*>(&mXNormal), sizeof(uint16_t));
    is.read(reinterpret_cast<char*>(&mYNormal), sizeof(uint16_t));
    is.read(reinterpret_cast<char*>(&mZNormal), sizeof(uint16_t));
    is.read(reinterpret_cast<char*>(&mData), sizeof(uint32_t));
}

void
Hermite::write(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(&mXNormal), sizeof(uint16_t));
    os.write(reinterpret_cast<const char*>(&mYNormal), sizeof(uint16_t));
    os.write(reinterpret_cast<const char*>(&mZNormal), sizeof(uint16_t));
    os.write(reinterpret_cast<const char*>(&mData), sizeof(uint32_t));
}


} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
