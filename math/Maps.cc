///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
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

#include "Maps.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


////////////////////////////////////////


MapRegistry* MapRegistry::mInstance = NULL;

// Declare this at file scope to ensure thread-safe initialization.
tbb::mutex sInitMapRegistryMutex;

MapRegistry*
MapRegistry::instance()
{
    Lock lock(sInitMapRegistryMutex);

    if(mInstance == NULL) {
        OPENVDB_START_THREADSAFE_STATIC_WRITE
            mInstance = new MapRegistry();
        OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
            return mInstance;
    }

    return mInstance;
}

MapBase::Ptr
MapRegistry::createMap(const Name& name)
{
    Lock lock(instance()->mMutex);
    MapDictionary::const_iterator iter = instance()->mMap.find(name);

    if (iter == instance()->mMap.end()) {
        OPENVDB_THROW(LookupError, "Cannot create map of unregistered type " << name);
    }

    return (iter->second)();
}

bool
MapRegistry::isRegistered(const Name& name)
{
    Lock lock(instance()->mMutex);
    return (instance()->mMap.find(name) != instance()->mMap.end());
}

void
MapRegistry::registerMap(const Name& name, MapBase::MapFactory factory)
{
    Lock lock(instance()->mMutex);

    if (instance()->mMap.find(name) != instance()->mMap.end()) {
        OPENVDB_THROW(KeyError, "Map type " << name << " is already registered");
    }

    instance()->mMap[name] = factory;
}

void
MapRegistry::unregisterMap(const Name& name)
{
    Lock lock(instance()->mMutex);
    instance()->mMap.erase(name);
}

void
MapRegistry::clear()
{
    Lock lock(instance()->mMutex);
    instance()->mMap.clear();
}


////////////////////////////////////////

// Utility methods for decomposition


SymmetricMap::Ptr
createSymmetricMap(const Mat3d& m)
{
    // test that the mat3 is a rotation || reflection
    if (!isSymmetric(m)) {
        OPENVDB_THROW(ArithmeticError,
                      "3x3 Matrix initializing symmetric map was not symmetric");
    }
    Vec3d eigenValues;
    Mat3d Umatrix;

    bool converged = math::diagonalizeSymmetricMatrix(m, Umatrix, eigenValues);
    if (!converged) {
        OPENVDB_THROW(ArithmeticError, "Diagonalization of the symmetric matrix failed");
    }

    UnitaryMap rotation(Umatrix);
    ScaleMap diagonal(eigenValues);
    CompoundMap<UnitaryMap, ScaleMap> first(rotation, diagonal);
   
    UnitaryMap rotationInv(Umatrix.transpose());
    return SymmetricMap::Ptr( new SymmetricMap(first, rotationInv));
}


PolarDecomposedMap::Ptr
createPolarDecomposedMap(const Mat3d& m)
{
    // Because our internal libary left-multiplies vectors against matrices
    // we are constructing  M  = Symmetric * Unitary instead of the more
    // standard M = Unitary * Symmetric
    Mat3d unitary, symmetric, mat3 = m.transpose();
    
    // factor mat3 = U * S  where U is unitary and S is symmetric
    bool gotPolar = math::polarDecomposition(mat3, unitary, symmetric);
    if (!gotPolar) {
        OPENVDB_THROW(ArithmeticError, "Polar decomposition of transform failed");
    }
    // put the result in a polar map and then copy it into the output polar
    UnitaryMap unitary_map(unitary.transpose());
    SymmetricMap::Ptr symmetric_map = createSymmetricMap(symmetric);
    
    return PolarDecomposedMap::Ptr(new PolarDecomposedMap(*symmetric_map, unitary_map));
}


FullyDecomposedMap::Ptr
createFullyDecomposedMap(const Mat4d& m)
{
    if (!isAffine(m)) {
        OPENVDB_THROW(ArithmeticError,
                 "4x4 Matrix initializing Decomposition map was not affine");
    }

    TranslationMap translate(m.getTranslation());
    PolarDecomposedMap::Ptr polar = createPolarDecomposedMap(m.getMat3());
    
    UnitaryAndTranslationMap rotationAndTranslate(polar->secondMap(), translate);

    return FullyDecomposedMap::Ptr(new FullyDecomposedMap(polar->firstMap(), rotationAndTranslate));
}


MapBase::Ptr
simplify(AffineMap::Ptr affine)
{
    if (affine->isScale()) { // can be simplified into a ScaleMap

        Vec3d scale = affine->applyMap(Vec3d(1,1,1));
        
        if (isApproxEqual(scale[0], scale[1]) && isApproxEqual(scale[0], scale[2])) {
            return MapBase::Ptr(new UniformScaleMap(scale[0]));
        } else {
            return MapBase::Ptr(new ScaleMap(scale));
        }
        
    } else if (affine->isScaleTranslate()) { // can be simplified into a ScaleTranslateMap

        Vec3d translate = affine->applyMap(Vec3d(0,0,0));
        Vec3d scale = affine->applyMap(Vec3d(1,1,1)) - translate;

        if (isApproxEqual(scale[0], scale[1]) && isApproxEqual(scale[0], scale[2])) {
            return MapBase::Ptr(new UniformScaleTranslateMap(scale[0], translate));
        } else {
            return MapBase::Ptr(new ScaleTranslateMap(scale, translate));
        }
    }

    // could not simplify the general Affine map.
    return boost::static_pointer_cast<MapBase, AffineMap>(affine);
}


////////////////////////////////////////


} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
