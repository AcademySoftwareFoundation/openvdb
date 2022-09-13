#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/LevelSetMeasure.h>


/* OpenVDBGrid public member function list
 
 double levelSetGridArea()
 
 mint levelSetGridEulerCharacteristic()
 
 mint levelSetGridGenus()
 
 double levelSetGridVolume()
 
 */


//////////// OpenVDBGrid public member function definitions

template<typename V>
double
OpenVDBGrid<V>::levelSetGridArea() const
{
    scalar_type_assert<V>();
    
    return levelSetArea(*grid(), true /* world space */);
}

template<typename V>
mint
OpenVDBGrid<V>::levelSetGridEulerCharacteristic() const
{
    scalar_type_assert<V>();
    
    return levelSetEulerCharacteristic(*grid());
}

template<typename V>
mint
OpenVDBGrid<V>::levelSetGridGenus() const
{
    scalar_type_assert<V>();
    
    return levelSetGenus(*grid());
}

template<typename V>
double
OpenVDBGrid<V>::levelSetGridVolume() const
{
    scalar_type_assert<V>();
    
    return levelSetVolume(*grid(), true /* world space */);
}
