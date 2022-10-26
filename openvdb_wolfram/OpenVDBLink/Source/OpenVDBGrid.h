// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_HAS_BEEN_INCLUDED

#include "GlueTensors.h"

#include "OpenVDBCommon.h"
#include "LTemplate.h"

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tools/LevelSetUtil.h>

#include <cmath>
#include <ctime>
#include <vector>

using namespace openvdb;
using namespace openvdb::tools;
using namespace openvdbmma::types;
using namespace openvdbmma::utils;


namespace openvdbmma {


template<typename ValueType>
class OpenVDBGrid {

    static_assert(std::is_same<ValueMask, ValueType>::value ||
        std::is_arithmetic<typename VecTraits<ValueType>::ElementType>::value,
        "OpenVDBGrid requires a mask, numeric, or vector type");

    // bool and mask will return integers (0 or 1)
    using mmaBaseValT = std::conditional_t<
        std::is_floating_point<typename VecTraits<ValueType>::ElementType>::value,
        double, mint>;

    static const int ValueLength = VecTraits<ValueType>::IsVec ?
        (int)VecTraits<ValueType>::Size : 0;

    using mmaGridT = OpenVDBGrid<ValueType>;

    using ValueT     = ValueType;
    using wlTreeType = typename tree::Tree4<ValueT, 5, 4, 3>::Type;
    using wlGridType = Grid<wlTreeType>;
    using wlGridPtr  = typename wlGridType::Ptr;

    /* ------------ return types to MMA ------------ */

    using GlueScalar = std::conditional_t<ValueLength == 0,
        mmaBaseValT, mma::VectorRef<mmaBaseValT>>;

    using GlueVector = std::conditional_t<ValueLength == 0,
        mma::VectorRef<mmaBaseValT>, mma::MatrixRef<mmaBaseValT>>;

    using GlueMatrix = std::conditional_t<ValueLength == 0,
        mma::MatrixRef<mmaBaseValT>, mma::CubeRef<mmaBaseValT>>;

    using GlueCube = std::conditional_t<ValueLength == 0,
        mma::CubeRef<mmaBaseValT>, mma::TensorRef<mmaBaseValT>>;

    /* ------------ glue tensor classes ------------ */

    using GScalar = std::conditional_t<ValueLength == 0,
        openvdbmma::glue::GlueScalar0<mmaBaseValT>,
        openvdbmma::glue::GlueScalar1<mmaBaseValT, ValueLength>>;

    using GVector = std::conditional_t<ValueLength == 0,
        openvdbmma::glue::GlueVector0<mmaBaseValT>,
        openvdbmma::glue::GlueVector1<mmaBaseValT, ValueLength>>;

    using GMatrix = std::conditional_t<ValueLength == 0,
        openvdbmma::glue::GlueMatrix0<mmaBaseValT>,
        openvdbmma::glue::GlueMatrix1<mmaBaseValT, ValueLength>>;

    using GCube = std::conditional_t<ValueLength == 0,
        openvdbmma::glue::GlueCube0<mmaBaseValT>,
        openvdbmma::glue::GlueCube1<mmaBaseValT, ValueLength>>;

public:

    /* ------------ class constructors / destructors ------------ */

    OpenVDBGrid() { createEmptyGrid_internal(); }

    ~OpenVDBGrid() { destructVDB_internal(); }

    OpenVDBGrid(const mmaGridT& vdb) { copyVDB_internal(vdb); }

    /* ------------ creation / deletion ------------ */

    void createEmptyGrid() { createEmptyGrid_internal(); }

    void deleteGrid() { destructVDB_internal(); }

    void copyGrid(const mmaGridT& vdb) { copyVDB_internal(vdb); }

    inline wlGridPtr grid() const { return mGrid; }

    /* ------------ IO ------------ */

    const char* importVDBType(const char* file_path, const char* grid_name);

    bool importVDB(const char* file_path, const char* grid_name);

    void exportVDB(const char* file_path) const;

    /* ------------ scalar grid setters ------------ */

    void setActiveStates(mma::IntCoordinatesRef coords, mma::IntVectorRef states);

    void setBackgroundValue(GlueScalar bg);

    void setGridClass(mint grid_class);

    void setCreator(const char* creator);

    void setName(const char* name);

    void setValues(mma::IntCoordinatesRef coords, GlueVector vals);

    void setVoxelSize(double spacing);

    /* ------------ scalar grid getters ------------ */

    mma::IntVectorRef getActiveStates(mma::IntCoordinatesRef coords) const;

    mint getActiveLeafVoxelCount() const;

    inline mint getActiveTileCount() const;

    inline mint getActiveVoxelCount() const;

    GlueScalar getBackgroundValue() const;

    mint getGridClass() const;

    // not a const function due to WLString memory management
    const char* getCreator();

    mma::IntMatrixRef getGridBoundingBox() const;

    mma::IntVectorRef getGridDimensions() const;

    // not a const function due to WLString memory management
    const char* getGridType();

    inline double getHalfwidth() const;

    inline bool getHasUniformVoxels() const;

    inline bool getIsEmpty() const;

    inline mint getMemoryUsage() const;

    // returns a GlueVector
    mma::TensorRef<mmaBaseValT> getMinMaxValues() const;

    // not a const function due to WLString memory management
    const char* getName();

    // returns a GlueVector
    mma::TensorRef<mmaBaseValT> getValues(mma::IntCoordinatesRef coords) const;

    inline double getVoxelSize() const;

    /* ------------ Metadata ------------ */

    bool getBooleanMetadata(const char* key) const;

    mint getIntegerMetadata(const char* key) const;

    double getRealMetadata(const char* key) const;

    // not a const function due to WLString memory management
    const char* getStringMetadata(const char* key);

    void setBooleanMetadata(const char* key, bool value);

    void setStringMetadata(const char* key, const char* value);

    void setDescription(const char* description);

    /* ------------ CSG ------------ */

    void gridUnion(mmaGridT& vdb);

    void gridIntersection(mmaGridT& vdb);

    void gridDifference(mmaGridT& vdb);

    void gridUnionCopy(mma::IntTensorRef ids);

    void gridIntersectionCopy(mma::IntTensorRef ids);

    void gridDifferenceCopy(mmaGridT& vdb1, mmaGridT& vdb2);

    void gridMax(mmaGridT& vdb);

    void gridMin(mmaGridT& vdb);

    void clipGrid(mmaGridT& vdb, mma::RealBounds3DRef bds);

    /* ------------ level set creation ------------ */

    void ballLevelSet(mma::RealVectorRef center, double radius,
        double spacing, double halfWidth, bool is_signed = true);

    void cuboidLevelSet(mma::RealBounds3DRef bounds,
        double spacing, double halfWidth, bool is_signed = true);

    void meshLevelSet(mma::RealCoordinatesRef pts, mma::IntMatrixRef tri_cells,
        double spacing, double halfWidth, bool is_signed = true);

    void offsetSurfaceLevelSet(mma::RealCoordinatesRef pts, mma::IntMatrixRef tri_cells,
        double offset, double spacing, double width, bool is_signed = true);

    /* ------------ level set measure ------------ */

    double levelSetGridArea() const;

    mint levelSetGridEulerCharacteristic() const;

    mint levelSetGridGenus() const;

    double  levelSetGridVolume() const;

    /* ------------ distance measure ------------ */

    mma::IntVectorRef gridMember(mma::IntCoordinatesRef pts, double isovalue) const;

    mma::RealCoordinatesRef gridNearest(mma::RealCoordinatesRef pts, double isovalue) const;

    mma::RealVectorRef gridDistance(mma::RealCoordinatesRef pts, double isovalue) const;

    mma::RealVectorRef gridSignedDistance(mma::RealCoordinatesRef pts, double isovalue) const;

    mma::RealMatrixRef fillWithBalls(mint bmin, mint bmax, bool overlapping,
        float rmin, float rmax, float isovalue, mint instanceCount) const;

    /* ------------ filters ------------ */

    void filterGrid(mint filter_type, mint width = 1, mint iter = 1);

    /* ------------ mesh creation ------------ */

    mma::IntVectorRef meshCellCount(double isovalue, double adaptivity, bool tri_only) const;

    mma::RealVectorRef meshData(double isovalue = 0, double adaptivity = 0,
        bool tri_only = true) const;

    /* ------------ level set <--> fog volume ------------ */

    void levelSetToFogVolume(double cutoff = -1.0);

    /* ------------ grid transformation ------------ */

    void transformGrid(mmaGridT& target_vdb, mma::RealMatrixRef mat, mint resampling = 1);

    void scalarMultiply(double fac);

    void gammaAdjustment(double gamma);

    /* ------------ morphology ------------ */

    void resizeBandwidth(double width);

    void offsetLevelSet(double r);

    /* ------------ Image ------------ */

    mma::ImageRef<mma::im_real32_t> depthMap(mma::IntBounds3DRef bds,
        const double gamma = 1.0, const double imin = 0.0, const double imax = 1.0) const;

    mma::GenericImageRef gridSliceImage(const mint z, mma::IntBounds2DRef bds,
        const bool mirror_image = false, const bool threaded = true) const;

    mma::GenericImage3DRef gridImage3D(mma::IntBounds3DRef bds) const;

    /* ------------ render ------------ */

    mma::ImageRef<mma::im_byte_t> renderGrid(
        double isovalue, mma::RGBRef color, mma::RGBRef color2, mma::RGBRef color3, mma::RGBRef background,
        mma::RealVectorRef translate, mma::RealVectorRef lookat, mma::RealVectorRef up,
        mma::RealVectorRef range, mma::RealVectorRef fov, mint shader, mint camera, mint samples,
        mma::IntVectorRef resolution, double frame, mma::RealVectorRef depthParams,
        mma::RealVectorRef lightdir, mma::RealVectorRef step, bool is_closed
    ) const;

    mma::ImageRef<mma::im_byte_t> renderGridPBR(
        double isovalue, mma::RGBRef background,
        mma::RealVectorRef translate, mma::RealVectorRef lookat, mma::RealVectorRef up,
        mma::RealVectorRef range, mma::RealVectorRef fov, mint camera, mint samples,
        mma::IntVectorRef resolution, double frame, bool is_closed,
        mma::RGBRef baseColorFront, mma::RGBRef baseColorBack, mma::RGBRef baseColorClosed,
        double metallic, double rough, double ani, double ref,
        mma::RGBRef coatColor, double coatRough, double coatAni, double coatRef,
        double fac_spec, double fac_diff, double fac_coat
    ) const;

    mma::ImageRef<mma::im_byte_t> renderGridVectorColor(
        double isovalue, OpenVDBGrid<Vec3s> cGrid,
        OpenVDBGrid<Vec3s> cGrid2, OpenVDBGrid<Vec3s> cGrid3, mma::RGBRef background,
        mma::RealVectorRef translate, mma::RealVectorRef lookat, mma::RealVectorRef up,
        mma::RealVectorRef range, mma::RealVectorRef fov, mint shader, mint camera, mint samples,
        mma::IntVectorRef resolution, double frame, mma::RealVectorRef depthParams,
        mma::RealVectorRef lightdir, mma::RealVectorRef step, bool is_closed
    ) const;

    /* ------------ aggregate data ------------ */

    mma::IntTensorRef sliceVoxelCounts(const mint zmin, const mint zmax) const;

    // returns a GlueVector
    mma::TensorRef<mmaBaseValT> sliceVoxelValueTotals(const mint zmin, const mint zmax) const;

    mma::IntCubeRef activeTiles(mma::IntBounds3DRef bds, const bool partial_overlap) const;

    mma::SparseArrayRef<mmaBaseValT> activeVoxels(mma::IntBounds3DRef bds) const;

    mma::IntCoordinatesRef activeVoxelPositions(mma::IntBounds3DRef bds) const;

    // returns a GlueVector
    mma::TensorRef<mmaBaseValT> activeVoxelValues(mma::IntBounds3DRef bds) const;

    // returns a GlueMatrix
    mma::TensorRef<mmaBaseValT> gridSlice(const mint z, mma::IntBounds2DRef bds,
        const bool mirror_slice, const bool threaded) const;

    // returns a GlueCube
    mma::TensorRef<mmaBaseValT> gridData(mma::IntBounds3DRef bds) const;

protected:

    /* ------------ class getters / setters ------------ */

    inline void setGrid(wlGridPtr grid, bool init_meta = true)
    {
        mGrid = grid;
        if(init_meta) {
            initializeGridMetadata();
        } else {
            setLastModified();
        }
    }

    inline void setLastModified()
    {
        mGrid->insertMeta(META_LAST_MODIFIED, openvdb::Int64Metadata((long int)(time(NULL))));
    }

    inline char* WLString(std::string str)
    {
        const int n = str.length();
        //delete mString;

        mString = new char[n + 1];
        strcpy(mString, str.c_str());

        return mString;
    }

private:

    wlGridPtr mGrid;

    char* mString;

    inline void initializeGridMetadata()
    {
        time_t sec = time(NULL);
        mGrid->insertMeta(META_CREATED, openvdb::Int64Metadata((long int)sec));
        mGrid->insertMeta(META_LAST_MODIFIED, openvdb::Int64Metadata((long int)sec));

        mGrid->insertMeta(META_DESCRIPTION, openvdb::StringMetadata(""));
    }

    inline void createEmptyGrid_internal()
    {
        openvdbmma::utils::initialize();

        mGrid = wlGridType::create();

        initializeGridMetadata();

        mString = new char[1];
    }

    inline void destructVDB_internal()
    {
        // unloading the library sometimes segfaults, so maybe something is needed here?
    }

    inline void copyVDB_internal(const mmaGridT& vdb)
    {
        openvdb::initialize();
        destructVDB_internal();

        mGrid = (vdb.mGrid)->deepCopy();
        mString = strdup(vdb.mString);
    }

    inline bool valid_glueScalar(mmaBaseValT scalar) const
    {
        return true;
    }

    inline bool valid_glueScalar(mma::VectorRef<mmaBaseValT> scalar) const
    {
        return scalar.size() == ValueLength;
    }

    inline bool valid_glueVector(mma::VectorRef<mmaBaseValT> vec, const int n) const
    {
        return vec.size() == n;
    }

    inline bool valid_glueVector(mma::MatrixRef<mmaBaseValT> vec, const int n) const
    {
        return vec.rows() == n && vec.cols() == ValueLength;
    }

    inline wlGridPtr instanceGrid(mint id)
    {
        return mma::getInstance<OpenVDBGrid>(id).grid();
    }
};

} // namespace openvdbmma

#include "OpenVDBGrid/AggregateData.h"
#include "OpenVDBGrid/CSG.h"
#include "OpenVDBGrid/DistanceMeasure.h"
#include "OpenVDBGrid/Filter.h"
#include "OpenVDBGrid/FogVolume.h"
#include "OpenVDBGrid/Getters.h"
#include "OpenVDBGrid/Image.h"
#include "OpenVDBGrid/IO.h"
#include "OpenVDBGrid/LevelSetCreation.h"
#include "OpenVDBGrid/Measure.h"
#include "OpenVDBGrid/Mesh.h"
#include "OpenVDBGrid/Metadata.h"
#include "OpenVDBGrid/Morphology.h"
#include "OpenVDBGrid/Render.h"
#include "OpenVDBGrid/Setters.h"
#include "OpenVDBGrid/Transform.h"

#endif // OPENVDBLINK_OPENVDBGRID_HAS_BEEN_INCLUDED
