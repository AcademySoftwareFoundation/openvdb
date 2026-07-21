#!/usr/bin/env python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import os

# If on Windows, add required dll directories from our binary build tree
if 'add_dll_directory' in dir(os):
    config = os.path.basename(os.getcwd())
    os.add_dll_directory(os.getcwd() + '\\..\\..\\..\\..\\openvdb\\openvdb\\' + config)
    for p in os.environ.get('PATH', '').split(os.pathsep):
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except OSError:
                pass

import nanovdb
import unittest
import tempfile


class TestVersion(unittest.TestCase):
    def test_operators(self):
        currentVersion = nanovdb.Version()
        initialVersion = nanovdb.Version(1, 0, 0)
        self.assertEqual(initialVersion.getMajor(), 1)
        self.assertEqual(initialVersion.getMinor(), 0)
        self.assertEqual(initialVersion.getPatch(), 0)
        self.assertTrue(currentVersion > initialVersion)
        self.assertFalse(currentVersion < initialVersion)
        self.assertFalse(currentVersion == initialVersion)
        self.assertTrue(initialVersion.age() < 0)

        alsoCurrentVersion = nanovdb.Version(currentVersion.id())
        self.assertTrue(currentVersion == alsoCurrentVersion)
        self.assertFalse(currentVersion > alsoCurrentVersion)
        self.assertTrue(currentVersion >= alsoCurrentVersion)
        self.assertFalse(currentVersion < alsoCurrentVersion)
        self.assertTrue(currentVersion <= alsoCurrentVersion)


class TestCoord(unittest.TestCase):
    def test_operators(self):
        ijk = nanovdb.math.Coord()
        self.assertEqual(ijk[0], 0)
        self.assertEqual(ijk[1], 0)
        self.assertEqual(ijk[2], 0)
        self.assertEqual(ijk.x, 0)
        self.assertEqual(ijk.y, 0)
        self.assertEqual(ijk.z, 0)

        ijk.x = 1
        ijk[1] = 2
        ijk.z += 3
        self.assertEqual(ijk, nanovdb.math.Coord(1, 2, 3))
        self.assertNotEqual(ijk, nanovdb.math.Coord(4))

    def test_hash(self):
        self.assertEqual(0, nanovdb.math.Coord(1, 2, 3).octant())
        self.assertEqual(0, nanovdb.math.Coord(1, 9, 3).octant())
        self.assertEqual(1, nanovdb.math.Coord(-1, 2, 3).octant())
        self.assertEqual(2, nanovdb.math.Coord(1, -2, 3).octant())
        self.assertEqual(3, nanovdb.math.Coord(-1, -2, 3).octant())
        self.assertEqual(4, nanovdb.math.Coord(1, 2, -3).octant())
        self.assertEqual(5, nanovdb.math.Coord(-1, 2, -3).octant())
        self.assertEqual(6, nanovdb.math.Coord(1, -2, -3).octant())
        self.assertEqual(7, nanovdb.math.Coord(-1, -2, -3).octant())


class TestVec3dBBox(unittest.TestCase):
    def test_functions(self):
        bbox = nanovdb.math.Vec3dBBox()
        self.assertTrue(bbox.empty())

        bbox.expand(nanovdb.math.Vec3d(57.0, -31.0, 60.0))
        self.assertTrue(bbox.empty())
        self.assertEqual(nanovdb.math.Vec3d(0.0), bbox.dim())
        self.assertEqual(57.0, bbox[0][0])
        self.assertEqual(-31.0, bbox[0][1])
        self.assertEqual(60.0, bbox[0][2])
        self.assertEqual(57.0, bbox[1][0])
        self.assertEqual(-31.0, bbox[1][1])
        self.assertEqual(60.0, bbox[1][2])

        bbox.expand(nanovdb.math.Vec3d(58.0, 0.0, 62.0))
        self.assertFalse(bbox.empty())
        self.assertEqual(nanovdb.math.Vec3d(1.0, 31.0, 2.0), bbox.dim())
        self.assertEqual(57.0, bbox[0][0])
        self.assertEqual(-31.0, bbox[0][1])
        self.assertEqual(60.0, bbox[0][2])
        self.assertEqual(58.0, bbox[1][0])
        self.assertEqual(0.0, bbox[1][1])
        self.assertEqual(62.0, bbox[1][2])


class TestCoordBBox(unittest.TestCase):
    def test_functions(self):
        bbox = nanovdb.math.CoordBBox()
        self.assertTrue(bbox.empty())

        bbox.expand(nanovdb.math.Coord(57, -31, 60))
        self.assertFalse(bbox.empty())
        self.assertEqual(nanovdb.math.Coord(1), bbox.dim())
        self.assertEqual(57, bbox[0][0])
        self.assertEqual(-31, bbox[0][1])
        self.assertEqual(60, bbox[0][2])
        self.assertEqual(57, bbox[1][0])
        self.assertEqual(-31, bbox[1][1])
        self.assertEqual(60, bbox[1][2])

        bbox.expand(nanovdb.math.Coord(58, 0, 62))
        self.assertFalse(bbox.empty())
        self.assertEqual(nanovdb.math.Coord(2, 32, 3), bbox.dim())
        self.assertEqual(57, bbox[0][0])
        self.assertEqual(-31, bbox[0][1])
        self.assertEqual(60, bbox[0][2])
        self.assertEqual(58, bbox[1][0])
        self.assertEqual(0, bbox[1][1])
        self.assertEqual(62, bbox[1][2])

    def test_convert(self):
        bbox = nanovdb.math.CoordBBox(
            nanovdb.math.Coord(57, -31, 60), nanovdb.math.Coord(58, 0, 62)
        )
        bbox2 = bbox.asDouble()
        self.assertFalse(bbox2.empty())
        self.assertEqual(nanovdb.math.Vec3d(57.0, -31.0, 60.0), bbox2.min)
        self.assertEqual(nanovdb.math.Vec3d(59.0, 1.0, 63.0), bbox2.max)

    def test_iterator(self):
        bbox = nanovdb.math.CoordBBox(
            nanovdb.math.Coord(57, -31, 60), nanovdb.math.Coord(58, 0, 62)
        )
        ijk = iter(bbox)
        for i in range(bbox.min[0], bbox.max[0] + 1):
            for j in range(bbox.min[1], bbox.max[1] + 1):
                for k in range(bbox.min[2], bbox.max[2] + 1):
                    self.assertTrue(iter)
                    coord = next(ijk)
                    self.assertTrue(bbox.isInside(coord))
                    self.assertEqual(coord.x, i)
                    self.assertEqual(coord.y, j)
                    self.assertEqual(coord.z, k)

    def test_create_cube(self):
        self.assertEqual(
            nanovdb.math.Coord(-7, -7, -7),
            nanovdb.math.CoordBBox.createCube(nanovdb.math.Coord(-7), 8).min,
        )
        self.assertEqual(
            nanovdb.math.Coord(0, 0, 0),
            nanovdb.math.CoordBBox.createCube(nanovdb.math.Coord(-7), 8).max,
        )
        self.assertEqual(
            nanovdb.math.Coord(-7, -7, -7), nanovdb.math.CoordBBox.createCube(-7, 0).min
        )
        self.assertEqual(
            nanovdb.math.Coord(0, 0, 0), nanovdb.math.CoordBBox.createCube(-7, 0).max
        )


class TestVec3(unittest.TestCase):
    def test_float(self):
        a = nanovdb.math.Vec3f()
        self.assertEqual(a, nanovdb.math.Vec3f(0.0))
        b = nanovdb.math.Vec3f(1.0)
        self.assertEqual(b[0], 1.0)
        self.assertEqual(b[1], 1.0)
        self.assertEqual(b[2], 1.0)
        c = a + b
        c[0] = 2.0
        c[1] = 3.0
        c[2] = 4.0

        self.assertEqual(a + b, b)
        self.assertEqual(b - c, -nanovdb.math.Vec3f(1.0, 2.0, 3.0))
        self.assertEqual(c * 5.0, nanovdb.math.Vec3f(10.0, 15.0, 20.0))
        self.assertEqual(b.cross(c), nanovdb.math.Vec3f(1.0, -2.0, 1.0))
        self.assertEqual(c.dot(c), c.lengthSqr())

    def test_double(self):
        a = nanovdb.math.Vec3d()
        self.assertEqual(a, nanovdb.math.Vec3d(0.0))
        b = nanovdb.math.Vec3d(1.0)
        self.assertEqual(b[0], 1.0)
        self.assertEqual(b[1], 1.0)
        self.assertEqual(b[2], 1.0)
        c = a + b
        c[0] = 2.0
        c[1] = 3.0
        c[2] = 4.0

        self.assertEqual(a + b, b)
        self.assertEqual(b - c, -nanovdb.math.Vec3d(1.0, 2.0, 3.0))
        self.assertEqual(c * 5.0, nanovdb.math.Vec3d(10.0, 15.0, 20.0))
        self.assertEqual(b.cross(c), nanovdb.math.Vec3d(1.0, -2.0, 1.0))
        self.assertEqual(c.dot(c), c.lengthSqr())


class TestVec4(unittest.TestCase):
    def test_float(self):
        a = nanovdb.math.Vec4f()
        self.assertEqual(a, nanovdb.math.Vec4f(0.0))
        b = nanovdb.math.Vec4f(1.0)
        self.assertEqual(b[0], 1.0)
        self.assertEqual(b[1], 1.0)
        self.assertEqual(b[2], 1.0)
        self.assertEqual(b[3], 1.0)
        c = a + b
        c[0] = 2.0
        c[1] = 3.0
        c[2] = 4.0
        c[3] = 5.0

        self.assertEqual(a + b, b)
        self.assertEqual(b - c, -nanovdb.math.Vec4f(1.0, 2.0, 3.0, 4.0))
        self.assertEqual(c * 5.0, nanovdb.math.Vec4f(10.0, 15.0, 20.0, 25.0))
        self.assertEqual(c.lengthSqr(), 54.0)

    def test_double(self):
        a = nanovdb.math.Vec4d()
        self.assertEqual(a, nanovdb.math.Vec4d(0.0))
        b = nanovdb.math.Vec4d(1.0)
        self.assertEqual(b[0], 1.0)
        self.assertEqual(b[1], 1.0)
        self.assertEqual(b[2], 1.0)
        self.assertEqual(b[3], 1.0)
        c = a + b
        c[0] = 2.0
        c[1] = 3.0
        c[2] = 4.0
        c[3] = 5.0

        self.assertEqual(a + b, b)
        self.assertEqual(b - c, -nanovdb.math.Vec4d(1.0, 2.0, 3.0, 4.0))
        self.assertEqual(c * 5.0, nanovdb.math.Vec4d(10.0, 15.0, 20.0, 25.0))
        self.assertEqual(c.lengthSqr(), 54.0)


class TestRgba8(unittest.TestCase):
    def test(self):
        a = nanovdb.math.Rgba8()
        self.assertEqual(a, nanovdb.math.Rgba8(0))
        b = nanovdb.math.Rgba8(127)
        self.assertEqual(b[0], 127)
        self.assertEqual(b[1], 127)
        self.assertEqual(b[2], 127)
        self.assertEqual(b[3], 127)
        b.r = 255
        b.g = 255
        b.b = 255
        b.a = 255
        self.assertEqual(b.packed, 2**32 - 1)
        self.assertEqual(b.asFloat(0), 1.0)
        self.assertEqual(b.asFloat(1), 1.0)
        self.assertEqual(b.asFloat(2), 1.0)
        self.assertEqual(b.asFloat(3), 1.0)
        self.assertEqual(b.asVec3f(), nanovdb.math.Vec3f(1.0))
        self.assertEqual(b.asVec4f(), nanovdb.math.Vec4f(1.0))


class TestMask(unittest.TestCase):
    def test_leaf_mask(self):
        self.assertEqual(8, nanovdb.LeafMask.wordCount())
        self.assertEqual(512, nanovdb.LeafMask.bitCount())
        self.assertEqual(8 * 8, nanovdb.LeafMask.memUsage())

        mask = nanovdb.LeafMask()
        self.assertEqual(0, mask.countOn())
        self.assertTrue(mask.isOff())
        self.assertFalse(mask.isOn())
        for i in range(nanovdb.LeafMask.bitCount()):
            self.assertFalse(mask.isOn(i))
            self.assertTrue(mask.isOff(i))

        for i in range(1000):
            self.assertEqual(512, mask.findNextOn(i))
            self.assertEqual(512, mask.findPrevOn(i))
            self.assertEqual(i if i < 512 else 512, mask.findNextOff(i))
            self.assertEqual(i if i < 512 else 512, mask.findPrevOff(i))

        mask.setOn(256)
        self.assertFalse(mask.isOff())
        self.assertFalse(mask.isOn())


class TestMap(unittest.TestCase):
    def test_functions(self):
        map1 = nanovdb.Map()
        map2 = nanovdb.Map()

        self.assertEqual(nanovdb.math.Vec3d(1.0), map1.getVoxelSize())
        map1.set(1.0, nanovdb.math.Vec3d(0.0))
        self.assertEqual(nanovdb.math.Vec3d(1.0), map1.getVoxelSize())
        map2.set(2.0, nanovdb.math.Vec3d(0.0))
        self.assertEqual(nanovdb.math.Vec3d(2.0), map2.getVoxelSize())
        map1 = map2
        self.assertEqual(nanovdb.math.Vec3d(2.0), map2.getVoxelSize())
        self.assertEqual(nanovdb.math.Vec3d(2.0), map1.getVoxelSize())


class TestGrid(unittest.TestCase):
    def test_float_grid(self):
        handle = nanovdb.tools.createFogVolumeSphere()
        self.assertEqual(handle.gridCount(), 1)
        for i in range(handle.gridCount()):
            self.assertTrue(handle.gridSize(i) > 0)
            self.assertEqual(handle.gridType(i), nanovdb.GridType.Float)
            grid = handle.grid(i)
            self.assertIsNotNone(grid)
            accessor = nanovdb.FloatReadAccessor(grid)
            coord = nanovdb.math.Coord(0)
            self.assertEqual(accessor.getValue(0, 0, 0), accessor.getValue(coord))
            self.assertEqual(accessor.getValue(coord), accessor(coord))
            self.assertEqual(accessor(coord), accessor(0, 0, 0))
            self.assertTrue(accessor.isActive(coord))
            nodeInfo = accessor.getNodeInfo(coord)
            self.assertTrue(nodeInfo.bbox.isInside(coord))
            self.assertEqual(accessor.probeValue(coord), (1.0, True))

    def test_checksum(self):
        handle = nanovdb.tools.createFogVolumeTorus()
        self.assertEqual(handle.gridCount(), 1)
        for i in range(handle.gridCount()):
            self.assertTrue(handle.gridSize(i) > 0)
            self.assertEqual(handle.gridType(i), nanovdb.GridType.Float)
            grid = handle.grid(i)
            self.assertIsNotNone(grid)
            checksum = grid.checksum()
            nanovdb.tools.updateChecksum(grid, nanovdb.CheckMode.Default)
            updatedChecksum = grid.checksum()
            self.assertEqual(checksum, updatedChecksum)


class TestGridHandleExchange(unittest.TestCase):
    def test_list_to_vector(self):
        handle = nanovdb.tools.createLevelSetTorus(nanovdb.GridType.Double)
        self.assertEqual(handle.gridCount(), 1)
        self.assertIsNotNone(handle.grid())
        handles = [handle, handle]
        dstFile = tempfile.NamedTemporaryFile(delete=False)
        dstFile.close()
        try:
            nanovdb.io.writeGrids(dstFile.name, handles)
            # Verify that both grids actually made it into the file. Previously
            # writeGrids re-opened the file per handle with std::ios::trunc, so
            # only the final grid was retained on disk.
            metadata = nanovdb.io.readGridMetaData(dstFile.name)
            self.assertEqual(len(metadata), len(handles))
            readHandles = nanovdb.io.readGrids(dstFile.name)
            self.assertEqual(len(readHandles), len(handles))
            for readHandle in readHandles:
                self.assertEqual(readHandle.gridCount(), 1)
                self.assertEqual(readHandle.gridType(0), nanovdb.GridType.Double)
                self.assertIsInstance(readHandle.grid(), nanovdb.DoubleGrid)
        finally:
            os.unlink(dstFile.name)


class TestPolymorphicGridAccess(unittest.TestCase):
    """handle.grid(n) returns the correct typed Grid subclass selected by
    gridType(n); the legacy per-type accessors (floatGrid(), etc.) are not
    bound."""

    def test_float_grid(self):
        h = nanovdb.tools.createFogVolumeSphere()
        self.assertIsInstance(h.grid(), nanovdb.FloatGrid)
        self.assertEqual(h.grid().gridType(), nanovdb.GridType.Float)

    def test_double_grid(self):
        h = nanovdb.tools.createLevelSetTorus(nanovdb.GridType.Double)
        self.assertIsInstance(h.grid(), nanovdb.DoubleGrid)
        self.assertEqual(h.grid().gridType(), nanovdb.GridType.Double)

    def test_out_of_range_returns_none(self):
        h = nanovdb.tools.createFogVolumeSphere()
        self.assertIsNone(h.grid(99))

    def test_empty_handle_returns_none(self):
        self.assertIsNone(nanovdb.GridHandle().grid())

    def test_typed_accessors_removed(self):
        # The legacy per-type accessors were replaced by handle.grid(n).
        h = nanovdb.tools.createFogVolumeSphere()
        self.assertFalse(hasattr(h, "floatGrid"))
        self.assertFalse(hasattr(h, "doubleGrid"))
        self.assertFalse(hasattr(h, "int32Grid"))
        self.assertFalse(hasattr(h, "vec3fGrid"))
        self.assertFalse(hasattr(h, "rgba8Grid"))


class TestGridBase(unittest.TestCase):
    """Methods that don't depend on BuildT (gridType, gridClass, voxelSize,
    isLevelSet/...) resolve via the Grid base class shared by every typed
    grid subclass."""

    def test_grid_base_class_name(self):
        # Typed grids inherit from a base class named "Grid" (no more "GridData").
        self.assertTrue(any(b.__name__ == "Grid" for b in nanovdb.FloatGrid.__bases__))
        self.assertFalse(hasattr(nanovdb, "GridData"))

    def test_lifted_methods_accessible_via_inheritance(self):
        h = nanovdb.tools.createFogVolumeSphere(name="probe")
        g = h.grid()
        self.assertEqual(g.gridType(), nanovdb.GridType.Float)
        self.assertEqual(g.gridClass(), nanovdb.GridClass.FogVolume)
        self.assertTrue(g.isFogVolume())
        self.assertFalse(g.isLevelSet())
        self.assertEqual(g.gridName(), "probe")
        self.assertEqual(g.shortGridName(), "probe")
        self.assertGreater(g.gridSize(), 0)
        self.assertEqual(g.gridCount(), 1)


class TestGridMetaData(unittest.TestCase):
    """nanovdb.GridMetaData is a type-erased introspector — construct from a
    Grid and query gridType/gridClass/voxelSize/etc. without knowing
    BuildT."""

    def test_constructed_from_grid(self):
        h = nanovdb.tools.createFogVolumeSphere(name="probe")
        m = nanovdb.GridMetaData(h.grid())
        self.assertEqual(m.gridType(), nanovdb.GridType.Float)
        self.assertEqual(m.gridClass(), nanovdb.GridClass.FogVolume)
        self.assertEqual(m.shortGridName(), "probe")
        self.assertTrue(m.isValid())
        self.assertTrue(m.isFogVolume())
        self.assertGreater(m.activeVoxelCount(), 0)
        self.assertEqual(m.blindDataCount(), 0)
        self.assertTrue(nanovdb.GridMetaData.safeCast(h.grid()))


class TestBlindDataEmpty(unittest.TestCase):
    """Blind data API (blindDataCount, blindMetaData, findBlindData,
    findBlindDataForSemantic, getBlindData) returns sensible None/-1
    sentinels on grids that have no blind data channels."""

    def test_no_blind_data(self):
        h = nanovdb.tools.createFogVolumeSphere()
        g = h.grid()
        self.assertEqual(g.blindDataCount(), 0)
        self.assertIsNone(g.blindMetaData(0))
        self.assertEqual(g.findBlindData("anything"), -1)
        self.assertEqual(
            g.findBlindDataForSemantic(nanovdb.GridBlindDataSemantic.PointPosition),
            -1,
        )
        self.assertIsNone(g.getBlindData(0))


class TestSplitMergeCopy(unittest.TestCase):
    """splitGrids(h) -> list of single-grid handles, mergeGrids([h1, h2])
    -> combined handle, h.copy() -> deep buffer copy. mergeGrids must not
    consume its input handles."""

    def test_split_and_merge_roundtrip(self):
        h1 = nanovdb.tools.createFogVolumeSphere(name="a")
        h2 = nanovdb.tools.createLevelSetTorus(nanovdb.GridType.Float, name="b")
        merged = nanovdb.mergeGrids([h1, h2])
        self.assertEqual(merged.gridCount(), 2)
        split = nanovdb.splitGrids(merged)
        self.assertEqual(len(split), 2)
        for s in split:
            self.assertEqual(s.gridCount(), 1)

    def test_merge_does_not_consume_inputs(self):
        # Regression: a previous mergeGrids implementation used
        # nb::cast<HandleT&&> which moved the underlying C++ handle out of
        # the Python wrapper, silently emptying h1/h2 after the call. The
        # binding now reads each handle by const reference.
        h1 = nanovdb.tools.createFogVolumeSphere(name="a")
        h2 = nanovdb.tools.createLevelSetTorus(nanovdb.GridType.Float, name="b")
        sz1, sz2 = h1.size(), h2.size()
        gc1, gc2 = h1.gridCount(), h2.gridCount()

        merged = nanovdb.mergeGrids([h1, h2])

        self.assertEqual(h1.gridCount(), gc1)
        self.assertEqual(h2.gridCount(), gc2)
        self.assertEqual(h1.size(), sz1)
        self.assertEqual(h2.size(), sz2)
        # merged still works
        self.assertEqual(merged.gridCount(), 2)
        self.assertEqual(merged.grid(0).gridName(), "a")
        self.assertEqual(merged.grid(1).gridName(), "b")

    def test_copy_is_deep(self):
        src = nanovdb.tools.createFogVolumeSphere(name="orig")
        cp = src.copy()
        self.assertIsNot(src, cp)
        self.assertEqual(cp.gridCount(), src.gridCount())
        self.assertEqual(cp.grid().gridName(), "orig")


class TestBuildTRegistrations(unittest.TestCase):
    """Every BuildT we bind exposes the right shape — a Grid class, a
    ReadAccessor, and (for arithmetic-valued scalars) a NodeInfo. Accessor
    surface depends on the type kind: scalar accessors have setVoxel +
    getNodeInfo; vector accessors have setVoxel only; read-only accessors
    (Boolean, Fp*, Index, OnIndex, Mask) have neither."""

    SCALARS = ["Int16", "Int64", "UInt8", "UInt32"]
    VECTORS = ["Vec3d", "Vec4f", "Vec4d", "Vec3u8", "Vec3u16"]
    READONLY = ["Boolean", "Fp4", "Fp8", "Fp16", "FpN", "Index", "OnIndex", "Mask"]

    def test_all_grid_classes_registered(self):
        for suffix in self.SCALARS + self.VECTORS + self.READONLY:
            cls = getattr(nanovdb, suffix + "Grid", None)
            self.assertIsNotNone(cls, f"{suffix}Grid missing")
            # All inherit from the type-erased Grid base.
            self.assertIn(nanovdb.Grid, cls.__mro__)

    def test_all_accessors_registered(self):
        for suffix in self.SCALARS + self.VECTORS + self.READONLY:
            acc = getattr(nanovdb, suffix + "ReadAccessor", None)
            self.assertIsNotNone(acc, f"{suffix}ReadAccessor missing")

    def test_scalar_accessors_have_setvoxel_and_nodeinfo(self):
        for suffix in self.SCALARS:
            acc = getattr(nanovdb, suffix + "ReadAccessor")
            self.assertTrue(hasattr(acc, "setVoxel"),
                            f"{suffix}ReadAccessor missing setVoxel")
            self.assertTrue(hasattr(acc, "getNodeInfo"),
                            f"{suffix}ReadAccessor missing getNodeInfo")
            self.assertIsNotNone(getattr(nanovdb, suffix + "NodeInfo", None),
                                 f"{suffix}NodeInfo missing")

    def test_vector_accessors_have_setvoxel_no_nodeinfo(self):
        # The newer vector accessor names follow the consistent
        # <Suffix>ReadAccessor pattern (Vec3dReadAccessor, ...); the
        # original Vec3f one is named Vec3fReadVectorAccessor for backwards
        # compatibility.
        for suffix in self.VECTORS:
            acc = getattr(nanovdb, suffix + "ReadAccessor")
            self.assertTrue(hasattr(acc, "setVoxel"))
            self.assertFalse(hasattr(acc, "getNodeInfo"))

    def test_all_grid_type_enums_reachable(self):
        # GridType enum binding must cover every BuildT we register —
        # otherwise Python users can't compare against handle.gridType(n).
        # Locks in the full set; missing entries (e.g. an unbound enumerator
        # for a freshly-added BuildT) get caught here.
        for name in ["Float", "Double", "Int16", "Int32", "Int64", "UInt8",
                     "UInt32", "Boolean", "Half", "RGBA8", "Vec3f", "Vec3d",
                     "Vec4f", "Vec4d", "Vec3u8", "Vec3u16", "Mask", "Fp4",
                     "Fp8", "Fp16", "FpN", "Index", "OnIndex", "PointIndex"]:
            self.assertTrue(hasattr(nanovdb.GridType, name),
                            f"nanovdb.GridType.{name} not bound")

    def test_readonly_accessors_have_neither_setvoxel_nor_nodeinfo(self):
        # Quantized types decode to float on read but don't bind setVoxel;
        # index types return uint64; ValueMask exposes only active-state
        # queries. All share a bare ReadAccessor.
        for suffix in self.READONLY:
            acc = getattr(nanovdb, suffix + "ReadAccessor")
            self.assertFalse(hasattr(acc, "setVoxel"),
                             f"{suffix}ReadAccessor should not have setVoxel")
            self.assertFalse(hasattr(acc, "getNodeInfo"),
                             f"{suffix}ReadAccessor should not have getNodeInfo")


class TestTreeNodeWalking(unittest.TestCase):
    """Walk a grid's tree from Python: Grid.tree(), Root/Upper/Lower/Leaf
    node access and metadata, per-leaf zero-copy values() and bulk
    grid.leaf_values() NumPy views, NodeManager + createNodeManager."""

    @classmethod
    def setUpClass(cls):
        cls.h = nanovdb.tools.createFogVolumeSphere(name="probe")
        cls.g = cls.h.grid()
        cls.tree = cls.g.tree()

    def test_grid_tree_basic(self):
        self.assertIsInstance(self.tree, nanovdb.FloatTree)
        self.assertEqual(self.tree.background(), 3.0)  # halfwidth*voxelsize default
        self.assertGreater(self.tree.activeVoxelCount(), 0)
        self.assertGreaterEqual(self.tree.totalNodeCount(), self.tree.nodeCount(0))

    def test_extrema(self):
        mn, mx = self.tree.extrema()
        # FogVolumeSphere produces values in [0, 1].
        self.assertGreaterEqual(mn, 0.0)
        self.assertLessEqual(mx, 1.0)
        self.assertLessEqual(mn, mx)

    def test_first_leaf_and_node_metadata(self):
        leaf = self.tree.getFirstLeaf()
        self.assertIsInstance(leaf, nanovdb.FloatLeaf)
        self.assertEqual(nanovdb.FloatLeaf.dim(), 8)
        self.assertEqual(nanovdb.FloatLeaf.voxelCount(), 512)
        # Origin should be aligned to LeafNode dim=8.
        for c in (leaf.origin().x, leaf.origin().y, leaf.origin().z):
            self.assertEqual(c % 8, 0)

    def test_root_metadata(self):
        root = self.tree.root()
        self.assertIsInstance(root, nanovdb.FloatRoot)
        self.assertGreater(root.tileCount(), 0)
        # Root bbox covers ALL active voxels — non-empty for a fog sphere.
        bb = root.bbox()
        self.assertFalse(bb.empty())
        self.assertEqual(root.background(), self.tree.background())

    def test_leaf_values_zero_copy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        leaf = self.tree.getFirstLeaf()
        vals = leaf.values()
        self.assertEqual(vals.shape, (512,))
        self.assertEqual(vals.dtype, np.float32)
        # Mutation through the view writes back into the grid buffer.
        original = float(vals[0])
        vals[0] = original + 1.0
        self.assertAlmostEqual(float(leaf.getValue(0)), original + 1.0)
        vals[0] = original  # restore

    def test_leaf_values_unavailable_for_special_buildts(self):
        # ValueIndex / ValueMask / bool / Fp* leaves don't carry T mValues[512],
        # so the `values` accessor is not bound for them.
        for cls_name in ("BooleanLeaf", "Fp4Leaf", "IndexLeaf", "MaskLeaf"):
            leaf_cls = getattr(nanovdb, cls_name)
            self.assertFalse(hasattr(leaf_cls, "values"),
                             f"{cls_name}.values should not be bound")

    def test_bulk_leaf_values(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        bulk = self.g.leaf_values()
        self.assertEqual(bulk.shape, (self.tree.nodeCount(0), 512))
        self.assertEqual(bulk.dtype, np.float32)
        # First row should match per-leaf values().
        first_via_bulk = np.asarray(bulk[0])
        first_via_leaf = np.asarray(self.tree.getFirstLeaf().values())
        self.assertTrue(np.array_equal(first_via_bulk, first_via_leaf))

    def test_bulk_leaf_values_empty_grid_returns_empty_array(self):
        # A grid with no leaves returns an empty (0, 512) NumPy view rather
        # than None, so callers can iterate / shape-test without a sentinel.
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        empty_bbox = nanovdb.math.CoordBBox()  # default-constructed = empty
        empty_h = nanovdb.tools.createFloatGrid(
            0.0, "empty", nanovdb.GridClass.Unknown,
            lambda ijk: 0.0, empty_bbox)
        bulk = empty_h.grid().leaf_values()
        self.assertEqual(bulk.shape, (0, 512))
        self.assertEqual(bulk.dtype, np.float32)

    def test_node_manager_round_trip(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        handle = nanovdb.createNodeManager(self.g)
        self.assertGreater(handle.size(), 0)
        self.assertTrue(bool(handle))
        nm = handle.mgr()
        self.assertIsInstance(nm, nanovdb.FloatNodeManager)
        self.assertTrue(nm.isLinear())  # createNanoGrid produces breadth-first
        self.assertEqual(nm.leafCount(), self.tree.nodeCount(0))
        self.assertEqual(nm.lowerCount(), self.tree.nodeCount(1))
        self.assertEqual(nm.upperCount(), self.tree.nodeCount(2))
        # NodeManager.leaf(0) should be the same leaf as tree.getFirstLeaf()
        # (breadth-first order).
        self.assertEqual(nm.leaf(0).origin(), self.tree.getFirstLeaf().origin())
        self.assertTrue(np.array_equal(
            nm.leaf(0).values(), self.tree.getFirstLeaf().values()))


class TestBoundsChecks(unittest.TestCase):
    """Out-of-range indices on Leaf / Tree / NodeManager raise Python
    exceptions rather than falling through into raw memory access. The
    underlying C++ uses NANOVDB_ASSERT which is a no-op in release builds,
    so the Python layer guards every entry point that takes a level or
    index argument.
    """

    @classmethod
    def setUpClass(cls):
        cls.h = nanovdb.tools.createFogVolumeSphere()
        cls.g = cls.h.grid()
        cls.tree = cls.g.tree()
        cls.leaf = cls.tree.getFirstLeaf()
        cls.nm_handle = nanovdb.createNodeManager(cls.g)
        cls.nm = cls.nm_handle.mgr()

    def test_leaf_offset_bounds(self):
        n = nanovdb.FloatLeaf.voxelCount()
        with self.assertRaises(IndexError):
            self.leaf.isActive(n)
        with self.assertRaises(IndexError):
            self.leaf.isActive(n + 1000)
        with self.assertRaises(IndexError):
            self.leaf.getValue(n)
        # In-range still works.
        self.assertIsNotNone(self.leaf.getValue(0))
        self.assertIsNotNone(self.leaf.getValue(n - 1))

    def test_tree_active_tile_count_level(self):
        # activeTileCount levels are 1..3 (level 0 is leaves, not tiles).
        with self.assertRaises(ValueError):
            self.tree.activeTileCount(0)
        with self.assertRaises(ValueError):
            self.tree.activeTileCount(4)
        # In-range still works.
        self.assertGreaterEqual(self.tree.activeTileCount(3), 0)

    def test_tree_node_count_level(self):
        # nodeCount levels are 0..2 (leaf / lower / upper).
        with self.assertRaises(ValueError):
            self.tree.nodeCount(-1)
        with self.assertRaises(ValueError):
            self.tree.nodeCount(3)
        self.assertGreater(self.tree.nodeCount(0), 0)

    def test_node_manager_indexed_access(self):
        with self.assertRaises(IndexError):
            self.nm.leaf(self.nm.leafCount())
        with self.assertRaises(IndexError):
            self.nm.lower(self.nm.lowerCount())
        with self.assertRaises(IndexError):
            self.nm.upper(self.nm.upperCount())
        with self.assertRaises(ValueError):
            self.nm.nodeCount(3)
        # In-range still works.
        self.assertEqual(self.nm.leaf(0).origin(), self.leaf.origin())


class TestZeroCopyViewLifetimes(unittest.TestCase):
    """Returned typed grids, trees, leaves, NodeManagers, and zero-copy
    NumPy views must keep their backing buffers alive across the chained
    temporary expressions used at the call site (e.g.
    `nanovdb.tools.createFogVolumeSphere().grid().tree().getFirstLeaf().values()`).
    Without explicit nb::keep_alive linkages the intermediate handle gets
    GC'd between expressions and the returned object reads freed memory.
    """

    def _force_gc(self):
        import gc
        for _ in range(3):
            gc.collect()

    def test_handle_grid_temporary(self):
        g = nanovdb.tools.createFogVolumeSphere(name="probe").grid()
        self._force_gc()
        self.assertEqual(g.gridName(), "probe")

    def test_handle_grid_tree_leaf_values_chain(self):
        # nb::ndarray<nb::numpy, ...> requires numpy at runtime, so the
        # binding raises TypeError if numpy isn't installed. Skip then.
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        vals = (nanovdb.tools.createFogVolumeSphere()
                .grid().tree().getFirstLeaf().values())
        self._force_gc()
        # Touching the view shouldn't crash.
        self.assertEqual(vals.shape, (512,))
        _ = float(vals[0])

    def test_grid_leaf_values_temporary(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        bulk = nanovdb.tools.createFogVolumeSphere().grid().leaf_values()
        self._force_gc()
        self.assertEqual(bulk.shape[1], 512)
        _ = float(bulk[0, 0])

    def test_node_manager_temporary_grid(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        nm = nanovdb.createNodeManager(
            nanovdb.tools.createFogVolumeSphere().grid()).mgr()
        self._force_gc()
        self.assertGreater(nm.leafCount(), 0)
        leaf0_vals = nm.leaf(0).values()
        self._force_gc()
        self.assertEqual(leaf0_vals.shape, (512,))

    def test_blind_data_temporary(self):
        # The grid has no blind data so getBlindData returns None — what we
        # care about here is that the temporary chain doesn't segfault.
        result = nanovdb.tools.createFogVolumeSphere().grid().getBlindData(0)
        self._force_gc()
        self.assertIsNone(result)


class TestVoxelBlockManager(unittest.TestCase):
    """nanovdb.tools.buildVoxelBlockManager + VoxelBlockManagerHandle +
    decodeInverseMaps and the createOnIndexGrid test-scaffold factory.

    NOTE: end-to-end decode verification across every block is intentionally
    deferred until the Phase 4 build::Grid bindings land. The C++
    buildVoxelBlockManager has an algorithmic gap when the source OnIndex
    grid is tile-compressed (blocks not reached by any leaf's iteration
    sweep are left with uninitialized firstLeafID). The current host-side
    createFloatGrid + createOnIndexGrid path triggers tile compression on
    uniform regions, so we only exercise decodeBlock(0) here — that block
    is guaranteed to be covered when the grid's firstOffset is 1. The
    bindings include a defensive check that raises ValueError if a
    user hits the uninitialized-block case rather than crashing.
    """

    def _make_cube_on_index_grid(self):
        # 21^3 fully-active cube — matches the C++ unit test's input.
        bbox = nanovdb.math.CoordBBox(
            nanovdb.math.Coord(125), nanovdb.math.Coord(145))
        float_h = nanovdb.tools.createFloatGrid(
            0.0, "cube", nanovdb.GridClass.Unknown,
            lambda ijk: 1.0, bbox)
        return nanovdb.tools.createOnIndexGrid(
            float_h.grid(), include_stats=False, include_tiles=False)

    def test_create_on_index_grid(self):
        h = self._make_cube_on_index_grid()
        g = h.grid()
        self.assertEqual(g.gridType(), nanovdb.GridType.OnIndex)
        self.assertEqual(g.gridClass(), nanovdb.GridClass.IndexGrid)
        self.assertGreater(g.activeVoxelCount(), 0)
        self.assertTrue(g.isSequential())

    def test_create_on_index_grid_rejects_unsupported_source(self):
        # createOnIndexGrid only accepts {float, double, int32, Vec3f}
        # source grids; passing None (or any non-grid object) should raise
        # TypeError at the first BuildT-isinstance check.
        with self.assertRaises(TypeError):
            nanovdb.tools.createOnIndexGrid(None)

    def test_build_voxel_block_manager_handle(self):
        h = self._make_cube_on_index_grid()
        g = h.grid()
        vbm = nanovdb.tools.buildVoxelBlockManager(g, log2_block_width=6)
        self.assertGreater(vbm.blockCount(), 0)
        self.assertEqual(vbm.firstOffset(), 1)
        self.assertEqual(vbm.lastOffset(), g.activeVoxelCount())
        self.assertTrue(bool(vbm))

    def test_buffers_zero_copy_shape_and_dtype(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        h = self._make_cube_on_index_grid()
        # log2_block_width=6 -> JumpMapLength=1.
        vbm6 = nanovdb.tools.buildVoxelBlockManager(h.grid(), log2_block_width=6)
        self.assertEqual(vbm6.log2_block_width, 6)
        self.assertEqual(vbm6.block_width, 64)
        self.assertEqual(vbm6.jump_map_length, 1)
        fl = vbm6.firstLeafID()
        self.assertEqual(fl.shape, (vbm6.blockCount(),))
        self.assertEqual(fl.dtype, np.uint32)
        jm = vbm6.jumpMap()
        self.assertEqual(jm.shape, (vbm6.blockCount(), 1))
        self.assertEqual(jm.dtype, np.uint64)
        # log2_block_width=7 -> JumpMapLength=2 (independent build, separate
        # allocation; the jumpMap shape comes from the handle, not the caller).
        vbm7 = nanovdb.tools.buildVoxelBlockManager(h.grid(), log2_block_width=7)
        self.assertEqual(vbm7.jump_map_length, 2)
        self.assertEqual(vbm7.jumpMap().shape, (vbm7.blockCount(), 2))

    def test_decode_block_zero(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        h = self._make_cube_on_index_grid()
        g = h.grid()
        vbm = nanovdb.tools.buildVoxelBlockManager(g, log2_block_width=6)
        leaf_index, voxel_offset = vbm.decodeBlock(g, 0)
        self.assertEqual(leaf_index.shape, (64,))
        self.assertEqual(leaf_index.dtype, np.uint32)
        self.assertEqual(voxel_offset.shape, (64,))
        self.assertEqual(voxel_offset.dtype, np.uint16)
        # Free function should produce the same result for the same input.
        fl = np.asarray(vbm.firstLeafID())
        jm = np.asarray(vbm.jumpMap())
        li_free, vo_free = nanovdb.tools.decodeInverseMaps(
            g, int(fl[0]), jm[0], vbm.firstOffset(), log2_block_width=6)
        self.assertTrue(np.array_equal(leaf_index, li_free))
        self.assertTrue(np.array_equal(voxel_offset, vo_free))

    def test_decode_block_out_of_range(self):
        h = self._make_cube_on_index_grid()
        g = h.grid()
        vbm = nanovdb.tools.buildVoxelBlockManager(g)
        with self.assertRaises(IndexError):
            vbm.decodeBlock(g, vbm.blockCount())

    def test_log2_block_width_out_of_range(self):
        h = self._make_cube_on_index_grid()
        g = h.grid()
        with self.assertRaises(ValueError):
            nanovdb.tools.buildVoxelBlockManager(g, log2_block_width=5)
        with self.assertRaises(ValueError):
            nanovdb.tools.buildVoxelBlockManager(g, log2_block_width=10)

    def test_build_voxel_block_manager_rejects_misaligned_first_offset(self):
        # first_offset must satisfy first_offset == 1 (mod BlockWidth).
        # For log2_block_width=6, BlockWidth=64, so 1, 65, 129, ... are valid
        # but 2 is not.
        h = self._make_cube_on_index_grid()
        g = h.grid()
        with self.assertRaises(ValueError):
            nanovdb.tools.buildVoxelBlockManager(
                g, log2_block_width=6, first_offset=2)
        # And the wider-block case: log2_block_width=7 -> BlockWidth=128,
        # first_offset=65 is valid for width=6 but misaligned for width=7.
        with self.assertRaises(ValueError):
            nanovdb.tools.buildVoxelBlockManager(
                g, log2_block_width=7, first_offset=65)

    def test_decode_inverse_maps_rejects_bad_first_leaf_id(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        h = self._make_cube_on_index_grid()
        g = h.grid()
        vbm = nanovdb.tools.buildVoxelBlockManager(g, log2_block_width=6)
        jm0 = np.asarray(vbm.jumpMap())[0]
        n_leaves = g.tree().nodeCount(0)
        with self.assertRaises(IndexError):
            nanovdb.tools.decodeInverseMaps(
                g, n_leaves, jm0, vbm.firstOffset(), log2_block_width=6)

    def test_build_voxel_block_manager_rejects_undersized_n_blocks(self):
        # Caller-supplied n_blocks must hold at least
        # ceil((last_offset - first_offset + 1) / BlockWidth) blocks;
        # smaller values would silently truncate coverage.
        h = self._make_cube_on_index_grid()
        g = h.grid()
        # The cube grid has ~9261 active voxels, so at log2_block_width=6
        # (BlockWidth=64) the minimum is roughly 145 blocks. Passing 1
        # must be rejected.
        with self.assertRaises(ValueError):
            nanovdb.tools.buildVoxelBlockManager(
                g, log2_block_width=6, n_blocks=1)

    def test_build_voxel_block_manager_rejects_non_on_index_grid(self):
        # FloatGrid is not an OnIndexGrid.
        h_float = nanovdb.tools.createFogVolumeSphere()
        with self.assertRaises(TypeError):
            nanovdb.tools.buildVoxelBlockManager(h_float.grid())

    def test_untouched_blocks_trip_sentinel_guard(self):
        # Build the cube VBM and probe every block. The Python binding
        # prefills firstLeafID with a sentinel (== nLeaves) before calling
        # the in-place builder, so any block the upstream algorithm doesn't
        # touch deterministically trips the firstLeafID >= nLeaves guard in
        # decodeBlock (raising ValueError). The sweep must therefore see
        # only two outcomes per block: a successful decode or a sentinel
        # ValueError — no segfaults, no IndexError (those would indicate
        # block_index out of range, not sentinel), and no silent wrong
        # decode.
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        h = self._make_cube_on_index_grid()
        g = h.grid()
        vbm = nanovdb.tools.buildVoxelBlockManager(g, log2_block_width=6)
        n_leaves = g.tree().nodeCount(0)
        fl = vbm.firstLeafID()
        for b in range(vbm.blockCount()):
            slot = int(fl[b])
            # Slot must be a real leaf id or the sentinel — never garbage.
            self.assertTrue(slot < n_leaves or slot == n_leaves,
                f"block {b}: firstLeafID={slot} is neither a real leaf id "
                f"(< {n_leaves}) nor the sentinel (== {n_leaves}); "
                "uninitialized memory leaked through.")
            if slot >= n_leaves:
                with self.assertRaises(ValueError):
                    vbm.decodeBlock(g, b)
            else:
                # Successful decode path — just confirm the shapes.
                li, vo = vbm.decodeBlock(g, b)
                self.assertEqual(li.shape, (64,))
                self.assertEqual(vo.shape, (64,))

    def test_default_constructed_handle_returns_empty_arrays(self):
        # A default-constructed VoxelBlockManagerHandle has null backing
        # buffers; firstLeafID() and jumpMap() should still return empty
        # ndarrays rather than crash on the null pointer.
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        vbm = nanovdb.tools.VoxelBlockManagerHandle()
        self.assertEqual(vbm.blockCount(), 0)
        self.assertFalse(bool(vbm))
        fl = np.asarray(vbm.firstLeafID())
        self.assertEqual(fl.shape, (0,))
        self.assertEqual(fl.dtype, np.uint32)
        jm = np.asarray(vbm.jumpMap())
        # Default-constructed handle uses log2_block_width=6 -> JumpMapLength=1.
        self.assertEqual(jm.shape, (0, 1))
        self.assertEqual(jm.dtype, np.uint64)

    def test_reset_handle_returns_empty_arrays(self):
        # Same guard but exercising reset() after a real build.
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        h = self._make_cube_on_index_grid()
        vbm = nanovdb.tools.buildVoxelBlockManager(h.grid(), log2_block_width=6)
        self.assertGreater(vbm.blockCount(), 0)
        vbm.reset()
        self.assertEqual(vbm.blockCount(), 0)
        self.assertEqual(np.asarray(vbm.firstLeafID()).shape, (0,))
        self.assertEqual(np.asarray(vbm.jumpMap()).shape, (0, 1))


class TestGridMetaDataGuards(unittest.TestCase):
    """GridMetaData() constructor and safeCast() reject bad input (None, a
    Grid wrapping an invalid buffer) with a Python exception or False
    rather than asserting / null-dereferencing inside NanoVDB."""

    def test_init_rejects_none(self):
        # nanobind's type system rejects None for const GridData* before our
        # validity check runs. Either is acceptable as long as we don't
        # crash / abort.
        with self.assertRaises((TypeError, ValueError)):
            nanovdb.GridMetaData(None)

    def test_safeCast_rejects_none(self):
        with self.assertRaises((TypeError, ValueError)):
            nanovdb.GridMetaData.safeCast(None)

    def test_valid_grid_still_works(self):
        h = nanovdb.tools.createFogVolumeSphere()
        m = nanovdb.GridMetaData(h.grid())
        self.assertTrue(m.isValid())
        self.assertTrue(nanovdb.GridMetaData.safeCast(h.grid()))


class TestReadWriteGrids(unittest.TestCase):
    def setUp(self):
        self.gridName = "sphere_ls"
        sphereHandle = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Float, name=self.gridName
        )
        self.srcFile = tempfile.NamedTemporaryFile(delete=False)
        self.srcFile.close()
        nanovdb.io.writeGrid(self.srcFile.name, sphereHandle)
        nanovdb.io.writeGrid(self.srcFile.name, sphereHandle)
        self.dstFile = tempfile.NamedTemporaryFile(delete=False)
        self.dstFile.close()

    def tearDown(self):
        os.unlink(self.srcFile.name)
        os.unlink(self.dstFile.name)

    def test_metadata(self):
        metadataList = nanovdb.io.readGridMetaData(self.srcFile.name)
        for metadata in metadataList:
            self.assertEqual(metadata.gridName, self.gridName)
            self.assertTrue(metadata.memUsage() > 0)
            self.assertTrue(metadata.gridSize > 0)
            self.assertTrue(metadata.fileSize > 0)
            self.assertTrue(metadata.voxelCount > 0)
            self.assertEqual(metadata.gridType, nanovdb.GridType.Float)
            self.assertEqual(metadata.gridClass, nanovdb.GridClass.LevelSet)
            self.assertTrue(metadata.indexBBox.isInside(nanovdb.math.Coord(0)))
            self.assertTrue(metadata.worldBBox.isInside(nanovdb.math.Vec3d(0.0)))
            self.assertEqual(metadata.voxelSize, nanovdb.math.Vec3d(1.0))
            self.assertEqual(metadata.nameSize, len(self.gridName) + 1)
            self.assertIsInstance(metadata.nodeCount, tuple)
            self.assertIsInstance(metadata.tileCount, tuple)
            self.assertEqual(metadata.codec, nanovdb.io.Codec.NONE)
            self.assertEqual(metadata.blindDataCount, 0)
            self.assertEqual(metadata.version, nanovdb.Version())

    def test_read_write_grid(self):
        self.assertTrue(nanovdb.io.hasGrid(self.srcFile.name, self.gridName))
        handle = nanovdb.io.readGrid(self.srcFile.name)
        self.assertEqual(handle.gridCount(), 1)
        for i in range(handle.gridCount()):
            self.assertTrue(handle.gridSize(i) > 0)
            self.assertEqual(handle.gridType(i), nanovdb.GridType.Float)
            grid = handle.grid(i)
            self.assertIsNotNone(grid)
            self.assertTrue(grid.activeVoxelCount() > 0)
            self.assertTrue(grid.isSequential())
            self.assertEqual(grid.gridName(), self.gridName)

        nanovdb.io.writeGrid(self.dstFile.name, handle, nanovdb.io.Codec.NONE)
        self.assertTrue(nanovdb.io.hasGrid(self.dstFile.name, self.gridName))
        nanovdb.io.writeGrid(self.dstFile.name, handle, nanovdb.io.Codec.BLOSC)
        self.assertTrue(nanovdb.io.hasGrid(self.dstFile.name, self.gridName))
        nanovdb.io.writeGrid(self.dstFile.name, handle, nanovdb.io.Codec.ZIP)
        self.assertTrue(nanovdb.io.hasGrid(self.dstFile.name, self.gridName))

    def test_read_write_grids(self):
        handles = nanovdb.io.readGrids(self.srcFile.name)
        self.assertEqual(len(handles), 1)
        nanovdb.io.writeGrids(self.dstFile.name, handles, nanovdb.io.Codec.NONE)

        try:
            nanovdb.io.writeGrids(self.dstFile.name, handles, nanovdb.io.Codec.BLOSC)
        except RuntimeError:
            print("BLOSC compression codec not supported. Skipping...")

        try:
            nanovdb.io.writeGrids(self.dstFile.name, handles, nanovdb.io.Codec.ZIP)
        except RuntimeError:
            print("ZIP compression codec not supported. Skipping...")


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestDeviceReadWriteGrids(unittest.TestCase):
    def setUp(self):
        self.gridName = "sphere_ls"
        sphereHandle = nanovdb.tools.cuda.createLevelSetSphere(
            nanovdb.GridType.Float, name=self.gridName
        )
        self.srcFile = tempfile.NamedTemporaryFile(delete=False)
        self.srcFile.close()
        nanovdb.io.deviceWriteGrid(self.srcFile.name, sphereHandle)
        nanovdb.io.deviceWriteGrid(self.srcFile.name, sphereHandle)
        self.dstFile = tempfile.NamedTemporaryFile(delete=False)
        self.dstFile.close()

    def tearDown(self):
        os.unlink(self.srcFile.name)
        os.unlink(self.dstFile.name)

    def test_metadata(self):
        metadataList = nanovdb.io.readGridMetaData(self.srcFile.name)
        for metadata in metadataList:
            self.assertEqual(metadata.gridName, self.gridName)
            self.assertTrue(metadata.memUsage() > 0)
            self.assertTrue(metadata.gridSize > 0)
            self.assertTrue(metadata.fileSize > 0)
            self.assertTrue(metadata.voxelCount > 0)
            self.assertEqual(metadata.gridType, nanovdb.GridType.Float)
            self.assertEqual(metadata.gridClass, nanovdb.GridClass.LevelSet)
            self.assertTrue(metadata.indexBBox.isInside(nanovdb.math.Coord(0)))
            self.assertTrue(metadata.worldBBox.isInside(nanovdb.math.Vec3d(0.0)))
            self.assertEqual(metadata.voxelSize, nanovdb.math.Vec3d(1.0))
            self.assertEqual(metadata.nameSize, len(self.gridName) + 1)
            self.assertIsInstance(metadata.nodeCount, tuple)
            self.assertIsInstance(metadata.tileCount, tuple)
            self.assertEqual(metadata.codec, nanovdb.io.Codec.NONE)
            self.assertEqual(metadata.blindDataCount, 0)
            self.assertEqual(metadata.version, nanovdb.Version())

    def test_read_write_grid(self):
        self.assertTrue(nanovdb.io.hasGrid(self.srcFile.name, self.gridName))
        handle = nanovdb.io.deviceReadGrid(self.srcFile.name)
        self.assertEqual(handle.gridCount(), 1)
        for i in range(handle.gridCount()):
            self.assertTrue(handle.gridSize(i) > 0)
            self.assertEqual(handle.gridType(i), nanovdb.GridType.Float)
            grid = handle.grid(i)
            self.assertIsNotNone(grid)
            deviceGrid = handle.deviceGrid(i)
            self.assertIsNone(deviceGrid)
            handle.deviceUpload()
            deviceGrid = handle.deviceGrid(i)
            handle.deviceDownload()
            grid = handle.grid(i)
            self.assertIsNotNone(grid)
            self.assertIsNotNone(deviceGrid)
            self.assertTrue(grid.activeVoxelCount() > 0)
            self.assertTrue(grid.isSequential())
            self.assertEqual(grid.gridName(), self.gridName)

        nanovdb.io.deviceWriteGrid(self.dstFile.name, handle, nanovdb.io.Codec.NONE)
        self.assertTrue(nanovdb.io.hasGrid(self.dstFile.name, self.gridName))
        nanovdb.io.deviceWriteGrid(self.dstFile.name, handle, nanovdb.io.Codec.BLOSC)
        self.assertTrue(nanovdb.io.hasGrid(self.dstFile.name, self.gridName))
        nanovdb.io.deviceWriteGrid(self.dstFile.name, handle, nanovdb.io.Codec.ZIP)
        self.assertTrue(nanovdb.io.hasGrid(self.dstFile.name, self.gridName))

    def test_read_write_grids(self):
        handles = nanovdb.io.deviceReadGrids(self.srcFile.name)
        self.assertEqual(len(handles), 1)
        nanovdb.io.deviceWriteGrids(self.dstFile.name, handles, nanovdb.io.Codec.NONE)

        try:
            nanovdb.io.deviceWriteGrids(
                self.dstFile.name, handles, nanovdb.io.Codec.BLOSC
            )
        except RuntimeError:
            print("BLOSC compression codec not supported. Skipping...")

        try:
            nanovdb.io.deviceWriteGrids(
                self.dstFile.name, handles, nanovdb.io.Codec.ZIP
            )
        except RuntimeError:
            print("ZIP compression codec not supported. Skipping...")

    def test_device_write_grids_multi(self):
        # Regression test for deviceWriteGrids: all grids in the list must end
        # up in the output file, not just the last one.
        handle = nanovdb.tools.cuda.createLevelSetSphere(
            nanovdb.GridType.Float, name=self.gridName
        )
        handles = [handle, handle]
        nanovdb.io.deviceWriteGrids(self.dstFile.name, handles)
        metadata = nanovdb.io.readGridMetaData(self.dstFile.name)
        self.assertEqual(len(metadata), len(handles))
        readHandles = nanovdb.io.deviceReadGrids(self.dstFile.name)
        self.assertEqual(len(readHandles), len(handles))
        for readHandle in readHandles:
            self.assertEqual(readHandle.gridCount(), 1)
            self.assertEqual(readHandle.gridType(0), nanovdb.GridType.Float)


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestPointsToGrid(unittest.TestCase):
    def test_points_to_grid(self):
        try:
            import torch

            tensor = torch.tensor(
                [[1, 2, 3]], dtype=torch.int32, device=torch.device("cuda", 0)
            )
            handle = nanovdb.tools.cuda.pointsToRGBA8Grid(tensor)
            deviceGrid = handle.deviceGrid()
            self.assertTrue(deviceGrid)
            grid = handle.grid()
            self.assertFalse(grid)
            handle.deviceDownload()
            grid = handle.grid()
            self.assertTrue(grid)
        except ImportError:
            print("PyTorch not found. Skipping...")
            pass


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestSignedFloodFill(unittest.TestCase):
    def test_signed_flood_fill_float(self):
        handle = nanovdb.tools.cuda.createLevelSetSphere(nanovdb.GridType.Float, 100)
        grid = handle.grid()
        self.assertIsNotNone(grid)
        accessor = grid.getAccessor()
        self.assertFalse(accessor.isActive(nanovdb.math.Coord(103, 0, 0)))
        self.assertTrue(accessor.isActive(nanovdb.math.Coord(100, 0, 0)))
        self.assertFalse(accessor.isActive(nanovdb.math.Coord(97, 0, 0)))
        self.assertEqual(3.0, accessor(103, 0, 0))
        self.assertEqual(0.0, accessor(100, 0, 0))
        self.assertEqual(-3.0, accessor(97, 0, 0))
        accessor.setVoxel(nanovdb.math.Coord(103, 0, 0), -1.0)
        accessor.setVoxel(nanovdb.math.Coord(97, 0, 0), 1.0)
        self.assertEqual(-1.0, accessor(103, 0, 0))
        self.assertEqual(0.0, accessor(100, 0, 0))
        self.assertEqual(1.0, accessor(97, 0, 0))
        handle.deviceUpload()
        deviceGrid = handle.deviceGrid(0)
        self.assertIsNotNone(deviceGrid)
        nanovdb.tools.cuda.signedFloodFill(deviceGrid)
        handle.deviceDownload()
        grid = handle.grid()
        self.assertIsNotNone(grid)
        accessor = grid.getAccessor()
        self.assertEqual(3.0, accessor(103, 0, 0))
        self.assertEqual(0.0, accessor(100, 0, 0))
        self.assertEqual(-3.0, accessor(97, 0, 0))
        # self.assertFalse(grid.isLexicographic())
        self.assertTrue(grid.isBreadthFirst())

    def test_signed_flood_fill_double(self):
        handle = nanovdb.tools.cuda.createLevelSetSphere(nanovdb.GridType.Double, 100)
        grid = handle.grid()
        self.assertIsNotNone(grid)
        accessor = grid.getAccessor()
        self.assertFalse(accessor.isActive(nanovdb.math.Coord(103, 0, 0)))
        self.assertTrue(accessor.isActive(nanovdb.math.Coord(100, 0, 0)))
        self.assertFalse(accessor.isActive(nanovdb.math.Coord(97, 0, 0)))
        self.assertEqual(3.0, accessor(103, 0, 0))
        self.assertEqual(0.0, accessor(100, 0, 0))
        self.assertEqual(-3.0, accessor(97, 0, 0))
        accessor.setVoxel(nanovdb.math.Coord(103, 0, 0), -1.0)
        accessor.setVoxel(nanovdb.math.Coord(97, 0, 0), 1.0)
        self.assertEqual(-1.0, accessor(103, 0, 0))
        self.assertEqual(0.0, accessor(100, 0, 0))
        self.assertEqual(1.0, accessor(97, 0, 0))
        handle.deviceUpload()
        deviceGrid = handle.deviceGrid(0)
        self.assertIsNotNone(deviceGrid)
        nanovdb.tools.cuda.signedFloodFill(deviceGrid)
        handle.deviceDownload()
        grid = handle.grid()
        self.assertIsNotNone(grid)
        accessor = grid.getAccessor()
        self.assertEqual(3.0, accessor(103, 0, 0))
        self.assertEqual(0.0, accessor(100, 0, 0))
        self.assertEqual(-3.0, accessor(97, 0, 0))
        # self.assertFalse(grid.isLexicographic())
        self.assertTrue(grid.isBreadthFirst())


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestSampleFromPoints(unittest.TestCase):
    def test_sample_from_points_float(self):
        try:
            import torch

            radius = 100.0
            world_space_pos = nanovdb.math.Vec3f(radius)
            voxelSize = 0.5
            halfWidth = 5.0
            value = halfWidth * voxelSize
            handle = nanovdb.tools.cuda.createLevelSetSphere(
                nanovdb.GridType.Float,
                radius=radius,
                halfWidth=halfWidth,
                voxelSize=voxelSize,
            )
            handle.deviceUpload()
            grid = handle.deviceGrid()
            self.assertIsNotNone(grid)

            points = torch.tensor(
                [
                    [10.0, 0.0, 0.0],  # interior point outside the narrow band
                    [100.0, 0.0, 0.0],  # on the boundary
                    [0.0, 0.0, 99.0],  # interior point inside the narrow band
                    [0.0, 101.0, 0.0],  # exterior point inside the narrow band
                    [110.0, 0.0, 0.0],  # exterior point ouside the narrow band
                ],
                dtype=torch.float32,
                device=torch.device("cuda", 0),
            )
            values = torch.zeros(
                [5], dtype=torch.float32, device=torch.device("cuda", 0)
            )
            gradients = torch.zeros(
                [5, 3], dtype=torch.float32, device=torch.device("cuda", 0)
            )
            expected_values = torch.tensor(
                [-value, 0.0, -1.0, 1.0, value], dtype=torch.float32
            )
            expected_gradients = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
                device=torch.device("cuda", 0),
            )
            nanovdb.tools.cuda.sampleFromVoxels(points, grid, values, gradients)
            for i in range(5):
                self.assertEqual(values[i], expected_values[i])
            for i in range(5):
                for j in range(3):
                    self.assertEqual(gradients[i][j], expected_gradients[i][j])

        except ImportError:
            print("PyTorch not found. Skipping...")
            pass

    def test_sample_from_points_double(self):
        try:
            import torch

            radius = 100.0
            world_space_pos = nanovdb.math.Vec3d(radius)
            voxelSize = 0.5
            halfWidth = 5.0
            value = halfWidth * voxelSize
            handle = nanovdb.tools.cuda.createLevelSetSphere(
                nanovdb.GridType.Double,
                radius=radius,
                halfWidth=halfWidth,
                voxelSize=voxelSize,
            )
            handle.deviceUpload()
            grid = handle.deviceGrid()
            self.assertIsNotNone(grid)

            points = torch.tensor(
                [
                    [10.0, 0.0, 0.0],
                    [100.0, 0.0, 0.0],
                    [0.0, 0.0, 99.0],
                    [0.0, 101.0, 0.0],
                    [110.0, 0.0, 0.0],
                ],
                dtype=torch.float64,
                device=torch.device("cuda", 0),
            )
            values = torch.zeros(
                [5], dtype=torch.float64, device=torch.device("cuda", 0)
            )
            expected_values = torch.tensor(
                [-value, 0.0, -1.0, 1.0, value], dtype=torch.float64
            )
            nanovdb.tools.cuda.sampleFromVoxels(points, grid, values)
            for i in range(5):
                self.assertEqual(values[i], expected_values[i])

        except ImportError:
            print("PyTorch not found. Skipping...")
            pass


class TestSampler(unittest.TestCase):
    def test_float_sampler(self):
        radius = 100.0
        world_space_pos = nanovdb.math.Vec3f(radius)
        voxelSize = 0.1
        halfWidth = 5.0
        value = halfWidth * voxelSize

        handle = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Float,
            radius=radius,
            halfWidth=halfWidth,
            voxelSize=voxelSize,
        )
        grid = handle.grid()
        xform = grid.map()
        index_space_pos = xform.applyInverseMap(world_space_pos)
        sampler = nanovdb.math.createNearestNeighborSampler(grid)
        self.assertEqual(value, sampler(index_space_pos))

        sampler = nanovdb.math.createTrilinearSampler(grid)
        self.assertEqual(value, sampler(index_space_pos))

        sampler = nanovdb.math.createTriquadraticSampler(grid)
        self.assertEqual(value, sampler(index_space_pos))

        sampler = nanovdb.math.createTricubicSampler(grid)
        self.assertEqual(value, sampler(index_space_pos))

    def test_double_sampler(self):
        radius = 100.0
        world_space_pos = nanovdb.math.Vec3d(radius)
        voxelSize = 0.1
        halfWidth = 5.0
        value = halfWidth * voxelSize

        handle = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Double,
            radius=radius,
            halfWidth=halfWidth,
            voxelSize=voxelSize,
        )
        grid = handle.grid()
        xform = grid.map()
        index_space_pos = xform.applyInverseMap(world_space_pos)
        sampler = nanovdb.math.createNearestNeighborSampler(grid)
        self.assertEqual(value, sampler(index_space_pos))

        sampler = nanovdb.math.createTrilinearSampler(grid)
        self.assertEqual(value, sampler(index_space_pos))

        sampler = nanovdb.math.createTriquadraticSampler(grid)
        self.assertEqual(value, sampler(index_space_pos))

        sampler = nanovdb.math.createTricubicSampler(grid)
        self.assertEqual(value, sampler(index_space_pos))


class TestCreateNanoGrid(unittest.TestCase):
    def test_create_float_nano_grid(self):
        bbox = nanovdb.math.CoordBBox(
            nanovdb.math.Coord(0), nanovdb.math.Coord(2 * 4 - 1)
        )
        handle = nanovdb.tools.createFloatGrid(
            0.0, "test_float", nanovdb.GridClass.Unknown, lambda ijk: 1.0, bbox
        )
        self.assertEqual(handle.gridCount(), 1)
        for i in range(handle.gridCount()):
            self.assertTrue(handle.gridSize(i) > 0)
            self.assertEqual(handle.gridType(i), nanovdb.GridType.Float)
            grid = handle.grid(i)
            self.assertIsNotNone(grid)
            self.assertTrue(grid.activeVoxelCount() > 0)
            self.assertTrue(grid.isSequential())
            self.assertEqual(grid.gridName(), "test_float")
            self.assertEqual(grid.gridClass(), nanovdb.GridClass.Unknown)

    def test_create_double_nano_grid(self):
        bbox = nanovdb.math.CoordBBox(
            nanovdb.math.Coord(0), nanovdb.math.Coord(2 * 4 - 1)
        )
        handle = nanovdb.tools.createDoubleGrid(
            0.0, "test_double", nanovdb.GridClass.Unknown, lambda ijk: 1.0, bbox
        )
        self.assertEqual(handle.gridCount(), 1)
        for i in range(handle.gridCount()):
            self.assertTrue(handle.gridSize(i) > 0)
            self.assertEqual(handle.gridType(i), nanovdb.GridType.Double)
            grid = handle.grid(i)
            self.assertIsNotNone(grid)
            self.assertTrue(grid.activeVoxelCount() > 0)
            self.assertTrue(grid.isSequential())
            self.assertEqual(grid.gridName(), "test_double")
            self.assertEqual(grid.gridClass(), nanovdb.GridClass.Unknown)

    def test_create_int_nano_grid(self):
        bbox = nanovdb.math.CoordBBox(
            nanovdb.math.Coord(0), nanovdb.math.Coord(2 * 4 - 1)
        )
        handle = nanovdb.tools.createInt32Grid(
            0, "test_int", nanovdb.GridClass.Unknown, lambda ijk: 1, bbox
        )
        self.assertEqual(handle.gridCount(), 1)
        for i in range(handle.gridCount()):
            self.assertTrue(handle.gridSize(i) > 0)
            self.assertEqual(handle.gridType(i), nanovdb.GridType.Int32)
            grid = handle.grid(i)
            self.assertIsNotNone(grid)
            self.assertTrue(grid.activeVoxelCount() > 0)
            self.assertTrue(grid.isSequential())
            self.assertEqual(grid.gridName(), "test_int")
            self.assertEqual(grid.gridClass(), nanovdb.GridClass.Unknown)

    def test_create_vec3f_nano_grid(self):
        bbox = nanovdb.math.CoordBBox(
            nanovdb.math.Coord(0), nanovdb.math.Coord(2 * 4 - 1)
        )
        handle = nanovdb.tools.createVec3fGrid(
            nanovdb.math.Vec3f(0.0),
            "test_vec3f",
            nanovdb.GridClass.Unknown,
            lambda ijk: nanovdb.math.Vec3f(1.0),
            bbox,
        )
        self.assertEqual(handle.gridCount(), 1)
        for i in range(handle.gridCount()):
            self.assertTrue(handle.gridSize(i) > 0)
            self.assertEqual(handle.gridType(i), nanovdb.GridType.Vec3f)
            grid = handle.grid(i)
            self.assertIsNotNone(grid)
            self.assertTrue(grid.activeVoxelCount() > 0)
            self.assertTrue(grid.isSequential())
            self.assertEqual(grid.gridName(), "test_vec3f")
            self.assertEqual(grid.gridClass(), nanovdb.GridClass.Unknown)


class TestNewPrimitives(unittest.TestCase):
    """Phase 5a: the 9 host primitives that didn't ship in Phase 0."""

    def test_create_level_set_box(self):
        h = nanovdb.tools.createLevelSetBox(width=10.0, height=15.0, depth=20.0)
        self.assertEqual(h.gridCount(), 1)
        self.assertEqual(h.gridType(0), nanovdb.GridType.Float)
        self.assertGreater(h.grid().activeVoxelCount(), 0)
        self.assertEqual(h.grid().gridClass(), nanovdb.GridClass.LevelSet)

    def test_create_level_set_box_double(self):
        h = nanovdb.tools.createLevelSetBox(
            gridType=nanovdb.GridType.Double, width=10.0)
        self.assertEqual(h.gridType(0), nanovdb.GridType.Double)

    def test_create_level_set_bbox(self):
        h = nanovdb.tools.createLevelSetBBox(
            width=40.0, height=40.0, depth=40.0, thickness=5.0)
        self.assertEqual(h.gridType(0), nanovdb.GridType.Float)
        self.assertGreater(h.grid().activeVoxelCount(), 0)

    def test_create_level_set_octahedron(self):
        h = nanovdb.tools.createLevelSetOctahedron(scale=20.0)
        self.assertEqual(h.gridType(0), nanovdb.GridType.Float)
        self.assertGreater(h.grid().activeVoxelCount(), 0)

    def test_create_fog_volume_box(self):
        h = nanovdb.tools.createFogVolumeBox(width=10.0)
        self.assertEqual(h.gridType(0), nanovdb.GridType.Float)
        self.assertEqual(h.grid().gridClass(), nanovdb.GridClass.FogVolume)
        self.assertGreater(h.grid().activeVoxelCount(), 0)

    def test_create_fog_volume_octahedron(self):
        h = nanovdb.tools.createFogVolumeOctahedron(scale=20.0)
        self.assertEqual(h.grid().gridClass(), nanovdb.GridClass.FogVolume)

    def test_create_point_sphere(self):
        h = nanovdb.tools.createPointSphere(pointsPerVoxel=2, radius=10.0)
        self.assertEqual(h.gridCount(), 1)
        # PointGrid stores point counts as UInt32 sequential indices.
        self.assertEqual(h.gridType(0), nanovdb.GridType.UInt32)
        self.assertEqual(h.grid().gridClass(), nanovdb.GridClass.PointData)
        self.assertGreater(h.grid().activeVoxelCount(), 0)

    def test_create_point_torus(self):
        h = nanovdb.tools.createPointTorus(
            pointsPerVoxel=1, majorRadius=10.0, minorRadius=3.0)
        self.assertEqual(h.gridType(0), nanovdb.GridType.UInt32)
        self.assertGreater(h.grid().activeVoxelCount(), 0)

    def test_create_point_box(self):
        # Box must be large enough to enclose at least one active voxel
        # for createPointScatter's internal "ActiveVoxelCount is required"
        # precondition to pass.
        h = nanovdb.tools.createPointBox(
            pointsPerVoxel=1, width=40.0, height=40.0, depth=40.0)
        self.assertEqual(h.gridType(0), nanovdb.GridType.UInt32)
        self.assertGreater(h.grid().activeVoxelCount(), 0)

    def test_create_point_scatter(self):
        # Source level set, then scatter points into it.
        sphere = nanovdb.tools.createLevelSetSphere(radius=10.0).grid()
        h = nanovdb.tools.createPointScatter(sphere, pointsPerVoxel=2)
        self.assertEqual(h.gridType(0), nanovdb.GridType.UInt32)
        self.assertGreater(h.grid().activeVoxelCount(), 0)

    def test_create_point_scatter_rejects_non_float_source(self):
        # The binding accepts NanoGrid<float> only — other source types
        # should raise TypeError at the conversion boundary.
        h_double = nanovdb.tools.createLevelSetSphere(
            gridType=nanovdb.GridType.Double, radius=10.0)
        with self.assertRaises(TypeError):
            nanovdb.tools.createPointScatter(h_double.grid())

    def test_create_point_scatter_rejects_fog_volume(self):
        # createPointScatter's C++ implementation requires the source to
        # pass srcGrid.isLevelSet(); fog volumes raise RuntimeError.
        h_fog = nanovdb.tools.createFogVolumeSphere(radius=10.0)
        with self.assertRaises(RuntimeError):
            nanovdb.tools.createPointScatter(h_fog.grid())


class TestCreateNanoGridQuantized(unittest.TestCase):
    """Phase 5b: tools.createNanoGridFp4 / Fp8 / Fp16 / FpN with AbsDiff/RelDiff."""

    def _float_sphere(self):
        return nanovdb.tools.createLevelSetSphere(radius=10.0).grid()

    def test_quantize_fp4(self):
        h = nanovdb.tools.createNanoGridFp4(self._float_sphere())
        self.assertEqual(h.gridType(0), nanovdb.GridType.Fp4)
        self.assertGreater(h.grid().activeVoxelCount(), 0)

    def test_quantize_fp8(self):
        h = nanovdb.tools.createNanoGridFp8(self._float_sphere())
        self.assertEqual(h.gridType(0), nanovdb.GridType.Fp8)

    def test_quantize_fp16(self):
        h = nanovdb.tools.createNanoGridFp16(self._float_sphere())
        self.assertEqual(h.gridType(0), nanovdb.GridType.Fp16)

    def test_quantize_fpn_absdiff(self):
        oracle = nanovdb.tools.AbsDiff(0.05)
        self.assertAlmostEqual(oracle.getTolerance(), 0.05, places=5)
        self.assertTrue(bool(oracle))
        h = nanovdb.tools.createNanoGridFpN(self._float_sphere(), oracle)
        self.assertEqual(h.gridType(0), nanovdb.GridType.FpN)

    def test_quantize_fpn_reldiff(self):
        oracle = nanovdb.tools.RelDiff(0.1)
        self.assertAlmostEqual(oracle.getTolerance(), 0.1, places=5)
        h = nanovdb.tools.createNanoGridFpN(self._float_sphere(), oracle)
        self.assertEqual(h.gridType(0), nanovdb.GridType.FpN)

    def test_oracle_default_tolerance(self):
        # Default-constructed oracle has tolerance == -1, which means
        # "uninitialized"; the operator bool() detects that.
        a = nanovdb.tools.AbsDiff()
        self.assertEqual(a.getTolerance(), -1.0)
        self.assertFalse(bool(a))
        a.setTolerance(0.5)
        self.assertEqual(a.getTolerance(), 0.5)
        self.assertTrue(bool(a))

    def test_quantize_rejects_double_source(self):
        # The C++ Fp{4,8,16,N} preProcess static-asserts SrcValueT == float;
        # Python must surface this as a TypeError at the conversion boundary.
        h_double = nanovdb.tools.createLevelSetSphere(
            gridType=nanovdb.GridType.Double, radius=5.0)
        with self.assertRaises(TypeError):
            nanovdb.tools.createNanoGridFp16(h_double.grid())
        with self.assertRaises(TypeError):
            nanovdb.tools.createNanoGridFpN(
                h_double.grid(), nanovdb.tools.AbsDiff(0.05))

    def test_quantize_from_build_grid(self):
        # Phase 5c: build::FloatGrid accepted as quantization source.
        bg = nanovdb.tools.build.FloatGrid(0.0)
        for i in range(5):
            bg.setValue(nanovdb.math.Coord(i, 0, 0), float(i + 1))
        h = nanovdb.tools.createNanoGridFp16(bg)
        self.assertEqual(h.gridType(0), nanovdb.GridType.Fp16)
        self.assertEqual(h.grid().activeVoxelCount(), 5)


class TestCreateNanoGridIndex(unittest.TestCase):
    """Phase 5b/5c: tools.createNanoGridIndex / OnIndex with broad source set."""

    def test_index_from_float_nanogrid(self):
        sphere = nanovdb.tools.createLevelSetSphere(radius=10.0).grid()
        h = nanovdb.tools.createNanoGridIndex(sphere)
        self.assertEqual(h.gridType(0), nanovdb.GridType.Index)

    def test_on_index_from_float_nanogrid(self):
        sphere = nanovdb.tools.createLevelSetSphere(radius=10.0).grid()
        h = nanovdb.tools.createNanoGridOnIndex(sphere)
        self.assertEqual(h.gridType(0), nanovdb.GridType.OnIndex)

    def test_index_from_double_nanogrid(self):
        sphere = nanovdb.tools.createLevelSetSphere(
            gridType=nanovdb.GridType.Double, radius=10.0).grid()
        h = nanovdb.tools.createNanoGridIndex(sphere)
        self.assertEqual(h.gridType(0), nanovdb.GridType.Index)

    def test_index_from_int32_build(self):
        # Phase 5c source: build::Int32Grid is accepted by the index path.
        bg = nanovdb.tools.build.Int32Grid(0)
        bg.setValue(nanovdb.math.Coord(0, 0, 0), 42)
        bg.setValue(nanovdb.math.Coord(1, 0, 0), -7)
        h = nanovdb.tools.createNanoGridOnIndex(bg)
        self.assertEqual(h.gridType(0), nanovdb.GridType.OnIndex)
        self.assertEqual(h.grid().activeVoxelCount(), 2)

    def test_index_from_vec3f_build(self):
        bv = nanovdb.tools.build.Vec3fGrid(nanovdb.math.Vec3f(0.0))
        bv.setValue(nanovdb.math.Coord(0, 0, 0), nanovdb.math.Vec3f(1, 2, 3))
        h = nanovdb.tools.createNanoGridOnIndex(bv)
        self.assertEqual(h.gridType(0), nanovdb.GridType.OnIndex)

    def test_index_rejects_none(self):
        # The conversion functions accept either a NanoGrid or a
        # build::Grid; None matches neither and is rejected at the
        # isinstance dispatch.
        with self.assertRaises(TypeError):
            nanovdb.tools.createNanoGridOnIndex(None)

    def test_index_rejects_unsupported_buildt(self):
        # The Phase 5 index conversion accepts float / double / int32 /
        # Vec3f sources (NanoGrid or build::Grid). A Vec3d build::Grid
        # is a structurally valid grid but a BuildT outside that set —
        # the try-each-SrcBuildT chain falls through and raises.
        bv = nanovdb.tools.build.Vec3dGrid(nanovdb.math.Vec3d(0.0))
        bv.setValue(nanovdb.math.Coord(0, 0, 0), nanovdb.math.Vec3d(1, 2, 3))
        with self.assertRaises(TypeError):
            nanovdb.tools.createNanoGridOnIndex(bv)


class TestGridStats(unittest.TestCase):
    """nanovdb.tools.Extrema*, Stats*, updateGridStats, getExtrema."""

    def _five_voxel_float_grid(self):
        g = nanovdb.tools.build.FloatGrid(0.0, "stats", nanovdb.GridClass.FogVolume)
        for i in range(5):
            g.setValue(nanovdb.math.Coord(i, 0, 0), float(i + 1))
        return g.to_nanovdb(sMode=nanovdb.tools.StatsMode.All)

    def test_extrema_default_and_add(self):
        ex = nanovdb.tools.FloatExtrema()
        self.assertFalse(bool(ex))
        ex.add(2.5)
        ex.add(1.0)
        ex.add(7.0)
        self.assertTrue(bool(ex))
        self.assertEqual(ex.min(), 1.0)
        self.assertEqual(ex.max(), 7.0)
        # Extrema doesn't compute averages or std deviation.
        self.assertTrue(nanovdb.tools.FloatExtrema.hasMinMax())
        self.assertFalse(nanovdb.tools.FloatExtrema.hasAverage())
        self.assertFalse(nanovdb.tools.FloatExtrema.hasStdDeviation())

    def test_stats_default_and_accumulate(self):
        st = nanovdb.tools.FloatStats()
        for v in (1.0, 2.0, 3.0, 4.0, 5.0):
            st.add(v)
        self.assertEqual(st.size(), 5)
        self.assertEqual(st.min(), 1.0)
        self.assertEqual(st.max(), 5.0)
        self.assertAlmostEqual(st.avg(), 3.0)
        self.assertAlmostEqual(st.mean(), 3.0)
        # Population variance of 1..5 = (((-2)^2 + (-1)^2 + 0 + 1 + 4) / 5) = 2
        self.assertAlmostEqual(st.var(), 2.0)
        self.assertAlmostEqual(st.std() ** 2, 2.0)
        self.assertTrue(nanovdb.tools.FloatStats.hasAverage())
        self.assertTrue(nanovdb.tools.FloatStats.hasStdDeviation())

    def test_get_extrema_strictly_inside_active_region(self):
        # Pick a bbox strictly inside the root's active bbox so the C++
        # implementation takes the recursive-traversal branch (the
        # "bbox contains root.bbox()" branch unconditionally adds the
        # background value, which would muddy this assertion). With
        # only the three active voxels at (1,0,0)..(3,0,0) sampled, the
        # extrema should be exactly their min and max.
        h = self._five_voxel_float_grid()
        ng = h.grid()
        ex = nanovdb.tools.getExtrema(
            ng, nanovdb.math.CoordBBox(
                nanovdb.math.Coord(1, 0, 0), nanovdb.math.Coord(3, 0, 0)))
        self.assertTrue(bool(ex))
        self.assertEqual(ex.min(), 2.0)
        self.assertEqual(ex.max(), 4.0)

    def test_update_grid_stats_polymorphic(self):
        # Building with StatsMode.Disable leaves stats uncomputed; calling
        # tools.updateGridStats on the resulting handle should populate
        # them in-place. Asserting "no exception" is the round-trip we
        # care about — the actual stats live inside the grid's nodes.
        g = nanovdb.tools.build.FloatGrid(0.0)
        for i in range(3):
            g.setValue(nanovdb.math.Coord(i, 0, 0), float(i + 10))
        h = g.to_nanovdb(sMode=nanovdb.tools.StatsMode.Disable)
        ng = h.grid()
        nanovdb.tools.updateGridStats(ng, nanovdb.tools.StatsMode.All)
        # checkGrid still passes after writing stats.
        ok, msg = nanovdb.tools.checkGrid(ng, nanovdb.CheckMode.Full)
        self.assertTrue(ok, msg)

    def test_update_grid_stats_on_index_grid(self):
        # OnIndexGrid is a special BuildT — MinMax and All raise because
        # Stats<uint64> isn't meaningful, but BBox (NoopStats) is still
        # accepted because it only touches node bounding boxes.
        bbox = nanovdb.math.CoordBBox(
            nanovdb.math.Coord(0), nanovdb.math.Coord(4))
        h_float = nanovdb.tools.createFloatGrid(
            0.0, "src", nanovdb.GridClass.Unknown, lambda ijk: 1.0, bbox)
        h_index = nanovdb.tools.createOnIndexGrid(h_float.grid())
        ng = h_index.grid()
        with self.assertRaises(ValueError):
            nanovdb.tools.updateGridStats(ng, nanovdb.tools.StatsMode.MinMax)
        with self.assertRaises(ValueError):
            nanovdb.tools.updateGridStats(ng, nanovdb.tools.StatsMode.All)
        # BBox mode is a NoopStats path — must succeed.
        nanovdb.tools.updateGridStats(ng, nanovdb.tools.StatsMode.BBox)
        # And Disable is a true no-op.
        nanovdb.tools.updateGridStats(ng, nanovdb.tools.StatsMode.Disable)


class TestGridValidate(unittest.TestCase):
    """nanovdb.tools.validateGrid, checkGrid, isValid."""

    def _good_handle(self):
        bbox = nanovdb.math.CoordBBox(
            nanovdb.math.Coord(0), nanovdb.math.Coord(3))
        return nanovdb.tools.createFloatGrid(
            0.0, "v", nanovdb.GridClass.Unknown, lambda ijk: 1.0, bbox)

    def test_checkGrid_on_valid_grid(self):
        h = self._good_handle()
        ok, msg = nanovdb.tools.checkGrid(h.grid(), nanovdb.CheckMode.Full)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_isValid_on_valid_grid(self):
        h = self._good_handle()
        self.assertTrue(nanovdb.tools.isValid(h.grid(), nanovdb.CheckMode.Default))

    def test_validateGrid_on_valid_handle(self):
        h = self._good_handle()
        self.assertTrue(nanovdb.tools.validateGrid(h, 0))
        # validateGrid with out-of-range gridID returns False, never raises.
        self.assertFalse(nanovdb.tools.validateGrid(h, 99))

    def test_validateGrid_disable_mode_always_true(self):
        h = self._good_handle()
        self.assertTrue(
            nanovdb.tools.validateGrid(h, 99, nanovdb.CheckMode.Disable))

    @unittest.skipIf(
        not nanovdb.isCudaAvailable(),
        "nanovdb module was compiled without CUDA support",
    )
    @unittest.skipIf(
        not nanovdb.isGpuAvailable(),
        "No CUDA-capable GPU available at runtime",
    )
    def test_validateGrid_on_device_handle(self):
        # validateGrid is bound for both host and device handles. The
        # device overload routes through the same callNanoGrid dispatch
        # against the host-resident copy of the grid metadata.
        h = nanovdb.tools.cuda.createLevelSetSphere()
        self.assertTrue(nanovdb.tools.validateGrid(h, 0))
        # Out-of-range gridID returns False (without raising); Disable
        # mode short-circuits to True even on an out-of-range gridID.
        self.assertFalse(nanovdb.tools.validateGrid(h, 99))
        self.assertTrue(
            nanovdb.tools.validateGrid(h, 99, nanovdb.CheckMode.Disable))


class TestGridChecksum(unittest.TestCase):
    """nanovdb.tools.evalChecksum and validateChecksum."""

    def _handle(self):
        bbox = nanovdb.math.CoordBBox(
            nanovdb.math.Coord(0), nanovdb.math.Coord(3))
        return nanovdb.tools.createFloatGrid(
            0.0, "cs", nanovdb.GridClass.Unknown, lambda ijk: 1.0, bbox)

    def test_eval_then_update_then_validate(self):
        h = self._handle()
        ng = h.grid()
        cs1 = nanovdb.tools.evalChecksum(ng, nanovdb.CheckMode.Full)
        nanovdb.tools.updateChecksum(ng, nanovdb.CheckMode.Full)
        cs2 = nanovdb.tools.evalChecksum(ng, nanovdb.CheckMode.Full)
        # Recomputing on an unchanged grid gives the same checksum.
        self.assertEqual(cs1, cs2)
        self.assertTrue(
            nanovdb.tools.validateChecksum(ng, nanovdb.CheckMode.Full))

    def test_validate_empty_stored_returns_true(self):
        # A grid with no stored checksum is considered valid by the C++
        # rule (Checksum.isEmpty() short-circuit).
        h = self._handle()
        self.assertTrue(
            nanovdb.tools.validateChecksum(h.grid(), nanovdb.CheckMode.Default))


class TestBuildGrid(unittest.TestCase):
    """nanovdb.tools.build.* — mutable voxel-by-voxel CPU grid builder."""

    def test_constructor_defaults_and_metadata(self):
        g = nanovdb.tools.build.FloatGrid(0.0)
        self.assertEqual(g.getName(), "")
        self.assertEqual(g.gridClass(), nanovdb.GridClass.Unknown)
        self.assertEqual(g.gridType(), nanovdb.GridType.Float)
        self.assertEqual(g.background, 0.0)
        self.assertEqual(g.nodeCount(), [0, 0, 0])
        g.setName("renamed")
        self.assertEqual(g.getName(), "renamed")

    def test_set_get_value_marks_active(self):
        g = nanovdb.tools.build.FloatGrid(0.0, "demo")
        ijk = nanovdb.math.Coord(1, 2, 3)
        self.assertFalse(g.isActive(ijk))
        self.assertEqual(g.getValue(ijk), 0.0)
        g.setValue(ijk, 4.5)
        self.assertTrue(g.isActive(ijk))
        self.assertEqual(g.getValue(ijk), 4.5)
        # An untouched voxel is still background-valued and inactive.
        self.assertEqual(g.getValue(nanovdb.math.Coord(10, 0, 0)), 0.0)
        self.assertFalse(g.isActive(nanovdb.math.Coord(10, 0, 0)))

    def test_set_value_on_keeps_background_value(self):
        g = nanovdb.tools.build.FloatGrid(-1.0, "demo")
        ijk = nanovdb.math.Coord(5, 6, 7)
        g.setValueOn(ijk)
        self.assertTrue(g.isActive(ijk))
        # setValueOn does not change the stored value — still background.
        self.assertEqual(g.getValue(ijk), -1.0)

    def test_value_accessor_parity_with_grid(self):
        g = nanovdb.tools.build.FloatGrid(0.0)
        acc = g.getAccessor()
        ijk = nanovdb.math.Coord(100, 200, 300)
        acc.setValue(ijk, 7.5)
        self.assertEqual(g.getValue(ijk), 7.5)
        self.assertEqual(acc.getValue(ijk), 7.5)
        self.assertTrue(acc.isActive(ijk))
        # isValueOn is an alias for isActive.
        self.assertEqual(acc.isValueOn(ijk), acc.isActive(ijk))

    def test_write_accessor_explicit_merge(self):
        g = nanovdb.tools.build.FloatGrid(0.0)
        ijk = nanovdb.math.Coord(50, 50, 50)
        wa = g.getWriteAccessor()
        wa.setValue(ijk, 9.0)
        # Before merge, the parent grid hasn't seen the change yet.
        self.assertEqual(g.getValue(ijk), 0.0)
        wa.merge()
        self.assertEqual(g.getValue(ijk), 9.0)
        self.assertTrue(g.isActive(ijk))

    def test_write_accessor_merges_on_destruction(self):
        # When the Python wrapper for a WriteAccessor is collected, the
        # C++ destructor runs merge() automatically. Force collection by
        # dropping the only reference and running the GC.
        import gc
        g = nanovdb.tools.build.FloatGrid(0.0)
        ijk = nanovdb.math.Coord(60, 60, 60)
        wa = g.getWriteAccessor()
        wa.setValue(ijk, 3.5)
        self.assertEqual(g.getValue(ijk), 0.0)
        del wa
        gc.collect()
        self.assertEqual(g.getValue(ijk), 3.5)
        self.assertTrue(g.isActive(ijk))

    def test_to_nanovdb_roundtrip(self):
        g = nanovdb.tools.build.FloatGrid(0.0, "trip", nanovdb.GridClass.FogVolume)
        g.setValue(nanovdb.math.Coord(0, 0, 0), 1.0)
        g.setValue(nanovdb.math.Coord(1, 0, 0), 2.0)
        g.setValue(nanovdb.math.Coord(2, 0, 0), 3.0)
        h = g.to_nanovdb()
        self.assertEqual(h.gridCount(), 1)
        ng = h.grid()
        self.assertEqual(ng.gridType(), nanovdb.GridType.Float)
        self.assertEqual(ng.gridClass(), nanovdb.GridClass.FogVolume)
        self.assertEqual(ng.gridName(), "trip")
        self.assertEqual(ng.activeVoxelCount(), 3)

    def test_to_nanovdb_does_not_consume_source(self):
        # Source build::Grid must remain usable after .to_nanovdb().
        g = nanovdb.tools.build.FloatGrid(0.0)
        g.setValue(nanovdb.math.Coord(0, 0, 0), 1.0)
        _ = g.to_nanovdb()
        g.setValue(nanovdb.math.Coord(1, 0, 0), 2.0)
        h2 = g.to_nanovdb()
        self.assertEqual(h2.grid().activeVoxelCount(), 2)

    def test_int32_build_grid(self):
        g = nanovdb.tools.build.Int32Grid(0, "ints", nanovdb.GridClass.Unknown)
        g.setValue(nanovdb.math.Coord(0, 0, 0), 42)
        g.setValue(nanovdb.math.Coord(1, 1, 1), -7)
        self.assertEqual(g.getValue(nanovdb.math.Coord(0, 0, 0)), 42)
        self.assertEqual(g.getValue(nanovdb.math.Coord(1, 1, 1)), -7)
        h = g.to_nanovdb()
        self.assertEqual(h.grid().gridType(), nanovdb.GridType.Int32)
        self.assertEqual(h.grid().activeVoxelCount(), 2)

    def test_vec3f_build_grid(self):
        g = nanovdb.tools.build.Vec3fGrid(
            nanovdb.math.Vec3f(0.0), "v", nanovdb.GridClass.Unknown)
        v = nanovdb.math.Vec3f(1.0, 2.0, 3.0)
        g.setValue(nanovdb.math.Coord(0, 0, 0), v)
        self.assertEqual(g.getValue(nanovdb.math.Coord(0, 0, 0)), v)
        h = g.to_nanovdb()
        self.assertEqual(h.grid().gridType(), nanovdb.GridType.Vec3f)

    def test_set_transform(self):
        g = nanovdb.tools.build.FloatGrid(0.0)
        g.setTransform(scale=0.5, translation=nanovdb.math.Vec3d(1.0, 2.0, 3.0))
        g.setValue(nanovdb.math.Coord(0, 0, 0), 1.0)
        h = g.to_nanovdb()
        ng = h.grid()
        vs = ng.voxelSize()
        self.assertAlmostEqual(vs[0], 0.5)
        self.assertAlmostEqual(vs[1], 0.5)
        self.assertAlmostEqual(vs[2], 0.5)
        # Index (0,0,0) mapped through (scale=0.5, translation=(1,2,3))
        # lands at world-space (1, 2, 3).
        w = ng.map().applyMap(nanovdb.math.Vec3d(0.0, 0.0, 0.0))
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], 2.0)
        self.assertAlmostEqual(w[2], 3.0)


class TestNanoToOpenVDB(unittest.TestCase):
    def test_function(self):
        handle = nanovdb.tools.createLevelSetSphere()
        try:
            import openvdb

            sphere = nanovdb.tools.nanoToOpenVDB(handle)
            self.assertEqual(sphere.name, "sphere_ls")
            self.assertFalse(sphere.empty())
        except ImportError:
            print("openvdb not found. Skipping...")
            pass


class TestOpenToNanoVDB(unittest.TestCase):
    def test_function(self):
        try:
            import openvdb

            sphere = openvdb.createLevelSetSphere(100.0)
            sphere.name = "test_sphere"
            handle = nanovdb.tools.openToNanoVDB(sphere)
            self.assertEqual(handle.gridCount(), 1)
            for i in range(handle.gridCount()):
                self.assertTrue(handle.gridSize(i) > 0)
                self.assertEqual(handle.gridType(i), nanovdb.GridType.Float)
                grid = handle.grid(i)
                self.assertIsNotNone(grid)
                self.assertTrue(grid.activeVoxelCount() > 0)
                self.assertTrue(grid.isSequential())
                self.assertEqual(grid.gridName(), "test_sphere")
                self.assertEqual(grid.gridClass(), nanovdb.GridClass.LevelSet)
        except ImportError:
            print("openvdb not found. Skipping...")
            pass


class TestGridTransformAliases(unittest.TestCase):
    """worldToIndex / indexToWorld and friends are aliases of the apply*
    transform family, mirroring the C++ Grid<TreeT> convenience names."""

    def setUp(self):
        self.handle = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Float, radius=10.0, voxelSize=0.5
        )
        self.grid = self.handle.grid()

    def test_world_index_point_aliases(self):
        p = nanovdb.math.Vec3d(1.5, -2.0, 3.25)
        self.assertEqual(self.grid.worldToIndex(p), self.grid.applyInverseMap(p))
        self.assertEqual(self.grid.indexToWorld(p), self.grid.applyMap(p))
        roundtrip = self.grid.indexToWorld(self.grid.worldToIndex(p))
        for i in range(3):
            self.assertAlmostEqual(roundtrip[i], p[i], places=12)

    def test_direction_and_gradient_aliases(self):
        d = nanovdb.math.Vec3d(0.25, 1.0, -0.5)
        self.assertEqual(self.grid.worldToIndexDir(d), self.grid.applyInverseJacobian(d))
        self.assertEqual(self.grid.indexToWorldDir(d), self.grid.applyJacobian(d))
        self.assertEqual(self.grid.indexToWorldGrad(d), self.grid.applyIJT(d))

    def test_single_precision_aliases(self):
        p = nanovdb.math.Vec3f(1.5, -2.0, 3.25)
        self.assertEqual(self.grid.worldToIndexF(p), self.grid.applyInverseMapF(p))
        self.assertEqual(self.grid.indexToWorldF(p), self.grid.applyMapF(p))
        self.assertEqual(self.grid.worldToIndexDirF(p), self.grid.applyInverseJacobianF(p))
        self.assertEqual(self.grid.indexToWorldDirF(p), self.grid.applyJacobianF(p))
        self.assertEqual(self.grid.indexToWorldGradF(p), self.grid.applyIJTF(p))


class TestGridValuePointCount(unittest.TestCase):
    """valueCount() on Index/OnIndex grids and pointCount() on PointGrid,
    mirroring the SFINAE-gated C++ Grid methods."""

    def setUp(self):
        self.src = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Float, radius=5.0, voxelSize=1.0
        )

    def test_on_index_value_count(self):
        handle = nanovdb.tools.createNanoGridOnIndex(self.src.grid())
        grid = handle.grid()
        self.assertIsInstance(grid, nanovdb.OnIndexGrid)
        self.assertGreaterEqual(grid.valueCount(), self.src.grid().activeVoxelCount())

    def test_index_value_count(self):
        handle = nanovdb.tools.createNanoGridIndex(self.src.grid())
        grid = handle.grid()
        self.assertIsInstance(grid, nanovdb.IndexGrid)
        self.assertGreaterEqual(grid.valueCount(), self.src.grid().activeVoxelCount())

    def test_point_count_bound_on_point_grid(self):
        # pointCount() lives on NanoGrid<Point> (GridType.PointIndex). The
        # point primitives bake UInt32 PointData grids, so only the binding's
        # presence can be verified host-side without an OpenVDB conversion.
        self.assertIn("pointCount", dir(nanovdb.PointGrid))
        self.assertNotIn("valueCount", dir(nanovdb.PointGrid))
        point_data_grid = nanovdb.tools.createPointSphere(
            pointsPerVoxel=2, radius=5.0, voxelSize=1.0
        ).grid()
        self.assertIsInstance(point_data_grid, nanovdb.UInt32Grid)
        self.assertFalse(hasattr(point_data_grid, "pointCount"))

    def test_gated_to_matching_buildts(self):
        float_grid = self.src.grid()
        self.assertFalse(hasattr(float_grid, "valueCount"))
        self.assertFalse(hasattr(float_grid, "pointCount"))
        on_index_grid = nanovdb.tools.createNanoGridOnIndex(float_grid).grid()
        self.assertFalse(hasattr(on_index_grid, "pointCount"))


class TestSamplerGradient(unittest.TestCase):
    """gradient() on the trilinear sampler and zeroCrossing() on the
    trilinear + triquadratic samplers, for floating-point grids only."""

    def setUp(self):
        self.radius = 10.0
        self.voxelSize = 0.5
        self.handle = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Float, radius=self.radius, voxelSize=self.voxelSize
        )
        self.grid = self.handle.grid()
        # Index-space position on the sphere surface, on the +x axis.
        self.surface = self.grid.worldToIndex(nanovdb.math.Vec3d(self.radius, 0.0, 0.0))

    def test_trilinear_gradient_points_outward(self):
        sampler = nanovdb.math.createTrilinearSampler(self.grid)
        surface_f = nanovdb.math.Vec3f(
            self.surface[0], self.surface[1], self.surface[2]
        )
        g = sampler.gradient(surface_f)
        # An SDF in world units sampled on an index-space lattice changes by
        # ~voxelSize per index step along the outward normal (+x here). The
        # tangential components pick up the sphere's curvature across the
        # stencil cell, so they are small but not zero.
        self.assertAlmostEqual(g[0], self.voxelSize, places=3)
        self.assertAlmostEqual(g[1], 0.0, delta=0.05)
        self.assertAlmostEqual(g[2], 0.0, delta=0.05)
        # The Vec3d overload agrees.
        gd = sampler.gradient(nanovdb.math.Vec3d(self.surface))
        for i in range(3):
            self.assertAlmostEqual(g[i], gd[i], places=5)

    def test_zero_crossing(self):
        # Probe just inside the surface: an exactly-zero stencil corner is
        # not a strict sign change, so the on-surface lattice point (20,0,0)
        # itself does not count as a crossing.
        inside = nanovdb.math.Vec3d(self.surface[0] - 0.5, 0.0, 0.0)
        for make in (
            nanovdb.math.createTrilinearSampler,
            nanovdb.math.createTriquadraticSampler,
        ):
            sampler = make(self.grid)
            self.assertTrue(abs(sampler(inside)) < self.voxelSize)
            self.assertTrue(sampler.zeroCrossing(inside))
            # Deep inside the narrow band there is no crossing.
            self.assertFalse(sampler.zeroCrossing(nanovdb.math.Vec3d(0.0, 0.0, 0.0)))

    def test_gated_to_matching_orders_and_buildts(self):
        nn = nanovdb.math.createNearestNeighborSampler(self.grid)
        self.assertFalse(hasattr(nn, "gradient"))
        self.assertFalse(hasattr(nn, "zeroCrossing"))
        tq = nanovdb.math.createTriquadraticSampler(self.grid)
        self.assertFalse(hasattr(tq, "gradient"))
        tc = nanovdb.math.createTricubicSampler(self.grid)
        self.assertFalse(hasattr(tc, "gradient"))
        self.assertFalse(hasattr(tc, "zeroCrossing"))
        bbox = nanovdb.math.CoordBBox(nanovdb.math.Coord(0), nanovdb.math.Coord(7))
        int_grid = nanovdb.tools.createInt32Grid(
            0, "ints", nanovdb.GridClass.Unknown, lambda ijk: 1, bbox
        ).grid()
        int_sampler = nanovdb.math.createTrilinearSampler(int_grid)
        self.assertFalse(hasattr(int_sampler, "gradient"))
        self.assertFalse(hasattr(int_sampler, "zeroCrossing"))


class TestChecksumMethods(unittest.TestCase):
    def test_mode_queries(self):
        handle = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Float, radius=5.0, voxelSize=1.0
        )
        grid = handle.grid()
        stored = grid.checksum()
        self.assertFalse(stored.isEmpty())
        self.assertNotEqual(stored.mode(), nanovdb.CheckMode.Disable)
        self.assertEqual(stored.isFull(), stored.mode() == nanovdb.CheckMode.Full)
        self.assertEqual(stored.isHalf(), stored.mode() == nanovdb.CheckMode.Partial)
        disabled = nanovdb.tools.evalChecksum(grid, nanovdb.CheckMode.Disable)
        self.assertTrue(disabled.isEmpty())
        self.assertEqual(disabled.mode(), nanovdb.CheckMode.Disable)


class TestCreateNanoGridClass(unittest.TestCase):
    """tools.CreateNanoGrid converter: bake with authored blind-data
    channels, filled through the writable getBlindData() NumPy view."""

    def _build_source(self):
        g = nanovdb.tools.build.FloatGrid(0.0, "blind_src", nanovdb.GridClass.Unknown)
        for i in range(8):
            g.setValue(nanovdb.math.Coord(i, 0, 0), float(i + 1))
        return g

    def test_bake_without_blind_data(self):
        src = self._build_source()
        handle = nanovdb.tools.CreateNanoGrid(src).getHandle()
        grid = handle.grid()
        self.assertIsInstance(grid, nanovdb.FloatGrid)
        acc = grid.getAccessor()
        for i in range(8):
            self.assertEqual(acc.getValue(nanovdb.math.Coord(i, 0, 0)), float(i + 1))
        self.assertEqual(grid.blindDataCount(), 0)

    def test_author_float_channel(self):
        import numpy as np

        conv = nanovdb.tools.CreateNanoGrid(self._build_source())
        channel = conv.addBlindData("uv", count=100)
        self.assertEqual(channel, 0)
        handle = conv.getHandle()
        grid = handle.grid()
        self.assertEqual(grid.blindDataCount(), 1)
        n = grid.findBlindData("uv")
        self.assertEqual(n, 0)
        meta = grid.blindMetaData(n)
        self.assertEqual(meta.valueCount, 100)
        self.assertEqual(meta.valueSize, 4)
        self.assertEqual(meta.dataType, nanovdb.GridType.Float)
        self.assertTrue(meta.isValid())
        view = grid.getBlindData(n)
        self.assertEqual(view.shape, (100,))
        self.assertTrue(np.all(view == 0.0))
        view[:] = np.arange(100, dtype=np.float32)
        again = grid.getBlindData(n)
        self.assertTrue(np.array_equal(again, np.arange(100, dtype=np.float32)))

    def test_author_vec3f_channel_with_semantic(self):
        conv = nanovdb.tools.CreateNanoGrid(self._build_source())
        conv.addBlindData(
            "N",
            count=10,
            dataType=nanovdb.GridType.Vec3f,
            dataSemantic=nanovdb.GridBlindDataSemantic.PointNormal,
        )
        grid = conv.getHandle().grid()
        n = grid.findBlindDataForSemantic(nanovdb.GridBlindDataSemantic.PointNormal)
        self.assertEqual(n, 0)
        self.assertEqual(grid.blindMetaData(n).valueSize, 12)
        self.assertEqual(grid.getBlindData(n).shape, (10, 3))

    def test_multiple_channels_from_nanogrid_source(self):
        src = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Float, radius=5.0, voxelSize=1.0
        )
        conv = nanovdb.tools.CreateNanoGrid(src.grid())
        self.assertEqual(conv.addBlindData("a", count=4), 0)
        self.assertEqual(
            conv.addBlindData("b", count=4, dataType=nanovdb.GridType.Int32), 1
        )
        grid = conv.getHandle().grid()
        self.assertEqual(grid.blindDataCount(), 2)
        self.assertEqual(grid.findBlindData("a"), 0)
        self.assertEqual(grid.findBlindData("b"), 1)
        # The baked grid still carries the source's values.
        self.assertEqual(grid.activeVoxelCount(), src.grid().activeVoxelCount())

    def test_rejects_invalid_specs(self):
        conv = nanovdb.tools.CreateNanoGrid(self._build_source())
        with self.assertRaises(ValueError):
            conv.addBlindData("x" * 300, count=1)
        with self.assertRaises(ValueError):
            conv.addBlindData(
                "bad", count=1, dataClass=nanovdb.GridBlindDataClass.GridName
            )
        with self.assertRaises(ValueError):
            conv.addBlindData("opaque", count=1, dataType=nanovdb.GridType.Unknown)
        # Unknown dataType is allowed when the element size is explicit.
        conv.addBlindData(
            "opaque", count=16, dataType=nanovdb.GridType.Unknown, size=1
        )
        self.assertEqual(conv.getHandle().grid().blindDataCount(), 1)

    def test_rejects_unsupported_source(self):
        src = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Float, radius=5.0, voxelSize=1.0
        )
        on_index = nanovdb.tools.createNanoGridOnIndex(src.grid()).grid()
        with self.assertRaises(TypeError):
            nanovdb.tools.CreateNanoGrid(on_index)
        with self.assertRaises(TypeError):
            nanovdb.tools.CreateNanoGrid(None)


class TestChannelAccessor(unittest.TestCase):
    """ChannelAccessor reads an Index/OnIndex grid's blind-data channel by
    Coord; createChannelAccessor dispatches on the channel's dataType."""

    def setUp(self):
        self.src = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Float, radius=5.0, voxelSize=1.0
        )
        self.surface = nanovdb.math.Coord(5, 0, 0)

    def test_factory_reads_channel_values(self):
        handle = nanovdb.tools.createNanoGridIndex(self.src.grid(), channels=1)
        grid = handle.grid()
        acc = nanovdb.createChannelAccessor(grid, 0)
        self.assertIsInstance(acc, nanovdb.IndexFloatChannelAccessor)
        self.assertTrue(bool(acc))
        self.assertEqual(acc.valueCount(), grid.valueCount())
        src_acc = self.src.grid().getAccessor()
        for ijk in (self.surface, nanovdb.math.Coord(0, 5, 0), nanovdb.math.Coord(0, 0, 5)):
            self.assertEqual(acc.getValue(ijk), src_acc.getValue(ijk))
            self.assertEqual(acc(ijk), src_acc(ijk))
        self.assertEqual(
            acc(self.surface.x, self.surface.y, self.surface.z),
            src_acc(self.surface),
        )
        self.assertTrue(acc.isActive(self.surface))
        value, is_on = acc.probeValue(self.surface)
        self.assertEqual(value, src_acc.getValue(self.surface))
        self.assertTrue(is_on)
        self.assertGreater(acc.getIndex(self.surface), 0)
        self.assertEqual(
            acc.getIndex(self.surface),
            acc.idx(self.surface.x, self.surface.y, self.surface.z),
        )

    def test_on_index_factory(self):
        handle = nanovdb.tools.createNanoGridOnIndex(self.src.grid(), channels=1)
        acc = nanovdb.createChannelAccessor(handle.grid())
        self.assertIsInstance(acc, nanovdb.OnIndexFloatChannelAccessor)
        src_acc = self.src.grid().getAccessor()
        self.assertEqual(acc.getValue(self.surface), src_acc.getValue(self.surface))

    def test_direct_constructor_and_set_channel(self):
        handle = nanovdb.tools.createNanoGridIndex(self.src.grid(), channels=2)
        grid = handle.grid()
        acc = nanovdb.IndexFloatChannelAccessor(grid, 1)
        self.assertTrue(bool(acc))
        acc.setChannel(0)
        self.assertTrue(bool(acc))
        with self.assertRaises(IndexError):
            acc.setChannel(2)

    def test_errors(self):
        handle = nanovdb.tools.createNanoGridIndex(self.src.grid(), channels=1)
        grid = handle.grid()
        with self.assertRaises(IndexError):
            nanovdb.createChannelAccessor(grid, 1)
        with self.assertRaises(TypeError):
            nanovdb.IndexDoubleChannelAccessor(grid, 0)
        with self.assertRaises(TypeError):
            nanovdb.createChannelAccessor(self.src.grid(), 0)
        bare = nanovdb.tools.createNanoGridIndex(self.src.grid(), channels=0)
        with self.assertRaises(IndexError):
            nanovdb.createChannelAccessor(bare.grid(), 0)


if __name__ == "__main__":
    unittest.main()
