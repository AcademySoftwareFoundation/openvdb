#!/usr/bin/env python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

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
            grid = handle.floatGrid(i)
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
            grid = handle.floatGrid(i)
            self.assertIsNotNone(grid)
            checksum = grid.checksum()
            nanovdb.tools.updateChecksum(grid, nanovdb.CheckMode.Default)
            updatedChecksum = grid.checksum()
            self.assertEqual(checksum, updatedChecksum)


class TestGridHandleExchange(unittest.TestCase):
    def test_list_to_vector(self):
        handle = nanovdb.tools.createLevelSetTorus(nanovdb.GridType.Double)
        self.assertEqual(handle.gridCount(), 1)
        self.assertIsNotNone(handle.doubleGrid())
        handles = [handle, handle]
        dstFile = tempfile.NamedTemporaryFile()
        nanovdb.io.writeGrids(dstFile.name, handles)


class TestReadWriteGrids(unittest.TestCase):
    def setUp(self):
        self.gridName = "sphere_ls"
        sphereHandle = nanovdb.tools.createLevelSetSphere(
            nanovdb.GridType.Float, name=self.gridName
        )
        self.srcFile = tempfile.NamedTemporaryFile()
        nanovdb.io.writeGrid(self.srcFile.name, sphereHandle)
        nanovdb.io.writeGrid(self.srcFile.name, sphereHandle)
        self.dstFile = tempfile.NamedTemporaryFile()

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
            self.assertEqual(metadata.padding, 0)
            self.assertEqual(metadata.version, nanovdb.Version())

    def test_read_write_grid(self):
        self.assertTrue(nanovdb.io.hasGrid(self.srcFile.name, self.gridName))
        handle = nanovdb.io.readGrid(self.srcFile.name)
        self.assertEqual(handle.gridCount(), 1)
        for i in range(handle.gridCount()):
            self.assertTrue(handle.gridSize(i) > 0)
            self.assertEqual(handle.gridType(i), nanovdb.GridType.Float)
            grid = handle.floatGrid(i)
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
class TestDeviceReadWriteGrids(unittest.TestCase):
    def setUp(self):
        self.gridName = "sphere_ls"
        sphereHandle = nanovdb.tools.cuda.createLevelSetSphere(
            nanovdb.GridType.Float, name=self.gridName
        )
        self.srcFile = tempfile.NamedTemporaryFile()
        nanovdb.io.deviceWriteGrid(self.srcFile.name, sphereHandle)
        nanovdb.io.deviceWriteGrid(self.srcFile.name, sphereHandle)
        self.dstFile = tempfile.NamedTemporaryFile()

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
            self.assertEqual(metadata.padding, 0)
            self.assertEqual(metadata.version, nanovdb.Version())

    def test_read_write_grid(self):
        self.assertTrue(nanovdb.io.hasGrid(self.srcFile.name, self.gridName))
        handle = nanovdb.io.deviceReadGrid(self.srcFile.name)
        self.assertEqual(handle.gridCount(), 1)
        for i in range(handle.gridCount()):
            self.assertTrue(handle.gridSize(i) > 0)
            self.assertEqual(handle.gridType(i), nanovdb.GridType.Float)
            grid = handle.floatGrid(i)
            self.assertIsNotNone(grid)
            deviceGrid = handle.deviceFloatGrid(i)
            self.assertIsNone(deviceGrid)
            handle.deviceUpload()
            deviceGrid = handle.deviceFloatGrid(i)
            handle.deviceDownload()
            grid = handle.floatGrid(i)
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


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
class TestPointsToGrid(unittest.TestCase):
    def test_points_to_grid(self):
        try:
            import torch

            tensor = torch.tensor(
                [[1, 2, 3]], dtype=torch.int32, device=torch.device("cuda", 0)
            )
            handle = nanovdb.tools.cuda.pointsToRGBA8Grid(tensor)
            deviceGrid = handle.deviceRGBA8Grid()
            self.assertTrue(deviceGrid)
            grid = handle.rgba8Grid()
            self.assertFalse(grid)
            handle.deviceDownload()
            grid = handle.rgba8Grid()
            self.assertTrue(grid)
        except ImportError:
            print("PyTorch not found. Skipping...")
            pass


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
class TestSignedFloodFill(unittest.TestCase):
    def test_signed_flood_fill_float(self):
        handle = nanovdb.tools.cuda.createLevelSetSphere(nanovdb.GridType.Float, 100)
        grid = handle.floatGrid()
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
        deviceGrid = handle.deviceFloatGrid(0)
        self.assertIsNotNone(deviceGrid)
        nanovdb.tools.cuda.signedFloodFill(deviceGrid)
        handle.deviceDownload()
        grid = handle.floatGrid()
        self.assertIsNotNone(grid)
        accessor = grid.getAccessor()
        self.assertEqual(3.0, accessor(103, 0, 0))
        self.assertEqual(0.0, accessor(100, 0, 0))
        self.assertEqual(-3.0, accessor(97, 0, 0))
        # self.assertFalse(grid.isLexicographic())
        self.assertTrue(grid.isBreadthFirst())

    def test_signed_flood_fill_double(self):
        handle = nanovdb.tools.cuda.createLevelSetSphere(nanovdb.GridType.Double, 100)
        grid = handle.doubleGrid()
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
        deviceGrid = handle.deviceDoubleGrid(0)
        self.assertIsNotNone(deviceGrid)
        nanovdb.tools.cuda.signedFloodFill(deviceGrid)
        handle.deviceDownload()
        grid = handle.doubleGrid()
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
            grid = handle.deviceFloatGrid()
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
            nanovdb.math.cuda.sampleFromVoxels(points, grid, values, gradients)
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
            grid = handle.deviceDoubleGrid()
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
            nanovdb.math.cuda.sampleFromVoxels(points, grid, values)
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
        grid = handle.floatGrid()
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
        grid = handle.doubleGrid()
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
            grid = handle.floatGrid(i)
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
            grid = handle.doubleGrid(i)
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
            grid = handle.int32Grid(i)
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
            grid = handle.vec3fGrid(i)
            self.assertIsNotNone(grid)
            self.assertTrue(grid.activeVoxelCount() > 0)
            self.assertTrue(grid.isSequential())
            self.assertEqual(grid.gridName(), "test_vec3f")
            self.assertEqual(grid.gridClass(), nanovdb.GridClass.Unknown)


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
                grid = handle.floatGrid(i)
                self.assertIsNotNone(grid)
                self.assertTrue(grid.activeVoxelCount() > 0)
                self.assertTrue(grid.isSequential())
                self.assertEqual(grid.gridName(), "test_sphere")
                self.assertEqual(grid.gridClass(), nanovdb.GridClass.LevelSet)
        except ImportError:
            print("openvdb not found. Skipping...")
            pass


if __name__ == "__main__":
    unittest.main()
