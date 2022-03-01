# vdb_tool

This repository hosts a command-line tool, dubbed vdb_tool, that can combine any sequence of the of high-level tools available in openvdb.
For instance, it can convert a sequence of polygon meshes and particles to level sets, and perform a large number of operations on these
level set surfaces. It can also generate adaptive polygon meshes from level sets, render images and write particles, meshes or VDBs to disk
or stream VDBs to stdout so other tools can render them (using pipelining). Currently the following list of actions are supported:

| Action | Description |
|-------|-------|
| **for/each/end** | Defines the scope of a for-loop or each-loop with loop-variables that can be used for other arguments or file names |
| **config** | Load a configuration file and add the actions for processing |
| **default** | Set default values used by all subsequent actions |
| **read** | Read mesh, points and level sets as obj, ply, abc, stl, pts, vdb or nvdb files |
| **write** | Write a polygon mesh, points or level set as a obj, ply, stl or vdb file |
| **vdb2points** | Extracts points from a VDB grid |
| **mesh2ls** | Convert a polygon mesh to a narrow-band level set |
| **points2ls** | Convert points into a narrow-band level set |
| **points2vdb** | Converts points into a VDB PointDataGrid |
| **iso2ls** | Convert an iso-surface of a scalar field into a level set |
| **ls2fog** | Convert a level set into a fog volume |
| **segment** | Segment level set and float grids into its disconnected parts |
| **sphere** | Create a narrow-band level set of a sphere |
| **platonic** | Create a narrow-band level set of a tetrahedron(4), cube(6), octahedron(8), dodecahedron(12) or icosahedron(2) |
| **dilate** | Dilate a level set surface |
| **erode** |  Erode a level set surface |
| **open** |  Morphological opening of a level set surface |
| **close** |  Morphological closing of a level set surface |
| **gauss** |  Gaussian convolution of a level set surface, i.e. surface smoothing |
| **mean** |   Mean-value filtering of a level set surface |
| **median** | Median-value filtering of a level set surface |
| **union** | Union of two narrow-band level sets |
| **intersection** | Intersection of two narrow-band level sets |
| **difference** | Difference of two narrow-band level sets |
| **prune** | Prune the VDB tree of a narrow-band level set |
| **flood** | Signed flood-fill of a narrow-band level set |
| **cpt** | Closest point transform of a narrow-band level set |
| **grad**| Gradient vector of a scalar VDB |
| **curl** | Curl of a vector VDB |
| **div** | Compute the divergence of a vector VDB |
| **curvature** | Mean curvature of a scalar VDB |
| **length** | Compute the magnitude of a vector VDB |
| **multires** | Compute multi-resolution grids |
| **enright** | Advects a level set in a periodic and divergence-free velocity field. Primarily intended for benchmarks |
| **expand** | Expand the narrow band of a level set |
| **resample** | Re-sample a scalar VDB grid |
| **ls2mesh** | Convert a level set surface into an adaptive polygon mesh surface |
| **clip** | Clips one VDB grid with another VDB grid |
| **clear** | Delete cached grids and geometry |
| **render**| Render and save an image of a level set or fog VDB |
| **clear** | Deletes cached VDB grids and geometry from memory |
| **print** | Print information about the cached geometries and VDBs |

For support, bug-reports or ideas for improvements please contact ken.museth@gmail.com

# Supported file formats
| Extension | Actions | Description |
|-------|-------|-------|
| vdb | read and write | OpenVDB sparse volume files with float, Vec3f and points |
| obj | read and write | ASCII OBJ mesh files with triangle, quad or points |
| ply | read and write | Binary and ASCII PLY mesh files with triangle, quad or points |
| stl | read and write | Binary STL mesh files with triangles |
| pts | read | ASCII PTS points files with one or more point clouds |
| abc | optional read | Alembic binary mesh files |
| nvdb| optional read | NanoVDB file with points |
| conf| read and write | ASCII configuration file for this tool |
| ppm | write | Binary PPM image file |
| png | optional write | Binary PNG image file |
| jpg | optional write | Binary JPEG image file |
| exr | optional write | Binary OpenEXR image file |

# Terminology

Note that **actions** always start with one or more "-" and (except for file names) its associated **options** always contain a "=" and an optional number of leading characters used for identification, e.g. "-erode r=2" is identical to "--erode radius=2.0", but "-erode rr=2" will produce an error since "rr" does not match the first two characters of the expected option "radius". Also note that this tool maintains two lists of primitives, namely geometry (i.e. points and meshes) and VDB volumes (that may contain voxels or points). They can be referenced with "geo=n" and "vdb=n" where the integer "n" refers to "age" of the primitive. That is, "n=0" means the most recent added primitive and "n=1" means the second primitive added to the internal list. So, "-mesh2ls g=1" means convert the second to last geometry (here a polygon mesh) to a level set. If no other VDB grid exists this level set can be referenced as "vdb=0". Thus, "-gauss v=0" means perform a gaussian filter on the most recent level set. By default the most recent geometry, i.e. "g=0, or most recent level set, i.e. "v=0", is selected for processing.

---

# Building this tool
At the moment we only provide a gnu Makefile, but it's simple so it shouldn't be hard to roll your own cmake. The only mandatory dependency of this command-line tool is [OpenVDB](http://www.openvdb.org). Optional dependencies include NanoVDB, libpng, libjpeg, OpenEXR, and Alembic (enable them at the top of the Makefile).

To build this command-line tool first edit the included Makefile and make sure the variables "INCLUDES" and "LIBRARY" are updated to point to your local installation of OpenVDB. Then simply type:
```
make
```
which builds the executable release/vdb_tool or
```
make debug
```
which builds the executable debug/vdb_tool. Other build targets are _archive_ which generates a tar-ball of this repository and _clean_ which deletes all the object, executable, volume, mesh and image files.

---

# Examples

* "Hello-world" example: Create a level set sphere and save it to a file
```
vdb_tool -sphere -write sphere.vdb
```
* Same example but with options to save the file in half-float precision
```
vdb_tool -sphere -write half=true sphere.vdb
```

* Convert a polygon mesh file into a narrow-band level and save it to a file:
```
vdb_tool -read mesh.obj -mesh2ls -write level_set.vdb
```

* Convert 5 polygon mesh files, "mesh_0{1,2,3,4,5}.obj", into separate narrow-band levels and save them to the files "level_set_0{1,2,3,4,5}.vdb":
```
vdb_tool -for f=1,6,1 -read mesh_%2f.obj -mesh2ls -write level_set_%2f.vdb -end
```
 
* Generate 5 sphere with different voxel sizes and save them all into a single vdb file:
```
vdb_tool -for v=0.01,0.06,0.01 -sphere voxel=%v name=sphere_%v -end -write vdb="*" spheres.vdb
```

* Converts input points in the file points.[vdb/ply/abc/obj/pts] to a level set, perform level set operations, and written to it the file surface.vdb:
```
vdb_tool -read points.[obj/ply/abc/vdb/pts] -points2ls -dilate -gauss -erode -write surface.vdb
```

* Example with many properties of scalar and vector fields
```
vdb_tool -default keep=true -sphere -curvature -grad -curl -div -length v=1 -debug
```

For more examples [click here](examples/EXAMPLES.md)

---
# Pipelining:

vdb_tool supports unix-style pipelining, which is especially useful for interactive viewing. Specifically,
vdb_tool can read VDB grids from stdin or write VDB grid to stdout. Here are some examples:

* Redirection of stdout and stdin:
```
vdb_tool -sphere -o stdout.vdb > sphere.vdb
vdb_tool -i stdin.vdb -print < bunny.vdb
cat bunny.vdb | vdb_tool -i stdin.vdb -print
vdb_tool -sphere -o stdout.vdb | gzip > sphere.vdb.gz
gzip -dc sphere.vdb.gz | vdb_tool -i stdin.vdb -print
```

* Pipelining multiple instances of vdb_tool (see note below):
```
vdb_tool -sphere -o stdout.vdb | vdb_tool -i stdin.vdb -dilate -o stdout.vdb > sphere.vdb
```
or with explicit semantics
```
vdb_tool -sphere -o stdout.vdb | vdb_tool -i stdin.vdb -dilate -o stdout.vdb > sphere.vdb
```

* Note that the example above is slow due to serialization of the VDB grid. A much faster alternative is:
```
vdb_tool -sphere -dilate -o stdout.vdb > sphere.vdb
```
or with explicit semantics
```
vdb_tool -sphere -dilate -o stdout.vdb > sphere.vdb
```

* Pipelining vdb_tool with vdb_view for interactive viewing
```
vdb_tool -sphere -dilate -o stdout.vdb | vdb_view
```

* View a sequence of animated level sets
```
vdb_tool -sphere d=80 r=0.15 c=0.35,0.35,0.35 -for i=1,20,1 -enright dt=0.05 k=1 -end -o stdout.vdb | vdb_view
vdb_tool -platonic d=128 f=8 s=0.15 c=0.35,0.35,0.35 -for i=1,20,1 -enright dt=0.05 k=1 -end -o stdout.vdb | vdb_view
```

Arguably the last example is the only application of pipelining that should be used in practice (since there is no faster alternative).

---
# To Do List:

- [x] vdb_tool::readGeo
- [x] vdb_tool::readVDB
- [x] vdb_tool::particlesToLevelSet
- [x] vdb_tool::processLevelSet
- [x] vdb_tool::offsetLevelSet
- [x] vdb_tool::filterLevelSet
- [x] vdb_tool::levelSetToMesh
- [x] vdb_tool::writeGeo
- [x] vdb_tool::writeVDB
- [x] read ASCI obj particle files
- [x] read ASCI ply particle files
- [x] read binary ply particle files
- [x] write binary ply mesh files
- [x] write ascii obj mesh files
- [x] Geometry::readVdb
- [x] Geometry::readPts
- [x] define time and space order
- [x] Mesh::readPly
- [x] vdb_tool::readMesh
- [x] vdb_tool::meshToLevelSet
- [x] Geometry::readObj
- [x] Geometry::readPly
- [x] Geometry::readNvdb
- [x] vdb_tool::writeVDB
- [x] allow actions to have multiple "-"
- [x] add "-sphere"
- [x] add volume/geometry ages to all actions
- [x] add CSG operations
- [x] "-read" supports multiple files
- [x] "-write" supports multiple files
- [x] added "-print"
- [x] works with tcsh, sh, ksh, and zsh shells
- [x] added "-default"
- [x] cache a list of base grids instead of FloatGrids
- [x] -points2vdb : points -> PointDataGrid
- [x] -vdb2points : PointDataGrid -> points
- [x] -write geo=1 vdb=1,3 file.ply file.vdb
- [x] -iso2ls, convert scalar field to level set
- [x] -ls2fog, convert level set to fog volume
- [x] -scatter, scatter points
- [x] -prune, prune level set
- [x] -flood, signed flood fill of level set
- [x] -multires, generate multi-resolution grids
- [x] -expand, expand narrow band of level set
- [x] -cpt, generate closest-point transfer
- [x] -grad, generate gradient field
- [x] -div, generate divergence from vector field
- [x] -curl, generate curl from vector field
- [x] -curvature, generate mean curvature from scalar field
- [x] -length, generate length of vector field
- [x] -render, render level set and fog volumes
- [x] -enright, performs advection test on level set
- [x] -for i=0,10,1 -end
- [x] -each s=str1,str2 -end
- [x] -read grids=sphere file_%4i.vdb
- [x] Geometry::readSTL
- [x] Geometry::writeSTL
- [x] -clip against either a mask grid, bbox or frustum
- [x] Added local counter "%I" to for-loops
- [x] Added global counter "%G"
- [x] add Tool::savePNG
- [x] add Tool::saveEXR
- [x] -platonic faces=4 name=\_DERIVE\_
- [x] -segment vdb=0 keep=0
- [x] -resample vdb=0[,1] scale=0 translate=0,0,0 order=1[0|2] keep=0
- [x] add Geometry::readABC
- [x] add support for unix pipelining
- [x] add Tool::saveJPG
- [x] add Geometry::read/write to support streaming
- [x] -read stdin.[ply,obj,stl,geo,vdb]
- [x] -write stdout.[ply,obj,stl,geo,vdb]
- [x] actions can now have an optional alias, e.g. -read, -i
- [x] -write file.nvdb stdout.nvdb
- [x] -write bits=32|16|8|N codec=blosc|zip|active
- [x] -help read,write,ls2mesh brief=true
- [x] use openvdb namespace
- [x] Major revision with Parser.h
- [ ] Combine: -min, -max, -sum
- [ ] -xform (translate and scale grid transforms)
- [ ] -merge
- [ ] -points2mask
- [ ] -erodeTopology
- [ ] use cmake
- [ ] add Geometry::readUSD

Private repository by Ken Museth