# vdb_tool

This command-line tool, dubbed vdb_tool, can combine any number of the of high-level tools available in openvdb/tools. For instance, it can convert a sequence of polygon meshes and particles to level sets, and perform a large number of operations on these level set surfaces. It can also generate adaptive polygon meshes from level sets, ray-trace images and export particles, meshes or VDBs to disk or even stream VDBs to STDOUT so other tools can render them (using pipelining). We denote the operations **actions**, and their arguments **options**. This command-line tool also supports a string-evaluation language that can be used to define procedural expressions for options of the actions. Currently the following list of actions are supported:

| Action | Description |
|-------|-------|
| **for/end** | Defines the scope of a for-loop with a range for a loop-variable |
| **each/end** | Defines the scope of an each-loop with a list for a loop-variable |
| **if/end** | If-statement used to enable/disable actions |
| **eval** | Evaluate an expression written in our Reverse Polish Notation (see below) |
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
| **min** | Composite two grid by means of min |
| **max** | Composite two grid by means of max |
| **sum** | Composite two grid by means of sum |
| **multires** | Compute multi-resolution grids |
| **enright** | Advects a level set in a periodic and divergence-free velocity field. Primarily intended for benchmarks |
| **expand** | Expand the narrow band of a level set |
| **resample** | Re-sample a scalar VDB grid |
| **transform** | Apply affine transformations to VDB grids |
| **ls2mesh** | Convert a level set surface into an adaptive polygon mesh surface |
| **clip** | Clips one VDB grid with another VDB grid or a bbox or frustum |
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
| nvdb| optional read and write | NanoVDB file with voxels or points |
| txt | read and write | ASCII configuration file for this tool |
| ppm | write | Binary PPM image file |
| png | optional write | Binary PNG image file |
| jpg | optional write | Binary JPEG image file |
| exr | optional write | Binary OpenEXR image file |

# Terminology

We introduce three terms: **actions**, **options** and **instructions**. Actions are high-level openvdb tools, which each have unique options, e.g. -mesh2ls geo=1 voxel=0.1, where "-mesh2ls" is an action with two options "geo" and "voxel". Operations are low-level functions in our stack-based programming language (see below). These expressions start with "{" and ends with "}", and ":" is used to separate values and instructions. E.g. {1:2:+} is an expression with two values (1 and 2) and one instruction "+", and it reduces to the string value "3".

Note that **actions** always start with one or more "-" and (except for file names) its associated **options** always contain a "=" and an optional number of leading characters used for identification, e.g. "-erode r=2" is identical to "-erode radius=2.0", but "-erode rr=2" will produce an error since "rr" does not match the first two characters of the expected option "radius". Also note that this tool maintains two lists of primitives, namely geometry (i.e. points and meshes) and VDB volumes (that may contain voxels or points). They can be referenced with "geo=n" and "vdb=n" where the integer "n" refers to "age" of the primitive. That is, "n=0" means the most recent added primitive and "n=1" means the second primitive added to the internal list. So, "-mesh2ls g=1" means convert the second to last geometry (here a polygon mesh) to a level set. If no other VDB grid exists this level set can be referenced as "vdb=0". Thus, "-gauss v=0" means perform a gaussian filter on the most recent level set. By default the most recent geometry, i.e. "g=0, or most recent level set, i.e. "v=0", is selected for processing.

# Stack-based string expressions

This tool supports its own light-weight stack-oriented programming language that is (very loosely) inspired by Forth. Specifically, it uses Reverse Polish Notation (RPN) to define instructions that are evaluated during paring of the command-line arguments (options to be precise). All such expressions start with the character "{", ends with "}", and arguments are separated by ":". Variables starting with "\$" are substituted by its (previously) defined values, and variables starting with "@" are stored in memory. So, "{1:2:+:@x}" is conceptually equivalent to "x = 1 + 2". Conversely, "{\$x:++}" is conceptually equivalent "2 + 1 = 3" since "x=2" was already saved to memory. This is especially useful in combination loops, e.g. "-quiet -for i=1,3,1 -eval {\$i:+} -end" will print 2 and 3 to the terminal. Branching is also supported, "{$x:1:>:if(0.5:sin?0.3:cos)}" is conceptually equal to "if (x>1) sin(0.5) else cos(0.3)". See the root-searching example below or run vdb_tool -eval help="*" to see a list of all instructions currently supported by this scripting language. Note that since this language uses characters that are interpreted by most shells it is necessary to use single quotes around strings! This is of course not the case when using config files.

# Building this tool

This tool is using CMake for build on Linux and Windows.
The only mandatory dependency of is [OpenVDB](http://www.openvdb.org). Optional dependencies include NanoVDB, libpng, libjpeg, OpenEXR, and Alembic. To enable them use the `-DUSE_<name>=ON` flags. See the CMakeLists.txt for details.

The included unit test are using Gtest. Add `-DBUILD_TEST=ON` to the cmake command line to build it.

## Building OpenVDB

Follow the instructions at OpenVDB`s [github page](https://github.com/AcademySoftwareFoundation/openvdb#developer-quick-start)

Make sure to build with NanoVDB support, if you intend to use vdb_tool's NanoVDB features.

## Building vdb_tool on Linux

To generate the makefile, navigate to the cloned directory of vdb_tool, then follow these steps:
```bash
mkdir build
cd build
cmake -DOPENVDB_CMAKE_PATH=/usr/local/lib/cmake/OpenVDB -DUSE_ALL=ON -DBUILD_TEST=ON ..
```
Update the OpenVDB cmake path above as needed.

To build in debug mode, add `-DCMAKE_BUILD_TYPE=Debug` to the cmake command above.

To build use
```bash
cmake --build . --parallel 2
```
or
```bash
make -j 2
```

## Building on Windows

### Install CMake

Install from cmake.org or with Chocolatey:
```bash
choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System'
```

### Install optional dependencies

Gtest for the unit tests
```bash
vcpkg install gtest:x64-windows
```

Other optional dependencies
```bash
vcpkg install libpng:x64-windows
vcpkg install libjpeg-turbo:x64-windows
vcpkg install openexr:x64-windows
vcpkg install alembic:x64-windows
```

### Building

```bash
mkdir build
cd build
cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_TOOLCHAIN_FILE=<path to vcpkg root>\scripts\buildsystems\vcpkg.cmake -A x64 -DOPENVDB_CMAKE_PATH=<OpenVDB install path>\lib\cmake\OpenVDB ..
cmake --build . --config Release --parallel 2
```


# Examples

* Get help on actions and their options
```
vdb_tool -help
vdb_tool -help read write
```

* Get help on instructions
```
vdb_tool -eval help="*"
vdb_tool -eval help=if,switch
```


* "Hello-world" example: Create a level set sphere and save it to a file
```
vdb_tool -sphere -write sphere.vdb
```
* Same example but with options to save the file in half-float precision
```
vdb_tool -sphere -write bits=16 sphere.vdb
```

* Convert a polygon mesh file into a narrow-band level and save it to a file:
```
vdb_tool -read mesh.obj -mesh2ls -write level_set.vdb
```

* Convert a polygon mesh file into a narrow-band level with a transform that matches another vdb:
```
vdb_tool -read mesh.obj,reference.vdb -mesh2ls vdb=0 -write level_set.vdb
```

* Convert 5 polygon mesh files, "mesh_0{1,2,3,4,5}.obj", into separate narrow-band levels and save them to the files "level_set_0{1,2,3,4,5}.vdb":
```
vdb_tool -for f=1,6 -read mesh_'{$f:2:pad0}'.obj -mesh2ls -write level_set_'{$f:2:pad0}'.vdb -end
```
 
* Generate 5 sphere with different voxel sizes and save them all into a single vdb file:
```
vdb_tool -for v=0.01,0.06,0.01 -sphere voxel='{$v}' name=sphere_%v -end -write vdb="*" spheres.vdb
```

* Generate spheres that are rotating along a parametric circle
```
vdb_tool -for degree=0,360,10 -eval '{$degree:d2r:@radian}' -sphere center='({$radian:cos},{$radian:sin},0)' name=sphere_'{$degree}' -end -write vdb="*" spheres.vdb
```

* Converts input points in the file points.[vdb/ply/abc/obj/pts] to a level set, perform level set actions, and written to it the file surface.vdb:
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
vdb_tool -sphere d=80 r=0.15 c=0.35,0.35,0.35 -for i=1,20 -enright dt=0.05 k=1 -end -o stdout.vdb | vdb_view
vdb_tool -platonic d=128 f=8 s=0.15 c=0.35,0.35,0.35 -for i=1,20 -enright dt=0.05 k=1 -end -o stdout.vdb | vdb_view
```

# String evaluation:

* Define, modify and access custom variables<br>
Define the local variable G=0 and increment it inside a for-loop.
```
vdb_tool -quiet -eval '{0:@G}' -for i=10,15 -eval '{$G:++:@G}G = {$G}' -end
```
which prints the output:<br>
G = 1<br>
G = 2<br>
G = 3<br>
G = 4<br>
G = 5<br>

* Find real roots of a quadratic polynomial:<br>
For convenience the computation is split in the following three steps:<br>
a=1, b=-8, c=5<br>
c=b^2-4ac, a=-2a<br>
if (c==0) one solution: b/a, else if (c>0) two solutions: (b+-sqrt(c))/a
```
vdb_tool -eval '{1:@a:-8:@b:5:@c}' '{$b:pow2:4:$a:*:$c:*:-:@c:-2:$a:*:@a}' '{$c:0:==:if($b:$a:/):$c:0:>:if($c:sqrt:dup:$b:+:$a:/:$b:rot:-:$a:/):squash}'
//vdb_tool -eval '(a=1;b=-8;c=5;c=b^2-4*a*c;a=-2*a;if(c==0){s1=b/a;}if(c>0){s1=(b+sqrt(c))/a;s2=b-sqrt(c))/a;}'
```
which prints the two real roots: 0.683375 7.316625. Changing b=c=4 prints the single real root -2.

* Use scripting to access properties of grids and geometry

Read both a vdb and mesh file and convert the mesh to a vdb with twice the voxel size of the input vdb.
```
vdb_tool -read bunny.vdb dragon.ply -mesh2ls voxel='{0:voxelSize:2:*}' -print
//vdb_tool -read bunny.vdb dragon.ply -mesh2ls voxel='(2*voxelSize(0);)' -print
```

* Loop skipping and expressions

```
vdb_tool -each i=1 -for j='{$i}',2 -end -end
```

which prints:<br>
Processing: i = 1 counter = 0<br>
Processing: j = 1 counter = 0

```
vdb_tool -each i=2 -for j='{$i}',2 -end -end
```
produces the output to:<br>
Processing: i = 2 counter = 0

```
vdb_tool -each i= -for j='{$i}',2 -end -end
```
produces no output, since both loops are skipped

* If-statement to level set grids

Read multiple grids, and render all level set grids

```
vdb_tool -read boat_points.vdb -for v=0,'{gridCount}' -if '{$v:isLS}' -render vdb='{$v}' -end -end
```

* Find and render thumbnails of all level sets in an entire directory structure
```
vdb_tool -each file=`find ~/dev/data -name '*.vdb'` -read '{$file}' -for grid=0,'{gridCount}' -if '{$grid:isLS}' -render vdb='{$grid}' thumbnail_'{$grid:gridName}'.ppm image=256x256 keep=1 -end -end -clear -end
```
Most of the arguments should be self-explanatory, but at least two deserve an explanation: -render has the option keep=1 because otherwise rendered grids are removed from the stack which invalidates {gridCount}, and -clear is added to avoid accumulating all grids as multiple files are loaded.

---
