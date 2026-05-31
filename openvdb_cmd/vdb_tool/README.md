# vdb_tool
The vdb_tool is a versatile command-line utility that chains together high-level operations from the OpenVDB library. It can convert polygon meshes and particles into level sets, perform complex volumetric transformations, and generate adaptive meshes or ray-traced images. Results can be exported as particles, meshes, or VDB files, or streamed directly to STDOUT for seamless pipelining with other renderers. We denote the operations **actions**, and their arguments **options**. Any sequence of **actions** and their **options** can be exported and imported to configuration files, which allows convenient reuse. This command-line tool also supports a string-evaluation language that can be used to define procedural expressions for options of the actions. Currently the following list of actions are supported:

| Action | Description |
|-------|-------|
| **for/end** | Defines the scope of a for-loop with a range for a loop-variable |
| **files/end** | Defines the scope of loop over all files in a specific directory structure |
| **each/end** | Defines the scope of an each-loop with a list for a loop-variable |
| **if/end** | If-statement used to enable/disable actions |
| **eval** | Evaluate an expression written in our Reverse Polish Notation (see below) |
| **config** | Load a configuration file and add the actions for processing |
| **default** | Set default values used by all subsequent actions |
| **read** | Read mesh, points, grids or config as obj, ply, abc, stl, off, pts, xyz, e57 (PDAL), usd, usda, usdc, usdz, vdb, nvdb or txt files |
| **write** | Write a polygon mesh, points, vdb or config as a obj, ply, stl, off, abc, vdb, or txt file |
| **vdb2points** | Extracts points from a VDB grid |
| **mesh2ls** | Convert a (water-tight) polygon mesh to a narrow-band signed distance field |
| **mesh2udf** | Convert an arbitrary polygon mesh to a narrow-band unsigned distance field |
| **soup2ls** | Convert an arbitrary polygon soup to a narrow-band signed distance field |
| **points2ls** | Convert points into a narrow-band level set |
| **points2vdb** | Converts points into a VDB PointDataGrid |
| **iso2ls** | Convert an iso-surface of a scalar field into a level set |
| **ls2fog** | Convert a level set into a fog volume |
| **quad2tri** | Convert all quads in a mesh to triangles |
| **segment** | Segment level set and float grids into its disconnected parts |
| **sphere** | Create a narrow-band level set of a sphere |
| **platonic** | Create a narrow-band level set of a tetrahedron(4), cube(6), octahedron(8), dodecahedron(12) or icosahedron(20) |
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
| **min** | Composite two grids by means of min |
| **max** | Composite two grids by means of max |
| **sum** | Composite two grids by means of sum |
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
| **slice** | Generate image files of slices through a VDB grid |
| **img2mpeg** | Convert multiple image files to an mpeg movie file |
| **calc** | Calculator |
| **forAllValues** | Apply a math kernel to every value in a grid (see "Per-voxel math kernels" below) |
| **forOnValues** | Apply a math kernel to every active value in a grid (see "Per-voxel math kernels" below) |
| **forOffValues** | Apply a math kernel to every inactive value in a grid (see "Per-voxel math kernels" below) |

For support, bug-reports or ideas for improvements please contact ken.museth@gmail.com

# Supported file formats
| Extension | Actions | Description |
|-------|-------|-------|
| vdb | read and write | OpenVDB sparse volume files with float, Vec3f and points |
| obj | read and write | ASCII OBJ mesh files with triangles, quads or points |
| ply | read and write | Binary and ASCII PLY mesh files with triangles, quads or points |
| stl | read and write | Binary STL mesh files with triangles |
| off | read and write | ASCII OFF mesh files with triangles, quads or points |
| xyz | read and write | ASCII XYZ files with x y z coordinates, |
| pts | read | ASCII PTS points files with one or more point clouds |
| abc | optional read and write | Alembic binary mesh files |
| usd, usda, usdc, usdz | optional read | OpenUSD scene files; reads UsdGeomMesh and UsdGeomPoints prims |
| nvdb| optional read and write | NanoVDB file with voxels or points |
| txt | read and write | ASCII configuration file for this tool |
| ppm | write | Binary PPM image file |
| png | optional write | Binary PNG image file |
| jpg | optional write | Binary JPEG image file |
| exr | optional write | Binary OpenEXR image file |

# Terminology

We introduce the following terms: **actions**, **options**, **expressions**, and **instructions**. Actions are high-level openvdb tools, which each have unique options, e.g. -mesh2ls geo=1 voxel=0.1, where "-mesh2ls" is an action with two options "geo" and "voxel". Expressions are strings of code with one or more low-level instructions in our stack-based programming language (see below). These expressions start with "{" and ends with "}", and ":" is used to separate values and instructions. E.g. {1:2:+} is an expression with two values (1 and 2) and one instruction "+", and it reduces to the string value "3". See section on the "Stack-based string expressions" below for more details.

Note that **actions** always start with one or more "-" and (except for file names) its associated **options** always contain a "=" and an optional number of leading characters used for identification, e.g. "-erode r=2" is identical to "-erode radius=2.0", but "-erode rr=2" will produce an error since "rr" does not match the first two characters of any option associated with the action "erode".

Note that this tool maintains two stacks of primitives, namely geometry (i.e. points and polygon meshes) and VDB volumes (that may contain voxels or points). They can be referenced respectively with "geo=n" and "vdb=n" where the integer "n" refers to "age" of the primitive, i.e. its order on the stack. That is, "n=0" means the most recently added primitive and "n=1" means the second primitive added to the internal stack. So, "-mesh2ls g=1" means convert the second to last geometry (here a polygon mesh) to a level set. If no other VDB grid exists this output level set can subsequently be referenced as "vdb=0". Thus, "-gauss v=0" means perform a gaussian filter on the most recently added level set VDB. By default the most recent geometry, i.e. "g=0, or most recent level set, i.e. "v=0", is selected for processing.

# Stack-based string expressions

This tool supports its own light-weight stack-oriented programming language that is (very loosely) inspired by Forth. Specifically, it uses Reverse Polish Notation (RPN) to define instructions that are evaluated during paring of the command-line arguments (options to be precise). All such expressions start with the character "{", ends with "}", and arguments are separated by ":". Variables starting with "\$" are substituted by its (previously) defined values, and variables starting with "@" are stored in memory. So, "{1:2:+:@x}" is conceptually equivalent to "x = 1 + 2". Conversely, "{\$x:++}" is conceptually equivalent "2 + 1 = 3" since "x=2" was already saved to memory. This is especially useful in combination with loops, e.g. "-quiet -for i=1,3,1 -eval {\$i:++} -end" will print 2 and 3 to the terminal. Branching is also supported, e.g. "radius={$x:1:>:if(0.5:sin?0.3:cos)}" is conceptually equal to "if (x>1) radius=sin(0.5) else radius=cos(0.3)". See the root-searching example below or run vdb_tool -eval help="*" to see a list of all instructions currently supported by this scripting language. Note that since this language uses characters that are interpreted by most shells it is necessary to use single quotes around strings! This is of course not the case when using config files.

# Standalone calculator (-calc)

The `-calc` action runs a single math expression through the same compiler used by the per-voxel kernels (see next section), but at command-line scope: input variables are read from the Processor's string memory (the same `{...}` namespace described above), and outputs (intermediate slot values and the trailing-LHS name) are written back to that memory. The numeric result is printed **only when the final statement is a plain expression** (no trailing `=`); a kernel that ends in an assignment is silent, since its outputs are already accessible via memory.

The expression can be supplied either as a bare positional argument (`-calc 'x=1+2'`) or via the explicit option syntax (`-calc kernel='x=1+2'`); the two are equivalent. The bare form is supported because `-calc`'s single option is registered with `Action::kAnonymousGreedy`, so the parser accepts tokens that contain `=` without trying to interpret the prefix as an option name.

Examples:

```bash
# Plain expression: result is echoed.
vdb_tool -calc '1+2+3'                          # prints 6

# Single assignment: silent on -calc; the trailing LHS stores the result
# in memory. Retrieval via {$x} comes from the stack-based expression
# language above.
vdb_tool -calc 'x=1+2' -eval str='{$x}'         # prints 3.000000

# Multi-statement: intermediate slots persist into the Processor memory
# too. The trailing assignment is silent.
vdb_tool -calc 'a=1+2; b=a*3' -eval str='a={$a} b={$b}'
# prints: a=3.000000 b=9.000000

# Inspect everything written to memory with -print mem=1.
vdb_tool -calc 'a=1+2;b=a+3' -print mem=1
# prints (no leading number; -calc was silent because the kernel ended in
# an assignment):
#         ... -print's "Variables" section:
#         a=3.000000
#         b=6.000000

# Drive -for's start, stop, step from values computed by -calc.
vdb_tool -calc 'a=1;b=5;c=1' -for x='{$a},{$b},{$c}' -end
# prints:
#         Processing: x = 1.000000, counter #x = 0
#         Processing: x = 2.000000, counter #x = 1
#         Processing: x = 3.000000, counter #x = 2
#         Processing: x = 4.000000, counter #x = 3

# Feed values into -calc from prior -eval set operations. The final
# statement is a plain expression, so the result is echoed.
vdb_tool -eval str='{2:@x}' -calc '3*sin(x)+1'  # prints 3*sin(2)+1 ≈ 3.727
```

A few rules:

- **Undefined variables are errors.** If the kernel reads a name that isn't in the Processor's memory, `-calc` throws with a message naming it. Set it first with `-eval str='{<value>:@<name>}'` (or via an earlier `-calc`). Reading a memory entry that exists but isn't a valid float (e.g. set by the typo `{n:@n}`) produces a diagnostic naming the variable and suggesting `{0:@n}`.
- **Reads don't rewrite memory.** A pure input read like `-calc n` leaves `mem["n"]` untouched, preserving the original string representation. This matters because `-for n=0,2,1` stores `n` as the int string `"0"`, which would break downstream int comparators if `-calc` rewrote it to `"0.000000"`. Only outputs (slots and the trailing-LHS) are written back.
- **Floats round-trip via `std::to_string`** (6 decimals). This is fine for casual chaining; for higher-precision pipelines, do all the math in one kernel and read only the final result.
- **Shell quoting.** Always single-quote the kernel value so `*`, `(`, `$`, `;`, and `=` aren't interpreted by the shell.

# Per-voxel math kernels (forAllValues / forOnValues / forOffValues)

The actions `-forAllValues`, `-forOnValues`, and `-forOffValues` apply a user-defined math expression to every value, every active value, or every inactive value in a `FloatGrid`. The expression is supplied via the `kernel` option, compiled once into a compact bytecode, and then evaluated in parallel across the grid &mdash; no JIT, no extra dependencies, no per-voxel string parsing.

The reserved variable `v` is bound to the current voxel value. Any other identifier in the expression is looked up once in the Processor's string memory (the same `{...}` namespace used by `-eval` and `-calc`) and bound as a per-voxel constant; a name that isn't in memory triggers an error before any voxels are touched. This lets a kernel pull scalars set by an earlier `-eval '{2:@scale}'` or `-calc 'scale=1.5'` and combine them with the voxel value, e.g. `-forOnValues 'scale*v + bias'`.

The voxel-variable name is configurable via the `use=` option (default `v`); for example `-forOnValues 'sin(x)+1' use=x` reads better if you prefer `x`, and is equivalent to `-forOnValues 'sin(v)+1'` &mdash; the chosen name is treated as the per-voxel input and excluded from the Processor-memory lookup performed for every other identifier.

The same expression can be written in any of three equivalent syntaxes:

| Syntax | Example |
|---|---|
| **Infix** (familiar to math users) | `'sin(v) + 2*v*v'` |
| **RPN** (same language as the rest of vdb_tool's expressions) | `'$v:sin:$v:pow2:2:*:+'` |
| **Infix multi-statement** (with assignment and reusable locals) | `'t = v*v; t + sin(t)'` |

All three compile to identical-shape bytecode. The compiler dispatches on the markers it sees: `=` or `;` &rarr; multi-statement infix; otherwise `:` or `$` &rarr; RPN; otherwise plain infix.

The kernel can be supplied either as a bare positional argument (`-forOnValues 'sin(v)+1'`) or via the explicit `kernel='...'` form (`-forOnValues kernel='sin(v)+1'`); the two are equivalent. Other options of the same action (e.g. `keep=true`, `class=ls`) parse normally regardless of which form you use, because the greedy fallback only kicks in for tokens whose `name=` prefix isn't a recognized option.

### Operators (infix)

| Op | Precedence | Associativity |
|----|-----------|---------------|
| `^` (power)      | 4 | right |
| unary `-` / `+`  | 5 | right (unary `+` is a no-op) |
| `*` `/`          | 3 | left |
| `+` `-` (binary) | 2 | left |

### Functions

| Unary  | `neg` `abs` `inv` `sqrt` `sin` `cos` `tan` `asin` `acos` `atan` `exp` `ln` `log` `floor` `ceil` `pow2` `pow3` |
|--------|---|
| **Binary** | `pow(a, b)` (also `a^b`), `min(a, b)`, `max(a, b)` |

### Constants

`pi` and `e` are recognized as named literals in all three syntaxes. They cannot be the target of an assignment.

### Multi-statement programs

Multi-statement kernels are separated by `;`. Each statement except the last must be an assignment `name = <expr>`, declaring a *local slot* whose value is reused by subsequent statements. The final statement may be either a plain expression or an assignment; either way its right-hand side is the value written back to the voxel. A trailing semicolon is fine.

```bash
# Reuse a squared subexpression instead of recomputing it.
vdb_tool -read in.vdb -forAllValues 't = v*v; t + sin(t)' -write out.vdb

# Multiple intermediate slots; the final assignment's LHS is documentation.
vdb_tool -read in.vdb -forOnValues 'a = sin(v); b = cos(v); v = a*a + b*b' -write out.vdb

# Pull scalar inputs from memory and combine with the voxel value: scale and
# bias were set earlier by -eval (or -calc) and applied uniformly to every
# active voxel.
vdb_tool -read in.vdb -eval '{2:@scale}' -eval '{0.5:@bias}' -forOnValues 'scale*v + bias' -write out.vdb
```

A slot name shadows any input variable of the same name from the point of its first assignment, mirroring ordinary scripting-language scoping. So `'v = v*2; v + 1'` reads the input `v` once on the right-hand side of the first statement, then reads the slot for every subsequent reference.

### Example commands

```bash
# Quadratic remap: y = sin(v) + 2*v^2
vdb_tool -read in.vdb -forAllValues 'sin(v) + 2*v*v' -write out.vdb

# Clamp negative values to zero (rectifier / ReLU-style):
vdb_tool -read in.vdb -forOnValues 'max(v, 0)' -write out.vdb

# Take the absolute value:
vdb_tool -read in.vdb -forAllValues 'abs(v)' -write out.vdb

# Smooth-step style mapping using pi:
vdb_tool -read in.vdb -forOnValues '0.5 - 0.5*cos(pi*v)' -write out.vdb

# Same kernel in RPN, for users who prefer the existing vdb_tool language:
vdb_tool -read in.vdb -forOnValues '0.5:0.5:$pi:$v:*:cos:*:-' -write out.vdb

# Combined with another option of the same action: the bare kernel still
# works because the greedy fallback only catches tokens whose `name=` prefix
# isn't a recognized option.
vdb_tool -read in.vdb -forOnValues 'max(v, 0)' keep=true -print
```

### Notes

- **Compile-time validation.** A typo such as `'sin(v'` (mismatched paren), `'1:2:3'` (leaves three values on the stack), or `'v + 1; v + 2'` (intermediate plain expression strands a value) is rejected before the grid is touched, with a clear error message identifying the offending token or statement.
- **Undefined-variable errors throw before any voxel is touched.** Compilation accepts arbitrary identifiers; the action then resolves every variable other than `v` against the Processor's string memory and throws with the offending name (`forValues: kernel references undefined variable "scale" …`) if the lookup fails. Set the value first via `-eval '{<value>:@<name>}'` or `-calc '<name>=<expr>'`.
- **Thread safety.** A compiled `kernel` is evaluated in parallel via TBB. The bytecode evaluator allocates its working stack &mdash; including the slot buffer used by multi-statement kernels &mdash; on the C stack at each call, so a single compiled kernel is safely shared across all worker threads.
- **Mixing syntaxes.** `=` and `;` require pure infix; combining them with `$` or `:` is rejected by the dispatcher.
- **Shell quoting.** Always single-quote the kernel value so the shell doesn't interpret `*`, `(`, `$`, `;`, etc.

# Building this tool

This tool is using CMake for build on Linux and Windows.
The only mandatory dependency is [OpenVDB](http://www.openvdb.org). Optional dependencies include NanoVDB, libpng, libjpeg, OpenEXR, Alembic, PDAL, and [OpenUSD](https://openusd.org). To enable them use the `-DOPENVDB_TOOL_USE_<name>=ON` flags (e.g. `-DOPENVDB_TOOL_USE_USD=ON` for USD support, or `-DOPENVDB_TOOL_USE_ALL=ON` to enable everything). See the CMakeLists.txt for details.

The included unit tests are using Gtest. Add `-DOPENVDB_BUILD_VDB_TOOL_UNITTESTS=ON` to the cmake command line to build it.

## Building OpenVDB

Follow the instructions at OpenVDB`s [github page](https://github.com/AcademySoftwareFoundation/openvdb#developer-quick-start)

Make sure to build with NanoVDB support, if you intend to use vdb_tool's NanoVDB features.

## Building vdb_tool on Linux

To generate the makefile, navigate to the cloned directory of vdb_tool, then follow these steps:
```bash
mkdir build
cd build
cmake -DOPENVDB_CMAKE_PATH=/usr/local/lib/cmake/OpenVDB -DUSE_ALL=ON -DOPENVDB_BUILD_VDB_TOOL_UNITTESTS=ON ..
```
Update the OpenVDB cmake path above as needed.

To build in debug mode, add `-DCMAKE_BUILD_TYPE=Debug` to the cmake command above. To build `vdb_tool` with NanoVDB support, pass in the `-DOPENVDB_BUILD_NANOVDB=ON` argument.

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
vcpkg install usd:x64-windows
```

### Building

```bash
mkdir build
cd build
cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_TOOLCHAIN_FILE=<path to vcpkg root>\scripts\buildsystems\vcpkg.cmake -A x64 -DOPENVDB_CMAKE_PATH=<OpenVDB install path>\lib\cmake\OpenVDB ..
cmake --build . --config Release --parallel 2
```
To build `vdb_tool` with NanoVDB support, pass in the `-DOPENVDB_BUILD_NANOVDB=ON` argument.


## Installing OpenUSD (optional)

USD read support (`.usd`, `.usda`, `.usdc`, `.usdz`) requires linking against [OpenUSD](https://openusd.org/release/index.html). After installing the library, enable it at configure time with `-DOPENVDB_TOOL_USE_USD=ON` (or `-DOPENVDB_TOOL_USE_ALL=ON`) and make sure CMake can locate the `pxrConfig.cmake` file shipped by OpenUSD &mdash; typically by adding the install root to `CMAKE_PREFIX_PATH` or by setting `-Dpxr_DIR=<install>/lib/cmake/pxr`. At runtime you may also need to point `LD_LIBRARY_PATH` (Linux), `DYLD_LIBRARY_PATH` (macOS), or `PATH` (Windows) at `<install>/lib` so the shared libraries are found.

### Linux / macOS (from source &mdash; recommended)

OpenUSD is not packaged in homebrew-core and is rarely packaged in distro repositories, so the most reliable install path on both Linux and macOS is to build it from source. The official build script bootstraps all third-party dependencies and produces a self-contained install (allow ~15&ndash;30 minutes on first build):

```bash
git clone https://github.com/PixarAnimationStudios/OpenUSD.git
python3 OpenUSD/build_scripts/build_usd.py \
    --no-imaging --no-usdview --no-alembic --no-draco --no-openimageio \
    --no-tutorials --no-examples ~/dev/src/openusd
cmake -DOPENVDB_TOOL_USE_USD=ON -Dpxr_DIR=$HOME/local/openusd/lib/cmake/pxr ..
```

For a slimmer build (skips Alembic, Draco, OpenImageIO, materials, imaging, etc., cutting build time substantially), pass `--no-imaging --no-alembic --no-draco --no-openimageio` to `build_usd.py`. vdb_tool only needs the core USD libraries (`usd`, `usdGeom`, `sdf`, `gf`, `vt`, `tf`).

### Windows (vcpkg)
```bash
vcpkg install usd:x64-windows
```
The vcpkg toolchain file already added to your CMake invocation will make OpenUSD discoverable; no extra `-Dpxr_DIR` is needed.

vcpkg also works on Linux and macOS if you prefer it over building from source &mdash; the package name is the same (`usd`), and you supply the appropriate triplet (e.g. `arm64-osx`, `x64-linux`).

### Verifying the install
```bash
vdb_tool -read scene.usda -print
```
should list the imported geometry on the stack. Per-prim world transforms are baked into the vertex positions; instancing, subdivision schemes, and animation are intentionally not handled by this minimal reader.


# Examples

## Getting help on all actions and their options
```
vdb_tool -help
```

## Getting help on specific actions and their options
```
vdb_tool -help read write
```

## Getting help on all actions
```
vdb_tool -eval help="*"
```

## Getting help on specific actions
```
vdb_tool -eval help=if,switch
```

## Hello-world example
Create a level set sphere and save it to a file
```
vdb_tool -sphere -write sphere.vdb
```
## Hello-world example with option
Same example but with options to save the file in half-float precision
```
vdb_tool -sphere -write bits=16 sphere.vdb
```

## Converting a mesh into a level set
Convert a polygon mesh file into a narrow-band level and save it to a file
```
vdb_tool -read mesh.obj -mesh2ls -write level_set.vdb
```

## Converting all quads in a mesh into triangles
Convert an obj file with n-gons into a ply file with only triangles
```
vdb_tool -read mesh.obj -quad2tri -write mesh.ply
```

## Generate image files from slices through a VDB grid
Generates a level set of a sphere and loops over multiple slices (in the yz plane) each generating an image files
```
vdb_tool -sphere -for x=0,1,0.01 -slice X='{$x}' -end
```

## Convert multiple images to a movie file
Reads multiple image files and converts them to an mpeg file
```
vdb_tool -img2mpeg input="slice_*.ppm" output=slices.mp4
```

## Read multiple specific files
Convert a polygon mesh file into a narrow-band level with a transform that matches a reference vdb
```
vdb_tool -read mesh.obj,reference.vdb -mesh2ls vdb=0 -write level_set.vdb
```

## Convert a sequence of files
Convert 5 polygon mesh files, "mesh_00{1,2,3,4,5}.obj", into separate narrow-band levels and save them to the files "level_set_0{1,2,3,4,5}.vdb". Note that the value of loop variables is accessible with a preceding "$" character and that the end of the for-loop (here 6) is exclusive.The instruction "pad0" adds zero-padding and takes two arguments, the string to pad and the desired length after padding.
```
vdb_tool -for n=1,6 -read mesh_'{$n:3:pad0}'.obj -mesh2ls -write level_set_'{$n:2:pad0}'.vdb -end
```

## Loop over specific files
Convert 3 polygon mesh files, "bunny.obj,teapot.ply,car.stl", into the Alembic files "mesh_0{1,2,3}.abc". Note that all loop variables have a matching counter defined with a preceding "#" character.
```
vdb_tool -each file=bunny.obj,teapot.ply,car.stl -read '{$file}' -write mesh_'{$#file:1:+:2:pad0}'.abc -end
```
 
## Define voxel size from a loop-variable
Generate 5 spheres with different voxel sizes and save them all into a single vdb file
```
vdb_tool -for v=0.01,0.06,0.01 -sphere voxel='{$v}' name=sphere_%v -end -write vdb="*" spheres.vdb
```

## Specify which grids to write into a single file
Generate 4 spheres named after their stack id, i.e. 3,2,1,0, and write only grid 0 and 2 to a file
```
vdb_tool -for i=0,5 -sphere name='{4:$i:-}' -end -write vdb=2,0 tmp.vdb
```

## Define options with simple math expression
Read both a vdb and mesh file and convert the mesh to a vdb with twice the voxel size of the input vdb.
```
vdb_tool -read bunny.vdb dragon.ply -mesh2ls voxel='{0:voxelSize:2:*}' -print
```

## Define options with complex math expressions
Generate spheres that are rotating along a parametric circle
```
vdb_tool -for degree=0,360,10 -eval '{$degree:d2r:@radian}' -sphere center='({$radian:cos},{$radian:sin},0)' name=sphere_'{$degree}' -end -write vdb="*" spheres.vdb
```

## Meshing of particles
Converts input points in the file points.[obj|ply|abc|pts] to a level set, perform level set actions, and written to it the file surface.vdb:
```
vdb_tool -read points.[obj|ply|abc|pts] -points2ls -dilate -gauss -erode -write surface.vdb
```

## Changing global default options
Example with many properties of scalar and vector fields
```
vdb_tool -default keep=true -sphere -curvature -grad -curl -div -length v=1 -debug
```

## If-statement to isolate level sets
Read multiple grids, and render only level set grids

```
vdb_tool -read boat_points.vdb -for v=0,'{gridCount}' -if '{$v:isLS}' -render vdb='{$v}' -end -end
```

## Use shell-script to define list of files
Find and render thumbnails of all level sets in an entire directory structure
```
vdb_tool -each file=`find ~/dev/data -name '*.vdb'` -read '{$file}' -for grid=0,'{gridCount}' -if '{$grid:isLS}' -render vdb='{$grid}' thumbnail_'{$grid:gridName}'.ppm image=256x256 keep=1 -end -end -clear -end
```
Most of the arguments should be self-explanatory, but at least two deserve an explanation: -render has the option keep=1 because otherwise rendered grids are removed from the stack which invalidates {gridCount}, and -clear is added to avoid accumulating all grids as multiple files are loaded.

For more examples [click here](examples/EXAMPLES.md)

---
# Pipelining:

vdb_tool supports unix-style pipelining, which is especially useful for interactive viewing. Specifically,
vdb_tool can read VDB grids from stdin or write VDB grid to stdout. Here are some examples:

## Redirection of stdout and stdin:
```
vdb_tool -sphere -o stdout.vdb > sphere.vdb
vdb_tool -i stdin.vdb -print < bunny.vdb
cat bunny.vdb | vdb_tool -i stdin.vdb -print
vdb_tool -sphere -o stdout.vdb | gzip > sphere.vdb.gz
gzip -dc sphere.vdb.gz | vdb_tool -i stdin.vdb -print
vdb_tool -sphere -o stdout.vdb | vdb_view
```

## Pipelining multiple instances of vdb_tool
```
vdb_tool -sphere -o stdout.vdb | vdb_tool -i stdin.vdb -dilate -o stdout.vdb > sphere.vdb
```
or with explicit semantics
```
vdb_tool -sphere -o stdout.vdb | vdb_tool -i stdin.vdb -dilate -o stdout.vdb > sphere.vdb
```
Note that the example above is slow due to serialization of the VDB grid.
```
vdb_tool -sphere -dilate -o stdout.vdb > sphere.vdb
```
or with explicit semantics
```
vdb_tool -sphere -dilate -o stdout.vdb > sphere.vdb
```

## Pipelining vdb_tool with vdb_view for interactive viewing
```
vdb_tool -sphere -dilate -o stdout.vdb | vdb_view
```

## View a sequence of scaling, rotating, and translated tetrahedra
```
vdb_tool -for t=0,6.28,0.2 -platonic f=4 -transform vdb=0 scale='{$t:sin:2:+}' rotate='(0,0,{$t})' translate='({$t:cos:5:*},{$t:sin:5:*},0)' -end -o stdout.vdb | vdb_view
```

## View a sequence of spheres deformed in an analytical fluid field
```
vdb_tool -sphere d=80 r=0.15 c=0.35,0.35,0.35 -for i=1,20 -enright dt=0.05 k=1 -end -o stdout.vdb | vdb_view
```

## View a sequence of octahedrons deformed in an analytical fluid field
```
vdb_tool -platonic d=128 f=8 s=0.15 c=0.35,0.35,0.35 -for i=1,20 -enright dt=0.05 k=1 -end -o stdout.vdb | vdb_view
```

## Production example of meshing of fluid particles
Generate adaptive meshes from a sequence of points files, points_0[200,299].vdb, and use mesh_mask.obj to clip off boundaries. Points are first rasterized as level set spheres, then dilates, filtered and eroded and finally meshed using the mask.
```
vdb_tool -read mesh_mask.obj -mesh2ls voxel=0.1 width=3 -for n=200,300,1 -read points_{$n:4:pad0}.vdb -vdb2points -points2ls voxel=0.035 radius=2.142 width=3 -dilate radius=2.5 space=5 time=1 -gauss iter=2 space=5 time=1 size=1 -erode radius=2.5 space=5 time=1 -ls2mesh vdb=0 mask=1 adapt=0.005 -write mesh_{$n:4:pad0}.abc -end
```

## Example of a configuration file performing Particle-to-Mesh generation
```
vdb_tool 10.8.0

# 1. LOAD A MASK (Optional)
# Used to clip the fluid so it doesn't leak out of the container
read collision_geo.obj 
mesh2ls voxel=0.1 width=3

# 2. LOOP THROUGH PARTICLE SEQUENCE
# Processing frames 200 to 300
for n=200,300,1
    
    # Read the particle VDB for the current frame
    read points_{$n:4:pad0}.vdb
    
    # Convert particles to a Level Set
    # 'radius' is the particle size; 'voxel' is the grid resolution
    points2ls voxel=0.035 radius=2.142 width=3
    
    # SURFACE REFINEMENT
    dilate radius=2.5         # Expand to merge gaps
    gauss iter=2              # Smooth out the "blobby" look
    erode radius=2.5          # Shrink back to original scale
    
    # 3. MESHING & CLIPPING
    # Convert to adaptive mesh, clipped by our collision mask (vdb=1)
    ls2mesh vdb=0 mask=1 adapt=0.005
    
    # 4. EXPORT
    write mesh_{$n:4:pad0}.abc
    
    # Clear the stack for the next frame to prevent memory bloat
    clear
end
```

## Production example with complex math using RPN syntax
Union 200 level set spheres scattered in a spiral pattern and ray-trace them into an image
```
vdb_tool -for n=0,200,1 -eval '{$n:137.5:*:@deg}' -eval '{$deg:d2r:@radian}' -eval '{$radian:cos:@x}' -eval '{$radian:sin:@y}' -eval '{$n:sqrt:@r}' -eval '{$r:5:+:@r_sum}' -eval '{$r_sum:0.25:pow:@pow_r}' -sphere voxel=0.1 radius='{$pow_r:0.5:*}' center='({$r:$x:*},{$r:$y:*},0)' -if '{$n:0:>}' -union -end -end -render spiral.ppm image=1024x1024 translate='(0,0,40)'
```

## Production example with complex math using InFix syntax
Union 200 level set spheres scattered in a spiral pattern and ray-trace them into an image
```
vdb_tool -for n=0,200,1  -calc 'radian=137.5*n*pi/180; r=sqrt(n); x=r*cos(radian); y=r*sin(radian); pow_r=0.5*(5+r)^0.25' -sphere voxel=0.1 radius='{$pow_r}' center='({$x},{$y},0)' -if '{$n:0:>}' -union -end -end -render spiral.ppm image=1024x1024 translate='(0,0,40)'
```

or as a config file:

## Production example with complex math in a configuration file using RPN syntax
```
vdb_tool 10.8.0
for n=0,200,1
    eval {$n:137.5:*:@deg}  # deg = 137.5 * n
    eval {$deg:d2r:@radian} # radian = d2r(deg)
    eval {$radian:cos:@x}   # x = cos(radian)
    eval {$radian:sin:@y}   # y = sin(radian)
    eval {$n:sqrt:@r}       # r = sqrt(n)
    eval {$r:5:+:@r_sum}    # r_sum = 5 + r
    eval {$r_sum:0.25:pow:@pow_r} # pow_r = pow(r_sum, 0.25)
    sphere voxel=0.1 radius={$pow_r:0.5:*} center=({$r:$x:*},{$r:$y:*},0) # radius=0.5*pow_r center=(r*x, r*x,0)
    if {$n:0:>} # if n > 0
        union
    end
end
render spiral.ppm image=1024x1024 translate=(0,0,40)
```

## Production example with complex math in a configuration file using InFix syntax
```
vdb_tool 10.8.0
for n=0,200,1
    calc a=137.5*n*pi/180;r=sqrt(n);x=r*cos(a);y=r*sin(a);r=0.5*(5+r)^0.25
    sphere voxel=0.1 radius={$r} center=({$x},{$y},0)
    if {$n:0:>} # if n > 0
        union # CSG union of spheres
    end
end
render spiral.ppm image=1024x1024 translate=(0,0,40)
```
---

