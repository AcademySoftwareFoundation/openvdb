# vdb_tool
The vdb_tool is a versatile command-line utility that chains together high-level operations from the OpenVDB library. It can convert polygon meshes and particles into level sets, perform complex volumetric transformations, and generate adaptive meshes or ray-traced images. Results can be exported as particles, meshes, or VDB files, or streamed directly to STDOUT for seamless pipelining with other renderers. We denote the operations **actions**, and their arguments **options**. Any sequence of **actions** and their **options** can be exported and imported to configuration files, which allows convenient reuse. This command-line tool also supports a string-evaluation language that can be used to define procedural expressions for options of the actions. Currently the following list of actions are supported:

<!-- BEGIN AUTO-GENERATED ACTION TABLE — do not edit by hand. Regenerate with:
       vdb_tool -help format=md
     and replace everything between the BEGIN/END markers with the output. -->
| Action | Description |
|---|---|
| **calc** | calculate string expression |
| **clear** | Deletes geometry, VDB grids and local variables |
| **clip** | Clip a VDB grid against another grid, a bbox or frustum |
| **close** | morphological closing, i.e. dilation followed by erosion, of level set surface by a fixed radius |
| **config** | Import and process one or more configuration files |
| **cpt** | generate a vector grid with the closest-point-transform to a level set surface |
| **curl** | generate a vector grid with the curl of another vector grid |
| **curvature** | generate scalar grid with the mean curvature of a level set surface |
| **debug** | print debugging information to the terminal |
| **default** | define default values to be used by subsequent actions |
| **difference** | CSG difference of two level sets surfaces |
| **dilate** | dilate level set surface by a fixed radius |
| **div** | generate a scalar grid with the divergence of a vector grid |
| **each** | start of each-loop over a user-defined loop variable and list of values. |
| **end** | marks the end scope of "-for,-each,and -if" control actions |
| **enright** | Performs Enright advection benchmark test on a level set |
| **erode** | erode level set surface by a fixed radius |
| **errorOnWarning** | stop on warnings, i.e. treat warnings as errors |
| **eval** | evaluate string expression |
| **examples** | print examples to the terminal and terminate |
| **expand** | expand narrow band of level set |
| **files** | start of files-loop in a directory. |
| **flood** | signed-flood filling of a level set VDB |
| **fog2mesh** | Convert a fog volume to an adaptive polygon mesh |
| **for** | start of for-loop over a user-defined loop variable and range. |
| **forAllValues** | Applied a simple computational kernel to ALL values in a grid. |
| **forOffValues** | Applied a simple computational kernel to OFF values in a grid. |
| **forOnValues** | Applied a simple computational kernel to ON values in a grid. |
| **gauss** | gaussian convolution of a level set surface |
| **grad** | generate a vector grid with the gradient of a scalar grid |
| **help** | Print documentation for one, multiple or all available actions |
| **if** | start of if-scope. If the value of its option, named test, evaluates to false the entire scope is skipped |
| **intersection** | CSG intersection of two level sets surfaces |
| **iso2ls** | Convert an iso-surface of a scalar field into a level set (i.e. SDF) |
| **length** | generate a scalar grid with the magnitude of a vector grid |
| **log** | enable logging to file |
| **ls2fog** | Convert a level set VDB into a VDB with a fog volume, i.e. normalized density. |
| **ls2mesh** | Convert a level set to an adaptive polygon mesh |
| **max** | Given grids A and B, compute max(a, b) per voxel |
| **mean** | mean value filtering of a level set surface |
| **median** | median value filtering of a level set surface |
| **mesh2ls** | Convert a watertight polygon surface into a narrow-band level set, i.e. a narrow-band signed distance to a polygon mesh |
| **min** | Given grids A and B, compute min(a, b) per voxel |
| **movie** | Convert image and movie files to mpeg or animated gif files |
| **multires** | construct a LoD sequences of VDB trees with powers of two refinements |
| **open** | morphological opening, i.e. erosion followed by dilation, of a level set surface by a fixed radius |
| **platonic** | Create a level set shape with the specified number of polygon faces |
| **points2ls** | Convert geometry points into a narrow-band level set |
| **points2vdb** | Encode geometry points into a VDB grid |
| **print** | prints information to the terminal about the current stack of VDB grids and Geometry |
| **prune** | prune away inactive values in a VDB grid |
| **quad2tri** | Convert all quads in mesh to triangles, assuming they are both planar and convex |
| **quiet** | disable printing to the terminal |
| **read** | Read one or more geometry or VDB files from disk or STDIN. |
| **render** | ray-tracing of level set surfaces and volume rendering of fog volumes |
| **resample** | resample one VDB grid into another VDB grid or a transformation of the input grid |
| **scatter** | Scatter point into the active values of an input VDB grid |
| **sdf2udf** | Converts a signed distance field into an unsigned distance field, i.e. performs the Abs of all values and changes GridClass to UNKNOWN. |
| **segment** | segment an input VDB into a list if topologically disconnected VDB grids |
| **slice** | Generate images of slices of a VDB grid |
| **soup2ls** | Convert a polygon soup into a narrow-band level set, i.e. a narrow-band signed distance to a polygon mesh |
| **soup2offset** | Convert a polygon soup into an offset narrow-band level set, i.e. a narrow-band signed distance to a polygon mesh |
| **soup2udf** | Convert a polygon soup into a to a unsigned distance field with an symmetrical narrow band |
| **sphere** | Create a level set sphere, i.e. a narrow-band signed distance to a sphere |
| **sum** | Given grids A and B, compute sum(a, b) per voxel |
| **transform** | apply affine transformations (uniform scale -> rotation -> translation) to a VDB grids and geometry |
| **union** | CSG union of two level sets surfaces |
| **vdb2points** | Extract points encoded in a VDB to points in a geometry format |
| **verbose** | print timing information to the terminal |
| **version** | write timing information to the terminal |
| **vol2mesh** | Convert a scalar volume to an adaptive polygon mesh |
| **write** | Write list of geometry, VDB or config files to disk or STDOUT |
<!-- END AUTO-GENERATED ACTION TABLE -->

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
| gltf, glb | optional read | glTF 2.0 and binary glTF mesh files via tinygltf (POSITION + indices, TRIANGLES mode) |
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

# Control flow: 'if(cond, then, else)' is a 3-arg expression. Both branches
# are evaluated eagerly; the result is selected.
vdb_tool -calc 'if(2>1, 10, 20)'                # prints 10

# Combine if() with multi-statement to compute |x|, set in memory:
vdb_tool -eval '{-5:@x}' -calc 'abs_x = if(x>=0, x, -x)' -eval str='|x|={$abs_x}'
# prints: |x|=5.000000

# Signed square-root in a single kernel (no memory needed; x is a local slot).
# Final statement is a plain expression, so the result is echoed.
vdb_tool -calc 'x=-9; if(x>=0, sqrt(x), -sqrt(-x))'   # prints -3

# Variadic switch(selector, k1, v1, ..., kN, vN, default): pick the value
# for the first ki that equals the selector, else default.
vdb_tool -eval '{2:@mode}' -calc 'switch(mode, 0, 100, 1, 200, 2, 300, -1)'
# prints: 300

# Classify each loop iteration with nested if():
vdb_tool -for n=-2,3,1 -calc 'label = if(n<0, -1, if(n==0, 0, 1))' \
                       -eval str='{$n}: label={$label}' -end
# prints:
#   Processing: n = -2, ...      -2: label=-1.000000
#   Processing: n = -1, ...      -1: label=-1.000000
#   Processing: n =  0, ...       0: label=0.000000
#   Processing: n =  1, ...       1: label=1.000000
#   Processing: n =  2, ...       2: label=1.000000
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

#### Stencil kernels: voxel-neighbor access

When the kernel calls the voxel-variable as a function with three integer-literal offsets, e.g. `v(1, 0, 0)`, the call expands to a relative neighbor read at index-space coordinate `(i+dx, j+dy, k+dz)` where `(i, j, k)` is the current voxel. `v(0, 0, 0)` is equivalent to bare `v` (the center). Neighbor reads go through a per-thread `ConstAccessor` for cache locality, and the grid is internally deep-copied before iteration so reads come from a stable snapshot of the original state &mdash; parallel writes to the iterator's grid don't race with neighbor reads.

```bash
# Finite-difference x-derivative.
vdb_tool -read in.vdb -forOnValues 'v(1,0,0) - v(-1,0,0)' -write out.vdb

# 6-point discrete Laplacian.
vdb_tool -read in.vdb -forOnValues 'v(1,0,0)+v(-1,0,0)+v(0,1,0)+v(0,-1,0)+v(0,0,1)+v(0,0,-1) - 6*v' -write out.vdb

# Jacobi smoothing of a cube: average each voxel with its 6 face neighbors.
vdb_tool -platonic faces=6 -forOnValues '(v + v(1,0,0)+v(-1,0,0)+v(0,1,0)+v(0,-1,0)+v(0,0,1)+v(0,0,-1)) / 7' -write smooth.vdb
```

Constraints and caveats:

- Offsets must be **integer literals** (or unary-negated integer literals); a runtime expression like `v(dx, 0, 0)` where `dx` is a variable is rejected at compile time. This keeps each neighbor reference resolvable to a single PushVar opcode with no per-voxel arithmetic.
- The renamed voxel variable participates: `-forOnValues '...' use=x` makes `x(1, 0, 0)` the +x neighbor.
- Reads at the boundary of the active region return the grid's background value. There's no `boundary=clamp|...` option (yet); structure the kernel to tolerate it or pre-pad the active region.
- The implicit deep-copy doubles memory for the duration of the action when any non-zero neighbor offset is used.

#### Multi-grid kernels

`use=` and `vdb=` accept comma-separated lists so the kernel can read from more than one grid in a single pass. The first entry is the **output** grid (iterated and written); the rest are **read-only inputs**. Each name becomes a kernel-side handle for that grid:

```bash
# Pointwise difference: write A - B into A.
vdb_tool -read a.vdb b.vdb -forOnValues 'a - b' use=a,b vdb=0,1 -write diff.vdb

# Cross-grid finite difference: write a result that depends on a neighbor of A
# and a neighbor of B.
vdb_tool -read a.vdb b.vdb -forOnValues 'a(1,2,3) + b(0,1,2)' use=a,b vdb=0,1 -write out.vdb

# Three-grid average.
vdb_tool -read a.vdb b.vdb c.vdb \
         -forOnValues '(a + b + c) / 3' use=a,b,c vdb=0,1,2 -write avg.vdb
```

Rules:

- `use=` and `vdb=` must have the same length (errors out otherwise).
- Iteration topology is the **output grid's** active voxels; reads from input grids at those coords return the input grid's stored value (or its background if inactive there).
- Only the output grid is deep-copied (and only if the kernel reads non-zero offsets from it). Inputs are read-only, no snapshot needed.
- Each input grid gets its own per-thread `ConstAccessor`, cached across sequential voxel reads.
- Duplicate names in `use=` are rejected — kernels must be unambiguous.

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
| unary `-` / `+` / `!`  | 8 | right (unary `+` is a no-op, `!` is logical NOT) |
| `^` (power)            | 7 | right |
| `*` `/` `%`            | 6 | left  |
| `+` `-` (binary)       | 5 | left  |
| `<` `>` `<=` `>=`      | 4 | left (return 1.0/0.0) |
| `==` `!=`              | 3 | left (return 1.0/0.0) |
| `&&`                   | 2 | left (return 1.0/0.0) |
| `||`                   | 1 | left (return 1.0/0.0) |

In RPN, the punctuation operators have word-form aliases: `mod`, `lt`/`gt`/`le`/`ge`/`eq`/`ne`, `and`/`or`/`not`.

### Functions

| Unary   | `neg` `abs` `inv` `sqrt` `sin` `cos` `tan` `asin` `acos` `atan` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` `exp` `ln` `log` `floor` `ceil` `pow2` `pow3` `sign` `round` `trunc` `not` |
|---------|---|
| **Binary**  | `pow(a, b)` (also `a^b`), `min(a, b)`, `max(a, b)`, `atan2(y, x)`, `hypot(a, b)`, `step(edge, x)`, `mod(a, b)` / `fmod(a, b)` |
| **Ternary** | `clamp(x, lo, hi)`, `lerp(a, b, t)` (alias `mix`), `smoothstep(e0, e1, x)`, `if(cond, then, else)` / `select(cond, then, else)` |
| **Variadic** | `switch(selector, k1, v1, ..., kN, vN, default)` |

`step(edge, x)` follows GLSL conventions: returns 1 when `x >= edge`, else 0. `lerp(a, b, t)` is `a*(1-t) + b*t`. `smoothstep` clamps to `[0,1]` then applies the Hermite polynomial `t*t*(3-2*t)`. `if`/`select` evaluates both branches eagerly &mdash; they're plain ternary, not short-circuit. `switch(s, k1, v1, ..., kN, vN, d)` returns `vi` for the first `ki == s` (exact equality), else `d`; like `if`, all case bodies are eagerly evaluated. The arg count must be even and at least 4.

### Constants

`pi`, `tau` (=2π), `e`, `phi` (golden ratio), `inf`, and `nan` are recognized as named literals in all three syntaxes. None of them can be the target of an assignment.

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

### Advanced features

The Calculator that drives `-calc` and the per-voxel kernels also supports several optimizations and language features beyond the basics above.

#### Lazy `if(...)` (short-circuit semantics)

`if(cond, then, else)` evaluates **only the taken branch**. The other branch is skipped at runtime, so kernels can guard against divisions by zero, square-roots of negatives, etc. without first evaluating the problematic expression:

```bash
vdb_tool -calc 'if(1, 42, 1/0)'                              # prints 42 — 1/0 is never evaluated
vdb_tool -calc 'x=-9; if(x>=0, sqrt(x), -sqrt(-x))'          # prints -3  — sqrt(-9) is never evaluated
vdb_tool -calc 'def safe_inv(x) = if(x==0, 0, 1/x); \
                 safe_inv(0) + safe_inv(2)'                  # prints 0.5 — 1/0 never runs
```

Nested `if()` calls and `if()` inside user-defined function bodies also short-circuit correctly. `switch(...)` is currently eager (all case values are computed); use nested `if()` if you need lazy semantics for many cases.

#### User-defined functions (`def`)

A `def name(params) = body` statement registers a function. Subsequent calls to it inline the body's bytecode with the arguments bound to the parameters &mdash; no runtime call overhead. Functions can call other previously-defined functions, but **recursion is not supported** (a function referencing itself fails with "unknown function"); free variables in the body (anything not in the parameter list) are also rejected at compile time.

```bash
# Single-parameter function
vdb_tool -calc 'def sq(x) = x*x; sq(3) + sq(4)'              # prints 25

# Two parameters
vdb_tool -calc 'def hyp(a, b) = sqrt(a*a + b*b); hyp(3, 4)'  # prints 5

# Composition: `cu` uses `sq` defined earlier
vdb_tool -calc 'def sq(x) = x*x; def cu(x) = x*sq(x); cu(3)' # prints 27

# Use a `def` inside a voxel kernel for readability:
vdb_tool -sphere -forOnValues 'def step01(t) = clamp(t, 0, 1); step01(v + 0.5)' -print
```

The `def` itself emits no bytecode; only call sites do. Therefore a `def` statement cannot be the final statement of a program (it has no return value).

#### Constant folding

Literal-only subexpressions are folded at compile time:

```bash
vdb_tool -calc '1 + 2 + 3'                                   # bytecode: PushLit 6 (single instruction)
vdb_tool -calc '2*pi + sqrt(16) + abs(-3)'                   # collapses to one literal
```

The fold pass runs after the parser and before lazy `if` rewriting; combined with the parser, a kernel like `kernel='sin(pi/4)*2 + a*v'` compiles down to two instructions: one `PushLit` (the precomputed `sin(pi/4)*2`) plus the per-voxel `a*v` chain.

#### Diagnostics: column-aware error messages

Calculator's tokenizer points at exactly where it stopped:

```
Calculator: unexpected character '@' in expression
  1 + @ + 2
      ^  (column 5)
```

#### Batched evaluation in C++

`Calculator::eval_n(in, out, n, varName="x")` applies a single-variable kernel across an array, suitable for vector-style transforms in tests or programmatic call-sites.

#### Bytecode inspection

`Calculator::disassemble()` returns a multi-line, human-readable dump of the compiled bytecode &mdash; useful when debugging kernel behavior or comparing the effect of the optimization passes.

# Building this tool

This tool is using CMake for build on Linux and Windows.
The only mandatory dependency is [OpenVDB](http://www.openvdb.org). Optional dependencies include NanoVDB, libpng, libjpeg, OpenEXR, Alembic, PDAL, [OpenUSD](https://openusd.org), and [tinygltf](https://github.com/syoyo/tinygltf) (the latter is fetched at configure time, no system install required). To enable them use the `-DOPENVDB_TOOL_USE_<name>=ON` flags (e.g. `-DOPENVDB_TOOL_USE_USD=ON` for USD support, `-DOPENVDB_TOOL_USE_GLTF=ON` for glTF read support, or `-DOPENVDB_TOOL_USE_ALL=ON` to enable everything). See the CMakeLists.txt for details.

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


## Enabling glTF support (optional)

glTF read support (`.gltf`, `.glb`) is provided by [tinygltf](https://github.com/syoyo/tinygltf), a single-header BSD-3 reader. Unlike OpenUSD, no system install is required &mdash; CMake's `FetchContent` pulls a pinned release at configure time. Simply enable the option:
```bash
cmake -DOPENVDB_TOOL_USE_GLTF=ON ..
```
(or pass `-DOPENVDB_TOOL_USE_ALL=ON` to enable every optional component, including glTF). The first configure downloads tinygltf into `<build>/_deps/tinygltf-src/`; subsequent configures reuse the cached checkout.

vdb_tool uses tinygltf in header-only mode (`TINYGLTF_HEADER_ONLY`), with image decoding disabled (`TINYGLTF_NO_STB_IMAGE` / `TINYGLTF_NO_STB_IMAGE_WRITE`) since vdb_tool only consumes mesh geometry &mdash; textures referenced by the glTF are silently skipped.

### What's imported
- Vertex positions (POSITION attribute) and indices (UBYTE / USHORT / UINT) from every mesh primitive.
- Both indexed and non-indexed primitives.
- Only TRIANGLES mode; POINTS / LINES / STRIPS / FANS are skipped with a warning when `-verbose` is on.

### What's not imported
- Node-graph transforms &mdash; meshes load in their local space.
- Materials, normals, UVs, vertex colors, animation, and skinning.

### Verifying the install
```bash
vdb_tool -read model.glb -print
```
should list the imported geometry on the stack.


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

## Production example with complex math using infix syntax
Union 200 level set spheres scattered in a spiral pattern and ray-trace them into an image
```
vdb_tool -for n=0,200,1  -calc 'radian=137.5*n*pi/180; r=sqrt(n); x=r*cos(radian); y=r*sin(radian); pow_r=0.5*(5+r)^0.25' -sphere voxel=0.1 radius='{$pow_r}' center='({$x},{$y},0)' -if 'n > 0' -union -end -end -render spiral.ppm image=1024x1024 translate='(0,0,40)'
```

or as a config file:

## Production example with complex math in a configuration file using infix syntax

Same 200-sphere phyllotaxis spiral as the RPN config above, written with the more readable infix `-calc` syntax (one multi-statement kernel replaces seven sequential `-eval` calls). Notice how `x` and `y` already include the radial factor (`x = r*cos(a)`), so the sphere's `center=` doesn't need to multiply by `r` again as the RPN version does — the two examples are mathematically equivalent.
```
vdb_tool 10.8.0
for n=0,200,1
    # Multi-statement calc kernel. Reads `n` from for-loop memory; writes
    # back a, r, x, y. The final assignment `r = 0.5*(5+r)^0.25` reuses
    # the `r` slot: its right-hand side reads the OLD value (sqrt(n)),
    # then overwrites the slot with the sphere radius for use below.
    calc a = 137.5*n*pi/180; r=sqrt(n); x = r*cos(a); y = r*sin(a); r = 0.5*(5+r)^0.25
    sphere voxel=0.1 radius={$r} center=({$x},{$y},0)
    if {$n:0:>}  # skip n==0: there's nothing to union with on the first iteration
        union    # CSG union of this sphere into the accumulator
    end
end
render spiral.ppm image=1024x1024 translate=(0,0,40)
```

## Production example with complex math in a configuration file using RPN syntax
This example, based on -eval vs -calc, is only included for completion! The example above using -calc is much more user-friendly.
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
    if n > 0
        union
    end
end
render spiral.ppm image=1024x1024 translate=(0,0,40)
```
---

