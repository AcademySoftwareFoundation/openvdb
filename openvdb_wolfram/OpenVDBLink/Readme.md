![OpenVDBLink](https://www.openvdb.org/images/openvdblink_logo.png)

# OpenVDBLink
OpenVDBLink provides a Mathematica interface to [OpenVDB](https://www.openvdb.org), an efficient sparse voxel data structure ubiquitous in the VFX industry. This link ports over access to various grid containers including level sets, fog volumes, vector grids, integer grids, Boolean grids, and mask grids. Construction, modification, combinations, visualisations, queries, import, export, etc. can be achieved over grids too. Any Mathematica 3D region that's ConstantRegionQ and BoundedRegionQ can be represented as a level set grid, providing a more seamless integration with OpenVDB.

OpenVDB is an Academy Award-winning open-source C++ library comprising a novel hierarchical data structure and a suite of tools for the efficient storage and manipulation of sparse volumetric data discretized on three-dimensional grids. It was developed by [DreamWorks Animation](http://www.dreamworksanimation.com/) for use in volumetric applications typically encountered in feature film production and is now maintained by the [Academy Software Foundation (ASWF)](https://www.aswf.io/).

### Dependencies and requirements

* Mathematica version 11.0 or higher
* OpenVDB itself and any downstream dependencies it has
* C++17 or higher

### Documentation

* [Web version](https://www.openvdb.org/documentation/wolfram)
* [Notebook version](https://www.openvdb.org/download/files/OpenVDBLink.nb.zip)
* Access documentation in a Wolfram notebook with `OpenVDBDocumentation[]`.

### Loading

In a Mathematica kernel, find the `openvdb_wolfram` directory and add this to `$Path`:

```
vdbpath = FileNameJoin[{"path", "to", "openvdb", "openvdb_wolfram"}];

$Path = DeleteDuplicates@Join[$Path, {vdbpath}];
```

Load the package:

```
<< OpenVDBLink`
```

### Compilation

Only during the first time loading the package, it will automatically compile the necessary binary:

```
<<OpenVDBLink`
```

After loading this will print in the notebook / console:

```
Current directory is: /path/to/openvdb/openvdb_wolfram/OpenVDBLink/Source/ExplicitGrids
Unloading library OpenVDBLink ...
Generating library code ...
Compiling library code ...
```

If compilation fails, you may need to add any include/link paths that are not currently in your path environment. These can be added in `BuildSettings.m`.

* For example on MacOS one may need to explictly add include and link paths like `-I/usr/local/include` in the `"CompileOptions"` field in the `"MacOSX"` area.
* On Windows `$vcpkgDir` might need to be changed if `vcpkg` is not in `$HomeDirectory`.

Once these are set, reload and then recompile. Below the input `True` is optional and tells the system to print the compile command as well as any warnings or errors during compile:

```
<<OpenVDBLink`
OpenVDBLink`Developer`Recompile[True]
```

OpenVDBLink has been compiled and tested on MacOS 10.14 and higher, Windows 10, and Ubuntu 20.04 and higher.

###  Basic example

Once loaded verify the link is working by creating a level set ball and retrieving its mesh:

```
grid = OpenVDBLevelSet[Ball[], 0.1, 3.0];
mesh = OpenVDBMesh[grid];

MeshRegionQ[mesh]
(* True *)

MeshCellCount[mesh]
(* {1832, 5490, 3660} *)
```
