# Documentation
All the configuration files in this directory can be executed as:
```
./vdb_tool -read points_to_mesh.txt
```

# Examples
Print documentation to the terminal (and terminate):
```
./vdb_tool --help
```

Convert one polygon mesh file into another polygon mesh file:
```
./vdb_tool --read mesh.[obj|ply|stl] --write mesh.[obj|ply|stl]
```

Convert multiple OBJ files (mesh_{00-09}.obj) into multiple PLY files (mesh_{00-09}.ply):
```
./vdb_tool --for f=0,10,1 --read mesh_{$f:2:pad0}.obj --write mesh_{$f:2:pad0}.ply -end
```

Convert a polygon mesh file into a narrow-band level set and save it to a file:
```
./vdb_tool --read mesh.obj -mesh2ls --write mesh.vdb
```

Convert a polygon mesh file into a narrow-band level set with half-width 2 voxels and a maximum dimension of 512 voxels, then save it to a file:
```
./vdb_tool --read mesh.obj --mesh2ls dim=512 width=2 --write mesh.vdb
```

Convert input points in points.[ply|obj|stl|pts] into a level set, apply a sequence of level-set operations, and write the result to surface.vdb:
```
./vdb_tool --read points.[obj|ply|stl|pts] --points2ls --dilate --gauss --erode --write surface.vdb
```

Convert input points in points.vdb into a level set, apply a sequence of level-set operations, and write the result to surface.vdb:
```
./vdb_tool -read points.vdb -vdb2points -points2ls -dilate -gauss -erode -write surface.vdb
```

Convert input points in points.[ply|obj|stl|pts] into a level set, apply level-set operations, extract a polygon mesh, and save the mesh to surface.[ply|obj|stl|abc|pts]:
```
./vdb_tool -read points.[ply|obj|stl|pts] -points2ls -dilate -gauss -erode -ls2mesh -write surface.[ply|obj|stl|abc|pts]
```

This example is more verbose and demonstrates how parameters of the level-set operations are specified. Note that the dilation operation uses 5th-order (WENO) spatial discretization and 2nd-order (TVD-RK) temporal discretization:
```
./vdb_tool -read points.[ply|obj|stl|pts] -points2ls dim=256 voxel=0.1 radius=0.2 width=3 -dilate radius=2 space=5 time=2 -gauss iter=1 width=1 -erode radius=2 -ls2mesh adapt=0.25 -write output.[ply|obj|stl|abc|pts]
```

This example shows how to save a sequence of operations as a config file:
```
./vdb_tool -read points.[ply|obj|stl|pts] -points2ls dim=256 voxel=0.1 radius=0.2 width=3 -dilate radius=2 space=5 time=2 -gauss iter=1 width=1 -erode radius=2 -ls2mesh adapt=0.25 -write output.[ply|obj|stl|abc|pts] -write conf.txt
```

This example shows how to read a config file and execute it. The file can of course be modified and re-run:
```
./vdb_tool -read conf.txt
```
