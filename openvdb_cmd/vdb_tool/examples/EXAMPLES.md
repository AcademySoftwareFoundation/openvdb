# Documentation
All the configuration files in this directory can be executed as
```
./vdb_tool -read points_to_mesh.txt
```

# Examples
Print documentation to the terminal (and terminate)
```
./vdb_tool --help
```

Convert a polygon mesh file into to another polygon mesh file
```
./vdb_tool --read mesh.[obj|ply|stl] --write mesh.[obj|ply|stl]
```

Convert multiple obj files (mesh_{00-09}.obj) into multiple ply files (mesh_{00-09}.ply)
```
./vdb_tool --for f=0,10,1 --read mesh_{$f:2:pad0}.obj --write mesh_{$f:2:pad0}.ply -end
```

Convert a polygon mesh file into a narrow-band level and save it to a file
```
./vdb_tool --read mesh.obj -mesh2ls --write mesh.vdb
```

Convert a polygon mesh file into a narrow-band level of width 2 with maximum voxels dimension of 512 and save it to a file
```
./vdb_tool --read mesh.obj --mesh2ls dim=512 width=2 --write mesh.vdb
```

Converts input points in the file points.[ply|obj|stl|pts] to a level set, perform level set operations, and written to it the file surface.vdb
```
./vdb_tool --read points.[obj|ply|stl|pts] --points2ls --dilate --gauss --erode --write surface.vdb
```

Converts input points in the file points.vdb to a level set, perform level set operations, and written to it the file surface.vdb
```
./vdb_tool -read points.vdb -vdb2points -points2ls -dilate -gauss -erode -write surface.vdb
```

Converts input points in the file points.[ply|obj|stl|pts] to a level set, perform level set operations, extract a polygon mesh, and save the mesh to the file surface.[ply|obj|stl|abc|pts]
```
./vdb_tool -read points.[ply|obj|stl|pts] -points2ls -dilate -gauss -erode -ls2mesh -write surface.[ply|obj|stl|abc|pts]
```

This examples is more verbose and demonstrates how parameters of the level set operations are specified. Note that the dilation operation is using 5'th order (WENO) spatial discretization and 2'nd order (TVD-RK) time discretization.
```
./vdb_tool -read points.[ply|obj|stl|pts] -points2ls dim=256 voxel=0.1 radius=0.2 width=3 -dilate radius=2 space=5 time=2 -gauss iter=1 width=1 -erode radius=2 -ls2mesh adapt=0.25 -write output.[ply|obj|stl|abc|pts]
```

This examples show how to save operations to a config file
```
./vdb_tool -read points.[ply|obj|stl|pts] -points2ls dim=256 voxel=0.1 radius=0.2 width=3 -dilate radius=2 space=5 time=2 -gauss iter=1 width=1 -erode radius=2 -ls2mesh adapt=0.25 -write output.[ply|obj|stl|abc|pts] -write conf.txt
```

This examples shows how to read operations to a config file and perform them. The file can of course be modified and re-run!
```
./vdb_tool -read conf.txt
```
