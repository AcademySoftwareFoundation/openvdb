# actions (-) and options (= except files)
```
vdb_tool -sphere
vdb_tool -sphere -print
vdb_tool -sphere dim=128 -print
vdb_tool -sphere d=128 -print
vdb_tool -sphere -write sphere.vdb
vdb_tool -sphere -render tmp.jpg
vdb_tool -sphere -render shader=normal tmp.jpg
```

# internal lists of VDBs and geometry and age
```
vdb_tool -sphere -platonic -print
vdb_tool -sphere -platonic -read bunny.vdb bunny.ply -print
vdb_tool -read teapot.ply -mesh2ls -render tmp.jpg
vdb_tool -read teapot.ply -points2ls v=0.1 -dilate -render tmp.jpg
vdb_tool -read teapot.ply -points2ls v=0.1 -dilate -gauss -erode -render tmp.jpg
vdb_tool -read teapot.ply -points2ls v=0.1 -dilate -gauss -erode -render tmp.jpg -write config.txt
vdb_tool -config config.txt
```

# keep
```
vdb_tool -sphere -ls2mesh -print
vdb_tool -sphere -ls2mesh keep=true -print
vdb_tool -sphere -ls2mesh k=1 -write sphere.vdb sphere.ply
```

# pipe
```
vdb_tool -sphere -o stdout.vdb > sphere.vdb
vdb_tool -i stdin.vdb -print < bunny.vdb
cat bunny.vdb | vdb_tool -i stdin.vdb -print
vdb_tool -sphere -o stdout.vdb | gzip > sphere.vdb.gz
gzip -dc sphere.vdb.gz | vdb_tool -i stdin.vdb -print
vdb_tool -sphere -o stdout.vdb | vdb_tool -i stdin.vdb -dilate > sphere.vdb
vdb_tool -sphere -dilate -o stdout.vdb >! sphere.vdb
vdb_tool -sphere -dilate -o stdout.vdb | vdb_view
wget -qO- https://artifacts.aswf.io/io/aswf/openvdb/models/bunny.vdb/1.0.0/bunny.vdb-1.0.0.zip | bsdtar -xvO | vdb_tool -i stdin.vdb -dilate -o stdout.vdb | vdb_view
wget -qO- https://people.sc.fsu.edu/~jburkardt/data/ply/cow.ply | vdb_tool -read stdin.ply -mesh2ls -o stdout.vdb | vdb_view
```


# non-linear workflows: for and each loops
```
vdb_tool -sphere -sphere c=0.5,0,0 -sphere c=-0.5,0,0 -union -union -debug -o stdout.vdb | vdb_view
vdb_tool -sphere -for x=-0.5,1,1 -sphere c={$x},0,0 -union -end -o stdout.vdb | vdb_view
vdb_tool -debug -sphere n=sphere_0 -for x=-0.5,1,1 -sphere n=sphere_{$#x} c={$x},0,0 -union -end -o stdout.vdb | vdb_view
```

 # example of double loop
 ```
vdb_tool -read ~/dev/data/mesh/teapot.ply -mesh2ls -print -render
vdb_tool -read ~/dev/data/mesh/*.ply -for i=0,2,1 -mesh2ls -end -print
vdb_tool -for v=0.5,2,0.5 -read ~/dev/data/mesh/teapot.ply -mesh2ls voxel={$v} -render test_{$v}.png -end
vdb_tool -for v=0.5,2,0.5 -each s=teapot,bunny -read ~/dev/data/mesh/{$s}.ply -mesh2ls voxel={$v} -render {$s}_{$v}.png -end -end
```

# Enright benchmark test
```
vdb_tool -sphere d=64 r=0.15 c=0.35,0.35,0.35 -enright -render test.jpg
vdb_tool -sphere d=64 r=0.15 c=0.35,0.35,0.35 -for i=1,10,1 -enright dt=0.1 -render enright_{$i}.png k=1 -end -o stdout.vdb | vdb_view
vdb_tool -sphere d=64 r=0.15 c=0.35,0.35,0.35 -for i=1,10,1 -enright dt=0.1 k=1 -end -o stdout.vdb | vdb_view
```

# For more documentation and help
```
vdb_tool -help
vdb_tool -h read write
vdb_tool -sphere -help -write sphere.vdb
```