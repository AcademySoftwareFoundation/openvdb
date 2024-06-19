# Sparse Grid I/O

We give an overview of ways to save and load sparse grids including how fVDB's serialized format relates to other libraries such as OpenVDB and NanoVDB.  All of the examples in this tutorial are available in the `examples/io.py` file of the fVDB repository.

In these examples we will be using tools which are part of the `NanoVDB` project such as `nanovdb_convert`.  It is assumed that the `NanoVDB` tools are available to call (i.e. findable on the system's `$PATH`).  If you have not already installed these tools, you can find instructions on building NanoVDB with OpenVDB from its documentation:

https://www.openvdb.org/documentation/doxygen/NanoVDB_HowToBuild.html

While not necessary for fVDB's functionality, they are useful utilities to have available for inspecting and manipulating sparse grids.

## Python Serialization

Batches of sparse grids can be serialized to a NanoVDB file using the `fvdb.save` method.  Here, we create two grids of different sizes and numbers of points and save them to a compressed NanoVDB file with specified names.  The names are optional.

```python
import torch
import fvdb

p = fvdb.JaggedTensor(
    [
        torch.randn(10, 3),
        torch.randn(100, 3),
    ]
)
grid = fvdb.sparse_grid_from_points(
    p, voxel_sizes=[[0.1, 0.1, 0.1], [0.15, 0.15, 0.15]], origins=[0.0] * 3
)

# save the grid and features to a compressed nvdb file
path = os.path.join(save_dir, "two_random_grids.nvdb")
fvdb.save(path, grid, names=["taco1", "taco2"], compressed=True)
```

We can use the `nanovdb_print` command line tool to show information about our saved file.  Note how our grids have the `INDEX` class since they have no features and only store the voxel indices.

```bash

The file "/tmp/tmpwnu7qc_7/two_random_grids.nvdb" contains the following 2 grids:
#  Name   Type     Class  Version  Codec  Size      File      Scale             # Voxels  Resolution
1  taco1  OnIndex  INDEX  32.6.0   BLOSC  1.453 MB  7.181 KB  (0.1,0.1,0.1)     10        22 x 43 x 37
2  taco2  OnIndex  INDEX  32.6.0   BLOSC  2.326 MB  12.7 KB   (0.15,0.15,0.15)  100       33 x 39 x 38
```

We can include N-dimensional features by passing a JaggedTensor as the second argument (or `data` kwarg) to `fvdb.save`.  Here, we create a grid with a single, float feature channel for our grids and save it to a `nvdb` file.

```python
# a single, scalar float feature per grid
feats = fvdb.JaggedTensor([torch.randn(x, 1) for x in grid.num_voxels])

# save the grid and features to a compressed nvdb file
path = os.path.join(save_dir, "two_random_grids.nvdb")
fvdb.save(path, grid, feats, names=["taco1", "taco2"], compressed=True)
```

Again, we can use the `nanovdb_print` command line tool to show information about our saved file.

```bash
The file "/tmp/tmpg9ol5841/two_random_grids.nvdb" contains the following 2 grids:
#  Name   Type   Class  Version  Codec  Size      File      Scale    # Voxels  Resolution
1  taco1  float  ?      32.6.0   BLOSC  1.47 MB   7.959 KB  (1,1,1)  10        26 x 30 x 36
2  taco2  float  ?      32.6.0   BLOSC  2.411 MB  16.48 KB  (1,1,1)  100       32 x 33 x 34
```

Note how our serialized NanoVDB grids are now of type `float`.  fVDB will automatically map N-dimensional features to appropriate NanoVDB types.  For feature sizes that don't naturally map to any NanoVDB data types, fVDB will save the feature data as NanoVDB blind-data which will be appropriately read back as N-dimensional features by fVDB.

Let's try to save the same two grids with a `Vec3d` type by creating a JaggedTensor of 3-dimensional double-precision features.

```python
# a 3-vector double feature per grid
feats = fvdb.JaggedTensor([torch.randn(x, 3, dtype=torch.float64) for x in grid.num_voxels])

# save the grid and features to a compressed nvdb file
path = os.path.join(save_dir, "two_random_vec3d_grids.nvdb")
fvdb.save(path, grid, feats, names=["taco1", "taco2"], compressed=True)
```

```bash
The file "/tmp/tmpwnu7qc_7/two_random_grids.nvdb" contains the following 2 grids:
#  Name   Type   Class  Version  Codec  Size      File      Scale    # Voxels  Resolution
1  taco1  Vec3d  ?      32.6.0   BLOSC  6.077 MB  28.18 KB  (1,1,1)  10        23 x 36 x 35
2  taco2  Vec3d  ?      32.6.0   BLOSC  7.346 MB  41.37 KB  (1,1,1)  100       37 x 40 x 34
```

## Loading NanoVDB Files

Loading NanoVDB files is as simple as calling `fvdb.load`.  You can optionally supply a PyTorch device you'd like the grids and features loaded onto.  Here, we load the two grids we saved in the previous section onto our GPU.

```python
# Load the grid and features from the compressed nvdb file
grid_batch, features, names = fvdb.load(saved_nvdb, device=torch.device("cuda:0"))
print("Loaded grid batch total number of voxels: ", grid_batch.total_voxels)
print("Loaded grid batch data type: %s, device: %s" % (features.dtype, features.device))
```

```bash
Loaded grid batch total number of voxels:  110
Loaded grid batch data type: torch.float64, device: cuda:0
```

## Saving/Loading OpenVDB Files

While saving and loading from OpenVDB files is not directly supported by fVDB, it is possible to easily convert between NanoVDB and OpenVDB files using the `nanovdb_convert` command line tool.  Here, we convert our previously saved NanoVDB file to an OpenVDB file.

```python
vdb_path = os.path.join(tmpdir, "two_random_grids.vdb")
convert_cmd = "nanovdb_convert -v %s %s" % (saved_nvdb, vdb_path)
print("nanovdb_convert our nvdb to vdb: ", convert_cmd)
print(subprocess.check_output(convert_cmd.split()).decode("utf-8"))
```

```bash
nanovdb_convert our nvdb to vdb:  nanovdb_convert -v /tmp/tmpnr7gk0tf/two_random_vec3d_grids.nvdb /tmp/tmpnr7gk0tf/two_random_grids.vdb
Opening NanoVDB file named "/tmp/tmpnr7gk0tf/two_random_vec3d_grids.nvdb"
Read 1 NanoGrid(s) from the file named "/tmp/tmpnr7gk0tf/two_random_vec3d_grids.nvdb"
Converting NanoVDB grid named "taco1" to OpenVDB
Converting NanoVDB grid named "taco2" to OpenVDB
```

From here, our grid can be loaded by OpenVDB tools.  Roundtripping our converted cache back to NanoVDB is possible with the `nanovdb_convert` tool as well.

Loading the converted OpenVDB file into fVDB shows our familiar grids and features as we expect:


```python
convert_cmd = "nanovdb_convert -v -f %s %s" % (  # -f flag forces overwriting existing file
    vdb_path,
    saved_nvdb,
)
print("nanovdb_convert roundtrip the vdb to nvdb: ", convert_cmd)
print(subprocess.check_output(convert_cmd.split()).decode("utf-8"))

# Load the nvdb file of the converted vdb
grid_batch, features, names = fvdb.load(saved_nvdb, device=torch.device("cuda:0"))
print("Loaded grid batch total number of voxels: ", grid_batch.total_voxels)
print("Loaded grid batch data type: %s, device: %s" % (features.dtype, features.device))
print("\n")
```

```bash
nanovdb_convert roundtrip the vdb to nvdb:  nanovdb_convert -v -f /tmp/tmpuwq2r5mx/two_random_grids.vdb /tmp/tmpuwq2r5mx/two_random_vec3d_grids.nvdb
Opening OpenVDB file named "/tmp/tmpuwq2r5mx/two_random_grids.vdb"
Converting OpenVDB grid named "taco1" to NanoVDB
Converting OpenVDB grid named "taco2" to NanoVDB

Loaded grid batch total number of voxels:  110
Loaded grid batch data type: torch.float64, device: cuda:0
```