# *f*(VDB)


#### The *f*VDB API is in alpha. If you depend on it for your project, expect it to change under you!

This repository contains the code for *f*VDB, a data structure for encoding and operating on *Sparse voxel hierarchies* of features in PyTorch. A sparse voxel hierarchy is a coarse-to-fine hierarchy of sparse voxel grids such that every fine voxel is contained within some coarse voxel. The image below illustrates an example. *f*VDB supports storing PyTorch tensors at the corners and centers of voxels in a hierarchy and enables a number of differentiable operations on these tensors (*e.g.* trilinear interpolation, splatting, ray tracing).

<p align="center">
  <img src="docs/imgs/fvdb_teaser.png" style="width: 40%;"alt="fVDB Teaser">
  <!-- <img src="data/av_screenshot.png" style="width: 100%;"alt="fVDB Teaser"> -->
  <figcaption style="text-align: center; font-style: italic;">An example of a sparse voxel hierarchy with 3 levels. Each fine voxel is contained within exactly one coarse voxel.</figcaption>
</p>


## Building *f*VDB
*f*VDB is a Python library implemented as a C++ Pytorch extension.

**(Optional) Install libMamba for a huge quality of life improvement when using Conda**
```
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

**Conda Environment.** Next, create the `fvdb` conda environment by running the following command from the root of this repository, and then grabbing a ☕:
```shell
conda env create -f env/test_environment.yml
```

Note:  You can optionally use the `env/build_environment.yml` environment file if you want a minimum set of dependencies needed to build *f*VDB and don't intend to run the tests.

Now activate the environment:
```shell
conda activate fvdb
```
PyTorch cannot find the conda `libcudart.so` when JIT compiling extensions, so create the following symlink:
```shell
ln -s ${CONDA_PREFIX}/lib ${CONDA_PREFIX}/lib64
```

**Building *f*VDB.**
You could either do an editable install with setuptools:
```shell
python setup.py develop
```
or directly install it to your site package folder if you are developing extensions:
```shell
pip install .
```
In both of the above cases, you should run from the root of the repository.

Note: Compilation can be very memory-consuming. Please add environment variable `MAX_JOBS=N` and set `N` to be a small value to reduce parallelism, so your compilation doesn't get killed due to OOM.

**Running Tests.** To make sure that everything works by running tests:
```shell
python setup.py test
```

**Building Docs.** To build the documentation, simply run:
```shell
# (Sphinx-7.0.0 works)
python setup.py build_ext --inplace
sphinx-build -E -a docs/ build/sphinx
# View the docs
open build/sphinx/index.html
```

**Docker Image.** To build and test feature-vdb, we have the dockerfile available:
```shell
# Build feature-vdb
docker build . -t fvdb-dev
# Run feature-vdb (or replace with your command)
docker run -it --gpus all --rm \
  --user $(id -u):$(id -g) \
  --mount type=bind,source="$HOME/.ssh",target=/root/.ssh \
  --mount type=bind,source="$(pwd)",target=/feature-vdb \
  fvdb-dev:latest \
  conda run -n fvdb_test --no-capture-output python setup.py test
```


## Code Structure
The main source code for fVDB lives in the [src](src) directory. There are several important files here:
* `src/PythonBindings.cpp` exposes functionality directly to Python. It is mainly a wrapper around the `SparseFeatureIndexGrid` class.
* `src/SparseFeatureIndexGrid.h` contains the implementation of `SparseFeatureIndexGrid` which is the core data structure on which fVDB is built. A `SparseFeatureIndexGrid` acts as a map between `(i, j, k)` integer coordinates and offsets in linear memory. This mapping can be used to perform a host of operations. The methods in this class are mostly lightweight wrappers around a set of CPU and CUDA *kernels*. The function prototypes for these kernels are defined in `src/Ops.h`.
* `src/Ops.h` contains the function prototypes for the main kernels used by fVDB. These are only prototypes since there are both CPU kernels (implemented in `src/ops/cpu`) and CUDA kernels (implemented in `src/ops/cuda`)
  * `src/ops/cpu/` contains CPU only implementations of the main kernels used by fVDB.
  * `src/ops/cuda` contains CUDA implementations of the main kernels used by fVDB.
* `src/autograd` contains C++ implementations of PyTorch autograd functions for differentiable operations. Including `autograd/Functions.h` includs all of the functions in this folder.
* `src/utils` contains a number of utilities which make it easier to use NanoVDB.




## Usage Examples
The [scripts](scripts) directory contains a number of examples of using the `fvdb` Python package. The sections below show some notable examples and their outputs. Run all commands from the root of the repository.

### Trilinear sampling of grids
```
python scripts/debug_trilerp.py
```
This script generates a grid with scalars at the corners of each voxel and samples this grid at points. The visualization below shows the points colored according to their sampled values as well as the grid corners.
<p align="center">
  <img src="data/trilerp.png" style="width: 40%;"alt="fVDB trilinear interpolation demo">
  <figcaption style="text-align: center; font-style: italic;">Trilinearly interpolate the corner values at the points.</figcaption>
</p>


### Trilinear splatting into grids
```
python scripts/debug_splat.py
```
This script splats normals of a point cloud onto grid centers. The green arrows are the normals splatted onto each grid center
<p align="center">
  <img src="data/splat.png" style="width: 40%;"alt="fVDB trilinear splatting demo">
  <figcaption style="text-align: center; font-style: italic;">Splat the normals at the blue points into the center of each grid cell. The green arrows are the splatted normals</figcaption>
</p>


### Tracing voxels along rays (hierarchical DDA)
```
python scripts/debug_ray_voxel_marching.py
```
This script demonstrates finding the first `N` voxels which lie along a ray (returning thier index as well as their entry and exit points).
<p align="center">
  <img src="data/rayvox.png" style="width: 70%;"alt="fVDB ray voxel marching">
  <figcaption style="text-align: center; font-style: italic;">Find the voxels (yellow) which intersect the pink rays eminating from the green dot.</figcaption>
</p>


### Tracing contiguous segments along rays
```
python scripts/debug_ray_segment_marching.py
```
This script demonstrates finding the first `N` continuous segments of voxels which lie along a ray (returning thier index as well as their entry and exit points).
<p align="center">
  <img src="data/rayseg.png" style="width: 70%;"alt="fVDB ray voxel marching">
  <figcaption style="text-align: center; font-style: italic;">Find the contiguous segments of voxels (red and blue lines) which intersect the cyan rays eminating from the pink dot.</figcaption>
</p>


### Backpropagating through sampling and splatting
```
python scripts/debug_overfit_sdf.py
```
This scripts fits SDF values at a grid corner to the SDF of a mesh using gradient descent.
<p align="center">
  <img src="data/fitsdf.png" style="width: 70%;"alt="fVDB SDF fitting">
  <figcaption style="text-align: center; font-style: italic;">SDF values at grid corners (colored dots) fitted using gradient descent to the SDF of a mesh.</figcaption>
</p>

The following scripts also show how to bakcprop through splatting and sampling with fVDB:
```
python scripts/debug_grad_trilerp.py
```
```
python scripts/debug_grad_splat.py
```

