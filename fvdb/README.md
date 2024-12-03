# *ƒ*(VDB)


This repository contains the code for *f*VDB, a data structure for encoding and operating on *sparse voxel hierarchies* of features in PyTorch. A sparse voxel hierarchy is a coarse-to-fine hierarchy of sparse voxel grids such that every fine voxel is contained within some coarse voxel. The image below illustrates an example. *f*VDB supports using PyTorch Tensors to represent features at the corners and centers of voxels in a hierarchy and enables a number of differentiable operations on these Tensors (*e.g.* trilinear interpolation, convolution, splatting, ray tracing).

<p align="center">
  <img src="docs/imgs/fvdb_teaser.png" style="width: 40%;"alt="fVDB Teaser">
  <!-- <img src="docs/imgs/readme/av_screenshot.png" style="width: 100%;"alt="fVDB Teaser"> -->
  <figcaption style="text-align: center; font-style: italic;">An example of a sparse voxel hierarchy with 3 levels. Each fine voxel is contained within exactly one coarse voxel.</figcaption>
</p>

## Learning to Use *f*VDB

After [installing *f*VDB](#installing-fvdb), we recommend starting with our walk-through [notebooks](notebooks) which provide a gentle, illustrated introduction to the main concepts and operations in *f*VDB.

Once familiar with the basics, [Usage Examples](#usage-examples) introduces a few of the practical python scripts that can be further explored in the [examples](examples) directory.

Lastly, our [documentation](docs) provides deeper details on the concepts as well as an exhaustive set of illustrations of all the operations available in *f*VDB and an API reference. The [documentation can be built locally](#building-documentation) or can be accessed online at [TODO: insert link to online documentation].

## Installing *f*VDB

During the project's initial stage of release, it is necessary to [run the build steps](#building-fvdb-from-source) to install ƒVDB. Eventually, ƒVDB will be provided as a pre-built, installable package from anaconda.  We support building the latest ƒVDB version for the following dependent library configurations:

|   PyTorch      | Python      | CUDA |
| -------------- | ----------- | ------------ |
|  2.4.0-2.4.1   | 3.10 - 3.12 | 12.1 - 12.4 |



** Notes:**
* Linux is the only platform currently supported (Ubuntu >= 20.04 recommended).
* A CUDA-capable GPU with Ampere architecture or newer (i.e. compute capability >=8.0) is required to run the CUDA-accelerated operations in ƒVDB.


## Building *f*VDB from Source

### Environment Management
ƒVDB is a Python library implemented as a C++ Pytorch extension.  Of course you can build ƒVDB in whatever environment suits you, but we provide two paths to constructing reliable environments for building and running ƒVDB:  using [docker](#setting-up-a-docker-container) and using [conda](#setting-up-a-conda-environment).

`conda` tends to be more flexible since reconfiguring toolchains and modules to suit your larger project can be dynamic, but at the same time this can be a more brittle experience compared to using a virtualized `docker` container.  Using `conda` is generally recommended for development and testing, while using `docker` is recommended for CI/CD and deployment.

#### Setting up a Docker Container

Running a docker container is a great way to ensure that you have a consistent environment for building and running ƒVDB.

Our provided [`Dockerfile`](Dockerfile) has two modes for building the image: `dev` and `production`.  `production` constructs an image capable of building ƒVDB, builds and installs the ƒVDB libraries and is ready for you to start running python code that uses the `fvdb` module.  `dev` mode constructs an image which is ready to build ƒVDB but does not build the ƒVDB libraries.

Building the docker image in `production` mode is the default and is as simple as running the following command from the root of this repository:
```shell
# Build the docker image in production mode
docker build -t fvdb/prod .
```

Building the docker mage in `dev` mode is done by setting the `MODE` argument to `dev`:
```shell
# Build the docker image in dev mode
docker build --build-arg  MODE=dev -t fvdb/dev .
```

Running the docker container is done with the following command:
```shell
# Run an interactive bash shell (or replace with your command)
docker run -it --gpus all --rm \
  fvdb/dev:latest \
  /bin/bash
```

When running the docker container in `dev` mode and when you are ready to build ƒVDB, you can run the following command to build ƒVDB for the recommended set of CUDA architectures:
```shell
MAX_JOBS=$(free -g | awk '/^Mem:/{jobs=int($4/2.5); if(jobs<1) jobs=1; print jobs}')  \
     TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX" \
     python setup.py install
```

#### Setting up a Conda Environment

In order to get resolved package versions in your conda environment consistent with our testing, it is necessary to configure your `.condarc` since not all package resolving behaviour can be controlled with an `environment.yml` file.  We recommend using `strict` channel priority in your conda configuration.  This can be done by running the following command:

```shell
conda config --set channel_priority strict
```

Further, it is recommend to not mix the `defaults` and `conda-forge` package channels when resolving environments.  We have generally used `conda-forge` as the primary channel for our dependencies.  You can remove the `defaults` channel and add `conda-forge` with the following command:

```shell
conda config --remove channels defaults
conda config --add channels conda-forge
```

With these changes, it is recommended that your `.condarc` file looks like the following:

```yaml
channel_priority: strict
channels:
  - conda-forge
```


**(Optional) Install libMamba for a huge quality of life improvement when using Conda**
```
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```


Next, create the `fvdb` conda environment by running the following command from the root of this repository, and then grabbing a ☕:
```shell
conda env create -f env/dev_environment.yml
```

**Notes:**
* You can optionally use the `env/build_environment.yml` environment file if you want a minimum set of dependencies needed just to build/package *f*VDB (note this environment won't have all the runtime dependencies needed to `import fvdb`).
* If you would like a runtime environment which has only the packages required to run the unit tests after building ƒVDB, you can use the `env/test_environment.yml`.  This is the environment used by the CI pipeline to run the tests after building ƒVDB in the `fvdb_build` environment.
* Use the `fvdb_learn` environment defined in `env/learn_environment.yml` if you would like an environment with the runtime requirements and the additional packages needed to run the [notebooks](notebooks) or [examples](examples) and view their visualizations.

Now activate the environment:
```shell
conda activate fvdb
```


### Building *f*VDB

**:warning: Note:** Compilation can be very memory-consuming. We recommend setting the `MAX_JOBS` environment variable to control compilation job parallelism with a value that allows for one job every 2.5GB of memory:

```bash
export MAX_JOBS=$(free -g | awk '/^Mem:/{jobs=int($4/2.5); if(jobs<1) jobs=1; print jobs}')
```

You could either perform an editable install with setuptools:
```shell
python setup.py develop
```
or install a 'read-only' copy to your site package folder:
```shell
pip install .
```

If you would like to build a packaged wheel for installing in other environments, you can run the following command:
```shell
python setup.py bdist_wheel
```


### Running Tests

To make sure that everything works by running tests:
```shell
cd tests
pytest unit
```

### Building Documentation

To build the documentation, simply run:
```shell
python setup.py build_ext --inplace
sphinx-build -E -a docs/ build/sphinx
# View the docs
open build/sphinx/index.html
```



## Usage Examples
The [examples](examples) directory contains a number of useful illustrations using the `fvdb` Python package. The sections below show some notable examples and their outputs. Run all commands from the root of the repository.

### Trilinear sampling of grids
```
python examples/sample_trilinear.py
```
This script generates a grid with scalars at the corners of each voxel and samples this grid at points. The visualization below shows the points colored according to their sampled values as well as the values at grid corners.
<p align="center">
  <img src="docs/imgs/readme/trilerp.png" style="width: 40%;"alt="fVDB trilinear interpolation demo">
  <figcaption style="text-align: center; font-style: italic;">Trilinearly interpolate the corner values at the points.</figcaption>
</p>


### Trilinear splatting into grids
```
python examples/splat_trilinear.py
```
This script splats normals of a point cloud onto grid centers. The green arrows represent the values of the normals splatted onto each grid center
<p align="center">
  <img src="docs/imgs/readme/splat.png" style="width: 40%;"alt="fVDB trilinear splatting demo">
  <figcaption style="text-align: center; font-style: italic;">Splat the normals at the blue points into the center of each grid cell. The green arrows are the splatted normals</figcaption>
</p>


### Tracing voxels along rays (hierarchical DDA)
```
python examples/ray_voxel_marching.py
```
This script demonstrates finding the first `N` voxels which lie along a ray (returning thier index as well as their entry and exit points).
<p align="center">
  <img src="docs/imgs/readme/rayvox.png" style="width: 70%;"alt="fVDB ray voxel marching">
  <figcaption style="text-align: center; font-style: italic;">Find the voxels (yellow) which intersect the pink rays eminating from the green dot.</figcaption>
</p>


### Tracing contiguous segments along rays
```
python examples/ray_segment_marching.py
```
This script demonstrates finding the first `N` continuous segments of voxels which lie along a ray (returning thier index as well as their entry and exit points).
<p align="center">
  <img src="docs/imgs/readme/rayseg.png" style="width: 70%;"alt="fVDB ray voxel marching">
  <figcaption style="text-align: center; font-style: italic;">Find the contiguous segments of voxels (red and blue lines) which intersect the cyan rays eminating from the pink dot.</figcaption>
</p>


### Backpropagating through sampling and splatting
```
python examples/overfit_sdf.py
```
This scripts fits SDF values at a grid corner to the SDF of a mesh using gradient descent.
<p align="center">
  <img src="docs/imgs/readme/fitsdf.png" style="width: 70%;"alt="fVDB SDF fitting">
  <figcaption style="text-align: center; font-style: italic;">SDF values at grid corners (colored dots) fitted using gradient descent to the SDF of a mesh.</figcaption>
</p>

The following scripts also show how to bakcprop through splatting and sampling with fVDB:
```
python scripts/debug_grad_trilerp.py
```
```
python scripts/debug_grad_splat.py
```

## Code Structure
The main source code for fVDB lives in the [src](src) directory. There are several important files here:
* `src/python/Bindings.cpp` exposes functionality directly to Python. It is mainly a wrapper around the core classes such as `fvdb::GridBatch` and `fvdb::JaggedTensor`.
* `src/GridBatch.h` contains the implementation of `fvdb::GridBatch` which is the core data structure on which fVDB is built. A `GridBatch` acts as a map between `(i, j, k)` integer coordinates and offsets in linear memory. This mapping can be used to perform a host of operations. The methods in this class are mostly lightweight wrappers around a set of CPU and CUDA *kernels*. The function prototypes for these kernels are defined in `src/detail/ops/Ops.h`.
* `src/detail/ops/Ops.h` contains the function prototypes for the main kernels used by fVDB. Host and device kernel implementations are provided in the `src/detail/ops/*.cu` source files.
* `src/detail/autograd` contains C++ implementations of PyTorch autograd functions for differentiable operations.  `#include <detail/autograd/Autograd.h>` includes all of the functions in this directory.
* `src/detail/utils/nanovdb` contains a number of utilities which make it easier to use NanoVDB.


## References

Please consider citing this when using *f*VDB in a project. You can use the citation BibTeX:

```bibtex
@article{williams2024fvdb,
  title={fVDB: A Deep-Learning Framework for Sparse, Large-Scale, and High-Performance Spatial Intelligence},
  author={Williams, Francis and Huang, Jiahui and Swartz, Jonathan and Klar, Gergely and Thakkar, Vijay and Cong, Matthew and Ren, Xuanchi and Li, Ruilong and Fuji-Tsang, Clement and Fidler, Sanja and Sifakis, Eftychios and Museth, Ken},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={4},
  pages={133:1--133:15},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

## Contact