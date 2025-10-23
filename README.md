# *ƒ*(VDB)

## A New Home for ƒVDB

:warning: The public home for fVDB development is now located in the Academy Software Foundation OpenVDB Github Organization [https://github.com/openvdb](https://github.com/openvdb). :warning:

All of the code and history for fVDB on the OpenVDB `feature/fvdb` branch has been moved to repositories in the OpenVDB Github Organization. In order to encourage movement to fVDB's new home in the OpenVDB Github Organization, we have removed the code from the head of this branch but the history remains.

The components of fVDB have been split into the following repositories:

### [fvdb-core](https://github.com/openvdb/fvdb-core)

The fvdb-core repository houses the core library and PyTorch extension for ƒVDB. ƒVDB is a framework that provides differentiable, sparse volumetric operators built on top of NanoVDB and enables PyTorch users to utilize the NanoVDB data structure to build powerful and scalable spatial intelligence applications.


### [fvdb-examples](https://github.com/openvdb/fvdb-examples)

The fvdb-examples repository contains examples of how to use the ƒVDB library to build spatial intelligence networks or pipelines. These are provided as reference for how to implement interesting or well-known methods and networks on top of ƒVDB including examples of pipelines that perform panoptic segmentation and depth reconstruction.


### [fvdb-reality-capture](https://github.com/openvdb/fvdb-reality-capture)

The fvdb-reality-capture repository contains examples of how to use the ƒVDB library to build reality capture pipelines centered on ƒVDB's 3D Gaussian splatting methods. This repository houses utilities and examples that illustrate how to train and render 3D Gaussian splatting models using ƒVDB as well as interesting methods that build upon the 3D Gaussian scene representation such as meshing.


### [nanovdb-editor](https://github.com/openvdb/nanovdb-editor)

The nanovdb-editor repository contains a library, python bindings and a standalone GUI application that allows you to view and edit NanoVDB.  nanovdb-editor also provides viewer functionality for fVDB.
