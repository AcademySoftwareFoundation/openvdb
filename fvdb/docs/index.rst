Welcome to fVDB!
=================

fVDB, inspired by function notation to resemble :math:`f(VDB)`, is a data structure for encoding and operating on *sparse* voxel hierarchies of features in PyTorch.
A sparse voxel hierarchy is a coarse-to-fine hierarchy of sparse voxel grids such that every fine voxel is contained within some coarse voxel.
fvdb supports storing PyTorch tensors at the corners and centers of voxels in a hierarchy and enables a number of differentiable operations on these tensors (e.g. trilinear interpolation, splatting, ray tracing).

.. image:: imgs/fvdb_teaser.png
   :align: center
   :width: 400

Please refer to the tutorials for examples of how to install and use fVDB to build your own sparse voxel pipelines.

.. toctree::
   :caption: Introduction
   :hidden:

   self

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/installation
   tutorials/basic_concepts
   tutorials/building_grids
   tutorials/basic_grid_ops
   tutorials/ray_tracing
   tutorials/simple_unet
   tutorials/io
   tutorials/mutable_grids
   tutorials/volume_rendering

.. toctree::
   :maxdepth: 1
   :caption: API References

   api/grid_batch
   api/jagged_tensor

.. toctree::
   :maxdepth: 2

   api/nn
   api/utils

.. raw:: html

   <hr>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
