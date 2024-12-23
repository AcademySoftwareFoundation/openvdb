# <img src="https://www.garfield.studio/data/favicon.png" height="30px"> GARField: Group Anything with Radiance Fields

This is the official implementation for [GARField](https://www.garfield.studio).

Tested on Python 3.10, cuda 12.0, using conda.

<div align='center'>
<img src="https://www.garfield.studio/data/garfield_training.jpg" height="230px">
</div>

## Installation
1. Install nerfstudio from source, and its dependencies. This project requires the latest version of nerfstudio
(more specifically, the new viewer based on viser).
```
# install dependencies
pip3 install torch torchvision torchaudio
conda install -c "nvidia/label/cuda-12.0.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install nerfstudio!
git clone git@github.com:nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install -e .
```

2. To use GARField with Gaussian Splatting, [`cuml`](https://docs.rapids.ai/install) is required (for global clustering).
The best way to install it is through conda: `conda install -c rapidsai -c conda-forge -c nvidia cuml`

, or with pip: `pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.2.* cuml-cu12==24.2.*`.

Important: I used [`libmamba`](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) for conda. I have been told multiple times that the conda solver is very slow / gets stuck, but this seems to be key.

If you get `ClobberError`, try `conda clean --all` -- see [here](https://stackoverflow.com/questions/51217876/conda-update-anaconda-fails-clobbererror). It seems that `pip` installed packages from `nerfstudio` may conflict with the `conda` install here.

3. Install GARField!
```
git clone git@github.com:chungmin99/garfield.git
pip install -e .
```

This installs both `garfield` (NeRF geometry), and `garfield-gauss` (Gaussian geometry).
Note that `garfield-gauss` requires reference to a fully trained `garfield` checkpoint,
as it relies on the affinity field from `garfield`. See the main paper for more details.

4. (Optional) If you wish to use a different version of the SAM model (by default, the Hugging Face Transformer's SAM model facebook/sam-vit-huge is used), please install the 'segment_anything' package.

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Running GARField

Note: using colmap-based image data makes it more convenient to run both `garfield` and `garfield-gauss` on the same dataset. Although `splatfacto` (Gaussian Splatting in nerfstudio) is supported with `NerfstudioDataParser`, and also supports random point initialization with non-colmap datasets, the NeRF and GS geometries will align better with colmap since 1) we will start from colmap points and 2) camera optimization is minimized.

You can use it like any other third-party nerfstudio project.
```
ns-train garfield --data /your/data/here
```
Note that GARField will pause to generate groups using Segment-Anything at around 2000 steps
(set by default, this can be set in GarfieldPipeline).
Afterwards, you can start interacting with the affinity field.
1. PCA visualization of affinity field: select `instance` as the output type,
   and change the value of `scale` slider.

https://github.com/chungmin99/garfield/assets/10284938/e193d7e8-da7c-4176-b7c5-a7ec75513c16

2. Affinity visualization between 3D point and scene: use "Click" button to
   select the point, and select `instance_interact` as the output type.
   You might need to drag the viewer window slightly to see this output type.
   Again, interact with the `scale` slider!
Here, with `invert` True and output unnormalized, red color means high affinity (i.e., features at click point and rendered point are close to each other). Blue means low affinity.

https://github.com/chungmin99/garfield/assets/10284938/6edbdad6-d356-4b32-b44e-0df8ec1dca16

Also, note: the results can change a lot between 2k to 30k steps.

Once the model is trained to completion, you can use the outputted config file for `garfield-gauss`.

## Running GARField with Gaussian Splatting geometry!
Although GARField's affinity field is optimized using NeRF geometry, it can be
used to group and cluster gaussians in 3D!
```
ns-train garfield-gauss --data /your/data/here --pipeline.garfield-ckpt outputs/your/data/garfield/.../config.yml
```

There are two main ways to interact with the scene -- make sure to pause training first!
1. Interactive selection: click anywhere in the scene, and use "Crop to Click" button to retrieve different groups (scale=group level*0.05). Use "Drag Current Crop" to move it around!


https://github.com/chungmin99/garfield/assets/10284938/82ea7145-d8d1-485d-bab2-f6e8b0ebd632


2. Global clustering: cluster the currently visible gaussians (either globally or just for the crop), at the scale specified by "Cluster Scale".


https://github.com/chungmin99/garfield/assets/10284938/541fe037-925c-418f-929d-a9397f8d57d3



## Citation
If you use this work or find it helpful, please consider citing: (bibtex)

```
@inproceedings{garfield2024,
 author = {Kim, Chung Min* and Wu, Mingxuan* and Kerr, Justin* and Tancik, Matthew and Goldberg, Ken and Kanazawa, Angjoo},
 title = {GARField: Group Anything with Radiance Fields},
 booktitle = {arXiv},
 year = {2024},
}
```
