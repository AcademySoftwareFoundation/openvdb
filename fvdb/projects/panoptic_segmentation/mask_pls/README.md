# Mask-PLS: Panoptic LiDAR Segmentation

This project implements [Mask-PLS (Mask-Based Panoptic LiDAR Segmentation)](https://github.com/PRBonn/MaskPLS) for panoptic LiDAR segmentation using fVDB. The model performs both semantic segmentation and instance segmentation of LiDAR point clouds.

## Requirements

Build an environment with the required dependencies for this project and install the `fVDB` package from a built wheel:

```bash
conda env create -f maskpls_envrionment.yml
conda activate maskpls
pip install /path/to/fVDB/dist/fvdb-0.2.1-cp311-cp311-linux_x86_64.whl # Replace with the correct wheel
```

## Usage

A basic example of training the model is contained in `train.py`. The script can be run with the following command:

```bash
python train.py --dataset-type SemanticKITTI \
                --dataset-path /path/to/datasets/SemanticKITTI \
                --dataset-spatial-normalization 82 80 32 # Magnitude of the spatial extents of the dataset for normalization
```

## Model Architecture

- **MaskPLS**: The main model class that implements the full architecture with sub-modules:
  - `MaskPLSEncoderDecoder` semantic segmentation head
  - Optional masked transformer decoder `MaskedTransformerDecoder` for instance segmentation


## Supported Datasets

- SemanticKITTI: Standard automotive LiDAR dataset
- E57: Generic point cloud format (random labels for testing)


## References

Based on the MaskPLS paper: [MaskPLS: Mask-Based Panoptic LiDAR Segmentation](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/marcuzzi2023ral.pdf)
