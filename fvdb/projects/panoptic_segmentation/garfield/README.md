# <img src="https://www.garfield.studio/data/favicon.png" height="30px"> GARField: Group Anything with Radiance Fields

This code is based on the official implementation for [GARField](https://github.com/chungmin99/garfield).

## Installation

1. Create the `fvdb_garfield` environment with conda.  This will install or build all the necessary dependencies.  This may take a while because some dependencies require building from source to build CUDA 12.1 versions.
   ```bash
   conda env create -f ./garfield_environment.yml
   ```

2. Activate the `fvdb_garfield` environment and install the `garfield` package.
   ```bash
   conda activate fvdb_garfield
   pip install -e .
   ```

## Running GARField

1. Download example image, camera info and COLMAP data
   ```bash
   ./download_example-data.py
   ```

2. Run the original GARField implementation on the example data
   ```bash
   ns-train garfield --data ./data/dozer_nerfgun_waldo
   ```

3. (Optional) Run GARField with Gaussian Splatting geometry
   ```bash
   ns-train garfield-gauss --data ./data/dozer_nerfgun_waldo --pipeline.garfield-ckpt outputs/dozer_nerfgun_waldo/garfied/[datetimestamp]/config.yml
   ```

4. Run fVDB 3D Gaussian Splatting on the example data
   ```bash
   python [fVDB_root]/projects/3d_gaussian_splatting/training/train_colmap.py --data ./data/dozer_nerfgun_waldo
   ```
