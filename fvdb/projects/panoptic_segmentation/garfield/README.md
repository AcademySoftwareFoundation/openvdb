# <img src="https://www.garfield.studio/data/favicon.png" height="30px"> GARField: Group Anything with Radiance Fields

This code is based on the official implementation for [GARField](https://github.com/chungmin99/garfield).

## Installation

1. Create the `fvdb_garfield` environment with conda.  This will install or build all the necessary dependencies.
   ```bash
   conda env create -f ./garfield_environment.yml
   ```

2. Activate the `fvdb_garfield` environment and install the `garfield` package.
   ```bash
   conda activate fvdb_garfield
   pip install -e .
   ```

## Running GARField

1. Download example colmap data
   ```bash
   ./download_example-data.py
   ```

2. Run GARField on the example data
   ```bash
   ns-train garfield --data ./data/figurines
   ```

3. (Optional) Run GARField with Gaussian Splatting geometry
   ```bash
   ns-train garfield-gauss --data ./data/figurines --pipeline.garfield-ckpt outputs/figurines/garfied/[datetimestamp]/config.yml
   ```
