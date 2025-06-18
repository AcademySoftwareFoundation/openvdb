# 3D Gaussian Splatting w/ƒVDB

## Getting Going

1. Setup the environment:

    ```bash
    conda env create -f ./3dgs_environment.yml
    conda activate fvdb_3dgs
    ```

2. Download the example data

    ```bash
    ./download_example-data.py
    ```

3. Run the `train_colmap.py` example
    ```bash
    python train_colmap.py --data-path data/360_v2/[scene_name]
    ```

4. View the results in a browser at `http://localhost:8080`
