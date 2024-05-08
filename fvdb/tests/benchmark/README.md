# fVDB Benchmarking Test Suite

## Introduction
### Important Files in this Directory

- `conftest.py`: Contains the necessary `pytest` configuration for Continuous Benchmarking
- `test_*.py`: Continuous Benchmarking `pytest` test definitions
- `data/`: Contains the data used for the benchmarking tests (see [Benchmark Data](#benchmark-data))
- `comparative_benchmark.py`: The script to run the standalone comparative benchmarking tests.

### Environment Setup

For the benchmark tests, we use the same environment as the testing environment.
One can simply install it by:
```bash
conda env create -f env/test_environment.yml
```

However, if running the `comparative_benchmark`, the baseline that we compare against, i.e. `torchsparse`, requires version >= 2.1.0, which is not included in the above yaml file. Hence, you have to install it manually:
```bash
git clone https://github.com/mit-han-lab/torchsparse
cd torchsparse
python setup.py install
```

### Benchmark Data

Please download the data from [here](https://drive.google.com/drive/folders/1zkmdmc-IxVqqkEnaeGwSV3Nef-V7UIM6?usp=sharing), and put all the folders within it into `tests/benchmark/data/` folder.

## Running the Continuous Benchmarking tests

The Continuous Benchmarking tests exist to catch regressions in performance in a set of performance measurements we take of individual operators, end-to-end networks, and other components of the library. The tests are run on every PR to the repository, and the results are displayed in the test results and as graphs that can be found in the Github Pages of the repository.

The pipeline that runs the continuous benchmarking tests as a Github Action is defined in `.github/workflows/benchmark.yml`. Running these tests yourself from the command line is also possible with `pytest-benchmark`:

```bash
pytest tests/benchmark
```

## Running the Comparative Benchmarking tests

The comparative benchmarking tests are designed to compare the performance of ƒVDB against other frameworks. These are the tests used to benchmark performance from our publication.  These tests are run manually.

The tests could be run with the following command:
```bash
python tests/benchmark/comparative_benchmark.py [experiment_name] \
    [--limit 10] \      # Limit the number of test samples to run
    [--detail]   \      # Dump detailed runtime comparison of each network component
    [--repeats 1]  \    # Number of repeats for each datum for each baseline
```

Up until now, available `experiment_name` are:
- `xcube` for running the XCube networks
- `kitti_segmentation` for running the KITTI segmentation networks (MinkUNet)

You will be able to see results dumped in the command line output in rich Tables.
