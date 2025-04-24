# ƒVDB Docker Help

Running a docker container is a great way to ensure that you have a consistent environment for building and running ƒVDB.

Our provided [`Dockerfile`](../../Dockerfile) constructs a Docker image which is ready to build ƒVDB.  The docker image is configured to install miniforge and the `fvdb` conda environment with all the dependencies needed to build and run ƒVDB.

## Setting up a Docker Container

Building and starting the docker image is done by running the following command from the fvdb directory:
```shell
docker compose run --rm fvdb-dev
```

When you are ready to build ƒVDB, run the following command within the docker container.  `TORCH_CUDA_ARCH_LIST` specifies which CUDA architectures to build for.
```shell
conda activate fvdb;
cd /openvdb/fvdb;
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" \
./build.sh install
```

### Troubleshooting

* **docker daemon runtime/driver errors**
Errors like these:
```shell
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
```shell
docker: Error response from daemon: Unknown runtime specified nvidia.
```
most likely indicate that the [`NVIDIA Container Toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) is not installed.
You can install it by following the [installation instructions here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

* **GPU access blocked by the operating system**
Errors like these:
```shell
Failed to initialize NVML: GPU access blocked by the operating system
Failed to properly shut down NVML: GPU access blocked by the operating system
```
may be solved by making a change to the file `/etc/nvidia-container-toolkit/config.toml`, setting
the property "no-cgroups" from "true" to "false":

```
# ...
[nvidia-container-cli]
# ...
# Below changed from previous no-cgroups = true
no-cgroups = false
# ...
```
