# Installation

## Pre-built wheels

fVDB is provided as an installable python package.
If you want to install the `fvdb` package into other python environments, we provide the following pre-built wheels.
Note that only Linux is supported for now (Ubuntu >= 20.04 recommended).


|                | Python     | `cu113` | `cu116` | `cu117` | `cu118` |
| -------------- | ---------- | ------- | ------- | ------- | ------- |
| PyTorch 2.0.0  | 3.8 - 3.11 |         |         | ✅       | ✅       |
| PyTorch 1.13.0 | 3.7 - 3.11 |         | ✅       | ✅       |         |
| PyTorch 1.12.0 | 3.7 - 3.10 | ✅       | ✅       |         |         |

Use the following command to install `fvdb`.

```bash
pip install -U fvdb -f https://fvdb.huangjh.tech/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
```

An example combination would be `TORCH_VERSION=2.0.0` and `CUDA_VERSION=cu118`.
