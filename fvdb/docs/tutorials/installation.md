# Installation

## Pre-built wheels

fVDB is provided as an installable python package.
If you want to install the `fvdb` package into other python environments, we provide the following pre-built wheels.
Note that only Linux is supported for now (Ubuntu >= 20.04 recommended).


|   PyTorch      | Python     | `cu121` |
| -------------- | ---------- | ------- |
|  2.0.0-2.0.3  | 3.8 - 3.11 |   ✅     |
|  2.1.0-2.1.3  | 3.8 - 3.12 |   ✅     |
|  2.3.0        | 3.8 - 3.12 |   ✅     |

Use the following command to install `fvdb`.

```bash
pip install -U fvdb -f https://fvdb.huangjh.tech/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
```

An example combination would be `TORCH_VERSION=2.0.0` and `CUDA_VERSION=cu121`.
