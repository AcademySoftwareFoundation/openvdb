# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#


def load_backend(backend: str):
    if backend == "hash_table":
        from backend import hash_table

        return hash_table
    elif backend == "fvdb":
        from backend import fvdb

        return fvdb
    else:
        raise NotImplementedError
